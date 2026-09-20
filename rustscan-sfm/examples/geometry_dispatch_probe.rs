//! Experimental synchronous dispatch probe, not a pair-pipeline benchmark.
//! No GPU object crosses a thread boundary; even the handler need not be Send.
use anyhow::{anyhow, Context, Result};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

struct Envelope<Q, A> {
    request: Q,
    reply: SyncSender<Result<A>>,
}

struct Worker<Q, A> {
    sender: Option<SyncSender<Envelope<Q, A>>>,
    thread: Option<JoinHandle<()>>,
}

fn serve<Q, A>(receiver: Receiver<Envelope<Q, A>>, mut handler: impl FnMut(Q) -> Result<A>) {
    while let Ok(message) = receiver.recv() {
        // A caller may abandon its reply without terminating the worker.
        let _ = message.reply.send(handler(message.request));
    }
}

impl<Q: Send + 'static, A: Send + 'static> Worker<Q, A> {
    fn start<H>(factory: impl FnOnce() -> Result<H> + Send + 'static) -> Result<Self>
    where
        H: FnMut(Q) -> Result<A> + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("geometry-dispatch-probe".into())
            .spawn(move || match factory() {
                Ok(handler) => {
                    if ready_tx.send(Ok(())).is_ok() {
                        serve(receiver, handler);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            })?;
        let mut worker = Self {
            sender: Some(sender),
            thread: Some(thread),
        };
        let ready = ready_rx
            .recv()
            .context("worker disconnected during startup");
        match ready.and_then(|result| result) {
            Ok(()) => Ok(worker),
            Err(error) => {
                worker.stop()?;
                Err(error)
            }
        }
    }

    fn call(&self, request: Q) -> Result<A> {
        let (reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .as_ref()
            .context("worker already shut down")?
            .send(Envelope { request, reply })
            .map_err(|_| anyhow!("worker request channel disconnected"))?;
        receiver
            .recv()
            .context("worker reply channel disconnected")?
    }
}

impl<Q, A> Worker<Q, A> {
    fn stop(&mut self) -> Result<()> {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| anyhow!("worker panicked"))?;
        }
        Ok(())
    }

    fn shutdown(mut self) -> Result<()> {
        self.stop()
    }
}

impl<Q, A> Drop for Worker<Q, A> {
    fn drop(&mut self) {
        // Explicit shutdown reports panics; Drop still joins on early-return paths.
        let _ = self.stop();
    }
}

#[derive(clap::Parser)]
#[command(
    about = "Experimental direct vs single-worker geometry dispatch (not pair pipeline)",
    long_about = "Runs actual GPU homography and Sampson score/mask calls: 512 models, 512 observations, four paired rounds with balanced alternating order. One iteration is four calls. Both paths reuse the same thread-local scorer, but public APIs recreate sessions per call. Forwarding uses preloaded inputs, a capacity-1 request queue and a capacity-1 reply per call, with one outstanding request. Startup, warmup and comparisons are excluded from timings. Direct timing is on the worker; forwarded timing includes caller/channel round trips. No GPU is created for --help. Iterations bound work, not driver stalls."
)]
struct Args {
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=100),
        help = "Iterations per path per round (1..=100; four calls per iteration)")]
    iterations: u32,
}

fn main() -> Result<()> {
    use clap::Parser;
    let args = Args::parse();
    #[cfg(feature = "gpu-wgpu")]
    return gpu::run(args.iterations);
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        let _ = args;
        anyhow::bail!("geometry_dispatch_probe requires --features gpu-wgpu");
    }
}

#[cfg(feature = "gpu-wgpu")]
mod gpu {
    use super::*;
    use rustscan_sfm::gpu::{TwoViewModelKind, WgpuContext, WgpuModelScorer};
    use std::time::{Duration, Instant};

    const SIZE: usize = 512;
    const ROUNDS: usize = 4;

    struct Workload {
        models: Vec<[f32; 9]>,
        left: Vec<[f32; 2]>,
        right: Vec<[f32; 2]>,
        kind: TwoViewModelKind,
    }

    impl Workload {
        fn new(kind: TwoViewModelKind) -> Self {
            let homography = kind == TwoViewModelKind::HomographyForward;
            let models = (0..SIZE)
                .map(|i| {
                    let delta = (i as f32 - 256.0) / 64.0;
                    if homography {
                        [
                            1.0,
                            0.0,
                            12.0 + delta,
                            0.0,
                            1.0,
                            -5.0 + delta * 0.2,
                            0.0,
                            0.0,
                            1.0,
                        ]
                    } else {
                        // Rectified stereo: y_left - y_right + delta = 0.
                        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, delta]
                    }
                })
                .collect();
            let left: Vec<_> = (0..SIZE)
                .map(|i| [24.0 + (i % 32) as f32 * 19.0, 20.0 + (i / 32) as f32 * 28.0])
                .collect();
            let right = left
                .iter()
                .enumerate()
                .map(|(i, &[x, y])| {
                    let noise = ((i * 17 % 23) as f32 - 11.0) * 0.04;
                    let outlier = if i % 7 == 0 { 18.0 } else { 0.0 };
                    if homography {
                        [x + 12.0 + noise, y - 5.0 - noise + outlier]
                    } else {
                        [x - 8.0 - (i % 13) as f32, y + noise + outlier]
                    }
                })
                .collect();
            Self {
                models,
                left,
                right,
                kind,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Output {
        // Preserve every residual bit, including signed zero and NaN payloads.
        Scores(Vec<(u32, u32)>),
        Mask(Vec<bool>),
    }

    struct Measurement {
        elapsed: Duration,
        outputs: Vec<Output>,
    }

    enum Request {
        Call(usize),
        Direct(u32),
    }

    enum Reply {
        Call(Output),
        Direct(Measurement),
    }

    fn execute(scorer: &WgpuModelScorer, inputs: &[Workload; 2], call: usize) -> Result<Output> {
        let input = &inputs[call / 2];
        if call % 2 == 0 {
            let scores = scorer.score_two_view_models(
                &input.models,
                &input.left,
                &input.right,
                1.5,
                input.kind,
            )?;
            anyhow::ensure!(scores.len() == SIZE, "incomplete score output");
            Ok(Output::Scores(
                scores
                    .into_iter()
                    .map(|s| (s.inliers, s.residual_sum.to_bits()))
                    .collect(),
            ))
        } else {
            let mask = scorer.inlier_mask(
                &input.models[SIZE / 2],
                &input.left,
                &input.right,
                1.5,
                input.kind,
            )?;
            anyhow::ensure!(mask.len() == SIZE, "incomplete mask output");
            Ok(Output::Mask(mask))
        }
    }

    // Identical loop, output retention and timing boundaries for both paths.
    fn bench(
        iterations: u32,
        mut call: impl FnMut(usize) -> Result<Output>,
    ) -> Result<Measurement> {
        let mut outputs = Vec::with_capacity(iterations as usize * 4);
        let start = Instant::now();
        for _ in 0..iterations {
            for operation in 0..4 {
                outputs.push(call(operation)?);
            }
        }
        Ok(Measurement {
            elapsed: start.elapsed(),
            outputs,
        })
    }

    fn direct(worker: &Worker<Request, Reply>, iterations: u32) -> Result<Measurement> {
        match worker.call(Request::Direct(iterations))? {
            Reply::Direct(measurement) => Ok(measurement),
            _ => anyhow::bail!("unexpected direct reply"),
        }
    }

    fn forwarded(worker: &Worker<Request, Reply>, iterations: u32) -> Result<Measurement> {
        bench(iterations, |operation| {
            match worker.call(Request::Call(operation))? {
                Reply::Call(output) => Ok(output),
                _ => anyhow::bail!("unexpected forwarded reply"),
            }
        })
    }

    pub fn run(iterations: u32) -> Result<()> {
        let worker = Worker::start(|| {
            // Creation, use and destruction all happen on this dedicated thread.
            // No Send bound on the handler: compatible with the test context's !Send lease.
            let context = WgpuContext::try_new()?;
            println!(
                "adapter={} backend={:?}",
                context.capabilities().device_name,
                context.backend()
            );
            let scorer = WgpuModelScorer::from_context(context)?;
            let inputs = [
                Workload::new(TwoViewModelKind::HomographyForward),
                Workload::new(TwoViewModelKind::Sampson),
            ];
            Ok(move |request| match request {
                Request::Call(operation) => execute(&scorer, &inputs, operation).map(Reply::Call),
                Request::Direct(iterations) => {
                    bench(iterations, |operation| execute(&scorer, &inputs, operation))
                        .map(Reply::Direct)
                }
            })
        })?;
        println!("512 models x 512 observations; iterations={iterations}; rounds={ROUNDS}; four calls/iteration (two scores, two masks)");
        println!("Same scorer/thread; fresh public-API session per call (private session reuse unavailable). Preloaded inputs; one outstanding request; no pair pipeline.");
        println!("Wall times include GPU completion/readback and full output retention; exclude startup, warmup, comparisons and direct-round control messages.");
        let warm_direct = direct(&worker, 1)?;
        let warm_forwarded = forwarded(&worker, 1)?;
        anyhow::ensure!(
            warm_direct.outputs == warm_forwarded.outputs,
            "warmup full-output mismatch"
        );
        let mut direct_total = Duration::ZERO;
        let mut forwarded_total = Duration::ZERO;
        for round in 0..ROUNDS {
            let (baseline, candidate) = if round % 2 == 0 {
                (
                    direct(&worker, iterations)?,
                    forwarded(&worker, iterations)?,
                )
            } else {
                let candidate = forwarded(&worker, iterations)?;
                (direct(&worker, iterations)?, candidate)
            };
            anyhow::ensure!(
                baseline.outputs == candidate.outputs,
                "round {} full-output mismatch",
                round + 1
            );
            direct_total += baseline.elapsed;
            forwarded_total += candidate.elapsed;
            println!("round={} order={} direct_ms={:.3} forwarded_ms={:.3} delta_ms={:.3} calls_per_path={} score_calls={} mask_calls={} forwarded_requests={} direct_control_requests=1 exact_full_outputs=true",
                round + 1, if round % 2 == 0 { "direct,forwarded" } else { "forwarded,direct" },
                baseline.elapsed.as_secs_f64() * 1000.0, candidate.elapsed.as_secs_f64() * 1000.0,
                (candidate.elapsed.as_secs_f64() - baseline.elapsed.as_secs_f64()) * 1000.0,
                iterations * 4, iterations * 2, iterations * 2, iterations * 4);
        }
        let calls = ROUNDS as u32 * iterations * 4;
        println!("measured_totals: calls_per_path={calls} direct_ms={:.3} forwarded_ms={:.3} delta_us_per_call={:.3}; warmup excluded: 4 calls/path, 1 direct control + 4 forwarded requests",
            direct_total.as_secs_f64() * 1000.0, forwarded_total.as_secs_f64() * 1000.0,
            (forwarded_total.as_secs_f64() - direct_total.as_secs_f64()) * 1e6 / f64::from(calls));
        println!("Delta includes scheduling, reply allocation and transport, not pure channel cost; four rounds are exploratory, not statistical or pipeline evidence.");
        worker.shutdown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::sync::{Arc, Barrier};

    #[test]
    fn backpressure_is_capacity_one() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let release = Arc::new(Barrier::new(2));
        let worker_release = release.clone();
        let thread = thread::spawn(move || {
            serve(receiver, |request| {
                if request == 0 {
                    entered_tx.send(()).unwrap();
                    worker_release.wait();
                }
                Ok(request)
            })
        });
        let message = |request| {
            let (reply, receiver) = mpsc::sync_channel(1);
            (Envelope { request, reply }, receiver)
        };
        let (first, first_reply) = message(0);
        sender.send(first).unwrap();
        entered_rx.recv().unwrap();
        let (second, second_reply) = message(1);
        sender.send(second).unwrap();
        let (third, third_reply) = message(2);
        let third = match sender.try_send(third) {
            Err(mpsc::TrySendError::Full(message)) => message,
            _ => panic!("handler held at barrier: second request must fill queue"),
        };
        release.wait();
        sender.send(third).unwrap();
        for (expected, reply) in [first_reply, second_reply, third_reply]
            .into_iter()
            .enumerate()
        {
            assert_eq!(reply.recv().unwrap().unwrap(), expected);
        }
        drop(sender);
        thread.join().unwrap();
    }

    #[test]
    fn handler_errors_propagate_and_worker_survives() {
        let worker = Worker::start(|| {
            Ok(|fail| {
                if fail {
                    anyhow::bail!("fake handler failure");
                }
                Ok(42)
            })
        })
        .unwrap();
        assert_eq!(
            worker.call(true).unwrap_err().to_string(),
            "fake handler failure"
        );
        assert_eq!(worker.call(false).unwrap(), 42);
        worker.shutdown().unwrap();
    }

    #[test]
    fn disconnect_releases_receiver() {
        let (sender, receiver) = mpsc::sync_channel::<Envelope<(), ()>>(1);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            ready_tx.send(()).unwrap();
            serve(receiver, |_| panic!("no requests expected"));
        });
        ready_rx.recv().unwrap();
        drop(sender);
        thread.join().unwrap();
    }

    #[test]
    fn shutdown_joins_and_non_send_handler_stays_on_owner_thread() {
        struct LocalGuard {
            owner: thread::ThreadId,
            dropped: SyncSender<thread::ThreadId>,
            _not_send: Rc<()>,
        }
        impl Drop for LocalGuard {
            fn drop(&mut self) {
                assert_eq!(self.owner, thread::current().id());
                self.dropped.send(self.owner).unwrap();
            }
        }
        let caller = thread::current().id();
        let (dropped, receiver) = mpsc::sync_channel(1);
        let worker = Worker::start(move || {
            let guard = LocalGuard {
                owner: thread::current().id(),
                dropped,
                _not_send: Rc::new(()),
            };
            Ok(move |()| {
                assert_eq!(guard.owner, thread::current().id());
                // Capture the whole !Send guard, not just its Send fields.
                let _ = &guard;
                Ok(guard.owner)
            })
        })
        .unwrap();
        let owner = worker.call(()).unwrap();
        assert_ne!(owner, caller);
        worker.shutdown().unwrap();
        assert_eq!(receiver.try_recv().unwrap(), owner);
    }

    #[test]
    fn abandoned_reply_does_not_kill_worker() {
        let worker = Worker::start(|| Ok(|value: u32| Ok(value + 1))).unwrap();
        let (reply, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        worker
            .sender
            .as_ref()
            .unwrap()
            .send(Envelope { request: 1, reply })
            .unwrap();
        assert_eq!(worker.call(2).unwrap(), 3);
        worker.shutdown().unwrap();
    }

    #[test]
    fn startup_error_is_propagated() {
        let result = Worker::<(), ()>::start(|| -> Result<fn(()) -> Result<()>> {
            anyhow::bail!("fake initialization failure")
        });
        assert_eq!(
            result.err().unwrap().to_string(),
            "fake initialization failure"
        );
    }

    #[test]
    fn missing_reply_is_an_error_and_panicked_worker_is_joined() {
        let worker = Worker::<(), ()>::start(|| Ok(|()| panic!("fake worker panic"))).unwrap();
        assert!(worker
            .call(())
            .unwrap_err()
            .to_string()
            .contains("reply channel disconnected"));
        assert_eq!(
            worker.shutdown().unwrap_err().to_string(),
            "worker panicked"
        );
    }

    #[test]
    fn cli_rejects_invalid_iterations_and_help_needs_no_gpu() {
        use clap::Parser;
        for value in ["0", "101", "-1", "nope"] {
            assert!(Args::try_parse_from(["probe", "--iterations", value]).is_err());
        }
        assert_eq!(Args::try_parse_from(["probe"]).unwrap().iterations, 3);
        assert_eq!(
            Args::try_parse_from(["probe", "--help"])
                .err()
                .unwrap()
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );
    }
}
