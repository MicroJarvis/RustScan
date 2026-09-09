//! Backend-independent example: one external thread simulates an asynchronous
//! GPU queue. This is a lifecycle demonstration, NOT a GPU performance benchmark.
use rayon::prelude::*;
use rustscan_taskflow::*;
use std::sync::{mpsc, Arc};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RuntimeConfig {
        budget: Budget {
            cpu_threads: 4,
            memory_bytes: 32 * 1024 * 1024,
            io_slots: 1,
        },
        gpus: vec![GpuCapacity {
            id: DeviceId(0),
            memory_bytes: 16 * 1024 * 1024,
            max_in_flight: 1,
            shared_host_memory: true,
        }],
        ..RuntimeConfig::default()
    };
    let runtime = Runtime::new(config)?;
    let (gpu_tx, gpu_rx) = mpsc::sync_channel::<Box<dyn FnOnce() + Send>>(8);
    let driver = std::thread::spawn(move || {
        for work in gpu_rx {
            work();
        }
    });
    let mut runs = Vec::new();
    for job in 0..3 {
        let mut graph = TaskGraph::new();
        let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
        request.output_memory_bytes = 80_000;
        let input = graph.task(
            "prepare",
            vec![TaskVariant::cpu("cpu", request, |_| {
                Ok((0..10_000u64).collect::<Vec<_>>())
            })],
        )?;
        let gpu_input = input.clone();
        let cpu_input = input.clone();
        let tx = gpu_tx.clone();
        let mut gpu_request = ResourceRequest::cpu(CpuRequest::fixed(1));
        gpu_request.output_memory_bytes = 80_000;
        gpu_request.gpu = Some(GpuRequest {
            device: None,
            working_memory_bytes: 80_000,
            output_memory_bytes: 0,
        });
        let mut cpu_request = ResourceRequest::cpu(CpuRequest::scalable(1, 2, 4));
        cpu_request.output_memory_bytes = 80_000;
        let scaled = graph.task(
            "scale",
            vec![
                TaskVariant::asynchronous("simulated-gpu", gpu_request, move |ctx, done| {
                    let input = match ctx.input(&gpu_input) {
                        Ok(input) => input,
                        Err(e) => {
                            done.complete(Err(e));
                            return;
                        }
                    };
                    let token = ctx.cancellation();
                    tx.send(Box::new(move || {
                        std::thread::sleep(Duration::from_millis(20));
                        done.complete(
                            token
                                .check()
                                .map(|_| input.iter().map(|x| x * 2).collect::<Vec<_>>()),
                        );
                    }))
                    .expect("simulation driver stopped");
                }),
                TaskVariant::cpu("cpu", cpu_request, move |ctx| {
                    let input = ctx.input(&cpu_input)?;
                    ctx.parallel(|| input.par_iter().map(|x| x * 2).collect::<Vec<_>>())
                }),
            ],
        )?;
        let scaled_input = scaled.clone();
        let checksum = graph.task(
            "checksum",
            vec![TaskVariant::cpu(
                "cpu",
                ResourceRequest::cpu(CpuRequest::fixed(1)),
                move |ctx| Ok(ctx.input(&scaled_input)?.iter().sum::<u64>()),
            )],
        )?;
        graph.depends_on(scaled.id(), input.id())?;
        graph.depends_on(checksum.id(), scaled.id())?;
        runs.push((job, runtime.submit(graph)?, checksum));
    }
    for (job, run, output) in runs {
        let report = run
            .wait_timeout(Duration::from_secs(10))
            .ok_or("workflow timeout")?;
        assert!(report.succeeded(), "{report:?}");
        assert_eq!(*run.output(&output)?, 99_990_000);
        println!(
            "job {job}: checksum={}, elapsed={:?}",
            *run.output(&output)?,
            report.elapsed
        );
        while let Some(event) = run.try_event() {
            if let Event::Started { task, grant, .. } = event {
                println!(
                    "  task {}: {} / {} CPU threads / GPU {:?}",
                    task.index(),
                    grant.variant,
                    grant.cpu_threads,
                    grant.gpu.map(|g| g.device)
                );
            }
        }
    }
    drop(gpu_tx);
    driver.join().unwrap();
    println!("remaining reservations: {:?}", runtime.snapshot()?);
    // The runtime can also be shared between independent submitting threads.
    let _shared = Arc::new(runtime);
    Ok(())
}
