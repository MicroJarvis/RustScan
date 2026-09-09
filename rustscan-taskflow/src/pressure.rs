use crate::{Budget, ResourceSnapshot};

/// Pressure from outside this runtime/process, normalized to [0, 1]. Do not
/// feed raw total CPU utilization here: our own useful work is not contention.
#[derive(Debug, Clone, Copy)]
pub struct PressureSample {
    pub external_cpu_pressure: f32,
    pub available_memory_bytes: u64,
}

/// Conservative hysteresis controller. Apply its result through set_budget.
/// It only affects admission; running work is never preempted or resized.
#[derive(Debug, Clone)]
pub struct AdaptivePolicy {
    ceiling: Budget,
    memory_headroom_bytes: u64,
    recovery_samples: usize,
    cool_samples: usize,
}
impl AdaptivePolicy {
    pub fn new(ceiling: Budget, memory_headroom_bytes: u64) -> Self {
        Self {
            ceiling,
            memory_headroom_bytes,
            recovery_samples: 4,
            cool_samples: 0,
        }
    }
    pub fn update(&mut self, snapshot: &ResourceSnapshot, sample: PressureSample) -> Budget {
        let pressure = if sample.external_cpu_pressure.is_finite() {
            sample.external_cpu_pressure.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let target = ((self.ceiling.cpu_threads as f32 * (1.0 - pressure)).floor() as usize)
            .max(1)
            .min(self.ceiling.cpu_threads);
        let mut cpu = snapshot.budget.cpu_threads;
        if target < cpu {
            cpu = cpu.saturating_sub(1).max(target);
            self.cool_samples = 0;
        } else if target > cpu {
            self.cool_samples += 1;
            if self.cool_samples >= self.recovery_samples {
                cpu += 1;
                self.cool_samples = 0;
            }
        } else {
            self.cool_samples = 0;
        }
        // Reservations and OS availability are different measurements. This is
        // a soft admission ceiling, never a claim that allocations cannot OOM.
        let memory = snapshot
            .memory_bytes
            .saturating_add(
                sample
                    .available_memory_bytes
                    .saturating_sub(self.memory_headroom_bytes),
            )
            .min(self.ceiling.memory_bytes);
        Budget {
            cpu_threads: cpu.min(self.ceiling.cpu_threads),
            memory_bytes: memory,
            io_slots: snapshot.budget.io_slots.min(self.ceiling.io_slots),
        }
    }
}

#[cfg(feature = "system-monitor")]
mod monitor {
    use crate::runtime::{set_budget, Command};
    use crate::{AdaptivePolicy, Error, PressureSample, Runtime};
    use std::sync::{mpsc, Arc, Condvar, Mutex};
    use std::thread::JoinHandle;
    use std::time::Duration;
    use sysinfo::System;

    /// Periodic best-effort OS feedback. GPU completion latency/VRAM and GUI
    /// frame-time monitoring belong to the device/application adapter.
    pub struct SystemMonitor {
        stop: Arc<(Mutex<bool>, Condvar)>,
        thread: Option<JoinHandle<()>>,
    }
    impl SystemMonitor {
        pub fn start(
            runtime: &Runtime,
            interval: Duration,
            headroom_bytes: u64,
        ) -> Result<Self, Error> {
            if interval < Duration::from_millis(250) {
                return Err(Error::Invalid(
                    "monitor interval must be at least 250 ms".into(),
                ));
            }
            let initial = runtime.snapshot()?;
            let mut policy = AdaptivePolicy::new(initial.budget, headroom_bytes);
            let stop = Arc::new((Mutex::new(false), Condvar::new()));
            let signal = stop.clone();
            let sender = runtime.sender.clone();
            let thread = std::thread::Builder::new()
                .name("taskflow-pressure".into())
                .spawn(move || {
                    let mut system = System::new();
                    let pid = sysinfo::get_current_pid().ok();
                    system.refresh_cpu();
                    if let Some(pid) = pid {
                        system.refresh_process(pid);
                    }
                    loop {
                        let guard = signal.0.lock().unwrap();
                        let (guard, _) = signal
                            .1
                            .wait_timeout_while(guard, interval, |stopped| !*stopped)
                            .unwrap();
                        if *guard {
                            break;
                        }
                        drop(guard);
                        system.refresh_cpu();
                        system.refresh_memory();
                        if let Some(pid) = pid {
                            system.refresh_process(pid);
                        }
                        let own = pid
                            .and_then(|p| system.process(p))
                            .map(|p| p.cpu_usage())
                            .unwrap_or(0.0);
                        let cores = system.cpus().len().max(1) as f32;
                        let external = ((system.global_cpu_info().cpu_usage() - own / cores)
                            / 100.0)
                            .clamp(0.0, 1.0);
                        let (tx, rx) = mpsc::sync_channel(1);
                        if sender.send(Command::Snapshot(tx)).is_err() {
                            break;
                        }
                        let Ok(snapshot) = rx.recv() else {
                            break;
                        };
                        let budget = policy.update(
                            &snapshot,
                            PressureSample {
                                external_cpu_pressure: external,
                                available_memory_bytes: system.available_memory(),
                            },
                        );
                        if set_budget(&sender, budget).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| Error::Executor(e.to_string()))?;
            Ok(Self {
                stop,
                thread: Some(thread),
            })
        }
    }
    impl Drop for SystemMonitor {
        fn drop(&mut self) {
            *self.stop.0.lock().unwrap() = true;
            self.stop.1.notify_all();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
}
#[cfg(feature = "system-monitor")]
pub use monitor::SystemMonitor;
