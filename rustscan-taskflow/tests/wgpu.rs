#![cfg(feature = "wgpu-backend")]

use rustscan_taskflow::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Explicit hardware gate; fails (does not silently pass) if no adapter exists.
#[test]
#[ignore = "requires a real compute-capable wgpu adapter"]
fn real_gpu_compute_then_async_readback() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no wgpu adapter");
    eprintln!("taskflow hardware test: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let device = Arc::new(device);
    let backend = WgpuBackend::new(DeviceId(0), device.clone(), Arc::new(queue));
    let runtime = Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 2,
            memory_bytes: 1024 * 1024,
            io_slots: 1,
        },
        gpus: vec![GpuCapacity {
            id: DeviceId(0),
            memory_bytes: 1024 * 1024,
            max_in_flight: 1,
            shared_host_memory: true,
        }],
        ..RuntimeConfig::default()
    })
    .unwrap();
    let mut graph = TaskGraph::new();
    let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
    request.gpu = Some(GpuRequest {
        device: Some(DeviceId(0)),
        working_memory_bytes: 256,
        output_memory_bytes: 256,
    });
    let gpu = graph.task("fill GPU buffer", vec![TaskVariant::asynchronous("wgpu", request, move |ctx, done| {
        let device = backend.device();
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taskflow-storage"), size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taskflow-readback"), size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taskflow shader"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> values: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { values[id.x] = id.x * 3u + 1u; }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taskflow pipeline"), layout: None, module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage.as_entire_binding() }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]); pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, 256);
        backend.submit(ctx, [encoder.finish()], readback, done);
    })]).unwrap();
    let input = gpu.clone();
    let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
    request.output_memory_bytes = 256;
    let output = graph
        .task(
            "map readback",
            vec![TaskVariant::asynchronous(
                "map",
                request,
                move |ctx, done| {
                    let buffer = match ctx.input(&input) {
                        Ok(buffer) => buffer,
                        Err(error) => {
                            done.complete(Err(error));
                            return;
                        }
                    };
                    let keep_alive = buffer.clone();
                    buffer
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let result =
                                result
                                    .map_err(|e| TaskError::Failed(e.to_string()))
                                    .map(|_| {
                                        let bytes = keep_alive.slice(..).get_mapped_range();
                                        let values: Vec<u32> = bytes
                                            .chunks_exact(4)
                                            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                                            .collect();
                                        drop(bytes);
                                        keep_alive.unmap();
                                        values
                                    });
                            done.complete(result);
                        });
                },
            )],
        )
        .unwrap();
    graph.depends_on(output.id(), gpu.id()).unwrap();
    let run = runtime.submit(graph).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let report = loop {
        device.poll(wgpu::PollType::Poll).unwrap();
        if let Some(report) = run.wait_timeout(Duration::from_millis(2)) {
            break report;
        }
        assert!(Instant::now() < deadline, "GPU completion timed out");
    };
    assert!(report.succeeded(), "{report:?}");
    assert_eq!(
        &**run.output(&output).unwrap(),
        &(0..64u32).map(|i| i * 3 + 1).collect::<Vec<_>>()
    );
    drop(gpu);
    drop(output);
    assert_eq!(runtime.snapshot().unwrap().gpus[&DeviceId(0)].in_flight, 0);
}
