//! RustViewer — Interactive 3D viewer for RustScan SLAM results.

fn main() {
    let startup_asset = startup_asset_from_args();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("RustViewer"),
        wgpu_options: cubecl_compatible_wgpu_options(),
        ..Default::default()
    };

    eframe::run_native(
        "RustViewer",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(
                rust_viewer::app::ViewerApp::new_with_startup_asset(cc, startup_asset.clone()),
            ))
        }),
    )
    .expect("Failed to start RustViewer");
}

fn startup_asset_from_args() -> Option<std::path::PathBuf> {
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--gaussian" || arg == "--scene" || arg == "--input" {
            return args.next().map(std::path::PathBuf::from);
        }
        if arg == "--help" || arg == "-h" {
            println!(
                "Usage: rust-viewer [--gaussian <scene.splat|scene.ply>|--scene <path>|--input <path>|<path>]"
            );
            std::process::exit(0);
        }
        if !arg.to_string_lossy().starts_with('-') {
            return Some(std::path::PathBuf::from(arg));
        }
    }
    None
}

fn cubecl_compatible_wgpu_options() -> eframe::egui_wgpu::WgpuConfiguration {
    let mut options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(create_new) = &mut options.wgpu_setup {
        create_new.device_descriptor = std::sync::Arc::new(|adapter| {
            let base_limits = if adapter.get_info().backend == eframe::wgpu::Backend::Gl {
                eframe::wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                adapter.limits()
            };
            eframe::wgpu::DeviceDescriptor {
                label: Some("RustViewer shared wgpu device"),
                required_features: adapter
                    .features()
                    .difference(eframe::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                required_limits: eframe::wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    ..base_limits
                },
                experimental_features: unsafe { eframe::wgpu::ExperimentalFeatures::enabled() },
                memory_hints: eframe::wgpu::MemoryHints::MemoryUsage,
                trace: eframe::wgpu::Trace::Off,
            }
        });
    }
    options
}
