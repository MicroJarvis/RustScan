#[cfg(target_os = "macos")]
use rustsfm::gpu::WgpuContext;

#[cfg(target_os = "macos")]
#[test]
fn reports_capabilities_for_an_available_wgpu_context() {
    let context = WgpuContext::try_new_optional()
        .expect("macOS RustViewer should probe wgpu adapters without an error");
    let Some(context) = context else {
        eprintln!("Skipping macOS wgpu context check: no compatible wgpu adapter is available");
        return;
    };

    assert!(
        !context.capabilities().device_name.trim().is_empty(),
        "a compatible wgpu context should report its device name"
    );
}
