#![cfg(not(any(
    target_vendor = "apple",
    target_os = "android",
    target_os = "linux",
    target_os = "redox",
)))]

use rust_viewer::project::{ProjectCreateRequest, ProjectStore, ProjectStoreError, SourceSpec};

#[test]
fn public_create_reports_the_unsupported_platform() {
    let result = ProjectStore::create(
        "Unsupported.rustscanproject",
        ProjectCreateRequest::new("Unsupported", SourceSpec::managed_images("source-a")),
    );

    assert!(matches!(
        result,
        Err(ProjectStoreError::UnsupportedPlatform)
    ));
}
