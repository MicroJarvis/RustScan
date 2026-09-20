//! Exact vs bounded forward pipeline parity (Task 3.1).

#![cfg(feature = "gpu")]

#[tokio::test(flavor = "current_thread")]
async fn bounded_forward_pipeline_parity() {
    rustscan_gs::run_bounded_forward_parity_suite()
        .await
        .expect("exact/bounded forward parity");
}
