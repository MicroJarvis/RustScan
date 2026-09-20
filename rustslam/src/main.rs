//! RustScan CLI entrypoint.

fn main() -> std::process::ExitCode {
    #[cfg(feature = "slam-pipeline")]
    {
        rustslam::cli::run()
    }
    #[cfg(not(feature = "slam-pipeline"))]
    {
        eprintln!("RustSLAM: slam-pipeline feature not enabled. Build with --features slam-pipeline to use CLI.");
        std::process::ExitCode::FAILURE
    }
}
