//! Pipeline module for multi-threaded SLAM processing

pub mod checkpoint;
#[cfg(feature = "slam-pipeline")]
pub mod realtime;
