# RustSFM

Pure-Rust COLMAP-style incremental SfM experiment for RustScan.

This crate intentionally does not call the external `colmap` executable. The
current implementation follows the COLMAP mapper shape: feature extraction,
geometric verification, initial pair selection, incremental PnP registration,
track triangulation, and COLMAP text export.
