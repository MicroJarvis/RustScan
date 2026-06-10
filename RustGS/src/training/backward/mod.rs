//! Backward rendering pipeline

pub mod autodiff;
pub mod project_bwd;
pub mod rasterize_bwd;

pub(crate) use autodiff::render_splats_with_visibility_active_sh;
