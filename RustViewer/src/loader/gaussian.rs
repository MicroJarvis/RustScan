//! Loader for Gaussian scene .splat and PLY files.

use std::path::Path;

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::{GaussianSplat, Scene};

/// Load a 3DGS scene.splat/scene.ply and add Gaussians to the scene.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use rust_viewer::loader::gaussian::load_gaussians;
/// use rust_viewer::renderer::scene::Scene;
///
/// let mut scene = Scene::default();
/// load_gaussians(Path::new("scene.splat"), &mut scene).unwrap();
/// println!("Loaded {} gaussians", scene.gaussians.len());
/// ```
pub fn load_gaussians(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    load_gaussians_with_splats(path, scene).map(|_| ())
}

/// Load a 3DGS scene and return the original RustGS splats when the file uses
/// RustGS's native layout. Legacy RustSLAM PLY files still load into `scene`,
/// but return `None` because they do not preserve the full RustGS SH layout.
pub fn load_gaussians_with_splats(
    path: &Path,
    scene: &mut Scene,
) -> Result<Option<rustgs::HostSplats>, LoadError> {
    match rustgs::load_splats(path) {
        Ok((splats, _meta)) => {
            append_rustgs_splats(&splats, scene);
            return Ok(Some(splats));
        }
        Err(err) if should_try_legacy_rustslam_loader(path, &err) => {}
        Err(err) => {
            return Err(LoadError::PlyParse(format!(
                "invalid RustGS splat file: {err}"
            )));
        }
    }

    let gaussians = load_legacy_rustslam_gaussians(path)?;
    for g in &gaussians {
        scene.gaussians.push(GaussianSplat {
            position: g.position,
            scale: g.scale,
            rotation: g.rotation,
            opacity: g.opacity,
            color: g.color,
        });
        scene.bounds.extend(g.position);
    }

    Ok(None)
}

fn append_rustgs_splats(splats: &rustgs::HostSplats, scene: &mut Scene) {
    for idx in 0..splats.len() {
        let position = splats.position(idx);
        scene.gaussians.push(GaussianSplat {
            position,
            scale: splats.scale(idx),
            rotation: splats.rotation(idx),
            opacity: splats.opacity(idx),
            color: splats.rgb_color(idx),
        });
        scene.bounds.extend(position);
    }
}

fn should_try_legacy_rustslam_loader(path: &Path, err: &rustgs::SceneIoError) -> bool {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| !extension.eq_ignore_ascii_case("ply"))
        .unwrap_or(true)
    {
        return false;
    }

    match err {
        rustgs::SceneIoError::InvalidFormat { message } => {
            message.contains("missing required PLY property f_dc_0")
                || message.contains("missing required PLY property scale_0")
                || message.contains("missing required PLY property opacity")
                || message.contains("missing required PLY property rot_0")
        }
        _ => false,
    }
}

fn load_legacy_rustslam_gaussians(
    path: &Path,
) -> Result<Vec<rustslam::fusion::Gaussian>, LoadError> {
    rustslam::fusion::load_scene_ply(path)
        .map(|(gaussians, _meta)| gaussians)
        .map_err(|e| {
            LoadError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustslam::fusion::{save_scene_ply, Gaussian, SceneMetadata};

    #[test]
    fn loads_full_gaussian_parameters_from_ply() {
        let tmp = tempfile::NamedTempFile::with_suffix(".ply").unwrap();
        let gaussians = vec![Gaussian::new(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [0.707, 0.0, 0.707, 0.0],
            0.42,
            [0.9, 0.8, 0.7],
        )];
        save_scene_ply(
            tmp.path(),
            &gaussians,
            &SceneMetadata {
                iterations: 1,
                final_loss: 0.5,
                gaussian_count: 1,
            },
        )
        .unwrap();

        let mut scene = Scene::default();
        load_gaussians(tmp.path(), &mut scene).unwrap();

        assert_eq!(scene.gaussians.len(), 1);
        assert_eq!(scene.gaussians[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(scene.gaussians[0].scale, [0.1, 0.2, 0.3]);
        assert_eq!(scene.gaussians[0].rotation, [0.707, 0.0, 0.707, 0.0]);
        assert!((scene.gaussians[0].opacity - 0.42).abs() < 1e-6);
        assert_eq!(scene.gaussians[0].color, [0.9, 0.8, 0.7]);
    }

    #[test]
    fn loads_rustgs_training_ply_layout() {
        const SH_C0: f32 = 0.282_094_8;
        let rgb = [0.25, 0.5, 0.75];
        let sh0 = rgb.map(|channel| (channel - 0.5) / SH_C0);
        let tmp = tempfile::NamedTempFile::with_suffix(".ply").unwrap();
        let splats = rustgs::HostSplats::from_components(
            vec![1.0, 2.0, 3.0],
            vec![0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()],
            vec![0.707, 0.0, 0.707, 0.0],
            vec![0.42],
            sh0.into(),
            0,
        )
        .unwrap();
        rustgs::save_splats_ply(
            tmp.path(),
            &splats,
            &rustgs::SplatMetadata {
                iterations: 1,
                final_loss: 0.5,
                gaussian_count: 1,
                sh_degree: 0,
            },
        )
        .unwrap();

        let mut scene = Scene::default();
        load_gaussians(tmp.path(), &mut scene).unwrap();

        assert_eq!(scene.gaussians.len(), 1);
        assert_eq!(scene.gaussians[0].position, [1.0, 2.0, 3.0]);
        assert!((scene.gaussians[0].scale[0] - 0.1).abs() < 1e-6);
        assert!((scene.gaussians[0].scale[1] - 0.2).abs() < 1e-6);
        assert!((scene.gaussians[0].scale[2] - 0.3).abs() < 1e-6);
        assert_eq!(scene.gaussians[0].rotation, [0.707, 0.0, 0.707, 0.0]);
        assert!((scene.gaussians[0].opacity - 0.60348326).abs() < 1e-6);
        for (actual, expected) in scene.gaussians[0].color.into_iter().zip(rgb) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn loads_rustgs_splat_layout() {
        const SH_C0: f32 = 0.282_094_8;
        let rgb = [0.25, 0.5, 0.75];
        let sh0 = rgb.map(|channel| (channel - 0.5) / SH_C0);
        let tmp = tempfile::NamedTempFile::with_suffix(".splat").unwrap();
        let splats = rustgs::HostSplats::from_components(
            vec![1.0, 2.0, 3.0],
            vec![0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.42],
            sh0.into(),
            0,
        )
        .unwrap();
        rustgs::save_splats(
            tmp.path(),
            &splats,
            &rustgs::SplatMetadata {
                iterations: 1,
                final_loss: 0.5,
                gaussian_count: 1,
                sh_degree: 0,
            },
        )
        .unwrap();

        let mut scene = Scene::default();
        load_gaussians(tmp.path(), &mut scene).unwrap();

        assert_eq!(scene.gaussians.len(), 1);
        assert_eq!(scene.gaussians[0].position, [1.0, 2.0, 3.0]);
        assert!((scene.gaussians[0].scale[0] - 0.1).abs() < 1e-6);
        assert!((scene.gaussians[0].scale[1] - 0.2).abs() < 1e-6);
        assert!((scene.gaussians[0].scale[2] - 0.3).abs() < 1e-6);
        assert!((scene.gaussians[0].opacity - 0.6039216).abs() < 1e-6);
        for (actual, expected) in scene.gaussians[0].color.into_iter().zip(rgb) {
            assert!((actual - expected).abs() <= 1.0 / 255.0 + 1e-6);
        }
    }

    #[test]
    fn rustgs_layout_parse_errors_do_not_fall_back_to_legacy_loader() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = String::from(
            "ply\n\
             format ascii 1.0\n\
             comment sh_degree 0\n\
             element vertex 1\n\
             property float x\n\
             property float y\n\
             property float z\n\
             property float f_dc_0\n\
             property float f_dc_1\n\
             property float f_dc_2\n",
        );
        for idx in 0..45 {
            header.push_str(&format!("property float f_rest_{idx}\n"));
        }
        header.push_str(
            "property float opacity\n\
             property float scale_0\n\
             property float scale_1\n\
             property float scale_2\n\
             property float rot_0\n\
             property float rot_1\n\
             property float rot_2\n\
             property float rot_3\n\
             end_header\n\
             1 2 3\n",
        );
        std::fs::write(tmp.path(), header).unwrap();

        let mut scene = Scene::default();
        let err = load_gaussians(tmp.path(), &mut scene).expect_err("malformed RustGS PLY");

        assert!(
            err.to_string().contains("invalid RustGS splat file"),
            "unexpected error: {err}"
        );
    }
}
