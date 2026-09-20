use anyhow::{bail, Context, Result};
use image::ImageReader;
use rustsfm::colmap::ColmapCamera;
use rustsfm::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
use std::path::Path;

pub(super) fn ensure_feature_extractor_database(
    database_path: &Path,
    image_path: &Path,
    camera_model: &str,
    single_camera: bool,
) -> Result<()> {
    if !matches!(camera_model, "PINHOLE" | "SIMPLE_PINHOLE") {
        bail!("RustSFM feature_extractor currently supports PINHOLE/SIMPLE_PINHOLE only");
    }
    let db = ColmapDatabase::open(database_path)?;
    if !db.read_all_images()?.is_empty() {
        return Ok(());
    }
    let mut images = std::fs::read_dir(image_path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
                })
        })
        .collect::<Vec<_>>();
    images.sort();
    if images.is_empty() {
        bail!("no images found under {}", image_path.display());
    }

    let mut shared_camera_id = None;
    for (index, path) in images.iter().enumerate() {
        let (width, height) = image_dimensions(path)?;
        let camera_id = if single_camera {
            if let Some(camera_id) = shared_camera_id {
                camera_id
            } else {
                let id = write_database_camera(&db, 0, width, height, camera_model)?;
                shared_camera_id = Some(id);
                id
            }
        } else {
            write_database_camera(&db, 0, width, height, camera_model)?
        };
        let name = path
            .file_name()
            .context("image path has no file name")?
            .to_string_lossy()
            .into_owned();
        db.write_image(
            &ColmapDatabaseImage {
                image_id: (index + 1) as u32,
                name,
                camera_id,
                frame_id: None,
            },
            true,
        )?;
    }
    Ok(())
}

fn image_dimensions(path: &Path) -> Result<(u32, u32)> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format for {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?;
    Ok((image.width(), image.height()))
}

fn write_database_camera(
    db: &ColmapDatabase,
    camera_id: u32,
    width: u32,
    height: u32,
    camera_model: &str,
) -> Result<u32> {
    let focal = width.max(height) as f64 * 1.2;
    let (model_id, params) = if camera_model == "SIMPLE_PINHOLE" {
        (
            rustsfm::types::COLMAP_SIMPLE_PINHOLE,
            vec![focal, width as f64 * 0.5, height as f64 * 0.5],
        )
    } else {
        (
            rustsfm::types::COLMAP_PINHOLE,
            vec![focal, focal, width as f64 * 0.5, height as f64 * 0.5],
        )
    };
    db.write_camera(
        &ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id,
                model_id,
                width,
                height,
                params,
            },
            has_prior_focal_length: false,
        },
        camera_id != 0,
    )
}
