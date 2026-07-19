use super::ColmapMapperArgs;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub(super) fn colmap_bool(value: i32, name: &str) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        other => bail!("{name} expects a COLMAP-style boolean value of 0 or 1, got {other}"),
    }
}

pub(super) fn colmap_optional_bool(value: Option<i32>, name: &str) -> Result<Option<bool>> {
    value.map(|value| colmap_bool(value, name)).transpose()
}

fn parse_colmap_project(path: &Path) -> Result<HashMap<String, String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read project_path {}", path.display()))?;
    let mut values = HashMap::new();
    let mut section: Option<String> = None;
    for (line_index, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(section_name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            section = Some(section_name.trim().to_string());
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            bail!(
                "invalid project_path line {} in {}: expected key=value",
                line_index + 1,
                path.display()
            );
        };
        let key = key.trim();
        if key.is_empty() {
            bail!(
                "invalid project_path line {} in {}: empty key",
                line_index + 1,
                path.display()
            );
        }
        let full_key = if let Some(section) = &section {
            format!("{section}.{}", key)
        } else {
            key.to_string()
        };
        values.insert(full_key, value.trim().to_string());
    }
    Ok(values)
}

fn project_value<'a>(project: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    project.get(key).map(String::as_str)
}

fn parse_project_value<T>(project: &HashMap<String, String>, key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    project
        .get(key)
        .map(|value| {
            value.parse::<T>().map_err(|err| {
                anyhow::anyhow!("failed to parse project_path option {key}={value:?}: {err}")
            })
        })
        .transpose()
}

fn parse_project_bool(project: &HashMap<String, String>, key: &str) -> Result<Option<bool>> {
    let Some(value) = project.get(key) else {
        return Ok(None);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(Some(true)),
        "0" | "false" => Ok(Some(false)),
        _ => bail!("failed to parse project_path option {key}={value:?} as boolean"),
    }
}

fn colmap_project_bool_to_i32(value: Option<bool>) -> Option<i32> {
    value.map(|value| if value { 1 } else { 0 })
}

fn colmap_num_threads(value: isize, name: &str) -> Result<Option<usize>> {
    if value < 0 {
        Ok(None)
    } else {
        usize::try_from(value)
            .with_context(|| format!("{name} is out of range: {value}"))
            .map(Some)
    }
}

#[derive(Debug)]
pub(super) struct ResolvedColmapMapperArgs {
    pub(super) database_path: PathBuf,
    pub(super) image_path: PathBuf,
    pub(super) output_path: PathBuf,
    pub(super) ba_refine_focal_length: i32,
    pub(super) ba_refine_principal_point: i32,
    pub(super) ba_refine_extra_params: i32,
    pub(super) multiple_models: i32,
    pub(super) min_num_matches: usize,
    pub(super) max_num_models: usize,
    pub(super) max_model_overlap: usize,
    pub(super) min_model_size: usize,
    pub(super) snapshot_path: Option<PathBuf>,
    pub(super) snapshot_frames_freq: usize,
    pub(super) fix_existing_frames: i32,
    pub(super) init_num_trials: usize,
    pub(super) init_min_num_inliers: usize,
    pub(super) init_max_error: f32,
    pub(super) init_max_forward_motion: f32,
    pub(super) init_min_tri_angle: f32,
    pub(super) init_max_reg_trials: usize,
    pub(super) abs_pose_max_error: f32,
    pub(super) abs_pose_min_num_inliers: usize,
    pub(super) abs_pose_min_inlier_ratio: f32,
    pub(super) use_gpu_pnp: Option<i32>,
    pub(super) max_reg_trials: usize,
    pub(super) local_ba_num_images: usize,
    pub(super) global_ba_images_ratio: f32,
    pub(super) global_ba_points_ratio: f32,
    pub(super) global_ba_images_freq: usize,
    pub(super) global_ba_points_freq: usize,
    pub(super) global_ba_iterations: usize,
    pub(super) local_ba_iterations: usize,
    pub(super) global_ba_max_refinements: usize,
    pub(super) local_ba_max_refinements: usize,
    pub(super) global_ba_max_refinement_change: f32,
    pub(super) local_ba_max_refinement_change: f32,
    pub(super) global_ba_ignore_redundant_points3d: i32,
    pub(super) global_ba_ignore_redundant_points3d_min_coverage_gain: f64,
    pub(super) extract_colors: i32,
    pub(super) min_focal_length_ratio: f64,
    pub(super) max_focal_length_ratio: f64,
    pub(super) max_extra_param: f64,
    pub(super) filter_max_reproj_error: f32,
    pub(super) tri_ignore_two_view_tracks: i32,
    pub(super) random_seed: i32,
    pub(super) num_threads: Option<usize>,
}

pub(super) fn resolve_colmap_mapper_args(
    args: &ColmapMapperArgs,
) -> Result<ResolvedColmapMapperArgs> {
    let project = if let Some(path) = &args.project_path {
        parse_colmap_project(path)?
    } else {
        HashMap::new()
    };

    let database_path = args
        .database_path
        .clone()
        .or_else(|| project_value(&project, "database_path").map(PathBuf::from))
        .context("missing required --database_path (or database_path in --project_path)")?;
    let image_path = args
        .image_path
        .clone()
        .or_else(|| project_value(&project, "image_path").map(PathBuf::from))
        .context("missing required --image_path (or image_path in --project_path)")?;
    let output_path = args
        .output_path
        .clone()
        .or_else(|| project_value(&project, "output_path").map(PathBuf::from))
        .context("missing required --output_path (or output_path in --project_path)")?;

    let project_num_threads = parse_project_value::<isize>(&project, "Mapper.num_threads")?;
    let num_threads = match args.num_threads.or(project_num_threads) {
        Some(value) => colmap_num_threads(value, "Mapper.num_threads")?,
        None => None,
    };
    let project_snapshot_path = project_value(&project, "Mapper.snapshot_path")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    Ok(ResolvedColmapMapperArgs {
        database_path,
        image_path,
        output_path,
        ba_refine_focal_length: args
            .ba_refine_focal_length
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_focal_length",
            )?))
            .unwrap_or(1),
        ba_refine_principal_point: args
            .ba_refine_principal_point
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_principal_point",
            )?))
            .unwrap_or(0),
        ba_refine_extra_params: args
            .ba_refine_extra_params
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_extra_params",
            )?))
            .unwrap_or(1),
        multiple_models: args
            .multiple_models
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.multiple_models",
            )?))
            .unwrap_or(1),
        min_num_matches: args
            .min_num_matches
            .or(parse_project_value(&project, "Mapper.min_num_matches")?)
            .unwrap_or(15),
        max_num_models: args
            .max_num_models
            .or(parse_project_value(&project, "Mapper.max_num_models")?)
            .unwrap_or(50),
        max_model_overlap: args
            .max_model_overlap
            .or(parse_project_value(&project, "Mapper.max_model_overlap")?)
            .unwrap_or(20),
        min_model_size: args
            .min_model_size
            .or(parse_project_value(&project, "Mapper.min_model_size")?)
            .unwrap_or(10),
        snapshot_path: args.snapshot_path.clone().or(project_snapshot_path),
        snapshot_frames_freq: args
            .snapshot_frames_freq
            .or(parse_project_value(
                &project,
                "Mapper.snapshot_frames_freq",
            )?)
            .unwrap_or(0),
        fix_existing_frames: args
            .fix_existing_frames
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.fix_existing_frames",
            )?))
            .unwrap_or(0),
        init_num_trials: args
            .init_num_trials
            .or(parse_project_value(&project, "Mapper.init_num_trials")?)
            .unwrap_or(200),
        init_min_num_inliers: args
            .init_min_num_inliers
            .or(parse_project_value(
                &project,
                "Mapper.init_min_num_inliers",
            )?)
            .unwrap_or(100),
        init_max_error: args
            .init_max_error
            .or(parse_project_value(&project, "Mapper.init_max_error")?)
            .unwrap_or(4.0),
        init_max_forward_motion: args
            .init_max_forward_motion
            .or(parse_project_value(
                &project,
                "Mapper.init_max_forward_motion",
            )?)
            .unwrap_or(0.95),
        init_min_tri_angle: args
            .init_min_tri_angle
            .or(parse_project_value(&project, "Mapper.init_min_tri_angle")?)
            .unwrap_or(16.0),
        init_max_reg_trials: args
            .init_max_reg_trials
            .or(parse_project_value(&project, "Mapper.init_max_reg_trials")?)
            .unwrap_or(2),
        abs_pose_max_error: args
            .abs_pose_max_error
            .or(parse_project_value(&project, "Mapper.abs_pose_max_error")?)
            .unwrap_or(12.0),
        abs_pose_min_num_inliers: args
            .abs_pose_min_num_inliers
            .or(parse_project_value(
                &project,
                "Mapper.abs_pose_min_num_inliers",
            )?)
            .unwrap_or(30),
        abs_pose_min_inlier_ratio: args
            .abs_pose_min_inlier_ratio
            .or(parse_project_value(
                &project,
                "Mapper.abs_pose_min_inlier_ratio",
            )?)
            .unwrap_or(0.25),
        use_gpu_pnp: args
            .use_gpu_pnp
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.use_gpu_pnp",
            )?)),
        max_reg_trials: args
            .max_reg_trials
            .or(parse_project_value(&project, "Mapper.max_reg_trials")?)
            .unwrap_or(3),
        local_ba_num_images: args
            .local_ba_num_images
            .or(parse_project_value(&project, "Mapper.ba_local_num_images")?)
            .unwrap_or(6),
        global_ba_images_ratio: args
            .global_ba_images_ratio
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_frames_ratio",
            )?)
            .unwrap_or(1.5),
        global_ba_points_ratio: args
            .global_ba_points_ratio
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_points_ratio",
            )?)
            .unwrap_or(1.5),
        global_ba_images_freq: args
            .global_ba_images_freq
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_frames_freq",
            )?)
            .unwrap_or(500),
        global_ba_points_freq: args
            .global_ba_points_freq
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_points_freq",
            )?)
            .unwrap_or(250_000),
        global_ba_iterations: args
            .global_ba_iterations
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_num_iterations",
            )?)
            .unwrap_or(50),
        local_ba_iterations: args
            .local_ba_iterations
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_num_iterations",
            )?)
            .unwrap_or(25),
        global_ba_max_refinements: args
            .global_ba_max_refinements
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_refinements",
            )?)
            .unwrap_or(5),
        local_ba_max_refinements: args
            .local_ba_max_refinements
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_refinements",
            )?)
            .unwrap_or(2),
        global_ba_max_refinement_change: args
            .global_ba_max_refinement_change
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_refinement_change",
            )?)
            .unwrap_or(0.0005),
        local_ba_max_refinement_change: args
            .local_ba_max_refinement_change
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_refinement_change",
            )?)
            .unwrap_or(0.001),
        global_ba_ignore_redundant_points3d: args
            .global_ba_ignore_redundant_points3d
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_global_ignore_redundant_points3D",
            )?))
            .unwrap_or(0),
        global_ba_ignore_redundant_points3d_min_coverage_gain: args
            .global_ba_ignore_redundant_points3d_min_coverage_gain
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain",
            )?)
            .unwrap_or(0.05),
        extract_colors: args
            .extract_colors
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.extract_colors",
            )?))
            .unwrap_or(1),
        min_focal_length_ratio: args
            .min_focal_length_ratio
            .or(parse_project_value(
                &project,
                "Mapper.min_focal_length_ratio",
            )?)
            .unwrap_or(0.1),
        max_focal_length_ratio: args
            .max_focal_length_ratio
            .or(parse_project_value(
                &project,
                "Mapper.max_focal_length_ratio",
            )?)
            .unwrap_or(10.0),
        max_extra_param: args
            .max_extra_param
            .or(parse_project_value(&project, "Mapper.max_extra_param")?)
            .unwrap_or(1.0),
        filter_max_reproj_error: args
            .filter_max_reproj_error
            .or(parse_project_value(
                &project,
                "Mapper.filter_max_reproj_error",
            )?)
            .unwrap_or(4.0),
        tri_ignore_two_view_tracks: args
            .tri_ignore_two_view_tracks
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.tri_ignore_two_view_tracks",
            )?))
            .unwrap_or(1),
        random_seed: args
            .random_seed
            .or(parse_project_value(&project, "Mapper.random_seed")?)
            .unwrap_or(-1),
        num_threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ColmapMapperArgs;
    use std::path::PathBuf;

    fn base_mapper_args(project_path: PathBuf) -> ColmapMapperArgs {
        ColmapMapperArgs {
            project_path: Some(project_path),
            database_path: None,
            image_path: None,
            output_path: None,
            ba_refine_focal_length: None,
            ba_refine_principal_point: None,
            ba_refine_extra_params: None,
            multiple_models: None,
            min_num_matches: None,
            max_num_models: None,
            max_model_overlap: None,
            min_model_size: None,
            snapshot_path: None,
            snapshot_frames_freq: None,
            fix_existing_frames: None,
            init_num_trials: None,
            init_min_num_inliers: None,
            init_max_error: None,
            init_max_forward_motion: None,
            init_min_tri_angle: None,
            init_max_reg_trials: None,
            abs_pose_max_error: None,
            abs_pose_min_num_inliers: None,
            abs_pose_min_inlier_ratio: None,
            use_gpu_pnp: None,
            max_reg_trials: None,
            local_ba_num_images: None,
            global_ba_images_ratio: None,
            global_ba_points_ratio: None,
            global_ba_images_freq: None,
            global_ba_points_freq: None,
            global_ba_iterations: None,
            local_ba_iterations: None,
            global_ba_max_refinements: None,
            local_ba_max_refinements: None,
            global_ba_max_refinement_change: None,
            local_ba_max_refinement_change: None,
            global_ba_ignore_redundant_points3d: None,
            global_ba_ignore_redundant_points3d_min_coverage_gain: None,
            extract_colors: None,
            min_focal_length_ratio: None,
            max_focal_length_ratio: None,
            max_extra_param: None,
            filter_max_reproj_error: None,
            tri_ignore_two_view_tracks: None,
            random_seed: None,
            num_threads: None,
            summary_json: None,
            log_level: "info".to_string(),
        }
    }

    #[test]
    fn colmap_mapper_project_path_overrides_defaults() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let project_path = dir.path().join("project.ini");
        std::fs::write(
            &project_path,
            "\
database_path=/tmp/project.db
image_path=/tmp/images
output_path=/tmp/sparse
[Mapper]
ba_refine_focal_length=false
ba_refine_extra_params=false
multiple_models=false
use_gpu_pnp=true
extract_colors=false
filter_max_reproj_error=4
tri_ignore_two_view_tracks=true
ba_global_frames_ratio=1.25
ba_global_points_ratio=1.35
num_threads=-1
",
        )?;

        let resolved = resolve_colmap_mapper_args(&base_mapper_args(project_path))?;

        assert_eq!(resolved.database_path, PathBuf::from("/tmp/project.db"));
        assert_eq!(resolved.image_path, PathBuf::from("/tmp/images"));
        assert_eq!(resolved.output_path, PathBuf::from("/tmp/sparse"));
        assert_eq!(resolved.ba_refine_focal_length, 0);
        assert_eq!(resolved.ba_refine_extra_params, 0);
        assert_eq!(resolved.multiple_models, 0);
        assert_eq!(resolved.use_gpu_pnp, Some(1));
        assert_eq!(resolved.extract_colors, 0);
        assert_eq!(resolved.filter_max_reproj_error, 4.0);
        assert_eq!(resolved.tri_ignore_two_view_tracks, 1);
        assert_eq!(resolved.global_ba_images_ratio, 1.25);
        assert_eq!(resolved.global_ba_points_ratio, 1.35);
        assert_eq!(resolved.num_threads, None);
        Ok(())
    }

    #[test]
    fn colmap_mapper_uses_native_global_ba_ratio_defaults_when_unspecified() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let project_path = dir.path().join("project.ini");
        std::fs::write(
            &project_path,
            "database_path=/tmp/project.db\nimage_path=/tmp/images\noutput_path=/tmp/sparse\n",
        )?;

        let resolved = resolve_colmap_mapper_args(&base_mapper_args(project_path))?;

        assert_eq!(resolved.global_ba_images_ratio, 1.5);
        assert_eq!(resolved.global_ba_points_ratio, 1.5);
        Ok(())
    }

    #[test]
    fn colmap_mapper_cli_values_override_project_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let project_path = dir.path().join("project.ini");
        std::fs::write(
            &project_path,
            "\
database_path=/tmp/project.db
image_path=/tmp/images
output_path=/tmp/sparse
[Mapper]
ba_refine_focal_length=false
extract_colors=false
use_gpu_pnp=true
num_threads=-1
",
        )?;
        let mut args = base_mapper_args(project_path);
        args.database_path = Some(PathBuf::from("/tmp/cli.db"));
        args.ba_refine_focal_length = Some(1);
        args.extract_colors = Some(1);
        args.use_gpu_pnp = Some(0);
        args.num_threads = Some(4);

        let resolved = resolve_colmap_mapper_args(&args)?;

        assert_eq!(resolved.database_path, PathBuf::from("/tmp/cli.db"));
        assert_eq!(resolved.ba_refine_focal_length, 1);
        assert_eq!(resolved.extract_colors, 1);
        assert_eq!(resolved.use_gpu_pnp, Some(0));
        assert_eq!(resolved.num_threads, Some(4));
        Ok(())
    }
}
