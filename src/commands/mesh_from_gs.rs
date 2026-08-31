use anyhow::{bail, Context};
use clap::Args;
use glam::{Mat4, Vec3};
use rustgs::{EvaluationDevice, GaussianCamera, HostSplats, SplatEvaluationRenderer};
use rustmesh::RustMesh;
use rustscan_types::{Intrinsics, ScenePose, TrainingDataset, SE3};
use rustslam::fusion::{Mesh, MeshExtractionConfig, MeshExtractor, TsdfConfig};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Args)]
pub struct MeshFromGsArgs {
    /// Trained RustGS scene .splat or PLY.
    #[arg(long)]
    scene: PathBuf,

    /// Original COLMAP training input directory used for the scene.
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for mesh.obj, mesh.ply, and metadata.
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Render scale for depth/color fusion.
    #[arg(long, default_value = "0.5")]
    render_scale: f32,

    /// Integrate every Nth pose from the dataset.
    #[arg(long, default_value = "5")]
    view_stride: usize,

    /// Maximum number of views to integrate after stride filtering (0 = all).
    #[arg(long, default_value = "0")]
    max_views: usize,

    /// TSDF voxel size in scene units.
    #[arg(long, default_value = "0.01")]
    voxel_size: f32,

    /// TSDF truncation distance. Defaults to 3x voxel size when omitted.
    #[arg(long)]
    truncation_distance: Option<f32>,

    /// Cubic TSDF volume size in scene units. Defaults to 2.5x scene extent.
    #[arg(long)]
    volume_size: Option<f32>,

    /// Override volume center as x,y,z. Defaults to the Gaussian centroid.
    #[arg(long, value_parser = parse_vec3)]
    volume_center: Option<Vec3>,

    /// Minimum connected triangle component size to keep.
    #[arg(long, default_value = "100")]
    min_cluster_size: usize,

    /// Number of largest connected components to keep (0 keeps all that pass min size).
    #[arg(long, default_value = "1")]
    num_largest_clusters: usize,

    /// Disable normal smoothing in the extracted mesh.
    #[arg(long, default_value_t = false)]
    no_smooth_normals: bool,

    /// Normal smoothing iterations.
    #[arg(long, default_value = "3")]
    normal_smoothing_iterations: usize,

    /// Also export through RustMesh IO as rustmesh.obj/rustmesh.ply.
    #[arg(long, default_value_t = false)]
    export_rustmesh_outputs: bool,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,
}

pub fn run_mesh_from_gs(args: MeshFromGsArgs) -> anyhow::Result<()> {
    let _ = env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .try_init();
    validate_args(&args)?;

    let (splats, metadata) = rustgs::load_splats(&args.scene)
        .with_context(|| format!("failed to load RustGS scene {}", args.scene.display()))?;
    if splats.is_empty() {
        bail!("scene {} contains no Gaussians", args.scene.display());
    }

    let (dataset, source) = rustgs::load_colmap_training_dataset_with_source(
        &args.input,
        &rustgs::ColmapConfig {
            max_frames: 0,
            frame_stride: 1,
            ..Default::default()
        },
    )
    .with_context(|| format!("failed to load training input {}", args.input.display()))?;
    if dataset.poses.is_empty() {
        bail!(
            "input {} resolved as {source} but contains no poses",
            args.input.display()
        );
    }

    let (scene_center, scene_extent) = scene_center_and_extent(&splats);
    let volume_center = args.volume_center.unwrap_or(scene_center);
    let volume_size = args
        .volume_size
        .unwrap_or_else(|| (scene_extent * 2.5).max(args.voxel_size * 32.0));
    let half = volume_size * 0.5;
    let truncation_distance = args
        .truncation_distance
        .unwrap_or(args.voxel_size * 3.0)
        .max(args.voxel_size);

    let config = MeshExtractionConfig {
        tsdf_config: TsdfConfig {
            voxel_size: args.voxel_size,
            sdf_trunc: truncation_distance,
            min_bound: volume_center - Vec3::splat(half),
            max_bound: volume_center + Vec3::splat(half),
            max_weight: 100.0,
            integration_weight: 1.0,
        },
        min_cluster_size: args.min_cluster_size,
        num_largest_clusters: args.num_largest_clusters,
        smooth_normals: !args.no_smooth_normals,
        normal_smoothing_iterations: args.normal_smoothing_iterations,
    };
    let mut extractor = MeshExtractor::new(config);

    let (render_width, render_height) = rustgs::scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        args.render_scale,
    );
    let scaled_intrinsics = scaled_intrinsics(
        dataset.intrinsics,
        render_width as u32,
        render_height as u32,
    );
    let mut renderer = SplatEvaluationRenderer::new(
        render_width,
        render_height,
        EvaluationDevice::Gpu,
        rustgs::DEFAULT_RASTER_COV_BLUR,
    )
    .map_err(anyhow::Error::from)?;
    let selected_poses = selected_poses(&dataset, args.view_stride, args.max_views);

    log::info!(
        "mesh-from-gs | scene={} | source={} | gaussians={} | sh_degree={} | views={} | render={}x{} | voxel_size={} | volume_center=({:.4},{:.4},{:.4}) | volume_size={:.4}",
        args.scene.display(),
        source,
        splats.len(),
        metadata.sh_degree,
        selected_poses.len(),
        render_width,
        render_height,
        args.voxel_size,
        volume_center.x,
        volume_center.y,
        volume_center.z,
        volume_size,
    );

    let mut integrated = 0usize;
    for pose in selected_poses {
        let camera = GaussianCamera::new(scaled_intrinsics, pose.pose.inverse());
        let rendered = renderer
            .render_with_depth(&splats, &camera)
            .with_context(|| format!("failed to render depth for frame {}", pose.frame_id))?;
        let color: Vec<[u8; 3]> = rendered
            .color
            .chunks_exact(3)
            .map(|rgb| {
                [
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect();
        let depth = rendered.depth;
        if depth.iter().all(|value| *value <= 0.0) {
            log::warn!("frame {} rendered no valid depth; skipping", pose.frame_id);
            continue;
        }
        extractor.integrate_frame(
            &depth,
            Some(&color),
            render_width,
            render_height,
            intrinsics_array(scaled_intrinsics),
            &pose_to_mat4(pose.pose),
        );
        integrated += 1;
    }

    if integrated == 0 {
        bail!(
            "no views produced valid depth; try a larger --volume-size, lower --view-stride, or verify the scene/dataset pair"
        );
    }

    let report = extractor.extract_with_postprocessing_report();
    if report.mesh.vertices.is_empty() || report.mesh.triangles.is_empty() {
        bail!(
            "mesh extraction produced an empty mesh after integrating {integrated} views; try a larger --volume-size or coarser --voxel-size"
        );
    }

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let (obj_path, ply_path) = extractor
        .export_mesh_files(&report.mesh, &args.output_dir)
        .with_context(|| {
            format!(
                "failed to export mesh files under {}",
                args.output_dir.display()
            )
        })?;
    let metadata_path = extractor
        .export_mesh_metadata_files(
            &report.mesh,
            report.isolated_triangle_percentage,
            &args.output_dir,
        )
        .with_context(|| {
            format!(
                "failed to export mesh metadata under {}",
                args.output_dir.display()
            )
        })?;

    if args.export_rustmesh_outputs {
        export_with_rustmesh(&report.mesh, &args.output_dir)?;
    }

    println!("mesh_obj={}", obj_path.display());
    println!("mesh_ply={}", ply_path.display());
    println!("mesh_metadata={}", metadata_path.display());
    println!("integrated_views={integrated}");
    println!("vertices={}", report.mesh.vertices.len());
    println!("triangles={}", report.mesh.triangles.len());

    Ok(())
}

fn validate_args(args: &MeshFromGsArgs) -> anyhow::Result<()> {
    ensure_positive("render-scale", args.render_scale)?;
    if args.render_scale > 1.0 {
        bail!("--render-scale must be <= 1");
    }
    if args.view_stride == 0 {
        bail!("--view-stride must be >= 1");
    }
    ensure_positive("voxel-size", args.voxel_size)?;
    if let Some(value) = args.truncation_distance {
        ensure_positive("truncation-distance", value)?;
    }
    if let Some(value) = args.volume_size {
        ensure_positive("volume-size", value)?;
    }
    Ok(())
}

fn ensure_positive(name: &str, value: f32) -> anyhow::Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!("--{name} must be finite and > 0");
    }
    Ok(())
}

fn selected_poses(
    dataset: &TrainingDataset,
    view_stride: usize,
    max_views: usize,
) -> Vec<&ScenePose> {
    let mut poses = dataset
        .poses
        .iter()
        .step_by(view_stride.max(1))
        .collect::<Vec<_>>();
    if max_views > 0 {
        poses.truncate(max_views);
    }
    poses
}

fn scaled_intrinsics(intrinsics: Intrinsics, width: u32, height: u32) -> Intrinsics {
    let sx = width as f32 / intrinsics.width as f32;
    let sy = height as f32 / intrinsics.height as f32;
    Intrinsics::new(
        intrinsics.fx * sx,
        intrinsics.fy * sy,
        intrinsics.cx * sx,
        intrinsics.cy * sy,
        width,
        height,
    )
}

fn intrinsics_array(intrinsics: Intrinsics) -> [f32; 4] {
    [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
}

fn pose_to_mat4(pose: SE3) -> Mat4 {
    Mat4::from_cols_array_2d(&pose.to_matrix())
}

fn scene_center_and_extent(splats: &HostSplats) -> (Vec3, f32) {
    if splats.is_empty() {
        return (Vec3::ZERO, 0.0);
    }

    let mut center = Vec3::ZERO;
    for idx in 0..splats.len() {
        let [x, y, z] = splats.position(idx);
        center += Vec3::new(x, y, z);
    }
    center /= splats.len() as f32;

    let mut extent = 0.0f32;
    for idx in 0..splats.len() {
        let [x, y, z] = splats.position(idx);
        extent = extent.max(Vec3::new(x, y, z).distance(center));
    }

    (center, extent.max(1e-6))
}

fn export_with_rustmesh(mesh: &Mesh, output_dir: &Path) -> anyhow::Result<()> {
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let triangles = mesh
        .triangles
        .iter()
        .map(|triangle| triangle.indices)
        .collect::<Vec<_>>();
    let normals = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.normal)
        .collect::<Vec<_>>();
    let colors = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.color)
        .collect::<Vec<_>>();

    let rustmesh =
        RustMesh::from_triangle_mesh(&vertices, &triangles, Some(&normals), Some(&colors));
    let obj_path = output_dir.join("rustmesh.obj");
    let ply_path = output_dir.join("rustmesh.ply");
    rustmesh::io::write_obj(&rustmesh, &obj_path)
        .with_context(|| format!("failed to write {}", obj_path.display()))?;
    rustmesh::io::write_ply(&rustmesh, &ply_path, rustmesh::io::PlyFormat::Ascii)
        .with_context(|| format!("failed to write {}", ply_path.display()))?;
    Ok(())
}

fn parse_vec3(value: &str) -> Result<Vec3, String> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<f32>()
                .map_err(|err| format!("invalid float '{part}': {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        return Err("expected x,y,z".to_string());
    }
    if parts.iter().any(|value| !value.is_finite()) {
        return Err("all components must be finite".to_string());
    }
    Ok(Vec3::new(parts[0], parts[1], parts[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vec3_accepts_xyz_triplet() {
        assert_eq!(
            parse_vec3("1.0, 2.5, -3").unwrap(),
            Vec3::new(1.0, 2.5, -3.0)
        );
    }

    #[test]
    fn selected_poses_applies_stride_and_limit() {
        let mut dataset = TrainingDataset::new(Intrinsics::from_focal(100.0, 10, 10));
        for idx in 0..6 {
            dataset.add_pose(ScenePose::new(
                idx,
                PathBuf::from(format!("frame_{idx}.png")),
                SE3::identity(),
                idx as f64,
            ));
        }

        let frames = selected_poses(&dataset, 2, 2)
            .into_iter()
            .map(|pose| pose.frame_id)
            .collect::<Vec<_>>();

        assert_eq!(frames, vec![0, 2]);
    }

    #[test]
    fn scaled_intrinsics_updates_principal_point_and_focal() {
        let intrinsics = Intrinsics::new(100.0, 200.0, 50.0, 80.0, 100, 200);
        let scaled = scaled_intrinsics(intrinsics, 50, 100);

        assert_eq!(scaled.width, 50);
        assert_eq!(scaled.height, 100);
        assert!((scaled.fx - 50.0).abs() < 1e-6);
        assert!((scaled.fy - 100.0).abs() < 1e-6);
        assert!((scaled.cx - 25.0).abs() < 1e-6);
        assert!((scaled.cy - 40.0).abs() < 1e-6);
    }
}
