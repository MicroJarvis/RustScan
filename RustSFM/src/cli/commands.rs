use super::feature_database::ensure_feature_extractor_database;
use super::project::{colmap_bool, colmap_optional_bool, resolve_colmap_mapper_args};
use super::support::{
    format_mask_overlap, matching_pair_strategy_from_name, parity_image_names, parse_sparse_format,
    sift_extraction_from_args, sift_matching_from_args,
};
use super::*;
use anyhow::{bail, Context, Result};
use clap::Parser;
use rustsfm::colmap::{read_colmap_sparse_files, write_colmap_sparse_model};
use rustsfm::feature_matching::MatchingPairStrategy;
use rustsfm::sift::SiftMatchingOptions;
use rustsfm::{
    benchmark_sift_extraction, compare_colmap_stages, compare_database_parity,
    compare_extracted_sift_features, debug_two_view_database_pair, extract_features_to_database,
    match_features_to_database, parse_compare_stages, run_reconstruction, DebugTwoViewOptions,
    MapperConfig, MatchFeaturesOptions,
};

fn run_reconstruct(args: ReconstructArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let matching_pair_strategy = matching_pair_strategy_from_name(
        &args.matching_strategy,
        args.local_window,
        args.sequential_overlap,
        args.sequential_quadratic_overlap,
        args.sequential_loop_detection,
        args.sequential_loop_detection_period,
        args.vocab_tree_num_images,
    );
    let config = MapperConfig {
        input: args.input,
        output: args.output,
        reference: args.reference,
        database: args.database,
        write_two_view_geometries: args.write_two_view_geometries,
        ignore_database_two_view_poses: args.ignore_database_two_view_poses,
        write_database: args.write_database,
        max_images: args.max_images,
        multiple_models: !args.single_model,
        max_num_models: args.max_num_models,
        max_model_overlap: args.max_model_overlap,
        min_model_size: args.min_model_size,
        snapshot_path: args.snapshot_path,
        snapshot_frames_freq: args.snapshot_frames_freq,
        extract_colors: !args.no_extract_colors,
        fix_existing_frames: args.fix_existing_frames,
        feature_type: args.feature_type,
        max_features: args.max_features,
        match_ratio: args.match_ratio,
        max_hamming_distance: args.max_hamming_distance,
        local_matching: args.local_matching,
        local_window: args.local_window,
        matching_pair_strategy,
        sift_matching: SiftMatchingOptions {
            guided_matching: args.guided_matching,
            max_guided_epipolar_error_px: args.guided_max_epipolar_error_px,
            cpu_brute_force_matcher: args.sift_cpu_brute_force_matcher,
            max_ratio: args.match_ratio as f32,
            ..Default::default()
        },
        sift_extraction: sift_extraction_from_args(
            args.max_features,
            args.sift_estimate_affine_shape,
            args.sift_domain_size_pooling,
            args.sift_force_covariant,
        ),
        experimental_sequence_heuristics: args.experimental_sequence_heuristics,
        experimental_ring_closure: args.experimental_ring_closure,
        experimental_structureless_pair_pose_fallback: args
            .experimental_structureless_pair_pose_fallback,
        min_matches: args.min_matches,
        min_inliers: args.min_inliers,
        min_triangulated: args.min_triangulated,
        init_min_num_inliers: args.init_min_num_inliers,
        init_min_tri_angle_deg: args.init_min_tri_angle_deg,
        init_max_forward_motion: args.init_max_forward_motion,
        init_num_trials: args.init_num_trials,
        init_max_reg_trials: args.init_max_reg_trials,
        essential_threshold_px: args.essential_threshold_px,
        essential_iterations: args.essential_iterations,
        pnp_threshold_px: args.pnp_threshold_px,
        pnp_iterations: args.pnp_iterations,
        abs_pose_min_num_inliers: args.abs_pose_min_num_inliers,
        abs_pose_min_inlier_ratio: args.abs_pose_min_inlier_ratio,
        image_selection_method: args.image_selection_method,
        random_seed: args.random_seed,
        max_reg_trials: args.max_reg_trials,
        local_ba: !args.no_local_ba,
        local_ba_num_images: args.local_ba_num_images,
        local_ba_min_shared_points: args.local_ba_min_shared_points,
        local_ba_iterations: args.local_ba_iterations,
        local_ba_max_refinements: args.local_ba_max_refinements,
        local_ba_max_refinement_change: args.local_ba_max_refinement_change,
        global_ba: !args.no_global_ba,
        global_ba_iterations: args.global_ba_iterations,
        global_ba_images_ratio: args.global_ba_images_ratio,
        global_ba_points_ratio: args.global_ba_points_ratio,
        global_ba_images_freq: args.global_ba_images_freq,
        global_ba_points_freq: args.global_ba_points_freq,
        global_ba_max_refinements: args.global_ba_max_refinements,
        global_ba_max_refinement_change: args.global_ba_max_refinement_change,
        global_ba_ignore_redundant_points3d: args.global_ba_ignore_redundant_points3d,
        global_ba_ignore_redundant_points3d_min_coverage_gain: args
            .global_ba_ignore_redundant_points3d_min_coverage_gain,
        ba_refine_focal_length: !args.no_ba_refine_focal_length,
        ba_refine_principal_point: args.ba_refine_principal_point,
        ba_refine_extra_params: !args.no_ba_refine_extra_params,
        ba_constant_rig_ids: args.ba_constant_rig_ids,
        ba_constant_camera_ids: args.ba_constant_camera_ids,
        min_focal_length_ratio: args.min_focal_length_ratio,
        max_focal_length_ratio: args.max_focal_length_ratio,
        max_extra_param: args.max_extra_param,
        max_reprojection_error_px: args.max_reprojection_error_px,
        ignore_two_view_tracks: args.ignore_two_view_tracks,
        global_mapper: args.global_mapper,
        pose_graph: args.pose_graph,
        copy_images: !args.no_copy_images,
        threads: args.threads,
        fx: args.fx,
        fy: args.fy,
        cx: args.cx,
        cy: args.cy,
        ..Default::default()
    };
    let summary = run_reconstruction(&config)?;
    println!(
        "RustSFM completed: images={} registered={} points={} pairs={} elapsed_ms={:.2}",
        summary.images,
        summary.registered_images,
        summary.points,
        summary.pairs,
        summary.elapsed_ms
    );
    if let Some(path) = args.summary_json {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

fn run_match_features(args: MatchFeaturesArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let report = match_features_to_database(
        &args.database,
        &MatchFeaturesOptions {
            pair_strategy: matching_pair_strategy_from_name(
                &args.matching_strategy,
                3,
                args.sequential_overlap,
                args.sequential_quadratic_overlap,
                args.sequential_loop_detection,
                args.sequential_loop_detection_period,
                args.vocab_tree_num_images,
            ),
            sift_matching: sift_matching_from_args(
                args.match_ratio,
                args.sift_cpu_brute_force_matcher,
            ),
            min_num_matches: args.min_num_matches,
            min_inliers: args.min_num_matches,
            essential_threshold_px: args.essential_threshold_px,
            essential_iterations: args.essential_iterations,
            clear_existing: args.clear_existing,
            use_existing_matches: args.use_existing_matches,
            existing_match_batch_size: args.existing_match_batch_size,
            random_seed: args.random_seed,
            ..MatchFeaturesOptions::default()
        },
    )?;
    println!(
        "match-features: pairs={} matched={} verified={} total_matches={} seconds={:.3}",
        report.pair_count,
        report.matched_pairs,
        report.verified_pairs,
        report.total_matches,
        report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_feature_extractor(args: ColmapFeatureExtractorArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let use_gpu = colmap_optional_bool(args.use_gpu, "SiftExtraction.use_gpu")?;
    let single_camera = colmap_bool(args.single_camera, "ImageReader.single_camera")?;
    let estimate_affine_shape = colmap_bool(
        args.estimate_affine_shape,
        "SiftExtraction.estimate_affine_shape",
    )?;
    let domain_size_pooling = colmap_bool(
        args.domain_size_pooling,
        "SiftExtraction.domain_size_pooling",
    )?;
    let mut options = sift_extraction_from_args(
        args.max_num_features,
        estimate_affine_shape,
        domain_size_pooling,
        estimate_affine_shape || domain_size_pooling,
    );
    options.use_gpu = use_gpu.unwrap_or(false);
    if options.use_gpu {
        rustsfm::gpu::validate_gpu_sift_options(&options)?;
    }
    ensure_feature_extractor_database(
        &args.database_path,
        &args.image_path,
        &args.camera_model,
        single_camera,
    )?;
    let report = extract_features_to_database(&args.database_path, &args.image_path, &options)?;
    println!(
        "feature_extractor: database={} images={} total_keypoints={} seconds={:.3}",
        args.database_path.display(),
        report.image_count,
        report.total_keypoints,
        report.extraction_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_matcher(
    database_path: PathBuf,
    strategy: MatchingPairStrategy,
    max_ratio: f64,
    use_gpu: Option<i32>,
    guided_matching: i32,
    min_num_inliers: usize,
    max_error: f32,
    max_num_trials: u32,
    random_seed: i32,
    output_json: Option<PathBuf>,
    log_level: String,
) -> Result<()> {
    let use_gpu = colmap_optional_bool(use_gpu, "SiftMatching.use_gpu")?;
    if use_gpu == Some(true) {
        bail!("SiftMatching.use_gpu is not implemented in RustSFM; use CPU matching");
    }
    let guided_matching = colmap_bool(guided_matching, "SiftMatching.guided_matching")?;
    let args = MatchFeaturesArgs {
        database: database_path,
        matching_strategy: "sequential".to_string(),
        sequential_overlap: 10,
        sequential_quadratic_overlap: true,
        sequential_loop_detection: false,
        sequential_loop_detection_period: 10,
        vocab_tree_num_images: 100,
        match_ratio: max_ratio,
        sift_cpu_brute_force_matcher: false,
        min_num_matches: min_num_inliers,
        essential_threshold_px: max_error,
        essential_iterations: max_num_trials,
        clear_existing: true,
        use_existing_matches: false,
        existing_match_batch_size: 1000,
        random_seed,
        output_json,
        log_level,
    };
    let mut options = MatchFeaturesOptions {
        pair_strategy: strategy,
        sift_matching: SiftMatchingOptions {
            guided_matching,
            max_ratio: max_ratio as f32,
            ..Default::default()
        },
        min_num_matches: min_num_inliers,
        min_inliers: min_num_inliers,
        essential_threshold_px: max_error,
        essential_iterations: max_num_trials,
        random_seed,
        ..MatchFeaturesOptions::default()
    };
    options.clear_existing = args.clear_existing;
    let report = match_features_to_database(&args.database, &options)?;
    println!(
        "matcher: pairs={} matched={} verified={} total_matches={} seconds={:.3}",
        report.pair_count,
        report.matched_pairs,
        report.verified_pairs,
        report.total_matches,
        report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_geometric_verifier(args: ColmapGeometricVerifierArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let guided_matching = colmap_bool(args.guided_matching, "SiftMatching.guided_matching")?;
    let report = match_features_to_database(
        &args.database_path,
        &MatchFeaturesOptions {
            sift_matching: SiftMatchingOptions {
                guided_matching,
                ..Default::default()
            },
            min_num_matches: args.min_num_inliers,
            min_inliers: args.min_num_inliers,
            essential_threshold_px: args.max_error,
            essential_iterations: args.max_num_trials,
            clear_existing: false,
            use_existing_matches: true,
            random_seed: args.random_seed,
            ..MatchFeaturesOptions::default()
        },
    )?;
    println!(
        "geometric_verifier: pairs={} verified={} total_matches={} seconds={:.3}",
        report.pair_count, report.verified_pairs, report.total_matches, report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_mapper(args: ColmapMapperArgs) -> Result<()> {
    let resolved = resolve_colmap_mapper_args(&args)?;
    let ba_refine_focal_length = colmap_bool(
        resolved.ba_refine_focal_length,
        "Mapper.ba_refine_focal_length",
    )?;
    let ba_refine_principal_point = colmap_bool(
        resolved.ba_refine_principal_point,
        "Mapper.ba_refine_principal_point",
    )?;
    let ba_refine_extra_params = colmap_bool(
        resolved.ba_refine_extra_params,
        "Mapper.ba_refine_extra_params",
    )?;
    let multiple_models = colmap_bool(resolved.multiple_models, "Mapper.multiple_models")?;
    let extract_colors = colmap_bool(resolved.extract_colors, "Mapper.extract_colors")?;
    let fix_existing_frames =
        colmap_bool(resolved.fix_existing_frames, "Mapper.fix_existing_frames")?;
    let global_ba_ignore_redundant_points3d = colmap_bool(
        resolved.global_ba_ignore_redundant_points3d,
        "Mapper.ba_global_ignore_redundant_points3D",
    )?;
    let ignore_two_view_tracks = colmap_bool(
        resolved.tri_ignore_two_view_tracks,
        "Mapper.tri_ignore_two_view_tracks",
    )?;
    let reconstruct_args = ReconstructArgs {
        input: resolved.image_path,
        output: resolved.output_path,
        reference: None,
        database: Some(resolved.database_path),
        write_two_view_geometries: false,
        ignore_database_two_view_poses: false,
        write_database: false,
        max_images: None,
        single_model: !multiple_models,
        max_num_models: resolved.max_num_models,
        max_model_overlap: resolved.max_model_overlap,
        min_model_size: resolved.min_model_size,
        snapshot_path: resolved.snapshot_path,
        snapshot_frames_freq: resolved.snapshot_frames_freq,
        no_extract_colors: !extract_colors,
        fix_existing_frames,
        feature_type: FeatureType::Sift,
        max_features: 8192,
        sift_estimate_affine_shape: false,
        sift_domain_size_pooling: false,
        sift_force_covariant: false,
        sift_cpu_brute_force_matcher: false,
        local_matching: false,
        local_window: 3,
        matching_strategy: "sequential".to_string(),
        sequential_overlap: 10,
        sequential_quadratic_overlap: true,
        sequential_loop_detection: false,
        sequential_loop_detection_period: 10,
        vocab_tree_num_images: 100,
        guided_matching: false,
        guided_max_epipolar_error_px: 2.0,
        experimental_sequence_heuristics: false,
        experimental_ring_closure: false,
        experimental_structureless_pair_pose_fallback: false,
        match_ratio: 0.8,
        max_hamming_distance: 160.0,
        min_matches: resolved.min_num_matches,
        min_inliers: resolved.min_num_matches,
        min_triangulated: 4,
        init_min_num_inliers: resolved.init_min_num_inliers,
        init_min_tri_angle_deg: resolved.init_min_tri_angle,
        init_max_forward_motion: resolved.init_max_forward_motion,
        init_num_trials: resolved.init_num_trials,
        init_max_reg_trials: resolved.init_max_reg_trials,
        essential_threshold_px: resolved.init_max_error,
        essential_iterations: 10_000,
        pnp_threshold_px: resolved.abs_pose_max_error,
        pnp_iterations: 10_000,
        abs_pose_min_num_inliers: resolved.abs_pose_min_num_inliers,
        abs_pose_min_inlier_ratio: resolved.abs_pose_min_inlier_ratio,
        image_selection_method: ImageSelectionMethod::MinUncertainty,
        random_seed: resolved.random_seed,
        max_reg_trials: resolved.max_reg_trials,
        no_local_ba: false,
        local_ba_num_images: resolved.local_ba_num_images,
        local_ba_min_shared_points: 15,
        local_ba_iterations: resolved.local_ba_iterations,
        local_ba_max_refinements: resolved.local_ba_max_refinements,
        local_ba_max_refinement_change: resolved.local_ba_max_refinement_change,
        no_global_ba: false,
        global_ba_iterations: resolved.global_ba_iterations,
        global_ba_images_ratio: resolved.global_ba_images_ratio,
        global_ba_points_ratio: resolved.global_ba_points_ratio,
        global_ba_images_freq: resolved.global_ba_images_freq,
        global_ba_points_freq: resolved.global_ba_points_freq,
        global_ba_max_refinements: resolved.global_ba_max_refinements,
        global_ba_max_refinement_change: resolved.global_ba_max_refinement_change,
        global_ba_ignore_redundant_points3d,
        global_ba_ignore_redundant_points3d_min_coverage_gain: resolved
            .global_ba_ignore_redundant_points3d_min_coverage_gain,
        no_ba_refine_focal_length: !ba_refine_focal_length,
        ba_refine_principal_point: ba_refine_principal_point,
        no_ba_refine_extra_params: !ba_refine_extra_params,
        ba_constant_rig_ids: Vec::new(),
        ba_constant_camera_ids: Vec::new(),
        min_focal_length_ratio: resolved.min_focal_length_ratio,
        max_focal_length_ratio: resolved.max_focal_length_ratio,
        max_extra_param: resolved.max_extra_param,
        max_reprojection_error_px: resolved.filter_max_reproj_error,
        ignore_two_view_tracks,
        pose_graph: false,
        global_mapper: false,
        fx: None,
        fy: None,
        cx: None,
        cy: None,
        threads: resolved.num_threads,
        no_copy_images: false,
        summary_json: args.summary_json,
        log_level: args.log_level,
    };
    run_reconstruct(reconstruct_args)
}

fn run_model_converter(args: ModelConverterArgs) -> Result<()> {
    let input_format = parse_sparse_format(args.input_type.as_deref())?;
    let output_format = parse_sparse_format(Some(&args.output_type))?;
    let sparse = if let Some(format) = input_format {
        rustsfm::colmap::read_colmap_sparse_files_with_format(&args.input_path, format)?
    } else {
        read_colmap_sparse_files(&args.input_path)?
    };
    write_colmap_sparse_model(
        &args.output_path,
        &sparse,
        output_format.with_context(|| {
            format!(
                "unsupported model_converter output_type '{}'; supported: TXT, BIN",
                args.output_type
            )
        })?,
    )?;
    println!(
        "model_converter: cameras={} images={} points3D={} output={}",
        sparse.cameras.len(),
        sparse.images.len(),
        sparse.points3d.len(),
        args.output_path.display()
    );
    Ok(())
}

pub(super) fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Reconstruct(args) => run_reconstruct(args)?,
        Commands::Compare(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let stages = parse_compare_stages(&args.stages)?;
            let report = compare_colmap_stages(
                &args.reference,
                &args.candidate,
                args.reference_database.as_deref(),
                args.candidate_database.as_deref(),
                &stages,
            )?;
            if let Some(features) = &report.features {
                println!(
                    "features: common_images={} ref_kpts={} cand_kpts={} max_diff={} pct_exact={:.3}",
                    features.common_images,
                    features.reference_total_keypoints,
                    features.candidate_total_keypoints,
                    features.keypoint_count_diff.max_abs_diff,
                    features.keypoint_count_diff.pct_exact
                );
            }
            if let Some(matches) = &report.matches {
                println!(
                    "matches: common_pairs={} ref_matches={} cand_matches={} max_diff={} pct_exact={:.3}",
                    matches.common_pairs,
                    matches.reference_total_matches,
                    matches.candidate_total_matches,
                    matches.common_pair_match_count_diff.max_abs_diff,
                    matches.common_pair_match_count_diff.pct_exact
                );
            }
            if let Some(twoview) = &report.twoview {
                println!(
                    "twoview: common_pairs={} config_agreement={:.3} max_inlier_diff={} inlier_overlap_mean={:.3} mismatches={}",
                    twoview.common_pairs,
                    twoview.config_agreement_rate,
                    twoview.inlier_count_diff.max_abs_diff,
                    twoview.inlier_set_overlap.mean_rate,
                    twoview.config_mismatches.len()
                );
            }
            if let Some(registration) = &report.registration {
                println!(
                    "registration: ref={} cand={} common={} ref_only={} cand_only={}",
                    registration.reference_registered,
                    registration.candidate_registered,
                    registration.common_registered,
                    registration.reference_only.len(),
                    registration.candidate_only.len()
                );
            }
            if let Some(tracks) = &report.tracks {
                println!(
                    "tracks: ref_points={} cand_points={} ref_len2={} cand_len2={} ref_len3_4={} cand_len3_4={}",
                    tracks.reference_points,
                    tracks.candidate_points,
                    tracks.reference_track_length_histogram.length_2,
                    tracks.candidate_track_length_histogram.length_2,
                    tracks.reference_track_length_histogram.length_3_4,
                    tracks.candidate_track_length_histogram.length_3_4
                );
            }
            if let Some(ba) = &report.ba {
                println!(
                    "ba: common={} translation_rmse={:.6} rotation_mean_deg={:.6} rotation_rmse_deg={:.6} rotation_max_deg={:.6} rel_rot_mean_deg={:.6} scale={:.6}",
                    ba.common_images,
                    ba.translation_error.rmse,
                    ba.rotation_error_deg.mean,
                    ba.rotation_error_deg.rmse,
                    ba.rotation_error_deg.max,
                    ba.adjacent_relative_rotation_error_deg.mean,
                    ba.similarity_scale
                );
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::Parity(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let image_names = parity_image_names(args.images.as_ref(), &args.image_names)?;
            let report = compare_database_parity(
                &args.database,
                image_names,
                args.min_matches,
                args.ignore_watermarks,
                args.load_all_images,
                args.convert_pose_priors_to_enu,
            )?;
            println!(
                "COLMAP database parity: raw_images={} cache_images={} raw_frames={} cache_frames={} raw_frame_data={} cache_frame_data={} raw_pairs={} cache_pairs={} bridge_pairs={} bridge_matches={} differences={}",
                report.raw.images,
                report.cache.images,
                report.raw.frames,
                report.cache.frames,
                report.raw.frame_data,
                report.cache.frame_data,
                report.raw.two_view_pairs,
                report.cache.two_view_pairs,
                report.bridge.frame_pairs,
                report.bridge.matches,
                report.differences.len()
            );
            if !report.differences.is_empty() {
                for diff in &report.differences {
                    println!("difference {}: {}", diff.kind, diff.detail);
                }
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::BenchmarkSift(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let mut options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            options.use_gpu = args.use_gpu;
            let report = benchmark_sift_extraction(&args.input, &options)?;
            println!(
                "SIFT benchmark: backend={} images={} total_features={} mean_features={:.1} seconds={:.3} covariant={}",
                report.backend,
                report.image_count,
                report.total_features,
                report.mean_features,
                report.extraction_seconds,
                report.uses_covariant_extractor
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::CompareExtract(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            let report =
                compare_extracted_sift_features(&args.reference_database, &args.images, &options)?;
            println!(
                "SIFT extract compare: common_images={} ref_kpts={} cand_kpts={} max_diff={} pct_exact={:.3} mean_abs_diff={:.1}",
                report.common_images,
                report.reference_total_keypoints,
                report.candidate_total_keypoints,
                report.keypoint_count_diff.max_abs_diff,
                report.keypoint_count_diff.pct_exact,
                report.keypoint_count_diff.mean_abs_diff
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::ExtractFeatures(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let mut options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            options.use_gpu = args.use_gpu;
            let report = extract_features_to_database(&args.database, &args.images, &options)?;
            println!(
                "extract-features: backend={} images={} total_keypoints={} mean_keypoints={:.1} seconds={:.3}",
                report.backend,
                report.image_count,
                report.total_keypoints,
                report.mean_keypoints,
                report.extraction_seconds
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::MatchFeatures(args) => run_match_features(args)?,
        Commands::DebugTwoview(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let report = debug_two_view_database_pair(
                &args.database,
                &args.image1,
                &args.image2,
                &DebugTwoViewOptions {
                    essential_threshold_px: args.essential_threshold_px,
                    essential_iterations: args.essential_iterations,
                    min_inliers: args.min_inliers,
                    min_triangulated: args.min_triangulated,
                    random_seed: args.random_seed,
                },
            )?;
            if let Some(diagnostics) = &report.diagnostics {
                let f_inliers = diagnostics
                    .fundamental
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let h_inliers = diagnostics
                    .homography
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let ef_overlap = diagnostics
                    .e_f_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                let eh_overlap = diagnostics
                    .e_h_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                let fh_overlap = diagnostics
                    .f_h_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                println!(
                    "debug-twoview: pair={} / {} indices=({}, {}) seed={} matches={} active={} E={} F={} H={} ratios(E/F={:.3},H/F={:.3},H/E={:.3}) overlaps(EF={},EH={},FH={}) selected={:?} config={:?} inliers={} estimate_config={:?} estimate_inliers={:?} estimate_triangulated={:?} estimate_tri_angle={:?} estimate_rot={:?} estimate_error={:?}",
                    report.left_image,
                    report.right_image,
                    report.left_index,
                    report.right_index,
                    report.sampler_seed,
                    report.num_matches,
                    diagnostics.active_observations,
                    diagnostics.essential.inliers,
                    f_inliers,
                    h_inliers,
                    diagnostics.e_f_inlier_ratio,
                    diagnostics.h_f_inlier_ratio,
                    diagnostics.h_e_inlier_ratio,
                    ef_overlap,
                    eh_overlap,
                    fh_overlap,
                    diagnostics.selected_source,
                    diagnostics.classified_config,
                    diagnostics.selected_inliers,
                    report.estimate_config,
                    report.estimate_inliers,
                    report.estimate_triangulated,
                    report.estimate_median_triangulation_angle_deg,
                    report.estimate_rotation_deg,
                    report.estimate_mean_reprojection_error_px
                );
            } else {
                println!(
                    "debug-twoview: pair={} / {} matches={} no estimate",
                    report.left_image, report.right_image, report.num_matches
                );
            }
            if let Some(stored_models) = &report.stored_models {
                let stored_e = stored_models
                    .essential
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let stored_f = stored_models
                    .fundamental
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let stored_h = stored_models
                    .homography
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                println!(
                    "debug-twoview stored-models: E={} F={} H={}",
                    stored_e, stored_f, stored_h
                );
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::FeatureExtractor(args) => run_colmap_feature_extractor(args)?,
        Commands::ExhaustiveMatcher(args) => run_colmap_matcher(
            args.database_path,
            MatchingPairStrategy::Exhaustive,
            args.max_ratio,
            args.use_gpu,
            args.guided_matching,
            args.min_num_inliers,
            args.max_error,
            args.max_num_trials,
            args.random_seed,
            args.output_json,
            args.log_level,
        )?,
        Commands::SequentialMatcher(args) => {
            let quadratic_overlap = colmap_bool(
                args.quadratic_overlap,
                "SequentialMatching.quadratic_overlap",
            )?;
            let loop_detection =
                colmap_bool(args.loop_detection, "SequentialMatching.loop_detection")?;
            run_colmap_matcher(
                args.database_path,
                MatchingPairStrategy::Sequential {
                    overlap: args.overlap.max(1),
                    quadratic_overlap,
                    loop_detection,
                    loop_detection_period: args.loop_detection_period.max(1),
                },
                args.max_ratio,
                args.use_gpu,
                args.guided_matching,
                args.min_num_inliers,
                args.max_error,
                args.max_num_trials,
                args.random_seed,
                args.output_json,
                args.log_level,
            )?
        }
        Commands::VocabTreeMatcher(args) => run_colmap_matcher(
            args.database_path,
            MatchingPairStrategy::VocabTree {
                num_images: args.num_images.max(1),
            },
            args.max_ratio,
            args.use_gpu,
            args.guided_matching,
            args.min_num_inliers,
            args.max_error,
            args.max_num_trials,
            args.random_seed,
            args.output_json,
            args.log_level,
        )?,
        Commands::GeometricVerifier(args) => run_colmap_geometric_verifier(args)?,
        Commands::Mapper(args) => run_colmap_mapper(args)?,
        Commands::ModelConverter(args) => run_model_converter(args)?,
    }
    Ok(())
}
