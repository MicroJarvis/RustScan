use anyhow::{bail, Result};
use rustsfm::colmap::ColmapSparseFormat;
use rustsfm::feature_matching::MatchingPairStrategy;
use rustsfm::sift::{SiftExtractionOptions, SiftMatchingOptions};
use std::path::PathBuf;

pub(super) fn sift_matching_from_args(
    match_ratio: f64,
    sift_cpu_brute_force_matcher: bool,
) -> SiftMatchingOptions {
    SiftMatchingOptions {
        max_ratio: match_ratio as f32,
        cpu_brute_force_matcher: sift_cpu_brute_force_matcher,
        ..Default::default()
    }
}

pub(super) fn sift_extraction_from_args(
    max_features: usize,
    sift_estimate_affine_shape: bool,
    sift_domain_size_pooling: bool,
    sift_force_covariant: bool,
) -> SiftExtractionOptions {
    SiftExtractionOptions {
        max_num_features: max_features,
        estimate_affine_shape: sift_estimate_affine_shape,
        domain_size_pooling: sift_domain_size_pooling,
        force_covariant_extractor: sift_force_covariant,
        ..Default::default()
    }
}

pub(super) fn matching_pair_strategy_from_name(
    matching_strategy: &str,
    local_window: usize,
    sequential_overlap: usize,
    sequential_quadratic_overlap: bool,
    sequential_loop_detection: bool,
    sequential_loop_detection_period: usize,
    vocab_tree_num_images: usize,
) -> MatchingPairStrategy {
    match matching_strategy.to_ascii_lowercase().as_str() {
        "exhaustive" => MatchingPairStrategy::Exhaustive,
        "local-window" | "local_window" => MatchingPairStrategy::LocalWindow {
            window: local_window.max(1),
        },
        "vocab-tree" | "vocab_tree" | "vocabtree" => MatchingPairStrategy::VocabTree {
            num_images: vocab_tree_num_images.max(1),
        },
        _ => MatchingPairStrategy::Sequential {
            overlap: sequential_overlap.max(1),
            quadratic_overlap: sequential_quadratic_overlap,
            loop_detection: sequential_loop_detection,
            loop_detection_period: sequential_loop_detection_period.max(1),
        },
    }
}

pub(super) fn parse_sparse_format(value: Option<&str>) -> Result<Option<ColmapSparseFormat>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_ascii_uppercase().as_str() {
        "TXT" | "TEXT" => Ok(Some(ColmapSparseFormat::Text)),
        "BIN" | "BINARY" => Ok(Some(ColmapSparseFormat::Binary)),
        other => bail!("unsupported sparse model format '{other}'; supported: TXT, BIN"),
    }
}

pub(super) fn format_mask_overlap(
    overlap: &rustsfm::two_view::TwoViewMaskOverlapDiagnostics,
) -> String {
    format!(
        "{}/{}@{:.3}",
        overlap.intersection, overlap.union, overlap.jaccard
    )
}

pub(super) fn parity_image_names(
    images: Option<&PathBuf>,
    explicit_names: &[String],
) -> Result<Vec<String>> {
    let mut names = explicit_names.to_vec();
    if let Some(root) = images {
        let mut from_dir = std::fs::read_dir(root)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_string())
            })
            .collect::<Vec<_>>();
        from_dir.sort();
        names.extend(from_dir);
    }
    names.sort();
    names.dedup();
    Ok(names)
}
