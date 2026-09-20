//! RustScan CLI entrypoint.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use log::{debug, error, info, warn};
use rustscan_types::MapPointData;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sysinfo::{get_current_pid, System};
use thiserror::Error;

use crate::config::SlamConfig;
use crate::core::{Camera, Frame, KeyFrame, Map, SE3};
use crate::io::{
    video_decoder as video, Dataset, DatasetConfig, EurocDataset, KittiDataset, TumRgbdDataset,
};
use crate::mapping::{LocalMapping, LocalMappingConfig};
use crate::tracker::VisualOdometry;
use pipeline_checkpoint::{
    load_pipeline_checkpoint, save_pipeline_checkpoint, slam_frames_dir, CameraCheckpoint,
    KeyframeCheckpoint, MapPointCheckpoint, PipelineCheckpoint, PipelineCheckpointError,
    PoseCheckpoint, SlamCheckpoint,
};

#[cfg(test)]
mod integration_tests;
mod pipeline_checkpoint;
mod slam_pipeline;

const MIB_BYTES: f64 = 1024.0 * 1024.0;

/// RustScan command-line arguments.
#[derive(Parser, Debug)]
#[command(name = "rustscan", version, about = "RustScan CLI")]
struct CliArgs {
    /// Input video file path or TUM dataset directory.
    #[arg(long, value_name = "FILE_OR_DIR")]
    input: Option<PathBuf>,
    /// Output directory path.
    #[arg(long, value_name = "DIR")]
    output: Option<PathBuf>,
    /// Path to TOML configuration file.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Output format for results.
    #[arg(long, value_enum)]
    output_format: Option<OutputFormat>,
    /// Log verbosity level.
    #[arg(long, value_enum)]
    log_level: Option<LogLevel>,
    /// Log format (text/json).
    #[arg(long, value_enum)]
    log_format: Option<LogFormat>,
    /// Video frame cache capacity (number of frames).
    #[arg(long, value_name = "N")]
    video_cache_capacity: Option<usize>,
    /// Prefer hardware-accelerated decoding (true/false).
    #[arg(long, value_name = "BOOL")]
    prefer_hardware: Option<bool>,
    /// Maximum number of frames to process (0 = no limit).
    #[arg(long, value_name = "N")]
    max_frames: Option<usize>,
    /// Process every Nth frame (stride).
    #[arg(long, value_name = "N")]
    frame_stride: Option<usize>,
    /// Mesh voxel size in meters (TSDF resolution).
    #[arg(long, value_name = "METERS")]
    mesh_voxel_size: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum OutputFormat {
    Json,
    Text,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Json
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum LogFormat {
    Text,
    Json,
}

impl Default for LogFormat {
    fn default() -> Self {
        Self::Text
    }
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// RustScan configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RustScanConfig {
    /// Input video file path.
    input: Option<PathBuf>,
    /// Output directory path.
    output: PathBuf,
    /// Output format for results.
    output_format: OutputFormat,
    /// Log verbosity.
    log_level: Option<LogLevel>,
    /// Log format (text/json).
    log_format: LogFormat,
    /// Non-interactive execution mode.
    non_interactive: bool,
    /// SLAM configuration block.
    slam: SlamConfig,
    /// Video decoding configuration.
    video: VideoConfig,
    /// Optional mesh voxel size override.
    mesh_voxel_size: Option<f32>,
}

impl Default for RustScanConfig {
    fn default() -> Self {
        Self {
            input: None,
            output: PathBuf::from("./output"),
            output_format: OutputFormat::Json,
            log_level: None,
            log_format: LogFormat::Text,
            non_interactive: true,
            slam: SlamConfig::default(),
            video: VideoConfig::default(),
            mesh_voxel_size: None,
        }
    }
}

struct ResolvedConfig {
    input: PathBuf,
    output: PathBuf,
    output_format: OutputFormat,
    slam: SlamConfig,
    video: VideoConfig,
    mesh_voxel_size: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct VideoConfig {
    cache_capacity: usize,
    prefer_hardware: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            cache_capacity: 100,
            prefer_hardware: true,
        }
    }
}

#[derive(Error, Debug)]
enum CliError {
    #[error("input path is required but was not provided")]
    InputNotProvided,
    #[error("input file not found: {0}")]
    InputMissing(PathBuf),
    #[error("input path is not a file: {0}")]
    InputNotFile(PathBuf),
    #[error("failed to read config file {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("output path exists but is not a directory: {0}")]
    OutputNotDirectory(PathBuf),
    #[error("failed to create output directory {path}: {source}")]
    OutputCreate {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write results to {path}: {source}")]
    OutputWrite {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("pipeline failed: {0}")]
    Pipeline(String),
}

impl CliError {
    fn exit_code(&self) -> ExitCode {
        match self {
            CliError::InputNotProvided
            | CliError::InputMissing(_)
            | CliError::InputNotFile(_)
            | CliError::ConfigRead { .. }
            | CliError::ConfigParse { .. }
            | CliError::OutputNotDirectory(_) => ExitCode::from(1),
            CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. }
            | CliError::Pipeline(_) => ExitCode::from(2),
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            CliError::InputNotProvided | CliError::InputMissing(_) | CliError::InputNotFile(_) => {
                "InputError"
            }
            CliError::ConfigRead { .. } | CliError::ConfigParse { .. } => "ConfigError",
            CliError::OutputNotDirectory(_)
            | CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. } => "OutputError",
            CliError::Pipeline(_) => "PipelineError",
        }
    }

    fn component(&self) -> &'static str {
        match self {
            CliError::InputNotProvided | CliError::InputMissing(_) | CliError::InputNotFile(_) => {
                "cli"
            }
            CliError::ConfigRead { .. } | CliError::ConfigParse { .. } => "config",
            CliError::OutputNotDirectory(_)
            | CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. } => "io",
            CliError::Pipeline(_) => "pipeline",
        }
    }

    fn suggestion(&self) -> &'static str {
        match self {
            CliError::InputNotProvided => {
                "Pass --input <video-file-or-dataset-dir> or set input in the TOML config."
            }
            CliError::InputMissing(_) => "Verify the input path and ensure the file exists.",
            CliError::InputNotFile(_) => "Provide a valid video file path (not a directory).",
            CliError::ConfigRead { .. } => "Verify the config path and file permissions.",
            CliError::ConfigParse { .. } => "Fix TOML syntax and ensure fields match the schema.",
            CliError::OutputNotDirectory(_) => "Choose an output path that is a directory.",
            CliError::OutputCreate { .. } => {
                "Check write permissions or select a different output directory."
            }
            CliError::OutputWrite { .. } => {
                "Ensure the output directory is writable and has free space."
            }
            CliError::Pipeline(_) => {
                "Run with RUST_LOG=debug for diagnostics and verify dependencies (OpenCV, ffmpeg)."
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct MeshStats {
    vertex_count: usize,
    triangle_count: usize,
    isolated_triangle_percent: f32,
}

impl Default for MeshStats {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            triangle_count: 0,
            isolated_triangle_percent: 0.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorInfo {
    error_type: String,
    root_cause: String,
    component: String,
    suggestion: String,
}

#[derive(Debug, Serialize)]
struct Diagnostics {
    os: String,
    arch: String,
    rustscan_version: String,
    cwd: String,
    input: Option<String>,
    output: Option<String>,
    config: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResultsJson {
    status: String,
    input: Option<String>,
    output: Option<String>,
    processing_time_ms: u128,
    camera_count: usize,
    slam_output: Option<String>,
    mesh: MeshStats,
    error: Option<ErrorInfo>,
    diagnostics: Diagnostics,
}

struct PipelineReport {
    camera_count: usize,
    slam_output_path: PathBuf,
    mesh: MeshStats,
}

struct DecodedVideo {
    decoder: video::VideoDecoder,
    camera: Camera,
    total_frames: usize,
}

enum InputSource {
    Video(DecodedVideo),
    Dataset(Box<dyn Dataset>),
}

struct KeyframeSample {
    index: usize,
    timestamp: f64,
    width: u32,
    height: u32,
    color: Vec<u8>,
    depth: Option<Vec<f32>>,
    pose: SE3,
}

struct SlamStageOutput {
    keyframes: Vec<KeyframeSample>,
    camera: Camera,
    frame_count: usize,
    map_points: Vec<MapPointData>,
}

pub fn run() -> ExitCode {
    let start = Instant::now();
    let cli = CliArgs::parse();
    let config_path = cli.config.clone();

    let config = match load_config(&cli) {
        Ok(config) => config,
        Err(err) => {
            return handle_error(
                &err,
                start,
                None,
                cli.output_format,
                cli.output,
                config_path,
            );
        }
    };

    let log_level = resolve_log_level(&cli, &config);
    let log_format = resolve_log_format(&cli, &config);
    init_logger(&log_level, log_format);

    let mut merged = config;
    let overrides = apply_overrides(&mut merged, &cli);
    for override_entry in overrides {
        debug!("CLI override: {}", override_entry);
    }

    if !merged.non_interactive {
        warn!("non_interactive=false in config; forcing non-interactive mode");
        merged.non_interactive = true;
    }

    let resolved = match finalize_config(merged) {
        Ok(resolved) => resolved,
        Err(err) => {
            return handle_error(
                &err,
                start,
                None,
                cli.output_format,
                cli.output,
                config_path,
            );
        }
    };

    let output_dir = match ensure_output_dir(&resolved.output) {
        Ok(dir) => dir,
        Err(err) => {
            return handle_error(
                &err,
                start,
                Some(&resolved),
                Some(resolved.output_format),
                Some(resolved.output.clone()),
                config_path,
            );
        }
    };

    info!("Starting RustScan SLAM pipeline");
    info!("Input: {}", resolved.input.display());
    info!("Output: {}", output_dir.display());
    debug!(
        "Loaded SLAM config camera: {}x{}",
        resolved.slam.camera.width, resolved.slam.camera.height
    );
    if let Some(voxel_size) = resolved.mesh_voxel_size {
        warn!(
            "mesh_voxel_size={} is ignored because RustSLAM now runs in SLAM-only mode",
            voxel_size
        );
    }

    let pipeline_report = match run_pipeline(&resolved) {
        Ok(report) => report,
        Err(err) => {
            return handle_error(
                &err,
                start,
                Some(&resolved),
                Some(resolved.output_format),
                Some(resolved.output.clone()),
                config_path,
            );
        }
    };

    let results = ResultsJson {
        status: "success".to_string(),
        input: Some(resolved.input.display().to_string()),
        output: Some(output_dir.display().to_string()),
        processing_time_ms: start.elapsed().as_millis(),
        camera_count: pipeline_report.camera_count,
        slam_output: Some(pipeline_report.slam_output_path.display().to_string()),
        mesh: pipeline_report.mesh,
        error: None,
        diagnostics: build_diagnostics(
            Some(&resolved.input),
            Some(&output_dir),
            config_path.as_deref(),
        ),
    };

    if let Err(err) = write_results(&results, &output_dir, resolved.output_format) {
        return handle_error(
            &err,
            start,
            Some(&resolved),
            Some(resolved.output_format),
            Some(resolved.output.clone()),
            config_path,
        );
    }

    if resolved.output_format == OutputFormat::Text {
        print_text_summary(&results);
    }

    ExitCode::SUCCESS
}

fn load_config(cli: &CliArgs) -> Result<RustScanConfig, CliError> {
    if let Some(path) = &cli.config {
        let content = fs::read_to_string(path).map_err(|source| CliError::ConfigRead {
            path: path.clone(),
            source,
        })?;
        let config: RustScanConfig =
            toml::from_str(&content).map_err(|source| CliError::ConfigParse {
                path: path.clone(),
                source,
            })?;
        Ok(config)
    } else {
        Ok(RustScanConfig::default())
    }
}

fn resolve_log_level(cli: &CliArgs, config: &RustScanConfig) -> String {
    if let Some(level) = cli.log_level {
        return level.as_str().to_string();
    }

    if let Ok(level) = std::env::var("RUST_LOG") {
        if !level.trim().is_empty() {
            return level;
        }
    }

    if let Some(level) = config.log_level {
        return level.as_str().to_string();
    }

    "info".to_string()
}

fn resolve_log_format(cli: &CliArgs, config: &RustScanConfig) -> LogFormat {
    if let Some(format) = cli.log_format {
        return format;
    }
    config.log_format
}

fn init_logger(level: &str, format: LogFormat) {
    let mut builder = env_logger::Builder::new();
    builder.target(env_logger::Target::Stderr);
    builder.filter_level(log::LevelFilter::Info);
    builder.parse_filters(level);
    builder.format(move |buf, record| {
        use std::io::Write;
        let module = record.module_path().unwrap_or(record.target());
        match format {
            LogFormat::Json => {
                let payload = json!({
                    "timestamp": buf.timestamp_millis().to_string(),
                    "level": record.level().to_string(),
                    "target": module,
                    "message": record.args().to_string(),
                });
                writeln!(buf, "{}", payload)
            }
            LogFormat::Text => {
                writeln!(
                    buf,
                    "{} [{}] {}: {}",
                    buf.timestamp_millis(),
                    record.level(),
                    module,
                    record.args()
                )
            }
        }
    });

    if let Err(err) = builder.try_init() {
        eprintln!("Failed to initialize logger: {}", err);
    }
}

fn apply_overrides(config: &mut RustScanConfig, cli: &CliArgs) -> Vec<String> {
    let mut overrides = Vec::new();

    if let Some(input) = cli.input.clone() {
        if config.input.as_ref() != Some(&input) {
            overrides.push(format!("input = {}", input.display()));
        }
        config.input = Some(input);
    }

    if let Some(output) = cli.output.clone() {
        if config.output != output {
            overrides.push(format!("output = {}", output.display()));
        }
        config.output = output;
    }

    if let Some(format) = cli.output_format {
        if config.output_format != format {
            overrides.push(format!("output_format = {:?}", format));
        }
        config.output_format = format;
    }

    if let Some(level) = cli.log_level {
        if config.log_level != Some(level) {
            overrides.push(format!("log_level = {:?}", level));
        }
        config.log_level = Some(level);
    }

    if let Some(format) = cli.log_format {
        if config.log_format != format {
            overrides.push(format!("log_format = {:?}", format));
        }
        config.log_format = format;
    }

    if let Some(capacity) = cli.video_cache_capacity {
        if config.video.cache_capacity != capacity {
            overrides.push(format!("video.cache_capacity = {}", capacity));
        }
        config.video.cache_capacity = capacity;
    }

    if let Some(prefer_hardware) = cli.prefer_hardware {
        if config.video.prefer_hardware != prefer_hardware {
            overrides.push(format!("video.prefer_hardware = {}", prefer_hardware));
        }
        config.video.prefer_hardware = prefer_hardware;
    }

    if let Some(max_frames) = cli.max_frames {
        if config.slam.dataset.max_frames != max_frames {
            overrides.push(format!("slam.dataset.max_frames = {}", max_frames));
        }
        config.slam.dataset.max_frames = max_frames;
    }

    if let Some(stride) = cli.frame_stride {
        let stride = stride.max(1);
        if config.slam.dataset.stride != stride {
            overrides.push(format!("slam.dataset.stride = {}", stride));
        }
        config.slam.dataset.stride = stride;
    }

    if let Some(voxel_size) = cli.mesh_voxel_size {
        if config.mesh_voxel_size != Some(voxel_size) {
            overrides.push(format!("mesh_voxel_size = {}", voxel_size));
        }
        config.mesh_voxel_size = Some(voxel_size);
    }

    overrides
}

fn finalize_config(config: RustScanConfig) -> Result<ResolvedConfig, CliError> {
    let input = match config.input {
        Some(path) => path,
        None => return Err(CliError::InputNotProvided),
    };

    if !input.exists() {
        return Err(CliError::InputMissing(input));
    }

    let metadata = fs::metadata(&input).map_err(|_| CliError::InputMissing(input.clone()))?;
    // Accept both files (videos) and directories (datasets like TUM RGB-D)
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(CliError::InputNotFile(input));
    }

    let output = config.output;

    Ok(ResolvedConfig {
        input,
        output,
        output_format: config.output_format,
        slam: config.slam,
        video: config.video,
        mesh_voxel_size: config.mesh_voxel_size,
    })
}

fn ensure_output_dir(output: &Path) -> Result<PathBuf, CliError> {
    if output.exists() {
        if output.is_dir() {
            return Ok(output.to_path_buf());
        }
        return Err(CliError::OutputNotDirectory(output.to_path_buf()));
    }

    fs::create_dir_all(output).map_err(|source| CliError::OutputCreate {
        path: output.to_path_buf(),
        source,
    })?;

    Ok(output.to_path_buf())
}

fn run_pipeline(config: &ResolvedConfig) -> Result<PipelineReport, CliError> {
    let checkpoint = match load_pipeline_checkpoint(&config.output) {
        Ok(checkpoint) => checkpoint,
        Err(err) => {
            warn!("Ignoring invalid pipeline checkpoint: {}", err);
            None
        }
    };

    let (slam, mut checkpoint_state) =
        match load_slam_from_checkpoint(&config.output, checkpoint.as_ref()) {
            Some(slam) => {
                info!("Stage 1/2: Input loading skipped (checkpoint)");
                info!("Stage 2/2: SLAM skipped (checkpoint)");
                (slam, checkpoint.unwrap_or_default())
            }
            None => {
                let input = load_input_source(config)?;
                let slam = run_slam_stage(input, config)?;
                let checkpoint_state = save_slam_checkpoint(&config.output, &slam, checkpoint)
                    .map_err(|err| CliError::Pipeline(format!("Checkpoint save: {err}")))?;
                (slam, checkpoint_state)
            }
        };
    let camera_count = slam.keyframes.len();
    reset_post_slam_checkpoint_state(&mut checkpoint_state);
    save_pipeline_checkpoint(&config.output, &checkpoint_state)
        .map_err(|err| CliError::Pipeline(format!("Checkpoint save: {err}")))?;
    let slam_output_path = export_slam_output_for_rustgs(&slam, &config.output)?;
    info!(
        "SLAM export ready for RustGS: {}",
        slam_output_path.display()
    );

    Ok(PipelineReport {
        camera_count,
        slam_output_path,
        mesh: MeshStats::default(),
    })
}

fn load_slam_from_checkpoint(
    output_dir: &Path,
    checkpoint: Option<&PipelineCheckpoint>,
) -> Option<SlamStageOutput> {
    let Some(checkpoint) = checkpoint else {
        return None;
    };
    if !checkpoint.slam_completed {
        return None;
    }

    let slam_checkpoint = checkpoint.slam.as_ref()?;
    if let Err(reason) = validate_slam_checkpoint(output_dir, slam_checkpoint) {
        warn!("SLAM checkpoint invalid: {}", reason);
        return None;
    }

    let camera = Camera::new(
        slam_checkpoint.camera.fx,
        slam_checkpoint.camera.fy,
        slam_checkpoint.camera.cx,
        slam_checkpoint.camera.cy,
        slam_checkpoint.camera.width,
        slam_checkpoint.camera.height,
    );

    let mut keyframes = Vec::with_capacity(slam_checkpoint.keyframes.len());
    for keyframe in &slam_checkpoint.keyframes {
        let path = resolve_checkpoint_path(output_dir, &keyframe.color_path);
        let color = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                warn!("Failed to read keyframe {}: {}", keyframe.index, err);
                return None;
            }
        };
        let pose =
            SE3::from_rotation_translation(&keyframe.pose.rotation, &keyframe.pose.translation);
        keyframes.push(KeyframeSample {
            index: keyframe.index,
            timestamp: keyframe.timestamp,
            width: keyframe.width,
            height: keyframe.height,
            color,
            depth: match &keyframe.depth_path {
                Some(depth_path) => {
                    let path = resolve_checkpoint_path(output_dir, depth_path);
                    match fs::read(&path) {
                        Ok(bytes) => {
                            if bytes.len() % std::mem::size_of::<f32>() != 0 {
                                warn!(
                                    "Depth checkpoint {} has invalid byte length {}",
                                    keyframe.index,
                                    bytes.len()
                                );
                                return None;
                            }
                            let mut depth =
                                Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
                            for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
                                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                depth.push(f32::from_le_bytes(arr));
                            }
                            Some(depth)
                        }
                        Err(err) => {
                            warn!("Failed to read depth keyframe {}: {}", keyframe.index, err);
                            return None;
                        }
                    }
                }
                None => None,
            },
            pose,
        });
    }

    info!(
        "Resuming from SLAM checkpoint with {} keyframes and {} sparse points",
        keyframes.len(),
        slam_checkpoint.map_points.len()
    );

    Some(SlamStageOutput {
        keyframes,
        camera,
        frame_count: slam_checkpoint.frame_count,
        map_points: slam_checkpoint
            .map_points
            .iter()
            .map(|point| MapPointData::new(point.position, point.color))
            .collect(),
    })
}

fn save_slam_checkpoint(
    output_dir: &Path,
    slam: &SlamStageOutput,
    existing: Option<PipelineCheckpoint>,
) -> Result<PipelineCheckpoint, PipelineCheckpointError> {
    let mut checkpoint = existing.unwrap_or_default();
    let frames_dir = slam_frames_dir(output_dir);
    if !frames_dir.exists() {
        fs::create_dir_all(&frames_dir).map_err(|source| PipelineCheckpointError::CreateDir {
            path: frames_dir.display().to_string(),
            source,
        })?;
    }

    let mut keyframes = Vec::with_capacity(slam.keyframes.len());
    for keyframe in &slam.keyframes {
        let file_name = format!("frame_{:06}.rgb", keyframe.index);
        let path = frames_dir.join(&file_name);
        fs::write(&path, &keyframe.color).map_err(|source| PipelineCheckpointError::Write {
            path: path.display().to_string(),
            source,
        })?;
        let relative = path
            .strip_prefix(output_dir)
            .unwrap_or(path.as_path())
            .display()
            .to_string();

        let depth_path = if let Some(depth) = &keyframe.depth {
            let depth_file_name = format!("frame_{:06}.depth", keyframe.index);
            let depth_full_path = frames_dir.join(&depth_file_name);
            let mut bytes = Vec::with_capacity(depth.len() * std::mem::size_of::<f32>());
            for value in depth {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(&depth_full_path, bytes).map_err(|source| {
                PipelineCheckpointError::Write {
                    path: depth_full_path.display().to_string(),
                    source,
                }
            })?;
            Some(
                depth_full_path
                    .strip_prefix(output_dir)
                    .unwrap_or(depth_full_path.as_path())
                    .display()
                    .to_string(),
            )
        } else {
            None
        };

        keyframes.push(KeyframeCheckpoint {
            index: keyframe.index,
            timestamp: keyframe.timestamp,
            width: keyframe.width,
            height: keyframe.height,
            pose: PoseCheckpoint {
                rotation: keyframe.pose.rotation(),
                translation: keyframe.pose.translation(),
            },
            color_path: relative,
            depth_path,
        });
    }

    checkpoint.video_completed = true;
    checkpoint.slam_completed = true;
    checkpoint.gaussian_completed = false;
    checkpoint.mesh_completed = false;
    checkpoint.slam = Some(SlamCheckpoint {
        camera: CameraCheckpoint {
            fx: slam.camera.focal.x,
            fy: slam.camera.focal.y,
            cx: slam.camera.principal.x,
            cy: slam.camera.principal.y,
            width: slam.camera.width,
            height: slam.camera.height,
        },
        frame_count: slam.frame_count,
        keyframes,
        map_points: slam
            .map_points
            .iter()
            .map(|point| MapPointCheckpoint {
                position: point.position,
                color: point.color,
            })
            .collect(),
    });

    save_pipeline_checkpoint(output_dir, &checkpoint)?;
    Ok(checkpoint)
}

fn reset_post_slam_checkpoint_state(checkpoint: &mut PipelineCheckpoint) {
    checkpoint.gaussian_completed = false;
    checkpoint.mesh_completed = false;
}

fn validate_slam_checkpoint(output_dir: &Path, checkpoint: &SlamCheckpoint) -> Result<(), String> {
    if checkpoint.keyframes.is_empty() {
        return Err("no keyframes in checkpoint".to_string());
    }
    if checkpoint.camera.width == 0 || checkpoint.camera.height == 0 {
        return Err("invalid camera dimensions".to_string());
    }
    if checkpoint.frame_count < checkpoint.keyframes.len() {
        return Err("frame count less than keyframe count".to_string());
    }

    for keyframe in &checkpoint.keyframes {
        if keyframe.width == 0 || keyframe.height == 0 {
            return Err(format!("invalid keyframe dimensions {}", keyframe.index));
        }
        let path = resolve_checkpoint_path(output_dir, &keyframe.color_path);
        let metadata =
            fs::metadata(&path).map_err(|_| format!("missing keyframe file {}", path.display()))?;
        let expected = keyframe
            .width
            .saturating_mul(keyframe.height)
            .saturating_mul(3) as u64;
        if metadata.len() != expected {
            return Err(format!(
                "keyframe {} size mismatch (expected {}, got {})",
                keyframe.index,
                expected,
                metadata.len()
            ));
        }
        if let Some(depth_path) = &keyframe.depth_path {
            let path = resolve_checkpoint_path(output_dir, depth_path);
            let metadata = fs::metadata(&path)
                .map_err(|_| format!("missing depth keyframe file {}", path.display()))?;
            let expected = keyframe
                .width
                .saturating_mul(keyframe.height)
                .saturating_mul(std::mem::size_of::<f32>() as u32)
                as u64;
            if metadata.len() != expected {
                return Err(format!(
                    "depth keyframe {} size mismatch (expected {}, got {})",
                    keyframe.index,
                    expected,
                    metadata.len()
                ));
            }
        }
    }

    Ok(())
}

fn resolve_checkpoint_path(output_dir: &Path, stored: &str) -> PathBuf {
    let candidate = PathBuf::from(stored);
    if candidate
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return output_dir.join("checkpoints").join("__blocked_path__");
    }

    let base = output_dir
        .canonicalize()
        .unwrap_or_else(|_| output_dir.to_path_buf());
    let joined = if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    };

    match joined.canonicalize() {
        Ok(canonical) if canonical.starts_with(&base) => canonical,
        _ => output_dir.join("checkpoints").join("__blocked_path__"),
    }
}

/// Check if a path is a TUM RGB-D dataset directory
fn is_tum_dataset(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }

    // Check for TUM dataset structure: rgb.txt or rgb/ directory
    let has_rgb_txt = path.join("rgb.txt").exists();
    let has_rgb_dir = path.join("rgb").is_dir();
    let has_depth_txt = path.join("depth.txt").exists();
    let has_depth_dir = path.join("depth").is_dir();

    // TUM dataset should have at least rgb.txt or both rgb/ and depth/ directories
    has_rgb_txt || (has_rgb_dir && (has_depth_txt || has_depth_dir))
}

fn is_kitti_dataset(path: &Path) -> bool {
    path.is_dir() && path.join("image_0").is_dir() && path.join("calib.txt").exists()
}

fn is_euroc_dataset(path: &Path) -> bool {
    let cam0 = path.join("mav0").join("cam0");
    cam0.join("data").is_dir() && cam0.join("data.csv").exists()
}

#[derive(Debug, Clone, Copy)]
enum DatasetKind {
    Tum,
    Kitti,
    Euroc,
}

fn detect_dataset_kind(config: &ResolvedConfig) -> Option<DatasetKind> {
    let configured = config.slam.dataset.dataset_type.trim().to_ascii_lowercase();
    match configured.as_str() {
        "tum" if is_tum_dataset(&config.input) => return Some(DatasetKind::Tum),
        "kitti" if is_kitti_dataset(&config.input) => return Some(DatasetKind::Kitti),
        "euroc" if is_euroc_dataset(&config.input) => return Some(DatasetKind::Euroc),
        _ => {}
    }

    if is_tum_dataset(&config.input) {
        Some(DatasetKind::Tum)
    } else if is_kitti_dataset(&config.input) {
        Some(DatasetKind::Kitti)
    } else if is_euroc_dataset(&config.input) {
        Some(DatasetKind::Euroc)
    } else {
        None
    }
}

fn load_input_source(config: &ResolvedConfig) -> Result<InputSource, CliError> {
    let dataset_config = DatasetConfig {
        root_path: config.input.clone(),
        load_depth: config.slam.dataset.load_depth,
        load_ground_truth: config.slam.dataset.load_ground_truth,
        max_frames: config.slam.dataset.max_frames,
        stride: config.slam.dataset.stride,
    };

    match detect_dataset_kind(config) {
        Some(DatasetKind::Tum) => {
            info!("Stage 1/2: Loading TUM RGB-D dataset");
            let dataset = TumRgbdDataset::load(dataset_config)
                .map_err(|err| CliError::Pipeline(format!("TUM dataset load: {err}")))?;
            let metadata = dataset.metadata();
            info!(
                "TUM dataset loaded: sequence={}, frames={}, has_depth={}, has_ground_truth={}",
                metadata.sequence,
                metadata.total_frames,
                metadata.has_depth,
                metadata.has_ground_truth
            );
            Ok(InputSource::Dataset(Box::new(dataset)))
        }
        Some(DatasetKind::Kitti) => {
            info!("Stage 1/2: Loading KITTI odometry dataset");
            let dataset = KittiDataset::load(dataset_config)
                .map_err(|err| CliError::Pipeline(format!("KITTI dataset load: {err}")))?;
            let metadata = dataset.metadata();
            info!(
                "KITTI dataset loaded: sequence={}, frames={}, has_depth={}, has_ground_truth={}",
                metadata.sequence,
                metadata.total_frames,
                metadata.has_depth,
                metadata.has_ground_truth
            );
            Ok(InputSource::Dataset(Box::new(dataset)))
        }
        Some(DatasetKind::Euroc) => {
            info!("Stage 1/2: Loading EuRoC MAV dataset");
            let dataset = EurocDataset::load(dataset_config)
                .map_err(|err| CliError::Pipeline(format!("EuRoC dataset load: {err}")))?;
            let metadata = dataset.metadata();
            info!(
                "EuRoC dataset loaded: sequence={}, frames={}, has_depth={}, has_ground_truth={}",
                metadata.sequence,
                metadata.total_frames,
                metadata.has_depth,
                metadata.has_ground_truth
            );
            Ok(InputSource::Dataset(Box::new(dataset)))
        }
        None => {
            let decoded = decode_video(config)?;
            Ok(InputSource::Video(decoded))
        }
    }
}

fn decode_video(config: &ResolvedConfig) -> Result<DecodedVideo, CliError> {
    info!("Stage 1/2: Video decode");
    debug!(
        "Video decoder config: cache_capacity={}, prefer_hardware={}",
        config.video.cache_capacity, config.video.prefer_hardware
    );

    let mut decoder = video::VideoDecoder::open(
        &config.input,
        video::VideoDecoderConfig {
            cache_capacity: config.video.cache_capacity,
            prefer_hardware: config.video.prefer_hardware,
        },
    )
    .map_err(|err| CliError::Pipeline(format!("Video decode: {err}")))?;

    let info = decoder.info().clone();
    let decoder_mode = if info.hardware_accel {
        "hardware"
    } else {
        "software"
    };
    info!(
        "Video stream: {}x{}, codec={}, container={}, fps={:.2}, decoder={}, mode={}",
        info.width,
        info.height,
        info.codec,
        info.container,
        info.frame_rate,
        info.decoder,
        decoder_mode
    );

    decoder
        .frame(0)
        .map_err(|err| CliError::Pipeline(format!("Video decode: {err}")))?;

    let total_frames = info.frame_count.unwrap_or(0);
    let camera = resolve_camera(&config.slam, info.width, info.height);

    info!("Video decode completed");
    Ok(DecodedVideo {
        decoder,
        camera,
        total_frames,
    })
}

fn run_slam_stage(
    input: InputSource,
    config: &ResolvedConfig,
) -> Result<SlamStageOutput, CliError> {
    info!("Stage 2/2: SLAM");
    let stage_start = Instant::now();

    match input {
        InputSource::Video(decoded) => run_slam_from_video(decoded, config, stage_start),
        InputSource::Dataset(dataset) => run_slam_from_dataset(dataset, config, stage_start),
    }
}

fn build_local_mapping(config: &ResolvedConfig, camera: Camera) -> LocalMapping {
    let defaults = LocalMappingConfig::default();
    let mut local_mapping = LocalMapping::new(LocalMappingConfig {
        max_keyframes: config.slam.mapper.local_mapping_window.max(2),
        max_map_points: config
            .slam
            .mapper
            .max_keyframes
            .max(1)
            .saturating_mul(config.slam.mapper.max_points_per_keyframe.max(1)),
        min_observations: defaults.min_observations,
        min_triangulation_angle: config.slam.mapper.min_triangulation_angle.max(0.1),
        min_triangulation_dist: config.slam.mapper.min_point_distance.max(0.0),
        max_reprojection_error: config.slam.mapper.max_reproj_error.max(0.1),
        local_ba_enabled: config.slam.mapper.use_local_mapping,
        local_ba_iterations: config.slam.optimizer.local_ba_iterations.max(1),
        local_ba_interval: defaults.local_ba_interval,
        culling_threshold: defaults.culling_threshold,
    });
    local_mapping.set_camera(camera);
    local_mapping.set_map(Map::new());
    local_mapping
}

fn build_mapping_keyframe(
    index: usize,
    timestamp: f64,
    width: u32,
    height: u32,
    pose: SE3,
    features: crate::core::FrameFeatures,
) -> KeyFrame {
    let mut frame = Frame::new(index as u64, timestamp, width, height);
    frame.set_pose(pose);
    frame.mark_as_keyframe();
    KeyFrame::new(frame, features)
}

fn collect_sparse_map_points(local_mapping: &LocalMapping) -> Vec<MapPointData> {
    let Some(map) = local_mapping.map() else {
        return Vec::new();
    };

    map.valid_points()
        .filter(|point| point.observations >= 2)
        .filter(|point| {
            point.position.x.is_finite()
                && point.position.y.is_finite()
                && point.position.z.is_finite()
        })
        .map(|point| {
            MapPointData::new(
                [point.position.x, point.position.y, point.position.z],
                point.color,
            )
        })
        .collect()
}

fn sparse_point_bin(point: [f32; 3]) -> (i32, i32, i32) {
    const BIN_SCALE: f32 = 100.0;
    (
        (point[0] * BIN_SCALE).round() as i32,
        (point[1] * BIN_SCALE).round() as i32,
        (point[2] * BIN_SCALE).round() as i32,
    )
}

fn accumulate_map_point(
    points: &mut Vec<MapPointData>,
    seen_bins: &mut HashSet<(i32, i32, i32)>,
    point: MapPointData,
) {
    if !point.position.iter().all(|value| value.is_finite()) {
        return;
    }
    let bin = sparse_point_bin(point.position);
    if seen_bins.insert(bin) {
        points.push(point);
    }
}

fn sparse_point_within_export_range(point: [f32; 3], max_distance: f32) -> bool {
    if !point.iter().all(|value| value.is_finite()) {
        return false;
    }

    let distance_sq = point.iter().map(|value| value * value).sum::<f32>();
    distance_sq.is_finite() && distance_sq.sqrt() <= max_distance.max(1.0)
}

fn run_slam_from_video(
    mut decoded: DecodedVideo,
    config: &ResolvedConfig,
    stage_start: Instant,
) -> Result<SlamStageOutput, CliError> {
    let mut vo = VisualOdometry::with_params(decoded.camera, config.slam.tracker.clone());
    let mut local_mapping = build_local_mapping(config, decoded.camera);
    let keyframe_interval = config.slam.mapper.keyframe_interval.max(1);
    let max_keyframes = config.slam.mapper.max_keyframes.max(1);
    let stride = config.slam.dataset.stride.max(1);
    let max_frames = config.slam.dataset.max_frames;
    let total_frames = decoded.total_frames;
    let total_available = if total_frames > 0 {
        (total_frames + stride - 1) / stride
    } else {
        0
    };
    let total_target = if max_frames > 0 {
        if total_available > 0 {
            total_available.min(max_frames)
        } else {
            max_frames
        }
    } else {
        total_available
    };
    let log_interval = progress_interval(total_target);

    let mut keyframes = Vec::new();
    let mut vo_sparse_points = Vec::new();
    let mut vo_sparse_bins = HashSet::new();
    let mut processed = 0usize;
    let mut success_frames = 0usize;
    let mut index = 0usize;

    loop {
        if max_frames > 0 && processed >= max_frames {
            break;
        }

        let frame = match decoded.decoder.frame(index) {
            Ok(frame) => frame,
            Err(video::VideoError::FrameIndex(_)) => break,
            Err(err) => {
                return Err(CliError::Pipeline(format!(
                    "SLAM: failed to decode frame {}: {err}",
                    index
                )));
            }
        };

        let gray = rgb_to_grayscale(&frame.data, frame.width as usize, frame.height as usize);
        let result = vo.process_frame(&gray, frame.width, frame.height);
        if result.success {
            success_frames += 1;
            for point in vo.last_sparse_points() {
                accumulate_map_point(
                    &mut vo_sparse_points,
                    &mut vo_sparse_bins,
                    MapPointData::new(point, None),
                );
            }
            if let Some(features) = vo.last_features() {
                local_mapping.insert_keyframe(build_mapping_keyframe(
                    frame.index,
                    frame.timestamp,
                    frame.width,
                    frame.height,
                    result.pose,
                    features,
                ));

                if processed % keyframe_interval == 0 && keyframes.len() < max_keyframes {
                    keyframes.push(KeyframeSample {
                        index: frame.index,
                        timestamp: frame.timestamp,
                        width: frame.width,
                        height: frame.height,
                        color: frame.data.as_ref().clone(),
                        depth: None,
                        pose: result.pose,
                    });
                }
            }
        }

        processed += 1;
        if should_log_progress(processed, total_target, log_interval) {
            log_progress("SLAM", processed, total_target, stage_start);
        }

        index = index.saturating_add(stride);
    }

    if processed == 0 {
        return Err(CliError::Pipeline("SLAM: no frames decoded".to_string()));
    }

    let tracking_success_ratio = if processed > 0 {
        Some(success_frames as f32 / processed as f32)
    } else {
        None
    };
    let mut map_points = collect_sparse_map_points(&local_mapping);
    let mut seen_sparse_bins = map_points
        .iter()
        .map(|point| sparse_point_bin(point.position))
        .collect::<HashSet<_>>();
    for point in vo_sparse_points {
        accumulate_map_point(&mut map_points, &mut seen_sparse_bins, point);
    }
    let max_export_distance = config.slam.mapper.max_point_distance.max(1.0);
    map_points
        .retain(|point| sparse_point_within_export_range(point.position, max_export_distance));

    log_progress("SLAM", processed, total_target, stage_start);
    if let Some(ratio) = tracking_success_ratio {
        info!(
            "SLAM completed: frames={}, keyframes={}, sparse_points={}, tracking_success={:.1}%",
            processed,
            keyframes.len(),
            map_points.len(),
            ratio * 100.0
        );
    } else {
        info!(
            "SLAM completed: frames={}, keyframes={}, sparse_points={}",
            processed,
            keyframes.len(),
            map_points.len()
        );
    }
    let relocalization_stats = vo.relocalization_stats();
    info!(
        "VO relocalization: lost_events={}, direct_retrack={}/{}, anchor_store={}, anchor_success={}/{}, anchor_candidates_tested={}, monocular_reinit={}/{}, cached_anchors={}",
        relocalization_stats.lost_events,
        relocalization_stats.direct_retrack_successes,
        relocalization_stats.direct_retrack_attempts,
        relocalization_stats.anchor_store_successes,
        relocalization_stats.anchor_relocalization_successes,
        relocalization_stats.anchor_relocalization_calls,
        relocalization_stats.anchor_candidates_tested,
        relocalization_stats.monocular_reinit_successes,
        relocalization_stats.monocular_reinit_attempts,
        relocalization_stats.cached_anchor_keyframes,
    );

    Ok(SlamStageOutput {
        keyframes,
        camera: decoded.camera,
        frame_count: processed,
        map_points,
    })
}

fn run_slam_from_dataset(
    dataset: Box<dyn Dataset>,
    config: &ResolvedConfig,
    stage_start: Instant,
) -> Result<SlamStageOutput, CliError> {
    let camera = dataset.camera();
    let mut vo = VisualOdometry::with_params(camera.clone(), config.slam.tracker.clone());
    let mut local_mapping = build_local_mapping(config, camera);

    let keyframe_interval = config.slam.mapper.keyframe_interval.max(1);
    let max_keyframes = config.slam.mapper.max_keyframes.max(1);
    let total_frames = dataset.len();
    let log_interval = progress_interval(total_frames);

    let mut keyframes = Vec::new();
    let mut vo_sparse_points = Vec::new();
    let mut vo_sparse_bins = HashSet::new();
    let mut processed = 0usize;
    let mut success_frames = 0usize;

    for index in 0..total_frames {
        let frame = match dataset.get_frame(index) {
            Ok(frame) => frame,
            Err(err) => {
                return Err(CliError::Pipeline(format!(
                    "SLAM: failed to load frame {}: {err}",
                    index
                )));
            }
        };

        let gray = rgb_to_grayscale(&frame.color, frame.width as usize, frame.height as usize);
        let result = vo.process_frame(&gray, frame.width, frame.height);

        if result.success {
            success_frames += 1;
            for point in vo.last_sparse_points() {
                accumulate_map_point(
                    &mut vo_sparse_points,
                    &mut vo_sparse_bins,
                    MapPointData::new(point, None),
                );
            }
            if let Some(features) = vo.last_features() {
                local_mapping.insert_keyframe(build_mapping_keyframe(
                    frame.index,
                    frame.timestamp,
                    frame.width,
                    frame.height,
                    result.pose,
                    features,
                ));

                if processed % keyframe_interval == 0 && keyframes.len() < max_keyframes {
                    keyframes.push(KeyframeSample {
                        index: frame.index,
                        timestamp: frame.timestamp,
                        width: frame.width,
                        height: frame.height,
                        color: frame.color,
                        depth: frame.depth,
                        pose: result.pose,
                    });
                }
            }
        }

        processed += 1;
        if should_log_progress(processed, total_frames, log_interval) {
            log_progress("SLAM", processed, total_frames, stage_start);
        }
    }

    if processed == 0 {
        return Err(CliError::Pipeline("SLAM: no frames loaded".to_string()));
    }

    let tracking_success_ratio = if processed > 0 {
        Some(success_frames as f32 / processed as f32)
    } else {
        None
    };
    let mut map_points = collect_sparse_map_points(&local_mapping);
    let mut seen_sparse_bins = map_points
        .iter()
        .map(|point| sparse_point_bin(point.position))
        .collect::<HashSet<_>>();
    for point in vo_sparse_points {
        accumulate_map_point(&mut map_points, &mut seen_sparse_bins, point);
    }
    let max_export_distance = config.slam.mapper.max_point_distance.max(1.0);
    map_points
        .retain(|point| sparse_point_within_export_range(point.position, max_export_distance));

    log_progress("SLAM", processed, total_frames, stage_start);
    if let Some(ratio) = tracking_success_ratio {
        info!(
            "SLAM completed: frames={}, keyframes={}, sparse_points={}, tracking_success={:.1}%",
            processed,
            keyframes.len(),
            map_points.len(),
            ratio * 100.0
        );
    } else {
        info!(
            "SLAM completed: frames={}, keyframes={}, sparse_points={}",
            processed,
            keyframes.len(),
            map_points.len()
        );
    }
    let relocalization_stats = vo.relocalization_stats();
    info!(
        "VO relocalization: lost_events={}, direct_retrack={}/{}, anchor_store={}, anchor_success={}/{}, anchor_candidates_tested={}, monocular_reinit={}/{}, cached_anchors={}",
        relocalization_stats.lost_events,
        relocalization_stats.direct_retrack_successes,
        relocalization_stats.direct_retrack_attempts,
        relocalization_stats.anchor_store_successes,
        relocalization_stats.anchor_relocalization_successes,
        relocalization_stats.anchor_relocalization_calls,
        relocalization_stats.anchor_candidates_tested,
        relocalization_stats.monocular_reinit_successes,
        relocalization_stats.monocular_reinit_attempts,
        relocalization_stats.cached_anchor_keyframes,
    );

    Ok(SlamStageOutput {
        keyframes,
        camera,
        frame_count: processed,
        map_points,
    })
}

fn build_rustgs_slam_output(
    slam: &SlamStageOutput,
    output_dir: &Path,
) -> Result<rustscan_types::SlamOutput, CliError> {
    let mut output = rustscan_types::SlamOutput::new(rustscan_types::Intrinsics::new(
        slam.camera.focal.x,
        slam.camera.focal.y,
        slam.camera.principal.x,
        slam.camera.principal.y,
        slam.camera.width,
        slam.camera.height,
    ));

    let frames_dir = slam_frames_dir(output_dir);
    fs::create_dir_all(&frames_dir)
        .map_err(|err| CliError::Pipeline(format!("prepare RustGS frame directory: {err}")))?;
    for keyframe in &slam.keyframes {
        let color_path = frames_dir.join(format!("frame_{:06}.rgb", keyframe.index));
        let expected_color = (keyframe.width as usize)
            .saturating_mul(keyframe.height as usize)
            .saturating_mul(3);
        if keyframe.color.len() != expected_color {
            return Err(CliError::Pipeline(format!(
                "keyframe {} RGB buffer size mismatch: got {}, expected {}",
                keyframe.index,
                keyframe.color.len(),
                expected_color,
            )));
        }
        fs::write(&color_path, &keyframe.color).map_err(|err| {
            CliError::Pipeline(format!(
                "write RustGS color frame {}: {err}",
                color_path.display(),
            ))
        })?;

        let pose = rustscan_types::SE3::from_rotation_translation(
            &keyframe.pose.rotation(),
            &keyframe.pose.translation(),
        );
        let mut scene_pose = rustscan_types::ScenePose::new(
            keyframe.index as u64,
            color_path,
            pose,
            keyframe.timestamp,
        );

        if let Some(depth) = &keyframe.depth {
            let depth_path = frames_dir.join(format!("frame_{:06}.depth", keyframe.index));
            let expected_depth = (keyframe.width as usize).saturating_mul(keyframe.height as usize);
            if depth.len() != expected_depth {
                return Err(CliError::Pipeline(format!(
                    "keyframe {} depth buffer size mismatch: got {}, expected {}",
                    keyframe.index,
                    depth.len(),
                    expected_depth,
                )));
            }

            let mut bytes = Vec::with_capacity(depth.len() * std::mem::size_of::<f32>());
            for value in depth {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(&depth_path, bytes).map_err(|err| {
                CliError::Pipeline(format!(
                    "write RustGS depth frame {}: {err}",
                    depth_path.display(),
                ))
            })?;

            scene_pose = scene_pose.with_depth_path(depth_path);
        }

        output.add_pose(scene_pose);
    }

    for point in &slam.map_points {
        output.add_map_point(*point);
    }

    Ok(output)
}

fn slam_output_json_path(output_dir: &Path) -> PathBuf {
    output_dir.join("slam_output.json")
}

fn export_slam_output_for_rustgs(
    slam: &SlamStageOutput,
    output_dir: &Path,
) -> Result<PathBuf, CliError> {
    let slam_output = build_rustgs_slam_output(slam, output_dir)?;
    let path = slam_output_json_path(output_dir);
    slam_output.save(&path).map_err(|err| {
        CliError::Pipeline(format!("write SLAM export {}: {err}", path.display()))
    })?;
    Ok(path)
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / MIB_BYTES
}

fn resolve_camera(config: &SlamConfig, width: u32, height: u32) -> Camera {
    let cam = &config.camera;
    if cam.width == width && cam.height == height {
        return Camera::new(cam.fx, cam.fy, cam.cx, cam.cy, width, height);
    }

    let scale_x = width as f32 / cam.width as f32;
    let scale_y = height as f32 / cam.height as f32;
    Camera::new(
        cam.fx * scale_x,
        cam.fy * scale_y,
        cam.cx * scale_x,
        cam.cy * scale_y,
        width,
        height,
    )
}

fn rgb_to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    if rgb.len() < expected {
        return Vec::new();
    }

    let mut gray = Vec::with_capacity(width * height);
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }
    gray
}

fn progress_interval(total: usize) -> usize {
    if total == 0 {
        return 30;
    }
    (total / 20).max(1)
}

fn should_log_progress(processed: usize, total: usize, interval: usize) -> bool {
    if processed == 0 {
        return false;
    }
    if total > 0 && processed == total {
        return true;
    }
    processed % interval == 0
}

fn log_progress(stage: &str, processed: usize, total: usize, started_at: Instant) {
    let percent = if total > 0 {
        (processed as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    let eta_ms = estimate_eta_ms(processed, total, started_at);
    let mem_mb = current_process_memory_mb();
    let gpu_util = current_gpu_utilization_percent();

    let eta_label = eta_ms
        .map(|ms| format!("{ms}ms"))
        .unwrap_or_else(|| "n/a".to_string());
    let mem_label = mem_mb
        .map(|mb| format!("{mb:.1}MB"))
        .unwrap_or_else(|| "n/a".to_string());
    let gpu_label = gpu_util
        .map(|u| format!("{u:.1}%"))
        .unwrap_or_else(|| "n/a".to_string());

    if total > 0 {
        info!(
            "{stage}: {processed}/{total} ({percent:.1}%), eta={eta}, mem={mem}, gpu={gpu}",
            stage = stage,
            processed = processed,
            total = total,
            percent = percent,
            eta = eta_label,
            mem = mem_label,
            gpu = gpu_label
        );
    } else {
        info!(
            "{stage}: {processed} frames processed, eta={eta}, mem={mem}, gpu={gpu}",
            stage = stage,
            processed = processed,
            eta = eta_label,
            mem = mem_label,
            gpu = gpu_label
        );
    }
}

fn estimate_eta_ms(processed: usize, total: usize, started_at: Instant) -> Option<u128> {
    if total == 0 || processed == 0 || processed >= total {
        return None;
    }
    let elapsed_ms = started_at.elapsed().as_millis();
    let remaining = total.saturating_sub(processed) as u128;
    let processed = processed as u128;
    Some(elapsed_ms.saturating_mul(remaining) / processed)
}

fn current_process_memory_mb() -> Option<f64> {
    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_process(pid);
    let process = system.process(pid)?;
    Some(bytes_to_mib(process.memory()))
}

fn current_gpu_utilization_percent() -> Option<f32> {
    None
}

fn write_results(
    results: &ResultsJson,
    output_dir: &Path,
    format: OutputFormat,
) -> Result<(), CliError> {
    if format != OutputFormat::Json {
        return Ok(());
    }

    let path = output_dir.join("results.json");
    let payload = serde_json::to_string_pretty(results)
        .map_err(|err| CliError::Pipeline(format!("Failed to serialize results: {err}")))?;

    fs::write(&path, payload).map_err(|source| CliError::OutputWrite { path, source })?;
    info!(
        "Results written to {}",
        output_dir.join("results.json").display()
    );

    Ok(())
}

fn print_text_summary(results: &ResultsJson) {
    println!("RustScan Results");
    println!("Status: {}", results.status);
    if let Some(input) = &results.input {
        println!("Input: {}", input);
    }
    if let Some(output) = &results.output {
        println!("Output: {}", output);
    }
    println!("Processing time (ms): {}", results.processing_time_ms);
    println!("Camera count: {}", results.camera_count);
    if let Some(slam_output) = &results.slam_output {
        println!("SLAM output: {}", slam_output);
    }
    println!("Mesh vertices: {}", results.mesh.vertex_count);
    println!("Mesh triangles: {}", results.mesh.triangle_count);
}

fn build_diagnostics(
    input: Option<&Path>,
    output: Option<&Path>,
    config: Option<&Path>,
) -> Diagnostics {
    Diagnostics {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        rustscan_version: env!("CARGO_PKG_VERSION").to_string(),
        cwd: std::env::current_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "<unavailable>".to_string()),
        input: input.map(|path| path.display().to_string()),
        output: output.map(|path| path.display().to_string()),
        config: config.map(|path| path.display().to_string()),
    }
}

fn handle_error(
    err: &CliError,
    start: Instant,
    resolved: Option<&ResolvedConfig>,
    output_format: Option<OutputFormat>,
    output_override: Option<PathBuf>,
    config_path: Option<PathBuf>,
) -> ExitCode {
    let input = resolved.map(|r| r.input.clone());
    let output = resolved.map(|r| r.output.clone()).or(output_override);

    let diagnostics =
        build_diagnostics(input.as_deref(), output.as_deref(), config_path.as_deref());

    let error_info = ErrorInfo {
        error_type: err.error_type().to_string(),
        root_cause: err.to_string(),
        component: err.component().to_string(),
        suggestion: err.suggestion().to_string(),
    };

    error!(
        "{}: {} (component: {})",
        error_info.error_type, error_info.root_cause, error_info.component
    );

    eprintln!("Error: {}", error_info.error_type);
    eprintln!("Component: {}", error_info.component);
    eprintln!("Cause: {}", error_info.root_cause);
    eprintln!("Suggestion: {}", error_info.suggestion);
    eprintln!(
        "Diagnostics: os={}, arch={}, cwd={}",
        diagnostics.os, diagnostics.arch, diagnostics.cwd
    );
    if let Some(input) = &diagnostics.input {
        eprintln!("Diagnostics: input={}", input);
    }
    if let Some(output) = &diagnostics.output {
        eprintln!("Diagnostics: output={}", output);
    }
    if let Some(config) = &diagnostics.config {
        eprintln!("Diagnostics: config={}", config);
    }

    let results = ResultsJson {
        status: "error".to_string(),
        input: input.map(|path| path.display().to_string()),
        output: output.as_ref().map(|path| path.display().to_string()),
        processing_time_ms: start.elapsed().as_millis(),
        camera_count: 0,
        slam_output: None,
        mesh: MeshStats::default(),
        error: Some(error_info),
        diagnostics,
    };

    let format = output_format.or_else(|| resolved.map(|r| r.output_format));
    if let Some(OutputFormat::Json) = format {
        if let Some(output_dir) = output {
            let _ = ensure_output_dir(&output_dir)
                .and_then(|dir| write_results(&results, &dir, OutputFormat::Json));
        }
    }

    err.exit_code()
}
