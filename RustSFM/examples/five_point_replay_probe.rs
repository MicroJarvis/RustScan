//! Read-only, CPU-only fixed sampler PREFIX replay; not an executed RANSAC trace.
//! Default work: 12 pairs x 512 solves per pass, one warmup per path and four
//! paired rounds. Release runtime is expected to be modest, not time-guaranteed.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use nalgebra::{Matrix3, Vector3};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use rustsfm::{
    correspondence_graph::pair_id_to_image_pair, database::ColmapDatabase,
    five_point::estimate_five_point_essential, types::CameraModel,
};
use rustslam::ColmapRandomSampler;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Parser)]
#[command(
    about = "CPU five-point fixed sampler PREFIX replay, not actual executed 960-trial RANSAC workload"
)]
pub(crate) struct Args {
    #[arg(long, help = "Required existing COLMAP database; opened read-only")]
    pub(crate) database: PathBuf,
    #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u32).range(1..=12))]
    pub(crate) pairs: u32,
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=512))]
    pub(crate) trials: u32,
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=512))]
    pub(crate) batch_size: u32,
}

#[derive(Clone)]
pub(crate) struct Input {
    pub(crate) indices: [usize; 5],
    pub(crate) left: [Vector3<f64>; 5],
    pub(crate) right: [Vector3<f64>; 5],
}

type Outputs = Vec<Vec<Matrix3<f64>>>;
type Bits = Vec<Vec<[u64; 9]>>;

fn stratified_indices(total: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            let start = i * total / count;
            let end = (i + 1) * total / count;
            start + (end - start) / 2
        })
        .collect()
}

// Mirrors two_view::vector3_from_ray / vector3_from_homogeneous_ray.
fn normalized_ray(ray: [f64; 3]) -> Vector3<f64> {
    let ray = Vector3::new(ray[0], ray[1], ray[2]);
    let norm = ray.norm();
    if norm > 1.0e-12 && norm.is_finite() {
        ray / norm
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}

fn gather(left: &[Vector3<f64>], right: &[Vector3<f64>], trials: usize) -> Vec<Input> {
    let active: Vec<_> = (0..left.len()).collect();
    // Exactly one stateful sampler per pair, seed 1; never reset per trial.
    let mut sampler = ColmapRandomSampler::new(1, &active);
    (0..trials)
        .map(|_| {
            let indices: [usize; 5] = sampler
                .sample(5)
                .try_into()
                .expect("five valid observations");
            Input {
                left: indices.map(|i| left[i]),
                right: indices.map(|i| right[i]),
                indices,
            }
        })
        .collect()
}

fn solve(input: &Input) -> Vec<Matrix3<f64>> {
    estimate_five_point_essential(&input.left, &input.right)
}

fn replay(inputs: &[Input], pool: Option<&ThreadPool>) -> Outputs {
    replay_batches(inputs, pool, inputs.len().max(1))
}

pub(crate) fn replay_batches(
    inputs: &[Input],
    pool: Option<&ThreadPool>,
    batch_size: usize,
) -> Outputs {
    let mut outputs = Vec::with_capacity(inputs.len());
    for batch in inputs.chunks(batch_size) {
        outputs.extend(match pool {
            // Slice par_iter is indexed: collect retains pair/trial order, regardless
            // of completion order. Each solver's model order is left untouched.
            Some(pool) => pool.install(|| batch.par_iter().map(solve).collect::<Outputs>()),
            None => batch.iter().map(solve).collect::<Outputs>(),
        });
    }
    outputs
}

pub(crate) fn output_bits(outputs: &Outputs) -> Bits {
    outputs
        .iter()
        .map(|models| {
            models
                .iter()
                .map(|m| {
                    // Explicit row-major coefficient order (nalgebra storage is column-major).
                    std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits())
                })
                .collect()
        })
        .collect()
}

fn word(hash: &mut blake3::Hasher, value: u64) {
    hash.update(&value.to_le_bytes());
}

pub(crate) fn input_digest(inputs: &[Input]) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(b"five-point-prefix-input-v1");
    word(&mut hash, inputs.len() as u64);
    for input in inputs {
        for i in input.indices {
            word(&mut hash, i as u64);
        }
        for ray in input.left.iter().chain(&input.right) {
            for x in ray.iter() {
                word(&mut hash, x.to_bits());
            }
        }
    }
    hash.finalize().to_hex().to_string()
}

fn model_digest(bits: &[Vec<[u64; 9]>]) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(b"five-point-prefix-models-row-major-v1");
    word(&mut hash, bits.len() as u64);
    for models in bits {
        word(&mut hash, models.len() as u64);
        for model in models {
            for &x in model {
                word(&mut hash, x);
            }
        }
    }
    hash.finalize().to_hex().to_string()
}

pub(crate) struct PairObservations {
    pub(crate) left: Vec<Vector3<f64>>,
    pub(crate) right: Vec<Vector3<f64>>,
}

fn load(args: &Args) -> Result<(Vec<Input>, Vec<Value>, usize)> {
    let (inputs, reports, eligible, _) = load_with_observations(args)?;
    Ok((inputs, reports, eligible))
}

// Shared read-only loader/sampler for the independent GPU replay example.
pub(crate) fn load_with_observations(
    args: &Args,
) -> Result<(Vec<Input>, Vec<Value>, usize, Vec<PairObservations>)> {
    ensure!(
        args.database.is_file(),
        "database unavailable: {}",
        args.database.display()
    );
    let db = ColmapDatabase::open_read_only(&args.database)?;
    let images = db.read_all_images()?;
    let mut pairs = db.read_num_matches()?;
    pairs.retain(|(_, count)| *count >= 5);
    pairs.sort_unstable_by_key(|(id, _)| *id);
    ensure!(
        pairs.len() >= args.pairs as usize,
        "need {} raw-match pairs with >=5 matches; found {}",
        args.pairs,
        pairs.len()
    );
    let mut inputs = Vec::new();
    let mut reports = Vec::new();
    let mut observations = Vec::new();
    for rank in stratified_indices(pairs.len(), args.pairs as usize) {
        let pair_id = pairs[rank].0;
        let (id1, id2) = pair_id_to_image_pair(pair_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let load_image = |id| -> Result<_> {
            let image = images
                .iter()
                .find(|image| image.image_id == id)
                .context("missing image")?;
            let camera = db
                .read_camera(image.camera_id)?
                .context("missing camera")?
                .camera;
            let camera = CameraModel::from_colmap(
                camera.model_id,
                camera.width,
                camera.height,
                &camera.params,
            )
            .context("unsupported camera")?;
            let keypoints = db
                .read_keypoints(id)?
                .into_iter()
                .map(|kp| kp.to_keypoint())
                .collect::<Vec<_>>();
            Ok((camera, keypoints, image.name.clone()))
        };
        let (camera1, keypoints1, name1) = load_image(id1)?;
        let (camera2, keypoints2, name2) = load_image(id2)?;
        let matches = db.read_matches(id1, id2)?; // Never read two_view_geometries/inliers.
        let prepare_start = Instant::now();
        let mut left = Vec::new();
        let mut right = Vec::new();
        // Mirrors prepare_pair_observations, including f32 projection validity
        // before f64 rays, preserving raw match order and invalid-index filtering.
        for m in &matches {
            let Some(lk) = keypoints1.get(m.point2d_idx1 as usize) else {
                continue;
            };
            let Some(rk) = keypoints2.get(m.point2d_idx2 as usize) else {
                continue;
            };
            if camera1.cam_from_img_f32(lk.x(), lk.y()).is_none() {
                continue;
            }
            if camera2.cam_from_img_f32(rk.x(), rk.y()).is_none() {
                continue;
            }
            let Some(lray) = camera1.cam_ray_from_img(lk.x() as f64, lk.y() as f64) else {
                continue;
            };
            let Some(rray) = camera2.cam_ray_from_img(rk.x() as f64, rk.y() as f64) else {
                continue;
            };
            left.push(normalized_ray(lray));
            right.push(normalized_ray(rray));
        }
        let preparation_seconds = prepare_start.elapsed().as_secs_f64();
        ensure!(left.len() >= 5, "selected pair {pair_id} has fewer than five valid observations; no replacement selected");
        let start = Instant::now();
        let gathered = gather(&left, &right, args.trials as usize);
        let sampling_gather_seconds = start.elapsed().as_secs_f64();
        reports.push(json!({
            "pair_id": pair_id, "sorted_eligible_rank": rank,
            "image_ids": [id1, id2], "image_names": [name1, name2],
            "raw_matches": matches.len(), "valid_observations": left.len(),
            "preparation_seconds": preparation_seconds,
            "sampling_gather_seconds": sampling_gather_seconds,
            "input_fingerprint": input_digest(&gathered)
        }));
        inputs.extend(gathered);
        observations.push(PairObservations { left, right });
    }
    Ok((inputs, reports, pairs.len(), observations))
}

fn measure(
    inputs: &[Input],
    pool1: &ThreadPool,
    pool4: &ThreadPool,
    batch_size: usize,
) -> Result<(Bits, Vec<Value>)> {
    // Full warmup pass per path, excluded from all reported measurements.
    let baseline = output_bits(&replay(inputs, None));
    for pool in [pool1, pool4] {
        ensure!(
            output_bits(&replay_batches(inputs, Some(pool), batch_size)) == baseline,
            "warmup coefficient/order mismatch"
        );
    }
    let mut rounds = Vec::new();
    for round in 0..4 {
        let order = if round % 2 == 0 { [0, 1, 4] } else { [4, 1, 0] };
        let mut measurements = Vec::new();
        for threads in order {
            let pool = match threads {
                1 => Some(pool1),
                4 => Some(pool4),
                _ => None,
            };
            let start = Instant::now();
            let outputs = replay_batches(inputs, pool, batch_size);
            let seconds = start.elapsed().as_secs_f64();
            // Conversion, full exact comparison, hashing and destruction excluded.
            let bits = output_bits(&outputs);
            ensure!(
                bits == baseline,
                "round {round}, threads {threads}: coefficient/model-count/order mismatch"
            );
            measurements.push(json!({
                "path": if threads == 0 { "serial" } else { "scoped_rayon" },
                "threads": threads, "solver_collect_seconds": seconds,
                "exact_match": true, "ordered_model_digest": model_digest(&bits)
            }));
        }
        rounds.push(json!({"round": round + 1, "order": order, "measurements": measurements}));
    }
    Ok((baseline, rounds))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (inputs, mut pairs, eligible_pairs) = load(&args)?;
    // Locally owned scoped pools only; no global pool or production dispatch.
    // Both pool creation and destruction sit outside measurement intervals.
    let (bits, rounds) = ThreadPoolBuilder::new().num_threads(1).build_scoped(
        |thread| thread.run(),
        |pool1| -> Result<_> {
            ThreadPoolBuilder::new().num_threads(4).build_scoped(
                |thread| thread.run(),
                |pool4| measure(&inputs, pool1, pool4, args.batch_size as usize),
            )?
        },
    )??;
    for (pair, models) in pairs.iter_mut().zip(bits.chunks(args.trials as usize)) {
        pair["model_counts"] = json!(models.iter().map(Vec::len).collect::<Vec<_>>());
        pair["ordered_model_digest"] = json!(model_digest(models));
    }
    let report = json!({
        "benchmark": "five_point_fixed_sampler_PREFIX", "database": args.database,
        "scope": "Existing CPU solver only. NOT actual executed960 RANSAC workload. No inferred trial counts or inlier causes. No GPU, CPU fallback implementation, or production Taskflow claims.",
        "timing_contract": "Pre-gathered inputs. Solver timings include full matrix allocation and indexed collection/retention overhead, not solver arithmetic alone. DB I/O, normalization, sampling/gather, pool setup, warmup, bit conversion, exact comparison, hashing, output destruction and JSON emission excluded. All outputs retained until timing ends.",
        "selection": "Midpoint of each equal contiguous stratum of sorted raw pair IDs with >=5 raw matches; insufficient valid selected pair is an error, not silently replaced.",
        "eligible_pairs": eligible_pairs, "seed_per_pair": 1,
        "prefix_trials_per_pair": args.trials, "solves_per_pass": inputs.len(),
                "batch_size": args.batch_size, "installs_per_parallel_pass": inputs.len().div_ceil(args.batch_size as usize),
        "warmup_passes_per_path": 1, "paired_rounds": 4,
        "input_fingerprint": input_digest(&inputs),
        "ordered_model_digest": model_digest(&bits),
        "digest_encoding": "BLAKE3 domain-tagged little-endian u64 words; lengths delimit trial/model lists; input words include sampled indices then left/right ray coefficient bits",
        "pairs": pairs, "rounds": rounds,
        "coefficient_encoding": "u64 f64::to_bits, row-major; nested pair-major/trial-major, solver model order; baseline output, all paths compared exactly",
        "model_coefficient_bits": bits
    });
    serde_json::to_writer(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_database_and_bounds_work() {
        assert!(Args::try_parse_from(["probe"]).is_err());
        for (flag, value) in [
            ("--pairs", "0"),
            ("--pairs", "13"),
            ("--trials", "0"),
            ("--trials", "513"),
        ] {
            assert!(
                Args::try_parse_from(["probe", "--database", "unused.db", flag, value]).is_err()
            );
        }
        let args = Args::try_parse_from(["probe", "--database", "unused.db"]).unwrap();
        assert_eq!((args.pairs, args.trials), (12, 512));
    }

    #[test]
    fn strata_cover_sorted_list() {
        let indices = stratified_indices(1200, 12);
        assert_eq!(indices, (0..12).map(|i| 50 + i * 100).collect::<Vec<_>>());
        assert_eq!(stratified_indices(12, 12), (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn sampling_is_stateful_prefix_and_gather_order_is_exact() {
        let rays: Vec<_> = (0..23)
            .map(|i| normalized_ray([i as f64, 1.0, 2.0]))
            .collect();
        let short = gather(&rays, &rays, 8);
        let long = gather(&rays, &rays, 16);
        let mut sampler = ColmapRandomSampler::new(1, &(0..23).collect::<Vec<_>>());
        for (a, b) in short.iter().zip(&long) {
            assert_eq!(a.indices.as_slice(), sampler.sample(5));
            assert_eq!(a.indices, b.indices);
            for (j, &index) in a.indices.iter().enumerate() {
                assert_eq!(a.left[j], rays[index]);
                assert_eq!(a.right[j], rays[index]);
            }
        }
        assert_ne!(short[0].indices, short[1].indices);
    }

    #[test]
    fn replay_retains_order_and_all_coefficient_bits() {
        let left: Vec<_> = (0..17)
            .map(|i| {
                normalized_ray([
                    (i as f64 * 1.7).sin(),
                    (i as f64 * 0.8).cos(),
                    2.0 + i as f64 * 0.1,
                ])
            })
            .collect();
        let right: Vec<_> = (0..17)
            .map(|i| {
                normalized_ray([
                    (i as f64 * 1.7).sin() + 0.3,
                    (i as f64 * 0.8).cos() - 0.1,
                    2.0 + i as f64 * 0.1,
                ])
            })
            .collect();
        let inputs = gather(&left, &right, 8);
        let baseline = output_bits(&replay(&inputs, None));
        for threads in [1, 4] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_scoped(
                    |thread| thread.run(),
                    |pool| {
                        for batch_size in [1, 3, 64, 512] {
                            assert_eq!(
                                baseline,
                                output_bits(&replay_batches(&inputs, Some(pool), batch_size))
                            );
                        }
                    },
                )
                .unwrap();
        }
        let mut m = Matrix3::zeros();
        m[(0, 1)] = -0.0;
        let bits = output_bits(&vec![vec![m], vec![]]);
        assert_eq!(bits[0][0][1], (-0.0f64).to_bits());
        assert_ne!(
            model_digest(&bits),
            model_digest(&[bits[1].clone(), bits[0].clone()])
        );
    }
}
