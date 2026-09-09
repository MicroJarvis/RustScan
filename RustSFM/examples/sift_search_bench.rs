//! Fixed real-descriptor benchmark. Extraction, index copies and parity checks
//! are outside search timing. The reference preserves the old float reduction.
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use rustsfm::sift::{SiftExtractionOptions, SiftFeatures, SiftMatchingOptions};
use serde::Serialize;
use std::{
    fs,
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

// Compile the production source without widening the library's public API.
#[path = "../src/feature/sift_index.rs"]
mod search;
use search::{SiftDescriptorIndex, SiftNearestNeighbors};
type Descriptor = [u8; 128];

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Extract {
        left: PathBuf,
        right: PathBuf,
        corpus: PathBuf,
    },
    Bench {
        corpus: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 6)]
        rounds: usize,
    },
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

fn extract(image: &Path) -> Result<Vec<Descriptor>> {
    let gray = rustsfm::colmap_image::load_colmap_grayscale_u8(image)?;
    Ok(rustsfm::sift::extract_sift_from_grayscale_u8(
        &gray.data,
        gray.width,
        gray.height,
        &SiftExtractionOptions {
            max_num_features: 8192,
            ..Default::default()
        },
    )?
    .descriptors_u8)
}

fn read_descriptors(path: &Path) -> Result<(Vec<Descriptor>, String)> {
    let bytes = fs::read(path)?;
    ensure!(
        !bytes.is_empty() && bytes.len() % 128 == 0,
        "invalid descriptor file"
    );
    let hash = blake3::hash(&bytes).to_hex().to_string();
    Ok((
        bytes
            .chunks_exact(128)
            .map(|row| row.try_into().unwrap())
            .collect(),
        hash,
    ))
}

#[inline(never)]
fn reference_nearest(query: &Descriptor, train: &[Descriptor]) -> SiftNearestNeighbors {
    let mut result = SiftNearestNeighbors {
        best_index: 0,
        second_best_index: 0,
        best_l2: f32::INFINITY,
        second_best_l2: f32::INFINITY,
    };
    for (index, descriptor) in train.iter().enumerate() {
        let distance: f32 = query
            .iter()
            .zip(descriptor)
            .map(|(a, b)| {
                let delta = i32::from(*a) - i32::from(*b);
                (delta * delta) as f32
            })
            .sum();
        if distance < result.best_l2 {
            result.second_best_l2 = result.best_l2;
            result.second_best_index = result.best_index;
            result.best_l2 = distance;
            result.best_index = index as u32;
        } else if distance < result.second_best_l2 {
            result.second_best_l2 = distance;
            result.second_best_index = index as u32;
        }
    }
    result
}

#[inline(never)]
fn reference_all(left: &[Descriptor], right: &[Descriptor]) -> Vec<SiftNearestNeighbors> {
    left.iter()
        .map(|q| reference_nearest(q, right))
        .chain(right.iter().map(|q| reference_nearest(q, left)))
        .collect()
}

#[inline(never)]
fn production_all(
    left: &[Descriptor],
    right: &[Descriptor],
    left_index: &SiftDescriptorIndex,
    right_index: &SiftDescriptorIndex,
) -> Vec<SiftNearestNeighbors> {
    left.iter()
        .map(|q| right_index.search_two_nearest(q).unwrap())
        .chain(
            right
                .iter()
                .map(|q| left_index.search_two_nearest(q).unwrap()),
        )
        .collect()
}

fn bits(rows: &[SiftNearestNeighbors]) -> Vec<(u32, u32, u32, u32)> {
    rows.iter()
        .map(|r| {
            (
                r.best_index,
                r.second_best_index,
                r.best_l2.to_bits(),
                r.second_best_l2.to_bits(),
            )
        })
        .collect()
}

fn features(rows: &[Descriptor]) -> SiftFeatures {
    SiftFeatures {
        descriptors: rows
            .iter()
            .map(|r| lowe_sift::Descriptor::new(r.map(|v| f32::from(v) / 512.0)))
            .collect(),
        descriptors_u8: rows.to_vec(),
        ..Default::default()
    }
}

#[derive(Serialize)]
struct Measurement {
    round: usize,
    mode: &'static str,
    elapsed_ms: f64,
}

fn main() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "use --release");
    match Args::parse().command {
        Command::Extract {
            left,
            right,
            corpus,
        } => {
            let mut inputs = Vec::new();
            for (name, source) in [("left.bin", left), ("right.bin", right)] {
                let rows = extract(&source)?;
                ensure!(!rows.is_empty(), "no descriptors extracted");
                let bytes: Vec<u8> = rows.iter().flatten().copied().collect();
                write_new(&corpus.join(name), &bytes)?;
                inputs.push(serde_json::json!({"image": source, "source_blake3": blake3::hash(&fs::read(&source)?).to_hex().to_string(), "file": name, "rows": rows.len(), "blake3": blake3::hash(&bytes).to_hex().to_string()}));
            }
            write_new(
                &corpus.join("metadata.json"),
                serde_json::to_string_pretty(&inputs)?.as_bytes(),
            )?;
            println!("{}", serde_json::to_string_pretty(&inputs)?);
        }
        Command::Bench {
            corpus,
            output,
            rounds,
        } => {
            ensure!(rounds > 0, "rounds must be positive");
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)?;
            let (left, left_hash) = read_descriptors(&corpus.join("left.bin"))?;
            let (right, right_hash) = read_descriptors(&corpus.join("right.bin"))?;
            let left_index = SiftDescriptorIndex::build(&left);
            let right_index = SiftDescriptorIndex::build(&right);
            // One full call per mode serves as warm-up and creates exact reference output.
            let expected = bits(&reference_all(&left, &right));
            ensure!(
                bits(&production_all(&left, &right, &left_index, &right_index)) == expected,
                "warm-up parity failed"
            );

            let mut measurements = Vec::new();
            for round in 0..rounds {
                for mode in if round % 2 == 0 {
                    ["legacy-float", "production"]
                } else {
                    ["production", "legacy-float"]
                } {
                    let start = Instant::now();
                    let rows = match mode {
                        "production" => production_all(
                            black_box(&left),
                            black_box(&right),
                            &left_index,
                            &right_index,
                        ),

                        _ => reference_all(black_box(&left), black_box(&right)),
                    };
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    ensure!(
                        bits(black_box(&rows)) == expected,
                        "nearest/second-nearest bitwise parity failed for {mode}"
                    );
                    println!("round={} mode={mode} search_ms={elapsed_ms:.3}", round + 1);
                    measurements.push(Measurement {
                        round: round + 1,
                        mode,
                        elapsed_ms,
                    });
                }
            }
            let matches = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()?
                .install(|| {
                    rustsfm::sift::match_sift_with_options(
                        &features(&left),
                        &features(&right),
                        &SiftMatchingOptions::default(),
                    )
                });
            let match_bits: Vec<_> = matches
                .iter()
                .map(|m| (m.query_idx, m.train_idx, m.distance.to_bits()))
                .collect();
            let report = serde_json::json!({
                "cpu_threads": 1, "left_rows": left.len(), "right_rows": right.len(),
                "left_blake3": left_hash, "right_blake3": right_hash,
                "measurement_boundary": "bidirectional exhaustive nearest/second-nearest search plus result allocation; excludes extraction, file IO, index build, parity checks and final match filtering",
                "nearest_bitwise_parity": true, "default_match_bits": match_bits, "measurements": measurements,
            });
            output.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
            println!(
                "Exact nearest-neighbor parity passed; default matches={}",
                matches.len()
            );
        }
    }
    Ok(())
}
