//! B1: frozen real bytes plus an independent, exhaustive scalar specification.
use super::*;

type Row = [u8; 128];
type Bits = (u32, u32, u32);

fn rows(bytes: &[u8]) -> Vec<Row> {
    assert_eq!(bytes.len() % 128, 0);
    bytes
        .chunks_exact(128)
        .map(|r| r.try_into().unwrap())
        .collect()
}

fn features(rows: &[Row]) -> SiftFeatures {
    SiftFeatures {
        descriptors: rows
            .iter()
            .map(|r| Descriptor::new(r.map(|v| f32::from(v) / 512.0)))
            .collect(),
        descriptors_u8: rows.to_vec(),
        ..Default::default()
    }
}

// Deliberately neither the production integer reduction nor its top-two update loop.
// Every partial sum is an exact integer below 2^24, including extreme u8 inputs.
fn scalar_distance(a: &Row, b: &Row) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..128 {
        let delta = f32::from(a[i]) - f32::from(b[i]);
        sum += delta * delta;
    }
    sum
}

fn ranked(query: &Row, train: &[Row]) -> Vec<(f32, u32)> {
    let mut all: Vec<_> = train
        .iter()
        .enumerate()
        .map(|(i, row)| (scalar_distance(query, row), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all
}

fn oracle_one_way(query: &[Row], train: &[Row], o: &SiftMatchingOptions) -> Vec<Bits> {
    query
        .iter()
        .enumerate()
        .filter_map(|(i, q)| {
            let all = ranked(q, train);
            let &(best, j) = all.first()?;
            let second = all.get(1).map_or(f32::INFINITY, |r| r.0);
            // Strict Lowe ratio; inclusive maximum distance, evaluated in squared units.
            if best <= 262_144.0 * o.max_distance * o.max_distance
                && best.sqrt() < o.max_ratio * second.sqrt()
            {
                Some((i as u32, j, (best / 262_144.0).sqrt().to_bits()))
            } else {
                None
            }
        })
        .collect()
}

fn oracle(left: &[Row], right: &[Row], o: &SiftMatchingOptions) -> Vec<Bits> {
    let mut accepted = oracle_one_way(left, right, o);
    if o.cross_check {
        let reverse = oracle_one_way(right, left, o);
        accepted.retain(|&(i, j, _)| reverse.iter().any(|&(rj, ri, _)| (i, j) == (ri, rj)));
    }
    // Explicit query-index secondary key specifies stable forward enumeration order,
    // independently of production's stable distance-only sort.
    accepted.sort_unstable_by(|a, b| {
        f32::from_bits(a.2)
            .total_cmp(&f32::from_bits(b.2))
            .then(a.0.cmp(&b.0))
    });
    if o.max_num_matches != 0 {
        accepted.truncate(o.max_num_matches);
    }
    accepted
}

fn bits(matches: &[Match]) -> Vec<Bits> {
    matches
        .iter()
        .map(|m| (m.query_idx, m.train_idx, m.distance.to_bits()))
        .collect()
}

fn check_neighbors(query: &[Row], train: &[Row]) {
    let index = SiftDescriptorIndex::build(train);
    for q in query {
        let expected = ranked(q, train);
        let actual = index.search_two_nearest(q);
        if expected.is_empty() {
            assert!(actual.is_none());
            continue;
        }
        let actual = actual.unwrap();
        let second = expected.get(1).copied().unwrap_or((f32::INFINITY, 0));
        assert_eq!(
            (actual.best_index, actual.best_l2.to_bits()),
            (expected[0].1, expected[0].0.to_bits())
        );
        assert_eq!(
            (actual.second_best_index, actual.second_best_l2.to_bits()),
            (second.1, second.0.to_bits())
        );
    }
}

// Check private uint8 entry points and public dispatch separately, including the
// public empty-input short circuit. This module shares the uint8 backend's cfg.
fn check_matchers(
    left: &[Row],
    right: &[Row],
    options: &SiftMatchingOptions,
    expected: &[Bits],
) -> Vec<Bits> {
    assert_eq!(
        oracle(left, right, options),
        expected,
        "independent specification"
    );
    let (left, right) = (features(left), features(right));
    let mut actual = Vec::new();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), threads);
            for brute in [false, true] {
                let options = SiftMatchingOptions {
                    cpu_brute_force_matcher: brute,
                    ..options.clone()
                };
                let plain = match_sift_colmap_uint8(&left, &right, &options, None);
                let mut timing = SiftMatchingTiming::default();
                let profiled = match_sift_colmap_uint8(&left, &right, &options, Some(&mut timing));
                assert_eq!(bits(&plain), expected, "threads={threads}, brute={brute}");
                assert_eq!(bits(&profiled), expected);
                assert!(timing.uint8_backend);
                for value in [timing.prepare_ms, timing.index_build_ms, timing.search_ms] {
                    assert!(value.is_finite() && value >= 0.0);
                }
                if brute {
                    assert_eq!(timing.index_build_ms, 0.0);
                }
                {
                    assert_eq!(
                        bits(&match_sift_with_options(&left, &right, &options)),
                        expected
                    );
                    let mut timing = SiftMatchingTiming::default();
                    assert_eq!(
                        bits(&match_sift_with_options_profiled(
                            &left,
                            &right,
                            &options,
                            Some(&mut timing)
                        )),
                        expected
                    );
                    assert_eq!(
                        timing.uint8_backend,
                        !left.descriptors.is_empty() && !right.descriptors.is_empty()
                    );
                }
                actual = bits(&plain);
            }
        });
    }
    actual
}

#[test]
fn b1_real_descriptor_complete_outputs() {
    let left_bytes = include_bytes!("fixtures/sift_b1/left.bin");
    let right_bytes = include_bytes!("fixtures/sift_b1/right.bin");
    let (left, right) = (rows(left_bytes), rows(right_bytes));
    assert_eq!((left.len(), right.len()), (128, 127));
    assert_eq!(
        blake3::hash(left_bytes).to_hex().as_str(),
        "4b2c698a07022a1e49bd3c9100652fa2c22388b9a133d7d5e25bf1dfd40ccea6"
    );
    assert_eq!(
        blake3::hash(right_bytes).to_hex().as_str(),
        "71bd4d0903fda84df8e6f38cd2c5d0ba08d380f1f977dc490f77903091ad92e1"
    );
    check_neighbors(&left, &right);
    check_neighbors(&right, &left);
    let mut output = Vec::new();
    for cross_check in [false, true] {
        for limit in [0, 7] {
            let options = SiftMatchingOptions {
                cross_check,
                max_num_matches: limit,
                ..Default::default()
            };
            let expected = oracle(&left, &right, &options);
            let count = if limit != 0 {
                7
            } else if cross_check {
                66
            } else {
                67
            };
            assert_eq!(expected.len(), count);
            let actual = check_matchers(&left, &right, &options, &expected);
            if cross_check && limit == 0 {
                // Pin the public default configuration as well as the unlimited sentinel.
                check_matchers(&left, &right, &SiftMatchingOptions::default(), &expected);
            }
            output.push(u8::from(cross_check));
            output.extend_from_slice(&(limit as u32).to_le_bytes());
            output.extend_from_slice(&(actual.len() as u32).to_le_bytes());
            for (i, j, d) in actual {
                for value in [i, j, d] {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    // Frozen only after all full production vectors matched the independent oracle.
    assert_eq!(
        blake3::hash(&output).to_hex().as_str(),
        "fbb55fac40c5e1a1d2504548b5246f331a7f4936c46735551440d32579ac6e52"
    );
}

fn row(first: u8) -> Row {
    let mut row = [0; 128];
    row[0] = first;
    row
}

fn expected(entries: &[(u32, u32, u8)]) -> Vec<Bits> {
    entries
        .iter()
        .map(|&(i, j, delta)| (i, j, (f32::from(delta) / 512.0).to_bits()))
        .collect()
}

#[test]
fn b1_explicit_empty_single_ties_and_boundaries() {
    let base = SiftMatchingOptions {
        cross_check: false,
        ..Default::default()
    };
    for cross_check in [false, true] {
        let options = SiftMatchingOptions {
            cross_check,
            ..base.clone()
        };
        for (left, right, want) in [
            (vec![], vec![], vec![]),
            (vec![row(0)], vec![], vec![]),
            (vec![], vec![row(0)], vec![]),
            (vec![row(0)], vec![row(1)], expected(&[(0, 0, 1)])),
        ] {
            check_neighbors(&[row(0)], &right);
            check_matchers(&left, &right, &options, &want);
        }
    }
    let left = [row(0)];
    let right = [row(1), row(2)];
    let ratio_boundary = SiftMatchingOptions {
        max_ratio: 0.5,
        ..base.clone()
    };
    check_matchers(&left, &right, &ratio_boundary, &[]);
    let ratio_above = SiftMatchingOptions {
        max_ratio: f32::from_bits(0.5f32.to_bits() + 1),
        ..base.clone()
    };
    check_matchers(&left, &right, &ratio_above, &expected(&[(0, 0, 1)]));
    let distance_boundary = SiftMatchingOptions {
        max_distance: 1.0 / 512.0,
        ..base.clone()
    };
    check_matchers(&left, &right, &distance_boundary, &expected(&[(0, 0, 1)]));
    let distance_below = SiftMatchingOptions {
        max_distance: f32::from_bits((1.0f32 / 512.0).to_bits() - 1),
        ..base.clone()
    };
    check_matchers(&left, &right, &distance_below, &[]);
    let tied = [row(2), row(0), row(2)];
    check_neighbors(&[row(1)], &tied);
    let equal_ratio = SiftMatchingOptions {
        max_ratio: 1.0,
        ..base.clone()
    };
    check_matchers(&[row(1)], &tied, &equal_ratio, &[]);
    let permissive = SiftMatchingOptions {
        max_ratio: 1.1,
        ..base
    };
    check_matchers(&[row(1)], &tied, &permissive, &expected(&[(0, 0, 1)]));
    check_neighbors(&left, &[row(0), row(0)]);
    check_matchers(&left, &[row(0), row(0)], &permissive, &[]);
}

#[test]
fn b1_explicit_reverse_ratio_stable_sort_and_truncation() {
    for cross_check in [false, true] {
        let base = SiftMatchingOptions {
            cross_check,
            max_num_matches: 0,
            ..Default::default()
        };
        // Forward ratio passes for both; reverse nearest ties and fails its ratio.
        let want = if cross_check {
            vec![]
        } else {
            expected(&[(0, 0, 1), (1, 0, 1)])
        };
        check_matchers(&[row(0), row(2)], &[row(1), row(20)], &base, &want);
        // Reverse ratio passes, but only the first forward pair is mutual.
        let want = if cross_check {
            expected(&[(0, 0, 1)])
        } else {
            expected(&[(0, 0, 1), (1, 0, 2)])
        };
        check_matchers(&[row(0), row(3)], &[row(1), row(20)], &base, &want);
        let left = [row(31), row(10), row(51), row(70)];
        let right = [row(0), row(10), row(30), row(50), row(70)];
        let all = expected(&[(1, 1, 0), (3, 4, 0), (0, 2, 1), (2, 3, 1)]);
        for limit in [0, 1, 3, 4, 8] {
            let options = SiftMatchingOptions {
                max_num_matches: limit,
                ..base.clone()
            };
            let count = if limit == 0 { 4 } else { limit.min(4) };
            check_matchers(&left, &right, &options, &all[..count]);
        }
    }
}
