use super::*;
use crate::five_point_generated::{build_elimination_matrix, determinant_coeffs};
use nalgebra::{DMatrix, Matrix3, Rotation3, Vector3};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[path = "five_point_q4_tests.rs"]
mod q4;

fn relative_error(gpu: &[f32], cpu: &[f64]) -> f64 {
    assert_eq!(gpu.len(), cpu.len());
    assert!(gpu.iter().all(|v| v.is_finite()));
    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
    gpu.iter()
        .zip(cpu)
        .map(|(&g, &c)| (f64::from(g) - c).abs() / scale)
        .fold(0.0, f64::max)
}

fn determinant_reference(solved: &DMatrix<f64>) -> [f64; 39] {
    let mut b = [0.0; 39];
    for col in 0..3 {
        for (offset, start, count) in [(0, 0, 3), (4, 3, 3), (8, 6, 4)] {
            for k in 0..count {
                b[13 * col + offset + k + 1] += solved[(2 * col + 4, start + k)];
                b[13 * col + offset + k] -= solved[(2 * col + 5, start + k)];
            }
        }
    }
    b
}

fn validate_roots(gpu: &WgpuFivePointF32) -> Result<()> {
    fn from_roots(roots: &[f64]) -> [f32; 11] {
        let mut coefficients = vec![1.0];
        for &root in roots {
            let mut next = vec![0.0; coefficients.len() + 1];
            for (i, c) in coefficients.iter().enumerate() {
                next[i] += c;
                next[i + 1] -= c * root;
            }
            coefficients = next;
        }
        let mut out = [0.0; 11];
        for (i, c) in coefficients.iter().enumerate() {
            out[11 - coefficients.len() + i] = *c as f32;
        }
        out
    }
    let expected = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.2, 0.6, 1.1, 1.6, 2.1];
    let mut complex = [0.0; 11];
    complex[8] = 1.0;
    complex[10] = 1.0;
    let polynomials = [
        from_roots(&expected),
        complex,
        from_roots(&[1.0, 1.0]),
        [0.0; 11],
        from_roots(&[1.0; 10]),
    ];
    let roots = gpu.compute_polynomial_roots(&polynomials)?;
    assert_eq!(roots.len(), polynomials.len());
    for (i, r) in roots.iter().enumerate() {
        eprintln!("root fixture {i}: degree={}, iterations={}, unconverged={}, complex={}, duplicate={}, failed={}, slots={:?}",r.degree,r.root_iterations,r.unconverged,r.complex_filtered,r.duplicate_roots,r.polynomial_failed,r.slots.iter().map(|s|(s.status,s.root)).collect::<Vec<_>>());
        assert!(r.root_iterations <= 384);
    }
    assert_eq!(roots[0].unconverged, 0);
    for expected in expected {
        assert!(
            roots[0]
                .slots
                .iter()
                .any(|s| s.status == FivePointSlotStatus::RealRoot
                    && (f64::from(s.root[0]) - expected).abs() < 2e-3),
            "missing root {expected}"
        );
    }
    assert_eq!(roots[1].complex_filtered + roots[1].unconverged, 2);
    assert_eq!(roots[1].dropped_leading, 8);
    assert!(
        roots[2].duplicate_roots + roots[2].unconverged >= 1,
        "repeated roots must not silently produce two independent roots"
    );
    assert!(roots[3].polynomial_failed);
    assert!(roots[4].unconverged + roots[4].complex_filtered + roots[4].duplicate_roots > 0);
    Ok(())
}

/// One synthetic output record: header floats plus ten sixteen-float slots.
/// Mirrors the layout documented in `five_point_recovery.wgsl`.
fn record(header: &[(usize, f32)], slots: &[(usize, f32, [f32; 9])]) -> Vec<f32> {
    let mut values = vec![0.0f32; complete::STRIDE];
    for &(index, value) in header {
        values[index] = value;
    }
    for &(slot, status, essential) in slots {
        let base = 16 + slot * 16;
        values[base] = status;
        values[base + 1] = 0.25 + slot as f32;
        values[base + 2] = -0.5 - slot as f32;
        values[base + 3] = 1e-7;
        values[base + 4..base + 13].copy_from_slice(&essential);
        values[base + 13] = 2e-6;
        values[base + 14] = 3e-6;
        values[base + 15] = 4e-6;
    }
    values
}

#[test]
fn decode_builds_fixed_slots_without_losing_state_or_order() -> Result<()> {
    assert!(complete::decode(&[])?.is_empty());

    // Every slot unused: the decoder still reports ten ordered slots.
    let empty = complete::decode(&record(&[], &[]))?;
    assert_eq!(empty.len(), 1);
    assert_eq!(empty[0].trial, 0);
    assert_eq!(empty[0].model_count, 0);
    assert_eq!(empty[0].upstream_status, FivePointStatus::Success);
    for (slot, s) in empty[0].slots.iter().enumerate() {
        assert_eq!(s.slot, slot);
        assert_eq!(s.status, FivePointSlotStatus::Unused);
        assert!(s.essential.is_none());
        assert_eq!(s.root, [0.0, 0.0]);
        assert_eq!(s.null_residual, 0.0);
    }

    // Mixed success and failure slots in one trial. Only Accepted slots carry
    // an essential matrix, and non-accepted slots keep their diagnostics.
    let accepted = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let last = [-1., -2., -3., -4., -5., -6., -7., -8., -9.];
    let mixed = complete::decode(&record(
        &[(1, 2.0), (2, 6.0), (4, 1.0), (5, 1.0), (7, 1.0), (11, 5.0)],
        &[
            (0, 1.0, accepted),
            (1, 2.0, [7.0; 9]),
            (2, 3.0, [7.0; 9]),
            (3, 4.0, [7.0; 9]),
            (4, 5.0, [7.0; 9]),
            (5, 6.0, [7.0; 9]),
            (6, 9.0, [7.0; 9]),
            (7, 10.0, [7.0; 9]),
            (9, 1.0, last),
        ],
    ))?;
    let slots = &mixed[0].slots;
    assert_eq!(mixed[0].model_count, 2);
    assert_eq!(mixed[0].rank, 5);
    assert_eq!(slots[0].essential, Some(accepted));
    assert_eq!(slots[9].essential, Some(last));
    assert_eq!(
        slots.iter().map(|s| s.status).collect::<Vec<_>>(),
        [
            FivePointSlotStatus::Accepted,
            FivePointSlotStatus::Complex,
            FivePointSlotStatus::RealAxisRejected,
            FivePointSlotStatus::DuplicateRoot,
            FivePointSlotStatus::NullVectorFailure,
            FivePointSlotStatus::InvalidEssential,
            FivePointSlotStatus::DuplicateModel,
            FivePointSlotStatus::RealRoot,
            FivePointSlotStatus::Unused,
            FivePointSlotStatus::Accepted,
        ]
    );
    assert_eq!(mixed[0].real_axis_rejected, 1);
    assert_eq!(mixed[0].root_not_done, 0);
    assert_eq!(mixed[0].polish_rejected, 0);

    // The two additional former-NotConverged codes decode and count.
    let split = complete::decode(&record(
        &[(4, 2.0)],
        &[(0, 7.0, [0.0; 9]), (1, 8.0, [0.0; 9])],
    ))?;
    assert_eq!(split[0].slots[0].status, FivePointSlotStatus::RootNotDone);
    assert_eq!(
        split[0].slots[1].status,
        FivePointSlotStatus::PolishRejected
    );
    assert_eq!(split[0].root_not_done, 1);
    assert_eq!(split[0].polish_rejected, 1);
    assert_eq!(split[0].unconverged, 2);
    assert!(FivePointSlotStatus::RealAxisRejected.is_unconverged());
    assert!(FivePointSlotStatus::RootNotDone.is_unconverged());
    assert!(FivePointSlotStatus::PolishRejected.is_unconverged());
    assert!(!FivePointSlotStatus::Complex.is_unconverged());
    for (slot, s) in slots.iter().enumerate() {
        assert_eq!(s.slot, slot);
        assert_eq!(
            s.essential.is_some(),
            s.status == FivePointSlotStatus::Accepted
        );
        if s.status != FivePointSlotStatus::Unused {
            assert_eq!(s.root, [0.25 + slot as f32, -0.5 - slot as f32]);
            assert_eq!(s.root_backward_error, 1e-7);
            assert_eq!(s.null_residual, 2e-6);
            assert_eq!(s.constraint_residual, 3e-6);
            assert_eq!(s.essential_residual, 4e-6);
        }
    }

    // A failed trial must not leak state into the following successful trial.
    let mut batch = record(&[(0, 2.0)], &[]);
    batch.extend(record(&[(1, 1.0), (15, 1.0)], &[(4, 1.0, accepted)]));
    batch.extend(record(&[(13, 3.0)], &[]));
    let trials = complete::decode(&batch)?;
    assert_eq!(trials.len(), 3);
    assert_eq!(
        trials.iter().map(|t| t.trial).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(trials[0].upstream_status, FivePointStatus::RankDeficient);
    assert!(trials[0].slots.iter().all(|s| s.essential.is_none()));
    assert!(trials[1].polynomial_failed);
    assert_eq!(trials[1].slots[4].essential, Some(accepted));
    assert!(trials[1]
        .slots
        .iter()
        .enumerate()
        .all(|(slot, s)| (slot == 4) == s.essential.is_some()));
    assert_eq!(
        trials[2].algebra_status,
        FivePointStatus::SingularElimination
    );
    assert!(trials[2].slots.iter().all(|s| s.essential.is_none()));

    // Illegal encodings are rejected, not silently mapped onto a valid state.
    // 7.0 and 8.0 are now RootNotDone / PolishRejected; keep them out of this list.
    for status in [11.0, 12.0, -1.0, f32::NAN] {
        let error = complete::decode(&record(&[], &[(3, status, [0.0; 9])]))
            .expect_err("invalid slot status must fail")
            .to_string();
        assert!(error.contains("invalid GPU slot status"), "{error}");
    }
    let error = complete::decode(&record(&[(0, 6.0)], &[]))
        .expect_err("invalid upstream status must fail")
        .to_string();
    assert!(error.contains("invalid GPU five-point status"), "{error}");
    // Header statuses are still checked before slot statuses.
    let error = complete::decode(&record(&[(0, 6.0)], &[(1, 11.0, [0.0; 9])]))
        .expect_err("invalid record must fail")
        .to_string();
    assert!(error.contains("invalid GPU five-point status"), "{error}");
    Ok(())
}

#[test]
fn five_point_f32_actual_gpu_stages() -> Result<()> {
    let context = match WgpuContext::try_new_optional()? {
        Some(context) => context,
        None => {
            eprintln!("five-point GPU unavailable; skipping integration test");
            return Ok(());
        }
    };
    eprintln!("five-point GPU: {:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let mut rng = StdRng::seed_from_u64(0x5f32);
    let mut rays1 = Vec::new();
    let mut rays2 = Vec::new();
    for sample in 0..16 {
        for _ in 0..5 {
            let p = Vector3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(2.0..5.0),
            );
            let q = if sample < 8 {
                Rotation3::from_euler_angles(0.1, -0.2, 0.07) * p + Vector3::new(0.7, -0.2, 0.3)
            } else {
                Vector3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(2.0..5.0),
                )
            };
            rays1.push(p.normalize().map(|x| x as f32).into());
            rays2.push(q.normalize().map(|x| x as f32).into());
        }
    }
    let matrices = gpu.compute_constraint_matrices(&rays1, &rays2)?;
    assert_eq!(matrices.len(), 16);
    for (sample, a) in matrices.iter().enumerate() {
        for r in 0..5 {
            for b in 0..3 {
                for c in 0..3 {
                    assert!(
                        (a[r * 9 + b * 3 + c]
                            - rays2[sample * 5 + r][b] * rays1[sample * 5 + r][c])
                            .abs()
                            < 1e-7
                    );
                }
            }
        }
    }
    let diagnostics = gpu.compute_nullspace_diagnostics(&matrices)?;
    let mut bases = Vec::new();
    let mut worst_residual = 0.0f64;
    for (sample, (a, d)) in matrices.iter().zip(&diagnostics).enumerate() {
        assert_eq!(d.status, FivePointStatus::Success, "sample {sample}: {d:?}");
        assert_eq!(d.rank, 5);
        assert!((1..=64).contains(&d.sweeps));
        assert!(
            d.min_diagonal_ratio > 1e-5,
            "sample {sample}: min_sigma_ratio={}",
            d.min_diagonal_ratio
        );
        let basis = d.basis.unwrap();
        let n = DMatrix::from_column_slice(9, 4, &basis.map(f64::from));
        let a64 = DMatrix::from_row_slice(5, 9, &a.map(f64::from));
        let residual = (&a64 * &n).norm() / a64.norm();
        worst_residual = worst_residual.max(residual);
        assert!(residual < 5e-4, "sample {sample}: A*N={residual}");
        assert!((n.transpose() * n - DMatrix::identity(4, 4)).norm() < 2e-4);
        bases.push(basis);
    }
    // Scale invariance, exact/near rank deficiency, and zero input are actual dispatches.
    let scaled = [
        matrices[0].map(|v| v * 1e-12),
        matrices[0].map(|v| v * 1e12),
    ];
    for d in gpu.compute_nullspace_diagnostics(&scaled)? {
        assert_eq!(d.status, FivePointStatus::Success);
    }
    // Mild rank-5 (σ5/σ1 = 1e-5): Q2 rescue band. Near-duplicate rows are rank-4
    // and must stay RankDeficient — they are not the lost-solution population.
    let mild = {
        let a = DMatrix::from_row_slice(5, 9, &matrices[0].map(f64::from));
        let svd = a.svd(true, true);
        let u = svd.u.as_ref().expect("SVD U");
        let v_t = svd.v_t.as_ref().expect("SVD V^T");
        let s1 = svd.singular_values[0];
        let sigma = nalgebra::DVector::from_iterator(
            5,
            [1.0, 0.5, 0.25, 0.1, 3e-5].into_iter().map(|t| s1 * t),
        );
        let rebuilt = u * nalgebra::DMatrix::from_diagonal(&sigma) * v_t;
        let mut out = [0.0f32; 45];
        for r in 0..5 {
            for c in 0..9 {
                out[r * 9 + c] = rebuilt[(r, c)] as f32;
            }
        }
        out
    };
    let mild_diag = gpu.compute_nullspace_diagnostics(&[mild])?.remove(0);
    assert_eq!(
        mild_diag.status,
        FivePointStatus::Success,
        "mild rank-5 should pass σ5/σ1 gate: {mild_diag:?}"
    );
    assert!(mild_diag.min_diagonal_ratio > 1e-5, "{mild_diag:?}");
    assert!(
        mild_diag.min_diagonal_ratio < 1e-3,
        "fixture should sit in the rescue band, got {mild_diag:?}"
    );
    // Hard degenerates: zeros, rank-1 all-ones, five identical rows.
    let mut five_same = matrices[0];
    let first: [f32; 9] = matrices[0][..9].try_into().unwrap();
    for r in 1..5 {
        five_same[r * 9..(r + 1) * 9].copy_from_slice(&first);
    }
    let invalid = [[0.0; 45], [1.0; 45], five_same];
    for d in gpu.compute_nullspace_diagnostics(&invalid)? {
        assert_eq!(d.status, FivePointStatus::RankDeficient, "{d:?}");
        assert!(d.basis.is_none());
        assert!(
            d.min_diagonal_ratio <= 1e-5,
            "hard degenerate σ5/σ1 should stay at/below the floor: {d:?}"
        );
    }
    assert!(gpu.compute_nullspace_basis(&invalid).is_err());
    assert!(gpu.compute_nullspace_basis(&[[f32::NAN; 45]]).is_err());
    assert!(gpu.compute_algebra(&[[f32::INFINITY; 36]]).is_err());
    assert!(gpu.compute_constraint_matrices(&[[1.0; 3]], &[]).is_err());
    assert!(gpu
        .compute_constraint_matrices(&[[1.0; 3]], &[[1.0; 3]])
        .is_err());
    assert!(gpu.compute_nullspace_basis(&[])?.is_empty());
    assert!(gpu.compute_algebra(&[])?.is_empty());
    assert!(gpu.compute_constraint_matrices(&[], &[])?.is_empty());

    let results = gpu.compute_algebra(&bases)?;
    let mut worst = [0.0f64; 4];
    let mut pivot_swaps_needed = 0;
    for (sample, (basis, result)) in bases.iter().zip(&results).enumerate() {
        assert_eq!(
            result.status,
            FivePointStatus::Success,
            "sample {sample}: pivot={}",
            result.minimum_relative_pivot
        );
        let a = build_elimination_matrix(&basis.map(f64::from));
        let left = DMatrix::from_column_slice(10, 10, &a[..100]);
        let right = DMatrix::from_column_slice(10, 10, &a[100..]);
        if (1..10).any(|r| left[(r, 0)].abs() > left[(0, 0)].abs()) {
            pivot_swaps_needed += 1;
        }
        let solved = left
            .clone()
            .lu()
            .solve(&right)
            .expect("CPU f64 reference nonsingular");
        let b = determinant_reference(&solved);
        let coeffs = determinant_coeffs(&b);
        let errors = [
            relative_error(&result.elimination, &a),
            relative_error(&result.solved, solved.as_slice()),
            relative_error(&result.determinant, &b),
            relative_error(&result.coefficients, &coeffs),
        ];
        eprintln!(
            "sample {sample}: [A, solve, B, coeff] relative errors={errors:?}, pivot={}",
            result.minimum_relative_pivot
        );
        for i in 0..4 {
            worst[i] = worst[i].max(errors[i]);
        }
        assert!(errors[0] < 3e-6, "elimination {errors:?}");
        assert!(errors[1] < 3e-3 && errors[2] < 3e-3, "solve/B {errors:?}");
        assert!(errors[3] < 1e-2, "polynomial {errors:?}");
        // Backward error against the actually generated GPU left/right matrix.
        let ga = result.elimination.map(f64::from);
        let gl = DMatrix::from_column_slice(10, 10, &ga[..100]);
        let gr = DMatrix::from_column_slice(10, 10, &ga[100..]);
        let gx = DMatrix::from_column_slice(10, 10, &result.solved.map(f64::from));
        let backward = (&gl * &gx - &gr).norm() / (gl.norm() * gx.norm() + gr.norm());
        assert!(backward < 2e-6, "solve backward error {backward}");
        // Isolate expansion error. Cancellation can make the final coefficient
        // tiny, so bound roundoff by the absolute sum of its triple products.
        let gb = result.determinant.map(f64::from);
        let reference = determinant_coeffs(&gb);
        let mut absolute_terms = [0.0; 11];
        for [r0, r1, r2] in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..5 {
                        absolute_terms[i + j + k] +=
                            (gb[r0 * 13 + i] * gb[r1 * 13 + 4 + j] * gb[r2 * 13 + 8 + k]).abs();
                    }
                }
            }
        }
        eprintln!(
            "sample {sample}: isolated polynomial relative error={}",
            relative_error(&result.coefficients, &reference)
        );
        for i in 0..11 {
            let error = (f64::from(result.coefficients[i]) - reference[i]).abs();
            assert!(
                error <= 128.0 * f64::from(f32::EPSILON) * absolute_terms[i].max(1e-30),
                "coefficient {i}: error={error}, absolute terms={}",
                absolute_terms[i]
            );
        }
        // Independent determinant evaluation checks descending coefficient order.
        for z in [-1.3, 0.0, 0.7, 1.4] {
            let mut bz = Matrix3::zeros();
            for row in 0..3 {
                for (col, offset, count) in [(0, 0, 4), (1, 4, 4), (2, 8, 5)] {
                    bz[(row, col)] =
                        (0..count).fold(0.0, |acc, k| acc * z + b[row * 13 + offset + k]);
                }
            }
            let value = coeffs.iter().fold(0.0, |acc, c| acc * z + c);
            let bound = coeffs
                .iter()
                .fold(0.0, |acc, c| acc * z.abs() + c.abs())
                .max(1e-25);
            assert!((value - bz.determinant()).abs() / bound < 1e-10);
        }
    }
    assert!(pivot_swaps_needed > 0);
    assert_eq!(
        gpu.compute_algebra(&[[0.0; 36]])?[0].status,
        FivePointStatus::SingularElimination
    );
    assert_eq!(
        gpu.compute_algebra(&[[1e30; 36]])?[0].status,
        FivePointStatus::NonFinite
    );
    validate_roots(&gpu)?;
    let complete = gpu.solve_essential(&rays1, &rays2)?;
    let mut total_models = 0;
    let mut close_trials = 0;
    for (trial, result) in complete.iter().enumerate() {
        assert_eq!(result.trial, trial);
        assert_eq!(
            result.model_count,
            result
                .slots
                .iter()
                .filter(|s| s.essential.is_some())
                .count()
        );
        assert!(result.root_iterations <= 384);
        assert!(result.model_count <= 10);
        let left: Vec<_> = rays1[trial * 5..trial * 5 + 5]
            .iter()
            .map(|r| Vector3::new(r[0] as f64, r[1] as f64, r[2] as f64))
            .collect();
        let right: Vec<_> = rays2[trial * 5..trial * 5 + 5]
            .iter()
            .map(|r| Vector3::new(r[0] as f64, r[1] as f64, r[2] as f64))
            .collect();
        let cpu = crate::five_point::estimate_five_point_essential(&left, &right);
        let mut best = f64::INFINITY;
        for (slot, s) in result.slots.iter().enumerate() {
            assert_eq!(s.slot, slot);
            if let Some(e) = s.essential {
                let e = Matrix3::from_row_slice(&e.map(f64::from));
                assert!((e.norm() - 1.0).abs() < 2e-5);
                assert!(s.constraint_residual < 5e-4 && s.essential_residual <= 2e-2);
                let residual = (left
                    .iter()
                    .zip(&right)
                    .map(|(l, r)| r.dot(&(e * l)).powi(2))
                    .sum::<f64>()
                    / 5.0)
                    .sqrt();
                assert!(residual < 5e-4, "independent A*E residual={residual}");
                let cubic = 2.0 * e * e.transpose() * e - e;
                assert!(cubic.norm() < 2.01e-2);
                for c in &cpu {
                    best = best.min((e - c.normalize()).norm().min((e + c.normalize()).norm()));
                }
                total_models += 1;
            }
        }
        if best < 0.02 {
            close_trials += 1;
        }
        eprintln!("full trial {trial}: CPU={}, GPU={}, iterations={}, unconverged={}, complex={}, duplicate={}, rejected={}, best={best}",cpu.len(),result.model_count,result.root_iterations,result.unconverged,result.complex_filtered,result.duplicate_roots,result.recovery_rejected);
        assert_eq!(
            result.degree,
            result.model_count
                + result.unconverged
                + result.complex_filtered
                + result.duplicate_roots
                + result.recovery_rejected
                + result.duplicate_models
        );
    }
    assert!(
        total_models >= 16,
        "too few complete models: {total_models}"
    );
    assert!(
        close_trials >= 12,
        "only {close_trials}/16 trials have a nearby CPU model"
    );
    let degenerate = gpu.solve_essential(&[[0.0; 3]; 10], &[[0.0; 3]; 10])?;
    for (i, r) in degenerate.iter().enumerate() {
        assert_eq!(r.trial, i);
        assert_eq!(r.model_count, 0);
        assert_eq!(r.upstream_status, FivePointStatus::RankDeficient);
    }
    let mut mixed_left = rays1[..5].to_vec();
    mixed_left.extend([[0.0; 3]; 5]);
    mixed_left.extend_from_slice(&rays1[5..10]);
    let mut mixed_right = rays2[..5].to_vec();
    mixed_right.extend([[0.0; 3]; 5]);
    mixed_right.extend_from_slice(&rays2[5..10]);
    let mixed = gpu.solve_essential(&mixed_left, &mixed_right)?;
    assert_eq!(mixed[1].trial, 1);
    assert_eq!(mixed[1].model_count, 0);
    assert_eq!(mixed[1].upstream_status, FivePointStatus::RankDeficient);
    for (source, dest) in [(0, 0), (1, 2)] {
        for slot in 0..10 {
            assert_eq!(
                mixed[dest].slots[slot].essential,
                complete[source].slots[slot].essential
            );
        }
    }
    assert!(gpu
        .solve_essential(&[[f32::NAN; 3]; 5], &[[0.0; 3]; 5])
        .is_err());
    assert!(gpu.solve_essential(&[], &[])?.is_empty());
    eprintln!("worst normalized A*N={worst_residual}; worst [A, solve, B, coeff]={worst:?}; pivoted samples={pivot_swaps_needed}");
    Ok(())
}

#[test]
fn five_point_session_reuses_buffers_without_stale_state() -> Result<()> {
    let context = match WgpuContext::try_new_optional()? {
        Some(context) => context,
        None => {
            eprintln!("five-point GPU unavailable; skipping session test");
            return Ok(());
        }
    };
    let gpu = WgpuFivePointF32::from_context(context)?;
    let mut rng = StdRng::seed_from_u64(7);
    let mut make = |trials: usize| {
        let mut left = Vec::with_capacity(trials * 5);
        let mut right = Vec::with_capacity(trials * 5);
        for _ in 0..trials {
            for _ in 0..5 {
                let p = Vector3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(2.0..5.0),
                );
                let q = Rotation3::from_euler_angles(0.1, -0.2, 0.07) * p
                    + Vector3::new(0.7, -0.2, 0.3);
                left.push(p.normalize().map(|x| x as f32).into());
                right.push(q.normalize().map(|x| x as f32).into());
            }
        }
        (left, right)
    };
    let bits = |results: &[FivePointTrialResult]| -> Vec<u32> {
        results
            .iter()
            .flat_map(|r| {
                let mut words = vec![
                    r.trial as u32,
                    r.model_count as u32,
                    r.upstream_status as u32,
                    r.algebra_status as u32,
                ];
                for s in &r.slots {
                    words.push(s.status as u32);
                    if let Some(e) = s.essential {
                        words.extend(e.map(f32::to_bits));
                    }
                }
                words
            })
            .collect()
    };

    let (small_l, small_r) = make(3);
    let (large_l, large_r) = make(40);
    let oneshot_small = gpu.solve_essential(&small_l, &small_r)?;
    let oneshot_large = gpu.solve_essential(&large_l, &large_r)?;

    let mut session = gpu.session(1)?;
    // small → large → small, and success → fail → success on the reused buffers.
    let s1 = session.solve_essential(&small_l, &small_r)?;
    assert_eq!(bits(&s1), bits(&oneshot_small), "small first call");
    assert!(session.capacity() >= 3);

    let s2 = session.solve_essential(&large_l, &large_r)?;
    assert_eq!(bits(&s2), bits(&oneshot_large), "large after grow");
    let grown = session.capacity();
    assert!(grown >= 40);

    let s3 = session.solve_essential(&small_l, &small_r)?;
    assert_eq!(
        bits(&s3),
        bits(&oneshot_small),
        "small after large (no stale)"
    );
    assert_eq!(session.capacity(), grown, "capacity is grow-only");

    let degenerate = session.solve_essential(&[[0.0; 3]; 10], &[[0.0; 3]; 10])?;
    assert!(degenerate
        .iter()
        .all(|r| r.upstream_status == FivePointStatus::RankDeficient && r.model_count == 0));

    let s4 = session.solve_essential(&small_l, &small_r)?;
    assert_eq!(bits(&s4), bits(&oneshot_small), "success after failure");

    assert!(session.solve_essential(&[], &[])?.is_empty());
    // Non-full workgroup: 33 trials ⇒ ceil(33/32)=2 groups with a partial lane.
    let (odd_l, odd_r) = make(33);
    let oneshot_odd = gpu.solve_essential(&odd_l, &odd_r)?;
    let s5 = session.solve_essential(&odd_l, &odd_r)?;
    assert_eq!(bits(&s5), bits(&oneshot_odd), "partial workgroup");
    // P3: one submit + one wait per unprofiled solve (compute+copy merged).
    session.reset_sync_counters();
    let n = 5usize;
    for _ in 0..n {
        let _ = session.solve_essential(&small_l, &small_r)?;
    }
    assert_eq!(
        session.submit_count(),
        n as u64,
        "expected one submit per solve"
    );
    assert_eq!(
        session.wait_count(),
        n as u64,
        "expected one wait per solve"
    );

    // P3-B: bounded double-buffer matches sequential results and still 1 wait/submit per batch.
    session.enable_double_buffer()?;
    assert!(session.double_buffer_enabled());
    let (b0_l, b0_r) = make(8);
    let (b1_l, b1_r) = make(8);
    let (b2_l, b2_r) = make(5);
    let seq0 = gpu.solve_essential(&b0_l, &b0_r)?;
    let seq1 = gpu.solve_essential(&b1_l, &b1_r)?;
    let seq2 = gpu.solve_essential(&b2_l, &b2_r)?;
    let mut expected = seq0;
    let base1 = expected.len();
    expected.extend(seq1.into_iter().map(|mut r| {
        r.trial += base1;
        r
    }));
    let base2 = expected.len();
    expected.extend(seq2.into_iter().map(|mut r| {
        r.trial += base2;
        r
    }));
    session.reset_sync_counters();
    let piped = session.solve_essential_batches(&[(b0_l, b0_r), (b1_l, b1_r), (b2_l, b2_r)])?;
    assert_eq!(bits(&piped), bits(&expected), "double-buffer vs sequential");
    assert_eq!(session.submit_count(), 3);
    assert_eq!(session.wait_count(), 3);

    // Mid-pipeline growth: batch 3 exceeds capacity while batch 2 is in
    // flight; the session must drain before rebuilding slots, or batch 2
    // would be decoded from a freshly zeroed staging buffer.
    let mut grow = gpu.session(1)?;
    grow.enable_double_buffer()?;
    let (g0_l, g0_r) = make(3);
    let (g1_l, g1_r) = make(8);
    let (g2_l, g2_r) = make(40);
    let (g3_l, g3_r) = make(6);
    let mut grow_expected = Vec::new();
    for (l, r) in [
        (&g0_l, &g0_r),
        (&g1_l, &g1_r),
        (&g2_l, &g2_r),
        (&g3_l, &g3_r),
    ] {
        let base = grow_expected.len();
        grow_expected.extend(gpu.solve_essential(l, r)?.into_iter().map(|mut t| {
            t.trial += base;
            t
        }));
    }
    grow.reset_sync_counters();
    let grow_piped =
        grow.solve_essential_batches(&[(g0_l, g0_r), (g1_l, g1_r), (g2_l, g2_r), (g3_l, g3_r)])?;
    assert_eq!(
        bits(&grow_piped),
        bits(&grow_expected),
        "double-buffer with mid-pipeline capacity growth"
    );
    assert!(grow.capacity() >= 40);
    assert_eq!(
        grow.submit_count(),
        4,
        "growth must not reset submit counter"
    );
    assert_eq!(grow.wait_count(), 4, "growth must not reset wait counter");
    Ok(())
}
