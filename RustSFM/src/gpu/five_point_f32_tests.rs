use super::*;
use crate::five_point_generated::{build_elimination_matrix, determinant_coeffs};
use nalgebra::{DMatrix, Matrix3, Rotation3, Vector3};
use rand::{rngs::StdRng, Rng, SeedableRng};

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
        assert!(d.relative_off_diagonal <= 2.01e-8);
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
    let mut duplicate = matrices[0];
    let first: [f32; 9] = duplicate[..9].try_into().unwrap();
    duplicate[9..18].copy_from_slice(&first);
    let mut near = duplicate;
    near[9] += 1e-7;
    let invalid = [[0.0; 45], [1.0; 45], duplicate, near];
    for d in gpu.compute_nullspace_diagnostics(&invalid)? {
        assert_eq!(d.status, FivePointStatus::RankDeficient, "{d:?}");
        assert!(d.basis.is_none());
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
