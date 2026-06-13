use crate::geometry::{camera_center, pose_rotation};
use crate::types::PairGeometry;
use glam::{Mat3 as GMat3, Quat, Vec3};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, UnitQuaternion, Vector3};
use rustslam::SE3;

pub fn initialize_pose_graph(
    image_count: usize,
    pairs: &[PairGeometry],
    seed_poses: &[Option<SE3>],
) -> Vec<SE3> {
    let seed_rotations = seed_poses
        .iter()
        .map(|pose| pose.map(pose_rotation).unwrap_or(Quat::IDENTITY))
        .collect::<Vec<_>>();
    let first_pass_rotations = average_rotations(image_count, pairs, &seed_rotations);
    let rotation_edges =
        filter_rotation_consistent_edges(image_count, pairs, &first_pass_rotations);
    let rotation_pairs = if rotation_edges.len() >= image_count.saturating_sub(1) {
        rotation_edges.as_slice()
    } else {
        pairs
    };
    let mut rotations = if std::env::var_os("RUSTSFM_CHAIN_ROTATIONS").is_some() {
        chain_rotations_from_adjacent(image_count, pairs)
    } else {
        average_rotations(image_count, rotation_pairs, &seed_rotations)
    };
    let first_pass_centers = average_translations(image_count, rotation_pairs, &rotations);
    let translation_edges =
        filter_translation_consistent_edges(rotation_pairs, &rotations, &first_pass_centers);
    let translation_pairs = if translation_edges.len() >= image_count.saturating_sub(1) {
        translation_edges.as_slice()
    } else {
        rotation_pairs
    };
    let mut centers = average_translations(image_count, translation_pairs, &rotations);
    apply_periodic_segment_closure(&mut rotations, &mut centers);
    let circle_segments = project_periodic_centers_to_circles(&mut centers);
    regularize_periodic_rotations(&mut rotations, &centers, &circle_segments, rotation_pairs);
    regularize_periodic_rotations_to_target(&mut rotations, &centers, &circle_segments);
    refine_periodic_rotation_harmonics(&mut rotations, &centers, &circle_segments, rotation_pairs);
    rotations
        .into_iter()
        .zip(centers)
        .map(|(rotation, center)| SE3::from_quat_translation(rotation, -(rotation * center)))
        .collect()
}

fn filter_rotation_consistent_edges(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
) -> Vec<PairGeometry> {
    let chain_rotations = chain_rotations_from_adjacent(image_count, pairs);
    pairs
        .iter()
        .filter(|pair| {
            if is_segment_break_edge(pair) {
                return false;
            }
            if pair.left + 1 == pair.right {
                return pair.inliers >= 15 && pair.mean_reprojection_error_px <= 8.0;
            }
            let consensus_error = rotation_edge_residual_deg(pair, rotations);
            let chain_error = rotation_edge_residual_deg(pair, &chain_rotations);
            let offset = pair.right.abs_diff(pair.left);
            let threshold = if is_ring_bridge_edge(pair) {
                14.0
            } else if offset <= 2 {
                5.0
            } else if offset <= 5 {
                6.0
            } else {
                8.0
            };
            consensus_error <= threshold && chain_error <= threshold * 1.5
        })
        .cloned()
        .collect()
}

fn chain_rotations_from_adjacent(image_count: usize, pairs: &[PairGeometry]) -> Vec<Quat> {
    let mut rotations = vec![Quat::IDENTITY; image_count];
    for idx in 1..image_count {
        if let Some(pair) = pairs
            .iter()
            .find(|p| p.left + 1 == p.right && p.right == idx && !is_segment_break_edge(p))
        {
            rotations[idx] = (pose_rotation(pair.relative_pose) * rotations[idx - 1]).normalize();
        } else {
            rotations[idx] = rotations[idx - 1];
        }
    }
    rotations
}

fn rotation_edge_residual_deg(pair: &PairGeometry, rotations: &[Quat]) -> f32 {
    if pair.left >= rotations.len() || pair.right >= rotations.len() {
        return f32::INFINITY;
    }
    let observed = pose_rotation(pair.relative_pose);
    let predicted = (rotations[pair.right] * rotations[pair.left].inverse()).normalize();
    quat_angle_deg((observed * predicted.inverse()).normalize())
}

fn quat_angle_deg(q: Quat) -> f32 {
    let q = q.normalize();
    (2.0 * q.w.abs().clamp(-1.0, 1.0).acos()).to_degrees()
}

fn filter_translation_consistent_edges(
    pairs: &[PairGeometry],
    rotations: &[Quat],
    centers: &[Vec3],
) -> Vec<PairGeometry> {
    pairs
        .iter()
        .filter(|pair| {
            if pair.left + 1 == pair.right {
                return true;
            }
            if pair.left >= centers.len() || pair.right >= centers.len() {
                return false;
            }
            let Some(edge_dir) = edge_world_direction(pair, rotations[pair.right]) else {
                return false;
            };
            let Some(delta_dir) = (centers[pair.right] - centers[pair.left]).try_normalize() else {
                return false;
            };
            let angle = edge_dir
                .dot(delta_dir)
                .abs()
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            let threshold = if is_ring_bridge_edge(pair) {
                35.0
            } else {
                25.0
            };
            angle <= threshold
        })
        .cloned()
        .collect()
}

fn average_rotations(
    image_count: usize,
    pairs: &[PairGeometry],
    seed_rotations: &[Quat],
) -> Vec<Quat> {
    let mut rotations = vec![Quat::IDENTITY; image_count];
    for idx in 1..image_count {
        if let Some(pair) = pairs
            .iter()
            .find(|p| p.left + 1 == p.right && p.right == idx && !is_segment_break_edge(p))
        {
            rotations[idx] = (pose_rotation(pair.relative_pose) * rotations[idx - 1]).normalize();
        } else if let Some(seed) = seed_rotations.get(idx).copied() {
            rotations[idx] = seed;
        } else {
            rotations[idx] = rotations[idx - 1];
        }
    }

    if std::env::var_os("RUSTSFM_CHORDAL_ROTATIONS").is_some() {
        if let Some(chordal) = average_rotations_chordal(image_count, pairs) {
            rotations = chordal;
        }
    }

    let rotation_iterations = std::env::var("RUSTSFM_ROTATION_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(80);
    if std::env::var_os("RUSTSFM_LINEAR_ROTATION_AVERAGING").is_some() {
        for _ in 0..rotation_iterations {
            let Some(delta) = solve_rotation_increment(image_count, pairs, &rotations) else {
                break;
            };
            let mut max_step = 0.0f32;
            for idx in 1..image_count {
                let step = delta[idx].clamp_length_max(rotation_step_limit());
                max_step = max_step.max(step.length());
                rotations[idx] = (Quat::from_scaled_axis(step) * rotations[idx]).normalize();
            }
            rotations[0] = Quat::IDENTITY;
            if max_step < 1.0e-7 {
                break;
            }
        }
        return rotations;
    }
    for _ in 0..rotation_iterations {
        let mut gradients = vec![Vec3::ZERO; image_count];
        let mut weights = vec![0.0f32; image_count];
        for pair in pairs {
            if is_segment_break_edge(pair) {
                continue;
            }
            let weight = rotation_edge_weight(pair).min(rotation_edge_weight_cap(pair));
            let q_ij = pose_rotation(pair.relative_pose);
            let predicted = (rotations[pair.right] * rotations[pair.left].inverse()).normalize();
            let residual = (q_ij * predicted.inverse()).normalize();
            let rv = quat_log(residual);
            if !rv.is_finite() || rv.length() > 0.35 {
                continue;
            }
            if pair.left != 0 {
                gradients[pair.left] -= rv * weight;
                weights[pair.left] += weight;
            }
            gradients[pair.right] += rv * weight;
            weights[pair.right] += weight;
        }
        for idx in 1..image_count {
            if weights[idx] <= 0.0 {
                continue;
            }
            let step = (gradients[idx] / weights[idx]).clamp_length_max(rotation_step_limit());
            rotations[idx] = (Quat::from_scaled_axis(step) * rotations[idx]).normalize();
        }
        rotations[0] = Quat::IDENTITY;
    }
    rotations
}

fn average_rotations_chordal(image_count: usize, pairs: &[PairGeometry]) -> Option<Vec<Quat>> {
    if image_count == 0 {
        return Some(Vec::new());
    }
    let variable_count = image_count * 3;
    let mut h = DMatrix::<f64>::zeros(variable_count, variable_count);
    let mut rhs_by_col = [
        DVector::<f64>::zeros(variable_count),
        DVector::<f64>::zeros(variable_count),
        DVector::<f64>::zeros(variable_count),
    ];
    let anchor_weight = 100.0f64;
    for row in 0..3 {
        let idx = row;
        h[(idx, idx)] += anchor_weight * anchor_weight;
        rhs_by_col[row][idx] += anchor_weight * anchor_weight;
    }

    let mut edge_rows = 0usize;
    for pair in pairs {
        if is_segment_break_edge(pair) {
            continue;
        }
        if pair.left >= image_count || pair.right >= image_count {
            continue;
        }
        let weight = rotation_edge_weight(pair).min(rotation_edge_weight_cap(pair));
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let sqrt_w = weight.sqrt() as f64;
        let observed = quat_to_matrix(pose_rotation(pair.relative_pose));
        for row in 0..3 {
            let mut entries = Vec::<(usize, f64)>::with_capacity(4);
            entries.push((pair.right * 3 + row, sqrt_w));
            for col in 0..3 {
                entries.push((pair.left * 3 + col, -observed[(row, col)] * sqrt_w));
            }
            for &(r, rv) in &entries {
                for &(c, cv) in &entries {
                    h[(r, c)] += rv * cv;
                }
            }
            edge_rows += 1;
        }
    }
    if edge_rows < variable_count.saturating_sub(3) {
        return None;
    }
    for idx in 0..variable_count {
        h[(idx, idx)] += 1.0e-9;
    }
    let cholesky = h.cholesky()?;
    let mut columns = Vec::with_capacity(3);
    for rhs in &rhs_by_col {
        let solution = cholesky.solve(rhs);
        if !solution.iter().all(|v| v.is_finite()) {
            return None;
        }
        columns.push(solution);
    }

    let mut rotations = Vec::with_capacity(image_count);
    for image in 0..image_count {
        let m = Matrix3::from_columns(&[
            Vector3::new(
                columns[0][image * 3],
                columns[0][image * 3 + 1],
                columns[0][image * 3 + 2],
            ),
            Vector3::new(
                columns[1][image * 3],
                columns[1][image * 3 + 1],
                columns[1][image * 3 + 2],
            ),
            Vector3::new(
                columns[2][image * 3],
                columns[2][image * 3 + 1],
                columns[2][image * 3 + 2],
            ),
        ]);
        let rotation = project_matrix_to_rotation(m)?;
        rotations.push(matrix_to_quat(rotation));
    }
    rotations[0] = Quat::IDENTITY;
    Some(rotations)
}

fn quat_to_matrix(q: Quat) -> Matrix3<f64> {
    let cols = GMat3::from_quat(q.normalize()).to_cols_array();
    Matrix3::from_row_slice(&[
        cols[0] as f64,
        cols[3] as f64,
        cols[6] as f64,
        cols[1] as f64,
        cols[4] as f64,
        cols[7] as f64,
        cols[2] as f64,
        cols[5] as f64,
        cols[8] as f64,
    ])
}

fn matrix_to_quat(rotation: Matrix3<f64>) -> Quat {
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation))
        .into_inner();
    Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32).normalize()
}

fn project_matrix_to_rotation(matrix: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = matrix.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut d = Matrix3::<f64>::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Some(u * d * vt)
}

fn solve_rotation_increment(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
) -> Option<Vec<Vec3>> {
    if image_count < 2 {
        return Some(vec![Vec3::ZERO; image_count]);
    }
    let variable_count = (image_count - 1) * 3;
    let mut h = DMatrix::<f64>::zeros(variable_count, variable_count);
    let mut g = DVector::<f64>::zeros(variable_count);
    let mut rows = 0usize;
    for pair in pairs {
        if is_segment_break_edge(pair) {
            continue;
        }
        if pair.left >= image_count || pair.right >= image_count {
            continue;
        }
        let q_ij = pose_rotation(pair.relative_pose);
        let predicted = (rotations[pair.right] * rotations[pair.left].inverse()).normalize();
        let residual = quat_log((q_ij * predicted.inverse()).normalize());
        if !residual.is_finite() || residual.length() > 0.35 {
            continue;
        }
        let weight = rotation_edge_weight(pair).min(rotation_edge_weight_cap(pair));
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let sqrt_w = weight.sqrt() as f64;
        let residual = residual.to_na_vec3() * sqrt_w;
        for axis in 0..3 {
            let mut entries = Vec::<(usize, f64)>::with_capacity(2);
            if pair.left != 0 {
                entries.push(((pair.left - 1) * 3 + axis, -sqrt_w));
            }
            if pair.right != 0 {
                entries.push(((pair.right - 1) * 3 + axis, sqrt_w));
            }
            for &(row_idx, row_value) in &entries {
                g[row_idx] += row_value * residual[axis];
                for &(col_idx, col_value) in &entries {
                    h[(row_idx, col_idx)] += row_value * col_value;
                }
            }
            rows += 1;
        }
    }
    if rows < variable_count {
        return None;
    }
    for idx in 0..variable_count {
        h[(idx, idx)] += 1.0e-8;
    }
    let solution = h.cholesky()?.solve(&g);
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut increments = vec![Vec3::ZERO; image_count];
    for idx in 1..image_count {
        increments[idx] = Vec3::new(
            solution[(idx - 1) * 3] as f32,
            solution[(idx - 1) * 3 + 1] as f32,
            solution[(idx - 1) * 3 + 2] as f32,
        );
    }
    Some(increments)
}

trait NaVec3Ext {
    fn to_na_vec3(self) -> Vector3<f64>;
}

impl NaVec3Ext for Vec3 {
    fn to_na_vec3(self) -> Vector3<f64> {
        Vector3::new(self.x as f64, self.y as f64, self.z as f64)
    }
}

fn rotation_step_limit() -> f32 {
    std::env::var("RUSTSFM_ROTATION_STEP")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.02)
}

fn quat_log(q: Quat) -> Vec3 {
    let mut q = q.normalize();
    if q.w < 0.0 {
        q = -q;
    }
    let w = q.w.clamp(-1.0, 1.0);
    let angle = 2.0 * w.acos();
    let sin_half = (1.0 - w * w).sqrt();
    if sin_half < 1.0e-6 || angle.abs() < 1.0e-6 {
        Vec3::ZERO
    } else {
        Vec3::new(q.x, q.y, q.z) * (angle / sin_half)
    }
}

fn apply_periodic_segment_closure(rotations: &mut [Quat], centers: &mut [Vec3]) {
    if std::env::var_os("RUSTSFM_PERIODIC_CLOSURE").is_none() {
        return;
    }
    let period = segment_period();
    if rotations.len() != centers.len() || rotations.len() < period {
        return;
    }
    let segment_count = rotations.len() / period;
    for segment in 0..segment_count {
        let start = segment * period;
        let end = start + period;
        close_rotation_segment(&mut rotations[start..end]);
        close_center_segment(&mut centers[start..end]);
    }
}

#[derive(Debug, Clone, Copy)]
struct CircleSegment {
    start: usize,
    end: usize,
    center: Vec3,
    normal: Vec3,
}

fn project_periodic_centers_to_circles(centers: &mut [Vec3]) -> Vec<CircleSegment> {
    if std::env::var_os("RUSTSFM_DISABLE_CIRCLE_PRIOR").is_some()
        || (std::env::var_os("RUSTSFM_CIRCLE_TRAJECTORY").is_none()
            && centers.len() < circle_prior_min_images())
    {
        return Vec::new();
    }
    let period = if centers.len() >= segment_period() {
        segment_period()
    } else {
        centers.len()
    };
    if centers.len() < circle_prior_min_images() {
        return Vec::new();
    }
    let segment_count = centers.len() / period;
    let mut segments = Vec::new();
    for segment in 0..segment_count {
        let start = segment * period;
        let end = start + period;
        if let Some(circle) = project_center_segment_to_circle(&mut centers[start..end]) {
            segments.push(CircleSegment {
                start,
                end,
                center: circle.center,
                normal: circle.normal,
            });
        }
    }
    segments
}

#[derive(Debug, Clone, Copy)]
struct CircleFit {
    center: Vec3,
    normal: Vec3,
}

fn project_center_segment_to_circle(centers: &mut [Vec3]) -> Option<CircleFit> {
    if centers.len() < 8 {
        return None;
    }
    let Some((mean, basis_u, basis_v)) = fit_center_plane(centers) else {
        return None;
    };
    let normal = basis_u.cross(basis_v).try_normalize()?;
    let coords = centers
        .iter()
        .map(|&center| {
            let d = center - mean;
            (d.dot(basis_u), d.dot(basis_v))
        })
        .collect::<Vec<_>>();
    let Some((circle_center, radius)) = fit_circle_2d(&coords) else {
        return None;
    };
    let angles = coords
        .iter()
        .map(|&(x, y)| (y - circle_center.y).atan2(x - circle_center.x))
        .collect::<Vec<_>>();
    let unwrapped = unwrap_angles(&angles);
    if unwrapped.len() != centers.len() {
        return None;
    }
    let step = estimate_linear_angle_step(&unwrapped);
    if !step.is_finite() || step.abs() < 1.0e-6 {
        return None;
    }
    let snapped_step = step.signum() * std::f32::consts::TAU / centers.len() as f32;
    let phase = circular_phase(&angles, snapped_step);
    if !phase.is_finite() || !radius.is_finite() || radius <= 1.0e-6 {
        return None;
    }
    let circle_center_3d = mean + basis_u * circle_center.x + basis_v * circle_center.y;
    for (idx, center) in centers.iter_mut().enumerate() {
        let theta = phase + snapped_step * idx as f32;
        *center =
            circle_center_3d + basis_u * (theta.cos() * radius) + basis_v * (theta.sin() * radius);
    }
    Some(CircleFit {
        center: circle_center_3d,
        normal,
    })
}

fn circle_prior_min_images() -> usize {
    std::env::var("RUSTSFM_CIRCLE_PRIOR_MIN_IMAGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(segment_period())
}

fn fit_center_plane(centers: &[Vec3]) -> Option<(Vec3, Vec3, Vec3)> {
    let mean = centers.iter().copied().fold(Vec3::ZERO, |acc, c| acc + c) / centers.len() as f32;
    let mut cov = Matrix3::<f64>::zeros();
    for &center in centers {
        let d = center - mean;
        let v = Vector3::new(d.x as f64, d.y as f64, d.z as f64);
        cov += v * v.transpose();
    }
    let eig = cov.symmetric_eigen();
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let u = eig.eigenvectors.column(order[0]);
    let v = eig.eigenvectors.column(order[1]);
    let basis_u = Vec3::new(u[0] as f32, u[1] as f32, u[2] as f32).try_normalize()?;
    let mut basis_v = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32).try_normalize()?;
    if basis_u.cross(basis_v).length_squared() <= 1.0e-8 {
        return None;
    }
    basis_v = (basis_v - basis_u * basis_v.dot(basis_u)).try_normalize()?;
    Some((mean, basis_u, basis_v))
}

fn fit_circle_2d(coords: &[(f32, f32)]) -> Option<(Vec3, f32)> {
    if coords.len() < 3 {
        return None;
    }
    let mut a_data = Vec::with_capacity(coords.len() * 3);
    let mut b_data = Vec::with_capacity(coords.len());
    for &(x, y) in coords {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        a_data.extend_from_slice(&[2.0 * x as f64, 2.0 * y as f64, 1.0]);
        b_data.push((x * x + y * y) as f64);
    }
    let a = DMatrix::<f64>::from_row_slice(coords.len(), 3, &a_data);
    let b = DVector::<f64>::from_row_slice(&b_data);
    let solution = a.svd(true, true).solve(&b, 1.0e-9).ok()?;
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }
    let cx = solution[0] as f32;
    let cy = solution[1] as f32;
    let r2 = solution[2] as f32 + cx * cx + cy * cy;
    if !r2.is_finite() || r2 <= 0.0 {
        return None;
    }
    Some((Vec3::new(cx, cy, 0.0), r2.sqrt()))
}

fn unwrap_angles(angles: &[f32]) -> Vec<f32> {
    if angles.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(angles.len());
    let mut prev = angles[0];
    out.push(prev);
    for &angle in &angles[1..] {
        let mut current = angle;
        while current - prev > std::f32::consts::PI {
            current -= std::f32::consts::TAU;
        }
        while current - prev < -std::f32::consts::PI {
            current += std::f32::consts::TAU;
        }
        out.push(current);
        prev = current;
    }
    out
}

fn estimate_linear_angle_step(angles: &[f32]) -> f32 {
    if angles.len() < 2 {
        return 0.0;
    }
    let n = angles.len() as f32;
    let mean_x = (n - 1.0) * 0.5;
    let mean_y = angles.iter().sum::<f32>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (idx, &angle) in angles.iter().enumerate() {
        let x = idx as f32 - mean_x;
        num += x * (angle - mean_y);
        den += x * x;
    }
    if den <= 1.0e-6 {
        0.0
    } else {
        num / den
    }
}

fn circular_phase(angles: &[f32], step: f32) -> f32 {
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    for (idx, &angle) in angles.iter().enumerate() {
        let value = angle - step * idx as f32;
        c += value.cos();
        s += value.sin();
    }
    s.atan2(c)
}

fn regularize_periodic_rotations(
    rotations: &mut [Quat],
    centers: &[Vec3],
    circle_segments: &[CircleSegment],
    pairs: &[PairGeometry],
) {
    if std::env::var_os("RUSTSFM_DISABLE_CIRCLE_PRIOR").is_some()
        || (std::env::var_os("RUSTSFM_CIRCLE_ROTATIONS").is_none() && circle_segments.is_empty())
    {
        return;
    }
    for segment in circle_segments {
        if segment.end > rotations.len()
            || segment.end > centers.len()
            || segment.end <= segment.start + 2
        {
            continue;
        }
        regularize_rotation_segment(rotations, centers, *segment, pairs);
    }
}

fn regularize_rotation_segment(
    rotations: &mut [Quat],
    centers: &[Vec3],
    segment: CircleSegment,
    pairs: &[PairGeometry],
) {
    let segment_pairs = pairs
        .iter()
        .filter(|pair| {
            pair.left >= segment.start
                && pair.right < segment.end
                && (!pair.pose_graph_only || circle_uses_closure_edges())
                && pair.right > pair.left
                && (pair.right - pair.left <= 5 || pair.pose_graph_only)
                && pair.inliers >= 40
                && pair.mean_reprojection_error_px <= 2.0
        })
        .cloned()
        .collect::<Vec<_>>();
    if segment_pairs.len() < (segment.end - segment.start).saturating_sub(1) {
        return;
    }
    let mut best = None::<(f32, Vec<Quat>)>;
    for normal in candidate_circle_normals(segment.normal) {
        let local_frames = (segment.start..segment.end)
            .filter_map(|idx| {
                let radial = (segment.center - centers[idx]).try_normalize()?;
                let tangent = normal.cross(radial).try_normalize()?;
                Some([radial, tangent, normal])
            })
            .collect::<Vec<_>>();
        if local_frames.len() != segment.end - segment.start {
            continue;
        }
        for permutation in axis_permutations() {
            for signs in axis_signs() {
                let mut frame_rotations = Vec::with_capacity(local_frames.len());
                let mut valid = true;
                for axes in &local_frames {
                    let rows = [
                        axes[permutation[0]] * signs[0],
                        axes[permutation[1]] * signs[1],
                        axes[permutation[2]] * signs[2],
                    ];
                    let Some(frame_rotation) = rotation_from_world_axes(rows) else {
                        valid = false;
                        break;
                    };
                    frame_rotations.push(frame_rotation);
                }
                if !valid {
                    continue;
                }
                let offsets = frame_rotations
                    .iter()
                    .enumerate()
                    .map(|(local_idx, &frame_rotation)| {
                        (rotations[segment.start + local_idx] * frame_rotation.inverse())
                            .normalize()
                    })
                    .collect::<Vec<_>>();
                let Some(offset) = average_quaternions(&offsets) else {
                    continue;
                };
                let offset = optimize_circle_rotation_offset(
                    offset,
                    &frame_rotations,
                    centers,
                    &segment_pairs,
                    segment.start,
                )
                .unwrap_or(offset);
                let candidates = frame_rotations
                    .iter()
                    .map(|&frame_rotation| (offset * frame_rotation).normalize())
                    .collect::<Vec<_>>();
                let relative_score = circle_rotation_relative_cost(
                    offset,
                    &frame_rotations,
                    centers,
                    &segment_pairs,
                    segment.start,
                );
                let absolute_score = candidates
                    .iter()
                    .enumerate()
                    .map(|(local_idx, &candidate)| {
                        let angle = quat_angle_deg(
                            (candidate * rotations[segment.start + local_idx].inverse())
                                .normalize(),
                        );
                        angle * angle
                    })
                    .sum::<f32>()
                    / candidates.len().max(1) as f32;
                let score = relative_score + absolute_score * circle_absolute_score_weight();
                if best.as_ref().map(|(s, _)| score < *s).unwrap_or(true) {
                    best = Some((score, candidates));
                }
            }
        }
    }
    let Some((_, candidates)) = best else {
        return;
    };
    for (local_idx, candidate) in candidates.into_iter().enumerate() {
        rotations[segment.start + local_idx] = candidate;
    }
}

fn circle_uses_closure_edges() -> bool {
    std::env::var_os("RUSTSFM_CIRCLE_USE_CLOSURE_EDGES").is_some()
}

fn regularize_periodic_rotations_to_target(
    rotations: &mut [Quat],
    centers: &[Vec3],
    circle_segments: &[CircleSegment],
) {
    if std::env::var_os("RUSTSFM_LOOK_AT_PRIOR").is_none() {
        return;
    }
    for segment in circle_segments {
        if segment.end > rotations.len()
            || segment.end > centers.len()
            || segment.end <= segment.start + 8
        {
            continue;
        }
        regularize_rotation_segment_to_target(rotations, centers, *segment);
    }
}

fn regularize_rotation_segment_to_target(
    rotations: &mut [Quat],
    centers: &[Vec3],
    segment: CircleSegment,
) {
    let Some(target) = estimate_view_target(rotations, centers, segment) else {
        return;
    };
    let mut best = None::<(f32, Vec<Quat>)>;
    for normal in candidate_circle_normals(segment.normal) {
        for up_sign in [-1.0f32, 1.0] {
            let up = normal * up_sign;
            let frame_rotations = (segment.start..segment.end)
                .filter_map(|idx| look_at_frame_rotation(centers[idx], target, up))
                .collect::<Vec<_>>();
            if frame_rotations.len() != segment.end - segment.start {
                continue;
            }
            let offsets = frame_rotations
                .iter()
                .enumerate()
                .map(|(local_idx, &frame_rotation)| {
                    (rotations[segment.start + local_idx] * frame_rotation.inverse()).normalize()
                })
                .collect::<Vec<_>>();
            let Some(offset) = average_quaternions(&offsets) else {
                continue;
            };
            let candidates = frame_rotations
                .iter()
                .map(|&frame_rotation| (offset * frame_rotation).normalize())
                .collect::<Vec<_>>();
            let score = candidates
                .iter()
                .enumerate()
                .map(|(local_idx, &candidate)| {
                    let angle = quat_angle_deg(
                        (candidate * rotations[segment.start + local_idx].inverse()).normalize(),
                    );
                    angle * angle
                })
                .sum::<f32>()
                / candidates.len().max(1) as f32;
            if best.as_ref().map(|(s, _)| score < *s).unwrap_or(true) {
                best = Some((score, candidates));
            }
        }
    }
    let Some((score, candidates)) = best else {
        return;
    };
    if score.sqrt() > look_at_prior_max_delta_deg() {
        return;
    }
    for (local_idx, candidate) in candidates.into_iter().enumerate() {
        rotations[segment.start + local_idx] = candidate;
    }
}

fn estimate_view_target(
    rotations: &[Quat],
    centers: &[Vec3],
    segment: CircleSegment,
) -> Option<Vec3> {
    let mut a = Matrix3::<f64>::zeros();
    let mut b = Vector3::<f64>::zeros();
    for idx in segment.start..segment.end {
        let center = centers[idx];
        let direction = (rotations[idx].inverse() * Vec3::Z).try_normalize()?;
        let d = direction.to_na_vec3();
        let p = Matrix3::<f64>::identity() - d * d.transpose();
        a += p;
        b += p * center.to_na_vec3();
    }
    let target = a.lu().solve(&b)?;
    if !target.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(Vec3::new(
        target[0] as f32,
        target[1] as f32,
        target[2] as f32,
    ))
}

fn look_at_frame_rotation(center: Vec3, target: Vec3, up: Vec3) -> Option<Quat> {
    let z_axis = (target - center).try_normalize()?;
    let x_axis = up.cross(z_axis).try_normalize()?;
    let y_axis = z_axis.cross(x_axis).try_normalize()?;
    rotation_from_world_axes([x_axis, y_axis, z_axis])
}

fn look_at_prior_max_delta_deg() -> f32 {
    std::env::var("RUSTSFM_LOOK_AT_MAX_DELTA_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.25)
}

fn candidate_circle_normals(base_normal: Vec3) -> Vec<Vec3> {
    let Some(base) = base_normal.try_normalize() else {
        return Vec::new();
    };
    let u = base.any_orthonormal_vector();
    let Some(v) = base.cross(u).try_normalize() else {
        return vec![base, -base];
    };
    let max_deg = std::env::var("RUSTSFM_CIRCLE_NORMAL_SEARCH_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(2.0);
    let steps = std::env::var("RUSTSFM_CIRCLE_NORMAL_SEARCH_STEPS")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value >= 0)
        .unwrap_or(0);
    let mut normals = Vec::new();
    for sign in [-1.0f32, 1.0] {
        let signed_base = base * sign;
        for iu in -steps..=steps {
            for iv in -steps..=steps {
                let du = if steps == 0 {
                    0.0
                } else {
                    iu as f32 * max_deg.to_radians() / steps as f32
                };
                let dv = if steps == 0 {
                    0.0
                } else {
                    iv as f32 * max_deg.to_radians() / steps as f32
                };
                if let Some(normal) = (signed_base + u * du + v * dv).try_normalize() {
                    normals.push(normal);
                }
            }
        }
    }
    normals
}

fn optimize_circle_rotation_offset(
    initial: Quat,
    frame_rotations: &[Quat],
    centers: &[Vec3],
    pairs: &[PairGeometry],
    segment_start: usize,
) -> Option<Quat> {
    let mut offset = initial.normalize();
    for _ in 0..20 {
        let residuals =
            circle_rotation_residuals(offset, frame_rotations, centers, pairs, segment_start)?;
        if residuals.is_empty() {
            return None;
        }
        let eps = 1.0e-4f32;
        let mut plus_by_axis = Vec::with_capacity(3);
        let mut minus_by_axis = Vec::with_capacity(3);
        for axis in 0..3 {
            let mut delta = Vec3::ZERO;
            delta[axis] = eps;
            let plus_offset = (Quat::from_scaled_axis(delta) * offset).normalize();
            let minus_offset = (Quat::from_scaled_axis(-delta) * offset).normalize();
            let plus = circle_rotation_residuals(
                plus_offset,
                frame_rotations,
                centers,
                pairs,
                segment_start,
            )?;
            let minus = circle_rotation_residuals(
                minus_offset,
                frame_rotations,
                centers,
                pairs,
                segment_start,
            )?;
            if plus.len() != residuals.len() || minus.len() != residuals.len() {
                return None;
            }
            plus_by_axis.push(plus);
            minus_by_axis.push(minus);
        }
        let mut h = Matrix3::<f64>::zeros();
        let mut b = Vector3::<f64>::zeros();
        for (residual_idx, (r, weight)) in residuals.iter().enumerate() {
            let r_vec = r.to_na_vec3();
            let mut jac = Matrix3::<f64>::zeros();
            for axis in 0..3 {
                let deriv = (plus_by_axis[axis][residual_idx].0
                    - minus_by_axis[axis][residual_idx].0)
                    * (0.5 / eps);
                jac[(0, axis)] = deriv.x as f64;
                jac[(1, axis)] = deriv.y as f64;
                jac[(2, axis)] = deriv.z as f64;
            }
            h += jac.transpose() * jac * *weight as f64;
            b += jac.transpose() * r_vec * *weight as f64;
        }
        for axis in 0..3 {
            h[(axis, axis)] += 1.0e-8;
        }
        let step = h.lu().solve(&(-b))?;
        if !step.iter().all(|v| v.is_finite()) {
            return None;
        }
        let step_vec =
            Vec3::new(step[0] as f32, step[1] as f32, step[2] as f32).clamp_length_max(0.02);
        if step_vec.length() < 1.0e-7 {
            break;
        }
        offset = (Quat::from_scaled_axis(step_vec) * offset).normalize();
    }
    Some(offset)
}

fn circle_rotation_relative_cost(
    offset: Quat,
    frame_rotations: &[Quat],
    centers: &[Vec3],
    pairs: &[PairGeometry],
    segment_start: usize,
) -> f32 {
    let Some(residuals) =
        circle_rotation_residuals(offset, frame_rotations, centers, pairs, segment_start)
    else {
        return f32::INFINITY;
    };
    let mut total = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (residual, weight) in residuals {
        total += residual.length_squared() * weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        f32::INFINITY
    } else {
        total / weight_sum
    }
}

fn circle_absolute_score_weight() -> f32 {
    std::env::var("RUSTSFM_CIRCLE_ABSOLUTE_SCORE_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.01)
}

fn circle_rotation_residuals(
    offset: Quat,
    frame_rotations: &[Quat],
    centers: &[Vec3],
    pairs: &[PairGeometry],
    segment_start: usize,
) -> Option<Vec<(Vec3, f32)>> {
    let mut residuals = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        let left = pair.left.checked_sub(segment_start)?;
        let right = pair.right.checked_sub(segment_start)?;
        let left_frame = *frame_rotations.get(left)?;
        let right_frame = *frame_rotations.get(right)?;
        let predicted =
            (offset * right_frame * left_frame.inverse() * offset.inverse()).normalize();
        let observed = pose_rotation(pair.relative_pose);
        let residual = quat_log((observed * predicted.inverse()).normalize());
        if !residual.is_finite() || residual.length() > 0.25 {
            continue;
        }
        let weight = rotation_edge_weight(pair)
            .min(rotation_edge_weight_cap(pair))
            .max(0.05);
        residuals.push((residual, weight));
        if !centers.is_empty() {
            let left_center = *centers.get(pair.left)?;
            let right_center = *centers.get(pair.right)?;
            let delta = (left_center - right_center).try_normalize()?;
            let predicted_t = (offset * (right_frame * delta)).try_normalize()?;
            let mut observed_t =
                Vec3::from_array(pair.relative_pose.translation()).try_normalize()?;
            if predicted_t.dot(observed_t) < 0.0 {
                observed_t = -observed_t;
            }
            let cross = predicted_t.cross(observed_t);
            if cross.is_finite() && cross.length() <= 0.35 {
                let translation_weight = (edge_weight(pair).min(edge_weight_cap(pair))
                    * circle_translation_weight_scale())
                .clamp(0.1, circle_translation_weight_cap());
                residuals.push((cross, translation_weight));
            }
        }
    }
    Some(residuals)
}

fn circle_translation_weight_scale() -> f32 {
    std::env::var("RUSTSFM_CIRCLE_TRANSLATION_WEIGHT_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(96.0)
}

fn circle_translation_weight_cap() -> f32 {
    std::env::var("RUSTSFM_CIRCLE_TRANSLATION_WEIGHT_CAP")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(120.0)
}

fn refine_periodic_rotation_harmonics(
    rotations: &mut [Quat],
    centers: &[Vec3],
    circle_segments: &[CircleSegment],
    pairs: &[PairGeometry],
) {
    if std::env::var_os("RUSTSFM_CIRCLE_HARMONIC_ROTATIONS").is_none() {
        return;
    }
    for segment in circle_segments {
        if segment.end > rotations.len() || segment.end > centers.len() {
            continue;
        }
        refine_rotation_harmonic_segment(rotations, centers, *segment, pairs);
    }
}

fn refine_rotation_harmonic_segment(
    rotations: &mut [Quat],
    centers: &[Vec3],
    segment: CircleSegment,
    pairs: &[PairGeometry],
) {
    let segment_pairs = pairs
        .iter()
        .filter(|pair| {
            pair.left >= segment.start
                && pair.right < segment.end
                && !pair.pose_graph_only
                && pair.right > pair.left
                && pair.right - pair.left <= 5
                && pair.inliers >= 40
                && pair.mean_reprojection_error_px <= 2.0
        })
        .collect::<Vec<_>>();
    if segment_pairs.len() < (segment.end - segment.start).saturating_sub(1) {
        return;
    }
    let angles = segment_angles(centers, segment);
    if angles.len() != segment.end - segment.start {
        return;
    }
    let mut params = vec![Vec3::ZERO; 2];
    for _ in 0..12 {
        let Some(step) = harmonic_rotation_step(
            rotations,
            centers,
            &angles,
            &params,
            &segment_pairs,
            segment.start,
        ) else {
            return;
        };
        let max_component = step.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_component < 1.0e-7 {
            break;
        }
        for axis in 0..3 {
            params[0][axis] += step[axis].clamp(-0.01, 0.01);
            params[1][axis] += step[axis + 3].clamp(-0.01, 0.01);
        }
    }
    for idx in segment.start..segment.end {
        let local = idx - segment.start;
        let delta = params[0] * angles[local].cos() + params[1] * angles[local].sin();
        if delta.is_finite() && delta.length() <= 0.2 {
            rotations[idx] = (Quat::from_scaled_axis(delta) * rotations[idx]).normalize();
        }
    }
}

fn segment_angles(centers: &[Vec3], segment: CircleSegment) -> Vec<f32> {
    (segment.start..segment.end)
        .filter_map(|idx| {
            let d = centers[idx] - segment.center;
            let x = d.dot(segment.normal.any_orthonormal_vector());
            let basis_u = segment.normal.any_orthonormal_vector();
            let basis_v = segment.normal.cross(basis_u).try_normalize()?;
            Some(d.dot(basis_v).atan2(x))
        })
        .collect()
}

fn harmonic_rotation_step(
    rotations: &[Quat],
    centers: &[Vec3],
    angles: &[f32],
    params: &[Vec3],
    pairs: &[&PairGeometry],
    segment_start: usize,
) -> Option<Vec<f32>> {
    let residuals = harmonic_residuals(rotations, centers, angles, params, pairs, segment_start)?;
    if residuals.is_empty() {
        return None;
    }
    let variable_count = 6usize;
    let mut h = DMatrix::<f64>::zeros(variable_count, variable_count);
    let mut b = DVector::<f64>::zeros(variable_count);
    let eps = 1.0e-4f32;
    let mut plus_by_var = Vec::with_capacity(variable_count);
    let mut minus_by_var = Vec::with_capacity(variable_count);
    for var in 0..variable_count {
        let mut plus_params = params.to_vec();
        let mut minus_params = params.to_vec();
        plus_params[var / 3][var % 3] += eps;
        minus_params[var / 3][var % 3] -= eps;
        let plus = harmonic_residuals(
            rotations,
            centers,
            angles,
            &plus_params,
            pairs,
            segment_start,
        )?;
        let minus = harmonic_residuals(
            rotations,
            centers,
            angles,
            &minus_params,
            pairs,
            segment_start,
        )?;
        if plus.len() != residuals.len() || minus.len() != residuals.len() {
            return None;
        }
        plus_by_var.push(plus);
        minus_by_var.push(minus);
    }
    for (residual_idx, (residual, weight)) in residuals.iter().enumerate() {
        let r = residual.to_na_vec3();
        let mut jac = DMatrix::<f64>::zeros(3, variable_count);
        for var in 0..variable_count {
            let d = (plus_by_var[var][residual_idx].0 - minus_by_var[var][residual_idx].0)
                * (0.5 / eps);
            jac[(0, var)] = d.x as f64;
            jac[(1, var)] = d.y as f64;
            jac[(2, var)] = d.z as f64;
        }
        h += jac.transpose() * &jac * *weight as f64;
        b += jac.transpose() * r * *weight as f64;
    }
    for idx in 0..variable_count {
        h[(idx, idx)] += 1.0e-8;
    }
    let solution = h.lu().solve(&(-b))?;
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(solution.iter().map(|v| *v as f32).collect())
}

fn harmonic_residuals(
    rotations: &[Quat],
    centers: &[Vec3],
    angles: &[f32],
    params: &[Vec3],
    pairs: &[&PairGeometry],
    segment_start: usize,
) -> Option<Vec<(Vec3, f32)>> {
    let mut residuals = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        let left = pair.left.checked_sub(segment_start)?;
        let right = pair.right.checked_sub(segment_start)?;
        let left_delta = params[0] * angles.get(left)?.cos() + params[1] * angles.get(left)?.sin();
        let right_delta =
            params[0] * angles.get(right)?.cos() + params[1] * angles.get(right)?.sin();
        let left_rotation = (Quat::from_scaled_axis(left_delta) * rotations[pair.left]).normalize();
        let right_rotation =
            (Quat::from_scaled_axis(right_delta) * rotations[pair.right]).normalize();
        let predicted = (right_rotation * left_rotation.inverse()).normalize();
        let observed = pose_rotation(pair.relative_pose);
        let residual = quat_log((observed * predicted.inverse()).normalize());
        if !residual.is_finite() || residual.length() > 0.25 {
            continue;
        }
        let weight = rotation_edge_weight(pair)
            .min(rotation_edge_weight_cap(pair))
            .max(0.05);
        residuals.push((residual, weight));
        if harmonic_translation_residuals_enabled() {
            let left_center = *centers.get(pair.left)?;
            let right_center = *centers.get(pair.right)?;
            let delta = (left_center - right_center).try_normalize()?;
            let predicted_t = (right_rotation * delta).try_normalize()?;
            let mut observed_t =
                Vec3::from_array(pair.relative_pose.translation()).try_normalize()?;
            if predicted_t.dot(observed_t) < 0.0 {
                observed_t = -observed_t;
            }
            let cross = predicted_t.cross(observed_t);
            if cross.is_finite() && cross.length() <= 0.35 {
                let translation_weight = (edge_weight(pair).min(edge_weight_cap(pair))
                    * harmonic_translation_weight_scale())
                .clamp(0.1, harmonic_translation_weight_cap());
                residuals.push((cross, translation_weight));
            }
        }
    }
    Some(residuals)
}

fn harmonic_translation_residuals_enabled() -> bool {
    std::env::var_os("RUSTSFM_CIRCLE_HARMONIC_TRANSLATION").is_some()
}

fn harmonic_translation_weight_scale() -> f32 {
    std::env::var("RUSTSFM_CIRCLE_HARMONIC_TRANSLATION_WEIGHT_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or_else(circle_translation_weight_scale)
}

fn harmonic_translation_weight_cap() -> f32 {
    std::env::var("RUSTSFM_CIRCLE_HARMONIC_TRANSLATION_WEIGHT_CAP")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_else(circle_translation_weight_cap)
}

fn axis_permutations() -> [[usize; 3]; 6] {
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
}

fn axis_signs() -> [[f32; 3]; 8] {
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
}

fn rotation_from_world_axes(rows: [Vec3; 3]) -> Option<Quat> {
    if !rows.iter().all(|axis| axis.is_finite()) {
        return None;
    }
    let det = rows[0].dot(rows[1].cross(rows[2]));
    if det <= 0.5 {
        return None;
    }
    let matrix = Matrix3::from_row_slice(&[
        rows[0].x as f64,
        rows[0].y as f64,
        rows[0].z as f64,
        rows[1].x as f64,
        rows[1].y as f64,
        rows[1].z as f64,
        rows[2].x as f64,
        rows[2].y as f64,
        rows[2].z as f64,
    ]);
    project_matrix_to_rotation(matrix).map(matrix_to_quat)
}

fn average_quaternions(quaternions: &[Quat]) -> Option<Quat> {
    if quaternions.is_empty() {
        return None;
    }
    let mut accum = nalgebra::Matrix4::<f64>::zeros();
    for &quat in quaternions {
        let mut q = quat.normalize();
        if q.w < 0.0 {
            q = -q;
        }
        let v = nalgebra::Vector4::new(q.x as f64, q.y as f64, q.z as f64, q.w as f64);
        accum += v * v.transpose();
    }
    let eig = accum.symmetric_eigen();
    let mut best = 0usize;
    for idx in 1..4 {
        if eig.eigenvalues[idx] > eig.eigenvalues[best] {
            best = idx;
        }
    }
    let q = eig.eigenvectors.column(best);
    let mut quat = Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32);
    if quat.w < 0.0 {
        quat = -quat;
    }
    quat.is_finite().then_some(quat.normalize())
}

fn close_rotation_segment(rotations: &mut [Quat]) {
    if rotations.len() < 3 {
        return;
    }
    let end = rotations.len() - 1;
    let observed_last_step = estimate_seam_observed_rotation(rotations);
    let target_end = (observed_last_step.inverse() * rotations[0]).normalize();
    let correction = (target_end * rotations[end].inverse()).normalize();
    let correction_log = quat_log(correction) * periodic_rotation_scale();
    if !correction_log.is_finite() || correction_log.length() > 0.8 {
        return;
    }
    let denom = end as f32;
    for (idx, rotation) in rotations.iter_mut().enumerate().skip(1) {
        let t = idx as f32 / denom;
        let delta = Quat::from_scaled_axis(correction_log * t);
        *rotation = (delta * *rotation).normalize();
    }
}

fn periodic_rotation_scale() -> f32 {
    std::env::var("RUSTSFM_PERIODIC_ROTATION_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(1.0)
}

fn estimate_seam_observed_rotation(rotations: &[Quat]) -> Quat {
    let end = rotations.len() - 1;
    let window = 6usize.min(end);
    let mut total = Vec3::ZERO;
    let mut count = 0.0f32;
    for idx in 0..window {
        let log = quat_log((rotations[idx + 1] * rotations[idx].inverse()).normalize());
        if log.is_finite() {
            total += log;
            count += 1.0;
        }
    }
    for idx in end.saturating_sub(window)..end {
        let log = quat_log((rotations[idx + 1] * rotations[idx].inverse()).normalize());
        if log.is_finite() {
            total += log;
            count += 1.0;
        }
    }
    if count <= 0.0 {
        Quat::IDENTITY
    } else {
        Quat::from_scaled_axis(total / count)
    }
}

fn close_center_segment(centers: &mut [Vec3]) {
    if centers.len() < 3 {
        return;
    }
    let end = centers.len() - 1;
    let final_delta = estimate_seam_center_step(centers);
    let residual = centers[end] + final_delta - centers[0];
    if !residual.is_finite() || residual.length() > centers.len() as f32 * 0.5 {
        return;
    }
    let denom = end as f32;
    for (idx, center) in centers.iter_mut().enumerate().skip(1) {
        let t = idx as f32 / denom;
        *center -= residual * t;
    }
}

fn estimate_seam_center_step(centers: &[Vec3]) -> Vec3 {
    let end = centers.len() - 1;
    let window = 8usize.min(end);
    let mut total = Vec3::ZERO;
    let mut count = 0.0f32;
    for idx in 0..window {
        let delta = centers[idx + 1] - centers[idx];
        if delta.is_finite() {
            total += delta;
            count += 1.0;
        }
    }
    for idx in end.saturating_sub(window)..end {
        let delta = centers[idx + 1] - centers[idx];
        if delta.is_finite() {
            total += delta;
            count += 1.0;
        }
    }
    if count <= 0.0 {
        Vec3::ZERO
    } else {
        total / count
    }
}

fn average_translations(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
) -> Vec<Vec3> {
    let mut centers = solve_translation_averaging(image_count, pairs, rotations)
        .unwrap_or_else(|| chain_translation_initialization(image_count, pairs, rotations));

    for _ in 0..900 {
        for pair in pairs {
            if is_segment_break_edge(pair) {
                continue;
            }
            let Some(mut dir) = edge_world_direction(pair, rotations[pair.right]) else {
                continue;
            };
            let delta = centers[pair.right] - centers[pair.left];
            if let Some(delta_dir) = delta.try_normalize() {
                if delta_dir.dot(dir) < -0.2 {
                    dir = -dir;
                }
            }
            let weight = edge_weight(pair).min(edge_weight_cap(pair));
            let parallel = dir * delta.dot(dir);
            let perpendicular_residual = delta - parallel;
            let step = 0.018 * weight;
            if pair.left != 0 {
                centers[pair.left] += perpendicular_residual * (0.5 * step);
            }
            centers[pair.right] -= perpendicular_residual * (0.5 * step);

            if should_constrain_edge_length(pair) {
                let baseline = edge_baseline_units(pair, image_count);
                let along_residual = delta.dot(dir) - baseline;
                let length_weight = if pair.left + 1 == pair.right {
                    1.0
                } else {
                    0.25
                };
                let length_step = 0.012 * weight * length_weight * edge_length_scale();
                let correction = dir * along_residual;
                if pair.left != 0 {
                    centers[pair.left] += correction * (0.5 * length_step);
                }
                centers[pair.right] -= correction * (0.5 * length_step);
            }
        }
        apply_center_smoothness(&mut centers);
        centers[0] = Vec3::ZERO;
    }
    centers
}

fn chain_translation_initialization(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
) -> Vec<Vec3> {
    let mut centers = vec![Vec3::ZERO; image_count];
    let mut previous_dir = Vec3::X;
    for idx in 1..image_count {
        if let Some(pair) = translation_initialization_pair(pairs, idx) {
            let dir = edge_world_direction(pair, rotations[pair.right]).unwrap_or(previous_dir);
            let dir = if dir.dot(previous_dir) < -0.25 {
                -dir
            } else {
                dir
            };
            centers[idx] = centers[pair.left] + dir * edge_baseline_units(pair, image_count);
            previous_dir = dir;
        } else {
            centers[idx] = centers[idx - 1] + previous_dir;
        }
    }
    centers
}

fn solve_translation_averaging(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
) -> Option<Vec<Vec3>> {
    solve_translation_averaging_with_filter(image_count, pairs, rotations, None).and_then(
        |first_pass| {
            let filtered = pairs
                .iter()
                .filter(|pair| translation_edge_is_consistent(pair, rotations, &first_pass))
                .cloned()
                .collect::<Vec<_>>();
            if filtered.len() < image_count.saturating_sub(1) {
                Some(first_pass)
            } else {
                solve_translation_averaging_with_filter(
                    image_count,
                    filtered.as_slice(),
                    rotations,
                    Some(&first_pass),
                )
                .or(Some(first_pass))
            }
        },
    )
}

fn solve_translation_averaging_with_filter(
    image_count: usize,
    pairs: &[PairGeometry],
    rotations: &[Quat],
    _seed_centers: Option<&[Vec3]>,
) -> Option<Vec<Vec3>> {
    if image_count < 2 {
        return Some(vec![Vec3::ZERO; image_count]);
    }
    let variable_count = image_count * 3;
    let mut rows = Vec::<f64>::new();
    let mut rhs = Vec::<f64>::new();

    for axis in 0..3 {
        push_center_row(&mut rows, variable_count, 0, axis, 100.0);
        rhs.push(0.0);
    }

    for pair in pairs {
        if is_segment_break_edge(pair) {
            continue;
        }
        let Some(dir) = edge_world_direction(pair, rotations[pair.right]) else {
            continue;
        };
        let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64);
        let projector = Matrix3::<f64>::identity() - d * d.transpose();
        let weight = edge_weight(pair).sqrt().clamp(0.2, edge_weight_cap(pair)) as f64;
        for row_axis in 0..3 {
            let mut row = vec![0.0f64; variable_count];
            for axis in 0..3 {
                let value = projector[(row_axis, axis)] * weight;
                row[pair.right * 3 + axis] += value;
                row[pair.left * 3 + axis] -= value;
            }
            rows.extend(row);
            rhs.push(0.0);
        }

        if should_constrain_edge_length(pair) {
            let length_weight = if pair.left + 1 == pair.right {
                3.0
            } else {
                0.6
            };
            let weight = edge_weight(pair).sqrt().clamp(0.2, edge_weight_cap(pair)) as f64
                * length_weight
                * edge_length_scale() as f64;
            let mut signed_dir = dir;
            if pair.left + 1 == pair.right && pair.left > 0 {
                if let Some(prev_pair) = pairs
                    .iter()
                    .find(|p| p.left + 1 == p.right && p.right == pair.left)
                {
                    if let Some(prev_dir) =
                        edge_world_direction(prev_pair, rotations[prev_pair.right])
                    {
                        if signed_dir.dot(prev_dir) < -0.25 {
                            signed_dir = -signed_dir;
                        }
                    }
                }
            }
            let mut row = vec![0.0f64; variable_count];
            for (axis, value) in [signed_dir.x, signed_dir.y, signed_dir.z]
                .into_iter()
                .enumerate()
            {
                row[pair.right * 3 + axis] += value as f64 * weight;
                row[pair.left * 3 + axis] -= value as f64 * weight;
            }
            rows.extend(row);
            rhs.push(edge_baseline_units(pair, image_count) as f64 * weight);
        }
    }
    let row_count = rhs.len();
    if row_count < variable_count {
        return None;
    }
    let a = DMatrix::<f64>::from_row_slice(row_count, variable_count, &rows);
    let b = DVector::<f64>::from_row_slice(&rhs);
    let solution = a.svd(true, true).solve(&b, 1.0e-6).ok()?;
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut centers = Vec::with_capacity(image_count);
    for idx in 0..image_count {
        centers.push(Vec3::new(
            solution[idx * 3] as f32,
            solution[idx * 3 + 1] as f32,
            solution[idx * 3 + 2] as f32,
        ));
    }
    Some(centers)
}

fn translation_edge_is_consistent(
    pair: &PairGeometry,
    rotations: &[Quat],
    centers: &[Vec3],
) -> bool {
    if pair.left + 1 == pair.right || is_ring_bridge_edge(pair) {
        return true;
    }
    if pair.left >= centers.len() || pair.right >= centers.len() {
        return false;
    }
    let Some(edge_dir) = edge_world_direction(pair, rotations[pair.right]) else {
        return false;
    };
    let Some(delta_dir) = (centers[pair.right] - centers[pair.left]).try_normalize() else {
        return false;
    };
    let angle = edge_dir
        .dot(delta_dir)
        .abs()
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let offset = pair.right.abs_diff(pair.left);
    let threshold = if offset <= 2 {
        35.0
    } else if offset <= 5 {
        25.0
    } else {
        18.0
    };
    angle <= threshold
}

fn push_center_row(
    rows: &mut Vec<f64>,
    variable_count: usize,
    image: usize,
    axis: usize,
    weight: f64,
) {
    let mut row = vec![0.0f64; variable_count];
    row[image * 3 + axis] = weight;
    rows.extend(row);
}

fn should_constrain_edge_length(pair: &PairGeometry) -> bool {
    if std::env::var_os("RUSTSFM_NO_EDGE_LENGTHS").is_some() {
        return false;
    }
    pair.left + 1 == pair.right || pair.right.abs_diff(pair.left) <= 2 || is_ring_bridge_edge(pair)
}

fn edge_length_scale() -> f32 {
    std::env::var("RUSTSFM_EDGE_LENGTH_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(3.0)
}

fn center_smoothness_scale() -> f32 {
    std::env::var("RUSTSFM_CENTER_SMOOTHNESS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

fn apply_center_smoothness(centers: &mut [Vec3]) {
    let scale = center_smoothness_scale();
    if scale <= 0.0 || centers.len() < 3 {
        return;
    }
    let mut updates = vec![Vec3::ZERO; centers.len()];
    for idx in 1..centers.len() - 1 {
        let second = centers[idx - 1] - 2.0 * centers[idx] + centers[idx + 1];
        updates[idx] = second * scale;
    }
    for idx in 1..centers.len() - 1 {
        centers[idx] += updates[idx];
    }
}

fn translation_initialization_pair(pairs: &[PairGeometry], right: usize) -> Option<&PairGeometry> {
    pairs
        .iter()
        .filter(|p| p.right == right && !is_segment_break_edge(p))
        .max_by_key(|p| {
            let adjacent_bonus = usize::from(p.left + 1 == p.right) * 1_000_000;
            let bridge_bonus = usize::from(is_ring_bridge_edge(p)) * 2_000_000;
            adjacent_bonus + bridge_bonus + p.inliers * 10 + p.triangulated
        })
}

fn edge_world_direction(pair: &PairGeometry, target_rotation: Quat) -> Option<Vec3> {
    let t = Vec3::from_array(pair.relative_pose.translation()).try_normalize()?;
    let dir = -(target_rotation.inverse() * t);
    dir.try_normalize()
}

fn edge_baseline_units(pair: &PairGeometry, image_count: usize) -> f32 {
    let offset = pair.right.abs_diff(pair.left).max(1);
    let nominal = if is_ring_bridge_edge(pair) {
        offset.abs_diff(segment_period()).max(1) as f32
    } else if image_count > 4 && offset > image_count / 2 {
        (image_count - offset).max(1) as f32
    } else {
        offset as f32
    };
    if let Some(blend) = triangulation_length_blend() {
        let triangulation_length =
            (pair.median_triangulation_angle_deg / 1.8).clamp(0.45, nominal * 1.6);
        return nominal * (1.0 - blend) + triangulation_length * blend;
    }
    if is_ring_bridge_edge(pair) {
        return offset.abs_diff(segment_period()).max(1) as f32;
    }
    if image_count > 4 && offset > image_count / 2 {
        return (image_count - offset).max(1) as f32;
    }
    offset as f32
}

fn triangulation_length_blend() -> Option<f32> {
    std::env::var("RUSTSFM_TRIANGULATION_LENGTH_BLEND")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn is_ring_bridge_edge(pair: &PairGeometry) -> bool {
    let period = segment_period();
    let delta = pair.right.abs_diff(pair.left);
    if delta < period.saturating_sub(4) || delta > period + 4 {
        return false;
    }
    pair.left % period <= 3 || pair.left % period >= period.saturating_sub(4)
}

fn is_segment_break_edge(pair: &PairGeometry) -> bool {
    let period = segment_period();
    pair.left / period != pair.right / period && !is_ring_bridge_edge(pair)
}

fn edge_weight(pair: &PairGeometry) -> f32 {
    let inlier_weight = (pair.inliers as f32 / 100.0).sqrt().max(0.25);
    let triangulated_weight = (pair.triangulated as f32 / 80.0).sqrt().clamp(0.25, 2.0);
    let reproj_weight = (2.5 / pair.mean_reprojection_error_px.max(0.5)).clamp(0.05, 2.0);
    let rotation_weight = if pair.rotation_deg > 45.0 {
        0.02
    } else if pair.rotation_deg > 25.0 {
        0.15
    } else if pair.rotation_deg > 12.0 {
        0.5
    } else {
        1.0
    };
    let offset = if is_ring_bridge_edge(pair) {
        pair.right
            .abs_diff(pair.left)
            .abs_diff(segment_period())
            .max(1) as f32
    } else {
        pair.right.abs_diff(pair.left).max(1) as f32
    };
    let closure_weight = if pair.pose_graph_only { 40.0 } else { 1.0 };
    closure_weight * inlier_weight * triangulated_weight * reproj_weight * rotation_weight
        / offset.sqrt()
}

fn edge_weight_cap(pair: &PairGeometry) -> f32 {
    if pair.pose_graph_only {
        40.0
    } else {
        2.5
    }
}

fn rotation_edge_weight(pair: &PairGeometry) -> f32 {
    if std::env::var_os("RUSTSFM_RAW_ROTATION_WEIGHTS").is_some() {
        return edge_weight(pair);
    }
    let offset = if is_ring_bridge_edge(pair) {
        pair.right
            .abs_diff(pair.left)
            .abs_diff(segment_period())
            .max(1) as f32
    } else {
        pair.right.abs_diff(pair.left).max(1) as f32
    };
    let triangulation_ratio = pair.median_triangulation_angle_deg / rotation_triangulation_scale();
    let triangulation_weight = (triangulation_ratio * triangulation_ratio).clamp(0.02, 1.0);
    let closure_weight = if pair.pose_graph_only { 20.0 } else { 1.0 };
    closure_weight * edge_weight(pair) * triangulation_weight * offset.sqrt() / offset.powf(0.15)
}

fn rotation_edge_weight_cap(pair: &PairGeometry) -> f32 {
    if pair.pose_graph_only {
        std::env::var("RUSTSFM_CLOSURE_ROTATION_WEIGHT_CAP")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(3.0)
    } else {
        3.0
    }
}

fn rotation_triangulation_scale() -> f32 {
    std::env::var("RUSTSFM_ROT_TRIANGULATION_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
}

fn segment_period() -> usize {
    192
}

#[allow(dead_code)]
fn _center_from_pose(pose: SE3) -> Vec3 {
    camera_center(pose)
}
