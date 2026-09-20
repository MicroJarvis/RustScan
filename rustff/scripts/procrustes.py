#!/usr/bin/env python3
"""Weighted Procrustes alignment for pose estimation from pointmaps.

This is a reference implementation that will be ported to Rust.
Given predicted 3D pointmaps and confidence weights, estimate the
optimal rigid body transformation (rotation + translation).

Usage:
    python scripts/procrustes.py  # runs self-test
"""

import numpy as np


def weighted_procrustes(X_src, X_tgt, weights=None):
    """Estimate rigid body transformation from source to target point cloud.

    Args:
        X_src: Source points [N, 3]
        X_tgt: Target points [N, 3]
        weights: Per-point weights [N] (default: uniform)

    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
    """
    N = X_src.shape[0]
    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.sum()

    # Weighted centroids
    src_centroid = (X_src * weights[:, None]).sum(axis=0)
    tgt_centroid = (X_tgt * weights[:, None]).sum(axis=0)

    # Center the point clouds
    X_src_centered = X_src - src_centroid
    X_tgt_centered = X_tgt - tgt_centroid

    # Weighted cross-covariance matrix
    W = np.diag(weights)
    H = X_src_centered.T @ W @ X_tgt_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Rotation
    R = Vt.T @ sign_matrix @ U.T

    # Translation
    t = tgt_centroid - R @ src_centroid

    return R, t


def pointmap_to_pose(pointmap, confidence, ref_pointmap=None):
    """Extract camera pose from predicted pointmap.

    The pointmap represents 3D points in the camera's coordinate frame.
    We estimate the transformation from camera frame to world frame
    by aligning with a reference or by analyzing the pointmap geometry.

    Args:
        pointmap: Predicted 3D points [H, W, 3] or [N, 3]
        confidence: Per-point confidence [H, W] or [N]
        ref_pointmap: Optional reference pointmap for alignment

    Returns:
        pose: 4x4 transformation matrix (camera to world)
    """
    # Flatten if needed
    if pointmap.ndim == 3:
        H, W, _ = pointmap.shape
        points = pointmap.reshape(-1, 3)
        conf = confidence.reshape(-1)
    else:
        points = pointmap
        conf = confidence

    # Filter by confidence
    valid_mask = conf > 0.1
    valid_points = points[valid_mask]
    valid_conf = conf[valid_mask]

    if len(valid_points) < 10:
        # Not enough valid points, return identity
        return np.eye(4)

    if ref_pointmap is not None:
        # Align with reference pointmap
        if ref_pointmap.ndim == 3:
            ref_points = ref_pointmap.reshape(-1, 3)
        else:
            ref_points = ref_pointmap

        ref_valid = ref_points[valid_mask]
        R, t = weighted_procrustes(valid_points, ref_valid, valid_conf)
    else:
        # Estimate pose from pointmap geometry
        # The pointmap is in camera frame, so we can derive pose from
        # the centroid and principal axes
        centroid = (valid_points * valid_conf[:, None]).sum(axis=0) / valid_conf.sum()

        # Centered points for covariance
        centered = valid_points - centroid
        cov = centered.T @ (centered * valid_conf[:, None])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Camera looks along -Z, so principal axis should align with camera forward
        R = eigenvectors
        t = centroid

    # Build 4x4 pose matrix
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    return pose


def test_procrustes():
    """Test weighted Procrustes alignment."""
    print("Testing Weighted Procrustes...")

    # Create random point cloud
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 3)

    # Known transformation
    R_true = np.array([
        [0.866, -0.5, 0],
        [0.5, 0.866, 0],
        [0, 0, 1]
    ])
    t_true = np.array([1.0, 2.0, 3.0])

    # Apply transformation
    Y = (R_true @ X.T).T + t_true

    # Add noise
    Y += np.random.randn(N, 3) * 0.01

    # Uniform weights
    weights = np.ones(N)

    # Estimate
    R_est, t_est = weighted_procrustes(X, Y, weights)

    # Check
    R_err = np.linalg.norm(R_est - R_true)
    t_err = np.linalg.norm(t_est - t_true)

    print(f"  Rotation error: {R_err:.6f} (should be < 0.01)")
    print(f"  Translation error: {t_err:.6f} (should be < 0.01)")

    assert R_err < 0.01, f"Rotation error too large: {R_err}"
    assert t_err < 0.01, f"Translation error too large: {t_err}"

    # Test with non-uniform weights
    weights_conf = np.exp(-np.linspace(0, 2, N))
    R_est2, t_est2 = weighted_procrustes(X, Y, weights_conf)
    R_err2 = np.linalg.norm(R_est2 - R_true)
    t_err2 = np.linalg.norm(t_est2 - t_true)
    print(f"  Weighted rotation error: {R_err2:.6f}")
    print(f"  Weighted translation error: {t_err2:.6f}")

    print("  PASSED")


def test_pointmap_to_pose():
    """Test pointmap to pose extraction."""
    print("\nTesting pointmap_to_pose...")

    # Create a synthetic pointmap (simulating a flat wall)
    H, W = 32, 32
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.ones_like(x) * 5.0  # 5m depth

    pointmap = np.stack([x - W/2, y - H/2, z], axis=-1)
    confidence = np.ones((H, W)) * 0.9

    pose = pointmap_to_pose(pointmap, confidence)
    print(f"  Pose shape: {pose.shape}")
    print(f"  Pose:\n{pose}")

    assert pose.shape == (4, 4)
    print("  PASSED")


if __name__ == "__main__":
    test_procrustes()
    test_pointmap_to_pose()
    print("\nAll tests passed!")
