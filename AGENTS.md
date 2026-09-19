# RustScan Repository Guidelines

## CPU linear-algebra representation

All CPU-side matrices, vectors, points, quaternions, and related
linear-algebra code in this repository MUST use `nalgebra` as the only
mathematical type library.

Use the appropriate `nalgebra` type for the problem:

- `Matrix2`, `Matrix3`, `Matrix4`, and other fixed-size matrix types for
  small, statically sized matrices.
- `SMatrix<T, R, C>` for fixed-size non-square matrices and Jacobians.
- `DMatrix<T>` for matrices whose dimensions are known only at runtime.
- `Vector2<T>`, `Vector3<T>`, and `Vector4<T>` for mathematical vectors.
- `Point2<T>` and `Point3<T>` for positions, where point/vector semantics
  matter.
- `UnitQuaternion<T>` for rotations; use `Quaternion<T>` only for raw,
  potentially non-unit quaternion algebra.

This rule applies to public CPU APIs, internal CPU implementations, geometry,
SfM, SLAM, bundle adjustment, mesh processing, numerical solvers, and tests.
New CPU code MUST NOT introduce or use `glam::Mat*`, `glam::Vec*`, `glam::Quat`,
another mathematics crate, or a hand-rolled general-purpose linear-algebra
type.

`glam` may remain only in an explicit graphics/GPU or third-party adapter
boundary. It MUST NOT be used as the owning type for CPU vectors, points,
matrices, or rotations. Convert explicitly at that boundary when necessary.

## Boundary exceptions

The following representations are permitted only at an explicit boundary and
MUST NOT leak into general CPU linear-algebra APIs:

- Burn `Tensor` values in GPU training code.
- WGSL `mat*` values in shader code.
- Eigen matrices in C/C++ FFI and native helper code.
- `glam` values required by a graphics or third-party API adapter.
- Fixed arrays such as `[[f32; 4]; 4]` or flat buffers used for serialization,
  GPU uniforms, wire formats, or FFI.
- UI-library types such as `egui::Vec2` when they represent screen/layout
  dimensions rather than mathematical geometry.

Boundary conversions MUST make layout and convention explicit. Use names such
as `from_row_major_slice`, `to_row_major_array`, or
`to_column_major_array`; do not pass an unlabeled matrix-shaped array between
subsystems.

## Layout and convention

Unless a boundary API explicitly documents otherwise:

- Matrix indexing is `(row, column)`.
- Vectors are column vectors and transforms use `p' = M * p`.
- CPU numerical code uses `f64` when precision is material; use `f32` where
  the surrounding algorithm or hardware interface requires it.
- Pose direction MUST be stated in the type or API documentation, such as
  `world_from_camera` or `camera_from_world`.

## Shared pose and geometry types

Do not add another `SE3` implementation. Reuse the shared pose type from
`rustscan-types` or an existing crate-level re-export. An `SE3` rotation MUST
use `nalgebra::UnitQuaternion` and its translation MUST use
`nalgebra::Vector3`; keep conversion helpers in one place.

When changing matrix, vector, point, or quaternion code, add or update tests
with a non-symmetric rotation, non-zero translation, point/vector operations,
and a round-trip through every affected boundary. Identity-only tests are
insufficient to detect transposition, row/column-layout, point/vector, and
quaternion-component-order errors.
