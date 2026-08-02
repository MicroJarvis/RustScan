# RustSFM GPU PnP-f Design

## Goal

Support absolute-pose estimation with an unknown, shared focal length on the
existing generic wgpu path. The solver must estimate pose and focal length for
one camera shared by an image sequence, so RustViewer can reconstruct image
sets without trusted focal-length metadata while retaining GPU acceleration.

## Scope

The first version supports a single camera with a shared focal length and a
known principal point. It runs the focal-aware RANSAC model generation,
candidate scoring, best-model selection, inlier-mask generation, and local
refinement on the GPU. It keeps the CPU solver as the reference implementation
and as an explicit fallback for unavailable GPUs, numerical failures, or
validation failures.

This change does not support per-image focal lengths, separate `fx` and `fy`,
variable-zoom sequences, multi-camera rigs, or a forced Vulkan or Metal
backend. Existing GPU SIFT extraction and matching settings remain independent
from the PnP-f backend.

## Existing Behavior

RustSFM currently estimates focal length with the CPU `PnPSolver` when a
camera lacks a prior focal length. The GPU scorer accepts only `SE3` models and
scores normalized observations with fixed intrinsics. Consequently, enabling
`use_gpu_pnp` while focal estimation is required produces the error `gpu pnp
does not support focal length estimation`.

The CPU focal route samples four correspondences, evaluates P4Pf candidates,
adds P3P-plus-focal-update candidates for numerical coverage, scores candidates
in pixel space, and refines the best model.

## Architecture

### Solver Interface

Add an internal focal-aware GPU solver that accepts centered image points,
object points, principal point, RANSAC settings, focal bounds, and a seed. Its
result contains a pose, a positive shared focal length, an inlier mask, and
backend telemetry. The existing fixed-intrinsic GPU PnP scorer remains
unchanged and continues to serve known-intrinsics requests.

The mapper selects the focal-aware GPU solver whenever focal estimation is
required and GPU PnP is enabled. It must not reject the request merely because
the focal route was selected.

### GPU Pipeline

The wgpu solver uses storage buffers for observations, sampled indices,
candidate models, support summaries, the best-model state, and the final
inlier mask. Every candidate is represented as rotation, translation, and
`log_focal`; using logarithmic focal length guarantees a positive focal value.

Compute passes run in this order:

1. Generate deterministic four-point samples from the configured seed.
2. Produce P4Pf focal-pose candidates. When that route is singular or yields
   invalid roots, generate P3P candidates followed by focal updates.
3. Reject non-finite, behind-camera, or out-of-range-focal candidates.
4. Score candidates in pixel coordinates, reduce each candidate to inlier
   count and residual sum, and atomically select the best valid candidate.
5. Generate an inlier mask for the selected model.
6. Run a fixed number of Gauss-Newton local-refinement iterations over pose and
   `log_focal`, with GPU reductions and a small GPU linear solve.
7. Re-score the refined model and write final pose, focal length, support, and
   mask to readback buffers.

The first version dispatches the configured maximum RANSAC trials. Dynamic
early stopping is deliberately out of scope because wgpu cannot recursively
dispatch new work from a compute shader; a later optimization can use indirect
dispatch after correctness is established.

### Numerical Rules

The solver keeps observations centered around the known principal point and
uses pixels for the inlier threshold, matching the CPU focal route. It rejects
focal values outside `MapperConfig` focal-ratio bounds, non-finite arithmetic,
negative depth, degenerate samples, and rank-deficient local-refinement
systems. Candidate comparisons use the same ordering as the CPU path: more
inliers first, lower residual sum second.

### Fallback and Diagnostics

If wgpu initialization, shader validation, dispatch, readback, or numerical
validation fails, the mapper executes the existing CPU PnP-f route. A fallback
is a successful result with a diagnostic event, not a silent backend change.
If CPU PnP-f also fails, the reconstruction stage fails with the actual cause.

RustViewer receives backend telemetry so its stage presentation distinguishes
GPU PnP-f, CPU PnP-f fallback, and terminal failures. A persisted failed stage
must replace any stale running operation in the UI.

## Validation

Unit and integration coverage must include:

- Deterministic synthetic scenes with known pose and focal length, checking
  focal error, angular pose error, translation error, inliers, and residuals.
- CPU/GPU differential tests on synthetic scenes, accepting documented numeric
  tolerances rather than identical RANSAC samples.
- Degenerate geometry, invalid observations, behind-camera points, and focal
  bounds violations.
- GPU initialization and dispatch failures, proving the CPU fallback is used
  and logged.
- Existing known-intrinsics GPU PnP tests, proving no regression.
- RustViewer orchestration for an imported project without a prior focal
  length, proving keyframe SfM proceeds rather than failing at GPU PnP.

## Success Criteria

- Unknown-focal single-camera reconstruction does not emit the current GPU PnP
  focal-estimation rejection.
- The generic wgpu backend performs the focal-aware solver without forcing a
  platform-specific graphics backend.
- Valid GPU results satisfy the configured quality gates and remain within the
  declared tolerances of the CPU reference on deterministic test scenes.
- Invalid GPU results are safely retried by CPU PnP-f and visibly reported.
