# RustViewer RustSFM Adaptive Keyframes Design

## Goal

RustViewer must stop treating every imported image as a keyframe. RustSFM will select
keyframes from the actual visual and geometric relationship between frames, so the
selected count varies naturally by dataset. All imported images remain available for
full-frame PnP registration and RustGS training.

This change must also reduce the keyframe reconstruction workload without imposing a
fixed keyframe count, a fixed sampling ratio, or a dataset-length-dependent cap.

## Current Problem

Image-folder import marks every frame as a keyframe. `RustSfmWorker` then passes every
frame ID to `run_keyframe_reconstruction`. RustSFM uses sequential matching with overlap
10 and quadratic expansion by default. A 960-image dataset therefore creates 14,045
pair verifications before full-frame PnP begins.

The current project configuration contains `use_all_images`, but setting it to false is
not sufficient for image folders because every imported image has `is_keyframe=true`.

## Selected Approach

Add an adaptive keyframe selection API to RustSFM. The selector will use SIFT feature
matches and verified two-view geometry, using the configured wgpu backend when GPU SIFT
and matching are enabled.

The selector has no maximum keyframe count. It scans frames in their stable sequence
order and advances a keyframe anchor only when the measured overlap with that anchor
falls below the configured retention range while geometric connectivity remains valid.
When connectivity is about to be lost, it retains the last connected bridge frame.

The first and last frames are always represented. The result is deterministic for a
fixed input order, configuration, and random seed.

## Selection Metrics

For an anchor/candidate pair, RustSFM records:

- descriptor match count;
- verified two-view inlier count;
- triangulated correspondence count;
- inlier ratio (`inliers / matches`);
- feature coverage (`inliers / min(anchor_features, candidate_features)`).

The policy classifies a candidate as follows:

1. **Redundant:** feature coverage remains above the retention threshold. Continue
   scanning without selecting the candidate. Low parallax alone does not turn a nearly
   identical frame into a keyframe.
2. **Connected transition:** coverage has fallen, but two-view geometry still meets the
   minimum inlier, inlier-ratio, and triangulation requirements. Select the candidate as
   the next keyframe and make it the new anchor.
3. **Connectivity loss:** coverage is low and geometry is no longer valid. Select the
   most recent candidate that was geometrically connected to the anchor, then retry the
   current candidate from the new anchor. If no bridge exists, select the current frame
   so the scan makes forward progress.

Thresholds are configuration values, not limits on output count. Defaults will follow
the existing RustSFM matching and geometry acceptance values where possible. Invalid or
non-finite threshold values are rejected before image processing begins.

## RustSFM API

RustSFM will expose:

- `AdaptiveKeyframeSelectionConfig` for overlap and geometry thresholds;
- `AdaptiveKeyframePairMetrics` for testable pair evidence;
- `AdaptiveKeyframeSelectionResult` containing selected frame IDs and diagnostics;
- a controlled selection entry point accepting `SequenceFrame`, mapper configuration,
  output/cache location, and `SfmTaskContext`.

Metric acquisition and policy evaluation are separate. The policy is a pure,
deterministic component that can be tested without a GPU or image fixtures. The runtime
adapter reuses existing RustSFM feature extraction, explicit-pair matching, two-view
verification, task cancellation, and progress events.

The selection database may contain features for all imported frames. The following
keyframe reconstruction and full-frame registration paths must reuse those database
features when they run in the same RustSFM operation, rather than extracting selected
features again.

## RustViewer Integration

`SfmConfigSnapshot` gains an adaptive selection mode with serde defaults so schema-v1
projects that do not contain the new field open in adaptive mode. The legacy
`use_all_images` value remains readable for compatibility, but adaptive mode takes
precedence unless an explicit all-images mode is selected.

When the keyframe stage starts, RustViewer calls the RustSFM selector and uses the
returned IDs. It does not derive adaptive keyframes from perceptual hashes, image count,
or a fixed stride.

The keyframe stage result records:

- imported frame count;
- selected keyframe count and IDs;
- selection thresholds;
- number of evaluated pairs;
- per-pair diagnostics needed to explain selection decisions.

Full-frame PnP receives the selected IDs and continues to register every imported frame.
RustGS remains blocked until complete pose coverage is committed.

## Reconstruction Workload Defaults

After adaptive selection, keyframe reconstruction uses local sequential matching with a
window of 5 and no quadratic expansion. This keeps pair generation linear in the number
of selected keyframes. SIFT extraction is capped at 4,096 features per image by the
RustViewer mapper preset. These are quality/performance parameters and remain distinct
from the adaptive selector's output count.

The existing generic wgpu backend policy remains unchanged. This work does not force
Metal or Vulkan and does not redesign GPU readback synchronization.

## Progress And Failure Behavior

Adaptive selection emits task progress for feature extraction, pair evaluation, and
selection completion. Progress includes the current anchor/candidate IDs and the number
of selected keyframes so the UI does not appear stalled.

Cancellation and pause requests are checked before extraction batches and before each
pair evaluation. Temporary selection output follows the existing stage workspace cleanup
rules.

Selection fails with a specific RustSFM error when fewer than two usable frames remain,
feature extraction cannot produce evidence, configuration is invalid, or no finite
ordered selection can be produced. RustViewer persists the failure using the existing
retryable pipeline error path.

## Tests

RustSFM policy tests cover:

- highly redundant sequences selecting only the required boundary representation;
- fast-changing sequences selecting more keyframes than redundant sequences of the same
  length;
- connected transitions advancing the anchor;
- connectivity loss retaining a bridge and terminating without loops;
- first/last preservation, stable ordering, uniqueness, and determinism;
- invalid thresholds and insufficient inputs.

RustSFM integration tests cover task cancellation and a small synthetic sequence through
the metric acquisition adapter when a compatible backend is available.

RustViewer tests cover:

- old manifests defaulting to adaptive mode;
- mapper configuration using 4,096 features and local window 5;
- the worker using RustSFM-selected IDs rather than all imported image IDs;
- full-frame PnP retaining complete imported-frame coverage.

The final verification set is the RustSFM targeted tests, the complete RustViewer test
suite, `cargo check` for both packages, and a RustViewer release build.

## Out Of Scope

- A fixed maximum keyframe count or fixed sampling ratio;
- perceptual-hash-only keyframe selection;
- forced Vulkan or Metal backend selection;
- asynchronous/batched GPU readback redesign;
- changing RustGS training hyperparameters.
