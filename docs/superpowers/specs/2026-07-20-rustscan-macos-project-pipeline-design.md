# RustScan macOS Project Pipeline Design

## Status

Approved product brief and visual direction. This document defines the implementation boundary for
turning RustViewer into a project-based macOS application that imports video or image sequences,
runs RustSFM, trains RustGS, and browses the resulting Gaussian scene.

## Product Summary

RustViewer becomes the user-facing RustScan desktop application. Its first screen is a quiet project
library. Opening a project reveals a production-style pipeline workspace with media on the left, a
large 3D Gaussian viewport in the center, a compact collapsible inspector on the right, and pipeline
progress along the bottom.

The default workflow is automatic:

1. Import a video or image sequence.
2. Normalize the media and create thumbnails.
3. Reconstruct video keyframes with RustSFM. Image sequences use all images as keyframes by default.
4. Register every remaining video frame with PnP.
5. Require complete pose coverage before automatic RustGS training.
6. Show live Gaussian snapshots while training.
7. Save and browse the final lossless PLY and reports.

The selected technical approach is in-process orchestration through typed Rust APIs. Task interfaces
remain independent of the UI so they can later move to a helper process without changing the project
format or presentation layer.

## Goals

- Make video or image-sequence reconstruction a single coherent desktop workflow.
- Produce a pose for every imported frame.
- Keep the UI responsive during long RustSFM and RustGS jobs.
- Preserve completed work and resume after pause, failure, application quit, or machine restart.
- Use Apple platform conventions while retaining the existing Rust/egui/wgpu renderer.
- Keep the central 3DGS preview visibly dominant.
- Preserve direct loading of existing COLMAP, PLY, SPLAT, checkpoint, and mesh artifacts.

## Non-goals

- Cloud synchronization, remote compute, or distributed scheduling.
- Concurrent execution of multiple projects in the first release.
- A SwiftUI rewrite or a new native Metal renderer.
- Manual per-camera pose editing.
- Mesh generation or texture baking.
- Windows or Linux native video decoding in the first release. Image-sequence projects remain
  portable; video import uses a macOS implementation behind a platform interface.

## Selected Visual Direction

### Project library

The launch screen follows the approved content-first library direction:

- Project thumbnails and names are the strongest first signal.
- Cards show a compact five-segment pipeline status, a short status line, and last-updated time.
- Sidebar filters include All Projects, Processing, Completed, and Recently Opened.
- New Project is the only emphasized command.
- Search, sort, reveal in Finder, duplicate, and delete use standard toolbar/menu placement.

### Project workspace

The workspace follows the approved production-pipeline direction:

- Left media sidebar: all frames, keyframes, and frames needing attention.
- Center: stable, full-size wgpu 3DGS viewport with orbit, pan, zoom, fit, and display-mode controls.
- Right: a 42 px inspector rail. Selecting a tool opens an approximately 260 px collapsible sidebar.
- Bottom: stable pipeline timeline containing current progress, iterations, loss, elapsed time, and
  estimated remaining time.
- When training completes, the timeline can collapse to a report summary and the 3D viewport becomes
  the primary project surface.

The inspector never permanently consumes a large content column. Metrics required to understand a
running task remain visible in the bottom timeline.

### macOS behavior

- Use the system font stack and restrained semantic colors in light and dark appearance.
- Replace emoji controls with a consistent Lucide icon set using SF Symbols-like semantics.
- Use native window title bars, menus, keyboard shortcuts, drag and drop, file dialogs, sheets, and
  confirmation alerts where eframe exposes them.
- Keep controls dense and work-focused. Avoid marketing composition, nested cards, decorative
  gradients, oversized headings, and text-heavy helper panels.

## Application Architecture

### Modules

`ProjectStore`

- Creates, opens, validates, migrates, and atomically saves project packages.
- Resolves artifact paths and source bookmarks.
- Publishes project summaries for the library.

`PipelineCoordinator`

- Owns the only authoritative project stage state machine.
- Computes dependencies and downstream invalidation.
- Starts one background job at a time.
- Translates typed worker events into manifest updates and UI events.
- Handles pause, cancel, retry, application termination, and recovery.

`MediaImporter`

- Imports image sequences through the existing image stack.
- Decodes video with a macOS AVFoundation adapter behind a platform-neutral trait.
- Generates normalized frames, timestamps, metadata, and thumbnails.
- Keeps every decoded video frame; keyframe selection does not discard non-keyframes.

`SfmTaskRunner`

- Drives RustSFM feature extraction, matching, incremental mapping, and export through library APIs.
- Emits typed stage and progress events.
- Checks cooperative control requests at bounded work boundaries.

`FullFrameRegistrationRunner`

- Registers non-keyframe video frames against the keyframe reconstruction.
- Writes complete camera poses and registration diagnostics.
- Blocks automatic training if any frame remains unregistered.

`GsTaskRunner`

- Extends the existing RustViewer `TrainingManager` and RustGS event API.
- Supports pause checkpoints, resume, progress, live snapshots, and final export.
- Shares the eframe wgpu device for training and live preview.

`SceneSession`

- Loads sparse reconstruction, training snapshots, and final Gaussian artifacts.
- Keeps viewport state independent from pipeline state.
- Preserves the existing direct artifact-loading workflow.

### Execution model

- UI and rendering stay on the eframe main thread.
- Media, RustSFM, full-frame PnP, and persistence work run on dedicated background threads.
- RustGS uses its existing training thread and shared wgpu context.
- Workers communicate only through bounded typed channels.
- Worker APIs accept control tokens and event sinks; they do not depend on egui types.
- Only one compute-heavy stage runs at a time in the first release.

## Project Package

A project is a Finder-visible directory package with the `.rustscanproject` extension:

```text
Flowers.rustscanproject/
  project.json
  Sources/
    source.json
    managed/
  Cache/
    frames/
    thumbnails/
    keyframes.json
    database.db
  Reconstruction/
    sparse/0/
    poses.json
    registration.json
    summary.json
  Training/
    checkpoints/
    scene.ply
    scene.parity.json
  Logs/
    events.jsonl
```

`project.json` is the source of truth for orchestration, not for large numeric data. It contains:

- Schema version, UUID, display name, creation/update timestamps.
- Source specification and source identity hash.
- Import, SFM, PnP, and training configuration snapshots.
- Stage records and their committed artifact references.
- Active and final scene references.
- Compatibility information including RustSFM and RustGS artifact versions.

Large frame, database, sparse, checkpoint, and PLY data remain in their native formats.

### Source ownership

- Managed copy is the default because it makes projects reliable and movable.
- Reference-in-place is an advanced option. On macOS it stores a security-scoped bookmark and source
  identity. A missing or changed source produces a recoverable source error.
- Imported images and decoded video frames receive stable frame IDs independent of filenames.

### Atomic updates

- Manifest and small JSON artifacts are written to a sibling temporary file, flushed, and renamed.
- A stage becomes `Succeeded` only after all declared artifacts pass validation.
- Partial outputs remain in a stage-specific temporary directory and are never advertised as final.
- Opening a project cleans abandoned temporary outputs only after preserving diagnostics.

## Pipeline State Model

### User-visible stages

1. Import
2. Keyframe SFM
3. Full-frame PnP
4. 3DGS Training
5. Complete

### Internal stages

The Keyframe SFM segment contains feature extraction, matching/geometric verification, incremental
mapping/BA, and COLMAP export. These remain separately checkpointed and observable without adding
more primary timeline segments.

### Stage states

```text
NotStarted -> Ready -> Queued -> Running
Running -> PauseRequested -> Paused -> Queued
Running -> CancelRequested -> Cancelled -> Queued
Running -> Succeeded
Running -> Failed -> Queued
Succeeded -> Stale -> Ready
```

`Stale` means a previously valid result no longer matches its source or configuration. It is distinct
from failure and retains the old artifacts until replacement succeeds.

### Invalidation

- Source changes invalidate every downstream stage.
- Keyframe selection changes invalidate keyframe SFM, PnP, and training.
- SFM configuration changes invalidate SFM, PnP, and training.
- PnP configuration changes invalidate PnP and training.
- Training-only changes invalidate training while preserving reconstruction.
- Viewer appearance changes invalidate nothing.

The UI explains the invalidation before applying a setting that would trigger recomputation.

## Media Import and Keyframes

### Image sequences

- Supported image files are normalized into stable frame order.
- Every image participates in SFM by default, matching the existing all-image RustSFM workflow.
- Timestamp metadata is retained when available but is not required.

### Video

- AVFoundation decodes every source frame and records presentation timestamps.
- The keyframe selector targets approximately three reconstruction keyframes per second while forcing
  a maximum one-second keyframe gap.
- It favors sharper candidates and avoids adjacent near-duplicates.
- Selection is deterministic for the same source and configuration.
- All non-keyframe frames remain available for PnP and RustGS.
- Advanced settings can change the keyframe rate but never silently discard source frames.

### Import completion gate

Import succeeds only when frame IDs are unique, referenced media is readable, dimensions are valid,
and at least two frames exist. The UI reports source duration, decoded frame count, resolution, and
selected keyframe count before SFM begins.

## RustSFM Changes

### Unified task events

Add a public event model covering extraction, matching, mapping, BA, PnP, and export. Events include:

- Stage and operation identifier.
- Completed and total work counts when total is knowable.
- Registered image and sparse point counts.
- Current image or pair IDs for diagnostics.
- Elapsed time and monotonic sequence number.
- Recoverable warning or terminal error details.

Existing mapper callbacks remain valid and are adapted into the unified event stream.

### Cooperative control

Add a cloneable control token supporting continue, pause request, and cancel request. Check it:

- Between decoded/extracted images.
- Between bounded match batches.
- Before and after registration attempts.
- Between local/global BA invocations and at solver boundaries where safe.
- Before artifact export and stage commit.

Pause does not serialize a half-finished BA solve. It finishes or cancels the current atomic solver
operation, commits the last valid sparse state, and then reports `Paused`.

### Keyframe reconstruction

- Build or reuse the feature database for selected keyframes.
- Run the existing sequential matching and optimized incremental mapper.
- Export one validated sparse model.
- Require finite camera poses, finite points, and a non-empty sparse point set.

### Full-frame registration

- Extract features for non-keyframes.
- First match each frame to nearby registered keyframes in timestamp order.
- Estimate PnP against existing 3D tracks using the wgpu scorer when enabled.
- Retry failed frames with a wider temporal neighborhood and relaxed but bounded thresholds.
- Permit successful neighboring non-keyframes to support later registrations.
- Write per-frame inlier count, inlier ratio, reprojection error, attempt count, and final status.
- Require `registered_frames == imported_frames` before the automatic RustGS stage.

Frames that still fail move the project to `NeedsAttention`. The user may retry with a wider search or
explicitly exclude frames. Exclusion is never automatic and changes the coverage claim shown in UI.

## RustGS Changes

RustGS already provides iteration events, cancellation, HostSplats snapshots, and a shared-wgpu
training path. Extend it with resumable checkpoints.

A checkpoint contains:

- Host splats and SH metadata.
- Optimizer moments and learning-rate schedule position.
- Topology accumulators, splat ages, and invisibility windows.
- Current iteration, active SH degree, deterministic RNG state, and configuration hash.
- Dataset identity and reconstruction identity.

Checkpoints are written every 1,000 iterations, on pause, and before an orderly application quit.
Progress events remain frequent, but live HostSplats snapshots default to every 100 iterations to
bound GPU readback cost.

Resume rejects a mismatched dataset, reconstruction, or incompatible configuration with a clear
explanation. A user may explicitly start a fresh training run while retaining the previous result.

The final artifact remains a lossless PLY plus parity report. A completed export must pass the
existing finite-value and round-trip checks before the project is marked complete.

## Pause, Cancel, Quit, and Resume

### Pause

- Pause is cooperative and stage-aware.
- The UI immediately shows `Pause Requested` and continues reporting the current atomic operation.
- The stage enters `Paused` only after a valid checkpoint is committed.
- A paused project may be closed and reopened.

### Cancel

- Cancel stops future work but does not delete committed artifacts.
- Current valid RustGS snapshot and RustSFM database/sparse outputs are retained.
- Retry starts from the last valid boundary.
- Destructive cleanup is a separate, confirmed action.

### Application quit

- If a task is running, closing the last project window presents Pause and Quit, Cancel Task and
  Quit, or Keep Running.
- Pause and Quit waits for a safe checkpoint with visible progress.
- An unclean termination is detected from the manifest lease on next launch and recovered from the
  last committed state.

### Resume

- The project library shows Paused, Interrupted, or Needs Attention without opening the project.
- Opening the project explains exactly which stage and artifact will resume.
- Automatic resume occurs only after user confirmation when the previous termination was unclean.

## Error Handling

Errors are structured records with code, stage, summary, detail, affected frame/pair when relevant,
retryability, and suggested actions.

The UI follows these rules:

- Keep the 3D/media workspace visible; do not replace it with a generic error page.
- Mark only the affected stage.
- Use plain summaries such as `3 frames could not be registered`.
- Offer contextual commands: Retry, Use Wider Search, Reveal Source, Open Log, or Start Fresh.
- Put full technical details in the inspector/log, not in primary alerts.
- Never report success from process exit alone; validate declared artifacts.
- Never silently skip missing frames, non-finite poses, empty sparse points, NaNs, or OOM.

## Compatibility and Migration

- Existing RustViewer command-line asset opening remains supported.
- Existing COLMAP datasets and PLY/SPLAT files can be opened as read-only ad hoc sessions.
- The user can choose Save as Project to wrap an ad hoc session in a `.rustscanproject` package.
- New RustSFM control/event APIs are additive; current CLI behavior remains supported.
- New RustGS checkpoint APIs are additive; PLY remains the stable interchange artifact.

## Testing Strategy

### Unit tests

- Project manifest round-trip, schema migration, atomic save, source identity, and artifact validation.
- Pipeline state transitions, invalidation, retry, pause/cancel races, and interrupted lease recovery.
- Deterministic video keyframe selection.
- RustSFM event ordering and cancellation at extraction, matching, mapping, and PnP boundaries.
- RustGS checkpoint round-trip, compatibility rejection, and resumed iteration continuity.

### Integration tests

- Fake workers drive every UI state without running GPU work.
- Image sequence creates a project and uses all images for SFM.
- Video fixture decodes all frames, reconstructs keyframes, and registers all remaining frames.
- Failed PnP blocks training and becomes retryable.
- Application restart resumes RustSFM and RustGS from committed boundaries.
- Completed training writes a loadable lossless PLY and passed parity report.

### Native visual QA

- Capture project library, running pipeline, paused, failed, and completed states on macOS.
- Verify light/dark appearance, 1280x800 and 1728x1117 windows, and narrow resizing.
- Verify the 3D viewport remains the dominant stable region with the inspector collapsed and expanded.
- Verify labels, buttons, and dynamic counts do not overlap or resize fixed tool surfaces.

### Acceptance runs

- A small generated video fixture for fast CI behavior.
- A 96-frame flowers2 preflight for real RustSFM/RustGS integration.
- The full 960-frame flowers2 project as a manual release acceptance run.

## Acceptance Criteria

- A user can create a project from a video or image sequence without using a terminal.
- Every imported frame either has a validated pose or is explicitly shown as unresolved; automatic
  training requires complete coverage.
- Project state survives an application restart at every stage boundary.
- Pause and cancel never advertise partial artifacts as completed.
- RustGS can resume from a compatible checkpoint without restarting at iteration zero.
- Live training snapshots appear in the central 3D viewport without blocking UI interaction.
- A successful project produces one loadable lossless PLY, a passed parity report, and persistent
  reconstruction artifacts.
- Existing direct artifact viewing remains functional.

