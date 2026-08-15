# RustViewer RustSFM-to-RustGS Orchestration Design

## Goal

Let an operator select a directory of JPEG or PNG images in RustViewer and run one
workflow that performs RustSFM sparse reconstruction, loads its COLMAP-compatible
output, and starts RustGS 3D Gaussian Splatting training only after reconstruction
succeeds.

## Scope

This change adds native RustSFM orchestration to RustViewer. It does not add video
decoding, EXIF calibration, persistent project files, pause or cancellation during
RustSFM reconstruction, or a second reconstruction backend.

## User Workflow

1. The operator selects a folder containing at least two `.jpg`, `.jpeg`, or `.png`
   images.
2. RustViewer creates a unique output directory below the selected folder:
   `.rustviewer/rustsfm/<run-id>/`.
3. A background worker runs RustSFM using its automatic single-camera intrinsics
   estimate and image-only local matching.
4. RustViewer reports RustSFM registration callbacks while reconstruction runs.
5. On success, RustViewer loads the generated COLMAP dataset from the run directory.
6. When that load succeeds, RustViewer starts the existing RustGS training manager.
7. A RustSFM or COLMAP-load failure prevents training, retains the run directory,
   and surfaces the error in the application state.

## Architecture

RustViewer will depend directly on the `rustsfm` library and invoke
`run_reconstruction_with_callbacks` from a worker thread. A narrow viewer-local
runner abstraction will own construction of the `MapperConfig` and translate
RustSFM callback events into viewer messages. This keeps the UI thread free and
makes pipeline sequencing unit-testable without invoking a real image
reconstruction.

The mapper configuration uses RustSFM defaults except for the values required for
raw image folders:

- `input`: the selected image directory.
- `output`: the newly created, unique app-managed run directory.
- `local_matching`: `true`, because no precomputed COLMAP database is required.
- `copy_images`: `true`, so the exported COLMAP dataset contains the image root
  expected by RustGS.

The camera values `fx`, `fy`, `cx`, and `cy` remain unset. RustSFM therefore uses
its existing automatic single-camera estimate. The change does not force GPU SIFT
or GPU PnP: on macOS wgpu uses Metal rather than Vulkan, and GPU PnP is incompatible
with focal-length estimation in RustSFM.

## Output and Safety

Each run receives a collision-free identifier, so running the same input again
never overwrites or deletes prior sparse reconstructions or trained assets. RustSFM
writes its standard COLMAP-compatible `images/` and `sparse/` layout, and the
existing `load_colmap_training_dataset` function remains the only conversion point
to RustGS `TrainingDataset`.

The worker must verify that RustSFM reported at least one registered image and one
sparse point before attempting COLMAP loading. It must also treat any loader error
as a terminal reconstruction error rather than falling through to training.

## State and Errors

RustViewer will add distinct reconstruction states for image selection, running,
completed, and failed. Existing `Start Training` remains available for manually
loaded COLMAP datasets. The one-click image workflow uses an internal command that
starts training only after its `ColmapLoaded` success message.

RustSFM's callback trait is notification-only; this scope intentionally disables
new reconstruction starts while one is active and does not expose a cancel button
for that stage. Existing cancellation behavior applies only after RustGS training
has begun.

## Testing

Tests will first define and verify the following behavior:

- A raw-image run builds a local-matching RustSFM configuration with a unique
  output directory below the source folder.
- Successful RustSFM completion loads the output and begins RustGS training.
- RustSFM failure, invalid summary, or invalid COLMAP output does not start RustGS.
- Callback data updates RustViewer's reconstruction progress without direct UI
  thread mutation.
- Existing manual COLMAP loading and RustGS training tests remain valid.

The RustSFM runner itself will be injected behind a small trait in unit tests;
real end-to-end reconstruction remains an integration concern because it requires
image content, CPU/GPU resources, and substantial runtime.
