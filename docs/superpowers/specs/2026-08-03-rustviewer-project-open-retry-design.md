# RustViewer Project Open And Retry Design

## Goal

Allow RustViewer to reopen an existing `.rustscanproject` package from either a startup argument or the file controls. A recovered failed pose solve must show the existing `Retry pose solve` primary command without changing the persisted project state.

## Approach

`AssetLoadKind` will gain a project-package variant. Extension detection will classify `.rustscanproject` before the generic Gaussian fallback. Its handler will construct the existing `PipelineCoordinator` through `new_project_pipeline`, then reuse the imported-project state reset and manifest-summary setup already used after image import.

The file controls will expose `Open RustScan Project`. The dialog will select project package directories rather than treating them as point-cloud files. Startup and UI loading therefore use the same `AppCommand::LoadAsset` path.

## Failure Handling

Opening a locked, malformed, or invalid package leaves the current project intact and reports the `ProjectStore` error in the load-error area. Opening a valid failed package does not enqueue a reconstruction command. The primary command comes solely from the recovered manifest, so it remains an explicit user action.

## Verification

Unit tests will create a small imported project, mark `KeyframeSfm` as retryable failed, then open it through the application handler. They will assert the project summary is restored and `WorkbenchSnapshot::primary_command()` is enabled with `Retry pose solve`. A separate classification test will protect `.rustscanproject` startup handling.
