# RustViewer Pipeline State Sync Design

## Goal

Keep the RustViewer workbench controls consistent with the persisted project stage
state when progress events fill the in-memory event queue.

## Problem

`PipelineCoordinator` emits progress and manifest events with bounded, nonblocking
channels. A backgrounded window can stop draining events while a long RustSFM
stage continues. Progress fills the queue and the later failure manifest can be
dropped. The on-disk manifest becomes failed while the workbench keeps its last
running summary and disables its retry command.

## Decision

The app refreshes `ProjectSessionSummary` from the coordinator store after every
pipeline drive. The manifest is the source of truth for stage controls; pipeline
events remain responsible for activity details and dataset loading. A transition
to a persisted pose failure also clears the loading state and exposes the saved
error.

## Alternatives

Increasing the event queue only delays the failure and cannot make UI state
durable. Making failure sends blocking risks stalling worker completion when the
window is suspended. A dedicated latest-state channel would work but adds a
second synchronization protocol. Reading the already-owned store is smaller and
ensures the UI uses the authoritative state.

## Verification

Add an app regression test that starts with an in-memory running summary and a
coordinator whose manifest is failed. After the synchronization step, the
snapshot must expose `Retry pose solve` and loading must be cleared. Run the
focused app test, the RustViewer test suite, and a release build.
