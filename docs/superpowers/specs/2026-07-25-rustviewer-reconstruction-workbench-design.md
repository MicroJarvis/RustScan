# RustViewer Reconstruction Workbench Design

## Goal

Create a browser-viewable HTML prototype for a redesigned RustViewer desktop workbench. It demonstrates a single coherent workflow from input selection through pose reconstruction, Gaussian training, and rendered scene inspection.

## Audience

Technical operators processing image sequences or video captures into a 3D Gaussian scene. They need to know which stage is active, whether it is healthy, and what the renderer is producing without navigating between separate pages.

## Visual Direction

- Dark, neutral workstation surface with blue as the primary action color and restrained green, amber, and red state colors.
- Dense but calm information hierarchy: command bar, reconstruction stage rail, primary viewport, inspector, and bottom activity strip.
- Panels use thin separators and subtle surface shifts rather than nested cards or decorative gradients.
- Controls are icon-led where familiar and expose browser-native tooltips through `title` attributes.

## Layout

1. Top command bar: project name, source summary, actions, GPU status.
2. Stage rail: Import, Pose Solve, Train, Render. Each stage exposes ready, running, completed, and blocked states.
3. Main viewport: Canvas-based point-cloud and Gaussian-scene representation with camera controls and a training overlay.
4. Right inspector: source metadata, current reconstruction metrics, and render controls.
5. Bottom activity strip: ordered event log and frame sequence progress.

## Interaction Model

- Selecting an input type changes the metadata shown in the inspector.
- Running the workflow advances through pose solving and training with visibly changing metrics and viewport content.
- Pause freezes progress; resume continues it; reset restores the import-ready state.
- Layer toggles show and hide camera, sparse point, and Gaussian representations.

## Scope

This is a visual and interaction prototype served only through the local brainstorming preview. It does not launch RustSFM, decode media, allocate GPU training resources, or write scene files. Those integrations remain a separate RustViewer implementation change.
