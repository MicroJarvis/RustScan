# RustViewer Design QA

- Source visual truth: user-provided workbench design screenshot in the conversation. The attachment was not materialized as a local file, so it cannot be included in a pixel-overlay comparison.
- Implementation screenshot: `/private/tmp/rustviewer-design-pass-02.png`
- Viewport: native window `1280 x 800` content, captured with its macOS title bar as `1280 x 832` logical points.
- State: empty project, no imported image sequence.

## Evidence

- Full-view evidence: `/private/tmp/rustviewer-design-pass-02.png` was captured from the native RustViewer process after the final build.
- Focused-region evidence: the top command bar, left pipeline rail, viewport toolbar and status card, inspector, and activity/frame strip were visually checked from the same capture.
- The source screenshot is unavailable as a local image, so a combined side-by-side evidence image cannot be produced in this workspace.

## Findings

- [P1] Project state differs from the reference.
  - Location: capture count, pipeline progress, viewport status card, and frame sequence.
  - Evidence: the reference uses an imported image-sequence state; the verified native capture has no active project and correctly displays zero frames.
  - Impact: an exact pixel comparison is not meaningful across these data states.
  - Fix: open a real imported project, then capture the same state for a final comparison. Do not inject mock frame data into the product UI.

- [P3] Native chrome remains platform-provided.
  - Location: macOS title bar and window controls.
  - Evidence: the capture includes standard native window chrome outside the egui workbench surface.
  - Impact: this does not affect the in-app layout.
  - Fix: compare only the workbench content region for pixel-level review.

## Patches Since Prior QA

- Added a macOS CJK fallback using `Hiragino Sans GB` for proportional and monospace egui text.
- Added a regression test that asserts the fallback is first in both font-family chains.
- Captured `/private/tmp/rustviewer-design-pass-02.png`; all Chinese workbench text now renders as glyphs rather than missing-glyph boxes.

## Implementation Checklist

- [x] Register the CJK font once per egui context.
- [x] Put the font before egui defaults for proportional and monospace text.
- [x] Build the native application and visually verify the rendered Chinese UI.
- [ ] Repeat the comparison with the original reference asset and the same imported-project state.

## Follow-up Polish

- Use the real imported sequence from the reference state to tune data-dependent labels, counts, and frame-strip density.

final result: blocked
