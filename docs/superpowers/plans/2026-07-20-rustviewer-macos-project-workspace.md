# RustViewer macOS Project Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RustViewer's fixed control-panel shell with the approved macOS project library and pipeline workspace while preserving the existing full-size wgpu 3DGS viewer and direct artifact opening.

**Architecture:** Split navigation and product surfaces out of the monolithic app while retaining one `ViewerApp` owner for renderer resources. The project library reads immutable summaries from `ProjectStore`; the workspace consumes typed coordinator events and gives the remaining rectangle to the existing viewport after fixed media, inspector-rail, and timeline bands are laid out. UI tests render deterministic fake project states without GPU work, followed by native screenshot and interaction verification.

**Tech Stack:** Rust 2021, eframe/egui 0.34, egui_extras SVG loader, official Lucide SVG assets, existing RustViewer wgpu renderer and RustGS `GpuViewportBridge`, macOS native menus/file dialogs through eframe/rfd.

**Approved visual reference:** Before editing UI code, read `product-design:image-to-code` and use `.superpowers/brainstorm/40861-1784554346/content/project-workspace-v3.html` plus `mac-workspace-directions.html` as the fidelity references. The written dimensions and behavior in the approved design specification take precedence if the temporary companion files are unavailable.

---

## File Map

- Modify `RustViewer/Cargo.toml`: add SVG image loader support.
- Modify `RustViewer/src/app.rs`: navigation shell, coordinator polling, route commands, and renderer ownership.
- Create `RustViewer/src/navigation.rs`: library/workspace/ad-hoc routes and selection state.
- Create `RustViewer/src/scene_session.rs`: independent sparse/snapshot/final scene lifecycle and viewport state.
- Modify `RustViewer/src/ui/mod.rs`: export the new surfaces.
- Modify `RustViewer/src/ui/theme.rs`: system-aware light/dark semantic theme and stable spacing.
- Create `RustViewer/src/ui/icons.rs`: embedded official Lucide icons and icon-button helper.
- Create `RustViewer/assets/icons/plus.svg`, `search.svg`, `folder-open.svg`, `more-horizontal.svg`, `chevron-left.svg`, `play.svg`, `pause.svg`, `square.svg`, `rotate-ccw.svg`, `maximize.svg`, `orbit.svg`, `move.svg`, `zoom-in.svg`, `layers.svg`, `sliders-horizontal.svg`, `info.svg`, `list-filter.svg`, `image.svg`, `video.svg`, `triangle-alert.svg`, `check.svg`, `x.svg`, `panel-right-open.svg`, and `panel-right-close.svg`: official Lucide toolbar assets.
- Create `RustViewer/assets/icons/LICENSE`: upstream Lucide ISC license.
- Create `RustViewer/src/ui/library.rs`: filters, search/sort, project grid, empty state, and context menu.
- Create `RustViewer/src/ui/workspace.rs`: responsive workspace composition.
- Create `RustViewer/src/ui/media_sidebar.rs`: all/keyframe/attention frame browser.
- Create `RustViewer/src/ui/inspector.rs`: 42 px rail and approximately 260 px expandable detail sidebar.
- Create `RustViewer/src/ui/timeline.rs`: stable pipeline segments, progress, iteration/loss/ETA, and controls.
- Create `RustViewer/src/ui/toolbar.rs`: project title, navigation, viewport tools, and display modes.
- Create `RustViewer/src/ui/dialogs.rs`: new project, source choice, invalidation confirmation, errors, quit, delete, and restart sheets.
- Modify `RustViewer/src/ui/viewport.rs`: project-aware empty/loading/error overlays without obscuring the workspace.
- Delete `RustViewer/src/ui/panel.rs` after its still-used robot/layer controls move into inspector tools.
- Modify `RustViewer/src/main.rs`: title, native window sizing, startup route, dropped files, and close behavior.
- Create `RustViewer/tests/ui_state.rs`: route, layout policy, actions, filters, and coordinator projection tests.

### Task 1: Navigation Model and Scene Session

**Files:**
- Create: `RustViewer/src/navigation.rs`
- Create: `RustViewer/src/scene_session.rs`
- Modify: `RustViewer/src/lib.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing route and scene tests**

```rust
#[test]
fn opening_and_closing_project_preserves_library_selection() {
    let id = uuid::Uuid::new_v4();
    let mut navigation = NavigationState::default();
    navigation.open_project(id);
    assert_eq!(navigation.route(), AppRoute::ProjectWorkspace(id));
    navigation.back_to_library();
    assert_eq!(navigation.route(), AppRoute::ProjectLibrary);
    assert_eq!(navigation.selected_project(), Some(id));
}

#[test]
fn scene_session_prefers_final_scene_over_training_snapshot() {
    let mut session = SceneSession::default();
    session.set_training_snapshot(host_splats(8));
    assert_eq!(session.active_source(), Some(SceneSource::TrainingSnapshot));
    session.set_final_scene(host_splats(12));
    assert_eq!(session.active_source(), Some(SceneSource::FinalScene));
}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state navigation_
cargo test -p rust-viewer --test ui_state scene_session_
```

Expected: FAIL because navigation and scene session types do not exist.

- [ ] **Step 3: Implement routes and independent scene state**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppRoute {
    ProjectLibrary,
    ProjectWorkspace(uuid::Uuid),
    AdHocViewer,
}

#[derive(Debug, Clone)]
pub struct NavigationState {
    route: AppRoute,
    selected_project: Option<uuid::Uuid>,
}
```

`SceneSession` owns loaded sparse scene, optional training/final splats, active source, camera, display layers, and dirty flags; it does not own the pipeline coordinator. Keep direct PLY/SPLAT/COLMAP/checkpoint/mesh loaders as methods that select `AdHocViewer`.

- [ ] **Step 4: Run route and scene tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state navigation_
cargo test -p rust-viewer --test ui_state scene_session_
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add RustViewer/src/navigation.rs RustViewer/src/scene_session.rs RustViewer/src/lib.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): separate navigation and scene sessions"
```

### Task 2: Semantic macOS Theme and Lucide Icon Controls

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Modify: `RustViewer/src/ui/theme.rs`
- Create: `RustViewer/src/ui/icons.rs`
- Create: `RustViewer/assets/icons/plus.svg`
- Create: `RustViewer/assets/icons/search.svg`
- Create: `RustViewer/assets/icons/folder-open.svg`
- Create: `RustViewer/assets/icons/more-horizontal.svg`
- Create: `RustViewer/assets/icons/chevron-left.svg`
- Create: `RustViewer/assets/icons/play.svg`
- Create: `RustViewer/assets/icons/pause.svg`
- Create: `RustViewer/assets/icons/square.svg`
- Create: `RustViewer/assets/icons/rotate-ccw.svg`
- Create: `RustViewer/assets/icons/maximize.svg`
- Create: `RustViewer/assets/icons/orbit.svg`
- Create: `RustViewer/assets/icons/move.svg`
- Create: `RustViewer/assets/icons/zoom-in.svg`
- Create: `RustViewer/assets/icons/layers.svg`
- Create: `RustViewer/assets/icons/sliders-horizontal.svg`
- Create: `RustViewer/assets/icons/info.svg`
- Create: `RustViewer/assets/icons/list-filter.svg`
- Create: `RustViewer/assets/icons/image.svg`
- Create: `RustViewer/assets/icons/video.svg`
- Create: `RustViewer/assets/icons/triangle-alert.svg`
- Create: `RustViewer/assets/icons/check.svg`
- Create: `RustViewer/assets/icons/x.svg`
- Create: `RustViewer/assets/icons/panel-right-open.svg`
- Create: `RustViewer/assets/icons/panel-right-close.svg`
- Create: `RustViewer/assets/icons/LICENSE`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Add SVG support and write failing theme tests**

Add:

```toml
egui_extras = { version = "0.34.3", features = ["svg"] }
```

Test that both appearances meet the fixed semantic token contract and that icon buttons remain 28x28 regardless of tooltip/label length:

```rust
assert_eq!(MacTheme::light().metrics.icon_button, egui::vec2(28.0, 28.0));
assert_eq!(MacTheme::dark().metrics.inspector_rail_width, 42.0);
assert!(contrast_ratio(MacTheme::light().text_primary, MacTheme::light().window_bg) >= 4.5);
assert!(contrast_ratio(MacTheme::dark().text_primary, MacTheme::dark().window_bg) >= 4.5);
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state theme_
```

Expected: FAIL because semantic light/dark tokens do not exist.

- [ ] **Step 3: Implement system-aware semantic tokens**

Define `MacTheme { appearance, window_bg, sidebar_bg, viewport_bg, panel_bg, separator, text_primary, text_secondary, accent, success, warning, error, metrics }`. Use restrained neutral grays, Apple blue only for selection/primary commands, green/warning/red only for status. Set system proportional fonts, font size 13 for body, 11 for captions, and 18 for page titles; keep letter spacing at zero.

Use `ctx.system_theme()` when available and fall back to dark only if the OS does not report an appearance. Do not apply the previous always-dark visual override.

- [ ] **Step 4: Embed official Lucide assets**

Vendor only the required upstream Lucide SVG files plus its ISC license: `plus`, `search`, `folder-open`, `more-horizontal`, `chevron-left`, `play`, `pause`, `square`, `rotate-ccw`, `maximize`, `orbit`, `move`, `zoom-in`, `layers`, `sliders-horizontal`, `info`, `list-filter`, `image`, `video`, `triangle-alert`, `check`, `x`, `panel-right-open`, and `panel-right-close`.

Register egui image loaders once and render icons with `egui::Image::from_bytes`. `icon_button(ui, icon, tooltip, selected)` always allocates 28x28, paints only hover/selected backgrounds, and calls `response.on_hover_text(tooltip)`. No emoji remains in visible controls.

- [ ] **Step 5: Run theme tests and scan emoji controls**

Run:

```bash
cargo test -p rust-viewer --test ui_state theme_
rg -n '📂|✨|🔷|🗂|⚠️|⏹|▶|🎯' RustViewer/src
```

Expected: tests PASS and `rg` prints no matches.

- [ ] **Step 6: Commit Task 2**

```bash
git add RustViewer/Cargo.toml Cargo.lock RustViewer/assets RustViewer/src/ui/theme.rs RustViewer/src/ui/icons.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): adopt macOS theme and Lucide controls"
```

### Task 3: Project Library State and Layout

**Files:**
- Create: `RustViewer/src/ui/library.rs`
- Modify: `RustViewer/src/ui/mod.rs`
- Modify: `RustViewer/src/app.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing filter/sort/action tests**

Create summaries for running, completed, paused, and recently opened projects. Require filters and case-insensitive search to compose, sorting to be stable, and clicking a card to emit `LibraryAction::OpenProject(id)`.

```rust
let visible = state.visible_projects(&projects);
assert_eq!(visible.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["Flowers", "Office"]);
assert_eq!(draw_library_for_test(&mut state, &projects, clicked_card), LibraryAction::OpenProject(flowers_id));
```

- [ ] **Step 2: Run library tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state library_
```

Expected: FAIL because library UI state does not exist.

- [ ] **Step 3: Implement library view model**

```rust
pub enum LibraryFilter { All, Processing, Completed, Recent }
pub enum LibrarySort { UpdatedDescending, NameAscending, CreatedDescending }
pub enum LibraryAction {
    None,
    NewProject,
    OpenProject(uuid::Uuid),
    Reveal(uuid::Uuid),
    Duplicate(uuid::Uuid),
    Delete(uuid::Uuid),
    OpenAdHoc,
}
```

Keep search, filter, and sort values when returning from a project. The store supplies summaries; UI filtering never touches manifests on disk.

- [ ] **Step 4: Implement the approved library composition**

Draw a 188 px left sidebar with All Projects, Processing, Completed, and Recently Opened. Draw one emphasized New Project button at the top. The unframed main area uses a compact toolbar with title, result count, search, sort menu, and direct-artifact open menu. Use a responsive grid with minimum 240 px columns and 12 px gaps.

Each project card has maximum 320 px width, 6 px radius, a 16:10 real project thumbnail, name, five fixed-width 28 px pipeline segments, one status line, and updated time. Context actions live in the ellipsis menu. Do not nest status cards inside the project card.

- [ ] **Step 5: Run library tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state library_
```

Expected: PASS for filters, sorting, actions, keyboard selection, and stable card dimensions.

- [ ] **Step 6: Commit Task 3**

```bash
git add RustViewer/src/ui/library.rs RustViewer/src/ui/mod.rs RustViewer/src/app.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): add project library"
```

### Task 4: Workspace Geometry with Dominant 3DGS Viewport

**Files:**
- Create: `RustViewer/src/ui/workspace.rs`
- Create: `RustViewer/src/ui/toolbar.rs`
- Modify: `RustViewer/src/app.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing layout-policy tests**

Test exact dimensions at 1280x800 and 1728x1117, both collapsed and expanded inspector states:

```rust
let collapsed = WorkspaceLayout::compute(egui::vec2(1280.0, 800.0), false);
assert_eq!(collapsed.inspector.width(), 42.0);
assert_eq!(collapsed.timeline.height(), 94.0);
assert!(collapsed.viewport.width() >= 800.0);
assert!(collapsed.viewport.area() > collapsed.media.area() + collapsed.inspector.area());

let expanded = WorkspaceLayout::compute(egui::vec2(1280.0, 800.0), true);
assert_eq!(expanded.inspector.width(), 260.0);
assert!(expanded.viewport.width() >= 620.0);
assert_eq!(collapsed.viewport.top(), expanded.viewport.top());
```

- [ ] **Step 2: Run layout tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state workspace_layout_
```

Expected: FAIL because workspace layout policy does not exist.

- [ ] **Step 3: Implement stable layout policy**

Use constants:

```rust
pub const TOOLBAR_HEIGHT: f32 = 44.0;
pub const MEDIA_WIDTH: f32 = 220.0;
pub const INSPECTOR_RAIL_WIDTH: f32 = 42.0;
pub const INSPECTOR_EXPANDED_WIDTH: f32 = 260.0;
pub const TIMELINE_RUNNING_HEIGHT: f32 = 94.0;
pub const TIMELINE_COMPLETE_HEIGHT: f32 = 42.0;
pub const MIN_VIEWPORT_WIDTH: f32 = 560.0;
```

The toolbar and timeline consume horizontal bands first. Media consumes a left strip. Inspector consumes only 42 px unless a tool is selected and width allows expansion. The viewport receives the remaining central rectangle and never sits inside a decorative card. Below 920 px, inspector expansion overlays the viewport with a solid panel rather than shrinking it below `MIN_VIEWPORT_WIDTH`; the 42 px rail remains fixed.

- [ ] **Step 4: Route existing renderer into the computed viewport**

Move the current central rendering callback, camera input, depth picking, robot navigation, and `GpuViewportBridge` drawing into `draw_project_viewport(ui, rect, scene_session)`. Preserve the exact allocated viewport rect across loading labels, snapshots, hover controls, and empty states so dynamic content cannot resize it.

- [ ] **Step 5: Implement the compact toolbar**

Use back chevron, project title/status, then grouped orbit/pan/zoom/fit and display mode controls. Icon tools have tooltips; display modes use a segmented control. Place destructive/task controls in timeline or menus, not in the central toolbar.

- [ ] **Step 6: Run workspace tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state workspace_layout_
cargo test -p rust-viewer --test loader_integration_test
```

Expected: PASS and existing scene loaders remain functional.

- [ ] **Step 7: Commit Task 4**

```bash
git add RustViewer/src/ui/workspace.rs RustViewer/src/ui/toolbar.rs RustViewer/src/app.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): center project workspace on 3D viewport"
```

### Task 5: Media Sidebar and Collapsible Inspector Rail

**Files:**
- Create: `RustViewer/src/ui/media_sidebar.rs`
- Create: `RustViewer/src/ui/inspector.rs`
- Modify: `RustViewer/src/ui/panel.rs`
- Modify: `RustViewer/src/ui/workspace.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing media/inspector state tests**

```rust
let mut media = MediaSidebarState::default();
media.filter = MediaFilter::NeedsAttention;
assert_eq!(media.visible_frames(&frames).iter().map(|f| f.id).collect::<Vec<_>>(), [7, 9]);

let mut inspector = InspectorState::default();
inspector.select(InspectorTool::Scene);
assert!(inspector.is_expanded());
inspector.select(InspectorTool::Scene);
assert!(!inspector.is_expanded());
assert_eq!(inspector.rail_width(), 42.0);
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state media_sidebar_
cargo test -p rust-viewer --test ui_state inspector_
```

Expected: FAIL because state models do not exist.

- [ ] **Step 3: Implement media sidebar**

Add All Frames, Keyframes, and Needs Attention tabs. Render virtualized 56 px rows with 48x36 thumbnails, stable frame number, keyframe mark, and registration status icon. Selection updates the inspector but does not change the 3D viewport size. A failed frame's context menu exposes Retry and Use Wider Search only.

- [ ] **Step 4: Implement inspector rail and panels**

The rail contains Scene, Reconstruction, Training, Project, and Log icon tools. Selecting the active icon collapses; selecting another swaps content without changing width. Move scene layers, camera statistics, robot controls, SFM diagnostics, training configuration, and technical log out of the old left panel. Use unframed sections with separators; numeric settings use sliders/steppers, binary settings use toggles, and option sets use menus.

Keep iteration, current loss, ETA, and task progress out of the inspector; they belong to the timeline.

- [ ] **Step 5: Remove the oversized legacy panel**

After all still-used controls have moved, remove `draw_side_panel` calls and delete `RustViewer/src/ui/panel.rs`. Re-export `UiState` replacements from their owning modules. Verify no fixed 300 px control panel remains.

- [ ] **Step 6: Run media/inspector tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state media_sidebar_
cargo test -p rust-viewer --test ui_state inspector_
rg -n 'exact_size\(300\.0\)|draw_side_panel' RustViewer/src
```

Expected: tests PASS and `rg` prints no matches.

- [ ] **Step 7: Commit Task 5**

```bash
git add RustViewer/src/ui/media_sidebar.rs RustViewer/src/ui/inspector.rs RustViewer/src/ui/workspace.rs RustViewer/src/ui/panel.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): add media and compact inspector sidebars"
```

### Task 6: Pipeline Timeline, Metrics, and Task Controls

**Files:**
- Create: `RustViewer/src/ui/timeline.rs`
- Modify: `RustViewer/src/ui/workspace.rs`
- Modify: `RustViewer/src/app.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing timeline projection tests**

Project manifests and coordinator progress into five segments. Require complete, running, paused, failed, and Needs Attention labels, plus ETA only after enough samples:

```rust
let model = TimelineModel::from_project(&manifest, &progress_history);
assert_eq!(model.segments.len(), 5);
assert_eq!(model.active.unwrap().title, "3DGS Training");
assert_eq!(model.iteration, Some((1_240, 30_000)));
assert_eq!(model.loss, Some(0.0312));
assert!(model.eta.is_some());
assert_eq!(model.primary_action, TimelineAction::Pause);
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state timeline_
```

Expected: FAIL because timeline projection does not exist.

- [ ] **Step 3: Implement model and robust ETA**

Keep the last 30 progress samples. Compute rate from the median of positive per-sample rates after at least five samples and 3 seconds; suppress ETA for unknown totals, stalled work, and non-finite rates. For training, base ETA on iterations. Expose elapsed independently.

- [ ] **Step 4: Draw the stable bottom timeline**

Draw five equal segment tracks with current stage and status. The second row contains current operation, progress fraction, iteration, loss, Gaussian count, elapsed, and ETA in fixed minimum-width cells, followed by icon controls for pause/resume, cancel, retry, or start. Long operation names elide with a hover tooltip and never increase timeline height.

When Complete succeeds, collapse to 42 px showing completion time, final iteration/loss/Gaussians, Reveal in Finder, and Export/Open commands.

- [ ] **Step 5: Wire typed coordinator actions**

Map timeline actions only to `PipelineCommand`; the UI must not mutate `ProjectManifest` directly. `PauseRequested` disables duplicate pause input but retains Cancel. Retry is shown only for Failed/Cancelled/Needs Attention and names the affected stage.

- [ ] **Step 6: Run timeline tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state timeline_
```

Expected: PASS for all states, ETA stability, fixed height, and action mapping.

- [ ] **Step 7: Commit Task 6**

```bash
git add RustViewer/src/ui/timeline.rs RustViewer/src/ui/workspace.rs RustViewer/src/app.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): add persistent pipeline timeline"
```

### Task 7: New Project, Errors, Invalidation, and Quit Sheets

**Files:**
- Create: `RustViewer/src/ui/dialogs.rs`
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/main.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Write failing dialog-decision tests**

Require New Project to distinguish video/file sequence, default managed copy, and refuse creation with fewer than two images. Require setting changes to list invalidated stages before applying. Require a running close request to offer Pause and Quit, Cancel Task and Quit, and Keep Running.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test ui_state dialog_
```

Expected: FAIL because dialog state is absent.

- [ ] **Step 3: Implement modal state and native pickers**

Use `rfd` for video and multi-image selection. Present Source, Ownership, Project Name, and Destination in a compact sheet. Do not add explanatory feature copy. Dragging a video/image group onto the library opens the same sheet; dragging PLY/SPLAT/COLMAP opens an ad-hoc session.

Invalidation confirmation lists exact affected stages and preserves old results until replacements succeed. Delete and Start Fresh require destructive confirmation; Reveal and Open Log are non-destructive.

- [ ] **Step 4: Handle errors in place**

Keep media and viewport visible. Mark only the affected timeline segment, show a one-line plain summary, and place technical detail in Log inspector. Map error suggestions to Retry, Use Wider Search, Reveal Source, Open Log, and Start Fresh. Never replace the workspace with a generic failure page.

- [ ] **Step 5: Handle close requests and unclean resume**

Intercept viewport close requests. If idle, persist navigation/project state and close. If running, cancel the close and show the three approved choices. Pause and Quit waits for `Paused` plus a committed checkpoint before issuing a new close command. On next open after an interrupted lease, require explicit Resume/Retry confirmation; never auto-resume.

- [ ] **Step 6: Run dialog and route tests**

Run:

```bash
cargo test -p rust-viewer --test ui_state dialog_
cargo test -p rust-viewer --test ui_state close_request_
```

Expected: PASS for validation, invalidation explanation, error commands, and all close outcomes.

- [ ] **Step 7: Commit Task 7**

```bash
git add RustViewer/src/ui/dialogs.rs RustViewer/src/app.rs RustViewer/src/main.rs RustViewer/tests/ui_state.rs
git commit -m "feat(viewer): add native project workflow sheets"
```

### Task 8: App Integration and Direct-Open Compatibility

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/main.rs`
- Modify: `RustViewer/src/ui/viewport.rs`
- Test: `RustViewer/tests/loader_integration_test.rs`
- Test: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Split `ViewerApp` responsibilities without changing renderer ownership**

Keep wgpu device/context, renderer bridges, and eframe callback resources in `ViewerApp`. Move view state into navigation/library/workspace/scene structs. Poll coordinator events once per frame, update the current manifest/timeline, and call `ctx.request_repaint()` only while a task, animation, or camera interaction is active.

- [ ] **Step 2: Preserve startup asset behavior**

`rust-viewer --gaussian scene.ply`, positional PLY/SPLAT/checkpoint/mesh paths, and COLMAP directories still open `AdHocViewer`. Add `.rustscanproject` detection to open a project workspace. The ad-hoc toolbar includes Save as Project, which creates a package referencing or copying the loaded artifacts without rerunning computation.

- [ ] **Step 3: Add application menus and shortcuts**

Implement File menu commands for New Project (`Cmd-N`), Open (`Cmd-O`), Close Project (`Cmd-W`), Save as Project, Reveal in Finder, and Quit (`Cmd-Q`); Edit search focus (`Cmd-F`); View fit scene, inspector toggle, and display modes. Commands dispatch through the same action enums used by buttons.

- [ ] **Step 4: Run all RustViewer tests**

Run:

```bash
cargo fmt --all -- --check
cargo test -p rust-viewer
cargo clippy -p rust-viewer --all-targets -- -D warnings
```

Expected: all commands exit 0, including existing loader and training session tests.

- [ ] **Step 5: Commit Task 8**

```bash
git add RustViewer/src/app.rs RustViewer/src/main.rs RustViewer/src/ui/viewport.rs RustViewer/tests
git commit -m "feat(viewer): integrate project workspace shell"
```

### Task 9: Native Visual and End-to-End Acceptance

**Files:**
- Modify: `RustViewer/tests/ui_state.rs`

- [ ] **Step 1: Build and start the native application**

Run:

```bash
cargo build --release -p rust-viewer
target/release/rust-viewer
```

Expected: the project library opens in a 1280x800 native window with no terminal interaction required.

- [ ] **Step 2: Capture required visual states**

Using deterministic fake project manifests, capture project library, running SFM, running training, paused, failed PnP, completed collapsed timeline, inspector collapsed, and inspector expanded at 1280x800 and 1728x1117 in light and dark appearances.

Verify for every capture:

- Central viewport is the largest workspace region.
- Inspector rail is 42 px collapsed and approximately 260 px expanded.
- No white placeholder appears when a scene/snapshot is available.
- Text and counts do not overlap, clip buttons, or resize tool surfaces.
- There are no emoji, nested cards, gradients, or one-hue full-window palettes.
- Project thumbnails show actual source imagery.

- [ ] **Step 3: Verify real interactive rendering**

Open `test_data/flowers2/rustgs_training/flowers2-3dgs.ply`, capture the viewport, and compare canvas pixels against the empty viewport background: at least 5% of central viewport pixels must differ. Orbit, pan, zoom, fit, switch display mode, expand/collapse inspector, and confirm the viewport remains nonblank and correctly framed after each action.

- [ ] **Step 4: Run the 96-frame automatic project workflow**

Create a project from the first 96 flowers2 images through the New Project sheet. Let it reach Complete. Expected: all five timeline segments succeed, live snapshots appear without blocking input, final PLY loads in the same central viewport, and reopening the application returns to the completed project.

- [ ] **Step 5: Run the full 960-frame manual release acceptance**

Import all images from `test_data/flowers2`, verify Import reports 960/960 and image SFM uses all 960 images, pause/resume once during SFM and once during RustGS, and finish. Expected: every frame has a finite validated pose, the final lossless PLY and passed parity report are present, and reopening browses the final scene without recomputation.

- [ ] **Step 6: Commit Task 9**

```bash
git add RustViewer/tests/ui_state.rs
git commit -m "test(viewer): verify native project experience"
```
