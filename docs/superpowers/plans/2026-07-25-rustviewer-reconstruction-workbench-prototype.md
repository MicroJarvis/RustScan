# RustViewer Reconstruction Workbench Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-viewable, interactive HTML prototype of the redesigned RustViewer reconstruction workbench.

**Architecture:** A single self-contained HTML document is served by the existing local brainstorming preview. CSS establishes the desktop workstation layout and JavaScript owns a small deterministic state machine for the simulated import, pose, training, and render stages. A Canvas renderer visualizes sparse points, cameras, and Gaussian splats without requiring a GPU service.

**Tech Stack:** Semantic HTML, CSS custom properties, Canvas 2D, vanilla JavaScript, Lucide icons loaded from CDN, local brainstorming preview server.

---

### Task 1: Build the workbench shell and static operational surfaces

**Files:**
- Create: `.superpowers/brainstorm/26647-1784874940/content/03-reconstruction-workbench.html`

- [ ] **Step 1: Add semantic application regions**

Create `header`, `nav`, `main`, `aside`, and `footer` regions with these labels: `Project`, `Capture`, `Pose`, `Train`, `Render`, `Scene Inspector`, and `Activity`. Add Lucide-backed icon buttons with `aria-label` and `title` for reset, layer visibility, fit view, and settings.

```html
<header class="command-bar">...</header>
<nav class="stage-rail" aria-label="Reconstruction stages">...</nav>
<main class="viewport-shell"><canvas id="scene-canvas"></canvas></main>
<aside class="inspector" aria-label="Scene Inspector">...</aside>
<footer class="activity-strip">...</footer>
```

- [ ] **Step 2: Add stable desktop layout styling**

Use CSS grid tracks `56px minmax(152px, 188px) minmax(0, 1fr) minmax(276px, 336px)` for the primary shell and reserve a 164px activity strip. Define neutral surface, divider, primary, success, warning, and danger tokens; constrain text with wrapping and overflow handling.

- [ ] **Step 3: Render a deterministic scene canvas**

Implement a canvas draw routine that projects seeded sparse points, camera frustums, and Gaussian splats into the viewport. Layer toggle state controls each group and status text reflects the current selected stage.

```js
function drawScene() {
  const { width, height } = resizeCanvas(canvas);
  clearViewport(width, height);
  if (workflow.showPoints) drawSparsePoints(width, height);
  if (workflow.showCameras) drawCameras(width, height);
  if (workflow.showGaussians) drawGaussians(width, height);
}
```

- [ ] **Step 4: Verify the static shell manually**

Open `http://localhost:50555` and confirm the viewport, inspector, stage rail, and activity strip fit in a 1440px-wide browser without clipped text or overlapping panels.

### Task 2: Add workflow state and realistic interactions

**Files:**
- Modify: `.superpowers/brainstorm/26647-1784874940/content/03-reconstruction-workbench.html`

- [ ] **Step 1: Define the prototype state machine**

Add a `workflow` object with `phase`, `progress`, `paused`, `inputKind`, `showCameras`, `showPoints`, and `showGaussians`. Define the legal phase order `import -> pose -> train -> render` and reset all derived metrics on `resetWorkflow()`.

```js
const workflow = {
  phase: 'import', progress: 0, paused: false, inputKind: 'sequence',
  showCameras: true, showPoints: true, showGaussians: true,
  poses: 0, gaussians: 0, psnr: 0, timer: null,
};
const phases = ['import', 'pose', 'train', 'render'];
```

- [ ] **Step 2: Implement user commands**

Wire `Run reconstruction`, `Pause`, `Resume`, `Reset`, input switching, and layer toggles. Each command updates the stage rail, activity log, inspector metrics, and canvas in the same animation frame.

- [ ] **Step 3: Simulate progress without runaway timers**

Use one retained interval handle. Advance pose progress to 100%, then transition into training; training increases Gaussian count and PSNR until render-ready. Clear the interval on pause, reset, and completed state.

```js
function tickWorkflow() {
  if (workflow.phase === 'pose') advancePose();
  else if (workflow.phase === 'train') advanceTraining();
  renderUi();
  drawScene();
}
function stopTimer() {
  window.clearInterval(workflow.timer);
  workflow.timer = null;
}
```

- [ ] **Step 4: Verify interactive behavior manually**

Confirm that running the workflow changes all four stage states, pause freezes values, resume continues them, reset restores the initial scene, and every layer toggle visibly changes the canvas.

### Task 3: Present the prototype for selection

**Files:**
- Modify: `.superpowers/brainstorm/26647-1784874940/content/03-reconstruction-workbench.html`

- [ ] **Step 1: Add a compact prototype note**

Add a non-intrusive status label in the activity strip identifying this as an interaction prototype, not a live reconstruction run.

- [ ] **Step 2: Check browser preview availability**

Verify `.superpowers/brainstorm/26647-1784874940/state/server-info` exists and that `http://localhost:50555` is the current preview URL.

- [ ] **Step 3: Hand off the preview**

Provide the local preview URL and ask for concrete feedback on hierarchy, density, and the next interaction to refine.
