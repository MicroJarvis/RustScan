# Q1 diagnostic fix: split NotConverged into three status codes (2026-09-14)

## Decision and scope

Single-variable round. The only change is the slot status value written by three
already-distinct rejection paths in `five_point_recovery.wgsl`, plus the host
decode of those values. **No threshold, no rejection logic, no arithmetic.**

**Retain.** Model signature is bit-identical to the pre-fix capture.
Diagnostic signature advances to v2. The previous zero-root inference of the
three path counts is reproduced exactly by the new status codes, and the
inference's unresolved split (RootNotDone vs PolishRejected at the iteration
cap) is now measured: all 20,786 at-cap after-store rejections are
`RootNotDone`; polish rejection occurs only below the cap (130 slots).

## What changed

| path | shader line | old status | new status | float code |
| --- | --- | --- | --- | --- |
| real-axis residual gate (`final_eval` at Re(z)) | 85 | `NotConverged` | `RealAxisRejected` | 3.0 |
| Aberth `done[i]` still false | 88 | `NotConverged` | `RootNotDone` | 7.0 |
| real Newton polish residual | 105 | `NotConverged` | `PolishRejected` | 8.0 |

Header field `unconverged` (`ob+4`) is still incremented on all three paths and
remains the sum. Decode derives `real_axis_rejected`, `root_not_done`, and
`polish_rejected` from slot statuses; they are required to sum to `unconverged`.

`FivePointSlotStatus::NotConverged` is removed. `is_unconverged()` covers the
three new variants. Protected historical harnesses that counted `NotConverged`
slots were updated to `is_unconverged()` so they keep compiling; their
behaviour is otherwise unchanged.

## Provenance

- Source commit baseline `3ab9c09`; solver arithmetic and thresholds unchanged.
- Changed: `five_point_recovery.wgsl`, `five_point_f32_complete.rs`,
  `five_point_f32_tests.rs`; minimal compile fixes in
  `five_point_gpu_profile_replay.rs`, `five_point_gpu_roots_layout_replay.rs`,
  `five_point_gpu_roots_attribution.rs`.
- New: `examples/five_point_gpu_q1_diagnostics.rs`;
  `model_bits` / `model_signature` helpers in
  `five_point_gpu_capacity_replay.rs`.
- Pre-fix model signature (essentials + model_count only), captured before
  the shader edit:
  `030f43068d9e8f78961142915162f20bef4ecb909d49d247a4f766a2ab5412df`.
- Diagnostic signature v1 (historical):
  `1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`.
- Diagnostic signature v2:
  `3cd8091d6033f2fdb0b598c27483e67c4dd3ee1af1232ff4f1f5ee1da86117cb`.
- Fixed input unchanged: digest `af07459d…c5`, CPU f64 signature `9e6764b4…3a`.

## Results

### Model identity

Model signature after the fix equals the pre-fix capture. Candidate essential
matrices and per-trial model counts are bit-identical. The diagnostic signature
changed, as required.

### Cross-check against the previous inference

| quantity | previous zero-root inference | status-code count | match |
| --- | --- | --- | --- |
| RealAxisRejected | 276,906 | 276,906 | yes |
| PolishRejected with `root_iterations < 384` | 130 | 130 | yes |
| RootNotDone ∪ PolishRejected at cap | 20,786 | 20,786 | yes |
| header `unconverged` | 297,822 | 297,822 = 276,906+20,786+130 | yes |

### New information the inference could not provide

| status | total | of which at iteration cap |
| --- | --- | --- |
| `RootNotDone` | **20,786** | **20,786** |
| `PolishRejected` | **130** | **0** |

Every post-store rejection that coincided with the iteration cap was the
loop's own `done[]` flag. Polish rejection is rare and only occurs after the
loop has already declared every root done.

### Slot status totals (650,240 = 65,024 × 10)

| status | slots |
| --- | --- |
| `RealAxisRejected` | 276,906 |
| `Accepted` | 209,600 |
| `Unused` | 100,761 |
| `Complex` | 21,468 |
| `RootNotDone` | 20,786 |
| `InvalidEssential` | 12,458 |
| `NullVectorFailure` | 7,733 |
| `DuplicateRoot` | 398 |
| `PolishRejected` | 130 |
| `DuplicateModel` | 0 |

### Lost solutions at the root stage (1,938 trials), now exact

Previously "NotConverged 13,823" of 19,380 slots; now:

| status | slots in the 1,938 trials |
| --- | --- |
| `RealAxisRejected` | 10,996 |
| `RootNotDone` | 2,757 |
| `InvalidEssential` | 2,620 |
| `NullVectorFailure` | 2,093 |
| `Complex` | 642 |
| `Unused` | 199 |
| `PolishRejected` | 70 |
| `DuplicateRoot` | 3 |

So of the unconverged slots inside the quality-highest losses, 79.6% are still
the real-axis gate acting as the complex filter, 20.0% are iteration-cap
`RootNotDone`, and 0.5% are polish. The ~4 real roots the CPU finds in these
trials continue to fail at recovery (`InvalidEssential` + `NullVectorFailure`
≈ 2.4 / trial) or at the real-axis residual bound — that residual-bound
component for *real* roots is still entangled with complex roots inside
`RealAxisRejected` and is a Q2/recovery item, not this round.

### Regression fixtures

`experiments/q1-fixtures-20260914.json`: 23 trials.

- 18 nullspace `RankDeficient` samples, three each from the σ₅/σ₁ buckets
  `[1e-3,1e-2)`, `[1e-4,1e-3)`, `[1e-5,1e-4)`, `(1e-7,1e-5)`, `≤1e-7`, `≤0`.
- 5 root-stage losses, one each labelled `real_axis_rejected`, `root_not_done`,
  `polish_rejected`, `invalid_essential`, `null_vector_failure`.

Each entry carries the five ray pairs and sampler indices so a later Q2 round
can reload them without the COLMAP database.

## Correctness gates

| gate | result |
| --- | --- |
| model signature vs pre-fix capture | identical |
| diagnostic signature ≠ v1 | yes (v2 recorded) |
| status-code counts vs zero-root inference | three of three |
| header `unconverged` == sum of three splits | every trial |
| `cargo test -p rustsfm --lib five_point` | 8/8 |
| `cargo fmt --check` | pass |
| `git diff --check` | pass |
| clippy on new example | no warning in the new file |

## Retain / rollback

**Retain.** The change is diagnostic-only, model-identical, and removes the
dependence of Q1 classification on buffer zero-initialization — which was a
P2 prerequisite. Rollback would be reverting the three float writes and the
enum variants; no other behaviour depends on the new codes yet.

## Consequences for the plan

- Q1 measurement items that depended on the zero-root trick are closed.
- The 1,938 root-stage losses now have an exact unconverged split; recovery
  (`InvalidEssential` / `NullVectorFailure`) remains the next diagnostic target
  under Q2 item 6.
- P2 no longer needs to preserve "unwritten root stays zero" for classification.
- Next round per TODO: **Q2 零空间直接方法**.

## Reproduction

```
POSELIB_ROOT=<main-checkout>/third_party/PoseLib \
  cargo build --release --target-dir target -p rustsfm \
  --example five_point_gpu_q1_diagnostics

./target/release/examples/five_point_gpu_q1_diagnostics \
  --database <fixed matching.db> \
  --output experiments/q1-diagnostics-<date>.json \
  --fixtures experiments/q1-fixtures-<date>.json
```

The harness refuses to overwrite, asserts the input digest, the model
signature, the three cross-check counts, and that the diagnostic signature
moved off v1.
