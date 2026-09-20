# Geometry dispatch forwarding feasibility probe

## Scope

Experimental example only; no production matching, Taskflow, RANSAC, GPU session
API or default behavior changes. This is a single-caller forwarding gate, not a
pair pipeline or CPU/GPU overlap benchmark.

The dedicated thread creates and owns the GPU scorer. A capacity-one blocking
request queue and a capacity-one per-request reply channel carry requests and
results. The caller has one outstanding request. Shutdown disconnects the queue
and joins the thread. Handler errors propagate; abandoned replies do not kill the
worker. GPU driver stalls are not bounded by the protocol; use an external timeout.

Eight GPU-free tests cover backpressure, disconnect, shutdown/thread ownership,
handler failure, startup failure, abandoned replies, panic and argument validation.
They do not prove Taskflow context propagation, pair cancellation, ordered database
commit, or a two-pair memory bound.

## Reproduction

From `rustsfm`, with native thread limits set to one:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 cargo test --example geometry_dispatch_probe --offline -j 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 cargo run --release --example geometry_dispatch_probe --offline -j 1 -- --iterations 100
```

Four paired rounds alternate direct/forwarded order equally. Each path performs
400 calls per round: 200 score and 200 scalar mask calls, over homography and
Sampson inputs. Scoring uses 512 models and 512 observations; scalar mask calls
use one selected model. Full support counts/residual bits and mask entries are
compared outside timing. Both paths run on the same GPU owner thread/scorer.
Direct-round control messages, startup, warmup and comparisons are excluded.

## Observed results (2026-09-13)

Device: Apple M5 Max, Metal. Final balanced run:

| Round | Order | Direct ms | Forwarded ms | Delta ms |
|---|---|---:|---:|---:|
| 1 | direct, forwarded | 553.421 | 555.694 | 2.273 |
| 2 | forwarded, direct | 548.350 | 554.721 | 6.371 |
| 3 | direct, forwarded | 545.043 | 554.843 | 9.800 |
| 4 | forwarded, direct | 538.749 | 540.155 | 1.406 |

All four complete output comparisons passed. Across 1,600 calls per path:
2,185.563 ms direct versus 2,205.412 ms forwarded, approximately 0.91% overhead,
12.406 microseconds per call. Per-round deltas vary; these are exploratory figures,
not confidence bounds. The preliminary three-round run had 34.302 microseconds
per call aggregate overhead, dominated by its first round; it is not discarded as
proof of stable low overhead. Four rounds correct its unbalanced execution order.

## Decision and limits

No large forwarding penalty appeared in the balanced run. Proceed to a
session-reusing representative probe before building a two-pair scheduler.

Public scorer APIs recreate observation/scratch sessions every call, unlike the
production RANSAC path. Inputs are preloaded on the worker, so this does not
measure transport of dynamically generated models. Masks are scalar, not R11's
bounded batched masks. Timing includes scheduling, reply allocation and output
transport; it is not pure channel overhead. Repeated inputs do not test stale
results across changing models. No descriptor matcher, Taskflow grant inheritance,
GPU queue contention, second pair, 960-frame dataset, or real overlap was exercised.
Do not multiply these microseconds by 960-frame request counts to claim speedup.

Next gate: preserve production session reuse, transfer changing model batches,
exercise scalar/batched masks, and compare full outputs. Then test two-pair
ordinal result/error ordering, bounded retained memory and scoped admission
lifetime before a first48 smoke and any 960-frame matching measurement.
