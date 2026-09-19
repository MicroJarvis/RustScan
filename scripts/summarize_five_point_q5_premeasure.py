#!/usr/bin/env python3
"""Q5 premeasure: offline analysis of Q4 verified artifacts (no GPU).

Reads trials.json (stage_errors) + bits.json (slot statuses) and reports:
  - NullVectorFailure vs InvalidEssential slot counts
  - B_gpu_vs_f64_same_basis error as a discriminator for loss / recovery fails
  - residual MissingNoSlot / RecoveryFailed guidance from existing report numbers

Refuses overwrite of --output.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

# Rust enum discriminant order in FivePointSlotStatus (gpu_bits uses `as u32`).
SLOT_STATUS = {
    0: "Unused",
    1: "Accepted",
    2: "Complex",
    3: "RealAxisRejected",
    4: "RootNotDone",
    5: "PolishRejected",
    6: "DuplicateRoot",
    7: "NullVectorFailure",
    8: "InvalidEssential",
    9: "DuplicateModel",
    10: "RealRoot",
}

# attribution::NAMES index
B_SAME_BASIS = 4
POLY_SAME_B = 5
POLY_TOTAL = 7


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[int(round((len(xs) - 1) * p))]


def summarize(xs: list[float]) -> dict:
    finite = [x for x in xs if math.isfinite(x)]
    return {
        "n": len(finite),
        "nonfinite": len(xs) - len(finite),
        "p50": percentile(finite, 0.5),
        "p90": percentile(finite, 0.9),
        "p99": percentile(finite, 0.99),
    }


def parse_bits_slots(words: list[int]) -> list[dict]:
    """Walk variable-length gpu_bits record; return slot dicts."""
    i = 15  # skip header
    slots = []
    while i < len(words):
        if i + 9 > len(words):
            break
        slot = words[i]
        status = words[i + 1]
        root0 = struct.unpack("<f", struct.pack("<I", words[i + 2] & 0xFFFFFFFF))[0]
        root1 = struct.unpack("<f", struct.pack("<I", words[i + 3] & 0xFFFFFFFF))[0]
        has_e = words[i + 8] != 0
        i += 9
        if has_e:
            i += 9
        slots.append(
            {
                "slot": slot,
                "status": SLOT_STATUS.get(status, f"unknown_{status}"),
                "status_code": status,
                "root": [root0, root1],
            }
        )
        if len(slots) >= 10:
            break
    return slots


def header_recovery_rejected(words: list[int]) -> int:
    return words[10]


def as_float(x) -> float:
    if x is None:
        return float("nan")
    return float(x)


def bucket(e: float) -> str:
    if not math.isfinite(e):
        return "nonfinite"
    if e >= 1e-1:
        return ">=1e-1"
    if e >= 1e-2:
        return "[1e-2,1e-1)"
    if e >= 1e-3:
        return "[1e-3,1e-2)"
    if e >= 1e-4:
        return "[1e-4,1e-3)"
    if e >= 1e-5:
        return "[1e-5,1e-4)"
    if e >= 1e-6:
        return "[1e-6,1e-5)"
    return "<1e-6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--bits", type=Path, required=True)
    ap.add_argument("--report", type=Path, help="optional Q4 verified JSON for class totals")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    trials = json.loads(args.trials.read_text())
    bits = json.loads(args.bits.read_text())
    assert len(trials) == len(bits) == 65024

    status_counts: Counter[str] = Counter()
    recovery_sub: Counter[str] = Counter()
    b_all: list[float] = []
    b_loss: list[float] = []
    b_no_loss: list[float] = []
    b_with_recovery_slot: list[float] = []
    b_without_recovery_slot: list[float] = []
    poly_b_all: list[float] = []
    poly_total_all: list[float] = []

    trials_with_null = 0
    trials_with_invalid = 0
    trials_with_either = 0

    for t, words in zip(trials, bits):
        slots = parse_bits_slots(words)
        statuses = [s["status"] for s in slots]
        for s in statuses:
            status_counts[s] += 1
        has_null = "NullVectorFailure" in statuses
        has_inv = "InvalidEssential" in statuses
        if has_null:
            trials_with_null += 1
            recovery_sub["NullVectorFailure_slots"] += statuses.count("NullVectorFailure")
        if has_inv:
            trials_with_invalid += 1
            recovery_sub["InvalidEssential_slots"] += statuses.count("InvalidEssential")
        if has_null or has_inv:
            trials_with_either += 1

        b = as_float(t["stage_errors"][B_SAME_BASIS])
        pb = as_float(t["stage_errors"][POLY_SAME_B])
        pt = as_float(t["stage_errors"][POLY_TOTAL])
        b_all.append(b)
        poly_b_all.append(pb)
        poly_total_all.append(pt)
        if t["loss"]:
            b_loss.append(b)
        else:
            b_no_loss.append(b)
        if has_null or has_inv:
            b_with_recovery_slot.append(b)
        else:
            b_without_recovery_slot.append(b)

    loss_buckets: Counter[str] = Counter()
    noloss_buckets: Counter[str] = Counter()
    for e in b_loss:
        loss_buckets[bucket(e)] += 1
    for e in b_no_loss:
        noloss_buckets[bucket(e)] += 1

    report_classes = None
    if args.report and args.report.exists():
        rep = json.loads(args.report.read_text())
        report_classes = rep.get("root_classes_of_f64_real_roots") or rep.get(
            "attribution", {}
        ).get("root_classes_of_f64_real_roots")

    # Heuristic recommendation
    b_loss_p90 = percentile([x for x in b_loss if math.isfinite(x)], 0.9)
    b_noloss_p90 = percentile([x for x in b_no_loss if math.isfinite(x)], 0.9)
    ratio = (
        b_loss_p90 / b_noloss_p90
        if b_noloss_p90 > 0 and math.isfinite(b_loss_p90) and math.isfinite(b_noloss_p90)
        else float("nan")
    )
    null_slots = status_counts["NullVectorFailure"]
    inv_slots = status_counts["InvalidEssential"]

    if ratio >= 10.0:
        recommendation = "Q5a_LU_B_precision"
        rationale = (
            f"B_same_basis p90 loss/no-loss ratio {ratio:.1f}x; "
            "same pattern as Q3 coefficient discriminator → fix upstream LU/B first"
        )
    elif null_slots + inv_slots > 0 and null_slots >= 2 * inv_slots:
        recommendation = "Q5b_recovery_nullvector"
        rationale = (
            f"B error weakly discriminates (ratio {ratio:.2f}x) but "
            f"NullVectorFailure slots ({null_slots}) dominate InvalidEssential ({inv_slots})"
        )
    elif inv_slots > null_slots:
        recommendation = "Q5b_recovery_invalid_essential"
        rationale = (
            f"B error weakly discriminates (ratio {ratio:.2f}x); "
            f"InvalidEssential ({inv_slots}) > NullVectorFailure ({null_slots})"
        )
    else:
        recommendation = "Q5a_LU_B_precision_or_mixed"
        rationale = (
            f"B ratio {ratio:.2f}x; recovery subtypes close "
            f"(null={null_slots}, invalid={inv_slots}); prefer LU/B unless matched-root "
            "attribution shows otherwise"
        )

    out = {
        "round": "Q5-premeasure",
        "kind": "offline-only",
        "inputs": {
            "trials": str(args.trials),
            "bits": str(args.bits),
            "n_trials": len(trials),
        },
        "slot_status_counts": dict(status_counts),
        "recovery_subtype": {
            "NullVectorFailure_slots": null_slots,
            "InvalidEssential_slots": inv_slots,
            "trials_with_NullVectorFailure": trials_with_null,
            "trials_with_InvalidEssential": trials_with_invalid,
            "trials_with_either": trials_with_either,
        },
        "B_gpu_vs_f64_same_basis": {
            "all": summarize(b_all),
            "loss_trials": summarize(b_loss),
            "no_loss_trials": summarize(b_no_loss),
            "trials_with_recovery_fail_slot": summarize(b_with_recovery_slot),
            "trials_without_recovery_fail_slot": summarize(b_without_recovery_slot),
            "p90_loss_over_noloss": ratio,
            "loss_buckets": dict(sorted(loss_buckets.items())),
            "no_loss_buckets": dict(sorted(noloss_buckets.items())),
        },
        "context_polynomial_errors": {
            "polynomial_gpu_vs_f64_same_B": summarize(poly_b_all),
            "polynomial_total": summarize(poly_total_all),
        },
        "q4_report_root_classes": report_classes,
        "recommendation": recommendation,
        "rationale": rationale,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
