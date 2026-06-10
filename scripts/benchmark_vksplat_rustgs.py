#!/usr/bin/env python3
"""Run comparable RustGS and VkSplat benchmarks on one COLMAP dataset.

The script is intentionally conservative: it prepares a local TUM-derived
COLMAP dataset, runs RustGS through the CLI, runs VkSplat through its Python
binding when the extension is available, and writes one summary JSON that can be
used as the optimization scoreboard.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "output" / "vksplat_rustgs_benchmark"
DEFAULT_TUM_ROOT = WORKSPACE_ROOT / "test_data" / "tum" / "rgbd_dataset_freiburg1_xyz"
DEFAULT_VKSPLAT_ROOT = Path("/tmp/vksplat-analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RustGS and VkSplat on the same prepared COLMAP dataset."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, help="Use an existing COLMAP dataset.")
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--vksplat-root", type=Path, default=DEFAULT_VKSPLAT_ROOT)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--tum-frame-stride", type=int, default=8)
    parser.add_argument("--tum-point-step", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--rustgs-render-scale", type=float, default=0.5)
    parser.add_argument("--eval-render-scale", type=float, default=0.5)
    parser.add_argument("--rustgs-init-point-scale-factor", type=float, default=0.5)
    parser.add_argument("--rustgs-init-point-opacity", type=float, default=0.5)
    parser.add_argument("--rustgs-init-vksplat-scale-estimator", action="store_true")
    parser.add_argument("--rustgs-init-random-rotations", action="store_true")
    parser.add_argument("--rustgs-init-rotation-seed", type=int, default=42)
    parser.add_argument("--rustgs-loss-l1-weight", type=float, default=0.8)
    parser.add_argument("--rustgs-loss-ssim-weight", type=float, default=0.2)
    parser.add_argument(
        "--rustgs-bin",
        type=Path,
        default=WORKSPACE_ROOT / "target" / "release" / "rustgs",
        help="RustGS binary to run. If missing, the script falls back to cargo run --release.",
    )
    parser.add_argument("--device-id", type=int, default=-1, help="VkSplat device id; -1 = auto.")
    parser.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    parser.add_argument("--mcmc-cap-max", type=int, default=250_000)
    parser.add_argument("--prepare-overwrite", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-rustgs", action="store_true")
    parser.add_argument("--skip-vksplat", action="store_true")
    parser.add_argument(
        "--build-vksplat",
        action="store_true",
        help="Try `pip install -e . --no-build-isolation --no-deps` before running VkSplat.",
    )
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(command, cwd=cwd, text=True, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "elapsed_wall_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def prepare_dataset(args: argparse.Namespace, output_root: Path) -> Path:
    if args.dataset:
        return args.dataset.resolve()
    dataset = output_root / "datasets" / "tum_freiburg1_xyz_colmap"
    if args.skip_prepare and dataset.exists():
        return dataset.resolve()
    if args.skip_prepare:
        raise FileNotFoundError(f"{dataset} does not exist and --skip-prepare was set")

    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "tum_to_colmap.py"),
        "--tum",
        str(args.tum_root),
        "--output",
        str(dataset),
        "--frame-stride",
        str(args.tum_frame_stride),
        "--point-step",
        str(args.tum_point_step),
    ]
    if args.max_frames > 0:
        command += ["--max-frames", str(args.max_frames)]
    if args.prepare_overwrite or not dataset.exists():
        command.append("--overwrite")
    if args.copy_images:
        command.append("--copy-images")

    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    result = run_command(
        command,
        cwd=WORKSPACE_ROOT,
        stdout_path=logs / "prepare_dataset.stdout.log",
        stderr_path=logs / "prepare_dataset.stderr.log",
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"dataset preparation failed; see {result['stderr']}")
    return dataset.resolve()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def frame_count_for_split(dataset: Path) -> int:
    images_txt = dataset / "sparse" / "0" / "images.txt"
    if images_txt.exists():
        count = 0
        for line in images_txt.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 10 and parts[0].isdigit():
                count += 1
        if count > 0:
            return count

    image_files = sorted(
        path
        for path in (dataset / "images").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return len(image_files)


def run_rustgs(args: argparse.Namespace, dataset: Path, run_root: Path) -> dict[str, Any]:
    out_dir = run_root / "rustgs"
    logs = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    output_ply = out_dir / "rustgs.ply"
    val_frame_ids = ",".join(
        str(idx) for idx in range(0, frame_count_for_split(dataset), args.eval_interval)
    )
    if args.rustgs_bin.exists():
        command = [str(args.rustgs_bin), "train"]
    else:
        command = [
            "cargo",
            "run",
            "-p",
            "rustgs",
            "--release",
            "--bin",
            "rustgs",
            "--",
            "train",
        ]
    command += [
        "--input",
        str(dataset),
        "--output",
        str(output_ply),
        "--iterations",
        str(args.iterations),
        "--max-frames",
        "0",
        "--frame-stride",
        "1",
        "--exclude-frame-ranges",
        val_frame_ids,
        "--render-scale",
        str(args.rustgs_render_scale),
        "--init-point-scale-factor",
        str(args.rustgs_init_point_scale_factor),
        "--init-point-opacity",
        str(args.rustgs_init_point_opacity),
        "--init-rotation-seed",
        str(args.rustgs_init_rotation_seed),
        "--loss-l1-weight",
        str(args.rustgs_loss_l1_weight),
        "--loss-ssim-weight",
        str(args.rustgs_loss_ssim_weight),
        "--eval-after-train",
        "--eval-render-scale",
        str(args.eval_render_scale),
        "--eval-frame-stride",
        str(args.eval_interval),
        "--eval-json",
        "--log-level",
        "info",
    ]
    if args.rustgs_init_vksplat_scale_estimator:
        command.append("--init-vksplat-scale-estimator")
    if args.rustgs_init_random_rotations:
        command.append("--init-random-rotations")
    result = run_command(
        command,
        cwd=WORKSPACE_ROOT,
        stdout_path=logs / "rustgs.stdout.log",
        stderr_path=logs / "rustgs.stderr.log",
    )
    parity_path = output_ply.with_name(f"{output_ply.stem}.parity.json")
    parity = load_json(parity_path)
    result.update(
        {
            "output_ply": str(output_ply),
            "parity_report": str(parity_path) if parity_path.exists() else None,
            "status": "ok" if result["returncode"] == 0 else "failed",
        }
    )
    if parity:
        result["training_seconds"] = (
            parity.get("timing", {}).get("training_ms", 0) / 1000.0
        )
        result["psnr_mean_db"] = parity.get("metrics", {}).get("final_psnr")
        result["gaussian_count"] = parity.get("topology", {}).get("final_gaussians")
    return result


def vksplat_config(args: argparse.Namespace, dataset: Path, out_dir: Path) -> dict[str, Any]:
    strategy = args.strategy
    config = {
        "output_dir": str(out_dir),
        "output_ply": str(out_dir / "vksplat.ply"),
        "dataset_dir": str(dataset) + "/",
        "image_dir": str(dataset / "images") + "/",
        "mask_dir": "",
        "sparse_dir": str(dataset / "sparse" / "0") + "/",
        "eval_interval": args.eval_interval,
        "image_cache_device": "cpu",
        "global_scale": 1.0,
        "init_scale": 1.0,
        "init_opacity": 0.1,
        "strategy": strategy,
        "max_steps": args.iterations,
        "ssim_lambda": 0.2,
        "means_lr": 1.6e-4,
        "means_lr_final": 1.6e-6,
        "features_dc_lr": 0.0025,
        "features_rest_lr": 0.0025 / 20.0,
        "opacities_lr": 0.05,
        "scales_lr": 0.005,
        "quats_lr": 0.001,
        "scale_reg": 0.0,
        "opacity_reg": 0.0,
        "refine_start_iter": 500,
        "refine_stop_iter": max(args.iterations, 500),
        "refine_every": 100,
        "prune_opa": 0.005,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "grow_scale2d": 0.05,
        "prune_scale3d": 0.1,
        "prune_scale2d": 0.15,
        "refine_scale2d_stop_iter": 0,
        "reset_every": 3000,
        "stop_reset_at": -1,
        "pause_refine_after_reset": 0,
        "noise_lr": 5e5,
        "min_opacity": 0.005,
        "grow_factor": 1.05,
        "cap_max": args.mcmc_cap_max,
    }
    if strategy == "mcmc":
        config.update(
            {
                "init_scale": 0.1,
                "init_opacity": 0.5,
                "scale_reg": 0.01,
                "opacity_reg": 0.01,
                "refine_stop_iter": max(args.iterations, 500),
            }
        )
    return config


VKSPLAT_RUNNER = r"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

config_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
shader_dir = Path(sys.argv[3])
device_id = int(sys.argv[4])
iterations = int(sys.argv[5])
eval_render_scale = float(sys.argv[6])

config = json.loads(config_path.read_text())

def resize_rgb_box(src, dst_width, dst_height):
    src_height, src_width, _ = src.shape
    if src_width == dst_width and src_height == dst_height:
        return src.astype(np.float32, copy=True)
    dst = np.zeros((dst_height, dst_width, 3), dtype=np.float32)
    for dy in range(dst_height):
        sy0 = dy * src_height // dst_height
        sy1 = min(src_height, max(sy0 + 1, (dy + 1) * src_height // dst_height))
        for dx in range(dst_width):
            sx0 = dx * src_width // dst_width
            sx1 = min(src_width, max(sx0 + 1, (dx + 1) * src_width // dst_width))
            dst[dy, dx] = src[sy0:sy1, sx0:sx1].mean(axis=(0, 1))
    return dst

try:
    import vksplat
    if not hasattr(vksplat, "VkSplat"):
        raise RuntimeError("imported vksplat module does not expose VkSplat; build the C++ extension")

    module = vksplat.VkSplat()
    module.initialize(str(shader_dir) + "/", device_id)
    train_meta = module.set_train_config(config)
    random.seed(0)
    shuffle_idx = list(range(module.num_train))
    start = time.perf_counter()
    for step in range(iterations):
        if step > 0 and step % len(shuffle_idx) == 0:
            random.shuffle(shuffle_idx)
        module.train_step(shuffle_idx[step % len(shuffle_idx)], step)
    train_seconds = time.perf_counter() - start

    val_metrics = []
    val_images = []
    for idx in range(module.num_val):
        module.render_val(idx)
        rendered = np.asarray(module.pixel_state, dtype=np.float32)[..., :3]
        info = dict(module.get_val_image_path(idx))
        target = np.asarray(Image.open(info["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        rendered = np.clip(rendered, 0.0, 1.0)
        eval_scale = min(1.0, max(0.0625, eval_render_scale))
        dst_width = max(1, round(target.shape[1] * eval_scale))
        dst_height = max(1, round(target.shape[0] * eval_scale))
        rendered = resize_rgb_box(rendered, dst_width, dst_height)
        target = resize_rgb_box(target, dst_width, dst_height)
        mse = float(np.mean((rendered - target) ** 2))
        psnr = float("inf") if mse <= 0.0 else -10.0 * math.log10(mse)
        val_metrics.append(psnr)
        val_images.append({
            "image_path": info["image_path"],
            "psnr_db": psnr,
            "eval_width": dst_width,
            "eval_height": dst_height,
        })

    num_splats = int(len(module.opacities))
    module.write_ply(config["output_ply"])
    summary = {
        "status": "ok",
        "training_seconds": train_seconds,
        "psnr_mean_db": float(np.mean(val_metrics)) if val_metrics else None,
        "psnr_min_db": float(np.min(val_metrics)) if val_metrics else None,
        "gaussian_count": num_splats,
        "vram": int(module.get_vram_usage()),
        "peak_vram": int(module.get_peak_vram_usage()),
        "timing_breakdown": json_safe(module.get_timing_breakdown()),
        "vram_breakdown": json_safe(module.get_vram_breakdown()),
        "train_meta": json_safe(dict(train_meta)),
        "val_images": val_images,
        "output_ply": config["output_ply"],
    }
    module.cleanup()
except Exception as exc:
    summary = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

summary_path.write_text(json.dumps(summary, indent=2) + "\n")
if summary["status"] != "ok":
    raise SystemExit(1)
"""


def build_vksplat(vksplat_package: Path, logs: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--no-build-isolation",
        "--no-deps",
    ]
    return run_command(
        command,
        cwd=vksplat_package,
        stdout_path=logs / "vksplat_build.stdout.log",
        stderr_path=logs / "vksplat_build.stderr.log",
    )


def run_vksplat(args: argparse.Namespace, dataset: Path, run_root: Path) -> dict[str, Any]:
    out_dir = run_root / "vksplat"
    logs = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    vksplat_package = args.vksplat_root / "vksplat"
    shader_dir = vksplat_package / "shader"
    result: dict[str, Any] = {
        "status": "skipped",
        "vksplat_package": str(vksplat_package),
    }
    if args.build_vksplat:
        result["build"] = build_vksplat(vksplat_package, logs)
        if result["build"]["returncode"] != 0:
            result["status"] = "failed"
            result["error"] = f"VkSplat build failed; see {result['build']['stderr']}"
            return result

    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.json"
    config_path.write_text(json.dumps(vksplat_config(args, dataset, out_dir), indent=2) + "\n")
    command = [
        sys.executable,
        "-c",
        VKSPLAT_RUNNER,
        str(config_path),
        str(summary_path),
        str(shader_dir),
        str(args.device_id),
        str(args.iterations),
        str(args.eval_render_scale),
    ]
    run = run_command(
        command,
        cwd=vksplat_package,
        stdout_path=logs / "vksplat.stdout.log",
        stderr_path=logs / "vksplat.stderr.log",
    )
    summary = load_json(summary_path) or {}
    result.update(run)
    result.update(summary)
    if run["returncode"] != 0 and result.get("status") != "failed":
        result["status"] = "failed"
    return result


def decide_gate(rustgs: dict[str, Any] | None, vksplat: dict[str, Any] | None) -> dict[str, Any]:
    if not rustgs or rustgs.get("status") != "ok":
        return {"status": "missing_rustgs"}
    if not vksplat or vksplat.get("status") != "ok":
        return {"status": "missing_vksplat"}
    rt = rustgs.get("training_seconds")
    vt = vksplat.get("training_seconds")
    rp = rustgs.get("psnr_mean_db")
    vp = vksplat.get("psnr_mean_db")
    if rt is None or vt is None or rp is None or vp is None:
        return {"status": "missing_metric"}
    return {
        "status": "pass" if rt <= vt and rp >= vp else "fail",
        "rustgs_not_slower": rt <= vt,
        "rustgs_psnr_not_lower": rp >= vp,
        "training_speedup_vs_vksplat": vt / rt if rt > 0 else None,
        "psnr_delta_db": rp - vp,
    }


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(args, output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / "runs" / f"tum_{args.iterations}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORKSPACE_ROOT),
        "dataset": str(dataset),
        "config": {
            "iterations": args.iterations,
            "eval_interval": args.eval_interval,
            "rustgs_render_scale": args.rustgs_render_scale,
            "eval_render_scale": args.eval_render_scale,
            "rustgs_init_point_scale_factor": args.rustgs_init_point_scale_factor,
            "rustgs_init_point_opacity": args.rustgs_init_point_opacity,
            "rustgs_init_vksplat_scale_estimator": args.rustgs_init_vksplat_scale_estimator,
            "rustgs_init_random_rotations": args.rustgs_init_random_rotations,
            "rustgs_init_rotation_seed": args.rustgs_init_rotation_seed,
            "rustgs_loss_l1_weight": args.rustgs_loss_l1_weight,
            "rustgs_loss_ssim_weight": args.rustgs_loss_ssim_weight,
            "strategy": args.strategy,
        },
        "rustgs": None,
        "vksplat": None,
        "gate": None,
    }

    if not args.skip_rustgs:
        summary["rustgs"] = run_rustgs(args, dataset, run_root)
    if not args.skip_vksplat:
        summary["vksplat"] = run_vksplat(args, dataset, run_root)
    summary["gate"] = decide_gate(summary["rustgs"], summary["vksplat"])

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)
    print(json.dumps(summary["gate"], indent=2))
    return 0 if summary["gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
