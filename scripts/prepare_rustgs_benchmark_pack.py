#!/usr/bin/env python3
"""Prepare a local RustGS benchmark pack manifest.

The script intentionally avoids downloading large datasets. It records the
recommended benchmark scenes, optionally converts a local TUM RGB-D sequence to
COLMAP text format, and writes command templates for RustGS training/eval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkScene:
    scene_id: str
    family: str
    priority: int
    input_kind: str
    status: str
    source_url: str
    prepared_input: str
    why: str
    setup: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a RustGS benchmark pack manifest and optional local TUM conversion."
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/runs/rustgs_benchmark_pack",
        help="Directory for generated manifests and local prepared datasets.",
    )
    parser.add_argument(
        "--tum-root",
        default="artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz",
        help="Local TUM RGB-D sequence root to convert when present.",
    )
    parser.add_argument(
        "--skip-tum-convert",
        action="store_true",
        help="Only write the benchmark manifest; do not run scripts/tum_to_colmap.py.",
    )
    parser.add_argument(
        "--max-tum-frames",
        type=int,
        default=0,
        help="Maximum TUM RGB frames to consider before stride (0 = all).",
    )
    parser.add_argument(
        "--tum-frame-stride",
        type=int,
        default=10,
        help="Keep every Nth TUM RGB frame for the generated quick benchmark.",
    )
    parser.add_argument(
        "--tum-point-step",
        type=int,
        default=96,
        help="Depth sampling step for TUM sparse point generation.",
    )
    parser.add_argument(
        "--copy-tum-images",
        action="store_true",
        help="Copy TUM RGB images instead of symlinking them.",
    )
    return parser.parse_args()


def benchmark_scenes(output_root: Path) -> list[BenchmarkScene]:
    tum_prepared = output_root / "tum_freiburg1_xyz_colmap"
    return [
        BenchmarkScene(
            scene_id="tum_freiburg1_xyz_colmap",
            family="TUM RGB-D",
            priority=1,
            input_kind="COLMAP text",
            status="local-convertible",
            source_url="https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
            prepared_input=str(tum_prepared),
            why="Continuity with the existing RustGS PSNR loop and fast RGB-D reprojection-derived sparse init.",
            setup="Generated from the local TUM RGB-D sequence by scripts/tum_to_colmap.py.",
        ),
        BenchmarkScene(
            scene_id="mipnerf360_room",
            family="Mip-NeRF 360",
            priority=2,
            input_kind="COLMAP",
            status="download-manual",
            source_url="https://jonbarron.info/mipnerf360/",
            prepared_input="<data_root>/mipnerf360/room",
            why="Indoor bounded scene for checking PSNR, SH color, and short-run convergence.",
            setup="Download the official Mip-NeRF 360 data and point RustGS at the scene root containing images and sparse/0.",
        ),
        BenchmarkScene(
            scene_id="mipnerf360_garden",
            family="Mip-NeRF 360",
            priority=3,
            input_kind="COLMAP",
            status="download-manual",
            source_url="https://jonbarron.info/mipnerf360/",
            prepared_input="<data_root>/mipnerf360/garden",
            why="Outdoor unbounded scene for testing scene scale, pruning, and densification stability.",
            setup="Download the official Mip-NeRF 360 data and use the COLMAP-format scene directory.",
        ),
        BenchmarkScene(
            scene_id="tandt_truck",
            family="Tanks and Temples",
            priority=4,
            input_kind="COLMAP",
            status="download-manual",
            source_url="https://www.tanksandtemples.org/download/",
            prepared_input="<data_root>/tanks_and_temples/truck",
            why="Large structured object with strong geometry cues; useful for topology growth and scale schedules.",
            setup="Use the official/Graphdeco COLMAP-prepared Truck scene when available.",
        ),
        BenchmarkScene(
            scene_id="deep_blending_playroom",
            family="Deep Blending",
            priority=5,
            input_kind="COLMAP",
            status="download-manual",
            source_url="https://github.com/graphdeco-inria/gaussian-splatting",
            prepared_input="<data_root>/deep_blending/playroom",
            why="Complex indoor light transport and occlusion; useful for SH/view-dependent color regressions.",
            setup="Use the Graphdeco 3DGS data package for the Playroom scene.",
        ),
        BenchmarkScene(
            scene_id="nerfstudio_blender_lego",
            family="NeRF Synthetic / Nerfstudio",
            priority=6,
            input_kind="Nerfstudio transforms + sparse_pc.ply",
            status="format-supported",
            source_url="https://docs.nerf.studio/quickstart/existing_dataset.html",
            prepared_input="<data_root>/nerfstudio/lego",
            why="Perfect-pose synthetic sanity check for projection, coordinate transforms, and image target loading.",
            setup="Place transforms.json plus sparse_pc.ply/point_cloud.ply under the scene root.",
        ),
    ]


def train_command(scene: BenchmarkScene, output_root: Path, iterations: int) -> str:
    scene_out = output_root / "runs" / f"{scene.scene_id}_{iterations}.ply"
    return (
        "cargo run -p rustscan-gs --release --bin rustgs -- train "
        f"--input {scene.prepared_input} "
        f"--output {scene_out} "
        f"--iterations {iterations} "
        "--render-scale 1.0 "
        "--eval-after-train "
        "--eval-render-scale 0.5 "
        "--eval-frame-stride 10 "
        "--eval-json"
    )


def convert_local_tum(args: argparse.Namespace, output_root: Path) -> Path | None:
    if args.skip_tum_convert:
        return None

    tum_root = (WORKSPACE_ROOT / args.tum_root).resolve()
    if not tum_root.exists():
        print(f"Skipping TUM conversion; {tum_root} does not exist.", file=sys.stderr)
        return None

    prepared = (output_root / "tum_freiburg1_xyz_colmap").resolve()
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "tum_to_colmap.py"),
        "--tum",
        str(tum_root),
        "--output",
        str(prepared),
        "--overwrite",
        "--frame-stride",
        str(args.tum_frame_stride),
        "--point-step",
        str(args.tum_point_step),
    ]
    if args.max_tum_frames > 0:
        command.extend(["--max-frames", str(args.max_tum_frames)])
    if args.copy_tum_images:
        command.append("--copy-images")

    subprocess.run(command, check=True, cwd=WORKSPACE_ROOT)
    return prepared


def write_manifest(output_root: Path, scenes: list[BenchmarkScene]) -> Path:
    manifest_path = output_root / "manifest.json"
    manifest = {
        "workspace_root": str(WORKSPACE_ROOT),
        "output_root": str(output_root.resolve()),
        "scenes": [asdict(scene) for scene in scenes],
        "commands": {
            scene.scene_id: {
                "quick_500": train_command(scene, output_root, 500),
                "long_30000": train_command(scene, output_root, 30000),
            }
            for scene in scenes
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def write_markdown(output_root: Path, scenes: list[BenchmarkScene]) -> Path:
    markdown_path = output_root / "README.md"
    lines = [
        "# RustGS Benchmark Pack",
        "",
        "Generated helper manifest for RustGS dataset expansion. Large public datasets are not downloaded by this script.",
        "",
        "## Scenes",
        "",
        "| Priority | Scene | Family | Input | Status |",
        "|---:|---|---|---|---|",
    ]
    for scene in scenes:
        lines.append(
            f"| {scene.priority} | `{scene.scene_id}` | {scene.family} | {scene.input_kind} | {scene.status} |"
        )
    lines.extend(
        [
            "",
            "## Quick Command",
            "",
            "Start with the local TUM conversion if it exists:",
            "",
            "```sh",
            train_command(scenes[0], output_root, 500),
            "```",
            "",
            "See `manifest.json` for all command templates.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines))
    return markdown_path


def main() -> int:
    args = parse_args()
    output_root = (WORKSPACE_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    converted_tum = convert_local_tum(args, output_root)
    scenes = benchmark_scenes(output_root)
    manifest_path = write_manifest(output_root, scenes)
    markdown_path = write_markdown(output_root, scenes)

    print(f"Benchmark manifest: {manifest_path}")
    print(f"Benchmark README: {markdown_path}")
    if converted_tum:
        print(f"Prepared TUM COLMAP dataset: {converted_tum}")
    print("Next: fill the manual-download scene paths in manifest.json as datasets land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
