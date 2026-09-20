#!/usr/bin/env python3
"""
Convert a TUM RGB-D dataset into a COLMAP text-format dataset.

This script writes a COLMAP-style directory tree:

  <output>/
    images/
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt
      points3d.txt
    manifest.json

It uses TUM RGB timestamps plus `groundtruth.txt` poses to create `images.txt`.
If depth frames are available, it can also synthesize a sparse point cloud by
back-projecting depth samples into world coordinates.

Example:
  python3 scripts/tum_to_colmap.py \
    --tum artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz \
    --output artifacts/runs/tum_freiburg1_xyz_colmap \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
T = TypeVar("T")


@dataclass(frozen=True)
class AssociationEntry:
    timestamp: float
    relative_path: str


@dataclass(frozen=True)
class PoseEntry:
    timestamp: float
    translation_c2w: tuple[float, float, float]
    quaternion_xyzw_c2w: tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectedFrame:
    dataset_index: int
    timestamp: float
    rgb_relative_path: str
    depth_relative_path: str | None
    translation_c2w: tuple[float, float, float]
    quaternion_xyzw_c2w: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TUM RGB-D dataset into COLMAP text format."
    )
    parser.add_argument(
        "--tum",
        required=True,
        help="Path to the TUM RGB-D dataset root (must contain rgb.txt and groundtruth.txt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the generated COLMAP dataset.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum RGB frames to consider before applying frame stride (0 = all).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth matched RGB frame.",
    )
    parser.add_argument(
        "--pose-tolerance-seconds",
        type=float,
        default=0.1,
        help="Maximum timestamp delta when matching RGB frames to poses.",
    )
    parser.add_argument(
        "--depth-tolerance-seconds",
        type=float,
        default=0.1,
        help="Maximum timestamp delta when matching RGB frames to depth frames.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=5000.0,
        help="Scale factor used to convert 16-bit TUM depth PNG values into meters.",
    )
    parser.add_argument(
        "--point-step",
        type=int,
        default=128,
        help=(
            "Sample one sparse point every N pixels in x/y from each depth frame. "
            "Set to 0 to skip points3D generation."
        ),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy RGB images instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    parser.add_argument("--fx", type=float, help="Override camera fx.")
    parser.add_argument("--fy", type=float, help="Override camera fy.")
    parser.add_argument("--cx", type=float, help="Override camera cx.")
    parser.add_argument("--cy", type=float, help="Override camera cy.")
    return parser.parse_args()


def require_tum_root(path: Path) -> Path:
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    if not path.joinpath("rgb.txt").exists():
        raise ValueError(f"missing rgb.txt under {path}")
    if not path.joinpath("groundtruth.txt").exists():
        raise ValueError(f"missing groundtruth.txt under {path}")
    return path


def parse_association_file(path: Path) -> list[AssociationEntry]:
    entries: list[AssociationEntry] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path} line {line_number}: expected 2 columns, got {len(parts)}"
            )
        entries.append(AssociationEntry(timestamp=float(parts[0]), relative_path=parts[1]))
    return entries


def parse_groundtruth(path: Path) -> list[PoseEntry]:
    poses: list[PoseEntry] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(
                f"{path} line {line_number}: expected 8 columns, got {len(parts)}"
            )
        quaternion = normalize_quaternion(
            (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )
        )
        poses.append(
            PoseEntry(
                timestamp=float(parts[0]),
                translation_c2w=(float(parts[1]), float(parts[2]), float(parts[3])),
                quaternion_xyzw_c2w=quaternion,
            )
        )
    poses.sort(key=lambda pose: pose.timestamp)
    return poses


def normalize_quaternion(
    quaternion_xyzw: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be > 0")
    return (x / norm, y / norm, z / norm, w / norm)


def closest_entry(
    entries: Iterable[T], timestamp: float, tolerance: float, key
) -> T | None:
    best = None
    best_diff = None
    for entry in entries:
        diff = abs(key(entry) - timestamp)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best = entry
            best_diff = diff
    return best


def select_frames(
    tum_root: Path,
    max_frames: int,
    frame_stride: int,
    pose_tolerance_seconds: float,
    depth_tolerance_seconds: float,
) -> list[SelectedFrame]:
    rgb_entries = parse_association_file(tum_root / "rgb.txt")
    depth_entries = (
        parse_association_file(tum_root / "depth.txt")
        if tum_root.joinpath("depth.txt").exists()
        else []
    )
    poses = parse_groundtruth(tum_root / "groundtruth.txt")

    considered = rgb_entries[: max_frames or None]
    stride = max(frame_stride, 1)
    selected: list[SelectedFrame] = []
    for dataset_index, rgb_entry in enumerate(considered[::stride]):
        pose = closest_entry(
            poses,
            rgb_entry.timestamp,
            pose_tolerance_seconds,
            key=lambda item: item.timestamp,
        )
        if pose is None:
            continue
        depth_entry = closest_entry(
            depth_entries,
            rgb_entry.timestamp,
            depth_tolerance_seconds,
            key=lambda item: item.timestamp,
        )
        selected.append(
            SelectedFrame(
                dataset_index=dataset_index,
                timestamp=rgb_entry.timestamp,
                rgb_relative_path=rgb_entry.relative_path,
                depth_relative_path=depth_entry.relative_path if depth_entry else None,
                translation_c2w=pose.translation_c2w,
                quaternion_xyzw_c2w=pose.quaternion_xyzw_c2w,
            )
        )
    if not selected:
        raise ValueError(
            f"no frames from {tum_root} could be matched to poses within tolerance"
        )
    return selected


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError(f"{path} is not a PNG file")
        ihdr_length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or ihdr_length < 8:
            raise ValueError(f"{path} does not contain a valid PNG IHDR chunk")
        width, height = struct.unpack(">II", handle.read(8))
        return width, height


def load_intrinsics(
    tum_root: Path,
    width: int,
    height: int,
    fx_override: float | None,
    fy_override: float | None,
    cx_override: float | None,
    cy_override: float | None,
) -> tuple[float, float, float, float]:
    overrides = [fx_override, fy_override, cx_override, cy_override]
    if any(value is not None for value in overrides):
        if not all(value is not None for value in overrides):
            raise ValueError("fx/fy/cx/cy overrides must be provided together")
        return (
            float(fx_override),
            float(fy_override),
            float(cx_override),
            float(cy_override),
        )

    calibration_path = tum_root / "calibration.txt"
    if calibration_path.exists():
        values = []
        for raw_line in calibration_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
        if len(values) >= 4:
            return values[0], values[1], values[2], values[3]

    sequence = tum_root.name.lower()
    if "freiburg1" in sequence:
        return (517.3, 516.5, 318.6, 255.3)
    if "freiburg2" in sequence:
        return (520.9, 521.0, 325.1, 249.7)
    if "freiburg3" in sequence:
        return (535.4, 539.2, 320.1, 247.6)
    return (525.0, 525.0, width / 2.0 - 0.5, height / 2.0 - 0.5)


def quaternion_xyzw_to_rotation_matrix(
    quaternion_xyzw: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    x, y, z, w = quaternion_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def matrix_vector_mul(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def invert_camera_to_world_pose(
    translation_c2w: tuple[float, float, float],
    quaternion_xyzw_c2w: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    qx, qy, qz, qw = quaternion_xyzw_c2w
    quaternion_xyzw_w2c = normalize_quaternion((-qx, -qy, -qz, qw))
    rotation_w2c = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw_w2c)
    translated = matrix_vector_mul(rotation_w2c, translation_c2w)
    return quaternion_xyzw_w2c, (-translated[0], -translated[1], -translated[2])


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ValueError(
                f"{path} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_camera_file(
    sparse_dir: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    content = (
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
    )
    sparse_dir.joinpath("cameras.txt").write_text(content)


def materialize_images(
    tum_root: Path, output_images_dir: Path, frames: list[SelectedFrame], copy_images: bool
) -> list[str]:
    image_names: list[str] = []
    for frame in frames:
        source = tum_root / frame.rgb_relative_path
        destination_name = f"frame_{frame.dataset_index:04d}{source.suffix.lower() or '.png'}"
        destination = output_images_dir / destination_name
        if copy_images:
            shutil.copy2(source, destination)
        else:
            os.symlink(source.resolve(), destination)
        image_names.append(destination_name)
    return image_names


def write_images_file(
    sparse_dir: Path, frames: list[SelectedFrame], image_names: list[str]
) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
    ]
    for image_id, (frame, image_name) in enumerate(zip(frames, image_names), start=1):
        quaternion_xyzw_w2c, translation_w2c = invert_camera_to_world_pose(
            frame.translation_c2w, frame.quaternion_xyzw_c2w
        )
        qx, qy, qz, qw = quaternion_xyzw_w2c
        tx, ty, tz = translation_w2c
        lines.append(
            f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
            f"{tx:.12f} {ty:.12f} {tz:.12f} 1 {image_name}"
        )
        lines.append("")
    sparse_dir.joinpath("images.txt").write_text("\n".join(lines) + "\n")


def write_points_file(
    tum_root: Path,
    sparse_dir: Path,
    frames: list[SelectedFrame],
    point_step: int,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> int:
    lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR",
    ]
    if point_step <= 0:
        content = "\n".join(lines) + "\n"
        sparse_dir.joinpath("points3D.txt").write_text(content)
        sparse_dir.joinpath("points3d.txt").write_text(content)
        return 0

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "point generation requires Pillow; install it or rerun with --point-step 0"
        ) from exc

    point_id = 1
    for frame in frames:
        if frame.depth_relative_path is None:
            continue
        rgb_path = tum_root / frame.rgb_relative_path
        depth_path = tum_root / frame.depth_relative_path
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        if rgb.size != depth.size:
            raise ValueError(
                f"RGB/depth size mismatch for {rgb_path} and {depth_path}: "
                f"{rgb.size} vs {depth.size}"
            )
        rgb_pixels = rgb.load()
        depth_pixels = depth.load()
        rotation_c2w = quaternion_xyzw_to_rotation_matrix(frame.quaternion_xyzw_c2w)
        tx, ty, tz = frame.translation_c2w
        width, height = rgb.size

        for y in range(0, height, point_step):
            for x in range(0, width, point_step):
                depth_raw = depth_pixels[x, y]
                if isinstance(depth_raw, tuple):
                    depth_raw = depth_raw[0]
                depth_m = float(depth_raw) / depth_scale
                if depth_m <= 0.0 or not math.isfinite(depth_m) or depth_m > 10.0:
                    continue

                camera_x = (x - cx) * depth_m / fx
                camera_y = (y - cy) * depth_m / fy
                camera_point = (camera_x, camera_y, depth_m)
                world_x, world_y, world_z = matrix_vector_mul(
                    rotation_c2w, camera_point
                )
                world_x += tx
                world_y += ty
                world_z += tz

                red, green, blue = rgb_pixels[x, y]
                lines.append(
                    f"{point_id} {world_x:.9f} {world_y:.9f} {world_z:.9f} "
                    f"{red} {green} {blue} 0.0"
                )
                point_id += 1

    content = "\n".join(lines) + "\n"
    sparse_dir.joinpath("points3D.txt").write_text(content)
    sparse_dir.joinpath("points3d.txt").write_text(content)
    return point_id - 1


def write_manifest(
    output_root: Path,
    tum_root: Path,
    frames: list[SelectedFrame],
    image_names: list[str],
    point_count: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "source_tum_root": str(tum_root.resolve()),
        "output_root": str(output_root.resolve()),
        "frame_count": len(frames),
        "point_count": point_count,
        "copy_images": args.copy_images,
        "max_frames": args.max_frames,
        "frame_stride": max(args.frame_stride, 1),
        "pose_tolerance_seconds": args.pose_tolerance_seconds,
        "depth_tolerance_seconds": args.depth_tolerance_seconds,
        "depth_scale": args.depth_scale,
        "point_step": args.point_step,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        },
        "frames": [
            {
                "dataset_index": frame.dataset_index,
                "timestamp": frame.timestamp,
                "rgb_relative_path": frame.rgb_relative_path,
                "depth_relative_path": frame.depth_relative_path,
                "output_image_name": image_name,
            }
            for frame, image_name in zip(frames, image_names)
        ],
    }
    output_root.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    args = parse_args()
    tum_root = require_tum_root(Path(args.tum))
    output_root = Path(args.output)
    prepare_output_directory(output_root, args.overwrite)

    frames = select_frames(
        tum_root=tum_root,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        pose_tolerance_seconds=args.pose_tolerance_seconds,
        depth_tolerance_seconds=args.depth_tolerance_seconds,
    )

    first_rgb_path = tum_root / frames[0].rgb_relative_path
    width, height = read_png_size(first_rgb_path)
    fx, fy, cx, cy = load_intrinsics(
        tum_root,
        width,
        height,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
    )

    images_dir = output_root / "images"
    sparse_dir = output_root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_camera_file(sparse_dir, width, height, fx, fy, cx, cy)
    image_names = materialize_images(tum_root, images_dir, frames, args.copy_images)
    write_images_file(sparse_dir, frames, image_names)
    point_count = write_points_file(
        tum_root=tum_root,
        sparse_dir=sparse_dir,
        frames=frames,
        point_step=args.point_step,
        depth_scale=args.depth_scale,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    write_manifest(
        output_root=output_root,
        tum_root=tum_root,
        frames=frames,
        image_names=image_names,
        point_count=point_count,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        args=args,
    )

    print(f"TUM root: {tum_root}")
    print(f"Output: {output_root}")
    print(f"Frames: {len(frames)}")
    print(f"Points: {point_count}")
    print(
        "Intrinsics: "
        f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, {width}x{height}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
