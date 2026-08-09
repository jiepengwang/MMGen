"""Shared file-selection contract for image pseudo-label preprocessors."""

import argparse
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_images(
    input_dir,
    output_dir,
    *,
    start=0,
    limit=None,
    shard_index=0,
    num_shards=1,
):
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input image directory not found: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit("Output directory must not be inside the input image tree")
    images = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    images = select_items(
        images,
        start=start,
        limit=limit,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return input_root, output_root, images


def select_items(items, *, start=0, limit=None, shard_index=0, num_shards=1):
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    selected = list(items)[shard_index::num_shards]
    end = None if limit is None else start + limit
    selected = selected[start:end]
    if not selected:
        raise SystemExit("No input items selected; check the input and shard arguments")
    return selected


def output_path(input_path, input_root, output_root):
    return (output_root / input_path.relative_to(input_root)).with_suffix(".png")


def to_uint8_rgb(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite RGB image, got {array.shape}")
    if array.dtype == np.uint8:
        return array
    if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
        array = array * 255
    return np.round(np.clip(array, 0, 255)).astype(np.uint8)


def add_image_selection_args(parser: argparse.ArgumentParser, paths_required=True):
    parser.add_argument(
        "--input", required=paths_required, help="Input RGB image directory."
    )
    parser.add_argument(
        "--output", required=paths_required, help="Output modality directory."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
