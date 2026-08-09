"""Filesystem configuration shared by training and preprocessing.

All values can be overridden with environment variables. Defaults live under
``data/local`` so imports never depend on a hostname or internal mount.
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(name, default):
    value = os.environ.get(name)
    return str(Path(value).expanduser() if value else Path(default))


data_root = _path("REPA_DATA_ROOT", REPO_ROOT / "data/local")

image_dir = _path("REPA_RGB_DIR", f"{data_root}/rgb/train")
depth_dir = _path("REPA_DEPTH_DIR", f"{data_root}/depth/train")
normal_dir = _path("REPA_NORMAL_DIR", f"{data_root}/normal/train")
mask_dir = _path("REPA_MASK_DIR", f"{data_root}/mask/train")

path_meta_data = _path("REPA_METADATA", f"{data_root}/metadata/train.jsonl")
path_h5_rgb = _path("REPA_VAE_RGB_H5", f"{data_root}/h5/vae/rgb.h5")
path_h5_depth = _path("REPA_VAE_DEPTH_H5", f"{data_root}/h5/vae/depth.h5")
path_h5_normal = _path("REPA_VAE_NORMAL_H5", f"{data_root}/h5/vae/normal.h5")
path_h5_mask = _path("REPA_VAE_MASK_H5", f"{data_root}/h5/vae/mask.h5")
path_h5_dino_rgb = _path("REPA_DINO_RGB_H5", f"{data_root}/h5/dino/rgb.h5")

hypersim_root = _path("REPA_HYPERSIM_ROOT", f"{data_root}/hypersim/processed")
hypersim_splits = _path("REPA_HYPERSIM_SPLITS", f"{data_root}/hypersim/splits")
path_meta_hypersim = _path(
    "REPA_HYPERSIM_SEG_METADATA", f"{hypersim_splits}/train_rgb_mask.jsonl"
)

semantic_sam_repo = _path(
    "SEMANTIC_SAM_REPO", REPO_ROOT / "third_party/Semantic-SAM"
)
semantic_sam_checkpoint = _path(
    "SEMANTIC_SAM_CHECKPOINT",
    f"{semantic_sam_repo}/checkpoints/swinl_only_sam_many2many.pth",
)
depth_anything_repo = _path(
    "DEPTH_ANYTHING_V2_REPO", REPO_ROOT / "third_party/Depth-Anything-V2"
)
depth_anything_checkpoint = _path(
    "DEPTH_ANYTHING_V2_CHECKPOINT",
    f"{depth_anything_repo}/checkpoints/depth_anything_v2_vitl.pth",
)

TASK_NAMES = ("rgb", "depth", "normal", "mask")


def map_idxtask_to_name(idx_task):
    try:
        return TASK_NAMES[idx_task]
    except (IndexError, TypeError):
        raise ValueError(f"task index {idx_task!r} is not supported") from None
