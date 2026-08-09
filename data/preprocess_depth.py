"""Generate ImageNet relative-depth pseudo-labels with Depth Anything V2."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.path as PathConfig
from data.preprocess_utils import (
    add_image_selection_args,
    output_path,
    select_images,
)


MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_model(repo, checkpoint, encoder, device):
    repo = Path(repo).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Depth Anything V2 repository not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repo))
    module = importlib.import_module("depth_anything_v2.dpt")
    config = MODEL_CONFIGS[encoder]
    model = module.DepthAnythingV2(encoder=encoder, **config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def render_depth(model, image, input_size):
    rgb = np.asarray(image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    depth = np.asarray(model.infer_image(bgr, input_size=input_size), dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError(f"Depth Anything V2 returned an invalid map: {depth.shape}")
    minimum = float(depth.min())
    scale = float(depth.max()) - minimum
    normalized = np.zeros_like(depth) if scale <= 1e-8 else (depth - minimum) / scale
    return Image.fromarray(np.round(normalized * 255).astype(np.uint8))


def main(args):
    input_root, output_root, images = select_images(
        args.input,
        args.output,
        start=args.start,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.input_size < 1:
        raise ValueError("input_size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model = load_model(args.repo, args.checkpoint, args.encoder, device)

    failed = 0
    for path in tqdm(images):
        target = output_path(path, input_root, output_root)
        if target.exists() and not args.overwrite:
            continue
        try:
            with Image.open(path) as image:
                depth = render_depth(model, image, args.input_size)
            target.parent.mkdir(parents=True, exist_ok=True)
            depth.save(target)
        except Exception as error:
            if not args.continue_on_error:
                raise RuntimeError(f"Failed to process {path}") from error
            failed += 1
            print(f"Failed to process {path}: {error}")
    print(f"Processed {len(images) - failed} images; failed {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_selection_args(parser)
    parser.add_argument("--repo", default=PathConfig.depth_anything_repo)
    parser.add_argument("--checkpoint", default=PathConfig.depth_anything_checkpoint)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
