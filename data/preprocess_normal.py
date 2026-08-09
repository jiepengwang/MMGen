"""Generate ImageNet surface-normal pseudo-labels with StableNormal."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocess_utils import (
    add_image_selection_args,
    output_path,
    select_images,
    to_uint8_rgb,
)


def main(args):
    input_root, output_root, images = select_images(
        args.input,
        args.output,
        start=args.start,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.hub_source == "local" and not Path(args.hub_repo).expanduser().is_dir():
        raise SystemExit(f"Local torch.hub repository not found: {args.hub_repo}")
    predictor = torch.hub.load(
        args.hub_repo,
        args.hub_model,
        source=args.hub_source,
        trust_repo=args.trust_repo,
    )

    failed = 0
    for path in tqdm(images):
        target = output_path(path, input_root, output_root)
        if target.exists() and not args.overwrite:
            continue
        try:
            with Image.open(path) as image:
                normal = predictor(image.convert("RGB"))
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(to_uint8_rgb(normal)).save(target)
        except Exception as error:
            if not args.continue_on_error:
                raise RuntimeError(f"Failed to process {path}") from error
            failed += 1
            print(f"Failed to process {path}: {error}")
    print(f"Processed {len(images) - failed} images; failed {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_selection_args(parser)
    parser.add_argument("--hub-repo", default="Stable-X/StableNormal")
    parser.add_argument("--hub-model", default="StableNormal")
    parser.add_argument("--hub-source", choices=["github", "local"], default="github")
    parser.add_argument(
        "--trust-repo", action=argparse.BooleanOptionalAction, default=True
    )
    main(parser.parse_args())
