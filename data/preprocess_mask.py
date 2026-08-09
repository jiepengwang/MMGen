"""Generate ImageNet segmentation pseudo-labels with Semantic-SAM level 2."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.path as PathConfig
from data.preprocess_utils import (
    add_image_selection_args,
    output_path,
    select_images,
    select_items,
)


def load_semantic_sam(repo):
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Semantic-SAM repository not found: {repo}")
    sys.path.insert(0, str(repo))
    return importlib.import_module("semantic_sam")


def render_masks(masks, seed):
    if not masks:
        return None
    ordered = sorted(masks, key=lambda item: item["area"], reverse=True)
    height, width = ordered[0]["segmentation"].shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for annotation in ordered:
        output[annotation["segmentation"]] = rng.integers(
            0, 256, size=3, dtype=np.uint8
        )
    background = output.sum(axis=-1) == 0
    output[background] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return output


def image_seed(base_seed, relative_path):
    digest = hashlib.blake2s(
        relative_path.as_posix().encode("utf-8"), digest_size=4
    ).digest()
    return (base_seed + int.from_bytes(digest, "big")) % (2**32)


def build_jobs(args):
    if args.metadata:
        if args.input or args.output:
            raise ValueError("Use either --metadata or --input/--output, not both")
        metadata = Path(args.metadata).expanduser().resolve()
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata}")
        with metadata.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        records = select_items(
            records,
            start=args.start,
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        return [
            (
                Path(record["path_rgb"]).expanduser().resolve(),
                Path(record["path_mask"]).expanduser().resolve(),
                Path(record.get("path_rel_rgb", record["path_rgb"])),
            )
            for record in records
        ]
    if not args.input or not args.output:
        raise ValueError("--input and --output are required without --metadata")
    input_root, output_root, images = select_images(
        args.input,
        args.output,
        start=args.start,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    return [
        (path, output_path(path, input_root, output_root), path.relative_to(input_root))
        for path in images
    ]


def main(args):
    jobs = build_jobs(args)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Semantic-SAM checkpoint not found: {checkpoint}")
    semantic_sam = load_semantic_sam(args.semantic_sam_repo)
    generator = semantic_sam.SemanticSamAutomaticMaskGenerator(
        semantic_sam.build_semantic_sam(
            model_type=args.model_type, ckpt=str(checkpoint)
        ),
        level=[args.level],
    )

    failed = 0
    for path, target, relative in tqdm(jobs):
        if target.exists() and not args.overwrite:
            continue
        try:
            _, image = semantic_sam.prepare_image(image_pth=str(path))
            rendered = render_masks(
                generator.generate(image), image_seed(args.seed, relative)
            )
            if rendered is None:
                raise RuntimeError("Semantic-SAM returned no masks")
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered).save(target)
        except Exception as error:
            if not args.continue_on_error:
                raise RuntimeError(f"Failed to process {path}") from error
            failed += 1
            print(f"Failed to process {path}: {error}")
    print(f"Processed {len(jobs) - failed} images; failed {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_selection_args(parser, paths_required=False)
    parser.add_argument(
        "--metadata",
        help="JSONL with path_rgb/path_mask pairs, used for layouts such as Hypersim.",
    )
    parser.add_argument("--semantic-sam-repo", default=PathConfig.semantic_sam_repo)
    parser.add_argument("--checkpoint", default=PathConfig.semantic_sam_checkpoint)
    parser.add_argument("--model-type", choices=["L", "T"], default="L")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
