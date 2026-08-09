"""Build JSONL metadata for inference conditioned on one image modality.

MMGenDataset (num_tasks=1) only needs these fields per line:
  path_<modality> absolute path to the condition image
  idx_cat        ImageNet class id fed to the label embedder (1000 = null/unconditional)
  cat            category name string (only used for logging)
  path_rel_rgb   relative path used to name the output files

Usage:
  python data/prepare_infer_meta.py --input <img_or_dir> --out <meta.jsonl> \
    --modality rgb [--idx-cat 1000]
"""
import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_path):
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported image extension: {source}")
        return source.parent, [source]
    if not source.is_dir():
        raise SystemExit(f"Input does not exist: {source}")
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return source, files


def main(args):
    root, files = collect_images(args.input)
    if not files:
        raise SystemExit(f"No images found under {args.input}")

    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(root)
            record = {
                f"path_{args.modality}": str(path),
                "idx_cat": args.idx_cat,
                "cat": path.parent.name or "na",
                "path_rel_rgb": str(relative),
            }
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(files)} entries to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="A single image file or a directory (scanned recursively).")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--modality", choices=("rgb", "depth", "normal", "mask"), default="rgb"
    )
    parser.add_argument("--idx-cat", type=int, default=1000,
                        help="ImageNet class id; 1000 is the null class.")
    args = parser.parse_args()
    main(args)
