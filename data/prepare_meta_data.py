"""Build the multi-modal JSONL index consumed by preprocessing and training."""

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODALITIES = ("depth", "normal", "mask")


def find_rgb_images(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main(args):
    rgb_root = Path(args.rgb_dir).expanduser().resolve()
    roots = {
        name: Path(getattr(args, f"{name}_dir")).expanduser().resolve()
        for name in MODALITIES
        if getattr(args, f"{name}_dir") is not None
    }
    required = ("depth", "normal", "mask")
    missing_roots = [name for name in required if name not in roots]
    if missing_roots:
        raise SystemExit(f"Missing required modality directories: {', '.join(missing_roots)}")

    images = find_rgb_images(rgb_root)
    if not images:
        raise SystemExit(f"No RGB images found under {rgb_root}")
    flat_images = [path for path in images if len(path.relative_to(rgb_root).parts) < 2]
    if flat_images:
        raise SystemExit(
            "RGB images must be grouped under class directories; "
            f"found a flat path such as {flat_images[0]}"
        )
    categories = sorted({path.relative_to(rgb_root).parts[0] for path in images})
    category_ids = {name: index for index, name in enumerate(categories)}
    records = []
    skipped = 0
    for rgb_path in images:
        relative = rgb_path.relative_to(rgb_root)
        relative_png = relative.with_suffix(".png")
        modality_paths = {name: root / relative_png for name, root in roots.items()}
        missing = [str(path) for path in modality_paths.values() if not path.is_file()]
        if missing:
            if args.strict:
                raise FileNotFoundError(f"Missing modalities for {rgb_path}: {missing}")
            skipped += 1
            continue

        category = relative.parts[0]
        record = {
            "cat": category,
            "idx_cat": category_ids[category],
            "path_rgb": str(rgb_path),
            "path_rel_rgb": str(relative_png),
        }
        record.update({f"path_{name}": str(path) for name, path in modality_paths.items()})
        records.append(record)

    if not records:
        raise SystemExit("No complete samples found; check modality roots or use --strict for details")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {output}; skipped {skipped} incomplete samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping incomplete samples.")
    main(parser.parse_args())
