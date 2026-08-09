"""Build Hypersim JSONL indices for mask generation and fine-tuning."""

import argparse
import json
from pathlib import Path


def iter_frames(processed_root, split_file):
    scenes = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    for scene in scenes:
        for camera in sorted((processed_root / scene).glob("cam_*")):
            for rgb_path in sorted((camera / "rgb").glob("*.png")):
                yield scene, camera, rgb_path


def main(args):
    processed_root = Path(args.processed_root).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not processed_root.is_dir():
        raise SystemExit(f"Processed Hypersim root not found: {processed_root}")
    if not split_file.is_file():
        raise SystemExit(f"Hypersim split file not found: {split_file}")
    records = []
    skipped = 0
    for scene, camera, rgb_path in iter_frames(processed_root, split_file):
        mask_path = camera / "mask" / rgb_path.name
        if args.mode == "seg":
            records.append({"path_rgb": str(rgb_path), "path_mask": str(mask_path)})
            continue

        visual_name = f"{rgb_path.stem}.vis.png"
        paths = {
            "path_rgb": rgb_path,
            "path_depth": camera / "depth" / visual_name,
            "path_normal": camera / "normal" / visual_name,
            "path_mask": mask_path,
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            if args.strict:
                raise FileNotFoundError(f"Missing modalities: {missing}")
            skipped += 1
            continue
        relative = Path(scene) / camera.name / rgb_path.name
        records.append({
            "cat": "indoor",
            "idx_cat": args.class_id,
            "path_rel_rgb": str(relative),
            **{name: str(path) for name, path in paths.items()},
        })

    if not records:
        raise SystemExit("No Hypersim records found; check the split and processed layout")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {output}; skipped {skipped} incomplete samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["seg", "train"], required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--split-file", required=True, help="Text file containing one scene id per line.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-id", type=int, default=1000)
    parser.add_argument("--strict", action="store_true")
    main(parser.parse_args())
