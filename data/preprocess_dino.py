"""Encode RGB images into DINOv2 patch-token features in HDF5."""

import argparse
import sys
from pathlib import Path

import h5py
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.path as PathConfig
from data.mmgen_dataset import MMGenDataset, center_crop_arr


def write_dataset(handle, key, value):
    group_name, dataset_name = key.rsplit("/", 1) if "/" in key else ("", key)
    group = handle.require_group(group_name) if group_name else handle
    if dataset_name in group:
        del group[dataset_name]
    group.create_dataset(dataset_name, data=value)


def load_encoder(args):
    if args.dinov2_repo:
        kwargs = {"source": "local"}
        repo = args.dinov2_repo
    else:
        kwargs = {"source": "github", "trust_repo": True}
        repo = "facebookresearch/dinov2"
    if args.dinov2_weights:
        kwargs["weights"] = args.dinov2_weights
    encoder = torch.hub.load(repo, args.model_name, **kwargs)
    if hasattr(encoder, "head"):
        del encoder.head
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [16, 16]
    )
    encoder.head = torch.nn.Identity()
    return encoder


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    output = Path(args.output).expanduser()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True,
        ),
    ])
    dataset = MMGenDataset(
        path_meta_data=args.metadata,
        transoform=transform,
        num_tasks=1,
        mode_load_1task="rgb",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    with h5py.File(output, "w") as handle:
        for batch in tqdm(loader):
            with torch.no_grad():
                features = encoder.forward_features(batch["x"].to(device))["x_norm_patchtokens"]
            for key, value in zip(batch["path_rel_rgb"], features.cpu().numpy()):
                write_dataset(handle, key, value)
    print(f"Wrote {len(dataset)} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=PathConfig.path_meta_data)
    parser.add_argument("--output", default=PathConfig.path_h5_dino_rgb)
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--dinov2-repo", help="Optional local DINOv2 repository checkout.")
    parser.add_argument("--dinov2-weights", help="Optional local DINOv2 checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
