"""Encode one image modality into Stable Diffusion VAE moments in HDF5."""

import argparse
import sys
from pathlib import Path

import h5py
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.path as PathConfig
from data.mmgen_dataset import MMGenDataset


DEFAULT_OUTPUTS = {
    "rgb": PathConfig.path_h5_rgb,
    "depth": PathConfig.path_h5_depth,
    "normal": PathConfig.path_h5_normal,
    "mask": PathConfig.path_h5_mask,
}


def write_dataset(handle, key, value):
    group_name, dataset_name = key.rsplit("/", 1) if "/" in key else ("", key)
    group = handle.require_group(group_name) if group_name else handle
    if dataset_name in group:
        del group[dataset_name]
    group.create_dataset(dataset_name, data=value)


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    output = Path(args.output or DEFAULT_OUTPUTS[args.modality]).expanduser()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = MMGenDataset(
        path_meta_data=args.metadata,
        num_tasks=1,
        mode_load_1task=args.modality,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    with h5py.File(output, "w") as handle:
        for batch in tqdm(loader):
            images = batch["x"].to(device)
            with torch.no_grad():
                distribution = vae.encode(images).latent_dist
                moments = torch.stack([distribution.mean, distribution.std], dim=-1)
            for key, value in zip(batch["path_rel_rgb"], moments.cpu().numpy()):
                write_dataset(handle, key, value)
    print(f"Wrote {len(dataset)} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modality", "--mode-data", dest="modality",
        choices=list(DEFAULT_OUTPUTS), required=True,
    )
    parser.add_argument("--metadata", default=PathConfig.path_meta_data)
    parser.add_argument("--output", help="HDF5 output path; defaults to data/path.py.")
    parser.add_argument("--vae-path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
