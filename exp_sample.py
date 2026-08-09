# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run MMGen category generation or generation conditioned on one modality."""

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from data.mmgen_dataset import MMGenDataset
from data.path import TASK_NAMES, map_idxtask_to_name
from models.sit import SiT_models
from samplers import euler_maruyama_sampler
from utils.checkpoints import load_checkpoint
from utils.latents import sample_posterior


RELEASE_RESOLUTION = 256
RELEASE_NUM_TASKS = 4


class CategoryDataset(Dataset):
    def __init__(self, num_samples, class_label):
        self.num_samples = num_samples
        self.class_label = class_label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "idx_cat": self.class_label,
            "path_rel_rgb": f"{index:06d}.png",
        }


def checkpoint_arg(args, name, default=None):
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


@torch.no_grad()
def encode_images(vae, images):
    distribution = vae.encode(images).latent_dist
    moments = torch.stack([distribution.mean, distribution.std], dim=-1)
    scale = torch.full(
        (1, 4, 1, 1), 0.18215, device=images.device, dtype=moments.dtype
    )
    return sample_posterior(moments, latents_scale=scale)


def output_relative_path(value):
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path_rel_rgb must be a safe relative path, got {value!r}")
    return relative.with_suffix(".png")


def setup_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("RGB-to-modalities inference requires a CUDA GPU")

    if args.is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        device = torch.device("cuda", local_rank)
        seed = args.global_seed * world_size + rank
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda", 0)
        seed = args.global_seed

    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    return device, rank, world_size


def build_model(args, device):
    checkpoint = load_checkpoint(args.ckpt, mmap=True)
    if "ema" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain EMA weights: {args.ckpt}")
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain training arguments: {args.ckpt}")

    checkpoint_args = checkpoint["args"]
    state_dict = checkpoint["ema"]
    model_name = checkpoint_arg(checkpoint_args, "model")
    resolution = checkpoint_arg(checkpoint_args, "resolution", RELEASE_RESOLUTION)
    num_tasks = checkpoint_arg(checkpoint_args, "num_tasks", 1)
    if model_name not in SiT_models:
        raise ValueError(f"Unsupported checkpoint model: {model_name!r}")
    if resolution != RELEASE_RESOLUTION:
        raise ValueError(f"Release inference requires {RELEASE_RESOLUTION}px checkpoints")
    if num_tasks != RELEASE_NUM_TASKS:
        raise ValueError(f"Release inference requires {RELEASE_NUM_TASKS} tasks")
    if not checkpoint_arg(checkpoint_args, "is_mmcat_diff", False):
        raise ValueError("Release checkpoint is not a channel-concatenated multi-task model")
    if not checkpoint_arg(checkpoint_args, "use_decouple_task", False):
        raise ValueError("Release checkpoint does not use task-decoupled timesteps")
    if not checkpoint_arg(checkpoint_args, "use_taskcond_emb", False):
        raise ValueError("Release checkpoint does not contain task-conditioning embeddings")
    if checkpoint_arg(checkpoint_args, "use_decouple_latent", False):
        raise ValueError("Latent-decoupled checkpoints are not supported by this inference entrypoint")

    num_classes = checkpoint_arg(checkpoint_args, "num_classes", 1000)
    label_count = state_dict["y_embedder.embedding_table.weight"].shape[0]
    class_dropout_prob = checkpoint_arg(
        checkpoint_args,
        "cfg_prob",
        0.1 if label_count == num_classes + 1 else 0.0,
    )
    z_dim = state_dict["projectors.0.4.weight"].shape[0]
    fused_attn = checkpoint_arg(checkpoint_args, "fused_attn", False)
    if args.fused_attn is not None:
        fused_attn = args.fused_attn
    model = SiT_models[model_name](
        input_size=resolution // 8,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        z_dims=[z_dim],
        encoder_depth=checkpoint_arg(checkpoint_args, "encoder_depth", 8),
        num_tasks=num_tasks,
        is_mmcat_diff=True,
        use_decouple_task=True,
        use_taskcond_emb=True,
        fused_attn=fused_attn,
        qk_norm=checkpoint_arg(checkpoint_args, "qk_norm", False),
    ).to(device)
    model.load_state_dict(state_dict)
    del state_dict, checkpoint
    path_type = checkpoint_arg(checkpoint_args, "path_type", "linear")
    return model.eval(), model_name, resolution, path_type, num_classes


def build_loader(args, rank, world_size, condition_task):
    if condition_task is None:
        dataset = CategoryDataset(args.num_samples, args.class_label)
    else:
        dataset = MMGenDataset(
            path_meta_data=args.path_meta_data,
            num_tasks=1,
            mode_load_1task=map_idxtask_to_name(condition_task),
        )
    if world_size > 1:
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
    return DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_output_dir(args, model_name, resolution, condition_task):
    model_name = model_name.replace("/", "-")
    checkpoint_name = Path(args.ckpt).stem
    experiment_name = args.exp_name_tag or checkpoint_name
    run_name = (
        f"{model_name}-{checkpoint_name}-size-{resolution}-"
        f"steps-{args.num_steps}-seed-{args.global_seed}-sde"
    )
    mode_name = "category" if condition_task is None else f"condition_{condition_task}"
    output = Path(args.sample_dir) / experiment_name / run_name / mode_name
    output.mkdir(parents=True, exist_ok=True)
    print(f"Writing samples to {output}")
    return output


@torch.no_grad()
def run_batch(
    args,
    model,
    vae,
    batch,
    output_dir,
    device,
    path_type,
    resolution,
    condition_task,
):
    relative_paths = [output_relative_path(value) for value in batch["path_rel_rgb"]]
    batch_size = len(relative_paths)
    latent_size = resolution // 8
    latents = torch.randn(
        batch_size, model.in_channels, latent_size, latent_size, device=device
    )

    if condition_task is not None:
        condition = encode_images(vae, batch["x"].to(device))
        if condition.shape[-2:] != (latent_size, latent_size):
            raise ValueError(f"Condition latent has an unexpected shape: {condition.shape}")
        start = condition_task * 4
        stop = start + 4
        latents[:, start:stop] = (
            latents[:, start:stop] * args.t_blend_start
            + condition * (1 - args.t_blend_start)
        )
    labels = batch["idx_cat"].to(device)
    samples = euler_maruyama_sampler(
        model=model,
        latents=latents,
        y=labels,
        num_steps=args.num_steps,
        path_type=path_type,
        condition_task=condition_task,
        t_blend_start=args.t_blend_start,
        cfg_scale=args.cfg_scale,
    ).to(torch.float32)

    scale = torch.full((1, 4, 1, 1), 0.18215, device=device)
    decoded_tasks = []
    for task_index, task_latents in enumerate(samples.chunk(model.num_tasks, dim=1)):
        decoded = vae.decode(task_latents / scale).sample
        decoded = (
            decoded.add(1)
            .div(2)
            .mul(255)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        decoded_tasks.append(decoded)
        task_dir = output_dir / f"task_{task_index}"
        for relative, image in zip(relative_paths, decoded):
            target = task_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)

    merge_dir = output_dir / "task_merge"
    for sample_index, relative in enumerate(relative_paths):
        merged = np.concatenate(
            [task[sample_index] for task in decoded_tasks], axis=1
        )
        target = merge_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(merged).save(target)


def main(args):
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.per_proc_batch_size < 1:
        raise ValueError("per_proc_batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 0.0 <= args.t_blend_start <= 1.0:
        raise ValueError("t_blend_start must be between 0 and 1")
    if args.cfg_scale < 1.0:
        raise ValueError("cfg_scale must be at least 1")
    condition_task = (
        None if args.condition_task == "category" else TASK_NAMES.index(args.condition_task)
    )
    if condition_task is None:
        if args.path_meta_data:
            raise ValueError("--path-meta-data is not used for category generation")
        if args.num_samples < 1:
            raise ValueError("num_samples must be positive")
    elif not args.path_meta_data:
        raise ValueError("--path-meta-data is required for modality-conditioned generation")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)
    device, rank, world_size = setup_runtime(args)
    model, model_name, resolution, path_type, num_classes = build_model(args, device)
    if condition_task is None and not 0 <= args.class_label < num_classes:
        raise ValueError(f"class_label must be in [0, {num_classes})")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    loader = build_loader(args, rank, world_size, condition_task)
    output_dir = build_output_dir(args, model_name, resolution, condition_task)

    iterator = tqdm(loader, disable=rank != 0)
    for batch in iterator:
        run_batch(
            args,
            model,
            vae,
            batch,
            output_dir,
            device,
            path_type,
            resolution,
            condition_task,
        )

    if args.is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", required=True, help="Path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", default="exps/inference")

    parser.add_argument(
        "--fused-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint attention implementation.",
    )

    parser.add_argument("--vae-path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--exp-name-tag")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale; the paper uses 1.8 for conditioned generation.",
    )

    parser.add_argument(
        "--condition-task",
        choices=("category", *TASK_NAMES),
        default="rgb",
        help="Use category for joint generation, or hold one input modality fixed.",
    )
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--t-blend-start", type=float, default=0.005)
    parser.add_argument("--is-distributed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--path-meta-data", "--path_meta_data", dest="path_meta_data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
