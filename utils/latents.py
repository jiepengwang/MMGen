"""Shared helpers for sampling Stable Diffusion VAE moment tensors."""

import torch


@torch.no_grad()
def sample_posterior(
    moments,
    latents_scale=1.0,
    latents_bias=0.0,
    is_mmcat_diff=False,
    num_tasks=4,
):
    """Sample latents from `[mean, std]` moments stored on the last axis."""
    if moments.shape[-1] != 2:
        raise ValueError(f"Expected VAE moments with last dimension 2, got {moments.shape}")
    mean, std = torch.chunk(moments, 2, dim=-1)
    mean = mean.squeeze(-1)
    std = std.squeeze(-1)

    if not is_mmcat_diff:
        return (mean + std * torch.randn_like(mean)) * latents_scale + latents_bias
    if mean.shape[1] % num_tasks:
        raise ValueError(
            f"Latent channels ({mean.shape[1]}) must be divisible by num_tasks ({num_tasks})"
        )
    samples = [
        (task_mean + task_std * torch.randn_like(task_mean))
        * latents_scale
        + latents_bias
        for task_mean, task_std in zip(
            mean.chunk(num_tasks, dim=1), std.chunk(num_tasks, dim=1)
        )
    ]
    return torch.cat(samples, dim=1)
