"""Restricted loading for REPA training checkpoints."""

import argparse

import torch


def load_checkpoint(path, *, mmap=False):
    """Load tensor state and allow only the legacy argparse metadata object."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=mmap,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(checkpoint)!r}")
    return checkpoint
