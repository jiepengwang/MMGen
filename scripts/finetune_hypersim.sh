#!/usr/bin/env bash
# Finetune the multi-task SiT-XL/2 on the hypersim 4-modality data.
#
# Modalities: num_tasks=4 -> 0=rgb, 1=depth, 2=normal, 3=mask.
# Data is read from precomputed HDF5 (VAE latents + DINO features). No raw
# images are loaded at training time.
#
# Architecture flags below MUST match the base checkpoint
# (is_mmcat_diff / num_tasks / use_decouple_task / use_taskcond_emb).
# The remaining flags preserve the verified joint fine-tuning recipe; lr=1e-4.
#
# Note: MAX_STEPS is the ABSOLUTE step count, not a delta. The base ckpt is at
# step 600000, so MAX_STEPS=605000 means "train 5k more steps".
set -euo pipefail
cd "$(dirname "$0")/.."

# ============ configurable ============
CKPT="${CKPT:-ckpt/0600000.pt}"              # base checkpoint (4-task, step 600000)
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
EXP_NAME="${EXP_NAME:-finetune_hypersim_joint}"
GPU="${GPU:-0,1}"                            # comma-separated GPU ids
NPROC="${NPROC:-2}"                          # must match the number of GPUs in $GPU
BATCH="${BATCH:-256}"                        # global batch (split across GPUs)
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-605000}"             # absolute target step (base is 600000)
CKPT_STEPS="${CKPT_STEPS:-1000}"             # save checkpoint every N steps (~11G each)
SAMPLE_STEPS="${SAMPLE_STEPS:-1000}"         # EMA sample visualization every N steps
NUM_WORKERS="${NUM_WORKERS:-12}"
METADATA="${METADATA:-${REPA_METADATA:-data/local/hypersim/splits/train.jsonl}}"
VAE_RGB_H5="${VAE_RGB_H5:-${REPA_VAE_RGB_H5:-data/local/hypersim/h5/vae/rgb.h5}}"
VAE_DEPTH_H5="${VAE_DEPTH_H5:-${REPA_VAE_DEPTH_H5:-data/local/hypersim/h5/vae/depth.h5}}"
VAE_NORMAL_H5="${VAE_NORMAL_H5:-${REPA_VAE_NORMAL_H5:-data/local/hypersim/h5/vae/normal.h5}}"
VAE_MASK_H5="${VAE_MASK_H5:-${REPA_VAE_MASK_H5:-data/local/hypersim/h5/vae/mask.h5}}"
DINO_RGB_H5="${DINO_RGB_H5:-${REPA_DINO_RGB_H5:-data/local/hypersim/h5/dino/rgb.h5}}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"
# ======================================

echo "Finetuning on hypersim: base=$CKPT  exp=$EXP_NAME  gpu=$GPU  lr=$LR  -> step $MAX_STEPS"

CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes="$NPROC" --mixed_precision=fp16 train.py \
  --report-to=tensorboard --allow-tf32 --mixed-precision=fp16 --seed=0 \
  --path-type=linear --prediction=v --weighting=uniform \
  --model=SiT-XL/2 --proj-coeff=0.5 --encoder-depth=8 \
  --metadata-path="$METADATA" --vae-rgb-h5="$VAE_RGB_H5" \
  --vae-depth-h5="$VAE_DEPTH_H5" --vae-normal-h5="$VAE_NORMAL_H5" \
  --vae-mask-h5="$VAE_MASK_H5" --dino-rgb-h5="$DINO_RGB_H5" --vae-path="$VAE_PATH" \
  --batch-size="$BATCH" --num-workers="$NUM_WORKERS" \
  --output-dir="$OUTPUT_DIR" --exp-name="$EXP_NAME" \
  --is-mmcat-diff --num-tasks=4 \
  --ratio-train-onlyrgb=0.5 --use-quarter-mixing \
  --use-decouple-task --use-decouple-rand1 --use-taskcond-emb --use-diff-task-weight \
  --learning-rate="$LR" --max-train-steps="$MAX_STEPS" \
  --ckpt="$CKPT" \
  --sampling-steps="$SAMPLE_STEPS" --checkpointing-steps="$CKPT_STEPS"
