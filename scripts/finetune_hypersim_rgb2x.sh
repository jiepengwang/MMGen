#!/usr/bin/env bash
# Finetune the multi-task SiT-XL/2 for rgb -> {depth, normal, mask} CONDITIONAL
# generation on hypersim (mode A): every sample treats rgb (task 0) as the clean
# generation condition and supervises only depth/normal/mask. This specializes
# the model for the understanding direction (rgb in, modalities out).
#
# vs scripts/finetune_hypersim.sh (joint generation):
#   + --cond-task-fixed 0   : lock the condition task to rgb (needs --use-decouple-rand1)
#   - --ratio-train-onlyrgb : set to 0 (the fixed-condition logic replaces it)
#   - --use-quater-mixing   : dropped (no mixing; all samples stay in rgb->X mode)
#
# Modalities: num_tasks=4 -> 0=rgb, 1=depth, 2=normal, 3=mask.
# Architecture flags MUST match the base checkpoint.
# MAX_STEPS is the ABSOLUTE step count (base ckpt is at 600000).
set -euo pipefail
cd "$(dirname "$0")/.."

# ============ configurable ============
CKPT="${CKPT:-ckpt/0600000.pt}"              # base checkpoint (4-task, step 600000)
OUTPUT_DIR="${OUTPUT_DIR:-exps}"                  # run folder root (save_dir = OUTPUT_DIR/EXP_NAME)
EXP_NAME="${EXP_NAME:-finetune_hypersim_rgb2x}"
GPU="${GPU:-0,1,2,3}"                        # comma-separated GPU ids
NPROC="${NPROC:-4}"                          # must match the number of GPUs in $GPU
BATCH="${BATCH:-256}"                        # global batch (split across GPUs)
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-620000}"             # absolute target step (base is 600000)
CKPT_STEPS="${CKPT_STEPS:-2000}"             # save checkpoint every N steps (~11G each)
SAMPLE_STEPS="${SAMPLE_STEPS:-2000}"         # EMA sample visualization every N steps
NUM_WORKERS="${NUM_WORKERS:-12}"
COND_TASK="${COND_TASK:-0}"                  # condition task idx (0=rgb)
METADATA="${METADATA:-${REPA_METADATA:-data/local/hypersim/splits/train.jsonl}}"
VAE_RGB_H5="${VAE_RGB_H5:-${REPA_VAE_RGB_H5:-data/local/hypersim/h5/vae/rgb.h5}}"
VAE_DEPTH_H5="${VAE_DEPTH_H5:-${REPA_VAE_DEPTH_H5:-data/local/hypersim/h5/vae/depth.h5}}"
VAE_NORMAL_H5="${VAE_NORMAL_H5:-${REPA_VAE_NORMAL_H5:-data/local/hypersim/h5/vae/normal.h5}}"
VAE_MASK_H5="${VAE_MASK_H5:-${REPA_VAE_MASK_H5:-data/local/hypersim/h5/vae/mask.h5}}"
DINO_RGB_H5="${DINO_RGB_H5:-${REPA_DINO_RGB_H5:-data/local/hypersim/h5/dino/rgb.h5}}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"
# ======================================

echo "rgb->X finetune on hypersim: base=$CKPT  out=$OUTPUT_DIR/$EXP_NAME  gpu=$GPU  lr=$LR  cond_task=$COND_TASK  -> step $MAX_STEPS"

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
  --ratio-train-onlyrgb=0.0 \
  --use-decouple-task --use-decouple-rand1 --use-taskcond-emb --use-diff-task-weight \
  --cond-task-fixed="$COND_TASK" \
  --learning-rate="$LR" --max-train-steps="$MAX_STEPS" \
  --ckpt="$CKPT" \
  --sampling-steps="$SAMPLE_STEPS" --checkpointing-steps="$CKPT_STEPS"
