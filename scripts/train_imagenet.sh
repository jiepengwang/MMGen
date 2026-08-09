#!/usr/bin/env bash
# Train the released four-task MMGen model on ImageNet pseudo-labels.
# BATCH is the global batch and is split evenly across NPROC processes.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-exps}"
EXP_NAME="${EXP_NAME:-train_imagenet_mmgen}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-600000}"
CKPT_STEPS="${CKPT_STEPS:-50000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-10000}"
NUM_WORKERS="${NUM_WORKERS:-12}"
METADATA="${METADATA:-${REPA_METADATA:-data/local/metadata/train.jsonl}}"
VAE_RGB_H5="${VAE_RGB_H5:-${REPA_VAE_RGB_H5:-data/local/h5/vae/rgb.h5}}"
VAE_DEPTH_H5="${VAE_DEPTH_H5:-${REPA_VAE_DEPTH_H5:-data/local/h5/vae/depth.h5}}"
VAE_NORMAL_H5="${VAE_NORMAL_H5:-${REPA_VAE_NORMAL_H5:-data/local/h5/vae/normal.h5}}"
VAE_MASK_H5="${VAE_MASK_H5:-${REPA_VAE_MASK_H5:-data/local/h5/vae/mask.h5}}"
DINO_RGB_H5="${DINO_RGB_H5:-${REPA_DINO_RGB_H5:-data/local/h5/dino/rgb.h5}}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"

IFS=',' read -r -a GPU_IDS <<< "$GPU"
if [[ "${#GPU_IDS[@]}" -ne "$NPROC" ]]; then
  echo "NPROC=$NPROC must match the number of GPU ids in GPU=$GPU" >&2
  exit 2
fi
if (( BATCH % NPROC != 0 )); then
  echo "BATCH=$BATCH must be divisible by NPROC=$NPROC" >&2
  exit 2
fi
for path in "$METADATA" "$VAE_RGB_H5" "$VAE_DEPTH_H5" \
  "$VAE_NORMAL_H5" "$VAE_MASK_H5" "$DINO_RGB_H5"; do
  if [[ ! -f "$path" ]]; then
    echo "Required preprocessed input not found: $path" >&2
    exit 2
  fi
done

echo "Training ImageNet MMGen: out=$OUTPUT_DIR/$EXP_NAME gpu=$GPU batch=$BATCH steps=$MAX_STEPS"

CUDA_VISIBLE_DEVICES="$GPU" accelerate launch \
  --num_processes="$NPROC" --mixed_precision=fp16 train.py \
  --report-to=tensorboard --allow-tf32 --mixed-precision=fp16 --seed=0 \
  --path-type=linear --prediction=v --weighting=uniform --cfg-prob=0.1 \
  --model=SiT-XL/2 --proj-coeff=0.5 --encoder-depth=8 \
  --metadata-path="$METADATA" --vae-rgb-h5="$VAE_RGB_H5" \
  --vae-depth-h5="$VAE_DEPTH_H5" --vae-normal-h5="$VAE_NORMAL_H5" \
  --vae-mask-h5="$VAE_MASK_H5" --dino-rgb-h5="$DINO_RGB_H5" \
  --vae-path="$VAE_PATH" --batch-size="$BATCH" --num-workers="$NUM_WORKERS" \
  --output-dir="$OUTPUT_DIR" --exp-name="$EXP_NAME" \
  --is-mmcat-diff --num-tasks=4 --ratio-train-onlyrgb=0.5 \
  --use-quarter-mixing --use-decouple-task --use-decouple-rand1 \
  --use-taskcond-emb --use-diff-task-weight \
  --learning-rate="$LR" --max-train-steps="$MAX_STEPS" \
  --sampling-steps="$SAMPLE_STEPS" --checkpointing-steps="$CKPT_STEPS"
