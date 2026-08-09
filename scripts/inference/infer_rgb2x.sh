#!/usr/bin/env bash
# RGB -> {depth, normal, mask} conditional inference pipeline.
#
# Pipeline:
#   1) prepare_infer_meta.py : turn an image / folder into the meta jsonl MMGenDataset expects
#   2) exp_sample.py         : condition on task 0 (rgb), generate the other tasks
#
# The released models are multi-task SiT-XL/2 checkpoints trained with:
#   num_tasks=4 (0=rgb,1=depth,2=normal,3=mask), is_mmcat_diff, use_decouple_task, use_taskcond_emb
# Those flags MUST match the checkpoint, so they are hard-set below.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ============ configurable ============
INPUT="${INPUT:?Set INPUT to an image or directory}"
CKPT="${CKPT:-ckpt/0620000.pt}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"   # local dir or HF repo id
SAMPLE_DIR="${SAMPLE_DIR:-exps/infer_rgb2x}"
META="${META:-${SAMPLE_DIR}/meta_infer.jsonl}"
IDX_CAT="${IDX_CAT:-1000}"                    # 1000 = null class (unconditional); set a real id to condition on class
NUM_STEPS="${NUM_STEPS:-250}"
BATCH="${BATCH:-4}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"                               # single GPU id for inference
EXP_TAG="${EXP_TAG:-0620000}"
# ======================================

mkdir -p "$SAMPLE_DIR"

echo "[1/2] Building meta jsonl from: $INPUT"
python data/prepare_infer_meta.py --input "$INPUT" --out "$META" --idx-cat "$IDX_CAT"

echo "[2/2] Running RGB->X conditional generation (condition = task 0 / rgb)"
CUDA_VISIBLE_DEVICES="$GPU" python exp_sample.py \
  --ckpt "$CKPT" \
  --vae-path "$VAE_PATH" \
  --exp-name-tag "$EXP_TAG" \
  --num-steps "$NUM_STEPS" \
  --per-proc-batch-size "$BATCH" \
  --global-seed "$SEED" \
  --sample-dir "$SAMPLE_DIR" \
  --condition-task rgb \
  --path-meta-data "$META" \
  --no-is-distributed

echo "Done. Results under: ${SAMPLE_DIR}/${EXP_TAG}/.../condition_0/"
echo "  task_0=rgb(recon)  task_1=depth  task_2=normal  task_3=mask  task_merge=side-by-side"

# Multi-GPU mode shards the metadata without padding or duplicate samples:
#   torchrun --nproc_per_node=8 exp_sample.py ... --is-distributed
