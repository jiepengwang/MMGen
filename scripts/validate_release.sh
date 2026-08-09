#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m compileall -q train.py exp_sample.py models data utils loss.py samplers.py

bash -n scripts/inference/infer_rgb2x.sh
bash -n scripts/train_imagenet.sh
if BATCH=255 NPROC=8 GPU=0,1,2,3,4,5,6,7 \
  bash scripts/train_imagenet.sh >/dev/null 2>&1; then
  echo "ImageNet launcher accepted a non-divisible global batch" >&2
  exit 1
fi
bash -n scripts/finetune_hypersim.sh
bash -n scripts/finetune_hypersim_rgb2x.sh

python train.py --help >/dev/null
python exp_sample.py --help >/dev/null
python data/prepare_meta_data.py --help >/dev/null
python data/gen_hypersim_jsonl.py --help >/dev/null
python data/preprocess_depth.py --help >/dev/null
python data/preprocess_normal.py --help >/dev/null
python data/preprocess_mask.py --help >/dev/null
python data/preprocess_vae.py --help >/dev/null
python data/preprocess_dino.py --help >/dev/null
python tests/smoke_data_pipeline.py
python tests/smoke_preprocessing.py
python tests/smoke_training_cli.py
python tests/smoke_model_training.py

if [[ -n "${REPA_RELEASE_CKPT:-}" ]]; then
  python tests/smoke_checkpoint.py --ckpt "$REPA_RELEASE_CKPT"
fi

if rg -n \
  -g '*.py' -g '*.sh' -g '*.md' -g '!scripts/validate_release.sh' \
  '(/vepfs/|/nvfile-|/tos-|/data_code/|/data_ckpt/|/home/|222\.201\.|100\.68\.|AKLT|T1RO)' \
  .; then
  echo "Found internal paths or credential-like strings" >&2
  exit 1
fi

echo "Release validation passed"
