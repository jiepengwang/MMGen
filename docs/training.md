# Training

## ImageNet base training

After completing the preprocessing pipeline in `docs/data_preparation.md`, set
the six input paths and launch the recipe used for the 600k checkpoint:

```bash
METADATA=/path/to/metadata/train.jsonl \
VAE_RGB_H5=/path/to/h5/vae/rgb.h5 \
VAE_DEPTH_H5=/path/to/h5/vae/depth.h5 \
VAE_NORMAL_H5=/path/to/h5/vae/normal.h5 \
VAE_MASK_H5=/path/to/h5/vae/mask.h5 \
DINO_RGB_H5=/path/to/h5/dino/rgb.h5 \
GPU=0,1,2,3,4,5,6,7 NPROC=8 BATCH=256 MAX_STEPS=600000 \
bash scripts/train_imagenet.sh
```

`BATCH` is global and must be divisible by `NPROC`. The wrapper uses
SiT-XL/2, fp16, learning rate `1e-4`, REPA coefficient `0.5`, encoder depth 8,
four task-concatenated latents, modality-decoupled timesteps, random single
condition tasks, quarter mixing, task-condition embeddings, and differentiated
task loss weights. These flags match the inspected 600k release checkpoint.

The paper reports training with global batch 256 on eight A100 GPUs. Runtime
depends on storage throughput and the local software stack.

## Direct training entry point

```bash
accelerate launch train.py \
  --report-to tensorboard \
  --allow-tf32 --mixed-precision fp16 \
  --seed 0 --path-type linear --prediction v --weighting uniform \
  --model SiT-XL/2 \
  --proj-coeff 0.5 --encoder-depth 8 \
  --output-dir exps --exp-name mm-repa \
  --metadata-path /path/to/train.jsonl \
  --vae-rgb-h5 /path/to/vae/rgb.h5 \
  --vae-depth-h5 /path/to/vae/depth.h5 \
  --vae-normal-h5 /path/to/vae/normal.h5 \
  --vae-mask-h5 /path/to/vae/mask.h5 \
  --dino-rgb-h5 /path/to/dino/rgb.h5 \
  --batch-size 256 --num-workers 12 \
  --is-mmcat-diff --num-tasks 4 \
  --ratio-train-onlyrgb 0.5 --use-quarter-mixing \
  --use-decouple-task --use-decouple-rand1 \
  --use-taskcond-emb --use-diff-task-weight \
  --max-train-steps 600000
```

The paths may also be set with `REPA_METADATA`, `REPA_VAE_RGB_H5`,
`REPA_VAE_DEPTH_H5`, `REPA_VAE_NORMAL_H5`, `REPA_VAE_MASK_H5`, and
`REPA_DINO_RGB_H5`.

## Checkpoints and resume

Checkpoints are written to
`<output-dir>/<exp-name>/checkpoints/<step:07d>.pt` and contain model, EMA,
optimizer, arguments, and global step.

- `--resume-step N` resumes a checkpoint from the current experiment
  directory.
- `--ckpt /path/to/file.pt` starts from an explicit checkpoint, including
  its optimizer and global step.
- `--max-train-steps` is an absolute global-step target.
- The requested `--learning-rate` overrides the learning rate restored from
  the optimizer state.

The architecture flags and task order must match the checkpoint.

## Hypersim fine-tuning

Set paths and launch the joint recipe:

```bash
CKPT=ckpt/0600000.pt \
METADATA=/path/to/hypersim/train.jsonl \
VAE_RGB_H5=/path/to/h5/vae/rgb.h5 \
VAE_DEPTH_H5=/path/to/h5/vae/depth.h5 \
VAE_NORMAL_H5=/path/to/h5/vae/normal.h5 \
VAE_MASK_H5=/path/to/h5/vae/mask.h5 \
DINO_RGB_H5=/path/to/h5/dino/rgb.h5 \
MAX_STEPS=605000 \
bash scripts/finetune_hypersim.sh
```

For RGB-conditioned fine-tuning:

```bash
CKPT=ckpt/0600000.pt \
METADATA=/path/to/hypersim/train.jsonl \
VAE_RGB_H5=/path/to/h5/vae/rgb.h5 \
VAE_DEPTH_H5=/path/to/h5/vae/depth.h5 \
VAE_NORMAL_H5=/path/to/h5/vae/normal.h5 \
VAE_MASK_H5=/path/to/h5/vae/mask.h5 \
DINO_RGB_H5=/path/to/h5/dino/rgb.h5 \
MAX_STEPS=620000 \
bash scripts/finetune_hypersim_rgb2x.sh
```

The RGB-to-X recipe fixes task 0 as the clean condition and supervises tasks
1-3. Its default architecture and loss flags match the released 600k base
checkpoint.
