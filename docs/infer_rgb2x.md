# MMGen inference

The released four-task checkpoints use task order `0=rgb`, `1=depth`,
`2=normal`, and `3=mask`. `exp_sample.py` restores the architecture from the
checkpoint and exposes the three inference modes described in the paper.

## Category-conditioned generation

Generate all four modalities jointly from an ImageNet class id:

```bash
CUDA_VISIBLE_DEVICES=0 python exp_sample.py \
  --ckpt ckpt/0600000.pt \
  --vae-path stabilityai/sd-vae-ft-mse \
  --condition-task category \
  --class-label 207 \
  --num-samples 4 \
  --num-steps 250
```

Class ids are in `[0, 1000)`. Category generation uses task-condition token 0
and a shared denoising schedule across all four modalities.

## RGB understanding

The convenience wrapper builds metadata, encodes RGB, holds task 0 near the
clean endpoint, and generates depth, normal, and mask:

```bash
INPUT=/path/to/image-or-directory \
CKPT=ckpt/0620000.pt \
VAE_PATH=stabilityai/sd-vae-ft-mse \
bash scripts/inference/infer_rgb2x.sh
```

`IDX_CAT` defaults to 1000, the null ImageNet class. Set it to a known class id
when the input label is available. Other wrapper variables are `SAMPLE_DIR`,
`META`, `NUM_STEPS`, `BATCH`, `SEED`, `GPU`, and `EXP_TAG`.

## Depth, normal, or mask conditioning

Build one-modality metadata, then select the same modality in the sampler:

```bash
python data/prepare_infer_meta.py \
  --input /path/to/depth-images \
  --modality depth \
  --idx-cat 1000 \
  --out /tmp/depth_condition.jsonl

CUDA_VISIBLE_DEVICES=0 python exp_sample.py \
  --ckpt ckpt/0600000.pt \
  --vae-path stabilityai/sd-vae-ft-mse \
  --condition-task depth \
  --path-meta-data /tmp/depth_condition.jsonl \
  --cfg-scale 1.8 \
  --num-steps 250
```

Replace `depth` with `normal` or `mask` in both commands. The condition image
is center-cropped to 256 pixels, encoded by the shared VAE, inserted into its
four-channel task slot, and held at `--t-blend-start` while all other tasks are
sampled from noise. The corresponding task-condition token is 1 through 4.
The paper's conditioned-generation table uses classifier-free guidance scale
`1.8`; this only changes the result when `idx_cat` is a known class rather
than the default null class 1000. Category-generation results in the main
paper table are reported without guidance, matching the default
`--cfg-scale 1.0`.

## Outputs and distributed execution

Each run writes `task_0` through `task_3` plus a horizontal `task_merge` under
`<sample-dir>/<experiment>/<run>/<mode>/`. Images retain safe relative names
from the metadata.

For distributed modality-conditioned inference, launch `exp_sample.py` with
`torchrun` and `--is-distributed`. The dataset is deterministically strided by
rank without padding or duplicate samples. Category generation uses the same
sharding behavior over `--num-samples`.

The one-step runs mentioned in the model card are compatibility checks only.
Use the documented 250 Euler-Maruyama steps for normal sampling.
