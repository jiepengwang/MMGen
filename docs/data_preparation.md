# Data preparation

## Path configuration

Every public entry point accepts explicit CLI paths. When a CLI path is not
provided, `data/path.py` resolves an environment variable and then falls back
to `<repo>/data/local`. The precedence is therefore:

```text
CLI argument > environment variable > repository-anchored default
```

The main variables are `REPA_DATA_ROOT`, `REPA_METADATA`,
`REPA_VAE_RGB_H5`, `REPA_VAE_DEPTH_H5`, `REPA_VAE_NORMAL_H5`,
`REPA_VAE_MASK_H5`, and `REPA_DINO_RGB_H5`. External pseudo-label tools use
`DEPTH_ANYTHING_V2_REPO`, `DEPTH_ANYTHING_V2_CHECKPOINT`,
`SEMANTIC_SAM_REPO`, and `SEMANTIC_SAM_CHECKPOINT`.

These module-level values are defaults evaluated at process startup, not
compile-time macros. Prefer explicit CLI paths in reproducible job scripts.

## Training contract

`MMGenDataset` consumes one JSON object per line with these fields:

| Field | Meaning |
|---|---|
| `cat` | Category name |
| `idx_cat` | ImageNet class id; 1000 is the null class |
| `path_rgb` | RGB image path |
| `path_depth` | Relative-depth visualization path |
| `path_normal` | Three-channel surface-normal path |
| `path_mask` | Three-channel segmentation-mask path |
| `path_rel_rgb` | Stable, unique key shared by every HDF5 file |

All modalities receive the same center crop to 256 pixels and are normalized
to `[-1, 1]` before VAE encoding. Derived modalities are PNG files and must
mirror the RGB relative path. Depth is the per-image relative visualization
used during training, not metric depth.

## ImageNet pseudo-labels

Start with the standard class-directory layout:

```text
dataset/
  rgb/train/<class>/<image>.(JPEG|png)
  depth/train/<class>/<image>.png
  normal/train/<class>/<image>.png
  mask/train/<class>/<image>.png
  metadata/train.jsonl
  h5/vae/{rgb,depth,normal,mask}.h5
  h5/dino/rgb.h5
```

The paper uses Depth Anything V2-Large, StableNormal, and Semantic-SAM-L
automatic level 2 as ImageNet pseudo-label experts. Their repositories and
weights are external assets and are not redistributed here.

```bash
python data/preprocess_depth.py \
  --input dataset/rgb/train --output dataset/depth/train \
  --repo /path/to/Depth-Anything-V2 \
  --checkpoint /path/to/depth_anything_v2_vitl.pth

python data/preprocess_normal.py \
  --input dataset/rgb/train --output dataset/normal/train

python data/preprocess_mask.py \
  --input dataset/rgb/train --output dataset/mask/train \
  --semantic-sam-repo /path/to/Semantic-SAM \
  --checkpoint /path/to/swinl_only_sam_many2many.pth
```

The three scripts share `--start`, `--limit`, `--shard-index`,
`--num-shards`, `--overwrite`, and `--continue-on-error`. Files are sorted
before deterministic strided sharding. For an eight-worker run, launch shard
indices 0 through 7 with `--num-shards 8`, assigning one GPU per process.

Build metadata only after all pseudo-label trees are complete:

```bash
python data/prepare_meta_data.py \
  --rgb-dir dataset/rgb/train \
  --depth-dir dataset/depth/train \
  --normal-dir dataset/normal/train \
  --mask-dir dataset/mask/train \
  --output dataset/metadata/train.jsonl \
  --strict
```

Without `--strict`, incomplete samples are skipped and counted.

## VAE and DINO features

Run the shared VAE preprocessor once per modality:

```bash
for modality in rgb depth normal mask; do
  python data/preprocess_vae.py \
    --modality "$modality" \
    --metadata dataset/metadata/train.jsonl \
    --output "dataset/h5/vae/$modality.h5" \
    --vae-path stabilityai/sd-vae-ft-mse
done

python data/preprocess_dino.py \
  --metadata dataset/metadata/train.jsonl \
  --output dataset/h5/dino/rgb.h5
```

Each VAE HDF5 key equals `path_rel_rgb` and stores posterior mean/std moments
with shape `[4, 32, 32, 2]`. The RGB DINO HDF5 uses the same key and stores
DINOv2-B/14 patch tokens with shape `[256, 768]`.

For offline DINO loading, pass `--dinov2-repo` and `--dinov2-weights`.
Existing HDF5 outputs are protected by default; use `--overwrite` to replace
them.

## Hypersim

The processed layout is
`processed/<scene>/cam_XX/{rgb,depth,normal,mask}/`. Depth and normal files use
the `frame.XXXX.vis.png` form. First build explicit RGB-to-mask jobs, because
the mask output is inside each camera directory:

```bash
python data/gen_hypersim_jsonl.py \
  --mode seg \
  --processed-root /path/to/processed \
  --split-file /path/to/splits/train.txt \
  --output /path/to/splits/train_rgb_mask.jsonl

python data/preprocess_mask.py \
  --metadata /path/to/splits/train_rgb_mask.jsonl \
  --semantic-sam-repo /path/to/Semantic-SAM \
  --checkpoint /path/to/semantic_sam.pth

python data/gen_hypersim_jsonl.py \
  --mode train \
  --processed-root /path/to/processed \
  --split-file /path/to/splits/train.txt \
  --output /path/to/splits/train.jsonl \
  --strict
```

Then run the same `preprocess_vae.py` and `preprocess_dino.py` commands with
the Hypersim training JSONL and output locations.
