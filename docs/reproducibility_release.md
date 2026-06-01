# Reproducibility Release Notes

This repository publishes the executable parts of the project that can be shared without redistributing third-party images.

## Included

- code under `src/bridge_retrieval/`
- all training and evaluation scripts under `scripts/`
- experiment configs under `configs/`
- sanitized split manifests under `data/processed/`
- metrics and analysis figures under `outputs/`

## Not Included

- raw source images from DACL10K, CODEBRIM, or SDI
- derived crop-image folders
- large model checkpoints

## Manifest Convention

The committed CSV files use relative paths such as:

- `images/train/...`
- `data/processed/dacl10k/crops/train/...`
- `classification_dataset/test/...`
- `SDI_DATASET_v1/...`

You should adapt these relative paths to your local dataset roots when reproducing the experiments.
