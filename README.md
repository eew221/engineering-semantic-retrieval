# Engineering-Semantic Retrieval of Bridge Defects for Visual Inspection Archives

This repository contains the code, experiment configs, public-data manifests, and result artifacts for our bridge-defect retrieval study.

The project reformulates bridge defect comparison as an **engineering-semantic retrieval** problem rather than a pure appearance-matching task. A query image should retrieve historical cases that are similar in:

- damage category
- structural component context
- visible defect extent

The implementation is built around a CLIP-initialized retrieval model with:

- weighted engineering-semantic pair supervision
- auxiliary visible-extent regression
- compositional text-anchor alignment

## Repository Scope

This repository is intended as a **code and reproducibility release**, not a raw-data mirror.

Included:

- training and evaluation code
- data preparation scripts
- experiment configs
- public release manifests with sanitized relative paths
- metrics JSON files used in the paper
- analysis figures used in the paper
- paired significance and bootstrap outputs for the revised baseline set
- citation and release metadata for future GitHub/Zenodo publication

Not included:

- raw DACL10K / CODEBRIM / SDI images
- processed crop images
- large checkpoints
- private journal submission workflow files

## Directory Layout

- `src/bridge_retrieval/`: reusable library code
- `scripts/`: data preparation, training, evaluation, and analysis scripts
- `configs/`: experiment configurations
- `data/processed/`: public manifests and split files with relative paths
- `outputs/metrics/`: saved evaluation outputs
- `outputs/figures/`: saved analysis figures

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Public Dataset Preparation

The repository does not redistribute third-party datasets. Download the public datasets separately and then use the preparation scripts here.

Main scripts:

- `scripts/prepare_dacl10k_retrieval.py`
- `scripts/prepare_codebrim_retrieval.py`
- `scripts/prepare_codebrim_from_extracted.py`
- `scripts/prepare_imagefolder_retrieval.py`

The committed CSV manifests under `data/processed/` use **relative paths** so they can serve as reference split files and examples of the expected format.

## Training

Example:

```bash
python scripts/train_retrieval.py --config configs/bridge_retrieval_lambda_t_05_3epoch.yaml
```

Other configs in `configs/` cover:

- zero-shot evaluation
- vanilla contrastive tuning
- triplet baseline
- supervised contrastive baseline
- hard-negative paired SupCon
- text-anchor ablations
- cross-dataset evaluation
- repeated-seed runs

## Evaluation

Example:

```bash
python scripts/evaluate_retrieval.py --config configs/bridge_retrieval_lambda_t_05_3epoch.yaml
```

Additional utilities:

- `scripts/bootstrap_retrieval_ci.py`
- `scripts/paired_retrieval_significance.py`
- `scripts/export_embedding_viz.py`
- `scripts/plot_lambda_sensitivity.py`
- `scripts/evaluate_checkpoint_series.py`

## Main Experimental Artifacts

Representative result files:

- `outputs/metrics/bridge_engineering_semantic_retrieval_lambda_t_05_3epoch_test_metrics.json`
- `outputs/metrics/bridge_engineering_semantic_retrieval_lambda_t_05_3epoch_seed7_test_metrics.json`
- `outputs/metrics/bridge_engineering_semantic_retrieval_lambda_t_05_3epoch_seed21_test_metrics.json`
- `outputs/metrics/bridge_engineering_semantic_retrieval_lambda_t_05_3epoch_bootstrap_ci.json`

Representative figures:

- `outputs/figures/lambda_t_sensitivity.png`
- `outputs/figures/full1to5epoch_convergence.png`

## Current Reproducibility Notes

- The pairwise engineering-semantic weights are implemented as reproducible design priors rather than expert-calibrated maintenance constants.
- The visible-extent target is an area-derived surrogate, not a structural severity label.
- Repeated-seed summaries are available for the main `lambda_t=0.5` configuration.
- Broader ablation families and some stronger baselines are still single-run comparisons.
- DOI minting is pending a formal GitHub release linked to Zenodo.

## Citation Note

This repository accompanies the manuscript:

**Engineering-Semantic Retrieval of Bridge Defects for Visual Inspection Archives**

prepared for submission to *The Visual Computer*.
