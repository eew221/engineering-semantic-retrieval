# Zenodo DOI Release Steps

This repository is prepared for Zenodo DOI minting through a GitHub release.

## What is already prepared

- `CITATION.cff`
- `.zenodo.json`
- public repository with code, configs, metrics, and figures

## What you need to do in the web UI

1. Log in to Zenodo.
2. In Zenodo, connect your GitHub account if it is not already connected.
3. In Zenodo's GitHub tab, enable the repository:
   - `eew221/engineering-semantic-retrieval`
4. Create a GitHub release for this repository.
5. Wait for Zenodo to archive the release and mint the DOI.
6. Copy the minted DOI back into the manuscript/revision materials if needed.

## Recommended first release

- Tag: `v1.0.0`
- Release title: `v1.0.0: TVC revision reproducibility release`

## Suggested release notes

This release contains the reproducibility materials for the revised submission of:

`Engineering-Semantic Retrieval of Bridge Defects for Visual Inspection Archives`

Included:

- training and evaluation scripts
- experiment configurations
- retrieval split manifests with relative paths
- revised metrics and significance outputs
- embedding visualizations
- qualitative figures
- citation metadata for reuse

Not included:

- raw third-party datasets
- processed crop images
- large checkpoints

## After DOI is minted

Update the following files if you want the DOI reflected in the repository text:

- `README.md`
- `docs/release_and_citation.md`
- `CITATION.cff`
