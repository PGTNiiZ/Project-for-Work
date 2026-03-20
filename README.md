# Project-for-Work

This repository mixes data preparation, image processing, feature engineering, and training experiments for social profile matching. The top level has been cleaned so the root now holds only high-signal entry points and project folders.

## Top-level layout

- `scripts/`: standalone utility and analysis scripts that used to float in the repo root
- `archive/`: old notebook copies and historical scratch files kept out of the active workflow
- `image_process/`: image download, embedding, captioning, and image matching pipelines
- `clean_data/`: dataset cleaning and location normalization scripts
- `bigram_username_and_fullname/`: username/full-name matching pipeline
- `bio_vector/`: notebook-based bio embedding experiments
- `train_data/`: feature generation, labeling, training, scoring, and experiment suites
- `data/`: source datasets, processed outputs, manifests, and matching results
- `data_for_project/`: project-specific exported CSVs used across stages
- `confix/`: one-off repair and migration utilities
- `docs/`: repository notes and structure guidance

## Main script entry points

- `scripts/image/advanced_embeddings.py`
- `scripts/analysis/compare_pipeline_metrics.py`
- `image_process/recover_profile_images.py`
- `image_process/image_matching_pipeline.py`
- `clean_data/preprocess_dataset.py`
- `train_data/stage10_13_training_pipeline.py`

## Working rules

- Put new standalone scripts in `scripts/` instead of the repository root.
- Move old notebook copies and ad-hoc backups into `archive/` instead of leaving them beside active files.
- Keep large generated artifacts under existing data folders, not beside source code.
- Keep notebooks inside the domain folder they belong to, for example `train_data/` or `bio_vector/`.
- Avoid adding cache, venv, or OS-generated files to git.
- Prefer project-root-based paths in scripts so they work regardless of the current working directory.

More detail is in `docs/PROJECT_STRUCTURE.md`.
