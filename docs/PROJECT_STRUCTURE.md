# Project Structure

## Current layout

- `scripts/analysis/`
  - Small reporting or comparison scripts.
- `scripts/image/`
  - Standalone image-related utilities that are not part of the main `image_process/` module folder.
- `archive/`
  - Old notebook copies and scratch artifacts that should stay out of active project folders.
- `image_process/`
  - Image download, profile recovery, feature extraction, and matching logic.
- `clean_data/`
  - Dataset cleanup and preprocessing code.
- `bigram_username_and_fullname/`
  - Username and full-name matching experiments and pipelines.
- `bio_vector/`
  - Bio/text embedding notebooks and experiments.
- `train_data/`
  - Candidate generation, labeling, training, evaluation, and export stages.
- `data/`
  - Raw and processed datasets plus generated outputs used by the pipelines.
- `data_for_project/`
  - Shared exported CSVs that are consumed across notebooks and scripts.
- `confix/`
  - Fix-up scripts for migration or repair tasks.
- `docs/`
  - Human-readable documentation about repo organization.

## Placement guide

- New reusable Python scripts: `scripts/<topic>/`
- New notebooks: keep them inside the closest domain folder, not at repo root
- Notebook copies, backups, and abandoned variants: `archive/`
- One-off repair scripts: `confix/`
- Generated reports and metrics: under the owning data folder
- Temporary files and caches: keep them ignored by git

## Root-level rule

Only keep these at repository root:

- repo metadata such as `.gitignore`
- shared dependency files such as `requirements.txt`
- high-level documentation such as `README.md`
- major domain folders
