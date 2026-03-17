# Stage 7-13 Multimodal Suite

This folder contains an isolated leak-safe experiment suite that compares:

- `text_attr_hybrid`: TF-IDF + SBERT + attribute features
- `image_stats`: baseline plus local image statistics
- `image_context`: baseline plus image statistics and caption-to-text cross features

## Structure

- `run_multimodal_suite.py`
  Runs the full suite and writes comparable reports for each experiment.
- `run_multimodal_suite.ps1`
  PowerShell entry point.
- `runs/<run_name>/artifacts`
  Pair data, feature matrices, and cached feature lists for one run.
- `runs/<run_name>/models`
  Trained model, scaler, calibrator.
- `runs/<run_name>/reports`
  Per-run `experiment_report.json`, score files, and modality manifest.
- `reports/leaderboard.csv`
  Comparable run summary across experiments.
- `reports/suite_report.json`
  Suite-level summary with best run and cache locations.
- `shared_cache`
  Reused SBERT embeddings and image profile features.

## Data Sources

- Profiles: `data-for-project/normalized_profiles_with_profile_id.csv`
- Base leak-safe logic: `Train-Data/stage7_8_rebuilt_experiment_hybrid/run_rebuilt_pipeline.py`
- Local profile images: `Image-Process/downloaded_images`
- Image metadata: `data/final/image_features_complete.csv`

## Notes

- The non-leaky split, pair construction, and calibration logic are inherited from the rebuilt hybrid pipeline.
- Local image coverage is partial and currently maps to Twitter profiles only.
- Reports explicitly record modality coverage so missing-image behavior is visible.
