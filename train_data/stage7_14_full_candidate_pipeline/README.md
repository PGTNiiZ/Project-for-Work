# Stage 7-14 Full Candidate Pipeline

This folder runs the leak-safe model on real candidate pairs at production-style scale.

## What it does

1. Loads the corrected `normalized_profiles_with_profile_id.csv`
2. Builds exact matches and blocking-based candidate pairs on all real profiles
3. Scores candidate pairs in chunks with the best leak-safe multimodal model
4. Writes exact matches, candidate chunks, score chunks, predicted matches, and a full pipeline report

## Key outputs

- `artifacts/exact_matches_part*.parquet`
- `artifacts/candidate_pairs_part*.parquet`
- `scores/candidate_scores_*.parquet`
- `artifacts/predicted_matches.parquet`
- `reports/top_5000_predictions.csv`
- `reports/full_pipeline_report.json`

## Notes

- This is not a toy sampled train/test run. It scores real blocking-generated candidate pairs from the full normalized profile table.
- Search-space reduction and ground-truth coverage are reported explicitly.
- The scoring model is loaded from the best multimodal suite run:
  `stage7_13_multimodal_suite/runs/image_context_r075_h20_s42`
