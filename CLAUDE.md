# Project Context: Identity Resolution Pipeline

## Project Goal
ระบบ **Identity Resolution** — จับคู่ social media profiles ข้าม platform (Twitter, Google+, Instagram) ว่าเป็นคนเดียวกันหรือเปล่า แล้ว merge เป็น unified customer profile พร้อม lead score

## Python Environment
ใช้ `python3.11` (homebrew) เสมอ — ไม่ใช่ `python3` หรือ `python`
```bash
python3.11 script.py
python3.11 -m papermill notebook.ipynb notebook.ipynb --kernel python3
pip3.11 install package
```

## Key Data Files
| File | Description |
|------|-------------|
| `data-for-project/merged_profiles.csv` | **Main input** — 36,807 profiles, 22 columns (normalized text + user_folder + platform) |
| `data-for-project/combined_profiles.csv` | Raw profiles + user_folder + platform |
| `data-for-project/nomalized_profiles.csv` | Normalized text + cleaned URLs |
| `data/processed/all_profiles_cleaned.csv` | Cleaned profiles ready for pipeline (generated from merged_profiles) |
| `data/processed/candidate_pairs.csv` | Blocking output — pairs to evaluate |
| `data/processed/labeled_pairs.csv` | Training pairs with label 0/1 |
| `data/processed/feature_matrix.csv` | Features per pair for model training |
| `data/processed/model.pt` | Trained IdentityMLP model |
| `data/processed/predictions.csv` | Inference output with MATCH/POSSIBLE_MATCH/NO_MATCH |
| `Train-Data/ground_truth_matrix.npz` | Sparse binary matrix 36,807×36,807 (ground truth) |
| `Train-Data/ground_truth_matrix_index.csv` | matrix_idx → userName, user_folder |

## Pipeline Stages (Train-Data/)
```
build_ground_truth_matrix.ipynb  ← ใช้ merged_profiles.csv
stage7_blocking.ipynb            ← candidate pair generation
stage8_labeling.ipynb            ← label pairs using ground truth (groupby user_folder)
stage9_features.ipynb            ← 17 similarity features per pair
stage10_dataset.ipynb            ← entity-aware train/val/test split
stage11_training.ipynb           ← train IdentityMLP (FocalLoss)
stage12_13_calibration_evaluation.ipynb  ← isotonic calibration + threshold
stage14_inference.ipynb          ← inference on candidate_pairs
stage15_16_17_merge_scoring_export.ipynb ← merge → lead score → export
```

## Important Conventions

### profile_id format
`profile_id = platform_userName_clean` (เช่น `twitter_aaronbird`)
- ใช้ใน stage 7, 9, 14 สำหรับ lookup
- stage 8 groupby `user_folder` เพื่อหา positive pairs

### Ground Truth
- `user_folder` = entity ID (คนเดียวกัน = user_folder เดียวกัน)
- `ground_truth_matrix[i,j] = 1` ถ้า profile i และ j มี user_folder เดียวกัน

### Model
- `IdentityMLP` — 4 layers (256→128→64→1), FocalLoss, 17 input features
- Threshold: MATCH ≥ 0.90, POSSIBLE_MATCH ≥ 0.70 (both included in merge)
- Test metrics (v2): F1=0.915, AUC=0.970, Precision=0.959, Recall=0.874

## Model Performance Notes
- Overall F1=0.915, Recall=0.874 (v2 model — ตัวเลขเก่า F1=0.95 มาจาก version ก่อน)
- Hard cases (username_jaro < 0.5 AND fullname_jaro < 0.5): F1=0.13 — เป็น data quality issue ไม่ใช่ model issue
- 3 platforms: Twitter (13,960), Google+ (11,890), Instagram (10,957)
- POSSIBLE_MATCH (prob 0.70–0.89): รวมใน merge ด้วย — stage15 ใช้ทั้ง MATCH + POSSIBLE_MATCH

## Output Files (data/processed/)
| File | Description |
|------|-------------|
| `unified_profiles.csv` | Merged customer profiles |
| `profile_mapping.csv` | original profile_id → unified_customer_id |
| `lead_scores.csv` | Lead score + tier (Hot/Warm/Cold) per unified profile |
| `customer_360.json` | Full customer profile JSON |
| `crm_export.csv` | CRM-ready flat file |
| `pipeline_report.json` | Pipeline run summary |
