# FILE MAP

หมายเหตุ: ไฟล์ใน `pub_multi` ใช้เป็นแพ็กเกจรวมสำหรับตรวจงานและเขียนรายงาน ไม่ใช่ตัวแทนหลักของโค้ดต้นฉบับทั้งหมด หากต้องอธิบายวิธีทำในเล่ม ควรอ้างไฟล์ต้นฉบับใน `clean_data/`, `image_process/` และ `train_data/` เป็นหลัก

## root

- `build_pkg.py` ตัวสร้างแพ็กเกจนี้
- `README.md` สรุปภาพใหญ่
- `USE.md` ขั้นตอนเปิดดูงาน
- `FILES.csv` ดัชนีไฟล์สั้น ๆ

## src

- `s01_prep.py` เตรียมข้อมูลดิบ
- `s02_loc.py` mapping location
- `s03_norm.py` สร้าง normalized profile DB
- `s04_img.py` กู้/ดึงรูปโปรไฟล์
- `s05_pair.py` สร้าง labeled pairs
- `s06_feat.py` สร้าง features แบบ chunked
- `s07_base.py` baseline leak-safe experiment
- `s08_multi.py` multimodal suite หลัก
- `s09_full.py` full candidate retrieval/scoring
- `s10_crm.py` CRM/entity merge/export
- `s11_train.py` strict MLP reference
- `s12_imgpair.py` helper ด้าน image pair features

## original source files to cite in the report

- `Project-for-Work/clean_data/preprocess_dataset.py` preprocess ต้นทาง
- `Project-for-Work/clean_data/location_mapping_pipeline.py` location normalization
- `Project-for-Work/image_process/create_normalized_db.py` normalized profile database
- `Project-for-Work/image_process/recover_profile_images.py` image recovery pipeline
- `stage8_pair_builder.py` pair building
- `Project-for-Work/train_data/stage9_features_pipeline_chunked.py` feature engineering
- `Project-for-Work/train_data/leakage_safe_experiment/run_experiment.py` leakage-safe baseline
- `Project-for-Work/train_data/stage10_13_training_pipeline.py` training/calibration pipeline เดิม
- `Project-for-Work/train_data/stage7_13_multimodal_suite/run_multimodal_suite.py` multimodal suite หลัก
- `Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py` full candidate scoring
- `Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py` CRM/entity pipeline

## data

- `train_all.parquet` ไฟล์รวม train/val/test พร้อม score และ metadata
- `train_all.csv` เวอร์ชัน CSV
- `train_all_sample.csv` sample สำหรับเปิดดูเร็ว
- `fp_top.csv` false positives สำคัญ
- `fn_top.csv` false negatives สำคัญ

## res

- `model_cmp.csv` เทียบ run/model สำคัญ
- `blocking.csv` ตัวเลข retrieval หลัก
- `blocking_keys.csv` contribution ของ exact/candidate keys
- `tiers.csv` precision/count ของ exact-match-review
- `thr_sweep.csv` ผลตาม threshold
- `feat_imp.csv` feature importance ของ main model
- `tune_rank.csv` ranking ของ tuning
- `modality.csv` coverage ของภาพ/metadata
- `prod.csv` ตัวเลขปลายทางด้าน CRM

## fig

ไฟล์รูปทุกไฟล์ในโฟลเดอร์นี้ถูกสร้างใหม่จาก artifact จริง เพื่อใช้ในเล่มหรือสไลด์ได้ทันที

## ref

copy ของ report/raw artifacts ที่ดึงมาจาก source เดิม เพื่อให้ตรวจย้อนกลับได้ว่า summary ทุกตัวมาจากไหน
