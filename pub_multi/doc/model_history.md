# Model History

## How Many Models Were Tried

- Rule baseline: `exact match` 1 วิธี
- Model families that existed in the repo before this round: `logreg`, `gb`, `rf`, `mlp` รวม 4 family
- Main benchmark added in `pub_multi`: `logreg`, `gb`, `rf`, `extra_trees`, `linear_svm`, `adaboost`, `mlp` รวม 7 family
- Feature-set variants in the multimodal suite: `text_attr_hybrid`, `image_stats`, `image_context` รวม 3 ชุด
- Negative-ratio tuning combinations from rebuilt experiment: 6 ค่า config หลัก

## What Was Adjusted Over Time

1. Pair construction
- tuned `random_neg_ratio` และ `hard_neg_ratio`
- best region อยู่แถว `random_neg_ratio = 0.75` และ `hard_neg_ratio = 2.0-2.5`

2. Feature set
- `text_attr_hybrid` ใช้ประมาณ 23 features
- `image_stats` ขยายเป็นประมาณ 37 features
- `image_context` ขยายเป็นประมาณ 41 features

3. Model family comparison
- เดิมในสาย multimodal/hybrid เทียบ `logreg`, `gb`, `rf`
- สาย strict training มี `mlp`
- รอบ benchmark ใหม่นี้เพิ่ม `extra_trees`, `linear_svm`, `adaboost`

4. Probability and threshold
- ใช้ isotonic calibration
- เลือก threshold จาก validation โดย optimize F1
- สำหรับ strict report line เพิ่มการเทียบ `isotonic` กับ `sigmoid`
- จุดที่เลือกใช้ในรายงานสำหรับ strict reference คือ `logreg + sigmoid @ 0.50`

5. Leakage control
- มีสาย `leakage-safe`
- มีสาย `strict no-leak` ที่ตัด feature overlap-heavy ออกเพิ่ม

## Final Main Model

- chosen run: `stage7_13_multimodal_suite/runs/image_context_r075_h20_s42`
- chosen feature setting: `image_context`
- chosen model family: `gb`

เหตุผลหลัก
- ชนะในการเทียบหลาย model บน feature/split เดียวกัน
- ชนะหรือใกล้ชนะในการ tune ของ family ตัวเอง
- เหมาะกับข้อมูลแบบ tabular similarity features
- อธิบายได้ง่ายกว่า neural network

## Where the Final Model Comes From

- source suite: `train_data/stage7_13_multimodal_suite`
- source run report: `pub_multi/ref/main_report.json`
- benchmark summary: `pub_multi/res/model_family_cmp.csv`
- tuning summary: `pub_multi/res/top_model_tune.csv`

## Do We Need a Neural Network

คำตอบคือ `ไม่จำเป็นต้องใช้เป็น final model เสมอ`

Neural network ถูกใช้เป็น reference ที่สำคัญ เพราะช่วยตอบคำถามว่า
- ถ้าเพิ่ม model complexity แล้วดีขึ้นจริงไหม
- หรือข้อมูลชุดนี้เหมาะกับ tree ensemble มากกว่า

จากหลักฐานตอนนี้
- main benchmark: `mlp` ตามหลัง `gb` และ `rf`
- strict no-leak comparison: strict `mlp` เดิมตามหลัง `rf`, `extra_trees`, `gb`, `logreg` ใน test F1/AP
- strict report operating point ที่สมจริงกว่า precision=1 คือ `logreg + sigmoid` ให้ `P 0.9900`, `R 0.8169`, `F1 0.8951`

ดังนั้น neural network `ควรมี` ในฐานะ comparative reference
แต่ยัง `ไม่ใช่ model หลักที่เหมาะที่สุด` สำหรับโปรเจกต์นี้จากหลักฐานที่มีอยู่ตอนนี้
