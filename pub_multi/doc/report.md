# REPORT

## 1. canonical pipeline

แพ็กเกจนี้ยึด pipeline เดียวเป็นหลัก คือ
- train/eval: `stage7_13_multimodal_suite`
- production retrieval/scoring: `stage7_14_full_candidate_pipeline`
- entity/export: `stage15_crm_entity_pipeline`

เหตุผลที่เลือกเส้นนี้ เพราะเป็นเส้นเดียวที่เชื่อมกันครบตั้งแต่ retrieval ไปจนถึง production decision และมี report พร้อมทั้งด้าน coverage, classification และ thresholding

## 2. why blocking is still needed

โจทย์นี้มี all cross-platform pairs = 449,149,239 คู่
ถ้า score ทุกคู่โดยตรงจะมีต้นทุนสูงมาก จึงต้องใช้ exact-first และ blocking เพื่อคัดคู่ที่น่าจะเกี่ยวข้องก่อน

retrieval summary:
- exact_match_pairs = 12,403
- candidate_pairs_for_model = 2,073,842
- search_space_reduction_pct = 99.5355
- ground_truth_coverage_pct = 88.6742

ประเด็นที่ต้องเขียนในเล่มคือ blocking stage ไม่ได้ optimize precision แต่ optimize candidate recall ภายใต้ข้อจำกัดด้านคอมพิวต์

## 3. why exact-first matters

exact stage ให้ precision สูงมากและควรถูกเล่าแยกออกจาก model stage
- exact precision = 0.9949
- exact recall_global = 0.4220

นี่คือคำตอบตรง ๆ ต่อคำถามว่า ทำไมไม่แยกคู่ที่มั่นใจแน่ ๆ ออกก่อนแล้วค่อยเอาที่เหลือเข้า model

## 4. why this multimodal run was chosen

main run คือ `image_context_r075_h20_s42`
- best model = gb
- feature_count = 41
- test AP = 0.9789
- test AUC = 0.9734
- test F1 = 0.9340
- test precision = 0.9507
- test recall = 0.9179
- confusion matrix = [[3717, 209], [360, 4027]]

ตัวนี้ถูกเลือกเพราะเป็น run ที่ได้ composite score สูงสุดใน suite, ใช้ image context เพิ่มจาก text-only baseline และยังสามารถอธิบายผ่าน feature importance ได้

## 5. why Gradient Boosting was chosen

Gradient Boosting ถูกเลือกเป็น best model ในทุก experiment ของ multimodal suite ไม่ใช่เฉพาะ main run
ข้อดีสำหรับเล่ม:
- เป็น tabular model ที่อธิบายง่ายกว่า deep neural net
- มี feature importance ให้ดูได้
- ให้ AP/AUC สูงกว่า logistic regression และชนะ random forest ใน suite นี้

## 5.1 strict no-leak result that should be written in the report

ถ้าต้องอ้างผลจากสาย strict no-leak ในเล่ม ไม่ควรใช้จุดที่ precision = 1 เป็นผลหลักเพียงอย่างเดียว เพราะจะดู conservative เกินไปและอธิบาย trade-off ได้ไม่ดี

strict point ที่แนะนำให้เขียนคือ
- model = `logreg + sigmoid`
- threshold = `0.50`
- precision = `0.9900`
- recall = `0.8169`
- f1 = `0.8951`
- confusion matrix = `[[461691, 10], [221, 986]]`

เหตุผลที่เลือกจุดนี้:
- recall เท่าเดิมกับ strict เดิม
- precision ไม่ตันที่ `1.0000`
- false positives มีเพียง `10` คู่ จึงยังคุมความผิดพลาดได้ดี
- อธิบายเชิงรายงานได้สมจริงกว่าจุด conservative ที่ `FP = 0`

## 6. tuning rationale

tuning จาก rebuilt experiment ชี้ว่าค่า negative sampling ที่เหมาะคือบริเวณ
- random_neg_ratio ~ 0.75
- hard_neg_ratio ~ 2.0-2.5
main multimodal suite จึงยึด `r075_h20`

## 7. production view

final production metrics:
- final_match_only_precision = 0.9550
- final_match_only_recall = 0.6711
- review_queue = 86,296
- unified_profiles = 19,799

สิ่งที่ควรอธิบายในเล่มคือ trade-off ระหว่าง precision สูงกับ review queue ที่ยังใหญ่ และเหตุผลเชิงระบบว่าทำไมต้องมี review tier
