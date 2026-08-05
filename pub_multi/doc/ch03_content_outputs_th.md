# สรุปเนื้อหาและผลลัพธ์ที่ควรอยู่ในบทที่ 3

เอกสารนี้สรุปว่าในบทที่ 3 ควรเขียนอะไรบ้าง และถ้าจะใส่ผลลัพธ์ในบทนี้ควรใส่ผลลัพธ์ระดับใด โดยยึดหลักว่า บทที่ 3 ตอบคำถามว่า “ทำอย่างไร” และ “แต่ละขั้นให้ artifact อะไรออกมา” ส่วนผลเชิงประสิทธิภาพละเอียด เช่น confusion matrix, ROC, PR curve, threshold sweep และการเทียบ metric แบบเต็ม ควรย้ายไปอยู่บทที่ 4

## ตารางที่ 1 เนื้อหาที่ควรมีในบทที่ 3

| หัวข้อในบทที่ 3 | สิ่งที่ควรอธิบาย | โค้ดต้นฉบับที่อ้าง | ผลลัพธ์หรือ artifact ที่ควรพูดถึง | รูปที่ควรใส่ |
| --- | --- | --- | --- | --- |
| `3.1 Business Understanding` | ปัญหา data silos ใน CRM, เหตุผลที่โจทย์ถูกนิยามเป็น identity resolution + lead scoring, เหตุผลที่ต้องมี retrieval ก่อน classification | ภาพรวมของระบบทั้งหมด | นิยาม output สุดท้ายเป็น `MATCH/REVIEW/NO_MATCH`, unified profiles, lead tiers | รูป 3.1, 3.2 |
| `3.2 Data Understanding` | แหล่งข้อมูล LinkSocial, แพลตฟอร์มที่ใช้, ฟิลด์หลักที่เกี่ยวข้องกับ identity และภาพรวมโครงสร้างข้อมูล | [preprocess_dataset.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/clean_data/preprocess_dataset.py) | `all_profiles_cleaned.csv`, `normalized_profiles_with_profile_id.csv` | รูป 3.3 |
| `3.3 Data Preparation` | การ clean ข้อความ, สร้าง `profile_id/profile_row_id`, ทำ location normalization, เตรียมข้อมูลภาพ, สร้าง positive/random negative/hard negative pairs, สร้าง feature matrix แบบ chunked | [preprocess_dataset.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/clean_data/preprocess_dataset.py), [location_mapping_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/clean_data/location_mapping_pipeline.py), [create_normalized_db.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/image_process/create_normalized_db.py), [stage8_pair_builder.py](/d:/66070260-Year3_Term2/Project1/Code/stage8_pair_builder.py), [stage9_features_pipeline_chunked.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage9_features_pipeline_chunked.py) | `all_profiles_cleaned.csv`, `location_mapping.csv`, normalized image DB, `labeled_pairs.parquet`, `train/val/test feature parquet` | รูป 3.4, 3.5 |
| `3.4 Modeling` | classical line, multimodal suite, neural reference, วิธีเลือกโมเดลหลัก, calibration, threshold selection | [run_experiment.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/leakage_safe_experiment/run_experiment.py), [run_multimodal_suite.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/run_multimodal_suite.py), [stage10_13_training_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage10_13_training_pipeline.py) | `leaderboard.csv`, `suite_report.json`, best model files, calibrator files, score files | รูป 3.6, 3.7 |
| `3.5 Evaluation` | อธิบายว่า evaluation มี 3 ชั้นคือ retrieval, model, production; อธิบาย metric ที่ใช้และเหตุผลของแต่ละ metric | [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py), [run_experiment.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/leakage_safe_experiment/run_experiment.py), [stage10_13_training_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage10_13_training_pipeline.py) | `evaluation_summary.json`, `experiment_report.json`, `full_pipeline_report.json` | รูป 3.8 |
| `3.6 Deployment` | exact-first + blocking production flow, candidate scoring, decision tiers, review queue, entity merge, lead scoring | [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py), [run_crm_entity_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py) | `candidate_scores_*.parquet`, `final_decisions.parquet`, `review_queue.parquet`, `unified_profiles.parquet`, `lead_scores.parquet` | รูป 3.9, 3.10 |
| `3.7 Summary` | สรุปว่าระบบนี้เป็น pipeline ครบวงจร ไม่ใช่โมเดลเดี่ยว | ใช้ภาพรวมทุกส่วน | สรุป flow จาก raw profiles ไปถึง CRM output | ไม่จำเป็นต้องมีรูปเพิ่ม |

## ตารางที่ 2 ผลลัพธ์ที่อ้างได้ในบทที่ 3

| ขั้นตอน | ผลลัพธ์ที่เกิดขึ้นจริง | ค่าจริงที่อ้างได้ | ไฟล์อ้างอิง |
| --- | --- | --- | --- |
| นำเข้าและ clean ข้อมูล | สร้างตารางโปรไฟล์ที่ผ่านการทำความสะอาดแล้ว | `24,729` แถวใน `all_profiles_cleaned.csv` | [all_profiles_cleaned.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data/processed/all_profiles_cleaned.csv) |
| ปรับมาตรฐานข้อมูลสำหรับ pipeline หลัก | สร้างชุดข้อมูลโปรไฟล์ที่มี `profile_id/profile_row_id` สำหรับใช้ตลอด pipeline | `36,807` แถวใน `normalized_profiles_with_profile_id.csv` | [normalized_profiles_with_profile_id.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv) |
| กระจายข้อมูลตามแพลตฟอร์มในชุดหลัก | ใช้ตรวจสอบว่าข้อมูลข้ามแพลตฟอร์มพร้อมสำหรับ pair generation | Twitter `13,960`, Google+ `11,890`, Instagram `10,957` | [normalized_profiles_with_profile_id.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv) |
| สร้างคู่ข้อมูลสำหรับ train | สร้าง positive, random negatives และ hard negatives | total `24,663,633`, positive `4,060,200`, negative `20,603,433`, random negatives `20,301,000`, hard negatives `302,433` | [pair_stats.json](/d:/66070260-Year3_Term2/Project1/Code/pair_stats.json) |
| multimodal model development | เปรียบเทียบ `text_attr_hybrid`, `image_stats`, `image_context` | best run คือ `image_context_r075_h20_s42`, feature count `41` | [suite_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/reports/suite_report.json) |
| selected model result | ใช้เป็นผลลัพธ์สรุประดับ model ในบทที่ 3 ได้แบบย่อ | test AP `0.9789`, test AUC `0.9734`, test F1 `0.9340` | [suite_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/reports/suite_report.json) |
| retrieval / blocking | สร้าง exact matches และ candidate pairs สำหรับ production pipeline | all cross-platform pairs `449,149,239`, exact pairs `12,403`, candidate pairs `2,073,842` | [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) |
| blocking coverage | ใช้ยืนยันว่า retrieval ไม่ได้คัดคู่แบบสุ่ม แต่ยังครอบคลุม ground truth ได้สูง | ground-truth positive pairs `29,243`, coverage `88.67%` | [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) |
| production decision | ใช้แสดงว่าระบบมี operating point สำหรับใช้งานจริง | match threshold `0.98`, review threshold `0.95` | [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) |
| production quality | ถ้าจะใส่ผลในบทที่ 3 ให้ใส่ระดับ workflow แบบสั้น ไม่ต้องลง confusion matrix | final match-only precision `0.9550`, recall `0.6711` | [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) |
| CRM decision output | สรุปผลว่าหลัง scoring แล้วระบบสร้างอะไรออกมาบ้าง | match decisions `2,086,245`, `MATCH = 20,549`, `REVIEW = 86,296`, `NO_MATCH = 1,979,400` | [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) |
| entity merge | แสดงว่าผลลัพธ์ถูกยกระดับจากคู่ไปเป็น customer entity | unified profiles `19,799` | [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) |
| lead scoring | แสดงว่าระบบปลายทางให้คะแนนลูกค้าได้จริง | lead scores `19,799`, `HOT = 4,937`, `WARM = 9,039`, `COLD = 5,823` | [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) |

## ตารางที่ 3 สิ่งที่ใส่ได้ในบทที่ 3 กับสิ่งที่ควรย้ายไปบทที่ 4

| ใส่ในบทที่ 3 ได้ | ควรย้ายไปบทที่ 4 |
| --- | --- |
| โครง pipeline ทั้งระบบ | confusion matrix ของแต่ละโมเดล |
| แหล่งข้อมูลและจำนวนข้อมูลหลัก | ROC curve |
| วิธี clean / normalize / pair building / feature engineering | PR curve |
| โมเดลที่ทดลองและเหตุผลที่เลือก final model | threshold sweep แบบละเอียด |
| exact-first, blocking keys, candidate generation flow | ตาราง metric เปรียบเทียบหลาย operating points |
| artifact ที่เกิดขึ้นในแต่ละขั้น เช่น cleaned profiles, labeled pairs, feature matrices, predicted matches, review queue | error analysis แบบ TP/FP/FN ละเอียด |
| ค่าระดับ workflow เช่น all pairs, exact pairs, candidate pairs, coverage, review queue, unified profiles | calibration plot และการวิเคราะห์ score distribution |

## รูปแบบที่แนะนำสำหรับการเขียนในเล่ม

ถ้าจะเขียนบทที่ 3 ให้กระชับและชัด ควรใช้รูปแบบนี้

1. อธิบายว่าขั้นนั้นทำอะไรและทำไมต้องทำ
2. อ้างโค้ดต้นฉบับที่รับผิดชอบขั้นนั้น
3. บอกว่า artifact หรือไฟล์อะไรถูกสร้างขึ้น
4. ถ้าจำเป็นค่อยใส่ตัวเลขสรุประดับระบบ เช่น จำนวนแถว จำนวนคู่ หรือ coverage
5. อย่าใส่ผลเชิงแข่งขันของโมเดลละเอียดเกินไปในบทนี้

## ย่อหน้าสรุปพร้อมใช้

ในบทที่ 3 ควรนำเสนอผลลัพธ์ในลักษณะของผลลัพธ์เชิงกระบวนการมากกว่าผลลัพธ์เชิงประสิทธิภาพ กล่าวคือ ควรแสดงให้เห็นว่าหลังจากแต่ละขั้นตอนของ pipeline ระบบสร้างข้อมูลหรือ artifact อะไรขึ้นมาบ้าง เช่น ชุดข้อมูลที่ผ่านการทำความสะอาดแล้ว ชุดข้อมูลที่ปรับมาตรฐานแล้ว ชุดคู่ข้อมูลสำหรับการฝึก ชุดคุณลักษณะสำหรับ train/validation/test ชุดคะแนนของ candidate pairs ชุดผลการตัดสินใจระดับ `MATCH/REVIEW/NO_MATCH` ตลอดจน unified profiles และ lead scores ในระดับ CRM ส่วนผลประเมินเชิงตัวเลขที่ใช้พิสูจน์ประสิทธิภาพของแบบจำลองโดยละเอียด เช่น confusion matrix, ROC-AUC, Precision-Recall curve และการวิเคราะห์ threshold ควรย้ายไปนำเสนออย่างเป็นระบบในบทที่ 4
