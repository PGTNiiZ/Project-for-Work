# แผนภาพความสัมพันธ์ของข้อมูลหลัก (Entity-Relationship Diagram)

ไฟล์ภาพหลักอยู่ที่ [data_er_latest.svg](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/fig/data_er_latest.svg)

ภาพนี้สรุปความสัมพันธ์ของตารางข้อมูลหลักที่ใช้จริงใน pipeline ปัจจุบัน โดยเน้น 3 ส่วนหลัก

1. ชุดข้อมูลต้นทางและการทำ normalization
2. ชุดข้อมูลสำหรับ pair building และ feature engineering
3. ชุดข้อมูลปลายทางสำหรับ decision, entity merge และ lead scoring

ตารางหลักที่แสดงในภาพประกอบด้วย

- `all_profiles_cleaned.csv` จำนวน `24,729` แถว
- `location_mapping.csv` จำนวน `4,589` แถว
- `normalized_profiles_with_profile_id.csv` จำนวน `36,807` แถว
- `train_pairs.parquet` จำนวน `10,133,486` แถวใน train split
- `feature_matrix_chunked_*_merged.parquet` โดย train split มี `10,133,486` แถว
- `match_decisions.parquet` จำนวน `2,086,245` แถว
- `review_queue.parquet` จำนวน `86,296` แถว
- `unified_profiles.parquet` จำนวน `19,799` แถว
- `profile_mapping.parquet` จำนวน `36,804` แถว
- `lead_scores.parquet` จำนวน `19,799` แถว

จุดที่สำคัญที่สุดของภาพนี้คือการแยกความหมายของคีย์ 2 ระดับ

- `profile_id` ใช้เป็นคีย์ระดับเอนทิตีต้นทางในบางสายของ pair building และ labeling
- `profile_row_id` เป็นคีย์ระดับระเบียนของ `normalized_profiles_with_profile_id.csv` และเป็นคีย์ที่ถูกใช้ต่อใน full candidate pipeline และ CRM pipeline

ข้อควรระวังในการอ้างอิงรายงานคือ ใน [run_crm_entity_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py) มีการ rename `profile_row_id_a` และ `profile_row_id_b` ให้เป็น `profile_id_a` และ `profile_id_b` ก่อนบันทึกลง `match_decisions.parquet` ดังนั้นคอลัมน์ `profile_id_a` และ `profile_id_b` ใน `match_decisions.parquet` จึงอ้างถึง `profile_row_id` ของชุดข้อมูล normalized ไม่ใช่ `profile_id` แบบคีย์เอนทิตีต้นทาง

ถ้าจะอ้างในบทที่ 3 แนะนำ caption แบบนี้

“รูปที่ 3.x แผนภาพความสัมพันธ์ของตารางข้อมูลหลักใน pipeline ปัจจุบัน แสดงการไหลของข้อมูลจากชุดข้อมูลโปรไฟล์ที่ผ่านการทำความสะอาดและปรับมาตรฐาน ไปสู่ตารางคู่ข้อมูล ตารางคุณลักษณะ ตารางการตัดสินใจ และตาราง unified customer profiles ที่ใช้ในขั้น lead scoring”
