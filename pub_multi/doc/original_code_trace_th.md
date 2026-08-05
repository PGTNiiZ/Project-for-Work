# Original Code Trace

เอกสารนี้ทำขึ้นเพื่อแยกให้ชัดว่า “โค้ดต้นฉบับของโปรเจกต์” อยู่ที่ไหน และ “โค้ดเสริมใน `pub_multi`” ถูกใช้เพื่ออะไร เพื่อไม่ให้รายงานไปอ้างไฟล์สรุปหรือไฟล์วิเคราะห์แทน pipeline ที่ทำจริง

## 1. หลักการใช้อ้างอิงในเล่ม

ถ้าในรายงานต้องอธิบายวิธีดำเนินงาน ควรอ้างจากไฟล์ต้นฉบับของโปรเจกต์ก่อนเสมอ ไม่ควรอ้างจากไฟล์วิเคราะห์ใน `pub_multi` เป็นหลัก เพราะไฟล์ใน `pub_multi` ส่วนหนึ่งเป็นเพียงตัวช่วยรวบรวมผล สร้างตาราง และทำรูปประกอบสำหรับการเขียนรายงาน

หลักง่าย ๆ คือ

- ถ้ากำลังอธิบาย “ระบบทำงานอย่างไร” ให้ใช้อ้างอิงจากไฟล์ต้นฉบับ
- ถ้ากำลังอธิบาย “สรุปผลหรือวิเคราะห์เพิ่มอย่างไร” จึงค่อยใช้อ้างอิงจากไฟล์ใน `pub_multi`

## 2. โค้ดต้นฉบับที่ควรใช้ในรายงาน

| ช่วงงาน | ไฟล์ต้นฉบับที่ควรอ้าง | หน้าที่ |
| --- | --- | --- |
| Data loading / preprocess | `Project-for-Work/clean_data/preprocess_dataset.py` | อ่านข้อมูลดิบและเตรียมข้อมูลตั้งต้น |
| Location normalization | `Project-for-Work/clean_data/location_mapping_pipeline.py` | map และ normalize location |
| Normalized profile DB | `Project-for-Work/image_process/create_normalized_db.py` | สร้างฐานข้อมูลโปรไฟล์ที่ผ่านการ normalize แล้ว |
| Image recovery / image assets | `Project-for-Work/image_process/recover_profile_images.py` | ดึง/กู้รูปโปรไฟล์เพื่อใช้ต่อใน pipeline ด้านภาพ |
| Pair building | `stage8_pair_builder.py` | สร้าง positive pairs, random negatives, hard negatives |
| Feature engineering | `Project-for-Work/train_data/stage9_features_pipeline_chunked.py` | สร้าง pair features แบบ chunked |
| Baseline / leakage-safe experiment | `Project-for-Work/train_data/leakage_safe_experiment/run_experiment.py` | เปรียบเทียบโมเดล baseline ในสาย leakage-safe |
| Main multimodal suite | `Project-for-Work/train_data/stage7_13_multimodal_suite/run_multimodal_suite.py` | สาย train/eval หลักของ multimodal |
| Main training notebook | `Project-for-Work/train_data/stage11_training.ipynb` | notebook ที่ใช้ตรวจและอธิบาย training logic |
| Main training pipeline | `Project-for-Work/train_data/stage10_13_training_pipeline.py` | training/calibration/evaluation pipeline เดิม |
| Full candidate scoring | `Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py` | exact-first, blocking, scoring ทั้ง candidate set |
| CRM / entity pipeline | `Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py` | รวมผลลัพธ์ไปสู่ review queue, unified profiles, lead outputs |
| Preprocess notebook reference | `Project-for-Work/train_data/preprocess_pipeline_backup.ipynb` | notebook อ้างอิงด้าน preprocess ที่คุณทำไว้ |

## 3. ไฟล์ใน `pub_multi/src` คืออะไร

ไฟล์ใน `pub_multi/src` ไม่ใช่ implementation ใหม่ที่ตั้งใจให้แทนโค้ดต้นฉบับ แต่เป็นการคัดลอกไฟล์เดิมมาเก็บรวมไว้เพื่อให้ตรวจงานง่ายขึ้นในแพ็กเกจเดียว ดังนั้นถ้าต้องเขียนรายงานหรืออธิบายวิธีทำ ควรย้อนกลับไปอ้างไฟล์ต้นฉบับตามตารางข้างบนมากกว่า

กล่าวอีกแบบหนึ่งคือ

- `pub_multi/src/*.py` = copy เพื่อรวบรวม
- ไฟล์ต้นฉบับใน `clean_data/`, `image_process/`, `train_data/` = แหล่งอ้างอิงหลักของวิธีทำ

## 4. ไฟล์เสริมใน `pub_multi` ที่เป็นแค่ analysis support

ไฟล์ต่อไปนี้ไม่ควรถูกอธิบายว่าเป็น main pipeline ของโปรเจกต์ แต่ใช้เพื่อสรุปผลและช่วยเขียนรายงาน

| ไฟล์ | บทบาทจริง |
| --- | --- |
| `Project-for-Work/pub_multi/bench_models.py` | benchmark เทียบหลาย model บน main run เดิม |
| `Project-for-Work/pub_multi/tune_top_models.py` | tune เพิ่มเพื่อสรุปในรายงาน |
| `Project-for-Work/pub_multi/bench_strict_models.py` | rerun strict comparison บน artifact เดิม |
| `Project-for-Work/pub_multi/strict_threshold_diag.py` | วิเคราะห์ว่าทำไม strict line ได้ precision = 1 |
| `Project-for-Work/pub_multi/strict_calibration_compare.py` | เทียบ isotonic กับ sigmoid สำหรับ strict report point |

ดังนั้น ถ้าเขียนในเล่ม ควรใช้ถ้อยคำลักษณะนี้

“ผู้วิจัยใช้ไฟล์ต้นฉบับในโฟลเดอร์ `clean_data`, `image_process` และ `train_data` เป็นแหล่งอ้างอิงหลักของขั้นตอนการพัฒนาแบบจำลอง ส่วนไฟล์ในแพ็กเกจ `pub_multi` ถูกใช้เพื่อรวบรวม artifact, สร้างตารางเปรียบเทียบ และจัดทำเอกสารประกอบรายงานเพิ่มเติม”

## 5. ที่มาของการปรับ strict operating point

ส่วนนี้สำคัญมาก เพราะเป็นจุดที่อาจทำให้สับสนว่าเป็น “การสร้างโค้ดใหม่” หรือ “การใช้โค้ดเดิมแล้วอธิบายผลเพิ่ม”

ที่มาจริงคือ

1. โค้ด strict เดิมและผลอ้างอิงดั้งเดิมอยู่ที่
   - `Project-for-Work/train_data/run_stage10_13_training_noleak_strict.ps1`
   - `Project-for-Work/train_data/stage10_13_training_noleak_strict/reports/evaluation_summary.json`

2. ไฟล์ `Project-for-Work/pub_multi/bench_strict_models.py` ใช้ feature set เดิม, exclusion list เดิม, และ merged parquet เดิม เพื่อสรุป comparison ของหลาย model family ใน strict setting เดียวกัน

3. ไฟล์ `Project-for-Work/pub_multi/strict_threshold_diag.py` ไม่ได้สร้าง preprocess ใหม่ แต่ขยายผลจาก strict setting เดิมให้ออกมาเป็น threshold sweep เพื่อดูว่าทำไม precision จึงตันที่ 1

4. ไฟล์ `Project-for-Work/pub_multi/strict_calibration_compare.py` ไม่ได้เปลี่ยนชุดข้อมูลหรือ feature set แต่ใช้ raw scores จาก strict setting เดิมมาเปรียบเทียบ post-hoc calibration แบบ `isotonic` กับ `sigmoid`

ดังนั้น การปรับ strict report point ที่ใช้ในรายงานจึงเป็น “การวิเคราะห์เพิ่มบนฐานของ artifact เดิม” ไม่ใช่การแทนที่ pipeline ดั้งเดิมของโปรเจกต์

## 6. วิธีเขียนในรายงานให้ไม่หลุดจากของเดิม

ถ้าต้องเขียนบทที่ 3 และบทที่ 4 ให้ตรงกับงานจริง ควรใช้หลักดังนี้

- บทที่ 3 อ้างจากไฟล์ต้นฉบับของคุณโดยตรง
- บทที่ 4 สามารถอ้างไฟล์สรุปใน `pub_multi` ได้ แต่ต้องระบุว่าเป็น report artifact หรือ analysis support
- ถ้ามีผลที่ได้จากการวิเคราะห์เพิ่ม เช่น strict report point ใหม่ ควรเขียนว่าเป็น “ผลวิเคราะห์เพิ่มเติมเพื่อใช้ในการตีความและเลือก operating point ที่เหมาะสมกว่า” ไม่ใช่บอกว่าเป็น training pipeline หลักอันใหม่

## 7. ข้อเสนอสำหรับเล่ม

ถ้าต้องการให้เล่มดูสะอาดและไม่ทำให้คนอ่านสับสน ควรใช้โครงนี้

- วิธีทำหลัก: อ้าง `clean_data/`, `image_process/`, `train_data/`
- ตารางและกราฟสรุป: อ้าง `pub_multi/res`, `pub_multi/fig`, `pub_multi/doc`
- ถ้าจำเป็นต้องพูดถึงสคริปต์วิเคราะห์ใหม่: เรียกว่า “auxiliary analysis scripts for reporting”
