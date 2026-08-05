# 3.6 การนำไปใช้งาน (Deployment)

ขั้น Deployment ของงานวิจัยนี้ไม่ได้หมายถึงการนำแบบจำลองไปเปิดเป็นบริการเพียงอย่างเดียว แต่หมายถึงการทำให้ผลลัพธ์จากขั้น retrieval และ scoring กลายเป็น workflow ที่ใช้งานได้จริงในระบบ CRM โดยเชื่อมต่อจาก candidate scoring ไปสู่การตัดสินใจเชิงปฏิบัติการ การตรวจสอบโดยมนุษย์ การรวมเอนทิตี และการสร้างคะแนนลูกค้าเป้าหมาย กระบวนการส่วนนี้อ้างอิงจาก [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py) และ [run_crm_entity_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py) ซึ่งเป็นสาย deployment หลักของระบบในปัจจุบัน

## 3.6.1 Full Candidate Scoring Pipeline

การนำแบบจำลองไปใช้จริงเริ่มจากการโหลดข้อมูลโปรไฟล์จาก [normalized_profiles_with_profile_id.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv) แล้วสร้าง search space ของคู่ข้ามแพลตฟอร์มในเชิงนิยามก่อน จากนั้นระบบใช้กลยุทธ์ `exact-first` เพื่อแยกคู่ที่มีหลักฐานตรงกันชัดเจนออกมาก่อน เช่น คู่ที่ `userName` ตรงกันหรือ `externalUrl` ตรงกัน แล้วจึงใช้ deterministic blocking ด้วย `username_prefix3`, `fullname_prefix3` และ `external_domain` เพื่อคัด candidate pairs สำหรับให้โมเดลประเมิน ขั้นตอนนี้ทำให้ระบบไม่ต้อง score ทุกคู่ใน search space แต่ score เฉพาะคู่ที่ retrieval ผ่านเข้ามาแล้ว

หลังได้ candidate set ระบบจะโหลด artifact ของ best run จาก multimodal suite ได้แก่ `best_model.pkl`, `scaler.pkl`, `calibrator.pkl` และ `feature_cols.pkl` ของ run `image_context_r075_h20_s42` แล้วคำนวณ feature และ score candidate pairs แบบเป็นช่วงย่อยหรือ chunked เพื่อให้รองรับข้อมูลระดับหลักล้านคู่ได้โดยไม่ใช้หน่วยความจำมากเกินไป ผลลัพธ์ของแต่ละช่วงถูกบันทึกเป็นไฟล์ `candidate_scores_*.parquet` และมีการสรุปคู่คะแนนสูงสุดออกเป็น [top_5000_predictions.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/top_5000_predictions.csv) สำหรับการตรวจสอบเชิงคุณภาพเพิ่มเติม

จากรายงาน [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) ระบบ score candidate pairs ทั้งหมด `2,073,842` คู่ โดยใช้ feature `41` ตัวจาก run หลัก และให้ค่า `candidate_avg_precision = 0.6017`, `candidate_roc_auc = 0.9600` ภายใน candidate set พร้อม `precision_at_100 = 0.98`, `precision_at_500 = 0.98`, `precision_at_1000 = 0.982` และ `precision_at_5000 = 0.9472` ตัวเลขเหล่านี้สะท้อนว่าคู่ที่ได้คะแนนสูงสุดมีคุณภาพดีมาก แม้ว่าหากใช้ threshold ภายใน candidate set แบบไม่ควบคุมในระดับ production จะยังมี false positives จำนวนมากก็ตาม

ตารางที่ 3.13 ผลลัพธ์หลักของ full candidate scoring pipeline

| รายการ | ค่า |
| --- | ---: |
| Candidate pairs scored | 2,073,842 |
| Feature count of deployed model | 41 |
| Candidate average precision | 0.6017 |
| Candidate ROC-AUC | 0.9600 |
| Precision@100 | 0.9800 |
| Precision@500 | 0.9800 |
| Precision@1000 | 0.9820 |
| Precision@5000 | 0.9472 |

## 3.6.2 Decision Tiers และ Human-in-the-Loop

หลังจากได้คะแนนของ candidate pairs แล้ว ระบบจะไม่ตัดสินใจแบบสองคลาสโดยตรง แต่แปลงคะแนนให้กลายเป็น decision tiers ที่พร้อมใช้งานจริง โดยใช้ `match_threshold = 0.98` และ `review_threshold = 0.95` สำหรับสาย production คู่ exact matches จะถูกกำหนดให้เป็นการตัดสินใจแบบอัตโนมัติทันทีและให้คะแนนเท่ากับ `1.0` ส่วนคู่ที่ได้คะแนนสูงกว่า `0.98` จะถูกจัดเป็น `MATCH` คู่ที่อยู่ระหว่าง `0.95` ถึงต่ำกว่า `0.98` จะถูกจัดเป็น `REVIEW` และคู่ที่ต่ำกว่านั้นจะถูกจัดเป็น `NO_MATCH`

การออกแบบเช่นนี้มีเหตุผลเชิงปฏิบัติการชัดเจน กล่าวคือ ระบบไม่พยายาม auto-merge ทุกคู่ที่คะแนนสูงพอประมาณ แต่เลือกจะยอมให้ recall ในชั้น `MATCH` ต่ำลงบางส่วนเพื่อรักษา precision ของคู่ที่ถูกยืนยันโดยอัตโนมัติให้อยู่ในระดับสูง และผลักคู่ที่ยังไม่แน่ชัดไปยัง review queue แทน ในรายงานของ production threshold พบว่า exact tier มี precision `0.9949`, model-based `MATCH` tier มี precision `0.8942` และ `REVIEW` tier มี precision เพียง `0.0471` ซึ่งยืนยันว่าการแยกชั้น `REVIEW` มีความจำเป็น เพราะถ้าระบบนำคู่กลุ่มนี้ไป merge อัตโนมัติจะเพิ่มความเสี่ยงของ false positives อย่างมาก

ใน CRM entity pipeline ระบบจะสร้างตาราง `match_decisions.parquet` เพื่อบันทึกการตัดสินใจทั้งหมด พร้อมระบุ `decision_id`, `profile_id_a`, `profile_id_b`, `score`, `decision`, `decision_source`, `review_status`, `reviewed_by` และ `reviewed_at` จากนั้นจะสร้าง `review_queue.parquet` สำหรับคู่ที่ถูกจัดอยู่ในช่วง `REVIEW` โดยดึง `profile_snapshot_a`, `profile_snapshot_b` และ `key_features` มาพร้อมกัน เพื่อให้มนุษย์สามารถตรวจสอบได้อย่างมีบริบท ไม่ใช่เพียงเห็นคะแนนตัวเลขอย่างเดียว การมี `decision_source` เช่น `AUTO_EXACT`, `AUTO_HIGH`, `AUTO_REVIEW` และ `AUTO_LOW` ยังช่วยให้ระบบสามารถ audit และย้อนตรวจที่มาของการตัดสินใจได้ชัดเจน

ตารางที่ 3.14 โครงสร้างการตัดสินใจเชิงปฏิบัติการในสาย deployment

| ระดับการตัดสินใจ | เงื่อนไข | วัตถุประสงค์ |
| --- | --- | --- |
| `AUTO_EXACT` | exact match rules | ยืนยันคู่ที่มีหลักฐานตรงกันชัดเจน |
| `AUTO_HIGH` | score ≥ 0.98 | auto-match เฉพาะคู่ที่มั่นใจสูงมาก |
| `AUTO_REVIEW` | 0.95 ≤ score < 0.98 | ส่งให้มนุษย์ตรวจ |
| `AUTO_LOW` | score < 0.95 | ปฏิเสธโดยอัตโนมัติ |

## 3.6.3 การรวมเอนทิตีและการสร้าง Customer 360 View

หลังจากระบบได้ผลการตัดสินใจระดับคู่แล้ว ขั้นถัดไปคือการรวมโปรไฟล์ที่ได้รับการยืนยันให้เป็นเอนทิตีระดับลูกค้า โดยใช้กลไก `union-find` ในฟังก์ชัน `build_unified_tables()` ของ [run_crm_entity_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/run_crm_entity_pipeline.py) เงื่อนไขที่ใช้สำหรับรวมเอนทิตีคือคู่ที่มี `decision = MATCH` และมี `decision_source` อยู่ในกลุ่ม `AUTO_EXACT`, `AUTO_HIGH` หรือ `HUMAN` เท่านั้น แนวทางนี้ทำให้ระบบสามารถรองรับความสัมพันธ์เชิงถ่ายโอนหรือ transitive closure ได้ เช่น หาก A เชื่อมกับ B และ B เชื่อมกับ C ระบบจะรวม A, B และ C เป็น unified profile เดียวกัน

ผลจากขั้นตอนนี้ถูกบันทึกเป็น `unified_profiles.parquet` และ `profile_mapping.parquet` โดย `unified_profiles` ทำหน้าที่เก็บข้อมูลระดับเอนทิตี เช่น `unified_id`, `canonical_name`, `platforms`, `all_usernames`, `merged_bio`, `location`, `all_urls`, `platform_count` และ `merge_confidence` ส่วน `profile_mapping` ใช้เก็บความสัมพันธ์ระหว่างโปรไฟล์ต้นทางกับเอนทิตีปลายทาง ทำให้สามารถย้อนกลับจาก customer 360 view ไปยังโปรไฟล์ต้นทางได้ในทุกกรณี

จากรายงาน [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) ระบบสร้าง `unified_profiles` ได้ `19,799` รายการ และ `profile_mapping` `36,804` รายการ โดย `merge_source_breakdown` แสดงว่ามี unified profiles ที่เกิดจาก `AUTO` จำนวน `14,105` รายการ และจาก `EXACT` จำนวน `5,694` รายการ ตัวเลขนี้แสดงให้เห็นว่าระบบไม่ได้ยึด exact matching เป็นวิธีหลักเพียงอย่างเดียว แต่ใช้ exact เป็นฐานความเชื่อมั่นสูง แล้วต่อยอดด้วย model-based matching เพื่อรวมข้อมูลลูกค้าให้ครบขึ้นในระดับเอนทิตี

## 3.6.4 การสร้าง Lead Scoring และผลลัพธ์ปลายทางของระบบ

ขั้นสุดท้ายของ deployment คือการคำนวณ lead score ในระดับ unified profile ผ่านฟังก์ชัน `build_lead_scores()` ซึ่งทำงานบน unified entity ที่รวมเรียบร้อยแล้ว แนวทางที่ใช้ในปัจจุบันเป็น heuristic lead scoring โดยพิจารณาจาก 3 องค์ประกอบหลัก คือ `completeness_score` สะท้อนความครบถ้วนของข้อมูลในเอนทิตีนั้น `platform_score` สะท้อนจำนวนแพลตฟอร์มที่เชื่อมโยงได้ และ `engagement_score` สะท้อนสัญญาณจาก mentions, hashtags และ URLs ในข้อมูลข้อความ จากนั้นจึงรวมเป็น `lead_score` และแบ่งระดับออกเป็น `HOT`, `WARM` และ `COLD`

ผลจาก [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) ระบุว่าระบบสร้าง `lead_scores` ได้ `19,799` รายการ เท่ากับจำนวน unified profiles ทั้งหมด โดยมีการกระจายของ tier เป็น `HOT = 4,937`, `WARM = 9,039` และ `COLD = 5,823` เมื่อพิจารณาร่วมกับ decision breakdown ที่ได้ `MATCH = 20,549`, `REVIEW = 86,296` และ `NO_MATCH = 1,979,400` จะเห็นว่าระบบที่พัฒนาขึ้นไม่ได้หยุดอยู่ที่การทำนายว่าคู่ใดควรถูกเชื่อมโยงกันเท่านั้น แต่สามารถเปลี่ยนผลลัพธ์ดังกล่าวไปสู่มุมมองลูกค้าแบบรวมและการจัดลำดับความสำคัญของลูกค้าเป้าหมายเพื่อใช้งานใน CRM ได้ต่อเนื่อง

ตารางที่ 3.15 ผลลัพธ์ปลายทางของ deployment pipeline

| รายการ | ค่า |
| --- | ---: |
| Match decisions | 2,086,245 |
| Review queue | 86,296 |
| Unified profiles | 19,799 |
| Profile mapping | 36,804 |
| Lead scores | 19,799 |
| MATCH decisions | 20,549 |
| REVIEW decisions | 86,296 |
| NO_MATCH decisions | 1,979,400 |
| HOT tier | 4,937 |
| WARM tier | 9,039 |
| COLD tier | 5,823 |

โดยสรุป ขั้น Deployment ของงานนี้ทำหน้าที่เปลี่ยนแบบจำลองจากการเป็นเพียงเครื่องมือให้คะแนนคู่ข้อมูล ไปสู่ระบบตัดสินใจที่ใช้งานได้จริงใน CRM ผ่านการผสาน retrieval, scoring, decision tiers, review queue, entity merge และ lead scoring เข้าด้วยกันอย่างเป็นระบบ จุดสำคัญของขั้นนี้คือการรักษาสมดุลระหว่างความแม่นยำของการจับคู่กับความปลอดภัยเชิงปฏิบัติการ กล่าวคือ คู่ที่มั่นใจสูงจะถูกยืนยันโดยอัตโนมัติ ขณะที่คู่ที่ยังไม่ชัดเจนจะถูกส่งต่อให้มนุษย์ตรวจสอบก่อน รวมถึงผลลัพธ์สุดท้ายจะถูกยกระดับเป็น unified customer profiles และ lead tiers ที่พร้อมนำไปใช้ในกระบวนการธุรกิจต่อไป
