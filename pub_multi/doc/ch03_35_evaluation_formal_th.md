# 3.5 การประเมินผล (Evaluation)

การประเมินผลในงานวิจัยนี้ถูกออกแบบให้สอดคล้องกับโครงสร้างของระบบทั้งเส้น ไม่ได้ประเมินเฉพาะความแม่นยำของแบบจำลองเพียงอย่างเดียว แต่แยกการประเมินออกเป็นหลายระดับเพื่อให้ตอบคำถามได้ครบว่า ขั้น retrieval ทำให้คู่จริงหลุดออกจากระบบหรือไม่ แบบจำลองสามารถแยกคู่ที่เป็นบุคคลเดียวกันกับคู่ที่ไม่ใช่ได้ดีเพียงใด คะแนนหลังการทำ calibration มีความน่าเชื่อถือเพียงใด และเมื่อนำระบบไปใช้กับ workflow จริงแล้วสามารถควบคุมผลลัพธ์ปลายทางได้หรือไม่ เหตุผลที่ต้องแยกการประเมินเป็นหลายชั้นเช่นนี้ เนื่องจากงาน Identity Resolution สำหรับระบบ CRM เป็นงานที่ผลลัพธ์สุดท้ายไม่ได้ขึ้นกับ classifier เพียงตัวเดียว แต่ขึ้นกับการทำงานร่วมกันของขั้นคัด candidate pairs ขั้นให้คะแนน และขั้นตัดสินใจเชิงปฏิบัติการ

## 3.5.1 กรอบการประเมินผลและตัวชี้วัดที่ใช้

ผู้วิจัยแบ่งการประเมินผลออกเป็น 4 ระดับ ได้แก่ การประเมินขั้น retrieval การประเมินขั้นแบบจำลอง การประเมินผลของ calibration และการประเมินในระดับ production workflow โดยการประเมินขั้น retrieval ใช้เพื่อตรวจว่าระบบ exact-first และ blocking สามารถลดจำนวนคู่เปรียบเทียบจาก search space ขนาดใหญ่ให้เหลือระดับที่คำนวณได้จริง พร้อมทั้งยังคงครอบคลุมคู่จริงไว้ได้มากเพียงใด การประเมินขั้นแบบจำลองใช้เพื่อตรวจว่าเมื่อคู่ข้อมูลผ่านเข้าสู่ candidate set แล้ว แบบจำลองสามารถแยกคู่ MATCH กับ NO_MATCH ได้ดีเพียงใด การประเมิน calibration ใช้เพื่อตรวจว่าคะแนนความน่าจะเป็นที่โมเดลให้มาสามารถนำไปใช้ตั้ง threshold สำหรับการใช้งานจริงได้ดีหรือไม่ ส่วนการประเมินในระดับ production workflow ใช้เพื่อตรวจว่าหลังเลือก threshold แล้ว ระบบให้จำนวน `MATCH`, `REVIEW` และ `NO_MATCH` อยู่ในระดับที่เหมาะสมต่อการใช้งานใน CRM หรือไม่

เมตริกหลักที่ใช้ในงานนี้ได้แก่ `Precision`, `Recall`, `F1-score`, `ROC-AUC`, `Average Precision`, `Confusion Matrix` และ `Classification Report` โดยเลือกใช้เมตริกเหล่านี้เนื่องจากปัญหาในงานมี class imbalance สูง โดยเฉพาะเมื่อพิจารณาในระดับ candidate scoring และ full search space ทำให้ `accuracy` เพียงอย่างเดียวไม่สามารถสะท้อนคุณภาพของระบบได้ดีพอ หากระบบทำนายว่าเกือบทุกคู่เป็น `NO_MATCH` ก็อาจได้ accuracy สูง แต่ไม่ได้ช่วยให้ระบบเชื่อมโยงลูกค้าได้จริง ในทางกลับกัน `Precision` และ `Recall` ช่วยให้เห็น trade-off ระหว่างการจับคู่ผิดกับการพลาดคู่จริงได้ชัดเจนกว่า ขณะที่ `Average Precision` เหมาะกับงานที่มีลักษณะเป็น ranking ภายใต้ class imbalance และ `ROC-AUC` ใช้ตรวจความสามารถของโมเดลในการแยกคู่บวกกับคู่ลบโดยรวมโดยไม่อิง threshold ค่าเดียว

ตารางที่ 3.9 สรุประดับการประเมินผลและตัวชี้วัดหลักที่ใช้ในงานวิจัย

| ระดับการประเมิน | คำถามที่ต้องการตอบ | ตัวชี้วัดหลัก |
| --- | --- | --- |
| Retrieval evaluation | retrieval ทำให้คู่จริงหลุดไปเท่าใด และลด search space ได้มากเพียงใด | exact matches, candidate pairs, search-space reduction, ground-truth coverage |
| Model evaluation | แบบจำลองแยก MATCH กับ NO_MATCH ได้ดีเพียงใด | Precision, Recall, F1-score, ROC-AUC, Average Precision, Confusion Matrix |
| Calibration evaluation | คะแนนหลัง calibration สื่อความมั่นใจได้ดีขึ้นหรือไม่ | ECE ก่อนและหลัง calibration, threshold stability |
| Production evaluation | threshold ที่เลือกให้ผลลัพธ์ปลายทางเหมาะกับการใช้งานจริงหรือไม่ | `MATCH`, `REVIEW`, `NO_MATCH`, review queue size, final precision/recall, unified profiles |

## 3.5.2 การประเมินขั้น Retrieval และ Candidate Generation

การประเมินขั้น retrieval อ้างอิงจาก [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py) และรายงาน [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) โดยเริ่มจากการคำนวณจำนวนคู่ข้ามแพลตฟอร์มทั้งหมดของโปรไฟล์ที่ผ่านการตรวจสอบคีย์อ้างอิงแล้วจำนวน `36,804` โปรไฟล์ ซึ่งประกอบด้วย Twitter `13,959` โปรไฟล์ Google+ `11,889` โปรไฟล์ และ Instagram `10,956` โปรไฟล์ เมื่อนับเฉพาะคู่แบบ cross-platform จะได้ search space ทั้งหมด `449,149,239` คู่ และมี ground-truth positive pairs จำนวน `29,243` คู่

ใน pipeline หลัก ขั้น retrieval ใช้กลยุทธ์ `exact-first` ร่วมกับ deterministic blocking โดย exact rules ใช้ดึงคู่ที่ `userName` ตรงกันหรือ `externalUrl` ตรงกันออกมาก่อน จากนั้นใช้ blocking keys ได้แก่ `username_prefix3`, `fullname_prefix3` และ `external_domain` เพื่อคัดกรองคู่ที่เหลือเข้าสู่ candidate set ผลลัพธ์ที่ได้คือ exact matches จำนวน `12,403` คู่ ซึ่งในจำนวนนี้เป็นคู่จริง `12,340` คู่ คิดเป็น precision `0.9949` และ recall ระดับ global `0.4220` หลังจากนั้นระบบสร้าง candidate pairs สำหรับเข้าสู่โมเดลได้อีก `2,073,842` คู่ โดยมาจาก `username_prefix3` จำนวน `1,026,466` คู่ `fullname_prefix3` จำนวน `1,041,315` คู่ และ `external_domain` จำนวน `6,061` คู่

เมื่อรวมผลของ exact และ candidate coverage เข้าด้วยกัน ระบบยังคงครอบคลุม ground-truth pairs ได้ `25,931` คู่ หรือ `88.67%` ของคู่จริงทั้งหมด พร้อมลด search space ลง `99.54%` เมื่อเทียบกับคู่ข้ามแพลตฟอร์มทั้งหมด ตัวเลขนี้แสดงให้เห็นว่า retrieval ของระบบลดภาระการคำนวณได้อย่างมาก ขณะเดียวกันยังรักษาคู่จริงไว้ส่วนใหญ่ได้ จึงเป็นเหตุผลว่าทำไมงานนี้จึงต้องวัด retrieval quality แยกจาก model quality อย่างชัดเจน หาก retrieval ทำให้คู่จริงหลุดไปตั้งแต่ต้น ต่อให้แบบจำลองมีประสิทธิภาพสูงเพียงใดก็ไม่สามารถกู้คู่ที่หายไปกลับมาได้

ตารางที่ 3.10 ผลการประเมินขั้น retrieval และ candidate generation

| รายการ | ค่า |
| --- | ---: |
| Valid profiles used in full pipeline | 36,804 |
| All cross-platform pairs | 449,149,239 |
| Ground-truth positive pairs | 29,243 |
| Exact match pairs | 12,403 |
| Exact true positives | 12,340 |
| Exact precision | 0.9949 |
| Exact recall_global | 0.4220 |
| Candidate pairs for model | 2,073,842 |
| Search-space reduction | 99.54% |
| Ground-truth pairs covered (exact + candidates) | 25,931 |
| Ground-truth coverage | 88.67% |

## 3.5.3 การประเมินแบบจำลองเชิงการจำแนก

การประเมินในระดับแบบจำลองไม่ได้ยึดเพียงผลจากการทดลองชุดเดียว แต่ใช้หลายสายการทดลองประกอบกันเพื่อให้ข้อสรุปมีความน่าเชื่อถือ สายแรกคือ `leakage-safe classical line` ใน [run_experiment.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/leakage_safe_experiment/run_experiment.py) ซึ่งใช้ feature จำนวน `22` ตัว และเปรียบเทียบ `Logistic Regression`, `Random Forest` และ `Gradient Boosting` ภายใต้ split ที่ควบคุม component overlap ให้เป็นศูนย์ ผลการทดลองใน [experiment_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/leakage_safe_experiment/reports/experiment_report.json) ระบุว่า `Gradient Boosting` เป็นโมเดลที่ดีที่สุดของสายนี้ โดยมีค่า `test_precision = 0.9657`, `test_recall = 0.9038`, `test_f1 = 0.9337`, `test_avg_precision = 0.9703` และ `test_roc_auc = 0.9772` พร้อม confusion matrix `[[9164, 141], [422, 3965]]`

สายที่สองคือ `multimodal suite` ใน [run_multimodal_suite.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/run_multimodal_suite.py) ซึ่งใช้เพื่อตรวจว่าการเพิ่ม feature ฝั่งภาพและบริบทภาพช่วยเพิ่มประสิทธิภาพจริงหรือไม่ โดยมีการเปรียบเทียบ 3 run ได้แก่ `text_attr_hybrid`, `image_stats` และ `image_context` ภายใต้เงื่อนไขเดียวกัน ผลใน [suite_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/reports/suite_report.json) ชี้ว่า run ที่ดีที่สุดคือ `image_context_r075_h20_s42` ซึ่งใช้ feature `41` ตัว และได้ค่า `test_precision = 0.9507`, `test_recall = 0.9179`, `test_f1 = 0.9340`, `test_avg_precision = 0.9789` และ `test_roc_auc = 0.9734` พร้อม confusion matrix `[[3717, 209], [360, 4027]]` ผลลัพธ์ดังกล่าวทำให้ผู้วิจัยเลือกสาย `image_context` เป็น main line ของระบบ เนื่องจากให้สมดุลที่ดีที่สุดระหว่างความแม่นยำ ความสามารถในการจัดอันดับ และความพร้อมในการใช้งานจริง

สายที่สามคือ `strict no-leak reference` ใน [stage10_13_training_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage10_13_training_pipeline.py) และรายงาน [evaluation_summary.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage10_13_training_noleak_strict/reports/evaluation_summary.json) ซึ่งใช้เพื่อตรวจว่าระบบยังคงให้ผลที่สมเหตุสมผลหรือไม่เมื่อ feature บางกลุ่มที่เสี่ยงต่อ leakage ถูกตัดออก ในสายนี้ `IdentityMLP` ให้ค่า `test_precision = 0.8427`, `test_recall = 0.8169`, `test_f1 = 0.8296`, `test_avg_precision = 0.7344` และ `test_roc_auc = 0.9960` พร้อม confusion matrix `[[461517, 184], [221, 986]]` แม้ผลจะต่ำกว่าสาย leakage-safe และ multimodal อย่างชัดเจน แต่ชุดนี้ทำหน้าที่เป็น reference สำคัญในการอภิปรายความแข็งแรงของระบบภายใต้เงื่อนไขที่เข้มงวดกว่า

ตารางที่ 3.11 ผลการประเมินแบบจำลองในสายการทดลองหลัก

| สายการทดลอง | โมเดล | จำนวน feature | Precision | Recall | F1-score | Avg Precision | ROC-AUC | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Leakage-safe classical | Gradient Boosting | 22 | 0.9657 | 0.9038 | 0.9337 | 0.9703 | 0.9772 | `[[9164, 141], [422, 3965]]` |
| Multimodal suite best run | Gradient Boosting (`image_context`) | 41 | 0.9507 | 0.9179 | 0.9340 | 0.9789 | 0.9734 | `[[3717, 209], [360, 4027]]` |
| Strict no-leak reference | IdentityMLP | 22 | 0.8427 | 0.8169 | 0.8296 | 0.7344 | 0.9960 | `[[461517, 184], [221, 986]]` |

การเปรียบเทียบทั้ง 3 สายนี้ทำให้ข้อสรุปของงานไม่ได้ยึดอยู่กับตัวเลขที่ดีที่สุดเพียงค่าเดียว แต่มีทั้ง baseline ที่ควบคุม leakage ได้ดี สาย multimodal ที่ใช้เป็น main line จริงของระบบ และสาย strict no-leak ที่ใช้ตรวจสอบความสมเหตุสมผลของผลลัพธ์ภายใต้เงื่อนไขที่เข้มงวดกว่า

## 3.5.4 การประเมิน Calibration และผลในระดับ Production Workflow

หลังจากได้คะแนนดิบจากแบบจำลอง ผู้วิจัยใช้ calibration เพื่อทำให้คะแนนสามารถตีความเป็นระดับความมั่นใจได้ดีขึ้น ในสายหลักของงานใช้ `Isotonic Regression` และเลือก threshold บน validation set ก่อนนำไปใช้กับ full candidate scoring ตัวอย่างจากสาย `strict no-leak` แสดงว่า Expected Calibration Error บน validation ลดจาก `0.4027` ก่อน calibration เหลือ `1.07e-07` หลัง calibration สะท้อนว่าคะแนนหลัง calibration มีความสอดคล้องกับความน่าจะเป็นมากขึ้น อย่างไรก็ตาม การประเมิน calibration ในงานนี้ไม่ได้จบที่การรายงาน ECE แต่ต้องพิจารณาต่อว่าคะแนนดังกล่าวสามารถใช้ตั้ง operating point สำหรับ `MATCH`, `REVIEW` และ `NO_MATCH` ได้อย่างเหมาะสมหรือไม่

ในระดับ production workflow ผู้วิจัยอ้างอิงผลจาก [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) และ [crm_entity_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage15_crm_entity_pipeline/reports/crm_entity_report.json) โดยใช้ `match_threshold = 0.98` และ `review_threshold = 0.95` เป็น threshold ปลายทาง ผลประเมินพบว่า exact tier มี precision `0.9949` ขณะที่ model-based `MATCH` tier มี precision `0.8942` และ `REVIEW` tier มี precision `0.0471` เมื่อรวม exact และ model-based high-confidence matches เข้าด้วยกัน ระบบได้ `final_match_only_count = 20,549`, `final_match_only_true_positive_count = 19,624`, `final_match_only_precision = 0.9550` และ `final_match_only_recall = 0.6711`

เมื่อผลดังกล่าวถูกส่งเข้าสู่ CRM entity pipeline ระบบสร้าง `match_decisions` ทั้งหมด `2,086,245` รายการ สร้าง `review_queue` `86,296` รายการ รวมโปรไฟล์เป็น `unified_profiles` `19,799` รายการ และสร้าง `lead_scores` `19,799` รายการ โดยแบ่งผลลัพธ์เชิงปฏิบัติการเป็น `MATCH = 20,549`, `REVIEW = 86,296` และ `NO_MATCH = 1,979,400` ส่วน lead tiers ถูกแบ่งเป็น `HOT = 4,937`, `WARM = 9,039` และ `COLD = 5,823` ตัวเลขเหล่านี้แสดงให้เห็นว่าการประเมินของงานไม่ได้หยุดอยู่ที่คะแนนเชิงสถิติของแบบจำลอง แต่ขยายไปถึงภาระของ review queue การรวมข้อมูลลูกค้า และความพร้อมของผลลัพธ์สำหรับใช้งานใน CRM จริง

ตารางที่ 3.12 ผลการประเมินในระดับ production workflow

| รายการ | ค่า |
| --- | ---: |
| Match threshold | 0.98 |
| Review threshold | 0.95 |
| Exact tier precision | 0.9949 |
| Match tier precision | 0.8942 |
| Review tier precision | 0.0471 |
| Final match-only count | 20,549 |
| Final match-only true positives | 19,624 |
| Final match-only precision | 0.9550 |
| Final match-only recall | 0.6711 |
| Review queue size | 86,296 |
| Unified profiles | 19,799 |
| Lead scores | 19,799 |

โดยสรุป ขั้น Evaluation ของงานนี้ถูกออกแบบให้พิสูจน์ประสิทธิภาพของระบบในลักษณะหลายชั้น เริ่มจากการวัดว่าระบบหา candidate pairs ได้ดีเพียงใด ต่อด้วยการวัดว่าแบบจำลองให้คะแนนคู่ข้อมูลได้ดีเพียงใด จากนั้นตรวจว่าคะแนนดังกล่าวผ่าน calibration แล้วมีความน่าเชื่อถือพอสำหรับการตั้ง threshold หรือไม่ และสุดท้ายวัดว่าระบบทั้งหมดสามารถแปลงผลลัพธ์ให้เป็น workflow ที่ใช้งานได้จริงใน CRM ได้หรือไม่ การออกแบบการประเมินเช่นนี้ทำให้ข้อสรุปของงานมีความสมบูรณ์มากกว่าการรายงานตัวเลขของแบบจำลองเพียงอย่างเดียว
