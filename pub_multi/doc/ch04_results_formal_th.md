# บทที่ 4 ผลการวิจัยและอภิปรายผล

บทนี้นำเสนอผลการวิจัยในขอบเขตของ Project 1 ซึ่งเป็นส่วนของระบบที่พัฒนาได้ไกลที่สุดและพร้อมเชื่อมต่อสู่การใช้งานจริงในระบบ CRM คือ `decision thresholding` โดยระบบใช้ `exact-first` ร่วมกับ `calibrated candidate scoring` เพื่อแบ่งผลลัพธ์เป็น `MATCH`, `REVIEW` และ `NO_MATCH` อย่างเป็นระบบ เนื้อหาทั้งหมดอ้างอิงจากผลรันจริงในโค้ดและ artifact ภายใน workspace เท่านั้น ไม่ใช้ตัวเลขสมมติหรือค่าที่แต่งขึ้นเพื่อประกอบการเขียนรายงาน

เพื่อให้การนำเสนอมีลำดับเชิงวิชาการที่สอดคล้องกับ pipeline จริง บทนี้จึงเริ่มจากการอธิบายการเตรียมข้อมูลและตัวชี้วัดที่ใช้ประเมินผลก่อน จากนั้นจึงอธิบาย candidate-scoring model ที่ถูกใช้ก่อน thresholding ว่า main run คืออะไรและทำงานอย่างไร แล้วจึงวิเคราะห์พฤติกรรมของแบบจำลองผ่าน confusion matrix, score distribution และ probability calibration ก่อนเชื่อมเข้าสู่ผลลัพธ์ระดับ production ของ `exact-first + calibrated candidate scoring` และปิดท้ายด้วยการวิเคราะห์ความสำคัญของคุณลักษณะทั้ง `41` ตัวที่สนับสนุนการตัดสินใจของระบบ

## 4.1 การเตรียมข้อมูล

ชุดข้อมูลที่ใช้ใน Project 1 เป็นข้อมูลโปรไฟล์ผู้ใช้ที่ผ่านการ normalize ให้อยู่ในโครงสร้างเดียวกันตามเทมเพลตข้อมูลที่ระบบใช้งานจริง โดยครอบคลุมข้อมูลจาก `Twitter`, `Instagram` และ `Google+ archive` ในขั้น full candidate pipeline ระบบคัดเฉพาะโปรไฟล์ที่พร้อมใช้งานจริงได้ `36,804` โปรไฟล์ ซึ่งแต่ละโปรไฟล์ประกอบด้วยข้อมูลชื่อผู้ใช้ (`Username`), ชื่อจริง (`Full Name`), ประวัติย่อ (`Bio`), ลิงก์อ้างอิง (`URL`) และตำแหน่งที่อยู่ (`Location`) รวมถึง metadata เสริมที่มีระดับความสมบูรณ์แตกต่างกันไปตามแต่ละแพลตฟอร์ม

ข้อมูลจาก `Google+ archive` ยังคงเป็นส่วนที่ท้าทายที่สุดของชุดข้อมูล เนื่องจากเป็นข้อมูลตกค้างจากแพลตฟอร์มที่ยุติการให้บริการไปแล้ว ทำให้ metadata, image coverage และสัญญาณเสริมหลายประเภทไม่สมบูรณ์เท่ากับ `Twitter` และ `Instagram` ประเด็นนี้มีความสำคัญต่อการตีความผล เพราะทำให้ Project 1 ต้องทำงานภายใต้ข้อจำกัดของข้อมูลไม่ครบจริง ไม่ใช่เพียงบนข้อมูลที่สะอาดสมบูรณ์แบบในทางทฤษฎี

ตารางที่ 4.1 จำนวน valid profiles แยกตามแพลตฟอร์มที่ใช้จริงใน Project 1

| แพลตฟอร์ม | จำนวนโปรไฟล์ | สัดส่วน (%) |
| --- | ---: | ---: |
| Twitter | 13,959 | 37.9 |
| Instagram | 10,956 | 29.8 |
| Google+ | 11,889 | 32.3 |
| รวม | 36,804 | 100.0 |

ก่อนที่โปรไฟล์ทั้งหมดจะเข้าสู่ขั้น candidate scoring ระบบต้องผ่านกระบวนการลดพื้นที่ค้นหาก่อน เพราะหากเปรียบเทียบทุกคู่แบบ cross-platform โดยตรงจะมีจำนวนคู่สูงถึง `449,149,239` คู่ การลด search space จึงเป็นเงื่อนไขจำเป็นของ Project 1 ไม่ใช่เพียงขั้นตอนเสริม โดย pipeline ใช้ `exact-first` ร่วมกับ blocking เพื่อดึงคู่ที่มีหลักฐานชัดมากออกมา merge อัตโนมัติทันที และกันเฉพาะคู่ที่ยังมีความกำกวมเข้าสู่ขั้น candidate scoring

ตารางที่ 4.2 ผลของขั้นเตรียม candidate set ก่อนเข้าสู่ candidate scoring

| รายการ | ค่าที่ได้ |
| --- | ---: |
| All cross-platform pairs | 449,149,239 |
| Ground-truth positive pairs | 29,243 |
| Exact-first auto-accepted pairs | 12,403 |
| Candidate pairs for model | 2,073,842 |
| Search-space reduction | 99.54% |
| Ground-truth coverage | 88.67% |

สำหรับชุดข้อมูลที่ใช้พัฒนา candidate-scoring model ระบบใช้ leak-safe split ที่ไม่มี component overlap ระหว่าง `train`, `validation` และ `test` และไม่มี self-pairs ค้างอยู่ในชุดข้อมูล การแบ่งลักษณะนี้ช่วยป้องกัน optimistic bias ที่มักเกิดขึ้นในงาน identity resolution เมื่อ entity เดียวกันหลุดไปอยู่หลาย split พร้อมกัน

ตารางที่ 4.3 โครงสร้างของ supervised split สำหรับ main run

| Split | Profiles | Positive Pairs | Random Negatives | Hard Negatives | Rows รวม |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 22,478 | 20,470 | 15,346 | 5,987 | 41,803 |
| Validation | 7,925 | 4,386 | 3,289 | 905 | 8,580 |
| Test | 6,401 | 4,387 | 3,290 | 636 | 8,313 |

ตารางที่ 4.3 แสดงว่าผู้วิจัยไม่ได้ฝึกแบบจำลองจาก positive pairs เพียงอย่างเดียว แต่ใช้ทั้ง `random negatives` และ `hard negatives` เพื่อให้แบบจำลองเห็นทั้งกรณีที่แยกได้ง่ายและกรณีที่มีความคล้ายสูงมากในเวลาเดียวกัน การออกแบบนี้ส่งผลโดยตรงต่อคุณภาพของ calibrated candidate scoring ในขั้นถัดไป

[แทรกรูปที่ 4.1 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_data_prep_summary.png
คำบรรยาย: รูปที่ 4.1 สรุปผลการเตรียมข้อมูลของ Project 1 แสดงจำนวน valid profiles แยกตามแพลตฟอร์ม โครงสร้างของ leak-safe supervised split และการเติบโตของ feature space จาก baseline ไปสู่ main run แบบ image_context

## 4.2 วิธีวัดประสิทธิภาพ

การประเมินผลใน Project 1 แบ่งออกเป็น 3 ระดับตามลำดับของ pipeline จริง ได้แก่ ระดับการลดพื้นที่ค้นหา ระดับ candidate scoring และระดับ decision thresholding สำหรับการใช้งานจริงใน CRM การแยกตัวชี้วัดออกตามหน้าที่ของแต่ละขั้นมีความสำคัญ เพราะค่า metric ที่ดีในระดับหนึ่งไม่ได้แปลว่าระบบพร้อมใช้งานจริงโดยอัตโนมัติเสมอไป

### 4.2.1 ตัวชี้วัดสำหรับ Search Space Reduction และ Candidate Retrieval

ในขั้นก่อนเข้าสู่ candidate scoring ผู้วิจัยใช้ตัวชี้วัดหลัก 2 ค่า คือ `Reduction Ratio` และ `Ground-truth Coverage`

Reduction Ratio = 1 - (Pairs_after / Pairs_before)

ค่าแรกสะท้อนว่าระบบลดจำนวนคู่ที่ต้องตรวจต่อได้มากเพียงใด ยิ่งมีค่าสูงยิ่งดี เพราะหมายถึง pipeline ช่วยลดภาระการคำนวณในขั้น scoring ได้มาก

Ground-truth Coverage = True matches retained after exact-first + blocking / True matches total

ค่านี้สะท้อนว่ายังเก็บคู่จริงที่ควรตรวจต่อไว้ได้มากเพียงใด หาก coverage ต่ำเกินไป แบบจำลองในขั้นถัดไปจะไม่มีโอกาสกู้คู่จริงที่ถูกตัดออกไปแล้ว

### 4.2.2 ตัวชี้วัดสำหรับ Candidate Scoring

ในระดับ candidate scoring ผู้วิจัยใช้ตัวชี้วัดมาตรฐานของงานจัดอันดับและจำแนก ได้แก่ `Precision`, `Recall`, `F1-score`, `Average Precision (AP)` และ `ROC-AUC`

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1-score = 2 x (Precision x Recall) / (Precision + Recall)

Average Precision (AP) = sum_n ((R_n - R_(n-1)) x P_n)

โดยที่ `P_n` คือค่า Precision ณ จุดลำดับที่ `n` และ `R_n` คือค่า Recall ณ จุดเดียวกัน ค่า AP ใช้ประเมินคุณภาพของการจัดอันดับคู่ข้อมูลจากคะแนนสูงไปต่ำ โดยเหมาะกับปัญหาที่มี class imbalance สูง

ROC-AUC เป็นพื้นที่ใต้เส้นโค้ง ROC ซึ่งสร้างจากความสัมพันธ์ระหว่าง `True Positive Rate (TPR)` และ `False Positive Rate (FPR)` ดังนี้

TPR = TP / (TP + FN)

FPR = FP / (FP + TN)

นอกจากนั้น ผู้วิจัยยังใช้ `Confusion Matrix` เพื่อวิเคราะห์โครงสร้างความผิดพลาดของแบบจำลอง โดยพิจารณาจำนวน `true positives`, `false positives`, `true negatives` และ `false negatives` ที่เกิดขึ้นจริงบนชุดทดสอบ ควบคู่กับการใช้ `precision@k` เพื่อประเมินคุณภาพของการจัดอันดับคู่ข้อมูลที่ได้รับคะแนนสูงสุดบน full candidate pairs ซึ่งมีความสำคัญอย่างยิ่งในระบบที่ต้องนำคะแนนของแบบจำลองไปใช้จัดลำดับคิวการตรวจสอบในขั้น review ต่อไป

### 4.2.3 ตัวชี้วัดสำหรับ Probability Calibration และ Decision Thresholding

เมื่อคะแนนของแบบจำลองถูกนำไปใช้ในบริบท production ตัวชี้วัดที่สำคัญจะเปลี่ยนจาก metric เชิงจำแนกทั่วไปไปสู่ metric เชิงการตัดสินใจ ได้แก่ `Expected Calibration Error (ECE)`, `final match-only precision`, `final match-only recall`, ขนาดของ `review queue` และผลลัพธ์ปลายทางของแต่ละ tier ได้แก่ `MATCH`, `REVIEW` และ `NO_MATCH`

Expected Calibration Error (ECE) = sum_(m=1..M) ((|B_m| / n) x |acc(B_m) - conf(B_m)|)

โดยที่ `B_m` คือชุดข้อมูลใน bin ที่ `m`, `|B_m|` คือจำนวนตัวอย่างใน bin นั้น, `acc(B_m)` คือ accuracy จริงของ bin และ `conf(B_m)` คือค่า confidence เฉลี่ยของ bin ดังกล่าว หากค่า ECE ต่ำ แสดงว่าค่าความน่าจะเป็นที่ผ่านการ calibration แล้วมีความสอดคล้องกับสัดส่วนความจริงมากขึ้น ทำให้ threshold ปลายทาง เช่น `0.98` หรือ `0.95` มีความหมายเชิงปฏิบัติการมากขึ้น

Final Match-only Precision = TP_match / (TP_match + FP_match)

Final Match-only Recall = TP_match / TP_all

Review Queue Size = จำนวนคู่ที่มีคะแนนอยู่ในช่วง Review Threshold <= score < Match Threshold

ตัวชี้วัดทั้งสามค่านี้ใช้ตอบคำถามปลายทางโดยตรงว่า operating point ที่เลือกไว้เหมาะกับการ merge อัตโนมัติใน CRM หรือไม่ กล่าวคือ precision ใช้ควบคุมความเสี่ยงของ false merge, recall ใช้สะท้อนความครอบคลุมของคู่จริงที่ระบบกู้ได้เอง และ review queue ใช้สะท้อนภาระงานของขั้น `Human-in-the-Loop`

## 4.3 Candidate-Scoring Model ที่ใช้ก่อน Decision Thresholding

main run ที่ถูกใช้เป็นฐานของ Project 1 คือ `image_context_r075_h20_s42` จากชุดทดลอง `multimodal suite` ซึ่งสืบทอด pair construction, leakage-safe split และ calibration logic มาจาก rebuilt hybrid pipeline เดิม ดังนั้นผลในบทนี้จึงไม่ใช่ผลจาก pair source ที่รั่วหรือ split ที่มี entity overlap ระหว่าง `train`, `validation` และ `test`

แบบจำลองในงานนี้ไม่ได้รับ raw profiles เข้าไปโดยตรง แต่แปลงโปรไฟล์สองฝั่งให้เป็น `pair-level feature vector` ก่อนเสมอ โดยโครงสร้างของ feature space เติบโตเป็นลำดับจาก `22` baseline text+attribute features ไปเป็น `23` features เมื่อเพิ่ม `SBERT`, ขยายเป็น `37` features เมื่อเพิ่ม image statistics และสิ้นสุดที่ `41` features เมื่อเพิ่ม caption-to-text cross signals ใน run แบบ `image_context` วิธีออกแบบเช่นนี้ทำให้ระบบทำงานในฐานะ `tabular candidate scorer` ที่เรียนรู้จากหลักฐานเชิงความคล้ายของคู่โปรไฟล์หลายประเภทพร้อมกัน

[แทรกรูปที่ 4.2 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_model_pipeline.png
คำบรรยาย: รูปที่ 4.2 โครงสร้างของ main-run model pipeline สำหรับ Project 1 เริ่มจากการสร้าง pair features จากข้อมูลชื่อ ข้อความ attribute ภาพ และ caption จากนั้นให้ candidate-scoring model ประเมินคะแนน ก่อนผ่าน probability calibration และ decision thresholding

จากรูปที่ 4.2 จะเห็นว่าตัวแบบในงานนี้ไม่ใช่ end-to-end deep model ที่เรียนรู้จากข้อมูลดิบทั้งก้อน แต่เป็น pipeline เชิงวิศวกรรมข้อมูลที่แยกขั้นชัดเจน กล่าวคือผู้วิจัยสร้าง feature table ระดับคู่ก่อน แล้วจึงให้ model family ที่เหมาะกับข้อมูลเชิงตารางเรียนรู้รูปแบบการแยก `MATCH / NO_MATCH` วิธีนี้เหมาะกับขอบเขตของ Project 1 เพราะอธิบายได้ชัด ตรวจสอบได้ง่าย และรองรับการนำคะแนนไป calibrate และ threshold ต่อในเชิงปฏิบัติการ

### 4.3.1 การเลือกโมเดลและการเปรียบเทียบ candidate-scoring configurations

ภายใน main run ระบบฝึกและเปรียบเทียบอย่างน้อย `Gradient Boosting`, `Random Forest` และ `Logistic Regression` แล้วเลือกโมเดลที่ให้ผลดีที่สุดบน validation set ซึ่ง `experiment_report.json` ระบุชัดว่า `best_model = gb` หรือ `Gradient Boosting`

ตารางที่ 4.4 ผล validation leaderboard ภายใน main run

| Model | Val AP | Val ROC-AUC | ข้อสรุปใน run |
| --- | ---: | ---: | --- |
| Gradient Boosting | 0.9795 | 0.9720 | ดีที่สุด |
| Random Forest | 0.9778 | 0.9696 | รองลงมา |
| Logistic Regression | 0.9431 | 0.9245 | ต่ำกว่าสองตัวแรกชัดเจน |

เพื่อไม่ให้การเลือก `Gradient Boosting` อิงเพียง leaderboard ภายใน run เดียว ผู้วิจัยยัง benchmark model families หลายชนิดบน final `41-feature split` เดียวกันเพิ่มเติม ซึ่งรวมถึง `MLP`, `Extra Trees`, `AdaBoost`, `Logistic Regression` และ `Linear SVM` ด้วย ผลในชุด benchmark นี้ชี้ว่า `Gradient Boosting` ให้สมดุลของ `AP`, `ROC-AUC`, `F1` และ `Precision` ดีที่สุดในภาพรวม แม้บางโมเดลจะมีค่าในบาง metric ใกล้เคียงกัน

[แทรกรูปที่ 4.3 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_model_family_heatmap.png
คำบรรยาย: รูปที่ 4.3 การเปรียบเทียบ model families บน final 41-feature split โดยใช้สีแสดงคุณภาพสัมพัทธ์ภายในแต่ละ metric และใช้ตัวเลขในเซลล์แสดงค่าจริงของ AP ROC-AUC F1 Precision และ Recall

พร้อมกันนั้น ผู้วิจัยยังเปรียบเทียบ candidate-scoring configurations ที่ใช้ pair set และ split เดียวกัน เพื่อพิสูจน์ว่า `image_context` เหมาะสมกว่าการหยุดที่ `text_attr_hybrid` หรือ `image_stats`

ตารางที่ 4.5 ผลเปรียบเทียบ candidate-scoring configurations ที่ใช้เลือก main run

| Configuration | Feature Count | AP | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text/Attribute Hybrid | 23 | 0.9782 | 0.9715 | 0.9330 | 0.9538 | 0.9132 |
| Image Statistics | 37 | 0.9789 | 0.9731 | 0.9340 | 0.9537 | 0.9152 |
| Image Context | 41 | 0.9789 | 0.9734 | 0.9340 | 0.9507 | 0.9179 |

[แทรกรูปที่ 4.4 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_metric_heatmap.png
คำบรรยาย: รูปที่ 4.4 การเปรียบเทียบ metric ของ candidate-scoring configurations โดยใช้สีแสดงคุณภาพสัมพัทธ์ภายในแต่ละ metric และใช้ตัวเลขในเซลล์แสดงค่าจริงของ AP ROC-AUC F1 Precision และ Recall

ผลในตารางที่ 4.5 และรูปที่ 4.4 แสดงว่า `image_context` เป็น configuration ที่เหมาะสมที่สุดสำหรับ Project 1 เพราะให้ `AP`, `ROC-AUC` และ `Recall` สูงที่สุดในภาพรวม แม้ `Image Statistics` จะมี `F1` สูงกว่าในหลักทศนิยมลึกบางจุด แต่ความต่างนั้นน้อยมากเมื่อเทียบกับข้อได้เปรียบของ `Image Context` ในด้าน ranking quality และความครอบคลุมของคู่จริง

เพื่อให้การเปรียบเทียบครบในระดับโครงสร้างความผิดพลาด ผู้วิจัยจึงพิจารณา confusion matrix ของทั้งสาม configuration ควบคู่ไปด้วย เพราะค่า `AP` หรือ `F1` เพียงอย่างเดียวอาจยังไม่อธิบายได้ชัดว่าความต่างของแต่ละ run เกิดจากการลด `false positives` หรือการลด `false negatives` เป็นหลัก

ตารางที่ 4.6 Confusion matrix เปรียบเทียบ candidate-scoring configurations บน test split

| Configuration | Threshold | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text/Attribute Hybrid | 0.44 | 3,732 | 194 | 381 | 4,006 |
| Image Statistics | 0.43 | 3,731 | 195 | 372 | 4,015 |
| Image Context | 0.35 | 3,717 | 209 | 360 | 4,027 |

[แทรกรูปที่ 4.5 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_suite_confusion_grid.png
คำบรรยาย: รูปที่ 4.5 Confusion matrices ของ candidate-scoring configurations ทั้งสามแบบบน test split แสดง trade-off ระหว่าง false positives และ false negatives ที่ใช้ประกอบการเลือก main run

ผลในตารางที่ 4.6 และรูปที่ 4.5 ชี้ให้เห็นลักษณะ trade-off ที่สำคัญ 2 ประเด็น ประเด็นแรก `Image Statistics` ลด `FN` ลงจาก `381` เหลือ `372` เมื่อเทียบกับ `Text/Attribute Hybrid` โดยแทบไม่เพิ่ม `FP` เลย ซึ่งยืนยันว่าการเพิ่มสัญญาณภาพระดับสถิติมีผลเชิงบวกจริงต่อการกู้คู่จริง ประเด็นที่สอง `Image Context` ลด `FN` ต่อเนื่องลงเหลือ `360` แต่ยอมให้ `FP` เพิ่มเป็น `209` ภายใต้ threshold ที่ต่ำกว่า (`0.35`) ผลเช่นนี้สอดคล้องกับค่า `Recall` ที่สูงที่สุดของ run นี้ และเป็นเหตุผลสำคัญที่ผู้วิจัยเลือก `image_context` เป็น main run แล้วนำไป calibrate และตั้ง threshold เชิง production ต่อ แทนที่จะเลือก configuration ที่เข้มงวดกว่าแต่กู้คู่จริงได้น้อยกว่า

### 4.3.2 ผลเชิงจำแนกของ Main Run บน Test Split

เมื่อใช้ threshold เชิงจำแนก `0.35` บน test split main run ให้ `Precision = 0.9507`, `Recall = 0.9179` และ `F1 = 0.9340` พร้อม confusion matrix `TN = 3,717`, `FP = 209`, `FN = 360` และ `TP = 4,027`

ตารางที่ 4.7 ผลของ main run บน test split ก่อนเข้าสู่ production thresholding

| รายการ | ค่าที่ได้ |
| --- | ---: |
| Best model | Gradient Boosting |
| Feature count | 41 |
| Threshold สำหรับ test classification | 0.35 |
| Test AP | 0.9789 |
| Test ROC-AUC | 0.9734 |
| Test F1 | 0.9340 |
| Test Precision | 0.9507 |
| Test Recall | 0.9179 |
| TN / FP / FN / TP | 3,717 / 209 / 360 / 4,027 |

[แทรกรูปที่ 4.6 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_main_confusion_matrix.png
คำบรรยาย: รูปที่ 4.6 Confusion matrix ของ main run บน test split ที่ threshold = 0.35 แสดงสมดุลระหว่าง true positives false positives true negatives และ false negatives ของแบบจำลองก่อนเข้าสู่ขั้น decision thresholding ระดับ production

ผลนี้ชี้ว่า main run สามารถรักษาสมดุลระหว่างความเข้มงวดและความครอบคลุมได้ดี กล่าวคือจำนวน false positives อยู่ในระดับต่ำเมื่อเทียบกับ true positives และจำนวน false negatives ก็ไม่ได้สูงเกินไปจนทำให้ recall ตกลงอย่างมีนัยสำคัญ หากมองในมุมของ pipeline ทั้งระบบ ผลในระดับนี้เหมาะกับการส่งต่อเข้าสู่ขั้น calibration และ thresholding มากกว่าการใช้ตัดสินใจ merge โดยตรง

### 4.3.3 การกระจายคะแนนและ Threshold Trade-off

เมื่อพิจารณาการกระจายของ calibrated probabilities บน test split จะเห็นว่าคู่ `NO_MATCH` ส่วนใหญ่กระจุกตัวอยู่ใกล้ศูนย์ ขณะที่คู่ `MATCH` กระจุกตัวอยู่ใกล้หนึ่งอย่างชัดเจน โดยค่าเฉลี่ยของกลุ่ม `NO_MATCH` อยู่ที่ประมาณ `0.1095` และค่ามัธยฐานอยู่ที่ `0.0684` ส่วนกลุ่ม `MATCH` มีค่าเฉลี่ย `0.9012` และค่ามัธยฐานสูงถึง `0.9966` การแยกตัวของสอง distribution นี้เป็นสัญญาณว่าตัวแบบมีความสามารถในการจัดลำดับคู่ที่น่าใช่และไม่น่าใช่ออกจากกันได้ดี

[แทรกรูปที่ 4.7 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_score_distribution.png
คำบรรยาย: รูปที่ 4.7 การกระจายของ calibrated scores บน test split แยกตาม Actual MATCH และ Actual NO_MATCH พร้อมเส้นอ้างอิงที่ threshold = 0.35 และเส้น production thresholds ที่ 0.95 และ 0.98

รูปที่ 4.7 ช่วยให้เห็นชัดว่า threshold สำหรับการประเมินเชิงจำแนก (`0.35`) และ threshold สำหรับการใช้งานจริง (`0.95 / 0.98`) อยู่คนละบริบทกัน โดย threshold `0.35` อยู่ในช่วงที่ optimize ความสมดุลของ `F1` บน test split ขณะที่ threshold ปลายทางถูกเลื่อนไปยังปลายขวาของ distribution เพื่อคุม precision สำหรับการ merge จริงให้เข้มงวดขึ้นมาก

หากพิจารณาความสัมพันธ์ระหว่าง `Precision`, `Recall` และ `F1` เมื่อขยับ threshold จะเห็น trade-off ชัดเจน กล่าวคือ threshold ต่ำช่วยให้ recall สูง แต่ precision ลดลงเร็ว ขณะที่ threshold สูงช่วยคุม precision ได้ดีขึ้นแลกกับ recall ที่ลดลง ซึ่งเป็นเหตุผลสำคัญที่ทำให้ production operating point ต้องแยก `MATCH` ออกจาก `REVIEW` แทนการใช้ threshold เดียวบังคับทั้งระบบ

[แทรกรูปที่ 4.8 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_threshold_tradeoff.png
คำบรรยาย: รูปที่ 4.8 พฤติกรรมของ Precision Recall และ F1 ของ main run เมื่อเปลี่ยน threshold บน test split แสดงความแตกต่างระหว่าง threshold เชิงจำแนกที่ 0.35 กับ production review threshold ที่ 0.95

ที่ threshold `0.35` ระบบได้ `Precision = 0.9507`, `Recall = 0.9179` และ `F1 = 0.9340` ซึ่งเหมาะสำหรับการประเมินเชิงแบบจำลอง แต่เมื่อยก threshold ไปถึง `0.95` ค่า precision บน test split จะสูงขึ้นเป็นประมาณ `0.9936` ขณะที่ recall ลดลงเหลือประมาณ `0.8110` และ `F1` ลดลงมาอยู่ที่ประมาณ `0.8931` ผลนี้อธิบายได้ชัดเจนว่าทำไม production จึงต้องเพิ่มชั้น `REVIEW` เข้ามาช่วยรับคู่ที่ยังมีศักยภาพแต่ไม่ควรถูก auto-merge ทันที

### 4.3.4 การสอบเทียบความน่าจะเป็นด้วย Isotonic Regression

หลังจากเลือกโมเดลที่ดีที่สุดแล้ว pipeline ไม่ได้นำค่าความน่าจะเป็นดิบจาก `predict_proba()` ไปใช้ตัดสินใจใน production ทันที แต่ใช้ `IsotonicRegression(out_of_bounds="clip")` เป็นตัว calibrate คะแนนก่อนเสมอ โดยขั้นตอนจริงคือ fit isotonic calibrator บน `validation raw scores` กับ `validation labels` จากนั้นจึงนำ calibrator เดียวกันไปแปลงคะแนนของ `train`, `validation` และ `test` ให้กลายเป็น calibrated probabilities ก่อนเลือก operating point ระดับ production

ผลจาก score files ของ main run แสดงว่า isotonic calibration ทำงานได้ดีมาก โดย validation set ซึ่งเป็นชุดที่ใช้ fit calibrator มี `ECE = 0.0000` และ test set มี `ECE = 0.0044` เท่านั้น ค่าที่ต่ำมากนี้สะท้อนว่า calibrated probabilities บน test ยังคงเกาะใกล้ observed positive rate ในแต่ละช่วงคะแนนอยู่มาก

[แทรกรูปที่ 4.9 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_calibration_curve.png
คำบรรยาย: รูปที่ 4.9 Reliability ของ calibrated probabilities หลังผ่าน Isotonic Regression เปรียบเทียบ validation set และ test set โดยใช้ขนาด marker แทนจำนวนคู่ในแต่ละ probability bin และเส้นตั้งแสดง production thresholds ที่ 0.95 และ 0.98

รูปที่ 4.9 ให้ข้อสรุปสำคัญว่า calibrated score ของ main run มีความน่าเชื่อถือเพียงพอสำหรับการกำหนด operating point เชิง production กล่าวคือ validation reliability curve เกือบซ้อนทับเส้นทแยงตามธรรมชาติของ isotonic fit ขณะที่ test curve ยังเกาะใกล้แนวอุดมคติอยู่มาก แม้บาง bin ขนาดเล็กจะมีความผันผวนตามจำนวนตัวอย่างที่น้อย

### 4.3.5 คุณภาพการจัดอันดับบน Full Candidate Pairs

นอกเหนือจากผลบน test split ผู้วิจัยยังประเมินคุณภาพการจัดอันดับของ calibrated candidate scores บน candidate pairs จริงทั้งหมด `2,073,842` คู่ที่ผ่าน blocking เข้ามา ผลลัพธ์แสดงว่า `candidate_avg_precision = 0.6017` และ `candidate_roc_auc = 0.9600` พร้อมค่า `precision@100 = 0.9800`, `precision@500 = 0.9800`, `precision@1000 = 0.9820` และ `precision@5000 = 0.9472`

[แทรกรูปที่ 4.10 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_ranking_quality.png
คำบรรยาย: รูปที่ 4.10 คุณภาพการจัดอันดับของ candidate scoring บน full candidate pairs โดยแสดง precision@100 precision@500 precision@1000 และ precision@5000 ร่วมกับค่า Candidate AP และ ROC-AUC

ค่าเหล่านี้มีความหมายเชิงระบบมาก เพราะชี้ว่าแม้ candidate pool จริงจะมี class imbalance สูงและมี candidate count ระดับล้านคู่ แต่คะแนนของแบบจำลองยังสามารถดันคู่จริงจำนวนมากขึ้นไปอยู่ในอันดับต้น ๆ ได้อย่างมีประสิทธิภาพ ซึ่งเป็นคุณสมบัติที่จำเป็นมากสำหรับ thresholding และ review workflow

## 4.4 Exact-First ร่วมกับ Calibrated Candidate Scoring สำหรับ Decision Thresholding

เมื่อได้ candidate-scoring model ที่ผ่านการ calibration แล้ว ขั้นถัดไปคือการนำคะแนนดังกล่าวไปเชื่อมกับ `exact-first` เพื่อสร้างกฎตัดสินใจที่ใช้ได้จริงใน CRM จุดตั้งต้นของปัญหานี้คือ valid profiles `36,804` โปรไฟล์ก่อให้เกิดคู่แบบ cross-platform ทั้งหมด `449,149,239` คู่ ซึ่งไม่สามารถเปรียบเทียบทุกคู่โดยตรงได้ในทางปฏิบัติ ดังนั้น pipeline จึงต้องลด search space ก่อน

ผลรันจริงแสดงว่า exact-first และ blocking สามารถคัดคู่ exact ที่ merge ได้ทันที `12,403` คู่ และลดคู่ที่ต้องให้แบบจำลองประเมินต่อเหลือ `2,073,842` คู่ คิดเป็นการลดพื้นที่ค้นหาลง `99.54%` พร้อมรักษา coverage ของ ground-truth matches ไว้ `88.67%`

ตารางที่ 4.8 Retrieval funnel ของ Project 1 ก่อนเข้าสู่การตัดสินใจปลายทาง

| ขั้นตอน | จำนวนคู่ |
| --- | ---: |
| All cross-platform pairs | 449,149,239 |
| Exact-first auto-accepted pairs | 12,403 |
| Candidate pairs for model | 2,073,842 |
| Final MATCH only | 20,549 |
| REVIEW queue | 86,296 |

[แทรกรูปที่ 4.11 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_retrieval_funnel.png
คำบรรยาย: รูปที่ 4.11 Retrieval funnel ของ Project 1 แสดงการลด search space จาก all cross-platform pairs ไปสู่ exact-first pairs model candidates final MATCH only และ REVIEW queue

[แทรกรูปที่ 4.12 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_threshold_flow.png
คำบรรยาย: รูปที่ 4.12 โครงสร้างของ Project 1 ในขั้น decision thresholding เริ่มจาก exact-first และ blocking ก่อนส่ง candidate pairs เข้าสู่ calibrated candidate scoring แล้วแยกผลลัพธ์เป็น MATCH REVIEW และ NO_MATCH

ตารางที่ 4.8 และรูปที่ 4.11-4.12 ชี้ให้เห็นว่า `exact-first` ไม่ได้เป็นเพียง heuristic เสริม แต่เป็นองค์ประกอบหลักของ Project 1 เพราะช่วยดึงคู่ที่ชัดมากออกจากระบบตั้งแต่ต้น และเปิดทางให้โมเดลโฟกัสเฉพาะคู่ที่กำกวมกว่า ซึ่งเป็นส่วนที่ calibration และ thresholding จะสร้างคุณค่าจริง

## 4.5 ผลลัพธ์ปลายทางของ MATCH, REVIEW และ NO_MATCH สำหรับ CRM

ในระดับ production operating point ที่ถูกนำไปใช้จริง ระบบกำหนด `match_threshold = 0.98` และ `review_threshold = 0.95` เพื่อควบคุมความเสี่ยงของการ merge ผิดในบริบท CRM ซึ่ง false merge ถือเป็นความเสียหายที่รุนแรงกว่าการส่งคู่ไปให้มนุษย์ตรวจซ้ำ กติกาการตัดสินใจจริงจึงเป็นดังนี้

1. ถ้า `username` หรือ `URL` ตรงกันแบบ exact ให้ตัดสินเป็น `MATCH` ทันที
2. ถ้าไม่ใช่ exact pair แต่ calibrated score `>= 0.98` ให้ตัดสินเป็น `MATCH`
3. ถ้าคะแนนอยู่ในช่วง `0.95 <= score < 0.98` ให้ส่งเข้า `REVIEW`
4. ถ้าคะแนนต่ำกว่า `0.95` ให้ตัดสินเป็น `NO_MATCH`

ตารางที่ 4.9 Operating point ที่ใช้จริงในขั้น decision thresholding

| ชั้นการตัดสินใจ | เงื่อนไข | จำนวนคู่ | บทบาทใน workflow |
| --- | --- | ---: | --- |
| Exact-First | Username หรือ URL ตรงกันแบบ exact | 12,403 | ส่งเข้า `MATCH` ทันที |
| High-score Match | Calibrated score `>= 0.98` | 8,146 | ส่งเข้า `MATCH` แบบอัตโนมัติ |
| Review Band | `0.95 <= score < 0.98` | 86,296 | ส่งเข้า `REVIEW` ให้ผู้เชี่ยวชาญ |
| Low-score Reject | `score < 0.95` | 1,979,400 | ตัดเป็น `NO_MATCH` |

เมื่อใช้ threshold ปลายทาง `0.98 / 0.95` กับคะแนนที่ผ่านการ calibration แล้ว ระบบได้ผลลัพธ์ปลายทางเป็น `MATCH = 20,549` คู่, `REVIEW = 86,296` คู่ และ `NO_MATCH = 1,979,400` คู่ จาก final decisions ทั้งหมด `2,086,245` คู่ กล่าวอีกนัยหนึ่งคือระบบสามารถจัดการคู่ส่วนใหญ่ได้อัตโนมัติ และกันคู่ที่ยังไม่แน่ชัดออกมาให้มนุษย์ตรวจเพียงประมาณ `4.14%` ของ final decisions เท่านั้น

คุณภาพของ `MATCH` ที่ระบบยอม merge อัตโนมัติเป็นจุดที่สำคัญที่สุดสำหรับ CRM หลังรวม exact matches กับ high-score matches เข้าด้วยกัน ระบบได้ `final match-only precision = 0.9550` และ `final match-only recall = 0.6711` ความหมายเชิงปฏิบัติของตัวเลขนี้คือ ในทุก 100 คู่ที่ระบบ auto-merge จะมี false merge ประมาณ 4-5 คู่ ขณะเดียวกันระบบกู้คู่จริงได้เองประมาณ `67.11%` ของคู่จริงทั้งหมด ส่วนที่ยังไม่มั่นใจพอจะไม่ถูก merge ทันที แต่ถูกส่งไปยัง review queue แทน

ตารางที่ 4.10 คุณภาพของแต่ละ tier ที่เกี่ยวข้องกับการตัดสินใจปลายทาง

| กลุ่มตัดสินใจ | จำนวนคู่ | คู่จริงในกลุ่ม | Precision | Recall ต่อคู่จริงทั้งหมด | ความหมายเชิงปฏิบัติ |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact auto | 12,403 | 12,340 | 0.9949 | 0.4220 | merge ได้ทันทีด้วยกฎ exact |
| Score `>= 0.98` | 8,146 | 7,284 | 0.8942 | 0.2491 | auto-merge จาก calibrated score |
| REVIEW | 86,296 | 4,065 | 0.0471 | 0.1390 | ส่งให้ผู้เชี่ยวชาญตรวจ |
| Final MATCH only | 20,549 | 19,624 | 0.9550 | 0.6711 | คุณภาพรวมของ auto-merge |

[แทรกรูปที่ 4.13 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_threshold_dashboard.png
คำบรรยาย: รูปที่ 4.13 ผลของ thresholding ต่อปริมาณงานและคุณภาพการตัดสินใจ แสดงจำนวนคู่ในแต่ละ tier precision ของกลุ่ม auto-merge และ recall contribution ของ exact auto score band สูง และกลุ่มที่ถูกส่งเข้า review

ผลในตารางที่ 4.10 และรูปที่ 4.13 สะท้อนบทบาทของแต่ละชั้นได้ชัดมาก โดย `exact-first` ช่วยกู้ recall ได้ `42.20%` ตั้งแต่ก่อนใช้แบบจำลอง และทำด้วย precision เกือบสมบูรณ์ (`0.9949`) จากนั้นคะแนน calibrated ช่วงสูง (`>= 0.98`) ช่วยเพิ่ม recall อัตโนมัติอีก `24.91%` ของคู่จริงทั้งหมด แม้ precision ของชั้นนี้จะต่ำกว่า exact rules แต่เมื่อนำมารวมกันใน `Final MATCH only` ระบบยังรักษา precision รวมไว้ได้ `0.9550`

ในอีกด้านหนึ่ง `REVIEW` tier มี precision เพียง `0.0471` ซึ่งไม่ควรถูกตีความว่าเป็นจุดอ่อนของระบบ แต่ควรถูกตีความว่าเป็น buffer zone ที่ถูกออกแบบไว้โดยเจตนา กล่าวคือระบบกันคู่ที่ยังไม่ควร auto-merge ออกไปให้มนุษย์ตัดสินแทน เพื่อรักษาคุณภาพข้อมูลลูกค้าในระดับองค์กร

### 4.5.1 Production Decision Matrix ที่ Operating Point สุดท้าย

หากพิจารณาผลลัพธ์ปลายทางในรูปของ decision matrix จะเห็นโครงสร้างการกระจายของคู่จริงและคู่ไม่จริงในแต่ละ tier ชัดเจนยิ่งขึ้น กล่าวคือจากคู่จริงทั้งหมด `29,243` คู่ ระบบจัดให้ `19,624` คู่เข้าสู่ `MATCH`, `4,065` คู่เข้าสู่ `REVIEW` และยังเหลือคู่จริง `5,554` คู่ที่ถูกตัดเป็น `NO_MATCH` ขณะเดียวกันในฝั่งคู่ที่ไม่จริง ระบบยังคงกันไว้ใน `NO_MATCH` ได้ `1,973,846` คู่ และมีคู่ไม่จริงที่หลุดเข้า `MATCH` เพียง `925` คู่

[แทรกรูปที่ 4.14 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_production_decision_matrix.png
คำบรรยาย: รูปที่ 4.14 Production decision matrix ที่ operating point สุดท้าย แสดงความสัมพันธ์ระหว่าง Actual MATCH และ Actual NO_MATCH กับผลการตัดสินใจใน tier MATCH REVIEW และ NO_MATCH

รูปที่ 4.14 ช่วยให้ตีความ operating point ได้ชัดขึ้นในเชิงปฏิบัติ กล่าวคือ `MATCH` tier ถูกออกแบบให้กินคู่จริงเป็นหลัก ขณะที่ `REVIEW` tier ทำหน้าที่เป็นเขตกันชนสำหรับคู่ที่ยังมีศักยภาพแต่ไม่ควรถูก merge อัตโนมัติ และ `NO_MATCH` tier รับภาระในการคัดคู่ไม่จริงส่วนใหญ่ของระบบไว้ ผลในลักษณะนี้สอดคล้องกับเป้าหมายของ CRM ซึ่งให้ความสำคัญกับการลด false merge มากกว่าการพยายาม merge ให้ได้ครอบคลุมที่สุดตั้งแต่รอบแรก

### 4.5.2 การวิเคราะห์ False Positives และ False Negatives จากตัวอย่างข้อผิดพลาด

เพื่อไม่ให้การอภิปรายผลหยุดอยู่เพียงที่ metric รวม ผู้วิจัยจึงวิเคราะห์ไฟล์ตัวอย่างข้อผิดพลาดจริง `fp_top.csv` และ `fn_top.csv` อย่างละ `200` คู่ ซึ่งเก็บคู่ที่แบบจำลองผิดพลาดเด่นที่สุดจาก test split ของ run หลัก ผลการสรุป pattern แสดงให้เห็นว่ากลุ่ม `false negatives` และ `false positives` มีธรรมชาติแตกต่างกันอย่างชัดเจน

ตารางที่ 4.11 สรุป pattern ของข้อผิดพลาดจากตัวอย่าง `fp_top.csv` และ `fn_top.csv`

| ตัวชี้วัด | False Positives | False Negatives |
| --- | ---: | ---: |
| ขนาดตัวอย่างที่วิเคราะห์ | 200 | 200 |
| คู่แพลตฟอร์มที่พบบ่อยที่สุด | `googleplus -> twitter` 50 คู่ (25.0%) | `googleplus -> twitter` 110 คู่ (55.0%) |
| ทั้ง `username_jaro` และ `fullname_jaro` ต่ำกว่า 0.5 | 17 คู่ (8.5%) | 82 คู่ (41.0%) |
| อย่างน้อยหนึ่งชื่อมี similarity ต่ำกว่า 0.5 | 69 คู่ (34.5%) | 135 คู่ (67.5%) |
| ทั้ง `username_jaro` และ `fullname_jaro` ตั้งแต่ 0.9 ขึ้นไป | 10 คู่ (5.0%) | 3 คู่ (1.5%) |
| มีภาพอย่างน้อยหนึ่งฝั่ง | 51 คู่ (25.5%) | 71 คู่ (35.5%) |
| มี caption signal | 20 คู่ (10.0%) | 20 คู่ (10.0%) |

[แทรกรูปที่ 4.15 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_error_analysis_fp_fn.png
คำบรรยาย: รูปที่ 4.15 การวิเคราะห์ข้อผิดพลาดจากตัวอย่าง false positives และ false negatives แสดงการกระจุกตัวตามคู่แพลตฟอร์มและลักษณะของสัญญาณชื่อ ภาพ และ caption ที่เกี่ยวข้องกับความผิดพลาด

ผลในตารางที่ 4.11 และรูปที่ 4.15 ให้ข้อสังเกตสำคัญ 2 ประเด็น ประเด็นแรก `false negatives` กระจุกตัวอยู่ในคู่ `googleplus -> twitter` อย่างเด่นชัด และส่วนใหญ่มีสัญญาณจากชื่ออ่อนมาก กล่าวคือในตัวอย่าง `67.5%` มีอย่างน้อยหนึ่งชื่อที่ similarity ต่ำกว่า `0.5` และ `41.0%` มีทั้ง `username` และ `fullname` ต่ำกว่า `0.5` พร้อมกัน ซึ่งสะท้อนว่าข้อผิดพลาดประเภทนี้มักเกิดเมื่อผู้ใช้เปลี่ยนชื่อผู้ใช้หรือชื่อแสดงผลข้ามแพลตฟอร์มจนสัญญาณแบบ name-based แทบไม่เหลือ

ประเด็นที่สอง `false positives` จำนวนมากกลับเกิดในบริบทตรงข้าม คือเป็นคู่ลบที่ดูเหมือนคู่จริงมากเกินไป โดยเฉพาะกลุ่ม `hard negatives` ที่มีชื่อหรือชื่อผู้ใช้คล้ายกันมาก ตัวอย่างเช่นกรณี `mitchellhall / mitchell hall` และ `pullbackes / pullbackes` ซึ่งได้คะแนนสูงมากแม้เป็นคนละ `user_folder` จริง รูปแบบนี้อธิบายได้ว่าทำไมระบบจึงยังต้องมี `REVIEW` tier เพื่อกันคู่ที่ดูน่าใช่มาก แต่ยังไม่ปลอดภัยพอสำหรับการ merge อัตโนมัติในทุกกรณี

เมื่อพิจารณาร่วมกับผล feature importance จะเห็นว่าข้อผิดพลาดทั้งสองแบบสอดคล้องกับธรรมชาติของแบบจำลอง กล่าวคือระบบเก่งมากในกรณีที่มีสัญญาณจากชื่อชัด แต่ยังมีข้อจำกัดเมื่อชื่ออ่อนเกินไปหรือเมื่อคู่ลบมีชื่อเหมือนกันผิดปกติ ดังนั้นข้อเสนอเชิงเทคนิคสำหรับการพัฒนาต่อจึงไม่ใช่เพียงการปรับ threshold เท่านั้น แต่ควรเพิ่ม coverage ของหลักฐานเสริมที่ช่วยแยกสองกรณีนี้ได้ เช่น image signal ที่มีทั้งสองฝั่ง, caption-context ที่สมบูรณ์ขึ้น หรือ contextual features อื่นที่ช่วยยืนยันตัวตนเมื่อสัญญาณจากชื่อไม่เพียงพอ

## 4.6 ความสำคัญของคุณลักษณะทั้ง 41 ตัวที่สนับสนุนการตัดสินใจ

เพื่อทำความเข้าใจว่าระบบอาศัยหลักฐานประเภทใดในการตัดสินใจ ผู้วิจัยวิเคราะห์ feature importance ของ main run ซึ่งใช้ feature ทั้งหมด `41` ตัว ผลที่ได้ชี้ชัดว่าระบบยังพึ่งพาสัญญาณจากชื่อเป็นหลักอย่างมาก โดยเฉพาะกลุ่ม `fullname_*` และ `username_*` ขณะที่สัญญาณจาก `bio`, `style` และ `image-cross-signal` ทำหน้าที่เป็นหลักฐานเสริมมากกว่าจะเป็นแกนหลักของการตัดสินใจ

ตารางที่ 4.12 คุณลักษณะที่สำคัญสูงสุดของ main run

| อันดับ | Feature | Importance |
| --- | --- | ---: |
| 1 | `fullname_token_sort` | 0.4165 |
| 2 | `username_token_sort` | 0.2263 |
| 3 | `fullname_jaro` | 0.1561 |
| 4 | `bio_tfidf_cosine` | 0.0588 |
| 5 | `username_jaro` | 0.0464 |
| 6 | `fullname_lev` | 0.0409 |
| 7 | `bio_sbert_cosine` | 0.0183 |
| 8 | `username_lev` | 0.0089 |
| 9 | `platform_pair_code` | 0.0076 |
| 10 | `style_biolen_ratio` | 0.0040 |

หากรวมความสำคัญตามตระกูลของ feature จะพบว่า `Name similarity` ครองสัดส่วนรวมถึง `89.51%` ของความสำคัญทั้งหมด ขณะที่ `Bio text` มี `7.71%`, `Style / writing` มี `1.21%`, `Platform pair` มี `0.76%` และ `Image-caption cross-signal` มี `0.65%` ที่สำคัญคือใน `41` features นี้มีถึง `18` ตัวที่ importance เป็นศูนย์ใน main run

[แทรกรูปที่ 4.16 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_feature_importance_all41.png
คำบรรยาย: รูปที่ 4.16 ความสำคัญของคุณลักษณะทั้ง 41 ตัวใน main run แสดงให้เห็นว่า feature กลุ่มชื่อและ username เป็นหลักฐานหลักของระบบ ขณะที่ feature จำนวนหนึ่งมีผลต่อการตัดสินใจต่ำมากหรือเป็นศูนย์ในรอบนี้

[แทรกรูปที่ 4.17 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_feature_family_importance.png
คำบรรยาย: รูปที่ 4.17 การรวม feature importance ตามตระกูลหลัก แสดงสัดส่วนของหลักฐานจากชื่อ bio style image-cross-signal และหลักฐานประเภทอื่นที่แบบจำลองใช้จริงใน main run

อย่างไรก็ตาม ผลในระดับ feature importance ต้องตีความร่วมกับข้อจำกัดของข้อมูลภาพด้วย โดย valid profiles ทั้งหมด `36,804` โปรไฟล์ มีเพียง `4,248` โปรไฟล์ที่มี local image, `2,173` โปรไฟล์ที่มี image metadata และ `2,173` โปรไฟล์ที่มี caption ใช้งานได้ อีกทั้งใน test pairs มีคู่ที่มีภาพอย่างน้อยหนึ่งฝั่ง `1,873` คู่ แต่ไม่มีคู่ที่มี local image พร้อมกันทั้งสองฝั่งเลย

[แทรกรูปที่ 4.18 ที่นี่]
ไฟล์รูป: d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_image_coverage.png
คำบรรยาย: รูปที่ 4.18 Coverage ของข้อมูลภาพใน main run แยกระดับ profile และ test pair แสดงให้เห็นว่าการทดลองในรอบนี้ยังเป็น partial multimodal setting มากกว่า full image-to-image matching

ผลในรูปที่ 4.18 อธิบายได้ว่าทำไม image-related features จำนวนมากยังมีน้ำหนักต่ำกว่ากลุ่มชื่อและข้อความอย่างชัดเจน กล่าวคือ image branch ช่วยเพิ่มคุณภาพของ main run จริง แต่ยังอยู่ในฐานะสัญญาณเสริมภายใต้ coverage ที่จำกัด

## 4.7 สรุปผลของ Project 1 ในขอบเขต Decision Thresholding

ผลการวิจัยในบทนี้ยืนยันว่าแกนหลักของ Project 1 คือ `decision thresholding` ที่ทำงานต่อจาก `exact-first + calibrated candidate scoring` โดยใช้ main run แบบ `image_context_r075_h20_s42` เป็นตัวให้คะแนนคู่โปรไฟล์ แล้วแปลงคะแนนนั้นไปเป็นกฎการตัดสินใจใน workflow จริงของ CRM

หากสรุปในเชิง implementation ที่พร้อมนำไปใช้ต่อได้โดยตรง สามารถสรุปได้ดังนี้

1. ใช้ `image_context_r075_h20_s42` เป็น candidate scorer หลัก เพราะให้สมดุลดีที่สุดในด้าน `AP`, `ROC-AUC` และ `Recall` บน feature space `41` ตัว
2. ใช้ `IsotonicRegression` เพื่อ calibrate ความน่าจะเป็นก่อนตั้ง threshold ปลายทาง เพื่อให้คะแนนมีความหมายเชิงปฏิบัติการมากขึ้น
3. ใช้กฎ `exact-first` เพื่อดึงคู่ที่ชัดมากออกมา merge อัตโนมัติก่อน แล้วค่อยใช้ calibrated score กับคู่ที่ยังมีความกำกวม
4. ใช้ thresholds `0.98` สำหรับ `MATCH` และ `0.95` สำหรับ `REVIEW` เพื่อควบคุมความเสี่ยงของ false merge ในบริบท CRM
5. ที่ operating point นี้ ระบบได้ `final match-only precision = 0.9550` และ `final match-only recall = 0.6711` พร้อม `review queue = 86,296` คู่ ซึ่งสะท้อนสมดุลที่ยอมรับได้ระหว่างการ merge อัตโนมัติและการส่งคู่กำกวมให้มนุษย์ตรวจสอบต่อ

ดังนั้น ข้อสรุปสุดท้ายของ Project 1 ไม่ใช่เพียงว่าแบบจำลองให้ metric ดีบน test split แต่คือระบบสามารถแปลงคะแนนของแบบจำลองให้กลายเป็นกระบวนการตัดสินใจแบบ `MATCH / REVIEW / NO_MATCH` ที่ใช้งานได้จริงใน CRM โดยรักษาสมดุลระหว่างการ merge อัตโนมัติและการส่งคู่กำกวมให้มนุษย์ตรวจสอบต่ออย่างเป็นระบบ
