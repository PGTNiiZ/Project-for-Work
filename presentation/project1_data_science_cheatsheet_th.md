# Project 1 Data Science Cheat Sheet

เอกสารนี้สรุปว่า Project 1 ควรถูกอธิบายใน “ภาษาของงาน data science” อย่างไร เพื่อให้เวลานำเสนอคุณไม่พูดแค่ว่า “เราเขียนโค้ดอะไร” แต่พูดได้ว่า “เราแก้ปัญหาอะไร ออกแบบการทดลองอย่างไร และผลลัพธ์ตอบโจทย์เชิงวิชาการและเชิงธุรกิจอย่างไร”

## 1. ถ้าต้องอธิบายงานนี้ในประโยคเดียว

งานนี้คือ **cross-platform identity resolution for CRM** หรือการเชื่อมโยงว่าโปรไฟล์จากหลายแพลตฟอร์มเป็นบุคคลเดียวกันหรือไม่ โดยออกแบบเป็น **retrieval + pairwise scoring + decision thresholding pipeline** เพื่อให้ใช้งานได้จริงในบริบท CRM

พูดแบบสั้น:

> งานนี้เป็นงานด้าน Entity Resolution หรือ Record Linkage โดยเราพัฒนา pipeline ที่เริ่มจากการลด search space ด้วย exact-first และ blocking จากนั้นใช้ machine learning สำหรับ pairwise matching และสุดท้ายทำ calibration กับ thresholding เพื่อแปลงคะแนนเป็นการตัดสินใจระดับ production

## 2. ชื่อเรียกงานนี้ในภาษาของ Data Science

ใช้คำเหล่านี้แทนคำอธิบายกว้าง ๆ

- **Entity Resolution (ER)**: การระบุว่า record หลายตัวเป็น entity เดียวกันหรือไม่
- **Record Linkage**: การเชื่อม record ข้ามแหล่งข้อมูล
- **Identity Resolution**: การเชื่อมตัวตนของผู้ใช้/ลูกค้าข้ามแพลตฟอร์ม
- **Cross-Platform Matching**: การจับคู่บัญชีผู้ใช้ข้ามแพลตฟอร์ม
- **Customer 360 / Unified Customer View**: ผลลัพธ์ปลายทางที่รวมข้อมูลลูกค้าเป็นมุมมองเดียว

ประโยคที่ใช้พูด:

> ในเชิง data science งานนี้ไม่ใช่แค่ binary classification ธรรมดา แต่เป็นงาน Entity Resolution ที่ต้องแก้ทั้ง retrieval, matching, calibration และ deployment decision ไปพร้อมกัน

## 3. โครงสร้างงานนี้ในเชิง Methodology

งานนี้ควรถูกเล่าว่าเป็น **end-to-end data science pipeline** ไม่ใช่แค่โมเดลตัวเดียว

### 3.1 Business Understanding

- ปัญหาคือ CRM เห็นลูกค้าคนเดียวเป็นหลาย record
- ผลกระทบคือ segmentation, personalization และ lead prioritization คลาดเคลื่อน
- เป้าหมายไม่ใช่แค่ “ทายคู่ได้ถูก” แต่คือ “รวมโปรไฟล์ให้ใช้จริงได้”

### 3.2 Data Understanding

- ใช้ข้อมูลจาก Twitter, Instagram และ Google+
- ข้อมูลมีความไม่สมบูรณ์สูง
- ชุดข้อมูลมีทั้ง text, metadata และ image-related signals

### 3.3 Data Preparation

- preprocessing และ normalization
- location normalization
- normalized profile database
- image artifact preparation

### 3.4 Modeling

- pair construction
- feature engineering
- model benchmarking
- multimodal comparison

### 3.5 Evaluation

- retrieval evaluation
- classification evaluation
- calibration evaluation
- production evaluation

### 3.6 Deployment Framing

- decision tiers: MATCH / REVIEW / NO_MATCH
- human-in-the-loop
- entity merge
- lead scoring

ประโยคที่ใช้พูด:

> เราวางงานตามกรอบ CRISP-DM เพราะโจทย์นี้ต้องตอบตั้งแต่ business problem ไปจนถึง deployment decision ไม่ใช่หยุดอยู่แค่ขั้น train model

## 4. สิ่งที่คุณ “ทดลอง” จริง ๆ คืออะไร

นี่คือส่วนที่ทำให้โปรเจกต์ดูเป็น data science project มากขึ้น เพราะคุณไม่ได้แค่ “ทำระบบ” แต่ **มีการทดลอง เปรียบเทียบ และคัดเลือกแบบมีเหตุผล**

### 4.1 Retrieval Experiment

คุณทดลองว่า exact-first และ blocking ลดจำนวนคู่ได้มากแค่ไหน โดยยังรักษาคู่จริงไว้ได้มากพอหรือไม่

ศัพท์ที่ควรใช้:

- **Search Space Reduction**
- **Candidate Retrieval**
- **Blocking**
- **Ground-Truth Coverage**

ประโยคที่ใช้พูด:

> ในส่วน retrieval เราประเมินความสามารถของ exact-first และ blocking ในการลด search space จาก 449 ล้านคู่ให้เหลือประมาณ 2.07 ล้านคู่ โดยยังรักษา ground-truth coverage ไว้ที่ 88.67%

### 4.2 Pair Construction Experiment

คุณไม่ได้สร้างแค่คู่บวก แต่สร้างคู่ลบหลายแบบ

ศัพท์ที่ควรใช้:

- **Positive pairs**
- **Random negatives**
- **Hard negatives**
- **Class imbalance**

ประโยคที่ใช้พูด:

> เราออกแบบ pair construction ให้สะท้อน difficulty ของปัญหาจริง โดยใช้ทั้ง positive pairs, random negatives และ hard negatives เพื่อให้โมเดลเรียนรู้ทั้งคู่ลบทั่วไปและคู่ลบที่คล้ายกันมาก

### 4.3 Feature Engineering Experiment

คุณไม่ได้ใช้ข้อมูลดิบตรง ๆ แต่แปลงเป็น pair-level features

ศัพท์ที่ควรใช้:

- **Pairwise feature vector**
- **Similarity features**
- **Feature engineering**
- **Multimodal features**
- **Selected feature set**

ประโยคที่ใช้พูด:

> โมเดลของเราไม่ได้รับ raw profile โดยตรง แต่รับ pairwise feature vector ที่สรุปความเหมือนและความต่างของโปรไฟล์สองฝั่งในหลายมิติ เช่น ชื่อ bio URL location และ image-derived signals

### 4.4 Model Comparison Experiment

คุณเปรียบเทียบหลาย model family

ศัพท์ที่ควรใช้:

- **Model benchmarking**
- **Baseline model**
- **Classical models**
- **Tree-based models**
- **Neural reference model**

ประโยคที่ใช้พูด:

> เรา benchmark หลาย model families ภายใต้ split และ feature space เดียวกัน เพื่อให้การเลือก final model เกิดจาก empirical comparison ไม่ใช่จาก intuition

### 4.5 Multimodal Ablation / Incremental Experiment

คุณไม่ได้พูดแค่ว่า “เพิ่มรูปภาพแล้วดีขึ้น” แต่ควรเรียกสิ่งนี้ว่า **ablation-style comparison** หรือ **incremental feature expansion**

ศัพท์ที่ควรใช้:

- **Ablation study**
- **Incremental improvement**
- **Text-only baseline**
- **Image-enhanced model**
- **Image-context model**

ประโยคที่ใช้พูด:

> เราทำการทดลองแบบ ablation โดยเริ่มจาก text-attribute baseline แล้วเพิ่ม image statistics และ image-context features ทีละขั้น เพื่อดูว่าการเพิ่ม modality ให้ประโยชน์เพิ่มขึ้นจริงหรือไม่

### 4.6 Calibration and Threshold Experiment

ส่วนนี้ทำให้โปรเจกต์ดู mature มากขึ้นในเชิง data science

ศัพท์ที่ควรใช้:

- **Probability calibration**
- **Isotonic Regression**
- **Expected Calibration Error (ECE)**
- **Threshold sweep**
- **Operating point**

ประโยคที่ใช้พูด:

> หลังจากเลือกโมเดลที่ดีที่สุดแล้ว เราไม่ได้ใช้ raw score ตัดสินใจทันที แต่ calibrate ความน่าจะเป็นด้วย Isotonic Regression และวิเคราะห์ threshold trade-off เพื่อหา operating point ที่เหมาะกับ production

## 5. ศัพท์เฉพาะที่ควรใส่บ่อย ๆ

ด้านล่างคือศัพท์ที่ควรแทรกเวลาเล่า

### กลุ่ม Problem Framing

- **Entity Resolution**
  ความหมาย: การระบุว่า record หลายตัวเป็นคนเดียวกันหรือไม่
- **Record Linkage**
  ความหมาย: การเชื่อม record ข้าม dataset หรือข้าม platform
- **Cross-Platform Identity Resolution**
  ความหมาย: การเชื่อมตัวตนข้ามแพลตฟอร์ม

### กลุ่ม Data Preparation

- **Preprocessing**
  ความหมาย: ขั้นเตรียมข้อมูลให้พร้อมใช้
- **Normalization**
  ความหมาย: ทำข้อมูลหลายรูปแบบให้อยู่ในรูปมาตรฐานเดียวกัน
- **Feature Engineering**
  ความหมาย: การสร้างตัวแปรเชิงตัวเลขจากข้อมูลดิบ

### กลุ่ม Pair Building

- **Positive Pair**
  ความหมาย: คู่ที่เป็นคนเดียวกันจริง
- **Negative Pair**
  ความหมาย: คู่ที่ไม่ใช่คนเดียวกัน
- **Hard Negative**
  ความหมาย: คู่ลบที่คล้ายกันมากและทำให้โมเดลสับสนได้

### กลุ่ม Retrieval

- **Exact Match Baseline**
  ความหมาย: กฎตรงตัวที่ใช้จับคู่ก่อนเข้าโมเดล
- **Blocking**
  ความหมาย: วิธีลดจำนวนคู่ที่ต้องพิจารณา
- **Candidate Retrieval**
  ความหมาย: ขั้นดึงคู่ที่ควรให้โมเดลตรวจต่อ
- **Search-Space Reduction**
  ความหมาย: การลดจำนวนคู่จากจักรวาลทั้งหมดลงเหลือชุดที่พอประมวลผลได้
- **Ground-Truth Coverage**
  ความหมาย: สัดส่วนของคู่จริงที่ยังรอดเข้ามาใน candidate set

### กลุ่ม Modeling

- **Pairwise Scoring Model**
  ความหมาย: โมเดลที่ให้คะแนนว่าคู่ข้อมูลน่าจะ match กันแค่ไหน
- **Leak-Safe Split**
  ความหมาย: การแบ่ง train/val/test แบบไม่ให้ identity เดียวกันรั่วข้าม split
- **Identity Leakage**
  ความหมาย: การรั่วของข้อมูล identity เดียวกันข้าม split จนทำให้ metric สูงเกินจริง
- **Benchmarking**
  ความหมาย: การเปรียบเทียบหลายโมเดลภายใต้เงื่อนไขเดียวกัน
- **Ablation Study**
  ความหมาย: การเพิ่มหรือลดองค์ประกอบของโมเดลเพื่อดูผลของแต่ละส่วน

### กลุ่ม Evaluation

- **Average Precision (AP)**
  ความหมาย: วัดคุณภาพการจัดอันดับ โดยเหมาะกับ class imbalance สูง
- **ROC-AUC**
  ความหมาย: วัดความสามารถในการแยก positive กับ negative โดยรวม
- **Precision**
  ความหมาย: ทายว่า match แล้วถูกจริงมากแค่ไหน
- **Recall**
  ความหมาย: ระบบเก็บคู่จริงได้มากแค่ไหน
- **F1-score**
  ความหมาย: ค่าเฉลี่ยเชิงสมดุลของ precision กับ recall
- **Confusion Matrix**
  ความหมาย: ตารางสรุป TP, FP, TN, FN

### กลุ่ม Production

- **Probability Calibration**
  ความหมาย: การทำให้ score ของโมเดลใกล้เคียง probability จริงมากขึ้น
- **Threshold Sweep**
  ความหมาย: การไล่ดู performance เมื่อเปลี่ยน threshold
- **Operating Point**
  ความหมาย: threshold ที่เลือกใช้จริง
- **Decision Thresholding**
  ความหมาย: การแปลง score ให้เป็น decision
- **Human-in-the-Loop**
  ความหมาย: ให้มนุษย์ช่วยตรวจคู่กำกวม
- **Review Queue**
  ความหมาย: คิวของคู่ข้อมูลที่ยังไม่ควร auto-merge
- **False Merge Risk**
  ความหมาย: ความเสี่ยงจากการรวมคนละคนเป็นคนเดียวกัน

## 6. คำพูดที่ทำให้ Presentation ดูเป็น Data Science มากขึ้น

แทนที่จะพูดว่า:

> เราลองหลายแบบแล้วตัวนี้ดีที่สุด

ให้พูดว่า:

> เรา benchmark หลาย model families ภายใต้ feature space และ split เดียวกัน แล้วเลือก final model จาก empirical performance บน validation และ test sets

แทนที่จะพูดว่า:

> เราเพิ่มรูปภาพเข้าไป

ให้พูดว่า:

> เราทำ incremental multimodal feature expansion โดยเริ่มจาก text-attribute baseline แล้วเพิ่ม image statistics และ caption-context signals เพื่อประเมิน marginal gain ของแต่ละ modality

แทนที่จะพูดว่า:

> เราตั้ง threshold เอง

ให้พูดว่า:

> เราวิเคราะห์ threshold trade-off หลัง probability calibration เพื่อหา operating point ที่เหมาะกับบริบท production ซึ่งต้องคุม false merge risk มากกว่าการ maximize F1 เพียงอย่างเดียว

แทนที่จะพูดว่า:

> เราแยก train test

ให้พูดว่า:

> เราใช้ leak-safe entity-level split เพื่อป้องกัน identity leakage และลด optimistic bias ในการประเมินผล

แทนที่จะพูดว่า:

> เราใช้ชื่อกับ bio มาช่วยกัน

ให้พูดว่า:

> เราสร้าง pairwise similarity features จาก identity strings, semantic bio signals, URL/domain overlap, stylometric cues และ image-derived context แล้วป้อนเข้า tabular scoring model

## 7. ประโยคพร้อมใช้ตอนอธิบายสิ่งที่คุณทำ

### เวอร์ชันเปิดโปรเจกต์

> โปรเจกต์นี้อยู่ในกลุ่มปัญหา Entity Resolution หรือ Record Linkage โดยมีเป้าหมายเพื่อเชื่อมโยงโปรไฟล์ผู้ใช้ข้ามแพลตฟอร์มให้กลายเป็น unified customer view สำหรับระบบ CRM

### เวอร์ชันอธิบายวิธีทำ

> ในเชิงระบบ เราออกแบบ pipeline แบบ retrieval-first เริ่มจาก exact-first และ blocking เพื่อลด search space จากนั้นจึงใช้ pairwise scoring model ให้คะแนน candidate pairs แล้วทำ calibration และ thresholding เพื่อแปลงคะแนนเป็น MATCH, REVIEW และ NO_MATCH

### เวอร์ชันอธิบายการทดลอง

> ในเชิงการทดลอง เราเปรียบเทียบหลาย model families และทำ ablation-style comparison กับ feature sets หลายระดับ ตั้งแต่ text-attribute baseline ไปจนถึง multimodal image-context model เพื่อประเมินว่าการเพิ่ม modality ให้ประโยชน์จริงหรือไม่

### เวอร์ชันอธิบายเรื่อง leakage

> จุดสำคัญของงานนี้คือการออกแบบ leak-safe split ในระดับ entity เพื่อหลีกเลี่ยง identity leakage ซึ่งเป็นสาเหตุที่ทำให้ metric สูงเกินจริงในงาน record linkage ได้ง่ายมาก

### เวอร์ชันอธิบายผล production

> เมื่อได้ calibrated probabilities แล้ว เราเลือก operating point ที่เหมาะกับบริบท CRM ซึ่งให้ความสำคัญกับการลด false merge มากกว่าการ maximize recall เพียงอย่างเดียว จึงออกแบบ review tier ไว้รองรับคู่กำกวมโดยเฉพาะ

## 8. ถ้าอาจารย์ถามว่า “แล้วนี่มันใช่ Data Science ตรงไหน”

ให้ตอบแบบนี้:

> ความเป็น data science ของงานนี้อยู่ที่ 4 ส่วนหลัก
>
> 1. เรามี problem framing ชัดว่าเป็น entity resolution สำหรับ CRM
> 2. เรามีการเตรียมข้อมูลและสร้าง feature อย่างเป็นระบบ
> 3. เรามีการออกแบบการทดลอง เปรียบเทียบโมเดล และทำ ablation / benchmarking
> 4. เรามีการประเมินผลหลายระดับตั้งแต่ retrieval, model performance, calibration ไปจนถึง production operating point

## 9. ถ้าอาจารย์ถามว่า “สิ่งที่ทำมามันออกจากการทดลองยังไง”

ให้ตอบแบบนี้:

> สิ่งที่สรุปเป็น pipeline สุดท้ายไม่ได้มาจากการเลือกแบบ intuition แต่เกิดจากผลการทดลองหลายชั้น ได้แก่
>
> - การทดลอง retrieval ว่า exact-first และ blocking ลด search space ได้มากแค่ไหน
> - การทดลอง pair construction ที่ใช้ทั้ง random negatives และ hard negatives
> - การ benchmark model families หลายแบบ
> - การ ablation เปรียบเทียบ text-only, image-stats และ image-context
> - การ calibration และ threshold sweep เพื่อหา operating point สำหรับ production

## 10. แก่นของงานนี้ที่ควรจำให้ได้

จำ 5 ข้อนี้ให้แม่นที่สุด

1. งานนี้คือ **Entity Resolution / Record Linkage**
2. ระบบนี้เป็น **Retrieval + Scoring + Decision Pipeline**
3. คุณมี **Leak-Safe Experimental Design**
4. คุณทำ **Model Benchmarking + Ablation Study**
5. คุณสรุปผลในระดับ **Production Operating Point** ไม่ใช่แค่ Test Metric

ถ้าพูด 5 ประเด็นนี้ได้ งานของคุณจะดูเป็น project data science มากขึ้นทันที
