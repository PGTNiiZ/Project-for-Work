# Prompt พร้อมใช้สำหรับสร้างบทที่ 4 ลง Word

คัดลอก prompt ด้านล่างไปใช้กับระบบที่สามารถสร้างไฟล์ Word หรือเอกสาร `.docx` ได้โดยตรง

---

คุณเป็นผู้ช่วยเขียนรายงานวิจัยภาษาไทยเชิงวิชาการ ให้สร้างเอกสาร Word บทที่ 4 จากไฟล์ต้นฉบับในเครื่อง โดยห้ามแต่งตัวเลขหรือสร้างผลลัพธ์ขึ้นเองเด็ดขาด ทุกตัวเลขต้องอ้างอิงจากไฟล์ที่ระบุเท่านั้น

## งานที่ต้องทำ

สร้างเอกสาร Word ชื่อ `บทที่4_ผลการวิจัยและอภิปรายผล.docx` โดยใช้โครงและเนื้อหาจากไฟล์ต่อไปนี้เป็นแหล่งข้อมูลหลัก

1. ต้นฉบับบทที่ 4 พร้อม marker จุดแทรกรูป  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\doc\ch04_results_formal_th.md`

2. แผนรูปประกอบบทที่ 4  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\doc\ch04_figure_plan_th.md`

3. ผล benchmark รายโมเดลของสาย Classical Leakage-Safe  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\res\classical_leaksafe_cmp.csv`

4. benchmark model families  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\res\model_family_cmp.csv`

5. strict neural/classical reference comparison  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\res\model_strict_cmp.csv`

6. ผลของ multimodal suite  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\train_data\stage7_13_multimodal_suite\reports\leaderboard.csv`

7. ผล production pipeline  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\train_data\stage7_14_full_candidate_pipeline\reports\full_pipeline_report.json`
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\train_data\stage7_14_full_candidate_pipeline\reports\operating_points.json`

8. ผล CRM pipeline  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\train_data\stage15_crm_entity_pipeline\reports\crm_entity_report.json`

## ข้อกำหนดสำคัญ

1. ใช้โครงหัวข้อ exactly ตามนี้
   - 4.1 ผลการเตรียมข้อมูล
   - 4.1.1 ผลการรวมและทำความสะอาดข้อมูล
   - 4.1.2 ผลการสร้าง labeled pairs และ feature matrix
   - 4.1.3 ผลการแบ่งชุดข้อมูลแบบ leakage-safe
   - 4.2 ผลการทดลองสาย Classical Leakage-Safe
   - 4.2.1 Logistic Regression
   - 4.2.2 Gradient Boosting
   - 4.2.3 Random Forest
   - 4.2.4 สรุปเปรียบเทียบผลของสาย Classical
   - 4.3 ผลการทดลองสาย Multimodal Suite
   - 4.3.1 Text/Attribute Hybrid
   - 4.3.2 Image Statistics
   - 4.3.3 Image Context
   - 4.3.4 สรุปเปรียบเทียบผลของสาย Multimodal
   - 4.4 ผลการทดลองสาย Neural-Network Reference
   - 4.4.1 IdentityMLP ภายใต้ strict setting
   - 4.4.2 เปรียบเทียบกับสาย Classical และ Multimodal
   - 4.5 สรุปผลการเปรียบเทียบทุกสายการทดลอง

2. ใช้ข้อความจากไฟล์ `ch04_results_formal_th.md` เป็นฐานหลักในการเขียน และต้องรักษาตัวเลขให้ตรง

3. ในจุดที่พบ marker รูปในลักษณะนี้
   - `[แทรกรูปที่ 4.x ที่นี่]`
   - `ไฟล์รูป: ...`
   - `คำบรรยาย: ...`

   ให้แทรกรูปตาม path ที่ระบุลงในเอกสาร Word จริง โดยจัดวางรูปกึ่งกลางหน้าและใส่คำบรรยายใต้ภาพตามข้อความที่ให้ไว้

4. หากพบ marker แบบ
   - `[อ้างถึงรูปที่ ... ในย่อหน้านี้]`
   - `[เน้นอ้างถึงรูปที่ ... ซ้ำในย่อหน้านี้ ไม่ต้องแทรกรูปใหม่]`

   ให้คงสาระในย่อหน้านั้นไว้ แต่ไม่ต้องแทรกรูปซ้ำ ให้เพียงอ้างถึงเลขรูปในประโยคให้กลมกลืน

5. ตารางทุกตารางต้องจัดเป็นตารางจริงใน Word ไม่ใช่แปลงเป็นข้อความธรรมดา

6. ห้ามใส่ข้อมูลที่ไม่มีในไฟล์ต้นทาง เช่น
   - ห้ามสร้าง ROC, confusion matrix หรือ precision ที่ไม่มีตัวเลขรองรับ
   - ห้ามเปลี่ยน threshold
   - ห้ามแก้จำนวน profiles, pairs, candidates, review queue หรือ unified profiles

7. โทนภาษา:
   - ภาษาไทยเชิงวิชาการ
   - กระชับแต่สมบูรณ์
   - อธิบายความหมายของผล ไม่ใช่เพียงคัดลอกตัวเลข
   - หลีกเลี่ยงภาษาพูด

## วิธีจัดรูปแบบใน Word

1. ใช้ Heading 1 สำหรับ “บทที่ 4 ผลการวิจัยและอภิปรายผล”
2. ใช้ Heading 2 สำหรับหัวข้อ `4.1`, `4.2`, `4.3`, `4.4`, `4.5`
3. ใช้ Heading 3 สำหรับหัวข้อย่อยระดับ `4.1.1`, `4.2.1` เป็นต้น
4. จัดตารางให้อยู่กึ่งกลางหน้า ความกว้างพอดีกับหน้าเอกสาร
5. รูปทุกภาพให้กึ่งกลางหน้า กว้างพออ่านรายละเอียดได้ชัดเจน
6. คำบรรยายรูปและตารางให้ใช้รูปแบบมาตรฐานเดียวกันทั้งเอกสาร
7. อย่าแสดง path ไฟล์ในเนื้อหาสุดท้ายของเอกสาร Word ให้ใช้ path เพื่อแทรกรูปเท่านั้น

## สิ่งที่ต้องระวังเป็นพิเศษ

1. ต้องแยกให้ชัดระหว่าง
   - threshold สำหรับ test classification ของ main run = `0.35`
   - threshold สำหรับ production workflow = `0.98 / 0.95`

2. ต้องอธิบายให้ชัดว่า
   - pipeline เดิมมีปัญหา leakage
   - ผล classical leak-safe ใช้ข้อมูลและ split ที่สะอาดกว่า
   - multimodal suite เป็นสายทดลองหลักที่ดีที่สุดในปัจจุบัน
   - neural network ถูกใช้เป็น reference line ไม่ใช่ผล production หลัก

3. ต้องคงข้อสรุปสุดท้ายนี้
   - main run ปัจจุบันคือ `image_context_r075_h20_s42`
   - production operating point ที่ใช้งานจริงคือ `0.98 / 0.95`
   - final match-only precision = `0.9550`
   - final match-only recall = `0.6711`
   - review queue = `86,296`
   - unified profiles = `19,799`

## รูปที่ต้องแทรกตามลำดับ

1. รูปที่ 4.1  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_data_prep_summary.png`

2. รูปที่ 4.2  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_split_design.png`

3. รูปที่ 4.3  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\classical_leaksafe_cmp.png`

4. รูปที่ 4.4  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\classical_leaksafe_cm_grid.png`

5. รูปที่ 4.5  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_sbert_gain.png`

6. รูปที่ 4.6  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\suite_cmp.png`

7. รูปที่ 4.7  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\cm_main.png`

8. รูปที่ 4.8  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\feat_imp.png`

9. รูปที่ 4.9  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\modality.png`

10. รูปที่ 4.10  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\model_strict_cmp.png`

11. รูปที่ 4.11  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\model_strict_cm_grid.png`

12. รูปที่ 4.12  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\model_family_cmp.png`

13. รูปที่ 4.13  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\funnel.png`

14. รูปที่ 4.14  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\thr_sweep.png`

15. รูปที่ 4.15  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\tiers_count.png`

16. รูปที่ 4.16  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\tiers_precision.png`

17. รูปที่ 4.17  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_crm_outcomes.png`

18. รูปที่ 4.18  
`d:\66070260-Year3_Term2\Project1\Code\Project-for-Work\pub_multi\fig\ch04_experiment_roadmap.png`

## ผลลัพธ์สุดท้ายที่ต้องส่งออก

1. ไฟล์ Word `.docx` ที่จัดรูปแบบเรียบร้อย
2. เนื้อหาบทที่ 4 ครบทุกหัวข้อ
3. ตารางและรูปแทรกครบตาม marker
4. ไม่เหลือข้อความ marker เช่น `[แทรกรูปที่ ...]` ในเอกสารสุดท้าย

---

ถ้าระบบที่ใช้สร้าง Word ไม่สามารถอ่านไฟล์บนเครื่องได้โดยตรง ให้ใช้ไฟล์ `ch04_results_formal_th.md` เป็นต้นฉบับในการคัดลอกเนื้อหา และใช้รายการรูปใน prompt นี้เป็นแผนแทรกรูปแบบ manual
