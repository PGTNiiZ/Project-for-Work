# แผนการใส่รูปประกอบบทที่ 4

เอกสารนี้จัดรูปประกอบของบทที่ 4 ให้ตรงกับโครงใหม่ของบท `4.1 ผลการเตรียมข้อมูล`, `4.2 Classical Leakage-Safe`, `4.3 Multimodal Suite`, `4.4 Neural-Network Reference` และ `4.5 สรุปผลรวม` โดยยึดหลักว่าแต่ละหัวข้อหลักต้องมีภาพที่ช่วยอธิบายความหมายของผล ไม่ใช่มีเพียงตารางตัวเลข

## 4.1 ผลการเตรียมข้อมูล

### รูปที่ 4.1 Data Preparation Summary

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.1.1

สิ่งที่รูปควรแสดง  
จำนวน valid profiles แยกตามแพลตฟอร์ม จำนวนแถวของ leak-safe supervised split และการเติบโตของ feature space จาก classical ไป multimodal

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผลลัพธ์หลังเตรียมข้อมูลจริง ทั้งในมิติของจำนวนข้อมูลที่พร้อมใช้ โครงสร้างชุด train/validation/test และจำนวนคุณลักษณะที่ถูกใช้จริงในแต่ละสายการทดลอง

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/ch04_data_prep_summary.png`

### รูปที่ 4.2 Split Design และองค์ประกอบของคู่ข้อมูล

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.1.3

สิ่งที่รูปควรแสดง  
จำนวนคู่ข้อมูลใน `train`, `val`, `test` แยกเป็น positive, random negative และ hard negative โดยเปรียบเทียบระหว่าง rebuilt baseline กับ main multimodal run

คำอธิบายใต้ภาพที่ควรใช้  
แสดงว่าการเพิ่ม modality ไม่ได้เปลี่ยนแนวคิดของ split แต่เปลี่ยนเฉพาะ feature space บนชุดคู่ข้อมูลที่คุม leakage แล้ว

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/ch04_split_design.png`

## 4.2 ผลการทดลองสาย Classical Leakage-Safe

### รูปที่ 4.3 Classical Model Comparison

ตำแหน่งที่ควรใส่  
วางก่อนหัวข้อ 4.2.1 หรือท้ายหัวข้อ 4.2.4

สิ่งที่รูปควรแสดง  
เปรียบเทียบ `Logistic Regression`, `Gradient Boosting` และ `Random Forest` บน `AP`, `AUC`, `F1`, `Precision`, `Recall`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผล benchmark จริงของสาย classical leakage-safe บน feature set เดียวกันและ split เดียวกัน เพื่อชี้ว่า `Gradient Boosting` ให้สมดุลดีที่สุด

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/classical_leaksafe_cmp.png`

### รูปที่ 4.4 Classical Confusion Matrix Grid

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.2.4

สิ่งที่รูปควรแสดง  
confusion matrix ของ `Logistic Regression`, `Gradient Boosting` และ `Random Forest` ในรูปเดียว

คำอธิบายใต้ภาพที่ควรใช้  
แสดงรูปแบบข้อผิดพลาดของแต่ละโมเดลในสาย classical leakage-safe เพื่อให้เห็นว่าทำไม `Gradient Boosting` จึงเหมาะเป็น baseline หลัก

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/classical_leaksafe_cm_grid.png`

## 4.3 ผลการทดลองสาย Multimodal Suite

### รูปที่ 4.5 SBERT Gain

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.3.1

สิ่งที่รูปควรแสดง  
ผลต่างของ `AP`, `AUC`, `F1`, `Precision`, `Recall` ระหว่าง tuned baseline กับ hybrid text ที่เพิ่ม `SBERT`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงว่าการเพิ่ม semantic text encoder ช่วยเพิ่ม recall และยกระดับประสิทธิภาพโดยรวมของสาย hybrid text

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/ch04_sbert_gain.png`

### รูปที่ 4.6 Multimodal Suite Comparison

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.3.3 หรือเปิดหัวข้อ 4.3.4

สิ่งที่รูปควรแสดง  
เปรียบเทียบ `text_attr_hybrid`, `image_stats`, `image_context` บน metrics หลัก

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผลของการเพิ่มภาพและ caption-cross signals ต่อประสิทธิภาพของระบบภายใต้ split เดียวกัน

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/suite_cmp.png`

### รูปที่ 4.7 Confusion Matrix ของ Main Run

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.3.3

สิ่งที่รูปควรแสดง  
confusion matrix ของ `image_context_r075_h20_s42`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผลการจำแนกของ main run ที่ถูกเลือกใช้ต่อใน production pipeline

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/cm_main.png`

### รูปที่ 4.8 Feature Importance ของ Main Run

ตำแหน่งที่ควรใส่  
วางกลางหัวข้อ 4.3.4

สิ่งที่รูปควรแสดง  
Top features ของ `image_context_r075_h20_s42`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงว่ากลุ่มชื่อยังเป็นสัญญาณหลักของระบบ ขณะที่ image-context signals ทำหน้าที่เป็นสัญญาณเสริมที่ช่วยยกระดับผล

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/feat_imp.png`

### รูปที่ 4.9 Modality Coverage

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.3.4

สิ่งที่รูปควรแสดง  
coverage ของ local image, image metadata และ caption ในข้อมูลจริง

คำอธิบายใต้ภาพที่ควรใช้  
แสดงข้อจำกัดของข้อมูลภาพที่ทำให้การทดลองครั้งนี้ยังเป็น partial multimodal setting

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/modality.png`

## 4.4 ผลการทดลองสาย Neural-Network Reference

### รูปที่ 4.10 Strict Reference Comparison

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.4.1

สิ่งที่รูปควรแสดง  
เปรียบเทียบ `IdentityMLP` กับ strict rerun models บน feature set `22` ตัวเดิม

คำอธิบายใต้ภาพที่ควรใช้  
แสดงว่า neural reference line ใน strict setting ไม่ได้ให้ข้อได้เปรียบเหนือ classical models โดยอัตโนมัติ

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/model_strict_cmp.png`

### รูปที่ 4.11 IdentityMLP Strict Confusion Matrix

ตำแหน่งที่ควรใส่  
วางต่อจากรูปที่ 4.10 หากมีพื้นที่

สิ่งที่รูปควรแสดง  
confusion matrix ของ `IdentityMLP` ภายใต้ strict reference setting

คำอธิบายใต้ภาพที่ควรใช้  
แสดงรูปแบบข้อผิดพลาดของ `IdentityMLP` ในเงื่อนไข strict setting เพื่ออธิบายว่าเหตุใดค่า AP และ F1 จึงยังตามหลัง tree-based reruns

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/strict_identitymlp_cm.png`

### รูปที่ 4.12 Model Family Benchmark

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.4.2

สิ่งที่รูปควรแสดง  
เปรียบเทียบ `GB`, `RF`, `MLP` และโมเดลอื่นบน final `41-feature split`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงว่าแม้ `MLP` จะทำได้ดีขึ้นบน split ที่สะอาด แต่ `Gradient Boosting` ยังให้สมดุลดีที่สุดเมื่อเทียบกับทุก model family

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/model_family_cmp.png`

## 4.5 สรุปผลการเปรียบเทียบทุกสายการทดลอง

### รูปที่ 4.13 Retrieval Funnel

ตำแหน่งที่ควรใส่  
วางต้นหัวข้อ 4.5

สิ่งที่รูปควรแสดง  
การลด search space จาก all cross-platform pairs ไปสู่ exact pairs, candidate pairs, match-only และ review queue

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผลของ exact-first และ blocking ที่ช่วยให้ระบบใช้งานได้จริงใน candidate pool ขนาดใหญ่

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/funnel.png`

### รูปที่ 4.14 Threshold Behavior ของ Main Run

ตำแหน่งที่ควรใส่  
วางกลางหัวข้อ 4.5

สิ่งที่รูปควรแสดง  
พฤติกรรมของ precision, recall และ F1 เมื่อเปลี่ยน threshold

คำอธิบายใต้ภาพที่ควรใช้  
อธิบายว่าทำไม threshold สำหรับ test classification (`0.35`) จึงไม่ใช่ threshold ที่เหมาะสำหรับ production workflow

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/thr_sweep.png`

### รูปที่ 4.15 Production Tier Counts

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 4.5 ก่อนสรุปย่อหน้า final run

สิ่งที่รูปควรแสดง  
จำนวนคู่ของ tier `exact`, `match` และ `review`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงขนาดของแต่ละ tier ภายใต้ operating point แบบ `0.98 / 0.95` ใน production

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/tiers_count.png`

### รูปที่ 4.16 Production Tier Precision

ตำแหน่งที่ควรใส่  
วางถัดจากรูปที่ 4.15

สิ่งที่รูปควรแสดง  
precision ของ tier `exact`, `match` และ `review`

คำอธิบายใต้ภาพที่ควรใช้  
แสดง trade-off ด้านความแม่นยำของแต่ละ tier ภายใต้ operating point แบบ `0.98 / 0.95`

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/tiers_precision.png`

### รูปที่ 4.17 CRM Output Summary

ตำแหน่งที่ควรใส่  
วางย่อหน้าปิดของหัวข้อ 4.5

สิ่งที่รูปควรแสดง  
decision breakdown (`MATCH/REVIEW/NO_MATCH`) และ lead tier breakdown (`HOT/WARM/COLD`)

คำอธิบายใต้ภาพที่ควรใช้  
แสดงผลลัพธ์ปลายทางของ production pipeline หลัง merge entity และทำ lead scoring

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/ch04_crm_outcomes.png`

### รูปที่ 4.18 Experiment Roadmap

ตำแหน่งที่ควรใส่  
วางท้ายบทเป็นภาพสรุปหรือย้ายไปเปิดบทก็ได้

สิ่งที่รูปควรแสดง  
ลำดับรอบทดลองตั้งแต่ `R1` ถึง production stage

คำอธิบายใต้ภาพที่ควรใช้  
สรุปเส้นทางการพัฒนางานจาก leakage diagnosis ไปสู่ final multimodal run และ production deployment

ไฟล์รูปที่ควรใช้  
`pub_multi/fig/ch04_experiment_roadmap.png`

## หมายเหตุการใช้งาน

1. รูปทุกใบในแผนนี้อ้างอิงจาก artifact จริงในโค้ดหรือสร้างจากผลรันจริงเท่านั้น
2. หากต้องลดจำนวนรูปลง ควรเก็บอย่างน้อย `4.1`, `4.3`, `4.4`, `4.6`, `4.10`, `4.12`, `4.13`, `4.15`, `4.16`
3. สำหรับหัวข้อย่อยของแต่ละโมเดล สามารถใช้รูป comparison ร่วมกันได้ แต่ในเนื้อหาควรชี้ให้ชัดว่ากำลังอธิบาย panel หรือ series ใดของรูป
