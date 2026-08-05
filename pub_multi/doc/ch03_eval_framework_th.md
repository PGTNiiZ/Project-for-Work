# รูปที่ 3.8 Evaluation Framework

ไฟล์ภาพหลักอยู่ที่ [ch03_eval_framework.svg](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/fig/ch03_eval_framework.svg)

รูปนี้ใช้ประกอบหัวข้อ `3.5 การประเมินผล` เพื่อแสดงให้เห็นว่าการประเมินในงานวิจัยนี้ไม่ได้มีเพียงชั้นเดียว แต่แบ่งเป็น 4 ชั้นที่ต่อเนื่องกัน ได้แก่

1. Retrieval evaluation
2. Model evaluation
3. Calibration evaluation
4. Production evaluation

caption ที่แนะนำ:

“รูปที่ 3.8 กรอบการประเมินผลของระบบ แสดงการประเมินคุณภาพของ pipeline ใน 4 ระดับ ได้แก่ retrieval, model, calibration และ production workflow เพื่อให้การสรุปผลสะท้อนทั้งประสิทธิภาพเชิงสถิติและความพร้อมสำหรับการใช้งานจริง”

จุดประสงค์ของรูปนี้คือช่วยให้ผู้อ่านเข้าใจว่า metric แต่ละตัวถูกใช้ในขั้นใดของระบบ เช่น `search-space reduction` และ `coverage` ใช้ใน retrieval, `Precision/Recall/F1-score/AP/AUC` ใช้ใน model evaluation, `ECE` ใช้ใน calibration และ `review queue size`, `final match-only precision/recall`, `unified profiles` ใช้ใน production evaluation
