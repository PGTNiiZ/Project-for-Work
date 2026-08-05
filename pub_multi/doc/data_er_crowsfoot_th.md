# แผนภาพ ER แบบ Crow's-Foot (เวอร์ชันเรียบง่าย)

ไฟล์ภาพหลักอยู่ที่ [data_er_crowsfoot.svg](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/fig/data_er_crowsfoot.svg)

ภาพนี้เป็น ER Diagram แบบเรียบง่ายที่เน้นเฉพาะตารางหลักในสาย production/CRM ซึ่งมีความสัมพันธ์เชิง PK/FK ชัดเจน เหมาะสำหรับใส่เล่มวิชาการมากกว่าภาพ flow ที่เน้น pipeline

ตารางที่แสดงในภาพประกอบด้วย

- `NORMALIZED_PROFILE`
- `MATCH_DECISION`
- `REVIEW_QUEUE`
- `PROFILE_MAPPING`
- `UNIFIED_PROFILE`
- `LEAD_SCORE`

เหตุผลที่เลือกเฉพาะ 6 ตารางนี้คือเป็นส่วนของระบบที่สามารถอธิบายความสัมพันธ์เชิงโครงสร้างได้ชัดที่สุด โดยเฉพาะเส้นทางจากโปรไฟล์ที่ผ่านการ normalize แล้ว ไปสู่การตัดสินใจจับคู่ การส่งเข้าคิวตรวจ การรวมเป็น unified customer profile และการคำนวณ lead score ในระดับเอนทิตี

ข้อควรระวังที่ต้องระบุใต้ภาพในรายงานคือ คอลัมน์ `profile_id_a` และ `profile_id_b` ใน `match_decisions.parquet` เป็นชื่อที่ถูก rename มาจาก `profile_row_id_a` และ `profile_row_id_b` ดังนั้นในเชิงความสัมพันธ์ของข้อมูล คอลัมน์ทั้งสองจึงอ้างถึง `NORMALIZED_PROFILE.profile_row_id` ไม่ใช่ `profile_id` แบบ source entity key

caption ที่แนะนำ:

“รูปที่ 3.x แผนภาพความสัมพันธ์ของตารางข้อมูลหลักแบบ Crow's-Foot ในสาย production/CRM แสดงความสัมพันธ์ระหว่างตารางโปรไฟล์ที่ผ่านการปรับมาตรฐาน ตารางผลการตัดสินใจจับคู่ คิวตรวจสอบ ตารางการแม็ปโปรไฟล์เข้าสู่ unified profile และตารางคะแนนลูกค้าเป้าหมาย”
