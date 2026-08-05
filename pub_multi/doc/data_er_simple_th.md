# ภาพโครงสร้างข้อมูลปลายทางของระบบ CRM แบบอ่านง่าย

ไฟล์ภาพ:
- [data_er_simple_th.svg](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/fig/data_er_simple_th.svg)

## ชื่อรูปที่แนะนำ

รูปที่ 3.4 โครงสร้างข้อมูลปลายทางของระบบ CRM แบบอ่านง่าย

## คำอธิบายใต้ภาพ

แสดงความสัมพันธ์ของข้อมูลในสาย production/CRM โดยเริ่มจากตารางโปรไฟล์มาตรฐาน (`NORMALIZED_PROFILE`) ซึ่งเป็นจุดตั้งต้นของทุกโปรไฟล์ จากนั้นระบบจะสร้างผลการตัดสินรายคู่ไว้ใน `MATCH_DECISION` และส่งเฉพาะคู่ที่ยังไม่มั่นใจไปยัง `REVIEW_QUEUE` หลังจากยืนยันผลแล้ว โปรไฟล์จะถูก map ไปยังลูกค้ารวมใน `PROFILE_MAPPING` และ `UNIFIED_PROFILE` ก่อนนำไปคำนวณคะแนนในระดับเอนทิตีผ่าน `LEAD_SCORE`

## คำอธิบายสั้นในเนื้อหา

รูปนี้ใช้เพื่ออธิบายความสัมพันธ์ของ artifact ปลายทางที่เกิดขึ้นหลังขั้น scoring และ decision making แล้ว โดยเน้นให้เห็นว่าโปรไฟล์ที่ผ่านการ normalize ถูกนำไปสร้างผลการตัดสินรายคู่ จากนั้นจึงถูก merge เป็น unified customer profile และนำไปคำนวณ lead score ต่อในระดับลูกค้ารวม

## หมายเหตุที่ควรใส่ใต้รูปหรือในย่อหน้าประกอบ

ในตารางผลลัพธ์ production คอลัมน์ `profile_id_a` และ `profile_id_b` ใน `MATCH_DECISION` เป็นชื่อที่ใช้แทน `profile_row_id_a` และ `profile_row_id_b` ดังนั้น foreign key เชิงปฏิบัติของตารางนี้จึงชี้กลับไปยัง `NORMALIZED_PROFILE.profile_row_id`
