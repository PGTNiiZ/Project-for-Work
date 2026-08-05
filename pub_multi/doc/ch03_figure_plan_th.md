# แผนการใส่รูปประกอบบทที่ 3

เอกสารนี้สรุปว่าบทที่ 3 ควรมีรูปอะไรบ้าง วางตรงไหน และแต่ละรูปควรอธิบายอะไร รูปในบทนี้ควรเน้นอธิบาย “ระบบทำงานอย่างไร” มากกว่า “ผลลัพธ์เป็นอย่างไร” ส่วน confusion matrix, PR curve, ROC curve, threshold sweep และกราฟเปรียบเทียบคะแนนของโมเดลควรย้ายไปอยู่บทที่ 4 เป็นหลัก

## รูปที่ 3.1 แผนภาพ CRISP-DM

ตำแหน่งที่ควรใส่  
วางหลังย่อหน้าเกริ่นนำของบทที่ 3

สิ่งที่รูปควรแสดง  
แสดงวงจร 6 ขั้นของ CRISP-DM ได้แก่ Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation และ Deployment

คำอธิบายใต้ภาพที่ควรใช้  
แสดงกรอบวิธีวิจัยที่ใช้เป็นโครงหลักของการพัฒนาระบบเชื่อมโยงอัตลักษณ์ลูกค้าและการให้คะแนนลูกค้าเป้าหมายในงานวิจัยนี้

## รูปที่ 3.2 ภาพรวม pipeline ของงานวิจัย

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.1

สิ่งที่รูปควรแสดง  
เริ่มจาก raw profiles จากหลายแพลตฟอร์ม แล้วไหลผ่าน cleaned profiles, normalized profiles, pair building, feature engineering, model training, calibration, exact-first + blocking, candidate scoring, decision tiers, review queue, unified profiles และ lead scores

คำอธิบายใต้ภาพที่ควรใช้  
แสดงลำดับการทำงานของระบบตั้งแต่การเตรียมข้อมูลต้นทางไปจนถึงผลลัพธ์ปลายทางในระบบ CRM

ไฟล์ร่างรูปที่ควรใช้  
ใช้ [pipeline_latest_actual_th.md](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/doc/pipeline_latest_actual_th.md) หรือ [pipeline_latest_actual.mmd](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/doc/pipeline_latest_actual.mmd) เป็นแหล่งอ้างอิงหลัก เพื่อหลีกเลี่ยงการใช้รูปเวอร์ชันเก่าที่ไม่ตรงกับ implementation ปัจจุบัน

## รูปที่ 3.3 Data Lineage และโครงสร้างข้อมูลกลาง

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.2

สิ่งที่รูปควรแสดง  
ความสัมพันธ์ระหว่างไฟล์และ artifact หลัก เช่น JSON ต้นทาง, `all_profiles_cleaned.csv`, `location_mapping.csv`, ฐานข้อมูลภาพที่ผ่านการปรับมาตรฐาน และ `normalized_profiles_with_profile_id.csv`

คำอธิบายใต้ภาพที่ควรใช้  
แสดงการไหลของข้อมูลจากแหล่งข้อมูลดิบไปสู่ชุดข้อมูลมาตรฐานที่ใช้ในขั้น pair building และ model training

## รูปที่ 3.4 Pair Building Pipeline

ตำแหน่งที่ควรใส่  
วางหลังย่อหน้าที่อธิบาย `stage8_pair_builder.py` ในหัวข้อ 3.3

สิ่งที่รูปควรแสดง  
การสร้าง positive pairs จากเอนทิตีเดียวกันข้ามแพลตฟอร์ม การสร้าง random negatives และการสร้าง hard negatives พร้อมบอกว่าขั้นตอนนี้ใช้เฉพาะคู่ข้ามแพลตฟอร์ม

คำอธิบายใต้ภาพที่ควรใช้  
แสดงกระบวนการสร้างชุดคู่ข้อมูลสำหรับการฝึกแบบจำลอง โดยครอบคลุมทั้งกรณีคู่บวก คู่ลบทั่วไป และคู่ลบแบบยาก

## รูปที่ 3.5 Feature Engineering และ Chunked Feature Pipeline

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.3

สิ่งที่รูปควรแสดง  
ลำดับจาก labeled pairs ไปสู่ train/val/test splits การสร้าง cache การคำนวณ text features, profile features, location features, URL/domain features และ multimodal features แล้วรวมเป็น feature matrix แบบ chunked

คำอธิบายใต้ภาพที่ควรใช้  
แสดงขั้นตอนการสร้างคุณลักษณะของคู่โปรไฟล์และการรวมผลลัพธ์เป็นชุดข้อมูลที่พร้อมใช้ในการฝึกและประเมินแบบจำลอง

## รูปที่ 3.6 Model Development Pipeline

ตำแหน่งที่ควรใส่  
วางในหัวข้อ 3.4 หลังอธิบายสาย classical และ multimodal เบื้องต้น

สิ่งที่รูปควรแสดง  
เส้นทางจาก feature matrix ไปยังการเปรียบเทียบหลายโมเดล การเลือก best model การทำ calibration และการเลือก threshold

คำอธิบายใต้ภาพที่ควรใช้  
แสดงกระบวนการพัฒนาแบบจำลอง ตั้งแต่การเตรียมข้อมูลสำหรับ train/validation/test ไปจนถึงการเลือก operating point สำหรับใช้งานจริง

ไฟล์ร่างรูปที่ควรใช้  
ใช้ส่วน “รูปที่ 3.6 เวอร์ชันล่าสุดจริง” ใน [pipeline_latest_actual_th.md](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/doc/pipeline_latest_actual_th.md) เป็นแหล่งร่างหลัก

## รูปที่ 3.7 ความสัมพันธ์ของสายการทดลองหลัก

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.4

สิ่งที่รูปควรแสดง  
ให้เห็น 3 สายหลักของการทดลอง ได้แก่ leakage-safe classical line, multimodal suite และ neural-network reference line พร้อมชี้ว่าผลจากแต่ละสายถูกใช้เพื่ออะไร

คำอธิบายใต้ภาพที่ควรใช้  
แสดงความสัมพันธ์ของชุดการทดลองที่ใช้ในการคัดเลือกแบบจำลองหลักและตรวจสอบความแข็งแรงของผลลัพธ์

## รูปที่ 3.8 Evaluation Framework

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.5

สิ่งที่รูปควรแสดง  
สามชั้นของการประเมิน คือ retrieval evaluation, model evaluation และ production evaluation โดยไม่ต้องใส่ตัวเลขผลลัพธ์ในรูป

คำอธิบายใต้ภาพที่ควรใช้  
แสดงกรอบการประเมินผลของระบบในหลายระดับ เพื่อให้การวัดผลสอดคล้องกับโครงสร้างของ pipeline ทั้งเส้น

## รูปที่ 3.9 Exact-First และ Blocking Production Pipeline

ตำแหน่งที่ควรใส่  
วางในหัวข้อ 3.6 หลังอธิบาย full candidate pipeline

สิ่งที่รูปควรแสดง  
จำนวนคู่ทั้งหมดในเชิงแนวคิด จากนั้นแยก exact matches ออกก่อน ใช้ blocking keys สร้าง candidate pairs ที่เหลือ แล้วส่งเข้า scoring และ thresholding

คำอธิบายใต้ภาพที่ควรใช้  
แสดงขั้นตอนการลด search space และการใช้โมเดลกับ candidate pairs ในระบบ production pipeline

## รูปที่ 3.10 Human-in-the-loop และ CRM Workflow

ตำแหน่งที่ควรใส่  
วางท้ายหัวข้อ 3.6

สิ่งที่รูปควรแสดง  
เส้นทางจาก match decisions ไปยัง review queue, decision log, entity merge, unified profiles และ lead scoring

คำอธิบายใต้ภาพที่ควรใช้  
แสดงการเชื่อมผลลัพธ์จากแบบจำลองเข้าสู่ workflow ปลายทางของระบบ CRM พร้อมกลไกการตรวจสอบโดยมนุษย์

## หมายเหตุการใช้รูปในบทที่ 3 และบทที่ 4

บทที่ 3 ควรใช้รูปที่ช่วยอธิบายลำดับขั้นและสถาปัตยกรรมของระบบ เช่น pipeline diagram, data lineage, pair-building diagram และ deployment workflow ส่วนรูปที่มีหน้าที่อธิบายผลลัพธ์เชิงตัวเลข เช่น confusion matrix, ROC curve, PR curve, threshold sweep, calibration plots, score distribution และกราฟเปรียบเทียบค่า metric ของแต่ละโมเดล ควรย้ายไปอยู่บทที่ 4 เพื่อให้โครงสร้างรายงานชัดว่า บทที่ 3 ตอบคำถามว่า “ทำอย่างไร” และบทที่ 4 ตอบคำถามว่า “ได้ผลอย่างไร”
