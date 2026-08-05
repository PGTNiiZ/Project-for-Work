# Project 1 Presentation Guide

เอกสารนี้เป็นแม่แบบนำเสนอประมาณ 100 สไลด์สำหรับ Project 1 โดยแยก 4 ส่วนให้ชัดเจนคือ ข้อความที่ควรอยู่บนสไลด์, แนวภาพประกอบ, ไฟล์ภาพใน repo ที่หยิบใช้ได้ และสคริปต์พูดสำหรับซ้อมนำเสนอ

## Slide 1: Cross-Platform Identity Resolution for CRM

- ประเภท: title
- หมวด: Opening
- ข้อความบนสไลด์:
- Guide deck 100 slides พร้อมสคริปต์พูด
- Project 1: exact-first, blocking, multimodal scoring และ CRM output
- ใช้ข้อมูล Twitter, Instagram และ Google+ รวม 36,804 โปรไฟล์
- ภาพประกอบที่ควรใช้: หน้าปกโทนวิจัย ใช้เส้นเชื่อม 3 แพลตฟอร์มเข้าสู่ customer profile เดียว
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_experiment_roadmap.png

**Speaker Script**
สไลด์เปิดนี้ใช้ตั้งภาพใหญ่ของงานก่อนว่าเราไม่ได้ทำแค่ classifier แต่กำลังสร้างระบบเชื่อมโยงตัวตนข้ามแพลตฟอร์มสำหรับ CRM ตั้งแต่การลด search space ไปจนถึงการรวมโปรไฟล์และสร้างผลลัพธ์ที่นำไปใช้จริงได้

## Slide 2: วิธีใช้ Deck ชุดนี้

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- ข้อความบนสไลด์ถูกตั้งใจให้สั้น
- สคริปต์พูดอยู่ในไฟล์คู่กัน
- ภาพประกอบแต่ละหน้าเป็นข้อเสนอแนะ ไม่จำเป็นต้องใช้ตามนี้ทั้งหมด
- ภาพประกอบที่ควรใช้: ใช้ภาพ flow สั้น ๆ อธิบายว่า slide = headline, guide = script, figure = visual support

**Speaker Script**
เด็คชุดนี้ตั้งใจให้เป็นแม่แบบนำเสนอขนาดใหญ่ก่อน เพื่อช่วยเห็นว่าควรเล่าอะไรบ้างในงานจริง หน้า slide จะคุมข้อความให้น้อย ส่วนรายละเอียดที่ต้องพูด เหตุผล และแนวภาพประกอบ ผมแยกให้อยู่ใน guide เพื่อหยิบไปตัดทอนหรือขยายต่อได้สะดวก

## Slide 3: Agenda

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- ปัญหา ข้อมูล และขอบเขตของงาน
- วิธีทำตั้งแต่ pipeline ถึง model
- ผลลัพธ์ production ข้อจำกัด และข้อเสนอแนะ
- ภาพประกอบที่ควรใช้: ใช้ timeline 3 ช่วงหรือ roadmap แบบกว้าง ๆ

**Speaker Script**
โครงเรื่องวันนี้จะแบ่งเป็นสามช่วงใหญ่ ช่วงแรกคือปัญหาและข้อมูล ช่วงที่สองคือวิธีทำและการพัฒนาโมเดล ส่วนช่วงสุดท้ายคือผลลัพธ์ระดับ production ใน CRM รวมถึงข้อจำกัดและทางต่อยอดของงาน

## Slide 4: ปัญหาทางธุรกิจของงานนี้

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- ลูกค้าคนเดียวกันกระจายอยู่หลายแพลตฟอร์ม
- CRM เห็นข้อมูลเป็นหลายระเบียน
- การวิเคราะห์และการสื่อสารทางการตลาดจึงคลาดเคลื่อน
- ภาพประกอบที่ควรใช้: ใช้ภาพ customer journey ที่แตกเป็นหลายบัญชีแล้วถูกรวมกลับมา

**Speaker Script**
แกนของปัญหาคือธุรกิจรู้จักลูกค้าคนหนึ่งผ่านหลายบัญชี แต่ข้อมูลเหล่านั้นไม่ถูกผูกเข้าหากันอย่างถูกต้อง ถ้า CRM มองลูกค้าคนเดียวเป็นหลายคน การแบ่งกลุ่มลูกค้า การ personalization และการจัดลำดับ lead ก็จะผิดพลาดตามไปด้วย

## Slide 5: ทำไม Duplicate Profile ถึงกระทบ CRM

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- เห็นพฤติกรรมลูกค้าไม่ครบ
- นับจำนวนลูกค้าซ้ำ
- วางแผน follow-up ผิดคนหรือผิดจังหวะ
- ภาพประกอบที่ควรใช้: ใช้ภาพก่อนรวมและหลังรวม customer 360 แบบ side-by-side

**Speaker Script**
สไลด์นี้ใช้เชื่อมโจทย์เทคนิคกับผลกระทบทางธุรกิจให้ชัดว่าเรื่อง identity resolution ไม่ใช่แค่ปัญหาความสวยงามของข้อมูล แต่กระทบต่อการมองลูกค้าแบบ 360 องศาโดยตรง รวมถึงการตัดสินใจเชิงการตลาดและงานขาย

## Slide 6: Research Gap ที่งานนี้พยายามตอบ

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- ไม่ได้จบที่ pair matching อย่างเดียว
- ต้องเชื่อมต่อกับ workflow จริงของ CRM
- ต้องคุมทั้ง coverage, precision และภาระ review
- ภาพประกอบที่ควรใช้: ใช้ไดอะแกรมจาก model output ไปสู่ human review และ CRM

**Speaker Script**
จุดต่างของงานนี้คือเราไม่ได้ถามแค่ว่าโมเดลทายคู่ไหนเหมือนกันได้แม่นแค่ไหน แต่ถามต่อด้วยว่าหลังทายแล้วจะเอาไปใช้จริงอย่างไรใน CRM เพราะระบบจริงต้องคุมความเสี่ยงของ false merge พร้อมกับรักษาความครอบคลุมและต้นทุนการตรวจสอบของคน

## Slide 7: คำถามวิจัยหลัก

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- จะเชื่อมโปรไฟล์ข้ามแพลตฟอร์มให้แม่นพอสำหรับ CRM ได้อย่างไร
- จะลดพื้นที่ค้นหาโดยไม่ทำให้คู่จริงหายไปมากเกินไปได้อย่างไร
- จะตั้ง threshold ให้เหมาะกับการใช้งานจริงอย่างไร
- ภาพประกอบที่ควรใช้: ใช้กล่องคำถาม 3 ข้อเรียงจาก retrieval ไป production

**Speaker Script**
คำถามวิจัยของงานนี้มีสามชั้น เริ่มจากการหาคู่ที่ควรพิจารณาให้เจอ ต่อมาคือการให้คะแนนคู่เหล่านั้นอย่างน่าเชื่อถือ และสุดท้ายคือการแปลงคะแนนให้เป็นกติกาการทำงานจริงในระบบ CRM

## Slide 8: วัตถุประสงค์ของโครงการ

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- พัฒนาระบบ cross-platform identity resolution
- เปรียบเทียบแบบจำลองหลายประเภทบน split ที่ปลอดจาก leakage
- ประเมินความเป็นไปได้ในการใช้งานจริงระดับ production
- ภาพประกอบที่ควรใช้: ใช้ objective cards 3 ใบพร้อมไอคอน data, model, deployment

**Speaker Script**
วัตถุประสงค์ของงานจึงไม่ได้มีแค่การทำโมเดลให้คะแนนสูง แต่รวมถึงการออกแบบ pipeline ให้ครบตั้งแต่ต้นน้ำถึงปลายน้ำ ทั้งในเชิงการทดลองที่น่าเชื่อถือและในเชิงการนำไปใช้จริงในระบบปลายทาง

## Slide 9: Contribution ของงานนี้

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- exact-first + blocking + calibrated scoring
- multimodal feature set สูงสุด 41 ฟีเจอร์
- workflow ปลายทางแบบ MATCH, REVIEW, NO_MATCH
- ภาพประกอบที่ควรใช้: ใช้ 3 กล่อง contribution พร้อมลูกศรเชื่อมต่อกัน

**Speaker Script**
ถ้าสรุปให้สั้นที่สุด contribution ของงานนี้มีสามส่วนคือ ออกแบบ retrieval ที่ลด search space ได้จริง พัฒนา candidate scorer แบบ multimodal ที่อธิบายได้ และเชื่อมผลลัพธ์เข้าสู่ workflow ปลายทางของ CRM อย่างเป็นระบบ

## Slide 10: ขอบเขตเชิงธุรกิจกับขอบเขตการทดลอง

- ประเภท: content
- หมวด: Opening
- ข้อความบนสไลด์:
- ธุรกิจอยากรวม identity ทุกกรณี
- การทดลองหลักของงานนี้เน้น cross-platform
- เพราะ same-platform positive มีเพียง 4 คู่
- ภาพประกอบที่ควรใช้: ใช้ตารางเปรียบเทียบ business scope กับ experimental scope

**Speaker Script**
ต้องแยกสองชั้นนี้ให้ชัดนะครับ ในเชิงธุรกิจระบบควรรองรับการรวม identity ทุกกรณี แต่ในเชิงการทดลองหลักของงานนี้เราโฟกัส cross-platform เพราะชุดข้อมูลรองรับคู่บวกแบบข้ามแพลตฟอร์มชัดกว่ามาก ขณะที่ same-platform positive มีเพียงสี่คู่เท่านั้น

## Slide 11: Problem & Data

- ประเภท: divider
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- จากโจทย์ CRM ไปสู่ข้อเท็จจริงของข้อมูลที่เรามี
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section แบบเรียบ ๆ มีไอคอน database และ social platform

**Speaker Script**
จากตรงนี้เราจะลงไปดูข้อมูลจริงของงาน ว่ามาจากไหน มีข้อจำกัดอะไร และตัวเลขตั้งต้นแบบไหนที่ทำให้ต้องออกแบบ pipeline เป็นหลายชั้น

## Slide 12: ชุดข้อมูลที่ใช้ในงาน

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- อ้างอิงจาก LinkSocial
- ใช้ 3 แพลตฟอร์ม: Twitter, Instagram, Google+
- หลังคัด valid profiles เหลือ 36,804 โปรไฟล์
- ภาพประกอบที่ควรใช้: ใช้ภาพ dataset overview หรือรูปสรุปจำนวนโปรไฟล์ตามแพลตฟอร์ม
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_data_prep_summary.png

**Speaker Script**
สไลด์นี้แนะนำแหล่งข้อมูลตั้งต้นของงาน เราใช้ข้อมูลโปรไฟล์สาธารณะจากชุด LinkSocial และคัดเฉพาะโปรไฟล์ที่พร้อมใช้งานจริงตามเงื่อนไขของ pipeline จนเหลือ 36,804 โปรไฟล์สำหรับการรัน full pipeline

## Slide 13: แพลตฟอร์มที่อยู่ในงานวิจัยนี้

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- Twitter 13,959 โปรไฟล์
- Instagram 10,956 โปรไฟล์
- Google+ 11,889 โปรไฟล์
- ภาพประกอบที่ควรใช้: ใช้ bar chart 3 แท่งหรือไอคอนแพลตฟอร์มพร้อมตัวเลข
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_data_prep_summary.png

**Speaker Script**
ตรงนี้ให้ชี้ให้เห็นว่าเราไม่ได้มีจำนวนโปรไฟล์เท่ากันทุกแพลตฟอร์ม และ Google+ ก็เป็นข้อมูล archive ที่มีลักษณะต่างจากอีกสองแพลตฟอร์ม ความไม่สมดุลนี้มีผลทั้งต่อคุณภาพสัญญาณและการตีความผลลัพธ์ภายหลัง

## Slide 14: ฟิลด์หลักของโปรไฟล์หลัง normalize

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- userName, fullName, bio
- location, externalUrl, pictureURL
- platform, profile_id, profile_row_id
- ภาพประกอบที่ควรใช้: ใช้ mock table ของหนึ่งโปรไฟล์แบบอ่านง่าย

**Speaker Script**
สไลด์นี้ทำหน้าที่ปูพื้นว่า pipeline ไม่ได้รับข้อมูลดิบกระจัดกระจายเข้าไปตรง ๆ แต่พยายาม normalize ให้อยู่ในโครงสร้างเดียวกันก่อน ฟิลด์สำคัญมีทั้งข้อความ ลิงก์ ตำแหน่ง และคีย์เชิงโครงสร้างที่ใช้ join กับ artifact อื่น ๆ

## Slide 15: ข้อท้าทายด้านคุณภาพข้อมูล

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- ข้อมูลหาย ไม่ครบ และไม่สม่ำเสมอ
- รูปแบบการเขียนชื่อและ bio เปลี่ยนตามแพลตฟอร์ม
- metadata ภาพและ URL มี coverage ไม่เท่ากัน
- ภาพประกอบที่ควรใช้: ใช้ภาพ puzzle pieces ที่หายบางส่วนหรือ table completeness heatmap

**Speaker Script**
จุดที่ต้องย้ำคือปัญหานี้ไม่ได้เกิดบนข้อมูลสะอาดสมบูรณ์แบบ ผู้ใช้คนเดียวอาจใช้ชื่อคนละแบบ bio คนละแนว หรือไม่มีรูปเลยในบางแพลตฟอร์ม งานนี้จึงเป็นงานเชิงระบบที่ต้องรับมือกับความไม่สมบูรณ์ของข้อมูลตั้งแต่ต้น

## Slide 16: Google+ คือแพลตฟอร์มที่ท้าทายที่สุด

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- เป็นข้อมูล archive จากแพลตฟอร์มที่ปิดตัวแล้ว
- metadata และภาพไม่สมบูรณ์เท่า Twitter และ Instagram
- ข้อผิดพลาดจำนวนมากกระจุกในคู่ Google+ -> Twitter
- ภาพประกอบที่ควรใช้: ใช้ callout box ชี้ให้เห็นบทบาทของ Google+ ในชุดข้อมูล

**Speaker Script**
ในเชิงการตีความผล Google+ เป็นตัวแปรสำคัญมาก เพราะข้อมูลส่วนนี้มีความสมบูรณ์ต่ำกว่าฝั่งอื่น ทำให้หลายข้อผิดพลาด โดยเฉพาะ false negatives กระจุกในคู่ที่เกี่ยวกับ Google+ อย่างชัดเจน

## Slide 17: ตัวเลขโปรไฟล์ valid ที่ใช้จริง

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- รวมทั้งหมด 36,804 โปรไฟล์
- Twitter สัดส่วน 37.9%
- Instagram 29.8% และ Google+ 32.3%
- ภาพประกอบที่ควรใช้: ใช้ pie chart หรือ stacked bar ที่อ่านสัดส่วนง่าย
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_data_prep_summary.png

**Speaker Script**
สไลด์นี้ใช้ตอบคำถามพื้นฐานว่าขนาดข้อมูลจริงที่เราใช้ใหญ่แค่ไหน และกระจายตัวอย่างไร แนะนำให้พูดเร็ว ๆ เพื่อพาผู้ฟังไปยังประเด็นสำคัญต่อไปคือจำนวนคู่ที่เป็นไปได้ซึ่งโตเร็วกว่าจำนวนโปรไฟล์มาก

## Slide 18: All-Platform กับ Cross-Platform ไม่ใช่เรื่องเดียวกัน

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- all unordered pairs = 677,248,806
- cross-platform pairs = 449,149,239
- same-platform pairs = 228,099,567
- ภาพประกอบที่ควรใช้: ใช้ number cards 3 ใบ หรือ Venn-like comparison

**Speaker Script**
สไลด์นี้ช่วยแก้ความสับสนเรื่องขอบเขตของปัญหา ถ้าพูดเชิงแนวคิดทุกคู่ที่เป็นไปได้มีมากกว่า 677 ล้านคู่ แต่ถ้าพูดตาม experimental scope หลักของงาน เรานับเฉพาะ cross-platform จึงเหลือ 449 ล้านคู่ ซึ่งยังใหญ่มากอยู่ดี

## Slide 19: Ground Truth ที่รองรับการทดลองหลัก

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- cross-platform positive pairs = 29,243
- same-platform positive pairs = 4
- จึงเหมาะจะสรุปผลหลักในกรอบ cross-platform
- ภาพประกอบที่ควรใช้: ใช้ตารางเปรียบเทียบ positive pairs สองกรณี

**Speaker Script**
เหตุผลที่เราสรุปผลงานหลักในกรอบ cross-platform ไม่ใช่เพราะ same-platform ไม่สำคัญ แต่เพราะข้อมูลกำกับที่มีอยู่จริงรองรับกรณี cross-platform แข็งแรงกว่ามาก ถ้าสรุปรวมทั้งหมดจะเสี่ยงตีความเกินหลักฐานที่มี

## Slide 20: ทำไม Search Space จึงเป็นปัญหาหลัก

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- เปรียบเทียบทุกคู่ตรง ๆ ไม่คุ้มทั้งเวลาและทรัพยากร
- 449 ล้านคู่ใหญ่เกินกว่าจะ score ทุกคู่แบบตรงไปตรงมา
- จึงต้องมี retrieval ก่อน scoring
- ภาพประกอบที่ควรใช้: ใช้ funnel เริ่มจาก 449M แล้วชี้ว่าต้องลดก่อนเข้า model
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_retrieval_funnel.png

**Speaker Script**
ตรงนี้เป็นเหตุผลเชิงวิศวกรรมของทั้งระบบ ถ้าเราพยายามสร้าง feature และ score ให้ทุกคู่โดยตรง ต้นทุนจะสูงเกินไป งานนี้จึงแยก retrieval ออกมาต่างหาก และ optimize ให้เก็บคู่จริงได้มากพอขณะเดียวกันก็ลดจำนวนคู่ลงอย่างรุนแรง

## Slide 21: ระบบต้องตอบ 3 งานต่อเนื่องกัน

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- Retrieval: จะดึงคู่ที่ควรดูต่ออย่างไร
- Scoring: จะให้คะแนนความน่าจะเป็นอย่างไร
- Decision: จะตัดสิน MATCH, REVIEW, NO_MATCH อย่างไร
- ภาพประกอบที่ควรใช้: ใช้ diagram 3 กล่องเรียงจากซ้ายไปขวา

**Speaker Script**
สไลด์นี้สำคัญมากเพราะเป็นกรอบคิดของทั้งโครงการ เราไม่ได้ทำงาน classification แบบแถวต่อแถว แต่ทำงานสามชั้นต่อเนื่องกัน คือ retrieval หา candidate, scoring ให้คะแนน และ decision แปลงคะแนนไปสู่กฎการทำงานจริง

## Slide 22: CRISP-DM คือกรอบวิธีวิจัยของงานนี้

- ประเภท: content
- หมวด: Problem & Data
- ข้อความบนสไลด์:
- Business Understanding
- Data Understanding และ Data Preparation
- Modeling, Evaluation และ Deployment
- ภาพประกอบที่ควรใช้: ใช้วงจร CRISP-DM แบบมาตรฐาน

**Speaker Script**
ถ้าจะเชื่อมปัญหาธุรกิจกับงานทดลองทั้งหมดให้เป็นเรื่องเดียวกัน กรอบที่เหมาะที่สุดคือ CRISP-DM เพราะมันไม่หยุดที่การฝึกโมเดล แต่ครอบคลุมตั้งแต่ความเข้าใจปัญหาไปจนถึงการนำผลไปใช้จริงใน workflow ปลายทาง

## Slide 23: Method & Pipeline

- ประเภท: divider
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- จากข้อมูลดิบไปสู่การสร้างคู่ข้อมูล ฟีเจอร์ และโมเดลที่พร้อมใช้จริง
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section แบบ pipeline หรือแถบขั้นตอนหลายช่วง

**Speaker Script**
จากตรงนี้เราจะเริ่มดูวิธีทำจริงของระบบ ตั้งแต่การเตรียมข้อมูล การสร้าง artifact มาตรฐาน การสร้างคู่ข้อมูล ไปจนถึงการฝึกและเลือกโมเดล

## Slide 24: ภาพรวม End-to-End Pipeline

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- raw profiles -> cleaned profiles -> normalized profiles
- pair building -> feature engineering -> modeling
- retrieval -> scoring -> CRM export
- ภาพประกอบที่ควรใช้: ใช้ pipeline overview ทั้งเส้นแบบ end-to-end
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/pipeline_latest_actual.svg

**Speaker Script**
สไลด์นี้ควรเป็นภาพใหญ่ของ implementation ล่าสุดจริง ให้ผู้ฟังเห็นว่าระบบไหลจากข้อมูลต้นทางไปสู่ผลลัพธ์ปลายทางอย่างไร และช่วยกันความสับสนว่าแต่ละไฟล์หรือแต่ละ experiment อยู่ตรงไหนของภาพรวม

## Slide 25: Stage Map ของระบบ

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- Stage 1-4: เตรียมข้อมูลและ artifact มาตรฐาน
- Stage 5-8: สร้างคู่ข้อมูลและฝึกโมเดล
- Stage 9-12: production retrieval, CRM export และ helper
- ภาพประกอบที่ควรใช้: ใช้ timeline ของ stage 1 ถึง 12 พร้อมชื่อสั้น ๆ

**Speaker Script**
สไลด์นี้เหมาะกับการเล่าเป็น roadmap ของโค้ดใน repo ว่างานไม่ได้กระจุกอยู่ไฟล์เดียว แต่จัดเป็นหลาย stage ชัดเจน ช่วยให้ผู้ฟังเห็นว่าระบบเติบโตจาก data pipeline ไปสู่ production pipeline อย่างเป็นลำดับ

## Slide 26: Stage 1: Data Preparation

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- โหลดและ clean raw profiles
- รวม schema ให้เป็นมาตรฐานเดียว
- ได้ all_profiles_cleaned.csv
- ภาพประกอบที่ควรใช้: ใช้ภาพ data lineage จาก raw JSON ไปสู่ cleaned CSV

**Speaker Script**
ขั้นแรกของระบบคือทำให้ข้อมูลจากหลายแพลตฟอร์มอยู่ในรูปที่คุยกันได้ก่อน งานส่วนนี้ดูพื้นฐานแต่สำคัญมาก เพราะถ้า schema ยังไม่เสถียร ขั้นต่อ ๆ ไปจะสร้างฟีเจอร์หรือ join artifact ได้ยาก

## Slide 27: Stage 2: Location Normalization

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- normalize location text
- สร้าง location_mapping.csv
- ลดความหลากหลายของ location ที่เขียนไม่เหมือนกัน
- ภาพประกอบที่ควรใช้: ใช้ before-after examples ของ location text

**Speaker Script**
สัญญาณ location มักเขียนหลากหลายมาก เช่น ชื่อเมืองย่อ สลับภาษา หรือมีคำเกินมา การ normalize location จึงทำหน้าที่เปลี่ยนข้อความที่ใกล้กันให้มาอยู่ในรูปที่เทียบกันได้ดีขึ้น ก่อนนำไปใช้สร้าง similarity features

## Slide 28: Stage 3: Normalized Profile Database

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- รวม profile fields กับ location mapping
- แนบ profile_id และ profile_row_id
- ได้ normalized_profiles_with_profile_id.csv
- ภาพประกอบที่ควรใช้: ใช้ภาพ data lineage เชื่อม cleaned profiles, location mapping และ normalized DB

**Speaker Script**
พอ location ถูกทำให้เป็นระบบแล้ว ขั้นต่อมาคือสร้างฐานข้อมูลโปรไฟล์มาตรฐานที่ใช้ร่วมกันได้ทั้งสาย training และสาย production จุดนี้สำคัญเพราะเป็น single source of truth ของ profile-level artifact หลายตัวในระบบ

## Slide 29: Stage 4: Image Recovery และ Image Artifacts

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- ดึงและจัดเก็บ local images
- สร้าง metadata, quality, face และ caption artifacts
- เตรียม image signal สำหรับสาย multimodal
- ภาพประกอบที่ควรใช้: ใช้ภาพ flow จาก pictureURL ไปสู่ images, metadata, captions, embeddings

**Speaker Script**
สายภาพของงานนี้ไม่ได้หยิบ raw picture URL ไปใช้ตรง ๆ แต่มีขั้นเตรียม artifact แยกต่างหาก ทั้งเรื่องการมีภาพจริง คุณภาพภาพ จำนวนใบหน้า และ caption เพื่อให้ feature engineering ใช้สัญญาณภาพได้อย่างเป็นระบบ

## Slide 30: ตัวอย่างโปรไฟล์หลังผ่านการ normalize

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- ข้อความถูกจัดให้อยู่ในฟิลด์เดียวกัน
- คีย์เชิงโครงสร้างถูกเพิ่มเข้ามา
- พร้อมสำหรับ pair building และ feature joins
- ภาพประกอบที่ควรใช้: ใช้ screenshot ของ 1-2 แถวจาก normalized_profiles_with_profile_id.csv แบบ crop แล้วอ่านง่าย

**Speaker Script**
สไลด์นี้ใช้ให้เห็นภาพว่าหลัง normalize แล้วแต่ละโปรไฟล์มีหน้าตาอย่างไร ผู้ฟังจะเข้าใจง่ายขึ้นเมื่อเห็น field จริง เช่น userName, fullName, bio, location, externalUrl, platform และคีย์อ้างอิงต่าง ๆ อยู่ในหนึ่งแถวเดียวกัน

## Slide 31: Stage 5: Pair Construction Overview

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- สร้าง positive pairs
- สร้าง random negatives
- สร้าง hard negatives
- ภาพประกอบที่ควรใช้: ใช้ pair-building diagram มี 3 แขนงแล้วรวมเป็น labeled pairs

**Speaker Script**
นี่คือจุดเปลี่ยนจากข้อมูลระดับโปรไฟล์ไปสู่ข้อมูลระดับคู่ งานนี้ไม่ได้มีเฉพาะคู่บวก แต่ตั้งใจออกแบบให้มีทั้งคู่ลบทั่วไปและคู่ลบแบบยาก เพื่อให้โมเดลเห็นขอบเขตของปัญหาในหลายระดับความยาก

## Slide 32: การสร้าง Positive Pairs

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- อิง user_folder เป็นตัวแทน identity ต้นทาง
- ใช้คู่ข้ามแพลตฟอร์มเป็นฐานของงานหลัก
- รองรับ ground truth 29,243 คู่
- ภาพประกอบที่ควรใช้: ใช้ภาพกลุ่มโปรไฟล์ของคนเดียวกันแล้วจับคู่ข้ามแพลตฟอร์ม

**Speaker Script**
หลักคิดของ positive pair คืองานนี้ใช้ user_folder เป็นตัวแทนของ entity ต้นทาง แล้วสร้างคู่โปรไฟล์ที่เป็นคนเดียวกันข้ามแพลตฟอร์ม สิ่งนี้ทำให้เส้นการทดลองหลักของงานยึดอยู่บน ground truth ที่ชัดและสอดคล้องกับ scope ของระบบ

## Slide 33: Random Negatives ใช้ทำอะไร

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- แทนคู่ลบทั่วไปที่พบได้ในโลกจริง
- ช่วยให้โมเดลเห็น base rate ของคู่ที่ไม่ใช่
- ลดการ bias ไปที่ตัวอย่างยากอย่างเดียว
- ภาพประกอบที่ควรใช้: ใช้ภาพกลุ่มคู่ข้อมูลจำนวนมากที่สุ่มมาจากคนละ identity

**Speaker Script**
ถ้าเราให้โมเดลเห็นแต่ positive กับ hard negatives อย่างเดียว โมเดลจะไม่เห็น distribution ของคู่ลบทั่วไป Random negatives จึงทำหน้าที่เป็นพื้นหลังของปัญหา ว่าคู่ส่วนใหญ่ในโลกจริงจริง ๆ แล้วไม่ใช่คู่เดียวกัน

## Slide 34: Hard Negatives สำคัญอย่างไร

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- เป็นคู่ลบที่คล้ายกันมาก
- ช่วยกด false positives
- จำเป็นมากสำหรับโจทย์ชื่อคล้ายข้ามแพลตฟอร์ม
- ภาพประกอบที่ควรใช้: ใช้ตัวอย่างคู่ชื่อคล้ายแต่คนละ identity แบบ side-by-side

**Speaker Script**
hard negatives เป็นส่วนที่ทำให้งานนี้มีความสมจริงมากขึ้น เพราะในงาน identity matching คู่ที่ยากที่สุดคือคู่ลบที่ดูเหมือนคู่จริงมาก ๆ เช่น username หรือ fullName ใกล้กันมาก ถ้าไม่ใส่ตัวอย่างแบบนี้เข้าไป โมเดลจะหลงเชื่อชื่อที่คล้ายกันง่ายเกินไป

## Slide 35: สรุปปริมาณ Pair Data ที่สร้างได้

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- positive pairs = 4,060,200
- random negatives = 20,301,000
- hard negatives = 302,433
- ภาพประกอบที่ควรใช้: ใช้ stacked bar หรือ 3 number cards พร้อมยอดรวม 24,663,633 คู่

**Speaker Script**
สไลด์นี้ใช้ย้ำว่าระดับ pair data ของงานใหญ่มาก ไม่ใช่แค่หลักหมื่นหรือหลักแสน แต่ระดับหลายสิบล้านคู่ การจัดการข้อมูลขนาดนี้จึงมีผลต่อทั้งวิธี split วิธีสร้างฟีเจอร์ และเหตุผลที่ต้องทำ chunked processing ในขั้นต่อไป

## Slide 36: หลักการ Split แบบ Leak-Safe

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- split ในระดับ entity ไม่ใช่สุ่มทีละคู่
- entity เดียวต้องอยู่ split เดียวกันทั้งหมด
- ลด optimistic bias จาก identity leakage
- ภาพประกอบที่ควรใช้: ใช้รูป split design ที่ชี้ให้เห็น component overlap เป็นศูนย์
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_split_design.png

**Speaker Script**
สไลด์นี้สำคัญในเชิงระเบียบวิธีมาก เพราะถ้า entity เดียวกันกระจายไปอยู่ทั้ง train และ test เราจะได้คะแนนสูงเกินจริงทันที งานนี้จึงเลือก split ในระดับ entity เพื่อป้องกัน identity leakage อย่างชัดเจน

## Slide 37: ทำไมต้องระวัง Leakage ขนาดนี้

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- run ดั้งเดิมเคยให้ metric สมบูรณ์แบบผิดปกติ
- เมื่อแก้ leakage แล้วคะแนนกลับมาอยู่ในช่วงสมเหตุสมผล
- ความน่าเชื่อถือของผลจึงสูงขึ้น
- ภาพประกอบที่ควรใช้: ใช้ภาพ leakage diagnosis หรือ before-after metric comparison
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_leakage_diagnosis.png

**Speaker Script**
ถ้าอาจารย์ถามว่าทำไมเราพูดเรื่อง leakage บ่อย สไลด์นี้คือคำตอบ งานเดิมเคยเจอผล AP AUC F1 ใกล้สมบูรณ์แบบ ซึ่งเป็นสัญญาณผิดธรรมชาติ พอออกแบบ split ใหม่แบบ leak-safe ผลลัพธ์จึงกลับมาอยู่ในระดับที่เชื่อได้มากขึ้น

## Slide 38: ขนาดข้อมูลหลัง Split

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- artifact-level pairs: train 10.13M, val 4.51M, test 0.46M
- main supervised run: 41,803 / 8,580 / 8,313 rows
- ใช้ทั้ง large artifact pipeline และ curated training split
- ภาพประกอบที่ควรใช้: ใช้สองชั้นของตัวเลข แยก artifact split กับ main run split

**Speaker Script**
ตัวเลขในงานนี้มีสองชั้นที่ควรอธิบายให้ชัด ชั้นแรกคือ artifact-level split สำหรับงาน feature pipeline ขนาดหลายล้านคู่ อีกชั้นคือ curated supervised split ที่ใช้ฝึกและประเมิน main run โดยตรง จึงไม่ควรเอาตัวเลขสองชุดไปปนกัน

## Slide 39: Chunked Feature Engineering

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- คำนวณฟีเจอร์ทีละก้อน
- ลดปัญหา memory footprint
- ทำให้จัดการคู่ข้อมูลระดับล้านได้จริง
- ภาพประกอบที่ควรใช้: ใช้ภาพ data chunks ไหลผ่าน feature extractor แล้ว merge กลับ

**Speaker Script**
เนื่องจากคู่ข้อมูลมีขนาดใหญ่ เราไม่สามารถคำนวณทุกอย่างพร้อมกันในหน่วยความจำครั้งเดียวได้ แนวทาง chunked processing จึงเป็นหัวใจด้านวิศวกรรมที่ทำให้ pipeline นี้รันได้จริงกับข้อมูลระดับล้านคู่

## Slide 40: จากโปรไฟล์ 2 ฝั่ง ไปสู่ Pair-Level Vector

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- โมเดลไม่ได้รับ raw profile ตรง ๆ
- แต่รับผลการเปรียบเทียบของสองโปรไฟล์
- หนึ่งคู่ = หนึ่งแถวของฟีเจอร์
- ภาพประกอบที่ควรใช้: ใช้รูป pair-to-vector diagram
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch03_pair_to_vector.svg

**Speaker Script**
นี่คือแนวคิดสำคัญของ modeling ในงานนี้ เราแปลงโปรไฟล์สองฝั่งให้กลายเป็นเวกเตอร์ของ similarity, overlap, difference และ context ต่าง ๆ เพราะสิ่งที่โมเดลต้องเรียนรู้คือความสัมพันธ์ของสองโปรไฟล์ ไม่ใช่ค่าดิบของโปรไฟล์เพียงด้านเดียว

## Slide 41: 48 -> 44 -> 41 คืออะไร

- ประเภท: content
- หมวด: Method & Pipeline
- ข้อความบนสไลด์:
- 48 คอลัมน์ใน merged feature matrix
- 44 ฟีเจอร์หลังตัดคอลัมน์ระบุตัวคู่
- 41 ฟีเจอร์ที่เลือกใช้จริงใน main run
- ภาพประกอบที่ควรใช้: ใช้ staircase diagram 48 ไป 44 ไป 41

**Speaker Script**
ตัวเลขสามชุดนี้มักทำให้คนสับสน จึงควรพูดให้ชัดว่า 48 คือระดับตารางหลัง merge 44 คือ feature pool สำหรับฝึก และ 41 คือ selected feature set ของ run หลัก ไม่ใช่ตัวเลขที่ขัดกัน แต่เป็นคนละชั้นของ pipeline

## Slide 42: Feature Design

- ประเภท: divider
- หมวด: Features
- ข้อความบนสไลด์:
- จากข้อความ ลิงก์ และภาพ ไปสู่หลักฐานเชิงตัวเลขที่โมเดลใช้จริง
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section ที่มีคำว่า text, image, metadata ลอยอยู่รอบ ๆ

**Speaker Script**
จากตรงนี้เราจะลงรายละเอียดเรื่องฟีเจอร์ ซึ่งเป็นหัวใจว่าโมเดลมองสองโปรไฟล์ผ่านหลักฐานแบบไหนบ้าง และทำไม run หลักจึงถูกเรียกว่า multimodal

## Slide 43: ภาพรวมของกลุ่มฟีเจอร์

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- Identity string, bio, URL/domain
- location, mention/hashtag, style
- image availability, image stats, image-caption context
- ภาพประกอบที่ควรใช้: ใช้ matrix หรือ family map ของ 10 กลุ่มฟีเจอร์
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/modality.png

**Speaker Script**
สไลด์นี้ควรใช้เป็นแผนที่ของ feature space ทั้งหมด ก่อนจะพาเจาะทีละกลุ่ม ผู้ฟังจะได้เห็นว่า 41 ฟีเจอร์ไม่ได้กระจัดกระจายแบบสุ่ม แต่รวมตัวกันเป็นกลุ่มตามชนิดของหลักฐาน

## Slide 44: Identity String Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- username_jaro, username_lev, username_token_sort
- fullname_jaro, fullname_lev, fullname_token_sort
- เป็นแกนหลักของการตัดสินใจ
- ภาพประกอบที่ควรใช้: ใช้ตัวอย่างชื่อสองฝั่งแล้วคำนวณ similarity 2-3 แบบ

**Speaker Script**
กลุ่มชื่อผู้ใช้และชื่อจริงเป็นสัญญาณที่ทรงพลังที่สุดในงานนี้ เพราะเมื่อชื่อสอดคล้องกันสูงมาก โอกาสเป็นคนเดียวกันก็สูงตามไปด้วย นี่คือเหตุผลที่ feature importance ภายหลังจะกระจุกอยู่ที่กลุ่ม fullname และ username อย่างชัดเจน

## Slide 45: Bio และ Semantic Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- bio_tfidf_cosine
- bio_sbert_cosine
- ช่วยจับความคล้ายเชิงเนื้อหา ไม่ใช่แค่คำที่ตรงกัน
- ภาพประกอบที่ควรใช้: ใช้ตัวอย่าง bio สองฝั่งที่คำไม่เหมือนกันแต่ความหมายใกล้กัน
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_sbert_gain.png

**Speaker Script**
ฟีเจอร์ bio มีประโยชน์ในกรณีที่ชื่ออาจไม่พอหรือมีการเปลี่ยนแปลงข้ามแพลตฟอร์ม โดยเฉพาะ SBERT ที่ช่วยจับ semantic similarity ของ bio ได้ดีกว่า token overlap ธรรมดา ทำให้ recall ดีขึ้นในหลายกรณี

## Slide 46: URL และ Domain Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- domain_jaccard และ url_jaccard
- domain_count_a และ domain_count_b
- ใช้ลิงก์ภายนอกเป็นสัญญาณยืนยันตัวตน
- ภาพประกอบที่ควรใช้: ใช้ภาพตัวอย่าง externalUrl สองฝั่งและโดเมนที่ซ้ำกัน

**Speaker Script**
แม้ URL จะไม่ได้มีครบทุกโปรไฟล์ แต่ถ้าสองบัญชีชี้ไปยังโดเมนหรือเว็บไซต์เดียวกัน มันเป็นสัญญาณที่มีประโยชน์มากในเชิง identity resolution สไลด์นี้จึงช่วยอธิบายว่าระบบไม่ได้พึ่งแต่ชื่ออย่างเดียว

## Slide 47: Location Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- location_jaro
- location_token_sort
- ใช้กับ location ที่ผ่านการ normalize แล้ว
- ภาพประกอบที่ควรใช้: ใช้ before-after ของ location เช่น Bangkok / BKK / Thailand

**Speaker Script**
location เป็นฟีเจอร์เสริมที่ช่วยได้เมื่อผู้ใช้บอกตำแหน่งคล้ายกันข้ามแพลตฟอร์ม แต่ถ้าไม่ normalize ก่อน ฟีเจอร์นี้จะ noisy มาก สไลด์นี้จึงควรโยงกลับไปที่ขั้น location normalization ก่อนหน้า

## Slide 48: Mention และ Hashtag Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- mention_jaccard
- hashtag_jaccard
- hashtag_count_a และ hashtag_count_b
- ภาพประกอบที่ควรใช้: ใช้ตัวอย่าง bio ที่มี mentions และ hashtags ซ้ำกัน

**Speaker Script**
ในบางแพลตฟอร์ม ผู้ใช้ทิ้ง context ไว้ใน bio ผ่านการ mention หรือใช้ hashtag บางชุดซ้ำกัน ฟีเจอร์กลุ่มนี้จึงทำหน้าที่เก็บบริบทการใช้งานของบัญชี ไม่ใช่แค่ข้อมูลชื่อหรือคำอธิบายตัวเองเท่านั้น

## Slide 49: Stylometric Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- style_caps_diff
- style_avgword_diff
- style_biolen_ratio และ style_punct_diff
- ภาพประกอบที่ควรใช้: ใช้ภาพเปรียบเทียบสไตล์การเขียนสองฝั่ง เช่น ตัวพิมพ์ใหญ่และความยาว bio

**Speaker Script**
Stylometric features เป็นสัญญาณอ่อนแต่มีประโยชน์ เพราะบางคนมีสไตล์การเขียนคงที่ข้ามแพลตฟอร์ม เช่น ชอบพิมพ์ใหญ่ ชอบใช้เครื่องหมาย หรือเขียน bio ยาวใกล้เคียงกัน แม้ฟีเจอร์กลุ่มนี้ไม่ใช่ตัวตัดสินหลัก แต่ช่วยเติมบริบทบางกรณีได้

## Slide 50: Platform Pair Code

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- ระบุว่าคู่ที่เทียบกันเป็นแพลตฟอร์มใด
- เช่น twitter-instagram หรือ googleplus-twitter
- ช่วยให้โมเดลเรียนรู้ความยากต่างกันของแต่ละคู่แพลตฟอร์ม
- ภาพประกอบที่ควรใช้: ใช้ matrix 3x3 ของคู่แพลตฟอร์มพร้อม color coding

**Speaker Script**
ฟีเจอร์นี้ดูเรียบง่าย แต่มีประโยชน์เพราะคู่แพลตฟอร์มแต่ละแบบมีพฤติกรรมต่างกัน เช่นคู่ที่เกี่ยวกับ Google+ มักยากกว่า ฟีเจอร์ platform pair code จึงช่วยให้โมเดลรู้บริบทของคู่ข้อมูลที่กำลังตัดสินอยู่

## Slide 51: Image Availability Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- image_any_local
- image_both_local
- image_one_local_only
- ภาพประกอบที่ควรใช้: ใช้ไอคอนรูปภาพ 2 ฝั่งแล้วไล่สถานะ none / one / both

**Speaker Script**
ก่อนจะพูดถึงความคล้ายของภาพ เราต้องรู้ก่อนว่าคู่นั้นมีภาพให้ใช้หรือไม่ ฟีเจอร์ availability จึงเป็นชั้นแรกของสายภาพ เพราะถ้าข้อมูลภาพไม่มีหรือมีเพียงข้างเดียว ความหมายของฟีเจอร์ภาพอื่น ๆ ก็เปลี่ยนไปทันที

## Slide 52: Image Statistics Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- phash, dhash, brightness, contrast
- entropy, RGB distance, file-size ratio
- face count, face area, blur และ metadata
- ภาพประกอบที่ควรใช้: ใช้แผนภาพ feature family ของสายภาพเชิงสถิติ

**Speaker Script**
สายในรอบนี้ไม่ได้พึ่ง image embedding หนัก ๆ อย่างเดียว แต่ใช้ฟีเจอร์ภาพเชิงสถิติที่อธิบายได้ เช่น perceptual hash ความสว่าง ความคมชัด หรือจำนวนใบหน้า สิ่งนี้ช่วยให้ multimodal branch ยังตีความได้ในเชิงวิจัย

## Slide 53: Image-Caption Context Features

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- image_caption_any
- image_caption_bio_sbert_cross
- caption เทียบกับ fullName และ userName ของอีกฝั่ง
- ภาพประกอบที่ควรใช้: ใช้ภาพ flow จาก caption ของรูปไปเทียบกับ bio และชื่อของอีกฝั่ง

**Speaker Script**
จุดที่ทำให้ run หลักเป็น image_context ไม่ใช่แค่ image_stats คือฟีเจอร์กลุ่มนี้ เราไม่ได้ถามแค่ว่ารูปเหมือนกันไหม แต่ถามต่อว่าคำอธิบายของภาพฝั่งหนึ่งสอดคล้องกับข้อความของอีกฝั่งหรือไม่ ซึ่งเป็นการเชื่อมภาพกับข้อความโดยตรง

## Slide 54: Multimodal ในงานนี้เป็นการเพิ่มหลักฐานทีละขั้น

- ประเภท: content
- หมวด: Features
- ข้อความบนสไลด์:
- 23 ฟีเจอร์: text_attr_hybrid
- 37 ฟีเจอร์: image_stats
- 41 ฟีเจอร์: image_context
- ภาพประกอบที่ควรใช้: ใช้ staircase diagram 23 -> 37 -> 41 พร้อมคำอธิบายสั้น ๆ
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_sbert_gain.png

**Speaker Script**
สไลด์นี้ช่วยให้ผู้ฟังเห็นว่าความเป็น multimodal ของงานไม่ได้เกิดแบบกระโดดทีเดียว แต่ค่อย ๆ เพิ่มหลักฐานบนฐานเดิม เริ่มจาก text กับ attributes แล้วเติมภาพเชิงสถิติ และสุดท้ายเติม caption-context เพื่อทดสอบ incremental gain อย่างเป็นระบบ

## Slide 55: Modeling & Evaluation

- ประเภท: divider
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- จาก feature space ไปสู่การเลือกโมเดลหลักและการอ่านผลอย่างถูกบริบท
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section แบบมีไอคอน model, chart และ threshold

**Speaker Script**
จากตรงนี้เราจะดูว่าฟีเจอร์ทั้งหมดถูกใช้ในสายการทดลองไหนบ้าง โมเดลใดชนะ และเราตีความผลอย่างไรทั้งในระดับ test split และระดับ candidate pool จริง

## Slide 56: ภาพรวมของสายการทดลอง

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- Classical leakage-safe line
- Multimodal suite line
- Neural-network reference line
- ภาพประกอบที่ควรใช้: ใช้ roadmap ของ experiment line ทั้ง 3 สาย
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_experiment_roadmap.png

**Speaker Script**
งานนี้ไม่ได้อิงผลจากการทดลองชุดเดียว แต่มีสามสายที่ทำหน้าที่ต่างกัน สาย classical ใช้เป็น baseline ที่ปลอดจาก leakage สาย multimodal ใช้หาชุดฟีเจอร์และ run หลัก ส่วนสาย neural เป็น reference ว่าความซับซ้อนแบบ deep model ให้ประโยชน์เพิ่มจริงหรือไม่

## Slide 57: Classical Leakage-Safe Line

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- ใช้ 22 ฟีเจอร์เป็น baseline สำคัญ
- Gradient Boosting ให้ Test AP 0.9703 และ F1 0.9337
- strict report reference: logreg + sigmoid ที่ threshold 0.50
- ภาพประกอบที่ควรใช้: ใช้กราฟเปรียบเทียบ classical models หรือ heatmap
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/classical_leaksafe_cmp.png

**Speaker Script**
สไลด์นี้ใช้เล่าว่าแม้ก่อนเพิ่มภาพและ context ระบบ baseline ที่ออกแบบอย่าง leak-safe ก็แข็งแรงอยู่แล้ว โดยเฉพาะ Gradient Boosting ส่วน strict logistic reference ถูกเก็บไว้เป็นจุดอ้างอิงเชิง conservative สำหรับการรายงานที่ต้องการความเข้มงวดสูง

## Slide 58: Multimodal Suite Line

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- text_attr_hybrid -> image_stats -> image_context
- ทุก run ใช้ pair set และ split เดียวกัน
- ทำให้เห็นผลของการเพิ่ม modality ได้ชัด
- ภาพประกอบที่ควรใช้: ใช้ตารางหรือ bar chart เปรียบเทียบ 3 run หลัก
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/suite_cmp.png

**Speaker Script**
จุดแข็งของ multimodal suite คือทุก run ถูกเทียบบนเงื่อนไขเดียวกัน เราจึงตอบได้ค่อนข้างตรงว่าการเพิ่ม SBERT ภาพเชิงสถิติ หรือ caption-context ช่วยอะไรบ้าง และ gain ที่เห็นไม่ได้เกิดจากการเปลี่ยนชุดข้อมูลหรือ split

## Slide 59: Neural Reference Line

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- IdentityMLP ใช้เป็น neural baseline
- ช่วยตอบว่าความซับซ้อนแบบ neural คุ้มไหม
- ผลดีขึ้นจาก feature space ที่กว้างขึ้น แต่ยังไม่ชนะ GB
- ภาพประกอบที่ควรใช้: ใช้กล่องสรุป MLP พร้อม confusion matrix หรือ metric card
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/strict_identitymlp_cm.png

**Speaker Script**
สาย neural มีบทบาทเป็น reference ที่สำคัญ เพราะช่วยให้เราพูดได้อย่างมีหลักฐานว่าในบริบทของ pair-level tabular features ขนาดนี้ โมเดลที่ซับซ้อนกว่าไม่ได้ชนะเสมอไป และ tree-based model ยังเหมาะกว่าในแง่ performance กับ interpretability

## Slide 60: เหตุผลของค่า Tuning r075_h20

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- random_neg_ratio = 0.75
- hard_neg_ratio = 2.0
- เป็นช่วงที่ให้ trade-off ดีใน rebuilt experiment
- ภาพประกอบที่ควรใช้: ใช้กราฟ tuning trade-off หรือ top tuning rows
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/top_model_tune.png

**Speaker Script**
ค่า r075_h20 ไม่ได้เลือกจากความรู้สึก แต่เกิดจาก rebuilt experiment ที่ sweep ค่า negative sampling หลายจุด แล้วพบว่าช่วง random negative ประมาณ 0.75 และ hard negative ประมาณ 2.0 ถึง 2.5 ให้สมดุลที่ดีในเชิงผลลัพธ์

## Slide 61: ทำไม Exact-First ยังจำเป็นแม้มีโมเดลแล้ว

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- exact precision สูงมาก
- แยกคู่ที่ชัดมากออกจาก model stage ได้
- ทำให้โมเดลโฟกัสเฉพาะคู่กำกวม
- ภาพประกอบที่ควรใช้: ใช้ split diagram exact path กับ model path

**Speaker Script**
ตรงนี้ต้องสื่อให้ชัดว่า exact-first ไม่ใช่ heuristic เล็ก ๆ แต่เป็นส่วนสำคัญของสถาปัตยกรรม เพราะมันดึงคู่ที่ชัดมากออกไปก่อนด้วย precision สูงมาก แล้วปล่อยให้โมเดลทำงานกับส่วนที่ยากกว่าและมีความกำกวมจริง ๆ

## Slide 62: Blocking Keys ที่ใช้ใน Production Retrieval

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- username_prefix3
- fullname_prefix3
- external_domain
- ภาพประกอบที่ควรใช้: ใช้ภาพ blocking keys และลูกศรเชื่อม candidate buckets
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/blocking_keys.png

**Speaker Script**
หลัง exact-first ขั้นตอนที่ช่วยลด search space ต่อคือ blocking งานนี้เลือกใช้คีย์ที่อธิบายได้และสอดคล้องกับข้อมูลจริง ได้แก่ prefix ของ username prefix ของ fullName และ external domain เพื่อเก็บคู่ที่น่าจะเกี่ยวข้องไว้ก่อน

## Slide 63: Retrieval Summary ก่อนเข้าโมเดล

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- all cross-platform pairs = 449,149,239
- exact matches = 12,403
- candidate pairs = 2,073,842
- ภาพประกอบที่ควรใช้: ใช้ funnel หรือ number cards 3 ชั้น
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_retrieval_funnel.png

**Speaker Script**
ตัวเลขชุดนี้เป็นหัวใจของ retrieval stage เราเริ่มจากจักรวาลคู่ข้ามแพลตฟอร์มกว่า 449 ล้านคู่ แต่หลัง exact-first และ blocking เหลือ candidate pairs ที่ต้องส่งเข้าโมเดลเพียงประมาณ 2.07 ล้านคู่ ซึ่งทำให้ขั้น scoring มีความเป็นไปได้ในเชิงปฏิบัติ

## Slide 64: คุณภาพของ Exact-First เพียงอย่างเดียว

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- exact precision = 0.9949
- exact recall_global = 0.4220
- ดึงคู่จริงจำนวนมากออกได้ตั้งแต่ต้น
- ภาพประกอบที่ควรใช้: ใช้ precision-recall cards ของ exact stage

**Speaker Script**
สไลด์นี้ใช้เน้นบทบาทของ exact stage ว่ามันไม่ได้เก็บคู่ได้น้อยเกินไป แต่ช่วยกู้ recall ได้ถึง 42.20% ด้วย precision เกือบสมบูรณ์ เพราะฉะนั้น exact-first จึงควรถูกเล่าแยกจาก model stage อย่างชัดเจน

## Slide 65: Ground-Truth Coverage หลัง Retrieval

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- search-space reduction = 99.54%
- ground-truth coverage = 88.67%
- blocking stage optimize candidate recall ไม่ใช่ precision
- ภาพประกอบที่ควรใช้: ใช้ scatter หรือ two-metric card ระหว่าง reduction กับ coverage

**Speaker Script**
ถ้าจะอธิบาย retrieval อย่างถูกบริบท ต้องพูดสองค่าไปพร้อมกัน คือระบบลดพื้นที่ค้นหาได้ 99.54% และยังเก็บคู่จริงไว้ได้ 88.67% ค่าชุดนี้บอกว่า blocking stage ถูกออกแบบมาเพื่อคุม candidate recall ภายใต้ข้อจำกัดด้านคอมพิวต์

## Slide 66: เปรียบเทียบ 7 โมเดลบน Main Multimodal Run

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- GB: AP 0.9797, AUC 0.9737, F1 0.9333
- RF: AP 0.9764, AUC 0.9706, F1 0.9289
- MLP: AP 0.9734, AUC 0.9649, F1 0.9255
- ภาพประกอบที่ควรใช้: ใช้ model family heatmap หรือ ranking table
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_model_family_heatmap.png

**Speaker Script**
จุดนี้คือหลักฐานว่าการเลือกโมเดลหลักไม่ได้มาจาก intuition อย่างเดียว เรา benchmark หลาย family บน feature set เดียวกัน split เดียวกัน และ calibration logic เดียวกัน แล้วพบว่า Gradient Boosting ให้สมดุลดีที่สุดโดยรวม

## Slide 67: ทำไม Gradient Boosting จึงชนะ

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- เหมาะกับ tabular similarity features
- จับ nonlinear interaction ได้ดี
- ยังพออธิบายผลผ่าน feature importance ได้
- ภาพประกอบที่ควรใช้: ใช้ comparison card ของ GB เทียบกับ linear และ neural models
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/model_family_cmp.png

**Speaker Script**
คำตอบสั้น ๆ คือฟีเจอร์ของงานนี้เป็น pair-level tabular features ที่มี interaction ซับซ้อนพอสมควร Gradient Boosting จึงได้เปรียบทั้งเรื่อง performance และการอธิบายผล ขณะที่ linear models ตรงเกินไป และ neural models ยังไม่ให้ประโยชน์มากพอเมื่อเทียบกับต้นทุนความซับซ้อน

## Slide 68: Run หลักที่เลือกใช้จริง

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- image_context_r075_h20_s42
- best model = Gradient Boosting
- selected features = 41
- ภาพประกอบที่ควรใช้: ใช้ model pipeline ของ main run พร้อม label ว่า chosen run
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_model_pipeline.png

**Speaker Script**
run หลักที่ถูกเลือกสำหรับงานนี้คือ image_context_r075_h20_s42 เพราะได้ composite score สูงสุดใน suite และสะท้อน narrative แบบ multimodal ได้ครบที่สุด ทั้งในเชิงตัวเลขและในเชิงการอธิบายต่อกรรมการ

## Slide 69: Main Run Test Metrics

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- AP = 0.9789
- AUC = 0.9734
- Precision = 0.9507, Recall = 0.9179, F1 = 0.9340
- ภาพประกอบที่ควรใช้: ใช้ metric cards ขนาดใหญ่ 3-5 ใบ

**Speaker Script**
สไลด์นี้ให้ใช้เล่าตัวเลขหลักแบบตรงไปตรงมา เพื่อให้ผู้ฟังเห็นว่าก่อนเข้าสู่เรื่อง thresholding ตัว candidate scorer เองก็มีคุณภาพดีอยู่แล้ว ทั้งในเชิงการจัดอันดับและการจำแนกบน test split

## Slide 70: Confusion Matrix ของ Main Run

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- TN = 3717, FP = 209
- FN = 360, TP = 4027
- สะท้อนสมดุล precision กับ recall ที่ค่อนข้างดี
- ภาพประกอบที่ควรใช้: ใช้ confusion matrix ของ main run แบบเต็มหน้า
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_main_confusion_matrix.png

**Speaker Script**
confusion matrix ช่วยให้เล่าเรื่องความผิดพลาดได้จับต้องขึ้น ไม่ใช่แค่พูด AP หรือ F1 อย่างเดียว เราจะเห็นว่าจำนวน false positives และ false negatives อยู่ในระดับที่สมเหตุสมผลสำหรับ model stage ก่อนการปรับ threshold ระดับ production

## Slide 71: การกระจายของคะแนนที่ผ่าน Calibration แล้ว

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- MATCH กระจุกอยู่ทางขวา
- NO_MATCH กระจุกอยู่ทางซ้าย
- threshold เชิงจำแนกกับ threshold production คนละบริบท
- ภาพประกอบที่ควรใช้: ใช้ score distribution plot พร้อมเส้น threshold 0.35, 0.95, 0.98
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_score_distribution.png

**Speaker Script**
สไลด์นี้เหมาะกับการอธิบายว่าทำไมเราไม่ใช้ threshold เดียวสำหรับทุกบริบท threshold 0.35 ดีในเชิงจำแนกบน test split แต่ถ้าจะใช้จริงใน CRM เราต้องเลื่อน threshold ไปทางขวาเพื่อคุมความเสี่ยงของ false merge ให้เข้มงวดขึ้นมาก

## Slide 72: Threshold Trade-Off

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- threshold ต่ำ -> recall สูง แต่ precision ลด
- threshold สูง -> precision ดี แต่ recall หาย
- จึงต้องมี REVIEW tier
- ภาพประกอบที่ควรใช้: ใช้ precision-recall-f1 threshold sweep chart
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_threshold_tradeoff.png

**Speaker Script**
ข้อความสำคัญของสไลด์นี้คือไม่มี threshold เดียวที่ดีที่สุดสำหรับทุกเป้าหมาย ถ้าเน้น classifier metric เราจะได้ threshold แบบหนึ่ง แต่ถ้าเน้นลด false merge ใน CRM เราต้องยอมเสีย recall บางส่วนและชดเชยด้วย review queue แทน

## Slide 73: Calibration คืออะไร และทำไมต้องทำ

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- คะแนนดิบจากโมเดลไม่เท่ากับ probability ที่เชื่อถือได้เสมอ
- ใช้ Isotonic Regression เพื่อปรับคะแนน
- ทำให้ threshold มีความหมายเชิงปฏิบัติการมากขึ้น
- ภาพประกอบที่ควรใช้: ใช้ schematic แสดง raw score -> calibrator -> calibrated probability

**Speaker Script**
ถ้าจะใช้คะแนนของโมเดลไปตั้ง threshold จริง เราต้องเชื่อได้ว่าคะแนน 0.98 แปลว่า confidence สูงจริง ไม่ใช่แค่ตัวเลขที่โมเดล output มา การ calibration จึงเป็นขั้นเชื่อมจาก model evaluation ไปสู่ production decision

## Slide 74: ผลของ Calibration

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- validation ECE = 0.0000
- test ECE = 0.0044
- คะแนนหลัง calibration เกาะ observed rate ได้ดี
- ภาพประกอบที่ควรใช้: ใช้ calibration curve พร้อม marker ของแต่ละ bin
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_calibration_curve.png

**Speaker Script**
ผล calibration ของงานนี้ถือว่าดีมาก โดยเฉพาะบน test set ที่ ECE ต่ำมาก สไลด์นี้ช่วยอธิบายว่าทำไมเราจึงกล้าตั้ง operating point ระดับ 0.95 และ 0.98 เพื่อแยก MATCH กับ REVIEW ในบริบท production

## Slide 75: คุณภาพการจัดอันดับบน Full Candidate Pool

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- candidate AP = 0.6017
- candidate ROC-AUC = 0.9600
- precision@100 = 0.98 และ precision@5000 = 0.9472
- ภาพประกอบที่ควรใช้: ใช้ ranking quality chart พร้อม precision@k
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_ranking_quality.png

**Speaker Script**
นอกเหนือจาก test split เราดูผลบน candidate pool จริงทั้งหมดด้วย เพราะระบบ production ต้องจัดการข้อมูลที่ class imbalance สูงกว่ามาก ตัวเลข precision@k ที่ยังสูงแสดงว่าโมเดลสามารถดันคู่จริงขึ้นมาอยู่บนสุดของ ranking ได้ดี

## Slide 76: Error Analysis Overview

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- วิเคราะห์ fp_top.csv และ fn_top.csv อย่างละ 200 คู่
- false negatives กระจุกใน Google+ -> Twitter
- false positives จำนวนมากเป็น hard negatives ที่ชื่อคล้ายมาก
- ภาพประกอบที่ควรใช้: ใช้ error analysis figure แบบสรุปสองฝั่ง FP/FN
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_error_analysis_fp_fn.png

**Speaker Script**
สไลด์นี้สำคัญเพราะทำให้การอภิปรายผลไม่หยุดอยู่แค่ metric รวม เราเห็นทั้งลักษณะของคู่ที่ระบบพลาด และสาเหตุเชิงข้อมูลว่าความผิดพลาดมักเกิดจากชื่ออ่อนเกินไปหรือชื่อเหมือนกันมากเกินไปในคนละ entity

## Slide 77: ตัวอย่าง False Positive

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- mitchellhall / mitchell hall vs mitchellhall / mitchell hall
- ชื่อและ username ตรงกันมาก
- แต่เป็นคนละ user_folder จริง
- ภาพประกอบที่ควรใช้: ใช้ profile cards สองฝั่งแบบ side-by-side พร้อม highlight similarity สูง

**Speaker Script**
ตัวอย่างนี้เหมาะมากในการอธิบายว่าทำไมระบบยังต้องมี REVIEW tier เพราะแม้ชื่อและ username จะตรงกันแทบสมบูรณ์ โมเดลก็ยังมีโอกาสหลงเชื่อได้ว่าคือคนเดียวกัน ทั้งที่ ground truth บอกว่าเป็นคนละ entity

## Slide 78: ตัวอย่าง False Negative

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- sarojbeheraofficial vs indialiveblog
- เป็นคู่จริง แต่ชื่ออ่อนมาก
- probability เพียง 0.0041
- ภาพประกอบที่ควรใช้: ใช้ profile cards สองฝั่งแล้ว highlight ว่าชื่อไม่ช่วย แต่ identity เดียวกัน

**Speaker Script**
ตัวอย่าง false negative นี้สะท้อนข้อจำกัดอีกด้านหนึ่งของระบบ คือถ้าผู้ใช้เปลี่ยนชื่อผู้ใช้และชื่อแสดงผลจนสัญญาณแบบ name-based แทบหายไป โมเดลก็อาจปล่อยคู่จริงหลุดได้ โดยเฉพาะเมื่อสัญญาณเสริมอย่างภาพหรือ URL ไม่เพียงพอ

## Slide 79: Top 10 Feature Importance

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- fullname_token_sort = 0.4165
- username_token_sort = 0.2263
- bio_tfidf_cosine = 0.0588
- ภาพประกอบที่ควรใช้: ใช้กราฟแท่ง top feature importance
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_feature_importance_all41.png

**Speaker Script**
feature importance ของ main run ชี้ชัดว่ากลุ่มชื่อยังเป็นแกนกลางของระบบ โดยเฉพาะ fullname_token_sort และ username_token_sort ขณะที่ bio features เป็นหลักฐานเสริมที่มีน้ำหนักรองลงมา แต่ยังมีประโยชน์ชัดเจน

## Slide 80: Family-Level Importance

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- Name similarity = 89.51%
- Bio text = 7.71%
- Image-caption cross-signal = 0.65%
- ภาพประกอบที่ควรใช้: ใช้ donut หรือ stacked bar แสดงสัดส่วนตาม family
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_feature_family_importance.png

**Speaker Script**
การรวม importance ตามตระกูลทำให้ narrative ของงานชัดขึ้น คือระบบนี้ยังเป็น text-and-name centric model โดยมี bio และสัญญาณอื่นเป็นตัวช่วย ข้อมูลภาพในรอบนี้ยังเป็นสัญญาณเสริมมากกว่าสัญญาณหลัก

## Slide 81: ข้อจำกัดของสายภาพในรอบนี้

- ประเภท: content
- หมวด: Modeling & Evaluation
- ข้อความบนสไลด์:
- profiles with local image = 4,248
- profiles with caption = 2,173
- test pairs with both local images = 0
- ภาพประกอบที่ควรใช้: ใช้ image coverage figure หรือ coverage ladder
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_image_coverage.png

**Speaker Script**
สไลด์นี้ช่วยกันการตีความเกินจริงเรื่อง multimodal เราเห็นว่า image branch ช่วยจริง แต่ coverage ของภาพยังจำกัดมาก โดยเฉพาะใน test pairs ที่ไม่มีคู่ไหนมี local image ครบทั้งสองฝั่งเลย ทำให้รอบนี้ยังเป็น partial multimodal มากกว่า full image-to-image matching

## Slide 82: Production & CRM

- ประเภท: divider
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- จากคะแนนของโมเดล ไปสู่การตัดสินใจจริงในระบบ CRM
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section แบบมีไอคอน funnel, review queue และ CRM

**Speaker Script**
จากตรงนี้เราจะย้ายจากมุมมองเชิงโมเดลไปสู่มุมมองเชิงระบบจริง ว่าหลังจากได้ calibrated scores แล้ว เราเปลี่ยนมันให้เป็น MATCH REVIEW และ NO_MATCH อย่างไร

## Slide 83: Production Flow ของระบบ

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- exact-first และ blocking มาก่อน
- candidate scoring ทำหลัง retrieval
- ผลลัพธ์สุดท้ายไปสู่ CRM workflow
- ภาพประกอบที่ควรใช้: ใช้ threshold flow หรือ production pipeline diagram
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_threshold_flow.png

**Speaker Script**
สไลด์นี้คือภาพย่อของ production line ทั้งหมด exact-first และ blocking ช่วยลดคู่ก่อน จากนั้น candidate scorer ที่ผ่าน calibration แล้วจึงให้คะแนน และผลลัพธ์จะถูกแปลงเป็น decision tiers ก่อนเข้าสู่ review และ entity merge

## Slide 84: กฎ Dual Threshold ที่ใช้จริง

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- exact pair -> MATCH ทันที
- score >= 0.98 -> MATCH
- 0.95 <= score < 0.98 -> REVIEW
- ภาพประกอบที่ควรใช้: ใช้ decision tree สั้น ๆ หรือ threshold ladder 3 ชั้น
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_threshold_flow.png

**Speaker Script**
จุดนี้ให้เล่า logic การตัดสินใจจริงอย่างชัดเจน exact pair ถูกยอมรับทันที ส่วนคู่ที่ไม่ exact จะถูกพิจารณาตาม calibrated score ถ้าคะแนนสูงมากถึง 0.98 จึง auto-match ได้ แต่ถ้าอยู่ช่วง 0.95 ถึง 0.98 จะถูกพักไว้ให้คนตรวจ

## Slide 85: Operating Point ที่เลือกใช้

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- match_threshold = 0.98
- review_threshold = 0.95
- ออกแบบเพื่อคุม false merge ใน CRM
- ภาพประกอบที่ควรใช้: ใช้ dashboard สรุป threshold, tier counts และ quality
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_threshold_dashboard.png

**Speaker Script**
เหตุผลของ operating point นี้ไม่ได้มาจาก F1 สูงสุดอย่างเดียว แต่ยึดบริบทธุรกิจเป็นหลัก ใน CRM ความเสียหายจาก false merge สูงกว่าการส่งคู่ไปให้คนดูซ้ำ เราจึงตั้ง threshold ที่เข้มงวดและเพิ่ม REVIEW tier เข้ามารับคู่กำกวม

## Slide 86: Retrieval Funnel

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- 449,149,239 คู่ -> 12,403 exact
- เหลือ 2,073,842 candidate pairs
- สุดท้ายได้ MATCH 20,549 และ REVIEW 86,296
- ภาพประกอบที่ควรใช้: ใช้ funnel เต็มรูปจาก all pairs ไปสู่ exact, candidate, match, review
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_retrieval_funnel.png

**Speaker Script**
funnel นี้คือภาพที่ช่วยให้กรรมการเห็นคุณค่าของ pipeline ได้ชัดที่สุด เพราะมันแสดงทั้งการลด search space และการไหลของคู่ข้อมูลไปสู่ผลลัพธ์ปลายทางในระบบ production โดยยังผูกกับตัวเลขจริงของงานทั้งหมด

## Slide 87: จำนวนคู่ในแต่ละ Tier

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- MATCH = 20,549
- REVIEW = 86,296
- NO_MATCH = 1,979,400
- ภาพประกอบที่ควรใช้: ใช้ bar chart 3 แท่งหรือ stacked bar ของ final decisions

**Speaker Script**
สไลด์นี้ใช้ให้เห็นภาระงานจริงหลังตั้ง threshold คู่ส่วนใหญ่ถูกตัดเป็น NO_MATCH ได้อัตโนมัติ คู่ที่ได้ MATCH มีเพียงส่วนเล็กของทั้งหมด และคู่ที่ต้องส่งต่อให้คนตรวจมีประมาณ 4.14% ของ final decisions เท่านั้น

## Slide 88: คุณภาพของ Final MATCH Only

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- final match-only precision = 0.9550
- final match-only recall = 0.6711
- exact และ high-score match ทำงานร่วมกัน
- ภาพประกอบที่ควรใช้: ใช้ precision-recall cards ของ final match tier

**Speaker Script**
นี่คือตัวเลขที่ควรใช้ตอบคำถามว่า production-ready แค่ไหน ในทุก 100 คู่ที่ระบบยอม auto-merge จะมี false merge ราว 4 ถึง 5 คู่ ขณะเดียวกันระบบกู้คู่จริงได้เองประมาณสองในสาม ส่วนที่ยังไม่มั่นใจจะถูกส่งเข้า review แทน

## Slide 89: Production Decision Matrix

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- คู่จริง 19,624 คู่ไปอยู่ใน MATCH
- คู่จริง 4,065 คู่ไปอยู่ใน REVIEW
- คู่ไม่จริงหลุดเข้า MATCH เพียง 925 คู่
- ภาพประกอบที่ควรใช้: ใช้ production decision matrix แบบ 3 tier
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_production_decision_matrix.png

**Speaker Script**
decision matrix ทำให้เราอ่าน operating point เชิงปฏิบัติได้ชัดขึ้น MATCH tier ถูกออกแบบให้กินคู่จริงเป็นหลัก REVIEW ทำหน้าที่เป็น buffer zone และ NO_MATCH รับภาระการคัดคู่ไม่จริงส่วนใหญ่ของระบบไว้

## Slide 90: REVIEW Queue ไม่ใช่จุดอ่อน แต่คือ Safety Buffer

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- review precision = 0.0471
- review recall contribution = 0.1390
- ตั้งใจกันคู่กำกวมไว้ให้คนตัดสิน
- ภาพประกอบที่ควรใช้: ใช้ workflow box ของ reviewer ระหว่าง MATCH กับ NO_MATCH

**Speaker Script**
เวลาเห็น precision ของ REVIEW ต่ำ อย่าเพิ่งตีความว่าโมเดลแย่ เพราะธรรมชาติของ REVIEW tier คือเขตกันชนสำหรับคู่ที่ยังพอมีศักยภาพแต่ไม่ควรถูก auto-merge ระบบจงใจยอมให้ queue นี้มีความกำกวม เพื่อรักษาคุณภาพของ auto-match ให้สูงพอ

## Slide 91: CRM Entity Merge

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- รวมผล MATCH เข้าสู่ entity เดียว
- ใช้ union-find สำหรับ transitive closure
- ได้ unified customer view
- ภาพประกอบที่ควรใช้: ใช้แผนภาพหลายโปรไฟล์ไหลเข้ากลุ่ม entity เดียว
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_crm_outcomes.png

**Speaker Script**
หลังตัดสินว่าใคร match กันแล้ว งานยังไม่จบ ระบบต้องรวมโปรไฟล์ที่เป็นคนเดียวกันให้กลายเป็น entity เดียวจริง ๆ ซึ่งขั้นนี้ทำให้ผลลัพธ์ของโมเดลแปลงเป็น customer record ที่ระบบ CRM ใช้งานต่อได้

## Slide 92: จำนวน Unified Profiles ที่ได้

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- unified profiles = 19,799
- เป็นผลจาก exact + high-score matches + entity merge
- พร้อมต่อยอดสู่ customer 360
- ภาพประกอบที่ควรใช้: ใช้ big number card กับภาพ customer clusters

**Speaker Script**
ตัวเลข unified profiles เป็นตัวบอกผลลัพธ์ในระดับเอนทิตี ไม่ใช่แค่ระดับคู่ข้อมูล มันช่วยแปลผลจากงาน identity resolution ให้คนสายธุรกิจเห็นภาพมากขึ้นว่าเรารวมข้อมูลกระจัดกระจายกลับมาเป็นมุมมองลูกค้าระดับบุคคลได้เท่าไร

## Slide 93: Lead Scoring Tiers

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- HOT = 4,937
- WARM = 9,039
- COLD = 5,823
- ภาพประกอบที่ควรใช้: ใช้ stacked bar หรือ 3 cards ของ lead tiers

**Speaker Script**
เมื่อได้ unified profiles แล้ว ระบบยังไปต่อถึงการสร้าง lead tiers เพื่อใช้ในการวางแผนงานขายและการตลาด สไลด์นี้ช่วยย้ำว่างานวิจัยไม่ได้หยุดที่ matching แต่ไปถึงการสร้างผลลัพธ์ปลายทางที่ธุรกิจหยิบไปใช้ได้ทันที

## Slide 94: Business Use Case ที่เกิดขึ้นได้จริง

- ประเภท: content
- หมวด: Production & CRM
- ข้อความบนสไลด์:
- เห็นลูกค้าคนเดียวกันแบบรวมศูนย์
- จัดลำดับความสำคัญของ lead ได้ดีขึ้น
- วางแผน follow-up และ personalization ได้แม่นขึ้น
- ภาพประกอบที่ควรใช้: ใช้ before-after ของ CRM dashboard หรือ customer 360 mockup
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_crm_outcomes.png

**Speaker Script**
สไลด์นี้ควรปิด section production ด้วยภาษาธุรกิจ ว่าสุดท้ายแล้วสิ่งที่ระบบให้ไม่ใช่แค่ไฟล์คะแนน แต่คือความสามารถในการเห็นลูกค้าเป็นคนเดียวกันจริง ๆ และใช้ข้อมูลนั้นวางแผนงานขายและการตลาดได้ดีขึ้น

## Slide 95: Conclusion

- ประเภท: divider
- หมวด: Conclusion
- ข้อความบนสไลด์:
- สรุปว่าอะไรคือคำตอบหลักของงานนี้ และอะไรคือขอบเขตที่ควรต่อยอด
- ภาพประกอบที่ควรใช้: ใช้ภาพคั่น section แบบ clean มีคำว่า summary และ roadmap
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_experiment_roadmap.png

**Speaker Script**
จากตรงนี้เราจะสรุปคำตอบของงานวิจัย ว่างานนี้พิสูจน์อะไรได้แล้วบ้าง อะไรยังเป็นข้อจำกัด และถ้าจะทำต่อควรไปทางไหน

## Slide 96: คำตอบต่อวัตถุประสงค์ที่ 1

- ประเภท: content
- หมวด: Conclusion
- ข้อความบนสไลด์:
- ระบบ cross-platform identity resolution ถูกพัฒนาได้จริง
- main run ให้ AP 0.9789 และ F1 0.9340
- retrieval ลด search space ได้ 99.54%
- ภาพประกอบที่ควรใช้: ใช้ summary card ผูก metric ของ model กับ metric ของ retrieval

**Speaker Script**
วัตถุประสงค์แรกคือพัฒนาระบบเชื่อมโยงตัวตนข้ามแพลตฟอร์ม สำหรับคำตอบของงานนี้คือทำได้จริงทั้งในระดับ retrieval และ scoring ไม่ใช่แค่ตัวโมเดลดี แต่ระบบทั้งเส้นสามารถลดปัญหาขนาดใหญ่ให้เข้าสู่ระดับที่จัดการได้

## Slide 97: คำตอบต่อวัตถุประสงค์ที่ 2

- ประเภท: content
- หมวด: Conclusion
- ข้อความบนสไลด์:
- Gradient Boosting ชนะโดยรวม
- tree-based models เหมาะกับ pair-level tabular features
- multimodal run แบบ image_context ให้ผลดีที่สุด
- ภาพประกอบที่ควรใช้: ใช้ model comparison summary แบบย่อ
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_model_family_heatmap.png

**Speaker Script**
วัตถุประสงค์ที่สองคือการเปรียบเทียบแบบจำลอง ผลของงานค่อนข้างชัดว่า Gradient Boosting เป็นตัวเลือกที่เหมาะที่สุดในบริบทนี้ และการเพิ่มหลักฐานแบบ multimodal โดยเฉพาะ image_context ช่วยยกระดับผลลัพธ์ขึ้นอีกเล็กน้อยแต่สม่ำเสมอ

## Slide 98: คำตอบต่อวัตถุประสงค์ที่ 3

- ประเภท: content
- หมวด: Conclusion
- ข้อความบนสไลด์:
- สามารถใช้งานจริงใน production ได้
- final match-only precision = 0.9550
- มี review queue รองรับ human-in-the-loop
- ภาพประกอบที่ควรใช้: ใช้ threshold dashboard หรือ CRM outcomes summary
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_threshold_dashboard.png

**Speaker Script**
วัตถุประสงค์ที่สามคือความเป็นไปได้ในการใช้งานจริง ตรงนี้คำตอบคือใช่ เพราะระบบมี operating point ชัดเจน มีการ calibrate score ก่อนตัดสินใจ และมี review queue ทำหน้าที่เป็น safety layer ระหว่างโมเดลกับการใช้งานจริง

## Slide 99: ข้อจำกัดที่ต้องยอมรับ

- ประเภท: content
- หมวด: Conclusion
- ข้อความบนสไลด์:
- แพลตฟอร์มยังมีเพียง 3 แหล่ง และมี Google+ เป็นข้อมูล archive
- สายภาพยังเป็น partial multimodal
- ยังไม่มี temporal validation และ case-level explanation
- ภาพประกอบที่ควรใช้: ใช้ limitation board 3 ช่อง พร้อมไอคอน platform, image, time
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_image_coverage.png

**Speaker Script**
เพื่อให้สรุปอย่างซื่อตรง เราควรย้ำข้อจำกัดของงานด้วย ได้แก่ ความหลากหลายของแพลตฟอร์มยังไม่มาก สัญญาณภาพยังมี coverage จำกัด และงานยังไม่ได้ทดสอบในมิติเวลา หรืออธิบายผลระดับรายคู่ด้วย XAI อย่างเต็มรูปแบบ

## Slide 100: Future Work และ Q&A

- ประเภท: closing
- หมวด: Conclusion
- ข้อความบนสไลด์:
- ขยายไปยังแพลตฟอร์มใหม่และข้อมูลภาษาไทย
- เพิ่ม full image-to-image matching, XAI และ active learning
- เปิดให้ถามต่อเรื่อง pipeline, model หรือ production decision
- ภาพประกอบที่ควรใช้: ใช้สไลด์ปิดที่มี roadmap สั้น ๆ กับคำว่า Q&A ชัดเจน
- ไฟล์ใน repo ที่ใช้ได้: pub_multi/fig/ch04_experiment_roadmap.png

**Speaker Script**
สไลด์สุดท้ายให้ปิดงานด้วยสองส่วน ส่วนแรกคือทิศทางต่อยอด เช่น เพิ่มแพลตฟอร์มใหม่ เพิ่มสายภาพแบบเต็มรูปแบบ และนำ active learning กลับมาลด review queue ส่วนที่สองคือเปิดรับคำถาม โดยแนะนำว่าถ้าถูกถามลึกให้ย้อนกลับไปใช้สไลด์กลาง ๆ เป็น backup ได้ทันที
