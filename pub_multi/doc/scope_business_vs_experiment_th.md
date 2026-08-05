# ขอบเขตเชิงธุรกิจกับขอบเขตการทดลองของงานวิจัย

ตัวอย่างข้อความในหัวข้อนี้เขียนขึ้นเพื่อใช้อธิบายว่าเหตุใดเป้าหมายเชิงธุรกิจของระบบจึงกว้างกว่าขอบเขตการทดลองหลัก และเหตุใดรายงานผลหลักของงานจึงยึดกรณี cross-platform เป็นสำคัญ ทั้งนี้ ตัวเลขทั้งหมดด้านล่างอ้างอิงจากการอ่านชุดข้อมูล [normalized_profiles_with_profile_id.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv) โดยใช้เงื่อนไขคัดโปรไฟล์ที่มี `profile_row_id` และ `profile_id` พร้อมใช้งานตามที่กำหนดในฟังก์ชัน `load_profiles()` ของ [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py#L78) และอ้างอิงค่ารายงานหลักจาก [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json)

ในเชิงเป้าหมายทางธุรกิจ ระบบบริหารลูกค้าสัมพันธ์ควรสามารถระบุได้ว่าโปรไฟล์ใดเป็นบุคคลเดียวกันโดยไม่จำกัดว่าโปรไฟล์เหล่านั้นจะอยู่คนละแพลตฟอร์มหรืออยู่ในแพลตฟอร์มเดียวกัน กล่าวอีกนัยหนึ่ง ขอบเขตเชิงธุรกิจของระบบคือการทำ identity resolution ในระดับเอนทิตีลูกค้า ไม่ใช่เพียงการเชื่อมโยงข้ามแพลตฟอร์มเท่านั้น หากพิจารณาจากชุดข้อมูลที่ผ่านการคัดโปรไฟล์ที่มีคีย์อ้างอิงพร้อมใช้งานแล้ว จะมีโปรไฟล์ทั้งหมด 36,804 รายการ ซึ่งในเชิงนิยามของปัญหา หากนับคู่แบบไม่เรียงลำดับทั้งหมดจะได้ `36,804 choose 2 = 677,248,806` คู่ ตัวเลขนี้สะท้อนขอบเขตเชิงแนวคิดของงานในมุมมอง all-platform identity resolution

อย่างไรก็ตาม เมื่อพิจารณาในเชิงการทดลองหลักของโครงการ โค้ดต้นฉบับถูกออกแบบให้มุ่งเน้นกรณี cross-platform เป็นหลักอย่างชัดเจน ทั้งในขั้น pair building และในขั้น retrieval โดยใน [stage8_pair_builder.py](/d:/66070260-Year3_Term2/Project1/Code/stage8_pair_builder.py#L54) มีการกำหนด `CROSS_PLATFORM_ONLY = True` และใน [run_full_candidate_pipeline.py](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/run_full_candidate_pipeline.py#L99) ฟังก์ชัน `count_cross_platform_all_pairs()` จะนับเฉพาะคู่ที่อยู่คนละแพลตฟอร์มเท่านั้น เมื่อใช้จำนวนโปรไฟล์รายแพลตฟอร์มจากรายงานหลัก ได้แก่ Twitter 13,959 โปรไฟล์, Google+ 11,889 โปรไฟล์ และ Instagram 10,956 โปรไฟล์ จะได้จำนวนคู่ข้ามแพลตฟอร์มทั้งหมดเท่ากับ `11,889 x 10,956 + 11,889 x 13,959 + 10,956 x 13,959 = 449,149,239` คู่ ซึ่งตรงกับค่าที่รายงานไว้ใน [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json)

หากคำนวณส่วนที่เหลือของ search space จะพบว่าคู่แบบ same-platform มีจำนวนรวม `228,099,567` คู่ ซึ่งเกิดจาก `11,889 choose 2 + 10,956 choose 2 + 13,959 choose 2` กล่าวคือ ในเชิงปริมาณ same-platform pairs มีอยู่จริงจำนวนมาก แต่ปัญหาสำคัญไม่ได้อยู่ที่จำนวนคู่ทั้งหมด แต่อยู่ที่จำนวนคู่เชิงบวกที่สามารถใช้เป็น ground truth ได้ เมื่อใช้ `user_folder` เป็นตัวแทนเอนทิตีต้นทางและนับคู่เชิงบวกจากชุดข้อมูลเดียวกันภายใต้เงื่อนไข valid profile แบบเดียวกับ pipeline หลัก จะพบว่าคู่เชิงบวกแบบ cross-platform มี 29,243 คู่ ขณะที่คู่เชิงบวกแบบ same-platform มีเพียง 4 คู่เท่านั้น ความแตกต่างนี้ทำให้กรณี same-platform ยังไม่มีตัวอย่างเชิงบวกเพียงพอสำหรับการฝึกและประเมินผลอย่างมั่นคงในระดับที่ใช้สรุปเชิงวิชาการได้

ด้วยเหตุนี้ ผู้วิจัยจึงกำหนดให้การทดลองหลักของโครงการมุ่งเน้น cross-platform identity linkage ซึ่งเป็นส่วนของปัญหาที่มีทั้งจำนวน ground truth และโครงสร้างการประเมินรองรับชัดเจนกว่า ผลการทดลองหลักของงาน เช่น จำนวนคู่ ground truth 29,243 คู่, จำนวน exact matches 12,403 คู่, จำนวน candidate pairs 2,073,842 คู่ และ retrieval coverage 88.67% ที่รายงานใน [full_pipeline_report.json](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_14_full_candidate_pipeline/reports/full_pipeline_report.json) จึงควรถูกตีความว่าเป็นผลลัพธ์สำหรับกรณี cross-platform โดยตรง ไม่ใช่ผลยืนยันประสิทธิภาพของ all-platform identity resolution ทั้งหมด

ดังนั้น หากจะสรุปให้ตรงที่สุด ขอบเขตเชิงธุรกิจของระบบในงานนี้คือการรวมโปรไฟล์ที่เป็นบุคคลเดียวกันให้ได้โดยไม่จำกัดแพลตฟอร์ม แต่ขอบเขตเชิงทดลองที่ใช้พิสูจน์ประสิทธิภาพในรายงานฉบับนี้เป็น cross-platform identity resolution เนื่องจากข้อจำกัดของข้อมูลกำกับในกรณี same-platform ที่มีอยู่น้อยมาก การแยกขอบเขตทั้งสองชั้นออกจากกันทำให้รายงานมีความน่าเชื่อถือมากขึ้น เพราะผู้อ่านจะเห็นชัดเจนว่า ตัวระบบสามารถถูกออกแบบให้รองรับเป้าหมายทางธุรกิจที่กว้างกว่าได้ แต่หลักฐานเชิงประจักษ์ที่มีอยู่ในงานปัจจุบันรองรับการสรุปผลในกรณี cross-platform ได้แข็งแรงกว่าอย่างมีนัยสำคัญ

## ตารางเปรียบเทียบขอบเขตของงาน

| มิติ | ขอบเขตเชิงธุรกิจ (Business Scope) | ขอบเขตการทดลองหลัก (Experimental Scope) | ข้อจำกัดของข้อมูล (Data Limitation) | ข้อสรุปที่ควรใช้ในรายงาน |
| --- | --- | --- | --- | --- |
| เป้าหมายของระบบ | ระบุว่าโปรไฟล์ใดเป็นบุคคลเดียวกัน ไม่จำกัดแพลตฟอร์ม | ประเมินและพัฒนาระบบโดยเน้นกรณีข้ามแพลตฟอร์ม | กรณี same-platform มี ground truth เชิงบวกน้อยมาก | ขอบเขตธุรกิจกว้างกว่าขอบเขตทดลอง |
| หน่วยของปัญหา | ทุกคู่ที่เป็นไปได้ของโปรไฟล์ valid | เฉพาะคู่ข้ามแพลตฟอร์ม | โค้ด pair building และ retrieval ตัด same-platform ออกเป็นหลัก | รายงานหลักควรระบุชัดว่าใช้ cross-platform search space |
| Search space | ทุกคู่ไม่ซ้ำของ 36,804 โปรไฟล์ เท่ากับ `677,248,806` คู่ | เฉพาะคู่ข้ามแพลตฟอร์ม `449,149,239` คู่ | same-platform pairs มี `228,099,567` คู่ แต่ไม่ใช่ search space หลักของการทดลอง | ค่า `449,149,239` ไม่ได้ผิด แต่เป็นค่าของ experimental scope |
| Positive pairs | ควรมีทั้ง same-platform และ cross-platform | ใช้ positive pairs ข้ามแพลตฟอร์มเป็นหลัก | same-platform positive มีเพียง `4` คู่ ขณะที่ cross-platform positive มี `29,243` คู่ | ไม่ควรอ้างว่าผลหลักครอบคลุม same-platform อย่างแข็งแรง |
| Pair building | ในเชิงระบบควรสร้างคู่ที่รองรับการรวม identity ทั้งหมด | ใช้ `CROSS_PLATFORM_ONLY = True` ในการสร้างคู่ฝึก | training data สำหรับ same-platform ไม่เพียงพอ | การฝึกหลักจึงเหมาะกับ cross-platform มากกว่า |
| Retrieval | ในทางธุรกิจควรสามารถค้นหาคู่จริงทุกแบบ | exact-first + blocking ถูกออกแบบสำหรับ cross-platform | coverage ที่รายงานเป็น coverage ของ cross-platform เท่านั้น | retrieval result ต้องตีความในบริบทเดียวกับ search space ที่ใช้ |
| Evaluation | ควรประเมินได้ทั้ง all-platform และ cross-platform | ประเมินหลักบน cross-platform ground truth | ชุดข้อมูลไม่เอื้อต่อการประเมิน same-platform อย่างน่าเชื่อถือ | ผลลัพธ์หลักของรายงานควรระบุขอบเขตการประเมินอย่างชัดเจน |
| การนำไปใช้จริง | สามารถต่อยอดสู่ all-platform identity resolution ได้ | pipeline ปัจจุบันเป็นฐานสำหรับ cross-platform deployment | ต้องมีข้อมูลกำกับ same-platform เพิ่มจึงจะพิสูจน์ได้ครบ | same-platform ควรถูกเสนอเป็นงานต่อยอด |

## ตัวเลขสำคัญที่ควรอ้างในรายงาน

| รายการ | ค่า |
| --- | ---: |
| โปรไฟล์ valid ที่ใช้ใน full pipeline | 36,804 |
| Twitter profiles | 13,959 |
| Google+ profiles | 11,889 |
| Instagram profiles | 10,956 |
| all unordered pairs | 677,248,806 |
| cross-platform pairs | 449,149,239 |
| same-platform pairs | 228,099,567 |
| cross-platform positive pairs | 29,243 |
| same-platform positive pairs | 4 |
| exact match pairs | 12,403 |
| candidate pairs for model | 2,073,842 |
| retrieval coverage | 88.67% |

## ประโยคสรุปสั้นสำหรับใช้ปิดย่อหน้า

กล่าวโดยสรุป งานวิจัยนี้มีขอบเขตเชิงธุรกิจที่มุ่งสู่การรวมอัตลักษณ์ของลูกค้าในทุกกรณี แต่มีขอบเขตเชิงทดลองที่มุ่งเน้น cross-platform identity linkage เนื่องจากข้อจำกัดของชุดข้อมูลกำกับในกรณี same-platform ดังนั้น ตัวเลข search space, retrieval coverage และผลประเมินหลักที่รายงานในงานฉบับนี้จึงควรถูกตีความภายใต้กรอบ cross-platform เป็นสำคัญ ขณะที่การขยายสู่ all-platform identity resolution ควรถูกนำเสนอในฐานะทิศทางการพัฒนาต่อไปของระบบ
