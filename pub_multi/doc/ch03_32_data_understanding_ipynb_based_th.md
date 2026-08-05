# 3.2 การทำความเข้าใจข้อมูล (Data Understanding)

หัวข้อนี้เขียนโดยยึด [preprocess_pipeline_backup.ipynb](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/preprocess_pipeline_backup.ipynb) เป็นแหล่งอ้างอิงหลักสำหรับอธิบายโครงสร้างข้อมูล การรวมไฟล์ JSON การสร้าง DataFrame กลาง และการทำ normalization ในช่วงต้นของ pipeline โดยเน้นอธิบายสิ่งที่ notebook ทำจริง มากกว่าการอธิบายจาก artifact ปลายทางเพียงอย่างเดียว

## 3.2.1 Data Description

ข้อมูลต้นทางของงานนี้มาจากชุดข้อมูล LinkSocial ซึ่งถูกจัดเก็บในรูปแบบไฟล์ JSON ของโปรไฟล์สาธารณะจากสื่อสังคมออนไลน์หลายแพลตฟอร์ม ภายใน [preprocess_pipeline_backup.ipynb](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/preprocess_pipeline_backup.ipynb) ขั้น `Stage 2: Data Loading` ใช้ฟังก์ชัน `load_all_profiles()` เพื่ออ่านไฟล์จากโฟลเดอร์ `1.profile.data`, `2.profile.data` และ `3.profile.data` แล้วรวมเป็น DataFrame เดียว โดยฟังก์ชันดังกล่าวถูกออกแบบให้รองรับทั้งโครงสร้างแบบ 2 ชั้น (`folder/user_folder/*.json`) และแบบ 3 ชั้น (`folder/pair_folder/user_folder/*.json`) พร้อมทั้งเพิ่มข้อมูลกำกับ เช่น `user_folder`, `platform` และ `source_folder` จากตำแหน่งของไฟล์ในโฟลเดอร์ต้นทาง

ในเชิงโครงสร้าง ข้อมูลที่ได้จากขั้นโหลดถูกเก็บอยู่ในตัวแปร `df_raw` ซึ่ง notebook แสดงสรุปไว้ใน `Step 2.2: DataFrame Summary` ว่ามีทั้งหมด `36,807` แถว และ `11` คอลัมน์ ประกอบด้วย `userName`, `fullName`, `bio`, `location`, `externalUrl`, `pictureURL`, `user_folder`, `outputProfileName`, `platform`, `source_folder` และ `bigrams` จาก output เดียวกันพบว่าในระดับดิบ `userName` มีข้อมูล `36,724` แถว `fullName` `36,381` แถว `bio` `30,380` แถว `location` `12,871` แถว `externalUrl` `30,580` แถว และ `pictureURL` `24,699` แถว ตัวเลขเหล่านี้สะท้อนให้เห็นตั้งแต่ต้นว่าข้อมูลมีปัญหาเรื่องความไม่ครบถ้วน (missingness) แตกต่างกันมากตามชนิดของฟิลด์ โดยเฉพาะ `location` และ `pictureURL` ที่ขาดหายบ่อยกว่าฟิลด์ระบุตัวตนพื้นฐาน

อย่างไรก็ตาม มีข้อสังเกตสำคัญจาก notebook คือใน output ของ `df_raw` ที่แสดงไว้ ค่า `platform` และ `source_folder` ออกมาเป็น `unknown` ทั้งหมด เนื่องจากใน flow ที่ notebook ใช้ระหว่างการทดลองมีการ reload ข้อมูลจาก `cleaned_social_data.csv` แล้วกำหนดค่า `platform = 'unknown'` และ `source_folder = 'unknown'` ซ้ำในบาง cell ดังนั้น หากต้องการอธิบาย “ขั้นตอนการรวมไฟล์และฟิลด์ที่ใช้งาน” ควรยึด logic ของ `load_all_profiles()` ใน notebook เป็นหลัก แต่หากต้องการอ้างจำนวนโปรไฟล์แยกตามแพลตฟอร์มในรายงาน ควรอ้างจากชุดข้อมูล normalized ปลายทางแทน ไม่ควรใช้ค่า `unknown` จาก snapshot ระหว่างทางใน notebook เป็นข้อสรุปเชิงข้อมูล

ตารางที่ 3.2 ข้อมูลดิบในระดับ DataFrame กลางจาก `preprocess_pipeline_backup.ipynb`

| รายการ | ค่า |
| --- | ---: |
| ชื่อ DataFrame กลาง | `df_raw` |
| จำนวนแถว | 36,807 |
| จำนวนคอลัมน์ | 11 |
| คอลัมน์หลัก | `userName`, `fullName`, `bio`, `location`, `externalUrl`, `pictureURL`, `user_folder`, `outputProfileName`, `platform`, `source_folder`, `bigrams` |
| `userName` non-null | 36,724 |
| `fullName` non-null | 36,381 |
| `bio` non-null | 30,380 |
| `location` non-null | 12,871 |
| `externalUrl` non-null | 30,580 |
| `pictureURL` non-null | 24,699 |

## 3.2.2 Data Dictionary ของฟิลด์สำคัญ

ในเชิงโครงสร้างข้อมูล ฟิลด์สำคัญที่เกี่ยวข้องกับงาน Identity Resolution ใน notebook มีทั้งข้อมูลระบุตัวตนโดยตรงและข้อมูลบริบทประกอบ ฟิลด์หลักที่เกิดขึ้นและถูกใช้งานจริงตั้งแต่ช่วงต้นของ pipeline ได้แก่ `userName`, `fullName`, `bio`, `location`, `externalUrl`, `pictureURL`, `user_folder`, `outputProfileName`, `platform`, `source_folder` และ `bigrams` จากนั้นเมื่อเข้าสู่ช่วง `Stage 3: Data Cleaning` และ `Stage 3.3: Normalization` notebook จะค่อย ๆ สร้าง field ที่ผ่านการแปลงแล้วเพิ่มเติม เช่น `bio_urls`, `bio_url_count`, `bio_mentions`, `bio_mentions_count`, `location_type`, `location_valid`, `latitude`, `longitude`, `externalUrl_clean`, `external_domain` และ `profile_id`

ในเชิงบทบาทของฟิลด์ `userName` และ `fullName` เป็นสัญญาณหลักของตัวตนและถูกนำไปใช้ทั้งใน exact matching และ string similarity, `bio` ถูกทำความสะอาดและแยกองค์ประกอบย่อยออกมาเพื่อใช้สร้าง text features และ semantic features, `location` ถูกส่งเข้าสู่ฟังก์ชัน `clean_location_full()` เพื่อจำแนกประเภทและสกัดพิกัดเมื่อเป็นไปได้, `externalUrl` ถูก normalize เป็น `externalUrl_clean` และ `external_domain` เพื่อใช้ในการจับคู่ URL/domain, ส่วน `pictureURL` ทำหน้าที่เป็นจุดเริ่มต้นของสาย image processing ภายนอก notebook ซึ่งต่อไปจะถูกใช้สร้าง multimodal evidence ใน pipeline ขั้นหลัง

นอกจากนี้ `user_folder` และ `outputProfileName` มีความสำคัญเชิงโครงสร้างมาก เพราะ notebook ใช้สองฟิลด์นี้เป็นฐานในการสร้าง `profile_id` ใน `Step 3.8: Create Profile ID` โดยแปลง `outputProfileName` ให้เป็น normalized text แล้วใช้เป็น key ระดับโปรไฟล์/เอนทิตีสำหรับการเชื่อมโยงข้อมูลในขั้นถัดไป ขณะที่ `profile_row_id` ไม่ได้ถูกสร้างใน notebook นี้โดยตรง แต่เป็น field ของสาย downstream ที่ถูกเพิ่มเข้ามาใน [normalized_profiles_with_profile_id.csv](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv) เพื่อใช้เชื่อม artifact ข้าม stage เช่น scoring, match decisions และ CRM entity pipeline ดังนั้นหากอ้างอิงจาก notebook โดยตรง ควรแยกให้ชัดว่า `profile_id` เป็นผลของ preprocessing notebook ส่วน `profile_row_id` เป็น row-level key ที่เพิ่มขึ้นภายหลังในสาย deployment

ตารางที่ 3.3 Data dictionary แบบย่อของฟิลด์สำคัญใน notebook และ downstream linkage

| Field Name | คำอธิบาย | ตัวอย่าง | ใช้ในขั้นตอน |
| --- | --- | --- | --- |
| `userName` | ชื่อบัญชีผู้ใช้จากโปรไฟล์ | `john_doe` | exact match, username similarity, blocking |
| `fullName` | ชื่อที่แสดงบนโปรไฟล์ | `John Doe` | full-name similarity, text matching |
| `bio` | คำอธิบายตัวตนหรือข้อความแนะนำตนเอง | `traveler, photographer` | text cleaning, semantic/text features |
| `location` | สถานที่ที่ผู้ใช้ระบุ | `Bangkok` | location normalization, location similarity |
| `externalUrl` | ลิงก์ภายนอกที่ผู้ใช้ระบุ | `https://john.example.com` | URL/domain matching, exact-first |
| `pictureURL` | ลิงก์รูปโปรไฟล์ต้นทาง | image URL | image normalization, multimodal features |
| `platform` | แพลตฟอร์มของโปรไฟล์ | `twitter` | platform-aware analysis, cross-platform pairing |
| `user_folder` | ตัวระบุเอนทิตีระดับต้นทางจากโครงสร้างโฟลเดอร์ | `johndoe` | ground-truth construction, source linkage |
| `outputProfileName` | ชื่อโปรไฟล์ปลายทางที่ notebook ใช้เป็นฐานสร้าง ID | `AaronCohenArts` | source linkage, profile ID creation |
| `profile_id` | รหัสอ้างอิงระดับโปรไฟล์/เอนทิตีที่สร้างใน notebook จาก `outputProfileName` | normalized id | pair building, joins, downstream linkage |
| `profile_row_id` | รหัสแถวที่ใช้เชื่อม artifact ข้าม stage ในสาย downstream | integer row id | feature joins, scoring, CRM entity pipeline |
| `bio_urls` | URL ที่ดึงออกจาก bio | `my.site.com` | bio-derived features |
| `bio_mentions` | mention ที่ดึงออกจาก bio | `john` | social-context features |
| `location_type` | ประเภทของ location หลัง normalize | `text_location` / `coordinates` | location QA, location features |
| `external_domain` | domain ที่สกัดจาก externalUrl | `linkedin.com` | URL/domain matching |

## 3.2.3 ความสัมพันธ์ของข้อมูล การรวมไฟล์ และ Entity-Relationship ในระดับ preprocessing

ในระดับ preprocessing notebook ไม่ได้มองข้อมูลเป็นเพียงตารางเดียว แต่เป็นการไหลของข้อมูลจากหลายแหล่งเข้าสู่ DataFrame กลางก่อน แล้วค่อยแตกออกเป็น artifact ที่มีความหมายต่างกันตามหน้าที่ จุดเริ่มต้นอยู่ที่ไฟล์ JSON ต้นทางใน LinkSocial ซึ่งถูกอ่านผ่าน `load_all_profiles()` และรวมเป็น `df_raw` จากนั้น notebook สร้าง `df_clean` เพื่อใช้จัดการค่าว่างและป้องกันการเขียนทับข้อมูลดิบ แล้วคัดบางคอลัมน์ออกมาเป็น `df_already` ซึ่งประกอบด้วย `userName`, `fullName`, `bio`, `location`, `externalUrl`, `pictureURL` และ `platform` เพื่อใช้เป็นฐานของการทำ normalization ต่อไป หลังจากนั้นจึงสร้าง `df_nomalized` เพื่อเก็บค่าที่ผ่านการ clean และ normalize แล้วในแต่ละฟิลด์ เช่น `bio_urls`, `bio_mentions`, `location_type`, `latitude`, `longitude`, `externalUrl_clean`, `external_domain` และ `profile_id`

ในเชิง Entity-Relationship ของขั้น preprocessing สามารถอธิบายได้ว่า “โปรไฟล์” เป็นเอนทิตีหลักหนึ่งเดียว โดยมีแอตทริบิวต์พื้นฐานมาจาก JSON ต้นทาง และถูก enrich ด้วยฟิลด์ที่ derive เพิ่มขึ้นระหว่างทาง เช่น URL ที่แยกจาก bio, mentions ที่แยกจาก bio, สถานะความถูกต้องของ location และ domain ที่สกัดจาก externalUrl หากมองในเชิงโครงสร้างข้อมูล ตารางกลางที่ notebook สร้างจึงยังไม่ได้แยกเป็นหลายตารางสัมพันธ์กันแบบฐานข้อมูลเชิงสัมพันธ์เต็มรูปแบบ แต่ทำงานในลักษณะของ wide table ที่ค่อย ๆ เพิ่ม derived columns เข้าไปบนเอนทิตี `profile` เดียว อย่างไรก็ดี เมื่อผลจาก notebook ถูกส่งต่อเข้าสู่สาย downstream แล้ว ระบบจะเริ่มแยกออกเป็นหลายเอนทิตี เช่น `normalized_profile`, `pair`, `feature_matrix`, `match_decision`, `unified_profile` และ `lead_score` ซึ่งสามารถอธิบายแบบ ER Diagram ได้ชัดเจนในระดับ deployment

กล่าวอีกนัยหนึ่ง ถ้าอ้างอิงตาม notebook โดยตรง ความสัมพันธ์หลักที่สำคัญมี 2 เส้น คือ หนึ่ง ความสัมพันธ์ระหว่าง `Raw JSON` กับ `Profile Record` ที่ถูกโหลดเข้า DataFrame กลาง และสอง ความสัมพันธ์ระหว่าง `Profile Record` กับ `Derived Attributes` ที่เกิดจากการ clean/normalize ฟิลด์ย่อยต่าง ๆ เช่น bio-derived fields, location-derived fields และ URL-derived fields ส่วนความสัมพันธ์ในรูปแบบ pair-wise หรือ entity-wise จะเกิดขึ้นในขั้น downstream หลังจากที่ notebook สร้าง `profile_id` และส่งข้อมูลต่อไปยัง pipeline ขั้น train/test และ CRM

เพื่อให้ผู้อ่านเห็นภาพชัดขึ้น รูปที่ควรใช้ประกอบหัวข้อนี้มี 2 แบบ แบบแรกคือ `Data Lineage Diagram` ที่แสดงการไหลของข้อมูลจาก `Raw LinkSocial JSON Files` ไปยัง `df_raw`, `df_clean`, `df_already`, `df_nomalized`, `cleaned_social_data.csv` และต่อไปยัง `normalized_profiles_with_profile_id.csv` แบบที่สองคือ `Entity-Relationship Diagram` แบบเรียบง่ายที่เน้นเอนทิตี `PROFILE` เป็นศูนย์กลาง แล้วแสดง derived fields หลักที่ถูกเพิ่มจาก bio, location และ externalUrl โดยอาจอ้างภาพที่สร้างไว้แล้วใน [data_er_crowsfoot.svg](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/pub_multi/fig/data_er_crowsfoot.svg) สำหรับสาย downstream และใช้คำอธิบายเสริมว่าในระดับ notebook เอนทิตียังอยู่ในรูป wide profile table มากกว่าฐานข้อมูลแบบแยกหลายตาราง

## 3.2.4 ข้อสรุปสำหรับการเขียนรายงาน

หากต้องการเขียนหัวข้อ `Data Description`, `Data Dictionary` และ `Entity-Relationship Diagram` โดยยึด [preprocess_pipeline_backup.ipynb](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/preprocess_pipeline_backup.ipynb) เป็นหลัก ควรยึดหลักดังนี้

1. ใช้ notebook เป็นแหล่งอ้างอิงสำหรับ “โครงสร้างข้อมูลดิบ”, “ชื่อคอลัมน์”, “ลอจิกการรวมไฟล์” และ “ขั้นตอนการ clean/normalize”
2. ใช้ notebook เป็นหลักในการอธิบายบทบาทของ `user_folder`, `outputProfileName`, `profile_id`, `bio_urls`, `bio_mentions`, `location_type`, `external_domain`
3. ระบุอย่างชัดเจนว่า `profile_row_id` เป็น field ของสาย downstream ไม่ได้ถูกสร้างใน notebook นี้โดยตรง
4. หากต้องอ้างจำนวนโปรไฟล์แยกตามแพลตฟอร์มเพื่อใช้ในผลลัพธ์ production ควรอ้างจาก artifact ปลายทาง ไม่ใช่จาก snapshot `platform = unknown` ที่เกิดขึ้นระหว่างการทดลองใน notebook

จากขั้นตอนทั้งหมดข้างต้น จะเห็นได้ว่า notebook นี้ทำหน้าที่เป็นแหล่งอ้างอิงสำคัญของช่วง Data Understanding และ Data Preparation เนื่องจากเป็นจุดที่แสดงให้เห็นอย่างเป็นรูปธรรมว่าข้อมูลดิบถูกอ่านอย่างไร คอลัมน์ใดถูกสร้างหรือแปลงเพิ่มขึ้นอย่างไร และ field ใดมีบทบาทเป็นฐานของการเชื่อมโยงข้อมูลใน pipeline ขั้นถัดไป การเขียนรายงานโดยอ้างอิง notebook นี้จึงช่วยให้คำอธิบายเรื่อง Data Description, Data Dictionary และ Entity-Relationship มีความผูกกับสิ่งที่ทำจริงมากกว่าการอธิบายจากไฟล์ผลลัพธ์ปลายทางเพียงอย่างเดียว
