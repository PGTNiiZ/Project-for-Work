# รายการ 41 Features ของ Main Run `image_context_r075_h20_s42`

เอกสารนี้สรุปรายการฟีเจอร์ที่ใช้จริงในโมเดลหลักของสาย multimodal ซึ่งมาจาก run [image_context_r075_h20_s42](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/runs/image_context_r075_h20_s42/reports/experiment_report.json) และอ้างอิงรายชื่อจาก [feature_list.txt](/d:/66070260-Year3_Term2/Project1/Code/Project-for-Work/train_data/stage7_13_multimodal_suite/runs/image_context_r075_h20_s42/reports/feature_list.txt) โดยตรง

จุดสำคัญคือ pipeline สร้างฟีเจอร์ได้มากกว่านี้ใน stage 9 แต่ **final run ไม่ได้ใช้ทุกฟีเจอร์ที่สร้างได้** โมเดลหลักเลือกใช้เพียง `41` ฟีเจอร์ที่เหลือหลังการคัดเลือกและการจัดชุดทดลองใน multimodal suite ดังนั้นเวลาเขียนรายงานควรแยกให้ชัดระหว่าง

- `candidate feature pool`: ฟีเจอร์ทั้งหมดที่ pipeline สร้างได้
- `selected features`: ฟีเจอร์ที่ถูกใช้จริงใน run หลัก

## สรุปเป็นกลุ่มฟีเจอร์

| Feature Group | จำนวน | ฟีเจอร์ในกลุ่ม | ความหมาย |
| --- | ---: | --- | --- |
| `Identity string` | 6 | `username_*`, `fullname_*` | วัดความคล้ายของชื่อบัญชีและชื่อแสดง |
| `Bio text semantics` | 2 | `bio_tfidf_cosine`, `bio_sbert_cosine` | วัดความคล้ายของ bio ทั้งเชิงคำและเชิงความหมาย |
| `URL / domain` | 4 | `domain_*`, `url_jaccard` | วัดการซ้ำกันของลิงก์ภายนอกและโดเมน |
| `Location` | 2 | `location_jaro`, `location_token_sort` | วัดความคล้ายของ location ที่ normalize แล้ว |
| `Mention / hashtag` | 4 | `mention_jaccard`, `hashtag_*` | วัด context overlap จาก mentions และ hashtags |
| `Stylometric` | 4 | `style_*` | วัดลักษณะการเขียน เช่น ตัวพิมพ์ใหญ่ ความยาวคำ เครื่องหมายวรรคตอน |
| `Platform / metadata` | 1 | `platform_pair_code` | ระบุชนิดของคู่แพลตฟอร์มที่กำลังเปรียบเทียบ |
| `Image availability` | 3 | `image_any_local`, `image_both_local`, `image_one_local_only` | บอกว่าคู่โปรไฟล์นี้มีภาพให้ใช้แค่ไหน |
| `Image statistics` | 11 | `image_phash_sim` ถึง `image_metadata_any` | วัดความใกล้ของภาพจาก hash, คุณภาพภาพ, สี, blur, face metadata |
| `Image-caption context` | 4 | `image_caption_*` | วัดความสอดคล้องระหว่าง caption ของภาพกับ bio / fullName / userName อีกฝั่ง |
| รวม | 41 | main selected features | ฟีเจอร์ทั้งหมดที่ใช้จริงในโมเดลหลัก |

## ความสัมพันธ์กับแต่ละ experiment

สาม experiment ของ multimodal suite ใช้ฟีเจอร์ต่างกันดังนี้

| Experiment | จำนวนฟีเจอร์ | กลุ่มฟีเจอร์ที่ใช้ |
| --- | ---: | --- |
| `text_attr_hybrid` | 23 | Identity string + Bio text semantics + URL/domain + Location + Mention/hashtag + Stylometric + Platform/meta |
| `image_stats` | 37 | `text_attr_hybrid` ทั้งหมด + Image availability + Image statistics |
| `image_context` | 41 | `image_stats` ทั้งหมด + Image-caption context |

ดังนั้น หากต้องการอธิบายว่าเหตุใด run หลักจึงมี `41` ฟีเจอร์ คำตอบคือ

1. เริ่มจากแกน `text_attr_hybrid` จำนวน `23` ฟีเจอร์
2. เพิ่มฟีเจอร์ภาพเชิงสถิติและสถานะการมีภาพอีก `14` ฟีเจอร์ จนเป็น `37`
3. เพิ่มฟีเจอร์ context จาก caption ของภาพอีก `4` ฟีเจอร์ รวมเป็น `41`

## ฟิลด์ใดเป็น “โครงสร้างข้อมูล” และฟิลด์ใดเป็น “โมเดลฟีเจอร์”

มีจุดที่ต้องแยกให้ชัดเวลาเขียนรายงาน คือฟิลด์สำคัญของระบบไม่ได้เท่ากับฟีเจอร์ของโมเดลทั้งหมดโดยตรง ฟิลด์อย่าง `userName`, `fullName`, `bio`, `location`, `externalUrl`, `pictureURL` และ `platform` เป็นฟิลด์ตั้งต้นของข้อมูล แล้ว pipeline จึงแตกออกมาเป็นฟีเจอร์เชิงตัวเลขหลายตัวสำหรับใช้ฝึกโมเดล ขณะที่ `user_folder`, `profile_id` และ `profile_row_id` มีบทบาทเป็นคีย์เชิงโครงสร้าง ใช้สำหรับสร้าง ground truth, pair building, joins และการเชื่อมผลลัพธ์กลับไปยังโปรไฟล์ต้นทาง แต่ **ไม่ได้ถูกป้อนเป็นโมเดลฟีเจอร์โดยตรง** ใน run หลัก

สรุปแบบสั้นคือ

- `userName` แตกเป็น `username_jaro`, `username_lev`, `username_token_sort`
- `fullName` แตกเป็น `fullname_jaro`, `fullname_lev`, `fullname_token_sort`
- `bio` แตกเป็น `bio_tfidf_cosine`, `bio_sbert_cosine`, `mention_jaccard`, `hashtag_*`, `style_*`
- `location` แตกเป็น `location_jaro`, `location_token_sort`
- `externalUrl` แตกเป็น `domain_jaccard`, `url_jaccard`, `domain_count_a`, `domain_count_b`
- `platform` ถูกแปลงเป็น `platform_pair_code`
- `pictureURL` และ image artifacts แตกเป็น `image_*` และ `image_caption_*`
- `user_folder`, `profile_id`, `profile_row_id` เป็น keys ไม่ใช่ predictive features

## ตารางแจกแจงฟีเจอร์ทั้ง 41 ตัว

| ลำดับ | Feature Name | กลุ่ม | คำอธิบาย | มาจากขั้นตอน |
| ---: | --- | --- | --- | --- |
| 1 | `username_jaro` | Identity string | ความคล้ายของ `userName` ด้วย Jaro-Winkler | `stage9_features_pipeline_chunked.py` |
| 2 | `username_lev` | Identity string | ความคล้ายของ `userName` ด้วย Levenshtein แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 3 | `username_token_sort` | Identity string | ความคล้ายของ `userName` หลัง token sort | `stage9_features_pipeline_chunked.py` |
| 4 | `fullname_jaro` | Identity string | ความคล้ายของ `fullName` ด้วย Jaro-Winkler | `stage9_features_pipeline_chunked.py` |
| 5 | `fullname_lev` | Identity string | ความคล้ายของ `fullName` ด้วย Levenshtein แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 6 | `fullname_token_sort` | Identity string | ความคล้ายของ `fullName` หลัง token sort | `stage9_features_pipeline_chunked.py` |
| 7 | `bio_tfidf_cosine` | Bio text semantics | cosine similarity ของ bio ใน TF-IDF space | `stage9_features_pipeline_chunked.py` |
| 8 | `bio_sbert_cosine` | Bio text semantics | cosine similarity ของ bio ใน SBERT embedding space | `stage9_features_pipeline_chunked.py` |
| 9 | `domain_jaccard` | URL / domain | Jaccard overlap ของชุดโดเมนจาก `externalUrl` | `stage9_features_pipeline_chunked.py` |
| 10 | `url_jaccard` | URL / domain | Jaccard overlap ของ URL เต็ม | `stage9_features_pipeline_chunked.py` |
| 11 | `domain_count_a` | URL / domain | จำนวนโดเมนของโปรไฟล์ฝั่ง A แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 12 | `domain_count_b` | URL / domain | จำนวนโดเมนของโปรไฟล์ฝั่ง B แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 13 | `location_jaro` | Location | ความคล้ายของ location text ด้วย Jaro-Winkler | selected from normalized location features |
| 14 | `location_token_sort` | Location | ความคล้ายของ location text ด้วย token-sort ratio | selected from normalized location features |
| 15 | `mention_jaccard` | Mention / hashtag | Jaccard overlap ของชุด mentions ใน bio | `stage9_features_pipeline_chunked.py` |
| 16 | `hashtag_jaccard` | Mention / hashtag | Jaccard overlap ของชุด hashtags ใน bio | `stage9_features_pipeline_chunked.py` |
| 17 | `hashtag_count_a` | Mention / hashtag | จำนวน hashtags ของฝั่ง A แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 18 | `hashtag_count_b` | Mention / hashtag | จำนวน hashtags ของฝั่ง B แบบ normalize | `stage9_features_pipeline_chunked.py` |
| 19 | `style_caps_diff` | Stylometric | ความต่างของสัดส่วนตัวพิมพ์ใหญ่ | `stage9_features_pipeline_chunked.py` |
| 20 | `style_avgword_diff` | Stylometric | ความต่างของความยาวเฉลี่ยของคำ | `stage9_features_pipeline_chunked.py` |
| 21 | `style_biolen_ratio` | Stylometric | อัตราส่วนความยาว bio ระหว่างสองฝั่ง | `stage9_features_pipeline_chunked.py` |
| 22 | `style_punct_diff` | Stylometric | ความต่างของสัดส่วนเครื่องหมายวรรคตอน | `stage9_features_pipeline_chunked.py` |
| 23 | `platform_pair_code` | Platform / metadata | รหัสแทนคู่แพลตฟอร์ม เช่น twitter-instagram | `stage9_features_pipeline_chunked.py` |
| 24 | `image_any_local` | Image availability | อย่างน้อยหนึ่งฝั่งมี local image | `run_multimodal_suite.py` wrapper |
| 25 | `image_both_local` | Image availability | ทั้งสองฝั่งมี local image | `run_multimodal_suite.py` wrapper |
| 26 | `image_one_local_only` | Image availability | มีภาพเพียงฝั่งเดียว | `run_multimodal_suite.py` wrapper |
| 27 | `image_phash_sim` | Image statistics | ความคล้ายของ perceptual hash | `run_multimodal_suite.py` wrapper |
| 28 | `image_dhash_sim` | Image statistics | ความคล้ายของ difference hash | `run_multimodal_suite.py` wrapper |
| 29 | `image_brightness_diff` | Image statistics | ความต่างของความสว่างเฉลี่ยของภาพ | `run_multimodal_suite.py` wrapper |
| 30 | `image_contrast_diff` | Image statistics | ความต่างของ contrast | `run_multimodal_suite.py` wrapper |
| 31 | `image_entropy_diff` | Image statistics | ความต่างของ entropy ของภาพ | `run_multimodal_suite.py` wrapper |
| 32 | `image_rgb_l1` | Image statistics | ระยะห่างเฉลี่ยของค่าสี RGB | `run_multimodal_suite.py` wrapper |
| 33 | `image_filesize_ratio` | Image statistics | อัตราส่วนขนาดไฟล์ของภาพสองฝั่ง | `run_multimodal_suite.py` wrapper |
| 34 | `image_face_count_diff` | Image statistics | ความต่างของจำนวนใบหน้าในภาพ | `run_multimodal_suite.py` wrapper |
| 35 | `image_face_area_diff` | Image statistics | ความต่างของสัดส่วนพื้นที่ใบหน้า | `run_multimodal_suite.py` wrapper |
| 36 | `image_blur_diff` | Image statistics | ความต่างของค่า blur score | `run_multimodal_suite.py` wrapper |
| 37 | `image_metadata_any` | Image statistics | อย่างน้อยหนึ่งฝั่งมี image metadata พร้อมใช้ | `run_multimodal_suite.py` wrapper |
| 38 | `image_caption_any` | Image-caption context | อย่างน้อยหนึ่งฝั่งมี caption embedding | `run_multimodal_suite.py` wrapper |
| 39 | `image_caption_bio_sbert_cross` | Image-caption context | ความสอดคล้องระหว่าง caption ของภาพฝั่งหนึ่งกับ bio embedding ของอีกฝั่ง | `run_multimodal_suite.py` wrapper |
| 40 | `image_caption_fullname_token_cross` | Image-caption context | ความคล้ายระหว่าง caption ของภาพกับ `fullName` ของอีกฝั่ง | `run_multimodal_suite.py` wrapper |
| 41 | `image_caption_username_token_cross` | Image-caption context | ความคล้ายระหว่าง caption ของภาพกับ `userName` ของอีกฝั่ง | `run_multimodal_suite.py` wrapper |

## ข้อสังเกตสำคัญสำหรับใส่รายงาน

1. ฟีเจอร์ `41` ตัวนี้ไม่ใช่ฟีเจอร์ดิบทั้งหมดที่ระบบสร้างได้ แต่เป็น **selected feature set** ของ run หลัก
2. ฟีเจอร์กลุ่มข้อความและแอตทริบิวต์ยังคงเป็นแกนของโมเดล ส่วนฟีเจอร์ภาพทำหน้าที่เสริมสัญญาณ
3. ฟีเจอร์ภาพที่ใช้จริงไม่ได้พึ่ง image-image deep embedding อย่างเดียว แต่ใช้ image statistics และ caption-to-text cross features เป็นหลัก
4. หากจะทำตารางในรายงาน ควรแยกอย่างน้อยเป็น 3 ชั้น
   - text / attribute baseline
   - image statistics addition
   - image context addition

## ประโยคพร้อมใช้ในรายงาน

“โมเดลหลักของงานนี้ใช้ฟีเจอร์ทั้งหมด 41 ตัวใน run `image_context_r075_h20_s42` โดยสามารถแบ่งออกเป็น 10 กลุ่มหลัก ได้แก่ ฟีเจอร์ความคล้ายของชื่อบัญชีและชื่อแสดง, ฟีเจอร์ความคล้ายของ bio, ฟีเจอร์ด้าน URL/domain, ฟีเจอร์ด้าน location, ฟีเจอร์ด้าน mentions และ hashtags, ฟีเจอร์ stylometric, ฟีเจอร์ metadata ของคู่แพลตฟอร์ม, ฟีเจอร์สถานะการมีภาพ, ฟีเจอร์สถิติของภาพ และฟีเจอร์ context ที่เชื่อม caption ของภาพเข้ากับข้อมูลข้อความของอีกฝั่ง ทั้งนี้ ฟีเจอร์ 41 ตัวดังกล่าวเป็น selected feature set ของ run หลัก ไม่ใช่ฟีเจอร์ทั้งหมดที่ pipeline สามารถสร้างได้”
