# Implementation Plan — ปรับปรุงโมเดล Identity Resolution (จับ False Negative ให้ได้มากขึ้น)

เป้าหมาย: เพิ่ม recall บนคู่ที่โมเดลปัจจุบันพลาด โดย **ไม่ทำลาย precision** และให้ได้ผลที่เขียนลงเล่มได้
ทุกตัวเลขในแผนนี้มาจากการวิเคราะห์ไฟล์จริง (`train_all.parquet`, main_report.json) ไม่ใช่ค่าคาดเดา

## สถานะปัจจุบัน (baseline ที่ต้องเอาชนะ / อ้างอิง)
- GB image_context (41 features): **test P=0.9507 · R=0.9179 · F1=0.9340 · AP=0.9789**
- FN บน test = **360 คู่** แบ่งเป็น EASY(name≥0.5)=240 · MID(0.3–0.5)=115 · HARD(zero-lexical)=5
- weak-name recall = 0.216 · easy-name recall = 0.943
- **กำแพงที่พิสูจน์แล้ว:** ใน 4,387 test positives มี url signal 5 คู่, geo signal 0 คู่ → สัญญาณ non-lexical ที่เชื่อถือได้ (r_url=0.93, r_geo=0.999) มี coverage ≈ 0%
- noisy-OR fusion ทดสอบแล้ว: AP 0.942 < GB → พิสูจน์ว่า "สมการไม่ใช่คอขวด, coverage คือกำแพง"

## หลักการควบคุมการทดลอง (บังคับใช้ทุก phase)
1. เลือก threshold / hyperparameter **บน validation เท่านั้น** แตะ test แค่ครั้งเดียวตอนรายงานผลสุดท้าย
2. ใช้ split เดิม (`split_name` ใน train_all) — component-aware, overlap = 0 ตาม main_report
3. ทุก phase ต้องเทียบกับ GB baseline บน **slice เดียวกัน** (easy/mid/hard, ต่อ platform-pair)
4. รายงาน precision คู่กับ recall เสมอ — ห้ามโชว์ recall ลอย ๆ

---

## Phase 0 — Evaluation Harness (รากฐาน, ครึ่งวัน)

**เป้า:** สคริปต์เดียวที่ผลิตตาราง slice-metrics + per-platform-pair ซ้ำได้ ใช้เป็นเครื่องวัดของทุก phase

**ไฟล์:** `stage10_eval_harness.py`

**ตรรกะ:**
- รับ: DataFrame ที่มี `label`, `probability` (หรือ score ใด ๆ), `split_name`, ฟีเจอร์ชื่อ, `platform_a/b`
- คำนวณ: `name_max = max(username/fullname × jaro/token_sort)`, `bio_max = max(bio_tfidf, bio_sbert)`
- นิยาม slice: EASY (name_max≥0.5), MID (0.3≤name_max<0.5), HARD (name_max<0.3 & bio_max<0.3)
- output ต่อ (โมเดล × threshold):
  - overall P/R/F1/AP
  - recall แยก 3 slice
  - recall แยก 6 platform-pair (พร้อม support)
  - FN count + FN breakdown ต่อ slice/platform
- เซฟเป็น `reports/eval_<tag>.csv` + `reports/eval_<tag>.json`

**Acceptance:** รันบน GB baseline แล้วได้เลขตรงกับ main_report (P=0.9507, R=0.9179) ± 0.001

**ลงเล่ม:** ตาราง slice-recall นี้คือ **หลักฐานใหม่ของบท 4** (เล่มปัจจุบันยังไม่มี)

---

## Phase 1 — Adaptive / Per-Platform Threshold (win จริง, effort ต่ำ, 1 วัน)

**เป้า:** กู้ FN ที่ borderline (prob 0.20–0.35, ~70 คู่) และ easy-slice ที่โมเดลกลัวเกิน โดยลด threshold
เฉพาะจุดที่ recall แย่ ไม่ใช่ทั้งระบบ — FN กระจุกที่ `googleplus→twitter` (191/360)

**ไฟล์:** `stage11_adaptive_threshold.py`

**ตรรกะ:**
1. บน **validation**: สำหรับแต่ละ platform-pair หา threshold ที่ max F1 (หรือ max recall ภายใต้ precision≥0.93)
2. เงื่อนไขกัน overfit: platform-pair ที่มี positive < 200 บน val → ใช้ global threshold แทน (ไม่ tune แยก)
3. บันทึก threshold map → `models/platform_thresholds.json`
4. บน **test**: apply threshold ตาม platform-pair ของแต่ละคู่ → วัดผลผ่าน harness (Phase 0)
5. เทียบ 3 แบบ: (a) global 0.35 เดิม, (b) global sweep-optimal, (c) per-platform

**Acceptance (gate):** per-platform ต้องได้ **recall เพิ่ม ≥ +1.5%** โดย precision ลด **≤ 1%** เทียบ baseline
ถ้าไม่ผ่าน gate → ทิ้งแนวนี้ ใช้ global sweep-optimal แทน

**คาดผล (จาก threshold sweep จริง):** recall 0.918 → ~0.93–0.94, precision ~0.93–0.94
FN 360 → ~275 (กู้ ~85 คู่ ส่วนใหญ่จาก easy-borderline)

**ลงเล่ม:** บท 4.4 (decision thresholding) — เพิ่มหัวข้อ "adaptive operating point"; ตารางเทียบ 3 แบบ

---

## Phase 2 — Noisy-OR Fusion เป็น Analytical Contribution (สร้างแล้ว, เหลือ write-up, 1–2 วัน)

**เป้า:** ใช้ noisy-OR (มีอยู่แล้วใน `stage9_noisy_or_fusion.py`) เป็น "สมการใหม่ที่ทดลองแล้ว" +
พิสูจน์เพดาน coverage เชิงประจักษ์ = contribution เชิงวิเคราะห์ของ thesis

**งานที่เหลือ:**
1. `stage9b_ceiling_analysis.py` — คำนวณ **recall ceiling** อย่างเป็นทางการ:
   - ต่อ slice: สัดส่วน positive ที่มี ≥1 สัญญาณ non-lexical ที่ discriminative (bio_sbert≥0.5 | url>0 | geo>0 | image-match)
   - พิสูจน์: "X% ของ weak-name positives ไม่มีสัญญาณเลย → recall ceiling = 1 − X%"
   - ตัวเลขที่มีแล้ว: weak-name positives 850, มีสัญญาณ 59.4%, ไม่มีเลย 40.6%
2. เทียบตาราง: GB vs Noisy-OR vs Ensemble (มีเลขแล้วใน `noisy_or_report.json`)
3. เขียน reliability table (r_url=0.93, r_geo=0.999, r_bio=0.07) + ตีความว่า "สัญญาณที่เชื่อได้มี coverage 0%"

**Acceptance:** ceiling analysis reproduce ได้ + ผลชี้ชัดว่า fusion ไม่ชนะเพราะ coverage (ไม่ใช่เพราะ tuning)

**ลงเล่ม:**
- บท 3 (methodology): สมการ noisy-OR + นิยาม availability-gated fusion
- บท 4: ผลเทียบ + reliability table
- บท 5.3 (ข้อจำกัด): เปลี่ยนจากคำพูดลอย ๆ เป็น "recall bounded by signal coverage (พิสูจน์เชิงประจักษ์)"

---

## Phase 3 — Data Enrichment (ทางเดียวที่ทะลุเพดาน, stretch, 3–5 วัน)

**เป้า:** ยก coverage ของสัญญาณ reliability สูง (url/geo/image) จาก ~0% ให้ noisy-OR มีของจริงให้ fuse
ทำเฉพาะถ้ามีเวลาเหลือ — เป็น future work ที่ทำจริงได้

**ไฟล์:** `stage12_enrich_signals.py`

**งานย่อย (แต่ละอันวัด coverage gain แยก):**
| งาน | วิธี | คาด coverage gain | ความเสี่ยง |
|---|---|---|---|
| Expand external URL | ขยาย short-link → registrable domain (tldextract) | url 0.1% → ? | Google+ ตาย, link เน่า |
| Geocode location | Nominatim (rate-limit 1 req/s, cache) | geo 0% → ~35% (location มี 35%) | free-text กำกวม |
| Both-side image | ดึงรูปให้ครบ 2 ฝั่ง + CLIP/face embedding cosine | both_local 0 → ? | pictureURL เน่า, default avatar |

**Acceptance:** อย่างน้อย 1 สัญญาณต้องได้ coverage บน weak-name positives > 15% ถึงจะคุ้มเทรนใหม่
ถ้าทุกตัว < 15% → รายงานว่า "ข้อมูล LinkSocial enrich ไม่ขึ้น" = ก็เป็นผลวิจัย

**ลงเล่ม:** บท 5.4 (future work) — เปลี่ยนจากข้อเสนอลอย ๆ เป็น "ทดลองแล้ว, coverage gain = ..."

---

## Phase 4 — Thesis Integration (2 วัน)

**Deliverables ที่ต้องผลิต:**
1. ตาราง slice-recall (Phase 0) → บท 4 รูป/ตารางใหม่
2. ตาราง adaptive-threshold เทียบ 3 แบบ (Phase 1) → บท 4.4
3. สมการ noisy-OR + reliability table + ceiling (Phase 2) → บท 3.4 + 4
4. กราฟ: recall-vs-name_similarity-slice (bar), FN composition (stacked), coverage waterfall
5. แก้ข้อความบท 5.2.2/5.3 ให้อ้างตัวเลขจริงแทนคำบรรยาย

**ประโยคหลักที่ได้ (เขียนได้เลย):**
> "ระบบได้ recall 0.94 บน 96.5% ของคู่ที่มีสัญญาณชื่อ แต่ recall ตกเหลือ 0.21 ใน 3.4% ที่ชื่อ
> แตกต่าง ซึ่ง 40.6% ของกลุ่มนี้ไม่มีสัญญาณ non-lexical ใด ๆ ให้ใช้ — เราออกแบบ noisy-OR
> evidence fusion และพิสูจน์เชิงประจักษ์ว่า recall ถูก bound ด้วย signal coverage ไม่ใช่ด้วย
> วิธีรวมสัญญาณ (url/geo coverage บน positive = 0.1%/0%)"

---

## ลำดับความสำคัญ & timeline (แนะนำ)
| ลำดับ | Phase | effort | ได้อะไร | must-do? |
|---|---|---|---|---|
| 1 | Phase 0 harness | 0.5 วัน | เครื่องวัด + ตารางบท 4 | ✅ บังคับ |
| 2 | Phase 1 adaptive threshold | 1 วัน | recall จริง +1.5–2% | ✅ บังคับ |
| 3 | Phase 2 noisy-OR writeup | 1–2 วัน | contribution + ceiling | ✅ บังคับ |
| 4 | Phase 4 integration | 2 วัน | เขียนลงเล่ม | ✅ บังคับ |
| 5 | Phase 3 enrichment | 3–5 วัน | ทะลุเพดาน | ⭕ ถ้ามีเวลา |

**Critical path = Phase 0 → 1 → 2 → 4** (~4.5 วัน) ได้ทั้ง recall เพิ่มจริง + contribution เชิงสมการ
+ การพิสูจน์เพดานที่ reviewer รับได้ Phase 3 เป็นของแถมที่ทำให้ future work มีน้ำหนัก

## สิ่งที่ตัดทิ้ง (พิสูจน์แล้วว่าไม่คุ้ม)
- Contrastive / two-tower encoder สำหรับ zero-overlap → hard slice 0.22%, ไม่คุ้มหนึ่งเทอม
- Redesign string similarity (char n-gram ฯลฯ) → char_cos ของ FN median 0.018, ชื่อไม่ overlap จริง
- Calibration เพิ่มเพื่อ "เพิ่ม probability" → monotonic ไม่เปลี่ยน ranking, จับ FN ใหม่ไม่ได้
