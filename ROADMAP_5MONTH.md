# Roadmap 5 เดือน — ปิดปัญหา Identity Resolution + สร้าง Lead Scoring ที่เป็น ML จริง

เป้าหมายปลายทาง: ระบบ end-to-end ที่ (1) IR แข็งแกร่ง+วัดผลครบ, (2) รวมเป็น Single Customer
View, (3) Lead Scoring เป็น **supervised ML ที่ validate ได้** ไม่ใช่ heuristic, (4) ปิด limitation
ทั้ง 5 ข้อในบท 5.3 ของเล่ม

## ความจริงที่กำหนดแผน (อย่าข้าม)
- **IR (Project 1) เกือบสมบูรณ์แล้ว**: GB image_context P=0.95 R=0.92, decision tiers, calibration พร้อม
  เพดาน recall พิสูจน์แล้วว่าถูก bound ด้วย signal coverage (url/geo ≈ 0%)
- **Lead Scoring ยังไม่ใช่ ML**: สูตร `0.4·completeness+0.3·platform+0.3·engagement`, `engagement`
  เป็นตัวเลขสังเคราะห์, **ไม่มี conversion label, ไม่มี behavioral data จริงใน LinkSocial**
- LinkSocial ไม่มี timestamp/session/transaction → Lead Scoring แบบ behavioral **ทำบน LinkSocial
  ตรง ๆ ไม่ได้** ต้องตัดสินใจเรื่องข้อมูลก่อน

---

## GATE ตัดสินแล้ว — เส้น A (ข้อมูลบริษัท) ปิดถาวร → ใช้เส้น B′

บริษัทไม่ปล่อย transactional/conversion data → **ต้องมี label conversion จริงจากที่อื่น** ไม่งั้น
Lead Scoring จะ validate ไม่ได้

**ทางที่ตัดทิ้ง:**
- เส้น A (ข้อมูลบริษัท) — ปิดถาวร
- เส้น B-เดิม (ปั้น conversion label เองบน LinkSocial) — **อันตราย: circularity** ถ้ากำหนดให้
  conversion ขึ้นกับ cross-platform consistency แล้วสรุปว่า IR ช่วย = bake คำตอบเอง reviewer ตก

**เส้นที่เลือก = B′ : Public lead dataset จริง + จำลอง data-silo**
ใช้ dataset lead scoring สาธารณะที่มี **conversion label จริง** (เช่น X-Education "Leads" ~9k leads
มีคอลัมน์ `Converted` 0/1 + behavioral: TotalVisits, Time-on-site, Last Activity, Lead Source; หรือ
UCI Online Shoppers Intention `Revenue`; หรือ Bank Marketing `y`) เป็นแกนของ Lead Scoring

**กุญแจ — วิธีพิสูจน์ "IR สร้างมูลค่า" บน label จริงโดยไม่ circular:**
1. เอา lead dataset จริง (มี label conversion จริง) มา
2. **จำลองปัญหา data silo**: หั่น feature ของแต่ละลูกค้ากระจายไปหลาย "แหล่ง/แพลตฟอร์ม" +
   ทิ้งบางส่วน + ใส่ noise ให้ชื่อเพี้ยน → ได้ fragmented view (เลียนแบบ LinkSocial)
3. **Baseline:** เทรน conversion model บน single-fragment (siloed) → AUC_silo
4. **Treatment:** รวม fragment กลับ (perfect merge + IR-predicted merge) → AUC_unified
5. พิสูจน์ **AUC_unified > AUC_silo อย่างมีนัยสำคัญ** = IR สร้างมูลค่าที่วัดได้ **บน label จริง**
   ไม่ต้องปั้น conversion เลย และไม่ circular เพราะ label มีมาก่อน ไม่ได้ผูกกับ merge

ผลพลอยได้: ได้ทดสอบ IR pipeline บน dataset ที่ 2 (external validation ปิด limitation บท 5.3 ข้อ 1)
**ทั้งแผนเดือน 2–3 ข้างล่างเขียนบนเส้น B′ นี้**

---

## เดือนที่ 1 — ปิดงาน IR + ตั้งฐาน Lead Scoring data

**สัปดาห์ 1–2: ปิด IR ให้สมบูรณ์** (ต่อยอดจาก IMPLEMENTATION_PLAN.md)
- Phase 0 eval harness (slice + per-platform metrics)
- Phase 1 adaptive threshold → recall +1.5–2% (deliverable: ตารางบท 4)
- Phase 2 noisy-OR + ceiling analysis → contribution เชิงสมการ + พิสูจน์เพดาน
- **Gate:** recall ต้องขึ้น ≥1.5% โดย precision ลด ≤1% ไม่งั้นคง baseline

**สัปดาห์ 1 (ขนาน): ตัดสิน GATE ข้อมูล Lead Scoring** — คุยบริษัท/อาจารย์ เลือกเส้น A/B/C

**สัปดาห์ 3–4: ออกแบบ Single Customer View (SCV) feature store**
- รวม unified_profiles (จาก IR MATCH tier) → 1 entity ต่อคน
- feature ต่อ entity: n_platforms, completeness, platform_presence, bio richness,
  external-domain diversity, location resolvability, image availability
- output: `scv_features.parquet` (entity-level, ไม่ใช่ pair-level)
- **Deliverable:** ERD + feature dictionary ของ SCV (เข้าบท 3)

---

## เดือนที่ 2 — เตรียม Public Lead Dataset + ออกแบบ Fragmentation

**สัปดาห์ 1: เลือก + ตรวจ dataset**
- เลือก 1 dataset lead ที่มี conversion label จริง (แนะนำ X-Education Leads: `Converted` 0/1,
  ~9k rows, behavioral columns) — ตรวจ license, class balance, missingness
- EDA + leakage audit (ตาม prompt_library Step 3): หา feature ที่รั่ว label ออกก่อน

**สัปดาห์ 2–3: ออกแบบ Fragmentation Protocol (หัวใจของ track นี้)**
- นิยาม "แหล่ง/แพลตฟอร์ม" สังเคราะห์ N แหล่ง แล้วกระจาย feature ของแต่ละลูกค้าไปตามแหล่ง
- ใส่ realism: drop feature บางส่วนต่อแหล่ง, เพี้ยนชื่อ/identifier (typo, abbreviation) เลียนแบบ
  LinkSocial, เก็บ ground-truth mapping (คนเดียวกันข้ามแหล่ง) ไว้เป็น label ของ IR
- output: `fragmented_leads.parquet` (siloed view) + `entity_map.csv` (ground truth merge)

**สัปดาห์ 4: รัน IR pipeline (จาก Project 1) บน fragmented_leads**
- ใช้ blocking + candidate scoring + decision tiers เดิม → ได้ merge ที่ทำนาย
- วัด IR performance บน dataset ที่ 2 = **external validation** (ปิด limitation บท 5.3 ข้อ 1)
- **Deliverable:** fragmentation spec + IR-on-dataset-2 report

---

## เดือนที่ 3 — Lead Scoring Model + พิสูจน์มูลค่าของ IR (บน label จริง)

- **Baseline:** rule-based เดิม (0.4/0.3/0.3) + logistic regression บน siloed view
- **Main model:** Gradient Boosting / calibrated classifier ทำนาย `Converted` (label จริง)
- **การทดลองหลักของเล่ม — 3 เงื่อนไข เทียบบน conversion label เดียวกัน:**
  1. **Siloed:** เทรนบน single-fragment view → AUC_silo
  2. **IR-merged:** รวม fragment ด้วย IR ที่ทำนาย → AUC_ir
  3. **Oracle-merged:** รวมด้วย ground-truth mapping → AUC_oracle (เพดานบน)
- **พิสูจน์:** AUC_ir > AUC_silo อย่างมีนัยสำคัญ (bootstrap CI / DeLong test) = **IR สร้างมูลค่าที่
  วัดได้บน conversion จริง โดยไม่ circular** (label มีก่อน ไม่ผูกกับ merge) และ AUC_ir ≈ AUC_oracle
  บอกว่า IR ดีพอ
- calibration + threshold → tier HOT/WARM/COLD เชิงความน่าจะเป็น
- **Gate:** (a) main model ชนะ rule-based บน AUC/AP มีนัยสำคัญ, (b) IR-merged ชนะ siloed มีนัยสำคัญ
- **Deliverable:** lead model + ตาราง 3-เงื่อนไข + SHAP + calibration report

---

## เดือนที่ 4 — End-to-End Integration + Robustness (ปิด limitation บท 5.3)

**เชื่อม pipeline เดียว:** raw profiles → IR (match/merge) → SCV → Lead Scoring → CRM tiers
- แต่ละ stage มี contract (schema validation) + audit log

**ปิด limitation ทีละข้อ:**
| Limitation (บท 5.3) | วิธีปิด |
|---|---|
| dataset เดียว | เพิ่ม second UIL dataset เป็น external validation ของ IR |
| image coverage ต่ำ | รายงาน coverage เป็น controlled variable + ablation image on/off |
| ไม่มี temporal split | ถ้าเส้น B: จำลอง profile drift ตามเวลา, ทำ temporal holdout |
| black-box | SHAP ทั้ง IR + Lead; minimal-evidence rule สำหรับ HITL |
| review queue ใหญ่ (86k) | active-learning loop + uncertainty routing ลดคิว |

**Robustness ของ model (ทำให้ "แข็งแกร่ง"):**
- seed stability (5 seeds, รายงาน mean±sd)
- subgroup fairness (platform-pair, completeness tier, non-Latin names)
- calibration บน holdout (ECE)
- **Deliverable:** robustness report + fairness matrix + model card (Mitchell et al.)

---

## เดือนที่ 5 — Evaluation, Ablation, เขียนเล่ม, เตรียม defense

**สัปดาห์ 1–2: การทดลองปิดท้าย**
- full ablation ladder ทั้ง IR (lexical→+image→+fusion→+adaptive-threshold) และ
  Lead (rule→+ML→+IR-features)
- end-to-end metric: precision@k ของ HOT tier, business-value proxy (conversion lift)

**สัปดาห์ 3–4: เขียนเล่ม + slide**
- อัปเดตบท 4 ด้วยตัวเลขใหม่ (slice recall, adaptive threshold, noisy-OR ceiling)
- เขียนบท Lead Scoring ใหม่ทั้งบท (ตอนนี้เล่มยอมรับว่ายังไม่ได้ทำ)
- บท 5: contribution 3 อย่าง — (1) coverage-bound recall finding, (2) availability-gated fusion,
  (3) IR→Lead value proof
- เตรียม defense: demo web prototype (มีอยู่แล้วในขอบเขต), Q&A จาก limitation

---

## เกณฑ์ "model แข็งแกร่งพอ" (definition of done)
1. IR: recall ≥0.93 @ precision ≥0.94, seed-stable, external-dataset validated, ceiling อธิบายได้
2. Lead Scoring: **เป็น supervised ML ที่ชนะ rule-based baseline อย่างมีนัยสำคัญ**, calibrated,
   ablation พิสูจน์ว่า IR features เพิ่มค่า
3. End-to-end: pipeline เดียวรันได้ reproducible, มี audit log + model card
4. ปิด limitation บท 5.3 ครบ 5 ข้อ (หรืออธิบายเชิงประจักษ์ว่าทำไมปิดไม่ได้)

## Risk register
| ความเสี่ยง | ผลกระทบ | แผนสำรอง |
|---|---|---|
| บริษัทไม่ปล่อยข้อมูลจริง (เส้น A ล่ม) | Lead Scoring ไม่มี label จริง | เส้น B synthetic validate — เตรียมตั้งแต่สัปดาห์ 1 |
| synthetic ไม่สมจริง | ผล lead scoring ไม่น่าเชื่อ | โปร่งใส generative spec + validate distribution + วางเป็น proof-of-concept |
| IR–Lead value ไม่มีนัยสำคัญ | สมมติฐานหลักเล่มล้ม | ก็เป็นผลวิจัย (negative result) — รายงานเงื่อนไขที่ IR ช่วย/ไม่ช่วย |
| เวลาไม่พอ | ทำไม่ครบ 5 เดือน | critical path = IR ปิด + Lead supervised + integration; robustness/2nd-dataset เป็น optional |

## Critical path (ห้ามหลุด)
เดือน1 IR ปิด + GATE ข้อมูล → เดือน2 lead dataset → เดือน3 lead model + IR-value proof →
เดือน4 integration → เดือน5 eval+เขียนเล่ม
งาน optional (ยกออกได้ถ้าเวลาไม่พอ): 2nd dataset, data enrichment, temporal drift sim
