# แผนการทดลอง R0–R5: Representation × Decision

> เป้าหมาย: พิสูจน์ด้วยตัวเลขว่า transformation แต่ละแบบ (GA decision, BERT representation, Bloom encoding)
> ทำให้ eval **ดีขึ้นหรือไม่ เท่าไหร่** เทียบกับ baseline production (รูป 4.14: FP 925 / FN 5,554 / REVIEW 86,296)

## 1. กรอบการทดลอง — ตาราง 2 แกน

แยก **representation** (แกนนอน) ออกจาก **decision** (แกนตั้ง) แล้ววัดทุกช่องด้วย harness เดียวกัน:

|                          | decision: threshold ทำมือ (เดิม) | decision: GA-optimized |
|--------------------------|----------------------------------|------------------------|
| repr: 17 feature (เดิม)  | **R0** = baseline (925/5,554/86,296) | **R1** |
| repr: +BERT cosine       | R2                               | R3 |
| repr: +Bloom filter      | R4                               | R5 |

- **R0→R1** ตอบ: "GA ช่วย FP/FN แค่ไหน" — *ไม่ต้องเทรน model ใหม่* (ใช้ score เดิมจาก `match_decisions.parquet`)
- **R0→R2** ตอบ: "transformer representation ช่วยแค่ไหน" — เทรน MLP ใหม่ (17→18 features)
- **R4/R5** ตอบ: "เข้ารหัสแล้ว accuracy ตกเท่าไหร่" — privacy/accuracy tradeoff curve

ลำดับลงมือ: **split → R1 → R2 → R4** (เร็วสุดก่อน, แต่ละตัวใช้ split + harness ชุดเดียวกัน)

## 2. ข้อมูลจริงที่ใช้ (ตรวจสอบแล้ว 2026-07-20)

| ไฟล์ | บทบาท |
|---|---|
| `Project-for-Work/train_data/stage15_crm_entity_pipeline/artifacts/match_decisions.parquet` | 2,086,245 คู่ (exact 12,403 + scored 2,073,842) คอลัมน์ `score, decision, decision_source` — `split_name='full'` ทั้งหมด จึงต้องแบ่ง split เอง |
| `Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv` | profile + `user_folder` (ground truth) — `profile_row_id` คือ key ที่ `profile_id_a/b` ชี้ถึง |
| `analysis_decision_matrix/blocking_missed_pairs.csv` | คู่จริง 3,316 คู่ที่ blocking พลาด (ไม่เคยถูก score) |

ตัวเลขอ้างอิง (GT จาก `user_folder`): คู่จริงทั้งหมด **29,247**
- FP 925 = 862 (AUTO_HIGH ≥0.98) + 63 (AUTO_EXACT ชนกันคนละคน)
- FN 5,554 = 2,242 (scored แล้วปัดตก) + 3,316 (blocking พลาด)
- REVIEW 86,296 = 4,065 จริง + 82,231 ไม่จริง

## 3. กันข้อผิดพลาด: Entity-aware split (ทำก่อนทุกอย่าง)

ถ้า GA จูน threshold บนข้อมูลชุดเดียวกับที่วัดผล = overfit → ตัวเลขเชื่อไม่ได้
แบ่งด้วย `user_folder` (ทั้ง entity อยู่ฝั่งเดียว ห้ามคร่อม) แบบ hash-based ให้ deterministic:

```
key = user_folder (ถ้ามี) หรือ profile_row_id (profile ที่ไม่มี folder — เป็น negative เท่านั้น)
h = md5(key) % 100  →  test ถ้า h < 30 (30%), ไม่งั้น val (70%)
คู่ (a,b): อยู่ split เดียวกันทั้งคู่ → ใช้ split นั้น, คนละ split → ทิ้ง (drop)
```

คุณสมบัติสำคัญ: คู่จริงทุกคู่มี folder เดียวกัน → ไม่โดน drop เลย (drop เฉพาะ negative ข้าม split)
→ **GA จูนบน val เท่านั้น, รายงานผลบน test เท่านั้น**

## 4. Evaluation harness (ใช้ร่วมทุก experiment — apple-to-apple)

ค่าคงที่ต่อ split (GA/model แตะไม่ได้): exact-tier TP/FP, blocking-missed FN, total positives

```
FP  = (decision==MATCH & actual==0) + FP_exact[split]
TP  = (decision==MATCH & actual==1) + TP_exact[split]
FN  = (decision==NO_MATCH & actual==1) + FN_blocking[split]
REV = (decision==REVIEW)
precision = TP/(TP+FP)   recall = TP/TOTAL_POS[split]   F1 = harmonic mean
```

**ข้อจำกัดที่ต้องเขียนในเล่ม:** GA/decision rule แตะได้เฉพาะคู่ที่ถูก score แล้ว
(FP 925, FN-in-candidate 2,242, REVIEW 86,296) — **FN จาก blocking พลาด 3,316 คู่ กู้ไม่ได้**
ต้องใช้ semantic blocking (experiment แยก) ถึงจะกู้ได้ → เพดาน recall ของ R1 คือ (29,247−3,316)/29,247 = 0.887

## 5. R1 — GA re-decision (`exp_r1_ga_redecision.py`)

Genome = generalization ของ rule มือใน stage18 (0.98/0.95/0.9/0.7):

```
g = [t_m, t_r, c_promote, c_demote]        constraint: t_r ≤ t_m, c_demote ≤ c_promote
score ≥ t_m                    → MATCH
t_r ≤ score < t_m (band):
    name_sim ≥ c_promote       → MATCH     (promote)
    name_sim <  c_demote       → NO_MATCH  (demote)
    else                       → REVIEW
score < t_r                    → NO_MATCH
```

Fitness (minimize บน **val** เท่านั้น):

```
cost(g) = W_FP·FP + W_FN·FN + W_REV·REVIEW      โดย W_FP=5.0, W_FN=1.0, W_REV=0.02
```

(W_FP > W_FN เพราะ merge ผิดคนใน CRM แก้ยากกว่าพลาด match; W_REV = ต้นทุนแรงงานคนตรวจต่อคู่
— จุดนี้อาจารย์ปรับได้ แล้วรันซ้ำ ได้ rule ใหม่ทันทีโดยไม่ต้องเทรนอะไร)

GA: population 60, generations 40, elite 12, uniform crossover, Gaussian mutation (σ=0.02, p=0.3), seed 42

รายงานบน **test**: production rule (R0) vs manual stage18 vs GA best → ตาราง before/after
+ projection เต็มระบบ (apply genome กับ scored ทั้งหมด เทียบ 925/5,554/86,296 ตรง ๆ)

`name_sim` = max Jaro-Winkler ของ 4 คู่ (userName/fullName ไขว้กัน) เหมือน stage18 —
คำนวณเฉพาะคู่ score ≥ 0.5 (785,208 คู่) เพราะ t_r ต่ำสุดในการค้นหาคือ 0.5 → ต่ำกว่านั้นเป็น NO_MATCH เสมอ

## 6. R2 — BERT representation (`exp_r2_bert_feature.py`)

- encode `fullName + bio + location` ด้วย `all-MiniLM-L6-v2` (sentence-transformers ติดตั้งแล้ว, CPU ได้)
- feature ใหม่ = cosine ของคู่ → เพิ่มเป็น feature ที่ 18 → retrain IdentityMLP โครงเดิม (input 17→18) บน split เดิม
- ได้ probability ชุดใหม่ → รัน harness แบบ R0 (threshold เดิม) = **R2**, รัน GA ทับ = **R3**
- ทางหนัก (ถ้ามีเวลา): fine-tune cross-encoder `[CLS] rec_a [SEP] rec_b` แทน MLP

## 7. R4/R5 — Bloom filter encoding (privacy/accuracy tradeoff)

- แทน jaro feature ด้วย Dice similarity ของ bigram Bloom filter (L bits, k=10 hash)
- sweep L ∈ {2000, 1000, 500, 250}: L เล็ก = ชนกันเยอะ = privacy สูง แต่ accuracy ตก
- plot F1 vs L = tradeoff curve

## 8. เกณฑ์ตัดสิน "ดีขึ้นหรือไม่"

ทุก experiment รายงาน (FP, FN, REVIEW, precision, recall, F1) บน **test** ผ่าน harness เดียวกัน:

- **R1 ดีขึ้น** ถ้า: cost(test) ลดลง และ REVIEW ลดมาก (>50%) โดย precision ไม่ตกต่ำกว่า ~0.94
- **R2 ดีขึ้น** ถ้า: กู้ FN-in-candidate ได้ (recall ↑) โดย FP ไม่บวมเกิน
- **R4 ยอมรับได้** ถ้า: F1 ตกน้อยกว่า ~2–3 จุดที่ระดับ privacy ที่ต้องการ

ไฟล์ output ทั้งหมด → `experiments/` (r1_results.json, r1_best_genome.json, …)
