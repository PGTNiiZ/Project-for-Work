# รายงานผลการทดลอง R0–R6: Representation × Decision (Identity Resolution)

> ทุกตัวเลขในรายงานนี้ดึงตรงจากไฟล์ JSON ที่สร้างโดย `exp_lib.evaluate()`:
> `experiments/r1_results.json`, `r2_results.json`, `r3_results.json`,
> `r4_privacy_tradeoff.json`, `r2_r3_fn_recovery.json`, `r6_results.json`, `r6_blocking_recovery.json`
> — ไม่มีตัวเลขพิมพ์มือ harness/entity-aware split เดียวกันทุก experiment
> (ดู `EXPERIMENT_PLAN.md`, `exp_lib.py`)

## 1. ตาราง R0–R5 บน TEST (unbiased, 191,693 scored pairs)

| Exp | Representation | Decision | FP | FN | REVIEW | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **R0** | 17-feature เดิม (production score) | threshold มือ (0.98/0.95) | 88 | 1,689 | 8,783 | 0.9852 | 0.6683 | 0.7964 |
| **R1** | 17-feature เดิม | GA re-decision | 23 | 1,885 | 3,116 | 0.9963 | 0.7101 | 0.8292 |
| **R2** | +BERT cosine (feature 18) | threshold มือเดิม (0.98/0.95) | 19 | 2,523 | 1,115 | 0.9963 | 0.5877 | 0.7393 |
| **R3** | +BERT cosine | GA re-decision | 38 | 1,873 | 640 | 0.9940 | 0.7233 | **0.8373** |
| R4 (L=2000) | +Bloom-Dice (privacy สูงสุดในชุดที่ทดสอบ) | threshold มือเดิม | 14 | 2,858 | 766 | 0.9973 | 0.5878 | 0.7397 |
| R4 (L=1000) | +Bloom-Dice | threshold มือเดิม | 14 | 2,931 | 766 | 0.9973 | 0.5795 | 0.7330 |
| R4 (L=500) | +Bloom-Dice | threshold มือเดิม | 14 | 2,453 | 1,354 | 0.9972 | 0.5688 | 0.7244 |
| R4 (L=250) | +Bloom-Dice (privacy ต่ำสุด/ชนบ่อยสุด) | threshold มือเดิม | 14 | 2,760 | 1,026 | 0.9972 | 0.5699 | 0.7253 |
| **R5 (L=2000)** | +Bloom-Dice | GA re-decision | 57 | 1,823 | 700 | 0.9910 | 0.7183 | 0.8329 |
| R5 (L=1000) | +Bloom-Dice | GA re-decision | 57 | 1,820 | 721 | 0.9910 | 0.7166 | 0.8317 |
| R5 (L=500) | +Bloom-Dice | GA re-decision | 57 | 1,818 | 738 | 0.9910 | 0.7153 | 0.8309 |
| R5 (L=250) | +Bloom-Dice | GA re-decision | 57 | 1,881 | 674 | 0.9910 | 0.7152 | 0.8308 |

**Best config = R3** (BERT cosine feature 18 + GA re-decision) F1=0.8373 — ดีกว่า R1 (GA บน representation เดิม)
ทุกตัวชี้วัด: REVIEW ลดจาก 3,116→640 (−79%), recall เพิ่ม 0.7101→0.7233
รองลงมาคือ **R5 (L=2000)** F1=0.8329 ซึ่งใกล้เคียง R3 มาก (ต่างกันแค่ 0.0044) แต่ **ไม่ต้องเก็บ userName/fullName
เป็น plaintext เลย** (เข้ารหัสด้วย Bloom filter ก่อนคำนวณ similarity) — เป็นตัวเลือกที่คุ้มที่สุดถ้าต้องการ privacy

### แถว projection เต็มระบบ (full = scored ทั้งหมด 2,073,842 คู่ + exact-tier) สำหรับ best config (R3)

| Exp | FP | FN | REVIEW | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 (baseline, รูป 4.14) | 925 | 5,558 | 86,296 | 19,624 | 0.9550 | 0.6710 | 0.7882 |
| **R3 (BERT + GA) — projection** | 465 | 6,179 | 2,751 | 21,274 | 0.9786 | 0.7274 | 0.8345 |

R3 projection: FP 925→465 (−50%), REVIEW 86,296→2,751 (**−96.8%**), F1 0.7882→0.8345
(genome จูนบน val แล้ว apply กับ full — ตัวเลข unbiased คือฝั่ง test ด้านบน)

## 2. FN-recovery diagnostic (คู่จริงที่เคย "ปัดตก" 2,242 คู่ — ตัวเลขที่อาจารย์สนใจ)

จาก `experiments/r2_r3_fn_recovery.json` — นับเฉพาะ 2,242 คู่ที่ actual=1 แต่ production decision เดิม = NO_MATCH:

| Rule | recover เป็น MATCH หรือ REVIEW | recover เป็น MATCH ล้วน |
|---|---:|---:|
| R2 (BERT + threshold มือ) | 282 / 2,242 (12.6%) | 37 |
| R3 (BERT + GA) | **728 / 2,242 (32.5%)** | 214 |

R3 กู้คืนได้เกือบ 3 เท่าของ R2 เพราะ threshold มือเดิม (0.98/0.95) ไม่ได้ re-calibrate ให้เข้ากับ scale
ของ probability ใหม่จาก MLP+BERT — ต้องใช้ GA จูน threshold ใหม่ (R3) ถึงจะดึงประโยชน์จาก representation
ใหม่ออกมาได้เต็มที่ (นี่คือสาเหตุที่ R2 เพียงอย่างเดียว F1 ต่ำกว่า R0 ด้วยซ้ำ)

## 3. ข้อจำกัดของทุก experiment ใน R0–R5

**FN จาก blocking พลาด 3,316 คู่ กู้ไม่ได้ด้วย decision rule หรือ representation ใดๆ ใน R0–R5**
เพราะคู่เหล่านี้ไม่เคยถูกสร้างเป็น candidate pair ตั้งแต่ขั้น blocking เลย (ไม่มี score ให้ปรับ) —
ต้องใช้ semantic/embedding blocking ถึงจะมีโอกาสกู้ได้ (ทดลองจริงใน R6 ด้านล่าง)
เพดาน recall ของทุก experiment ใน R0–R5 คือ **(29,247−3,316)/29,247 = 0.8866**

## 4. R6 — BERT-ER semantic blocking (ความพยายามกู้ 3,316 คู่ที่ blocking เดิมมองข้าม)

แรงบันดาลใจจาก Arford et al. (2025 IEEE ISI) "Assessing the De-anonymization Risk of Social
Media Users: A BERT-Based Entity Resolution Approach" ซึ่งใช้ BERT-ER (Li et al. 2021 AAAI)
เข้ารหัสโปรไฟล์ทีละตัวด้วย BERT แล้วใช้ **learnable hash blocking decoder** (แปลง embedding เป็น
k-bit code แล้ว bucket ตาม Hamming distance) แทน string-based blocking — เป้าหมายคือให้จับคู่ที่
username/fullname ต่างกันโดยสิ้นเชิงได้ ถ้าความหมาย/context อื่นยังคล้ายกัน

**สถาปัตยกรรมที่ implement จริง** (`exp_r6_bert_er.py`): DistilBERT (bi-encoder, encode ทีละ
โปรไฟล์) → hash head (Linear→LayerNorm→tanh, 64 บิต, เทรนด้วย cosine contrastive loss) สำหรับ
blocking + match head (interaction features `[h_a,h_b,h_a·h_b,|h_a-h_b|]` → MLP) สำหรับตัดสิน
MATCH — ตาม Fig.3 ของ paper ต้นฉบับ (Encoding → Blocking decoder → Matching decoder)

**ข้อจำกัดที่ปรับจากต้นฉบับ (บันทึกไว้ตรงไปตรงมา)**: วัดจริงบนเครื่องนี้พบว่า fine-tune DistilBERT
ทั้งตัวบน CPU ใช้เวลา ~6.3 วินาที/step (1 epoch ≈ 3.6 ชั่วโมง) ไม่ practical จึงเปลี่ยนเป็น
**frozen backbone**: encode โปรไฟล์ทั้ง 36,807 ตัวครั้งเดียว (ไม่ fine-tune) แล้วเทรนเฉพาะ
hash/match head บน embedding ที่ตายตัว — mean pooled output ของ DistilBERT pretrained ตรงๆ
ไม่ได้ปรับให้เข้ากับ domain โปรไฟล์โซเชียลมีเดีย

**กฎที่รักษาเข้มงวด**: 3,316 คู่ blocking-missed **ไม่ถูกใช้ train เลย** (held-out ล้วน) — ผลด้านล่าง
คือการทดสอบว่า hash ที่เทรนจากคู่อื่นทั้งหมดจะ generalize มาจับคู่เหล่านี้ได้หรือไม่ (ไม่ใช่ circular)

### ผล blocking recovery (จาก `r6_blocking_recovery.json`)

| Radius (Hamming) | พบเป็น candidate ใหม่ / 3,316 | จัดเป็น MATCH ถูกด้วย (end-to-end) |
|---|---:|---:|
| 0 (bucket เดียวกันเป๊ะ) | 0 (0.00%) | 0 |
| 1 (ต่างกัน ≤1 บิต) | 4 (0.12%) | 3 |

**ผลลัพธ์คือลบ (negative)** — hash ที่เทรนได้ไม่สามารถกู้ 3,316 คู่นี้ได้จริง วินิจฉัยเพิ่มเติมโดยเช็คว่า
hash จับคู่จริงกลุ่มอื่น (ที่ไม่ใช่ held-out แต่เป็นคู่จริงที่ผ่าน blocking แล้ว actual=1 ทั้งหมด 13,591 คู่)
ไว้ bucket เดียวกันได้ดีแค่ไหน — พบว่า hash **เรียนรู้โครงสร้างความคล้ายจริงบ้าง** (คู่จริงอยู่ bucket
ใกล้กัน radius≤3 ที่ 1.53% เทียบกับคู่สุ่มที่ 0.03% — ต่างกัน ~50 เท่า) แต่ **ความถี่สัมบูรณ์ต่ำเกินจะใช้
ทำ blocking จริงได้** (98.47% ของคู่จริงยังไม่ถูกจับ แม้ที่ radius 3) สาเหตุน่าจะมาจาก embedding ที่ไม่ได้
fine-tune + เทรน hash head แค่ 27 epoch บน positive pairs เพียง ~9,558 คู่ + hash space 64 มิติกว้าง
เกินไปเทียบกับข้อมูล — ไม่ใช่ว่าแนวทางนี้ผิดหลักการ แต่ระดับ investment (compute/epoch/fine-tuning)
ที่ทำได้จริงบน CPU ในเซสชันนี้ยังไม่พอ

### ผล R6/R6-GA บน apples-to-apples harness เดิม (จาก `r6_results.json`)

| Exp | Decision | FP (test) | REVIEW (test) | Precision | Recall | F1 (test) |
|---|---|---:|---:|---:|---:|---:|
| R6 | threshold มือ (0.98/0.95) | 19 | 224 | 0.9950 | 0.4350 | 0.6053 |
| R6-GA | GA re-decision | 13 | 3,712 | 0.9977 | 0.6405 | **0.7801** |

R6-GA (F1=0.7801) **แย่กว่า R3 (F1=0.8373)** และแย่กว่า R1 (F1=0.8292) ด้วยซ้ำ — สมเหตุสมผล เพราะ
R6 ใช้ embedding จาก DistilBERT ที่ไม่ได้ fine-tune (frozen) ล้วนๆ ไม่มี hand-crafted feature 17 ตัว
ของ R2/R3 ช่วยเสริมเลย ขณะที่ R2/R3 ผสมทั้งสองอย่าง (string-similarity features ที่แม่นมาก + BERT
cosine เสริม) — สรุปคือ **representation ที่ fine-tune หรือผสมกับ feature เดิมสำคัญกว่า pretrained
embedding ล้วนๆ** สำหรับงานนี้

**สรุป R6**: ตอบคำถามตั้งต้น (จะกู้ 3,316 คู่ blocking-missed ที่สัญญาณชื่ออ่อนได้ไหมด้วย transformer
blocking) ได้ว่า **ยังกู้ไม่ได้จริงในทางปฏิบัติด้วย setup นี้** (3/3,316 end-to-end) แนวทางถูกทิศทาง
(hash เรียนรู้ความคล้ายได้จริงในทางสถิติ) แต่ต้องการ fine-tune backbone จริง (ต้องมี GPU) + เทรนนาน
ขึ้นมาก + อาจต้องลดมิติ hash หรือทำ hard-negative mining เพิ่ม ถึงจะใช้งานได้จริง — เหมาะเป็นหัวข้อ
วิจัยต่อยอด ไม่ใช่ deliverable ที่ deploy ได้ในรอบนี้

## 5. สรุปและข้อเสนอแนะ

การเพิ่ม BERT cosine เป็น feature ที่ 18 (R2/R3) ช่วยให้ representation แยกคู่จริง/ปลอมได้ดีขึ้นจริง
แต่ประโยชน์จะออกมาก็ต่อเมื่อ **re-tune decision rule ด้วย GA เท่านั้น** — ใช้ threshold มือเดิมตรงๆ (R2)
กลับแย่กว่า baseline เพราะ scale ของ probability เปลี่ยน R3 (BERT+GA) ให้ F1 ดีที่สุดในทุก experiment
(0.8373 บน test, REVIEW ลด 79% จาก R1) และกู้ FN ที่เคยปัดตกได้ 728/2,242 คู่ (32.5%)
ฝั่ง privacy, Bloom filter ที่ L=2000 (R5) แทบไม่เสีย F1 เลยเทียบกับ R3 (0.8329 vs 0.8373 ต่างกัน 0.44 จุด)
ในขณะที่ L เล็กลง (1000/500/250) ทำให้ recall ตกเพิ่มขึ้นเรื่อย ๆ ตามที่คาด (bit collision มากขึ้น)
**ข้อเสนอแนะให้ deploy**: ถ้าไม่มีข้อกำหนดเรื่อง privacy ของ plaintext name ให้ใช้ **R3** (BERT+GA);
ถ้าต้องปฏิบัติตามข้อกำหนด privacy (เช่นไม่เก็บชื่อ plaintext ข้ามระบบ) ให้ใช้ **R5 ที่ L=2000** ซึ่งแทบไม่เสีย
ความแม่นยำเลยแต่ปลอดภัยกว่ามาก ทั้งสองกรณีต้องรัน GA ใหม่ (ไม่ใช้ threshold มือ 0.98/0.95 เดิม) และ
ต้องแก้ปัญหา 3,316 คู่ blocking พลาดด้วย semantic blocking แยกต่างหากก่อนถึงจะดัน recall เกิน 0.887 ได้
**ส่วน R6 (transformer semantic blocking) ยังไม่พร้อม deploy** — ทิศทางถูกต้อง (paper อ้างอิงพิสูจน์
แล้วว่า BERT-ER ทำได้จริงบน GPU) แต่บน CPU-only + frozen backbone ของรอบทดลองนี้ยังกู้ 3,316 คู่
blocking-missed ได้แค่ 3 คู่ (0.09%) — ต้องมี GPU สำหรับ fine-tune backbone จริงก่อนถึงจะเป็น
ทางเลือกที่ใช้งานได้จริงสำหรับปัญหานี้ (แผนละเอียดในหัวข้อ 6)

## 6. ข้อเสนอแนะงานวิจัยต่อไป: ทำ R6 ให้ใช้งานได้จริงด้วย GPU fine-tuning

> **แผน implement ฉบับละเอียดระดับโค้ด** (code diff ทีละจุด, hyperparameter sweep plan,
> evaluation protocol, timeline 3–4 วัน, ความเสี่ยง+ทางแก้) อยู่ที่ `experiments/R6_GPU_IMPLEMENT_PLAN.md`

โครงสร้างโค้ดของ R6 (`exp_r6_bert_er.py`) ออกแบบไว้ให้สลับจาก frozen-backbone กลับไปเป็น
full fine-tuning ได้โดยไม่ต้องเขียนใหม่ทั้งหมด — งานที่เหลือคือปรับ config/hyperparameter แล้วรัน
บนเครื่องที่มี GPU ขั้นตอนที่แนะนำมีดังนี้ (ไม่ใช่สิ่งที่ทำในรอบทดลองนี้ เก็บไว้เป็น future work)

**1) สภาพแวดล้อม**: GPU ตัวเดียวระดับ consumer ก็เพียงพอ (เช่น RTX 3060/4060 8–12GB VRAM) สำหรับ
fine-tune DistilBERT/BERT-base ที่ batch ขนาดพอเหมาะ (64–128 คู่) — เทียบกับ CPU ที่วัดได้จริง
6.3 วินาที/step (batch=16) คาดว่า GPU จะเร็วขึ้น ~30–60 เท่า (~0.1–0.2 วินาที/step) ทำให้ 1 epoch
(~2,000 step) เหลือ ~5–10 นาที และเทรนหลาย epoch เสร็จได้ภายใน 1–2 ชั่วโมง

**2) การแก้โค้ด** (ใน `exp_r6_bert_er.py`):
   - `BertERModel.__init__`: เอา `self.backbone.eval()` และ `p.requires_grad_(False)` ออก
   - `train_heads()` → เปลี่ยนกลับเป็นแบบ fine-tune ทั้งตัว: ใส่ `model.backbone.parameters()`
     กลับเข้า optimizer พร้อม differential LR (backbone `lr=2e-5`, head `lr=1e-3` — ค่าที่เตรียม
     ไว้แล้วในโค้ดฉบับแรกก่อนเปลี่ยนมาเป็น frozen mode)
   - เปลี่ยน batch construction จาก tensor-indexing บน embedding cache กลับไปเป็น tokenized-batch
     forward ผ่าน backbone ทุก step (แบบที่เคย implement ไว้ตอนแรกใน git history ของไฟล์นี้)
   - เพิ่ม mixed-precision training (`torch.cuda.amp`) เพื่อความเร็วเพิ่มอีก ~2 เท่าบน GPU
   - implement straight-through estimator (STE) ให้ hash head จริงจัง (ตอนนี้ใช้ continuous
     relaxation tanh ล้วน ๆ เป็นการลดทอนจาก paper ต้นฉบับ)

**3) ปรับ hyperparameter ที่ควร sweep**:
   - `HASH_BITS`: ลองทั้ง 32/48/64/128 บิต (bucket ที่แน่นเกินไปหรือหลวมเกินไปกระทบ recall ทั้งคู่)
   - `HASH_LOSS_WEIGHT`: เพิ่มน้ำหนักให้ hash loss มากกว่า match loss (ตอนนี้ 0.5:1) เพราะ
     blocking recall คือจุดที่ล้มเหลวหนักสุด ไม่ใช่ match head (holdout AP ของ match head
     ทำได้ถึง 0.923 อยู่แล้วแม้ backbone จะ frozen)
   - เพิ่ม hard-negative mining: สุ่ม negative ที่ cosine similarity สูงแต่ actual=0 (คนละ
     user_folder) แทน/เสริมจาก random negative ปัจจุบัน เพื่อบีบ decision boundary ให้คมขึ้น
   - ทำ Cartesian negative sampling แบบเต็มรูปแบบตาม paper (R×I ทุกคู่) แทนการสุ่มตัวอย่างย่อ
     ถ้า GPU มีกำลังพอ

**4) Evaluation protocol (ต้องคงเดิมเป๊ะเพื่อเทียบผลกับรอบนี้ได้)**:
   - ห้ามใช้ 3,316 คู่ blocking-missed train เด็ดขาด (held-out ตลอด กัน circular/data leakage)
   - รายงานผ่าน `exp_lib.evaluate()` + `split_constants()` เดิม สำหรับ R6/R6-GA
   - รายงาน blocking recovery ที่ radius 0/1/2/3 เหมือนรอบนี้ เพื่อเทียบ trend ได้ตรง ๆ

**5) เกณฑ์ตัดสินว่า "ใช้งานได้จริง"**: กู้คืน 3,316 คู่ได้อย่างน้อย ~10–20% (300–650 คู่) ถือว่าคุ้ม
ค่าเทียบกับความเสี่ยง (false candidate เพิ่มที่ต้องกรองอีกชั้น) และ R6-GA test F1 ต้องขึ้นมาทัดเทียม
หรือดีกว่า R3 (F1 0.8373) ไม่งั้นใช้ R3 ต่อไปคุ้มกว่า — ถ้าเข้าเกณฑ์นี้ แนะนำให้ใช้ hash-based
blocking เป็น **ส่วนเสริม** ของ blocking เดิม (union กัน) ไม่ใช่แทนที่ทั้งหมด เพราะ blocking เดิม
ยังคุม precision/recall ของคู่ที่รู้จักอยู่แล้วได้ดี การเสริมจึงมีแต่ได้ไม่มีเสีย

---
*ไฟล์ประกอบ: `r1_results.json`, `r2_results.json`, `r3_results.json`, `r4_privacy_tradeoff.json`
(+ `.png` กราฟ F1 vs L), `r2_r3_fn_recovery.json`, `r3_ga_history.json`, `r6_results.json`,
`r6_blocking_recovery.json` — โค้ด: `exp_r1_ga_redecision.py` (ของเดิม ไม่แก้), `exp_r2_bert_feature.py`,
`exp_r4_bloom_privacy.py`, `exp_r6_bert_er.py`*
