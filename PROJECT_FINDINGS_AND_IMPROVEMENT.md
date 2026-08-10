# เอกสารสรุปปัญหา ประเด็นที่พบ และแนวทางพัฒนา — Identity Resolution Project

เอกสารนี้รวบรวมทุกสิ่งที่วิเคราะห์และ **verify กับไฟล์จริง** ตลอดการตรวจสอบ ทุกตัวเลขมีที่มา
ระบุแหล่ง ไม่ใช่ค่าคาดเดา ใช้เป็นฐานตัดสินใจพัฒนาต่อ + เนื้อหาสำหรับบท 4–5 ของปริญญานิพนธ์

**อัปเดต:** 2026-07 · **โมเดลอ้างอิง:** GB image_context (41 features), run `image_context_r075_h20_s42`

---

## 0. สรุปผู้บริหาร (Executive Summary)

โมเดล Identity Resolution ปัจจุบัน **แข็งแรงบนเคสง่ายแล้ว** (คู่ที่ชื่อคล้าย ~96.5% ได้ recall 0.94)
ปัญหาที่เหลือ **ไม่ได้แก้ด้วยการเปลี่ยนโมเดลหรือสมการ** แต่เป็น 3 เรื่องคนละชนิด:

1. **เพดานข้อมูล (data-limited):** 40.6% ของคู่จริงที่ชื่อต่างกันไม่มีสัญญาณ non-lexical ใด ๆ ให้ลิงก์
   — พิสูจน์แล้วว่า noisy-OR fusion และ similarity ใหม่ก็แก้ไม่ได้
2. **Threshold ตั้งสูงเกิน (policy):** FN 2,242 คู่ถูก score แล้วแต่ปัดตก ทั้งที่ median score 0.83 —
   แก้ได้ด้วย re-decision / ลด threshold
3. **Blocking พลาด (retrieval):** FN 3,316 คู่ไม่เคยเข้าโมเดลเลย — ต้องแก้ที่ blocking key

**Improvement ที่หลักฐานรองรับและ reproduce ได้จริง:** REVIEW re-decision ด้วย strict name-match
ยก **recall 0.671 → 0.760 โดย precision ลดแค่ 0.955 → 0.948** (แต่มี landmine เรื่อง metric — ดู §4)

---

## 1. สถานะปัจจุบัน (Baseline ที่ verify แล้ว)

**บน supervised test split (8,313 คู่):**
| metric | ค่า | แหล่ง |
|---|---|---|
| Precision / Recall / F1 | 0.9507 / 0.9179 / 0.9340 | main_report.json |
| Average Precision / ROC-AUC | 0.9789 / 0.9734 | main_report.json |
| FN / FP | 360 / 209 | reproduce จาก train_all.parquet |

**บน production (full candidate set 2,086,245 คู่):**
| tier | จำนวน | แหล่ง |
|---|---|---|
| MATCH | 20,549 (exact 12,403 + high-score 8,146) | match_decisions.parquet |
| REVIEW | 86,296 | match_decisions.parquet |
| NO_MATCH | 1,979,400 | match_decisions.parquet |
| Final match-only precision / recall | 0.9550 / 0.6710 | reproduce |
| Ground-truth positives ทั้งหมด | 29,243–29,247 | leak_safe_report / stage15 |

---

## 2. ปัญหาที่พบ (แต่ละข้อมีหลักฐาน + ระบุว่าแก้ได้ที่ไหน)

### ปัญหา 2.1 — FN บน test กระจุกที่ slice ที่ชื่ออ่อน (แก้ที่โมเดลได้จำกัด)
แยก 360 FN ตาม name similarity:
| slice | นิยาม | n_pos | recall | FN |
|---|---|---|---|---|
| EASY | name_max ≥ 0.5 | 4,234 | 0.943 | 240 |
| MID | name_max < 0.5 | 148 | 0.223 | 115 |
| HARD | name<0.3 & bio<0.3 (zero-lexical) | 5 | 0.000 | 5 |

- FN กระจุกที่ platform-pair `googleplus × twitter`: recall 0.894, FN 191/360
- **จุดพลิกความเข้าใจ:** zero-lexical แท้ ๆ มีแค่ 5 คู่ (0.1%) — การทุ่มทำ contrastive encoder เพื่อ
  เคสนี้ไม่คุ้ม (แหล่ง: stage10_eval_harness.py, train_all.parquet)

### ปัญหา 2.2 — เพดาน signal coverage (แก้ที่โมเดล/สมการไม่ได้ ต้องแก้ที่ข้อมูล)
ในคู่จริงที่ชื่ออ่อน (name<0.5) 850 คู่ทั้ง dataset:
| สัญญาณ non-lexical | coverage |
|---|---|
| bio semantic ≥0.3 | 36.6% |
| image (มีรูป) | 36.1% |
| url / domain ตรง | **0.0%** |
| location ตรง | **0.0%** |
| **มีสัญญาณอย่างน้อย 1** | 59.4% |
| **ไม่มีสัญญาณเลย** | **40.6%** |

- บน test positives 4,387 คู่: มี url signal 5 คู่, geo signal 0 คู่
- bio_sbert **แยก FN จาก negative ไม่ได้** (FN median 0.190 vs negative 0.179)
- **สรุป:** 40.6% ของเคสยากลิงก์ไม่ได้ด้วยข้อมูลที่มี = เพดานเชิงข้อมูล ไม่ใช่ข้อบกพร่องอัลกอริทึม

### ปัญหา 2.3 — FN production 5,554 คู่ เป็น "สองปัญหาคนละเรื่อง"
| กลุ่ม | จำนวน | ลักษณะ | แก้ที่ไหน |
|---|---|---|---|
| ถูก score แล้วปัดตก | 2,242 | median score 0.83, มี 765 คู่ score≥0.90 | **threshold / re-decision** |
| ไม่เคยถูก score (blocking miss) | 3,316 | 64% name<0.5 (ชื่อคนละเรื่อง) | **blocking key** |

- แหล่ง: analysis_decision_matrix/fn_nomatch_but_actually_yes.csv (2,242) +
  blocking_missed_pairs.csv (3,316); 2,242+3,316 = 5,558 ≈ 5,554

### ปัญหา 2.4 — False Positive 925 คู่ (auto-merge ผิด — อันตรายสุดใน CRM)
| ที่มา | จำนวน | สาเหตุ |
|---|---|---|
| จากโมเดล (score≥0.98) | 862 | name_sim แค่ 0.5–0.7 แต่โมเดลมั่นใจเกิน = **calibration ที่ปลาย distribution** |
| จาก exact-match rule | 63 | username ชนกันแต่คนละคน (เช่น `pullbackes` IG vs TW) = exact-first precision 0.9949 ≠ 1.0 |

- กระจุกที่ `googleplus × twitter` 431 คู่ (47%) · เช็กแล้วไม่ใช่ label noise (มีแค่ 1 คู่ต่างแค่พิมพ์เล็ก-ใหญ่)
- แหล่ง: analysis_decision_matrix/fp_match_but_actually_no.csv

### ปัญหา 2.5 — REVIEW band 86,296 คู่: score แยกไม่ได้ แต่ strict name แยกได้ขาด
- calibrated score ใน band [0.95, 0.98) มี precision แค่ 0.005–0.14 ทุก sub-band = **score หมดพลังแยก**
- แต่ **strict name-match แยกได้ชัด** → เป็นโอกาส re-decision (ดู §3 Exp D)

---

## 3. การทดลองที่รันแล้ว (method + result + verdict)

### Exp A — Noisy-OR Evidence Fusion (สมการรวมหลักฐานแบบ availability-aware)
- สมการ: `P = 1 − ∏_k (1 − s_k·r_k)`; fit reliability r_k บน train
- reliability ที่เรียนได้: r_geo=0.999, r_url=0.93, r_name=0.70, **r_bio=0.07**, r_img=0.001, r_cap=0
- ผล: test AP **0.942 < GB 0.979** · ensemble (GB∨NoisyOR) กู้ 85 TP แต่เพิ่ม **557 FP**
- **Verdict:** ❌ แพ้ GB — พิสูจน์ว่า "สมการไม่ใช่คอขวด, coverage คือกำแพง" (สัญญาณที่เชื่อได้
  url/geo มี coverage 0% พอดีในจุดที่ต้องใช้) · แหล่ง: stage9_noisy_or_fusion.py

### Exp B — Redesign name similarity (char n-gram cosine)
- char-ngram(2-4) cosine ของชื่อคู่ FN: **median 0.018** (TP median 0.825) → ชื่อไม่ overlap จริง
- rescue rule char_cos≥0.6 กู้ได้แค่ 30/360 FN
- **Verdict:** ❌ สมการ string ใหม่สร้าง similarity จากชื่อที่ไม่เหมือนไม่ได้

### Exp C — Adaptive / per-platform threshold
| config | P | R | F1 | slice recall E/M/H |
|---|---|---|---|---|
| baseline @0.35 | 0.9507 | 0.9179 | 0.9340 | 0.94/0.22/0.00 |
| per-pair recall@P≥0.93 | 0.9345 | 0.9296 | 0.9320 | 0.95/**0.27**/**0.20** |
- ΔRecall +1.17% แต่ ΔPrecision −1.62% → **GATE FAIL** (งบ precision −1.0%)
- **Verdict:** 🟡 ตอกย้ำว่า recall ซื้อถูกไม่ได้บน test split — แต่ operating point นี้ดัน MID/HARD slice
  ได้ (เป็นทางเลือกถ้าจะเอา recall เคสยากแลก precision) · แหล่ง: stage11_adaptive_threshold.py

### Exp D — REVIEW re-decision ด้วย strict name-match ✅ (ตัวที่เวิร์ค)
- insight: ใน review band score ไร้พลัง แต่ strict name-match (SequenceMatcher.ratio) แยกได้
- rule: name_sim≥0.9 → MATCH · 0.7–0.9 → คงไว้ · <0.7 → NO_MATCH
- ผล (verify โดยรันสคริปต์จริง): promote 2,920 คู่ (TP 2,612 / FP 308, precision 0.895);
  queue เหลือ 6.3%; เสียคู่จริง 695
- **ระบบรวม: recall 0.6710 → 0.7603, precision 0.9550 → 0.9475** ✅
- **Verdict:** ✅ improvement จริงและ reproduce ได้ — **แต่มี landmine (§4)** · แหล่ง: stage18_review_redecision.py

---

## 4. ⚠️ Reproducibility Landmine (ต้องปิดก่อนสร้างต่อ)

ผล Exp D ที่เวิร์ค **ขึ้นกับ metric ที่มาแบบบังเอิญ**: `jellyfish` ไม่ได้ติดตั้งใน venv →
สคริปต์ fallback เงียบไปใช้ `difflib.SequenceMatcher.ratio()` แทน Jaro-Winkler ที่โค้ดตั้งใจ

ทดสอบทั้งสอง metric บนข้อมูลเดียวกัน:
| name_sim metric | pairs ≥0.9 | precision rescue | precision รวมหลัง rule |
|---|---|---|---|
| SequenceMatcher.ratio (fallback ที่รันจริง) | 2,920 | **0.895** | **0.9475** ✅ |
| Jaro-Winkler (ถ้ามี jellyfish) | 7,300 | 0.438 | **0.819** ❌ พัง |

**ผลกระทบ:** ถ้าใครลง jellyfish หรือรันเครื่องอื่นที่มี lib นี้ โค้ดจะสลับไป JW เงียบ ๆ →
recall +9 จุดหายทันที precision ตกเหลือ 0.82 → **reproduce ไม่ได้ตอน defense**

**ทำไม SequenceMatcher เวิร์ค:** JW ให้น้ำหนัก prefix เยอะ → ชื่อคล้ายปานกลางได้ ≥0.9 ง่าย → FP;
SequenceMatcher (อิง LCS) เข้มกว่า ≥0.9 ≈ เกือบตรงเป๊ะ → precision สูง
**insight ที่ถูกต้อง = "ใช้ strict subsequence-ratio เป็น re-ranker" ไม่ใช่ "name_sim ช่วย" ลอย ๆ**

**ต้องแก้:** (1) ลบ try/except jellyfish เลือก metric แบบตั้งใจ deterministic ·
(2) เลือก threshold 0.9 บน validation ไม่ใช่ปรับบน production ·
(3) เขียนในเล่มเป็นตารางเทียบ metric เพื่อพิสูจน์ว่าทำไมเลือกตัวเข้ม

---

## 5. แผนพัฒนาต่อ (เรียงตามผลตอบแทน/หลักฐาน)

| ลำดับ | งาน | ผลคาด (verify แล้ว) | effort | gate |
|---|---|---|---|---|
| 1 | **REVIEW re-decision** + fix metric ให้ explicit (§4) | recall 0.671→0.760, prec 0.948 | ต่ำ (เกือบเสร็จ) | metric deterministic + threshold จาก val |
| 2 | **ลด match/review threshold** 0.98/0.95 + re-decision | กู้ 765 คู่ score≥0.90 ที่เกือบถึง band | ต่ำ | precision รวม ≥0.94 |
| 3 | **FP fix — recalibrate model tail** (862 คู่) | ลด false auto-merge | กลาง | precision auto-merge ↑ |
| 4 | **FP fix — username-collision guard** บน exact-first (63 คู่) | exact-first precision 0.9949→~1.0 | ต่ำ | เพิ่มเงื่อนไข secondary signal |
| 5 | **Blocking redesign** (3,316 never-scored) | ยกเพดาน recall retrieval | สูง | 64% name<0.5 → ต้อง non-name key |
| — | **Data enrichment** (url/geo/image) | ยกเพดาน §2.2 | สูงมาก | coverage weak-name >15% |

**Critical path:** งาน 1 → 2 → 3+4 (ปิด FP) → เขียนเล่ม · งาน 5 และ enrichment เป็น stretch

---

## 6. สิ่งที่ **ไม่ควรทำ** (พิสูจน์แล้วว่าไม่คุ้ม)
| แนวทาง | เหตุผลเชิงตัวเลข |
|---|---|
| Contrastive / two-tower encoder สำหรับ zero-overlap | hard slice = 0.22% ของ positive (test 5 คู่) |
| Redesign string similarity (char n-gram ฯลฯ) | char_cos ของ FN median 0.018 — ชื่อไม่ overlap จริง |
| Noisy-OR / fusion equation ใหม่ | AP 0.942 < GB — coverage เป็นกำแพง ไม่ใช่วิธีรวม |
| หา dataset โปรไฟล์แบบเดิมเพิ่ม (หวังแม่นขึ้น) | เพดานคือ signal type ไม่ใช่ sample size — ช่วยแค่ generalization |
| Calibration เพิ่มเพื่อ "เพิ่ม probability" | monotonic ไม่เปลี่ยน ranking — จับ FN ใหม่ไม่ได้ |

---

## 7. แม็ปเข้าบทของเล่ม
| เนื้อหา | เข้าบท |
|---|---|
| ตาราง slice-recall + per-platform (§1, §2.1) | บท 4 (ตารางใหม่) |
| การแยก FN 2 ปัญหา + FP 2 ที่มา (§2.3, §2.4) | บท 4 (error analysis) |
| Noisy-OR + ceiling analysis (§3 Exp A, §2.2) | บท 3 (สมการ) + บท 4 (ผล) |
| REVIEW re-decision + ตารางเทียบ metric (§3 Exp D, §4) | บท 4 (improvement) |
| coverage-bound recall เป็น finding | บท 5.3 (ข้อจำกัด แบบมีตัวเลข) |
| สิ่งที่ไม่ทำ + เหตุผล (§6) | บท 5.4 (future work / เหตุผลตัดสินใจ) |

---

## 8. ดัชนีไฟล์/สคริปต์ที่เกี่ยวข้อง
| ไฟล์ | หน้าที่ |
|---|---|
| stage10_eval_harness.py | slice + per-platform metrics (Phase 0) |
| stage11_adaptive_threshold.py | adaptive threshold experiment (Exp C) |
| stage9_noisy_or_fusion.py | noisy-OR fusion (Exp A) |
| stage16_error_groups.py / stage17_error_analysis.py | reproduce decision matrix + error groups |
| stage18_review_redecision.py | REVIEW re-decision (Exp D) — **ต้อง fix metric** |
| analysis_decision_matrix/ | CSV ทุกกลุ่ม (fp/fn/review/blocking-missed) + error_analysis_report.md |
| reports_eval_gb_baseline.json / reports_adaptive_threshold.json | ผล Phase 0/1 |
| IMPLEMENTATION_PLAN.md / ROADMAP_5MONTH.md | แผนสั้น/ยาว |
