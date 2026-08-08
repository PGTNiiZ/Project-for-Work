# แผน Implement ละเอียด: R6 BERT-ER แบบ GPU Fine-tuning (Future Work)

> ขยายจากหัวข้อ 6 ของ `REPORT_R0_R5.md` — เอกสารนี้คือ blueprint ระดับโค้ดสำหรับคนที่จะสานต่อ
> ทุกการแก้ไขอ้างอิงไฟล์ `exp_r6_bert_er.py` เวอร์ชันปัจจุบัน (frozen-backbone) เป็นจุดตั้งต้น
> เป้าหมาย: กู้ 3,316 คู่ blocking-missed ให้ได้ ≥10–20% และดัน R6-GA test F1 ≥ 0.8373 (เท่า R3)

---

## Phase 0 — เตรียมสภาพแวดล้อม (ครึ่งวัน)

### 0.1 Hardware ที่ต้องการ

| ระดับ | GPU | VRAM | batch ที่รันได้ | เวลาโดยประมาณ/epoch |
|---|---|---|---|---|
| ขั้นต่ำ | RTX 3060 / T4 (Colab ฟรี) | 8–16GB | 32–64 คู่ | ~10–15 นาที |
| แนะนำ | RTX 4070 / A10 / Colab Pro (A100) | 12–40GB | 128–256 คู่ | ~3–6 นาที |

เหตุผลตัวเลข: บน CPU เครื่องเดิมวัดจริงได้ 6.3 วินาที/step ที่ batch=16 (2,031 step/epoch = ~3.6 ชม.)
GPU consumer เร็วกว่า CPU สำหรับ transformer ~30–60 เท่า → step ละ ~0.1–0.2 วินาที

### 0.2 Software setup

```bash
# บนเครื่อง GPU (หรือ Colab)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # ให้ตรง CUDA version
pip install transformers pandas pyarrow scikit-learn jellyfish
# copy ไฟล์ที่ต้องใช้จากโปรเจกต์ (ไม่ต้อง copy ทั้ง repo):
#   exp_lib.py, exp_r1_ga_redecision.py, exp_r2_bert_feature.py, exp_r6_bert_er.py
#   experiments/scored_pairs_enriched.parquet        (cache กลาง — สร้างครั้งเดียวจาก exp_lib)
#   analysis_decision_matrix/blocking_missed_pairs.csv (held-out 3,316 คู่)
#   Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv
```

ตรวจว่า GPU มองเห็น: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

---

## Phase 1 — แก้โค้ดจาก frozen → full fine-tune (1 วัน)

### 1.1 ปลด freeze backbone

ใน `BertERModel.__init__` (บรรทัดที่มี `requires_grad_(False)`):

```python
# เดิม (frozen mode):
self.backbone = AutoModel.from_pretrained(model_name)
self.backbone.eval()
for p in self.backbone.parameters():
    p.requires_grad_(False)

# ใหม่ (fine-tune mode):
self.backbone = AutoModel.from_pretrained(model_name)
# ไม่ freeze — ปล่อยให้ gradient ไหลผ่าน backbone ทุก step
```

และเอา `@torch.no_grad()` ออกจากเมธอด `encode()` (จำเป็น ไม่งั้น gradient ไม่ไหลเข้า backbone):

```python
# เดิม:
@torch.no_grad()
def encode(self, input_ids, attention_mask): ...

# ใหม่ (ไม่มี decorator):
def encode(self, input_ids, attention_mask): ...
```

### 1.2 Optimizer แบบ differential learning rate

backbone ต้องใช้ LR ต่ำ (ปรับน้ำหนัก pretrained เบา ๆ) ส่วน head ที่เริ่มจากศูนย์ใช้ LR สูงกว่า:

```python
LR_BACKBONE = 2e-5      # มาตรฐาน BERT fine-tuning (Devlin et al. แนะนำ 2e-5 ถึง 5e-5)
LR_HEAD = 1e-3

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE, "weight_decay": 0.01},
    {"params": list(model.hash_head.parameters()) + list(model.match_head.parameters()),
     "lr": LR_HEAD, "weight_decay": 1e-5},
])
# warmup 10% ของ total steps แล้ว decay เชิงเส้น (มาตรฐาน BERT fine-tuning)
from transformers import get_linear_schedule_with_warmup
total_steps = (len(train_df) // BATCH_SIZE) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
# เรียก scheduler.step() ทุก step (ไม่ใช่ทุก epoch แบบ CosineAnnealing เดิม)
```

### 1.3 Training loop: กลับไป forward ผ่าน backbone ทุก step + mixed precision

แทนที่ `train_heads()` (ที่ index embedding cache) ด้วย loop ที่ tokenize-forward จริง
โครงนี้เคยมีอยู่ในไฟล์เวอร์ชันแรก (ก่อนเปลี่ยนเป็น frozen) — ดูใน git history หรือเขียนตามนี้:

```python
scaler = torch.cuda.amp.GradScaler()          # mixed precision — เร็วขึ้น ~2x, VRAM ลดครึ่ง
device = torch.device("cuda")

for epoch in range(1, EPOCHS + 1):
    model.train()
    rng = np.random.default_rng(SEED + epoch)
    rng.shuffle(idx_all)
    for s in range(0, len(idx_all), BATCH_SIZE):
        idx = idx_all[s:s + BATCH_SIZE]
        ids_a, mask_a, ids_b, mask_b, y = make_batch(train_df, idx, tok, device)
        # encode ทั้งสองฝั่งใน forward เดียว (concat) — ประหยัดกว่าเรียก backbone 2 รอบ
        ids = torch.cat([ids_a, ids_b], 0)
        masks = torch.cat([mask_a, mask_b], 0)

        with torch.cuda.amp.autocast():
            h = model.encode(ids, masks)
            h_a, h_b = h[:len(idx)], h[len(idx):]
            match_loss = focal(model.match_logit(h_a, h_b), y)
            z_a, z_b = model.hash_embed(h_a), model.hash_embed(h_b)
            hash_loss = cosine_contrastive_loss(z_a, z_b, y) \
                        + QUANT_REG_WEIGHT * quantization_reg(torch.cat([z_a, z_b], 0))
            loss = match_loss + HASH_LOSS_WEIGHT * hash_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # กัน gradient ระเบิดตอน fine-tune
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

ค่า config ที่แนะนำเริ่มต้น:

```python
BATCH_SIZE = 64          # ปรับตาม VRAM: 32 (8GB) / 64 (12GB) / 128+ (24GB+)
EPOCHS = 5               # BERT fine-tuning ปกติ 3–5 epoch ก็อิ่มตัว (ต่างจาก head-only ที่ต้อง 40)
PATIENCE = 2
MAX_LEN = 96             # คงเดิม — โปรไฟล์ส่วนใหญ่สั้น
```

### 1.4 Straight-Through Estimator (STE) สำหรับ hash head

ปัจจุบันใช้ tanh relaxation แล้ว sign() เฉพาะตอน inference — ปัญหาคือสิ่งที่ optimize (cosine ของ
ค่าต่อเนื่อง) ไม่ตรงกับสิ่งที่ใช้จริง (Hamming ของบิต) ทำเต็มรูปแบบตาม paper ต้นฉบับด้วย STE:

```python
class STEBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        return torch.sign(z)              # forward: บิตจริง {-1, +1}
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output                # backward: ส่ง gradient ผ่านเหมือน identity

# ใน hash_embed():
def hash_embed(self, h):
    z = self.hash_head(h)                # tanh output (-1, 1)
    return STEBinarize.apply(z)          # เทรนบนบิตจริง แต่ gradient ยังไหลได้
```

ผลคือ contrastive loss คำนวณบน binary code จริง → ช่องว่างระหว่าง train กับ inference หายไป

### 1.5 Hard-negative mining (สำคัญมากสำหรับ blocking recall)

random negative ปัจจุบันง่ายเกินไป (โปรไฟล์สุ่มสองคนแทบไม่มีอะไรเหมือนกัน) hash เลยเรียนแค่
"แยกของที่ต่างกันสุดขั้ว" ไม่ได้เรียน "แยกของที่คล้ายกันแต่ไม่ใช่" เพิ่ม mining step ทุก ๆ 1 epoch:

```python
@torch.no_grad()
def mine_hard_negatives(model, embs_current, ids, folder_map, k=5, n_max=20000):
    """หาคู่ที่ embedding ใกล้กัน (top-k neighbors) แต่คนละ user_folder -> negative ที่ยากจริง"""
    import faiss                                    # หรือ sklearn NearestNeighbors ถ้าข้อมูลเล็ก
    z = torch.nn.functional.normalize(embs_current, dim=1).cpu().numpy()
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    _, nbr = index.search(z, k + 1)                 # +1 เพราะตัวเองจะติดมาด้วย
    rows = []
    for i, neighbors in enumerate(nbr):
        for j in neighbors[1:]:
            a, b = int(ids[i]), int(ids[j])
            if folder_map.get(a) != folder_map.get(b):   # คนละคนจริง
                rows.append((min(a, b), max(a, b), 0))
    df = pd.DataFrame(rows, columns=["profile_id_a", "profile_id_b", "label"]).drop_duplicates()
    return df.sample(n=min(n_max, len(df)), random_state=SEED)
```

แล้วผสม hard negatives เหล่านี้เข้า train_df ก่อนเริ่ม epoch ถัดไป (สัดส่วนแนะนำ:
positive 1 : random negative 2 : hard negative 1) — **ระวัง: ต้องกรอง 3,316 คู่ held-out ออกจาก
ผล mining ทุกครั้ง** (โค้ดปัจจุบันมี `bm_pairs` set ไว้เช็คแล้ว ใช้ต่อได้เลย)

---

## Phase 2 — Hyperparameter sweep (1–2 วัน, รันคู่ขนานได้)

รันทีละตัวแปรโดยตรึงที่เหลือ (ไม่ต้อง grid search เต็ม):

| ตัวแปร | ค่าที่ลอง | ตัวชี้วัดที่ดู |
|---|---|---|
| `HASH_BITS` | 32 / **48** / 64 / 128 | blocking recovery@radius≤2 บน val-positives (ไม่ใช่ 3,316 held-out!) |
| `HASH_LOSS_WEIGHT` | 0.5 / **1.0** / 2.0 | trade-off: recovery ขึ้นแต่ match AP ต้องไม่ตกเกิน 2 จุด |
| `COSINE_MARGIN` | 0.1 / **0.3** / 0.5 | margin สูง = บีบ negative ออกไกลขึ้น |
| hard-neg ratio | 0 / **1x** / 2x ของ positive | recovery + FP ของ candidate ใหม่ |

ตัวหนา = ค่าที่คาดว่าดีสุดตามสัญชาตญาณ ใช้เป็นจุดเริ่ม
**หลักการสำคัญ: จูนทุกอย่างบน val-positives (คู่จริงใน val split ที่ blocking เดิมเจอ) เท่านั้น
ห้ามแตะ 3,316 คู่ held-out จนกว่าจะ freeze config สุดท้ายแล้ววัดครั้งเดียว** — ไม่งั้นตัวเลข
recovery จะ overfit กับ test pool และเชื่อไม่ได้

## Phase 3 — Evaluation (ครึ่งวัน) — protocol ต้องเหมือนรอบ CPU เป๊ะ

ลำดับที่ต้องรัน (ทุกข้อมีโค้ดอยู่แล้วใน `exp_r6_bert_er.py` ส่วน `main()`):

1. encode โปรไฟล์ทั้ง 36,807 ด้วย backbone ที่ fine-tune แล้ว → cache `.npy`
2. `compute_hash_codes()` + `build_buckets()` → รายงาน bucket size distribution
3. `blocking_recovery_eval()` ที่ radius 0/1/2/3 บน 3,316 held-out (วัด **ครั้งเดียว** หลัง freeze config)
4. end-to-end: คู่ที่ hash เจอ + match_head ≥ 0.5 → นับเป็น "กู้ได้จริง"
5. score 2,073,842 คู่เดิม → `exp_lib.evaluate()` บน test/full + รัน GA (`r1.run_ga`) → R6-GA ใหม่
6. `enumerate_bucket_pairs()` → นับ candidate ใหม่ทั้งหมดที่ hash สร้าง แล้ววัด precision ของ
   กลุ่มนี้ (คู่ใหม่กี่ % เป็นคู่จริง) — ตัวเลขนี้บอกต้นทุน review ที่เพิ่มขึ้น

### เกณฑ์ตัดสิน (กำหนดล่วงหน้า กันตีความเข้าข้างตัวเอง)

| ผลลัพธ์ | การตัดสิน |
|---|---|
| recovery ≥ 20% (≥663 คู่) และ R6-GA F1 ≥ 0.8373 | **สำเร็จ** — เสนอ deploy เป็น blocking เสริม (union กับเดิม) |
| recovery 10–20% หรือ F1 ใกล้ R3 (±0.01) | **สำเร็จบางส่วน** — คุ้มถ้าคิว review รับไหว, รายงานเป็น option |
| recovery < 10% แม้ fine-tune แล้ว | **ยืนยันผลลบ** — สรุปว่า dataset นี้ (คู่ blocking-missed ส่วนใหญ่ bio/location ว่าง) ไม่มีสัญญาณพอสำหรับ semantic blocking ไม่ว่าโมเดลแรงแค่ไหน |

หมายเหตุแถวสุดท้าย: มีความเป็นไปได้จริง เพราะจากการวิเคราะห์ error ในเล่ม (ตาราง 4.11) คู่ FN กลุ่ม
googleplus→twitter จำนวนมากมีแต่ username/fullname ที่ต่างกัน และ bio ว่างเปล่า — ถ้า field อื่น
ไม่มีข้อมูลเลย semantic model ก็ไม่มีอะไรให้จับ นี่คือขีดจำกัดของ **ข้อมูล** ไม่ใช่ของโมเดล

---

## Timeline รวม

| Phase | งาน | เวลา (มี GPU แล้ว) |
|---|---|---|
| 0 | setup environment + ย้ายไฟล์ | 0.5 วัน |
| 1 | แก้โค้ด 5 จุด (unfreeze, optimizer, loop+AMP, STE, hard-neg mining) | 1 วัน |
| 2 | sweep 4 ตัวแปร (รันคู่ขนานบน Colab ได้) | 1–2 วัน |
| 3 | final evaluation + เขียนสรุป | 0.5 วัน |
| **รวม** | | **3–4 วันทำงาน** |

## ความเสี่ยงหลักและทางแก้

1. **Colab ฟรี timeout กลางคัน** → โค้ดมี checkpoint/resume อยู่แล้ว (`CKPT_PATH`) ใช้ต่อได้เลย
   แค่ mount Google Drive แล้วชี้ `R6_DIR` ไปที่ Drive
2. **fine-tune แล้ว match head เก่งขึ้นแต่ hash แย่ลง** (loss สองตัวแย่งกัน) → เทรนสองเฟส:
   เฟสแรก fine-tune backbone + match head ก่อน (HASH_LOSS_WEIGHT=0), เฟสสอง freeze backbone
   แล้วเทรน hash head อย่างเดียวบน embedding ใหม่
3. **คู่ held-out รั่วเข้า training โดยไม่ตั้งใจ** (ผ่าน hard-neg mining หรือ random sampling)
   → assert เช็ค `bm_pairs` ทุกครั้งที่สร้าง pair ใหม่ (มี pattern นี้ในโค้ดแล้ว ห้ามลบ)

---
*อ้างอิง: Li et al. 2021 (AAAI) "Improving the Efficiency and Effectiveness for BERT-based
Entity Resolution"; Arford et al. 2025 (IEEE ISI) "Assessing the De-anonymization Risk of
Social Media Users" — ผล baseline ที่ต้องเอาชนะ: R3 test F1=0.8373 (`r3_results.json`),
R6 frozen-backbone recovery 3/3,316 (`r6_blocking_recovery.json`)*
