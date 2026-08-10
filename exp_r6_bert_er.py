"""
Experiment R6: BERT-ER style semantic blocking + matching
(Li, Miao, Wang, Sun, Wang, "Improving the Efficiency and Effectiveness for
BERT-based Entity Resolution," AAAI 2021 -- referenced via Arford et al. 2025
ISI paper "Assessing the De-anonymization Risk of Social Media Users")

ต่างจาก R2/R3 (ที่ใช้ SentenceTransformer แค่เป็น similarity feature ตัวเดียวใน 18-feature MLP)
ตัวนี้คือความพยายาม implement สถาปัตยกรรมตามรูปที่ 3 ของ paper จริง ๆ:

  A) Encoding module  : DistilBERT encode "ทีละโปรไฟล์" (bi-encoder, ไม่ใช่ยัดคู่เข้าไปพร้อมกัน
                        ตามที่ paper อธิบายจริง -- ดู Fig.3 A: Reddit records กับ Instagram bios
                        เข้า BERT แยกกัน ได้ embedding matrix H, H')
  B) Blocking decoder  : learnable hash head (Linear->LayerNorm->tanh) แปลง embedding เป็น
                        k-bit code แล้ว bucket ตาม Hamming space -- ใช้แทน exact/username-prefix
                        blocking เดิม เป้าหมายคือกู้คู่ที่ blocking เดิม "มองข้าม" (3,316 คู่)
                        โดยเฉพาะกลุ่มที่ username/fullname string ไม่เหมือนกันเลย (name_sim ต่ำ)
  C) Matching decoder  : interaction features [h_a, h_b, h_a*h_b, |h_a-h_b|] -> MLP -> MATCH prob
                        (แทน "cross encoding / comparison / concatenation / linear layer" ใน Fig.3 C)

ข้อจำกัด/ปรับจากต้นฉบับ (บันทึกไว้ตรงนี้เพื่อความโปร่งใส เพราะ CPU-only ไม่มี GPU):
  - ใช้ distilbert-base-uncased (6 layer) แทน BERT-base เพื่อความเร็วบน CPU
  - **backbone แช่แข็ง (frozen) ไม่ fine-tune** -- วัดจริงบนเครื่องนี้แล้วว่า fine-tune ทั้งตัว
    (fwd+bwd ผ่าน distilbert ทุก step) ใช้เวลา ~6.3 วินาที/step ที่ batch=16 คู่ -> 1 epoch
    (~2,031 step) จะกิน ~3.6 ชั่วโมง ไม่ practical บน CPU-only ของเครื่องนี้ จึงเปลี่ยนมาเป็น
    encode โปรไฟล์ทั้งหมดครั้งเดียว (no_grad) แล้วเทรนเฉพาะ hash_head/match_head (เล็ก เร็วมาก
    เหมือน MLP ทั่วไปใน R2/R4) บน embedding ที่ตายตัวแล้ว -- เป็นเทคนิคมาตรฐาน "frozen BERT
    feature extractor + trainable heads" ไม่ใช่การ implement ผิดเจตนาแต่เป็นการปรับให้ทำงานได้
    จริงภายใต้ทรัพยากรที่มี ผลคือ hash/match head เรียนรู้จาก representation ของ DistilBERT
    pretrained ตรง ๆ โดยไม่ได้ fine-tune ให้เข้ากับ domain โปรไฟล์โซเชียลมีเดียเพิ่มเติม
  - hash head เทรนด้วย continuous relaxation (tanh) + cosine contrastive loss + quantization
    regularizer แล้ว sign() ตอน inference เท่านั้น (ไม่ได้ implement straight-through estimator
    เต็มรูปแบบตาม paper ต้นฉบับ)
  - Cartesian negative sampling ของ paper (R×I ทุกคู่) ทำแบบย่อ: สุ่ม negative pairs เพิ่มจากทั้ง
    universe ของโปรไฟล์ (ไม่ใช่แค่คู่ที่ blocking เดิมหาเจอ) แทนการ enumerate ทุกคู่จริง (677M คู่)

กฎที่ต้องรักษาเข้มงวด (ไม่งั้นผล "กู้คืนได้" จะเป็น data leakage/circular):
  3,316 คู่ blocking-missed (BLOCKING_MISSED) ห้ามใช้ train เด็ดขาด -- ใช้เป็น held-out
  recovery test เท่านั้น (ตรวจว่า hash ที่เทรนจากคู่อื่นล้วน ๆ จะจับคู่เหล่านี้ไว้ bucket เดียวกัน
  โดยไม่เคยเห็นมันมาก่อนหรือไม่)

Checkpointing: profile embeddings cache แยกจาก head-training checkpoint -- resume ได้ทั้งคู่
"""
from __future__ import annotations

import json
import os

# distilbert-base-uncased ถูก cache ไว้ครบแล้ว (config/weights/tokenizer) -- บังคับ offline กัน
# from_pretrained() ไปค้าง network/lock ระหว่างเช็ค revision บน Hub (เจอปัญหานี้จริงระหว่างพัฒนา)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from exp_lib import (EXP, MATCH, NO_MATCH, REVIEW, BLOCKING_MISSED, PROFILES,
                     build_cache, evaluate, save_json, split_constants)
import exp_r1_ga_redecision as r1
from exp_r2_bert_feature import FocalLoss, decide_manual

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
HASH_BITS = 64
MAX_LEN = 96
ENCODE_BATCH = 128          # batch สำหรับ encode โปรไฟล์ทั้งหมดครั้งเดียว (no_grad, เร็ว)
HEAD_BATCH = 2048           # batch สำหรับเทรน hash/match head บน embedding cache (เร็วมาก)
EPOCHS = 40
PATIENCE = 5
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-5
NEG_RATIO = 3.0
EXTRA_RANDOM_NEG_RATIO = 1.0   # เพิ่ม negative แบบสุ่มทั้ง universe เท่ากับจำนวน cache negative
HASH_LOSS_WEIGHT = 0.5
QUANT_REG_WEIGHT = 0.01
COSINE_MARGIN = 0.1

SMOKE = os.environ.get("R6_SMOKE") == "1"
if SMOKE:
    EPOCHS = 2

R6_DIR = EXP / "r6_bert_er"
R6_DIR.mkdir(exist_ok=True)
LOG_PATH = EXP / ("r6_smoke_run.log" if SMOKE else "r6_run.log")
CKPT_NAME = "heads_checkpoint_smoke.pt" if SMOKE else "heads_checkpoint.pt"
BEST_NAME = "heads_best_smoke.pt" if SMOKE else "heads_best.pt"
CKPT_PATH = R6_DIR / CKPT_NAME
EMB_PATH = R6_DIR / ("profile_embeddings_smoke.npy" if SMOKE else "profile_embeddings.npy")
EMB_IDS_PATH = R6_DIR / ("profile_embeddings_ids_smoke.npy" if SMOKE else "profile_embeddings_ids.npy")
PROB_PATH = R6_DIR / "r6_probabilities.parquet"
BLOCKING_RECOVERY_PATH = EXP / "r6_blocking_recovery.json"
NEW_CANDIDATES_PATH = R6_DIR / "new_candidate_pairs.parquet"
R6_RESULTS = EXP / "r6_results.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- profile text + tokenization ----
def clean_field(x) -> str:
    s = str(x).strip()
    return "" if s.lower() in ("", "nan") else s


def build_profile_texts() -> pd.DataFrame:
    df = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna()].copy()

    def row_text(r):
        parts = []
        for tag, col in [("username", "userName"), ("fullname", "fullName"),
                         ("bio", "bio"), ("location", "location"), ("platform", "platform")]:
            v = clean_field(r.get(col, ""))
            if v:
                parts.append(f"[{tag}] {v}")
        return " ".join(parts) if parts else "[empty]"

    texts = df.apply(row_text, axis=1)
    return pd.DataFrame({"profile_row_id": df["profile_row_id"].astype(int).values, "text": texts.values})


def tokenize_all_profiles(tokenizer) -> dict:
    prof_text = build_profile_texts()
    ids = prof_text["profile_row_id"].tolist()
    texts = prof_text["text"].tolist()
    enc = tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True,
                    return_tensors="pt")
    out = {}
    for i, pid in enumerate(ids):
        out[int(pid)] = (enc["input_ids"][i], enc["attention_mask"][i])
    return out


# ---------------------------------------------------------------- model ----
class BertERModel(nn.Module):
    def __init__(self, model_name: str, hash_bits: int):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        h = self.backbone.config.hidden_size
        self.hash_head = nn.Sequential(nn.Linear(h, hash_bits), nn.LayerNorm(hash_bits), nn.Tanh())
        self.match_head = nn.Sequential(
            nn.Linear(4 * h, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.hidden_size = h

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (tok * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-6)
        return summed / counts  # mean pooling

    def hash_embed(self, h: torch.Tensor) -> torch.Tensor:
        return self.hash_head(h)

    def match_logit(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([h_a, h_b, h_a * h_b, (h_a - h_b).abs()], dim=-1)
        return self.match_head(feat).squeeze(-1)


def cosine_contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, labels: torch.Tensor,
                            margin: float = COSINE_MARGIN) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1)
    pos_loss = (1.0 - cos)
    neg_loss = torch.clamp(cos - margin, min=0.0)
    per_pair = torch.where(labels == 1, pos_loss, neg_loss)
    return per_pair.mean()


def quantization_reg(z: torch.Tensor) -> torch.Tensor:
    return (1.0 - z.pow(2)).mean()


# ---------------------------------------------------------------- encode all profiles (ONE-TIME, frozen backbone) ----
@torch.no_grad()
def encode_all_profiles(model: BertERModel, tok: dict, device) -> tuple[np.ndarray, np.ndarray]:
    if EMB_PATH.exists() and EMB_IDS_PATH.exists():
        log(f"embeddings cache found -> {EMB_PATH}")
        return np.load(EMB_PATH), np.load(EMB_IDS_PATH)
    model.eval()
    ids = list(tok.keys())
    if SMOKE:
        ids = ids[:2000]
    embs = np.zeros((len(ids), model.hidden_size), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(ids), ENCODE_BATCH):
        chunk = ids[s:s + ENCODE_BATCH]
        input_ids = torch.stack([tok[i][0] for i in chunk]).to(device)
        attn = torch.stack([tok[i][1] for i in chunk]).to(device)
        h = model.encode(input_ids, attn)
        embs[s:s + len(chunk)] = h.cpu().numpy()
        if s and (s // ENCODE_BATCH) % 20 == 0:
            elapsed = time.time() - t0
            log(f"  encoding profiles {s:,}/{len(ids):,} ({elapsed:.0f}s, "
                f"{elapsed/(s/ENCODE_BATCH):.2f}s/batch, ETA {elapsed/s*(len(ids)-s):.0f}s)")
    ids = np.array(ids, dtype=np.int64)
    np.save(EMB_PATH, embs)
    np.save(EMB_IDS_PATH, ids)
    log(f"encoded {len(ids):,} profiles in {time.time()-t0:.0f}s -> {EMB_PATH}")
    return embs, ids


# ---------------------------------------------------------------- training pairs ----
def build_training_pairs(cache: pd.DataFrame) -> pd.DataFrame:
    bm = pd.read_csv(BLOCKING_MISSED, keep_default_na=False)
    bm_pairs = set(zip(bm["profile_id_a"].astype(int), bm["profile_id_b"].astype(int)))

    scored = cache[cache.decision_source != "AUTO_EXACT"]
    val = scored[scored.split == "val"][["profile_id_a", "profile_id_b", "actual"]].copy()
    val = val.rename(columns={"actual": "label"})
    before = len(val)
    mask_overlap = val.apply(
        lambda r: (int(r.profile_id_a), int(r.profile_id_b)) in bm_pairs
        or (int(r.profile_id_b), int(r.profile_id_a)) in bm_pairs, axis=1)
    assert mask_overlap.sum() == 0, "cache val pairs overlap with held-out blocking_missed pairs!"
    val["source"] = "cache"

    prof = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    prof["profile_row_id"] = pd.to_numeric(prof["profile_row_id"], errors="coerce").astype("Int64")
    prof = prof[prof["profile_row_id"].notna()].set_index("profile_row_id")
    folder = prof["user_folder"]
    platform = prof["platform"] if "platform" in prof.columns else pd.Series(dtype=object)

    from exp_lib import bucket as split_bucket
    keys = folder.where(folder != "", "pid:" + folder.index.astype(str))
    prof_split = keys.map(split_bucket)
    val_ids = prof_split[prof_split == "val"].index.values

    # ขนาดตาม target หลัง undersample (pos*NEG_RATIO) ไม่ใช่จำนวน negative ดิบก่อน undersample
    n_extra = int((val.label == 1).sum() * NEG_RATIO * EXTRA_RANDOM_NEG_RATIO)
    rng = np.random.default_rng(SEED)
    extra_rows = []
    tries = 0
    while len(extra_rows) < n_extra and tries < n_extra * 20:
        tries += 1
        a, b = rng.choice(val_ids, size=2, replace=False)
        if platform.get(a) == platform.get(b):
            continue
        fa, fb = folder.get(a, ""), folder.get(b, "")
        if fa and fa == fb:
            continue  # ไม่ใช่ negative จริง ข้าม
        pa, pb = int(a), int(b)
        if (pa, pb) in bm_pairs or (pb, pa) in bm_pairs:
            continue
        extra_rows.append((pa, pb, 0))
    extra = pd.DataFrame(extra_rows, columns=["profile_id_a", "profile_id_b", "label"])
    extra["source"] = "random_extra"

    pairs = pd.concat([val, extra], ignore_index=True)
    pos = pairs[pairs.label == 1]
    neg = pairs[pairs.label == 0]
    n_neg_keep = min(len(neg), int(len(pos) * NEG_RATIO))
    neg = neg.sample(n=n_neg_keep, random_state=SEED)
    out = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log(f"training pairs: total={len(out):,} pos={int((out.label==1).sum()):,} "
        f"neg={int((out.label==0).sum()):,} (cache negs before extra={before - int(val.label.sum()):,}, "
        f"extra random negs added={len(extra):,})")
    return out


# ---------------------------------------------------------------- train heads on frozen embeddings ----
def pairs_to_row_indices(pairs: pd.DataFrame, id_to_row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_idx = pairs["profile_id_a"].map(id_to_row).values.astype(np.int64)
    b_idx = pairs["profile_id_b"].map(id_to_row).values.astype(np.int64)
    labels = pairs["label"].values.astype(np.float32)
    return a_idx, b_idx, labels


@torch.no_grad()
def eval_ap(model, embs: torch.Tensor, a_idx, b_idx, labels) -> float:
    model.eval()
    h_a, h_b = embs[a_idx], embs[b_idx]
    logit = model.match_logit(h_a, h_b)
    probs = torch.sigmoid(logit).cpu().numpy()
    return float(average_precision_score(labels, probs))


def train_heads(model: BertERModel, embs: torch.Tensor, id_to_row: dict,
                train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    params = list(model.hash_head.parameters()) + list(model.match_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    a_tr, b_tr, y_tr = pairs_to_row_indices(train_df, id_to_row)
    a_ho, b_ho, y_ho = pairs_to_row_indices(holdout_df, id_to_row)

    start_epoch = 1
    best_ap, patience_left = -1.0, PATIENCE
    if CKPT_PATH.exists():
        ck = torch.load(CKPT_PATH, map_location="cpu")
        model.hash_head.load_state_dict(ck["hash_head"])
        model.match_head.load_state_dict(ck["match_head"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_ap = ck.get("best_ap", -1.0)
        log(f"resumed heads from checkpoint epoch {ck['epoch']}, best_ap={best_ap:.4f}")

    n = len(a_tr)
    idx_all = np.arange(n)
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        model.backbone.eval()  # backbone แช่แข็งเสมอ ไม่ปิด eval mode
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(idx_all)
        total_loss, total_match, total_hash = 0.0, 0.0, 0.0
        n_batches = 0
        t0 = time.time()
        for s in range(0, n, HEAD_BATCH):
            bidx = idx_all[s:s + HEAD_BATCH]
            h_a = embs[a_tr[bidx]]
            h_b = embs[b_tr[bidx]]
            y = torch.from_numpy(y_tr[bidx])

            logit = model.match_logit(h_a, h_b)
            match_loss = focal(logit, y)

            z_a = model.hash_embed(h_a)
            z_b = model.hash_embed(h_b)
            z_cat = torch.cat([z_a, z_b], 0)
            hash_loss = cosine_contrastive_loss(z_a, z_b, y) + QUANT_REG_WEIGHT * quantization_reg(z_cat)

            loss = match_loss + HASH_LOSS_WEIGHT * hash_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_match += match_loss.item()
            total_hash += hash_loss.item()
            n_batches += 1
        scheduler.step()

        val_ap = eval_ap(model, embs, a_ho, b_ho, y_ho)
        log(f"epoch {epoch:03d} loss={total_loss/max(n_batches,1):.4f} "
            f"match={total_match/max(n_batches,1):.4f} hash={total_hash/max(n_batches,1):.4f} "
            f"holdout_AP={val_ap:.4f} ({time.time()-t0:.1f}s)")

        torch.save({"hash_head": model.hash_head.state_dict(), "match_head": model.match_head.state_dict(),
                   "optimizer": optimizer.state_dict(), "epoch": epoch, "best_ap": max(best_ap, val_ap)},
                  CKPT_PATH)

        if val_ap > best_ap:
            best_ap = val_ap
            patience_left = PATIENCE
            torch.save({"hash_head": model.hash_head.state_dict(), "match_head": model.match_head.state_dict(),
                       "epoch": epoch, "best_ap": best_ap}, R6_DIR / BEST_NAME)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"early stopping at epoch {epoch}")
                break
    return model


# ---------------------------------------------------------------- hashing + blocking ----
def compute_hash_codes(model: BertERModel, embs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model.hash_head(embs).cpu().numpy()
    bits = (z >= 0).astype(np.uint64)
    powers = (2 ** np.arange(HASH_BITS, dtype=np.uint64))
    codes = (bits * powers).sum(axis=1)
    return codes


def build_buckets(codes: np.ndarray, ids: np.ndarray) -> dict:
    buckets: dict[int, list[int]] = {}
    for code, pid in zip(codes.tolist(), ids.tolist()):
        buckets.setdefault(code, []).append(pid)
    return buckets


def neighbor_codes(code: int, radius: int = 1):
    yield code
    if radius >= 1:
        for i in range(HASH_BITS):
            yield code ^ (1 << i)


def blocking_recovery_eval(id_to_code: dict, radius: int) -> dict:
    bm = pd.read_csv(BLOCKING_MISSED, keep_default_na=False)
    n = len(bm)
    found = 0
    missing_ids = 0
    for a, b in zip(bm["profile_id_a"].astype(int), bm["profile_id_b"].astype(int)):
        if a not in id_to_code or b not in id_to_code:
            missing_ids += 1
            continue
        ca, cb = id_to_code[a], id_to_code[b]
        neigh = set(neighbor_codes(ca, radius))
        if cb in neigh:
            found += 1
    return {"radius": radius, "pool_size": n, "recovered_as_candidate": found,
           "recovery_rate": round(found / n, 4) if n else 0.0, "ids_not_encoded": missing_ids}


def enumerate_bucket_pairs(buckets: dict, max_bucket_size: int = 2000) -> pd.DataFrame:
    rows_a, rows_b = [], []
    skipped_large = 0
    for code, members in buckets.items():
        if len(members) < 2:
            continue
        if len(members) > max_bucket_size:
            skipped_large += 1
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rows_a.append(members[i])
                rows_b.append(members[j])
    if skipped_large:
        log(f"skipped {skipped_large} buckets larger than {max_bucket_size} (too expensive to enumerate)")
    a = np.array(rows_a, dtype=np.int64)
    b = np.array(rows_b, dtype=np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return pd.DataFrame({"profile_id_a": lo, "profile_id_b": hi}).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------- match scoring (fast: tensor indexing only) ----
def score_pairs_with_match_head(model: BertERModel, embs: torch.Tensor, id_to_row: dict,
                                pairs: pd.DataFrame, chunk: int = 50000) -> np.ndarray:
    """out[i] = probability for pairs.iloc[i]; NaN if either id was never encoded
    (only happens in R6_SMOKE mode where profile encoding is subsampled -- in the
    full run every profile_id in `pairs` is guaranteed to have been encoded)."""
    model.eval()
    a_row = pairs["profile_id_a"].map(id_to_row)
    b_row = pairs["profile_id_b"].map(id_to_row)
    valid = a_row.notna().values & b_row.notna().values
    out = np.full(len(pairs), np.nan, dtype=np.float32)
    if not valid.any():
        return out
    a_idx = a_row[valid].astype(np.int64).values
    b_idx = b_row[valid].astype(np.int64).values
    valid_positions = np.where(valid)[0]
    with torch.no_grad():
        for s in range(0, len(a_idx), chunk):
            e = min(s + chunk, len(a_idx))
            h_a = embs[a_idx[s:e]]
            h_b = embs[b_idx[s:e]]
            logit = model.match_logit(h_a, h_b)
            out[valid_positions[s:e]] = torch.sigmoid(logit).cpu().numpy()
    return out


def main():
    LOG_PATH.write_text("", encoding="utf-8")
    log(f"=== R6 BERT-ER start (SMOKE={SMOKE}, frozen-backbone mode) ===")
    set_seed(SEED)
    device = torch.device("cpu")

    cache = build_cache()
    consts = split_constants(cache)
    assert consts["full"]["total_pos"] == 29247

    from transformers import AutoTokenizer
    log(f"loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok = tokenize_all_profiles(tokenizer)
    log(f"tokenized {len(tok):,} profiles")

    model = BertERModel(MODEL_NAME, HASH_BITS).to(device)
    log("model constructed (backbone frozen)")

    embs_np, ids = encode_all_profiles(model, tok, device)
    embs = torch.from_numpy(embs_np)
    id_to_row = {int(pid): i for i, pid in enumerate(ids)}

    train_pairs = build_training_pairs(cache)
    # เก็บเฉพาะคู่ที่ profile ทั้งสองฝั่งถูก encode แล้ว (สำคัญเฉพาะตอน SMOKE ที่ encode ไม่ครบ)
    keep = train_pairs["profile_id_a"].isin(id_to_row) & train_pairs["profile_id_b"].isin(id_to_row)
    train_pairs = train_pairs[keep].reset_index(drop=True)
    idx = np.arange(len(train_pairs))
    idx_train, idx_holdout = train_test_split(idx, test_size=0.15, random_state=SEED,
                                              stratify=train_pairs["label"].values)
    train_df = train_pairs.iloc[idx_train].reset_index(drop=True)
    holdout_df = train_pairs.iloc[idx_holdout].reset_index(drop=True)
    log(f"train={len(train_df):,} holdout={len(holdout_df):,}")

    model = train_heads(model, embs, id_to_row, train_df, holdout_df)

    best_path = R6_DIR / BEST_NAME
    if best_path.exists():
        ck = torch.load(best_path, map_location=device)
        model.hash_head.load_state_dict(ck["hash_head"])
        model.match_head.load_state_dict(ck["match_head"])
        log(f"loaded {BEST_NAME} (epoch {ck['epoch']}, holdout_AP={ck['best_ap']:.4f})")

    codes = compute_hash_codes(model, embs)
    id_to_code = {int(pid): int(c) for pid, c in zip(ids, codes)}
    buckets = build_buckets(codes, ids)
    bucket_sizes = np.array([len(v) for v in buckets.values()])
    log(f"hash buckets: {len(buckets):,} unique codes over {len(ids):,} profiles "
        f"(max bucket={bucket_sizes.max()}, mean nonempty={bucket_sizes.mean():.2f})")

    recovery_r0 = blocking_recovery_eval(id_to_code, radius=0)
    recovery_r1 = blocking_recovery_eval(id_to_code, radius=1)
    log(f"blocking recovery radius=0: {recovery_r0}")
    log(f"blocking recovery radius=1: {recovery_r1}")

    # ของคู่ blocking-missed ที่ hash เจอ (radius<=1) -> เช็คว่า match_head ให้คะแนน MATCH ถูกไหม
    bm = pd.read_csv(BLOCKING_MISSED, keep_default_na=False)
    bm_pairs = bm[["profile_id_a", "profile_id_b"]].astype(int)
    bm_probs = score_pairs_with_match_head(model, embs, id_to_row, bm_pairs)
    bm_out = bm_pairs.copy()
    bm_out["probability_r6"] = bm_probs
    bm_out["recovered_radius0"] = [int(cb in set(neighbor_codes(id_to_code.get(a, -1), 0)))
                                   for a, cb in zip(bm_out.profile_id_a, [id_to_code.get(b, -2) for b in bm_out.profile_id_b])]
    bm_out["recovered_radius1"] = [int(cb in set(neighbor_codes(id_to_code.get(a, -1), 1)))
                                   for a, cb in zip(bm_out.profile_id_a, [id_to_code.get(b, -2) for b in bm_out.profile_id_b])]
    n_end_to_end_r0 = int(((bm_out.recovered_radius0 == 1) & (bm_out.probability_r6 >= 0.5)).sum())
    n_end_to_end_r1 = int(((bm_out.recovered_radius1 == 1) & (bm_out.probability_r6 >= 0.5)).sum())
    save_json(BLOCKING_RECOVERY_PATH, {
        "pool_size": len(bm_out),
        "radius0": recovery_r0, "radius1": recovery_r1,
        "end_to_end_recovered_as_MATCH_radius0": n_end_to_end_r0,
        "end_to_end_recovered_as_MATCH_radius1": n_end_to_end_r1,
    })
    log(f"end-to-end recovery (found by hash AND classified MATCH by match_head): "
        f"radius0={n_end_to_end_r0}/{len(bm_out)} radius1={n_end_to_end_r1}/{len(bm_out)}")

    # apples-to-apples กับ R0-R5: score ทุกคู่ที่ blocking เดิมเจอ (2,073,842 คู่) ด้วย match_head
    scored = cache[cache.decision_source != "AUTO_EXACT"][["profile_id_a", "profile_id_b", "actual", "split", "name_sim"]]
    probs_orig = score_pairs_with_match_head(model, embs, id_to_row, scored[["profile_id_a", "profile_id_b"]])
    prob_df = pd.DataFrame({"profile_id_a": scored.profile_id_a.values, "profile_id_b": scored.profile_id_b.values,
                            "probability_r6": probs_orig})
    prob_df.to_parquet(PROB_PATH, index=False)

    scored_p = scored.merge(prob_df, on=["profile_id_a", "profile_id_b"], how="left")
    results = {"thresholds": {"MATCH": 0.98, "REVIEW": 0.95}, "splits": {}}
    for sp in ["test", "full"]:
        sub = scored_p if sp == "full" else scored_p[scored_p.split == sp]
        code = decide_manual(sub["probability_r6"].values)
        actual = sub["actual"].values.astype(np.int8)
        m = evaluate(code, actual, consts[sp])
        results["splits"][sp] = m
        log(f"R6 [{sp}] manual-threshold {m}")

    val = scored_p[scored_p.split == "val"]
    best, history = r1.run_ga(val["probability_r6"].values, val["name_sim"].values,
                              val["actual"].values.astype(np.int8), 0)
    results["ga_genome"] = dict(t_m=float(best[0]), t_r=float(best[1]), c_promote=float(best[2]), c_demote=float(best[3]))
    results["ga_rules"] = {}
    for sp in ["test", "full"]:
        sub = scored_p if sp == "full" else scored_p[scored_p.split == sp]
        code = r1.decide_code(sub["probability_r6"].values, sub["name_sim"].values, best)
        actual = sub["actual"].values.astype(np.int8)
        m = evaluate(code, actual, consts[sp])
        results["ga_rules"][sp] = m
        log(f"R6-GA [{sp}] {m}")

    # workload/quality ของ blocking ใหม่ (นอกเหนือจาก recovery diagnostic ด้านบน)
    if not SMOKE:
        new_pairs = enumerate_bucket_pairs(buckets)
        new_pairs.to_parquet(NEW_CANDIDATES_PATH, index=False)
        orig_pairs = set(zip(cache.profile_id_a.astype(int), cache.profile_id_b.astype(int)))
        orig_pairs |= set(zip(cache.profile_id_b.astype(int), cache.profile_id_a.astype(int)))
        genuinely_new = new_pairs[~new_pairs.apply(
            lambda r: (int(r.profile_id_a), int(r.profile_id_b)) in orig_pairs, axis=1)]
        results["new_blocking"] = {
            "total_bucket_pairs": len(new_pairs),
            "genuinely_new_vs_original_2073842": len(genuinely_new),
        }
        log(f"new hash-based blocking: {len(new_pairs):,} total pairs in shared buckets, "
            f"{len(genuinely_new):,} not in original 2,073,842-pair candidate set")

    save_json(R6_RESULTS, results)
    log(f"saved -> {R6_RESULTS}")
    log("=== R6 done ===")


if __name__ == "__main__":
    main()
