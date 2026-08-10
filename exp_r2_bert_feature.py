"""
Experiment R2/R3: BERT representation (ดู EXPERIMENT_PLAN.md ข้อ 6)

R2 = MLP ใหม่ (17 feature เดิม + bert_cos = feature ที่ 18) + threshold มือเดิม (0.98/0.95)
R3 = GA re-decision (จาก exp_r1_ga_redecision.py) บน probability_r2

หมายเหตุสำคัญ: labeled_pairs.parquet (ทั้ง root และ Project-for-Work/train_data/) ใช้ profile_id
ช่วง 0-9999 ซึ่งเป็นคนละ id space กับ profile_row_id ใน scored_pairs_enriched.parquet (0-36806)
-- ตรวจสอบแล้วว่า join ไม่ได้ (เป็น dataset ทดลองคนละชุด) จึงคำนวณ feature 17 ตัวเองจากข้อความ
โปรไฟล์ดิบ (userName/fullName/bio/location/platform) แทน แล้วรักษา "โครง IdentityMLP เดิม" ตาม
stage10_13_training_pipeline.py (hidden [256,128,64], FocalLoss, Adam, cosine LR, isotonic
calibration, undersample negative train-neg-ratio=3.0) ตามที่ prompt สั่งให้ copy สถาปัตยกรรม

Checkpointing: ทุกขั้นตอนเช็คไฟล์ output ก่อน ถ้ามีแล้วข้าม (rerun ต่อได้ ไม่ต้องเริ่มใหม่)
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from exp_lib import (EXP, MATCH, NO_MATCH, REVIEW, DEC_CODE, PROFILES,
                     build_cache, evaluate, save_json, split_constants)
import exp_r1_ga_redecision as r1

SEED = 42
LOG_PATH = EXP / "r2_run.log"

EMB_PATH = EXP / "profile_embeddings_minilm.npy"
EMB_IDS_PATH = EXP / "profile_embeddings_minilm_ids.npy"
BERT_COS_PATH = EXP / "pair_bert_cos.parquet"
FEAT17_PATH = EXP / "pair_features17.parquet"
MODEL_DIR = EXP / "r2_model"
PROB_PATH = EXP / "r2_probabilities.parquet"
R2_RESULTS = EXP / "r2_results.json"
R3_RESULTS = EXP / "r3_results.json"
R3_HISTORY = EXP / "r3_ga_history.json"
FN_RECOVERY = EXP / "r2_r3_fn_recovery.json"

FEATURE17_COLS = [
    "username_jaro", "username_lev_sim", "username_token_sort",
    "fullname_jaro", "fullname_lev_sim", "fullname_token_sort",
    "name_sim",
    "bio_jaccard", "bio_len_ratio", "both_have_bio",
    "location_jaro", "location_exact", "both_have_location",
    "same_platform",
    "username_len_diff", "fullname_wordcount_diff", "username_prefix3_match",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- step 1/2: embeddings ----
def build_profile_texts() -> pd.DataFrame:
    df = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna()].copy()

    def clean(s):
        s = s.astype(str).str.strip()
        return s.where(~s.str.lower().isin(["", "nan"]), "")

    full = clean(df["fullName"])
    bio = clean(df["bio"])
    loc = clean(df["location"])
    text = (full + " " + bio + " " + loc).str.strip()
    text = text.where(text != "", "[empty]")
    return pd.DataFrame({"profile_row_id": df["profile_row_id"].astype(int).values, "text": text.values})


def encode_profiles() -> tuple[np.ndarray, np.ndarray]:
    if EMB_PATH.exists() and EMB_IDS_PATH.exists():
        log(f"embeddings cache found -> {EMB_PATH}")
        return np.load(EMB_PATH), np.load(EMB_IDS_PATH)

    from sentence_transformers import SentenceTransformer

    prof_text = build_profile_texts()
    log(f"encoding {len(prof_text):,} profiles ด้วย all-MiniLM-L6-v2 (CPU) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        prof_text["text"].tolist(), batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    ids = prof_text["profile_row_id"].values.astype(np.int64)
    np.save(EMB_PATH, emb)
    np.save(EMB_IDS_PATH, ids)
    log(f"saved embeddings {emb.shape} -> {EMB_PATH}")
    return emb, ids


def compute_bert_cos(cache: pd.DataFrame, emb: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    if BERT_COS_PATH.exists():
        log(f"bert_cos cache found -> {BERT_COS_PATH}")
        return pd.read_parquet(BERT_COS_PATH)

    id_to_row = {int(pid): i for i, pid in enumerate(ids)}
    scored = cache[cache.decision_source != "AUTO_EXACT"]
    a_idx = scored["profile_id_a"].map(id_to_row)
    b_idx = scored["profile_id_b"].map(id_to_row)
    assert a_idx.notna().all() and b_idx.notna().all(), "some profile_id missing from embeddings index"
    a_idx = a_idx.astype(np.int64).values
    b_idx = b_idx.astype(np.int64).values
    bert_cos = np.sum(emb[a_idx] * emb[b_idx], axis=1).astype(np.float32)

    out = pd.DataFrame({
        "profile_id_a": scored["profile_id_a"].values,
        "profile_id_b": scored["profile_id_b"].values,
        "bert_cos": bert_cos,
    })
    out.to_parquet(BERT_COS_PATH, index=False)
    log(f"saved bert_cos for {len(out):,} pairs -> {BERT_COS_PATH}")
    return out


# ---------------------------------------------------------------- step: 17 hand features ----
def _clean(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.lower().str.strip()
    return s.where(~s.isin(["", "nan"]), "").values


def compute_features17(cache: pd.DataFrame) -> pd.DataFrame:
    if FEAT17_PATH.exists():
        log(f"features17 cache found -> {FEAT17_PATH}")
        return pd.read_parquet(FEAT17_PATH)

    prof = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    prof["profile_row_id"] = pd.to_numeric(prof["profile_row_id"], errors="coerce").astype("Int64")
    prof = prof[prof["profile_row_id"].notna()].set_index("profile_row_id")

    scored = cache[cache.decision_source != "AUTO_EXACT"].reset_index(drop=True)
    n = len(scored)
    log(f"computing 17 hand features สำหรับ {n:,} scored pairs ...")

    ua = _clean(prof["userName"].reindex(scored["profile_id_a"]))
    ub = _clean(prof["userName"].reindex(scored["profile_id_b"]))
    fa = _clean(prof["fullName"].reindex(scored["profile_id_a"]))
    fb = _clean(prof["fullName"].reindex(scored["profile_id_b"]))
    ba = _clean(prof["bio"].reindex(scored["profile_id_a"]))
    bb = _clean(prof["bio"].reindex(scored["profile_id_b"]))
    la = _clean(prof["location"].reindex(scored["profile_id_a"]))
    lb = _clean(prof["location"].reindex(scored["profile_id_b"]))
    pa = _clean(prof["platform"].reindex(scored["profile_id_a"])) if "platform" in prof.columns else np.full(n, "")
    pb = _clean(prof["platform"].reindex(scored["profile_id_b"])) if "platform" in prof.columns else np.full(n, "")

    jaro = jellyfish.jaro_winkler_similarity
    lev = jellyfish.levenshtein_distance

    username_jaro = np.zeros(n, dtype=np.float32)
    username_lev_sim = np.zeros(n, dtype=np.float32)
    username_token_sort = np.zeros(n, dtype=np.float32)
    fullname_jaro = np.zeros(n, dtype=np.float32)
    fullname_lev_sim = np.zeros(n, dtype=np.float32)
    fullname_token_sort = np.zeros(n, dtype=np.float32)
    location_jaro = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i in range(n):
        x, y = ua[i], ub[i]
        if x and y:
            username_jaro[i] = jaro(x, y)
            m = max(len(x), len(y))
            username_lev_sim[i] = 1.0 - lev(x, y) / m if m else 0.0
            username_token_sort[i] = 1.0 if x == y else jaro("".join(sorted(x)), "".join(sorted(y)))
        x, y = fa[i], fb[i]
        if x and y:
            fullname_jaro[i] = jaro(x, y)
            m = max(len(x), len(y))
            fullname_lev_sim[i] = 1.0 - lev(x, y) / m if m else 0.0
            tx = " ".join(sorted(x.split()))
            ty = " ".join(sorted(y.split()))
            fullname_token_sort[i] = jaro(tx, ty)
        x, y = la[i], lb[i]
        if x and y:
            location_jaro[i] = jaro(x, y)
        if i and i % 500000 == 0:
            log(f"  ... {i:,}/{n:,} ({time.time()-t0:.0f}s)")
    log(f"loop features เสร็จใน {time.time()-t0:.0f}s")

    def word_set(arr):
        return [set(s.split()) for s in arr]

    bio_wa, bio_wb = word_set(ba), word_set(bb)
    bio_jaccard = np.array([
        (len(x & y) / len(x | y)) if (x or y) else 0.0
        for x, y in zip(bio_wa, bio_wb)
    ], dtype=np.float32)
    len_ba = np.array([len(s) for s in ba])
    len_bb = np.array([len(s) for s in bb])
    bio_len_ratio = np.where(
        np.maximum(len_ba, len_bb) > 0,
        np.minimum(len_ba, len_bb) / np.maximum(np.maximum(len_ba, len_bb), 1), 0.0,
    ).astype(np.float32)
    both_have_bio = ((len_ba > 0) & (len_bb > 0)).astype(np.float32)

    location_exact = np.array([1.0 if (x and x == y) else 0.0 for x, y in zip(la, lb)], dtype=np.float32)
    both_have_location = np.array([1.0 if (x and y) else 0.0 for x, y in zip(la, lb)], dtype=np.float32)
    same_platform = np.array([1.0 if (x and x == y) else 0.0 for x, y in zip(pa, pb)], dtype=np.float32)

    len_ua = np.array([len(s) for s in ua])
    len_ub = np.array([len(s) for s in ub])
    username_len_diff = np.where(
        np.maximum(len_ua, len_ub) > 0,
        np.abs(len_ua - len_ub) / np.maximum(np.maximum(len_ua, len_ub), 1), 0.0,
    ).astype(np.float32)

    wc_fa = np.array([len(s.split()) for s in fa])
    wc_fb = np.array([len(s.split()) for s in fb])
    fullname_wordcount_diff = np.where(
        np.maximum(wc_fa, wc_fb) > 0,
        np.abs(wc_fa - wc_fb) / np.maximum(np.maximum(wc_fa, wc_fb), 1), 0.0,
    ).astype(np.float32)

    username_prefix3_match = np.array([
        1.0 if (len(x) >= 3 and len(y) >= 3 and x[:3] == y[:3]) else 0.0
        for x, y in zip(ua, ub)
    ], dtype=np.float32)

    out = pd.DataFrame({
        "profile_id_a": scored["profile_id_a"].values,
        "profile_id_b": scored["profile_id_b"].values,
        "username_jaro": username_jaro,
        "username_lev_sim": username_lev_sim,
        "username_token_sort": username_token_sort,
        "fullname_jaro": fullname_jaro,
        "fullname_lev_sim": fullname_lev_sim,
        "fullname_token_sort": fullname_token_sort,
        "name_sim": scored["name_sim"].values.astype(np.float32),
        "bio_jaccard": bio_jaccard,
        "bio_len_ratio": bio_len_ratio,
        "both_have_bio": both_have_bio,
        "location_jaro": location_jaro,
        "location_exact": location_exact,
        "both_have_location": both_have_location,
        "same_platform": same_platform,
        "username_len_diff": username_len_diff,
        "fullname_wordcount_diff": fullname_wordcount_diff,
        "username_prefix3_match": username_prefix3_match,
    })
    out.to_parquet(FEAT17_PATH, index=False)
    log(f"saved features17 -> {FEAT17_PATH}")
    return out


# ---------------------------------------------------------------- MLP: copy จาก stage10_13_training_pipeline.py (โครง/hparam เดิม, ห้าม redesign) ----
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, torch.full_like(targets, self.alpha), torch.full_like(targets, 1 - self.alpha))
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class IdentityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


TRAIN_NEG_RATIO = 3.0
BATCH_SIZE = 8192
EVAL_BATCH_SIZE = 16384
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


@dataclass(frozen=True)
class R2TrainingConfig:
    """Reproducible settings for a fresh validation-only R2 model training run."""

    seed: int = SEED
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    train_neg_ratio: float = TRAIN_NEG_RATIO
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    dropout: float = DROPOUT
    focal_alpha: float = FOCAL_ALPHA
    focal_gamma: float = FOCAL_GAMMA

    def validate(self):
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be at least 1")
        if self.batch_size < 2 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive and training batch_size at least 2")
        if self.train_neg_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("train_neg_ratio and learning_rate must be positive")
        if self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("weight_decay must be non-negative and dropout in [0, 1)")
        return self


def undersample_negatives(x: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(keep)
    return x[keep], y[keep]


def make_loader(x, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_probs(model, x, batch_size, device):
    loader = make_loader(x, np.zeros(len(x), dtype=np.float32), batch_size, shuffle=False)
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def train_mlp(train_x, train_y, val_x, val_y, input_dim: int, seed: int = SEED,
              training_config: R2TrainingConfig | None = None, logger=log):
    """Train the existing IdentityMLP architecture with configurable, reproducible settings."""
    cfg = (training_config or R2TrainingConfig(seed=seed)).validate()
    set_seed(cfg.seed)
    device = torch.device("cpu")
    model = IdentityMLP(input_dim=input_dim, dropout=cfg.dropout).to(device)
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loader = make_loader(train_x, train_y, cfg.batch_size, shuffle=True)

    best_state, best_ap, best_epoch, patience_left = None, -1.0, -1, cfg.patience
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        val_probs = predict_probs(model, val_x, cfg.eval_batch_size, device)
        val_ap = average_precision_score(val_y, val_probs)
        if logger:
            logger(f"  epoch {epoch:02d} loss={total_loss/max(len(train_x),1):.4f} val_ap={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, best_epoch, patience_left = float(val_ap), epoch, PATIENCE
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                if logger:
                    logger(f"  early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_ap": best_ap}


def train_r2_probabilities(cache: pd.DataFrame, feat17: pd.DataFrame, bert_cos: pd.DataFrame,
                           output_dir: Path, training_config: R2TrainingConfig | None = None,
                           logger=None) -> tuple[pd.DataFrame, dict]:
    """Train a fresh R2 model on validation only and save all artifacts below ``output_dir``.

    Cached hand/BERT features are inputs, not fitted artifacts. Test rows are transformed and
    predicted only after the scaler, MLP, and calibrator have been fitted on validation rows.
    """
    cfg = (training_config or R2TrainingConfig()).validate()
    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    probability_path = output_dir / "r2_probabilities.parquet"
    metadata_path = output_dir / "r2_training.json"
    if probability_path.exists() or model_dir.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite R2 training artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = cache[cache.decision_source != "AUTO_EXACT"][
        ["profile_id_a", "profile_id_b", "actual", "split"]
    ]
    feats = scored.merge(feat17, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one") \
                  .merge(bert_cos, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one")
    feature_cols = FEATURE17_COLS + ["bert_cos"]
    if len(feats) != len(scored) or feats[feature_cols].isna().any().any():
        raise ValueError("R2 training feature joins are incomplete or non-unique")

    val_mask = feats["split"].values == "val"
    if not val_mask.any():
        raise ValueError("R2 training requires validation rows")
    X_val_all = feats.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
    y_val_all = feats.loc[val_mask, "actual"].to_numpy(dtype=np.float32)
    if len(np.unique(y_val_all)) != 2:
        raise ValueError("R2 validation labels need both classes")

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_val_all))
    idx_train, idx_holdout = train_test_split(
        idx, test_size=0.15, random_state=cfg.seed, stratify=y_val_all)
    scaler = StandardScaler()
    Xtr_raw, ytr = X_val_all[idx_train], y_val_all[idx_train]
    Xtr_bal, ytr_bal = undersample_negatives(Xtr_raw, ytr, cfg.train_neg_ratio, cfg.seed)
    Xtr_bal = scaler.fit_transform(Xtr_bal)
    Xho = scaler.transform(X_val_all[idx_holdout])
    yho = y_val_all[idx_holdout]
    if logger:
        logger(f"R2 train-inner balanced={len(Xtr_bal):,}, holdout={len(Xho):,}; validation only")
    model, training_meta = train_mlp(
        Xtr_bal, ytr_bal, Xho, yho, input_dim=len(feature_cols),
        training_config=cfg, logger=logger,
    )

    device = torch.device("cpu")
    holdout_probs_raw = predict_probs(model, Xho, cfg.eval_batch_size, device)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(holdout_probs_raw, yho)
    cal_probs = calibrator.predict(holdout_probs_raw)
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1s = [f1_score(yho, (cal_probs >= threshold).astype(int), zero_division=0) for threshold in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    best_f1 = float(max(f1s))

    # The model has already been fitted. This is inference only for both validation and test rows.
    X_all_scaled = scaler.transform(feats[feature_cols].to_numpy(dtype=np.float32))
    probabilities = calibrator.predict(
        predict_probs(model, X_all_scaled, cfg.eval_batch_size, device)
    ).astype(np.float32)
    output = pd.DataFrame({
        "profile_id_a": feats["profile_id_a"].values,
        "profile_id_b": feats["profile_id_b"].values,
        "probability_r2": probabilities,
    })

    model_dir.mkdir()
    torch.save({"state_dict": model.state_dict(), "feature_dim": len(feature_cols)}, model_dir / "identity_mlp_r2.pt")
    import pickle
    with (model_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    with (model_dir / "isotonic_calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)
    with (model_dir / "feature_cols.pkl").open("wb") as handle:
        pickle.dump(feature_cols, handle)
    output.to_parquet(probability_path, index=False)
    metadata = {
        **training_meta,
        "training_config": asdict(cfg),
        "feature_columns": feature_cols,
        "validation_only_training": True,
        "inner_holdout_fraction": 0.15,
        "holdout_best_f1": best_f1,
        "holdout_best_threshold": best_threshold,
        "n_validation": int(len(X_val_all)),
        "n_train": int(len(Xtr_bal)),
        "n_inner_holdout": int(len(Xho)),
        "n_predicted_all_splits": int(len(output)),
        "probability_path": str(probability_path),
    }
    save_json(metadata_path, metadata)
    if logger:
        logger(f"saved fresh R2 model/probabilities -> {output_dir}")
    return output, metadata


def build_r2_probabilities(cache: pd.DataFrame, feat17: pd.DataFrame, bert_cos: pd.DataFrame) -> pd.DataFrame:
    if PROB_PATH.exists():
        log(f"probability_r2 cache found -> {PROB_PATH}")
        return pd.read_parquet(PROB_PATH)

    MODEL_DIR.mkdir(exist_ok=True)
    scored = cache[cache.decision_source != "AUTO_EXACT"][
        ["profile_id_a", "profile_id_b", "actual", "split"]
    ]
    feats = scored.merge(feat17, on=["profile_id_a", "profile_id_b"], how="left") \
                   .merge(bert_cos, on=["profile_id_a", "profile_id_b"], how="left")
    assert len(feats) == len(scored), "merge changed row count -- duplicate keys?"
    feature_cols = FEATURE17_COLS + ["bert_cos"]
    assert feats[feature_cols].isna().sum().sum() == 0, "missing features after merge"

    val_mask = feats["split"].values == "val"
    X_val_all = feats.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
    y_val_all = feats.loc[val_mask, "actual"].to_numpy(dtype=np.float32)

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_val_all))
    idx_train, idx_holdout = train_test_split(
        idx, test_size=0.15, random_state=SEED, stratify=y_val_all)

    scaler = StandardScaler()
    Xtr_raw, ytr = X_val_all[idx_train], y_val_all[idx_train]
    Xtr_bal, ytr_bal = undersample_negatives(Xtr_raw, ytr, TRAIN_NEG_RATIO, SEED)
    Xtr_bal = scaler.fit_transform(Xtr_bal)
    Xho = scaler.transform(X_val_all[idx_holdout])
    yho = y_val_all[idx_holdout]
    log(f"train_inner balanced={len(Xtr_bal):,} (pos={int(ytr_bal.sum()):,}) holdout={len(Xho):,} (pos={int(yho.sum()):,})")

    model, meta = train_mlp(Xtr_bal, ytr_bal, Xho, yho, input_dim=len(feature_cols))

    device = torch.device("cpu")
    holdout_probs_raw = predict_probs(model, Xho, EVAL_BATCH_SIZE, device)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(holdout_probs_raw, yho)

    thresholds = np.arange(0.01, 1.00, 0.01)
    cal_probs = calibrator.predict(holdout_probs_raw)
    f1s = [f1_score(yho, (cal_probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_thr, best_f1 = float(thresholds[int(np.argmax(f1s))]), float(max(f1s))
    log(f"holdout best_val_f1={best_f1:.4f} @ threshold={best_thr:.2f} (informational only; R2 uses fixed 0.98/0.95)")

    X_all = feats[feature_cols].to_numpy(dtype=np.float32)
    X_all_scaled = scaler.transform(X_all)
    probs_raw_all = predict_probs(model, X_all_scaled, EVAL_BATCH_SIZE, device)
    probs_cal_all = calibrator.predict(probs_raw_all).astype(np.float32)

    torch.save({"state_dict": model.state_dict(), "feature_dim": len(feature_cols)}, MODEL_DIR / "identity_mlp_r2.pt")
    import pickle
    with (MODEL_DIR / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)
    with (MODEL_DIR / "isotonic_calibrator.pkl").open("wb") as fh:
        pickle.dump(calibrator, fh)
    with (MODEL_DIR / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(feature_cols, fh)
    save_json(MODEL_DIR / "train_meta.json", {**meta, "holdout_best_f1": best_f1, "holdout_best_threshold": best_thr,
                                              "train_neg_ratio": TRAIN_NEG_RATIO, "n_train": int(len(Xtr_bal)),
                                              "n_holdout": int(len(Xho))})

    out = pd.DataFrame({
        "profile_id_a": feats["profile_id_a"].values,
        "profile_id_b": feats["profile_id_b"].values,
        "probability_r2": probs_cal_all,
    })
    out.to_parquet(PROB_PATH, index=False)
    log(f"saved probability_r2 for {len(out):,} pairs -> {PROB_PATH}")
    return out


# ---------------------------------------------------------------- R2: threshold มือเดิม ----
MATCH_T, REVIEW_T = 0.98, 0.95


def decide_manual(prob: np.ndarray) -> np.ndarray:
    d = np.full(len(prob), NO_MATCH, dtype=np.int8)
    d[prob >= REVIEW_T] = REVIEW
    d[prob >= MATCH_T] = MATCH
    return d


def run_r2(cache: pd.DataFrame, consts: dict, prob_df: pd.DataFrame) -> dict:
    scored = cache[cache.decision_source != "AUTO_EXACT"].merge(
        prob_df, on=["profile_id_a", "profile_id_b"], how="left")
    assert scored["probability_r2"].isna().sum() == 0

    results = {"thresholds": {"MATCH": MATCH_T, "REVIEW": REVIEW_T}, "splits": {}}
    for sp in ["test", "full"]:
        sub = scored if sp == "full" else scored[scored.split == sp]
        code = decide_manual(sub["probability_r2"].values)
        actual = sub["actual"].values.astype(np.int8)
        m = evaluate(code, actual, consts[sp])
        results["splits"][sp] = m
        log(f"R2 [{sp}] {m}")
    save_json(R2_RESULTS, results)
    log(f"saved -> {R2_RESULTS}")
    return results


# ---------------------------------------------------------------- R3: GA (reuse exp_r1_ga_redecision, ห้ามแก้ไฟล์นั้น) ----
def run_r3(cache: pd.DataFrame, consts: dict, prob_df: pd.DataFrame) -> tuple[dict, list, np.ndarray]:
    scored = cache[cache.decision_source != "AUTO_EXACT"].merge(
        prob_df, on=["profile_id_a", "profile_id_b"], how="left")
    assert scored["probability_r2"].isna().sum() == 0

    results = {"weights": dict(W_FP=r1.W_FP, W_FN=r1.W_FN, W_REV=r1.W_REV), "rules": {}}

    val = scored[scored.split == "val"]
    const_fn_below = 0  # probability_r2 ไม่มี floor แบบ score>=0.5 ของ R1 -> ใช้ทุกคู่ val ตรงๆ
    log(f"GA (R3) บน val: {len(val):,} คู่")
    best, history = r1.run_ga(val["probability_r2"].values, val["name_sim"].values,
                              val["actual"].values.astype(np.int8), const_fn_below)
    log(f"R3 best genome: t_m={best[0]:.4f} t_r={best[1]:.4f} c_promote={best[2]:.4f} c_demote={best[3]:.4f}")
    results["best_genome"] = dict(t_m=float(best[0]), t_r=float(best[1]),
                                  c_promote=float(best[2]), c_demote=float(best[3]))

    for sp in ["test", "full"]:
        sub = scored if sp == "full" else scored[scored.split == sp]
        prob, nsim = sub["probability_r2"].values, sub["name_sim"].values
        actual = sub["actual"].values.astype(np.int8)
        code = r1.decide_code(prob, nsim, best)
        m = evaluate(code, actual, consts[sp])
        results["rules"].setdefault(sp, {})["R3_ga"] = m
        log(f"R3 [{sp}] {m}")
    save_json(R3_RESULTS, results)
    save_json(R3_HISTORY, history)
    log(f"saved -> {R3_RESULTS}, {R3_HISTORY}")

    # decision code สำหรับทุกคู่ scored (ใช้ full projection genome) -> ใช้ทำ FN-recovery diagnostic
    full_code = r1.decide_code(scored["probability_r2"].values, scored["name_sim"].values, best)
    return results, history, full_code


# ---------------------------------------------------------------- FN-recovery diagnostic ----
def fn_recovery(cache: pd.DataFrame, prob_df: pd.DataFrame, r3_full_code: np.ndarray) -> dict:
    scored = cache[cache.decision_source != "AUTO_EXACT"].merge(
        prob_df, on=["profile_id_a", "profile_id_b"], how="left").reset_index(drop=True)
    fn_pool_mask = (scored["actual"] == 1) & (scored["decision"] == "NO_MATCH")
    n_pool = int(fn_pool_mask.sum())

    r2_code_all = decide_manual(scored["probability_r2"].values)
    r2_recovered = int(((r2_code_all != NO_MATCH) & fn_pool_mask.values).sum())
    r3_recovered = int(((r3_full_code != NO_MATCH) & fn_pool_mask.values).sum())
    r2_recovered_match = int(((r2_code_all == MATCH) & fn_pool_mask.values).sum())
    r3_recovered_match = int(((r3_full_code == MATCH) & fn_pool_mask.values).sum())

    out = {
        "fn_pool_size": n_pool,
        "R2_recovered_any (MATCH or REVIEW)": r2_recovered,
        "R2_recovered_to_MATCH": r2_recovered_match,
        "R3_recovered_any (MATCH or REVIEW)": r3_recovered,
        "R3_recovered_to_MATCH": r3_recovered_match,
    }
    save_json(FN_RECOVERY, out)
    log(f"FN recovery -> {out}")
    return out


def main():
    LOG_PATH.write_text("", encoding="utf-8")
    log("=== R2/R3 pipeline start ===")
    set_seed(SEED)

    cache = build_cache()
    consts = split_constants(cache)
    assert consts["full"]["total_pos"] == 29247, consts["full"]["total_pos"]
    # sanity gate (hard rule 3): R0 บน full ต้องได้ FP=925, REVIEW=86296, TP=19624
    scored_all = cache[cache.decision_source != "AUTO_EXACT"]
    prod_code_full = scored_all["decision"].map(DEC_CODE).values.astype(np.int8)
    r0_full = evaluate(prod_code_full, scored_all["actual"].values.astype(np.int8), consts["full"])
    assert r0_full["FP"] == 925 and r0_full["REVIEW"] == 86296 and r0_full["TP"] == 19624, r0_full
    log(f"sanity gate OK: R0 full = {r0_full}")

    emb, ids = encode_profiles()
    bert_cos = compute_bert_cos(cache, emb, ids)
    feat17 = compute_features17(cache)
    prob_df = build_r2_probabilities(cache, feat17, bert_cos)

    r2_results = run_r2(cache, consts, prob_df)
    r3_results, r3_history, r3_full_code = run_r3(cache, consts, prob_df)
    fn_recovery(cache, prob_df, r3_full_code)

    log("=== R2/R3 pipeline done ===")


if __name__ == "__main__":
    pass  # Legacy entrypoint disabled; use run_ga_experiments.py.
# Publication-grade R2 trainer override: explicit nested entity splits and provenance.

@dataclass(frozen=True)
class R2TrainingConfig:
    """Deterministic configuration for one fresh, nested-split R2 model artifact."""

    seed: int = SEED
    split_seed: int = 42
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    train_neg_ratio: float = TRAIN_NEG_RATIO
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    dropout: float = DROPOUT
    focal_alpha: float = FOCAL_ALPHA
    focal_gamma: float = FOCAL_GAMMA
    num_threads: int = 1

    def validate(self):
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be at least 1")
        if self.batch_size < 2 or self.eval_batch_size < 1 or self.num_threads < 1:
            raise ValueError("batch sizes must be positive and num_threads at least 1")
        if self.train_neg_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("train_neg_ratio and learning_rate must be positive")
        if self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("weight_decay must be non-negative and dropout in [0, 1)")
        return self


def set_deterministic_seed(seed: int, num_threads: int = 1) -> None:
    """Set every local RNG and PyTorch execution mode used by a fresh R2 run."""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        # PyTorch permits this global setting only before parallel work begins.
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def environment_snapshot(num_threads: int) -> dict:
    import platform
    import sklearn
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cpu": platform.processor() or platform.machine(),
        "torch_num_threads": int(num_threads),
        "deterministic_algorithms": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_feature_frame(frame: pd.DataFrame, columns: list[str], label: str,
                            lower: float, upper: float) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite feature values")
    # Dot products of float32-normalised embeddings can exceed one by a few ULPs.
    # Treat that numerical drift as in-range, but reject any meaningful violation.
    tolerance = 1e-6
    if (values < lower - tolerance).any() or (values > upper + tolerance).any():
        raise ValueError(f"{label} contains features outside [{lower}, {upper}]")


def _preflight_r2_inputs(cache: pd.DataFrame, feat17: pd.DataFrame, bert_cos: pd.DataFrame) -> None:
    key_cols = ["profile_id_a", "profile_id_b"]
    _require = lambda frame, cols, label: (
        (_ for _ in ()).throw(ValueError(f"{label} is missing required columns: {sorted(set(cols) - set(frame.columns))}"))
        if set(cols) - set(frame.columns) else None
    )
    _require(cache, key_cols + ["actual", "split", "decision_source", "experiment_split"], "scored cache")
    _require(feat17, key_cols + FEATURE17_COLS, "17-feature cache")
    _require(bert_cos, key_cols + ["bert_cos"], "BERT cosine cache")
    for frame, label in ((cache, "scored cache"), (feat17, "17-feature cache"), (bert_cos, "BERT cosine cache")):
        if frame.duplicated(key_cols).any():
            raise ValueError(f"{label} contains duplicate pair keys")
    _validate_feature_frame(feat17, FEATURE17_COLS, "17-feature cache", 0.0, 1.0)
    _validate_feature_frame(bert_cos, ["bert_cos"], "BERT cosine cache", -1.0, 1.0)


def train_mlp(train_x, train_y, val_x, val_y, input_dim: int, seed: int = SEED,
              training_config: R2TrainingConfig | None = None, logger=log):
    """Train IdentityMLP and choose a checkpoint solely by calibration AP."""
    cfg = (training_config or R2TrainingConfig(seed=seed)).validate()
    set_deterministic_seed(cfg.seed, cfg.num_threads)
    device = torch.device("cpu")
    model = IdentityMLP(input_dim=input_dim, dropout=cfg.dropout).to(device)
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loader = make_loader(train_x, train_y, cfg.batch_size, shuffle=True)

    best_state, best_ap, best_epoch, patience_left = None, -1.0, -1, cfg.patience
    history = []
    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        val_probs = predict_probs(model, val_x, cfg.eval_batch_size, device)
        val_ap = float(average_precision_score(val_y, val_probs))
        improved = val_ap > best_ap
        if improved:
            best_ap, best_epoch, patience_left = val_ap, epoch, cfg.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(total_loss / max(len(train_x), 1)),
            "calibration_average_precision": val_ap,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.perf_counter() - started),
            "checkpoint_improved": bool(improved),
        }
        history.append(row)
        if logger:
            logger("  epoch {epoch:02d} loss={train_loss:.4f} cal_ap={calibration_average_precision:.4f} lr={learning_rate:.7f} improved={checkpoint_improved}".format(**row))
        if not improved and patience_left <= 0:
            if logger:
                logger(f"  early stopping at epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError("R2 did not produce a calibration checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_calibration_ap": best_ap,
        "epochs_completed": len(history),
        "training_history": history,
    }


def train_r2_probabilities(cache: pd.DataFrame, feat17: pd.DataFrame, bert_cos: pd.DataFrame,
                           output_dir: Path, training_config: R2TrainingConfig | None = None,
                           logger=None) -> tuple[pd.DataFrame, dict]:
    """Train and calibrate R2 without exposing GA-validation or test labels."""
    from sklearn.calibration import calibration_curve

    cfg = (training_config or R2TrainingConfig()).validate()
    set_deterministic_seed(cfg.seed, cfg.num_threads)
    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    probability_path = output_dir / "r2_probabilities.parquet"
    metadata_path = output_dir / "r2_training.json"
    history_path = output_dir / "training_history.csv"
    if any(path.exists() for path in (probability_path, model_dir, metadata_path, history_path)):
        raise FileExistsError(f"refusing to overwrite R2 training artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _preflight_r2_inputs(cache, feat17, bert_cos)
    scored = cache.loc[cache["decision_source"] != "AUTO_EXACT",
                       ["profile_id_a", "profile_id_b", "actual", "split", "experiment_split"]].copy()
    if scored.empty:
        raise ValueError("R2 training requires scored candidate pairs")
    feats = scored.merge(feat17, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one")
    if len(feats) != len(scored):
        raise ValueError("17-feature join changed row count")
    feats = feats.merge(bert_cos, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one")
    if len(feats) != len(scored):
        raise ValueError("BERT cosine join changed row count")
    feature_cols = FEATURE17_COLS + ["bert_cos"]
    if feats[feature_cols].isna().any().any():
        raise ValueError("R2 feature join is incomplete")
    _validate_feature_frame(feats, FEATURE17_COLS, "joined 17 features", 0.0, 1.0)
    _validate_feature_frame(feats, ["bert_cos"], "joined BERT cosine", -1.0, 1.0)

    train = feats.loc[feats["experiment_split"] == "model_train"]
    calibration = feats.loc[feats["experiment_split"] == "model_calibration"]
    ga_validation = feats.loc[feats["experiment_split"] == "ga_validation"]
    test = feats.loc[feats["experiment_split"] == "test"]
    if train.empty or calibration.empty or ga_validation.empty or test.empty:
        raise ValueError("nested split must contain model_train, model_calibration, ga_validation, and test rows")
    y_train = train["actual"].to_numpy(dtype=np.float32)
    y_calibration = calibration["actual"].to_numpy(dtype=np.float32)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_calibration)) != 2:
        raise ValueError("model_train and model_calibration each require both label classes")

    X_train, y_train = undersample_negatives(
        train[feature_cols].to_numpy(dtype=np.float32), y_train, cfg.train_neg_ratio, cfg.seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_calibration = scaler.transform(calibration[feature_cols].to_numpy(dtype=np.float32))
    model, train_meta = train_mlp(
        X_train, y_train, X_calibration, y_calibration, input_dim=len(feature_cols),
        training_config=cfg, logger=logger,
    )
    raw_calibration = predict_probs(model, X_calibration, cfg.eval_batch_size, torch.device("cpu"))
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calibration, y_calibration)
    calibrated_calibration = calibrator.predict(raw_calibration)
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_values = [f1_score(y_calibration, (calibrated_calibration >= threshold).astype(int), zero_division=0)
                 for threshold in thresholds]
    curve_true, curve_pred = calibration_curve(y_calibration, calibrated_calibration, n_bins=10, strategy="uniform")

    X_all = scaler.transform(feats[feature_cols].to_numpy(dtype=np.float32))
    probabilities = calibrator.predict(
        predict_probs(model, X_all, cfg.eval_batch_size, torch.device("cpu"))
    ).astype(np.float32)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("R2 probabilities are not finite values in [0, 1]")
    output = pd.DataFrame({
        "profile_id_a": feats["profile_id_a"].to_numpy(),
        "profile_id_b": feats["profile_id_b"].to_numpy(),
        "probability_r2": probabilities,
    })
    if len(output) != len(scored) or output.duplicated(["profile_id_a", "profile_id_b"]).any():
        raise ValueError("R2 probability artifact is incomplete or has duplicate pair keys")

    model_dir.mkdir()
    torch.save({"state_dict": model.state_dict(), "feature_dim": len(feature_cols)}, model_dir / "identity_mlp_r2.pt")
    import pickle
    for value, name in ((scaler, "scaler.pkl"), (calibrator, "isotonic_calibrator.pkl"), (feature_cols, "feature_cols.pkl")):
        with (model_dir / name).open("wb") as handle:
            pickle.dump(value, handle)
    output.to_parquet(probability_path, index=False)
    pd.DataFrame(train_meta["training_history"]).to_csv(history_path, index=False)
    split_counts = {role: int((feats["experiment_split"] == role).sum()) for role in
                    ("model_train", "model_calibration", "ga_validation", "test", "drop")}
    metadata = {
        **{key: value for key, value in train_meta.items() if key != "training_history"},
        "training_config": asdict(cfg),
        "feature_columns": feature_cols,
        "split_column": "experiment_split",
        "split_counts": split_counts,
        "label_access": {
            "model_fit": "model_train only",
            "early_stopping_and_calibration": "model_calibration only",
            "ga_optimisation": "ga_validation only (performed after this function)",
            "held_out_test": "features predicted only; labels never read by this function",
        },
        "calibration_diagnostics": {
            "best_threshold_diagnostic_only": float(thresholds[int(np.argmax(f1_values))]),
            "best_threshold_f1_diagnostic_only": float(max(f1_values)),
            "curve_fraction_positive": [float(value) for value in curve_true],
            "curve_mean_predicted_value": [float(value) for value in curve_pred],
        },
        "n_train_after_undersample": int(len(X_train)),
        "n_calibration": int(len(calibration)),
        "n_predicted_all_scored_pairs": int(len(output)),
        "environment": environment_snapshot(cfg.num_threads),
        "artifact_hashes": {
            "identity_mlp_r2.pt": _sha256_file(model_dir / "identity_mlp_r2.pt"),
            "scaler.pkl": _sha256_file(model_dir / "scaler.pkl"),
            "isotonic_calibrator.pkl": _sha256_file(model_dir / "isotonic_calibrator.pkl"),
            "feature_cols.pkl": _sha256_file(model_dir / "feature_cols.pkl"),
            "r2_probabilities.parquet": _sha256_file(probability_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if logger:
        logger(f"saved frozen R2 artifacts and {len(output):,} probabilities -> {output_dir}")
    return output, metadata

