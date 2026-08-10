"""
Experiment R4/R5: Bloom filter privacy/accuracy tradeoff (EXPERIMENT_PLAN.md ข้อ 7)

R4 = repr:+Bloom filter, decision: threshold มือเดิม (0.98/0.95)
R5 = repr:+Bloom filter, decision: GA (จาก exp_r1_ga_redecision.py)

Bloom เข้ารหัสเฉพาะ userName/fullName (ตาม prompt) แล้วแทนที่ "name-similarity feature"
(jaro/lev/token-sort ของ username/fullname + name_sim รวม 7 คอลัมน์ใน pair_features17.parquet)
ด้วย 3 คอลัมน์ dice-on-bloom (dice_username, dice_fullname, name_sim_bloom=max ของสองตัว)
feature ที่เหลือ (bio/location/platform/length ฯลฯ) ใช้ pair_features17.parquet เดิมตรงๆ
(non-name features "ไม่แตะ" ตามที่ prompt สั่ง) -> รวม 13 features ต่อค่า L

sweep L ใน {2000, 1000, 500, 250}, k=10 hash/gram, salt คงที่ (ทุกฝั่งใช้ salt เดียวกัน
เหมือนสถานการณ์ linkage ที่ตกลง salt ร่วมกันไว้ล่วงหน้า -- ไม่ใช่ salt ต่างฝั่งซึ่งจะจับคู่ไม่ได้เลย)

train_neg_ratio=3.0 (มาจาก stage10_13 เดิม) ทำหน้าที่ subsample negative สำหรับ train อยู่แล้ว
ตาม fallback ที่ prompt อนุญาต (ข้อ 3 ของ R4/R5) -- ไม่แตะ eval set

Checkpointing: ทุกไฟล์ intermediate เช็คก่อนสร้างซ้ำ
"""
from __future__ import annotations

import hashlib
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from exp_lib import EXP, MATCH, NO_MATCH, REVIEW, PROFILES, build_cache, evaluate, save_json, split_constants
import exp_r1_ga_redecision as r1
from exp_r2_bert_feature import (
    FEAT17_PATH, IdentityMLP, FocalLoss, log as _log_base, set_seed,
    undersample_negatives, make_loader, predict_probs, train_mlp,
    decide_manual, TRAIN_NEG_RATIO, EVAL_BATCH_SIZE, SEED,
)

L_VALUES = [2000, 1000, 500, 250]
K_HASHES = 10
SALT = "identity-resolution-r4-bloom-salt-fixed"

BLOOM_DIR = EXP / "bloom"
BLOOM_DIR.mkdir(exist_ok=True)
LOG_PATH = EXP / "r4_run.log"
RESULTS_PATH = EXP / "r4_privacy_tradeoff.json"
PLOT_PATH = EXP / "r4_privacy_tradeoff.png"

NON_NAME_COLS = [
    "bio_jaccard", "bio_len_ratio", "both_have_bio",
    "location_jaro", "location_exact", "both_have_location",
    "same_platform", "username_len_diff", "fullname_wordcount_diff",
    "username_prefix3_match",
]

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def bigrams(s: str) -> list[str]:
    padded = f" {s} "
    return [padded[i:i + 2] for i in range(len(padded) - 1)] if len(padded) >= 2 else []


def bloom_packed(s: str, L: int, k: int, salt: str) -> tuple[np.ndarray, int]:
    """คืน packed bits (uint8, ceil(L/8) bytes) + popcount ของ string s"""
    bits = np.zeros(L, dtype=bool)
    for g in bigrams(s):
        for j in range(k):
            h = hashlib.md5(f"{salt}|{j}|{g}".encode("utf-8")).hexdigest()
            pos = int(h, 16) % L
            bits[pos] = True
    packed = np.packbits(bits)
    return packed, int(bits.sum())


def build_profile_bloom(L: int) -> dict:
    cache_path = BLOOM_DIR / f"profiles_L{L}.npz"
    if cache_path.exists():
        log(f"L={L}: bloom profile cache found -> {cache_path}")
        with np.load(cache_path) as z:
            return dict(ids=z["ids"], username_packed=z["username_packed"], username_pop=z["username_pop"],
                       fullname_packed=z["fullname_packed"], fullname_pop=z["fullname_pop"])

    prof = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    prof["profile_row_id"] = pd.to_numeric(prof["profile_row_id"], errors="coerce").astype("Int64")
    prof = prof[prof["profile_row_id"].notna()].copy()

    def clean(s):
        s = s.astype(str).str.lower().str.strip()
        return s.where(~s.isin(["", "nan"]), "").values

    uname = clean(prof["userName"])
    fname = clean(prof["fullName"])
    ids = prof["profile_row_id"].astype(int).values
    n = len(ids)
    nbytes = (L + 7) // 8

    username_packed = np.zeros((n, nbytes), dtype=np.uint8)
    fullname_packed = np.zeros((n, nbytes), dtype=np.uint8)
    username_pop = np.zeros(n, dtype=np.int32)
    fullname_pop = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    for i in range(n):
        p, c = bloom_packed(uname[i], L, K_HASHES, SALT)
        username_packed[i], username_pop[i] = p, c
        p, c = bloom_packed(fname[i], L, K_HASHES, SALT)
        fullname_packed[i], fullname_pop[i] = p, c
        if i and i % 10000 == 0:
            log(f"  L={L} bloom-encode {i:,}/{n:,} ({time.time()-t0:.0f}s)")
    log(f"L={L}: bloom-encoded {n:,} profiles in {time.time()-t0:.0f}s")

    np.savez(cache_path, ids=ids, username_packed=username_packed, username_pop=username_pop,
             fullname_packed=fullname_packed, fullname_pop=fullname_pop)
    return dict(ids=ids, username_packed=username_packed, username_pop=username_pop,
               fullname_packed=fullname_packed, fullname_pop=fullname_pop)


def dice_chunked(packed_a, packed_b, pop_a, pop_b, chunk=200_000):
    n = len(pop_a)
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        inter = POPCOUNT[np.bitwise_and(packed_a[s:e], packed_b[s:e])].sum(axis=1).astype(np.float32)
        denom = (pop_a[s:e] + pop_b[s:e]).astype(np.float32)
        out[s:e] = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 0.0)
    return out


def compute_pair_bloom_features(L: int, cache: pd.DataFrame) -> pd.DataFrame:
    out_path = BLOOM_DIR / f"pairs_L{L}.parquet"
    if out_path.exists():
        log(f"L={L}: pair bloom-dice cache found -> {out_path}")
        return pd.read_parquet(out_path)

    prof_bloom = build_profile_bloom(L)
    id_to_row = {int(pid): i for i, pid in enumerate(prof_bloom["ids"])}
    scored = cache[cache.decision_source != "AUTO_EXACT"]
    a_idx = scored["profile_id_a"].map(id_to_row)
    b_idx = scored["profile_id_b"].map(id_to_row)
    assert a_idx.notna().all() and b_idx.notna().all()
    a_idx, b_idx = a_idx.astype(np.int64).values, b_idx.astype(np.int64).values

    dice_username = dice_chunked(
        prof_bloom["username_packed"][a_idx], prof_bloom["username_packed"][b_idx],
        prof_bloom["username_pop"][a_idx], prof_bloom["username_pop"][b_idx])
    dice_fullname = dice_chunked(
        prof_bloom["fullname_packed"][a_idx], prof_bloom["fullname_packed"][b_idx],
        prof_bloom["fullname_pop"][a_idx], prof_bloom["fullname_pop"][b_idx])
    name_sim_bloom = np.maximum(dice_username, dice_fullname)

    out = pd.DataFrame({
        "profile_id_a": scored["profile_id_a"].values,
        "profile_id_b": scored["profile_id_b"].values,
        "dice_username": dice_username,
        "dice_fullname": dice_fullname,
        "name_sim_bloom": name_sim_bloom,
    })
    out.to_parquet(out_path, index=False)
    log(f"L={L}: saved pair bloom-dice -> {out_path}")
    return out


def build_probabilities_for_L(L: int, cache: pd.DataFrame, feat17: pd.DataFrame) -> pd.DataFrame:
    prob_path = BLOOM_DIR / f"probability_r4_L{L}.parquet"
    if prob_path.exists():
        log(f"L={L}: probability cache found -> {prob_path}")
        return pd.read_parquet(prob_path)

    bloom_feat = compute_pair_bloom_features(L, cache)
    scored = cache[cache.decision_source != "AUTO_EXACT"][["profile_id_a", "profile_id_b", "actual", "split"]]
    feats = scored.merge(feat17[["profile_id_a", "profile_id_b"] + NON_NAME_COLS],
                         on=["profile_id_a", "profile_id_b"], how="left") \
                   .merge(bloom_feat, on=["profile_id_a", "profile_id_b"], how="left")
    assert len(feats) == len(scored)
    feature_cols = ["dice_username", "dice_fullname", "name_sim_bloom"] + NON_NAME_COLS
    assert feats[feature_cols].isna().sum().sum() == 0

    val_mask = feats["split"].values == "val"
    X_val = feats.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
    y_val = feats.loc[val_mask, "actual"].to_numpy(dtype=np.float32)

    idx = np.arange(len(X_val))
    idx_train, idx_holdout = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y_val)

    scaler = StandardScaler()
    Xtr, ytr = undersample_negatives(X_val[idx_train], y_val[idx_train], TRAIN_NEG_RATIO, SEED)
    Xtr = scaler.fit_transform(Xtr)
    Xho = scaler.transform(X_val[idx_holdout])
    yho = y_val[idx_holdout]
    log(f"L={L}: train_inner={len(Xtr):,} (pos={int(ytr.sum()):,}) holdout={len(Xho):,} (pos={int(yho.sum()):,})")

    set_seed(SEED)
    model, meta = train_mlp(Xtr, ytr, Xho, yho, input_dim=len(feature_cols))

    device = torch.device("cpu")
    holdout_raw = predict_probs(model, Xho, EVAL_BATCH_SIZE, device)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(holdout_raw, yho)

    X_all = scaler.transform(feats[feature_cols].to_numpy(dtype=np.float32))
    probs_raw_all = predict_probs(model, X_all, EVAL_BATCH_SIZE, device)
    probs_cal_all = calibrator.predict(probs_raw_all).astype(np.float32)

    model_dir = BLOOM_DIR / f"model_L{L}"
    model_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_dim": len(feature_cols)}, model_dir / "identity_mlp.pt")
    with (model_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)
    with (model_dir / "isotonic_calibrator.pkl").open("wb") as fh:
        pickle.dump(calibrator, fh)
    save_json(model_dir / "train_meta.json", {**meta, "L": L, "n_train": int(len(Xtr)), "n_holdout": int(len(Xho))})

    out = pd.DataFrame({
        "profile_id_a": feats["profile_id_a"].values,
        "profile_id_b": feats["profile_id_b"].values,
        "probability": probs_cal_all,
        "name_sim_bloom": feats["name_sim_bloom"].values,
    })
    out.to_parquet(prob_path, index=False)
    log(f"L={L}: saved probability -> {prob_path}")
    return out


def eval_manual(cache, consts, prob_df) -> dict:
    scored = cache[cache.decision_source != "AUTO_EXACT"].merge(prob_df, on=["profile_id_a", "profile_id_b"], how="left")
    out = {}
    for sp in ["test", "full"]:
        sub = scored if sp == "full" else scored[scored.split == sp]
        code = decide_manual(sub["probability"].values)
        actual = sub["actual"].values.astype(np.int8)
        out[sp] = evaluate(code, actual, consts[sp])
    return out


def eval_ga(cache, consts, prob_df) -> tuple[dict, dict]:
    # หมายเหตุ: ใช้ name_sim_bloom (ไม่ใช่ name_sim plaintext จาก cache) เป็นสัญญาณ promote/demote
    # ของ GA ด้วย -- ไม่งั้นจะแอบใช้ plaintext name ผ่านทาง GA ซึ่งขัดกับโจทย์ privacy ของ R4/R5
    scored = cache[cache.decision_source != "AUTO_EXACT"].merge(prob_df, on=["profile_id_a", "profile_id_b"], how="left")
    val = scored[scored.split == "val"]
    best, history = r1.run_ga(val["probability"].values, val["name_sim_bloom"].values,
                              val["actual"].values.astype(np.int8), 0)
    genome = dict(t_m=float(best[0]), t_r=float(best[1]), c_promote=float(best[2]), c_demote=float(best[3]))
    out = {}
    for sp in ["test", "full"]:
        sub = scored if sp == "full" else scored[scored.split == sp]
        code = r1.decide_code(sub["probability"].values, sub["name_sim_bloom"].values, best)
        actual = sub["actual"].values.astype(np.int8)
        out[sp] = evaluate(code, actual, consts[sp])
    return out, genome


def main():
    LOG_PATH.write_text("", encoding="utf-8")
    log("=== R4/R5 bloom privacy sweep start ===")

    cache = build_cache()
    consts = split_constants(cache)
    assert consts["full"]["total_pos"] == 29247

    if not FEAT17_PATH.exists():
        raise RuntimeError(f"{FEAT17_PATH} not found -- run exp_r2_bert_feature.py first (rule: reuse, don't recompute)")
    feat17 = pd.read_parquet(FEAT17_PATH)

    results = {"L_values": L_VALUES, "k_hashes": K_HASHES, "per_L": {}}
    for L in L_VALUES:
        log(f"--- L={L} ---")
        prob_df = build_probabilities_for_L(L, cache, feat17)
        r4 = eval_manual(cache, consts, prob_df)
        r5, genome = eval_ga(cache, consts, prob_df)
        results["per_L"][str(L)] = {"R4_manual": r4, "R5_ga": r5, "R5_best_genome": genome}
        log(f"L={L} R4 test={r4['test']} R5 test={r5['test']}")
        save_json(RESULTS_PATH, results)

    # plot F1 (test) vs L
    Ls = sorted(L_VALUES)
    f1_r4 = [results["per_L"][str(L)]["R4_manual"]["test"]["F1"] for L in Ls]
    f1_r5 = [results["per_L"][str(L)]["R5_ga"]["test"]["F1"] for L in Ls]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ls, f1_r4, marker="o", label="R4 (Bloom + threshold มือ)", color="#4C72B0")
    ax.plot(Ls, f1_r5, marker="o", label="R5 (Bloom + GA)", color="#DD8452")
    ax.set_xlabel("Bloom filter length L (bit)")
    ax.set_ylabel("F1 (test)")
    ax.set_title("Privacy/accuracy tradeoff: F1 vs Bloom filter length L")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    log(f"saved plot -> {PLOT_PATH}")

    save_json(RESULTS_PATH, results)
    log(f"saved -> {RESULTS_PATH}")
    log("=== R4/R5 done ===")


if __name__ == "__main__":
    main()
