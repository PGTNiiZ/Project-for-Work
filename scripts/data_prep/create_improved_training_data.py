"""
สร้าง training data ที่ดีขึ้นจาก combined_all_platforms_training.csv
- เพิ่ม features: Levenshtein ratio, token overlap, number-stripped similarity, prefix match
- Hard Negative Mining: เก็บ negative pairs ที่ bigram สูง (ยากขึ้น)
- บันทึก train/val/test พร้อม usernames สำหรับ error analysis
"""

import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import train_test_split

SRC = Path("data/processed/training_matrices/combined_all_platforms_training.csv")
OUT_DIR = Path("data/processed/training_matrices/improved_training")
HARD_NEG_THRESHOLD = 0.3  # merged_bigram_jaccard >= threshold → hard negative


# ─── Feature helpers ──────────────────────────────────────────────────────────

def lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return round(1 - dist / max_len, 6)


def token_overlap(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 0.0
    union = ta | tb
    return round(len(ta & tb) / len(union), 6) if union else 0.0


def strip_numbers(s: str) -> str:
    return re.sub(r"\d+", "", s).strip()


def strip_separators(s: str) -> str:
    return re.sub(r"[_.\-@\s]+", "", s)


def prefix_match_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    length = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            length += 1
        else:
            break
    return round(length / max(len(a), len(b)), 6)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    un_a = df["profile_a_userName"].fillna("").str.lower()
    un_b = df["profile_b_userName"].fillna("").str.lower()
    fn_a = df["profile_a_fullName"].fillna("").str.lower()
    fn_b = df["profile_b_fullName"].fillna("").str.lower()

    # Levenshtein ratio
    df["username_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a, un_b)]
    df["fullname_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(fn_a, fn_b)]

    # Token overlap (ชื่อเต็มแบ่งเป็น token)
    df["fullname_token_overlap"] = [token_overlap(a, b) for a, b in zip(fn_a, fn_b)]
    df["username_token_overlap"] = [token_overlap(a, b) for a, b in zip(un_a, un_b)]

    # Number-stripped username similarity
    un_a_ns = un_a.apply(strip_numbers)
    un_b_ns = un_b.apply(strip_numbers)
    df["username_nonum_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a_ns, un_b_ns)]

    # Separator-stripped username similarity  (john_smith → johnsmith)
    un_a_ss = un_a.apply(strip_separators)
    un_b_ss = un_b.apply(strip_separators)
    df["username_nosep_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a_ss, un_b_ss)]
    df["username_nosep_exact"] = (un_a_ss == un_b_ss).astype(int)

    # Prefix match ratio
    df["username_prefix_ratio"] = [prefix_match_ratio(a, b) for a, b in zip(un_a, un_b)]
    df["fullname_prefix_ratio"] = [prefix_match_ratio(a, b) for a, b in zip(fn_a, fn_b)]

    return df


# ─── Hard Negative Mining ─────────────────────────────────────────────────────

def mine_hard_negatives(df: pd.DataFrame, n_hard: int) -> pd.DataFrame:
    """เลือก negative pairs ที่ bigram similarity สูง (คนละคนแต่ชื่อคล้าย)"""
    negatives = df[df["label"] == 0].copy()
    hard = negatives[negatives["merged_bigram_jaccard"] >= HARD_NEG_THRESHOLD]
    easy = negatives[negatives["merged_bigram_jaccard"] < HARD_NEG_THRESHOLD]

    print(f"  Hard negatives available (bigram >= {HARD_NEG_THRESHOLD}): {len(hard):,}")
    print(f"  Easy negatives available: {len(easy):,}")

    # สัดส่วน: hard 50%, easy 50%
    n_hard_take = min(len(hard), n_hard // 2)
    n_easy_take = min(len(easy), n_hard - n_hard_take)

    hard_sample = hard.sample(n=n_hard_take, random_state=42)
    easy_sample = easy.sample(n=n_easy_take, random_state=42)
    return pd.concat([hard_sample, easy_sample])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("โหลด combined training data...")
    df = pd.read_csv(SRC)
    print(f"  ทั้งหมด: {len(df):,} rows  (pos={( df['label']==1).sum():,}, neg={(df['label']==0).sum():,})")

    positives = df[df["label"] == 1].copy()
    n_pos = len(positives)

    # Hard negative mining: ดึง negatives ที่ยากขึ้น (5x จำนวน pos เหมือนเดิม)
    print(f"\nHard Negative Mining (target {n_pos * 5:,} negatives)...")
    negatives_mined = mine_hard_negatives(df, n_pos * 5)
    print(f"  ได้ negatives: {len(negatives_mined):,}")

    combined = pd.concat([positives, negatives_mined]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nDataset ใหม่: {len(combined):,} rows  (pos={n_pos:,}, neg={len(negatives_mined):,})")

    # เพิ่ม features
    print("\nสร้าง features เพิ่ม...")
    combined = add_features(combined)

    # แบ่ง train/val/test (70/15/15) แบบ stratified
    train_df, temp_df = train_test_split(combined, test_size=0.30, stratify=combined["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    print(f"\nSplit:")
    print(f"  Train : {len(train_df):,}  pos={train_df['label'].sum():,}")
    print(f"  Val   : {len(val_df):,}   pos={val_df['label'].sum():,}")
    print(f"  Test  : {len(test_df):,}   pos={test_df['label'].sum():,}")

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    combined.to_csv(OUT_DIR / "all.csv", index=False)

    print(f"\nบันทึกแล้วใน {OUT_DIR}/")

    # สรุป features ใหม่
    new_features = [
        "username_lev_ratio", "fullname_lev_ratio",
        "fullname_token_overlap", "username_token_overlap",
        "username_nonum_lev_ratio", "username_nosep_lev_ratio",
        "username_nosep_exact", "username_prefix_ratio", "fullname_prefix_ratio",
    ]
    print("\n=== Feature stats (pos vs neg) ===")
    all_features = [
        "username_exact_match", "fullname_exact_match",
        "username_bigram_jaccard", "fullname_bigram_jaccard", "merged_bigram_jaccard",
    ] + new_features
    for col in all_features:
        if col not in combined.columns:
            continue
        pos_m = combined.loc[combined["label"] == 1, col].mean()
        neg_m = combined.loc[combined["label"] == 0, col].mean()
        print(f"  {col:<35} pos={pos_m:.4f}  neg={neg_m:.4f}  diff={pos_m-neg_m:+.4f}")


if __name__ == "__main__":
    main()
