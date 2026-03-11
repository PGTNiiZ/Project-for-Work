"""
สร้าง Synthetic Hard Negatives
คนละคน แต่มีชื่อคล้ายกัน เพื่อฝึกโมเดลให้แยกแยะเคสยากได้

กลยุทธ์:
1. Cross-pair negatives: คนที่ชื่อคล้ายกันใน folder ต่างกัน (จากคนละ identity)
2. ค้นหา profiles ที่ bigram similarity สูงแต่ไม่ใช่คนเดียวกัน
"""

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

SRC_COMBINED = Path("data/processed/training_matrices/combined_all_platforms_training.csv")
SRC_INDEX    = Path("data/processed/training_matrices/combined_profile_index.csv")
OUT_DIR      = Path("data/processed/training_matrices/improved_training")

TARGET_HARD_NEG = 5000  # จำนวน synthetic hard negatives ที่ต้องการ


def to_bigrams(s: str) -> set:
    s = s.strip().lower()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i: i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def lev_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return round(1 - dist / max(len(a), len(b)), 6)


def strip_numbers(s: str) -> str:
    return re.sub(r"\d+", "", s).strip()


def strip_separators(s: str) -> str:
    return re.sub(r"[_.\-@\s]+", "", s)


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 0.0
    union = ta | tb
    return round(len(ta & tb) / len(union), 6) if union else 0.0


def compute_features(pa: dict, pb: dict) -> dict:
    un_a = pa["userName"]
    un_b = pb["userName"]
    fn_a = pa["fullName"]
    fn_b = pb["fullName"]
    bg_ua, bg_ub = to_bigrams(un_a), to_bigrams(un_b)
    bg_fa, bg_fb = to_bigrams(fn_a), to_bigrams(fn_b)
    mg_a = to_bigrams(f"{un_a} {fn_a}")
    mg_b = to_bigrams(f"{un_b} {fn_b}")
    un_a_ns, un_b_ns = strip_numbers(un_a), strip_numbers(un_b)
    un_a_ss, un_b_ss = strip_separators(un_a), strip_separators(un_b)

    return {
        "profile_a_platform": pa["platform"],
        "profile_b_platform": pb["platform"],
        "profile_a_folder":   pa["folder"],
        "profile_b_folder":   pb["folder"],
        "profile_a_userName": un_a,
        "profile_a_fullName": fn_a,
        "profile_b_userName": un_b,
        "profile_b_fullName": fn_b,
        "username_exact_match":    int(un_a == un_b) if un_a and un_b else 0,
        "fullname_exact_match":    int(fn_a == fn_b) if fn_a and fn_b else 0,
        "username_bigram_jaccard": round(jaccard(bg_ua, bg_ub), 6),
        "fullname_bigram_jaccard": round(jaccard(bg_fa, bg_fb), 6),
        "merged_bigram_jaccard":   round(jaccard(mg_a, mg_b), 6),
        "same_platform":           int(pa["platform"] == pb["platform"]),
        "username_lev_ratio":      lev_ratio(un_a, un_b),
        "fullname_lev_ratio":      lev_ratio(fn_a, fn_b),
        "fullname_token_overlap":  token_overlap(fn_a, fn_b),
        "username_token_overlap":  token_overlap(un_a, un_b),
        "username_nonum_lev_ratio": lev_ratio(un_a_ns, un_b_ns),
        "username_nosep_lev_ratio": lev_ratio(un_a_ss, un_b_ss),
        "username_nosep_exact":    int(un_a_ss == un_b_ss) if un_a_ss and un_b_ss else 0,
        "username_prefix_ratio":   round(
            sum(1 for a, b in zip(un_a, un_b) if a == b) / max(len(un_a), len(un_b), 1), 6
        ),
        "fullname_prefix_ratio":   round(
            sum(1 for a, b in zip(fn_a, fn_b) if a == b) / max(len(fn_a), len(fn_b), 1), 6
        ),
        "label": 0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("โหลด profile index...")
    idx_df = pd.read_csv(SRC_INDEX)
    idx_df["userName"] = idx_df["userName"].fillna("").str.lower()
    idx_df["fullName"] = idx_df["fullName"].fillna("").str.lower()

    profiles = idx_df.to_dict("records")
    n = len(profiles)
    print(f"  profiles: {n:,}")

    # สร้าง set ของ positive pairs (คนเดียวกัน = folder เดียวกันใน pair_source เดียวกัน)
    print("สร้าง positive pair set...")
    pos_pairs = set()
    folder_groups: dict = {}
    for p in profiles:
        key = f"{p['pair_source']}::{p['folder']}"
        folder_groups.setdefault(key, []).append(p["idx"])
    for indices in folder_groups.values():
        for i in indices:
            for j in indices:
                pos_pairs.add((i, j))

    print(f"  positive pairs: {len(pos_pairs):,}")

    # หา candidate hard negatives โดยเรียงตาม fullname similarity
    print(f"\nค้นหา hard negative candidates (target={TARGET_HARD_NEG:,})...")
    np.random.seed(42)

    hard_rows = []
    checked = set()
    attempts = 0
    max_attempts = TARGET_HARD_NEG * 50

    while len(hard_rows) < TARGET_HARD_NEG and attempts < max_attempts:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i >= j:
            attempts += 1
            continue
        if (i, j) in checked:
            attempts += 1
            continue
        if (i, j) in pos_pairs or (j, i) in pos_pairs:
            attempts += 1
            continue

        checked.add((i, j))
        pa, pb = profiles[i], profiles[j]

        fn_a, fn_b = pa["fullName"], pb["fullName"]
        if not fn_a or not fn_b:
            attempts += 1
            continue

        # กรองเฉพาะที่ชื่อคล้ายกัน (fullname similarity สูง)
        sim = lev_ratio(fn_a, fn_b)
        if sim < 0.5:
            attempts += 1
            continue

        row = compute_features(pa, pb)
        hard_rows.append(row)
        attempts += 1

        if len(hard_rows) % 500 == 0:
            print(f"  พบแล้ว: {len(hard_rows):,} / {TARGET_HARD_NEG:,}")

    print(f"\nSynthetic hard negatives: {len(hard_rows):,}")

    if not hard_rows:
        print("ไม่พบ hard negatives — ลองลด threshold fullname_lev_ratio")
        return

    hard_df = pd.DataFrame(hard_rows)

    print("\nStats:")
    print(f"  fullname_lev_ratio mean: {hard_df['fullname_lev_ratio'].mean():.4f}")
    print(f"  merged_bigram_jaccard mean: {hard_df['merged_bigram_jaccard'].mean():.4f}")

    # เพิ่มเข้าไปใน train set ของ improved_training
    train_path = OUT_DIR / "train.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        print(f"\nเพิ่ม {len(hard_df):,} hard negatives เข้า train set (เดิม {len(train_df):,})")
        train_df = pd.concat([train_df, hard_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        train_df.to_csv(train_path, index=False)
        print(f"Train ใหม่: {len(train_df):,}  pos={train_df['label'].sum():,}")

    # บันทึก hard negatives แยก
    out = OUT_DIR / "synthetic_hard_negatives.csv"
    hard_df.to_csv(out, index=False)
    print(f"บันทึกแล้ว: {out}")


if __name__ == "__main__":
    main()
