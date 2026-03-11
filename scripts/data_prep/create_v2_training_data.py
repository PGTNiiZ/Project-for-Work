"""
create_v2_training_data.py

ต่อยอดจาก create_improved_training_data.py โดยเพิ่ม features:
  1. username_clean_*   — ลบ @ และ symbols ออกก่อนเปรียบเทียบ
  2. fullname_clean_*   — ลบ emoji/symbols ออกก่อนเปรียบเทียบ
  3. firstname_match    — เปรียบเทียบเฉพาะคำแรก
  4. lastname_match     — เปรียบเทียบเฉพาะคำสุดท้าย
  5. name_token_set_ratio — token set overlap (ไม่สนลำดับ)
  6. fullname_rarity    — ชื่อนี้ปรากฏกี่ครั้งใน dataset (ชื่อสามัญ → ลด confidence)

บันทึกไปที่: data/processed/training_matrices/v2_training/
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import train_test_split

SRC = Path("data/processed/training_matrices/combined_all_platforms_training.csv")
OUT_DIR = Path("data/processed/training_matrices/v2_training")
HARD_NEG_THRESHOLD = 0.3


# ─── Clean helpers ────────────────────────────────────────────────────────────

def remove_emoji(s: str) -> str:
    """ลบ emoji และ symbols พิเศษ เหลือเฉพาะตัวอักษร ตัวเลข และช่องว่าง"""
    s = unicodedata.normalize("NFC", s)
    # ลบ emoji range
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    # ลบ @ นำหน้า
    s = re.sub(r"^\s*@+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_username(s: str) -> str:
    """ลบ @ นำหน้า, ลบ . _ - แล้วทำให้เป็น lowercase"""
    s = re.sub(r"^\s*@+", "", s)               # ลบ @ นำหน้า
    s = re.sub(r"[^a-z0-9]", "", s.lower())    # เหลือแค่ alphanumeric
    return s.strip()


def clean_fullname(s: str) -> str:
    """ลบ emoji/symbols เหลือแค่ unicode letters, digits, spaces"""
    s = remove_emoji(s)
    # ลบ non-letter/digit/space ที่เหลือ
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip().lower()


# ─── Feature helpers ──────────────────────────────────────────────────────────

def lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return round(1 - dist / max(len(a), len(b)), 6)


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 0.0
    union = ta | tb
    return round(len(ta & tb) / len(union), 6) if union else 0.0


def token_set_ratio(a: str, b: str) -> float:
    """Jaccard ของ token set (ไม่สนลำดับคำ)"""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 6) if (ta | tb) else 0.0


def strip_numbers(s: str) -> str:
    return re.sub(r"\d+", "", s).strip()


def strip_separators(s: str) -> str:
    return re.sub(r"[_.\-@\s]+", "", s)


def prefix_match_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    length = sum(1 for ca, cb in zip(a, b) if ca == cb)
    # นับเฉพาะ prefix ต่อเนื่อง
    plen = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            plen += 1
        else:
            break
    return round(plen / max(len(a), len(b)), 6)


def to_bigrams(s: str) -> set:
    s = s.strip().lower()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def add_features(df: pd.DataFrame, name_freq: dict) -> pd.DataFrame:
    df = df.copy()

    # raw (lowercase)
    un_a = df["profile_a_userName"].fillna("").str.lower()
    un_b = df["profile_b_userName"].fillna("").str.lower()
    fn_a = df["profile_a_fullName"].fillna("").str.lower()
    fn_b = df["profile_b_fullName"].fillna("").str.lower()

    # ── เดิม (คง compatibility) ──────────────────────────────────────────────
    df["username_lev_ratio"]       = [lev_ratio(a, b) for a, b in zip(un_a, un_b)]
    df["fullname_lev_ratio"]       = [lev_ratio(a, b) for a, b in zip(fn_a, fn_b)]
    df["fullname_token_overlap"]   = [token_overlap(a, b) for a, b in zip(fn_a, fn_b)]
    df["username_token_overlap"]   = [token_overlap(a, b) for a, b in zip(un_a, un_b)]

    un_a_ns = un_a.apply(strip_numbers)
    un_b_ns = un_b.apply(strip_numbers)
    df["username_nonum_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a_ns, un_b_ns)]

    un_a_ss = un_a.apply(strip_separators)
    un_b_ss = un_b.apply(strip_separators)
    df["username_nosep_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a_ss, un_b_ss)]
    df["username_nosep_exact"]     = (un_a_ss == un_b_ss).astype(int)

    df["username_prefix_ratio"]    = [prefix_match_ratio(a, b) for a, b in zip(un_a, un_b)]
    df["fullname_prefix_ratio"]    = [prefix_match_ratio(a, b) for a, b in zip(fn_a, fn_b)]

    # ── ใหม่ 1: clean username (ลบ @ และ non-alphanumeric) ─────────────────
    un_a_cl = un_a.apply(clean_username)
    un_b_cl = un_b.apply(clean_username)
    df["username_clean_exact"]     = (un_a_cl == un_b_cl).astype(int)
    df["username_clean_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(un_a_cl, un_b_cl)]
    df["username_clean_bigram"]    = [
        round(jaccard(to_bigrams(a), to_bigrams(b)), 6)
        for a, b in zip(un_a_cl, un_b_cl)
    ]

    # ── ใหม่ 2: clean fullname (ลบ emoji/symbols) ────────────────────────────
    fn_a_cl = fn_a.apply(clean_fullname)
    fn_b_cl = fn_b.apply(clean_fullname)
    df["fullname_clean_exact"]     = (fn_a_cl == fn_b_cl).astype(int)
    df["fullname_clean_lev_ratio"] = [lev_ratio(a, b) for a, b in zip(fn_a_cl, fn_b_cl)]
    df["fullname_clean_bigram"]    = [
        round(jaccard(to_bigrams(a), to_bigrams(b)), 6)
        for a, b in zip(fn_a_cl, fn_b_cl)
    ]
    df["fullname_clean_token_set"] = [token_set_ratio(a, b) for a, b in zip(fn_a_cl, fn_b_cl)]

    # ── ใหม่ 3: firstname / lastname match ───────────────────────────────────
    def first_token(s: str) -> str:
        t = s.split()
        return t[0] if t else ""

    def last_token(s: str) -> str:
        t = s.split()
        return t[-1] if t else ""

    fn_a_first = fn_a_cl.apply(first_token)
    fn_b_first = fn_b_cl.apply(first_token)
    fn_a_last  = fn_a_cl.apply(last_token)
    fn_b_last  = fn_b_cl.apply(last_token)

    df["firstname_exact_match"]    = ((fn_a_first == fn_b_first) & (fn_a_first != "")).astype(int)
    df["lastname_exact_match"]     = ((fn_a_last == fn_b_last) & (fn_a_last != "")).astype(int)
    df["firstname_lev_ratio"]      = [lev_ratio(a, b) for a, b in zip(fn_a_first, fn_b_first)]
    df["lastname_lev_ratio"]       = [lev_ratio(a, b) for a, b in zip(fn_a_last, fn_b_last)]

    # ── ใหม่ 4: fullname rarity (ชื่อที่พบบ่อย → ลด confidence) ─────────────
    # log1p เพื่อ scale ไม่ให้ค่าสูงเกินไป
    df["fullname_a_rarity"] = fn_a_cl.map(name_freq).fillna(1).apply(lambda x: round(np.log1p(x), 4))
    df["fullname_b_rarity"] = fn_b_cl.map(name_freq).fillna(1).apply(lambda x: round(np.log1p(x), 4))
    df["fullname_max_rarity"] = df[["fullname_a_rarity", "fullname_b_rarity"]].max(axis=1)

    return df


# ─── Hard Negative Mining ─────────────────────────────────────────────────────

def mine_hard_negatives(df: pd.DataFrame, n_target: int) -> pd.DataFrame:
    negatives = df[df["label"] == 0].copy()
    hard = negatives[negatives["merged_bigram_jaccard"] >= HARD_NEG_THRESHOLD]
    easy = negatives[negatives["merged_bigram_jaccard"] < HARD_NEG_THRESHOLD]

    print(f"  Hard negatives (bigram >= {HARD_NEG_THRESHOLD}): {len(hard):,}")
    print(f"  Easy negatives: {len(easy):,}")

    n_hard_take = min(len(hard), n_target // 2)
    n_easy_take = min(len(easy), n_target - n_hard_take)

    return pd.concat([
        hard.sample(n=n_hard_take, random_state=42),
        easy.sample(n=n_easy_take, random_state=42),
    ])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("โหลด combined training data...")
    df = pd.read_csv(SRC)
    print(f"  ทั้งหมด: {len(df):,}  (pos={( df['label']==1).sum():,}, neg={(df['label']==0).sum():,})")

    positives = df[df["label"] == 1].copy()
    n_pos = len(positives)

    print(f"\nHard Negative Mining (target {n_pos * 5:,})...")
    negatives_mined = mine_hard_negatives(df, n_pos * 5)
    print(f"  ได้ negatives: {len(negatives_mined):,}")

    combined = pd.concat([positives, negatives_mined]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nDataset: {len(combined):,}  (pos={n_pos:,}, neg={len(negatives_mined):,})")

    # สร้าง name_freq จาก fullname ทั้งหมดใน dataset (ก่อน clean)
    print("\nคำนวณ fullname rarity...")
    all_names = pd.concat([
        combined["profile_a_fullName"].fillna("").str.lower().apply(clean_fullname),
        combined["profile_b_fullName"].fillna("").str.lower().apply(clean_fullname),
    ])
    name_freq = all_names[all_names != ""].value_counts().to_dict()
    print(f"  unique names: {len(name_freq):,}")

    print("\nสร้าง features...")
    combined = add_features(combined, name_freq)

    # แบ่ง train/val/test (70/15/15)
    train_df, temp_df = train_test_split(combined, test_size=0.30, stratify=combined["label"], random_state=42)
    val_df, test_df   = train_test_split(temp_df,  test_size=0.50, stratify=temp_df["label"],  random_state=42)

    print(f"\nSplit:")
    print(f"  Train : {len(train_df):,}  pos={train_df['label'].sum():,}")
    print(f"  Val   : {len(val_df):,}   pos={val_df['label'].sum():,}")
    print(f"  Test  : {len(test_df):,}   pos={test_df['label'].sum():,}")

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv",   index=False)
    test_df.to_csv(OUT_DIR / "test.csv",  index=False)
    combined.to_csv(OUT_DIR / "all.csv",  index=False)
    print(f"\nบันทึกแล้วใน {OUT_DIR}/")

    # สรุป feature stats
    ALL_FEATURES = [
        "username_exact_match", "fullname_exact_match",
        "username_bigram_jaccard", "fullname_bigram_jaccard", "merged_bigram_jaccard",
        "username_lev_ratio", "fullname_lev_ratio",
        "fullname_token_overlap", "username_nonum_lev_ratio",
        "username_nosep_lev_ratio", "username_nosep_exact",
        "username_prefix_ratio", "fullname_prefix_ratio",
        # ใหม่
        "username_clean_exact", "username_clean_lev_ratio", "username_clean_bigram",
        "fullname_clean_exact", "fullname_clean_lev_ratio", "fullname_clean_bigram",
        "fullname_clean_token_set",
        "firstname_exact_match", "lastname_exact_match",
        "firstname_lev_ratio", "lastname_lev_ratio",
        "fullname_max_rarity",
    ]
    print("\n=== Feature stats (pos vs neg, diff) ===")
    rows = []
    for col in ALL_FEATURES:
        if col not in combined.columns:
            continue
        pos_m = combined.loc[combined["label"] == 1, col].mean()
        neg_m = combined.loc[combined["label"] == 0, col].mean()
        rows.append({"feature": col, "pos_mean": round(pos_m, 4), "neg_mean": round(neg_m, 4), "diff": round(pos_m - neg_m, 4)})
    stats = pd.DataFrame(rows).sort_values("diff", ascending=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
