"""
สร้าง training data matrix รวมทุก platform เข้าด้วยกัน

รวม profiles จาก Google+, Instagram, Twitter ทั้งหมดจาก 2.profile.data
เป็น matrix เดียว n×n:
- แถว = profile จากทุก platform
- คอลัมน์ = profile จากทุก platform
- label: 1 = คนเดียวกัน (อยู่ folder เดียวกัน), 0 = คนละคน
- features: username match, fullname match, bigram Jaccard similarity
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/data/Dataset-LinkSocial")
OUT_DIR = Path("data/processed/training_matrices")

PAIR_DIRS = {
    "Google_Insta": ("googlePlus", "instagram"),
    "Google_Twitter": ("googlePlus", "twitter"),
    "Insta_Twitter": ("instagram", "twitter"),
}


def to_bigrams(s: str) -> set:
    s = s.strip().lower()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def load_profile(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict, key: str) -> str:
    v = d.get(key, "")
    return str(v).strip() if v else ""


def collect_all_profiles():
    """
    รวบรวม profile ทั้งหมดจาก 2.profile.data ทุก platform pair
    แต่ละ profile จะมี identity_key = (pair_name, folder_name) เพื่อระบุว่าเป็นคนเดียวกัน
    """
    profiles = []  # list of dicts
    # identity_group: folder_name -> list of profile indices (คนเดียวกัน)
    identity_groups = {}

    for pair_name, (prefix_a, prefix_b) in PAIR_DIRS.items():
        pair_dir = BASE / "2.profile.data" / pair_name
        if not pair_dir.exists():
            continue

        print(f"กำลังโหลด {pair_name}...")

        for folder_name in sorted(os.listdir(pair_dir)):
            folder_path = pair_dir / folder_name
            if not folder_path.is_dir():
                continue

            for fname in os.listdir(folder_path):
                if not fname.endswith(".json"):
                    continue
                if fname in ("score_file.json", "filename.json"):
                    continue

                fpath = folder_path / fname
                flow = fname.lower()

                # ระบุ platform
                if "google" in flow:
                    platform = "googlePlus"
                elif "instagram" in flow:
                    platform = "instagram"
                elif "twitter" in flow:
                    platform = "twitter"
                else:
                    continue

                try:
                    data = load_profile(str(fpath))
                except Exception:
                    continue

                idx = len(profiles)
                username = safe_get(data, "userName").lower()
                fullname = safe_get(data, "fullName").lower()

                profiles.append({
                    "idx": idx,
                    "platform": platform,
                    "pair_source": pair_name,
                    "folder": folder_name,
                    "userName": username,
                    "fullName": fullname,
                    "bigrams_user": to_bigrams(username),
                    "bigrams_full": to_bigrams(fullname),
                    "bigrams_merged": to_bigrams(f"{username} {fullname}"),
                })

                # เก็บ identity group (คนเดียวกัน = folder เดียวกันใน pair เดียวกัน)
                group_key = f"{pair_name}::{folder_name}"
                identity_groups.setdefault(group_key, []).append(idx)

    return profiles, identity_groups


def build_combined_matrix(profiles, identity_groups):
    """
    สร้าง n×n matrix รวมทุก profile
    """
    n = len(profiles)
    print(f"\nจำนวน profile ทั้งหมด: {n}")
    print(f"จำนวน identity groups: {len(identity_groups)}")
    print(f"ขนาด matrix: {n}×{n} = {n*n:,} pairs")

    # สร้าง set ของ indices ที่เป็นคนเดียวกัน
    same_person_pairs = set()
    for group_key, indices in identity_groups.items():
        for i in indices:
            for j in indices:
                if i != j:
                    same_person_pairs.add((i, j))
            same_person_pairs.add((i, i))  # ตัวเองก็ match

    print(f"จำนวน positive pairs (คนเดียวกัน): {len(same_person_pairs):,}")

    # สร้าง flat training table (ไม่สร้าง full n×n matrix เพราะจะใหญ่เกิน)
    # ใช้วิธี: เก็บ positive pairs ทั้งหมด + sample negative pairs
    print("\nกำลังสร้าง training data...")

    # --- 1) Positive pairs (คนเดียวกัน) ---
    pos_rows = []
    for group_key, indices in identity_groups.items():
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                pa = profiles[i]
                pb = profiles[j]
                pos_rows.append({
                    "profile_a_idx": i,
                    "profile_a_platform": pa["platform"],
                    "profile_a_folder": pa["folder"],
                    "profile_a_pair_source": pa["pair_source"],
                    "profile_a_userName": pa["userName"],
                    "profile_a_fullName": pa["fullName"],
                    "profile_b_idx": j,
                    "profile_b_platform": pb["platform"],
                    "profile_b_folder": pb["folder"],
                    "profile_b_pair_source": pb["pair_source"],
                    "profile_b_userName": pb["userName"],
                    "profile_b_fullName": pb["fullName"],
                    "username_exact_match": int(pa["userName"] == pb["userName"]) if pa["userName"] and pb["userName"] else 0,
                    "fullname_exact_match": int(pa["fullName"] == pb["fullName"]) if pa["fullName"] and pb["fullName"] else 0,
                    "username_bigram_jaccard": round(jaccard(pa["bigrams_user"], pb["bigrams_user"]), 6),
                    "fullname_bigram_jaccard": round(jaccard(pa["bigrams_full"], pb["bigrams_full"]), 6),
                    "merged_bigram_jaccard": round(jaccard(pa["bigrams_merged"], pb["bigrams_merged"]), 6),
                    "same_platform": int(pa["platform"] == pb["platform"]),
                    "label": 1,
                })

    print(f"  Positive pairs: {len(pos_rows):,}")

    # --- 2) Negative pairs (คนละคน) - sample เพื่อ balance ---
    # Sample negative pairs: ~5x จำนวน positive
    np.random.seed(42)
    neg_target = min(len(pos_rows) * 5, n * n - len(same_person_pairs))
    neg_rows = []
    attempts = 0
    max_attempts = neg_target * 3

    while len(neg_rows) < neg_target and attempts < max_attempts:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            attempts += 1
            continue
        if (i, j) in same_person_pairs:
            attempts += 1
            continue

        pa = profiles[i]
        pb = profiles[j]
        neg_rows.append({
            "profile_a_idx": i,
            "profile_a_platform": pa["platform"],
            "profile_a_folder": pa["folder"],
            "profile_a_pair_source": pa["pair_source"],
            "profile_a_userName": pa["userName"],
            "profile_a_fullName": pa["fullName"],
            "profile_b_idx": j,
            "profile_b_platform": pb["platform"],
            "profile_b_folder": pb["folder"],
            "profile_b_pair_source": pb["pair_source"],
            "profile_b_userName": pb["userName"],
            "profile_b_fullName": pb["fullName"],
            "username_exact_match": int(pa["userName"] == pb["userName"]) if pa["userName"] and pb["userName"] else 0,
            "fullname_exact_match": int(pa["fullName"] == pb["fullName"]) if pa["fullName"] and pb["fullName"] else 0,
            "username_bigram_jaccard": round(jaccard(pa["bigrams_user"], pb["bigrams_user"]), 6),
            "fullname_bigram_jaccard": round(jaccard(pa["bigrams_full"], pb["bigrams_full"]), 6),
            "merged_bigram_jaccard": round(jaccard(pa["bigrams_merged"], pb["bigrams_merged"]), 6),
            "same_platform": int(pa["platform"] == pb["platform"]),
            "label": 0,
        })
        attempts += 1

        if len(neg_rows) % 10000 == 0:
            print(f"    Negative pairs สร้างแล้ว: {len(neg_rows):,}")

    print(f"  Negative pairs: {len(neg_rows):,}")

    # รวมเข้าด้วยกัน
    all_rows = pos_rows + neg_rows
    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    return df, profiles


def build_label_matrix_small(profiles, identity_groups, pair_name=None):
    """
    สร้าง n×n label matrix สำหรับแต่ละ pair (ขนาดเล็กพอที่จะเก็บเป็น CSV ได้)
    """
    if pair_name:
        # กรองเฉพาะ profiles จาก pair นี้
        relevant_indices = set()
        for group_key, indices in identity_groups.items():
            if group_key.startswith(pair_name):
                relevant_indices.update(indices)
        sub_profiles = [p for p in profiles if p["idx"] in relevant_indices]
    else:
        sub_profiles = profiles

    return sub_profiles


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) รวบรวม profiles ทั้งหมด
    profiles, identity_groups = collect_all_profiles()

    # 2) สร้าง combined training data
    df, profiles = build_combined_matrix(profiles, identity_groups)

    # 3) บันทึก
    out_path = OUT_DIR / "combined_all_platforms_training.csv"
    df.to_csv(out_path, index=False)
    print(f"\nบันทึกแล้ว: {out_path}")
    print(f"จำนวนแถวทั้งหมด: {len(df):,}")
    print(f"Label=1 (คนเดียวกัน): {(df['label']==1).sum():,}")
    print(f"Label=0 (คนละคน): {(df['label']==0).sum():,}")

    # 4) สร้าง profile index table
    prof_df = pd.DataFrame([
        {
            "idx": p["idx"],
            "platform": p["platform"],
            "pair_source": p["pair_source"],
            "folder": p["folder"],
            "userName": p["userName"],
            "fullName": p["fullName"],
        }
        for p in profiles
    ])
    prof_path = OUT_DIR / "combined_profile_index.csv"
    prof_df.to_csv(prof_path, index=False)
    print(f"บันทึก profile index: {prof_path}")

    # 5) สรุปสถิติ
    print("\n" + "=" * 60)
    print("สรุปสถิติ")
    print("=" * 60)
    print(f"Platform breakdown:")
    print(prof_df["platform"].value_counts().to_string())
    print(f"\nPair source breakdown:")
    print(prof_df["pair_source"].value_counts().to_string())
    print(f"\nFeature statistics (label=1 vs label=0):")
    for col in ["username_exact_match", "fullname_exact_match",
                 "username_bigram_jaccard", "fullname_bigram_jaccard",
                 "merged_bigram_jaccard"]:
        pos_mean = df.loc[df["label"] == 1, col].mean()
        neg_mean = df.loc[df["label"] == 0, col].mean()
        print(f"  {col}: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}")


if __name__ == "__main__":
    main()
