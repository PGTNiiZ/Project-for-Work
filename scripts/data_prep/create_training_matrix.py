"""
Create n×n training data matrix from 1.profile.data and 2.profile.data.

For each platform pair (Google_Insta, Google_Twitter, Insta_Twitter):
- Build an n×n comparison matrix of all profiles from platform A vs platform B
- Label: 1 if same person (same folder), 0 otherwise
- Features: username exact match, fullname exact match, bigram Jaccard similarity
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


def extract_profiles_from_pair(pair_dir: Path, prefix_a: str, prefix_b: str):
    """Load all profile pairs from a 2.profile.data sub-directory."""
    profiles_a = []
    profiles_b = []
    user_folders = []

    for folder_name in sorted(os.listdir(pair_dir)):
        folder_path = pair_dir / folder_name
        if not folder_path.is_dir():
            continue

        prof_a = None
        prof_b = None

        for fname in os.listdir(folder_path):
            if not fname.endswith(".json"):
                continue
            if fname in ("score_file.json", "filename.json"):
                continue

            fpath = folder_path / fname
            flow = fname.lower()

            if prefix_a == "googlePlus" and "google" in flow:
                prof_a = load_profile(str(fpath))
                prof_a["_file"] = fname
            elif prefix_a == "instagram" and "instagram" in flow:
                prof_a = load_profile(str(fpath))
                prof_a["_file"] = fname
            elif prefix_a == "twitter" and "twitter" in flow:
                prof_a = load_profile(str(fpath))
                prof_a["_file"] = fname

            if prefix_b == "instagram" and "instagram" in flow:
                prof_b = load_profile(str(fpath))
                prof_b["_file"] = fname
            elif prefix_b == "twitter" and "twitter" in flow:
                prof_b = load_profile(str(fpath))
                prof_b["_file"] = fname
            elif prefix_b == "googlePlus" and "google" in flow:
                prof_b = load_profile(str(fpath))
                prof_b["_file"] = fname

        if prof_a and prof_b:
            profiles_a.append(prof_a)
            profiles_b.append(prof_b)
            user_folders.append(folder_name)

    return profiles_a, profiles_b, user_folders


def safe_get(d: dict, key: str) -> str:
    v = d.get(key, "")
    return str(v).strip() if v else ""


def build_nxn_matrix(profiles_a, profiles_b, user_folders, pair_name, prefix_a, prefix_b):
    """
    Build n×n comparison matrix.
    Row i = profile from platform A (index i)
    Col j = profile from platform B (index j)
    Label = 1 if same folder (same person), 0 otherwise
    """
    n = len(profiles_a)
    print(f"  [{pair_name}] Building {n}×{n} matrix ({n*n:,} pairs)...")

    # Pre-compute bigrams
    usernames_a = [safe_get(p, "userName").lower() for p in profiles_a]
    fullnames_a = [safe_get(p, "fullName").lower() for p in profiles_a]
    bigrams_user_a = [to_bigrams(u) for u in usernames_a]
    bigrams_full_a = [to_bigrams(f) for f in fullnames_a]
    bigrams_merged_a = [to_bigrams(f"{u} {f}") for u, f in zip(usernames_a, fullnames_a)]

    usernames_b = [safe_get(p, "userName").lower() for p in profiles_b]
    fullnames_b = [safe_get(p, "fullName").lower() for p in profiles_b]
    bigrams_user_b = [to_bigrams(u) for u in usernames_b]
    bigrams_full_b = [to_bigrams(f) for f in fullnames_b]
    bigrams_merged_b = [to_bigrams(f"{u} {f}") for u, f in zip(usernames_b, fullnames_b)]

    # Build label matrix: label[i][j] = 1 if i==j (same person), 0 otherwise
    label_matrix = np.eye(n, dtype=int)

    # Build feature matrices
    username_exact = np.zeros((n, n), dtype=int)
    fullname_exact = np.zeros((n, n), dtype=int)
    username_bigram_sim = np.zeros((n, n), dtype=float)
    fullname_bigram_sim = np.zeros((n, n), dtype=float)
    merged_bigram_sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            # Exact match
            if usernames_a[i] and usernames_b[j]:
                username_exact[i, j] = int(usernames_a[i] == usernames_b[j])
            if fullnames_a[i] and fullnames_b[j]:
                fullname_exact[i, j] = int(fullnames_a[i] == fullnames_b[j])

            # Bigram Jaccard similarity
            username_bigram_sim[i, j] = jaccard(bigrams_user_a[i], bigrams_user_b[j])
            fullname_bigram_sim[i, j] = jaccard(bigrams_full_a[i], bigrams_full_b[j])
            merged_bigram_sim[i, j] = jaccard(bigrams_merged_a[i], bigrams_merged_b[j])

        if (i + 1) % 200 == 0:
            print(f"    row {i+1}/{n} done")

    return {
        "label": label_matrix,
        "username_exact": username_exact,
        "fullname_exact": fullname_exact,
        "username_bigram_sim": username_bigram_sim,
        "fullname_bigram_sim": fullname_bigram_sim,
        "merged_bigram_sim": merged_bigram_sim,
        "usernames_a": usernames_a,
        "fullnames_a": fullnames_a,
        "usernames_b": usernames_b,
        "fullnames_b": fullnames_b,
        "user_folders": user_folders,
    }


def matrices_to_flat_table(result, pair_name, prefix_a, prefix_b):
    """Convert n×n matrices to flat training table (one row per pair)."""
    n = len(result["user_folders"])
    rows = []

    for i in range(n):
        for j in range(n):
            rows.append({
                "pair_type": pair_name,
                f"{prefix_a}_idx": i,
                f"{prefix_a}_folder": result["user_folders"][i],
                f"{prefix_a}_userName": result["usernames_a"][i],
                f"{prefix_a}_fullName": result["fullnames_a"][i],
                f"{prefix_b}_idx": j,
                f"{prefix_b}_folder": result["user_folders"][j],
                f"{prefix_b}_userName": result["usernames_b"][j],
                f"{prefix_b}_fullName": result["fullnames_b"][j],
                "username_exact_match": result["username_exact"][i, j],
                "fullname_exact_match": result["fullname_exact"][i, j],
                "username_bigram_jaccard": round(result["username_bigram_sim"][i, j], 6),
                "fullname_bigram_jaccard": round(result["fullname_bigram_sim"][i, j], 6),
                "merged_bigram_jaccard": round(result["merged_bigram_sim"][i, j], 6),
                "label": result["label"][i, j],
            })

    return pd.DataFrame(rows)


def save_matrix_csvs(result, pair_name, prefix_a, prefix_b, out_dir):
    """Save n×n matrices as individual CSVs (row/col headers = user folders)."""
    folders = result["user_folders"]
    idx_a = [f"{prefix_a}:{f}" for f in folders]
    idx_b = [f"{prefix_b}:{f}" for f in folders]

    for name, matrix in [
        ("label", result["label"]),
        ("username_exact", result["username_exact"]),
        ("fullname_exact", result["fullname_exact"]),
        ("username_bigram_sim", result["username_bigram_sim"]),
        ("fullname_bigram_sim", result["fullname_bigram_sim"]),
        ("merged_bigram_sim", result["merged_bigram_sim"]),
    ]:
        df = pd.DataFrame(matrix, index=idx_a, columns=idx_b)
        path = out_dir / f"{pair_name}_{name}_matrix.csv"
        df.to_csv(path)
        print(f"    Saved {path.name} ({df.shape})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_flat = []

    for pair_name, (prefix_a, prefix_b) in PAIR_DIRS.items():
        pair_dir = BASE / "2.profile.data" / pair_name
        if not pair_dir.exists():
            print(f"Skipping {pair_name}: dir not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {pair_name} ({prefix_a} vs {prefix_b})")
        print(f"{'='*60}")

        profiles_a, profiles_b, user_folders = extract_profiles_from_pair(
            pair_dir, prefix_a, prefix_b
        )
        print(f"  Loaded {len(profiles_a)} profile pairs")

        if not profiles_a:
            print("  No profiles found, skipping")
            continue

        # Build n×n matrices
        result = build_nxn_matrix(
            profiles_a, profiles_b, user_folders, pair_name, prefix_a, prefix_b
        )

        # Save n×n matrix CSVs
        pair_out = OUT_DIR / pair_name
        pair_out.mkdir(exist_ok=True)
        save_matrix_csvs(result, pair_name, prefix_a, prefix_b, pair_out)

        # Convert to flat training table
        flat_df = matrices_to_flat_table(result, pair_name, prefix_a, prefix_b)
        flat_path = pair_out / f"{pair_name}_training_flat.csv"
        flat_df.to_csv(flat_path, index=False)
        print(f"    Saved {flat_path.name} ({len(flat_df):,} rows)")

        # Summary stats
        pos = int(flat_df["label"].sum())
        neg = len(flat_df) - pos
        print(f"    Positive (match): {pos}, Negative (no match): {neg}")
        print(f"    Ratio: 1:{neg//pos if pos else 'inf'}")

        all_flat.append(flat_df)

    # Combined flat table
    if all_flat:
        combined = pd.concat(all_flat, ignore_index=True)
        combined_path = OUT_DIR / "all_pairs_training_flat.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"Combined training data: {len(combined):,} rows")
        print(f"Saved to: {combined_path}")
        pos = int(combined["label"].sum())
        neg = len(combined) - pos
        print(f"Positive: {pos}, Negative: {neg}, Ratio: 1:{neg//pos if pos else 'inf'}")


if __name__ == "__main__":
    main()
