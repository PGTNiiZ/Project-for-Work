"""
Stage 8: Pair Construction & Label Building

สร้าง labeled_pairs.parquet สำหรับ train ML model

3 ประเภท pairs:
  (1) Positive pairs   — profiles ที่ user_folder เดียวกัน คนละ platform → label=1
  (2) Random negatives — profiles ที่ user_folder ต่างกัน สุ่มจับคู่     → label=0
  (3) Hard negatives   — profiles ที่ชื่อ/username คล้ายกันแต่คนละคน    → label=0

Input  : nomalized_profiles.csv (มี profile_id + platform + user_folder)
Output : labeled_pairs.parquet, pair_stats.json

Feature Summary (ตามที่อธิบายในเมสเสจก่อนหน้า):
  - B1 (String similarity): Jaro-Winkler / token-sort / ratio
  - A1 (URL domain, exact match, jaccard)
  - A2 (Location similarity / fuzzy location)
  - A3 (Mention overlap / username-in-mention)
  - A4 (Hashtag overlap)
  - B3 (text shape features: caps, punctuation, length)
  - META (platform, pair encoding)

SBERT (Optional):
  - ต้องติดตั้ง: pip install sentence-transformers
  - ถ้าต้องการใช้ SBERT ให้ uncomment ส่วน `compute_sbert_embeddings` และ `sbert_emb`
  - ใช้ model: all-mpnet-base-v2 (ดาวน์โหลดอัตโนมัติ ~420MB)
  - แนะนำให้ใช้ GPU ถ้ามี (เพิ่มความเร็วมาก)

Usage:
  python stage8_pair_builder.py

"""

import json
import math
import random
import warnings
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

warnings.filterwarnings("ignore")

# ─── config ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
NEG_TO_POS_RATIO = 5  # random negatives = 5x positive pairs
HARD_NEG_RATIO = 2  # hard negatives = 2x positive pairs
HARD_NEG_THRESHOLD = 0.65  # Jaro-Winkler threshold สำหรับ hard negative candidate
CROSS_PLATFORM_ONLY = True  # positive pairs ข้าม platform เท่านั้น (ไม่จับ same platform)

# สำหรับ SBERT (comment out ถ้าไม่ต้องการใช้)
USE_SBERT = False
SBERT_MODEL = "all-mpnet-base-v2"

# ─── utilities ───────────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip().lower()


def _name_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity ระหว่าง 2 strings"""
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a, b)


# ─── pair generators ─────────────────────────────────────────────────────────

class PairBuilder:
    """สร้าง labeled pairs จาก DataFrame ที่มี profile_id + user_folder + platform"""

    def __init__(self, df: pd.DataFrame, seed: int = RANDOM_SEED):
        self.df = df.copy().reset_index(drop=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # index lookups
        self._build_indexes()

    def _build_indexes(self):
        """สร้าง lookup tables สำหรับ fast access"""

        self.entity_to_rows: dict[str, list[int]] = {}
        for i, row in self.df.iterrows():
            key = (
                str(row["user_folder"])
                if pd.notna(row.get("user_folder"))
                else str(row["profile_id"])
            )
            self.entity_to_rows.setdefault(key, []).append(i)

        self.pid_to_idx: dict[int, int] = {
            int(r["profile_id"]): i
            for i, r in self.df.iterrows()
        }

        self.platform_to_rows: dict[str, list[int]] = {}
        for i, row in self.df.iterrows():
            plat = _safe_str(row.get("platform", ""))
            self.platform_to_rows.setdefault(plat, []).append(i)

        print(
            f"[PairBuilder] Loaded {len(self.df)} profiles, "
            f"{len(self.entity_to_rows)} entities, "
            f"{len(self.platform_to_rows)} platforms"
        )

    # ──────────────────────────────────────────────────────────
    # (1) POSITIVE PAIRS
    # ──────────────────────────────────────────────────────────

    def generate_positive_pairs(self) -> pd.DataFrame:
        """สร้าง positive pairs: profiles ที่ user_folder เดียวกัน คนละ platform"""
        pairs = []
        skipped_same_platform = 0

        for entity, row_idxs in self.entity_to_rows.items():
            if len(row_idxs) < 2:
                continue

            profiles = [self.df.loc[i] for i in row_idxs]

            for r_a, r_b in itertools.combinations(profiles, 2):
                plat_a = _safe_str(r_a.get("platform", ""))
                plat_b = _safe_str(r_b.get("platform", ""))

                if CROSS_PLATFORM_ONLY and plat_a == plat_b:
                    skipped_same_platform += 1
                    continue

                pairs.append(
                    {
                        "profile_id_a": int(r_a["profile_id"]),
                        "profile_id_b": int(r_b["profile_id"]),
                        "platform_a": plat_a,
                        "platform_b": plat_b,
                        "label": 1,
                        "pair_type": "positive",
                    }
                )

        df_pos = pd.DataFrame(pairs)
        print(
            f"[PairBuilder] Positive pairs: {len(df_pos)}"
            f"  (skipped same-platform: {skipped_same_platform})"
        )
        return df_pos

    # ──────────────────────────────────────────────────────────
    # (2) RANDOM NEGATIVE PAIRS
    # ──────────────────────────────────────────────────────────

    def generate_random_negatives(
        self,
        n_positive: int,
        ratio: float = NEG_TO_POS_RATIO,
        positive_pid_set: set = None,
    ) -> pd.DataFrame:
        """สุ่ม pairs ที่ entity ต่างกัน"""
        n_target = int(n_positive * ratio)
        all_ids = self.df["profile_id"].tolist()

        pid_to_folder = {
            int(self.df.loc[i, "profile_id"]): str(self.df.loc[i, "user_folder"])
            for i in range(len(self.df))
            if pd.notna(self.df.loc[i, "user_folder"])
        }
        pid_to_plat = {
            int(self.df.loc[i, "profile_id"]): _safe_str(
                self.df.loc[i, "platform"]
            )
            for i in range(len(self.df))
        }

        positive_set = positive_pid_set or set()
        generated = set()
        pairs = []
        attempts = 0
        max_attempts = n_target * 20

        while len(pairs) < n_target and attempts < max_attempts:
            attempts += 1
            idx_a = random.randint(0, len(all_ids) - 1)
            idx_b = random.randint(0, len(all_ids) - 1)

            pid_a = int(all_ids[idx_a])
            pid_b = int(all_ids[idx_b])

            if pid_a == pid_b:
                continue

            key = (min(pid_a, pid_b), max(pid_a, pid_b))
            if key in generated or key in positive_set:
                continue

            folder_a = pid_to_folder.get(pid_a, f"__unk_{pid_a}")
            folder_b = pid_to_folder.get(pid_b, f"__unk_{pid_b}")
            if folder_a == folder_b:
                continue

            generated.add(key)
            pairs.append(
                {
                    "profile_id_a": pid_a,
                    "profile_id_b": pid_b,
                    "platform_a": pid_to_plat.get(pid_a, ""),
                    "platform_b": pid_to_plat.get(pid_b, ""),
                    "label": 0,
                    "pair_type": "random_negative",
                }
            )

        df_neg = pd.DataFrame(pairs)
        print(
            f"[PairBuilder] Random negatives: {len(df_neg)}"
            f"  (target={n_target}, attempts={attempts})"
        )
        return df_neg

    # ──────────────────────────────────────────────────────────
    # (3) HARD NEGATIVE PAIRS
    # ──────────────────────────────────────────────────────────

    def generate_hard_negatives(
        self,
        n_positive: int,
        ratio: float = HARD_NEG_RATIO,
        threshold: float = HARD_NEG_THRESHOLD,
        positive_pid_set: set = None,
        random_neg_set: set = None,
    ) -> pd.DataFrame:
        """Hard negatives = profiles ที่เหมือนแต่คนละ entity"""
        n_target = int(n_positive * ratio)
        existing = (positive_pid_set or set()) | (random_neg_set or set())

        self.df["_name_prefix3"] = (
            self.df["userName"].fillna("").str.lower().str[:3]
        )

        has_name = self.df[
            self.df["userName"].notna() & (self.df["userName"].str.strip() != "")
        ].copy()

        blocks = has_name.groupby("_name_prefix3", group_keys=False)

        pid_to_folder = {
            int(r["profile_id"]): str(r["user_folder"])
            for _, r in self.df.iterrows()
            if pd.notna(r.get("user_folder"))
        }
        pid_to_plat = {
            int(r["profile_id"]): _safe_str(r.get("platform", ""))
            for _, r in self.df.iterrows()
        }

        candidates = []
        for prefix, group in blocks:
            if len(prefix) < 2 or len(group) < 2:
                continue

            rows = group.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    ra, rb = rows[i], rows[j]
                    pid_a = int(ra["profile_id"])
                    pid_b = int(rb["profile_id"])

                    folder_a = pid_to_folder.get(pid_a, f"__unk_{pid_a}")
                    folder_b = pid_to_folder.get(pid_b, f"__unk_{pid_b}")
                    if folder_a == folder_b:
                        continue

                    key = (min(pid_a, pid_b), max(pid_a, pid_b))
                    if key in existing:
                        continue

                    ua = _safe_str(ra.get("userName", ""))
                    ub = _safe_str(rb.get("userName", ""))
                    sim_name = _name_similarity(ua, ub)

                    if sim_name >= threshold:
                        candidates.append((key, pid_a, pid_b, sim_name))

        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[:n_target]

        pairs = []
        for key, pid_a, pid_b, sim in candidates:
            pairs.append(
                {
                    "profile_id_a": pid_a,
                    "profile_id_b": pid_b,
                    "platform_a": pid_to_plat.get(pid_a, ""),
                    "platform_b": pid_to_plat.get(pid_b, ""),
                    "label": 0,
                    "pair_type": "hard_negative",
                }
            )

        if "_name_prefix3" in self.df.columns:
            self.df.drop(columns=["_name_prefix3"], inplace=True)

        df_hard = pd.DataFrame(pairs)
        print(
            f"[PairBuilder] Hard negatives: {len(df_hard)}"
            f"  (target={n_target}, candidates={len(candidates)})"
        )
        return df_hard

    # ──────────────────────────────────────────────────────────
    # BUILD ALL PAIRS
    # ──────────────────────────────────────────────────────────

    def build(
        self,
        neg_ratio: float = NEG_TO_POS_RATIO,
        hard_neg_ratio: float = HARD_NEG_RATIO,
        hard_threshold: float = HARD_NEG_THRESHOLD,
    ) -> pd.DataFrame:
        """สร้าง labeled_pairs ครบทั้ง 3 ประเภท แล้ว concat + shuffle"""
        print("\n" + "=" * 55)
        print("Stage 8: Pair Construction & Label Building")
        print("=" * 55)

        print("\n[1/3] Generating positive pairs ...")
        df_pos = self.generate_positive_pairs()

        pos_set = {
            (min(int(r["profile_id_a"]), int(r["profile_id_b"])),
             max(int(r["profile_id_a"]), int(r["profile_id_b"])))
            for _, r in df_pos.iterrows()
        }

        print("\n[2/3] Generating random negatives ...")
        df_rand = self.generate_random_negatives(
            n_positive=len(df_pos),
            ratio=neg_ratio,
            positive_pid_set=pos_set,
        )

        rand_set = {
            (min(int(r["profile_id_a"]), int(r["profile_id_b"])),
             max(int(r["profile_id_a"]), int(r["profile_id_b"])))
            for _, r in df_rand.iterrows()
        }

        print("\n[3/3] Generating hard negatives ...")
        df_hard = self.generate_hard_negatives(
            n_positive=len(df_pos),
            ratio=hard_neg_ratio,
            threshold=hard_threshold,
            positive_pid_set=pos_set,
            random_neg_set=rand_set,
        )

        df_all = pd.concat([df_pos, df_rand, df_hard], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print("\n" + "─" * 45)
        print("Label distribution:")
        print(f"  Positive (label=1):  {(df_all['label'] == 1).sum():>7,}")
        print(f"  Negative (label=0):  {(df_all['label'] == 0).sum():>7,}")
        print(f"  Total pairs:         {len(df_all):>7,}")
        print(
            f"  Pos/Neg ratio:       1 : {(df_all['label'] == 0).sum() / max((df_all['label'] == 1).sum(), 1):.1f}"
        )
        print()
        print("Pair type breakdown:")
        for ptype, cnt in df_all["pair_type"].value_counts().items():
            print(f"  {ptype:<20} {cnt:>7,}")
        print()
        print("Platform pair distribution (positive):")
        pos_only = df_all[df_all["label"] == 1].copy()
        pos_only["plat_pair"] = pos_only.apply(
            lambda r: " x ".join(sorted([r["platform_a"], r["platform_b"]])), axis=1
        )
        for pp, cnt in pos_only["plat_pair"].value_counts().items():
            print(f"  {pp:<35} {cnt:>6,}")

        return df_all


# ─── quality checks ──────────────────────────────────────────────────────────

def run_quality_checks(labeled_pairs: pd.DataFrame, df_profiles: pd.DataFrame) -> dict:
    """ตรวจสอบคุณภาพของ labeled_pairs ก่อน save"""
    print("\n" + "=" * 45)
    print("Quality Checks")
    print("=" * 45)

    results = {}

    pair_keys = labeled_pairs.apply(
        lambda r: (
            min(int(r["profile_id_a"]), int(r["profile_id_b"])),
            max(int(r["profile_id_a"]), int(r["profile_id_b"]))
        ),
        axis=1,
    )
    n_dupes = pair_keys.duplicated().sum()
    results["no_duplicate_pairs"] = n_dupes == 0
    print(f"  [{'PASS' if n_dupes == 0 else 'FAIL'}] No duplicate pairs: {n_dupes} duplicates found")

    self_pairs = (labeled_pairs["profile_id_a"] == labeled_pairs["profile_id_b"]).sum()
    results["no_self_pairs"] = self_pairs == 0
    print(f"  [{'PASS' if self_pairs == 0 else 'FAIL'}] No self-pairs: {self_pairs}")

    pos_keys = set(pair_keys[labeled_pairs["label"] == 1])
    neg_keys = set(pair_keys[labeled_pairs["label"] == 0])
    overlap = len(pos_keys & neg_keys)
    results["no_pos_neg_overlap"] = overlap == 0
    print(f"  [{'PASS' if overlap == 0 else 'FAIL'}] No pos/neg overlap: {overlap}")

    valid_pids = set(df_profiles["profile_id"].astype(int).tolist())
    all_pair_pids = (
        set(labeled_pairs["profile_id_a"].astype(int))
        | set(labeled_pairs["profile_id_b"].astype(int))
    )
    orphan = all_pair_pids - valid_pids
    results["all_pids_valid"] = len(orphan) == 0
    print(
        f"  [{'PASS' if len(orphan) == 0 else 'FAIL'}] All profile_ids valid: {len(orphan)} orphan IDs"
    )

    n_pos = (labeled_pairs["label"] == 1).sum()
    n_neg = (labeled_pairs["label"] == 0).sum()
    ratio = n_neg / max(n_pos, 1)
    results["class_ratio"] = round(ratio, 2)
    print(f"  [INFO] Class ratio (neg/pos): {ratio:.1f} : 1")

    print(
        f"  [INFO] Hard negatives generated: "
        f"{(labeled_pairs['pair_type'] == 'hard_negative').sum():,}"
    )

    all_pass = all(v for k, v in results.items() if k != "class_ratio")
    results["all_pass"] = all_pass
    print(
        f"\n  {'✅ All checks passed!' if all_pass else '❌ Some checks FAILED'}"
    )

    return results


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    # Try a few common locations for the input CSVs (workspace root + data folders)
    candidate_paths = [
        "nomalized_profiles.csv",
        "combined_profiles.csv",
        "Project-for-Work/data-for-project/nomalized_profiles.csv",
        "Project-for-Work/data/data/Dataset-LinkSocial/data/nomalized_profiles.csv",
    ]

    CSV_PATH = None
    for p in candidate_paths:
        if Path(p).exists():
            CSV_PATH = p
            break

    if CSV_PATH is None:
        raise FileNotFoundError(
            "Could not find nomalized_profiles.csv or combined_profiles.csv in expected locations. "
            "Please place the input file in the workspace root or update the script."
        )

    df = pd.read_csv(CSV_PATH)
    entity_col = "user_folder"
    if "profile_id" not in df.columns:
        df["profile_id"] = range(len(df))
    print(f"Loaded: {CSV_PATH}  shape={df.shape}")

    df["user_folder"] = df["user_folder"].fillna(df["profile_id"].astype(str))

    print(f"Profiles: {len(df):,}  |  Entities: {df['user_folder'].nunique():,}")
    print(f"Platforms: {df['platform'].value_counts().to_dict()}")

    builder = PairBuilder(df, seed=RANDOM_SEED)

    labeled_pairs = builder.build(
        neg_ratio=NEG_TO_POS_RATIO,
        hard_neg_ratio=HARD_NEG_RATIO,
        hard_threshold=HARD_NEG_THRESHOLD,
    )

    qa_results = run_quality_checks(labeled_pairs, df)

    labeled_pairs.to_parquet("labeled_pairs.parquet", index=False)

    stats = {
        "total_pairs": int(len(labeled_pairs)),
        "positive_pairs": int((labeled_pairs["label"] == 1).sum()),
        "negative_pairs": int((labeled_pairs["label"] == 0).sum()),
        "random_negatives": int(
            (labeled_pairs["pair_type"] == "random_negative").sum()
        ),
        "hard_negatives": int(
            (labeled_pairs["pair_type"] == "hard_negative").sum()
        ),
        "class_ratio": round(
            (labeled_pairs["label"] == 0).sum()
            / max((labeled_pairs["label"] == 1).sum(), 1),
            2,
        ),
        "qa": qa_results,
        "config": {
            "random_seed": RANDOM_SEED,
            "neg_to_pos_ratio": NEG_TO_POS_RATIO,
            "hard_neg_ratio": HARD_NEG_RATIO,
            "hard_neg_threshold": HARD_NEG_THRESHOLD,
            "cross_platform_only": CROSS_PLATFORM_ONLY,
        },
    }
    def _json_default(o):
        # Convert numpy scalar types (and other non-serializable objects) to plain Python
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        return str(o)

    with open("pair_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=_json_default)

    print("\n✅ Stage 8 Complete!")
    print(f"   labeled_pairs.parquet  ({len(labeled_pairs):,} pairs)")
    print(f"   pair_stats.json")
    print(f"\n   Positive : {stats['positive_pairs']:,}")
    print(f"   Random-  : {stats['random_negatives']:,}")
    print(f"   Hard-    : {stats['hard_negatives']:,}")
    print(f"   Ratio    : 1 : {stats['class_ratio']}")
