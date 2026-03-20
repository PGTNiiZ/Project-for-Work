"""
Image Matching Pipeline — Pairwise Embedding Comparison + Classification
=========================================================================
Pipeline นี้เทียบ image embeddings (CLIP + ArcFace) แบบ pairwise ก่อน
แล้วค่อยเข้า classification — เหมือน approach ของ bio vector embedding

ขั้นตอน:
    1. Scan embedding directories → ค้นหา profiles ทั้งหมดที่มี embeddings
    2. โหลด CLIP + ArcFace embeddings (.npy) → fuse เป็น 1024-dim vector
    3. เทียบ embeddings แบบ pairwise (all-vs-all cosine similarity)
    4. Candidate selection — เลือก Top-K ที่คล้ายที่สุด
    5. สร้าง Feature vectors → Train Logistic Regression + Random Forest
    6. Evaluate Top-1 accuracy + Candidate recall
    7. predict_similarity() → เทียบ 2 profiles ว่าเป็นคนเดียวกันไหม

Usage:
    python image_matching_pipeline.py
    python image_matching_pipeline.py --top-k 50
    python image_matching_pipeline.py --predict profileA profileB
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"  # data/ อยู่ใน Project-for-Work/

DIRS = {
    "embeddings_clip": DATA_DIR / "embeddings" / "clip",
    "embeddings_face": DATA_DIR / "embeddings" / "face",
    "results": DATA_DIR / "processed" / "image_pipeline_results",
}

CLIP_DIM = 512
ARCFACE_DIM = 512
FUSED_DIM = CLIP_DIM + ARCFACE_DIM  # 1024


def log_stage(stage: str, start_ts: float) -> None:
    """แสดงเวลาที่ใช้ในแต่ละขั้นตอน"""
    elapsed = time.time() - start_ts
    print(f"  [{stage}] done in {elapsed:.2f}s")


# ============================================================
# 1. DATA STRUCTURES
# ============================================================

@dataclass
class ImageProfile:
    """โปรไฟล์ที่มี image embeddings"""
    profile_id: str
    platform: str
    clip_emb: Optional[np.ndarray] = None      # CLIP embedding (512-dim)
    face_emb: Optional[np.ndarray] = None       # ArcFace embedding (512-dim)
    fused_emb: Optional[np.ndarray] = None      # Fused [clip; face] (1024-dim)
    has_clip: bool = False
    has_face: bool = False


# ============================================================
# 2. EMBEDDING LOADER — scan directories เพื่อค้นหา profiles ทั้งหมด
# ============================================================

def load_and_normalize(npy_path: Path) -> Optional[np.ndarray]:
    """โหลด .npy file แล้ว normalize เป็น unit vector"""
    if not npy_path.exists():
        return None
    emb = np.load(npy_path).flatten().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def fuse_embeddings(
    clip_emb: Optional[np.ndarray],
    face_emb: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Fuse CLIP + ArcFace → 1024-dim vector (concat + normalize)"""
    if clip_emb is None and face_emb is None:
        return None
    if clip_emb is None:
        clip_emb = np.zeros(CLIP_DIM, dtype=np.float32)
    if face_emb is None:
        face_emb = np.zeros(ARCFACE_DIM, dtype=np.float32)
    fused = np.concatenate([clip_emb, face_emb])
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def discover_profiles() -> List[ImageProfile]:
    """
    Scan embedding directories เพื่อค้นหา profiles ทั้งหมด
    ไฟล์ตั้งชื่อตาม pattern: {profile_id}_{platform}.npy
    เช่น: adamthede_twitter.npy → profile_id=adamthede, platform=twitter

    Returns:
        List ของ ImageProfile ที่มี embedding อย่างน้อย 1 ชนิด (CLIP หรือ ArcFace)
    """
    clip_dir = DIRS["embeddings_clip"]
    face_dir = DIRS["embeddings_face"]

    # รวม profile IDs จากทั้ง 2 directories
    seen = {}  # key = (profile_id, platform) → dict of paths
    for emb_dir, emb_type in [(clip_dir, "clip"), (face_dir, "face")]:
        if not emb_dir.exists():
            print(f"  ⚠ Directory not found: {emb_dir}")
            continue
        for npy_file in emb_dir.glob("*.npy"):
            # Parse filename: {profile_id}_{platform}.npy
            stem = npy_file.stem  # e.g. "adamthede_twitter"
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            pid, plat = parts
            key = (pid, plat)
            if key not in seen:
                seen[key] = {}
            seen[key][emb_type] = npy_file

    # สร้าง ImageProfile objects
    profiles = []
    for (pid, plat), paths in seen.items():
        clip_emb = load_and_normalize(paths["clip"]) if "clip" in paths else None
        face_emb = load_and_normalize(paths["face"]) if "face" in paths else None
        fused_emb = fuse_embeddings(clip_emb, face_emb)

        if clip_emb is None and face_emb is None:
            continue  # ข้าม profile ที่ไม่มี embedding เลย

        profiles.append(ImageProfile(
            profile_id=pid,
            platform=plat,
            clip_emb=clip_emb,
            face_emb=face_emb,
            fused_emb=fused_emb,
            has_clip=clip_emb is not None,
            has_face=face_emb is not None,
        ))

    # เรียงตาม profile_id เพื่อ reproducibility
    profiles.sort(key=lambda p: (p.profile_id, p.platform))
    return profiles


# ============================================================
# 3. SIMILARITY COMPUTATION
# ============================================================

def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """คำนวณ Cosine Similarity ระหว่าง 2 vectors (normalized แล้ว = dot product)"""
    if a is None or b is None:
        return 0.0
    return float(max(-1.0, min(1.0, np.dot(a, b))))


def l2_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """คำนวณ Euclidean (L2) distance ระหว่าง 2 vectors"""
    if a is None or b is None:
        return 2.0  # max distance สำหรับ normalized vectors
    return float(np.linalg.norm(a - b))


# ============================================================
# 4. PAIRWISE COMPARISON — เทียบ embeddings ทั้งหมดแบบ all-vs-all
#    ⬇ ขั้นตอนนี้ทำก่อน classification (ตามที่ต้องการ)
# ============================================================

def compute_pairwise_similarity(
    profiles: Sequence[ImageProfile],
) -> np.ndarray:
    """
    คำนวณ pairwise cosine similarity ของ fused embeddings ทั้งหมด
    ใช้ matrix multiplication สำหรับความเร็ว

    Returns:
        similarity matrix (N x N) dtype=float32
    """
    n = len(profiles)
    # สร้าง fused matrix
    fused_matrix = np.zeros((n, FUSED_DIM), dtype=np.float32)
    for i, p in enumerate(profiles):
        if p.fused_emb is not None:
            fused_matrix[i] = p.fused_emb

    # Cosine similarity = dot product (เพราะ vectors normalized แล้ว)
    sim_matrix = fused_matrix @ fused_matrix.T

    # Clip ค่าให้อยู่ในช่วง [-1, 1]
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)

    return sim_matrix


def build_pairwise_results(
    profiles: Sequence[ImageProfile],
    sim_matrix: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """
    สร้าง DataFrame แสดงผลลัพธ์ pairwise comparison — Top-K matches ต่อ profile

    Columns:
        source_id, source_platform, rank, target_id, target_platform,
        fused_cosine, clip_cosine, face_cosine, clip_l2, face_l2
    """
    rows = []
    n = len(profiles)

    for i in range(n):
        src = profiles[i]
        # เลือก top-K (ไม่รวมตัวเอง)
        sims = sim_matrix[i].copy()
        sims[i] = -2.0  # exclude self
        k = min(top_k, n - 1)
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        for rank, j in enumerate(top_indices, start=1):
            tgt = profiles[j]
            rows.append({
                "source_id": src.profile_id,
                "source_platform": src.platform,
                "rank": rank,
                "target_id": tgt.profile_id,
                "target_platform": tgt.platform,
                "fused_cosine": round(float(sims[j]), 6),
                "clip_cosine": round(cosine_similarity(src.clip_emb, tgt.clip_emb), 6),
                "face_cosine": round(cosine_similarity(src.face_emb, tgt.face_emb), 6),
                "clip_l2": round(l2_distance(src.clip_emb, tgt.clip_emb), 6),
                "face_l2": round(l2_distance(src.face_emb, tgt.face_emb), 6),
                "is_same_person": src.profile_id == tgt.profile_id,
            })

    return pd.DataFrame(rows)


# ============================================================
# 5. CANDIDATE SELECTION — เลือก Top-K จาก similarity matrix
# ============================================================

def select_candidates_from_matrix(
    sim_matrix: np.ndarray,
    src_idx: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    เลือก Top-K candidates สำหรับ source profile จาก similarity matrix

    Returns:
        (top_indices, top_similarities)
    """
    sims = sim_matrix[src_idx].copy()
    sims[src_idx] = -2.0  # exclude self
    k = min(k, len(sims) - 1)
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    return top_indices.astype(np.int32), sims[top_indices]


# ============================================================
# 6. FEATURE BUILDING — สร้าง features สำหรับ classification
# ============================================================

def build_features_for_pair(
    src: ImageProfile,
    tgt: ImageProfile,
) -> np.ndarray:
    """
    สร้าง feature vector สำหรับคู่ source-target (7 features)

    Features:
        0: clip_cosine    — Cosine similarity CLIP embeddings
        1: face_cosine    — Cosine similarity ArcFace embeddings
        2: fused_cosine   — Cosine similarity Fused embeddings
        3: clip_l2        — L2 distance CLIP
        4: face_l2        — L2 distance ArcFace
        5: has_both_clip  — ทั้งคู่มี CLIP embedding (1/0)
        6: has_both_face  — ทั้งคู่มี ArcFace embedding (1/0)
    """
    return np.array([
        cosine_similarity(src.clip_emb, tgt.clip_emb),
        cosine_similarity(src.face_emb, tgt.face_emb),
        cosine_similarity(src.fused_emb, tgt.fused_emb),
        l2_distance(src.clip_emb, tgt.clip_emb),
        l2_distance(src.face_emb, tgt.face_emb),
        1.0 if (src.has_clip and tgt.has_clip) else 0.0,
        1.0 if (src.has_face and tgt.has_face) else 0.0,
    ], dtype=np.float64)


def build_features_for_candidates(
    src: ImageProfile,
    candidate_indices: np.ndarray,
    all_profiles: Sequence[ImageProfile],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    สร้าง Feature matrix สำหรับ source vs ทุก candidates

    Returns:
        X (n_candidates, 7), y (n_candidates,)
    """
    n = len(candidate_indices)
    X = np.zeros((n, 7), dtype=np.float64)
    y = np.zeros(n, dtype=np.int8)

    for i, tgt_idx in enumerate(candidate_indices):
        tgt = all_profiles[int(tgt_idx)]
        X[i] = build_features_for_pair(src, tgt)
        y[i] = 1 if src.profile_id == tgt.profile_id else 0

    return X, y


# ============================================================
# 7. TRAINING — สร้าง balanced training set
# ============================================================

def build_balanced_train_set(
    profiles: Sequence[ImageProfile],
    sim_matrix: np.ndarray,
    train_indices: Sequence[int],
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    สร้าง balanced training data (positive + negative pairs)
    ใช้ผลจาก pairwise comparison เพื่อเลือก hard negatives
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for src_i in train_indices:
        src = profiles[src_i]
        cand_indices, _ = select_candidates_from_matrix(sim_matrix, src_i, k)
        X_cand, y_cand = build_features_for_candidates(src, cand_indices, profiles)

        pos_mask = y_cand == 1
        neg_mask = y_cand == 0

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Positive: เลือกตัวที่ fused_cosine สูงสุด
        pos_idx = np.where(pos_mask)[0]
        best_pos = int(pos_idx[np.argmax(X_cand[pos_idx, 2])])

        # Negative: เลือกแบบสุ่ม (hard negative จาก top-K candidates)
        neg_idx = np.where(neg_mask)[0]
        rand_neg = int(rng.choice(neg_idx))

        X_list.append(X_cand[[best_pos, rand_neg]])
        y_list.append(np.array([1, 0], dtype=np.int8))

    if not X_list:
        raise RuntimeError("ไม่สามารถสร้าง training samples ได้ — อาจไม่มี positive pairs ที่มี embeddings")

    return np.vstack(X_list).astype(np.float64), np.concatenate(y_list).astype(np.int8)


# ============================================================
# 8. EVALUATION — Top-1 accuracy + Candidate recall
# ============================================================

def evaluate_top1(
    model,
    model_name: str,
    profiles: Sequence[ImageProfile],
    sim_matrix: np.ndarray,
    test_indices: Sequence[int],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate model: Top-1 accuracy + Candidate recall"""
    rows = []
    correct = 0
    total = 0
    candidate_recall_count = 0

    for src_i in test_indices:
        src = profiles[src_i]
        cand_indices, _ = select_candidates_from_matrix(sim_matrix, src_i, k)
        X_cand, y_cand = build_features_for_candidates(src, cand_indices, profiles)

        has_true = int((y_cand == 1).any())
        candidate_recall_count += has_true

        if len(X_cand) == 0:
            continue

        probs = model.predict_proba(X_cand)[:, 1]
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        best_local = int(np.argmax(probs))
        best_tgt_idx = int(cand_indices[best_local])
        best_tgt = profiles[best_tgt_idx]
        is_correct = 1 if best_tgt.profile_id == src.profile_id else 0
        correct += is_correct
        total += 1

        rows.append({
            "model": model_name,
            "source_id": src.profile_id,
            "source_platform": src.platform,
            "predicted_id": best_tgt.profile_id,
            "predicted_platform": best_tgt.platform,
            "is_correct_top1": is_correct,
            "match_probability": round(float(probs[best_local]), 6),
            "clip_cosine": round(float(X_cand[best_local, 0]), 6),
            "face_cosine": round(float(X_cand[best_local, 1]), 6),
            "fused_cosine": round(float(X_cand[best_local, 2]), 6),
            "clip_l2": round(float(X_cand[best_local, 3]), 6),
            "face_l2": round(float(X_cand[best_local, 4]), 6),
            "true_in_topk": has_true,
        })

    acc = (correct / total) if total else 0.0
    cand_rec = (candidate_recall_count / len(test_indices)) if len(test_indices) else 0.0

    metrics = {
        "model": model_name,
        "test_size": int(len(test_indices)),
        "top1_accuracy": round(acc, 6),
        "candidate_recall_topk": round(cand_rec, 6),
    }

    return pd.DataFrame(rows), metrics


# ============================================================
# 9. PREDICT SIMILARITY — เทียบ 2 profiles
# ============================================================

def predict_similarity(
    profile_a_id: str,
    profile_b_id: str,
    platform_a: str = "twitter",
    platform_b: str = "twitter",
    model=None,
) -> Dict:
    """
    เปรียบเทียบ 2 profiles ว่าเป็นคนเดียวกันไหม

    Args:
        profile_a_id: profile_id ตัวแรก
        profile_b_id: profile_id ตัวที่สอง
        platform_a: platform ของ profile A
        platform_b: platform ของ profile B
        model: trained model (ถ้ามี จะให้ probability ด้วย)

    Returns:
        dict ที่มี similarity scores ทั้งหมด
    """
    clip_a = load_and_normalize(DIRS["embeddings_clip"] / f"{profile_a_id}_{platform_a}.npy")
    clip_b = load_and_normalize(DIRS["embeddings_clip"] / f"{profile_b_id}_{platform_b}.npy")
    face_a = load_and_normalize(DIRS["embeddings_face"] / f"{profile_a_id}_{platform_a}.npy")
    face_b = load_and_normalize(DIRS["embeddings_face"] / f"{profile_b_id}_{platform_b}.npy")
    fused_a = fuse_embeddings(clip_a, face_a)
    fused_b = fuse_embeddings(clip_b, face_b)

    result = {
        "profile_a": f"{profile_a_id} ({platform_a})",
        "profile_b": f"{profile_b_id} ({platform_b})",
        "clip_cosine": round(cosine_similarity(clip_a, clip_b), 6),
        "face_cosine": round(cosine_similarity(face_a, face_b), 6),
        "fused_cosine": round(cosine_similarity(fused_a, fused_b), 6),
        "clip_l2": round(l2_distance(clip_a, clip_b), 6),
        "face_l2": round(l2_distance(face_a, face_b), 6),
        "has_clip_a": clip_a is not None,
        "has_clip_b": clip_b is not None,
        "has_face_a": face_a is not None,
        "has_face_b": face_b is not None,
    }

    if model is not None:
        features = np.array([[
            result["clip_cosine"],
            result["face_cosine"],
            result["fused_cosine"],
            result["clip_l2"],
            result["face_l2"],
            1.0 if (clip_a is not None and clip_b is not None) else 0.0,
            1.0 if (face_a is not None and face_b is not None) else 0.0,
        ]])
        prob = model.predict_proba(features)[0, 1]
        result["match_probability"] = round(float(prob), 6)
        result["is_same_person"] = bool(prob > 0.5)
    else:
        threshold = 0.7
        result["is_same_person"] = result["fused_cosine"] > threshold

    return result


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def run_pipeline(
    output_dir: Path,
    top_k: int,
    candidate_pct: float,
    random_state: int,
) -> None:
    """
    Main pipeline:
        1. Discover profiles → โหลด embeddings ทั้งหมด
        2. Pairwise comparison → เทียบทุกคู่
        3. Candidate selection → เลือก top candidates
        4. Train classifiers → Logistic Regression + Random Forest
        5. Evaluate → Top-1 accuracy + Candidate recall
    """
    total_ts = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # STEP 1: Discover & Load all profiles
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading embeddings...")
    print("=" * 60)

    step_ts = time.time()
    profiles = discover_profiles()
    n = len(profiles)

    clip_count = sum(1 for p in profiles if p.has_clip)
    face_count = sum(1 for p in profiles if p.has_face)
    both_count = sum(1 for p in profiles if p.has_clip and p.has_face)
    platforms = set(p.platform for p in profiles)

    print(f"  Profiles found: {n}")
    print(f"  With CLIP: {clip_count}")
    print(f"  With ArcFace: {face_count}")
    print(f"  With both: {both_count}")
    print(f"  Platforms: {platforms}")
    log_stage("load_embeddings", step_ts)

    if n < 2:
        print("❌ ต้องมีอย่างน้อย 2 profiles ถึงจะเทียบได้")
        return

    # ============================================================
    # STEP 2: Pairwise Comparison — เทียบ embeddings ทั้งหมด
    #          ⬇ ทำก่อน classification
    # ============================================================
    print("\n" + "=" * 60)
    print(f"STEP 2: Pairwise comparison ({n} × {n} = {n*n:,} pairs)...")
    print("=" * 60)

    step_ts = time.time()
    sim_matrix = compute_pairwise_similarity(profiles)

    # สถิติ similarity
    upper_triangle = sim_matrix[np.triu_indices(n, k=1)]
    print(f"  Similarity stats (upper triangle, {len(upper_triangle):,} pairs):")
    print(f"    Mean:   {upper_triangle.mean():.4f}")
    print(f"    Std:    {upper_triangle.std():.4f}")
    print(f"    Min:    {upper_triangle.min():.4f}")
    print(f"    Max:    {upper_triangle.max():.4f}")
    print(f"    Median: {np.median(upper_triangle):.4f}")
    log_stage("pairwise_similarity", step_ts)

    # ============================================================
    # STEP 3: Save pairwise results (Top-K matches per profile)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"STEP 3: Building Top-{top_k} matches per profile...")
    print("=" * 60)

    step_ts = time.time()
    pairwise_df = build_pairwise_results(profiles, sim_matrix, top_k)

    pairwise_out = output_dir / "pairwise_top_matches.csv"
    pairwise_df.to_csv(pairwise_out, index=False)
    print(f"  Saved: {pairwise_out}")
    print(f"  Total rows: {len(pairwise_df):,}")

    # แสดงตัวอย่างผลลัพธ์
    top1_only = pairwise_df[pairwise_df["rank"] == 1].head(10)
    if not top1_only.empty:
        print("\n  Sample Top-1 matches:")
        for _, row in top1_only.iterrows():
            marker = "✓" if row["is_same_person"] else " "
            print(f"    [{marker}] {row['source_id']} → {row['target_id']}  "
                  f"fused={row['fused_cosine']:.4f}")
    log_stage("save_pairwise", step_ts)

    # ============================================================
    # STEP 4: Candidate selection for classification
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Preparing candidates for classification...")
    print("=" * 60)

    step_ts = time.time()
    k_candidates = max(1, int(candidate_pct * n))
    print(f"  Candidate pool: top {candidate_pct*100:.0f}% = {k_candidates} per profile")

    # Train/Test split (60/40)
    all_indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.4, random_state=random_state, shuffle=True,
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    log_stage("candidate_setup", step_ts)

    # ============================================================
    # STEP 5: Build balanced training set
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Building training set...")
    print("=" * 60)

    step_ts = time.time()
    rng = np.random.default_rng(random_state)

    try:
        X_train, y_train = build_balanced_train_set(
            profiles=profiles,
            sim_matrix=sim_matrix,
            train_indices=train_idx,
            k=k_candidates,
            rng=rng,
        )
        print(f"  Train samples: {len(X_train)} "
              f"(pos={y_train.sum()}, neg={len(y_train) - y_train.sum()})")
        log_stage("build_trainset", step_ts)
    except RuntimeError as e:
        print(f"  ⚠ {e}")
        print("  ℹ Pipeline จะสร้างเฉพาะ pairwise results (ข้าม classification)")
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE (pairwise only)")
        log_stage("total", total_ts)
        print(f"{'='*60}")
        return

    # ============================================================
    # STEP 6: Train classifiers
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training classifiers...")
    print("=" * 60)

    # Model 1: Logistic Regression
    lr_ts = time.time()
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    lr_model.fit(X_train, y_train)
    log_stage("train_logistic_regression", lr_ts)

    # Model 2: Random Forest
    rf_ts = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    log_stage("train_random_forest", rf_ts)

    # ============================================================
    # STEP 7: Evaluate
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Evaluating models...")
    print("=" * 60)

    eval_ts = time.time()
    lr_df, lr_metrics = evaluate_top1(
        model=lr_model,
        model_name="LogisticRegression",
        profiles=profiles,
        sim_matrix=sim_matrix,
        test_indices=test_idx,
        k=k_candidates,
    )
    rf_df, rf_metrics = evaluate_top1(
        model=rf_model,
        model_name="RandomForest",
        profiles=profiles,
        sim_matrix=sim_matrix,
        test_indices=test_idx,
        k=k_candidates,
    )
    log_stage("evaluation", eval_ts)

    # ============================================================
    # STEP 8: Save all results
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 8: Saving results...")
    print("=" * 60)

    save_ts = time.time()

    # Predictions
    lr_out = output_dir / "predictions_logistic_regression.csv"
    rf_out = output_dir / "predictions_random_forest.csv"
    lr_df.to_csv(lr_out, index=False)
    rf_df.to_csv(rf_out, index=False)

    # Metrics
    metrics_df = pd.DataFrame([lr_metrics, rf_metrics])
    metrics_out = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)

    # Trained model
    import joblib
    model_out = output_dir / "rf_model.joblib"
    joblib.dump(rf_model, model_out)

    log_stage("save_outputs", save_ts)

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  LogisticRegression:")
    print(f"    Top-1 Accuracy:     {lr_metrics['top1_accuracy']:.4f}")
    print(f"    Candidate Recall:   {lr_metrics['candidate_recall_topk']:.4f}")
    print(f"  RandomForest:")
    print(f"    Top-1 Accuracy:     {rf_metrics['top1_accuracy']:.4f}")
    print(f"    Candidate Recall:   {rf_metrics['candidate_recall_topk']:.4f}")
    print()
    print(f"  Output files:")
    print(f"    {pairwise_out}")
    print(f"    {lr_out}")
    print(f"    {rf_out}")
    print(f"    {metrics_out}")
    print(f"    {model_out}")
    print(f"{'='*60}")
    log_stage("TOTAL PIPELINE", total_ts)
    print(f"{'='*60}")


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Matching Pipeline: Pairwise Embedding Comparison + Classification"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/processed/image_pipeline_results)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="จำนวน Top-K matches ที่จะแสดงต่อ profile (default: 20)",
    )
    parser.add_argument(
        "--candidate-pct", type=float, default=0.10,
        help="Candidate pool percent สำหรับ classification (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--predict", nargs=2, metavar=("PROFILE_A", "PROFILE_B"),
        help="Predict similarity ระหว่าง 2 profiles",
    )
    parser.add_argument(
        "--predict-platforms", nargs=2, default=["twitter", "twitter"],
        metavar=("PLATFORM_A", "PLATFORM_B"),
        help="Platforms สำหรับ --predict (default: twitter twitter)",
    )
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        output_dir = DIRS["results"]
    else:
        output_dir = Path(args.output_dir)

    # ---------- Predict mode ----------
    if args.predict:
        pid_a, pid_b = args.predict
        plat_a, plat_b = args.predict_platforms

        # Try to load trained model
        model_path = output_dir / "rf_model.joblib"
        model = None
        if model_path.exists():
            import joblib
            model = joblib.load(model_path)
            print(f"✓ Loaded trained model: {model_path}")

        result = predict_similarity(pid_a, pid_b, plat_a, plat_b, model=model)

        print("\n" + "=" * 50)
        print("SIMILARITY PREDICTION")
        print("=" * 50)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 50)
        return

    # ---------- Full pipeline mode ----------
    print("=" * 60)
    print("IMAGE MATCHING PIPELINE")
    print("  Pairwise Embedding Comparison → Classification")
    print("  CLIP + ArcFace (fused 1024-dim)")
    print("=" * 60)

    run_pipeline(
        output_dir=output_dir,
        top_k=args.top_k,
        candidate_pct=args.candidate_pct,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
