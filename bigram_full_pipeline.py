import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PLATFORMS = ("instagram", "twitter", "googleplus")

def norm_text(v: object) -> str:
    if pd.isna(v):
        return ""
    return " ".join(str(v).strip().lower().split())


def to_bigrams(s: str) -> Set[str]:
    s = norm_text(s)
    if not s:
        return set()
    if len(s) < 2:
        return {s}
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def build_index(gram_sets: Sequence[Set[str]]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {}
    for i, grams in enumerate(gram_sets):
        for g in grams:
            idx.setdefault(g, []).append(i)
    return idx


def infer_pair_prefixes(df: pd.DataFrame) -> Tuple[str, str]:
    found = []
    for c in df.columns:
        for p in PLATFORMS:
            if c.startswith(f"{p}_userName_clean") and p not in found:
                found.append(p)
    if len(found) != 2:
        raise ValueError(f"Expected 2 platform prefixes, found: {found}")
    return found[0], found[1]


@dataclass
class ProfileRow:
    profile_id: str
    username_clean: str
    fullname_clean: str
    user_grams: Set[str]
    full_grams: Set[str]
    merged_grams: Set[str]


def make_source_rows(pair_df: pd.DataFrame, src_prefix: str) -> List[ProfileRow]:
    rows: List[ProfileRow] = []
    src_user_col = f"{src_prefix}_userName_clean"
    src_full_col = (
        f"{src_prefix}_fullName_clean"
        if f"{src_prefix}_fullName_clean" in pair_df.columns
        else f"{src_prefix}_fullName"
    )

    for _, r in pair_df.iterrows():
        pid = str(r["profile_id"])
        u = norm_text(r[src_user_col])
        n = norm_text(r[src_full_col])
        rows.append(
            ProfileRow(
                profile_id=pid,
                username_clean=u,
                fullname_clean=n,
                user_grams=to_bigrams(u),
                full_grams=to_bigrams(n),
                merged_grams=to_bigrams((u + " " + n).strip()),
            )
        )
    return rows


def make_target_rows(all_df: pd.DataFrame, tgt_prefix: str) -> List[ProfileRow]:
    tgt_df = all_df[all_df["platform"].str.lower() == tgt_prefix].copy()
    rows: List[ProfileRow] = []
    for _, r in tgt_df.iterrows():
        pid = str(r["profile_id"])
        u = norm_text(r["userName_clean"])
        n = norm_text(r["fullName_clean"])
        rows.append(
            ProfileRow(
                profile_id=pid,
                username_clean=u,
                fullname_clean=n,
                user_grams=to_bigrams(u),
                full_grams=to_bigrams(n),
                merged_grams=to_bigrams((u + " " + n).strip()),
            )
        )
    return rows


def top_k_candidates_by_ub(
    src_merged_grams: Set[str],
    tgt_merged_sets: Sequence[Set[str]],
    tgt_inv_merged: Dict[str, List[int]],
    k: int,
    all_target_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    overlap: Dict[int, int] = {}
    for g in src_merged_grams:
        for cand in tgt_inv_merged.get(g, []):
            overlap[cand] = overlap.get(cand, 0) + 1

    # rank by UB Jaccard; only non-zero overlaps first
    scored = []
    src_size = len(src_merged_grams)
    for cand, inter in overlap.items():
        union = src_size + len(tgt_merged_sets[cand]) - inter
        sim = 0.0 if union <= 0 else inter / union
        scored.append((cand, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [i for i, _ in scored[:k]]
    selected_set = set(selected)

    if len(selected) < k:
        for cand in all_target_indices:
            if cand not in selected_set:
                selected.append(cand)
                selected_set.add(cand)
                if len(selected) >= k:
                    break

    selected = selected[:k]
    selected_arr = np.array(selected, dtype=np.int32)
    ub_sims = np.array(
        [jaccard(src_merged_grams, tgt_merged_sets[i]) for i in selected_arr],
        dtype=np.float32,
    )
    return selected_arr, ub_sims


def build_features_for_candidates(
    src: ProfileRow,
    selected_idxs: np.ndarray,
    ub_sims: np.ndarray,
    target_rows: Sequence[ProfileRow],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu = np.zeros(len(selected_idxs), dtype=np.float32)
    un = np.zeros(len(selected_idxs), dtype=np.float32)
    y = np.zeros(len(selected_idxs), dtype=np.int8)

    for i, tgt_idx in enumerate(selected_idxs):
        tgt = target_rows[int(tgt_idx)]
        uu[i] = jaccard(src.user_grams, tgt.user_grams)
        un[i] = jaccard(src.full_grams, tgt.full_grams)
        y[i] = 1 if src.profile_id == tgt.profile_id else 0

    X = np.column_stack([uu, un, ub_sims]).astype(np.float32)
    return X, y, selected_idxs


def build_balanced_train_set(
    src_rows: Sequence[ProfileRow],
    target_rows: Sequence[ProfileRow],
    candidates_cache: Sequence[Tuple[np.ndarray, np.ndarray]],
    train_indices: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for src_i in train_indices:
        src = src_rows[src_i]
        selected_idxs, ub_sims = candidates_cache[src_i]
        Xc, yc, _ = build_features_for_candidates(src, selected_idxs, ub_sims, target_rows)

        pos_idx = np.where(yc == 1)[0]
        neg_idx = np.where(yc == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        # Pick one strong positive and one random negative (balanced like paper setup)
        p = int(pos_idx[np.argmax(Xc[pos_idx, 2])])
        n = int(rng.choice(neg_idx))

        X_list.append(Xc[[p, n]])
        y_list.append(np.array([1, 0], dtype=np.int8))

    if not X_list:
        raise RuntimeError("No balanced training samples were generated.")

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.concatenate(y_list).astype(np.int8)
    return X_train, y_train


def evaluate_top1(
    model,
    model_name: str,
    src_rows: Sequence[ProfileRow],
    target_rows: Sequence[ProfileRow],
    candidates_cache: Sequence[Tuple[np.ndarray, np.ndarray]],
    test_indices: Sequence[int],
    pair_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    correct = 0
    total = 0
    candidate_recall = 0

    for src_i in test_indices:
        src = src_rows[src_i]
        selected_idxs, ub_sims = candidates_cache[src_i]
        Xc, yc, _ = build_features_for_candidates(src, selected_idxs, ub_sims, target_rows)

        has_true_candidate = int((yc == 1).any())
        candidate_recall += has_true_candidate
        if len(Xc) == 0:
            continue

        probs = model.predict_proba(Xc)[:, 1]
        best_local = int(np.argmax(probs))
        best_tgt_idx = int(selected_idxs[best_local])
        best_tgt = target_rows[best_tgt_idx]
        is_correct = 1 if best_tgt.profile_id == src.profile_id else 0
        correct += is_correct
        total += 1

        rows.append(
            {
                "pair_name": pair_name,
                "model": model_name,
                "source_profile_id": src.profile_id,
                "predicted_profile_id": best_tgt.profile_id,
                "is_correct_top1": is_correct,
                "predicted_match_probability": round(float(probs[best_local]), 6),
                "uu_sim": round(float(Xc[best_local, 0]), 6),
                "un_sim": round(float(Xc[best_local, 1]), 6),
                "ub_sim": round(float(Xc[best_local, 2]), 6),
                "true_candidate_in_topk": has_true_candidate,
            }
        )

    acc = (correct / total) if total else 0.0
    cand_rec = (candidate_recall / len(test_indices)) if len(test_indices) else 0.0
    metrics = {
        "pair_name": pair_name,
        "model": model_name,
        "test_size": int(len(test_indices)),
        "top1_accuracy": round(acc, 6),
        "candidate_recall_topk": round(cand_rec, 6),
    }

    return pd.DataFrame(rows), metrics


def run_pair_pipeline(
    pair_file: Path,
    all_profiles: Path,
    output_dir: Path,
    cluster_pct: float,
    random_state: int,
) -> List[Dict[str, float]]:
    pair_df = pd.read_csv(pair_file)
    all_df = pd.read_csv(all_profiles, usecols=["profile_id", "platform", "userName_clean", "fullName_clean"])
    all_df["platform"] = all_df["platform"].astype(str).str.lower()

    src_prefix, tgt_prefix = infer_pair_prefixes(pair_df)
    pair_name = pair_file.stem

    src_rows = make_source_rows(pair_df, src_prefix)
    target_rows = make_target_rows(all_df, tgt_prefix)
    if not target_rows:
        raise RuntimeError(f"No target rows for platform={tgt_prefix} in {all_profiles}")

    tgt_merged_sets = [r.merged_grams for r in target_rows]
    tgt_inv_merged = build_index(tgt_merged_sets)
    all_target_indices = list(range(len(target_rows)))
    k = max(1, math.ceil(cluster_pct * len(target_rows)))

    # Candidate profile selection (paper's clustering stage)
    candidates_cache: List[Tuple[np.ndarray, np.ndarray]] = []
    for src in src_rows:
        selected_idxs, ub_sims = top_k_candidates_by_ub(
            src_merged_grams=src.merged_grams,
            tgt_merged_sets=tgt_merged_sets,
            tgt_inv_merged=tgt_inv_merged,
            k=k,
            all_target_indices=all_target_indices,
        )
        candidates_cache.append((selected_idxs, ub_sims))

    # 60-40 split by source profiles (paper setting)
    src_indices = np.arange(len(src_rows))
    train_idx, test_idx = train_test_split(
        src_indices, test_size=0.4, random_state=random_state, shuffle=True
    )

    rng = np.random.default_rng(random_state)
    X_train, y_train = build_balanced_train_set(
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        train_indices=train_idx,
        rng=rng,
    )

    # Model 1: SGD (as in paper)
    sgd = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-4,
            random_state=random_state,
        ),
    )
    sgd.fit(X_train, y_train)

    # Model 2: Random Forest (paper comparison baseline)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    sgd_df, sgd_metrics = evaluate_top1(
        model=sgd,
        model_name="SGD",
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        test_indices=test_idx,
        pair_name=pair_name,
    )
    rf_df, rf_metrics = evaluate_top1(
        model=rf,
        model_name="RF",
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        test_indices=test_idx,
        pair_name=pair_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sgd_out = output_dir / f"{pair_name}_top1_predictions_sgd.csv"
    rf_out = output_dir / f"{pair_name}_top1_predictions_rf.csv"
    sgd_df.to_csv(sgd_out, index=False)
    rf_df.to_csv(rf_out, index=False)

    metrics_df = pd.DataFrame([sgd_metrics, rf_metrics])
    metrics_out = output_dir / f"{pair_name}_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    print(f"[{pair_name}] source={src_prefix}, target={tgt_prefix}, top_k={k}")
    print(f"[{pair_name}] train_samples={len(X_train)}, test_profiles={len(test_idx)}")
    print(f"[{pair_name}] SGD top1={sgd_metrics['top1_accuracy']:.4f}, candidate_recall={sgd_metrics['candidate_recall_topk']:.4f}")
    print(f"[{pair_name}] RF  top1={rf_metrics['top1_accuracy']:.4f}, candidate_recall={rf_metrics['candidate_recall_topk']:.4f}")
    print(f"[{pair_name}] saved: {sgd_out}")
    print(f"[{pair_name}] saved: {rf_out}")
    print(f"[{pair_name}] saved: {metrics_out}")

    return [sgd_metrics, rf_metrics]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-aligned bigram pipeline: candidate selection + SGD/RF matching"
    )
    parser.add_argument(
        "--all-profiles",
        default="data/processed/all_profiles_cleaned.csv",
        help="Master cleaned profile file",
    )
    parser.add_argument(
        "--pair-files",
        nargs="+",
        default=[
            "data/processed/pairs_instagram_googleplus.csv",
            "data/processed/pairs_twitter_googleplus.csv",
            "data/processed/pairs_twitter_instagram.csv",
        ],
        help="Pair files",
    )
    parser.add_argument(
        "--cluster-pct",
        type=float,
        default=0.10,
        help="Top candidate percent used in bigram clustering stage (default: 0.10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/bigram_pipeline_results",
        help="Output directory",
    )
    args = parser.parse_args()

    all_profiles = Path(args.all_profiles)
    output_dir = Path(args.output_dir)
    all_metrics: List[Dict[str, float]] = []

    for pair in args.pair_files:
        pair_path = Path(pair)
        all_metrics.extend(
            run_pair_pipeline(
                pair_file=pair_path,
                all_profiles=all_profiles,
                output_dir=output_dir,
                cluster_pct=args.cluster_pct,
                random_state=args.random_state,
            )
        )

    all_metrics_df = pd.DataFrame(all_metrics)
    summary_out = output_dir / "bigram_pipeline_metrics_summary.csv"
    all_metrics_df.to_csv(summary_out, index=False)
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()
