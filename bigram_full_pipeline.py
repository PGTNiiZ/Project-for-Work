import argparse
import math
import re
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._linear_loss"
)


PLATFORMS = ("instagram", "twitter", "googleplus")


def log_stage(pair_name: str, stage: str, start_ts: float) -> None:
    elapsed = time.time() - start_ts
    print(f"[{pair_name}] {stage} done in {elapsed:.2f}s")

def norm_text(v: object) -> str:
    if pd.isna(v):
        return ""
    return " ".join(str(v).strip().lower().split())


def normalize_location(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s,.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        "bkk": "bangkok",
        "nyc": "new york city",
        "la": "los angeles",
        "sf": "san francisco",
        "uk": "united kingdom",
        "usa": "united states",
        "us": "united states",
    }
    tokens = s.split()
    tokens = [replacements.get(t, t) for t in tokens]
    return " ".join(tokens)


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


def quality_filter(
    df: pd.DataFrame, username_col: str, fullname_col: str, label: str
) -> pd.DataFrame:
    """
    Keep rows that have at least one usable name field.
    Drop only rows where both username and fullname are empty/NaN.
    """
    u = df[username_col].map(norm_text)
    n = df[fullname_col].map(norm_text)
    keep = (u != "") | (n != "")
    dropped = int((~keep).sum())
    if dropped > 0:
        print(f"[{label}] quality_filter dropped {dropped} rows (no username/fullname)")
    return df.loc[keep].copy()


@dataclass
class ProfileRow:
    profile_id: str
    username_clean: str
    fullname_clean: str
    user_grams: Set[str]
    full_grams: Set[str]
    merged_grams: Set[str]
    location_raw: str
    country_code: str
    city_name: str
    lat: float
    lon: float
    loc_confidence: float


def _to_float_or_nan(v: object) -> float:
    try:
        if pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def load_location_lookup(mapping_file: Path) -> Dict[str, Dict[str, object]]:
    mapping_df = pd.read_csv(mapping_file)
    lookup: Dict[str, Dict[str, object]] = {}

    for _, r in mapping_df.iterrows():
        canonical = {
            "canonical_id": "" if pd.isna(r.get("canonical_id")) else str(r.get("canonical_id")).strip(),
            "canonical_name": "" if pd.isna(r.get("canonical_name")) else str(r.get("canonical_name")).strip().lower(),
            "country_code": "" if pd.isna(r.get("country_code")) else str(r.get("country_code")).strip().upper(),
            "lat": _to_float_or_nan(r.get("lat")),
            "lon": _to_float_or_nan(r.get("lon")),
            "confidence": _to_float_or_nan(r.get("confidence")),
        }
        raw = "" if pd.isna(r.get("raw_location")) else str(r.get("raw_location")).strip()
        raw_norm = "" if pd.isna(r.get("raw_location_norm")) else str(r.get("raw_location_norm")).strip()

        if raw:
            lookup[f"raw::{raw}"] = canonical
        if raw_norm:
            lookup[f"norm::{raw_norm}"] = canonical

    return lookup


def resolve_location(raw_location: object, location_lookup: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    raw = "" if pd.isna(raw_location) else str(raw_location).strip()
    raw_norm = normalize_location(raw)

    row = None
    if raw:
        row = location_lookup.get(f"raw::{raw}")
    if row is None and raw_norm:
        row = location_lookup.get(f"norm::{raw_norm}")

    if row is None:
        return {
            "country_code": "",
            "city_name": "",
            "lat": float("nan"),
            "lon": float("nan"),
            "loc_confidence": 0.0,
        }

    return {
        "country_code": str(row.get("country_code", "") or ""),
        "city_name": str(row.get("canonical_name", "") or ""),
        "lat": _to_float_or_nan(row.get("lat")),
        "lon": _to_float_or_nan(row.get("lon")),
        "loc_confidence": _to_float_or_nan(row.get("confidence")),
    }


def make_source_rows(
    pair_df: pd.DataFrame, src_prefix: str, location_lookup: Dict[str, Dict[str, object]]
) -> List[ProfileRow]:
    rows: List[ProfileRow] = []
    src_user_col = f"{src_prefix}_userName_clean"
    src_full_col = (
        f"{src_prefix}_fullName_clean"
        if f"{src_prefix}_fullName_clean" in pair_df.columns
        else f"{src_prefix}_fullName"
    )
    src_loc_col = f"{src_prefix}_location"

    for _, r in pair_df.iterrows():
        pid = str(r["profile_id"])
        u = norm_text(r[src_user_col])
        n = norm_text(r[src_full_col])
        loc_raw = r[src_loc_col] if src_loc_col in pair_df.columns else ""
        loc = resolve_location(loc_raw, location_lookup)
        rows.append(
            ProfileRow(
                profile_id=pid,
                username_clean=u,
                fullname_clean=n,
                user_grams=to_bigrams(u),
                full_grams=to_bigrams(n),
                merged_grams=to_bigrams((u + " " + n).strip()),
                location_raw="" if pd.isna(loc_raw) else str(loc_raw),
                country_code=loc["country_code"],
                city_name=loc["city_name"],
                lat=loc["lat"],
                lon=loc["lon"],
                loc_confidence=0.0 if pd.isna(loc["loc_confidence"]) else float(loc["loc_confidence"]),
            )
        )
    return rows


def make_target_rows(
    all_df: pd.DataFrame, tgt_prefix: str, location_lookup: Dict[str, Dict[str, object]]
) -> List[ProfileRow]:
    tgt_df = all_df[all_df["platform"].str.lower() == tgt_prefix].copy()
    rows: List[ProfileRow] = []
    for _, r in tgt_df.iterrows():
        pid = str(r["profile_id"])
        u = norm_text(r["userName_clean"])
        n = norm_text(r["fullName_clean"])
        loc_raw = r["location"] if "location" in tgt_df.columns else ""
        loc = resolve_location(loc_raw, location_lookup)
        rows.append(
            ProfileRow(
                profile_id=pid,
                username_clean=u,
                fullname_clean=n,
                user_grams=to_bigrams(u),
                full_grams=to_bigrams(n),
                merged_grams=to_bigrams((u + " " + n).strip()),
                location_raw="" if pd.isna(loc_raw) else str(loc_raw),
                country_code=loc["country_code"],
                city_name=loc["city_name"],
                lat=loc["lat"],
                lon=loc["lon"],
                loc_confidence=0.0 if pd.isna(loc["loc_confidence"]) else float(loc["loc_confidence"]),
            )
        )
    return rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")

    r = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


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
    location_conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu = np.zeros(len(selected_idxs), dtype=np.float64)
    un = np.zeros(len(selected_idxs), dtype=np.float64)
    same_country = np.zeros(len(selected_idxs), dtype=np.float64)
    same_city = np.zeros(len(selected_idxs), dtype=np.float64)
    geo_distance_km = np.zeros(len(selected_idxs), dtype=np.float64)
    loc_conf_src = np.zeros(len(selected_idxs), dtype=np.float64)
    loc_conf_tgt = np.zeros(len(selected_idxs), dtype=np.float64)
    loc_conf_min = np.zeros(len(selected_idxs), dtype=np.float64)
    y = np.zeros(len(selected_idxs), dtype=np.int8)

    for i, tgt_idx in enumerate(selected_idxs):
        tgt = target_rows[int(tgt_idx)]
        uu[i] = jaccard(src.user_grams, tgt.user_grams)
        un[i] = jaccard(src.full_grams, tgt.full_grams)
        src_conf = max(0.0, min(1.0, src.loc_confidence))
        tgt_conf = max(0.0, min(1.0, tgt.loc_confidence))
        conf_min = min(src_conf, tgt_conf)
        loc_conf_src[i] = src_conf
        loc_conf_tgt[i] = tgt_conf
        loc_conf_min[i] = conf_min

        loc_gate = conf_min >= location_conf_threshold
        same_country[i] = (
            1.0 if loc_gate and src.country_code and src.country_code == tgt.country_code else 0.0
        )
        same_city[i] = 1.0 if loc_gate and src.city_name and src.city_name == tgt.city_name else 0.0

        if loc_gate:
            d = haversine_km(src.lat, src.lon, tgt.lat, tgt.lon)
            # Fill missing distance with large value and clip outliers for numerical stability.
            geo_distance_km[i] = 5000.0 if pd.isna(d) else min(float(d), 5000.0)
        else:
            geo_distance_km[i] = 5000.0
        y[i] = 1 if src.profile_id == tgt.profile_id else 0

    X = np.column_stack(
        [uu, un, ub_sims, same_country, same_city, geo_distance_km, loc_conf_src, loc_conf_tgt, loc_conf_min]
    ).astype(np.float64)
    return X, y, selected_idxs


def build_balanced_train_set(
    src_rows: Sequence[ProfileRow],
    target_rows: Sequence[ProfileRow],
    candidates_cache: Sequence[Tuple[np.ndarray, np.ndarray]],
    train_indices: Sequence[int],
    rng: np.random.Generator,
    location_conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for src_i in train_indices:
        src = src_rows[src_i]
        selected_idxs, ub_sims = candidates_cache[src_i]
        Xc, yc, _ = build_features_for_candidates(
            src, selected_idxs, ub_sims, target_rows, location_conf_threshold
        )

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

    X_train = np.vstack(X_list).astype(np.float64)
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
    location_conf_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    correct = 0
    total = 0
    candidate_recall = 0

    for src_i in test_indices:
        src = src_rows[src_i]
        selected_idxs, ub_sims = candidates_cache[src_i]
        Xc, yc, _ = build_features_for_candidates(
            src, selected_idxs, ub_sims, target_rows, location_conf_threshold
        )

        has_true_candidate = int((yc == 1).any())
        candidate_recall += has_true_candidate
        if len(Xc) == 0:
            continue

        probs = model.predict_proba(Xc)[:, 1]
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
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
                "same_country": int(Xc[best_local, 3]),
                "same_city": int(Xc[best_local, 4]),
                "geo_distance_km": round(float(Xc[best_local, 5]), 6),
                "loc_conf_src": round(float(Xc[best_local, 6]), 6),
                "loc_conf_tgt": round(float(Xc[best_local, 7]), 6),
                "loc_conf_min": round(float(Xc[best_local, 8]), 6),
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
    location_mapping_file: Path,
    output_dir: Path,
    cluster_pct: float,
    random_state: int,
    location_conf_threshold: float,
) -> List[Dict[str, float]]:
    stage_ts = time.time()
    pair_df = pd.read_csv(pair_file)
    all_df = pd.read_csv(
        all_profiles, usecols=["profile_id", "platform", "userName_clean", "fullName_clean", "location"]
    )
    all_df["platform"] = all_df["platform"].astype(str).str.lower()
    location_lookup = load_location_lookup(location_mapping_file)
    log_stage(pair_file.stem, "load_data", stage_ts)

    src_prefix, tgt_prefix = infer_pair_prefixes(pair_df)
    pair_name = pair_file.stem

    src_user_col = f"{src_prefix}_userName_clean"
    src_full_col = (
        f"{src_prefix}_fullName_clean"
        if f"{src_prefix}_fullName_clean" in pair_df.columns
        else f"{src_prefix}_fullName"
    )
    pair_df = quality_filter(
        pair_df, username_col=src_user_col, fullname_col=src_full_col, label=f"{pair_name}:source"
    )

    src_rows = make_source_rows(pair_df, src_prefix, location_lookup)
    all_df_tgt = all_df[all_df["platform"].str.lower() == tgt_prefix].copy()
    all_df_tgt = quality_filter(
        all_df_tgt,
        username_col="userName_clean",
        fullname_col="fullName_clean",
        label=f"{pair_name}:target",
    )
    # Reuse builder on filtered target frame
    target_rows = make_target_rows(all_df_tgt, tgt_prefix, location_lookup)
    if not target_rows:
        raise RuntimeError(f"No target rows for platform={tgt_prefix} in {all_profiles}")
    log_stage(pair_name, "prepare_rows", stage_ts)

    tgt_merged_sets = [r.merged_grams for r in target_rows]
    tgt_inv_merged = build_index(tgt_merged_sets)
    all_target_indices = list(range(len(target_rows)))
    k = max(1, math.ceil(cluster_pct * len(target_rows)))

    # Candidate profile selection (paper's clustering stage)
    cand_ts = time.time()
    candidates_cache: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, src in enumerate(src_rows, start=1):
        selected_idxs, ub_sims = top_k_candidates_by_ub(
            src_merged_grams=src.merged_grams,
            tgt_merged_sets=tgt_merged_sets,
            tgt_inv_merged=tgt_inv_merged,
            k=k,
            all_target_indices=all_target_indices,
        )
        candidates_cache.append((selected_idxs, ub_sims))
        if i % 1000 == 0:
            print(f"[{pair_name}] candidate_progress {i}/{len(src_rows)}")
    log_stage(pair_name, "candidate_selection", cand_ts)

    # 60-40 split by source profiles (paper setting)
    src_indices = np.arange(len(src_rows))
    train_idx, test_idx = train_test_split(
        src_indices, test_size=0.4, random_state=random_state, shuffle=True
    )

    rng = np.random.default_rng(random_state)
    trainset_ts = time.time()
    X_train, y_train = build_balanced_train_set(
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        train_indices=train_idx,
        rng=rng,
        location_conf_threshold=location_conf_threshold,
    )
    log_stage(pair_name, "build_trainset", trainset_ts)

    # Model 1: stable linear model (SGD-equivalent role in pipeline).
    train_ts = time.time()
    sgd = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    sgd.fit(X_train, y_train)
    sgd_model_name = "Linear"
    log_stage(pair_name, "train_linear", train_ts)

    # Model 2: Random Forest (paper comparison baseline)
    rf_ts = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    log_stage(pair_name, "train_rf", rf_ts)

    eval_sgd_ts = time.time()
    sgd_df, sgd_metrics = evaluate_top1(
        model=sgd,
        model_name=sgd_model_name,
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        test_indices=test_idx,
        pair_name=pair_name,
        location_conf_threshold=location_conf_threshold,
    )
    log_stage(pair_name, "eval_linear", eval_sgd_ts)
    eval_rf_ts = time.time()
    rf_df, rf_metrics = evaluate_top1(
        model=rf,
        model_name="RF",
        src_rows=src_rows,
        target_rows=target_rows,
        candidates_cache=candidates_cache,
        test_indices=test_idx,
        pair_name=pair_name,
        location_conf_threshold=location_conf_threshold,
    )
    log_stage(pair_name, "eval_rf", eval_rf_ts)

    save_ts = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    sgd_out = output_dir / f"{pair_name}_top1_predictions_sgd.csv"
    rf_out = output_dir / f"{pair_name}_top1_predictions_rf.csv"
    sgd_df.to_csv(sgd_out, index=False)
    rf_df.to_csv(rf_out, index=False)

    metrics_df = pd.DataFrame([sgd_metrics, rf_metrics])
    metrics_out = output_dir / f"{pair_name}_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    log_stage(pair_name, "save_outputs", save_ts)

    print(
        f"[{pair_name}] source={src_prefix}, target={tgt_prefix}, top_k={k}, "
        f"loc_conf_threshold={location_conf_threshold}"
    )
    print(f"[{pair_name}] train_samples={len(X_train)}, test_profiles={len(test_idx)}")
    print(f"[{pair_name}] {sgd_model_name} top1={sgd_metrics['top1_accuracy']:.4f}, candidate_recall={sgd_metrics['candidate_recall_topk']:.4f}")
    print(f"[{pair_name}] RF  top1={rf_metrics['top1_accuracy']:.4f}, candidate_recall={rf_metrics['candidate_recall_topk']:.4f}")
    print(f"[{pair_name}] saved: {sgd_out}")
    print(f"[{pair_name}] saved: {rf_out}")
    print(f"[{pair_name}] saved: {metrics_out}")

    return [sgd_metrics, rf_metrics]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-aligned bigram pipeline: candidate selection + linear/RF matching"
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
        "--location-mapping-file",
        default="data/processed/location_mapping.csv",
        help="Location mapping file generated by location_mapping_pipeline.py",
    )
    parser.add_argument(
        "--cluster-pct",
        type=float,
        default=0.10,
        help="Top candidate percent used in bigram clustering stage (default: 0.10)",
    )
    parser.add_argument(
        "--location-conf-threshold",
        type=float,
        default=0.75,
        help="Use location features only when min(src_conf, tgt_conf) >= threshold",
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
    location_mapping_file = Path(args.location_mapping_file)
    output_dir = Path(args.output_dir)
    all_metrics: List[Dict[str, float]] = []

    for pair in args.pair_files:
        pair_path = Path(pair)
        all_metrics.extend(
            run_pair_pipeline(
                pair_file=pair_path,
                all_profiles=all_profiles,
                location_mapping_file=location_mapping_file,
                output_dir=output_dir,
                cluster_pct=args.cluster_pct,
                random_state=args.random_state,
                location_conf_threshold=args.location_conf_threshold,
            )
        )

    all_metrics_df = pd.DataFrame(all_metrics)
    summary_out = output_dir / "bigram_pipeline_metrics_summary.csv"
    all_metrics_df.to_csv(summary_out, index=False)
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()
