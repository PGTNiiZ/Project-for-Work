from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROFILES_PATH = PROJECT_ROOT / "data_for_project" / "normalized_profiles_with_profile_id.csv"

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "reports"
SBERT_EMBEDDINGS_PATH = ARTIFACTS_DIR / "sbert_embeddings.npy"
SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_LOCAL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / "e8c3b32edf5434bc2275fc9bab85f82640a19130"

KEY_COLS = ["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name", "profile_id"]
PLATFORM_PAIR_CODES = {
    tuple(sorted(["googleplus", "instagram"])): 0,
    tuple(sorted(["googleplus", "twitter"])): 1,
    tuple(sorted(["instagram", "twitter"])): 2,
}


def ensure_dirs() -> None:
    for path in [SCRIPT_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def parse_list_field(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = eval(text, {"__builtins__": {}})
            return [clean_text(item) for item in parsed if clean_text(item)]
        except Exception:
            return [text]
    return [text]


def parse_urls(value: object) -> set[str]:
    urls = set()
    for item in parse_list_field(value):
        normalized = clean_text(item).lower().rstrip("/")
        if normalized:
            urls.add(normalized)
    return urls


def extract_domains(urls: set[str]) -> set[str]:
    domains = set()
    for url in urls:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.netloc.lower().replace("www.", "")
        if host:
            domains.add(host)
    return domains


def parse_mentions(value: object) -> set[str]:
    text = lower_text(value)
    if not text:
        return set()
    return {part.strip().lstrip("@") for part in re.split(r"\s*\|\s*|\s+", text) if part.strip()}


def extract_hashtags(text: str) -> set[str]:
    return {tag.lower() for tag in re.findall(r"#(\w+)", text or "")}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def biolen_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return punct / max(len(text), 1)


def avg_word_len(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_PATH, keep_default_na=False, low_memory=False)
    df["userName_norm"] = df["userName"].map(lower_text)
    df["fullName_norm"] = df["fullName"].map(lower_text)
    df["bio_norm"] = df["bio"].map(lower_text)
    df["location_norm"] = df["location"].map(lower_text)
    df["bio_mentions_set"] = df["bio_mentions"].map(parse_mentions)
    df["hashtags_set"] = df["bio_norm"].map(extract_hashtags)
    df["urls_set"] = df["bio_urls"].map(parse_urls)
    df["domains_set"] = df["urls_set"].map(extract_domains)
    df["caps_ratio"] = df["bio"].map(lambda x: sum(1 for ch in clean_text(x) if ch.isupper()) / max(len(clean_text(x)), 1))
    df["avg_word_len"] = df["bio_norm"].map(avg_word_len)
    df["punct_ratio"] = df["bio_norm"].map(punct_ratio)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df["profile_id"] = pd.to_numeric(df["profile_id"], errors="coerce").astype("Int64")
    return df


def build_ground_truth_pairs(profiles: pd.DataFrame) -> pd.DataFrame:
    df = profiles[profiles["profile_id"].notna()].copy()
    rows = []
    for pid, group in df.groupby("profile_id", sort=True):
        recs = group[["profile_row_id", "platform"]].to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i]["platform"] == recs[j]["platform"]:
                    continue
                a = int(recs[i]["profile_row_id"])
                b = int(recs[j]["profile_row_id"])
                rows.append({
                    "profile_row_id_a": min(a, b),
                    "profile_row_id_b": max(a, b),
                    "profile_id": int(pid),
                    "label": 1,
                    "pair_type": "positive",
                })
    gt = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return gt


def save_ground_truth_matrix(profiles: pd.DataFrame, gt_pairs: pd.DataFrame) -> None:
    valid = profiles[profiles["profile_id"].notna()].copy()
    n = len(valid)
    rid_to_pos = {int(rid): idx for idx, rid in enumerate(valid["profile_row_id"].astype(int).tolist())}
    row_idx = []
    col_idx = []
    for row in gt_pairs[["profile_row_id_a", "profile_row_id_b"]].itertuples(index=False):
        a = rid_to_pos[int(row.profile_row_id_a)]
        b = rid_to_pos[int(row.profile_row_id_b)]
        row_idx.extend([a, b])
        col_idx.extend([b, a])
    data = np.ones(len(row_idx), dtype=np.uint8)
    matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    sparse.save_npz(ARTIFACTS_DIR / "ground_truth_matrix.npz", matrix)
    valid[["profile_row_id", "profile_id", "userName", "platform", "user_folder"]].to_parquet(
        ARTIFACTS_DIR / "ground_truth_profiles.parquet", index=False
    )


def assign_splits(profiles: pd.DataFrame, seed: int) -> dict[int, str]:
    stats = (
        profiles[profiles["profile_id"].notna()]
        .groupby("profile_id")
        .agg(profile_count=("profile_row_id", "size"), platform_count=("platform", "nunique"))
        .reset_index()
    )
    stats["positive_pair_count"] = stats["platform_count"].map({1: 0, 2: 1, 3: 3}).fillna(0) + (stats["profile_count"] == 4).astype(int) * 2
    stats = stats.sort_values(["positive_pair_count", "profile_count", "profile_id"], ascending=[False, False, True]).reset_index(drop=True)
    rng = random.Random(seed)
    split_targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    totals = {k: 0 for k in split_targets}
    target_total = max(stats["positive_pair_count"].sum(), 1)
    mapping: dict[int, str] = {}
    for row in stats.itertuples(index=False):
        choices = list(split_targets.keys())
        rng.shuffle(choices)
        chosen = max(
            choices,
            key=lambda split: (split_targets[split] * target_total - totals[split], -totals[split]),
        )
        mapping[int(row.profile_id)] = chosen
        totals[chosen] += int(row.positive_pair_count)
    return mapping


def generate_negatives(
    split_name: str,
    split_profiles: pd.DataFrame,
    positive_pair_set: set[tuple[int, int]],
    random_neg_ratio: float,
    hard_neg_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos_count = 0
    if "profile_id" in split_profiles.columns:
        counts = split_profiles.groupby("profile_id")["platform"].nunique()
        pos_count = int((counts == 2).sum() + (counts == 3).sum() * 3)
    recs = split_profiles[["profile_row_id", "platform", "profile_id", "userName_norm", "fullName_norm"]].to_dict("records")
    rng = random.Random(seed)
    random_target = int(pos_count * random_neg_ratio)
    hard_target = int(pos_count * hard_neg_ratio)

    random_pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(random_pairs) < random_target and attempts < max(random_target * 50, 10000):
        a = recs[rng.randrange(len(recs))]
        b = recs[rng.randrange(len(recs))]
        attempts += 1
        if a["profile_row_id"] == b["profile_row_id"] or a["platform"] == b["platform"] or a["profile_id"] == b["profile_id"]:
            continue
        pair = tuple(sorted((int(a["profile_row_id"]), int(b["profile_row_id"]))))
        if pair in positive_pair_set:
            continue
        random_pairs.add(pair)

    buckets: defaultdict[str, list[dict]] = defaultdict(list)
    for rec in recs:
        uname = rec["userName_norm"]
        for key in {uname[:1], uname[:2], uname[:3]}:
            if key:
                buckets[key].append(rec)
    hard_candidates: list[tuple[float, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for bucket_recs in buckets.values():
        bucket_recs = sorted(bucket_recs, key=lambda item: item["userName_norm"])[:120]
        for i in range(len(bucket_recs)):
            for j in range(i + 1, len(bucket_recs)):
                a = bucket_recs[i]
                b = bucket_recs[j]
                if a["platform"] == b["platform"] or a["profile_id"] == b["profile_id"]:
                    continue
                pair = tuple(sorted((int(a["profile_row_id"]), int(b["profile_row_id"]))))
                if pair in positive_pair_set or pair in random_pairs or pair in seen:
                    continue
                score = max(
                    fuzz.ratio(a["userName_norm"], b["userName_norm"]) / 100.0,
                    fuzz.ratio(a["fullName_norm"], b["fullName_norm"]) / 100.0,
                )
                if score < 0.72:
                    continue
                seen.add(pair)
                hard_candidates.append((score, pair))
    hard_candidates.sort(reverse=True)
    hard_pairs = hard_candidates[:hard_target]

    random_df = pd.DataFrame(
        [{"profile_row_id_a": a, "profile_row_id_b": b, "label": 0, "pair_type": "random_negative", "split_name": split_name}
         for a, b in sorted(random_pairs)]
    )
    hard_df = pd.DataFrame(
        [{"profile_row_id_a": a, "profile_row_id_b": b, "label": 0, "pair_type": "hard_negative", "split_name": split_name, "hard_score": score}
         for score, (a, b) in hard_pairs]
    )
    return random_df, hard_df


def make_profile_bundle(profiles: pd.DataFrame) -> dict[str, object]:
    subset = profiles[profiles["profile_id"].notna()].copy()
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(subset["bio_norm"].fillna(""))
    if SBERT_EMBEDDINGS_PATH.exists():
        sbert_embeddings = np.load(SBERT_EMBEDDINGS_PATH)
    else:
        model_path = str(SBERT_LOCAL_PATH if SBERT_LOCAL_PATH.exists() else SBERT_MODEL_NAME)
        model = SentenceTransformer(model_path, local_files_only=True)
        sbert_embeddings = model.encode(
            subset["bio_norm"].fillna("").tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.save(SBERT_EMBEDDINGS_PATH, sbert_embeddings)
    idx_map = {int(rid): idx for idx, rid in enumerate(subset["profile_row_id"].astype(int).tolist())}
    lookup = {}
    for record in subset.to_dict("records"):
        rid = int(record["profile_row_id"])
        record["_tfidf_idx"] = idx_map[rid]
        lookup[rid] = record
    return {
        "records": lookup,
        "tfidf_matrix": tfidf,
        "tfidf_vectorizer": vectorizer,
        "sbert_embeddings": sbert_embeddings,
    }


def compute_features(pairs: pd.DataFrame, profile_bundle: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    records = profile_bundle["records"]
    tfidf_matrix = profile_bundle["tfidf_matrix"]
    sbert_embeddings = profile_bundle["sbert_embeddings"]
    rows = []
    for row in pairs.itertuples(index=False):
        a = records[int(row.profile_row_id_a)]
        b = records[int(row.profile_row_id_b)]
        user_a, user_b = a["userName_norm"], b["userName_norm"]
        full_a, full_b = a["fullName_norm"], b["fullName_norm"]
        loc_a, loc_b = a["location_norm"], b["location_norm"]
        max_user_len = max(len(user_a), len(user_b), 1)
        max_full_len = max(len(full_a), len(full_b), 1)
        bio_cosine = float(tfidf_matrix[a["_tfidf_idx"]].multiply(tfidf_matrix[b["_tfidf_idx"]]).sum())
        bio_sbert_cosine = float(np.dot(sbert_embeddings[a["_tfidf_idx"]], sbert_embeddings[b["_tfidf_idx"]]))
        platform_pair = PLATFORM_PAIR_CODES.get(tuple(sorted([a["platform"], b["platform"]])), -1)
        rows.append({
            "profile_row_id_a": int(row.profile_row_id_a),
            "profile_row_id_b": int(row.profile_row_id_b),
            "label": int(row.label),
            "pair_type": row.pair_type,
            "split_name": row.split_name,
            "profile_id": int(a["profile_id"]) if a["profile_id"] == b["profile_id"] else -1,
            "username_jaro": JaroWinkler.normalized_similarity(user_a, user_b),
            "username_lev": 1.0 - Levenshtein.distance(user_a, user_b) / max_user_len,
            "username_token_sort": fuzz.token_sort_ratio(user_a, user_b) / 100.0,
            "fullname_jaro": JaroWinkler.normalized_similarity(full_a, full_b),
            "fullname_lev": 1.0 - Levenshtein.distance(full_a, full_b) / max_full_len,
            "fullname_token_sort": fuzz.token_sort_ratio(full_a, full_b) / 100.0,
            "bio_tfidf_cosine": bio_cosine,
            "bio_sbert_cosine": bio_sbert_cosine,
            "domain_jaccard": jaccard(a["domains_set"], b["domains_set"]),
            "url_jaccard": jaccard(a["urls_set"], b["urls_set"]),
            "domain_count_a": min(len(a["domains_set"]), 10) / 10.0,
            "domain_count_b": min(len(b["domains_set"]), 10) / 10.0,
            "location_jaro": JaroWinkler.normalized_similarity(loc_a, loc_b) if loc_a and loc_b else 0.0,
            "location_token_sort": fuzz.token_sort_ratio(loc_a, loc_b) / 100.0 if loc_a and loc_b else 0.0,
            "mention_jaccard": jaccard(a["bio_mentions_set"], b["bio_mentions_set"]),
            "hashtag_jaccard": jaccard(a["hashtags_set"], b["hashtags_set"]),
            "hashtag_count_a": min(len(a["hashtags_set"]), 10) / 10.0,
            "hashtag_count_b": min(len(b["hashtags_set"]), 10) / 10.0,
            "style_caps_diff": abs(a["caps_ratio"] - b["caps_ratio"]),
            "style_avgword_diff": abs(a["avg_word_len"] - b["avg_word_len"]),
            "style_biolen_ratio": biolen_ratio(a["bio_norm"], b["bio_norm"]),
            "style_punct_diff": abs(a["punct_ratio"] - b["punct_ratio"]),
            "platform_pair_code": platform_pair,
        })
    df = pd.DataFrame(rows)
    feature_cols = [col for col in df.columns if col not in KEY_COLS]
    return df, feature_cols


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def single_feature_leakage_check(test_df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, object]]:
    y = test_df["label"].to_numpy()
    findings = []
    for col in feature_cols:
        values = pd.to_numeric(test_df[col], errors="coerce").fillna(-999999.0).to_numpy()
        if len(np.unique(values)) <= 1:
            continue
        best = 0.0
        best_rule = ""
        for thr in np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19))):
            acc_ge = float((((values >= thr).astype(int)) == y).mean())
            acc_lt = float((((values < thr).astype(int)) == y).mean())
            if acc_ge > best:
                best = acc_ge
                best_rule = f">= {thr:.4f}"
            if acc_lt > best:
                best = acc_lt
                best_rule = f"< {thr:.4f}"
        findings.append({"feature": col, "best_threshold_rule": best_rule, "best_accuracy": best})
    findings.sort(key=lambda item: item["best_accuracy"], reverse=True)
    return findings[:10]


def train_and_eval(feature_df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict[str, object]:
    splits = {name: feature_df[feature_df["split_name"] == name].reset_index(drop=True) for name in ["train", "val", "test"]}
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df[feature_cols].fillna(0.0))
    val_x = scaler.transform(val_df[feature_cols].fillna(0.0))
    test_x = scaler.transform(test_df[feature_cols].fillna(0.0))
    train_y, val_y, test_y = train_df["label"].to_numpy(), val_df["label"].to_numpy(), test_df["label"].to_numpy()

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "gb": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
    }
    leaderboard = []
    fitted = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        probs = model.predict_proba(val_x)[:, 1]
        leaderboard.append({"model": name, "val_ap": float(average_precision_score(val_y, probs)), "val_auc": float(roc_auc_score(val_y, probs))})
        fitted[name] = model
    leaderboard.sort(key=lambda item: (item["val_ap"], item["val_auc"]), reverse=True)
    best_name = leaderboard[0]["model"]
    best_model = fitted[best_name]
    val_raw = best_model.predict_proba(val_x)[:, 1]
    test_raw = best_model.predict_proba(test_x)[:, 1]
    train_raw = best_model.predict_proba(train_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_probs = calibrator.predict(val_raw)
    test_probs = calibrator.predict(test_raw)
    train_probs = calibrator.predict(train_raw)
    threshold, best_val_f1 = choose_threshold(val_y, val_probs)
    test_pred = (test_probs >= threshold).astype(int)
    val_pred = (val_probs >= threshold).astype(int)

    summary = {
        "best_model": best_name,
        "leaderboard": leaderboard,
        "feature_count": len(feature_cols),
        "threshold": threshold,
        "best_val_f1": best_val_f1,
        "val_roc_auc": float(roc_auc_score(val_y, val_probs)),
        "val_avg_precision": float(average_precision_score(val_y, val_probs)),
        "val_f1": float(f1_score(val_y, val_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(test_y, test_probs)),
        "test_avg_precision": float(average_precision_score(test_y, test_probs)),
        "test_f1": float(f1_score(test_y, test_pred, zero_division=0)),
        "test_precision": float(precision_score(test_y, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_y, test_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_y, test_pred).tolist(),
        "classification_report": classification_report(test_y, test_pred, target_names=["NO_MATCH", "MATCH"], zero_division=0, output_dict=True),
    }

    with (MODELS_DIR / "best_model.pkl").open("wb") as fh:
        pickle.dump(best_model, fh)
    with (MODELS_DIR / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)
    with (MODELS_DIR / "calibrator.pkl").open("wb") as fh:
        pickle.dump(calibrator, fh)
    with (MODELS_DIR / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(feature_cols, fh)

    for name, df, probs in [("train", train_df, train_probs), ("val", val_df, val_probs), ("test", test_df, test_probs)]:
        scores = df[["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name"]].copy()
        scores["probability"] = probs
        scores.to_parquet(REPORTS_DIR / f"{name}_scores.parquet", index=False)
    return summary


def build_dataset(profiles: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    gt_pairs = build_ground_truth_pairs(profiles)
    gt_pairs.to_parquet(ARTIFACTS_DIR / "ground_truth_pairs.parquet", index=False)
    gt_pairs.to_csv(ARTIFACTS_DIR / "ground_truth_pairs.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    save_ground_truth_matrix(profiles, gt_pairs)

    split_map = assign_splits(profiles, seed=args.seed)
    valid_profiles = profiles[profiles["profile_id"].notna()].copy()
    valid_profiles["split_name"] = valid_profiles["profile_id"].astype(int).map(split_map)
    gt_pairs["split_name"] = gt_pairs["profile_id"].astype(int).map(split_map)
    positive_pair_set = {tuple(sorted((int(r.profile_row_id_a), int(r.profile_row_id_b)))) for r in gt_pairs.itertuples(index=False)}

    parts = []
    split_stats = {}
    for split_name in ["train", "val", "test"]:
        split_pos = gt_pairs[gt_pairs["split_name"] == split_name].copy()
        split_profiles = valid_profiles[valid_profiles["split_name"] == split_name].copy()
        random_df, hard_df = generate_negatives(
            split_name, split_profiles, positive_pair_set,
            random_neg_ratio=args.random_neg_ratio,
            hard_neg_ratio=args.hard_neg_ratio,
            seed=args.seed + {"train": 11, "val": 17, "test": 23}[split_name],
        )
        split_stats[split_name] = {
            "profiles": int(len(split_profiles)),
            "positive_pairs": int(len(split_pos)),
            "random_negatives": int(len(random_df)),
            "hard_negatives": int(len(hard_df)),
        }
        parts.extend([split_pos, random_df, hard_df])

    labeled_pairs = pd.concat(parts, ignore_index=True)
    labeled_pairs.to_parquet(ARTIFACTS_DIR / "labeled_pairs.parquet", index=False)
    labeled_pairs.to_csv(ARTIFACTS_DIR / "labeled_pairs.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    profile_bundle = make_profile_bundle(profiles)
    feature_df, feature_cols = compute_features(labeled_pairs, profile_bundle)
    feature_df.to_parquet(ARTIFACTS_DIR / "pair_features.parquet", index=False)
    with (ARTIFACTS_DIR / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(feature_cols, fh)
    with (ARTIFACTS_DIR / "tfidf_vectorizer.pkl").open("wb") as fh:
        pickle.dump(profile_bundle["tfidf_vectorizer"], fh)
    (ARTIFACTS_DIR / "sbert_model_name.txt").write_text(SBERT_MODEL_NAME, encoding="utf-8")

    overlap = {
        split: set(valid_profiles[valid_profiles["split_name"] == split]["profile_id"].astype(int).tolist())
        for split in ["train", "val", "test"]
    }
    leakage = {
        "component_overlap": {
            "train_val": int(len(overlap["train"] & overlap["val"])),
            "train_test": int(len(overlap["train"] & overlap["test"])),
            "val_test": int(len(overlap["val"] & overlap["test"])),
        },
        "self_pairs": int((labeled_pairs["profile_row_id_a"] == labeled_pairs["profile_row_id_b"]).sum()),
        "rows_by_split": {split: int((feature_df["split_name"] == split).sum()) for split in ["train", "val", "test"]},
        "label_dist_by_split": {split: feature_df[feature_df["split_name"] == split]["label"].value_counts().to_dict() for split in ["train", "val", "test"]},
        "single_feature_check": single_feature_leakage_check(feature_df[feature_df["split_name"] == "test"], feature_cols),
        "split_stats": split_stats,
    }
    return feature_df, feature_cols, leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuilt Stage7/8 leak-safe pipeline and training")
    parser.add_argument("--random-neg-ratio", type=float, default=2.0)
    parser.add_argument("--hard-neg-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)
    print("=" * 60)
    print("Rebuilt Stage7/8 Leak-Safe Pipeline")
    print("=" * 60)
    profiles = load_profiles()
    feature_df, feature_cols, leakage = build_dataset(profiles, args)
    metrics = train_and_eval(feature_df, feature_cols, seed=args.seed)
    report = {"config": vars(args), "metrics": metrics, "leakage_report": leakage}
    (REPORTS_DIR / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Train rows : {(feature_df['split_name'] == 'train').sum():,}")
    print(f"Val rows   : {(feature_df['split_name'] == 'val').sum():,}")
    print(f"Test rows  : {(feature_df['split_name'] == 'test').sum():,}")
    print(f"Features   : {len(feature_cols)}")
    print(f"Best model : {metrics['best_model']}")
    print(f"Test AP    : {metrics['test_avg_precision']:.4f}")
    print(f"Test AUC   : {metrics['test_roc_auc']:.4f}")
    print(f"Test F1    : {metrics['test_f1']:.4f}")
    print(f"Report     : {REPORTS_DIR / 'experiment_report.json'}")


if __name__ == "__main__":
    main()
