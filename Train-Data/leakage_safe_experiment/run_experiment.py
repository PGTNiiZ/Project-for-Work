from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
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
PROFILES_PATH = PROJECT_ROOT / "data-for-project" / "nomalized_profiles.csv"
GROUND_TRUTH_PAIRS_PATH = PROJECT_ROOT / "Train-Data" / "ground_truth_pairs.csv"

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "reports"

KEY_COLS = ["profile_id_a", "profile_id_b", "label", "pair_type", "split_name", "component_a", "component_b"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
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


def normalize_username(value: object) -> str:
    return str(value or "").strip().lower().lstrip("@")


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
    if not union:
        return 0.0
    return len(a & b) / len(union)


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


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ExperimentData:
    pairs: pd.DataFrame
    feature_cols: list[str]
    split_frames: dict[str, pd.DataFrame]
    leakage_report: dict[str, object]


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_PATH)
    df["source_profile_id"] = df["profile_id"].astype(str)
    df["profile_id"] = df.index.astype(str)
    df["userName_norm"] = df["userName"].map(normalize_username)
    df["fullName_norm"] = df["fullName"].map(lower_text)
    df["bio_norm"] = df["bio"].map(lower_text)
    df["location_norm"] = df["location"].map(lower_text)
    df["bio_mentions_set"] = df["bio_mentions"].map(parse_mentions)
    df["hashtags_set"] = df["bio_norm"].map(extract_hashtags)
    df["urls_set"] = df["bio_urls"].map(parse_urls)
    df["domains_set"] = df["urls_set"].map(extract_domains)
    df["caps_ratio"] = df["bio"].map(lambda x: sum(1 for ch in clean_text(x) if ch.isupper()) / max(len(clean_text(x)), 1))
    df["avg_word_len"] = df["bio_norm"].map(avg_word_len)
    df["bio_len"] = df["bio_norm"].map(len)
    df["punct_ratio"] = df["bio_norm"].map(punct_ratio)
    return df


def build_positive_pairs(profiles: pd.DataFrame) -> pd.DataFrame:
    profile_map = profiles.set_index("profile_id")[["platform", "userName_norm"]].to_dict("index")
    gt = pd.read_csv(GROUND_TRUTH_PAIRS_PATH)
    positives: set[tuple[str, str]] = set()
    for row in gt[gt["match"] == 1][["idx_i", "idx_j"]].itertuples(index=False):
        pid_a = str(row.idx_i)
        pid_b = str(row.idx_j)
        if pid_a == pid_b:
            continue
        if pid_a not in profile_map or pid_b not in profile_map:
            continue
        plat_a = profile_map[pid_a]["platform"]
        plat_b = profile_map[pid_b]["platform"]
        if plat_a == plat_b:
            continue
        positives.add(tuple(sorted((pid_a, pid_b))))

    positive_rows = [
        {
            "profile_id_a": pid_a,
            "profile_id_b": pid_b,
            "label": 1,
            "pair_type": "positive",
            "platform_a": profile_map[pid_a]["platform"],
            "platform_b": profile_map[pid_b]["platform"],
        }
        for pid_a, pid_b in sorted(positives)
    ]
    return pd.DataFrame(positive_rows)


def assign_components(positive_pairs: pd.DataFrame) -> dict[str, str]:
    uf = UnionFind()
    for row in positive_pairs[["profile_id_a", "profile_id_b"]].itertuples(index=False):
        uf.union(row.profile_id_a, row.profile_id_b)
    components = {}
    for pid in pd.unique(pd.concat([positive_pairs["profile_id_a"], positive_pairs["profile_id_b"]], ignore_index=True)):
        components[str(pid)] = uf.find(str(pid))
    return components


def assign_splits_by_component(positive_pairs: pd.DataFrame, components: dict[str, str], seed: int) -> tuple[dict[str, str], dict[str, object]]:
    rng = random.Random(seed)
    comp_pos_counts: Counter[str] = Counter()
    comp_profiles: defaultdict[str, set[str]] = defaultdict(set)
    for row in positive_pairs[["profile_id_a", "profile_id_b"]].itertuples(index=False):
        comp = components[row.profile_id_a]
        comp_pos_counts[comp] += 1
        comp_profiles[comp].add(row.profile_id_a)
        comp_profiles[comp].add(row.profile_id_b)

    ordered_components = list(comp_pos_counts.keys())
    rng.shuffle(ordered_components)
    ordered_components.sort(key=lambda comp: comp_pos_counts[comp], reverse=True)

    total_pos = max(sum(comp_pos_counts.values()), 1)
    target_pos = {split: SPLIT_RATIOS[split] * total_pos for split in SPLIT_RATIOS}
    current_pos = {split: 0 for split in SPLIT_RATIOS}
    split_components: dict[str, str] = {}

    for comp in ordered_components:
        deficits = {
            split: (target_pos[split] - current_pos[split]) / max(target_pos[split], 1.0)
            for split in SPLIT_RATIOS
        }
        chosen_split = max(deficits.items(), key=lambda item: (item[1], -current_pos[item[0]]))[0]
        split_components[comp] = chosen_split
        current_pos[chosen_split] += comp_pos_counts[comp]

    report = {
        "component_count": len(ordered_components),
        "positive_pairs_by_split_target": target_pos,
        "positive_pairs_by_split_actual": current_pos,
        "largest_components": [
            {
                "component": comp,
                "positive_pairs": int(comp_pos_counts[comp]),
                "profiles": int(len(comp_profiles[comp])),
                "split": split_components[comp],
            }
            for comp in ordered_components[:10]
        ],
    }
    return split_components, report


def make_profile_lookup(profiles: pd.DataFrame, keep_ids: set[str]) -> dict[str, object]:
    subset = profiles[profiles["profile_id"].isin(keep_ids)].copy()
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(subset["bio_norm"].fillna(""))
    idx_map = {pid: idx for idx, pid in enumerate(subset["profile_id"].tolist())}
    lookup: dict[str, dict] = {}
    for record in subset.to_dict("records"):
        pid = record["profile_id"]
        record["_tfidf_idx"] = idx_map[pid]
        lookup[pid] = record
    return {"records": lookup, "tfidf_matrix": tfidf, "tfidf_vectorizer": vectorizer}


def generate_random_negatives(
    split_name: str,
    profiles_by_split: pd.DataFrame,
    components: dict[str, str],
    positive_pairs: set[tuple[str, str]],
    target_count: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    records = profiles_by_split[["profile_id", "platform"]].to_dict("records")
    negatives: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(target_count * 40, 20000)
    while len(negatives) < target_count and attempts < max_attempts:
        left = records[rng.randrange(len(records))]
        right = records[rng.randrange(len(records))]
        attempts += 1
        if left["profile_id"] == right["profile_id"]:
            continue
        if left["platform"] == right["platform"]:
            continue
        comp_left = components[left["profile_id"]]
        comp_right = components[right["profile_id"]]
        if comp_left == comp_right:
            continue
        pair = tuple(sorted((left["profile_id"], right["profile_id"])))
        if pair in positive_pairs:
            continue
        negatives.add(pair)

    rows = [
        {
            "profile_id_a": pid_a,
            "profile_id_b": pid_b,
            "label": 0,
            "pair_type": "random_negative",
            "split_name": split_name,
            "component_a": components[pid_a],
            "component_b": components[pid_b],
        }
        for pid_a, pid_b in sorted(negatives)
    ]
    return pd.DataFrame(rows)


def generate_hard_negatives(
    split_name: str,
    profiles_by_split: pd.DataFrame,
    components: dict[str, str],
    positive_pairs: set[tuple[str, str]],
    already_used: set[tuple[str, str]],
    target_count: int,
) -> pd.DataFrame:
    buckets: defaultdict[str, list[dict]] = defaultdict(list)
    for record in profiles_by_split[["profile_id", "platform", "userName_norm", "fullName_norm"]].to_dict("records"):
        uname = record["userName_norm"]
        for key in {uname[:1], uname[:2], uname[:3]}:
            if key:
                buckets[key].append(record)

    candidates: list[tuple[float, tuple[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for bucket_records in buckets.values():
        if len(bucket_records) < 2:
            continue
        limit = min(len(bucket_records), 120)
        bucket_records = sorted(bucket_records, key=lambda item: item["userName_norm"])[:limit]
        for idx in range(len(bucket_records)):
            left = bucket_records[idx]
            for jdx in range(idx + 1, len(bucket_records)):
                right = bucket_records[jdx]
                if left["platform"] == right["platform"]:
                    continue
                pair = tuple(sorted((left["profile_id"], right["profile_id"])))
                if pair in seen or pair in positive_pairs or pair in already_used:
                    continue
                if components[left["profile_id"]] == components[right["profile_id"]]:
                    continue
                username_score = fuzz.ratio(left["userName_norm"], right["userName_norm"]) / 100.0
                fullname_score = fuzz.ratio(left["fullName_norm"], right["fullName_norm"]) / 100.0
                score = max(username_score, fullname_score)
                if score < 0.72:
                    continue
                seen.add(pair)
                candidates.append((score, pair))

    candidates.sort(reverse=True)
    selected = candidates[:target_count]
    rows = [
        {
            "profile_id_a": pid_a,
            "profile_id_b": pid_b,
            "label": 0,
            "pair_type": "hard_negative",
            "split_name": split_name,
            "component_a": components[pid_a],
            "component_b": components[pid_b],
            "hard_score": score,
        }
        for score, (pid_a, pid_b) in selected
    ]
    return pd.DataFrame(rows)


def build_pairs(profiles: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    positive_pairs_df = build_positive_pairs(profiles)
    if positive_pairs_df.empty:
        raise RuntimeError("No valid positive pairs could be built from ground truth data.")

    components = assign_components(positive_pairs_df)
    split_components, split_report = assign_splits_by_component(positive_pairs_df, components, seed=args.seed)

    positive_pairs_df["component_a"] = positive_pairs_df["profile_id_a"].map(components)
    positive_pairs_df["component_b"] = positive_pairs_df["profile_id_b"].map(components)
    positive_pairs_df["split_name"] = positive_pairs_df["component_a"].map(split_components)

    component_to_split = split_components
    positive_pair_set = {tuple(sorted((row.profile_id_a, row.profile_id_b))) for row in positive_pairs_df.itertuples(index=False)}
    platform_map = profiles.set_index("profile_id")["platform"].to_dict()

    all_profiles = profiles.copy()
    all_profiles["component"] = all_profiles["profile_id"].map(components)
    all_profiles = all_profiles[all_profiles["component"].notna()].copy()
    all_profiles["split_name"] = all_profiles["component"].map(component_to_split)

    all_parts = [positive_pairs_df.copy()]
    split_stats: dict[str, object] = {}

    for split_name in ["train", "val", "test"]:
        split_profiles = all_profiles[all_profiles["split_name"] == split_name].copy()
        split_positive = positive_pairs_df[positive_pairs_df["split_name"] == split_name].copy()
        target_random = int(len(split_positive) * args.random_neg_ratio)
        target_hard = int(len(split_positive) * args.hard_neg_ratio)

        random_neg = generate_random_negatives(
            split_name=split_name,
            profiles_by_split=split_profiles,
            components=components,
            positive_pairs=positive_pair_set,
            target_count=target_random,
            seed=args.seed + {"train": 11, "val": 17, "test": 23}[split_name],
        )
        random_neg["platform_a"] = random_neg["profile_id_a"].map(platform_map)
        random_neg["platform_b"] = random_neg["profile_id_b"].map(platform_map)

        used_pairs = set(tuple(sorted((row.profile_id_a, row.profile_id_b))) for row in random_neg.itertuples(index=False))
        hard_neg = generate_hard_negatives(
            split_name=split_name,
            profiles_by_split=split_profiles,
            components=components,
            positive_pairs=positive_pair_set,
            already_used=used_pairs,
            target_count=target_hard,
        )
        hard_neg["platform_a"] = hard_neg["profile_id_a"].map(platform_map)
        hard_neg["platform_b"] = hard_neg["profile_id_b"].map(platform_map)

        all_parts.extend([random_neg, hard_neg])
        split_stats[split_name] = {
            "profiles": int(len(split_profiles)),
            "positive_pairs": int(len(split_positive)),
            "random_negatives": int(len(random_neg)),
            "hard_negatives": int(len(hard_neg)),
        }

    pairs = pd.concat(all_parts, ignore_index=True)
    pairs["profile_id_a"] = pairs["profile_id_a"].astype(str)
    pairs["profile_id_b"] = pairs["profile_id_b"].astype(str)
    pairs = pairs.drop_duplicates(subset=["profile_id_a", "profile_id_b", "label", "pair_type", "split_name"]).reset_index(drop=True)

    leakage_report = {
        "pair_builder": {
            "positive_pairs": int(len(positive_pairs_df)),
            "positive_components": int(len(set(components.values()))),
            "split_assignment": split_report,
            "split_stats": split_stats,
        }
    }
    return pairs, leakage_report


def compute_features(pairs: pd.DataFrame, profile_bundle: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    records: dict[str, dict] = profile_bundle["records"]
    tfidf_matrix = profile_bundle["tfidf_matrix"]

    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        a = records[row.profile_id_a]
        b = records[row.profile_id_b]

        user_a = a["userName_norm"]
        user_b = b["userName_norm"]
        full_a = a["fullName_norm"]
        full_b = b["fullName_norm"]
        loc_a = a["location_norm"]
        loc_b = b["location_norm"]

        max_user_len = max(len(user_a), len(user_b), 1)
        max_full_len = max(len(full_a), len(full_b), 1)
        bio_cosine = float(tfidf_matrix[a["_tfidf_idx"]].multiply(tfidf_matrix[b["_tfidf_idx"]]).sum())
        platform_pair = PLATFORM_PAIR_CODES.get(tuple(sorted([a["platform"], b["platform"]])), -1)

        feat = {
            "profile_id_a": row.profile_id_a,
            "profile_id_b": row.profile_id_b,
            "label": row.label,
            "pair_type": row.pair_type,
            "split_name": row.split_name,
            "component_a": row.component_a,
            "component_b": row.component_b,
            "username_jaro": JaroWinkler.normalized_similarity(user_a, user_b),
            "username_lev": 1.0 - Levenshtein.distance(user_a, user_b) / max_user_len,
            "username_token_sort": fuzz.token_sort_ratio(user_a, user_b) / 100.0,
            "fullname_jaro": JaroWinkler.normalized_similarity(full_a, full_b),
            "fullname_lev": 1.0 - Levenshtein.distance(full_a, full_b) / max_full_len,
            "fullname_token_sort": fuzz.token_sort_ratio(full_a, full_b) / 100.0,
            "bio_tfidf_cosine": bio_cosine,
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
        }
        rows.append(feat)

    feature_df = pd.DataFrame(rows)
    feature_cols = [col for col in feature_df.columns if col not in KEY_COLS]
    return feature_df, feature_cols


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def train_and_evaluate(data: ExperimentData, seed: int) -> dict[str, object]:
    split_frames = data.split_frames
    feature_cols = data.feature_cols

    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df[feature_cols].fillna(0.0))
    val_x = scaler.transform(val_df[feature_cols].fillna(0.0))
    test_x = scaler.transform(test_df[feature_cols].fillna(0.0))
    train_y = train_df["label"].to_numpy()
    val_y = val_df["label"].to_numpy()
    test_y = test_df["label"].to_numpy()

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "gb": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=1,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
    }

    leaderboard = []
    fitted = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        val_probs = model.predict_proba(val_x)[:, 1]
        val_ap = average_precision_score(val_y, val_probs)
        val_auc = roc_auc_score(val_y, val_probs)
        leaderboard.append({"model": name, "val_ap": float(val_ap), "val_auc": float(val_auc)})
        fitted[name] = model

    leaderboard.sort(key=lambda item: (item["val_ap"], item["val_auc"]), reverse=True)
    best_name = leaderboard[0]["model"]
    best_model = fitted[best_name]

    val_probs_raw = best_model.predict_proba(val_x)[:, 1]
    test_probs_raw = best_model.predict_proba(test_x)[:, 1]
    train_probs_raw = best_model.predict_proba(train_x)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, val_y)
    val_probs = calibrator.predict(val_probs_raw)
    test_probs = calibrator.predict(test_probs_raw)
    train_probs = calibrator.predict(train_probs_raw)

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
        "classification_report": classification_report(
            test_y, test_pred, target_names=["NO_MATCH", "MATCH"], zero_division=0, output_dict=True
        ),
    }

    with (MODELS_DIR / "best_model.pkl").open("wb") as fh:
        pickle.dump(best_model, fh)
    with (MODELS_DIR / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)
    with (MODELS_DIR / "calibrator.pkl").open("wb") as fh:
        pickle.dump(calibrator, fh)
    with (MODELS_DIR / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(feature_cols, fh)

    train_scores = train_df[["profile_id_a", "profile_id_b", "label", "pair_type", "split_name"]].copy()
    train_scores["probability"] = train_probs
    val_scores = val_df[["profile_id_a", "profile_id_b", "label", "pair_type", "split_name"]].copy()
    val_scores["probability"] = val_probs
    test_scores = test_df[["profile_id_a", "profile_id_b", "label", "pair_type", "split_name"]].copy()
    test_scores["probability"] = test_probs
    train_scores.to_parquet(REPORTS_DIR / "train_scores.parquet", index=False)
    val_scores.to_parquet(REPORTS_DIR / "val_scores.parquet", index=False)
    test_scores.to_parquet(REPORTS_DIR / "test_scores.parquet", index=False)

    return summary


def single_feature_leakage_check(test_df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, object]]:
    y = test_df["label"].to_numpy()
    findings: list[dict[str, object]] = []
    for col in feature_cols:
        values = pd.to_numeric(test_df[col], errors="coerce").fillna(-999999.0).to_numpy()
        if len(np.unique(values)) <= 1:
            continue
        best_acc = 0.0
        best_rule = ""
        quantiles = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
        for thr in quantiles:
            pred_ge = (values >= thr).astype(int)
            acc_ge = float((pred_ge == y).mean())
            pred_lt = (values < thr).astype(int)
            acc_lt = float((pred_lt == y).mean())
            if acc_ge > best_acc:
                best_acc = acc_ge
                best_rule = f">= {thr:.4f}"
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_rule = f"< {thr:.4f}"
        findings.append({"feature": col, "best_threshold_rule": best_rule, "best_accuracy": best_acc})
    findings.sort(key=lambda item: item["best_accuracy"], reverse=True)
    return findings[:10]


def build_experiment_data(args: argparse.Namespace) -> ExperimentData:
    profiles = load_profiles()
    pairs, leakage_report = build_pairs(profiles, args)

    keep_ids = set(pd.unique(pd.concat([pairs["profile_id_a"], pairs["profile_id_b"]], ignore_index=True)))
    profile_bundle = make_profile_lookup(profiles, keep_ids)
    feature_df, feature_cols = compute_features(pairs, profile_bundle)

    split_frames = {
        split_name: feature_df[feature_df["split_name"] == split_name].reset_index(drop=True)
        for split_name in ["train", "val", "test"]
    }

    overlap = {
        split: set(split_frames[split]["component_a"].unique()) | set(split_frames[split]["component_b"].unique())
        for split in split_frames
    }
    component_overlap = {
        "train_val": int(len(overlap["train"] & overlap["val"])),
        "train_test": int(len(overlap["train"] & overlap["test"])),
        "val_test": int(len(overlap["val"] & overlap["test"])),
    }

    leakage_report["split_leakage"] = {
        "component_overlap": component_overlap,
        "self_pairs": int((pairs["profile_id_a"] == pairs["profile_id_b"]).sum()),
        "rows_by_split": {split: int(len(frame)) for split, frame in split_frames.items()},
        "label_dist_by_split": {split: frame["label"].value_counts().to_dict() for split, frame in split_frames.items()},
    }
    leakage_report["single_feature_check"] = single_feature_leakage_check(split_frames["test"], feature_cols)

    feature_df.to_parquet(ARTIFACTS_DIR / "pair_features.parquet", index=False)
    pairs.to_parquet(ARTIFACTS_DIR / "pairs.parquet", index=False)
    with (ARTIFACTS_DIR / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(feature_cols, fh)
    with (ARTIFACTS_DIR / "tfidf_vectorizer.pkl").open("wb") as fh:
        pickle.dump(profile_bundle["tfidf_vectorizer"], fh)

    return ExperimentData(pairs=pairs, feature_cols=feature_cols, split_frames=split_frames, leakage_report=leakage_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained leakage-safe identity matching experiment")
    parser.add_argument("--random-neg-ratio", type=float, default=4.0)
    parser.add_argument("--hard-neg-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    print("=" * 60)
    print("Leakage-Safe Identity Experiment")
    print("=" * 60)
    data = build_experiment_data(args)
    metrics = train_and_evaluate(data, seed=args.seed)

    report = {
        "metrics": metrics,
        "leakage_report": data.leakage_report,
    }
    (REPORTS_DIR / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Train rows : {len(data.split_frames['train']):,}")
    print(f"Val rows   : {len(data.split_frames['val']):,}")
    print(f"Test rows  : {len(data.split_frames['test']):,}")
    print(f"Features   : {len(data.feature_cols)}")
    print(f"Best model : {metrics['best_model']}")
    print(f"Test AP    : {metrics['test_avg_precision']:.4f}")
    print(f"Test AUC   : {metrics['test_roc_auc']:.4f}")
    print(f"Test F1    : {metrics['test_f1']:.4f}")
    print(f"Report     : {REPORTS_DIR / 'experiment_report.json'}")


if __name__ == "__main__":
    main()
