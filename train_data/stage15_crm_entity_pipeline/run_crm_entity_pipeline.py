from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROFILES_PATH = PROJECT_ROOT / "data_for_project" / "normalized_profiles_with_profile_id.csv"
FULL_PIPELINE_ROOT = PROJECT_ROOT / "train_data" / "stage7_14_full_candidate_pipeline"
BEST_RUN_PATH = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42"
BASE_PIPELINE_PATH = PROJECT_ROOT / "train_data" / "stage7_8_rebuilt_experiment_hybrid" / "run_rebuilt_pipeline.py"
MULTIMODAL_SUITE_PATH = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "run_multimodal_suite.py"

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
REPORTS_DIR = SCRIPT_DIR / "reports"


def ensure_dirs() -> None:
    for path in [SCRIPT_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_PATH, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df["profile_id"] = pd.to_numeric(df["profile_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna() & df["profile_id"].notna()].copy()
    df["platform"] = df["platform"].map(lower_text)
    df["userName"] = df["userName"].map(clean_text)
    df["fullName"] = df["fullName"].map(clean_text)
    df["bio"] = df["bio"].map(clean_text)
    df["location"] = df["location"].map(clean_text)
    df["externalUrl"] = df["externalUrl"].map(clean_text)
    df["pictureURL"] = df["pictureURL"].map(clean_text)
    df["bio_mentions_count"] = pd.to_numeric(df["bio_mentions_count"], errors="coerce").fillna(0).astype(int)
    df["url_count"] = pd.to_numeric(df["url_count"], errors="coerce").fillna(0).astype(int)
    df["hashtag_count"] = df["bio"].map(lambda text: len(re.findall(r"#(\w+)", text or "")))
    return df


def build_feature_engine():
    base = load_module("crm_entity_base", BASE_PIPELINE_PATH)
    suite = load_module("crm_entity_suite", MULTIMODAL_SUITE_PATH)
    base.SCRIPT_DIR = BEST_RUN_PATH
    base.PROJECT_ROOT = PROJECT_ROOT
    base.PROFILES_PATH = PROFILES_PATH
    base.ARTIFACTS_DIR = BEST_RUN_PATH / "artifacts"
    base.MODELS_DIR = BEST_RUN_PATH / "models"
    base.REPORTS_DIR = BEST_RUN_PATH / "reports"
    profiles = base.load_profiles()
    image_df, caption_embeddings = suite.build_image_profile_features(base, profiles)
    wrapped_make, wrapped_compute = suite.make_multimodal_wrappers(base, "image_context", image_df, caption_embeddings)
    profile_bundle = wrapped_make(profiles)
    with (BEST_RUN_PATH / "models" / "best_model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    with (BEST_RUN_PATH / "models" / "feature_cols.pkl").open("rb") as fh:
        feature_cols = pickle.load(fh)
    feature_importances = getattr(model, "feature_importances_", np.ones(len(feature_cols), dtype=float))
    return wrapped_compute, profile_bundle, feature_cols, feature_importances


def load_exact_and_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_paths = sorted((FULL_PIPELINE_ROOT / "artifacts").glob("exact_matches_part*.parquet"))
    if not exact_paths:
        raise FileNotFoundError("Exact match artifacts not found.")
    exact_df = pd.concat([pd.read_parquet(path) for path in exact_paths], ignore_index=True)
    exact_df["score"] = 1.0
    exact_df["decision"] = "MATCH"
    exact_df["decision_source"] = "AUTO_EXACT"
    exact_df["review_status"] = "APPROVED"
    exact_df["reviewed_by"] = None
    exact_df["reviewed_at"] = pd.NaT
    exact_df["decision_id"] = [str(uuid.uuid4()) for _ in range(len(exact_df))]

    score_paths = sorted((FULL_PIPELINE_ROOT / "scores").glob("candidate_scores_*.parquet"))
    if not score_paths:
        raise FileNotFoundError("Candidate score chunks not found.")
    scored_df = pd.concat([pd.read_parquet(path) for path in score_paths], ignore_index=True)
    return exact_df, scored_df


def build_match_decisions(scored_df: pd.DataFrame, match_threshold: float, review_threshold: float) -> pd.DataFrame:
    decisions = scored_df.rename(columns={"probability": "score"}).copy()
    decisions["decision"] = np.where(
        decisions["score"] >= match_threshold,
        "MATCH",
        np.where(decisions["score"] >= review_threshold, "REVIEW", "NO_MATCH"),
    )
    decisions["decision_source"] = np.where(
        decisions["decision"] == "MATCH",
        "AUTO_HIGH",
        np.where(decisions["decision"] == "REVIEW", "AUTO_REVIEW", "AUTO_LOW"),
    )
    decisions["review_status"] = np.where(
        decisions["decision"] == "MATCH",
        "APPROVED",
        np.where(decisions["decision"] == "REVIEW", "PENDING", "SKIPPED"),
    )
    decisions["reviewed_by"] = None
    decisions["reviewed_at"] = pd.NaT
    decisions["decision_id"] = [str(uuid.uuid4()) for _ in range(len(decisions))]
    return decisions[[
        "decision_id", "profile_row_id_a", "profile_row_id_b", "score",
        "decision", "decision_source", "review_status", "reviewed_by",
        "reviewed_at", "label", "pair_type", "split_name"
    ]]


def heuristic_support_score(feature_name: str, value: float) -> float:
    if pd.isna(value):
        return 0.0
    name = feature_name.lower()
    val = float(value)
    if any(token in name for token in ["jaro", "lev", "token", "jaccard", "cosine", "ratio", "sim"]):
        return max(0.0, min(val, 1.0))
    if "diff" in name:
        return max(0.0, 1.0 - min(abs(val), 1.0))
    if "count" in name or "code" in name:
        return max(0.0, min(abs(val), 1.0))
    if name.startswith("image_any") or name.startswith("image_both") or name.startswith("image_caption_any"):
        return max(0.0, min(val, 1.0))
    return max(0.0, min(abs(val), 1.0))


def build_review_queue(
    review_df: pd.DataFrame,
    profiles: pd.DataFrame,
    compute_features_fn,
    profile_bundle: dict[str, object],
    feature_cols: list[str],
    feature_importances: np.ndarray,
    snapshot_limit: int,
) -> pd.DataFrame:
    if review_df.empty:
        return pd.DataFrame(columns=[
            "queue_id", "decision_id", "score", "priority", "profile_snapshot_a",
            "profile_snapshot_b", "key_features", "status", "assigned_to"
        ])

    profile_lookup = {
        int(row["profile_row_id"]): row
        for row in profiles.to_dict("records")
    }
    feature_map = dict(zip(feature_cols, feature_importances))
    rows = []
    batch_size = 5000
    for start in range(0, len(review_df), batch_size):
        stop = min(start + batch_size, len(review_df))
        batch = review_df.iloc[start:stop].copy()
        pair_input = batch[["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name"]].copy()
        pair_input["profile_id"] = -1
        feat_df, _ = compute_features_fn(pair_input, profile_bundle)
        for idx, decision_row in batch.reset_index(drop=True).iterrows():
            feat_row = feat_df.iloc[idx]
            contribs = []
            for feature in feature_cols:
                raw_value = feat_row[feature]
                support = heuristic_support_score(feature, raw_value)
                contrib = float(feature_map.get(feature, 0.0)) * support
                contribs.append({
                    "feature": feature,
                    "value": None if pd.isna(raw_value) else float(raw_value),
                    "importance": float(feature_map.get(feature, 0.0)),
                    "support": support,
                    "contribution": contrib,
                })
            contribs.sort(key=lambda item: item["contribution"], reverse=True)
            rid_a = int(decision_row["profile_row_id_a"])
            rid_b = int(decision_row["profile_row_id_b"])
            a = profile_lookup[rid_a]
            b = profile_lookup[rid_b]
            snapshot_a = {
                "profile_row_id": rid_a,
                "platform": a["platform"],
                "userName": a["userName"],
                "fullName": a["fullName"],
                "bio": a["bio"],
                "location": a["location"],
                "externalUrl": a["externalUrl"],
                "pictureURL": a["pictureURL"],
            }
            snapshot_b = {
                "profile_row_id": rid_b,
                "platform": b["platform"],
                "userName": b["userName"],
                "fullName": b["fullName"],
                "bio": b["bio"],
                "location": b["location"],
                "externalUrl": b["externalUrl"],
                "pictureURL": b["pictureURL"],
            }
            rows.append({
                "queue_id": str(uuid.uuid4()),
                "decision_id": decision_row["decision_id"],
                "score": float(decision_row["score"]),
                "priority": float(decision_row["score"]),
                "profile_snapshot_a": json_text(snapshot_a),
                "profile_snapshot_b": json_text(snapshot_b),
                "key_features": json_text(contribs[:5]),
                "status": "PENDING",
                "assigned_to": None,
            })
        print(f"[REVIEW] processed {stop:,}/{len(review_df):,}")

    review_queue = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    if snapshot_limit > 0:
        review_queue.head(snapshot_limit).to_csv(
            REPORTS_DIR / "review_queue_sample.csv",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
    return review_queue


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def choose_canonical_name(names: list[str]) -> str:
    valid = [clean_text(name) for name in names if clean_text(name)]
    if not valid:
        return ""
    counts = Counter(valid)
    return max(valid, key=lambda name: (counts[name], len(name)))


def build_unified_tables(match_decisions: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = match_decisions[
        ((match_decisions["decision"] == "MATCH") & (match_decisions["decision_source"].isin(["AUTO_EXACT", "AUTO_HIGH", "HUMAN"])))
    ].copy()
    uf = UnionFind()
    for row in approved.itertuples(index=False):
        uf.union(int(row.profile_id_a), int(row.profile_id_b))

    # include isolated profiles too
    for rid in profiles["profile_row_id"].astype(int).tolist():
        uf.find(rid)

    members: dict[int, list[int]] = defaultdict(list)
    for rid in profiles["profile_row_id"].astype(int).tolist():
        members[uf.find(rid)].append(rid)

    profile_lookup = {int(row["profile_row_id"]): row for row in profiles.to_dict("records")}
    edge_lookup: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in approved.to_dict("records"):
        key = tuple(sorted((int(row["profile_id_a"]), int(row["profile_id_b"]))))
        edge_lookup[key].append(row)

    unified_rows = []
    profile_mapping_rows = []
    for root, rids in sorted(members.items()):
        unified_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"crm-unified-{root}"))
        member_profiles = [profile_lookup[rid] for rid in sorted(rids)]
        platforms = sorted({row["platform"] for row in member_profiles})
        usernames_by_platform: dict[str, list[str]] = defaultdict(list)
        urls = []
        bios = []
        locations = []
        merge_scores = []
        merge_sources = set()
        for row in member_profiles:
            if row["userName"]:
                usernames_by_platform[row["platform"]].append(row["userName"])
            if row["externalUrl"]:
                urls.append(row["externalUrl"])
            if row["bio"]:
                bios.append(row["bio"])
            if row["location"]:
                locations.append(row["location"])
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                key = tuple(sorted((rids[i], rids[j])))
                for edge in edge_lookup.get(key, []):
                    merge_scores.append(float(edge["score"]))
                    merge_sources.add(edge["decision_source"])

        if merge_sources == {"AUTO_EXACT"}:
            merge_source = "EXACT"
        elif "HUMAN" in merge_sources:
            merge_source = "HUMAN_APPROVED"
        else:
            merge_source = "AUTO"

        canonical_name = choose_canonical_name([row["fullName"] for row in member_profiles] + [row["userName"] for row in member_profiles])
        best_location = choose_canonical_name(locations)
        merged_bio = max(bios, key=len) if bios else ""
        merge_confidence = float(np.mean(merge_scores)) if merge_scores else 1.0

        unified_rows.append({
            "unified_id": unified_id,
            "profile_ids": json_text(sorted(rids)),
            "platforms": json_text(platforms),
            "canonical_name": canonical_name,
            "all_usernames": json_text({k: sorted(set(v)) for k, v in usernames_by_platform.items()}),
            "merged_bio": merged_bio,
            "location": best_location,
            "all_urls": json_text(sorted(set(urls))),
            "platform_count": int(len(platforms)),
            "merge_confidence": merge_confidence,
            "merge_source": merge_source,
        })

        mapping_source = merge_source
        mapped_at = pd.Timestamp.utcnow().isoformat()
        for row in member_profiles:
            profile_mapping_rows.append({
                "profile_id": int(row["profile_row_id"]),
                "unified_id": unified_id,
                "platform": row["platform"],
                "userName": row["userName"],
                "mapping_source": mapping_source,
                "mapped_at": mapped_at,
            })

    return pd.DataFrame(unified_rows), pd.DataFrame(profile_mapping_rows)


def build_lead_scores(unified_profiles: pd.DataFrame, profile_mapping: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    profile_lookup = {int(row["profile_row_id"]): row for row in profiles.to_dict("records")}
    rows = []
    for row in unified_profiles.to_dict("records"):
        profile_ids = json.loads(row["profile_ids"])
        members = [profile_lookup[int(pid)] for pid in profile_ids]
        has_bio = 1 if any(clean_text(member["bio"]) for member in members) else 0
        has_location = 1 if any(clean_text(member["location"]) for member in members) else 0
        has_url = 1 if any(clean_text(member["externalUrl"]) for member in members) else 0
        completeness_score = (has_bio + has_location + has_url) / 3.0
        platform_score = min(float(row["platform_count"]) / 3.0, 1.0)
        engagement_raw = sum(int(member.get("bio_mentions_count", 0)) + int(member.get("hashtag_count", 0)) + int(member.get("url_count", 0)) for member in members)
        engagement_score = min(engagement_raw / 10.0, 1.0)
        lead_score = 100.0 * (0.4 * completeness_score + 0.3 * platform_score + 0.3 * engagement_score)
        tier = "HOT" if lead_score >= 80 else "WARM" if lead_score >= 50 else "COLD"
        rows.append({
            "score_id": str(uuid.uuid4()),
            "unified_id": row["unified_id"],
            "completeness_score": completeness_score,
            "platform_score": platform_score,
            "engagement_score": engagement_score,
            "lead_score": lead_score,
            "tier": tier,
            "score_breakdown": json_text({
                "completeness_score": completeness_score,
                "platform_score": platform_score,
                "engagement_score": engagement_score,
                "formula": "0.4*completeness + 0.3*platform + 0.3*engagement",
            }),
        })
    return pd.DataFrame(rows).sort_values("lead_score", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CRM entity tables from full candidate pipeline outputs")
    parser.add_argument("--match-threshold", type=float, default=0.98)
    parser.add_argument("--review-threshold", type=float, default=0.95)
    parser.add_argument("--review-sample-limit", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    profiles = load_profiles()
    exact_df, scored_df = load_exact_and_scores()
    match_decisions = build_match_decisions(scored_df, args.match_threshold, args.review_threshold)

    exact_df = exact_df[["decision_id", "profile_row_id_a", "profile_row_id_b", "score", "decision", "decision_source", "review_status", "reviewed_by", "reviewed_at"]].copy()
    exact_df["label"] = None
    exact_df["pair_type"] = "exact_match"
    exact_df["split_name"] = "full"

    all_decisions = pd.concat([exact_df, match_decisions], ignore_index=True)
    all_decisions["created_at"] = pd.Timestamp.now("UTC").isoformat()
    all_decisions = all_decisions.rename(columns={"profile_row_id_a": "profile_id_a", "profile_row_id_b": "profile_id_b"})
    all_decisions.to_parquet(ARTIFACTS_DIR / "match_decisions.parquet", index=False)
    all_decisions.head(10000).to_csv(REPORTS_DIR / "match_decisions_sample.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    review_df = all_decisions[all_decisions["decision"] == "REVIEW"].copy()
    compute_features_fn, profile_bundle, feature_cols, feature_importances = build_feature_engine()
    feature_ready_review = review_df.rename(columns={"profile_id_a": "profile_row_id_a", "profile_id_b": "profile_row_id_b"})
    review_queue_path = ARTIFACTS_DIR / "review_queue.parquet"
    if review_queue_path.exists():
        review_queue = pd.read_parquet(review_queue_path)
    else:
        review_queue = build_review_queue(
            feature_ready_review,
            profiles,
            compute_features_fn,
            profile_bundle,
            feature_cols,
            feature_importances,
            args.review_sample_limit,
        )
    review_queue.to_parquet(ARTIFACTS_DIR / "review_queue.parquet", index=False)

    unified_profiles, profile_mapping = build_unified_tables(all_decisions, profiles)
    unified_profiles.to_parquet(ARTIFACTS_DIR / "unified_profiles.parquet", index=False)
    profile_mapping.to_parquet(ARTIFACTS_DIR / "profile_mapping.parquet", index=False)

    lead_scores = build_lead_scores(unified_profiles, profile_mapping, profiles)
    lead_scores.to_parquet(ARTIFACTS_DIR / "lead_scores.parquet", index=False)
    lead_scores.head(10000).to_csv(REPORTS_DIR / "lead_scores_top_sample.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    report = {
        "config": {
            "match_threshold": args.match_threshold,
            "review_threshold": args.review_threshold,
        },
        "counts": {
            "match_decisions": int(len(all_decisions)),
            "review_queue": int(len(review_queue)),
            "unified_profiles": int(len(unified_profiles)),
            "profile_mapping": int(len(profile_mapping)),
            "lead_scores": int(len(lead_scores)),
        },
        "decision_breakdown": all_decisions["decision"].value_counts().to_dict(),
        "decision_source_breakdown": all_decisions["decision_source"].value_counts().to_dict(),
        "merge_source_breakdown": unified_profiles["merge_source"].value_counts().to_dict(),
        "lead_tier_breakdown": lead_scores["tier"].value_counts().to_dict(),
        "paths": {
            "match_decisions": str(ARTIFACTS_DIR / "match_decisions.parquet"),
            "review_queue": str(ARTIFACTS_DIR / "review_queue.parquet"),
            "unified_profiles": str(ARTIFACTS_DIR / "unified_profiles.parquet"),
            "profile_mapping": str(ARTIFACTS_DIR / "profile_mapping.parquet"),
            "lead_scores": str(ARTIFACTS_DIR / "lead_scores.parquet"),
        },
    }
    (REPORTS_DIR / "crm_entity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Stage 15 CRM Entity Pipeline")
    print("=" * 72)
    print(f"match_decisions : {len(all_decisions):,}")
    print(f"review_queue    : {len(review_queue):,}")
    print(f"unified_profiles: {len(unified_profiles):,}")
    print(f"profile_mapping : {len(profile_mapping):,}")
    print(f"lead_scores     : {len(lead_scores):,}")
    print(f"Report          : {REPORTS_DIR / 'crm_entity_report.json'}")


if __name__ == "__main__":
    main()
