from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROFILES_PATH = PROJECT_ROOT / "data-for-project" / "normalized_profiles_with_profile_id.csv"
BASE_PIPELINE_PATH = PROJECT_ROOT / "Train-Data" / "stage7_8_rebuilt_experiment_hybrid" / "run_rebuilt_pipeline.py"
MULTIMODAL_SUITE_PATH = PROJECT_ROOT / "Train-Data" / "stage7_13_multimodal_suite" / "run_multimodal_suite.py"
BEST_RUN_PATH = PROJECT_ROOT / "Train-Data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42"

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
REPORTS_DIR = SCRIPT_DIR / "reports"
SCORES_DIR = SCRIPT_DIR / "scores"
TEMP_DIR = SCRIPT_DIR / "temp"

GENERIC_EXTERNAL_DOMAINS = {
    "about.me", "twitter.com", "facebook.com", "instagram.com", "youtube.com",
    "linkedin.com", "plus.google.com", "google.com", "bit.ly", "goo.gl",
    "youtu.be", "fb.com", "t.co", "ow.ly",
}

BLOCKING_KEYS = ["username_prefix3", "fullname_prefix3", "external_domain"]


def ensure_dirs() -> None:
    for path in [SCRIPT_DIR, ARTIFACTS_DIR, REPORTS_DIR, SCORES_DIR, TEMP_DIR]:
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


def prefix3(value: object) -> str:
    text = lower_text(value)
    text = re.sub(r"^@", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text[:3] if len(text) >= 3 else ""


def pair_key(a: int, b: int) -> int:
    x, y = (a, b) if a < b else (b, a)
    return (x << 32) | y


def unpack_pair_key(key: int) -> tuple[int, int]:
    return key >> 32, key & 0xFFFFFFFF


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_PATH, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df["profile_id"] = pd.to_numeric(df["profile_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna()].copy()
    df = df[df["profile_id"].notna()].copy()
    df["platform"] = df["platform"].map(lower_text)
    df["userName_norm"] = df["userName"].map(lower_text)
    df["fullName_norm"] = df["fullName"].map(lower_text)
    df["bio_norm"] = df["bio"].map(lower_text)
    df["username_prefix3"] = df["userName"].map(prefix3)
    df["fullname_prefix3"] = df["fullName"].map(prefix3)
    df["external_domain"] = df["external_domain"].map(lower_text)
    df["external_domain"] = df["external_domain"].map(
        lambda x: "" if (not x or x in GENERIC_EXTERNAL_DOMAINS or x == "nan") else x
    )
    df["externalUrl_clean"] = df["externalUrl_clean"].map(lower_text)
    df["externalUrl_clean"] = df["externalUrl_clean"].map(lambda x: "" if x == "nan" else x)
    return df


def count_cross_platform_all_pairs(df: pd.DataFrame) -> int:
    counts = df["platform"].value_counts().to_dict()
    platforms = sorted(counts)
    total = 0
    for i in range(len(platforms)):
        for j in range(i + 1, len(platforms)):
            total += int(counts[platforms[i]]) * int(counts[platforms[j]])
    return total


def build_ground_truth_set(base_module, profiles: pd.DataFrame) -> tuple[set[int], pd.DataFrame]:
    gt_pairs = base_module.build_ground_truth_pairs(profiles)
    gt_set = {
        pair_key(int(row.profile_row_id_a), int(row.profile_row_id_b))
        for row in gt_pairs.itertuples(index=False)
    }
    return gt_set, gt_pairs


def build_exact_match_pairs(df: pd.DataFrame) -> tuple[set[int], dict[str, int]]:
    exact_keys: set[int] = set()
    counters = {"username_exact": 0, "external_url_exact": 0}

    def add_group_pairs(group: pd.DataFrame, counter_name: str) -> None:
        added = 0
        recs = group[["profile_row_id", "platform"]].to_records(index=False)
        n = len(recs)
        for i in range(n):
            a_id = int(recs[i][0])
            a_platform = recs[i][1]
            for j in range(i + 1, n):
                b_id = int(recs[j][0])
                b_platform = recs[j][1]
                if a_platform == b_platform:
                    continue
                key = pair_key(a_id, b_id)
                if key not in exact_keys:
                    exact_keys.add(key)
                    added += 1
        counters[counter_name] += added

    uname_df = df[df["userName_norm"].str.len() >= 3].copy()
    for _, group in uname_df.groupby("userName_norm", sort=False):
        if len(group) >= 2:
            add_group_pairs(group, "username_exact")

    url_df = df[df["externalUrl_clean"].str.len() > 0].copy()
    for _, group in url_df.groupby("externalUrl_clean", sort=False):
        if len(group) >= 2:
            add_group_pairs(group, "external_url_exact")

    return exact_keys, counters


def build_candidate_pairs(df: pd.DataFrame, exact_keys: set[int]) -> tuple[set[int], dict[str, int]]:
    candidate_keys: set[int] = set()
    counts_by_key: dict[str, int] = {}

    for key_col in BLOCKING_KEYS:
        added = 0
        keyed = df[df[key_col].map(bool)].copy()
        for _, group in keyed.groupby(key_col, sort=False):
            recs = group[["profile_row_id", "platform"]].to_records(index=False)
            n = len(recs)
            if n < 2:
                continue
            for i in range(n):
                a_id = int(recs[i][0])
                a_platform = recs[i][1]
                for j in range(i + 1, n):
                    b_id = int(recs[j][0])
                    b_platform = recs[j][1]
                    if a_platform == b_platform:
                        continue
                    key = pair_key(a_id, b_id)
                    if key in exact_keys or key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    added += 1
        counts_by_key[key_col] = added
    return candidate_keys, counts_by_key


def export_pair_keys(keys: set[int], output_path: Path, pair_type: str, chunk_size: int = 500_000) -> int:
    rows = []
    total = 0
    part = 0
    for key in sorted(keys):
        a, b = unpack_pair_key(key)
        rows.append({"profile_row_id_a": a, "profile_row_id_b": b, "pair_type": pair_type})
        if len(rows) >= chunk_size:
            part_path = output_path.parent / f"{output_path.stem}_part{part:03d}.parquet"
            pd.DataFrame(rows).to_parquet(part_path, index=False)
            total += len(rows)
            rows = []
            part += 1
    if rows:
        part_path = output_path.parent / f"{output_path.stem}_part{part:03d}.parquet"
        pd.DataFrame(rows).to_parquet(part_path, index=False)
        total += len(rows)
    return total


def load_best_model_artifacts():
    with (BEST_RUN_PATH / "models" / "best_model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    with (BEST_RUN_PATH / "models" / "scaler.pkl").open("rb") as fh:
        scaler = pickle.load(fh)
    with (BEST_RUN_PATH / "models" / "calibrator.pkl").open("rb") as fh:
        calibrator = pickle.load(fh)
    with (BEST_RUN_PATH / "models" / "feature_cols.pkl").open("rb") as fh:
        feature_cols = pickle.load(fh)
    report = json.loads((BEST_RUN_PATH / "reports" / "experiment_report.json").read_text(encoding="utf-8"))
    threshold = float(report["metrics"]["threshold"])
    return model, scaler, calibrator, feature_cols, threshold


def load_scored_candidates() -> pd.DataFrame:
    paths = sorted(SCORES_DIR.glob("candidate_scores_*.parquet"))
    if not paths:
        raise FileNotFoundError("No candidate score chunks found.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def build_scoring_engine():
    base = load_module("full_candidate_base_hybrid", BASE_PIPELINE_PATH)
    suite = load_module("full_candidate_multimodal_suite", MULTIMODAL_SUITE_PATH)
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
    return base, profiles, profile_bundle, wrapped_compute


def score_candidates_chunked(
    candidate_keys: list[int],
    gt_set: set[int],
    compute_features_fn,
    profile_bundle: dict[str, object],
    model,
    scaler,
    calibrator,
    feature_cols: list[str],
    threshold: float,
    chunk_size: int,
) -> dict[str, object]:
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    predicted_match_rows = []
    top_rows = []
    predicted_count = 0
    tp = 0
    fp = 0

    for chunk_idx, start in enumerate(range(0, len(candidate_keys), chunk_size)):
        stop = min(start + chunk_size, len(candidate_keys))
        rows = []
        for key in candidate_keys[start:stop]:
            a, b = unpack_pair_key(key)
            rows.append({
                "profile_row_id_a": a,
                "profile_row_id_b": b,
                "label": 1 if key in gt_set else 0,
                "pair_type": "candidate",
                "split_name": "full",
                "profile_id": -1,
            })
        pair_df = pd.DataFrame(rows)
        feature_df, _ = compute_features_fn(pair_df, profile_bundle)
        x = scaler.transform(feature_df[feature_cols].fillna(0.0))
        raw = model.predict_proba(x)[:, 1]
        probs = calibrator.predict(raw)
        preds = (probs >= threshold).astype(int)
        labels = feature_df["label"].to_numpy()

        y_all.append(labels)
        p_all.append(np.asarray(probs))

        score_df = feature_df[["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name"]].copy()
        score_df["probability"] = probs
        score_df["pred_match"] = preds
        score_path = SCORES_DIR / f"candidate_scores_{start:09d}_{stop:09d}.parquet"
        score_df.to_parquet(score_path, index=False)

        pred_df = score_df[score_df["pred_match"] == 1].copy()
        if not pred_df.empty:
            predicted_count += int(len(pred_df))
            tp += int(((pred_df["label"] == 1)).sum())
            fp += int(((pred_df["label"] == 0)).sum())
            predicted_match_rows.append(pred_df)

        top_rows.append(score_df.nlargest(min(1000, len(score_df)), "probability"))
        print(f"[SCORE] chunk {chunk_idx + 1}: {start:,} - {stop:,} / {len(candidate_keys):,}")

    y = np.concatenate(y_all) if y_all else np.array([], dtype=int)
    probs = np.concatenate(p_all) if p_all else np.array([], dtype=float)
    pred_matches = pd.concat(predicted_match_rows, ignore_index=True) if predicted_match_rows else pd.DataFrame(
        columns=["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name", "probability", "pred_match"]
    )
    top_scores = pd.concat(top_rows, ignore_index=True).sort_values("probability", ascending=False).head(5000)

    pred_matches.to_parquet(ARTIFACTS_DIR / "predicted_matches.parquet", index=False)
    top_scores.to_csv(REPORTS_DIR / "top_5000_predictions.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    metrics = {
        "candidate_count_scored": int(len(y)),
        "candidate_positive_rate": float(y.mean()) if len(y) else 0.0,
        "candidate_avg_precision": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else None,
        "candidate_roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None,
        "predicted_match_count": int(predicted_count),
        "predicted_true_positive_count": int(tp),
        "predicted_false_positive_count": int(fp),
        "threshold_precision": float(precision_score(y, (probs >= threshold).astype(int), zero_division=0)),
        "threshold_recall_within_candidates": float(recall_score(y, (probs >= threshold).astype(int), zero_division=0)),
        "threshold_f1_within_candidates": float(f1_score(y, (probs >= threshold).astype(int), zero_division=0)),
    }

    for k in [100, 500, 1000, 5000]:
        if len(top_scores) >= k:
            metrics[f"precision_at_{k}"] = float(top_scores.head(k)["label"].mean())

    return {"metrics": metrics, "predicted_matches": pred_matches, "top_scores": top_scores}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full leak-safe candidate pipeline on real candidate pairs")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--match-threshold", type=float, default=0.98)
    parser.add_argument("--review-threshold", type=float, default=0.95)
    parser.add_argument("--reuse-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    base = load_module("full_candidate_base", BASE_PIPELINE_PATH)
    profiles = load_profiles()
    gt_set, gt_pairs = build_ground_truth_set(base, profiles)
    total_cross_platform_pairs = count_cross_platform_all_pairs(profiles)

    print("=" * 72)
    print("Stage 7-14 Full Candidate Pipeline")
    print("=" * 72)
    print(f"Profiles loaded            : {len(profiles):,}")
    print(f"Ground-truth positive pairs: {len(gt_set):,}")
    print(f"All cross-platform pairs   : {total_cross_platform_pairs:,}")

    exact_keys, exact_counts = build_exact_match_pairs(profiles)
    candidate_keys, candidate_counts = build_candidate_pairs(profiles, exact_keys)

    export_pair_keys(exact_keys, ARTIFACTS_DIR / "exact_matches.parquet", "exact_match")
    export_pair_keys(candidate_keys, ARTIFACTS_DIR / "candidate_pairs.parquet", "candidate")

    gt_exact = len(exact_keys & gt_set)
    gt_candidate = len(candidate_keys & gt_set)
    gt_total_covered = len((exact_keys | candidate_keys) & gt_set)

    model, scaler, calibrator, feature_cols, threshold = load_best_model_artifacts()
    candidate_key_list = sorted(candidate_keys)
    if args.reuse_scores and sorted(SCORES_DIR.glob("candidate_scores_*.parquet")):
        scored_df = load_scored_candidates()
        scoring_result = {
            "metrics": {
                "candidate_count_scored": int(len(scored_df)),
                "candidate_positive_rate": float(scored_df["label"].mean()) if len(scored_df) else 0.0,
                "candidate_avg_precision": float(average_precision_score(scored_df["label"], scored_df["probability"])) if len(scored_df["label"].unique()) > 1 else None,
                "candidate_roc_auc": float(roc_auc_score(scored_df["label"], scored_df["probability"])) if len(scored_df["label"].unique()) > 1 else None,
                "predicted_match_count": int((scored_df["probability"] >= threshold).sum()),
                "predicted_true_positive_count": int(((scored_df["probability"] >= threshold) & (scored_df["label"] == 1)).sum()),
                "predicted_false_positive_count": int(((scored_df["probability"] >= threshold) & (scored_df["label"] == 0)).sum()),
                "threshold_precision": float(precision_score(scored_df["label"], (scored_df["probability"] >= threshold).astype(int), zero_division=0)),
                "threshold_recall_within_candidates": float(recall_score(scored_df["label"], (scored_df["probability"] >= threshold).astype(int), zero_division=0)),
                "threshold_f1_within_candidates": float(f1_score(scored_df["label"], (scored_df["probability"] >= threshold).astype(int), zero_division=0)),
            },
            "predicted_matches": scored_df[scored_df["probability"] >= threshold].copy(),
            "top_scores": scored_df.sort_values("probability", ascending=False).head(5000).copy(),
        }
        for k in [100, 500, 1000, 5000]:
            if len(scoring_result["top_scores"]) >= k:
                scoring_result["metrics"][f"precision_at_{k}"] = float(scoring_result["top_scores"].head(k)["label"].mean())
        scoring_result["predicted_matches"].to_parquet(ARTIFACTS_DIR / "predicted_matches.parquet", index=False)
        scoring_result["top_scores"].to_csv(REPORTS_DIR / "top_5000_predictions.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    else:
        scoring_base, scoring_profiles, profile_bundle, compute_features_fn = build_scoring_engine()
        del scoring_profiles, scoring_base
        scoring_result = score_candidates_chunked(
            candidate_key_list,
            gt_set,
            compute_features_fn,
            profile_bundle,
            model,
            scaler,
            calibrator,
            feature_cols,
            threshold,
            args.chunk_size,
        )

    pred_matches = scoring_result["predicted_matches"]
    model_tp = int(pred_matches["label"].sum()) if not pred_matches.empty else 0
    final_tp = gt_exact + model_tp
    final_predicted = len(exact_keys) + int(len(pred_matches))
    final_precision = final_tp / final_predicted if final_predicted else 0.0
    final_recall = final_tp / len(gt_set) if gt_set else 0.0

    scored_df = load_scored_candidates()
    if args.review_threshold > args.match_threshold:
        raise ValueError("review-threshold must be <= match-threshold")
    scored_df["decision"] = np.where(
        scored_df["probability"] >= args.match_threshold,
        "MATCH",
        np.where(scored_df["probability"] >= args.review_threshold, "REVIEW", "NON_MATCH"),
    )
    match_df = scored_df[scored_df["decision"] == "MATCH"].copy()
    review_df = scored_df[scored_df["decision"] == "REVIEW"].copy()

    exact_df = pd.concat([pd.read_parquet(path) for path in sorted(ARTIFACTS_DIR.glob("exact_matches_part*.parquet"))], ignore_index=True)
    exact_df["probability"] = 1.0
    exact_df["decision"] = "MATCH_EXACT"
    exact_df["label"] = exact_df.apply(lambda row: 1 if pair_key(int(row["profile_row_id_a"]), int(row["profile_row_id_b"])) in gt_set else 0, axis=1)

    final_decisions = pd.concat(
        [
            exact_df[["profile_row_id_a", "profile_row_id_b", "pair_type", "label", "probability", "decision"]],
            match_df[["profile_row_id_a", "profile_row_id_b", "pair_type", "label", "probability", "decision"]],
            review_df[["profile_row_id_a", "profile_row_id_b", "pair_type", "label", "probability", "decision"]],
        ],
        ignore_index=True,
    )
    final_decisions.to_parquet(ARTIFACTS_DIR / "final_decisions.parquet", index=False)
    final_decisions.head(10000).to_csv(REPORTS_DIR / "final_decisions_sample.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    tier_metrics = {}
    for name, df in [("exact", exact_df), ("match", match_df), ("review", review_df)]:
        tp_count = int(df["label"].sum()) if not df.empty else 0
        count = int(len(df))
        tier_metrics[name] = {
            "count": count,
            "true_positive_count": tp_count,
            "precision": float(tp_count / count) if count else 0.0,
            "recall_global": float(tp_count / len(gt_set)) if gt_set else 0.0,
        }

    final_prod_count = int(len(exact_df) + len(match_df))
    final_prod_tp = int(exact_df["label"].sum() + match_df["label"].sum())
    final_prod_precision = float(final_prod_tp / final_prod_count) if final_prod_count else 0.0
    final_prod_recall = float(final_prod_tp / len(gt_set)) if gt_set else 0.0

    report = {
        "data_report": {
            "profiles_total": int(len(profiles)),
            "platform_counts": profiles["platform"].value_counts().to_dict(),
            "ground_truth_positive_pairs": int(len(gt_set)),
            "all_cross_platform_pairs": int(total_cross_platform_pairs),
        },
        "blocking_report": {
            "exact_match_pairs": int(len(exact_keys)),
            "exact_match_breakdown": exact_counts,
            "candidate_pairs_for_model": int(len(candidate_keys)),
            "candidate_breakdown_by_key": candidate_counts,
            "search_space_reduction_pct": float(100.0 * (1.0 - (len(exact_keys) + len(candidate_keys)) / total_cross_platform_pairs)),
            "ground_truth_covered_by_exact": int(gt_exact),
            "ground_truth_covered_by_candidates": int(gt_candidate),
            "ground_truth_covered_total": int(gt_total_covered),
            "ground_truth_coverage_pct": float(100.0 * gt_total_covered / len(gt_set)) if gt_set else 0.0,
        },
        "model_report": {
            "source_run": str(BEST_RUN_PATH),
            "feature_count": int(len(feature_cols)),
            "threshold": float(threshold),
            **scoring_result["metrics"],
        },
        "final_decision_report": {
            "exact_matches_auto_accepted": int(len(exact_keys)),
            "model_matches_above_threshold": int(len(pred_matches)),
            "final_predicted_matches": int(final_predicted),
            "final_true_positive_matches": int(final_tp),
            "final_precision_against_ground_truth": float(final_precision),
            "final_recall_against_ground_truth": float(final_recall),
        },
        "production_threshold_report": {
            "match_threshold": float(args.match_threshold),
            "review_threshold": float(args.review_threshold),
            "tiers": tier_metrics,
            "final_match_only_count": final_prod_count,
            "final_match_only_true_positive_count": final_prod_tp,
            "final_match_only_precision": final_prod_precision,
            "final_match_only_recall": final_prod_recall,
        },
        "paths": {
            "exact_matches_dir": str(ARTIFACTS_DIR),
            "candidate_pairs_dir": str(ARTIFACTS_DIR),
            "scores_dir": str(SCORES_DIR),
            "predicted_matches": str(ARTIFACTS_DIR / "predicted_matches.parquet"),
            "top_predictions": str(REPORTS_DIR / "top_5000_predictions.csv"),
            "final_decisions": str(ARTIFACTS_DIR / "final_decisions.parquet"),
        },
    }

    (REPORTS_DIR / "full_pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    gt_pairs.to_parquet(ARTIFACTS_DIR / "ground_truth_pairs.parquet", index=False)
    gt_pairs.to_csv(ARTIFACTS_DIR / "ground_truth_pairs.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(f"Exact matches             : {len(exact_keys):,}")
    print(f"Candidate pairs scored    : {len(candidate_keys):,}")
    print(f"Ground-truth coverage     : {report['blocking_report']['ground_truth_coverage_pct']:.2f}%")
    print(f"Final precision           : {final_precision:.4f}")
    print(f"Final recall              : {final_recall:.4f}")
    print(f"Prod precision            : {final_prod_precision:.4f}")
    print(f"Prod recall               : {final_prod_recall:.4f}")
    print(f"Report                    : {REPORTS_DIR / 'full_pipeline_report.json'}")


if __name__ == "__main__":
    main()
