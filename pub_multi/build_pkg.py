from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve


PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DOC_DIR = PKG_DIR / "doc"
SRC_DIR = PKG_DIR / "src"
RUN_DIR = PKG_DIR / "run"
DATA_DIR = PKG_DIR / "data"
RES_DIR = PKG_DIR / "res"
FIG_DIR = PKG_DIR / "fig"
REF_DIR = PKG_DIR / "ref"
LOG_DIR = PKG_DIR / "log"

MAIN_RUN = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42"
BASE_RUN = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "text_attr_hybrid_r075_h20_s42"
IMG_STATS_RUN = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_stats_r075_h20_s42"
SUITE_ROOT = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite"
FULL_ROOT = PROJECT_ROOT / "train_data" / "stage7_14_full_candidate_pipeline"
CRM_ROOT = PROJECT_ROOT / "train_data" / "stage15_crm_entity_pipeline"
LEAK_SAFE_ROOT = PROJECT_ROOT / "train_data" / "leakage_safe_experiment"
STRICT_ROOT = PROJECT_ROOT / "train_data" / "stage10_13_training_noleak_strict"
TUNED_ROOT = PROJECT_ROOT / "train_data" / "stage7_8_rebuilt_experiment_tuned"

# This package is centered on the canonical multimodal line:
#   stage7_13_multimodal_suite -> stage7_14_full_candidate_pipeline -> stage15_crm_entity_pipeline
# Those pipelines use normalized_profiles_with_profile_id.csv because it includes both
# profile_id and profile_row_id, which are required to join back onto pair_features.parquet
# and score files from the chosen main run. In build_pkg.py this file is used as a lookup/
# enrichment table for the consolidated package, not as a new upstream training source.
PROFILE_LOOKUP_CSV = PROJECT_ROOT / "data_for_project" / "normalized_profiles_with_profile_id.csv"
PAIR_FEATURES = MAIN_RUN / "artifacts" / "pair_features.parquet"
TRAIN_SCORES = MAIN_RUN / "reports" / "train_scores.parquet"
VAL_SCORES = MAIN_RUN / "reports" / "val_scores.parquet"
TEST_SCORES = MAIN_RUN / "reports" / "test_scores.parquet"
MODEL_PKL = MAIN_RUN / "models" / "best_model.pkl"
FEATURE_COLS_PKL = MAIN_RUN / "models" / "feature_cols.pkl"
MAIN_REPORT = MAIN_RUN / "reports" / "experiment_report.json"
BASE_REPORT = BASE_RUN / "reports" / "experiment_report.json"
IMG_STATS_REPORT = IMG_STATS_RUN / "reports" / "experiment_report.json"
SUITE_REPORT = SUITE_ROOT / "reports" / "suite_report.json"
SUITE_LB = SUITE_ROOT / "reports" / "leaderboard.csv"
FULL_REPORT = FULL_ROOT / "reports" / "full_pipeline_report.json"
TOP_PREDS = FULL_ROOT / "reports" / "top_5000_predictions.csv"
OP_POINTS = FULL_ROOT / "reports" / "operating_points.csv"
FINAL_DEC_SAMPLE = FULL_ROOT / "reports" / "final_decisions_sample.csv"
CRM_REPORT = CRM_ROOT / "reports" / "crm_entity_report.json"
LEAK_SAFE_REPORT = LEAK_SAFE_ROOT / "reports" / "experiment_report.json"
STRICT_REPORT = STRICT_ROOT / "reports" / "evaluation_summary.json"
TUNING_SUMMARY = TUNED_ROOT / "tuning_summary.csv"

SOURCE_MAP = {
    "s01_prep.py": PROJECT_ROOT / "clean_data" / "preprocess_dataset.py",
    "s02_loc.py": PROJECT_ROOT / "clean_data" / "location_mapping_pipeline.py",
    "s03_norm.py": PROJECT_ROOT / "image_process" / "create_normalized_db.py",
    "s04_img.py": PROJECT_ROOT / "image_process" / "recover_profile_images.py",
    "s05_pair.py": WORKSPACE_ROOT / "stage8_pair_builder.py",
    "s06_feat.py": PROJECT_ROOT / "train_data" / "stage9_features_pipeline_chunked.py",
    "s07_base.py": PROJECT_ROOT / "train_data" / "leakage_safe_experiment" / "run_experiment.py",
    "s08_multi.py": PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "run_multimodal_suite.py",
    "s09_full.py": PROJECT_ROOT / "train_data" / "stage7_14_full_candidate_pipeline" / "run_full_candidate_pipeline.py",
    "s10_crm.py": PROJECT_ROOT / "train_data" / "stage15_crm_entity_pipeline" / "run_crm_entity_pipeline.py",
    "s11_train.py": PROJECT_ROOT / "train_data" / "stage10_13_training_pipeline.py",
    "s12_imgpair.py": PROJECT_ROOT / "image_process" / "profile_image_pair_features.py",
}

RUNNER_MAP = {
    "r_feat.ps1": PROJECT_ROOT / "train_data" / "run_stage9_pipeline.ps1",
    "r_multi.ps1": SUITE_ROOT / "run_multimodal_suite.ps1",
    "r_full.ps1": FULL_ROOT / "run_full_candidate_pipeline.ps1",
    "r_crm.ps1": CRM_ROOT / "run_crm_entity_pipeline.ps1",
}

REF_MAP = {
    "main_report.json": MAIN_REPORT,
    "base_report.json": BASE_REPORT,
    "img_stats_report.json": IMG_STATS_REPORT,
    "suite_report.json": SUITE_REPORT,
    "suite_lb.csv": SUITE_LB,
    "full_report.json": FULL_REPORT,
    "crm_report.json": CRM_REPORT,
    "leak_safe_report.json": LEAK_SAFE_REPORT,
    "strict_report.json": STRICT_REPORT,
    "tune.csv": TUNING_SUMMARY,
    "top_preds.csv": TOP_PREDS,
    "op_points.csv": OP_POINTS,
    "final_dec_sample.csv": FINAL_DEC_SAMPLE,
}


def ensure_dirs() -> None:
    for path in [PKG_DIR, DOC_DIR, SRC_DIR, RUN_DIR, DATA_DIR, RES_DIR, FIG_DIR, REF_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def copy_assets() -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for dst_name, src in SOURCE_MAP.items():
        dst = SRC_DIR / dst_name
        shutil.copy2(src, dst)
        copied.append({"dst": str(dst.relative_to(PKG_DIR)), "src": str(src)})
    for dst_name, src in RUNNER_MAP.items():
        dst = RUN_DIR / dst_name
        shutil.copy2(src, dst)
        copied.append({"dst": str(dst.relative_to(PKG_DIR)), "src": str(src)})
    for dst_name, src in REF_MAP.items():
        dst = REF_DIR / dst_name
        shutil.copy2(src, dst)
        copied.append({"dst": str(dst.relative_to(PKG_DIR)), "src": str(src)})
    return copied


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILE_LOOKUP_CSV)
    keep_cols = [
        "profile_row_id",
        "profile_id",
        "platform",
        "user_folder",
        "userName",
        "fullName",
        "bio",
        "location",
        "externalUrl_clean",
        "external_domain",
        "pictureURL",
    ]
    return df[keep_cols].copy()


def build_train_all(main_report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = pd.read_parquet(PAIR_FEATURES)
    score_df = pd.concat(
        [pd.read_parquet(TRAIN_SCORES), pd.read_parquet(VAL_SCORES), pd.read_parquet(TEST_SCORES)],
        ignore_index=True,
    )
    merged = feature_df.merge(
        score_df,
        on=["profile_row_id_a", "profile_row_id_b", "label", "pair_type", "split_name"],
        how="left",
    )

    profiles = load_profiles()
    left = profiles.add_suffix("_a")
    right = profiles.add_suffix("_b")
    merged = merged.merge(left, left_on="profile_row_id_a", right_on="profile_row_id_a", how="left")
    merged = merged.merge(right, left_on="profile_row_id_b", right_on="profile_row_id_b", how="left")

    threshold = float(main_report["metrics"]["threshold"])
    merged["pred_label"] = (merged["probability"] >= threshold).astype(int)
    merged["pred_class"] = np.where(merged["pred_label"] == 1, "MATCH", "NO_MATCH")
    merged["actual_class"] = np.where(merged["label"] == 1, "MATCH", "NO_MATCH")
    merged["is_fp"] = ((merged["label"] == 0) & (merged["pred_label"] == 1)).astype(int)
    merged["is_fn"] = ((merged["label"] == 1) & (merged["pred_label"] == 0)).astype(int)
    merged["is_tp"] = ((merged["label"] == 1) & (merged["pred_label"] == 1)).astype(int)
    merged["is_tn"] = ((merged["label"] == 0) & (merged["pred_label"] == 0)).astype(int)

    merged = merged.sort_values(["split_name", "label", "probability"], ascending=[True, False, False]).reset_index(drop=True)

    out_cols = [
        "split_name", "pair_type", "label", "pred_label", "pred_class", "probability",
        "profile_row_id_a", "platform_a", "userName_a", "fullName_a", "user_folder_a",
        "profile_row_id_b", "platform_b", "userName_b", "fullName_b", "user_folder_b",
        "username_jaro", "username_lev", "username_token_sort",
        "fullname_jaro", "fullname_lev", "fullname_token_sort",
        "bio_tfidf_cosine", "bio_sbert_cosine", "domain_jaccard", "url_jaccard",
        "location_jaro", "location_token_sort", "mention_jaccard", "hashtag_jaccard",
        "style_caps_diff", "style_avgword_diff", "style_biolen_ratio", "style_punct_diff",
        "platform_pair_code", "image_any_local", "image_both_local", "image_one_local_only",
        "image_phash_sim", "image_dhash_sim", "image_brightness_diff", "image_contrast_diff",
        "image_entropy_diff", "image_rgb_l1", "image_filesize_ratio", "image_face_count_diff",
        "image_face_area_diff", "image_blur_diff", "image_metadata_any", "image_caption_any",
        "image_caption_bio_sbert_cross", "image_caption_fullname_token_cross",
        "image_caption_username_token_cross", "is_tp", "is_tn", "is_fp", "is_fn",
    ]
    merged = merged[[col for col in out_cols if col in merged.columns]].copy()

    merged.to_parquet(DATA_DIR / "train_all.parquet", index=False)
    save_csv(merged, DATA_DIR / "train_all.csv")
    save_csv(merged.head(500), DATA_DIR / "train_all_sample.csv")

    fp_df = merged[(merged["split_name"] == "test") & (merged["is_fp"] == 1)].sort_values("probability", ascending=False).head(200)
    fn_df = merged[(merged["split_name"] == "test") & (merged["is_fn"] == 1)].sort_values("probability", ascending=True).head(200)
    save_csv(fp_df, DATA_DIR / "fp_top.csv")
    save_csv(fn_df, DATA_DIR / "fn_top.csv")
    return merged, fp_df, fn_df


def threshold_sweep(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = scores["label"].to_numpy()
    probs = scores["probability"].to_numpy()
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (probs >= thr).astype(int)
        rows.append(
            {
                "threshold": round(float(thr), 2),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "match_count": int(pred.sum()),
            }
        )
    df = pd.DataFrame(rows)
    save_csv(df, RES_DIR / "thr_sweep.csv")
    return df


def build_results(main_report: dict, full_report: dict, crm_report: dict) -> dict[str, pd.DataFrame]:
    suite_lb = pd.read_csv(SUITE_LB)
    tuning_df = pd.read_csv(TUNING_SUMMARY)
    leak_safe = read_json(LEAK_SAFE_REPORT)
    strict = read_json(STRICT_REPORT)
    base = read_json(BASE_REPORT)
    img_stats = read_json(IMG_STATS_REPORT)

    blocking = pd.DataFrame(
        [
            {"item": "profiles_total", "value": full_report["data_report"]["profiles_total"]},
            {"item": "all_cross_platform_pairs", "value": full_report["data_report"]["all_cross_platform_pairs"]},
            {"item": "ground_truth_positive_pairs", "value": full_report["data_report"]["ground_truth_positive_pairs"]},
            {"item": "exact_match_pairs", "value": full_report["blocking_report"]["exact_match_pairs"]},
            {"item": "candidate_pairs_for_model", "value": full_report["blocking_report"]["candidate_pairs_for_model"]},
            {"item": "search_space_reduction_pct", "value": full_report["blocking_report"]["search_space_reduction_pct"]},
            {"item": "ground_truth_coverage_pct", "value": full_report["blocking_report"]["ground_truth_coverage_pct"]},
        ]
    )
    save_csv(blocking, RES_DIR / "blocking.csv")

    blocking_keys = pd.DataFrame(
        [
            {"group": "exact", "name": key, "count": value}
            for key, value in full_report["blocking_report"]["exact_match_breakdown"].items()
        ]
        + [
            {"group": "candidate", "name": key, "count": value}
            for key, value in full_report["blocking_report"]["candidate_breakdown_by_key"].items()
        ]
    )
    save_csv(blocking_keys, RES_DIR / "blocking_keys.csv")

    tiers = pd.DataFrame(
        [
            {"tier": "exact", **full_report["production_threshold_report"]["tiers"]["exact"]},
            {"tier": "match", **full_report["production_threshold_report"]["tiers"]["match"]},
            {"tier": "review", **full_report["production_threshold_report"]["tiers"]["review"]},
        ]
    )
    save_csv(tiers, RES_DIR / "tiers.csv")

    save_csv(suite_lb, RES_DIR / "suite_cmp.csv")

    model_cmp = pd.DataFrame(
        [
            {
                "name": "text_attr_hybrid",
                "scope": "multimodal_suite",
                "model": base["metrics"]["best_model"],
                "feature_count": base["metrics"]["feature_count"],
                "test_ap": base["metrics"]["test_avg_precision"],
                "test_auc": base["metrics"]["test_roc_auc"],
                "test_f1": base["metrics"]["test_f1"],
                "test_precision": base["metrics"]["test_precision"],
                "test_recall": base["metrics"]["test_recall"],
            },
            {
                "name": "image_stats",
                "scope": "multimodal_suite",
                "model": img_stats["metrics"]["best_model"],
                "feature_count": img_stats["metrics"]["feature_count"],
                "test_ap": img_stats["metrics"]["test_avg_precision"],
                "test_auc": img_stats["metrics"]["test_roc_auc"],
                "test_f1": img_stats["metrics"]["test_f1"],
                "test_precision": img_stats["metrics"]["test_precision"],
                "test_recall": img_stats["metrics"]["test_recall"],
            },
            {
                "name": "image_context",
                "scope": "multimodal_suite",
                "model": main_report["metrics"]["best_model"],
                "feature_count": main_report["metrics"]["feature_count"],
                "test_ap": main_report["metrics"]["test_avg_precision"],
                "test_auc": main_report["metrics"]["test_roc_auc"],
                "test_f1": main_report["metrics"]["test_f1"],
                "test_precision": main_report["metrics"]["test_precision"],
                "test_recall": main_report["metrics"]["test_recall"],
            },
            {
                "name": "leakage_safe_gb",
                "scope": "baseline_ref",
                "model": leak_safe["metrics"]["best_model"],
                "feature_count": leak_safe["metrics"]["feature_count"],
                "test_ap": leak_safe["metrics"]["test_avg_precision"],
                "test_auc": leak_safe["metrics"]["test_roc_auc"],
                "test_f1": leak_safe["metrics"]["test_f1"],
                "test_precision": leak_safe["metrics"]["test_precision"],
                "test_recall": leak_safe["metrics"]["test_recall"],
            },
            {
                "name": "strict_noleak_mlp",
                "scope": "strict_ref",
                "model": "mlp",
                "feature_count": strict["n_features"],
                "test_ap": strict["test_avg_precision"],
                "test_auc": strict["test_roc_auc"],
                "test_f1": strict["test_f1"],
                "test_precision": strict["test_precision"],
                "test_recall": strict["test_recall"],
            },
        ]
    )
    save_csv(model_cmp, RES_DIR / "model_cmp.csv")

    tuning_rank = tuning_df.sort_values(["score", "test_ap", "test_f1"], ascending=False).reset_index(drop=True)
    save_csv(tuning_rank, RES_DIR / "tune_rank.csv")

    prod = pd.DataFrame(
        [
            {"metric": "match_decisions", "value": crm_report["counts"]["match_decisions"]},
            {"metric": "review_queue", "value": crm_report["counts"]["review_queue"]},
            {"metric": "unified_profiles", "value": crm_report["counts"]["unified_profiles"]},
            {"metric": "profile_mapping", "value": crm_report["counts"]["profile_mapping"]},
            {"metric": "lead_scores", "value": crm_report["counts"]["lead_scores"]},
        ]
    )
    save_csv(prod, RES_DIR / "prod.csv")

    modality = pd.DataFrame(
        [
            {"metric": "valid_profiles", "value": main_report["modality_report"]["profile_coverage"]["valid_profiles"]},
            {"metric": "profiles_with_local_image", "value": main_report["modality_report"]["profile_coverage"]["profiles_with_local_image"]},
            {"metric": "profiles_with_image_metadata", "value": main_report["modality_report"]["profile_coverage"]["profiles_with_image_metadata"]},
            {"metric": "profiles_with_caption", "value": main_report["modality_report"]["profile_coverage"]["profiles_with_caption"]},
        ]
    )
    save_csv(modality, RES_DIR / "modality.csv")

    return {
        "blocking": blocking,
        "blocking_keys": blocking_keys,
        "tiers": tiers,
        "suite_cmp": suite_lb,
        "model_cmp": model_cmp,
        "tuning": tuning_rank,
        "prod": prod,
        "modality": modality,
    }


def save_feature_importance() -> pd.DataFrame:
    import pickle

    with MODEL_PKL.open("rb") as fh:
        model = pickle.load(fh)
    with FEATURE_COLS_PKL.open("rb") as fh:
        feature_cols = pickle.load(fh)

    values = getattr(model, "feature_importances_", None)
    if values is None:
        df = pd.DataFrame(columns=["feature", "importance"])
    else:
        df = pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False)
    save_csv(df, RES_DIR / "feat_imp.csv")
    return df


def plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def save_fig(fig_name: str) -> Path:
    path = FIG_DIR / fig_name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def make_figures(
    main_report: dict,
    strict_report: dict,
    full_report: dict,
    suite_cmp: pd.DataFrame,
    tiers: pd.DataFrame,
    sweep_df: pd.DataFrame,
    feat_imp: pd.DataFrame,
) -> list[str]:
    plot_style()
    created: list[str] = []

    plt.figure()
    funnel_labels = ["all_cross", "exact", "candidates", "match_only", "review"]
    funnel_values = [
        full_report["data_report"]["all_cross_platform_pairs"],
        full_report["blocking_report"]["exact_match_pairs"],
        full_report["blocking_report"]["candidate_pairs_for_model"],
        full_report["production_threshold_report"]["final_match_only_count"],
        full_report["production_threshold_report"]["tiers"]["review"]["count"],
    ]
    sns.barplot(x=funnel_labels, y=funnel_values, hue=funnel_labels, palette="Blues_r", legend=False)
    plt.title("Retrieval Funnel")
    plt.ylabel("pairs")
    plt.xticks(rotation=20)
    created.append(str(save_fig("funnel.png").relative_to(PKG_DIR)))

    plt.figure()
    bk = pd.read_csv(RES_DIR / "blocking_keys.csv")
    sns.barplot(data=bk, x="name", y="count", hue="group")
    plt.title("Blocking Keys Contribution")
    plt.ylabel("pairs")
    plt.xlabel("key")
    plt.xticks(rotation=25)
    created.append(str(save_fig("blocking_keys.png").relative_to(PKG_DIR)))

    plt.figure()
    cmp_plot = suite_cmp.melt(
        id_vars=["experiment_kind"],
        value_vars=["test_avg_precision", "test_roc_auc", "test_f1"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=cmp_plot, x="experiment_kind", y="value", hue="metric")
    plt.title("Multimodal Suite Comparison")
    plt.ylabel("score")
    plt.xlabel("run")
    created.append(str(save_fig("suite_cmp.png").relative_to(PKG_DIR)))

    plt.figure()
    cm = np.array(main_report["metrics"]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Main Multimodal Run")
    plt.xlabel("pred")
    plt.ylabel("true")
    created.append(str(save_fig("cm_main.png").relative_to(PKG_DIR)))

    plt.figure()
    cm2 = np.array(strict_report["confusion_matrix"])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Strict No-Leak Reference")
    plt.xlabel("pred")
    plt.ylabel("true")
    created.append(str(save_fig("cm_strict.png").relative_to(PKG_DIR)))

    scores = pd.read_parquet(TEST_SCORES)
    plt.figure()
    sns.histplot(data=scores, x="probability", hue="label", bins=40, stat="density", common_norm=False)
    plt.title("Test Score Distribution")
    created.append(str(save_fig("score_hist.png").relative_to(PKG_DIR)))

    y_true = scores["label"].to_numpy()
    y_prob = scores["probability"].to_numpy()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, color="#0b6e4f", linewidth=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("recall")
    plt.ylabel("precision")
    created.append(str(save_fig("pr_curve.png").relative_to(PKG_DIR)))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color="#1d4ed8", linewidth=2, label=f"AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    created.append(str(save_fig("roc_curve.png").relative_to(PKG_DIR)))

    plt.figure()
    sweep_plot = sweep_df.melt(id_vars=["threshold", "match_count"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
    sns.lineplot(data=sweep_plot, x="threshold", y="value", hue="metric", marker="o")
    plt.title("Threshold Sweep")
    created.append(str(save_fig("thr_sweep.png").relative_to(PKG_DIR)))

    plt.figure()
    sns.barplot(data=tiers, x="tier", y="precision", color="#ef4444")
    plt.title("Production Tier Precision")
    plt.ylabel("precision")
    created.append(str(save_fig("tiers_precision.png").relative_to(PKG_DIR)))

    plt.figure()
    sns.barplot(data=tiers, x="tier", y="count", color="#f59e0b")
    plt.title("Production Tier Counts")
    plt.ylabel("pairs")
    created.append(str(save_fig("tiers_count.png").relative_to(PKG_DIR)))

    plt.figure()
    mod_df = pd.read_csv(RES_DIR / "modality.csv")
    sns.barplot(data=mod_df, x="metric", y="value", color="#7c3aed")
    plt.title("Modality Coverage")
    plt.ylabel("profiles")
    plt.xticks(rotation=20)
    created.append(str(save_fig("modality.png").relative_to(PKG_DIR)))

    if not feat_imp.empty:
        plt.figure()
        top = feat_imp.head(15).iloc[::-1]
        sns.barplot(data=top, x="importance", y="feature", color="#0f766e")
        plt.title("Top Feature Importance - Main Model")
        created.append(str(save_fig("feat_imp.png").relative_to(PKG_DIR)))

    tune_df = pd.read_csv(RES_DIR / "tune_rank.csv").head(10)
    plt.figure()
    sns.scatterplot(data=tune_df, x="test_recall", y="test_precision", size="score", hue="random_neg_ratio", legend="brief")
    plt.title("Tuning Trade-off")
    created.append(str(save_fig("tune_tradeoff.png").relative_to(PKG_DIR)))

    return created


def write_docs(main_report: dict, full_report: dict, crm_report: dict) -> list[str]:
    created: list[str] = []

    suite_cmp = pd.read_csv(RES_DIR / "suite_cmp.csv")
    nearest = suite_cmp.sort_values("composite_score", ascending=False).iloc[1]
    tune_best = pd.read_csv(RES_DIR / "tune_rank.csv").iloc[0]

    readme = textwrap.dedent(
        f"""\
        # pub_multi

        แพ็กเกจนี้เป็นเวอร์ชันหลักของ pipeline แบบ multimodal สำหรับใช้ทำเล่ม รายงาน และตรวจงาน โดยรวม code, data, results, plots และ docs ไว้ในจุดเดียว

        ## แกนหลักที่เลือก

        - main retrieval + production pipeline: `stage7_14_full_candidate_pipeline`
        - main training pipeline: `stage7_13_multimodal_suite`
        - main run: `image_context_r075_h20_s42`
        - chosen model inside the run: `{main_report["metrics"]["best_model"]}`

        ## ทำไมเลือกตัวนี้

        - test AP = {main_report["metrics"]["test_avg_precision"]:.4f}
        - test AUC = {main_report["metrics"]["test_roc_auc"]:.4f}
        - test F1 = {main_report["metrics"]["test_f1"]:.4f}
        - ได้ composite score สูงสุดใน suite
        - ตัวใกล้เคียงที่สุดคือ `{nearest["experiment_kind"]}` ซึ่งคะแนนใกล้มาก แต่ `image_context` ให้ AP/AUC สูงกว่าเล็กน้อยและสะท้อนแนวคิด multimodal ได้ครบกว่า

        ## ตัวเลขสำคัญ

        - all cross-platform pairs: {full_report["data_report"]["all_cross_platform_pairs"]:,}
        - exact matches: {full_report["blocking_report"]["exact_match_pairs"]:,}
        - candidate pairs for model: {full_report["blocking_report"]["candidate_pairs_for_model"]:,}
        - ground-truth coverage: {full_report["blocking_report"]["ground_truth_coverage_pct"]:.2f}%
        - final match-only precision: {full_report["production_threshold_report"]["final_match_only_precision"]:.4f}
        - final match-only recall: {full_report["production_threshold_report"]["final_match_only_recall"]:.4f}
        - review queue: {crm_report["counts"]["review_queue"]:,}

        ## tuning ที่อ้างอิง

        best tuning row จาก rebuilt experiment:
        - random_neg_ratio = {tune_best["random_neg_ratio"]}
        - hard_neg_ratio = {tune_best["hard_neg_ratio"]}
        - test_f1 = {tune_best["test_f1"]:.4f}
        """
    )
    (PKG_DIR / "README.md").write_text(readme, encoding="utf-8")
    created.append("README.md")

    use_md = textwrap.dedent(
        """\
        # USE

        1. อ่าน `README.md` ก่อน
        2. ดู `doc/stage.md` เพื่อ map stage 1-17 กับ code จริง
        3. เปิด `data/train_all.parquet` ถ้าต้องการชุดข้อมูลรวมที่ใช้ train/evaluate
        4. เปิด `res/model_cmp.csv`, `res/blocking.csv`, `res/tiers.csv` ถ้าต้องการ metric หลัก
        5. ใช้ภาพจาก `fig/` สำหรับรายงาน/สไลด์
        6. เริ่มอ่าน source หลักจาก `src/s08_multi.py`, `src/s09_full.py`, `src/s10_crm.py`
        7. ถ้าต้องการ flow การรันแบบเดิม ให้ใช้ไฟล์ใน `run/`
        """
    )
    (PKG_DIR / "USE.md").write_text(use_md, encoding="utf-8")
    created.append("USE.md")

    stage_md = textwrap.dedent(
        """\
        # STAGE MAP

        - Stage 1: data prep -> `src/s01_prep.py`
        - Stage 2: location normalization -> `src/s02_loc.py`
        - Stage 3: normalized profile DB -> `src/s03_norm.py`
        - Stage 4: image recovery -> `src/s04_img.py`
        - Stage 5: pair construction -> `src/s05_pair.py`
        - Stage 6: feature engineering -> `src/s06_feat.py`
        - Stage 7: leak-safe baseline training -> `src/s07_base.py`
        - Stage 8: multimodal training suite -> `src/s08_multi.py`
        - Stage 9: full candidate retrieval + scoring -> `src/s09_full.py`
        - Stage 10: CRM/entity export -> `src/s10_crm.py`
        - Stage 11: strict MLP reference -> `src/s11_train.py`
        - Stage 12: image pair helper -> `src/s12_imgpair.py`

        หมายเหตุ: งานเดิมมีหลาย notebook/หลายเวอร์ชันปนกันอยู่ แพ็กเกจนี้เลือกเส้นทางหลักแบบ multimodal เพื่อให้เล่มมี version เดียวและตัวเลขไม่ขัดกันเอง
        """
    )
    (DOC_DIR / "stage.md").write_text(stage_md, encoding="utf-8")
    created.append("doc/stage.md")

    method_md = textwrap.dedent(
        f"""\
        # METHOD

        แนวคิดหลักคือ exact-first + blocking + multimodal scoring + production tiers

        - exact matches = {full_report["blocking_report"]["exact_match_pairs"]:,}
        - candidates for model = {full_report["blocking_report"]["candidate_pairs_for_model"]:,}
        - coverage = {full_report["blocking_report"]["ground_truth_coverage_pct"]:.2f}%
        - search-space reduction = {full_report["blocking_report"]["search_space_reduction_pct"]:.2f}%
        """
    )
    (DOC_DIR / "method.md").write_text(method_md, encoding="utf-8")
    created.append("doc/method.md")

    eval_md = textwrap.dedent(
        f"""\
        # EVAL

        retrieval:
        - all cross-platform pairs = {full_report["data_report"]["all_cross_platform_pairs"]:,}
        - exact recall_global = {full_report["production_threshold_report"]["tiers"]["exact"]["recall_global"]:.4f}
        - exact + candidate coverage = {full_report["blocking_report"]["ground_truth_coverage_pct"]:.2f}%

        classification:
        - best model = {main_report["metrics"]["best_model"]}
        - test precision = {main_report["metrics"]["test_precision"]:.4f}
        - test recall = {main_report["metrics"]["test_recall"]:.4f}
        - test f1 = {main_report["metrics"]["test_f1"]:.4f}
        - confusion matrix = {main_report["metrics"]["confusion_matrix"]}

        production:
        - final match-only precision = {full_report["production_threshold_report"]["final_match_only_precision"]:.4f}
        - final match-only recall = {full_report["production_threshold_report"]["final_match_only_recall"]:.4f}
        - review queue = {crm_report["counts"]["review_queue"]:,}
        """
    )
    (DOC_DIR / "eval.md").write_text(eval_md, encoding="utf-8")
    created.append("doc/eval.md")

    model_md = textwrap.dedent(
        f"""\
        # MODEL

        main run: `image_context_r075_h20_s42`

        เหตุผล:
        - AP สูงสุดใน suite: {main_report["metrics"]["test_avg_precision"]:.4f}
        - AUC สูงสุดใน suite: {main_report["metrics"]["test_roc_auc"]:.4f}
        - F1 สูงมากและใกล้กับตัวที่ดีที่สุดในเชิง F1
        - ใช้ context จาก caption-to-text cross features ทำให้ narrative แบบ multimodal สมบูรณ์กว่า `image_stats`

        best model ภายใน run คือ Gradient Boosting เพราะถูกเลือกเป็น best model ในทั้ง 3 experiments ของ suite และตีความได้ผ่าน feature importance
        """
    )
    (DOC_DIR / "model.md").write_text(model_md, encoding="utf-8")
    created.append("doc/model.md")

    tune_md = textwrap.dedent(
        """\
        # TUNE

        tuning summary มาจาก `stage7_8_rebuilt_experiment_tuned/tuning_summary.csv`

        ประเด็นสำคัญ:
        - random negative ratio ที่ดีที่สุดอยู่แถว 0.75
        - hard negative ratio ที่ดีที่สุดอยู่แถว 2.0-2.5
        - multimodal suite ที่เลือกใช้จึงยึด `r075_h20`
        """
    )
    (DOC_DIR / "tune.md").write_text(tune_md, encoding="utf-8")
    created.append("doc/tune.md")

    report_md = textwrap.dedent(
        f"""\
        # REPORT

        ## 1. canonical pipeline

        แพ็กเกจนี้ยึด pipeline เดียวเป็นหลัก คือ
        - train/eval: `stage7_13_multimodal_suite`
        - production retrieval/scoring: `stage7_14_full_candidate_pipeline`
        - entity/export: `stage15_crm_entity_pipeline`

        เหตุผลที่เลือกเส้นนี้ เพราะเป็นเส้นเดียวที่เชื่อมกันครบตั้งแต่ retrieval ไปจนถึง production decision และมี report พร้อมทั้งด้าน coverage, classification และ thresholding

        ## 2. why blocking is still needed

        โจทย์นี้มี all cross-platform pairs = {full_report["data_report"]["all_cross_platform_pairs"]:,} คู่
        ถ้า score ทุกคู่โดยตรงจะมีต้นทุนสูงมาก จึงต้องใช้ exact-first และ blocking เพื่อคัดคู่ที่น่าจะเกี่ยวข้องก่อน

        retrieval summary:
        - exact_match_pairs = {full_report["blocking_report"]["exact_match_pairs"]:,}
        - candidate_pairs_for_model = {full_report["blocking_report"]["candidate_pairs_for_model"]:,}
        - search_space_reduction_pct = {full_report["blocking_report"]["search_space_reduction_pct"]:.4f}
        - ground_truth_coverage_pct = {full_report["blocking_report"]["ground_truth_coverage_pct"]:.4f}

        ประเด็นที่ต้องเขียนในเล่มคือ blocking stage ไม่ได้ optimize precision แต่ optimize candidate recall ภายใต้ข้อจำกัดด้านคอมพิวต์

        ## 3. why exact-first matters

        exact stage ให้ precision สูงมากและควรถูกเล่าแยกออกจาก model stage
        - exact precision = {full_report["production_threshold_report"]["tiers"]["exact"]["precision"]:.4f}
        - exact recall_global = {full_report["production_threshold_report"]["tiers"]["exact"]["recall_global"]:.4f}

        นี่คือคำตอบตรง ๆ ต่อคำถามว่า ทำไมไม่แยกคู่ที่มั่นใจแน่ ๆ ออกก่อนแล้วค่อยเอาที่เหลือเข้า model

        ## 4. why this multimodal run was chosen

        main run คือ `image_context_r075_h20_s42`
        - best model = {main_report["metrics"]["best_model"]}
        - feature_count = {main_report["metrics"]["feature_count"]}
        - test AP = {main_report["metrics"]["test_avg_precision"]:.4f}
        - test AUC = {main_report["metrics"]["test_roc_auc"]:.4f}
        - test F1 = {main_report["metrics"]["test_f1"]:.4f}
        - test precision = {main_report["metrics"]["test_precision"]:.4f}
        - test recall = {main_report["metrics"]["test_recall"]:.4f}
        - confusion matrix = {main_report["metrics"]["confusion_matrix"]}

        ตัวนี้ถูกเลือกเพราะเป็น run ที่ได้ composite score สูงสุดใน suite, ใช้ image context เพิ่มจาก text-only baseline และยังสามารถอธิบายผ่าน feature importance ได้

        ## 5. why Gradient Boosting was chosen

        Gradient Boosting ถูกเลือกเป็น best model ในทุก experiment ของ multimodal suite ไม่ใช่เฉพาะ main run
        ข้อดีสำหรับเล่ม:
        - เป็น tabular model ที่อธิบายง่ายกว่า deep neural net
        - มี feature importance ให้ดูได้
        - ให้ AP/AUC สูงกว่า logistic regression และชนะ random forest ใน suite นี้

        ## 6. tuning rationale

        tuning จาก rebuilt experiment ชี้ว่าค่า negative sampling ที่เหมาะคือบริเวณ
        - random_neg_ratio ~ 0.75
        - hard_neg_ratio ~ 2.0-2.5
        main multimodal suite จึงยึด `r075_h20`

        ## 7. production view

        final production metrics:
        - final_match_only_precision = {full_report["production_threshold_report"]["final_match_only_precision"]:.4f}
        - final_match_only_recall = {full_report["production_threshold_report"]["final_match_only_recall"]:.4f}
        - review_queue = {crm_report["counts"]["review_queue"]:,}
        - unified_profiles = {crm_report["counts"]["unified_profiles"]:,}

        สิ่งที่ควรอธิบายในเล่มคือ trade-off ระหว่าง precision สูงกับ review queue ที่ยังใหญ่ และเหตุผลเชิงระบบว่าทำไมต้องมี review tier
        """
    )
    (DOC_DIR / "report.md").write_text(report_md, encoding="utf-8")
    created.append("doc/report.md")

    file_map_md = textwrap.dedent(
        """\
        # FILE MAP

        ## root

        - `build_pkg.py` ตัวสร้างแพ็กเกจนี้
        - `README.md` สรุปภาพใหญ่
        - `USE.md` ขั้นตอนเปิดดูงาน
        - `FILES.csv` ดัชนีไฟล์สั้น ๆ

        ## src

        - `s01_prep.py` เตรียมข้อมูลดิบ
        - `s02_loc.py` mapping location
        - `s03_norm.py` สร้าง normalized profile DB
        - `s04_img.py` กู้/ดึงรูปโปรไฟล์
        - `s05_pair.py` สร้าง labeled pairs
        - `s06_feat.py` สร้าง features แบบ chunked
        - `s07_base.py` baseline leak-safe experiment
        - `s08_multi.py` multimodal suite หลัก
        - `s09_full.py` full candidate retrieval/scoring
        - `s10_crm.py` CRM/entity merge/export
        - `s11_train.py` strict MLP reference
        - `s12_imgpair.py` helper ด้าน image pair features

        ## data

        - `train_all.parquet` ไฟล์รวม train/val/test พร้อม score และ metadata
        - `train_all.csv` เวอร์ชัน CSV
        - `train_all_sample.csv` sample สำหรับเปิดดูเร็ว
        - `fp_top.csv` false positives สำคัญ
        - `fn_top.csv` false negatives สำคัญ

        ## res

        - `model_cmp.csv` เทียบ run/model สำคัญ
        - `blocking.csv` ตัวเลข retrieval หลัก
        - `blocking_keys.csv` contribution ของ exact/candidate keys
        - `tiers.csv` precision/count ของ exact-match-review
        - `thr_sweep.csv` ผลตาม threshold
        - `feat_imp.csv` feature importance ของ main model
        - `tune_rank.csv` ranking ของ tuning
        - `modality.csv` coverage ของภาพ/metadata
        - `prod.csv` ตัวเลขปลายทางด้าน CRM

        ## fig

        ไฟล์รูปทุกไฟล์ในโฟลเดอร์นี้ถูกสร้างใหม่จาก artifact จริง เพื่อใช้ในเล่มหรือสไลด์ได้ทันที

        ## ref

        copy ของ report/raw artifacts ที่ดึงมาจาก source เดิม เพื่อให้ตรวจย้อนกลับได้ว่า summary ทุกตัวมาจากไหน
        """
    )
    (DOC_DIR / "file_map.md").write_text(file_map_md, encoding="utf-8")
    created.append("doc/file_map.md")

    files_df = pd.DataFrame(
        [
            {"path": "src/", "meaning": "code Python ที่คัดแล้ว"},
            {"path": "run/", "meaning": "runner .ps1 แบบสั้น"},
            {"path": "data/train_all.parquet", "meaning": "ไฟล์เดียวรวมข้อมูล train/val/test + score + metadata"},
            {"path": "data/train_all.csv", "meaning": "CSV version ของ train_all"},
            {"path": "data/fp_top.csv", "meaning": "false positives สำคัญจาก test split"},
            {"path": "data/fn_top.csv", "meaning": "false negatives สำคัญจาก test split"},
            {"path": "res/model_cmp.csv", "meaning": "ตารางเปรียบเทียบ model/run สำคัญ"},
            {"path": "res/blocking.csv", "meaning": "summary ของ retrieval/blocking"},
            {"path": "res/tiers.csv", "meaning": "production tier metrics"},
            {"path": "res/thr_sweep.csv", "meaning": "precision/recall/f1 ตาม threshold"},
            {"path": "res/feat_imp.csv", "meaning": "feature importance ของ main model"},
            {"path": "fig/", "meaning": "visualize สำคัญสำหรับเล่ม"},
            {"path": "ref/", "meaning": "copy ของ report/raw artifacts จาก source เดิม"},
            {"path": "doc/", "meaning": "เอกสารวิธีอ่านและเหตุผลการเลือก"},
        ]
    )
    save_csv(files_df, PKG_DIR / "FILES.csv")
    created.append("FILES.csv")

    return created


def main() -> None:
    ensure_dirs()

    copied = copy_assets()
    main_report = read_json(MAIN_REPORT)
    full_report = read_json(FULL_REPORT)
    crm_report = read_json(CRM_REPORT)
    strict_report = read_json(STRICT_REPORT)

    train_all, fp_df, fn_df = build_train_all(main_report)
    sweep_df = threshold_sweep(pd.read_parquet(TEST_SCORES))
    tables = build_results(main_report, full_report, crm_report)
    feat_imp = save_feature_importance()
    figs = make_figures(
        main_report=main_report,
        strict_report=strict_report,
        full_report=full_report,
        suite_cmp=tables["suite_cmp"],
        tiers=tables["tiers"],
        sweep_df=sweep_df,
        feat_imp=feat_imp,
    )
    docs = write_docs(main_report, full_report, crm_report)

    manifest = {
        "package": str(PKG_DIR),
        "built_at": datetime.now().isoformat(),
        "main_run": str(MAIN_RUN),
        "main_model": main_report["metrics"]["best_model"],
        "rows_train_all": int(len(train_all)),
        "fp_rows_saved": int(len(fp_df)),
        "fn_rows_saved": int(len(fn_df)),
        "copied_assets": copied,
        "figures": figs,
        "docs": docs,
    }
    write_json(LOG_DIR / "build_log.json", manifest)
    print(f"Package built at: {PKG_DIR}")
    print(f"Main dataset   : {DATA_DIR / 'train_all.parquet'}")
    print(f"Figures        : {len(figs)}")
    print(f"Docs           : {len(docs)}")


if __name__ == "__main__":
    main()
