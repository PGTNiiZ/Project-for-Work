from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FIG_DIR = ROOT / "fig"
REF_DIR = ROOT / "ref"
RES_DIR = ROOT / "res"

FULL_REPORT = REF_DIR / "full_report.json"
CRM_REPORT = REF_DIR / "crm_report.json"
FEATURE_IMPORTANCE = RES_DIR / "feat_imp.csv"
SUITE_COMPARE = RES_DIR / "suite_cmp.csv"
MODEL_FAMILY_COMPARE = RES_DIR / "model_family_cmp.csv"
RUN_REPORTS_DIR = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42" / "reports"
SUITE_RUNS_DIR = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs"


RUN_LABELS = {
    "text_attr_hybrid_r075_h20_s42": "Text / Attribute\nHybrid",
    "image_stats_r075_h20_s42": "Image\nStatistics",
    "image_context_r075_h20_s42": "Image\nContext",
}

METRIC_LABELS = {
    "test_avg_precision": "AP",
    "test_roc_auc": "ROC-AUC",
    "test_f1": "F1",
    "test_precision": "Precision",
    "test_recall": "Recall",
}

GROUP_COLORS = {
    "Name similarity": "#0f766e",
    "Bio text": "#14b8a6",
    "Style / writing": "#38bdf8",
    "Platform pair": "#6366f1",
    "Image-caption cross-signal": "#f59e0b",
    "URL / domain": "#f97316",
    "Image signal": "#ef4444",
    "Mention / hashtag": "#8b5cf6",
    "Location": "#94a3b8",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fbfdff"


def save(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, x: float, y: float, w: float, h: float, title: str, lines: list[str], facecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor="#0f172a",
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontsize=14, fontweight="bold", color="#0f172a")
    for idx, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.12 - idx * 0.06, line, ha="center", va="top", fontsize=11, color="#0f172a")


def arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#475569"))


def feature_group(name: str) -> str:
    if name.startswith(("fullname_", "username_")):
        return "Name similarity"
    if name.startswith("bio_"):
        return "Bio text"
    if name.startswith("style_"):
        return "Style / writing"
    if name.startswith("platform_"):
        return "Platform pair"
    if name.startswith("image_caption_"):
        return "Image-caption cross-signal"
    if name.startswith("domain_") or name.startswith("url_"):
        return "URL / domain"
    if name.startswith("image_"):
        return "Image signal"
    if name.startswith("mention_") or name.startswith("hashtag_"):
        return "Mention / hashtag"
    if name.startswith("location_"):
        return "Location"
    return "Location"


def make_threshold_flow(full_report: dict, crm_report: dict) -> None:
    data = full_report["data_report"]
    blocking = full_report["blocking_report"]
    model = full_report["model_report"]
    threshold = full_report["production_threshold_report"]
    breakdown = crm_report["decision_breakdown"]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.04,
        0.30,
        0.20,
        0.40,
        "Search Space",
        [
            f"Cross-platform pairs = {data['all_cross_platform_pairs']:,}",
            f"Valid profiles = {data['profiles_total']:,}",
            f"Ground-truth matches = {data['ground_truth_positive_pairs']:,}",
        ],
        "#dbeafe",
    )
    add_box(
        ax,
        0.31,
        0.22,
        0.24,
        0.56,
        "Exact-First + Blocking",
        [
            f"Exact auto-match = {blocking['exact_match_pairs']:,}",
            f"Model candidates = {blocking['candidate_pairs_for_model']:,}",
            f"Reduction = {blocking['search_space_reduction_pct']:.2f}%",
            f"Coverage = {blocking['ground_truth_coverage_pct']:.2f}%",
        ],
        "#e0f2fe",
    )
    add_box(
        ax,
        0.62,
        0.22,
        0.22,
        0.56,
        "Calibrated Candidate Scoring",
        [
            f"Feature count = {model['feature_count']}",
            f"Candidate AP = {model['candidate_avg_precision']:.4f}",
            f"Candidate ROC-AUC = {model['candidate_roc_auc']:.4f}",
            f"Decision thresholds = {threshold['match_threshold']:.2f} / {threshold['review_threshold']:.2f}",
        ],
        "#dcfce7",
    )

    add_box(ax, 0.87, 0.62, 0.11, 0.16, "MATCH", [f"{breakdown['MATCH']:,} pairs", "Auto-merge"], "#bbf7d0")
    add_box(ax, 0.87, 0.40, 0.11, 0.16, "REVIEW", [f"{breakdown['REVIEW']:,} pairs", "Human review"], "#fde68a")
    add_box(ax, 0.87, 0.18, 0.11, 0.16, "NO_MATCH", [f"{breakdown['NO_MATCH']:,} pairs", "Auto discard"], "#fecaca")

    arrow(ax, 0.24, 0.50, 0.31, 0.50)
    arrow(ax, 0.55, 0.50, 0.62, 0.50)
    arrow(ax, 0.84, 0.60, 0.87, 0.70)
    arrow(ax, 0.84, 0.50, 0.87, 0.48)
    arrow(ax, 0.84, 0.40, 0.87, 0.26)

    ax.text(
        0.50,
        0.06,
        "Exact rules accept unambiguous pairs first, then the calibrated model routes the remaining candidates into MATCH, REVIEW, and NO_MATCH.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    ax.set_title("Decision Thresholding Pipeline Used for the CRM Operating Workflow", pad=18, fontweight="bold")
    save(fig, "ch04_threshold_flow.png")


def make_model_pipeline(experiment_report: dict) -> None:
    metrics = experiment_report["metrics"]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.03,
        0.26,
        0.19,
        0.48,
        "Input Evidence",
        [
            "username / fullname",
            "bio / URL / domain",
            "location / mentions / hashtags",
            "image / caption metadata",
        ],
        "#dbeafe",
    )
    add_box(
        ax,
        0.28,
        0.16,
        0.24,
        0.68,
        "Pair Feature Engineering",
        [
            "22 baseline text + attribute features",
            "1 SBERT semantic feature",
            "14 image presence / stats / metadata features",
            "4 caption cross-signal features",
            "Total = 41 features",
        ],
        "#e0f2fe",
    )
    add_box(
        ax,
        0.58,
        0.16,
        0.20,
        0.68,
        "Candidate Scoring Model",
        [
            "Train: GB / RF / LogReg",
            f"Best model = {str(metrics['best_model']).upper()}",
            f"Val AP = {metrics['val_avg_precision']:.4f}",
            f"Val ROC-AUC = {metrics['val_roc_auc']:.4f}",
            f"Val F1 = {metrics['val_f1']:.4f}",
        ],
        "#dcfce7",
    )
    add_box(
        ax,
        0.83,
        0.16,
        0.14,
        0.68,
        "Calibration + Decision",
        [
            "Probability calibration",
            "Threshold for test F1 = 0.35",
            "Production thresholds =",
            "0.98 MATCH / 0.95 REVIEW",
        ],
        "#fee2e2",
    )

    arrow(ax, 0.22, 0.50, 0.28, 0.50)
    arrow(ax, 0.52, 0.50, 0.58, 0.50)
    arrow(ax, 0.78, 0.50, 0.83, 0.50)
    ax.text(
        0.5,
        0.05,
        "The main run is a leak-safe tabular candidate scorer: engineered pair features are scored by the best validation model, then calibrated and converted into CRM decisions.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    ax.set_title("Main-Run Model Pipeline Behind the Final Thresholding Stage", pad=18, fontweight="bold")
    save(fig, "ch04_model_pipeline.png")


def make_threshold_dashboard(full_report: dict, crm_report: dict) -> None:
    threshold = full_report["production_threshold_report"]
    breakdown = crm_report["decision_breakdown"]
    total = sum(breakdown.values())

    count_df = pd.DataFrame(
        [
            {"tier": "MATCH", "count": breakdown["MATCH"], "share": breakdown["MATCH"] / total},
            {"tier": "REVIEW", "count": breakdown["REVIEW"], "share": breakdown["REVIEW"] / total},
            {"tier": "NO_MATCH", "count": breakdown["NO_MATCH"], "share": breakdown["NO_MATCH"] / total},
        ]
    )

    precision_df = pd.DataFrame(
        [
            {"label": "Exact auto", "score": threshold["tiers"]["exact"]["precision"]},
            {"label": "Score >= 0.98", "score": threshold["tiers"]["match"]["precision"]},
            {"label": "Final MATCH", "score": threshold["final_match_only_precision"]},
            {"label": "REVIEW rate", "score": threshold["tiers"]["review"]["precision"]},
        ]
    )

    recall_df = pd.DataFrame(
        [
            {"label": "Exact auto", "score": threshold["tiers"]["exact"]["recall_global"]},
            {"label": "Score >= 0.98", "score": threshold["tiers"]["match"]["recall_global"]},
            {"label": "Final MATCH", "score": threshold["final_match_only_recall"]},
            {"label": "Sent to REVIEW", "score": threshold["tiers"]["review"]["recall_global"]},
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.5, 1.1, 1.1]})
    count_ax, precision_ax, recall_ax = axes

    colors = {"MATCH": "#16a34a", "REVIEW": "#f59e0b", "NO_MATCH": "#ef4444"}
    sns.barplot(data=count_df, y="tier", x="count", hue="tier", palette=colors, legend=False, ax=count_ax)
    count_ax.set_xscale("log")
    count_ax.set_xlabel("pair count (log scale)")
    count_ax.set_ylabel("")
    count_ax.set_title("Final decision volume")
    for idx, row in count_df.iterrows():
        count_ax.text(row["count"] * 1.08, idx, f"{row['count']:,}\n({row['share'] * 100:.2f}%)", va="center", ha="left", fontsize=10)

    for ax, df, title in [(precision_ax, precision_df, "Auto-merge precision"), (recall_ax, recall_df, "Recall contribution")]:
        ax.axvspan(0.00, 0.60, color="#fecaca", alpha=0.45)
        ax.axvspan(0.60, 0.85, color="#fde68a", alpha=0.35)
        ax.axvspan(0.85, 1.00, color="#bbf7d0", alpha=0.40)
        sns.barplot(data=df, y="label", x="score", hue="label", legend=False, palette=["#0f766e", "#14b8a6", "#16a34a", "#f59e0b"], ax=ax)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("score")
        ax.set_ylabel("")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=10, pad=6)
        for i, row in df.iterrows():
            ax.text(min(row["score"] + 0.03, 1.03), i, f"{row['score']:.4f}", va="center", ha="left", fontsize=10)

    fig.suptitle("Threshold Outcomes that Drive Auto-Merge and Human Review", y=1.02, fontweight="bold")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "ch04_threshold_dashboard.png")


def make_production_decision_matrix(full_report: dict) -> None:
    total_pos = full_report["data_report"]["ground_truth_positive_pairs"]
    total_decisions = (
        full_report["production_threshold_report"]["final_match_only_count"]
        + full_report["production_threshold_report"]["tiers"]["review"]["count"]
        + 1979400
    )
    total_neg = total_decisions - total_pos

    exact = full_report["production_threshold_report"]["tiers"]["exact"]
    match = full_report["production_threshold_report"]["tiers"]["match"]
    review = full_report["production_threshold_report"]["tiers"]["review"]

    match_tp = exact["true_positive_count"] + match["true_positive_count"]
    match_fp = (exact["count"] - exact["true_positive_count"]) + (match["count"] - match["true_positive_count"])
    review_tp = review["true_positive_count"]
    review_fp = review["count"] - review["true_positive_count"]
    no_match_total = 1979400
    no_match_tp = total_pos - match_tp - review_tp
    no_match_fp = total_neg - match_fp - review_fp

    matrix = np.array(
        [
            [match_tp, review_tp, no_match_tp],
            [match_fp, review_fp, no_match_fp],
        ]
    )
    annot = np.array(
        [
            [f"MATCH\n{match_tp:,}", f"REVIEW\n{review_tp:,}", f"NO_MATCH\n{no_match_tp:,}"],
            [f"MATCH\n{match_fp:,}", f"REVIEW\n{review_fp:,}", f"NO_MATCH\n{no_match_fp:,}"],
        ]
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(["Decision = MATCH", "Decision = REVIEW", "Decision = NO_MATCH"], rotation=0)
    ax.set_yticklabels(["Actual MATCH", "Actual NO_MATCH"], rotation=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Production Decision Matrix at the Final Operating Point", pad=14, fontweight="bold")
    save(fig, "ch04_production_decision_matrix.png")


def make_error_analysis_summary() -> None:
    fp = pd.read_csv(ROOT / "data" / "fp_top.csv")
    fn = pd.read_csv(ROOT / "data" / "fn_top.csv")

    def platform_counts(df: pd.DataFrame) -> pd.Series:
        return (df["platform_a"].astype(str) + " -> " + df["platform_b"].astype(str)).value_counts().sort_values(ascending=True)

    fp_pairs = platform_counts(fp)
    fn_pairs = platform_counts(fn)
    all_pairs = sorted(set(fp_pairs.index) | set(fn_pairs.index))
    pair_df = pd.DataFrame(
        {
            "platform_pair": all_pairs,
            "FP": [int(fp_pairs.get(k, 0)) for k in all_pairs],
            "FN": [int(fn_pairs.get(k, 0)) for k in all_pairs],
        }
    )

    def err_stats(df: pd.DataFrame) -> dict:
        n = len(df)
        return {
            "Both name sims < 0.5": int(((df["username_jaro"] < 0.5) & (df["fullname_jaro"] < 0.5)).sum()) / n * 100,
            "Either name sim < 0.5": int(((df["username_jaro"] < 0.5) | (df["fullname_jaro"] < 0.5)).sum()) / n * 100,
            "Both name sims >= 0.9": int(((df["username_jaro"] >= 0.9) & (df["fullname_jaro"] >= 0.9)).sum()) / n * 100,
            "Any local image": int((df["image_any_local"] == 1).sum()) / n * 100,
            "Caption available": int((df["image_caption_any"] == 1).sum()) / n * 100,
        }

    stat_df = pd.DataFrame([err_stats(fp), err_stats(fn)], index=["FP", "FN"]).T.reset_index().rename(columns={"index": "metric"})

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))

    ax = axes[0]
    y = np.arange(len(pair_df))
    ax.barh(y - 0.18, pair_df["FP"], height=0.34, color="#ef4444", label="False Positives")
    ax.barh(y + 0.18, pair_df["FN"], height=0.34, color="#0ea5e9", label="False Negatives")
    ax.set_yticks(y)
    ax.set_yticklabels(pair_df["platform_pair"])
    ax.set_xlabel("count in top-200 sample")
    ax.set_ylabel("")
    ax.set_title("Platform-pair concentration of errors")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    ax = axes[1]
    y = np.arange(len(stat_df))
    ax.barh(y - 0.18, stat_df["FP"], height=0.34, color="#ef4444", label="False Positives")
    ax.barh(y + 0.18, stat_df["FN"], height=0.34, color="#0ea5e9", label="False Negatives")
    ax.set_yticks(y)
    ax.set_yticklabels(stat_df["metric"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of top-200 sample (%)")
    ax.set_ylabel("")
    ax.set_title("Observed error characteristics")
    for i, row in stat_df.iterrows():
        ax.text(row["FP"] + 1.0, i - 0.18, f"{row['FP']:.1f}%", va="center", ha="left", fontsize=9)
        ax.text(row["FN"] + 1.0, i + 0.18, f"{row['FN']:.1f}%", va="center", ha="left", fontsize=9)

    fig.suptitle("Error Analysis of False Positives and False Negatives from Top Error Samples", y=1.02, fontweight="bold")
    fig.tight_layout()
    save(fig, "ch04_error_analysis_fp_fn.png")


def make_feature_importance_all(features: pd.DataFrame) -> None:
    df = features.copy()
    df["group"] = df["feature"].map(feature_group)
    df["color"] = df["group"].map(GROUP_COLORS)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.5, 13.8))
    ax.barh(df["feature"], df["importance"], color=df["color"], edgecolor="white", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("feature importance")
    ax.set_ylabel("")
    ax.set_title("Importance of All 41 Features Used in the Final Candidate-Scoring Model", pad=16, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    for idx, row in df.head(10).iterrows():
        ax.text(row["importance"] + 0.006, idx, f"{row['importance']:.4f}", va="center", ha="left", fontsize=9)

    zero_count = int((df["importance"] == 0).sum())
    top10_share = df.head(10)["importance"].sum()
    ax.text(
        0.98,
        0.02,
        f"Top 10 features = {top10_share * 100:.2f}% of total importance\nZero-importance features in this run = {zero_count}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
    )

    handles = [plt.Line2D([0], [0], color=color, lw=8) for color in GROUP_COLORS.values()]
    ax.legend(handles, list(GROUP_COLORS.keys()), loc="lower right", frameon=True, fontsize=9, title="feature family")
    save(fig, "ch04_feature_importance_all41.png")


def make_feature_family_importance(features: pd.DataFrame) -> None:
    df = features.copy()
    df["group"] = df["feature"].map(feature_group)
    agg = (
        df.groupby("group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    agg["share_pct"] = agg["importance"] / agg["importance"].sum() * 100
    colors = agg["group"].map(GROUP_COLORS)

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    ax.barh(agg["group"], agg["importance"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("aggregated importance")
    ax.set_ylabel("")
    ax.set_title("Feature Importance Aggregated by Evidence Family", pad=14, fontweight="bold")
    for pos, row in agg.iterrows():
        ax.text(row["importance"] + 0.008, pos, f"{row['importance']:.4f} ({row['share_pct']:.2f}%)", va="center", ha="left", fontsize=10)
    save(fig, "ch04_feature_family_importance.png")


def make_metric_heatmap(suite_df: pd.DataFrame) -> None:
    metric_cols = ["test_avg_precision", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    display = suite_df.copy()
    display["configuration"] = display["run_name"].map(RUN_LABELS)
    actual = display.set_index("configuration")[metric_cols].rename(columns=METRIC_LABELS)
    relative = actual.copy()
    for col in relative.columns:
        min_v = relative[col].min()
        max_v = relative[col].max()
        if max_v == min_v:
            relative[col] = 1.0
        else:
            relative[col] = (relative[col] - min_v) / (max_v - min_v)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    annot = actual.copy()
    for col in annot.columns:
        annot[col] = annot[col].map(lambda v: f"{v:.4f}")
    sns.heatmap(
        relative,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "relative quality within each metric"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Metric Comparison of the Final Candidate-Scoring Configurations", pad=14, fontweight="bold")
    save(fig, "ch04_metric_heatmap.png")


def make_model_family_heatmap(model_df: pd.DataFrame) -> None:
    metric_cols = ["test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]
    actual = model_df.set_index("model")[metric_cols].rename(columns={
        "test_ap": "AP",
        "test_auc": "ROC-AUC",
        "test_f1": "F1",
        "test_precision": "Precision",
        "test_recall": "Recall",
    })
    actual.index = [str(idx).upper() if str(idx).lower() != "extra_trees" else "EXTRA TREES" for idx in actual.index]
    relative = actual.copy()
    for col in relative.columns:
        min_v = relative[col].min()
        max_v = relative[col].max()
        if max_v == min_v:
            relative[col] = 1.0
        else:
            relative[col] = (relative[col] - min_v) / (max_v - min_v)
    annot = actual.copy()
    for col in annot.columns:
        annot[col] = annot[col].map(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    sns.heatmap(
        relative,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        linewidths=1.4,
        linecolor="white",
        cbar_kws={"label": "relative quality within each metric"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Model-Family Comparison on the Final 41-Feature Split", pad=14, fontweight="bold")
    save(fig, "ch04_model_family_heatmap.png")


def make_main_confusion_matrix(experiment_report: dict) -> None:
    cm = np.array(experiment_report["metrics"]["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]
    annot = np.array(
        [
            [f"TN\n{tn:,}", f"FP\n{fp:,}"],
            [f"FN\n{fn:,}", f"TP\n{tp:,}"],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        cbar=False,
        square=True,
        linewidths=1.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(["Predicted NO_MATCH", "Predicted MATCH"], rotation=0)
    ax.set_yticklabels(["Actual NO_MATCH", "Actual MATCH"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    precision = experiment_report["metrics"]["test_precision"]
    recall = experiment_report["metrics"]["test_recall"]
    f1 = experiment_report["metrics"]["test_f1"]
    ax.set_title(
        f"Confusion Matrix of the Main Run at Threshold = {experiment_report['metrics']['threshold']:.2f}\n"
        f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}",
        pad=14,
        fontweight="bold",
    )
    save(fig, "ch04_main_confusion_matrix.png")


def make_suite_confusion_grid() -> None:
    suite_runs = [
        ("text_attr_hybrid_r075_h20_s42", "Text / Attribute\nHybrid"),
        ("image_stats_r075_h20_s42", "Image\nStatistics"),
        ("image_context_r075_h20_s42", "Image\nContext"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6))
    for ax, (run_name, label) in zip(axes, suite_runs):
        report = read_json(SUITE_RUNS_DIR / run_name / "reports" / "experiment_report.json")
        cm = np.array(report["metrics"]["confusion_matrix"])
        tn, fp = cm[0]
        fn, tp = cm[1]
        annot = np.array(
            [
                [f"TN\n{tn:,}", f"FP\n{fp:,}"],
                [f"FN\n{fn:,}", f"TP\n{tp:,}"],
            ]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            cbar=False,
            square=True,
            linewidths=1.4,
            linecolor="white",
            ax=ax,
        )
        ax.set_xticklabels(["Pred NO", "Pred MATCH"], rotation=0)
        ax.set_yticklabels(["Actual NO", "Actual MATCH"], rotation=0)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"{label}\n"
            f"thr = {report['metrics']['threshold']:.2f}, "
            f"F1 = {report['metrics']['test_f1']:.4f}\n"
            f"P = {report['metrics']['test_precision']:.4f}, R = {report['metrics']['test_recall']:.4f}",
            fontsize=11.5,
            pad=10,
            fontweight="bold",
        )
    fig.suptitle(
        "Confusion Matrices of the Candidate-Scoring Configurations Used to Select the Main Run",
        y=1.03,
        fontweight="bold",
    )
    save(fig, "ch04_suite_confusion_grid.png")


def make_calibration_curve() -> None:
    val_df = pd.read_parquet(RUN_REPORTS_DIR / "val_scores.parquet")
    test_df = pd.read_parquet(RUN_REPORTS_DIR / "test_scores.parquet")

    def bin_curve(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        cut = pd.cut(df["probability"], bins=[i / bins for i in range(bins + 1)], include_lowest=True)
        rows = []
        for interval, part in df.groupby(cut, observed=False):
            if len(part) == 0:
                continue
            rows.append(
                {
                    "bin": str(interval),
                    "count": len(part),
                    "avg_prob": float(part["probability"].mean()),
                    "pos_rate": float(part["label"].mean()),
                }
            )
        return pd.DataFrame(rows)

    val_curve = bin_curve(val_df)
    test_curve = bin_curve(test_df)

    def ece(df: pd.DataFrame) -> float:
        total = float(len(df))
        score = 0.0
        curve = bin_curve(df)
        for row in curve.itertuples(index=False):
            score += abs(row.pos_rate - row.avg_prob) * (row.count / total)
        return score

    val_ece = ece(val_df)
    test_ece = ece(test_df)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
    for ax, curve, title, ece_score, color in [
        (axes[0], val_curve, "Validation set", val_ece, "#0f766e"),
        (axes[1], test_curve, "Test set", test_ece, "#0ea5e9"),
    ]:
        ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color="#64748b", label="Ideal calibration")
        ax.scatter(curve["avg_prob"], curve["pos_rate"], s=curve["count"] / 8, color=color, alpha=0.85, edgecolor="white", linewidth=1.0)
        ax.axvline(0.95, color="#f59e0b", ls=":", lw=1.4)
        ax.axvline(0.98, color="#ef4444", ls=":", lw=1.4)
        ax.set_title(f"{title}\nECE = {ece_score:.4f}")
        ax.set_xlabel("calibrated probability")
        ax.set_ylabel("observed positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Reliability of Isotonic-Calibrated Candidate Scores", y=1.02, fontweight="bold")
    fig.text(0.5, 0.01, "Marker size is proportional to the number of pairs in each probability bin. Vertical lines mark the production thresholds 0.95 and 0.98.", ha="center", fontsize=10.5, color="#334155")
    save(fig, "ch04_calibration_curve.png")


def make_retrieval_funnel(full_report: dict) -> None:
    data = [
        ("All cross-platform", full_report["data_report"]["all_cross_platform_pairs"]),
        ("Exact-first", full_report["blocking_report"]["exact_match_pairs"]),
        ("Model candidates", full_report["blocking_report"]["candidate_pairs_for_model"]),
        ("Final MATCH only", full_report["production_threshold_report"]["final_match_only_count"]),
        ("REVIEW queue", full_report["production_threshold_report"]["tiers"]["review"]["count"]),
    ]
    df = pd.DataFrame(data, columns=["stage", "count"])
    colors = ["#1d4ed8", "#0f766e", "#0ea5e9", "#16a34a", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    sns.barplot(data=df, y="stage", x="count", hue="stage", palette=colors, legend=False, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("pair count (log scale)")
    ax.set_ylabel("")
    ax.set_title("Retrieval Funnel from All Cross-Platform Pairs to Final Decision Tiers", pad=14, fontweight="bold")
    for i, row in df.iterrows():
        ax.text(row["count"] * 1.06, i, f"{row['count']:,}", va="center", ha="left", fontsize=10)
    save(fig, "ch04_retrieval_funnel.png")


def make_threshold_tradeoff() -> None:
    test_df = pd.read_parquet(RUN_REPORTS_DIR / "test_scores.parquet")
    rows = []
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (test_df["probability"] >= thr).astype(int)
        tp = int(((pred == 1) & (test_df["label"] == 1)).sum())
        fp = int(((pred == 1) & (test_df["label"] == 0)).sum())
        fn = int(((pred == 0) & (test_df["label"] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"threshold": thr, "precision": precision, "recall": recall, "f1": f1})
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(curve["threshold"], curve["precision"], color="#0f766e", lw=2.5, label="Precision")
    ax.plot(curve["threshold"], curve["recall"], color="#0ea5e9", lw=2.5, label="Recall")
    ax.plot(curve["threshold"], curve["f1"], color="#f59e0b", lw=2.5, label="F1")
    ax.axvline(0.35, color="#64748b", ls="--", lw=1.4, label="test threshold 0.35")
    ax.axvline(0.95, color="#f59e0b", ls=":", lw=1.4, label="production review 0.95")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title("Threshold Trade-off of the Main Run on the Test Split", pad=14, fontweight="bold")
    ax.legend(loc="lower left", ncol=4, frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    save(fig, "ch04_threshold_tradeoff.png")


def make_score_distribution() -> None:
    test_df = pd.read_parquet(RUN_REPORTS_DIR / "test_scores.parquet")
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    bins = np.linspace(0, 1, 25)
    ax.hist(test_df.loc[test_df["label"] == 0, "probability"], bins=bins, alpha=0.65, color="#ef4444", label="Actual NO_MATCH", density=True)
    ax.hist(test_df.loc[test_df["label"] == 1, "probability"], bins=bins, alpha=0.65, color="#0ea5e9", label="Actual MATCH", density=True)
    ax.axvline(0.35, color="#475569", ls="--", lw=1.6, label="test threshold 0.35")
    ax.axvline(0.95, color="#f59e0b", ls=":", lw=1.6, label="review threshold 0.95")
    ax.axvline(0.98, color="#16a34a", ls=":", lw=1.6, label="match threshold 0.98")
    ax.set_xlabel("calibrated probability")
    ax.set_ylabel("density")
    ax.set_title("Distribution of Calibrated Scores on the Test Split", pad=14, fontweight="bold")
    ax.legend(loc="upper center", ncol=5, fontsize=9, frameon=True)
    save(fig, "ch04_score_distribution.png")


def make_image_coverage(experiment_report: dict) -> None:
    prof = experiment_report["modality_report"]["profile_coverage"]
    pair = experiment_report["modality_report"]["pair_coverage"]["test"]
    profile_df = pd.DataFrame(
        [
            ("Valid profiles", prof["valid_profiles"]),
            ("Profiles with local image", prof["profiles_with_local_image"]),
            ("Profiles with image metadata", prof["profiles_with_image_metadata"]),
            ("Profiles with caption", prof["profiles_with_caption"]),
        ],
        columns=["metric", "count"],
    )
    pair_df = pd.DataFrame(
        [
            ("Test pairs", pair["rows"]),
            ("Any local image", pair["pairs_with_any_local_image"]),
            ("Both local images", pair["pairs_with_both_local_images"]),
            ("Caption signal", pair["pairs_with_caption_signal"]),
        ],
        columns=["metric", "count"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for ax, df, title, color in [
        (axes[0], profile_df, "Profile-level image coverage", "#0ea5e9"),
        (axes[1], pair_df, "Test-pair image coverage", "#f59e0b"),
    ]:
        sns.barplot(data=df, y="metric", x="count", hue="metric", legend=False, palette=[color] * len(df), ax=ax)
        ax.set_xlabel("count")
        ax.set_ylabel("")
        ax.set_title(title)
        for i, row in df.iterrows():
            ax.text(row["count"] + max(df["count"]) * 0.02, i, f"{row['count']:,}", va="center", ha="left", fontsize=10)
    fig.suptitle("Image Coverage that Constrains the Current Main Run", y=1.02, fontweight="bold")
    save(fig, "ch04_image_coverage.png")


def make_ranking_quality(full_report: dict) -> None:
    model = full_report["model_report"]
    df = pd.DataFrame(
        [
            ("P@100", model["precision_at_100"]),
            ("P@500", model["precision_at_500"]),
            ("P@1000", model["precision_at_1000"]),
            ("P@5000", model["precision_at_5000"]),
        ],
        columns=["metric", "score"],
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.barplot(data=df, x="metric", y="score", hue="metric", legend=False, palette=["#0f766e", "#14b8a6", "#0ea5e9", "#6366f1"], ax=ax)
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("precision")
    ax.set_title("Ranking Quality of Candidate Scoring on Full Candidate Pairs", pad=14, fontweight="bold")
    for i, row in df.iterrows():
        ax.text(i, row["score"] + 0.002, f"{row['score']:.4f}", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.98,
        0.04,
        f"Candidate AP = {model['candidate_avg_precision']:.4f}\nCandidate ROC-AUC = {model['candidate_roc_auc']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cbd5e1'),
    )
    save(fig, "ch04_ranking_quality.png")


def main() -> None:
    setup_theme()
    full_report = read_json(FULL_REPORT)
    crm_report = read_json(CRM_REPORT)
    experiment_report = read_json(
        PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42" / "reports" / "experiment_report.json"
    )
    features = pd.read_csv(FEATURE_IMPORTANCE)
    suite_df = pd.read_csv(SUITE_COMPARE)
    model_family_df = pd.read_csv(MODEL_FAMILY_COMPARE)

    make_model_pipeline(experiment_report)
    make_threshold_flow(full_report, crm_report)
    make_threshold_dashboard(full_report, crm_report)
    make_production_decision_matrix(full_report)
    make_error_analysis_summary()
    make_retrieval_funnel(full_report)
    make_threshold_tradeoff()
    make_score_distribution()
    make_image_coverage(experiment_report)
    make_ranking_quality(full_report)
    make_main_confusion_matrix(experiment_report)
    make_suite_confusion_grid()
    make_calibration_curve()
    make_feature_importance_all(features)
    make_feature_family_importance(features)
    make_metric_heatmap(suite_df)
    make_model_family_heatmap(model_family_df)


if __name__ == "__main__":
    main()
