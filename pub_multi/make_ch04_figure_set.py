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


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FIG_DIR = ROOT / "fig"

R1_REPORT = PROJECT_ROOT / "train_data" / "stage10_13_training" / "reports" / "evaluation_summary.json"
R2_REPORT = PROJECT_ROOT / "train_data" / "stage10_13_training_noleak" / "reports" / "evaluation_summary.json"
R3_REPORT = PROJECT_ROOT / "train_data" / "stage10_13_training_noleak_strict" / "reports" / "evaluation_summary.json"
R4_REPORT = PROJECT_ROOT / "train_data" / "stage7_8_rebuilt_experiment" / "reports" / "experiment_report.json"
R6_REPORT = PROJECT_ROOT / "train_data" / "stage7_8_rebuilt_experiment_hybrid" / "reports" / "experiment_report.json"
R7_REPORT = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42" / "reports" / "experiment_report.json"
TUNE_CSV = PROJECT_ROOT / "train_data" / "stage7_8_rebuilt_experiment_tuned" / "tuning_summary.csv"
CRM_REPORT = PROJECT_ROOT / "train_data" / "stage15_crm_entity_pipeline" / "reports" / "crm_entity_report.json"
FULL_PIPELINE_REPORT = PROJECT_ROOT / "train_data" / "stage7_14_full_candidate_pipeline" / "reports" / "full_pipeline_report.json"
SUITE_LEADERBOARD = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "reports" / "leaderboard.csv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (11, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["font.size"] = 10


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_experiment_roadmap(r1: dict, r2: dict, r3: dict, r4: dict, tune: pd.DataFrame, r6: dict, r7: dict, crm: dict) -> None:
    rows = [
        ("R1", "Original", f"F1 {r1['test_f1']:.3f}", "leak"),
        ("R2", "No-Leak Init", f"F1 {r2['test_f1']:.3f}", "still leak"),
        ("R3", "Strict No-Leak", f"F1 {r3['test_f1']:.3f}", "diagnostic"),
        ("R4", "Rebuilt Safe", f"F1 {r4['metrics']['test_f1']:.3f}", "trustable"),
        ("R5", "Tuned Negatives", f"F1 {tune.iloc[0]['test_f1']:.3f}", "better AP"),
        ("R6", "+ SBERT", f"F1 {r6['metrics']['test_f1']:.3f}", "better recall"),
        ("R7", "Image Context", f"F1 {r7['metrics']['test_f1']:.3f}", "main run"),
        ("P", "Production", f"P {0.9550:.3f}", f"review {crm['counts']['review_queue']:,}"),
    ]
    colors = ["#ef4444", "#f97316", "#f59e0b", "#0ea5e9", "#14b8a6", "#22c55e", "#0f766e", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.07, 0.93, len(rows))
    y = 0.5
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.055, y), xytext=(xs[i] + 0.055, y), arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
    for i, (code, title, metric, note) in enumerate(rows):
        box = FancyBboxPatch(
            (xs[i] - 0.055, y - 0.16),
            0.11,
            0.32,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#0f172a",
            facecolor=colors[i],
            alpha=0.95,
        )
        ax.add_patch(box)
        ax.text(xs[i], y + 0.10, code, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(xs[i], y + 0.03, title, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(xs[i], y - 0.04, metric, ha="center", va="center", color="white", fontsize=9)
        ax.text(xs[i], y - 0.10, note, ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Chapter 4 Experiment Roadmap from Leakage Diagnosis to Production")
    save(fig, "ch04_experiment_roadmap.png")


def make_data_prep_summary(full_pipeline: dict, r4: dict, r6: dict, suite: pd.DataFrame) -> None:
    platform_df = (
        pd.Series(full_pipeline["data_report"]["platform_counts"])
        .rename_axis("platform")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    split_rows = pd.Series(r4["leakage_report"]["rows_by_split"]).rename_axis("split").reset_index(name="rows")
    feature_rows = pd.DataFrame(
        [
            {"line": "Classical", "feature_count": r4["metrics"]["feature_count"]},
            {"line": "Hybrid + SBERT", "feature_count": r6["metrics"]["feature_count"]},
            {
                "line": "Image Stats",
                "feature_count": int(suite.loc[suite["run_name"] == "image_stats_r075_h20_s42", "feature_count"].iloc[0]),
            },
            {
                "line": "Image Context",
                "feature_count": int(suite.loc[suite["run_name"] == "image_context_r075_h20_s42", "feature_count"].iloc[0]),
            },
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))

    sns.barplot(data=platform_df, x="platform", y="count", hue="platform", legend=False, palette=["#0f766e", "#0ea5e9", "#f59e0b"], ax=axes[0])
    for i, row in platform_df.reset_index(drop=True).iterrows():
        axes[0].text(i, row["count"] + platform_df["count"].max() * 0.02, f"{int(row['count']):,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Valid Profiles After Preparation")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("profile count")
    axes[0].text(
        0.5,
        0.96,
        f"Total valid = {full_pipeline['data_report']['profiles_total']:,}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#334155",
    )

    sns.barplot(data=split_rows, x="split", y="rows", hue="split", legend=False, palette=["#0f766e", "#38bdf8", "#f59e0b"], ax=axes[1])
    for i, row in split_rows.reset_index(drop=True).iterrows():
        axes[1].text(i, row["rows"] + split_rows["rows"].max() * 0.02, f"{int(row['rows']):,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Leak-Safe Supervised Split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("pair rows")
    axes[1].text(0.5, 0.96, "component overlap = 0", transform=axes[1].transAxes, ha="center", va="top", fontsize=9, color="#334155")

    sns.barplot(data=feature_rows, x="line", y="feature_count", hue="line", legend=False, palette=["#0f766e", "#14b8a6", "#0ea5e9", "#7c3aed"], ax=axes[2])
    for i, row in feature_rows.reset_index(drop=True).iterrows():
        axes[2].text(i, row["feature_count"] + feature_rows["feature_count"].max() * 0.03, f"{int(row['feature_count'])}", ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Feature-Space Growth Across Lines")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("selected features")
    axes[2].tick_params(axis="x", rotation=18)

    fig.suptitle("Data Preparation Outcomes Used in Chapter 4", y=1.03)
    save(fig, "ch04_data_prep_summary.png")


def make_split_design(r4: dict, r7: dict) -> None:
    def split_df(report: dict) -> pd.DataFrame:
        rows = []
        for split, stats in report["leakage_report"]["split_stats"].items():
            rows.append(
                {
                    "split": split,
                    "positive": stats["positive_pairs"],
                    "random_negative": stats["random_negatives"],
                    "hard_negative": stats["hard_negatives"],
                    "profiles": stats["profiles"],
                }
            )
        return pd.DataFrame(rows)

    left = split_df(r4)
    right = split_df(r7)
    colors = {"positive": "#0f766e", "random_negative": "#38bdf8", "hard_negative": "#f59e0b"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    for ax, df, title in zip(
        axes,
        [left, right],
        ["R4 Rebuilt Leak-Safe", "R7-C Image Context"],
    ):
        bottom = np.zeros(len(df))
        for col in ["positive", "random_negative", "hard_negative"]:
            ax.bar(df["split"], df[col], bottom=bottom, color=colors[col], label=col.replace("_", " "))
            bottom += df[col].to_numpy()
        for idx, total in enumerate(bottom):
            ax.text(idx, total + max(bottom) * 0.02, f"{int(total):,}", ha="center", va="bottom", fontsize=9)
            ax.text(idx, max(bottom) * 0.04, f"profiles\n{int(df.iloc[idx]['profiles']):,}", ha="center", va="bottom", fontsize=8, color="#334155")
        ax.set_title(title)
        ax.set_xlabel("split")
    axes[0].set_ylabel("pair count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Split Composition by Positive, Random Negative, and Hard Negative Pairs", y=1.03)
    save(fig, "ch04_split_design.png")


def make_leakage_diagnosis(r1: dict, r2: dict, r3: dict) -> None:
    df = pd.DataFrame(
        [
            {"run": "R1 Original", "AP": r1["test_avg_precision"], "AUC": r1["test_roc_auc"], "F1": r1["test_f1"]},
            {"run": "R2 No-Leak Init", "AP": r2["test_avg_precision"], "AUC": r2["test_roc_auc"], "F1": r2["test_f1"]},
            {"run": "R3 Strict", "AP": r3["test_avg_precision"], "AUC": r3["test_roc_auc"], "F1": r3["test_f1"]},
        ]
    )
    plot_df = df.melt(id_vars="run", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(data=plot_df, x="run", y="value", hue="metric", palette=["#0ea5e9", "#22c55e", "#f59e0b"], ax=ax)
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, ls="--", lw=1, color="#64748b")
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.02,
            f"{patch.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Leakage Diagnosis: Perfect Metrics Collapse After Strict Control")
    ax.set_xlabel("")
    ax.set_ylabel("score")
    save(fig, "ch04_leakage_diagnosis.png")


def make_cm_rebuilt(r4: dict) -> None:
    cm = np.array(r4["metrics"]["confusion_matrix"])
    annot = np.array([[f"{v:,}" for v in row] for row in cm])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, square=True, linewidths=1.5, linecolor="white", ax=ax)
    ax.set_xticklabels(["Pred NO_MATCH", "Pred MATCH"], rotation=0)
    ax.set_yticklabels(["Actual NO_MATCH", "Actual MATCH"], rotation=0)
    ax.set_title("Confusion Matrix - Rebuilt Leak-Safe Baseline")
    save(fig, "ch04_cm_rebuilt.png")


def make_sbert_gain(tune: pd.DataFrame, r6: dict) -> None:
    base = tune.loc[tune["run_name"] == "r075_h200_s42"].iloc[0]
    df = pd.DataFrame(
        [
            {"metric": "AP", "baseline": base["test_ap"], "hybrid": r6["metrics"]["test_avg_precision"]},
            {"metric": "AUC", "baseline": base["test_auc"], "hybrid": r6["metrics"]["test_roc_auc"]},
            {"metric": "F1", "baseline": base["test_f1"], "hybrid": r6["metrics"]["test_f1"]},
            {"metric": "Precision", "baseline": base["test_precision"], "hybrid": r6["metrics"]["test_precision"]},
            {"metric": "Recall", "baseline": base["test_recall"], "hybrid": r6["metrics"]["test_recall"]},
        ]
    )
    df["delta"] = df["hybrid"] - df["baseline"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.8),
        gridspec_kw={"width_ratios": [2.6, 1.4]},
        sharey=True,
    )
    ax_score, ax_delta = axes
    y = np.arange(len(df))

    ax_score.hlines(y=y, xmin=df["baseline"], xmax=df["hybrid"], color="#cbd5e1", lw=2.5, zorder=1)
    ax_score.scatter(df["baseline"], y, color="#0ea5e9", s=80, zorder=3)
    ax_score.scatter(df["hybrid"], y, color="#0f766e", s=80, zorder=3)
    for i, row in df.iterrows():
        ax_score.text(row["baseline"] - 0.0014, i - 0.13, f"{row['baseline']:.4f}", ha="right", va="center", fontsize=8.8, color="#0369a1")
        ax_score.text(row["hybrid"] + 0.0014, i + 0.13, f"{row['hybrid']:.4f}", ha="left", va="center", fontsize=8.8, color="#115e59")
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(df["metric"])
    ax_score.set_xlim(0.898, 0.982)
    ax_score.set_xlabel("score")
    ax_score.set_title("Absolute scores", fontsize=13, pad=10)
    ax_score.text(
        0.5,
        1.01,
        "blue = baseline, green = + SBERT",
        transform=ax_score.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    ax_score.grid(axis="x", alpha=0.25)

    delta_colors = ["#0f766e" if value >= 0 else "#dc2626" for value in df["delta"]]
    ax_delta.axvline(0, color="#64748b", lw=1.2, ls="--")
    ax_delta.barh(y, df["delta"], color=delta_colors, height=0.46)
    for i, row in df.iterrows():
        x = row["delta"]
        offset = 0.00045 if x >= 0 else -0.00045
        ax_delta.text(x + offset, i, f"{x:+.4f}", ha="left" if x >= 0 else "right", va="center", fontsize=9, color="#334155")
    ax_delta.set_xlim(-0.0125, 0.0135)
    ax_delta.set_xlabel("delta from baseline")
    ax_delta.set_title("Incremental change")
    ax_delta.grid(axis="x", alpha=0.25)

    fig.suptitle("Effect of Adding SBERT to the Tuned Rebuilt Baseline", fontsize=15)
    save(fig, "ch04_sbert_gain.png")


def make_crm_outcomes(crm: dict) -> None:
    decision = pd.Series(crm["decision_breakdown"]).rename_axis("decision").reset_index(name="count")
    decision["share_pct"] = decision["count"] / decision["count"].sum() * 100
    lead = pd.Series(crm["lead_tier_breakdown"]).rename_axis("tier").reset_index(name="count")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    sns.barplot(
        data=decision,
        x="decision",
        y="share_pct",
        hue="decision",
        palette=["#ef4444", "#f59e0b", "#0f766e"],
        legend=False,
        ax=axes[0],
    )
    for i, row in decision.iterrows():
        axes[0].text(i, row["share_pct"] + 1, f"{row['count']:,}\n{row['share_pct']:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Decision Breakdown After Production Thresholding")
    axes[0].set_ylabel("share of all decisions (%)")
    axes[0].set_xlabel("")

    sns.barplot(
        data=lead,
        x="tier",
        y="count",
        hue="tier",
        palette=["#ef4444", "#f59e0b", "#0ea5e9"],
        legend=False,
        ax=axes[1],
    )
    for i, row in lead.iterrows():
        axes[1].text(i, row["count"] + lead["count"].max() * 0.02, f"{row['count']:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Lead Tier Output for Unified Profiles")
    axes[1].set_ylabel("profile count")
    axes[1].set_xlabel("")

    fig.suptitle("CRM Output Summary from the Final Production Pipeline", y=1.02)
    save(fig, "ch04_crm_outcomes.png")


def main() -> None:
    setup_theme()
    r1 = read_json(R1_REPORT)
    r2 = read_json(R2_REPORT)
    r3 = read_json(R3_REPORT)
    r4 = read_json(R4_REPORT)
    r6 = read_json(R6_REPORT)
    r7 = read_json(R7_REPORT)
    tune = pd.read_csv(TUNE_CSV)
    crm = read_json(CRM_REPORT)
    full_pipeline = read_json(FULL_PIPELINE_REPORT)
    suite = pd.read_csv(SUITE_LEADERBOARD)

    make_experiment_roadmap(r1, r2, r3, r4, tune, r6, r7, crm)
    make_data_prep_summary(full_pipeline, r4, r6, suite)
    make_split_design(r4, r7)
    make_leakage_diagnosis(r1, r2, r3)
    make_cm_rebuilt(r4)
    make_sbert_gain(tune, r6)
    make_crm_outcomes(crm)
    print("Created chapter 4 figure set.")


if __name__ == "__main__":
    main()
