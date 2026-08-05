from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "fig"
RES_DIR = ROOT / "res"


def setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["font.size"] = 11


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def metric_label(col: str) -> str:
    return {
        "test_ap": "AP",
        "test_auc": "AUC",
        "test_f1": "F1",
        "test_precision": "Precision",
        "test_recall": "Recall",
    }[col]


def model_label(name: str) -> str:
    mapping = {
        "gb": "Gradient Boosting",
        "rf": "Random Forest",
        "logreg": "Logistic Regression",
        "mlp": "IdentityMLP",
        "extra_trees": "Extra Trees",
        "adaboost": "AdaBoost",
        "linear_svm": "Linear SVM",
        "text_attr_hybrid_r075_h20_s42": "Text + Attr Hybrid",
        "image_stats_r075_h20_s42": "Image Statistics",
        "image_context_r075_h20_s42": "Image Context",
    }
    return mapping.get(name, name)


def feature_label(name: str) -> str:
    mapping = {
        "fullname_token_sort": "Full name token-sort similarity",
        "username_token_sort": "Username token-sort similarity",
        "fullname_jaro": "Full name Jaro similarity",
        "bio_tfidf_cosine": "Bio TF-IDF cosine similarity",
        "username_jaro": "Username Jaro similarity",
        "fullname_lev": "Full name Levenshtein similarity",
        "bio_sbert_cosine": "Bio SBERT cosine similarity",
        "username_lev": "Username Levenshtein similarity",
        "platform_pair_code": "Platform-pair code",
        "style_biolen_ratio": "Bio length ratio",
        "style_punct_diff": "Punctuation difference",
        "style_avgword_diff": "Average word-length difference",
        "image_caption_bio_sbert_cross": "Caption-Bio SBERT cross-signal",
        "image_caption_fullname_token_cross": "Caption-Full-name token cross-signal",
        "image_caption_username_token_cross": "Caption-Username token cross-signal",
    }
    return mapping.get(name, name.replace("_", " "))


def annotated_heatmap(df: pd.DataFrame, title: str, file_name: str, vmin: float, vmax: float) -> None:
    plot_df = df.copy()
    annot = plot_df.apply(lambda col: col.map(lambda v: f"{v:.4f}"))
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.7 * len(plot_df) + 2.4)))
    sns.heatmap(
        plot_df,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        linewidths=1,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    save(fig, file_name)


def draw_cm(ax, cm: np.ndarray, title: str) -> None:
    annot = np.array([[f"{int(v):,}" for v in row] for row in cm])
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 12},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticklabels(["NO_MATCH", "MATCH"], rotation=0, fontsize=9.5)
    ax.set_yticklabels(["NO_MATCH", "MATCH"], rotation=0, fontsize=9.5)
    ax.set_xlabel("Predicted class", fontsize=10, labelpad=6)
    ax.set_ylabel("Actual class", fontsize=10, labelpad=6)
    ax.tick_params(length=0, pad=4)


def save_single_cm(cm: np.ndarray, title: str, file_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw_cm(ax, cm, title)
    save(fig, file_name)


def make_classical_figures() -> None:
    df = pd.read_csv(RES_DIR / "classical_leaksafe_cmp.csv")
    order = ["gb", "rf", "logreg"]
    metric_cols = ["test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]
    heat_df = (
        df.set_index("model")
        .loc[order, metric_cols]
        .rename(index=model_label, columns=metric_label)
    )
    annotated_heatmap(
        heat_df,
        "Classical Leakage-Safe Comparison on the Same 22-Feature Split",
        "classical_leaksafe_cmp.png",
        vmin=0.86,
        vmax=0.98,
    )

    ordered = df.set_index("model").loc[order].reset_index()
    classical_cms = {
        "Gradient Boosting": np.array([[ordered.iloc[0].test_tn, ordered.iloc[0].test_fp], [ordered.iloc[0].test_fn, ordered.iloc[0].test_tp]]),
        "Random Forest": np.array([[ordered.iloc[1].test_tn, ordered.iloc[1].test_fp], [ordered.iloc[1].test_fn, ordered.iloc[1].test_tp]]),
        "Logistic Regression": np.array([[ordered.iloc[2].test_tn, ordered.iloc[2].test_fp], [ordered.iloc[2].test_fn, ordered.iloc[2].test_tp]]),
    }
    save_single_cm(classical_cms["Gradient Boosting"], "Gradient Boosting", "classical_gb_cm.png")
    save_single_cm(classical_cms["Random Forest"], "Random Forest", "classical_rf_cm.png")
    save_single_cm(classical_cms["Logistic Regression"], "Logistic Regression", "classical_logreg_cm.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.8))
    draw_cm(axes[0, 0], classical_cms["Gradient Boosting"], "Gradient Boosting")
    draw_cm(axes[0, 1], classical_cms["Random Forest"], "Random Forest")
    draw_cm(axes[1, 0], classical_cms["Logistic Regression"], "Logistic Regression")
    axes[1, 1].axis("off")
    note = (
        "Key reading points\n\n"
        "1. Gradient Boosting gives the best AP and F1.\n"
        "2. Random Forest recalls slightly more true matches,\n"
        "   but at the cost of more false positives.\n"
        "3. Logistic Regression is easier to interpret,\n"
        "   yet it underfits the non-linear patterns in the pair features."
    )
    axes[1, 1].text(0.0, 0.95, note, ha="left", va="top", fontsize=12)
    fig.suptitle("Confusion Matrices for the Classical Leakage-Safe Line", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "classical_leaksafe_cm_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_suite_figure() -> None:
    df = pd.read_csv(RES_DIR / "suite_cmp.csv")
    order = [
        "text_attr_hybrid_r075_h20_s42",
        "image_stats_r075_h20_s42",
        "image_context_r075_h20_s42",
    ]
    metric_cols = ["test_avg_precision", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    rename_cols = {
        "test_avg_precision": "AP",
        "test_roc_auc": "AUC",
        "test_f1": "F1",
        "test_precision": "Precision",
        "test_recall": "Recall",
    }
    heat_df = df.set_index("run_name").loc[order, metric_cols].rename(index=model_label, columns=rename_cols)
    annotated_heatmap(
        heat_df,
        "Multimodal Suite Comparison",
        "suite_cmp.png",
        vmin=0.91,
        vmax=0.98,
    )


def make_model_family_figure() -> None:
    df = pd.read_csv(RES_DIR / "model_family_cmp.csv").sort_values("rank")
    metric_cols = ["test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]
    heat_df = df.set_index("model")[metric_cols].rename(index=model_label, columns=metric_label)
    annotated_heatmap(
        heat_df,
        "Model Family Benchmark on the Final 41-Feature Split",
        "model_family_cmp.png",
        vmin=0.82,
        vmax=0.98,
    )


def make_model_strict_figures() -> None:
    df = pd.read_csv(RES_DIR / "model_strict_cmp.csv")
    order = ["rf", "extra_trees", "gb", "logreg", "mlp"]
    metric_cols = ["test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]
    heat_df = df.set_index("model").loc[order, metric_cols].rename(index=model_label, columns=metric_label)
    annotated_heatmap(
        heat_df,
        "Strict No-Leak Reference Comparison",
        "model_strict_cmp.png",
        vmin=0.73,
        vmax=1.0,
    )

    ordered = df.set_index("model").loc[order].reset_index()
    strict_cms = {
        "Random Forest": np.array([[ordered.iloc[0]["test_tn"], ordered.iloc[0]["test_fp"]], [ordered.iloc[0]["test_fn"], ordered.iloc[0]["test_tp"]]]),
        "Extra Trees": np.array([[ordered.iloc[1]["test_tn"], ordered.iloc[1]["test_fp"]], [ordered.iloc[1]["test_fn"], ordered.iloc[1]["test_tp"]]]),
        "Gradient Boosting": np.array([[ordered.iloc[2]["test_tn"], ordered.iloc[2]["test_fp"]], [ordered.iloc[2]["test_fn"], ordered.iloc[2]["test_tp"]]]),
        "Logistic Regression": np.array([[ordered.iloc[3]["test_tn"], ordered.iloc[3]["test_fp"]], [ordered.iloc[3]["test_fn"], ordered.iloc[3]["test_tp"]]]),
        "IdentityMLP": np.array([[ordered.iloc[4]["test_tn"], ordered.iloc[4]["test_fp"]], [ordered.iloc[4]["test_fn"], ordered.iloc[4]["test_tp"]]]),
    }
    save_single_cm(strict_cms["IdentityMLP"], "IdentityMLP (strict reference)", "strict_identitymlp_cm.png")

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 12.2))
    axes = axes.flatten()
    titles = ["Random Forest", "Extra Trees", "Gradient Boosting", "Logistic Regression", "IdentityMLP"]
    for ax, title in zip(axes, titles):
        draw_cm(ax, strict_cms[title], title)
    axes[-1].axis("off")
    axes[-1].text(
        0.0,
        0.95,
        "Strict-setting takeaway\n\n"
        "IdentityMLP keeps very high AUC,\n"
        "but its AP and F1 remain lower than\n"
        "the tree-based reruns under the same\n"
        "22-feature strict reference setting.",
        ha="left",
        va="top",
        fontsize=11.5,
    )
    fig.suptitle("Strict No-Leak Confusion-Matrix Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "model_strict_cm_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_feature_importance_figure() -> None:
    df = pd.read_csv(RES_DIR / "feat_imp.csv").head(12).iloc[::-1].copy()
    df["feature_clean"] = df["feature"].map(feature_label)
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    sns.barplot(data=df, x="importance", y="feature_clean", hue="feature_clean", legend=False, palette="crest", ax=ax)
    for patch, value in zip(ax.patches, df["importance"]):
        ax.text(value + 0.004, patch.get_y() + patch.get_height() / 2, f"{value:.3f}", va="center", fontsize=10)
    ax.set_title("Top Feature Importance of the Main Model")
    ax.set_xlabel("importance")
    ax.set_ylabel("")
    ax.set_xlim(0, max(df["importance"]) * 1.20)
    save(fig, "feat_imp.png")


def make_modality_figure() -> None:
    df = pd.read_csv(RES_DIR / "modality.csv").copy()
    total = float(df.loc[df["metric"] == "valid_profiles", "value"].iloc[0])
    label_map = {
        "valid_profiles": "Valid profiles",
        "profiles_with_local_image": "Profiles with local image",
        "profiles_with_image_metadata": "Profiles with image metadata",
        "profiles_with_caption": "Profiles with caption",
    }
    df["label"] = df["metric"].map(label_map)
    df["share"] = df["value"] / total * 100.0
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sns.barplot(data=df, x="value", y="label", hue="label", legend=False, palette="viridis", ax=ax)
    for patch, value, share in zip(ax.patches, df["value"], df["share"]):
        ax.text(value + total * 0.01, patch.get_y() + patch.get_height() / 2, f"{int(value):,} ({share:.1f}%)", va="center", fontsize=10)
    ax.set_title("Image-Modality Coverage in the Main Run")
    ax.set_xlabel("profiles")
    ax.set_ylabel("")
    ax.set_xlim(0, total * 1.22)
    save(fig, "modality.png")


def make_main_cm_figure() -> None:
    report = ROOT.parent / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42" / "reports" / "experiment_report.json"
    with report.open(encoding="utf-8") as fh:
        metrics = json.load(fh)["metrics"]
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    draw_cm(ax, cm, "Main multimodal run")
    ax.set_title("Main multimodal run (threshold = 0.35)", fontsize=13, fontweight="bold")
    save(fig, "cm_main.png")


def main() -> None:
    setup_theme()
    make_classical_figures()
    make_suite_figure()
    make_model_family_figure()
    make_model_strict_figures()
    make_feature_importance_figure()
    make_modality_figure()
    make_main_cm_figure()
    print("Refreshed clean Chapter 4 figures.")


if __name__ == "__main__":
    main()
