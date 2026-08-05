from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ARTIFACT_DIR = PROJECT_ROOT / "train_data" / "leakage_safe_experiment" / "artifacts"
REPORT_PATH = PROJECT_ROOT / "train_data" / "leakage_safe_experiment" / "reports" / "experiment_report.json"
RES_DIR = ROOT / "res"
FIG_DIR = ROOT / "fig"


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (11, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["font.size"] = 10


def load_inputs() -> tuple[pd.DataFrame, list[str], dict]:
    pair_features = pd.read_parquet(ARTIFACT_DIR / "pair_features.parquet")
    with (ARTIFACT_DIR / "feature_cols.pkl").open("rb") as fh:
        feature_cols = pickle.load(fh)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return pair_features, feature_cols, report


def fit_models(pair_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    split_frames = {
        split: pair_features[pair_features["split_name"] == split].reset_index(drop=True)
        for split in ["train", "val", "test"]
    }

    scaler = StandardScaler()
    train_x = scaler.fit_transform(split_frames["train"][feature_cols].fillna(0.0))
    val_x = scaler.transform(split_frames["val"][feature_cols].fillna(0.0))
    test_x = scaler.transform(split_frames["test"][feature_cols].fillna(0.0))
    train_y = split_frames["train"]["label"].to_numpy()
    val_y = split_frames["val"]["label"].to_numpy()
    test_y = split_frames["test"]["label"].to_numpy()

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "gb": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(train_x, train_y)
        val_probs_raw = model.predict_proba(val_x)[:, 1]
        test_probs_raw = model.predict_proba(test_x)[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs_raw, val_y)
        val_probs = calibrator.predict(val_probs_raw)
        test_probs = calibrator.predict(test_probs_raw)

        threshold, best_val_f1 = choose_threshold(val_y, val_probs)
        val_pred = (val_probs >= threshold).astype(int)
        test_pred = (test_probs >= threshold).astype(int)
        cm = confusion_matrix(test_y, test_pred)

        rows.append(
            {
                "model": model_name,
                "feature_count": len(feature_cols),
                "threshold": float(threshold),
                "val_ap": float(average_precision_score(val_y, val_probs)),
                "val_auc": float(roc_auc_score(val_y, val_probs)),
                "val_f1": float(f1_score(val_y, val_pred, zero_division=0)),
                "best_val_f1": float(best_val_f1),
                "test_ap": float(average_precision_score(test_y, test_probs)),
                "test_auc": float(roc_auc_score(test_y, test_probs)),
                "test_f1": float(f1_score(test_y, test_pred, zero_division=0)),
                "test_precision": float(precision_score(test_y, test_pred, zero_division=0)),
                "test_recall": float(recall_score(test_y, test_pred, zero_division=0)),
                "test_tn": int(cm[0, 0]),
                "test_fp": int(cm[0, 1]),
                "test_fn": int(cm[1, 0]),
                "test_tp": int(cm[1, 1]),
            }
        )

    result = pd.DataFrame(rows).sort_values(["test_f1", "test_ap", "test_auc"], ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def validate_against_report(result: pd.DataFrame, report: dict) -> None:
    best_row = result.iloc[0]
    metrics = report["metrics"]
    if best_row["model"] != metrics["best_model"]:
        raise RuntimeError(f"Best model mismatch: computed {best_row['model']} vs report {metrics['best_model']}")
    numeric_checks = {
        "threshold": metrics["threshold"],
        "test_ap": metrics["test_avg_precision"],
        "test_auc": metrics["test_roc_auc"],
        "test_f1": metrics["test_f1"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
    }
    for key, expected in numeric_checks.items():
        if not np.isclose(float(best_row[key]), float(expected), atol=1e-9):
            raise RuntimeError(f"Mismatch for {key}: computed {best_row[key]} vs report {expected}")


def save_outputs(result: pd.DataFrame) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(RES_DIR / "classical_leaksafe_cmp.csv", index=False)

    detail = {
        "source": "train_data/leakage_safe_experiment/artifacts/pair_features.parquet",
        "models": result.to_dict(orient="records"),
    }
    (RES_DIR / "classical_leaksafe_cmp.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")


def make_metric_figure(result: pd.DataFrame) -> None:
    metric_map = {
        "test_ap": "AP",
        "test_auc": "AUC",
        "test_f1": "F1",
        "test_precision": "Precision",
        "test_recall": "Recall",
    }
    plot_df = result[["model", *metric_map.keys()]].melt(id_vars="model", var_name="metric", value_name="score")
    plot_df["metric"] = plot_df["metric"].map(metric_map)
    model_label_map = {
        "gb": "Gradient Boosting",
        "rf": "Random Forest",
        "logreg": "Logistic Regression",
    }
    plot_df["model"] = plot_df["model"].map(model_label_map)

    palette = {
        "Gradient Boosting": "#0f766e",
        "Random Forest": "#0ea5e9",
        "Logistic Regression": "#f59e0b",
    }

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model", palette=palette, ax=ax)
    ax.set_ylim(0.82, 1.0)
    ax.set_title("Classical Leakage-Safe Comparison on the Same 22-Feature Split")
    ax.set_xlabel("")
    ax.set_ylabel("score")
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.003,
            f"{patch.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax.legend(frameon=False, title="")
    fig.subplots_adjust(bottom=0.16, top=0.90)
    fig.savefig(FIG_DIR / "classical_leaksafe_cmp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_cm_figure(result: pd.DataFrame) -> None:
    ordered = result.copy()
    ordered["label"] = ordered["model"].map(
        {
            "gb": "Gradient Boosting",
            "rf": "Random Forest",
            "logreg": "Logistic Regression",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    cmap = sns.light_palette("#0f766e", as_cmap=True)
    for ax, (_, row) in zip(axes, ordered.iterrows()):
        cm = np.array([[row["test_tn"], row["test_fp"]], [row["test_fn"], row["test_tp"]]])
        annot = np.array([[f"{int(v):,}" for v in line] for line in cm])
        sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, cbar=False, square=True, linewidths=1.5, linecolor="white", ax=ax)
        ax.set_title(f"{row['label']}\nF1={row['test_f1']:.3f}")
        ax.set_xticklabels(["Pred NO_MATCH", "Pred MATCH"], rotation=0)
        ax.set_yticklabels(["Actual NO_MATCH", "Actual MATCH"], rotation=0)
    fig.suptitle("Confusion Matrices for the Classical Leakage-Safe Line", y=1.03)
    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.35)
    fig.savefig(FIG_DIR / "classical_leaksafe_cm_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_theme()
    pair_features, feature_cols, report = load_inputs()
    result = fit_models(pair_features, feature_cols)
    validate_against_report(result, report)
    save_outputs(result)
    make_metric_figure(result)
    make_cm_figure(result)
    print("Saved classical leakage-safe benchmark outputs.")


if __name__ == "__main__":
    main()
