from __future__ import annotations

import itertools
import json
import pickle
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
RES_DIR = PKG_DIR / "res"
FIG_DIR = PKG_DIR / "fig"
DOC_DIR = PKG_DIR / "doc"
LOG_DIR = PKG_DIR / "log"

MAIN_RUN = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42"
PAIR_FEATURES = MAIN_RUN / "artifacts" / "pair_features.parquet"
FEATURE_COLS_PKL = MAIN_RUN / "models" / "feature_cols.pkl"


def ensure_dirs() -> None:
    for path in [RES_DIR, FIG_DIR, DOC_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    feature_df = pd.read_parquet(PAIR_FEATURES)
    with FEATURE_COLS_PKL.open("rb") as fh:
        feature_cols = pickle.load(fh)
    return feature_df, feature_cols


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def get_configs(seed: int) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []

    gb_grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6},
        {"n_estimators": 300, "learning_rate": 0.08, "max_depth": 4},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 4},
    ]
    for idx, params in enumerate(gb_grid, start=1):
        configs.append(
            {
                "family": "gb",
                "config_id": f"gb_{idx}",
                "use_scaler": False,
                "model": GradientBoostingClassifier(random_state=seed, **params),
                "params": params,
            }
        )

    rf_grid = [
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 18, "min_samples_leaf": 2},
        {"n_estimators": 600, "max_depth": 18, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": 24, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 14, "min_samples_leaf": 4},
    ]
    for idx, params in enumerate(rf_grid, start=1):
        configs.append(
            {
                "family": "rf",
                "config_id": f"rf_{idx}",
                "use_scaler": False,
                "model": RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=seed,
                    **params,
                ),
                "params": params,
            }
        )

    mlp_grid = [
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-3, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 3e-4},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-3, "learning_rate_init": 3e-4},
    ]
    for idx, params in enumerate(mlp_grid, start=1):
        configs.append(
            {
                "family": "mlp",
                "config_id": f"mlp_{idx}",
                "use_scaler": True,
                "model": MLPClassifier(
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    max_iter=220,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed,
                    **params,
                ),
                "params": params,
            }
        )

    return configs


def evaluate_config(
    config: dict[str, object],
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, object]:
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    train_x = train_df[feature_cols].fillna(0.0).to_numpy()
    val_x = val_df[feature_cols].fillna(0.0).to_numpy()
    test_x = test_df[feature_cols].fillna(0.0).to_numpy()
    train_y = train_df["label"].to_numpy()
    val_y = val_df["label"].to_numpy()
    test_y = test_df["label"].to_numpy()

    if config["use_scaler"]:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)

    model = config["model"]
    started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started

    val_raw = model.predict_proba(val_x)[:, 1]
    test_raw = model.predict_proba(test_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_probs = calibrator.predict(val_raw)
    test_probs = calibrator.predict(test_raw)
    threshold, best_val_f1 = choose_threshold(val_y, val_probs)
    test_pred = (test_probs >= threshold).astype(int)

    return {
        "family": config["family"],
        "config_id": config["config_id"],
        "params": json.dumps(config["params"], sort_keys=True),
        "fit_seconds": float(fit_seconds),
        "threshold": threshold,
        "val_ap": float(average_precision_score(val_y, val_probs)),
        "val_auc": float(roc_auc_score(val_y, val_probs)),
        "val_f1": best_val_f1,
        "test_ap": float(average_precision_score(test_y, test_probs)),
        "test_auc": float(roc_auc_score(test_y, test_probs)),
        "test_f1": float(f1_score(test_y, test_pred, zero_division=0)),
        "test_precision": float(precision_score(test_y, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_y, test_pred, zero_division=0)),
    }


def save_plot(df: pd.DataFrame) -> None:
    best_df = df.sort_values(["family", "val_composite"], ascending=[True, False]).groupby("family").head(1).copy()
    families = list(best_df["family"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(families))
    width = 0.24
    for idx, col in enumerate(["test_ap", "test_auc", "test_f1"]):
        axes[0].bar(x + (idx - 1) * width, best_df[col], width=width, label=col.replace("test_", "").upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families)
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("Best Config per Model Family")
    axes[0].legend()

    for family, group in df.groupby("family"):
        axes[1].scatter(group["fit_seconds"], group["val_composite"], label=family, s=60)
    axes[1].set_xlabel("Fit Seconds")
    axes[1].set_ylabel("Validation Composite")
    axes[1].set_title("Tuning Trade-off")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_model_tune.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown(df: pd.DataFrame) -> None:
    top_df = df.sort_values(["family", "val_composite"], ascending=[True, False]).groupby("family").head(1).copy()
    lines = [
        "# Top Model Tuning",
        "",
        "ตารางนี้เก็บเฉพาะค่าที่ดีที่สุดของแต่ละ family หลัง tune บน main multimodal run เดิม",
        "",
    ]
    for row in top_df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.family}",
                "",
                f"- best config: `{row.config_id}`",
                f"- params: `{row.params}`",
                f"- val composite: `{row.val_composite:.4f}`",
                f"- test AP/AUC/F1: `{row.test_ap:.4f} / {row.test_auc:.4f} / {row.test_f1:.4f}`",
                f"- test Precision/Recall: `{row.test_precision:.4f} / {row.test_recall:.4f}`",
                "",
            ]
        )
    (DOC_DIR / "top_model_tune.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feature_df, feature_cols = load_dataset()
    splits = {name: feature_df[feature_df["split_name"] == name].reset_index(drop=True) for name in ["train", "val", "test"]}

    rows = []
    for config in get_configs(seed=42):
        print(f"Running {config['config_id']} ...")
        rows.append(evaluate_config(config, splits, feature_cols))

    df = pd.DataFrame(rows)
    df["val_composite"] = 0.45 * df["val_ap"] + 0.35 * df["val_f1"] + 0.20 * df["val_auc"]
    df = df.sort_values(["val_composite", "test_ap", "test_f1"], ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    df.to_csv(RES_DIR / "top_model_tune.csv", index=False, encoding="utf-8-sig")
    save_plot(df)
    save_markdown(df)

    best_by_family = df.groupby("family").head(1)[["family", "config_id", "val_composite", "test_ap", "test_auc", "test_f1"]]
    summary = {
        "best_overall": df.loc[0, ["family", "config_id", "val_composite", "test_ap", "test_auc", "test_f1"]].to_dict(),
        "best_by_family": best_by_family.to_dict(orient="records"),
    }
    (LOG_DIR / "top_model_tune.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df[["rank", "family", "config_id", "val_composite", "test_ap", "test_auc", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
