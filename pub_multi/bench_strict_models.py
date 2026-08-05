from __future__ import annotations

"""Auxiliary strict benchmark on top of existing strict artifacts.

This script is not the original preprocess/train pipeline. It reuses the
feature matrices and strict exclusion list that already exist in the project
to prepare report-facing comparison tables only.
"""

import json
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
TRAIN_DATA_DIR = PROJECT_ROOT / "train_data"
RES_DIR = PKG_DIR / "res"
FIG_DIR = PKG_DIR / "fig"
DOC_DIR = PKG_DIR / "doc"
LOG_DIR = PKG_DIR / "log"

FEATURES_DIR = TRAIN_DATA_DIR / "stage9_pipeline_chunked" / "features"
ARTIFACTS_DIR = TRAIN_DATA_DIR / "stage9_pipeline_chunked" / "artifacts"
STRICT_REPORT = TRAIN_DATA_DIR / "stage10_13_training_noleak_strict" / "reports" / "evaluation_summary.json"
STRICT_EXCLUDE = TRAIN_DATA_DIR / "leakage_excluded_features_strict.txt"

KEY_COLS = ["profile_id_a", "profile_id_b", "label", "pair_type"]
MERGED_FILES = {
    "train": FEATURES_DIR / "feature_matrix_chunked_train_merged.parquet",
    "val": FEATURES_DIR / "feature_matrix_chunked_val_merged.parquet",
    "test": FEATURES_DIR / "feature_matrix_chunked_test_merged.parquet",
}


def ensure_dirs() -> None:
    for path in [RES_DIR, FIG_DIR, DOC_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.01, 0.51, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probs >= left) & (probs <= right if right == 1.0 else probs < right)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        total += mask.mean() * abs(acc - conf)
    return float(total)


def load_feature_cols() -> list[str]:
    with (ARTIFACTS_DIR / "feature_cols.pkl").open("rb") as fh:
        cols = pickle.load(fh)
    excluded = {
        line.strip()
        for line in STRICT_EXCLUDE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    schema_cols = set(pq.ParquetFile(MERGED_FILES["train"]).schema.names)
    return [col for col in cols if col not in excluded and col in schema_cols]


def iter_batches(path: Path, columns: list[str], batch_size: int = 250_000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def build_balanced_train(path: Path, columns: list[str], neg_ratio: float, seed: int) -> pd.DataFrame:
    positives: list[pd.DataFrame] = []
    pos_total = 0
    neg_total = 0
    for batch in iter_batches(path, columns):
        pos = batch[batch["label"] == 1]
        neg = batch[batch["label"] == 0]
        if not pos.empty:
            positives.append(pos)
            pos_total += len(pos)
        neg_total += len(neg)

    target_neg = min(neg_total, int(pos_total * neg_ratio))
    if target_neg == 0:
        raise ValueError("No negatives available for strict benchmark.")

    neg_parts: list[pd.DataFrame] = []
    seen_neg = 0
    taken_neg = 0
    for idx, batch in enumerate(iter_batches(path, columns)):
        neg = batch[batch["label"] == 0]
        if neg.empty:
            continue
        seen_neg += len(neg)
        target_cum = round(target_neg * seen_neg / neg_total)
        take = int(target_cum - taken_neg)
        if take > 0:
            take = min(take, len(neg))
            neg_parts.append(neg.sample(n=take, random_state=seed + idx))
            taken_neg += take
        if taken_neg >= target_neg:
            break

    train_df = pd.concat([*positives, *neg_parts], ignore_index=True)
    return train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def collect_predictions(
    model: object,
    scaler: StandardScaler | None,
    path: Path,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    raw_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    columns = KEY_COLS + feature_cols
    for batch in iter_batches(path, columns):
        x = batch[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if scaler is not None:
            x = scaler.transform(x)
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(x)[:, 1]
        else:
            raw = model.decision_function(x)
        raw_list.append(np.asarray(raw, dtype=np.float32))
        label_list.append(batch["label"].to_numpy(dtype=np.int8))
    return np.concatenate(raw_list), np.concatenate(label_list)


def get_models(seed: int) -> dict[str, tuple[object, bool]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            True,
        ),
        "gb": (
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=seed,
            ),
            True,
        ),
        "rf": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=seed,
            ),
            True,
        ),
        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=seed,
            ),
            True,
        ),
    }


def fit_and_eval(model_name: str, model: object, use_scaler: bool, train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    train_x = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    train_y = train_df["label"].to_numpy(dtype=np.int8)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)

    started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started

    val_raw, val_y = collect_predictions(model, scaler, MERGED_FILES["val"], feature_cols)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_probs = calibrator.predict(val_raw)
    threshold, best_val_f1 = choose_threshold(val_y, val_probs)

    test_raw, test_y = collect_predictions(model, scaler, MERGED_FILES["test"], feature_cols)
    test_probs = calibrator.predict(test_raw)
    test_pred = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()

    return {
        "model": model_name,
        "source": "strict_rerun",
        "feature_count": len(feature_cols),
        "fit_seconds": float(fit_seconds),
        "threshold": threshold,
        "val_ap": float(average_precision_score(val_y, val_probs)),
        "val_auc": float(roc_auc_score(val_y, val_probs)),
        "val_f1": float(best_val_f1),
        "val_ece": expected_calibration_error(val_y, val_probs),
        "test_ap": float(average_precision_score(test_y, test_probs)),
        "test_auc": float(roc_auc_score(test_y, test_probs)),
        "test_f1": float(f1_score(test_y, test_pred, zero_division=0)),
        "test_precision": float(precision_score(test_y, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_y, test_pred, zero_division=0)),
        "test_ece": expected_calibration_error(test_y, test_probs),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
    }


def load_existing_mlp() -> dict[str, object]:
    report = json.loads(STRICT_REPORT.read_text(encoding="utf-8"))
    tn, fp = report["confusion_matrix"][0]
    fn, tp = report["confusion_matrix"][1]
    return {
        "model": "mlp",
        "source": "strict_existing_report",
        "feature_count": int(report["n_features"]),
        "fit_seconds": np.nan,
        "threshold": float(report["threshold"]),
        "val_ap": float(report["train_meta"]["best_val_ap"]),
        "val_auc": float(report["train_meta"]["history"][0]["val_auc"]),
        "val_f1": float(report["best_val_f1"]),
        "val_ece": float(report["val_ece_after"]),
        "test_ap": float(report["test_avg_precision"]),
        "test_auc": float(report["test_roc_auc"]),
        "test_f1": float(report["test_f1"]),
        "test_precision": float(report["test_precision"]),
        "test_recall": float(report["test_recall"]),
        "test_ece": np.nan,
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
    }


def save_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(plot_df))
    width = 0.24
    for idx, col in enumerate(["test_ap", "test_auc", "test_f1"]):
        axes[0].bar(x + (idx - 1) * width, plot_df[col], width=width, label=col.replace("test_", "").upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["model"], rotation=20)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("Strict No-Leak Test Metrics")
    axes[0].legend()

    axes[1].scatter(plot_df["test_recall"], plot_df["test_precision"], s=90)
    for row in plot_df.itertuples(index=False):
        axes[1].annotate(f"{row.model}\n{row.source}", (row.test_recall, row.test_precision), xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Test Recall")
    axes[1].set_ylabel("Test Precision")
    axes[1].set_title("Strict No-Leak Precision vs Recall")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_strict_cmp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_grid(df: pd.DataFrame) -> None:
    plot_df = df.reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for ax, row in zip(axes, plot_df.itertuples(index=False)):
        cm = np.array([[row.test_tn, row.test_fp], [row.test_fn, row.test_tp]], dtype=int)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{row.model} ({row.test_f1:.3f})")
        ax.set_xticks([0, 1], labels=["NO_MATCH", "MATCH"])
        ax.set_yticks([0, 1], labels=["NO_MATCH", "MATCH"])
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("on")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_strict_cm_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown(df: pd.DataFrame) -> None:
    lines = [
        "# Strict No-Leak Model Comparison",
        "",
        "ตารางนี้ใช้ strict feature exclusion ชุดเดียวกับ `run_stage10_13_training_noleak_strict.ps1` และเปรียบเทียบโมเดล classical ที่ rerun ใหม่กับ strict MLP reference ที่มีอยู่เดิม",
        "",
    ]
    for row in df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.model}",
                "",
                f"- source: `{row.source}`",
                f"- test AP/AUC/F1: `{row.test_ap:.4f} / {row.test_auc:.4f} / {row.test_f1:.4f}`",
                f"- test Precision/Recall: `{row.test_precision:.4f} / {row.test_recall:.4f}`",
                f"- threshold: `{row.threshold:.4f}`",
                "",
            ]
        )
    (DOC_DIR / "model_strict_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(42)
    feature_cols = load_feature_cols()
    columns = KEY_COLS + feature_cols

    print("Building strict balanced train set ...")
    train_df = build_balanced_train(MERGED_FILES["train"], columns, neg_ratio=3.0, seed=42)
    print(f"Train balanced rows: {len(train_df):,}")

    rows = []
    for model_name, (model, use_scaler) in get_models(seed=42).items():
        print(f"Running strict {model_name} ...")
        rows.append(fit_and_eval(model_name, model, use_scaler, train_df, feature_cols))

    rows.append(load_existing_mlp())
    df = pd.DataFrame(rows).sort_values(["test_f1", "test_ap", "test_auc"], ascending=False).reset_index(drop=True)
    df.to_csv(RES_DIR / "model_strict_cmp.csv", index=False, encoding="utf-8-sig")
    save_plot(df)
    save_confusion_grid(df)
    save_markdown(df)

    summary = {
        "feature_count": len(feature_cols),
        "train_balanced_rows": int(len(train_df)),
        "models": df[["model", "source", "test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]].to_dict(orient="records"),
    }
    (LOG_DIR / "model_strict_cmp.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df[["model", "source", "test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
