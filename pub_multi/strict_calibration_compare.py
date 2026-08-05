from __future__ import annotations

"""Auxiliary calibration comparison for strict no-leak reporting.

This script is only used to compare post-hoc calibration choices on top of the
existing strict feature setting. It should be cited as analysis support, not as
the main preprocess or training implementation of the project.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import bench_strict_models as strict


PKG_DIR = Path(__file__).resolve().parent
RES_DIR = PKG_DIR / "res"
FIG_DIR = PKG_DIR / "fig"
DOC_DIR = PKG_DIR / "doc"
LOG_DIR = PKG_DIR / "log"


def fit_sigmoid(raw_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(raw_scores.reshape(-1, 1), labels)
    return model


def predict_sigmoid(model: LogisticRegression, raw_scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def threshold_table(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    thresholds = np.arange(0.01, 0.51, 0.01)
    rows = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tn, fp, fn, tp = strict.confusion_matrix(labels, pred).ravel()
        rows.append(
            {
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(strict.precision_score(labels, pred, zero_division=0)),
                "recall": float(strict.recall_score(labels, pred, zero_division=0)),
                "f1": float(strict.f1_score(labels, pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = strict.confusion_matrix(labels, pred).ravel()
    return {
        "threshold": float(threshold),
        "test_ap": float(strict.average_precision_score(labels, probs)),
        "test_auc": float(strict.roc_auc_score(labels, probs)),
        "test_f1": float(strict.f1_score(labels, pred, zero_division=0)),
        "test_precision": float(strict.precision_score(labels, pred, zero_division=0)),
        "test_recall": float(strict.recall_score(labels, pred, zero_division=0)),
        "test_ece": strict.expected_calibration_error(labels, probs),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
        "unique_calibrated_prob_count": int(len(np.unique(np.round(probs, 6)))),
    }


def summarize_model(
    model_name: str,
    calibrator_name: str,
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    threshold, best_val_f1 = strict.choose_threshold(val_labels, val_probs)
    sweep = threshold_table(test_labels, test_probs)
    sweep.insert(0, "calibrator", calibrator_name)
    sweep.insert(0, "model", model_name)

    best_test_f1 = sweep.iloc[sweep["f1"].idxmax()]
    p95 = sweep[sweep["precision"] >= 0.95]
    best_p95 = p95.iloc[p95["recall"].idxmax()] if not p95.empty else None

    row = {
        "model": model_name,
        "calibrator": calibrator_name,
        "val_f1": float(best_val_f1),
        "val_ece": strict.expected_calibration_error(val_labels, val_probs),
        **evaluate(test_labels, test_probs, threshold),
        "best_test_f1_threshold_diag": float(best_test_f1["threshold"]),
        "best_test_f1_precision_diag": float(best_test_f1["precision"]),
        "best_test_f1_recall_diag": float(best_test_f1["recall"]),
        "best_test_f1_fp_diag": int(best_test_f1["fp"]),
        "best_precision_ge_095_threshold_diag": None if best_p95 is None else float(best_p95["threshold"]),
        "best_precision_ge_095_precision_diag": None if best_p95 is None else float(best_p95["precision"]),
        "best_precision_ge_095_recall_diag": None if best_p95 is None else float(best_p95["recall"]),
        "best_precision_ge_095_fp_diag": None if best_p95 is None else int(best_p95["fp"]),
    }
    return row, sweep


def fit_base_model(model: object, use_scaler: bool, train_df: pd.DataFrame, feature_cols: list[str]):
    train_x = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    train_y = train_df["label"].to_numpy(dtype=np.int8)

    scaler = None
    if use_scaler:
        scaler = strict.StandardScaler()
        train_x = scaler.fit_transform(train_x)
    model.fit(train_x, train_y)
    return model, scaler


def save_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name in plot_df["model"].unique():
        sub = plot_df[plot_df["model"] == model_name]
        axes[0].plot(sub["test_recall"], sub["test_precision"], marker="o", label=model_name)
        for row in sub.itertuples(index=False):
            axes[0].annotate(row.calibrator, (row.test_recall, row.test_precision), xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("Test Recall")
    axes[0].set_ylabel("Test Precision")
    axes[0].set_title("Strict Calibration: Precision vs Recall")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x = np.arange(len(plot_df))
    width = 0.22
    axes[1].bar(x - width, plot_df["test_f1"], width=width, label="F1")
    axes[1].bar(x, plot_df["test_precision"], width=width, label="Precision")
    axes[1].bar(x + width, plot_df["test_recall"], width=width, label="Recall")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{m}\n{c}" for m, c in zip(plot_df["model"], plot_df["calibrator"])], rotation=20)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Strict Calibration Metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_calibration_cmp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown(df: pd.DataFrame) -> None:
    lines = [
        "# Strict Calibration Comparison",
        "",
        "เอกสารนี้เปรียบเทียบการทำ calibration แบบ `isotonic` และ `sigmoid` บน strict no-leak setting เดียวกัน โดยใช้ model family และ feature set เดิม เพื่อดูว่าการปรับ calibration ทำให้ operating point สมจริงขึ้นหรือไม่",
        "",
        "| model | calibrator | threshold | AP | AUC | F1 | Precision | Recall | FP | unique probs | หมายเหตุ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in df.itertuples(index=False):
        if row.test_precision == 1.0:
            note = "ยังเป็น conservative point"
        elif row.test_precision >= 0.95:
            note = "สมจริงขึ้นและยังคุม FP ได้ดี"
        else:
            note = "recall สูงขึ้นแต่ precision ลดลงมาก"
        lines.append(
            f"| `{row.model}` | `{row.calibrator}` | {row.threshold:.2f} | {row.test_ap:.4f} | {row.test_auc:.4f} | {row.test_f1:.4f} | "
            f"{row.test_precision:.4f} | {row.test_recall:.4f} | {int(row.test_fp)} | {int(row.unique_calibrated_prob_count)} | {note} |"
        )

    lines.extend(
        [
            "",
            "## ข้อสรุป",
            "",
            "หาก isotonic ทำให้คะแนน calibrated probability มีค่าอยู่เพียงไม่กี่ระดับ จะเกิด threshold jump ได้ง่าย กล่าวคือ threshold ขยับเล็กน้อยแต่ precision และจำนวน false positives เปลี่ยนแบบก้าวกระโดด ในกรณีดังกล่าว sigmoid calibration มักให้คะแนนที่ละเอียดกว่าและช่วยให้เลือก operating point ที่ไม่ตันอยู่ที่ precision เท่ากับ 1 ได้ง่ายขึ้น",
            "",
            "อย่างไรก็ตาม การเลือก calibration ที่ดีที่สุดควรพิจารณาร่วมกันทั้ง F1, precision, recall, จำนวน false positives และความสามารถในการอธิบายเชิงระบบ ไม่ควรพิจารณาเพียงว่า precision ลดลงจาก 1 หรือไม่เท่านั้น",
        ]
    )
    (DOC_DIR / "strict_calibration_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    strict.ensure_dirs()
    strict.set_seed(42)
    for path in [RES_DIR, FIG_DIR, DOC_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    feature_cols = strict.load_feature_cols()
    columns = strict.KEY_COLS + feature_cols
    train_df = strict.build_balanced_train(strict.MERGED_FILES["train"], columns, neg_ratio=3.0, seed=42)

    rows: list[dict[str, object]] = []
    sweep_frames: list[pd.DataFrame] = []
    for model_name, (model, use_scaler) in strict.get_models(seed=42).items():
        print(f"Running strict calibration compare for {model_name} ...")
        base_model, scaler = fit_base_model(model, use_scaler, train_df, feature_cols)
        val_raw, val_y = strict.collect_predictions(base_model, scaler, strict.MERGED_FILES["val"], feature_cols)
        test_raw, test_y = strict.collect_predictions(base_model, scaler, strict.MERGED_FILES["test"], feature_cols)

        isotonic = strict.IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(val_raw, val_y)
        iso_val_probs = isotonic.predict(val_raw)
        iso_test_probs = isotonic.predict(test_raw)
        row, sweep = summarize_model(model_name, "isotonic", val_y, iso_val_probs, test_y, iso_test_probs)
        rows.append(row)
        sweep_frames.append(sweep)

        sigmoid = fit_sigmoid(val_raw, val_y)
        sig_val_probs = predict_sigmoid(sigmoid, val_raw)
        sig_test_probs = predict_sigmoid(sigmoid, test_raw)
        row, sweep = summarize_model(model_name, "sigmoid", val_y, sig_val_probs, test_y, sig_test_probs)
        rows.append(row)
        sweep_frames.append(sweep)

    df = pd.DataFrame(rows).sort_values(["model", "calibrator"]).reset_index(drop=True)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)

    df.to_csv(RES_DIR / "strict_calibration_cmp.csv", index=False, encoding="utf-8-sig")
    sweep_df.to_csv(RES_DIR / "strict_calibration_sweep.csv", index=False, encoding="utf-8-sig")
    save_plot(df)
    save_markdown(df)

    summary = {
        "feature_count": len(feature_cols),
        "train_balanced_rows": int(len(train_df)),
        "rows": json.loads(df.to_json(orient="records")),
    }
    (LOG_DIR / "strict_calibration_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df[["model", "calibrator", "threshold", "test_f1", "test_precision", "test_recall", "test_fp", "unique_calibrated_prob_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
