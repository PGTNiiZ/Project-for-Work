from __future__ import annotations

"""Auxiliary threshold diagnostic for report writing.

This file does not replace the original training code. It reads predictions
from the same strict feature setting and expands them into threshold-sweep
tables so the report can explain why precision reached 1.0.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bench_strict_models as strict


PKG_DIR = Path(__file__).resolve().parent
RES_DIR = PKG_DIR / "res"
DOC_DIR = PKG_DIR / "doc"
LOG_DIR = PKG_DIR / "log"


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


def fit_model_diag(model_name: str, model: object, use_scaler: bool, train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    train_x = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    train_y = train_df["label"].to_numpy(dtype=np.int8)

    scaler = None
    if use_scaler:
        scaler = strict.StandardScaler()
        train_x = scaler.fit_transform(train_x)

    model.fit(train_x, train_y)

    val_raw, val_y = strict.collect_predictions(model, scaler, strict.MERGED_FILES["val"], feature_cols)
    calibrator = strict.IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_probs = calibrator.predict(val_raw)
    current_threshold, _ = strict.choose_threshold(val_y, val_probs)

    test_raw, test_y = strict.collect_predictions(model, scaler, strict.MERGED_FILES["test"], feature_cols)
    test_probs = calibrator.predict(test_raw)
    sweep = threshold_table(test_y, test_probs)
    sweep.insert(0, "model", model_name)

    current_row = sweep.loc[np.isclose(sweep["threshold"], current_threshold)].iloc[0]
    lower_than_one = sweep[sweep["precision"] < 1.0]
    first_lt_one = lower_than_one.iloc[-1] if not lower_than_one.empty else None
    best_f1 = sweep.iloc[sweep["f1"].idxmax()]
    p95 = sweep[sweep["precision"] >= 0.95]
    best_p95 = p95.iloc[p95["recall"].idxmax()] if not p95.empty else None

    unique_probs = np.unique(np.round(test_probs, 6))

    diag = {
        "model": model_name,
        "current_threshold": float(current_threshold),
        "current_tp": int(current_row["tp"]),
        "current_fp": int(current_row["fp"]),
        "current_fn": int(current_row["fn"]),
        "current_tn": int(current_row["tn"]),
        "current_precision": float(current_row["precision"]),
        "current_recall": float(current_row["recall"]),
        "current_f1": float(current_row["f1"]),
        "first_threshold_below_precision_1_test_diag": None if first_lt_one is None else float(first_lt_one["threshold"]),
        "first_threshold_below_precision_1_precision_test_diag": None if first_lt_one is None else float(first_lt_one["precision"]),
        "first_threshold_below_precision_1_recall_test_diag": None if first_lt_one is None else float(first_lt_one["recall"]),
        "first_threshold_below_precision_1_fp_test_diag": None if first_lt_one is None else int(first_lt_one["fp"]),
        "best_test_f1_threshold_diag": float(best_f1["threshold"]),
        "best_test_f1_precision_diag": float(best_f1["precision"]),
        "best_test_f1_recall_diag": float(best_f1["recall"]),
        "best_test_f1_fp_diag": int(best_f1["fp"]),
        "best_precision_ge_095_threshold_diag": None if best_p95 is None else float(best_p95["threshold"]),
        "best_precision_ge_095_precision_diag": None if best_p95 is None else float(best_p95["precision"]),
        "best_precision_ge_095_recall_diag": None if best_p95 is None else float(best_p95["recall"]),
        "best_precision_ge_095_fp_diag": None if best_p95 is None else int(best_p95["fp"]),
        "unique_calibrated_prob_count": int(len(unique_probs)),
        "unique_calibrated_prob_values": [float(v) for v in unique_probs.tolist()],
    }
    return diag, sweep


def save_markdown(diag_df: pd.DataFrame) -> None:
    lines = [
        "# Strict Threshold Diagnostic",
        "",
        "เอกสารนี้ใช้สำหรับอธิบายว่าทำไม strict rerun หลายโมเดลจึงมี precision เท่ากับ `1.0000` และถ้าต้องการ operating point ที่สมจริงขึ้นควรขยับ threshold ไปอย่างไร",
        "",
        "| model | current threshold | current P | current R | current FP | first threshold with P<1 (test diag) | P | R | FP | unique calibrated probs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diag_df.itertuples(index=False):
        lines.append(
            f"| `{row.model}` | {row.current_threshold:.2f} | {row.current_precision:.4f} | {row.current_recall:.4f} | {row.current_fp} | "
            f"{'' if pd.isna(row.first_threshold_below_precision_1_test_diag) else f'{row.first_threshold_below_precision_1_test_diag:.2f}'} | "
            f"{'' if pd.isna(row.first_threshold_below_precision_1_precision_test_diag) else f'{row.first_threshold_below_precision_1_precision_test_diag:.4f}'} | "
            f"{'' if pd.isna(row.first_threshold_below_precision_1_recall_test_diag) else f'{row.first_threshold_below_precision_1_recall_test_diag:.4f}'} | "
            f"{'' if pd.isna(row.first_threshold_below_precision_1_fp_test_diag) else int(row.first_threshold_below_precision_1_fp_test_diag)} | "
            f"{int(row.unique_calibrated_prob_count)} |"
        )
    lines.extend(
        [
            "",
            "## ข้อสังเกต",
            "",
            "ผลวิเคราะห์ชี้ว่าคะแนน calibrated probability ของ strict rerun มีค่าอยู่เพียงไม่กี่ระดับเท่านั้น ทำให้ threshold ขยับเพียงเล็กน้อยก็อาจทำให้จำนวน false positives กระโดดขึ้นทันที แทนที่จะค่อย ๆ เปลี่ยนอย่างต่อเนื่อง ปัญหานี้สอดคล้องกับการใช้ isotonic calibration บน validation set ที่แยกคู่บวกและคู่ลบได้ง่ายมาก จึงเกิด probability แบบ stepwise และทำให้ operating point ที่ได้มีลักษณะ conservative มาก",
            "",
            "ดังนั้น หากต้องการผลที่สมจริงขึ้น ไม่ควรรายงานเฉพาะจุดที่ precision เท่ากับ 1 เพียงจุดเดียว แต่ควรรายงาน threshold sweep ร่วมด้วย และพิจารณาเปลี่ยน calibration strategy หรือใช้ validation set ที่ยากขึ้นเพื่อให้ probability distribution ละเอียดและสะท้อนความไม่แน่นอนมากขึ้น",
        ]
    )
    (DOC_DIR / "strict_threshold_diag.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    strict.ensure_dirs()
    strict.set_seed(42)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = strict.load_feature_cols()
    columns = strict.KEY_COLS + feature_cols
    train_df = strict.build_balanced_train(strict.MERGED_FILES["train"], columns, neg_ratio=3.0, seed=42)

    diag_rows: list[dict[str, object]] = []
    sweep_frames: list[pd.DataFrame] = []
    for model_name, (model, use_scaler) in strict.get_models(seed=42).items():
        diag, sweep = fit_model_diag(model_name, model, use_scaler, train_df, feature_cols)
        diag_rows.append(diag)
        sweep_frames.append(sweep)

    diag_df = pd.DataFrame(diag_rows).sort_values("current_f1", ascending=False).reset_index(drop=True)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)

    diag_df.to_csv(RES_DIR / "strict_threshold_diag.csv", index=False, encoding="utf-8-sig")
    sweep_df.to_csv(RES_DIR / "strict_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    save_markdown(diag_df)

    log_payload = {
        "feature_count": len(feature_cols),
        "train_balanced_rows": int(len(train_df)),
        "diag_rows": json.loads(diag_df.to_json(orient="records")),
    }
    (LOG_DIR / "strict_threshold_diag.json").write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    print(diag_df[[
        "model",
        "current_threshold",
        "current_precision",
        "current_recall",
        "first_threshold_below_precision_1_test_diag",
        "first_threshold_below_precision_1_precision_test_diag",
        "first_threshold_below_precision_1_recall_test_diag",
        "unique_calibrated_prob_count",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
