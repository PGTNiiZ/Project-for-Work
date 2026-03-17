from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

import run_rebuilt_pipeline as base


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
SUMMARY_PATH = SCRIPT_DIR / "tuning_summary.csv"
BEST_PATH = SCRIPT_DIR / "best_run.json"


TRIALS = [
    {"run_name": "r075_h200_s42", "random_neg_ratio": 0.75, "hard_neg_ratio": 2.0, "seed": 42},
    {"run_name": "r100_h200_s42", "random_neg_ratio": 1.0, "hard_neg_ratio": 2.0, "seed": 42},
    {"run_name": "r125_h200_s42", "random_neg_ratio": 1.25, "hard_neg_ratio": 2.0, "seed": 42},
    {"run_name": "r150_h200_s42", "random_neg_ratio": 1.5, "hard_neg_ratio": 2.0, "seed": 42},
    {"run_name": "r100_h250_s42", "random_neg_ratio": 1.0, "hard_neg_ratio": 2.5, "seed": 42},
    {"run_name": "r075_h250_s42", "random_neg_ratio": 0.75, "hard_neg_ratio": 2.5, "seed": 42},
]


def score_trial(report: dict) -> float:
    metrics = report["metrics"]
    return (
        0.45 * metrics["val_avg_precision"]
        + 0.35 * metrics["best_val_f1"]
        + 0.20 * metrics["val_roc_auc"]
    )


def run_trial(cfg: dict) -> dict:
    run_dir = RUNS_DIR / cfg["run_name"]
    base.ARTIFACTS_DIR = run_dir / "artifacts"
    base.MODELS_DIR = run_dir / "models"
    base.REPORTS_DIR = run_dir / "reports"
    base.ensure_dirs()
    base.set_seed(cfg["seed"])

    profiles = base.load_profiles()
    args = type("Args", (), {
        "random_neg_ratio": cfg["random_neg_ratio"],
        "hard_neg_ratio": cfg["hard_neg_ratio"],
        "seed": cfg["seed"],
    })()
    feature_df, feature_cols, leakage = base.build_dataset(profiles, args)
    metrics = base.train_and_eval(feature_df, feature_cols, seed=cfg["seed"])
    report = {"config": cfg, "metrics": metrics, "leakage_report": leakage}
    (base.REPORTS_DIR / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best_report = None
    best_score = None

    for cfg in TRIALS:
        print("=" * 70)
        print(f"Trial: {cfg['run_name']}")
        report = run_trial(cfg)
        metrics = report["metrics"]
        leakage = report["leakage_report"]
        score = score_trial(report)
        row = {
            "run_name": cfg["run_name"],
            "random_neg_ratio": cfg["random_neg_ratio"],
            "hard_neg_ratio": cfg["hard_neg_ratio"],
            "seed": cfg["seed"],
            "score": score,
            "val_ap": metrics["val_avg_precision"],
            "val_auc": metrics["val_roc_auc"],
            "val_f1": metrics["val_f1"],
            "test_ap": metrics["test_avg_precision"],
            "test_auc": metrics["test_roc_auc"],
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "self_pairs": leakage["self_pairs"],
            "train_val_overlap": leakage["component_overlap"]["train_val"],
            "train_test_overlap": leakage["component_overlap"]["train_test"],
            "val_test_overlap": leakage["component_overlap"]["val_test"],
        }
        rows.append(row)
        print(
            f"val_ap={row['val_ap']:.4f} val_f1={row['val_f1']:.4f} "
            f"test_ap={row['test_ap']:.4f} test_f1={row['test_f1']:.4f}"
        )
        if best_score is None or score > best_score:
            best_score = score
            best_report = report

    summary_df = pd.DataFrame(rows).sort_values(
        ["score", "val_ap", "val_f1", "test_ap", "test_f1"], ascending=False
    ).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    if best_report is None:
        raise RuntimeError("No tuning runs completed.")

    best_cfg = best_report["config"]
    best_payload = {
        "best_run_name": best_cfg["run_name"],
        "selection_score": best_score,
        "config": best_cfg,
        "metrics": best_report["metrics"],
        "leakage_report": best_report["leakage_report"],
    }
    BEST_PATH.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"Best run: {best_cfg['run_name']}")
    print(f"Summary : {SUMMARY_PATH}")
    print(f"Best    : {BEST_PATH}")


if __name__ == "__main__":
    main()
