"""Fair Gradient Boosting comparison for the nested MiniLM experiment.

This runner compares GB-17, GB-18, and GB-18-GA using the exact nested
entity-aware split used by the latest R3 run.  It deliberately keeps test
labels out of model fitting, calibration, GA optimisation, and deployment
selection.  The held-out test labels are fetched only after the deployment
genome has been selected.

The GradientBoostingClassifier settings are the explicit project-standard
configuration in ``Project-for-Work/pub_multi/bench_classical_leaksafe.py``;
no test-driven tuning is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

import exp_r1_ga_redecision as ga
from exp_lib import (
    BLOCKING_MISSED,
    MATCH,
    NO_MATCH,
    REVIEW,
    build_nested_entity_split,
    evaluate,
    nested_split_constants,
    PROFILES,
)
from exp_r2_bert_feature import FEATURE17_COLS, undersample_negatives


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
DEFAULT_OUTPUT = EXP / "automation" / "gb_transformer_primary_20260809"
KEY_COLS = ["profile_id_a", "profile_id_b"]
ROLES = ["model_train", "model_calibration", "ga_validation", "test", "drop"]
DEV_ROLES = ["model_train", "model_calibration", "ga_validation"]
SCORED_SOURCE = "decision_source"
SEED = 42
SPLIT_SEED = 42
TRAIN_NEG_RATIO = 3.0
GA_SEEDS = [7, 42, 123, 999, 2025]
MANUAL_MATCH_THRESHOLD = 0.98
MANUAL_REVIEW_THRESHOLD = 0.95
GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": SEED,
}
GA_WEIGHTS = ga.CostWeights(fp=5.0, fn=1.0, review=0.02)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def environment_snapshot() -> dict[str, Any]:
    import sklearn

    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cwd": str(Path.cwd().resolve()),
        "deterministic_numpy_seed": SEED,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    try:
        import torch

        result.update({
            "pytorch": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
        })
    except ImportError:
        result["pytorch"] = "not installed"
    return result


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_unique_keys(frame: pd.DataFrame, label: str) -> None:
    if frame.duplicated(KEY_COLS).any():
        raise AssertionError(f"{label} contains duplicate pair keys")


def validate_ranges(frame: pd.DataFrame, columns: list[str], lower: float,
                    upper: float, label: str) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    tolerance = 1e-6
    if (values < lower - tolerance).any() or (values > upper + tolerance).any():
        raise ValueError(f"{label} contains values outside [{lower}, {upper}]")


def pair_key_digest(frame: pd.DataFrame) -> str:
    ordered = frame[KEY_COLS].astype("int64").sort_values(KEY_COLS)
    payload = "\n".join(
        f"{int(a)},{int(b)}" for a, b in ordered.itertuples(index=False, name=None)
    ).encode("utf-8")
    return sha256_bytes(payload)


def labels_for(assigned: pd.DataFrame, keys: pd.DataFrame, role: str,
               allow_test: bool = False) -> np.ndarray:
    """Fetch labels for a role; test is hard-blocked until final evaluation."""
    if role == "test" and not allow_test:
        raise AssertionError("test labels requested before final evaluation")
    source = assigned.loc[assigned["experiment_split"] == role,
                          KEY_COLS + ["actual"]]
    joined = keys[KEY_COLS].merge(source, on=KEY_COLS, how="left",
                                  validate="one_to_one", sort=False)
    if joined["actual"].isna().any():
        raise AssertionError(f"label alignment failed for role {role}")
    values = joined["actual"].to_numpy(dtype=np.int8)
    if not np.isin(values, [0, 1]).all():
        raise AssertionError(f"role {role} contains non-binary labels")
    return values


def validate_inputs(cache: pd.DataFrame, feat17: pd.DataFrame,
                    bert_cos: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """Validate pair keys, joins, feature ranges, and development labels."""
    require_columns(cache, KEY_COLS + ["actual", "split", "decision_source", "name_sim"],
                    "scored cache")
    require_columns(feat17, KEY_COLS + FEATURE17_COLS, "17-feature cache")
    require_columns(bert_cos, KEY_COLS + ["bert_cos"], "MiniLM cosine cache")
    for frame, label in ((cache, "scored cache"), (feat17, "17-feature cache"),
                         (bert_cos, "MiniLM cosine cache")):
        assert_unique_keys(frame, label)
        if frame[KEY_COLS].isna().any().any():
            raise ValueError(f"{label} contains missing pair keys")

    scored = assigned.loc[assigned[SCORED_SOURCE] != "AUTO_EXACT",
                          KEY_COLS + ["experiment_split", "name_sim"]].copy()
    if len(scored) != len(feat17) or len(scored) != len(bert_cos):
        raise AssertionError(
            f"row-count mismatch: scored={len(scored)}, feat17={len(feat17)}, "
            f"bert_cos={len(bert_cos)}"
        )

    # Merge validation is also the exact pair-key alignment check.  No labels
    # are carried into this feature frame, so test labels stay sealed here.
    work = scored.merge(feat17, on=KEY_COLS, how="left", validate="one_to_one",
                        suffixes=("_cache", "_features"), sort=False)
    if len(work) != len(scored):
        raise AssertionError("17-feature join changed the scored row count")
    work = work.merge(bert_cos, on=KEY_COLS, how="left", validate="one_to_one", sort=False)
    if len(work) != len(scored):
        raise AssertionError("MiniLM cosine join changed the scored row count")
    merged_feature_columns = [
        "name_sim_features" if column == "name_sim" else column
        for column in FEATURE17_COLS
    ] + ["bert_cos"]
    if work[merged_feature_columns].isna().any().any():
        raise ValueError("feature join contains missing values")
    validate_ranges(work, [
        "name_sim_features" if column == "name_sim" else column
        for column in FEATURE17_COLS
    ], 0.0, 1.0, "joined 17 features")
    validate_ranges(work, ["bert_cos"], -1.0, 1.0, "joined MiniLM cosine")
    validate_ranges(work, ["name_sim_cache", "name_sim_features"], 0.0, 1.0,
                    "name similarity")
    if not np.allclose(work["name_sim_cache"].to_numpy(),
                       work["name_sim_features"].to_numpy(), atol=1e-6):
        raise AssertionError("cached name_sim and feature-17 name_sim are misaligned")
    work["name_sim"] = work["name_sim_features"].astype(np.float32)
    work = work.drop(columns=["name_sim_cache", "name_sim_features"])
    if work[KEY_COLS].duplicated().any():
        raise AssertionError("joined work frame contains duplicate pair keys")

    # Before training, only development labels are inspected.  The test branch
    # is intentionally not included in this validation loop.
    for role in DEV_ROLES:
        role_keys = work.loc[work["experiment_split"] == role, KEY_COLS]
        role_labels = labels_for(assigned, role_keys, role)
        if len(np.unique(role_labels)) != 2:
            raise ValueError(f"{role} must contain both label classes")
    return work


def build_nested_constants_without_test(assigned: pd.DataFrame) -> dict:
    """Use exp_lib's exact harness constants while keeping test labels sealed."""
    development = assigned.loc[assigned["experiment_split"] != "test"].copy()
    return nested_split_constants(development, seed=SPLIT_SEED)


def metric_with_cost(decision: np.ndarray, actual: np.ndarray,
                     constants: dict[str, int], weights: ga.CostWeights) -> dict[str, Any]:
    metric = evaluate(decision, actual, constants)
    metric["cost"] = round(
        weights.fp * (metric["FP"]) + weights.fn * metric["FN"]
        + weights.review * metric["REVIEW"], 2
    )
    return metric


def manual_decision(probabilities: np.ndarray) -> np.ndarray:
    decision = np.full(len(probabilities), NO_MATCH, dtype=np.int8)
    decision[probabilities >= MANUAL_REVIEW_THRESHOLD] = REVIEW
    decision[probabilities >= MANUAL_MATCH_THRESHOLD] = MATCH
    return decision


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(value): int((labels == value).sum()) for value in (0, 1)}


def deterministic_sample_indices(labels: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Return indices using exp_r2's exact deterministic undersampling policy."""
    row_ids = np.arange(len(labels), dtype=np.float32).reshape(-1, 1)
    sampled_ids, sampled_labels = undersample_negatives(row_ids, labels, ratio, seed)
    indices = sampled_ids[:, 0].astype(np.int64)
    if not np.array_equal(sampled_labels.astype(np.int8), labels[indices].astype(np.int8)):
        raise AssertionError("undersampling labels do not match sampled row ids")
    return indices


def train_gb_variant(name: str, feature_columns: list[str], work: pd.DataFrame,
                     assigned: pd.DataFrame, output_dir: Path, model_seed: int,
                     smoke: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one GB variant and calibrate only on model_calibration."""
    train = work.loc[work["experiment_split"] == "model_train"].copy()
    calibration = work.loc[work["experiment_split"] == "model_calibration"].copy()
    y_train_raw = labels_for(assigned, train, "model_train")
    y_calibration = labels_for(assigned, calibration, "model_calibration")
    sample_indices = deterministic_sample_indices(y_train_raw, TRAIN_NEG_RATIO, model_seed)
    X_train_raw = train[feature_columns].to_numpy(dtype=np.float32)
    X_calibration = calibration[feature_columns].to_numpy(dtype=np.float32)
    X_train = X_train_raw[sample_indices]
    y_train = y_train_raw[sample_indices]

    params = dict(GB_PARAMS)
    if smoke:
        params["n_estimators"] = 5
    model = GradientBoostingClassifier(**params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    raw_calibration = model.predict_proba(X_calibration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calibration, y_calibration)
    calibrated_calibration = calibrator.predict(raw_calibration)
    calibration_ap = float(average_precision_score(y_calibration, calibrated_calibration))

    X_all = work[feature_columns].to_numpy(dtype=np.float32)
    raw_all = model.predict_proba(X_all)[:, 1]
    probabilities = calibrator.predict(raw_all).astype(np.float32)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{name} probabilities are not finite values in [0, 1]")

    artifact_stem = "gb17" if name == "GB-17" else "gb18"
    model_path = output_dir / "model" / f"{artifact_stem}_model.pkl"
    calibrator_path = output_dir / "model" / f"{artifact_stem}_calibrator.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with calibrator_path.open("wb") as handle:
        pickle.dump(calibrator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    probability_name = "gb17" if name == "GB-17" else "gb18"
    probability_path = output_dir / f"{probability_name}_probabilities.parquet"
    probability_frame = work[KEY_COLS + ["experiment_split"]].copy()
    probability_frame[f"probability_{probability_name}"] = probabilities
    probability_frame.to_parquet(probability_path, index=False)

    train_key_hash = pair_key_digest(train.iloc[sample_indices])
    metadata = {
        "configuration": name,
        "feature_columns": feature_columns,
        "model": "sklearn.ensemble.GradientBoostingClassifier",
        "hyperparameters": params,
        "project_standard_source": "Project-for-Work/pub_multi/bench_classical_leaksafe.py",
        "model_seed": int(model_seed),
        "calibration": "sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')",
        "train_neg_ratio": TRAIN_NEG_RATIO,
        "class_counts_before_undersampling": class_counts(y_train_raw),
        "class_counts_after_undersampling": class_counts(y_train),
        "n_model_train_before_undersampling": int(len(y_train_raw)),
        "n_model_train_after_undersampling": int(len(y_train)),
        "n_model_calibration": int(len(y_calibration)),
        "sampled_model_train_pair_key_sha256": train_key_hash,
        "calibration_average_precision": calibration_ap,
        "fit_seconds": float(fit_seconds),
        "probability_artifact": probability_path.name,
        "model_artifact": str(model_path.relative_to(output_dir)),
        "calibrator_artifact": str(calibrator_path.relative_to(output_dir)),
        "label_access": {
            "model_fit": "model_train only",
            "probability_calibration": "model_calibration only",
            "ga_optimisation": "ga_validation only after model fitting",
            "held_out_test": "not accessed by this training function",
        },
    }
    return probabilities, metadata


def probability_artifact_check(path: Path, work: pd.DataFrame,
                               column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    require_columns(frame, KEY_COLS + ["experiment_split", column], path.name)
    assert_unique_keys(frame, path.name)
    if len(frame) != len(work):
        raise AssertionError(f"{path.name} has {len(frame)} rows; expected {len(work)}")
    expected = work[KEY_COLS + ["experiment_split"]].reset_index(drop=True)
    observed = frame[KEY_COLS + ["experiment_split"]].reset_index(drop=True)
    if not expected.equals(observed):
        raise AssertionError(f"{path.name} pair order/split manifest differs from work frame")
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError(f"{path.name} contains invalid probabilities")
    return frame


def run_ga_trials(work: pd.DataFrame, assigned: pd.DataFrame,
                  probability_frame: pd.DataFrame, constants: dict,
                  ga_seeds: list[int], output_dir: Path,
                  smoke: bool = False) -> tuple[list[dict], list[dict], dict]:
    validation = work.loc[work["experiment_split"] == "ga_validation"].copy()
    labels = labels_for(assigned, validation, "ga_validation")
    probabilities = probability_frame.loc[validation.index, "probability_gb18"].to_numpy(dtype=float)
    nsim = validation["name_sim"].to_numpy(dtype=float)
    ga_configs = {
        "population_size": 10 if smoke else 60,
        "generations": 2 if smoke else 40,
        "elite_size": 3 if smoke else 12,
        "mutation_probability": 0.3,
        "mutation_sigma": 0.02,
    }
    trials: list[dict] = []
    generations: list[dict] = []
    for seed in ga_seeds:
        config = ga.GAConfig(
            seed=int(seed),
            population_size=ga_configs["population_size"],
            generations=ga_configs["generations"],
            elite_size=ga_configs["elite_size"],
            mutation_probability=ga_configs["mutation_probability"],
            mutation_sigma=ga_configs["mutation_sigma"],
            weights=GA_WEIGHTS,
        ).validate()
        started = time.perf_counter()
        best, history = ga.run_ga(probabilities, nsim, labels, config=config, verbose=False)
        best_code = ga.decide_code(probabilities, nsim, best)
        validation_metrics = metric_with_cost(best_code, labels, constants["ga_validation"], GA_WEIGHTS)
        objective_cost = ga.raw_cost(best_code, labels, GA_WEIGHTS)
        trial_runtime = time.perf_counter() - started
        trial = {
            "experiment": "GB-18-GA",
            "seed": int(seed),
            "t_m": float(best[0]),
            "t_r": float(best[1]),
            "c_promote": float(best[2]),
            "c_demote": float(best[3]),
            "validation_TP": validation_metrics["TP"],
            "validation_FP": validation_metrics["FP"],
            "validation_FN": validation_metrics["FN"],
            "validation_REVIEW": validation_metrics["REVIEW"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_F1": validation_metrics["F1"],
            "val_cost": validation_metrics["cost"],
            "objective_val_cost": round(float(objective_cost), 2),
            "runtime_seconds": round(float(trial_runtime), 6),
            "test_labels_accessed": False,
        }
        trials.append(trial)
        for entry in history:
            genome = np.asarray(entry["genome"], dtype=float)
            generation_code = ga.decide_code(probabilities, nsim, genome)
            generation_metrics = metric_with_cost(
                generation_code, labels, constants["ga_validation"], GA_WEIGHTS)
            generations.append({
                "experiment": "GB-18-GA",
                "seed": int(seed),
                "generation": int(entry["gen"]),
                "objective_val_cost": float(entry["cost"]),
                "val_cost": generation_metrics["cost"],
                "t_m": float(genome[0]),
                "t_r": float(genome[1]),
                "c_promote": float(genome[2]),
                "c_demote": float(genome[3]),
            })

    selected = min(trials, key=lambda row: (float(row["val_cost"]), int(row["seed"])))
    numeric_trial_columns = [
        "val_cost", "objective_val_cost", "t_m", "t_r", "c_promote", "c_demote",
        "validation_TP", "validation_FP", "validation_FN", "validation_REVIEW",
        "validation_precision", "validation_recall", "validation_F1",
    ]
    stats = {}
    for column in numeric_trial_columns:
        values = np.asarray([float(row[column]) for row in trials], dtype=float)
        stats[column] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    selection = {
        "policy": "minimum ga_validation weighted cost; ties resolved by smallest GA seed",
        "selected_seed": int(selected["seed"]),
        "selected_genome": {key: float(selected[key]) for key in
                            ("t_m", "t_r", "c_promote", "c_demote")},
        "selected_validation_cost": float(selected["val_cost"]),
        "seed_statistics": stats,
        "test_evaluations_during_selection": 0,
    }
    return trials, generations, selection


def metric_row(label: str, source: str, metric: dict[str, Any],
               test_eval_mode: str) -> dict[str, Any]:
    return {
        "experiment": label,
        "score_model": source,
        "TP": metric["TP"],
        "FP": metric["FP"],
        "FN": metric["FN"],
        "REVIEW": metric["REVIEW"],
        "precision": metric["precision"],
        "recall": metric["recall"],
        "F1": metric["F1"],
        "cost": metric["cost"],
        "test_eval_mode": test_eval_mode,
    }


def load_comparison_sources() -> tuple[dict, dict, dict]:
    r0_path = EXP / "automation" / "r1_nested_primary_20260809" / "summary.json"
    r3_path = EXP / "automation" / "r3_primary_20260809" / "summary.json"
    if not r0_path.exists() or not r3_path.exists():
        raise FileNotFoundError("R0/R3 comparison artifacts are missing")
    r0_summary = json.loads(r0_path.read_text(encoding="utf-8"))
    r3_summary = json.loads(r3_path.read_text(encoding="utf-8"))
    r0 = r0_summary["baselines"]["A_current"]
    r3_metrics = r3_summary["aggregates"]["A_current"]["metrics"]
    r3 = {
        "TP": r3_metrics["test_TP"]["mean"],
        "FP": r3_metrics["test_FP"]["mean"],
        "FN": r3_metrics["test_FN"]["mean"],
        "REVIEW": r3_metrics["test_REVIEW"]["mean"],
        "precision": r3_metrics["test_precision"]["mean"],
        "recall": r3_metrics["test_recall"]["mean"],
        "F1": r3_metrics["test_F1"]["mean"],
        "cost": r3_metrics["test_cost"]["mean"],
    }
    source_hashes = {
        "r0_summary": {"path": str(r0_path.resolve()), "sha256": sha256_file(r0_path)},
        "r3_summary": {"path": str(r3_path.resolve()), "sha256": sha256_file(r3_path)},
    }
    return r0, r3, source_hashes


def delta_effect(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ["TP", "FP", "FN", "REVIEW", "precision", "recall", "F1", "cost"]
    result = {field: float(right[field]) - float(left[field]) for field in fields}
    result["cost_reduction"] = float(left["cost"]) - float(right["cost"])
    return result


def build_summary(output_dir: Path, test_rows: list[dict], trials: list[dict],
                  selection: dict, training_meta: dict, split_manifest: dict,
                  input_hashes: dict, runtime_seconds: float) -> dict:
    r0, r3, comparison_sources = load_comparison_sources()
    rows_by_label = {row["experiment"]: row for row in test_rows}
    comparison = [
        metric_row("R0", "Original production baseline", r0, "existing sealed R0 artifact"),
        metric_row("R3", "MiniLM + IdentityMLP + GA", r3, "mean over existing R3 GA seeds"),
        metric_row("GB-17", "Gradient Boosting + 17 handcrafted features", rows_by_label["GB-17"],
                   "single final test evaluation"),
        metric_row("GB-18", "Gradient Boosting + 17 features + MiniLM cosine", rows_by_label["GB-18"],
                   "single final test evaluation"),
        metric_row("GB-18-GA", "Same calibrated GB-18 probabilities + GA rule",
                   rows_by_label["GB-18-GA"], "selected genome evaluated once after selection"),
    ]
    by_name = {row["experiment"]: row for row in comparison}
    effects = {
        "GB-17_to_GB-18_MiniLM_effect": delta_effect(by_name["GB-17"], by_name["GB-18"]),
        "GB-18_to_GB-18-GA_decision_layer_effect": delta_effect(by_name["GB-18"], by_name["GB-18-GA"]),
        "R3_to_GB-18-GA_model_family_effect": delta_effect(by_name["R3"], by_name["GB-18-GA"]),
    }
    winner = min((row for row in comparison if row["experiment"] in {"R0", "R3", "GB-17", "GB-18", "GB-18-GA"}),
                 key=lambda row: (float(row["cost"]), -float(row["F1"])))
    return {
        "run_id": output_dir.name,
        "experiment": "GB-17 vs GB-18 vs GB-18-GA",
        "status": "completed",
        "selection_policy": "No test labels were used for fitting, calibration, threshold choice, GA optimisation, or seed/genome selection.",
        "test_evaluation_policy": "Test labels were fetched once after GA selection; GB-18-GA was evaluated exactly once on test.",
        "comparison": comparison,
        "isolated_effects": effects,
        "winner_by_lowest_weighted_cost": winner["experiment"],
        "selected_ga": selection,
        "ga_trials": trials,
        "training": training_meta,
        "split_manifest": split_manifest,
        "input_artifacts": input_hashes,
        "comparison_artifacts": comparison_sources,
        "runtime_seconds": float(runtime_seconds),
        "verification": {
            "gb17_gb18_same_sampled_train_pair_hash": training_meta["GB-17"]["sampled_model_train_pair_key_sha256"] == training_meta["GB-18"]["sampled_model_train_pair_key_sha256"],
            "gb17_gb18_same_labels": True,
            "gb17_gb18_same_hyperparameters": training_meta["GB-17"]["hyperparameters"] == training_meta["GB-18"]["hyperparameters"],
            "gb18_and_ga_probability_artifact_same": True,
            "test_labels_accessed_before_final_evaluation": False,
            "selected_ga_test_evaluations": 1,
            "drop_pairs_used_for_training_or_evaluation": False,
        },
    }


def write_summary_markdown(path: Path, summary: dict) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# GB-17 / GB-18 / GB-18-GA Experiment",
        "",
        "Nested entity-aware split seed 42; outer test remained sealed until final evaluation.",
        "GB configuration: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42).",
        "",
        "## Test comparison",
        "",
        "| Experiment | TP | FP | FN | REVIEW | Precision | Recall | F1 | Weighted cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append("| {experiment} | {TP} | {FP} | {FN} | {REVIEW} | {precision} | {recall} | {F1} | {cost} |".format(
            **{key: fmt(row[key]) for key in ("experiment", "TP", "FP", "FN", "REVIEW", "precision", "recall", "F1", "cost")}))
    lines.extend([
        "",
        f"Winner by weighted cost: **{summary['winner_by_lowest_weighted_cost']}**.",
        "",
        "## Isolated effects (right minus left; cost reduction is positive when cost falls)",
        "",
    ])
    for name, effect in summary["isolated_effects"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(", ".join(f"{field}={fmt(effect[field])}" for field in
                               ("TP", "FP", "FN", "REVIEW", "precision", "recall", "F1", "cost", "cost_reduction")))
        lines.append("")
    lines.extend([
        "## GA validation seeds",
        "",
        "Test metrics were not calculated per GA seed. The selected seed/genome was chosen from ga_validation weighted cost only, then evaluated on test once.",
        "",
        "| Seed | Validation cost | t_m | t_r | c_promote | c_demote |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["ga_trials"]:
        lines.append(f"| {row['seed']} | {row['val_cost']:.2f} | {row['t_m']:.6f} | {row['t_r']:.6f} | {row['c_promote']:.6f} | {row['c_demote']:.6f} |")
    selected = summary["selected_ga"]
    lines.extend([
        "",
        f"Selected seed: {selected['selected_seed']}; selected validation cost: {selected['selected_validation_cost']:.2f}.",
        f"Selected genome: `{json.dumps(selected['selected_genome'], sort_keys=True)}`.",
        "",
        "## Verification",
        "",
        "- GB-17 and GB-18 share the same sampled training pair hash, labels, model seed, and hyperparameters.",
        "- GB-18 manual and GB-18-GA use the same `gb18_probabilities.parquet` artifact.",
        "- Test labels were not accessed before final selection; drop pairs were excluded.",
        "- See `artifact_hashes.json` for SHA-256 hashes.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run(args: argparse.Namespace) -> dict:
    set_seed(SEED)
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    cache_path = EXP / "scored_pairs_enriched.parquet"
    feat17_path = EXP / "pair_features17.parquet"
    bert_path = EXP / "pair_bert_cos.parquet"
    input_paths = {
        "scored_pairs_enriched": cache_path,
        "pair_features17": feat17_path,
        "pair_bert_cos": bert_path,
        "blocking_missed_pairs": BLOCKING_MISSED,
        "profiles": PROFILES,
        "runner_source": Path(__file__).resolve(),
        "exp_lib_source": ROOT / "exp_lib.py",
        "ga_source": ROOT / "exp_r1_ga_redecision.py",
        "r2_source": ROOT / "exp_r2_bert_feature.py",
        "gb_benchmark_source": ROOT / "Project-for-Work" / "pub_multi" / "bench_classical_leaksafe.py",
        "r0_nested_summary": EXP / "automation" / "r1_nested_primary_20260809" / "summary.json",
        "r3_summary": EXP / "automation" / "r3_primary_20260809" / "summary.json",
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    input_hashes = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
               "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    write_json(output_dir / "environment.json", environment_snapshot())

    cache = pd.read_parquet(cache_path)
    feat17 = pd.read_parquet(feat17_path)
    bert_cos = pd.read_parquet(bert_path)
    assigned, split_manifest = build_nested_entity_split(cache, seed=SPLIT_SEED)
    split_manifest["source_inputs"] = input_hashes
    split_manifest["scored_feature_row_count"] = int(len(feat17))
    split_manifest["feature_columns"] = FEATURE17_COLS + ["bert_cos"]
    split_manifest["roles"] = ROLES
    split_manifest["test_sealed_until"] = "final_test_evaluation"
    work = validate_inputs(cache, feat17, bert_cos, assigned)
    if args.smoke:
        # Keep the same assigned roles while making a small, deterministic code-path test.
        frames = []
        for role in ["model_train", "model_calibration", "ga_validation", "test"]:
            role_frame = work.loc[work["experiment_split"] == role]
            n = min(len(role_frame), int(args.smoke_rows))
            frames.append(role_frame.sample(n=n, random_state=SEED).sort_index())
        work = pd.concat(frames, ignore_index=True)
        split_manifest["smoke"] = True
        split_manifest["smoke_rows_per_role"] = int(args.smoke_rows)
    else:
        split_manifest["smoke"] = False
    write_json(output_dir / "split_manifest.json", split_manifest)

    # Constants for GA validation are computed from a development-only frame.
    # This calls exp_lib's harness without reading any test labels.
    development_constants = build_nested_constants_without_test(assigned)
    training_meta: dict[str, Any] = {}
    model17_features = FEATURE17_COLS.copy()
    model18_features = FEATURE17_COLS + ["bert_cos"]
    prob17, training_meta["GB-17"] = train_gb_variant(
        "GB-17", model17_features, work, assigned, output_dir, SEED, args.smoke)
    prob18, training_meta["GB-18"] = train_gb_variant(
        "GB-18", model18_features, work, assigned, output_dir, SEED, args.smoke)
    if training_meta["GB-17"]["sampled_model_train_pair_key_sha256"] != training_meta["GB-18"]["sampled_model_train_pair_key_sha256"]:
        raise AssertionError("GB-17 and GB-18 did not use identical sampled model_train rows")
    if training_meta["GB-17"]["class_counts_before_undersampling"] != training_meta["GB-18"]["class_counts_before_undersampling"]:
        raise AssertionError("GB-17 and GB-18 did not use identical pre-sampling labels")

    prob17_frame = probability_artifact_check(output_dir / "gb17_probabilities.parquet", work, "probability_gb17")
    prob18_frame = probability_artifact_check(output_dir / "gb18_probabilities.parquet", work, "probability_gb18")
    if not np.array_equal(prob18, prob18_frame["probability_gb18"].to_numpy(dtype=np.float32)):
        raise AssertionError("GB-18 probability artifact changed after writing")
    trials, generations, selection = run_ga_trials(
        work, assigned, prob18_frame, development_constants, GA_SEEDS, output_dir, args.smoke)
    write_csv(output_dir / "trials.csv", trials)
    write_csv(output_dir / "ga_generations.csv", generations)

    # No test labels have been accessed to this point.  Select the genome from
    # validation cost, then evaluate all three final configurations together.
    test_mask = work["experiment_split"].to_numpy() == "test"
    test_work = work.loc[test_mask].copy()
    selected_genome = np.asarray([selection["selected_genome"][key]
                                  for key in ("t_m", "t_r", "c_promote", "c_demote")], dtype=float)
    test_code17 = manual_decision(prob17[test_mask])
    test_code18 = manual_decision(prob18_frame.loc[test_mask, "probability_gb18"].to_numpy(dtype=float))
    test_code18_ga = ga.decide_code(
        prob18_frame.loc[test_mask, "probability_gb18"].to_numpy(dtype=float),
        test_work["name_sim"].to_numpy(dtype=float), selected_genome,
    )
    # This is the single point at which the sealed test labels are opened.
    test_actual = labels_for(assigned, test_work, "test", allow_test=True)
    full_constants = nested_split_constants(assigned, seed=SPLIT_SEED)
    for role in DEV_ROLES:
        for key in ("tp_exact", "fp_exact", "fn_blocking", "total_pos", "n_scored"):
            if development_constants[role][key] != full_constants[role][key]:
                raise AssertionError(f"development/test-deferred constants changed for {role}/{key}")
    final_specs = [
        ("GB-17", "manual_thresholds_0.98_0.95", test_code17, training_meta["GB-17"]["calibration_average_precision"], "gb17_probabilities.parquet"),
        ("GB-18", "manual_thresholds_0.98_0.95", test_code18, training_meta["GB-18"]["calibration_average_precision"], "gb18_probabilities.parquet"),
        ("GB-18-GA", "ga_selected_validation_cost", test_code18_ga, training_meta["GB-18"]["calibration_average_precision"], "gb18_probabilities.parquet"),
    ]
    test_results: list[dict] = []
    for label, rule, decision, calibration_ap, probability_artifact in final_specs:
        metrics = metric_with_cost(decision, test_actual, full_constants["test"], GA_WEIGHTS)
        test_results.append({
            "experiment": label,
            "decision_rule": rule,
            "probability_artifact": probability_artifact,
            "calibration_average_precision": float(calibration_ap),
            "selected_ga_seed": selection["selected_seed"] if label == "GB-18-GA" else "",
            **metrics,
        })
    write_csv(output_dir / "test_results.csv", test_results)

    training_meta["shared_fairness_assertions"] = {
        "same_model_train_rows": True,
        "same_sampled_rows": True,
        "same_labels": True,
        "same_hyperparameters": True,
        "same_model_seed": True,
        "only_feature_difference": "bert_cos feature 18",
        "calibrators_are_separate": True,
    }
    training_meta["role_counts_scored_pairs"] = {
        role: int((work["experiment_split"] == role).sum())
        for role in ROLES
    }
    write_json(output_dir / "training_metadata.json", training_meta)

    runtime_seconds = time.perf_counter() - started
    summary = build_summary(
        output_dir, test_results, trials, selection, training_meta,
        split_manifest, input_hashes, runtime_seconds,
    )
    write_json(output_dir / "summary.json", summary)
    write_summary_markdown(output_dir / "summary.md", summary)
    config = {
        "run_id": output_dir.name,
        "experiment": "GB-17 vs GB-18 vs GB-18-GA",
        "smoke": bool(args.smoke),
        "split": {
            "version": "entity_nested_v1",
            "seed": SPLIT_SEED,
            "roles": ROLES,
            "outer_test_sealed": True,
            "cross_entity_or_cross_split_pairs": "drop and excluded",
        },
        "features": {
            "gb17": model17_features,
            "gb18": model18_features,
            "feature_18": "bert_cos from cached pair_bert_cos.parquet",
        },
        "gb_configuration": GB_PARAMS if not args.smoke else {**GB_PARAMS, "n_estimators": 5},
        "training": {
            "model_seed": SEED,
            "train_neg_ratio": TRAIN_NEG_RATIO,
            "calibration": "isotonic on model_calibration",
            "hyperparameter_selection": "none; project-standard benchmark configuration",
        },
        "manual_thresholds": {"MATCH": MANUAL_MATCH_THRESHOLD, "REVIEW": MANUAL_REVIEW_THRESHOLD},
        "ga": {
            "implementation": "exp_r1_ga_redecision.py",
            "seeds": GA_SEEDS,
            "population": 10 if args.smoke else 60,
            "generations": 2 if args.smoke else 40,
            "elite": 3 if args.smoke else 12,
            "mutation_probability": 0.3,
            "mutation_sigma": 0.02,
            "weights": GA_WEIGHTS.as_dict(),
            "selection": selection,
        },
        "input_artifacts": input_hashes,
        "output_artifacts": {
            "probabilities": ["gb17_probabilities.parquet", "gb18_probabilities.parquet"],
            "models": ["model/gb17_model.pkl", "model/gb17_calibrator.pkl", "model/gb18_model.pkl", "model/gb18_calibrator.pkl"],
        },
        "label_access": {
            "before_final_test_evaluation": ["model_train", "model_calibration", "ga_validation"],
            "test": "fetched once after selected genome; no test metric used for selection",
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "config.json", config)
    generated_files = [
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    ]
    artifact_hashes = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(generated_files)
    }
    write_json(output_dir / "artifact_hashes.json", {
        "hash_algorithm": "SHA-256",
        "files": artifact_hashes,
    })
    (output_dir / "status.json").write_text(json.dumps({
        "status": "completed",
        "updated_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": runtime_seconds,
    }, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true",
                        help="run a small deterministic path test with reduced GB/GA settings")
    parser.add_argument("--smoke-rows", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(json.dumps({
        "status": result["status"],
        "winner": result["winner_by_lowest_weighted_cost"],
        "output_dir": str(Path(arguments.output_dir).resolve()),
    }, indent=2))
