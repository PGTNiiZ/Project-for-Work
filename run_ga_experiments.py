"""Reproducible, validation-only GA threshold experiment runner.

Examples
--------
    .\\.venv\\Scripts\\python.exe run_ga_experiments.py --experiment r1 --seeds 7 42 123 999 2025
    .\\.venv\\Scripts\\python.exe run_ga_experiments.py --experiment r1-sensitivity --seeds 7 42 123 999 2025
    .\\.venv\\Scripts\\python.exe run_ga_experiments.py --experiment r3 --seeds 7 42 123 999 2025
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

import exp_r1_ga_redecision as ga
from exp_lib import (CACHE, DEC_CODE, EXP, MATCH, NO_MATCH, REVIEW,
                     NSIM_MIN_SCORE, build_cache, evaluate, split_constants)


AUTOMATION_ROOT = EXP / "automation"
R2_PROBABILITIES = EXP / "r2_probabilities.parquet"
R2_FEATURES = EXP / "pair_features17.parquet"
R2_BERT_COS = EXP / "pair_bert_cos.parquet"
R2_MATCH_T, R2_REVIEW_T = 0.98, 0.95
METRIC_COLUMNS = ("TP", "FP", "FN", "REVIEW", "precision", "recall", "F1", "cost")
SUMMARY_METRICS = ("test_precision", "test_recall", "test_F1", "test_cost",
                   "test_TP", "test_FP", "test_FN", "test_REVIEW")


@dataclass
class ExperimentData:
    """Scored pairs and rule metadata for a single representation/decision experiment."""

    experiment: str
    scored: pd.DataFrame
    score_column: str
    baseline_label: str
    baseline_decider: Callable[[pd.DataFrame], np.ndarray]
    optimise_score_floor: Optional[float]
    input_manifest: dict


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_manifest(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _r1_baseline(sub: pd.DataFrame) -> np.ndarray:
    codes = sub["decision"].map(DEC_CODE)
    if codes.isna().any():
        raise ValueError("R0 baseline contains an unknown decision code")
    return codes.to_numpy(dtype=np.int8)


def _r2_baseline(sub: pd.DataFrame) -> np.ndarray:
    prob = sub["probability_r2"].to_numpy()
    codes = np.full(len(sub), NO_MATCH, dtype=np.int8)
    codes[prob >= R2_REVIEW_T] = REVIEW
    codes[prob >= R2_MATCH_T] = MATCH
    return codes


def make_r1_data(cache: pd.DataFrame, input_manifest: Optional[dict] = None) -> ExperimentData:
    _require_columns(cache, ("decision_source", "split", "score", "name_sim", "actual", "decision"), "cache")
    scored = cache.loc[cache["decision_source"] != "AUTO_EXACT"].copy()
    return ExperimentData(
        experiment="r1",
        scored=scored,
        score_column="score",
        baseline_label="R0_production",
        baseline_decider=_r1_baseline,
        optimise_score_floor=NSIM_MIN_SCORE,
        input_manifest=input_manifest or {},
    )


def make_r3_data(cache: pd.DataFrame, probability_path: Path,
                 input_manifest: Optional[dict] = None) -> ExperimentData:
    _require_columns(cache, ("profile_id_a", "profile_id_b", "decision_source", "split",
                             "name_sim", "actual"), "cache")
    probability_path = Path(probability_path)
    if not probability_path.exists():
        raise FileNotFoundError(f"R3 requires the existing BERT probability artifact: {probability_path}")
    probs = pd.read_parquet(probability_path)
    _require_columns(probs, ("profile_id_a", "profile_id_b", "probability_r2"), "R2 probabilities")
    if probs.duplicated(["profile_id_a", "profile_id_b"]).any():
        raise ValueError("R2 probabilities contain duplicate pair keys")

    scored = cache.loc[cache["decision_source"] != "AUTO_EXACT"].copy()
    scored = scored.merge(probs, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one")
    if len(scored) == 0 or scored["probability_r2"].isna().any():
        raise ValueError("R2 probability join is incomplete; R3 will not retrain or fill values")
    return ExperimentData(
        experiment="r3",
        scored=scored,
        score_column="probability_r2",
        baseline_label="R2_manual",
        baseline_decider=_r2_baseline,
        optimise_score_floor=None,
        input_manifest=input_manifest or {},
    )


def train_fresh_r2_for_run(cache: pd.DataFrame, output_dir: Path, model_seed: int,
                           model_epochs: Optional[int], model_patience: Optional[int]) -> tuple[Path, dict, dict]:
    """Create run-scoped R2 artifacts from existing feature caches; never overwrite global R2 files."""
    missing = [path for path in (R2_FEATURES, R2_BERT_COS) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"fresh R2 training requires existing feature caches: {joined}")
    # Import torch/R2 code only for the explicit training path; R1 stays lightweight.
    import exp_r2_bert_feature as r2

    defaults = r2.R2TrainingConfig()
    config = r2.R2TrainingConfig(
        seed=model_seed,
        epochs=model_epochs if model_epochs is not None else defaults.epochs,
        patience=model_patience if model_patience is not None else defaults.patience,
    ).validate()
    print("Training a fresh validation-only R2 model ...")
    probabilities, metadata = r2.train_r2_probabilities(
        cache,
        pd.read_parquet(R2_FEATURES),
        pd.read_parquet(R2_BERT_COS),
        output_dir,
        training_config=config,
        logger=print,
    )
    probability_path = output_dir / "r2_probabilities.parquet"
    if len(probabilities) == 0 or not probability_path.exists():
        raise RuntimeError("fresh R2 training did not produce probability output")
    manifest = {
        "pair_features17": _artifact_manifest(R2_FEATURES),
        "pair_bert_cos": _artifact_manifest(R2_BERT_COS),
        "fresh_r2_probabilities": _artifact_manifest(probability_path),
        "fresh_r2_training_metadata": _artifact_manifest(output_dir / "r2_training.json"),
    }
    return probability_path, metadata, manifest


def _validate_experiment_data(data: ExperimentData, consts: dict) -> None:
    _require_columns(data.scored, ("split", "actual", "name_sim", data.score_column), data.experiment)
    present = set(data.scored["split"].unique())
    required = {"val", "test"}
    if not required.issubset(present):
        raise ValueError(f"{data.experiment} needs both validation and test pairs; found {sorted(present)}")
    if not required.issubset(consts):
        raise ValueError("split constants must contain validation and test")


def _metric_with_cost(code: np.ndarray, actual: np.ndarray, consts: dict,
                      weights: ga.CostWeights) -> dict:
    metrics = evaluate(code, actual.astype(np.int8), consts)
    metrics["cost"] = round(float(ga.cost_from_counts(metrics, weights)), 6)
    return metrics


def _to_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_to_json_value), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_config(seed: int, settings: dict, weights: ga.CostWeights) -> ga.GAConfig:
    default = ga.default_config()
    return ga.GAConfig(
        seed=int(seed),
        population_size=int(settings.get("population_size") or default.population_size),
        generations=int(settings.get("generations") or default.generations),
        elite_size=int(settings.get("elite_size") or default.elite_size),
        mutation_probability=float(settings.get("mutation_probability")
                                   if settings.get("mutation_probability") is not None
                                   else default.mutation_probability),
        mutation_sigma=float(settings.get("mutation_sigma")
                             if settings.get("mutation_sigma") is not None
                             else default.mutation_sigma),
        weights=weights,
    ).validate()


def _scenario_specs(experiment: str, override_weights: Optional[ga.CostWeights]) -> list[tuple[str, ga.CostWeights]]:
    if experiment == "r1-sensitivity":
        if override_weights is not None:
            raise ValueError("weight overrides are not valid for r1-sensitivity; it always runs A/B/C")
        return [
            ("A_current", ga.CostWeights(5.0, 1.0, 0.02)),
            ("B_fp10", ga.CostWeights(10.0, 1.0, 0.02)),
            ("C_fn2", ga.CostWeights(5.0, 2.0, 0.02)),
        ]
    return [("custom" if override_weights is not None else "A_current",
             override_weights or ga.default_weights())]


def _percent_change(baseline: float, candidate: float):
    if baseline == 0:
        return None
    return round((candidate - baseline) / baseline * 100.0, 6)


def _format_pct(value) -> str:
    return "N/A" if value is None else f"{value:+.3f}%"


def _full_validation_cost(data: ExperimentData, consts: dict, genome: np.ndarray,
                          weights: ga.CostWeights) -> float:
    val = data.scored.loc[data.scored["split"] == "val"]
    code = ga.decide_code(val[data.score_column].to_numpy(), val["name_sim"].to_numpy(), genome)
    return _metric_with_cost(code, val["actual"].to_numpy(), consts["val"], weights)["cost"]


def _run_trial(data: ExperimentData, consts: dict, scenario: str, config: ga.GAConfig) -> tuple[dict, list[dict]]:
    """Select a GA genome using validation only, then evaluate it on held-out test."""
    val = data.scored.loc[data.scored["split"] == "val"]
    test = data.scored.loc[data.scored["split"] == "test"]
    if data.optimise_score_floor is None:
        ga_val = val
        const_fn_below = 0
    else:
        ga_val = val.loc[val[data.score_column] >= data.optimise_score_floor]
        const_fn_below = int(((val[data.score_column] < data.optimise_score_floor)
                              & (val["actual"] == 1)).sum())
    if ga_val.empty:
        raise ValueError("validation data contains no pairs in the GA search range")

    started = time.perf_counter()
    best, history = ga.run_ga(
        ga_val[data.score_column].to_numpy(),
        ga_val["name_sim"].to_numpy(),
        ga_val["actual"].to_numpy(dtype=np.int8),
        const_fn_below=const_fn_below,
        config=config,
        verbose=False,
    )
    # The test frame is first referenced only after the selected genome exists.
    test_code = ga.decide_code(test[data.score_column].to_numpy(), test["name_sim"].to_numpy(), best)
    test_metrics = _metric_with_cost(test_code, test["actual"].to_numpy(), consts["test"], config.weights)
    runtime_seconds = time.perf_counter() - started

    objective_cost = ga.raw_cost(
        ga.decide_code(ga_val[data.score_column].to_numpy(), ga_val["name_sim"].to_numpy(), best),
        ga_val["actual"].to_numpy(dtype=np.int8), config.weights,
    ) + config.weights.fn * const_fn_below
    record = {
        "experiment": data.experiment,
        "scenario": scenario,
        "seed": config.seed,
        "t_m": float(best[0]),
        "t_r": float(best[1]),
        "c_promote": float(best[2]),
        "c_demote": float(best[3]),
        "val_cost": _full_validation_cost(data, consts, best, config.weights),
        "objective_val_cost": float(objective_cost),
        "test_TP": test_metrics["TP"],
        "test_FP": test_metrics["FP"],
        "test_FN": test_metrics["FN"],
        "test_REVIEW": test_metrics["REVIEW"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_F1": test_metrics["F1"],
        "test_cost": test_metrics["cost"],
        "runtime_seconds": round(runtime_seconds, 6),
    }
    generation_rows = []
    for entry in history:
        genome = np.asarray(entry["genome"], dtype=float)
        generation_rows.append({
            "experiment": data.experiment,
            "scenario": scenario,
            "seed": config.seed,
            "generation": entry["gen"],
            "objective_val_cost": entry["cost"],
            "val_cost": _full_validation_cost(data, consts, genome, config.weights),
            "t_m": float(genome[0]),
            "t_r": float(genome[1]),
            "c_promote": float(genome[2]),
            "c_demote": float(genome[3]),
        })
    return record, generation_rows


def _baseline_metrics(data: ExperimentData, consts: dict, weights: ga.CostWeights) -> dict:
    test = data.scored.loc[data.scored["split"] == "test"]
    return _metric_with_cost(data.baseline_decider(test), test["actual"].to_numpy(), consts["test"], weights)


def _aggregate(records: list[dict], baseline: dict) -> dict:
    by_metric = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in records], dtype=float)
        by_metric[metric] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    selected = min(records, key=lambda row: (row["val_cost"], row["seed"]))
    delta = {}
    baseline_aliases = {"test_TP": "TP", "test_FP": "FP", "test_FN": "FN", "test_REVIEW": "REVIEW",
                        "test_precision": "precision", "test_recall": "recall", "test_F1": "F1",
                        "test_cost": "cost"}
    for trial in records:
        delta[str(trial["seed"])] = {
            metric: _percent_change(float(baseline[baseline_aliases[metric]]), float(trial[metric]))
            for metric in SUMMARY_METRICS
        }
    mean_delta = {
        metric: _percent_change(float(baseline[baseline_aliases[metric]]), by_metric[metric]["mean"])
        for metric in SUMMARY_METRICS
    }
    return {
        "metrics": by_metric,
        "min_max": {
            "test_cost": {"min": by_metric["test_cost"]["min"], "max": by_metric["test_cost"]["max"]},
            "test_F1": {"min": by_metric["test_F1"]["min"], "max": by_metric["test_F1"]["max"]},
        },
        "best_genome_selected_by_validation_cost_only": selected,
        "percent_change_from_baseline_per_seed": delta,
        "percent_change_from_baseline_mean": mean_delta,
    }


def _markdown_summary(experiment: str, data: ExperimentData, baselines: dict,
                      trials: list[dict], aggregates: dict, run_id: str) -> str:
    lines = [
        f"# GA Automation: {run_id}",
        "",
        "## ขั้นตอนการทดลอง",
        "",
        "1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.",
        "2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.",
        "3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.",
        "4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.",
        "",
        f"Baseline: **{data.baseline_label}** | GA experiment: **{experiment}**",
        "",
        "## Baseline บน held-out test",
        "",
        "| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, metric in baselines.items():
        lines.append("| {scenario} | {TP} | {FP} | {FN} | {REVIEW} | {precision:.4f} | {recall:.4f} | {F1:.4f} | {cost:.2f} |".format(
            scenario=scenario, **metric))

    lines.extend([
        "",
        "## ผลราย seed (held-out test)",
        "",
        "| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in trials:
        lines.append("| {scenario} | {seed} | {t_m:.4f} | {t_r:.4f} | {c_promote:.4f} | {c_demote:.4f} | {val_cost:.2f} | {test_F1:.4f} | {test_cost:.2f} | {runtime_seconds:.2f} |".format(**row))

    lines.extend(["", "## สรุป mean ± std และการเลือก genome", ""])
    for scenario, aggregate in aggregates.items():
        metric = aggregate["metrics"]
        best = aggregate["best_genome_selected_by_validation_cost_only"]
        delta = aggregate["percent_change_from_baseline_mean"]
        lines.extend([
            f"### {scenario}",
            "",
            f"- Test F1: {metric['test_F1']['mean']:.4f} ± {metric['test_F1']['std']:.4f} "
            f"(min {metric['test_F1']['min']:.4f}, max {metric['test_F1']['max']:.4f})",
            f"- Test cost: {metric['test_cost']['mean']:.2f} ± {metric['test_cost']['std']:.2f} "
            f"(min {metric['test_cost']['min']:.2f}, max {metric['test_cost']['max']:.2f})",
            f"- เปลี่ยนจาก baseline (mean): F1 {_format_pct(delta['test_F1'])} | "
            f"cost {_format_pct(delta['test_cost'])}",
            f"- Genome ที่เลือกจาก validation cost เท่านั้น: seed {best['seed']} "
            f"[t_m={best['t_m']:.4f}, t_r={best['t_r']:.4f}, "
            f"c_promote={best['c_promote']:.4f}, c_demote={best['c_demote']:.4f}] "
            f"(val cost {best['val_cost']:.2f})",
            "",
        ])
    return "\n".join(lines) + "\n"


def run_experiments(data: ExperimentData, consts: dict, seeds: Iterable[int], settings: Optional[dict],
                    scenarios: list[tuple[str, ga.CostWeights]], output_dir: Path,
                    run_id: str = "manual", allow_precreated_output: bool = False,
                    extra_config: Optional[dict] = None) -> dict:
    """Run and persist all independent trials. Public for synthetic integration tests."""
    _validate_experiment_data(data, consts)
    output_dir = Path(output_dir)
    if output_dir.exists():
        final_files = ("config.json", "trials.csv", "generations.csv", "summary.json", "summary.md")
        if not allow_precreated_output or not output_dir.is_dir() or any((output_dir / name).exists() for name in final_files):
            raise FileExistsError(f"refusing to overwrite existing experiment output: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    settings = settings or {}
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one seed is required")

    trials, generations, baselines = [], [], {}
    for scenario, weights in scenarios:
        for seed in seeds:
            config = _make_config(seed, settings, weights)
            trial, history = _run_trial(data, consts, scenario, config)
            trials.append(trial)
            generations.extend(history)
        # This baseline is reported after GA selection and never participates in it.
        baselines[scenario] = _baseline_metrics(data, consts, weights)

    trial_fields = ["experiment", "scenario", "seed", "t_m", "t_r", "c_promote", "c_demote",
                    "val_cost", "objective_val_cost", "test_TP", "test_FP", "test_FN", "test_REVIEW",
                    "test_precision", "test_recall", "test_F1", "test_cost", "runtime_seconds"]
    generation_fields = ["experiment", "scenario", "seed", "generation", "objective_val_cost", "val_cost",
                         "t_m", "t_r", "c_promote", "c_demote"]
    _write_csv(output_dir / "trials.csv", trials, trial_fields)
    _write_csv(output_dir / "generations.csv", generations, generation_fields)

    grouped = {
        scenario: _aggregate([row for row in trials if row["scenario"] == scenario], baselines[scenario])
        for scenario, _ in scenarios
    }
    summary = {
        "run_id": run_id,
        "experiment": data.experiment,
        "baseline": data.baseline_label,
        "selection_policy": "GA genome and best seed are selected by validation cost only; test is held out.",
        "baselines": baselines,
        "trials": trials,
        "aggregates": grouped,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        _markdown_summary(data.experiment, data, baselines, trials, grouped, run_id), encoding="utf-8")

    config_document = {
        "run_id": run_id,
        "experiment": data.experiment,
        "seeds": seeds,
        "ga_overrides": settings,
        "scenarios": [{"name": name, "weights": weights.as_dict()} for name, weights in scenarios],
        "defaults": asdict(ga.default_config()),
        "input_artifacts": data.input_manifest,
        "split": {key: consts[key] for key in ("val", "test")},
        "split_rule": "entity-aware md5(user_folder or negative profile id), test when bucket < 30; cross-split pairs dropped",
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    if extra_config:
        config_document.update(extra_config)
    _write_json(output_dir / "config.json", config_document)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible, validation-only GA threshold experiments.")
    parser.add_argument("--experiment", choices=("r1", "r1-sensitivity", "r3"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--population", dest="population_size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--elite-size", dest="elite_size", type=int)
    parser.add_argument("--mutation-probability", type=float)
    parser.add_argument("--mutation-sigma", type=float)
    parser.add_argument("--w-fp", type=float)
    parser.add_argument("--w-fn", type=float)
    parser.add_argument("--w-review", type=float)
    parser.add_argument("--output-root", type=Path, default=AUTOMATION_ROOT)
    parser.add_argument("--run-id", help="Optional leaf directory name; defaults to <experiment>_YYYYMMDD_HHMMSS")
    parser.add_argument("--bert-probabilities", type=Path, default=R2_PROBABILITIES)
    parser.add_argument("--train-r2", action="store_true",
                        help="R3 only: train a fresh validation-only R2 MLP into this run before GA")
    parser.add_argument("--model-seed", type=int, default=42,
                        help="Seed for a fresh R2 training run (default: 42)")
    parser.add_argument("--model-epochs", type=int,
                        help="Override fresh R2 training epochs; default preserves the R2 setting")
    parser.add_argument("--model-patience", type=int,
                        help="Override fresh R2 early-stopping patience; default preserves the R2 setting")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    weight_args = (args.w_fp, args.w_fn, args.w_review)
    override_weights = None
    if any(value is not None for value in weight_args):
        default = ga.default_weights()
        override_weights = ga.CostWeights(
            args.w_fp if args.w_fp is not None else default.fp,
            args.w_fn if args.w_fn is not None else default.fn,
            args.w_review if args.w_review is not None else default.review,
        )
    try:
        scenarios = _scenario_specs(args.experiment, override_weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.train_r2 and args.experiment != "r3":
        parser.error("--train-r2 is valid only with --experiment r3")
    if (args.model_epochs is not None or args.model_patience is not None) and not args.train_r2:
        parser.error("--model-epochs and --model-patience require --train-r2")

    cache = build_cache()
    manifest = {"scored_cache": _artifact_manifest(CACHE)}
    run_id = args.run_id or f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(run_id).name != run_id:
        parser.error("--run-id must be a directory name, not a path")
    output_dir = args.output_root / run_id
    extra_config = {}
    allow_precreated_output = False
    if args.experiment == "r3":
        if args.train_r2:
            if output_dir.exists():
                raise FileExistsError(f"refusing to overwrite existing experiment output: {output_dir}")
            output_dir.mkdir(parents=True)
            probability_path, training_meta, training_manifest = train_fresh_r2_for_run(
                cache, output_dir, args.model_seed, args.model_epochs, args.model_patience)
            manifest.update(training_manifest)
            manifest["r2_probabilities"] = _artifact_manifest(probability_path)
            extra_config["fresh_r2_training"] = training_meta
            allow_precreated_output = True
        else:
            probability_path = args.bert_probabilities
            manifest["r2_probabilities"] = _artifact_manifest(probability_path)
        data = make_r3_data(cache, probability_path, manifest)
    else:
        data = make_r1_data(cache, manifest)
    consts = split_constants(cache)
    settings = {
        "population_size": args.population_size,
        "generations": args.generations,
        "elite_size": args.elite_size,
        "mutation_probability": args.mutation_probability,
        "mutation_sigma": args.mutation_sigma,
    }
    summary = run_experiments(
        data, consts, args.seeds, settings, scenarios, output_dir, run_id,
        allow_precreated_output=allow_precreated_output, extra_config=extra_config,
    )
    print(f"Saved {len(summary['trials'])} trial(s) -> {output_dir}")
    for scenario, aggregate in summary["aggregates"].items():
        metrics = aggregate["metrics"]
        print(f"{scenario}: test F1 {metrics['test_F1']['mean']:.4f} +/- {metrics['test_F1']['std']:.4f}; "
              f"cost {metrics['test_cost']['mean']:.2f} +/- {metrics['test_cost']['std']:.2f}")
    return 0


if __name__ == "__main__":
    pass  # Entrypoint moved below the publication-grade overrides.
# Publication-grade runner override: frozen R2 probabilities + nested entity validation.

import sys

from exp_lib import build_nested_entity_split, nested_split_constants, PROFILES


def make_r3_data(cache: pd.DataFrame, probability_path: Path,
                 input_manifest: Optional[dict] = None) -> ExperimentData:
    _require_columns(cache, ("profile_id_a", "profile_id_b", "decision_source", "experiment_split",
                             "name_sim", "actual"), "nested cache")
    if cache.duplicated(["profile_id_a", "profile_id_b"]).any():
        raise ValueError("nested cache contains duplicate pair keys")
    probability_path = Path(probability_path)
    if not probability_path.exists():
        raise FileNotFoundError(f"R3 requires the frozen R2 probability artifact: {probability_path}")
    probs = pd.read_parquet(probability_path)
    _require_columns(probs, ("profile_id_a", "profile_id_b", "probability_r2"), "R2 probabilities")
    if probs.duplicated(["profile_id_a", "profile_id_b"]).any():
        raise ValueError("R2 probabilities contain duplicate pair keys")
    if not np.isfinite(probs["probability_r2"].to_numpy(dtype=float)).all() or (
            (probs["probability_r2"] < 0).any() or (probs["probability_r2"] > 1).any()):
        raise ValueError("R2 probabilities must be finite values in [0, 1]")

    all_scored = cache.loc[cache["decision_source"] != "AUTO_EXACT"].copy()
    joined = all_scored.merge(probs, on=["profile_id_a", "profile_id_b"], how="left", validate="one_to_one")
    if len(joined) != len(all_scored) or joined["probability_r2"].isna().any():
        raise ValueError("R2 probability join is incomplete; R3 will not retrain or fill values")
    if len(probs) != len(all_scored):
        raise ValueError("R2 probability artifact must contain exactly all scored pair keys")
    scored = joined.loc[joined["experiment_split"].isin(["ga_validation", "test"])].copy()
    if scored.empty:
        raise ValueError("nested R3 data has no GA-validation or held-out-test pairs")
    data = ExperimentData(
        experiment="r3",
        scored=scored,
        score_column="probability_r2",
        baseline_label="R2_manual",
        baseline_decider=_r2_baseline,
        optimise_score_floor=None,
        input_manifest=input_manifest or {},
    )
    data.validation_split = "ga_validation"
    data.test_split = "test"
    data.nested_split = True
    return data


def _validation_name(data: ExperimentData) -> str:
    return getattr(data, "validation_split", "val")


def _test_name(data: ExperimentData) -> str:
    return getattr(data, "test_split", "test")


def _split_column(data: ExperimentData) -> str:
    return "experiment_split" if getattr(data, "nested_split", False) else "split"


def _validate_experiment_data(data: ExperimentData, consts: dict) -> None:
    column, validation_name, test_name = _split_column(data), _validation_name(data), _test_name(data)
    _require_columns(data.scored, (column, "actual", "name_sim", data.score_column), data.experiment)
    present = set(data.scored[column].unique())
    if not {validation_name, test_name}.issubset(present):
        raise ValueError(f"{data.experiment} needs {validation_name} and {test_name}; found {sorted(present)}")
    if not {validation_name, test_name}.issubset(consts):
        raise ValueError("split constants do not cover validation and held-out test")


def _full_validation_cost(data: ExperimentData, consts: dict, genome: np.ndarray,
                          weights: ga.CostWeights) -> float:
    column, validation_name = _split_column(data), _validation_name(data)
    validation = data.scored.loc[data.scored[column] == validation_name]
    code = ga.decide_code(validation[data.score_column].to_numpy(), validation["name_sim"].to_numpy(), genome)
    return _metric_with_cost(code, validation["actual"].to_numpy(), consts[validation_name], weights)["cost"]


def _run_trial(data: ExperimentData, consts: dict, scenario: str, config: ga.GAConfig) -> tuple[dict, list[dict]]:
    """Optimise only the designated validation split, then open held-out test once."""
    column, validation_name, test_name = _split_column(data), _validation_name(data), _test_name(data)
    validation = data.scored.loc[data.scored[column] == validation_name]
    if data.optimise_score_floor is None:
        ga_validation, const_fn_below = validation, 0
    else:
        ga_validation = validation.loc[validation[data.score_column] >= data.optimise_score_floor]
        const_fn_below = int(((validation[data.score_column] < data.optimise_score_floor)
                              & (validation["actual"] == 1)).sum())
    if ga_validation.empty:
        raise ValueError("validation data contains no pairs in the GA search range")

    started = time.perf_counter()
    best, history = ga.run_ga(
        ga_validation[data.score_column].to_numpy(),
        ga_validation["name_sim"].to_numpy(),
        ga_validation["actual"].to_numpy(dtype=np.int8),
        const_fn_below=const_fn_below,
        config=config,
        verbose=False,
    )
    # This is deliberately the first test-label access in the trial.
    test = data.scored.loc[data.scored[column] == test_name]
    test_code = ga.decide_code(test[data.score_column].to_numpy(), test["name_sim"].to_numpy(), best)
    test_metrics = _metric_with_cost(test_code, test["actual"].to_numpy(), consts[test_name], config.weights)
    objective_cost = ga.raw_cost(
        ga.decide_code(ga_validation[data.score_column].to_numpy(), ga_validation["name_sim"].to_numpy(), best),
        ga_validation["actual"].to_numpy(dtype=np.int8), config.weights,
    ) + config.weights.fn * const_fn_below
    record = {
        "experiment": data.experiment, "scenario": scenario, "seed": config.seed,
        "t_m": float(best[0]), "t_r": float(best[1]), "c_promote": float(best[2]), "c_demote": float(best[3]),
        "val_cost": _full_validation_cost(data, consts, best, config.weights),
        "objective_val_cost": float(objective_cost),
        "test_TP": test_metrics["TP"], "test_FP": test_metrics["FP"], "test_FN": test_metrics["FN"],
        "test_REVIEW": test_metrics["REVIEW"], "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"], "test_F1": test_metrics["F1"], "test_cost": test_metrics["cost"],
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
    generations = []
    for entry in history:
        genome = np.asarray(entry["genome"], dtype=float)
        generations.append({
            "experiment": data.experiment, "scenario": scenario, "seed": config.seed, "generation": entry["gen"],
            "objective_val_cost": entry["cost"],
            "val_cost": _full_validation_cost(data, consts, genome, config.weights),
            "t_m": float(genome[0]), "t_r": float(genome[1]),
            "c_promote": float(genome[2]), "c_demote": float(genome[3]),
        })
    return record, generations


def _baseline_metrics(data: ExperimentData, consts: dict, weights: ga.CostWeights) -> dict:
    column, test_name = _split_column(data), _test_name(data)
    test = data.scored.loc[data.scored[column] == test_name]
    return _metric_with_cost(data.baseline_decider(test), test["actual"].to_numpy(), consts[test_name], weights)


def _runner_environment() -> dict:
    import platform
    import sklearn
    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import torch
        environment.update({
            "pytorch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "torch_num_threads": int(torch.get_num_threads()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        })
    except ImportError:
        environment["pytorch"] = "not installed"
    return environment


def run_experiments(data: ExperimentData, consts: dict, seeds: Iterable[int], settings: Optional[dict],
                    scenarios: list[tuple[str, ga.CostWeights]], output_dir: Path,
                    run_id: str = "manual", allow_precreated_output: bool = False,
                    extra_config: Optional[dict] = None) -> dict:
    """Run reproducible GA trials and persist their validation-only provenance."""
    _validate_experiment_data(data, consts)
    output_dir = Path(output_dir)
    final_files = ("config.json", "trials.csv", "generations.csv", "summary.json", "summary.md")
    if output_dir.exists():
        if not allow_precreated_output or not output_dir.is_dir() or any((output_dir / name).exists() for name in final_files):
            raise FileExistsError(f"refusing to overwrite existing experiment output: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    settings = settings or {}
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one seed is required")
    (output_dir / "status.json").write_text(json.dumps({
        "status": "running", "stage": "ga_optimisation", "run_id": run_id,
        "updated_at": datetime.now().astimezone().isoformat(),
    }, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(_runner_environment(), indent=2), encoding="utf-8")
    if extra_config and extra_config.get("split_manifest"):
        _write_json(output_dir / "split_manifest.json", extra_config["split_manifest"])

    trials, generations, baselines = [], [], {}
    for scenario, weights in scenarios:
        for seed in seeds:
            trial, history = _run_trial(data, consts, scenario, _make_config(seed, settings, weights))
            trials.append(trial)
            generations.extend(history)
        baselines[scenario] = _baseline_metrics(data, consts, weights)

    trial_fields = ["experiment", "scenario", "seed", "t_m", "t_r", "c_promote", "c_demote",
                    "val_cost", "objective_val_cost", "test_TP", "test_FP", "test_FN", "test_REVIEW",
                    "test_precision", "test_recall", "test_F1", "test_cost", "runtime_seconds"]
    generation_fields = ["experiment", "scenario", "seed", "generation", "objective_val_cost", "val_cost",
                         "t_m", "t_r", "c_promote", "c_demote"]
    _write_csv(output_dir / "trials.csv", trials, trial_fields)
    _write_csv(output_dir / "generations.csv", generations, generation_fields)
    grouped = {name: _aggregate([row for row in trials if row["scenario"] == name], baselines[name])
               for name, _ in scenarios}
    summary = {
        "run_id": run_id, "experiment": data.experiment, "baseline": data.baseline_label,
        "selection_policy": "GA genome and deployment seed are selected by validation cost only; held-out test is never used for selection.",
        "baselines": baselines, "trials": trials, "aggregates": grouped,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        _markdown_summary(data.experiment, data, baselines, trials, grouped, run_id), encoding="utf-8")

    validation_name, test_name = _validation_name(data), _test_name(data)
    result_artifacts = {name: _artifact_manifest(output_dir / name)
                        for name in ("trials.csv", "generations.csv", "summary.json", "summary.md")}
    config_document = {
        "run_id": run_id, "experiment": data.experiment, "seeds": seeds,
        "ga_overrides": settings,
        "scenarios": [{"name": name, "weights": weights.as_dict()} for name, weights in scenarios],
        "defaults": asdict(ga.default_config()), "input_artifacts": data.input_manifest,
        "provenance_chain": {
            "input_cache_and_features": data.input_manifest,
            "ga_config": {"seeds": seeds, "overrides": settings},
            "result_artifacts": result_artifacts,
        },
        "split": {"validation": consts[validation_name], "test": consts[test_name]},
        "split_names": {"validation": validation_name, "test": test_name},
        "selection_policy": summary["selection_policy"],
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    if extra_config:
        config_document.update({key: value for key, value in extra_config.items() if key != "split_manifest"})
    _write_json(output_dir / "config.json", config_document)
    (output_dir / "status.json").write_text(json.dumps({
        "status": "completed", "stage": "complete", "run_id": run_id,
        "updated_at": datetime.now().astimezone().isoformat(),
    }, indent=2), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible GA threshold experiments.")
    parser.add_argument("--experiment", choices=("r1", "r1-sensitivity", "r3"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--population", dest="population_size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--elite-size", dest="elite_size", type=int)
    parser.add_argument("--mutation-probability", type=float)
    parser.add_argument("--mutation-sigma", type=float)
    parser.add_argument("--w-fp", type=float)
    parser.add_argument("--w-fn", type=float)
    parser.add_argument("--w-review", type=float)
    parser.add_argument("--output-root", type=Path, default=AUTOMATION_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--bert-probabilities", type=Path, default=R2_PROBABILITIES)
    parser.add_argument("--train-r2", action="store_true")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Deterministic entity allocation seed for model/calibration/GA validation.")
    parser.add_argument("--model-epochs", type=int)
    parser.add_argument("--model-patience", type=int)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    weights_in = (args.w_fp, args.w_fn, args.w_review)
    override_weights = None
    if any(value is not None for value in weights_in):
        default = ga.default_weights()
        override_weights = ga.CostWeights(
            args.w_fp if args.w_fp is not None else default.fp,
            args.w_fn if args.w_fn is not None else default.fn,
            args.w_review if args.w_review is not None else default.review,
        )
    try:
        scenarios = _scenario_specs(args.experiment, override_weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.train_r2 and args.experiment != "r3":
        parser.error("--train-r2 is valid only with --experiment r3")
    if (args.model_epochs is not None or args.model_patience is not None) and not args.train_r2:
        parser.error("--model-epochs and --model-patience require --train-r2")

    cache = build_cache()
    manifest = {"scored_cache": _artifact_manifest(CACHE)}
    run_id = args.run_id or f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(run_id).name != run_id:
        parser.error("--run-id must be a directory name, not a path")
    output_dir = args.output_root / run_id
    allow_precreated_output = False
    extra_config = {}
    if args.experiment == "r3":
        nested_cache, split_manifest = build_nested_entity_split(cache, seed=args.split_seed)
        consts = nested_split_constants(nested_cache, seed=args.split_seed)
        split_manifest["source_profiles"] = _artifact_manifest(PROFILES)
        extra_config["split_manifest"] = split_manifest
        extra_config["nested_split_seed"] = args.split_seed
        if args.train_r2:
            if output_dir.exists():
                raise FileExistsError(f"refusing to overwrite existing experiment output: {output_dir}")
            output_dir.mkdir(parents=True)
            _write_json(output_dir / "split_manifest.json", split_manifest)
            import exp_r2_bert_feature as r2
            _write_json(output_dir / "environment.json", r2.environment_snapshot(1))
            _write_json(output_dir / "status.json", {
                "status": "running", "stage": "r2_training", "run_id": run_id,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            probability_path, training_meta, training_manifest = train_fresh_r2_for_run(
                nested_cache, output_dir, args.model_seed, args.model_epochs, args.model_patience)
            manifest.update(training_manifest)
            manifest["r2_probabilities"] = _artifact_manifest(probability_path)
            extra_config["fresh_r2_training"] = training_meta
            allow_precreated_output = True
        else:
            probability_path = args.bert_probabilities
            manifest["r2_probabilities"] = _artifact_manifest(probability_path)
        data = make_r3_data(nested_cache, probability_path, manifest)
    else:
        # R1 uses the same nested GA-validation and sealed test entities as R3.
        # Its score remains the original 17-feature production score; only its
        # decision rule is optimized.
        nested_cache, split_manifest = build_nested_entity_split(cache, seed=args.split_seed)
        split_manifest["source_profiles"] = _artifact_manifest(PROFILES)
        extra_config["split_manifest"] = split_manifest
        extra_config["nested_split_seed"] = args.split_seed
        data = make_r1_data(nested_cache, manifest)
        data.scored = data.scored.loc[data.scored["experiment_split"].isin(["ga_validation", "test"])].copy()
        data.validation_split = "ga_validation"
        data.test_split = "test"
        data.nested_split = True
        consts = nested_split_constants(nested_cache, seed=args.split_seed)
    settings = {
        "population_size": args.population_size, "generations": args.generations,
        "elite_size": args.elite_size, "mutation_probability": args.mutation_probability,
        "mutation_sigma": args.mutation_sigma,
    }
    summary = run_experiments(data, consts, args.seeds, settings, scenarios, output_dir, run_id,
                              allow_precreated_output=allow_precreated_output, extra_config=extra_config)
    print(f"Saved {len(summary['trials'])} trial(s) -> {output_dir}")
    for scenario, aggregate in summary["aggregates"].items():
        metrics = aggregate["metrics"]
        print(f"{scenario}: test F1 {metrics['test_F1']['mean']:.4f} +/- {metrics['test_F1']['std']:.4f}; "
              f"cost {metrics['test_cost']['mean']:.2f} +/- {metrics['test_cost']['std']:.2f}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
