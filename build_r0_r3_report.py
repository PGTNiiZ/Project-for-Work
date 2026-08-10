"""Build the publication-facing R0/R1/R2/R3 comparison from completed run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, source: str, metric: dict, variability: dict | None = None) -> dict:
    row = {
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
        "F1_std": 0.0,
        "cost_std": 0.0,
    }
    if variability:
        row["F1_std"] = variability["test_F1"]["std"]
        row["cost_std"] = variability["test_cost"]["std"]
    return row


def aggregate_row(label: str, source: str, aggregate: dict) -> dict:
    metric = aggregate["metrics"]
    return {
        "experiment": label,
        "score_model": source,
        "TP": metric["test_TP"]["mean"],
        "FP": metric["test_FP"]["mean"],
        "FN": metric["test_FN"]["mean"],
        "REVIEW": metric["test_REVIEW"]["mean"],
        "precision": metric["test_precision"]["mean"],
        "recall": metric["test_recall"]["mean"],
        "F1": metric["test_F1"]["mean"],
        "cost": metric["test_cost"]["mean"],
        "F1_std": metric["test_F1"]["std"],
        "cost_std": metric["test_cost"]["std"],
    }


def write_charts(r1_dir: Path, r3_dir: Path, output_dir: Path) -> list[str]:
    training = pd.read_csv(r3_dir / "training_history.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(training["epoch"], training["train_loss"], marker="o", color="#1f77b4", label="train focal loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train focal loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(training["epoch"], training["calibration_average_precision"], marker="s", color="#d62728",
             label="calibration AP")
    ax2.set_ylabel("Calibration average precision", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.tight_layout()
    training_chart = output_dir / "r2_training_history.png"
    fig.savefig(training_chart, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, directory in (("R1", r1_dir), ("R3", r3_dir)):
        history = pd.read_csv(directory / "generations.csv")
        for seed, group in history.groupby("seed"):
            ax.plot(group["generation"], group["val_cost"], alpha=0.55, linewidth=1,
                    label=f"{label} seed {seed}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation business cost")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    ga_chart = output_dir / "r1_r3_ga_history.png"
    fig.savefig(ga_chart, dpi=160)
    plt.close(fig)
    return [training_chart.name, ga_chart.name]


def model_seed_robustness(run_dirs: list[Path]) -> dict:
    rows = []
    for directory in run_dirs:
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
        metrics = summary["aggregates"]["A_current"]["metrics"]
        rows.append({
            "model_seed": config["fresh_r2_training"]["training_config"]["seed"],
            "run": str(directory.resolve()),
            "ga_seed_count": len(summary["trials"]),
            "F1_mean_over_ga_seeds": metrics["test_F1"]["mean"],
            "F1_std_within_ga_seeds": metrics["test_F1"]["std"],
            "cost_mean_over_ga_seeds": metrics["test_cost"]["mean"],
            "cost_std_within_ga_seeds": metrics["test_cost"]["std"],
        })
    frame = pd.DataFrame(rows).sort_values("model_seed")
    return {
        "per_model_seed": frame.to_dict(orient="records"),
        "between_model_seed_variation": {
            "F1_mean": float(frame["F1_mean_over_ga_seeds"].mean()),
            "F1_sample_std": float(frame["F1_mean_over_ga_seeds"].std(ddof=1)) if len(frame) > 1 else 0.0,
            "cost_mean": float(frame["cost_mean_over_ga_seeds"].mean()),
            "cost_sample_std": float(frame["cost_mean_over_ga_seeds"].std(ddof=1)) if len(frame) > 1 else 0.0,
        },
    }


def build_report(r1_dir: Path, r3_dir: Path, output_dir: Path,
                 model_seed_runs: list[Path] | None = None) -> dict:
    r1_summary = read_json(r1_dir / "summary.json")
    r3_summary = read_json(r3_dir / "summary.json")
    r1_aggregate = r1_summary["aggregates"]["A_current"]
    r3_aggregate = r3_summary["aggregates"]["A_current"]
    r0 = metric_row("R0", "Original 17-feature score / production manual rule",
                    r1_summary["baselines"]["A_current"])
    r1 = aggregate_row("R1", "Original 17-feature score / GA rule", r1_aggregate)
    r2 = metric_row("R2", "Frozen 18-feature MLP + MiniLM cosine / manual 0.98/0.95",
                    r3_summary["baselines"]["A_current"])
    r3 = aggregate_row("R3", "Same frozen R2 probabilities / GA rule", r3_aggregate)
    table = pd.DataFrame([r0, r1, r2, r3])
    for target in ("R2", "R3"):
        index = table.index[table["experiment"] == target][0]
        table.loc[index, "F1_change_from_R0_pct"] = (table.loc[index, "F1"] - r0["F1"]) / r0["F1"] * 100
        table.loc[index, "cost_change_from_R0_pct"] = (table.loc[index, "cost"] - r0["cost"]) / r0["cost"] * 100
    r3_index = table.index[table["experiment"] == "R3"][0]
    table.loc[r3_index, "F1_change_from_R2_pct"] = (r3["F1"] - r2["F1"]) / r2["F1"] * 100
    table.loc[r3_index, "cost_change_from_R2_pct"] = (r3["cost"] - r2["cost"]) / r2["cost"] * 100

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "r0_r3_experimental_matrix.csv", index=False)
    charts = write_charts(r1_dir, r3_dir, output_dir)
    report = {
        "selection_policy": "R1 and R3 deployment genomes were selected from GA-validation cost only; test metrics were not used for seed/genome selection.",
        "r1_run": str(r1_dir.resolve()),
        "r3_run": str(r3_dir.resolve()),
        "r3_probability_sha256": sha256(r3_dir / "r2_probabilities.parquet"),
        "table": table.to_dict(orient="records"),
        "r1_selected_genome": r1_aggregate["best_genome_selected_by_validation_cost_only"],
        "r3_selected_genome": r3_aggregate["best_genome_selected_by_validation_cost_only"],
        "charts": charts,
    }
    if model_seed_runs:
        report["model_seed_robustness"] = model_seed_robustness(model_seed_runs)
    (output_dir / "r0_r3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# R0/R1/R2/R3 Experiment Report", "", report["selection_policy"], "",
          "| Experiment | Score/model | F1 (mean +/- sd) | Cost (mean +/- sd) |", "|---|---|---:|---:|"]
    for row in report["table"]:
        md.append(f"| {row['experiment']} | {row['score_model']} | {row['F1']:.4f} +/- {row['F1_std']:.4f} | {row['cost']:.2f} +/- {row['cost_std']:.2f} |")
    md.extend(["", "## Selected validation-only genomes", "",
               f"- R1: seed {report['r1_selected_genome']['seed']} (validation cost {report['r1_selected_genome']['val_cost']:.2f})",
               f"- R3: seed {report['r3_selected_genome']['seed']} (validation cost {report['r3_selected_genome']['val_cost']:.2f})",
               "", "## Figures", "", *[f"- {name}" for name in charts], ""])
    if report.get("model_seed_robustness"):
        variance = report["model_seed_robustness"]["between_model_seed_variation"]
        md.extend(["## Model-seed robustness", "",
                   "Within-model GA-seed standard deviations are recorded per model below; the final line is the sample standard deviation across model-seed means.",
                   "", "| Model seed | F1 mean over GA seeds | F1 SD within GA seeds | Cost mean over GA seeds | Cost SD within GA seeds |",
                   "|---:|---:|---:|---:|---:|"])
        for row in report["model_seed_robustness"]["per_model_seed"]:
            md.append(f"| {row['model_seed']} | {row['F1_mean_over_ga_seeds']:.4f} | {row['F1_std_within_ga_seeds']:.4f} | {row['cost_mean_over_ga_seeds']:.2f} | {row['cost_std_within_ga_seeds']:.2f} |")
        md.extend(["", f"Across model-seed means: F1 SD {variance['F1_sample_std']:.4f}; cost SD {variance['cost_sample_std']:.2f}.", ""])
    (output_dir / "r0_r3_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r3-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed-runs", type=Path, nargs="*")
    args = parser.parse_args()
    build_report(args.r1_run, args.r3_run, args.output_dir, args.model_seed_runs)
    print(f"Saved report -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
