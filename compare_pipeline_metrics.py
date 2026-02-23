import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline pipeline metrics vs location-enhanced metrics"
    )
    parser.add_argument(
        "--baseline",
        default="data/processed/bigram_pipeline_results/bigram_pipeline_metrics_summary.csv",
        help="Baseline metrics summary CSV",
    )
    parser.add_argument(
        "--with-location",
        default="data/processed/bigram_pipeline_results_with_location/bigram_pipeline_metrics_summary.csv",
        help="Location-enhanced metrics summary CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/bigram_pipeline_results_with_location/bigram_vs_location_comparison.csv",
        help="Output comparison CSV",
    )
    args = parser.parse_args()

    base = pd.read_csv(args.baseline)
    loc = pd.read_csv(args.with_location)

    # Align model naming for linear branch
    base = base.copy()
    loc = loc.copy()
    base["model_norm"] = base["model"].replace({"SGD": "Linear"})
    loc["model_norm"] = loc["model"].replace({"SGD": "Linear", "SGD_fallback_LR": "Linear"})

    base_small = base[
        ["pair_name", "model_norm", "top1_accuracy", "candidate_recall_topk"]
    ].rename(
        columns={
            "top1_accuracy": "top1_accuracy_baseline",
            "candidate_recall_topk": "candidate_recall_baseline",
        }
    )
    loc_small = loc[
        ["pair_name", "model_norm", "top1_accuracy", "candidate_recall_topk"]
    ].rename(
        columns={
            "top1_accuracy": "top1_accuracy_with_location",
            "candidate_recall_topk": "candidate_recall_with_location",
        }
    )

    out = base_small.merge(loc_small, on=["pair_name", "model_norm"], how="inner")
    out["delta_top1_accuracy"] = (
        out["top1_accuracy_with_location"] - out["top1_accuracy_baseline"]
    )
    out["delta_candidate_recall"] = (
        out["candidate_recall_with_location"] - out["candidate_recall_baseline"]
    )
    out = out.sort_values(["pair_name", "model_norm"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
