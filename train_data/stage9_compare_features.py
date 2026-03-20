from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = ["profile_id_a", "profile_id_b", "label", "pair_type"]


def _read_parquet_compat(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if exc.__class__.__name__ not in {"ArrowKeyError", "ImportError"}:
            raise
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return pd.concat([_read_parquet_compat(part) for part in parts], ignore_index=True)
    return _read_parquet_compat(path)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in KEY_COLS]


def _safe_numeric_diff(old: pd.Series, new: pd.Series) -> dict[str, float]:
    old_num = pd.to_numeric(old, errors="coerce").astype(float)
    new_num = pd.to_numeric(new, errors="coerce").astype(float)
    diff = np.abs(old_num.to_numpy() - new_num.to_numpy())
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {"max_abs_diff": math.nan, "mean_abs_diff": math.nan, "mismatch_count": int((old.fillna("__NA__") != new.fillna("__NA__")).sum())}
    return {
        "max_abs_diff": float(finite.max()),
        "mean_abs_diff": float(finite.mean()),
        "mismatch_count": int(np.count_nonzero(finite > 1e-9)),
    }


def compare_frames(old_df: pd.DataFrame, new_df: pd.DataFrame, top_k: int) -> dict:
    result: dict[str, object] = {
        "old_rows": int(len(old_df)),
        "new_rows": int(len(new_df)),
        "same_row_count": bool(len(old_df) == len(new_df)),
    }

    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    result["old_only_cols"] = sorted(old_cols - new_cols)
    result["new_only_cols"] = sorted(new_cols - old_cols)

    common_cols = [col for col in old_df.columns if col in new_df.columns]
    missing_keys = [col for col in KEY_COLS if col not in common_cols]
    if missing_keys:
        raise ValueError(f"Missing key columns for comparison: {missing_keys}")

    old_idx = old_df[common_cols].set_index(KEY_COLS, drop=True).sort_index()
    new_idx = new_df[common_cols].set_index(KEY_COLS, drop=True).sort_index()

    old_keys = set(old_idx.index.tolist())
    new_keys = set(new_idx.index.tolist())
    result["old_only_rows"] = int(len(old_keys - new_keys))
    result["new_only_rows"] = int(len(new_keys - old_keys))

    shared_keys = sorted(old_keys & new_keys)
    result["shared_rows"] = int(len(shared_keys))
    if not shared_keys:
        result["numeric_diffs"] = {}
        result["worst_features"] = []
        return result

    old_shared = old_idx.loc[shared_keys]
    new_shared = new_idx.loc[shared_keys]
    feature_cols = [col for col in _feature_cols(old_shared.reset_index()) if col in new_shared.columns]

    numeric_diffs: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        numeric_diffs[col] = _safe_numeric_diff(old_shared[col], new_shared[col])

    worst_features = sorted(
        (
            {"feature": col, **stats}
            for col, stats in numeric_diffs.items()
            if not math.isnan(stats["max_abs_diff"])
        ),
        key=lambda item: item["max_abs_diff"],
        reverse=True,
    )[:top_k]

    result["numeric_diffs"] = numeric_diffs
    result["worst_features"] = worst_features
    result["exact_match_features"] = sorted(
        [col for col, stats in numeric_diffs.items() if stats["mismatch_count"] == 0]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 9 feature outputs")
    parser.add_argument("--old", required=True, help="Old/original parquet file or directory of parquet chunks")
    parser.add_argument("--new", required=True, help="New/chunked parquet file or directory of parquet chunks")
    parser.add_argument("--report", default="", help="Optional JSON output path")
    parser.add_argument("--top-k", type=int, default=10, help="Show top K features with largest max abs diff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_path = Path(args.old)
    new_path = Path(args.new)

    print(f"[Compare] Loading old features from {old_path}")
    old_df = _load_frame(old_path)
    print(f"[Compare] Loading new features from {new_path}")
    new_df = _load_frame(new_path)

    result = compare_frames(old_df, new_df, top_k=args.top_k)

    print("\nComparison summary")
    print(json.dumps(
        {
            "old_rows": result["old_rows"],
            "new_rows": result["new_rows"],
            "same_row_count": result["same_row_count"],
            "old_only_cols": result["old_only_cols"],
            "new_only_cols": result["new_only_cols"],
            "old_only_rows": result["old_only_rows"],
            "new_only_rows": result["new_only_rows"],
            "shared_rows": result["shared_rows"],
            "worst_features": result["worst_features"],
        },
        indent=2,
    ))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[Compare] Report written to {report_path}")


if __name__ == "__main__":
    main()
