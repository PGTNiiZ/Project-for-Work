from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKED_ROOT = SCRIPT_DIR / "stage9_pipeline_chunked"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "stage10_13_training"

KEY_COLS = ["profile_id_a", "profile_id_b", "label", "pair_type"]


def _read_parquet_compat(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if exc.__class__.__name__ not in {"ArrowKeyError", "ImportError"}:
            raise
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()


def ensure_dirs(training_root: Path) -> tuple[Path, Path]:
    models_dir = training_root / "models"
    reports_dir = training_root / "reports"
    for path in [training_root, models_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_frame(split_name: str, prefix: str, features_dir: Path) -> pd.DataFrame:
    merged_path = features_dir / f"{prefix}_{split_name}_merged.parquet"
    if merged_path.exists():
        return _read_parquet_compat(merged_path)

    split_dir = features_dir / split_name
    if split_dir.exists():
        parts = sorted(split_dir.glob(f"{prefix}_{split_name}_*.parquet"))
        if parts:
            return pd.concat([_read_parquet_compat(part) for part in parts], ignore_index=True)

    raise FileNotFoundError(f"Could not find features for split '{split_name}' under {features_dir}")


def load_feature_cols(prefix: str, artifacts_dir: Path, train_df: pd.DataFrame | None = None, features_dir: Path | None = None) -> list[str]:
    candidates = [artifacts_dir / f"{prefix}_feature_cols.pkl"]
    if prefix == "feature_matrix_chunked":
        candidates.append(artifacts_dir / "feature_cols.pkl")

    selected_cols: list[str] | None = None
    for path in candidates:
        if not path.exists():
            continue
        with path.open("rb") as fh:
            cols = pickle.load(fh)
        if train_df is None:
            return cols
        missing = [col for col in cols if col not in train_df.columns]
        if not missing:
            return cols
        if selected_cols is None:
            selected_cols = [col for col in cols if col in train_df.columns]

    if selected_cols:
        return selected_cols

    if train_df is None:
        if features_dir is None:
            raise ValueError("features_dir is required when train_df is not provided")
        train_df = load_feature_frame("train", prefix, features_dir)
    return [col for col in train_df.columns if col not in KEY_COLS]


def undersample_negatives(train_df: pd.DataFrame, target_ratio: float, seed: int) -> pd.DataFrame:
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return train_df.copy()
    target_neg = min(len(neg), int(len(pos) * target_ratio))
    neg_sampled = neg.sample(n=target_neg, random_state=seed)
    return pd.concat([pos, neg_sampled], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_excluded_features(args: argparse.Namespace) -> set[str]:
    excluded: set[str] = set()
    if args.exclude_features:
        excluded.update([item.strip() for item in args.exclude_features.split(",") if item.strip()])
    if args.exclude_features_file:
        path = Path(args.exclude_features_file)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        excluded.update([line for line in lines if line and not line.startswith("#")])
    return excluded


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    scaler: StandardScaler
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_eval_x: np.ndarray
    train_eval_y: np.ndarray


def build_data_bundle(
    prefix: str,
    neg_ratio: float,
    seed: int,
    features_dir: Path,
    artifacts_dir: Path,
    excluded_features: set[str],
) -> DataBundle:
    train_df = load_feature_frame("train", prefix, features_dir)
    val_df = load_feature_frame("val", prefix, features_dir)
    test_df = load_feature_frame("test", prefix, features_dir)
    feature_cols = load_feature_cols(prefix, artifacts_dir, train_df=train_df, features_dir=features_dir)
    feature_cols = [col for col in feature_cols if col not in excluded_features]
    if not feature_cols:
        raise ValueError("No features left after exclusions.")

    train_balanced = undersample_negatives(train_df, target_ratio=neg_ratio, seed=seed)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_balanced[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))
    val_x = scaler.transform(val_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))
    test_x = scaler.transform(test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))
    train_eval_x = scaler.transform(train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))

    return DataBundle(
        train_df=train_balanced,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        scaler=scaler,
        train_x=train_x,
        train_y=train_balanced["label"].to_numpy(dtype=np.float32),
        val_x=val_x,
        val_y=val_df["label"].to_numpy(dtype=np.float32),
        test_x=test_x,
        test_y=test_df["label"].to_numpy(dtype=np.float32),
        train_eval_x=train_eval_x,
        train_eval_y=train_df["label"].to_numpy(dtype=np.float32),
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, torch.full_like(targets, self.alpha), torch.full_like(targets, 1 - self.alpha))
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


class IdentityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_probs(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_loader(x, np.zeros(len(x), dtype=np.float32), batch_size=batch_size, shuffle=False)
    model.eval()
    probs: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.empty((0,), dtype=np.float32)


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.01, 1.00, 0.01)
    scores = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (probs >= start) & (probs < end if end < 1.0 else probs <= end)
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(probs))
    return float(ece)


def train_model(bundle: DataBundle, args: argparse.Namespace) -> tuple[IdentityMLP, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = IdentityMLP(input_dim=len(bundle.feature_cols), dropout=args.dropout).to(device)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = make_loader(bundle.train_x, bundle.train_y, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_ap = -1.0
    best_epoch = -1
    patience_left = args.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        scheduler.step()
        val_probs = predict_probs(model, bundle.val_x, batch_size=args.eval_batch_size, device=device)
        val_ap = average_precision_score(bundle.val_y, val_probs)
        val_auc = roc_auc_score(bundle.val_y, val_probs)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(bundle.train_x), 1),
            "val_ap": float(val_ap),
            "val_auc": float(val_auc),
        }
        history.append(epoch_summary)
        print(f"[Epoch {epoch:03d}] loss={epoch_summary['train_loss']:.4f} val_ap={val_ap:.4f} val_auc={val_auc:.4f}")

        if val_ap > best_ap:
            best_ap = float(val_ap)
            best_epoch = epoch
            patience_left = args.patience
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_ap": best_ap, "history": history}


def evaluate_and_save(
    model: IdentityMLP,
    bundle: DataBundle,
    args: argparse.Namespace,
    train_meta: dict,
    models_dir: Path,
    reports_dir: Path,
    features_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    val_probs_raw = predict_probs(model, bundle.val_x, batch_size=args.eval_batch_size, device=device)
    test_probs_raw = predict_probs(model, bundle.test_x, batch_size=args.eval_batch_size, device=device)
    train_probs_raw = predict_probs(model, bundle.train_eval_x, batch_size=args.eval_batch_size, device=device)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, bundle.val_y)
    val_probs_cal = calibrator.predict(val_probs_raw)
    test_probs_cal = calibrator.predict(test_probs_raw)
    train_probs_cal = calibrator.predict(train_probs_raw)

    threshold, best_val_f1 = choose_threshold(bundle.val_y, val_probs_cal)
    test_preds = (test_probs_cal >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "best_val_f1": best_val_f1,
        "val_ece_before": expected_calibration_error(val_probs_raw, bundle.val_y),
        "val_ece_after": expected_calibration_error(val_probs_cal, bundle.val_y),
        "test_roc_auc": float(roc_auc_score(bundle.test_y, test_probs_cal)),
        "test_avg_precision": float(average_precision_score(bundle.test_y, test_probs_cal)),
        "test_f1": float(f1_score(bundle.test_y, test_preds, zero_division=0)),
        "test_precision": float(precision_score(bundle.test_y, test_preds, zero_division=0)),
        "test_recall": float(recall_score(bundle.test_y, test_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(bundle.test_y, test_preds).tolist(),
        "classification_report": classification_report(
            bundle.test_y, test_preds, target_names=["NO_MATCH", "MATCH"], zero_division=0, output_dict=True
        ),
        "train_meta": train_meta,
        "n_features": len(bundle.feature_cols),
        "train_size": int(len(bundle.train_df)),
        "val_size": int(len(bundle.val_df)),
        "test_size": int(len(bundle.test_df)),
    }

    model_path = models_dir / "identity_mlp.pt"
    scaler_path = models_dir / "scaler.pkl"
    calibrator_path = models_dir / "isotonic_calibrator.pkl"
    metrics_path = reports_dir / "evaluation_summary.json"
    train_scores_path = reports_dir / "train_scores.parquet"
    val_scores_path = reports_dir / "val_scores.parquet"
    test_scores_path = reports_dir / "test_scores.parquet"

    torch.save({"state_dict": model.state_dict(), "feature_dim": len(bundle.feature_cols)}, model_path)
    with scaler_path.open("wb") as fh:
        pickle.dump(bundle.scaler, fh)
    with calibrator_path.open("wb") as fh:
        pickle.dump(calibrator, fh)
    with (models_dir / "feature_cols.pkl").open("wb") as fh:
        pickle.dump(bundle.feature_cols, fh)
    feature_config = {
        "feature_prefix": args.feature_prefix,
        "feature_count": len(bundle.feature_cols),
        "excluded_features": sorted(parse_excluded_features(args)),
        "features_dir": str(features_dir),
    }
    (reports_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def write_scores(df: pd.DataFrame, probs_raw: np.ndarray, probs_cal: np.ndarray, path: Path) -> None:
        out = df[KEY_COLS].copy()
        out["prob_raw"] = probs_raw
        out["prob_calibrated"] = probs_cal
        out["pred"] = (probs_cal >= threshold).astype(int)
        out.to_parquet(path, index=False)

    write_scores(load_feature_frame("train", args.feature_prefix, features_dir), train_probs_raw, train_probs_cal, train_scores_path)
    write_scores(bundle.val_df, val_probs_raw, val_probs_cal, val_scores_path)
    write_scores(bundle.test_df, test_probs_raw, test_probs_cal, test_scores_path)

    print("\nTraining complete")
    print(f"Model      : {model_path}")
    print(f"Scaler     : {scaler_path}")
    print(f"Calibrator : {calibrator_path}")
    print(f"Metrics    : {metrics_path}")
    print(f"Threshold  : {threshold:.2f}")
    print(f"Test F1    : {metrics['test_f1']:.4f}")
    print(f"Test AUC   : {metrics['test_roc_auc']:.4f}")
    print(f"Test AP    : {metrics['test_avg_precision']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10-13 training pipeline")
    parser.add_argument("--feature-prefix", default="feature_matrix_chunked")
    parser.add_argument("--chunked-root", default=str(DEFAULT_CHUNKED_ROOT))
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--train-neg-ratio", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--exclude-features", default="")
    parser.add_argument("--exclude-features-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunked_root = Path(args.chunked_root)
    features_dir = chunked_root / "features"
    artifacts_dir = chunked_root / "artifacts"
    training_root = Path(args.training_root)
    models_dir, reports_dir = ensure_dirs(training_root)
    excluded_features = parse_excluded_features(args)
    set_seed(args.seed)

    bundle = build_data_bundle(
        prefix=args.feature_prefix,
        neg_ratio=args.train_neg_ratio,
        seed=args.seed,
        features_dir=features_dir,
        artifacts_dir=artifacts_dir,
        excluded_features=excluded_features,
    )
    print(f"Train balanced: {len(bundle.train_df):,} rows")
    print(f"Val          : {len(bundle.val_df):,} rows")
    print(f"Test         : {len(bundle.test_df):,} rows")
    print(f"Features     : {len(bundle.feature_cols)}")
    if excluded_features:
        print(f"Excluded     : {len(excluded_features)} features")

    model, train_meta = train_model(bundle, args)
    evaluate_and_save(model, bundle, args, train_meta, models_dir, reports_dir, features_dir)


if __name__ == "__main__":
    main()
