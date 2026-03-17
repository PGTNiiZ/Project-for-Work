from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from rapidfuzz import fuzz


SUITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUITE_DIR.parent.parent
BASE_PIPELINE_PATH = PROJECT_ROOT / "Train-Data" / "stage7_8_rebuilt_experiment_hybrid" / "run_rebuilt_pipeline.py"
PROFILES_PATH = PROJECT_ROOT / "data-for-project" / "normalized_profiles_with_profile_id.csv"
LOCAL_IMAGE_DIR = PROJECT_ROOT / "Image-Process" / "downloaded_images"
IMAGE_META_PATH = PROJECT_ROOT / "data" / "final" / "image_features_complete.csv"

RUNS_DIR = SUITE_DIR / "runs"
REPORTS_DIR = SUITE_DIR / "reports"
SHARED_CACHE_DIR = SUITE_DIR / "shared_cache"
SHARED_SBERT_PATH = SHARED_CACHE_DIR / "sbert_embeddings.npy"
SHARED_IMAGE_PROFILE_PATH = SHARED_CACHE_DIR / "image_profile_features.parquet"
SHARED_IMAGE_CAPTION_PATH = SHARED_CACHE_DIR / "image_caption_embeddings.npy"


def ensure_dirs() -> None:
    for path in [SUITE_DIR, RUNS_DIR, REPORTS_DIR, SHARED_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_base_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, BASE_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base pipeline from {BASE_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base_module(base, run_dir: Path) -> None:
    base.SCRIPT_DIR = run_dir
    base.PROJECT_ROOT = PROJECT_ROOT
    base.PROFILES_PATH = PROFILES_PATH
    base.ARTIFACTS_DIR = run_dir / "artifacts"
    base.MODELS_DIR = run_dir / "models"
    base.REPORTS_DIR = run_dir / "reports"
    base.SBERT_EMBEDDINGS_PATH = SHARED_SBERT_PATH
    base.ensure_dirs()


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def image_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0


def compute_local_image_descriptor(image_path: Path) -> dict[str, object]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        gray = rgb.convert("L")
        arr_rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        arr_gray = np.asarray(gray, dtype=np.float32)
        width, height = rgb.size
        brightness = float(arr_gray.mean() / 255.0)
        contrast = float(arr_gray.std() / 255.0)
        avg_rgb = arr_rgb.reshape(-1, 3).mean(axis=0)
        return {
            "local_image_path": str(image_path),
            "local_width": int(width),
            "local_height": int(height),
            "local_aspect_ratio": float(width / max(height, 1)),
            "local_file_size_kb": float(image_path.stat().st_size / 1024.0),
            "local_brightness": brightness,
            "local_contrast": contrast,
            "local_entropy": image_entropy(arr_gray),
            "local_avg_r": float(avg_rgb[0]),
            "local_avg_g": float(avg_rgb[1]),
            "local_avg_b": float(avg_rgb[2]),
            "local_phash": str(imagehash.phash(rgb)),
            "local_dhash": str(imagehash.dhash(rgb)),
            "has_local_image": 1,
        }


def scan_local_image_map() -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in LOCAL_IMAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        prefix = stem.split("_", 1)[0]
        try:
            row_id = int(prefix)
        except ValueError:
            continue
        mapping[row_id] = path
    return mapping


def build_image_profile_features(base, profiles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if SHARED_IMAGE_PROFILE_PATH.exists():
        image_df = pd.read_parquet(SHARED_IMAGE_PROFILE_PATH)
        if SHARED_IMAGE_CAPTION_PATH.exists():
            caption_embeddings = np.load(SHARED_IMAGE_CAPTION_PATH)
        else:
            caption_embeddings = np.zeros((0, 768), dtype=np.float32)
        return image_df, caption_embeddings

    local_map = scan_local_image_map()
    image_rows = []
    for row_id, image_path in sorted(local_map.items()):
        record = compute_local_image_descriptor(image_path)
        record["profile_row_id"] = row_id
        image_rows.append(record)
    local_df = pd.DataFrame(image_rows)
    if local_df.empty:
        local_df = pd.DataFrame(columns=[
            "profile_row_id", "local_image_path", "local_width", "local_height",
            "local_aspect_ratio", "local_file_size_kb", "local_brightness",
            "local_contrast", "local_entropy", "local_avg_r", "local_avg_g",
            "local_avg_b", "local_phash", "local_dhash", "has_local_image",
        ])

    meta_cols = [
        "profile_id",
        "platform",
        "blur_score",
        "num_faces",
        "face_confidence",
        "face_area_ratio",
        "caption_en",
        "status",
    ]
    meta_df = pd.read_csv(IMAGE_META_PATH, usecols=meta_cols, low_memory=False)
    meta_df = meta_df.rename(columns={"profile_id": "user_folder"})
    meta_df["user_folder"] = meta_df["user_folder"].map(clean_text)
    meta_df["platform"] = meta_df["platform"].map(lower_text)
    meta_df["caption_en"] = meta_df["caption_en"].map(lower_text)
    meta_df["has_image_metadata"] = 1
    meta_df["has_caption"] = meta_df["caption_en"].map(lambda x: 1 if x else 0)
    meta_df["status_done"] = meta_df["status"].map(lambda x: 1 if lower_text(x) == "done" else 0)

    profile_keys = profiles[["profile_row_id", "user_folder", "platform"]].copy()
    profile_keys["user_folder"] = profile_keys["user_folder"].map(clean_text)
    profile_keys["platform"] = profile_keys["platform"].map(lower_text)
    image_df = profile_keys.merge(meta_df, on=["user_folder", "platform"], how="left")
    image_df = image_df.merge(local_df, on="profile_row_id", how="left")

    fill_defaults = {
        "blur_score": 0.0,
        "num_faces": 0.0,
        "face_confidence": 0.0,
        "face_area_ratio": 0.0,
        "has_image_metadata": 0,
        "has_caption": 0,
        "status_done": 0,
        "has_local_image": 0,
        "local_width": 0.0,
        "local_height": 0.0,
        "local_aspect_ratio": 0.0,
        "local_file_size_kb": 0.0,
        "local_brightness": 0.0,
        "local_contrast": 0.0,
        "local_entropy": 0.0,
        "local_avg_r": 0.0,
        "local_avg_g": 0.0,
        "local_avg_b": 0.0,
        "caption_en": "",
        "local_image_path": "",
        "local_phash": "",
        "local_dhash": "",
    }
    for col, default in fill_defaults.items():
        image_df[col] = image_df[col].fillna(default)

    caption_mask = image_df["has_caption"].astype(bool)
    image_df["caption_emb_idx"] = -1
    caption_embeddings = np.zeros((0, 768), dtype=np.float32)
    if int(caption_mask.sum()) > 0:
        model_path = str(base.SBERT_LOCAL_PATH if base.SBERT_LOCAL_PATH.exists() else base.SBERT_MODEL_NAME)
        model = base.SentenceTransformer(model_path, local_files_only=True)
        caption_embeddings = model.encode(
            image_df.loc[caption_mask, "caption_en"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        image_df.loc[caption_mask, "caption_emb_idx"] = np.arange(len(caption_embeddings))
        np.save(SHARED_IMAGE_CAPTION_PATH, caption_embeddings)
    else:
        np.save(SHARED_IMAGE_CAPTION_PATH, caption_embeddings)

    image_df.to_parquet(SHARED_IMAGE_PROFILE_PATH, index=False)
    return image_df, caption_embeddings


def hash_similarity(hash_a: str, hash_b: str) -> float:
    if not hash_a or not hash_b:
        return 0.0
    distance = sum(ch1 != ch2 for ch1, ch2 in zip(hash_a, hash_b))
    return 1.0 - distance / max(len(hash_a), 1)


def make_multimodal_wrappers(base, experiment_kind: str, image_df: pd.DataFrame, caption_embeddings: np.ndarray):
    orig_make = base.make_profile_bundle
    orig_compute = base.compute_features
    image_lookup = {
        int(row["profile_row_id"]): row
        for row in image_df.to_dict("records")
    }

    def wrapped_make_profile_bundle(profiles: pd.DataFrame):
        bundle = orig_make(profiles)
        bundle["image_lookup"] = image_lookup
        bundle["caption_embeddings"] = caption_embeddings
        return bundle

    def wrapped_compute_features(pairs: pd.DataFrame, profile_bundle: dict[str, object]):
        df, _ = orig_compute(pairs, profile_bundle)
        records = profile_bundle["records"]
        bio_embeddings = profile_bundle["sbert_embeddings"]
        image_rows = []
        for pair in pairs.itertuples(index=False):
            rid_a = int(pair.profile_row_id_a)
            rid_b = int(pair.profile_row_id_b)
            rec_a = records[rid_a]
            rec_b = records[rid_b]
            img_a = profile_bundle["image_lookup"].get(rid_a)
            img_b = profile_bundle["image_lookup"].get(rid_b)
            has_local_a = int(img_a["has_local_image"]) if img_a else 0
            has_local_b = int(img_b["has_local_image"]) if img_b else 0
            has_meta_a = int(img_a["has_image_metadata"]) if img_a else 0
            has_meta_b = int(img_b["has_image_metadata"]) if img_b else 0
            both_local = int(has_local_a and has_local_b)
            any_local = int(has_local_a or has_local_b)
            one_local = int(any_local and not both_local)

            row = {
                "image_any_local": any_local,
                "image_both_local": both_local,
                "image_one_local_only": one_local,
            }

            if experiment_kind in {"image_stats", "image_context"}:
                row.update({
                    "image_phash_sim": hash_similarity(img_a["local_phash"], img_b["local_phash"]) if both_local else 0.0,
                    "image_dhash_sim": hash_similarity(img_a["local_dhash"], img_b["local_dhash"]) if both_local else 0.0,
                    "image_brightness_diff": abs(float(img_a["local_brightness"]) - float(img_b["local_brightness"])) if both_local else 0.0,
                    "image_contrast_diff": abs(float(img_a["local_contrast"]) - float(img_b["local_contrast"])) if both_local else 0.0,
                    "image_entropy_diff": abs(float(img_a["local_entropy"]) - float(img_b["local_entropy"])) if both_local else 0.0,
                    "image_rgb_l1": (
                        abs(float(img_a["local_avg_r"]) - float(img_b["local_avg_r"])) +
                        abs(float(img_a["local_avg_g"]) - float(img_b["local_avg_g"])) +
                        abs(float(img_a["local_avg_b"]) - float(img_b["local_avg_b"]))
                    ) / 3.0 if both_local else 0.0,
                    "image_filesize_ratio": (
                        min(float(img_a["local_file_size_kb"]), float(img_b["local_file_size_kb"])) /
                        max(float(img_a["local_file_size_kb"]), float(img_b["local_file_size_kb"]), 1e-6)
                    ) if both_local else 0.0,
                    "image_face_count_diff": abs(float(img_a["num_faces"]) - float(img_b["num_faces"])) if has_meta_a and has_meta_b else 0.0,
                    "image_face_area_diff": abs(float(img_a["face_area_ratio"]) - float(img_b["face_area_ratio"])) if has_meta_a and has_meta_b else 0.0,
                    "image_blur_diff": abs(float(img_a["blur_score"]) - float(img_b["blur_score"])) if has_meta_a and has_meta_b else 0.0,
                    "image_metadata_any": int(has_meta_a or has_meta_b),
                })

            if experiment_kind == "image_context":
                cap_idx_a = int(img_a["caption_emb_idx"]) if img_a else -1
                cap_idx_b = int(img_b["caption_emb_idx"]) if img_b else -1
                cap_text_a = img_a["caption_en"] if img_a else ""
                cap_text_b = img_b["caption_en"] if img_b else ""
                bio_idx_a = rec_a["_tfidf_idx"]
                bio_idx_b = rec_b["_tfidf_idx"]
                cap_bio_a = float(np.dot(caption_embeddings[cap_idx_a], bio_embeddings[bio_idx_b])) if cap_idx_a >= 0 else 0.0
                cap_bio_b = float(np.dot(caption_embeddings[cap_idx_b], bio_embeddings[bio_idx_a])) if cap_idx_b >= 0 else 0.0
                row.update({
                    "image_caption_any": int((cap_idx_a >= 0) or (cap_idx_b >= 0)),
                    "image_caption_bio_sbert_cross": max(cap_bio_a, cap_bio_b),
                    "image_caption_fullname_token_cross": max(
                        fuzz.token_sort_ratio(cap_text_a, rec_b["fullName_norm"]) / 100.0 if cap_text_a else 0.0,
                        fuzz.token_sort_ratio(cap_text_b, rec_a["fullName_norm"]) / 100.0 if cap_text_b else 0.0,
                    ),
                    "image_caption_username_token_cross": max(
                        fuzz.token_sort_ratio(cap_text_a, rec_b["userName_norm"]) / 100.0 if cap_text_a else 0.0,
                        fuzz.token_sort_ratio(cap_text_b, rec_a["userName_norm"]) / 100.0 if cap_text_b else 0.0,
                    ),
                })
            image_rows.append(row)

        image_feature_df = pd.DataFrame(image_rows)
        df = pd.concat([df.reset_index(drop=True), image_feature_df], axis=1)
        feature_cols = [col for col in df.columns if col not in base.KEY_COLS]
        return df, feature_cols

    return wrapped_make_profile_bundle, wrapped_compute_features


def summarize_modalities(profiles: pd.DataFrame, image_df: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, object]:
    valid_profiles = profiles[profiles["profile_id"].notna()].copy()
    valid_rids = set(valid_profiles["profile_row_id"].astype(int).tolist())
    image_df = image_df[image_df["profile_row_id"].isin(valid_rids)].copy()
    pair_summary = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_df[feature_df["split_name"] == split_name]
        pair_summary[split_name] = {
            "rows": int(len(split_df)),
            "pairs_with_any_local_image": int(split_df.get("image_any_local", pd.Series(dtype=int)).sum()) if "image_any_local" in split_df else 0,
            "pairs_with_both_local_images": int(split_df.get("image_both_local", pd.Series(dtype=int)).sum()) if "image_both_local" in split_df else 0,
            "pairs_with_caption_signal": int(split_df.get("image_caption_any", pd.Series(dtype=int)).sum()) if "image_caption_any" in split_df else 0,
        }
    return {
        "profile_coverage": {
            "valid_profiles": int(len(valid_profiles)),
            "profiles_with_local_image": int(image_df["has_local_image"].astype(int).sum()),
            "profiles_with_image_metadata": int(image_df["has_image_metadata"].astype(int).sum()),
            "profiles_with_caption": int(image_df["has_caption"].astype(int).sum()),
            "platform_counts_with_local_image": (
                valid_profiles.merge(image_df[["profile_row_id", "has_local_image"]], on="profile_row_id", how="left")
                .fillna({"has_local_image": 0})
                .query("has_local_image == 1")["platform"]
                .value_counts()
                .to_dict()
            ),
        },
        "pair_coverage": pair_summary,
    }


def composite_score(metrics: dict[str, object]) -> float:
    return (
        0.45 * float(metrics["val_avg_precision"]) +
        0.35 * float(metrics["val_f1"]) +
        0.20 * float(metrics["val_roc_auc"])
    )


def run_experiment(run_name: str, experiment_kind: str, args: argparse.Namespace) -> dict[str, object]:
    run_dir = RUNS_DIR / run_name
    base = load_base_module(f"stage7_8_rebuilt_experiment_hybrid_{run_name}")
    configure_base_module(base, run_dir)
    base.set_seed(args.seed)

    profiles = base.load_profiles()
    image_df, caption_embeddings = build_image_profile_features(base, profiles)

    if experiment_kind != "text_attr_hybrid":
        wrapped_make, wrapped_compute = make_multimodal_wrappers(base, experiment_kind, image_df, caption_embeddings)
        base.make_profile_bundle = wrapped_make
        base.compute_features = wrapped_compute

    feature_df, feature_cols, leakage = base.build_dataset(profiles, args)
    metrics = base.train_and_eval(feature_df, feature_cols, seed=args.seed)
    modality_summary = summarize_modalities(profiles, image_df, feature_df)

    report = {
        "config": {
            "run_name": run_name,
            "experiment_kind": experiment_kind,
            "random_neg_ratio": args.random_neg_ratio,
            "hard_neg_ratio": args.hard_neg_ratio,
            "seed": args.seed,
        },
        "metrics": metrics,
        "leakage_report": leakage,
        "modality_report": modality_summary,
        "paths": {
            "run_dir": str(run_dir),
            "artifacts_dir": str(base.ARTIFACTS_DIR),
            "models_dir": str(base.MODELS_DIR),
            "reports_dir": str(base.REPORTS_DIR),
        },
    }
    (base.REPORTS_DIR / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (base.REPORTS_DIR / "feature_list.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    (base.REPORTS_DIR / "modality_manifest.json").write_text(json.dumps(modality_summary, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "experiment_kind": experiment_kind,
        "feature_count": len(feature_cols),
        "composite_score": composite_score(metrics),
        "metrics": metrics,
        "report_path": str(base.REPORTS_DIR / "experiment_report.json"),
        "run_dir": str(run_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-safe multimodal suite with TF-IDF + SBERT + optional image branch")
    parser.add_argument("--random-neg-ratio", type=float, default=0.75)
    parser.add_argument("--hard-neg-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["text_attr_hybrid", "image_stats", "image_context"],
        choices=["text_attr_hybrid", "image_stats", "image_context"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    results = []
    for experiment_kind in args.experiments:
        run_name = f"{experiment_kind}_r{str(args.random_neg_ratio).replace('.', '')}_h{str(args.hard_neg_ratio).replace('.', '')}_s{args.seed}"
        print("=" * 72)
        print(f"Running experiment: {run_name}")
        print("=" * 72)
        results.append(run_experiment(run_name, experiment_kind, args))

    baseline = next((r for r in results if r["experiment_kind"] == "text_attr_hybrid"), None)
    leaderboard_rows = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "run_name": result["run_name"],
            "experiment_kind": result["experiment_kind"],
            "feature_count": result["feature_count"],
            "composite_score": result["composite_score"],
            "val_avg_precision": metrics["val_avg_precision"],
            "val_roc_auc": metrics["val_roc_auc"],
            "val_f1": metrics["val_f1"],
            "test_avg_precision": metrics["test_avg_precision"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "report_path": result["report_path"],
        }
        if baseline is not None:
            base_metrics = baseline["metrics"]
            row["delta_test_avg_precision"] = metrics["test_avg_precision"] - base_metrics["test_avg_precision"]
            row["delta_test_roc_auc"] = metrics["test_roc_auc"] - base_metrics["test_roc_auc"]
            row["delta_test_f1"] = metrics["test_f1"] - base_metrics["test_f1"]
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["composite_score", "test_avg_precision", "test_f1"], ascending=False)
    leaderboard.to_csv(REPORTS_DIR / "leaderboard.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    best_row = leaderboard.iloc[0].to_dict()
    suite_report = {
        "config": vars(args),
        "best_run": best_row,
        "runs": leaderboard_rows,
        "shared_cache": {
            "sbert_embeddings": str(SHARED_SBERT_PATH),
            "image_profile_features": str(SHARED_IMAGE_PROFILE_PATH),
            "image_caption_embeddings": str(SHARED_IMAGE_CAPTION_PATH),
        },
    }
    (REPORTS_DIR / "suite_report.json").write_text(json.dumps(suite_report, indent=2), encoding="utf-8")

    print("\nSuite completed")
    print(f"Leaderboard : {REPORTS_DIR / 'leaderboard.csv'}")
    print(f"Suite report : {REPORTS_DIR / 'suite_report.json'}")
    print(f"Best run     : {best_row['run_name']}")
    print(f"Best Test AP : {best_row['test_avg_precision']:.4f}")
    print(f"Best Test AUC: {best_row['test_roc_auc']:.4f}")
    print(f"Best Test F1 : {best_row['test_f1']:.4f}")


if __name__ == "__main__":
    main()
