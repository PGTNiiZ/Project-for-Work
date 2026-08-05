"""
Profile image feature pipeline for pairwise identity resolution.

This module adapts the ArcFace + CLIP + pHash pipeline to the local project
layout:
  - profiles come from data_for_project/nomalized_profiles.csv
  - images are reused from image_process/downloaded_images
  - pairs come from candidate/exact/labeled pair CSVs with profile_id_a/b

Typical usage:
  1. Precompute profile-level cache once
     python profile_image_pair_features.py precompute

  2. Build pair-level image features from a pair CSV
     python profile_image_pair_features.py pair-features \
       --pairs-csv ..\data_for_project\candidate_pairs_advanced.csv \
       --output-csv ..\data_for_project\candidate_pairs_advanced_with_image_features.csv
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_PROFILES_CSV = PROJECT_DIR / "data_for_project" / "nomalized_profiles.csv"
DEFAULT_PAIRS_CSV = PROJECT_DIR / "data_for_project" / "candidate_pairs_advanced.csv"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "downloaded_images"
DEFAULT_CACHE_PATH = SCRIPT_DIR / "image_cache.pkl"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data_for_project" / "candidate_pairs_advanced_with_image_features.csv"

DEFAULT_PROFILE_ID_COL = "profile_id"
DEFAULT_URL_COL = "pictureURL"
DEFAULT_BIO_COL = "bio"

CLIP_CATEGORIES = [
    "a selfie photo of a person",
    "a photo of a pet or animal",
    "a logo or brand icon",
    "a landscape or nature photo",
    "a food or drink photo",
    "an abstract or artistic image",
    "a group photo with multiple people",
]
CATEGORY_LABELS = [
    "selfie",
    "pet",
    "logo",
    "landscape",
    "food",
    "abstract",
    "group",
]
CATEGORY_KEYWORDS = {
    "selfie": {"selfie", "portrait", "face", "me", "myself", "photographer", "artist"},
    "pet": {"dog", "cat", "pet", "puppy", "kitten", "animal"},
    "logo": {"brand", "official", "company", "studio", "agency", "shop", "store"},
    "landscape": {"travel", "nature", "outdoors", "mountain", "beach", "landscape"},
    "food": {"food", "coffee", "chef", "drink", "baker", "restaurant"},
    "abstract": {"design", "art", "abstract", "illustration", "creative"},
    "group": {"team", "family", "friends", "community", "band", "crew"},
}

_deepface = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_imagehash = None
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace

        _deepface = DeepFace
    return _deepface


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip

        torch = _get_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=device,
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_imagehash():
    global _imagehash
    if _imagehash is None:
        import imagehash

        _imagehash = imagehash
    return _imagehash


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:8]


def build_image_hash_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not image_dir.exists():
        return index
    for path in image_dir.iterdir():
        if not path.is_file() or "_" not in path.stem:
            continue
        image_hash = path.stem.split("_")[-1]
        index.setdefault(image_hash, path)
    return index


def resolve_local_image_path(picture_url: str, image_hash_index: dict[str, Path]) -> Path | None:
    if not picture_url or not isinstance(picture_url, str):
        return None
    key = get_url_hash(picture_url.strip())
    return image_hash_index.get(key)


def normalize_bio_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def bio_mentions_selfie(text: str) -> float:
    bio = normalize_bio_text(text)
    if not bio:
        return 0.0
    keywords = {"selfie", "portrait", "photographer", "model", "actor", "artist"}
    return float(any(word in bio for word in keywords))


def clip_bio_consistency(category: str | None, bio: str) -> float:
    if not category:
        return 0.0
    bio_norm = normalize_bio_text(bio)
    if not bio_norm:
        return 0.0
    keywords = CATEGORY_KEYWORDS.get(category, set())
    return float(any(word in bio_norm for word in keywords))


class ImageFeatureExtractor:
    def __init__(self, device: str = "auto"):
        torch = _get_torch()
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._clip_text_embeddings = None

    def precompute_from_path(self, image_path: Path | None, bio: str = "", picture_url: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {
            "url": picture_url,
            "image_path": str(image_path) if image_path else None,
            "available": False,
            "arcface_embedding": None,
            "clip_embedding": None,
            "phash": None,
            "clip_category": None,
            "has_face": False,
            "has_pet": False,
            "has_logo": False,
            "bio_mentions_selfie": bio_mentions_selfie(bio),
            "clip_bio_consistency": 0.0,
        }

        img = self._load_image(image_path)
        if img is None:
            return result

        result["available"] = True
        result["arcface_embedding"], result["has_face"] = self._extract_arcface(img)
        result["clip_embedding"], result["clip_category"] = self._extract_clip(img)
        result["phash"] = self._compute_phash(img)
        result["has_pet"] = result["clip_category"] == "pet"
        result["has_logo"] = result["clip_category"] == "logo"
        result["clip_bio_consistency"] = clip_bio_consistency(result["clip_category"], bio)
        return result

    def compute_pair_features(self, cache_a: dict[str, Any], cache_b: dict[str, Any]) -> dict[str, Any]:
        av_a = bool(cache_a.get("available", False))
        av_b = bool(cache_b.get("available", False))
        feats: dict[str, Any] = {
            "pic_available_a": float(av_a),
            "pic_available_b": float(av_b),
            "pic_available_both": float(av_a and av_b),
            "modality_mask": float(int(av_a) + int(av_b)),
        }

        feats.update(self._feat_arcface(cache_a, cache_b))
        feats.update(self._feat_clip(cache_a, cache_b))
        feats.update(self._feat_phash(cache_a, cache_b))

        feats["has_face_a"] = float(cache_a.get("has_face", False))
        feats["has_face_b"] = float(cache_b.get("has_face", False))
        feats["has_pet_a"] = float(cache_a.get("has_pet", False))
        feats["has_pet_b"] = float(cache_b.get("has_pet", False))
        feats["has_logo_a"] = float(cache_a.get("has_logo", False))
        feats["has_logo_b"] = float(cache_b.get("has_logo", False))

        cat_a = cache_a.get("clip_category") or "unknown"
        cat_b = cache_b.get("clip_category") or "unknown"
        feats["clip_category_a"] = cat_a
        feats["clip_category_b"] = cat_b
        feats["clip_category_match"] = float(cat_a == cat_b and cat_a != "unknown" and av_a and av_b)
        feats["pic_type_match"] = feats["clip_category_match"]

        feats["bio_mentions_selfie_a"] = float(cache_a.get("bio_mentions_selfie", 0.0))
        feats["bio_mentions_selfie_b"] = float(cache_b.get("bio_mentions_selfie", 0.0))
        feats["clip_bio_consistency_a"] = float(cache_a.get("clip_bio_consistency", 0.0))
        feats["clip_bio_consistency_b"] = float(cache_b.get("clip_bio_consistency", 0.0))
        feats["clip_bio_consistency"] = max(
            feats["clip_bio_consistency_a"],
            feats["clip_bio_consistency_b"],
        )
        return feats

    def _load_image(self, image_path: Path | None):
        if image_path is None or not image_path.exists():
            return None
        from PIL import Image

        try:
            with Image.open(image_path) as img:
                pil_img = img.convert("RGB")
                pil_img.thumbnail((512, 512), Image.LANCZOS)
                return pil_img.copy()
        except Exception:
            return None

    def _extract_arcface(self, img) -> tuple[np.ndarray | None, bool]:
        try:
            DeepFace = _get_deepface()
            img_np = np.array(img)
            result = DeepFace.represent(
                img_path=img_np,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )
            if result and len(result) > 0:
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb, True
        except Exception:
            pass
        return None, False

    def _get_clip_category_embeddings(self):
        if self._clip_text_embeddings is not None:
            return self._clip_text_embeddings

        torch = _get_torch()
        model, _, tokenizer = _get_clip()
        with torch.no_grad():
            tokens = tokenizer(CLIP_CATEGORIES).to(self.device)
            text_embs = model.encode_text(tokens)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            self._clip_text_embeddings = text_embs.cpu().float().numpy()
        return self._clip_text_embeddings

    def _extract_clip(self, img) -> tuple[np.ndarray | None, str | None]:
        try:
            torch = _get_torch()
            model, preprocess, _ = _get_clip()
            tensor = preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_emb = model.encode_image(tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                img_emb_np = img_emb.cpu().float().numpy()[0]

            category_embeddings = self._get_clip_category_embeddings()
            scores = img_emb_np @ category_embeddings.T
            best_idx = int(np.argmax(scores))
            return img_emb_np, CATEGORY_LABELS[best_idx]
        except Exception:
            return None, None

    def _compute_phash(self, img):
        try:
            ih = _get_imagehash()
            return ih.phash(img)
        except Exception:
            return None

    def _feat_arcface(self, cache_a: dict[str, Any], cache_b: dict[str, Any]) -> dict[str, Any]:
        emb_a = cache_a.get("arcface_embedding")
        emb_b = cache_b.get("arcface_embedding")
        feats = {
            "face_detected_a": float(cache_a.get("has_face", False)),
            "face_detected_b": float(cache_b.get("has_face", False)),
            "both_face_detected": float(cache_a.get("has_face", False) and cache_b.get("has_face", False)),
        }
        if emb_a is not None and emb_b is not None:
            cosine = float(np.dot(emb_a, emb_b))
            cosine = max(-1.0, min(1.0, cosine))
            feats["face_arcface_cosine"] = cosine
            feats["face_arcface_match"] = float(cosine > 0.28)
        else:
            feats["face_arcface_cosine"] = 0.0
            feats["face_arcface_match"] = 0.0
        return feats

    def _feat_clip(self, cache_a: dict[str, Any], cache_b: dict[str, Any]) -> dict[str, Any]:
        emb_a = cache_a.get("clip_embedding")
        emb_b = cache_b.get("clip_embedding")
        if emb_a is not None and emb_b is not None:
            cosine = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
            cosine = max(-1.0, min(1.0, cosine))
        else:
            cosine = 0.0
        return {"clip_cosine_sim": cosine}

    def _feat_phash(self, cache_a: dict[str, Any], cache_b: dict[str, Any]) -> dict[str, Any]:
        h_a = cache_a.get("phash")
        h_b = cache_b.get("phash")
        if h_a is not None and h_b is not None:
            hamming = int(h_a - h_b)
            return {
                "phash_hamming": hamming,
                "phash_exact_match": float(hamming == 0),
                "phash_near_dup": float(hamming <= 10),
                "phash_sim": 1.0 - hamming / 64.0,
            }
        return {
            "phash_hamming": 64,
            "phash_exact_match": 0.0,
            "phash_near_dup": 0.0,
            "phash_sim": 0.0,
        }


def _import_pandas():
    import pandas as pd

    return pd


def load_profiles_df(csv_path: Path):
    pd = _import_pandas()
    return pd.read_csv(csv_path)


def load_pairs_df(csv_path: Path):
    pd = _import_pandas()
    return pd.read_csv(csv_path)


def precompute_all_profiles(
    df,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    url_col: str = DEFAULT_URL_COL,
    bio_col: str = DEFAULT_BIO_COL,
    pid_col: str = DEFAULT_PROFILE_ID_COL,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    force: bool = False,
) -> dict[Any, dict[str, Any]]:
    cache_path = Path(cache_path)
    existing: dict[Any, dict[str, Any]] = {}
    if cache_path.exists() and not force:
        with cache_path.open("rb") as fh:
            existing = pickle.load(fh)

    extractor = ImageFeatureExtractor()
    image_hash_index = build_image_hash_index(image_dir)
    result = dict(existing)

    for step, (_, row) in enumerate(df.iterrows(), start=1):
        pid = row.get(pid_col)
        if pid in result and not force:
            continue
        picture_url = str(row.get(url_col, "") or "")
        bio = str(row.get(bio_col, "") or "")
        image_path = resolve_local_image_path(picture_url, image_hash_index)
        cache = extractor.precompute_from_path(image_path=image_path, bio=bio, picture_url=picture_url)
        result[pid] = cache
        if step % 200 == 0:
            with cache_path.open("wb") as fh:
                pickle.dump(result, fh)

    with cache_path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def add_image_features_to_pairs(pairs_df, image_cache: dict[Any, dict[str, Any]], extractor: ImageFeatureExtractor | None = None):
    pd = _import_pandas()
    extractor = extractor or ImageFeatureExtractor()

    rows: list[dict[str, Any]] = []
    for _, pair in pairs_df.iterrows():
        pid_a = pair["profile_id_a"]
        pid_b = pair["profile_id_b"]
        cache_a = image_cache.get(pid_a, {"available": False})
        cache_b = image_cache.get(pid_b, {"available": False})
        rows.append(extractor.compute_pair_features(cache_a, cache_b))

    img_df = pd.DataFrame(rows)
    result = pairs_df.reset_index(drop=True).copy()
    for col in img_df.columns:
        result[col] = img_df[col].values
    return result


def run_precompute(args: argparse.Namespace) -> None:
    df = load_profiles_df(args.profiles_csv)
    cache = precompute_all_profiles(
        df=df,
        image_dir=args.image_dir,
        url_col=args.url_col,
        bio_col=args.bio_col,
        pid_col=args.profile_id_col,
        cache_path=args.cache_path,
        force=args.force,
    )
    available = sum(1 for value in cache.values() if value.get("available"))
    print(f"cache saved: {args.cache_path}")
    print(f"profiles cached: {len(cache)}")
    print(f"profiles with local images: {available}")


def run_pair_features(args: argparse.Namespace) -> None:
    pairs_df = load_pairs_df(args.pairs_csv)
    with Path(args.cache_path).open("rb") as fh:
        image_cache = pickle.load(fh)
    result = add_image_features_to_pairs(pairs_df, image_cache)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"pair features saved: {args.output_csv}")
    print(f"rows: {len(result)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile image feature pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    precompute_parser = subparsers.add_parser("precompute", help="Build profile-level image cache")
    precompute_parser.add_argument("--profiles-csv", type=Path, default=DEFAULT_PROFILES_CSV)
    precompute_parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    precompute_parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    precompute_parser.add_argument("--profile-id-col", default=DEFAULT_PROFILE_ID_COL)
    precompute_parser.add_argument("--url-col", default=DEFAULT_URL_COL)
    precompute_parser.add_argument("--bio-col", default=DEFAULT_BIO_COL)
    precompute_parser.add_argument("--force", action="store_true")
    precompute_parser.set_defaults(func=run_precompute)

    pair_parser = subparsers.add_parser("pair-features", help="Append image features to pair CSV")
    pair_parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    pair_parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    pair_parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    pair_parser.set_defaults(func=run_pair_features)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
