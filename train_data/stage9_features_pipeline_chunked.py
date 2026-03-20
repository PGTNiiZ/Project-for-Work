from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import pickle
import re
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data_for_project"
IMAGE_PROCESS_DIR = PROJECT_DIR / "image_process"
PIPELINE_ROOT_DIR = SCRIPT_DIR / "stage9_pipeline_chunked"
ARTIFACTS_DIR = PIPELINE_ROOT_DIR / "artifacts"
FEATURES_DIR = PIPELINE_ROOT_DIR / "features"
CACHE_DIR = PIPELINE_ROOT_DIR / "cache"
IMAGES_DIR = PIPELINE_ROOT_DIR / "images"

PROFILES_PATH = DATA_DIR / "nomalized_profiles.csv"
PAIRS_PATH = SCRIPT_DIR / "labeled_pairs.parquet"
IMAGE_CACHE_PATH = CACHE_DIR / "image_cache.pkl"
DOWNLOADED_IMAGES_DIR = IMAGES_DIR / "downloaded_images"
SBERT_EMBEDDINGS_PATH = ARTIFACTS_DIR / "sbert_embeddings.npy"
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.pkl"
SPLIT_ARTIFACTS = {
    "train_df": ARTIFACTS_DIR / "train_profiles.parquet",
    "val_df": ARTIFACTS_DIR / "val_profiles.parquet",
    "test_df": ARTIFACTS_DIR / "test_profiles.parquet",
    "train_pairs": ARTIFACTS_DIR / "train_pairs.parquet",
    "val_pairs": ARTIFACTS_DIR / "val_pairs.parquet",
    "test_pairs": ARTIFACTS_DIR / "test_pairs.parquet",
}
PROFILE_CACHE_PATHS = {
    "train": ARTIFACTS_DIR / "train_profile_cache.pkl",
    "val": ARTIFACTS_DIR / "val_profile_cache.pkl",
    "test": ARTIFACTS_DIR / "test_profile_cache.pkl",
}


GENERIC_EXTERNAL_DOMAINS = {
    "about.me", "twitter.com", "facebook.com", "instagram.com", "youtube.com",
    "linkedin.com", "plus.google.com", "google.com", "bit.ly", "goo.gl",
    "youtu.be", "fb.com", "t.co", "ow.ly",
}

PLATFORM_PAIR_CODES = {
    tuple(sorted(["googleplus", "instagram"])): 0,
    tuple(sorted(["googleplus", "twitter"])): 1,
    tuple(sorted(["instagram", "twitter"])): 2,
}


def _load_profile_image_module():
    module_path = IMAGE_PROCESS_DIR / "profile_image_pair_features.py"
    spec = importlib.util.spec_from_file_location("profile_image_pair_features", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMG_MOD = _load_profile_image_module()

_rapidfuzz_jw = None
_rapidfuzz_lev = None
_rapidfuzz_fuzz = None


def _get_rapidfuzz():
    global _rapidfuzz_jw, _rapidfuzz_lev, _rapidfuzz_fuzz
    if _rapidfuzz_jw is None:
        from rapidfuzz.distance import JaroWinkler, Levenshtein as Lev
        from rapidfuzz import fuzz

        _rapidfuzz_jw = JaroWinkler
        _rapidfuzz_lev = Lev
        _rapidfuzz_fuzz = fuzz
    return _rapidfuzz_jw, _rapidfuzz_lev, _rapidfuzz_fuzz


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def _safe_str_lower(val) -> str:
    return _safe_str(val).lower()


def _parse_list_field(val) -> list[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s in {"nan", "NaN"}:
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [s]


def _parse_mentions(val) -> set[str]:
    s = _safe_str(val)
    if not s:
        return set()
    return {p.strip().lower() for p in re.split(r"\s*\|\s*|\s+", s) if p.strip()}


def _extract_hashtags(bio: str) -> set[str]:
    if not bio:
        return set()
    return {tag.lower() for tag in re.findall(r"#(\w+)", bio)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_picture_url(url: str) -> str:
    return _safe_str(url)


def _picture_stem(url: str) -> str:
    parsed = urlparse(_normalize_picture_url(url))
    stem = Path(parsed.path).stem.lower()
    return re.sub(r"_(400x400|200x200|bigger|normal)$", "", stem)


def _read_parquet_compat(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if exc.__class__.__name__ not in {"ArrowKeyError", "ImportError"}:
            raise
        import pyarrow.parquet as pq

        print(f"[IO] pandas.read_parquet failed for {path.name}; falling back to pyarrow.")
        return pq.read_table(path).to_pandas()


def _write_parquet_compat(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        if exc.__class__.__name__ not in {"ArrowKeyError", "ImportError"}:
            raise
        import pyarrow as pa
        import pyarrow.parquet as pq

        print(f"[IO] pandas.to_parquet failed for {path.name}; falling back to pyarrow.")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def ensure_output_dirs() -> None:
    for path in [PIPELINE_ROOT_DIR, ARTIFACTS_DIR, FEATURES_DIR, CACHE_DIR, IMAGES_DIR, DOWNLOADED_IMAGES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _find_resume_offset(split_name: str, output_prefix: str) -> int:
    split_output_dir = FEATURES_DIR / split_name
    if not split_output_dir.exists():
        return 0
    pattern = re.compile(rf"^{re.escape(output_prefix)}_{re.escape(split_name)}_(\d{{9}})_(\d{{9}})\.parquet$")
    max_end = 0
    for path in split_output_dir.glob(f"{output_prefix}_{split_name}_*.parquet"):
        match = pattern.match(path.name)
        if match:
            max_end = max(max_end, int(match.group(2)))
    return max_end


def save_split_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> None:
    _write_parquet_compat(train_df, SPLIT_ARTIFACTS["train_df"])
    _write_parquet_compat(val_df, SPLIT_ARTIFACTS["val_df"])
    _write_parquet_compat(test_df, SPLIT_ARTIFACTS["test_df"])
    _write_parquet_compat(train_pairs, SPLIT_ARTIFACTS["train_pairs"])
    _write_parquet_compat(val_pairs, SPLIT_ARTIFACTS["val_pairs"])
    _write_parquet_compat(test_pairs, SPLIT_ARTIFACTS["test_pairs"])


def load_split_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in SPLIT_ARTIFACTS.values() if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing split artifacts: {missing_names}. Run --stage split first.")
    return (
        _read_parquet_compat(SPLIT_ARTIFACTS["train_df"]),
        _read_parquet_compat(SPLIT_ARTIFACTS["val_df"]),
        _read_parquet_compat(SPLIT_ARTIFACTS["test_df"]),
        _read_parquet_compat(SPLIT_ARTIFACTS["train_pairs"]),
        _read_parquet_compat(SPLIT_ARTIFACTS["val_pairs"]),
        _read_parquet_compat(SPLIT_ARTIFACTS["test_pairs"]),
    )


def build_profile_lookup(split_df: pd.DataFrame) -> dict[str, dict]:
    src = split_df.reset_index(drop=True)
    src_records = []
    for idx, row in enumerate(src.to_dict("records")):
        source_idx = row.get("_source_idx", idx)
        src_records.append(_build_row_cache(row, int(source_idx)))
    return {str(row["profile_id"]): row for row in src_records}


def save_profile_cache(split_name: str, split_df: pd.DataFrame) -> None:
    lookup = build_profile_lookup(split_df)
    with PROFILE_CACHE_PATHS[split_name].open("wb") as fh:
        pickle.dump(lookup, fh)


def load_profile_cache(split_name: str) -> dict[str, dict]:
    path = PROFILE_CACHE_PATHS[split_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing profile cache: {path.name}. Run --stage profile-cache first.")
    with path.open("rb") as fh:
        return pickle.load(fh)


def merge_feature_chunks(split_name: str, output_prefix: str) -> Path:
    split_output_dir = FEATURES_DIR / split_name
    if not split_output_dir.exists():
        raise FileNotFoundError(f"Missing feature directory for split '{split_name}'. Run --stage features first.")

    pattern = re.compile(rf"^{re.escape(output_prefix)}_{re.escape(split_name)}_(\d{{9}})_(\d{{9}})\.parquet$")
    chunk_paths: list[tuple[int, Path]] = []
    for path in split_output_dir.glob(f"{output_prefix}_{split_name}_*.parquet"):
        match = pattern.match(path.name)
        if match:
            chunk_paths.append((int(match.group(1)), path))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found for split '{split_name}'.")

    chunk_paths.sort(key=lambda item: item[0])
    frames = [_read_parquet_compat(path) for _, path in chunk_paths]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged_path = FEATURES_DIR / f"{output_prefix}_{split_name}_merged.parquet"
    _write_parquet_compat(merged, merged_path)
    return merged_path


def _build_row_cache(row: dict, source_idx: int) -> dict:
    url = _normalize_picture_url(row.get("pictureURL", ""))
    parsed = urlparse(url) if url else None
    bio = _safe_str(row.get("bio", ""))
    letters = [c for c in bio if c.isalpha()]
    words = bio.split()
    return {
        **row,
        "__idx": int(source_idx),
        "_domains": set(_parse_list_field(row.get("external_domain", ""))) - GENERIC_EXTERNAL_DOMAINS,
        "_urls": set(_parse_list_field(row.get("externalUrl_clean", ""))),
        "_location_type": _safe_str_lower(row.get("location_type", "unknown")),
        "_location_text": _safe_str_lower(row.get("location", "")),
        "_location_valid": int(float(row.get("location_valid", 0) or 0)),
        "_mentions": _parse_mentions(row.get("bio_mentions", "")),
        "_mentions_count": int(float(row.get("bio_mentions_count", 0) or 0)),
        "_username_lower": _safe_str_lower(row.get("userName", "")),
        "_hashtags": _extract_hashtags(bio),
        "_style_stats": {
            "caps": sum(c.isupper() for c in letters) / max(len(letters), 1) if bio else 0.0,
            "avgw": float(np.mean([len(w) for w in words])) if words else 0.0,
            "blen": len(bio),
            "pct": sum(1 for c in bio if c in "!?.,;:-_()[]{}") / max(len(bio), 1) if bio else 0.0,
        },
        "_platform": _safe_str_lower(row.get("platform", "")),
        "_picture_url": url,
        "_picture_host": (parsed.netloc or "").lower() if parsed else "",
        "_picture_stem": _picture_stem(url) if url else "",
        "_picture_hash": IMG_MOD.get_url_hash(url) if url else "",
    }


class PairFeatureComputer:
    @staticmethod
    def feat_B1_text(record_a: dict, record_b: dict, tfidf_matrix=None, idx_a: int | None = None, idx_b: int | None = None) -> dict:
        JW, Lev, fuzz = _get_rapidfuzz()
        feats = {}
        for field in ["userName", "fullName"]:
            a = _safe_str(record_a.get(field, ""))
            b = _safe_str(record_b.get(field, ""))
            max_len = max(len(a), len(b), 1)
            feats[f"{field}_jaro"] = JW.normalized_similarity(a, b)
            feats[f"{field}_lev"] = 1.0 - Lev.distance(a, b) / max_len
            feats[f"{field}_token_sort"] = fuzz.token_sort_ratio(a, b) / 100.0
            feats[f"{field}_exact"] = float(a == b and a != "")

        if tfidf_matrix is not None and idx_a is not None and idx_b is not None:
            try:
                cosine = tfidf_matrix[idx_a].multiply(tfidf_matrix[idx_b]).sum()
                feats["bio_tfidf_cosine"] = float(cosine)
            except Exception:
                feats["bio_tfidf_cosine"] = 0.0
        else:
            feats["bio_tfidf_cosine"] = 0.0
        return feats

    @staticmethod
    def feat_A1_url(record_a: dict, record_b: dict) -> dict:
        dom_a = record_a.get("_domains", set())
        dom_b = record_b.get("_domains", set())
        urls_a = record_a.get("_urls", set())
        urls_b = record_b.get("_urls", set())
        return {
            "domain_jaccard": _jaccard(dom_a, dom_b),
            "domain_exact_match": float(bool(dom_a & dom_b)),
            "url_exact_match": float(bool(urls_a & urls_b)),
            "url_jaccard": _jaccard(urls_a, urls_b),
            "domain_count_a": min(len(dom_a), 10) / 10.0,
            "domain_count_b": min(len(dom_b), 10) / 10.0,
        }

    @staticmethod
    def feat_A2_location(record_a: dict, record_b: dict) -> dict:
        JW, _, fuzz = _get_rapidfuzz()
        lt_a = record_a.get("_location_type", _safe_str_lower(record_a.get("location_type", "unknown")))
        lt_b = record_b.get("_location_type", _safe_str_lower(record_b.get("location_type", "unknown")))
        la_a = float(record_a.get("latitude", 0.0) or 0.0)
        lo_a = float(record_a.get("longitude", 0.0) or 0.0)
        la_b = float(record_b.get("latitude", 0.0) or 0.0)
        lo_b = float(record_b.get("longitude", 0.0) or 0.0)
        coord_types = {"coordinates", "coordinates_dms"}

        if lt_a in coord_types and lt_b in coord_types and abs(la_a) > 0.001 and abs(la_b) > 0.001:
            dist = _haversine_km(la_a, lo_a, la_b, lo_b)
            coord_sim = float(np.exp(-dist / 100.0))
            same_city = float(dist < 50.0)
        else:
            coord_sim = 0.0
            same_city = 0.0

        loc_a = record_a.get("_location_text", _safe_str_lower(record_a.get("location", "")))
        loc_b = record_b.get("_location_text", _safe_str_lower(record_b.get("location", "")))
        if loc_a and loc_b:
            tj = JW.normalized_similarity(loc_a, loc_b)
            tt = fuzz.token_sort_ratio(loc_a, loc_b) / 100.0
            te = float(loc_a == loc_b)
        else:
            tj = 0.0
            tt = 0.0
            te = 0.0

        lv_a = record_a.get("_location_valid", int(float(record_a.get("location_valid", 0) or 0)))
        lv_b = record_b.get("_location_valid", int(float(record_b.get("location_valid", 0) or 0)))
        return {
            "location_coord_sim": coord_sim,
            "location_same_city": same_city,
            "location_text_jaro": tj,
            "location_text_token": tt,
            "location_text_exact": te,
            "both_have_location": float(lv_a == 1 and lv_b == 1),
        }

    @staticmethod
    def feat_A3_mention(record_a: dict, record_b: dict) -> dict:
        mentions_a = record_a.get("_mentions", set())
        mentions_b = record_b.get("_mentions", set())
        user_a = record_a.get("_username_lower", _safe_str_lower(record_a.get("userName", "")))
        user_b = record_b.get("_username_lower", _safe_str_lower(record_b.get("userName", "")))
        count_a = record_a.get("_mentions_count", int(float(record_a.get("bio_mentions_count", 0) or 0)))
        count_b = record_b.get("_mentions_count", int(float(record_b.get("bio_mentions_count", 0) or 0)))
        return {
            "mention_jaccard": _jaccard(mentions_a, mentions_b),
            "mention_exact_overlap": float(bool(mentions_a & mentions_b)),
            "both_have_mentions": float(count_a > 0 and count_b > 0),
            "username_in_other_mentions": float(
                (user_a != "" and user_a in mentions_b) or (user_b != "" and user_b in mentions_a)
            ),
        }

    @staticmethod
    def feat_A4_hashtag(record_a: dict, record_b: dict) -> dict:
        tags_a = record_a.get("_hashtags", set())
        tags_b = record_b.get("_hashtags", set())
        return {
            "hashtag_jaccard": _jaccard(tags_a, tags_b),
            "hashtag_exact_overlap": float(bool(tags_a & tags_b)),
            "both_have_hashtags": float(bool(tags_a) and bool(tags_b)),
            "hashtag_count_a": min(len(tags_a), 10) / 10.0,
            "hashtag_count_b": min(len(tags_b), 10) / 10.0,
        }

    @staticmethod
    def feat_B2_sbert(idx_a: int, idx_b: int, sbert_embeddings) -> dict:
        if sbert_embeddings is None:
            return {"bio_sbert_cosine": 0.0}
        try:
            emb_a = sbert_embeddings[idx_a]
            emb_b = sbert_embeddings[idx_b]
            return {"bio_sbert_cosine": float(np.dot(emb_a, emb_b))}
        except Exception:
            return {"bio_sbert_cosine": 0.0}

    @staticmethod
    def feat_B3_stylometric(record_a: dict, record_b: dict) -> dict:
        stats_a = record_a.get("_style_stats")
        stats_b = record_b.get("_style_stats")
        if stats_a is None or stats_b is None:
            raise ValueError("Missing cached stylometric stats.")
        return {
            "style_caps_diff": abs(stats_a["caps"] - stats_b["caps"]),
            "style_avgword_diff": abs(stats_a["avgw"] - stats_b["avgw"]) / 10.0,
            "style_biolen_ratio": min(stats_a["blen"], stats_b["blen"]) / max(stats_a["blen"], stats_b["blen"], 1),
            "style_punct_diff": abs(stats_a["pct"] - stats_b["pct"]),
        }

    @staticmethod
    def feat_meta(record_a: dict, record_b: dict) -> dict:
        plat_a = record_a.get("_platform", _safe_str_lower(record_a.get("platform", "")))
        plat_b = record_b.get("_platform", _safe_str_lower(record_b.get("platform", "")))
        return {
            "same_platform": float(plat_a == plat_b and plat_a != ""),
            "platform_pair": PLATFORM_PAIR_CODES.get(tuple(sorted([plat_a, plat_b])), 3),
        }

    @staticmethod
    def feat_picture_url(record_a: dict, record_b: dict) -> dict:
        url_a = record_a.get("_picture_url", _normalize_picture_url(record_a.get("pictureURL", "")))
        url_b = record_b.get("_picture_url", _normalize_picture_url(record_b.get("pictureURL", "")))
        host_a = record_a.get("_picture_host", "")
        host_b = record_b.get("_picture_host", "")
        stem_a = record_a.get("_picture_stem", "")
        stem_b = record_b.get("_picture_stem", "")
        hash_a = record_a.get("_picture_hash", "")
        hash_b = record_b.get("_picture_hash", "")
        return {
            "picture_url_available_a": float(bool(url_a)),
            "picture_url_available_b": float(bool(url_b)),
            "picture_url_available_both": float(bool(url_a) and bool(url_b)),
            "picture_url_exact_match": float(url_a != "" and url_a == url_b),
            "picture_url_hash_match": float(hash_a != "" and hash_a == hash_b),
            "picture_url_host_match": float(host_a != "" and host_a == host_b),
            "picture_url_stem_match": float(stem_a != "" and stem_a == stem_b),
        }


def compute_sbert_embeddings(df: pd.DataFrame, model_name: str = "all-mpnet-base-v2", batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SBERT requires CUDA in this pipeline, but no GPU was found.")

    print(f"[SBERT] Loading {model_name} on cuda ...")
    model = SentenceTransformer(model_name, device="cuda")
    bios = df["bio"].fillna("").astype(str).tolist()
    print(f"[SBERT] Encoding {len(bios)} bios on GPU ...")
    return model.encode(
        bios,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


class FeatureEngineer:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[list[str]] = None,
        train_bios: Optional[pd.Series] = None,
        sbert_embeddings: Optional[np.ndarray] = None,
        image_cache: Optional[dict] = None,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.feature_groups = feature_groups or ["B1", "A1", "A2", "A3", "A4", "B3", "META", "PICURL"]
        self.sbert_embeddings = sbert_embeddings
        self.image_cache = image_cache or {}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

        if "B1" in self.feature_groups and train_bios is not None:
            print("[FE] Fitting TF-IDF on train bios only ...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )
            self.tfidf_vectorizer.fit(train_bios.fillna("").astype(str).tolist())
            print(f"[FE] TF-IDF vocab size: {len(self.tfidf_vectorizer.vocabulary_)}")

    def compute_pairs_chunked(
        self,
        labeled_pairs: pd.DataFrame,
        split_df: pd.DataFrame,
        split_name: str,
        output_prefix: str,
        profile_lookup: Optional[dict[str, dict]] = None,
        chunk_size: int = 100_000,
        pair_start: int = 0,
        max_pairs: int | None = None,
        skip_existing: bool = False,
        progress_every: int = 10_000,
        verbose: bool = True,
    ) -> tuple[int, Optional[pd.DataFrame]]:
        src = split_df.reset_index(drop=True)
        if profile_lookup is None:
            profile_lookup = build_profile_lookup(src)

        tfidf_mat = None
        if "B1" in self.feature_groups and self.tfidf_vectorizer is not None:
            tfidf_mat = self.tfidf_vectorizer.transform(src["bio"].fillna("").astype(str).tolist())

        total_available = len(labeled_pairs)
        pair_start = min(max(pair_start, 0), total_available)
        pair_stop = total_available if max_pairs is None else min(total_available, pair_start + max_pairs)
        total_pairs = max(pair_stop - pair_start, 0)
        img_ext = IMG_MOD.ImageFeatureExtractor() if "IMG" in self.feature_groups else None
        split_output_dir = FEATURES_DIR / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        first_chunk_df: Optional[pd.DataFrame] = None

        for chunk_start in range(pair_start, pair_stop, chunk_size):
            chunk_end = min(chunk_start + chunk_size, pair_stop)
            chunk_path = split_output_dir / f"{output_prefix}_{split_name}_{chunk_start:09d}_{chunk_end:09d}.parquet"
            if skip_existing and chunk_path.exists():
                if verbose:
                    print(f"[FE] Skipping existing chunk: {chunk_path.name}")
                existing_chunk = _read_parquet_compat(chunk_path)
                total_rows += len(existing_chunk)
                if first_chunk_df is None and not existing_chunk.empty:
                    first_chunk_df = existing_chunk.head(1).copy()
                continue
            chunk = labeled_pairs.iloc[chunk_start:chunk_end]
            rows: list[dict] = []
            pair_cols = list(chunk.columns)
            idx_pid_a = pair_cols.index("profile_id_a")
            idx_pid_b = pair_cols.index("profile_id_b")
            idx_label = pair_cols.index("label") if "label" in pair_cols else None
            idx_pair_type = pair_cols.index("pair_type") if "pair_type" in pair_cols else None

            for offset, pair in enumerate(chunk.itertuples(index=False, name=None), start=chunk_start):
                if verbose and (((offset - pair_start + 1) % progress_every == 0) or offset == chunk_start):
                    print(f"   {offset + 1}/{pair_stop} pairs", end="\r")

                pid_a = str(pair[idx_pid_a])
                pid_b = str(pair[idx_pid_b])
                record_a = profile_lookup.get(pid_a)
                record_b = profile_lookup.get(pid_b)
                if record_a is None or record_b is None:
                    continue

                idx_a = int(record_a["__idx"])
                idx_b = int(record_b["__idx"])
                feats = {
                    "profile_id_a": pid_a,
                    "profile_id_b": pid_b,
                    "label": int(pair[idx_label]) if idx_label is not None else -1,
                    "pair_type": str(pair[idx_pair_type]) if idx_pair_type is not None else "",
                }
                if "B1" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_B1_text(record_a, record_b, tfidf_mat, idx_a, idx_b))
                if "A1" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_A1_url(record_a, record_b))
                if "A2" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_A2_location(record_a, record_b))
                if "A3" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_A3_mention(record_a, record_b))
                if "A4" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_A4_hashtag(record_a, record_b))
                if "B2" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_B2_sbert(idx_a, idx_b, self.sbert_embeddings))
                if "B3" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_B3_stylometric(record_a, record_b))
                if "META" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_meta(record_a, record_b))
                if "PICURL" in self.feature_groups:
                    feats.update(PairFeatureComputer.feat_picture_url(record_a, record_b))
                if "IMG" in self.feature_groups and img_ext is not None:
                    cache_a = self.image_cache.get(pid_a, self.image_cache.get(str(pid_a), {"available": False}))
                    cache_b = self.image_cache.get(pid_b, self.image_cache.get(str(pid_b), {"available": False}))
                    feats.update(img_ext.compute_pair_features(cache_a, cache_b))
                rows.append(feats)

            chunk_df = pd.DataFrame(rows)
            _write_parquet_compat(chunk_df, chunk_path)
            total_rows += len(chunk_df)
            if first_chunk_df is None and not chunk_df.empty:
                first_chunk_df = chunk_df.head(1).copy()
            if verbose:
                print(f"\n[FE] Chunk done: {chunk_start:,} - {chunk_end:,} -> {chunk_path.name}")

        return total_rows, first_chunk_df

    @staticmethod
    def get_feature_cols(feature_matrix: pd.DataFrame) -> list[str]:
        skip = {"profile_id_a", "profile_id_b", "label", "pair_type", "clip_category_a", "clip_category_b"}
        return [col for col in feature_matrix.columns if col not in skip]


def build_entity_split(df: pd.DataFrame, pairs: pd.DataFrame, random_seed: int = 42):
    entity_col = "user_folder" if "user_folder" in df.columns else "profile_id"
    profile_id_series = df["profile_id"].map(lambda x: str(x))
    entity_series = df[entity_col].fillna(profile_id_series).map(lambda x: str(x))
    unique_entities = entity_series.drop_duplicates().tolist()

    train_entities, tmp_entities = train_test_split(unique_entities, test_size=0.30, random_state=random_seed)
    val_entities, test_entities = train_test_split(tmp_entities, test_size=0.50, random_state=random_seed)

    df = df.copy()
    df["profile_id"] = profile_id_series
    df["_entity_key"] = entity_series

    def profile_subset(entities):
        entity_set = set(entities)
        subset = df[df["_entity_key"].isin(entity_set)].copy()
        subset["_source_idx"] = subset.index.astype(int)
        subset = subset.reset_index(drop=True)
        pid_set = set(subset["profile_id"].tolist())
        pair_subset = pairs[pairs["profile_id_a"].isin(pid_set) & pairs["profile_id_b"].isin(pid_set)].copy()
        return subset.drop(columns=["_entity_key"]), pair_subset

    train_df, train_pairs = profile_subset(train_entities)
    val_df, val_pairs = profile_subset(val_entities)
    test_df, test_pairs = profile_subset(test_entities)
    return train_df, val_df, test_df, train_pairs, val_pairs, test_pairs


def build_or_load_image_cache(df: pd.DataFrame, force: bool = False) -> dict:
    return IMG_MOD.precompute_all_profiles(
        df=df,
        image_dir=DOWNLOADED_IMAGES_DIR,
        url_col="pictureURL",
        bio_col="bio",
        pid_col="profile_id",
        cache_path=IMAGE_CACHE_PATH,
        force=force,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Chunked Stage 9 pipeline")
    parser.add_argument("--stage", choices=["all", "split", "embeddings", "profile-cache", "features", "merge"], default="all")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from the highest completed chunk for each split.")
    parser.add_argument("--output-prefix", default="feature_matrix_chunked")
    parser.add_argument("--splits", default="train", help="Comma-separated splits to run: train,val,test")
    parser.add_argument("--force-sbert", action="store_true", help="Recompute SBERT embeddings even if cached.")
    parser.add_argument("--force-profile-cache", action="store_true", help="Rebuild cached per-profile features.")
    return parser.parse_args()


def main():
    args = parse_args()
    use_sbert = not args.skip_sbert
    use_image = not args.skip_image
    random_seed = 42
    requested_splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    valid_splits = {"train", "val", "test"}
    invalid_splits = [s for s in requested_splits if s not in valid_splits]
    if invalid_splits:
        raise ValueError(f"Invalid split names: {', '.join(invalid_splits)}")
    if not requested_splits:
        raise ValueError("At least one split must be requested.")
    if args.pair_start < 0:
        raise ValueError("--pair-start must be >= 0")

    print("=" * 60)
    print("Stage 9 Pipeline (Chunked)")
    print("=" * 60)
    ensure_output_dirs()
    print(f"[Output] Root directory: {PIPELINE_ROOT_DIR}")

    df = pd.read_csv(PROFILES_PATH)
    if "profile_id" not in df.columns:
        raise ValueError("profiles file must contain profile_id")
    df["profile_id"] = df["profile_id"].map(lambda x: str(x))
    train_df = val_df = test_df = None
    train_pairs = val_pairs = test_pairs = None

    if args.stage in {"all", "split"}:
        pairs = _read_parquet_compat(PAIRS_PATH)
        pairs["profile_id_a"] = pairs["profile_id_a"].map(lambda x: str(x))
        pairs["profile_id_b"] = pairs["profile_id_b"].map(lambda x: str(x))
        train_df, val_df, test_df, train_pairs, val_pairs, test_pairs = build_entity_split(df, pairs, random_seed=random_seed)
        save_split_artifacts(train_df, val_df, test_df, train_pairs, val_pairs, test_pairs)
        print(f"Profiles: {len(df):,} | Pairs: {len(pairs):,}")
        print(f"Entity split -> Train: {len(train_pairs):,}, Val: {len(val_pairs):,}, Test: {len(test_pairs):,}")
        print(f"[Stage: split] Saved artifacts to {ARTIFACTS_DIR}")
        if args.stage == "split":
            return
    else:
        train_df, val_df, test_df, train_pairs, val_pairs, test_pairs = load_split_artifacts()
        print(f"[Stage: split] Loaded artifacts from {ARTIFACTS_DIR}")

    sbert_embeddings = None
    if use_sbert:
        if args.stage in {"all", "embeddings"} or args.force_sbert or not SBERT_EMBEDDINGS_PATH.exists():
            sbert_embeddings = compute_sbert_embeddings(df)
            np.save(SBERT_EMBEDDINGS_PATH, sbert_embeddings)
            print(f"[Stage: embeddings] Saved embeddings to {SBERT_EMBEDDINGS_PATH}")
        else:
            sbert_embeddings = np.load(SBERT_EMBEDDINGS_PATH)
            print(f"[Stage: embeddings] Loaded cached embeddings from {SBERT_EMBEDDINGS_PATH}")
        if args.stage == "embeddings":
            return
    elif args.stage == "embeddings":
        print("[Stage: embeddings] Skipped because --skip-sbert was set.")
        return

    if args.stage in {"all", "profile-cache"}:
        split_dfs = {"train": train_df, "val": val_df, "test": test_df}
        for split_name in requested_splits:
            if args.force_profile_cache or not PROFILE_CACHE_PATHS[split_name].exists():
                save_profile_cache(split_name, split_dfs[split_name])
                print(f"[Stage: profile-cache] Saved {PROFILE_CACHE_PATHS[split_name].name}")
            else:
                print(f"[Stage: profile-cache] Reusing {PROFILE_CACHE_PATHS[split_name].name}")
        if args.stage == "profile-cache":
            return

    if args.stage == "merge":
        for split_name in requested_splits:
            merged_path = merge_feature_chunks(split_name, args.output_prefix)
            print(f"[Stage: merge] Wrote {merged_path}")
        return

    image_cache = {}
    if use_image:
        image_cache = build_or_load_image_cache(df)

    feature_groups = ["B1", "A1", "A2", "A3", "A4", "B3", "META", "PICURL"]
    if use_sbert:
        feature_groups.append("B2")
    if use_image:
        feature_groups.append("IMG")

    fe = FeatureEngineer(
        df=df,
        feature_groups=feature_groups,
        train_bios=train_df["bio"],
        sbert_embeddings=sbert_embeddings,
        image_cache=image_cache,
    )

    split_payloads = {
        "train": (train_df, train_pairs),
        "val": (val_df, val_pairs),
        "test": (test_df, test_pairs),
    }
    profile_lookups = {split_name: load_profile_cache(split_name) for split_name in requested_splits}
    row_counts: dict[str, int] = {}
    samples: list[Optional[pd.DataFrame]] = []

    for split_name in requested_splits:
        split_df, split_pairs = split_payloads[split_name]
        effective_pair_start = args.pair_start
        if args.auto_resume:
            effective_pair_start = _find_resume_offset(split_name, args.output_prefix)
        print(f"\n--- {split_name.title()} ---")
        if args.auto_resume:
            print(f"[Resume] Starting from pair offset {effective_pair_start:,}")
        rows, sample = fe.compute_pairs_chunked(
            split_pairs,
            split_df=split_df,
            split_name=split_name,
            output_prefix=args.output_prefix,
            profile_lookup=profile_lookups[split_name],
            chunk_size=args.chunk_size,
            pair_start=effective_pair_start,
            max_pairs=args.max_pairs,
            skip_existing=args.skip_existing,
            progress_every=args.progress_every,
        )
        row_counts[split_name] = rows
        samples.append(sample)

    feature_source = next((sample for sample in samples if sample is not None), None)
    if feature_source is None:
        raise RuntimeError("No feature rows were produced.")
    feature_cols = fe.get_feature_cols(feature_source)
    with TFIDF_VECTORIZER_PATH.open("wb") as fh:
        pickle.dump(fe.tfidf_vectorizer, fh)
    with FEATURE_COLS_PATH.open("wb") as fh:
        pickle.dump(feature_cols, fh)

    print("\nDone")
    for split_name in requested_splits:
        print(f"{split_name.title():<5}: {row_counts[split_name]:,} rows")
    print(f"Features: {len(feature_cols)} columns")

    if args.stage == "all":
        for split_name in requested_splits:
            merged_path = merge_feature_chunks(split_name, args.output_prefix)
            print(f"[Stage: merge] Wrote {merged_path}")


if __name__ == "__main__":
    main()
