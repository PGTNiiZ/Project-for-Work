"""
exp_lib: ของกลางสำหรับการทดลอง R0-R5 (ดู EXPERIMENT_PLAN.md)

- entity-aware split (hash user_folder -> val 70% / test 30%, คู่คร่อม split ถูก drop)
- name_sim (max Jaro-Winkler 4 คู่ userName/fullName ไขว้ เหมือน stage18)
- cache scored pairs + actual + name_sim + split -> experiments/scored_pairs_enriched.parquet
- evaluation harness เดียวกันทุก experiment (รวมค่าคงที่ exact tier + blocking-missed FN ต่อ split)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import jellyfish

ROOT = Path(__file__).resolve().parent
PFW = ROOT / "Project-for-Work"
DECISIONS = PFW / "train_data" / "stage15_crm_entity_pipeline" / "artifacts" / "match_decisions.parquet"
PROFILES = PFW / "data_for_project" / "normalized_profiles_with_profile_id.csv"
BLOCKING_MISSED = ROOT / "analysis_decision_matrix" / "blocking_missed_pairs.csv"
EXP = ROOT / "experiments"
EXP.mkdir(exist_ok=True)
CACHE = EXP / "scored_pairs_enriched.parquet"

TOTAL_POS_FULL = 29247      # คู่จริงทั้งหมดจาก ground truth (user_folder)
TEST_PCT = 30               # hash % 100 < 30 -> test
NSIM_MIN_SCORE = 0.5        # ต่ำกว่านี้ decision เป็น NO_MATCH เสมอ (t_r >= 0.5) ไม่ต้องคำนวณ name_sim

# รหัส decision แบบ int ให้ GA เร็ว
NO_MATCH, REVIEW, MATCH = 0, 1, 2
DEC_CODE = {"NO_MATCH": NO_MATCH, "REVIEW": REVIEW, "MATCH": MATCH}
CODE_DEC = {v: k for k, v in DEC_CODE.items()}


def bucket(key: str) -> str:
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 100
    return "test" if h < TEST_PCT else "val"


def load_profiles():
    df = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna()].set_index("profile_row_id")
    return df[["userName", "fullName", "user_folder"]]


def _name_sim_arrays(ua, fa, ub, fb):
    """max Jaro-Winkler ของ 4 คู่ (userName/fullName ไขว้กัน) — logic เดียวกับ stage18"""
    jaro = jellyfish.jaro_winkler_similarity
    out = np.zeros(len(ua))
    for i, (a1, a2, b1, b2) in enumerate(zip(ua, fa, ub, fb)):
        best = 0.0
        for x in (a1, a2):
            if not x:
                continue
            for y in (b1, b2):
                if not y:
                    continue
                s = jaro(x, y)
                if s > best:
                    best = s
        out[i] = best
    return out


def _clean_names(series):
    s = series.astype(str).str.lower().str.strip()
    return s.where(~s.isin(["", "nan"]), "").values


def build_cache(force=False):
    """สร้าง/โหลด scored pairs พร้อม actual, split, name_sim (คำนวณครั้งเดียว ใช้ทุก experiment)"""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    prof = load_profiles()
    dec = pd.read_parquet(DECISIONS, columns=[
        "profile_id_a", "profile_id_b", "score", "decision", "decision_source"])

    folder = prof["user_folder"]
    fa = dec["profile_id_a"].map(folder)
    fb = dec["profile_id_b"].map(folder)
    dec["actual"] = ((fa == fb) & fa.notna() & (fa != "")).astype(np.int8)

    # split ต่อ profile: ใช้ user_folder ถ้ามี ไม่งั้น hash profile_row_id (มีแต่ negative)
    keys = folder.where(folder != "", "pid:" + folder.index.astype(str))
    prof_split = keys.map(bucket)
    sa = dec["profile_id_a"].map(prof_split)
    sb = dec["profile_id_b"].map(prof_split)
    dec["split"] = np.where(sa == sb, sa, "drop")

    # name_sim เฉพาะคู่ scored ที่ score >= NSIM_MIN_SCORE
    need = (dec["decision_source"] != "AUTO_EXACT") & (dec["score"] >= NSIM_MIN_SCORE)
    idx = dec.index[need]
    ua = _clean_names(prof["userName"].reindex(dec.loc[idx, "profile_id_a"]))
    fna = _clean_names(prof["fullName"].reindex(dec.loc[idx, "profile_id_a"]))
    ub = _clean_names(prof["userName"].reindex(dec.loc[idx, "profile_id_b"]))
    fnb = _clean_names(prof["fullName"].reindex(dec.loc[idx, "profile_id_b"]))
    dec["name_sim"] = 0.0
    dec.loc[idx, "name_sim"] = _name_sim_arrays(ua, fna, ub, fnb)

    dec.to_parquet(CACHE, index=False)
    return dec


def split_constants(cache):
    """ค่าคงที่ต่อ split: exact-tier TP/FP, blocking-missed FN, total positives"""
    bm = pd.read_csv(BLOCKING_MISSED, keep_default_na=False)
    bm["split"] = bm["user_folder_a"].map(bucket)   # คู่จริงทั้งคู่ folder เดียวกันเสมอ

    consts = {}
    for sp in ["val", "test", "full"]:
        ex = cache[cache.decision_source == "AUTO_EXACT"]
        sc = cache[cache.decision_source != "AUTO_EXACT"]
        if sp != "full":
            ex, sc = ex[ex.split == sp], sc[sc.split == sp]
            fn_block = int((bm.split == sp).sum())
        else:
            fn_block = len(bm)
        tp_exact = int((ex.actual == 1).sum())
        fp_exact = int((ex.actual == 0).sum())
        consts[sp] = dict(
            tp_exact=tp_exact, fp_exact=fp_exact, fn_blocking=fn_block,
            total_pos=int((sc.actual == 1).sum()) + tp_exact + fn_block,
            n_scored=len(sc))
    return consts


def evaluate(decision_code, actual, consts):
    """harness กลาง — decision_code: array int (0/1/2) ของ scored pairs, consts: dict ของ split นั้น"""
    fp = int(((decision_code == MATCH) & (actual == 0)).sum()) + consts["fp_exact"]
    tp = int(((decision_code == MATCH) & (actual == 1)).sum()) + consts["tp_exact"]
    fn = int(((decision_code == NO_MATCH) & (actual == 1)).sum()) + consts["fn_blocking"]
    rev = int((decision_code == REVIEW).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / consts["total_pos"] if consts["total_pos"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, FP=fp, FN=fn, REVIEW=rev,
                precision=round(prec, 4), recall=round(rec, 4), F1=round(f1, 4))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
NESTED_SPLIT_VERSION = "entity_nested_v1"
NESTED_DEV_PCTS = (70, 15, 15)  # model_train / model_calibration / ga_validation
def entity_keys_by_profile() -> pd.Series:
    """Return the entity used for every profile, including a negative-only fallback."""
    profiles = load_profiles()
    folders = profiles["user_folder"].fillna("").astype(str).str.strip()
    keys = folders.where(folders != "", "pid:" + folders.index.astype(str))
    keys.name = "entity_key"
    return keys


def _nested_entity_bucket(key: str, seed: int) -> str:
    encoded = f"{NESTED_SPLIT_VERSION}:{int(seed)}:{key}".encode("utf-8")
    value = int(hashlib.md5(encoded).hexdigest(), 16) % 100
    if value < NESTED_DEV_PCTS[0]:
        return "model_train"
    if value < NESTED_DEV_PCTS[0] + NESTED_DEV_PCTS[1]:
        return "model_calibration"
    return "ga_validation"


def build_nested_entity_split(cache: pd.DataFrame, seed: int = 42,
                              entity_keys: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Create entity-disjoint model, calibration, GA-validation, and test pair roles."""
    required = {"profile_id_a", "profile_id_b", "split"}
    missing = sorted(required - set(cache.columns))
    if missing:
        raise ValueError(f"cannot create nested split; cache is missing {missing}")
    if cache.duplicated(["profile_id_a", "profile_id_b"]).any():
        raise ValueError("cannot create nested split from duplicate pair keys")

    keys = entity_keys if entity_keys is not None else entity_keys_by_profile()
    keys = keys.copy()
    keys.index = pd.to_numeric(keys.index, errors="raise").astype("int64")
    keys = keys.fillna("").astype(str)
    if (keys == "").any() or keys.index.duplicated().any():
        raise ValueError("entity mapping must contain one non-empty key per profile")

    out = cache.copy()
    a_key = out["profile_id_a"].map(keys)
    b_key = out["profile_id_b"].map(keys)
    if a_key.isna().any() or b_key.isna().any():
        raise ValueError("cache contains profile ids absent from the entity mapping")
    a_role = a_key.map(lambda key: _nested_entity_bucket(key, seed))
    b_role = b_key.map(lambda key: _nested_entity_bucket(key, seed))
    split = np.full(len(out), "drop", dtype=object)
    outer_test = out["split"].to_numpy() == "test"
    split[outer_test] = "test"
    same_dev_role = (out["split"].to_numpy() == "val") & (a_role.to_numpy() == b_role.to_numpy())
    split[same_dev_role] = a_role.to_numpy()[same_dev_role]
    out["experiment_split"] = split

    entity_sets, pair_counts = {}, {}
    for role in ("model_train", "model_calibration", "ga_validation", "test"):
        mask = out["experiment_split"] == role
        entity_sets[role] = set(a_key.loc[mask]).union(set(b_key.loc[mask]))
        pair_counts[role] = int(mask.sum())
    roles = list(entity_sets)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            overlap = entity_sets[left] & entity_sets[right]
            if overlap:
                raise AssertionError(f"entity leakage between {left} and {right}: {len(overlap)} entities")
    manifest = {
        "version": NESTED_SPLIT_VERSION,
        "seed": int(seed),
        "outer_split": "existing entity-aware cache split; test remains sealed",
        "development_entity_allocation": {
            "model_train": NESTED_DEV_PCTS[0],
            "model_calibration": NESTED_DEV_PCTS[1],
            "ga_validation": NESTED_DEV_PCTS[2],
        },
        "pair_counts": {**pair_counts, "dropped": int((out["experiment_split"] == "drop").sum())},
        "entity_counts": {role: len(values) for role, values in entity_sets.items()},
        "entity_overlap_check": "passed",
    }
    return out, manifest

def nested_split_constants(cache: pd.DataFrame, seed: int = 42) -> dict:
    """Compute exact-tier and blocking constants for an assigned nested split."""
    if "experiment_split" not in cache:
        raise ValueError("nested split constants require the experiment_split column")
    bm = pd.read_csv(BLOCKING_MISSED, keep_default_na=False)
    outer = bm["user_folder_a"].map(bucket)
    bm["experiment_split"] = np.where(
        outer.to_numpy() == "test",
        "test",
        bm["user_folder_a"].map(lambda key: _nested_entity_bucket(str(key), seed)).to_numpy(),
    )
    consts = {}
    for role in ("model_train", "model_calibration", "ga_validation", "test"):
        exact = cache[(cache["decision_source"] == "AUTO_EXACT") & (cache["experiment_split"] == role)]
        scored = cache[(cache["decision_source"] != "AUTO_EXACT") & (cache["experiment_split"] == role)]
        fn_blocking = int((bm["experiment_split"] == role).sum())
        tp_exact = int((exact["actual"] == 1).sum())
        consts[role] = {
            "tp_exact": tp_exact,
            "fp_exact": int((exact["actual"] == 0).sum()),
            "fn_blocking": fn_blocking,
            "total_pos": int((scored["actual"] == 1).sum()) + tp_exact + fn_blocking,
            "n_scored": int(len(scored)),
        }
    return consts

