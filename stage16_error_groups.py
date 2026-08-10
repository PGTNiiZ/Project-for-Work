"""
Stage 16: ดึงกลุ่มผิดพลาดจาก Production Decision Matrix (รูป 4.14) มาวิเคราะห์

Actual มาจาก ground truth: user_folder_a == user_folder_b (คนเดียวกัน)
Decision มาจาก stage15 CRM pipeline (match_decisions.parquet รวม exact + scored)

กลุ่มที่ export (ตามโน้ตอาจารย์):
  FP     (925)    : decision MATCH   แต่จริงๆ ไม่ใช่คนเดียวกัน -> ดูว่าทำไม match
  FN     (5,554)  : decision NO_MATCH แต่จริงๆ คือคนเดียวกัน   -> เช็ก data
  REVIEW (86,296) : 4,065 คู่จริง + 82,231 คู่ไม่จริง            -> re-decide ใหม่ตั้งแต่ต้น
  TN     (~1.97M) : ข้าม ไม่ export
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PFW = ROOT / "Project-for-Work"
DECISIONS = PFW / "train_data" / "stage15_crm_entity_pipeline" / "artifacts" / "match_decisions.parquet"
PROFILES = PFW / "data_for_project" / "normalized_profiles_with_profile_id.csv"
OUT = ROOT / "analysis_decision_matrix"
OUT.mkdir(exist_ok=True)

PROF_COLS = ["platform", "userName", "fullName", "bio", "location",
             "externalUrl", "user_folder"]


def load_profiles():
    df = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    df["profile_row_id"] = pd.to_numeric(df["profile_row_id"], errors="coerce").astype("Int64")
    df = df[df["profile_row_id"].notna()].set_index("profile_row_id")
    return df[PROF_COLS]


def main():
    prof = load_profiles()
    dec = pd.read_parquet(DECISIONS)

    # actual จาก ground truth (user_folder เดียวกัน = คนเดียวกัน)
    folder = prof["user_folder"]
    dec["folder_a"] = dec["profile_id_a"].map(folder)
    dec["folder_b"] = dec["profile_id_b"].map(folder)
    dec["actual"] = ((dec["folder_a"] == dec["folder_b"])
                     & dec["folder_a"].notna() & (dec["folder_a"] != "")).astype(int)

    cm = pd.crosstab(dec["decision"], dec["actual"])
    print("=== Production Decision Matrix (actual จาก user_folder) ===")
    print(cm, "\n")

    # join ข้อมูล profile สองฝั่ง
    a = prof.add_suffix("_a").reindex(dec["profile_id_a"].values).reset_index(drop=True)
    b = prof.add_suffix("_b").reindex(dec["profile_id_b"].values).reset_index(drop=True)
    full = pd.concat([dec.reset_index(drop=True), a, b], axis=1)
    keep = ["profile_id_a", "profile_id_b", "score", "decision", "decision_source",
            "actual", "pair_type", "split_name",
            "platform_a", "userName_a", "fullName_a", "bio_a", "location_a", "externalUrl_a",
            "platform_b", "userName_b", "fullName_b", "bio_b", "location_b", "externalUrl_b",
            "user_folder_a", "user_folder_b"]
    full = full[keep]

    groups = {
        "fp_match_but_actually_no": full[(full.decision == "MATCH") & (full.actual == 0)],
        "fn_nomatch_but_actually_yes": full[(full.decision == "NO_MATCH") & (full.actual == 1)],
        "review_actual_match": full[(full.decision == "REVIEW") & (full.actual == 1)],
        "review_actual_nomatch": full[(full.decision == "REVIEW") & (full.actual == 0)],
    }
    summary = {"confusion_matrix": {d: cm.loc[d].to_dict() for d in cm.index}}
    for name, g in groups.items():
        g = g.sort_values("score", ascending=False)
        g.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        summary[name] = {
            "n": int(len(g)),
            "by_decision_source": g.decision_source.value_counts().to_dict(),
            "by_platform_pair": g.apply(
                lambda r: " x ".join(sorted([str(r.platform_a), str(r.platform_b)])), axis=1
            ).value_counts().head(6).to_dict() if len(g) else {},
            "score": {k: round(float(v), 4) for k, v in
                      g.score.describe()[["min", "25%", "50%", "75%", "max"]].items()},
        }
        print(f"{name:32s} n={len(g):>7,}  -> {OUT / (name + '.csv')}")

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
