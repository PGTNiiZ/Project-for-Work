"""
Stage 10: Evaluation Harness — slice-aware + per-platform-pair metrics

ผลิตตารางวัดผลที่ซ้ำได้ ใช้เป็นเครื่องวัดของทุก phase ถัดไป
- slice ตาม name similarity: EASY (name>=0.5) / MID (0.3-0.5) / HARD (name<0.3 & bio<0.3)
- per-platform-pair (canonical, unordered)
- overall P/R/F1/AP + FN breakdown

ใช้เป็น library ได้ (import ฟังก์ชัน) หรือรันตรงเพื่อ dump baseline
Input : Project-for-Work/pub_multi/data/train_all.parquet
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

TA = "Project-for-Work/pub_multi/data/train_all.parquet"
NAME_COLS = ["username_jaro", "fullname_jaro", "username_token_sort", "fullname_token_sort"]
BIO_COLS = ["bio_tfidf_cosine", "bio_sbert_cosine"]


def load(path=TA):
    return pd.read_parquet(path)


def add_derived(df):
    df = df.copy()
    df["name_max"] = df[NAME_COLS].fillna(0).max(axis=1)
    df["bio_max"] = df[BIO_COLS].fillna(0).max(axis=1)
    # canonical unordered platform-pair
    df["plat_pair"] = df.apply(
        lambda r: " x ".join(sorted([str(r["platform_a"]), str(r["platform_b"])])), axis=1)
    return df


def slice_of(df):
    s = pd.Series("EASY", index=df.index)
    s[df["name_max"] < 0.5] = "MID"
    s[(df["name_max"] < 0.3) & (df["bio_max"] < 0.3)] = "HARD"
    return s


def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    return dict(
        precision=round(precision_score(y, pred, zero_division=0), 4),
        recall=round(recall_score(y, pred, zero_division=0), 4),
        f1=round(f1_score(y, pred, zero_division=0), 4),
        FN=int(((pred == 0) & (y == 1)).sum()),
        FP=int(((pred == 1) & (y == 0)).sum()),
    )


def full_report(df, score_col="probability", thr=0.35, split="test"):
    d = add_derived(df)
    d = d[d.split_name == split].copy()
    d["slice"] = slice_of(d)
    y = d.label.values
    score = d[score_col].values
    rep = {"split": split, "threshold": thr, "n": int(len(d)),
           "n_pos": int((y == 1).sum()),
           "test_ap": round(float(average_precision_score(y, score)), 4),
           "overall": metrics_at(y, score, thr)}
    # per slice (recall over positives in slice)
    rep["by_slice"] = {}
    for sl in ["EASY", "MID", "HARD"]:
        m = d["slice"] == sl
        pos = m & (d.label == 1)
        if pos.sum() == 0:
            continue
        pred = (score[pos.values] >= thr).astype(int)
        rep["by_slice"][sl] = {"n_pos": int(pos.sum()),
                               "recall": round(float(pred.mean()), 4),
                               "FN": int((pred == 0).sum())}
    # per platform-pair
    rep["by_platform_pair"] = {}
    for pp, g in d.groupby("plat_pair"):
        yg = g.label.values
        sg = g[score_col].values
        pos = int((yg == 1).sum())
        if pos == 0:
            continue
        mm = metrics_at(yg, sg, thr)
        mm["n_pos"] = pos
        rep["by_platform_pair"][pp] = mm
    return rep


def print_report(rep, tag=""):
    o = rep["overall"]
    print(f"\n===== {tag}  (thr={rep['threshold']}) =====")
    print(f"AP={rep['test_ap']}  P={o['precision']}  R={o['recall']}  F1={o['f1']}  FN={o['FN']}  FP={o['FP']}")
    print("  by slice:")
    for sl, m in rep["by_slice"].items():
        print(f"    {sl:5s} n_pos={m['n_pos']:4d}  recall={m['recall']:.3f}  FN={m['FN']}")
    print("  by platform-pair:")
    for pp, m in rep["by_platform_pair"].items():
        print(f"    {pp:26s} n_pos={m['n_pos']:4d}  P={m['precision']:.3f} R={m['recall']:.3f} FN={m['FN']}")


if __name__ == "__main__":
    df = load()
    rep = full_report(df, "probability", 0.35, "test")
    print_report(rep, "GB baseline @0.35 (test)")
    with open("reports_eval_gb_baseline.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\nSaved reports_eval_gb_baseline.json")
