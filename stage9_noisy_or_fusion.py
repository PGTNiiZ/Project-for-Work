"""
Stage 9: Noisy-OR Evidence Fusion — prototype & evaluation vs GB baseline

แนวคิด: รวมหลักฐานหลายสัญญาณแบบ availability-aware
    P(match) = 1 - prod_k (1 - s_k * r_k)
โดย s_k = similarity ของสัญญาณ k (name/bio/url/geo/image/caption) in [0,1]
    r_k = reliability ที่เรียนรู้จาก train (sigmoid param)
สัญญาณที่ขาด -> s_k=0 -> term=1 -> ไม่กระทบผล (จัดการ missing เชิงความน่าจะเป็น)

เป้า: กู้ false negative ที่ GB (name-dominated) พลาด โดยเฉพาะ weak-name slice
Input : Project-for-Work/pub_multi/data/train_all.parquet  (มี split/label/features/GB prob)
Output: noisy_or_report.json + สรุปบนหน้าจอ
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

TA = "Project-for-Work/pub_multi/data/train_all.parquet"
SIGNALS = ["name", "bio", "url", "geo", "img", "cap"]


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    def mx(*cols):
        return df[list(cols)].fillna(0).clip(0, 1).max(axis=1)
    s = pd.DataFrame(index=df.index)
    s["name"] = mx("username_jaro", "fullname_jaro", "username_token_sort", "fullname_token_sort")
    s["bio"] = mx("bio_tfidf_cosine", "bio_sbert_cosine")
    s["url"] = mx("domain_jaccard", "url_jaccard")
    s["geo"] = mx("location_jaro", "location_token_sort")
    s["img"] = mx("image_phash_sim", "image_dhash_sim")
    s["cap"] = mx("image_caption_bio_sbert_cross",
                  "image_caption_fullname_token_cross",
                  "image_caption_username_token_cross")
    return s


def noisy_or_prob(S: np.ndarray, r: np.ndarray) -> np.ndarray:
    # S: (n, k) in [0,1], r: (k,) in [0,1]
    log1m = np.log1p(-np.clip(S * r, 0, 1 - 1e-9))  # log(1 - s_k r_k)
    return 1.0 - np.exp(log1m.sum(axis=1))


def fit_noisy_or(S: np.ndarray, y: np.ndarray):
    def bce(theta):
        r = 1.0 / (1.0 + np.exp(-theta))
        p = np.clip(noisy_or_prob(S, r), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = minimize(bce, np.zeros(S.shape[1]), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
    return 1.0 / (1.0 + np.exp(-res.x))


def best_threshold(y, p, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def name_max(df):
    return df[["username_jaro", "fullname_jaro",
               "username_token_sort", "fullname_token_sort"]].max(axis=1)


def report_block(tag, y, pred, name_mx):
    out = {
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
    }
    pos = y == 1
    easy = pos & (name_mx >= 0.5)
    weak = pos & (name_mx < 0.5)
    out["recall_easy_name>=0.5"] = round(pred[easy].mean(), 4) if easy.sum() else None
    out["recall_weak_name<0.5"] = round(pred[weak].mean(), 4) if weak.sum() else None
    out["n_pos"] = int(pos.sum())
    out["FN"] = int(((pred == 0) & pos).sum())
    print(f"  [{tag}] P={out['precision']:.4f} R={out['recall']:.4f} F1={out['f1']:.4f} "
          f"| easyR={out['recall_easy_name>=0.5']} weakR={out['recall_weak_name<0.5']} | FN={out['FN']}")
    return out


def main():
    ta = pd.read_parquet(TA)
    tr = ta[ta.split_name == "train"].reset_index(drop=True)
    va = ta[ta.split_name == "val"].reset_index(drop=True)
    te = ta[ta.split_name == "test"].reset_index(drop=True)

    Str, Sva, Ste = build_signals(tr), build_signals(va), build_signals(te)
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values

    r = fit_noisy_or(Str.values, ytr)
    print("Learned reliabilities r_k:")
    for k, v in zip(SIGNALS, r):
        print(f"  r_{k:4s} = {v:.4f}")

    p_va = noisy_or_prob(Sva.values, r)
    p_te = noisy_or_prob(Ste.values, r)
    t_nor = best_threshold(yva, p_va)
    print(f"\nNoisy-OR threshold (val F1-opt) = {t_nor:.3f}")
    print(f"Noisy-OR test AP = {average_precision_score(yte, p_te):.4f}")

    nm_te = name_max(te).values
    gb_pred = (te.probability.values >= 0.35).astype(int)   # GB baseline @0.35
    nor_pred = (p_te >= t_nor).astype(int)

    print("\n=== TEST comparison ===")
    rep = {"reliabilities": {k: round(float(v), 4) for k, v in zip(SIGNALS, r)},
           "noisy_or_threshold": round(t_nor, 3),
           "noisy_or_test_ap": round(float(average_precision_score(yte, p_te)), 4)}
    rep["gb_baseline"] = report_block("GB @0.35", yte, gb_pred, nm_te)
    rep["noisy_or"] = report_block("NoisyOR ", yte, nor_pred, nm_te)

    # Ensemble rescue: MATCH if GB>=0.35 OR NoisyOR>=t_nor
    ens_pred = ((te.probability.values >= 0.35) | (p_te >= t_nor)).astype(int)
    rep["ensemble_OR"] = report_block("Ens(OR) ", yte, ens_pred, nm_te)

    # What did the ensemble recover / cost, vs GB alone?
    pos = yte == 1
    recovered = int(((gb_pred == 0) & (ens_pred == 1) & pos).sum())
    added_fp = int(((gb_pred == 0) & (ens_pred == 1) & (~pos)).sum())
    print(f"\nEnsemble vs GB alone: recovered {recovered} true positives, added {added_fp} false positives")
    rep["ensemble_recovered_TP"] = recovered
    rep["ensemble_added_FP"] = added_fp

    with open("noisy_or_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\nSaved noisy_or_report.json")


if __name__ == "__main__":
    main()
