"""
Stage 11: Adaptive / Per-Platform-Pair Threshold

แนวคิด: FN กระจุกที่บาง platform-pair (googleplus x twitter) การใช้ threshold เดียว 0.35
ทั้งระบบทำให้ต้องเลือกระหว่าง recall กับ precision รวม แต่ถ้าตั้ง threshold แยกต่อ platform-pair
(เลือกบน validation) จะลด threshold เฉพาะจุดที่ recall แย่ได้โดยไม่กระทบทั้งระบบ

เลือก threshold บน VALIDATION เท่านั้น -> apply บน TEST ครั้งเดียว
กัน overfit: platform-pair ที่มี positive < MIN_POS บน val ใช้ global threshold แทน

Objectives ที่ลอง:
  - global@0.35        : baseline เดิม
  - global F1-opt      : threshold เดียว optimize F1 บน val
  - per-pair F1-opt    : threshold ต่อ pair optimize F1 บน val
  - per-pair recall@P  : threshold ต่อ pair max recall ภายใต้ precision>=PREC_FLOOR บน val
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from stage10_eval_harness import load, add_derived, slice_of

MIN_POS = 200          # guard: pair ที่ positive < นี้ บน val -> ใช้ global
PREC_FLOOR = 0.93      # สำหรับ objective recall@precision-floor
GRID = np.round(np.arange(0.05, 0.951, 0.01), 3)


def best_f1_threshold(y, s, grid=GRID):
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]), float(max(f1s))


def best_recall_at_precision(y, s, floor=PREC_FLOOR, grid=GRID):
    best_t, best_r = None, -1
    for t in grid:
        pred = (s >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        if p >= floor and r > best_r:
            best_t, best_r = float(t), r
    if best_t is None:                      # ไม่มี t ที่ถึง floor -> ใช้ F1-opt
        best_t, _ = best_f1_threshold(y, s, grid)
    return best_t


def fit_pair_thresholds(val, objective):
    """คืน dict plat_pair -> threshold (เลือกบน val)"""
    g_t, _ = best_f1_threshold(val.label.values, val.probability.values)
    thr = {"__global__": g_t}
    for pp, g in val.groupby("plat_pair"):
        pos = int((g.label == 1).sum())
        if pos < MIN_POS:
            thr[pp] = g_t                    # guard
            continue
        y, s = g.label.values, g.probability.values
        thr[pp] = (best_f1_threshold(y, s)[0] if objective == "f1"
                   else best_recall_at_precision(y, s))
    return thr


def apply_pair_thresholds(df, thr):
    t = df["plat_pair"].map(lambda p: thr.get(p, thr["__global__"])).values
    return (df.probability.values >= t).astype(int)


def eval_pred(df, pred):
    y = df.label.values
    return dict(precision=round(precision_score(y, pred, zero_division=0), 4),
                recall=round(recall_score(y, pred, zero_division=0), 4),
                f1=round(f1_score(y, pred, zero_division=0), 4),
                FN=int(((pred == 0) & (y == 1)).sum()),
                FP=int(((pred == 1) & (y == 0)).sum()))


def slice_recall(df, pred):
    df = df.copy(); df["slice"] = slice_of(df); df["pred"] = pred
    out = {}
    for sl in ["EASY", "MID", "HARD"]:
        pos = (df["slice"] == sl) & (df.label == 1)
        if pos.sum():
            out[sl] = round(df.loc[pos, "pred"].mean(), 4)
    return out


def main():
    d = add_derived(load())
    va = d[d.split_name == "val"].copy()
    te = d[d.split_name == "test"].copy()
    yte = te.label.values

    # baselines
    base035 = (te.probability.values >= 0.35).astype(int)
    g_t, _ = best_f1_threshold(va.label.values, va.probability.values)
    base_gopt = (te.probability.values >= g_t).astype(int)

    thr_f1 = fit_pair_thresholds(va, "f1")
    thr_rp = fit_pair_thresholds(va, "recall@p")
    pred_f1 = apply_pair_thresholds(te, thr_f1)
    pred_rp = apply_pair_thresholds(te, thr_rp)

    configs = {
        "global@0.35 (baseline)": (base035, {"__global__": 0.35}),
        f"global F1-opt@{g_t:.2f}": (base_gopt, {"__global__": g_t}),
        "per-pair F1-opt": (pred_f1, thr_f1),
        f"per-pair recall@P>={PREC_FLOOR}": (pred_rp, thr_rp),
    }

    print("=" * 78)
    print(f"{'config':32s} {'P':>7} {'R':>7} {'F1':>7} {'FN':>5} {'FP':>6}  slice-recall(E/M/H)")
    base = eval_pred(te, base035)
    rows = []
    for name, (pred, thr) in configs.items():
        m = eval_pred(te, pred)
        sr = slice_recall(te, pred)
        srs = f"{sr.get('EASY',0):.2f}/{sr.get('MID',0):.2f}/{sr.get('HARD',0):.2f}"
        rec = int(((base035 == 0) & (pred == 1) & (yte == 1)).sum())
        fp_add = int(((base035 == 0) & (pred == 1) & (yte == 0)).sum())
        print(f"{name:32s} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f} "
              f"{m['FN']:5d} {m['FP']:6d}  {srs}")
        rows.append({"config": name, **m, "slice_recall": sr,
                     "thresholds": {k: round(v, 3) for k, v in thr.items()},
                     "recovered_TP_vs_base": rec, "added_FP_vs_base": fp_add})

    # gate check on best recall-oriented config
    cand = eval_pred(te, pred_rp)
    dR = cand["recall"] - base["recall"]
    dP = cand["precision"] - base["precision"]
    gate = (dR >= 0.015) and (dP >= -0.01)
    print("\n--- Acceptance gate (per-pair recall@P vs baseline) ---")
    print(f"  ΔRecall = {dR:+.4f} (ต้อง >= +0.015)   ΔPrecision = {dP:+.4f} (ต้อง >= -0.010)")
    print(f"  GATE: {'PASS ✅' if gate else 'FAIL ❌ -> คง baseline / ใช้ global F1-opt'}")

    out = {"gate_pass": bool(gate), "delta_recall": round(dR, 4),
           "delta_precision": round(dP, 4), "configs": rows,
           "pair_thresholds_recall@p": {k: round(v, 3) for k, v in thr_rp.items()}}
    with open("reports_adaptive_threshold.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open("models_platform_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thr_rp if gate else {"__global__": g_t}, f, indent=2, ensure_ascii=False)
    print("\nSaved reports_adaptive_threshold.json + models_platform_thresholds.json")


if __name__ == "__main__":
    main()
