"""
Stage 18: Re-decide กลุ่ม REVIEW (86,296 คู่) ใหม่ตั้งแต่ต้น ตามโน้ตอาจารย์

Insight จาก stage17: score ใน band [0.95, 0.98) แยกคู่จริง/ไม่จริงแทบไม่ได้
(precision 0.005-0.14) แต่ name similarity แยกได้ชัด:
  >=0.9  -> precision 0.90
  0.7-0.9-> precision 0.16
  <0.7   -> precision <0.01  (แทบไม่มีคู่จริง)

Rule ที่เสนอ:
  PROMOTE  name_sim >= 0.9            -> MATCH
  KEEP     0.7 <= name_sim < 0.9      -> REVIEW (ให้คนตรวจ เหลือน้อยลงมาก)
  DEMOTE   name_sim < 0.7             -> NO_MATCH

เทียบผลรวมทั้งระบบ (MATCH tier เดิม + rule ใหม่) กับของเดิม
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import jellyfish
    jaro = jellyfish.jaro_winkler_similarity
except ImportError:
    from difflib import SequenceMatcher
    jaro = lambda a, b: SequenceMatcher(None, a, b).ratio()

OUT = Path(__file__).resolve().parent / "analysis_decision_matrix"

# ตัวเลขระบบเดิม (จากรูป 4.14 / stage16)
OLD_TP, OLD_FP = 19624, 925
TOTAL_POS = 29247          # คู่จริงทั้งหมดจาก ground truth


def name_sim(row):
    cands = []
    for fa, fb in [("userName_a", "userName_b"), ("fullName_a", "fullName_b"),
                   ("userName_a", "fullName_b"), ("fullName_a", "userName_b")]:
        x, y = str(row[fa]).lower().strip(), str(row[fb]).lower().strip()
        if x and y and x != "nan" and y != "nan":
            cands.append(jaro(x, y))
    return max(cands) if cands else 0.0


def main():
    rv = pd.concat([
        pd.read_csv(OUT / "review_actual_match.csv", keep_default_na=False).assign(actual=1),
        pd.read_csv(OUT / "review_actual_nomatch.csv", keep_default_na=False).assign(actual=0),
    ], ignore_index=True)
    rv["name_sim_max"] = rv.apply(name_sim, axis=1)

    rv["new_decision"] = np.where(rv.name_sim_max >= 0.9, "MATCH",
                          np.where(rv.name_sim_max >= 0.7, "REVIEW", "NO_MATCH"))

    print("=== Re-decision ของกลุ่ม REVIEW (86,296 คู่) ===")
    tab = pd.crosstab(rv.new_decision, rv.actual, margins=True)
    print(tab, "\n")

    promoted = rv[rv.new_decision == "MATCH"]
    kept = rv[rv.new_decision == "REVIEW"]
    demoted = rv[rv.new_decision == "NO_MATCH"]

    add_tp = int((promoted.actual == 1).sum())
    add_fp = int((promoted.actual == 0).sum())
    lost = int((demoted.actual == 1).sum())

    new_tp, new_fp = OLD_TP + add_tp, OLD_FP + add_fp
    print(f"PROMOTE -> MATCH : {len(promoted):>6,} คู่  (จริง {add_tp:,} / ผิด {add_fp:,}, precision {add_tp/len(promoted):.3f})")
    print(f"KEEP    -> REVIEW: {len(kept):>6,} คู่  (จริง {(kept.actual==1).sum():,}) — คิวคนตรวจลดจาก 86,296 เหลือ {len(kept):,} ({100*len(kept)/len(rv):.1f}%)")
    print(f"DEMOTE  -> NO    : {len(demoted):>6,} คู่  (เสียคู่จริงไป {lost:,} = {100*lost/len(rv[rv.actual==1]):.1f}% ของคู่จริงใน REVIEW)")
    print()
    print("=== ผลรวมทั้งระบบ (MATCH tier) ===")
    print(f"เดิม : TP {OLD_TP:,}  FP {OLD_FP:,}  precision {OLD_TP/(OLD_TP+OLD_FP):.4f}  recall {OLD_TP/TOTAL_POS:.4f}")
    print(f"ใหม่ : TP {new_tp:,}  FP {new_fp:,}  precision {new_tp/(new_tp+new_fp):.4f}  recall {new_tp/TOTAL_POS:.4f}")

    rv.sort_values(["new_decision", "name_sim_max"], ascending=[True, False]).to_csv(
        OUT / "review_redecision.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {OUT / 'review_redecision.csv'}")


if __name__ == "__main__":
    main()
