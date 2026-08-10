"""
Stage 17: วิเคราะห์ pattern ของกลุ่มผิดพลาดจาก stage16 + คู่ที่ blocking พลาด

Output: analysis_decision_matrix/
  - blocking_missed_pairs.csv   (คู่จริงที่ไม่เคยถูก score — ส่วนที่หายของ FN 5,554)
  - error_analysis_report.md    (สรุป pattern ทุกกลุ่ม)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import jellyfish
    jaro = jellyfish.jaro_winkler_similarity
except ImportError:
    from difflib import SequenceMatcher
    jaro = lambda a, b: SequenceMatcher(None, a, b).ratio()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_decision_matrix"
PROFILES = ROOT / "Project-for-Work" / "data_for_project" / "normalized_profiles_with_profile_id.csv"

PROF_COLS = ["platform", "userName", "fullName", "bio", "location",
             "externalUrl", "user_folder"]


def name_sim(row):
    cands = []
    for fa, fb in [("userName_a", "userName_b"), ("fullName_a", "fullName_b"),
                   ("userName_a", "fullName_b"), ("fullName_a", "userName_b")]:
        x, y = str(row[fa]).lower().strip(), str(row[fb]).lower().strip()
        if x and y and x != "nan" and y != "nan":
            cands.append(jaro(x, y))
    return max(cands) if cands else 0.0


def bucket(s):
    return pd.cut(s, [-0.01, 0.3, 0.5, 0.7, 0.9, 1.01],
                  labels=["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])


def plat_pair(df):
    return df.apply(lambda r: " x ".join(sorted([str(r.platform_a), str(r.platform_b)])), axis=1)


def field_presence(df):
    out = {}
    for f in ["bio", "location", "externalUrl"]:
        both = ((df[f + "_a"].astype(str).str.strip() != "") &
                (df[f + "_b"].astype(str).str.strip() != ""))
        out[f + "_both_present_pct"] = round(100 * both.mean(), 1)
    return out


def main():
    lines = ["# Error Analysis — Production Decision Matrix", ""]

    # ---------- คู่ที่ blocking พลาด (ส่วนที่หายของ FN) ----------
    prof = pd.read_csv(PROFILES, keep_default_na=False, low_memory=False)
    prof["profile_row_id"] = pd.to_numeric(prof["profile_row_id"], errors="coerce").astype("Int64")
    prof = prof[prof["profile_row_id"].notna()].set_index("profile_row_id")[PROF_COLS]

    missed = json.loads((OUT / "blocking_missed_pairs.json").read_text())
    mdf = pd.DataFrame(missed, columns=["profile_id_a", "profile_id_b"])
    a = prof.add_suffix("_a").reindex(mdf.profile_id_a.values).reset_index(drop=True)
    b = prof.add_suffix("_b").reindex(mdf.profile_id_b.values).reset_index(drop=True)
    mdf = pd.concat([mdf, a, b], axis=1)
    mdf["name_sim_max"] = mdf.apply(name_sim, axis=1)
    mdf.to_csv(OUT / "blocking_missed_pairs.csv", index=False, encoding="utf-8-sig")

    lines += [f"## 0) คู่จริงที่ blocking พลาด (ไม่เคยถูก score) — {len(mdf):,} คู่",
              "ส่วนที่หายของ FN 5,554 ในรูป 4.14 (= 2,242 โมเดลปัดตก + {:,} ไม่เข้า candidate)".format(len(mdf)),
              "", "name similarity (max ของ 4 ช่องชื่อ):",
              bucket(mdf.name_sim_max).value_counts().sort_index().to_string(),
              "", "platform pair:", plat_pair(mdf).value_counts().to_string(),
              "", "field ครบทั้งคู่ (%): " + json.dumps(field_presence(mdf)), ""]

    # ---------- โหลดกลุ่มจาก stage16 ----------
    fp = pd.read_csv(OUT / "fp_match_but_actually_no.csv", keep_default_na=False)
    fn = pd.read_csv(OUT / "fn_nomatch_but_actually_yes.csv", keep_default_na=False)
    rv1 = pd.read_csv(OUT / "review_actual_match.csv", keep_default_na=False)
    rv0 = pd.read_csv(OUT / "review_actual_nomatch.csv", keep_default_na=False)
    for d in (fp, fn, rv1, rv0):
        d["score"] = pd.to_numeric(d.score, errors="coerce")
        d["name_sim_max"] = d.apply(name_sim, axis=1)

    # ---------- FP ----------
    same_un = (fp.userName_a.str.lower().str.strip() == fp.userName_b.str.lower().str.strip()) & (fp.userName_a.str.strip() != "")
    lines += [f"## 1) False Positive — MATCH แต่ไม่ใช่คนเดียวกัน ({len(fp):,} คู่)",
              "", "decision_source:", fp.decision_source.value_counts().to_string(),
              "", f"userName เหมือนกันเป๊ะ: {same_un.sum():,} คู่ ({100*same_un.mean():.1f}%)",
              "", "name similarity:", bucket(fp.name_sim_max).value_counts().sort_index().to_string(),
              "", "platform pair:", plat_pair(fp).value_counts().head(6).to_string(),
              "", "ตัวอย่าง exact-FP (username ชนกันแต่คนละคน):", ""]
    ex = fp[fp.decision_source == "AUTO_EXACT"].head(8)
    for _, r in ex.iterrows():
        lines.append(f"- `{r.userName_a}` ({r.platform_a}, folder={r.user_folder_a}) vs "
                     f"`{r.userName_b}` ({r.platform_b}, folder={r.user_folder_b})")
    lines.append("")

    # ---------- FN (scored) ----------
    lines += [f"## 2) False Negative (scored) — NO_MATCH แต่คือคนเดียวกัน ({len(fn):,} คู่)",
              "", "score distribution:",
              fn.score.describe()[["min", "25%", "50%", "75%", "max"]].round(4).to_string(),
              "", f"score >= 0.5 (เกือบถึง review band 0.95): {(fn.score >= 0.5).sum():,} คู่",
              f"score >= 0.90: {(fn.score >= 0.90).sum():,} คู่",
              "", "name similarity:", bucket(fn.name_sim_max).value_counts().sort_index().to_string(),
              "", "platform pair:", plat_pair(fn).value_counts().head(6).to_string(),
              "", "field ครบทั้งคู่ (%): " + json.dumps(field_presence(fn)), ""]

    # ---------- REVIEW ----------
    both = pd.concat([rv1.assign(actual=1), rv0.assign(actual=0)])
    lines += [f"## 3) REVIEW queue — {len(both):,} คู่ (จริง {len(rv1):,} / ไม่จริง {len(rv0):,} — precision {len(rv1)/len(both):.3f})",
              ""]
    # precision ตาม score band ภายใน [0.95, 0.98)
    both["band"] = pd.cut(both.score, np.round(np.arange(0.95, 0.981, 0.005), 3))
    tab = both.groupby("band", observed=True).agg(n=("actual", "size"), true=("actual", "sum"))
    tab["precision"] = (tab.true / tab.n).round(4)
    lines += ["precision ตาม score band ใน REVIEW:", tab.to_string(), ""]
    # ถ้าจัดลำดับ queue ด้วย name_sim แทน score
    both["ns_band"] = bucket(both.name_sim_max)
    tab2 = both.groupby("ns_band", observed=True).agg(n=("actual", "size"), true=("actual", "sum"))
    tab2["precision"] = (tab2.true / tab2.n).round(4)
    lines += ["precision ตาม name similarity ใน REVIEW:", tab2.to_string(), "",
              "platform pair (เฉพาะคู่จริงใน REVIEW):", plat_pair(rv1).value_counts().head(6).to_string(), ""]

    report = "\n".join(str(x) for x in lines)
    (OUT / "error_analysis_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {OUT / 'error_analysis_report.md'}")


if __name__ == "__main__":
    main()
