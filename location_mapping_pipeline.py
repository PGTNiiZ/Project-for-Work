import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def normalize_location(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s,.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        "bkk": "bangkok",
        "nyc": "new york city",
        "la": "los angeles",
        "sf": "san francisco",
        "uk": "united kingdom",
        "usa": "united states",
        "us": "united states",
    }
    tokens = s.split()
    tokens = [replacements.get(t, t) for t in tokens]
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_set_ratio(a, b)) / 100.0
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()


def prepare_ontology(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["name"].map(normalize_location)
    out["alt_names_norm"] = out["alt_names"].fillna("").map(normalize_location)
    return out


def best_match(raw_norm: str, onto: pd.DataFrame) -> Tuple[Optional[pd.Series], float]:
    best_score = 0.0
    best_row = None

    for _, row in onto.iterrows():
        s1 = similarity(raw_norm, row["name_norm"])
        s2 = similarity(raw_norm, row["alt_names_norm"])
        score = max(s1, s2)
        if score > best_score:
            best_score = score
            best_row = row

    return best_row, best_score


def run(
    profiles_file: Path,
    ontology_file: Path,
    output_file: Path,
    threshold: float,
) -> pd.DataFrame:
    profiles = pd.read_csv(profiles_file, usecols=["profile_id", "platform", "location"])
    onto = pd.read_csv(ontology_file)
    onto = prepare_ontology(onto)

    loc_df = profiles[["location"]].dropna().copy()
    loc_df["raw_location"] = loc_df["location"].map(str)
    loc_df["raw_location_norm"] = loc_df["raw_location"].map(normalize_location)
    loc_df = loc_df[loc_df["raw_location_norm"] != ""]
    loc_df = loc_df[["raw_location", "raw_location_norm"]].drop_duplicates()

    rows: List[dict] = []
    for _, r in loc_df.iterrows():
        raw = r["raw_location"]
        raw_norm = r["raw_location_norm"]
        matched, score = best_match(raw_norm, onto)

        if matched is None or score < threshold:
            rows.append(
                {
                    "raw_location": raw,
                    "raw_location_norm": raw_norm,
                    "canonical_id": None,
                    "canonical_name": None,
                    "country_code": None,
                    "admin1": None,
                    "lat": None,
                    "lon": None,
                    "confidence": round(score, 6),
                    "match_method": "no_match",
                }
            )
        else:
            rows.append(
                {
                    "raw_location": raw,
                    "raw_location_norm": raw_norm,
                    "canonical_id": matched["canonical_id"],
                    "canonical_name": matched["name"],
                    "country_code": matched.get("country_code"),
                    "admin1": matched.get("admin1"),
                    "lat": matched.get("lat"),
                    "lon": matched.get("lon"),
                    "confidence": round(score, 6),
                    "match_method": "fuzzy_name_or_alt",
                }
            )

    mapping = pd.DataFrame(rows).sort_values(["confidence", "raw_location"], ascending=[False, True])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"rows={len(mapping)}, matched={(mapping['match_method'] != 'no_match').sum()}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Map raw profile locations to ontology world locations")
    parser.add_argument(
        "--profiles-file",
        default="data/processed/all_profiles_cleaned.csv",
        help="Input profiles file with location column",
    )
    parser.add_argument(
        "--ontology-file",
        default="data/ontology/location_ontology.csv",
        help="Ontology world location file",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/location_mapping.csv",
        help="Output mapping file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum similarity threshold to accept mapping",
    )
    args = parser.parse_args()

    run(
        profiles_file=Path(args.profiles_file),
        ontology_file=Path(args.ontology_file),
        output_file=Path(args.output_file),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
