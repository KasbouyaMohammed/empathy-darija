"""Validate langid_darija.identify() on independent human-written text (not our benchmark).

Gold varieties come from DarijaBench (Atlas-Chat), which repackages human translations:
  - FLORES+ devtest (long, formal sentences): en, fr, msa, ary
  - MADAR (short travel-domain sentences): msa, ary
  - DODa (short everyday phrases): en, fr, msa, ary
  - DODa transliteration: ary_latn (Arabizi)
Each item is assigned once to a dev split (used to design identifier rules) or a test split
(reported in the paper), stratified by variety and source.

Usage:
    python scripts/04a_langid_validation.py build      # data/processed/langid_gold.csv
    python scripts/04a_langid_validation.py eval [dev|test|all]
    python scripts/04a_langid_validation.py report   # both identifiers on test -> results/langid_test_predictions.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/darijabench"
GOLD = ROOT / "data/processed/langid_gold.csv"
CODE = {"en": "en", "fr": "fr", "msa": "msa", "dr": "ary"}
PER_CELL = {"flores_plus": 200, "madar": 300, "doda": 200, "transliteration": 600}


def _pairs(name):
    """(variety, text) for both sides of every translation pair in one DarijaBench subset."""
    d = pd.read_parquet(RAW / f"{name}.parquet")
    rows = []
    for r in d.itertuples():
        src, tgt = r.direction.split("_")
        user = r.messages[0]["content"].split("\n", 1)[1].strip()
        answer = r.messages[1]["content"].strip()
        if name == "transliteration":  # ar_dr: Arabizi -> Arabic script; dr_ar: the reverse
            latn, arab = (user, answer) if r.direction == "ar_dr" else (answer, user)
            rows += [("ary_latn", latn), ("ary", arab)]
        else:
            rows += [(CODE[src], user), (CODE[tgt], answer)]
    out = pd.DataFrame(rows, columns=["variety", "text"]).drop_duplicates("text")
    out["source"] = name
    return out


def build(seed=0):
    parts = []
    for name, n in PER_CELL.items():
        d = _pairs(name)
        if name == "transliteration":
            d = d[d.variety == "ary_latn"]
        for _, g in d.groupby("variety"):
            parts.append(g.sample(min(n, len(g)), random_state=seed))
    gold = pd.concat(parts, ignore_index=True)
    gold["split"] = "test"
    dev = gold.groupby(["variety", "source"], group_keys=False).sample(frac=0.5, random_state=seed).index
    gold.loc[dev, "split"] = "dev"
    gold.to_csv(GOLD, index=False)
    print(gold.groupby(["variety", "source", "split"]).size().unstack().to_string())
    print(f"saved -> {GOLD}")


def evaluate_split(split):
    from langid_darija import evaluate
    gold = pd.read_csv(GOLD)
    if split != "all":
        gold = gold[gold.split == split]
    acc, confusion = evaluate(gold.text, gold.variety)
    print(f"{split}: n={len(gold)} accuracy={acc:.3f}")
    print(confusion.to_string())
    gold = gold.assign(pred=[None] * len(gold))
    from langid_darija import identify
    gold["pred"] = [identify(t)[0] for t in gold.text]
    by_src = gold.assign(ok=gold.pred == gold.variety).groupby(["source", "variety"]).ok.mean()
    print(by_src.round(3).unstack().to_string())
    return gold


def report():
    """Per-item predictions of the baseline and final identifiers on the held-out test split."""
    import langid_darija
    import langid_darija_v1
    gold = pd.read_csv(GOLD)
    gold = gold[gold.split == "test"].copy()
    gold["pred_baseline"] = [langid_darija_v1.identify(t)[0] for t in gold.text]
    gold["pred_final"] = [langid_darija.identify(t)[0] for t in gold.text]
    gold.to_csv(ROOT / "results/langid_test_predictions.csv", index=False)
    for col in ("pred_baseline", "pred_final"):
        print(col, round((gold[col] == gold.variety).mean(), 3),
              (gold[col] == gold.variety).groupby(gold.variety).mean().round(3).to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "report":
        report()
    else:
        evaluate_split(sys.argv[2] if len(sys.argv) > 2 else "dev")
