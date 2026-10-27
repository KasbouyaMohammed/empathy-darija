"""Select the seed scenarios for the benchmark.

- 112 emotional complaints from CFPB: 7 categories x 4 emotion groups x 4 seeds.
- 40 informational (low-urgency) queries written in Darija, from DarijaBanking: 8 intents x 5.

Output: data/processed/scenarios_seed.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PER_CELL = 4
PER_INTENT = 5

EMOTION_GROUPS = {"anger": "anger", "disgust": "anger", "fear": "anxiety",
                  "sadness": "distress", "neutral": "neutral"}
CATEGORIES = ["fraud_unauthorized", "card_problem", "funds_transfer", "account_closed",
              "fees_charges", "billing_dispute", "payment_problem"]
INFO_INTENTS = ["activate_my_card", "age_limit", "create_account", "edit_account",
                "exchange", "get_invoice", "insurance", "loan"]
# DarijaBanking rows removed after manual screening (split:row -> reason).
EXCLUDE = {
    "train:522": "MSA, and an angry fee complaint mislabelled as the 'exchange' intent",
    "train:2848": "US-specific product (529 college savings plan)",
}
DARIJA_MARKERS = (r"ديال|واش|بغيت|فين|شنو|علاش|دابا|كيفاش|نقدر|ماشي|راه|حيت|كنت|عافاك|"
                  r"بزاف|ديالي|كاين|غادي|تقدرو|معرفتش")


def select_cfpb():
    df = pd.read_parquet(ROOT / "data/processed/cfpb_candidates.parquet")
    df = df[df["category"].isin(CATEGORIES) & df["n_words"].between(50, 180)]
    df["emotion_group"] = df["emotion_auto"].map(EMOTION_GROUPS)
    df = df.dropna(subset=["emotion_group"])
    df = df[df["emotion_conf"] >= 0.5]
    # Most confident exemplars per cell, ties broken by a fixed shuffle.
    df = df.sample(frac=1, random_state=SEED).sort_values("emotion_conf", ascending=False, kind="stable")
    picked = df.groupby(["category", "emotion_group"]).head(PER_CELL)
    short = picked.groupby(["category", "emotion_group"]).size()
    assert (short == PER_CELL).all() and len(short) == len(CATEGORIES) * 4, short[short < PER_CELL]
    return pd.DataFrame({
        "source": "cfpb",
        "source_id": picked["complaint_id"].astype(str),
        "category": picked["category"],
        "urgency": picked["urgency"],
        "emotion_target": picked["emotion_group"],
        "emotion_conf": picked["emotion_conf"].round(3),
        "seed_lang": "en",
        "seed_text": picked["narrative"],
    })


def select_darijabanking():
    parts = [pd.read_parquet(ROOT / f"data/raw/darijabanking/data/{s}-00000-of-00001.parquet")
             .assign(split=s) for s in ("train", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = df.groupby("split").cumcount().astype(str).radd(df["split"] + ":")
    df = df[~df["row_id"].isin(EXCLUDE)]
    # Underscored tokens (e.g. "النشاط_البنكي") are template slots left in the dataset.
    df = df[df["label"].isin(INFO_INTENTS) & df["text"].str.contains(DARIJA_MARKERS)
            & ~df["text"].str.contains("_")]
    df = df[df["text"].str.split().str.len().between(4, 30)].drop_duplicates("text")
    picked = df.sample(frac=1, random_state=SEED).groupby("label").head(PER_INTENT)
    assert len(picked) == len(INFO_INTENTS) * PER_INTENT
    return pd.DataFrame({
        "source": "darijabanking",
        "source_id": picked["row_id"],
        "category": "info_" + picked["label"],
        "urgency": "low",
        "emotion_target": "neutral",
        "emotion_conf": float("nan"),
        "seed_lang": "ary",
        "seed_text": picked["text"],
    })


def main():
    seeds = pd.concat([select_cfpb(), select_darijabanking()], ignore_index=True)
    seeds = seeds.sort_values(["source", "category", "emotion_target"], kind="stable").reset_index(drop=True)
    seeds.insert(0, "scenario_id", [f"S{i:03d}" for i in range(1, len(seeds) + 1)])
    out = ROOT / "data/processed/scenarios_seed.csv"
    seeds.to_csv(out, index=False)
    print(pd.crosstab(seeds["category"], seeds["emotion_target"], margins=True).to_string())
    print(f"saved {len(seeds)} scenarios -> {out}")


if __name__ == "__main__":
    main()
