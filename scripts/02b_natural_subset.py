"""Select 120 human-written Darija messages from DarijaBanking (urgent or emotional intents).

These are written by people, not translated, and test whether benchmark findings on
LLM-drafted Darija hold on natural Darija (RQ1). They are never used for training: the RQ4
training-data builder excludes every row selected here, in 02_select_scenarios.py, or used
as a style example in 03_build_messages.py.

Output: data/processed/natural_subset.csv
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PER_INTENT = 15
INTENTS = {   # DarijaBanking intent -> benchmark category and urgency
    "compromised_card": ("fraud_unauthorized", "high"),
    "card_swallowed": ("card_problem", "high"),
    "lost_or_stolen_phone": ("fraud_unauthorized", "high"),
    "cancel_transfer": ("funds_transfer", "high"),
    "get_refund": ("billing_dispute", "medium"),
    "fee": ("fees_charges", "medium"),
    "general_negative_feedback": ("customer_service", "medium"),
    "contact_human_agent": ("customer_service", "medium"),
}

_spec = importlib.util.spec_from_file_location("sel", Path(__file__).with_name("02_select_scenarios.py"))
_sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sel)


def main():
    parts = [pd.read_parquet(ROOT / f"data/raw/darijabanking/data/{s}-00000-of-00001.parquet").assign(split=s)
             for s in ("train", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = df.groupby("split").cumcount().astype(str).radd(df["split"] + ":")
    used = set(pd.read_csv(ROOT / "data/processed/scenarios_seed.csv")["source_id"])
    df = df[df["label"].isin(INTENTS) & df["text"].str.contains(_sel.DARIJA_MARKERS)
            & ~df["text"].str.contains("_") & ~df["row_id"].isin(used)]
    df = df[df["text"].str.split().str.len().between(4, 40)].drop_duplicates("text")
    picked = df.sample(frac=1, random_state=SEED).groupby("label").head(PER_INTENT)
    assert (picked["label"].value_counts() == PER_INTENT).all(), picked["label"].value_counts()

    out = pd.DataFrame({
        "natural_id": [f"N{i:03d}" for i in range(1, len(picked) + 1)],
        "source_id": picked["row_id"].values,
        "intent": picked["label"].values,
        "category": picked["label"].map(lambda l: INTENTS[l][0]).values,
        "urgency": picked["label"].map(lambda l: INTENTS[l][1]).values,
        "variety": "ary",
        "text": picked["text"].str.strip().values,
    })
    out.to_csv(ROOT / "data/processed/natural_subset.csv", index=False)
    print(out.groupby(["intent", "urgency"]).size().to_string())
    print(f"saved {len(out)} natural Darija messages")


if __name__ == "__main__":
    main()
