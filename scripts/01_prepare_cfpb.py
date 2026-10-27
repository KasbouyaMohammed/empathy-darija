"""Filter CFPB bank complaints, map them to scenario categories, and label emotion.

Input : data/raw/cfpb/has-text/train-00002-of-00003.parquet (CC0, BEE-spoke-data mirror)
Output: data/processed/cfpb_candidates.parquet

Emotion is labelled with j-hartmann/emotion-english-distilroberta-base (7 classes),
used only to stratify seed selection; final labels come from human annotation.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cfpb/has-text/train-00002-of-00003.parquet"
OUT = ROOT / "data/processed/cfpb_candidates.parquet"
SEED = 42
PER_CATEGORY = 800

# CFPB "Issue" -> (scenario category, urgency). First match wins.
CATEGORY_RULES = [
    (r"fraud|scam|unauthorized|identity theft", "fraud_unauthorized", "high"),
    (r"trouble using your card|debit or atm card|lost|stolen", "card_problem", "high"),
    (r"money was not available|deposits and withdrawals|purchase or transfer|"
     r"other transaction problem|receiving payments, sending money", "funds_transfer", "high"),
    (r"closing an account|closing your account|closing/cancelling", "account_closed", "high"),
    (r"fees or interest|funds being low|charging your account", "fees_charges", "medium"),
    (r"purchase shown on your statement|billing dispute", "billing_dispute", "medium"),
    (r"problem when making payments", "payment_problem", "medium"),
    (r"customer service|customer relations", "customer_service", "medium"),
]
BANK_PRODUCTS = r"checking|savings|credit card|prepaid|money transfer|bank account"


def categorize(issue: str):
    issue = (issue or "").lower()
    for pattern, cat, urg in CATEGORY_RULES:
        if re.search(pattern, issue):
            return cat, urg
    return None, None


def main():
    cols = ["Date received", "Product", "Issue", "Consumer complaint narrative", "Complaint ID"]
    df = pd.read_parquet(RAW, columns=cols).rename(columns={
        "Date received": "date", "Product": "product", "Issue": "issue",
        "Consumer complaint narrative": "narrative", "Complaint ID": "complaint_id"})
    df = df[df["product"].str.contains(BANK_PRODUCTS, case=False, na=False)]
    df = df[df["date"] >= "2018-01-01"]

    df["n_words"] = df["narrative"].str.split().str.len()
    df["redaction_ratio"] = df["narrative"].str.count(r"XX+") / df["n_words"]
    df = df[df["n_words"].between(40, 220) & (df["redaction_ratio"] < 0.08)]

    df[["category", "urgency"]] = df["issue"].apply(lambda s: pd.Series(categorize(s)))
    df = df.dropna(subset=["category"]).drop_duplicates("narrative")
    print("after filtering:", len(df), file=sys.stderr)
    print(df["category"].value_counts().to_string(), file=sys.stderr)

    df = df.sample(frac=1, random_state=SEED).groupby("category").head(PER_CATEGORY)

    device = "mps" if torch.backends.mps.is_available() else -1
    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                   top_k=None, device=device, truncation=True, max_length=512)
    scores = clf(df["narrative"].tolist(), batch_size=32)
    probs = pd.DataFrame([{s["label"]: s["score"] for s in row} for row in scores], index=df.index)
    df = df.join(probs.add_prefix("p_"))
    df["emotion_auto"] = probs.idxmax(axis=1)
    df["emotion_conf"] = probs.max(axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(OUT, index=False)
    print(pd.crosstab(df["category"], df["emotion_auto"]).to_string(), file=sys.stderr)
    print(f"saved {len(df)} rows -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
