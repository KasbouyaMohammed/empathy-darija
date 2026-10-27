"""Replies of the closed models (Claude Sonnet 5.5, Claude Haiku 5.5) under P0 / P1 / P2.

Inputs : data/processed/messages_final.csv (benchmark, after QC) and natural_subset.csv
Output : data/processed/responses_claude.csv
All requests go through the Batches API (50% price) and the on-disk cache.

Usage:
    python scripts/07_generate_claude.py --pilot 10   # 10 messages per set, synchronous
    python scripts/07_generate_claude.py              # everything, batched
"""
import argparse
from pathlib import Path

import pandas as pd

from llm import batch, chat, params
from prompts import POLICIES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5"]
# Thinking counts toward max_tokens; replies themselves are capped at 120 words by the prompt.
MAX_TOKENS = 2048


def load_items(pilot=0):
    bench = pd.read_csv(ROOT / "data/processed/messages_final.csv")
    bench = bench.assign(set="benchmark", item_id=bench["scenario_id"] + "_" + bench["variety"])
    nat = pd.read_csv(ROOT / "data/processed/natural_subset.csv")
    nat = nat.assign(set="natural", item_id=nat["natural_id"], scenario_id=nat["natural_id"])
    cols = ["set", "item_id", "scenario_id", "category", "urgency", "variety", "text"]
    items = pd.concat([bench[cols], nat[cols]], ignore_index=True)
    if pilot:
        items = items.groupby("set").sample(n=pilot, random_state=0)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()

    items = load_items(args.pilot)
    reqs = {}
    for it in items.itertuples():
        for model in MODELS:
            for pol, system in POLICIES.items():
                reqs[f"{it.item_id}|{model}|{pol}"] = params(model, it.text, system=system,
                                                             max_tokens=MAX_TOKENS, effort="low")
    print(f"{len(items)} messages x {len(MODELS)} models x {len(POLICIES)} policies = {len(reqs)} requests")
    res = {k: chat(p) for k, p in reqs.items()} if args.pilot else batch(reqs)

    rows = []
    for it in items.itertuples():
        for model in MODELS:
            for pol in POLICIES:
                rec = res.get(f"{it.item_id}|{model}|{pol}")
                rows.append({"set": it.set, "item_id": it.item_id, "scenario_id": it.scenario_id,
                             "category": it.category, "urgency": it.urgency, "variety": it.variety,
                             "message": it.text, "system": model, "policy": pol,
                             "reply": rec["output"].strip() if rec else None,
                             "stop_reason": rec["stop_reason"] if rec else "missing",
                             "model_id": rec["model_id"] if rec else None})
    out = pd.DataFrame(rows)
    name = "responses_claude_pilot.csv" if args.pilot else "responses_claude.csv"
    out.to_csv(ROOT / "data/processed" / name, index=False)
    print(out.groupby(["system", "policy"])["stop_reason"].value_counts().unstack(fill_value=0).to_string())
    print(f"saved {len(out)} replies -> {name}")


if __name__ == "__main__":
    main()
