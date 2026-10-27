"""Turn seed scenarios into short customer chat messages in 5 language varieties.

Step A (CFPB seeds only): condense + localise the English narrative into a 1-3 sentence
        chat message to a Moroccan bank's chatbot.
Step B: draft the remaining varieties from the source message.

Drafts are written to an Excel sheet for native-speaker post-editing; the post-edited
column, not the draft, is what later steps use.

Usage:
    python scripts/03_build_messages.py --dry-run      # print prompts for 2 scenarios
    python scripts/03_build_messages.py                # run all (needs ANTHROPIC_API_KEY or .env)
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from llm import chat, params

ROOT = Path(__file__).resolve().parents[1]
DRAFT_MODEL = "claude-opus-5-5"
VARIETIES = {
    "en": "English",
    "fr": "French (as written by a Moroccan customer, everyday register)",
    "msa": "Modern Standard Arabic (Arabic script)",
    "ary": "Moroccan Darija in Arabic script, as Moroccans actually type in chat apps",
    "ary_latn": ("Moroccan Darija in Latin script (Arabizi, using digits like 3, 7, 9 for "
                 "Arabic sounds), with the natural French code-switching typical of Moroccan "
                 "banking vocabulary (e.g. carte, compte, virement, agence)"),
}

CONDENSE_PROMPT = """You are helping build a research benchmark of customer messages sent to the chatbot of a Moroccan retail bank.

Below is a real consumer complaint (US, anonymised; "XXXX" marks redactions).
Rewrite it as ONE chat message (1-3 sentences, max 60 words) that a customer would type to the bank's chatbot.

Rules:
- First person, informal chat style, no greeting formulas beyond what a real customer would type.
- Keep the core problem and the customer's emotional tone and intensity ({emotion}); do not exaggerate or soften it.
- Localise to Morocco: amounts in dirhams (DH), "my bank"/"your bank" instead of US bank names, no US-specific institutions, no personal names, no reconstruction of redacted parts.
- Output only the message, nothing else.

Complaint:
\"\"\"{seed}\"\"\""""

TRANSLATE_PROMPT = """Translate the customer chat message below into each requested language variety.
Keep the meaning, the facts and the emotional tone exactly; keep it a short chat message as a real Moroccan customer would type it. Do not add or remove information.

Requested varieties:
{targets}

Reference examples of authentic Darija (Arabic script) banking messages, for style only:
{examples}

Source message ({source_name}):
\"\"\"{text}\"\"\"

Return a JSON object whose keys are exactly: {keys}. No other text."""


def darija_examples(n=6, seed=0):
    df = pd.read_csv(ROOT / "data/processed/scenarios_seed.csv")
    # Style examples come from DarijaBanking sentences NOT used as scenarios.
    used = set(df.loc[df["source"] == "darijabanking", "seed_text"])
    db = pd.read_parquet(ROOT / "data/raw/darijabanking/data/train-00000-of-00001.parquet")
    db = db[db["text"].str.contains(r"ديال|واش|بغيت|عافاك|شنو|كيفاش") & ~db["text"].isin(used)
            & ~db["text"].str.contains("_")]
    db = db[db["text"].str.split().str.len().between(6, 25)]
    return db.sample(n, random_state=seed)["text"].tolist()


def parse_json(text, keys):
    """Last JSON object in `text` that has all `keys` (the model sometimes self-corrects)."""
    decoder, found = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in keys):
            found = obj
    if found is None:
        raise ValueError(f"no JSON object with keys {keys} in output: {text[:200]!r}")
    return found


def build(row, examples, dry_run):
    out = {}
    if row.source == "cfpb":
        prompt = CONDENSE_PROMPT.format(seed=row.seed_text, emotion=row.emotion_target)
        if dry_run:
            print(prompt, "\n" + "-" * 80)
            out["en"] = "<condensed English message>"
        else:
            out["en"] = chat(params(DRAFT_MODEL, prompt, effort="medium"))["output"].strip().strip('"')
        source_code = "en"
    else:
        out["ary"] = row.seed_text
        source_code = "ary"

    targets = [k for k in VARIETIES if k != source_code]
    prompt = TRANSLATE_PROMPT.format(
        targets="\n".join(f"- {k}: {VARIETIES[k]}" for k in targets),
        examples="\n".join(f"- {e}" for e in examples),
        source_name=VARIETIES[source_code], text=out[source_code], keys=", ".join(targets))
    if dry_run:
        print(prompt, "\n" + "=" * 80)
        return {k: f"<{k}>" for k in VARIETIES} | {source_code: out[source_code]}
    drafts = parse_json(chat(params(DRAFT_MODEL, prompt, max_tokens=4096, effort="medium"))["output"], targets)
    out.update({k: drafts[k].strip() for k in targets})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", type=int, default=0,
                    help="draft only N scenarios (spread across sources) and print them")
    args = ap.parse_args()

    seeds = pd.read_csv(ROOT / "data/processed/scenarios_seed.csv")
    if args.dry_run:
        seeds = pd.concat([seeds[seeds.source == "cfpb"].head(1), seeds[seeds.source != "cfpb"].head(1)])
    elif args.pilot:
        seeds = seeds.iloc[::max(1, len(seeds) // args.pilot)].head(args.pilot)
    examples = darija_examples()

    rows = []
    for row in seeds.itertuples():
        msgs = build(row, examples, args.dry_run)
        source_code = "en" if row.source == "cfpb" else "ary"
        for code in VARIETIES:
            rows.append({"scenario_id": row.scenario_id, "category": row.category,
                         "urgency": row.urgency, "emotion_target": row.emotion_target,
                         "variety": code, "is_source": code == source_code,
                         "draft": msgs[code], "draft_model": DRAFT_MODEL,
                         "final": msgs[code] if code == source_code and row.source != "cfpb" else "",
                         "edited_by": "", "notes": ""})
    if args.dry_run:
        return
    out = pd.DataFrame(rows)
    if args.pilot:
        for sid, g in out.groupby("scenario_id", sort=False):
            print(f"\n=== {sid} | {g.category.iloc[0]} | {g.emotion_target.iloc[0]}")
            for r in g.itertuples():
                print(f"  [{r.variety:8}] {r.draft}")
        return
    out.to_csv(ROOT / "data/processed/messages_draft.csv", index=False)
    with pd.ExcelWriter(ROOT / "data/processed/messages_postedit.xlsx") as xw:
        for code in VARIETIES:
            out[out.variety == code].to_excel(xw, sheet_name=code, index=False)
    print(f"saved {len(out)} message drafts")


if __name__ == "__main__":
    main()
