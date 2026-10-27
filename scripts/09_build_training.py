"""Training data for RQ4: customer messages answered by the calibrated teacher (Claude Sonnet 5.5, P2).

Never overlaps the test material:
  - CFPB complaints not used as seeds, excluding any with TF-IDF cosine > 0.6 to a seed narrative;
  - DarijaBanking Darija rows not used as seeds, in the natural subset or as style examples.
About 800 scenarios (640 CFPB, stratified by category x emotion; 160 DarijaBanking, 10 per intent)
x 5 varieties. Messages are drafted with the benchmark prompts of 03_build_messages.py (CFPB: condense,
then translate; DarijaBanking: translate from Darija), by Sonnet instead of Opus. Messages whose
identified variety is wrong are dropped. The teacher answers each message under P2; replies are dropped
if truncated, in the wrong variety, or if Claude Haiku confirms a request for a secret flagged by
SAFETY_PATTERNS. Split 90/10 by scenario.

Usage:
    python scripts/09_build_training.py
Output: data/processed/train/{train,val}.jsonl, data/processed/train/build_report.csv
"""
import importlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langid_darija import identify
from llm import batch, params
from prompts import POLICIES, SAFETY_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/train"
TEACHER = "claude-sonnet-5-5"
CHECKER = "claude-haiku-5-5"
SEED = 7
N_CFPB, PER_INTENT = 640, 10
MAX_SIM = 0.6
bm = importlib.import_module("03_build_messages")
sel = importlib.import_module("02_select_scenarios")
nat = importlib.import_module("02b_natural_subset")
VARIETIES = list(bm.VARIETIES)
SAFETY_CHECK = """Does the following customer-service reply ASK the customer to give or type a PIN, full card number, CVV, password or one-time code? Telling the customer never to share them is NOT asking. Answer only "yes" or "no".

Reply:
\"\"\"{reply}\"\"\""""


def cfpb_pool():
    seeds = pd.read_csv(ROOT / "data/processed/scenarios_seed.csv")
    df = pd.read_parquet(ROOT / "data/processed/cfpb_candidates.parquet")
    df = df[df.category.isin(sel.CATEGORIES) & df.n_words.between(50, 180)]
    df["emotion_group"] = df.emotion_auto.map(sel.EMOTION_GROUPS)
    df = df.dropna(subset=["emotion_group"])
    seed_cfpb = seeds[seeds.source == "cfpb"]
    df = df[~df.complaint_id.astype(str).isin(seed_cfpb.source_id.astype(str))]
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(pd.concat([df.narrative, seed_cfpb.seed_text]))
    sim = cosine_similarity(vec.transform(df.narrative), vec.transform(seed_cfpb.seed_text)).max(axis=1)
    print(f"CFPB pool: {len(df)} candidates, {(sim > MAX_SIM).sum()} removed for TF-IDF cosine > {MAX_SIM}")
    df = df[sim <= MAX_SIM].sample(frac=1, random_state=SEED)
    per_cell = N_CFPB // (len(sel.CATEGORIES) * 4)
    picked = df.groupby(["category", "emotion_group"]).head(per_cell)
    picked = pd.concat([picked, df.drop(picked.index).head(N_CFPB - len(picked))])   # fill thin cells
    return pd.DataFrame({"source": "cfpb", "source_id": picked.complaint_id.astype(str),
                         "category": picked.category, "urgency": picked.urgency,
                         "emotion_target": picked.emotion_group, "seed_text": picked.narrative})


def darijabanking_pool():
    parts = [pd.read_parquet(ROOT / f"data/raw/darijabanking/data/{s}-00000-of-00001.parquet").assign(split=s)
             for s in ("train", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = df.groupby("split").cumcount().astype(str).radd(df["split"] + ":")
    used = set(pd.read_csv(ROOT / "data/processed/scenarios_seed.csv").source_id) | \
        set(pd.read_csv(ROOT / "data/processed/natural_subset.csv").source_id) | set(sel.EXCLUDE)
    style = set(bm.darija_examples())
    intents = {i: ("info_" + i, "low") for i in sel.INFO_INTENTS} | nat.INTENTS
    df = df[df.label.isin(intents) & df.text.str.contains(sel.DARIJA_MARKERS) & ~df.text.str.contains("_")
            & ~df.row_id.isin(used) & ~df.text.isin(style)]
    df = df[df.text.str.split().str.len().between(4, 30)].drop_duplicates("text")
    picked = df.sample(frac=1, random_state=SEED).groupby("label").head(PER_INTENT)
    return pd.DataFrame({"source": "darijabanking", "source_id": picked.row_id,
                         "category": picked.label.map(lambda l: intents[l][0]),
                         "urgency": picked.label.map(lambda l: intents[l][1]),
                         "emotion_target": "", "seed_text": picked.text})


def draft_messages(scen):
    """Long table (scenario_id, variety, text) with the benchmark drafting prompts, Sonnet as drafter."""
    cfpb = scen[scen.source == "cfpb"]
    res = batch({f"{r.scenario_id}|condense": params(
        TEACHER, bm.CONDENSE_PROMPT.format(seed=r.seed_text, emotion=r.emotion_target), effort="low")
        for r in cfpb.itertuples()})
    src = {r.scenario_id: ("en", res.get(f"{r.scenario_id}|condense", {}).get("output", "").strip().strip('"'))
           for r in cfpb.itertuples()}
    src |= {r.scenario_id: ("ary", r.seed_text) for r in scen[scen.source != "cfpb"].itertuples()}
    examples = "\n".join(f"- {e}" for e in bm.darija_examples())
    reqs = {}
    for sid, (code, text) in src.items():
        if not text:
            continue
        targets = [v for v in VARIETIES if v != code]
        reqs[f"{sid}|translate"] = params(TEACHER, bm.TRANSLATE_PROMPT.format(
            targets="\n".join(f"- {k}: {bm.VARIETIES[k]}" for k in targets), examples=examples,
            source_name=bm.VARIETIES[code], text=text, keys=", ".join(targets)), max_tokens=4096, effort="low")
    res = batch(reqs)
    rows = []
    for sid, (code, text) in src.items():
        rows.append((sid, code, text))
        try:
            out = bm.parse_json(res[f"{sid}|translate"]["output"], [v for v in VARIETIES if v != code])
            rows += [(sid, v, str(t).strip()) for v, t in out.items() if v in VARIETIES]
        except (KeyError, ValueError):
            pass
    return pd.DataFrame(rows, columns=["scenario_id", "variety", "text"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scen = pd.concat([cfpb_pool(), darijabanking_pool()], ignore_index=True)
    scen.insert(0, "scenario_id", [f"T{i:04d}" for i in range(1, len(scen) + 1)])
    print(scen.groupby(["source", "category"]).size().to_string())

    msgs = draft_messages(scen)
    msgs = msgs[msgs.text.str.len() > 0]
    msgs["lid"] = [identify(t)[0] for t in msgs.text]
    report = {"scenarios": len(scen), "messages_drafted": len(msgs), "messages_wrong_variety": int((msgs.lid != msgs.variety).sum())}
    msgs = msgs[msgs.lid == msgs.variety]

    keys = {f"{r.scenario_id}|{r.variety}|teacher": r for r in msgs.itertuples()}
    res = batch({k: params(TEACHER, r.text, system=POLICIES["P2"], max_tokens=2048, effort="low") for k, r in keys.items()})
    rows = []
    for k, r in keys.items():
        rec = res.get(k)
        if rec:
            rows.append({"scenario_id": r.scenario_id, "variety": r.variety, "message": r.text,
                         "reply": rec["output"].strip(), "stop_reason": rec["stop_reason"]})
    rep = pd.DataFrame(rows)
    rep["reply_lid"] = [identify(t)[0] for t in rep.reply]
    rep["safety_hit"] = rep.reply.apply(lambda t: any(re.search(p, t, re.I) for p in SAFETY_PATTERNS))
    check = batch({f"{r.scenario_id}|{r.variety}|safety": params(CHECKER, SAFETY_CHECK.format(reply=r.reply),
                                                                 max_tokens=1024, effort="low")
                   for r in rep[rep.safety_hit].itertuples()})
    rep["safety_violation"] = [r.safety_hit and check.get(f"{r.scenario_id}|{r.variety}|safety", {})
                               .get("output", "").strip().lower().startswith("yes") for r in rep.itertuples()]
    report |= {"replies": len(rep), "dropped_truncated": int((rep.stop_reason != "end_turn").sum()),
               "dropped_wrong_variety": int((rep.reply_lid != rep.variety).sum()),
               "safety_regex_hits": int(rep.safety_hit.sum()), "dropped_safety": int(rep.safety_violation.sum())}
    keep = rep[(rep.stop_reason == "end_turn") & (rep.reply_lid == rep.variety) & ~rep.safety_violation]

    ids = sorted(keep.scenario_id.unique())
    val_ids = set(pd.Series(ids).sample(frac=0.1, random_state=SEED))
    for name, part in (("train", keep[~keep.scenario_id.isin(val_ids)]), ("val", keep[keep.scenario_id.isin(val_ids)])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part.itertuples():
                f.write(json.dumps({"prompt": [{"role": "user", "content": r.message}],
                                    "completion": [{"role": "assistant", "content": r.reply}],
                                    "scenario_id": r.scenario_id, "variety": r.variety}, ensure_ascii=False) + "\n")
        report[f"{name}_examples"] = len(part)
        report[f"{name}_scenarios"] = part.scenario_id.nunique()
    scen.drop(columns="seed_text").to_csv(OUT / "scenarios.csv", index=False)
    rep.to_csv(OUT / "teacher_replies.csv", index=False)
    pd.Series(report).to_csv(OUT / "build_report.csv", header=["value"])
    print(pd.Series(report).to_string())


if __name__ == "__main__":
    main()
