"""Finalize translated texts after automatic QC: regenerate failures once, re-check, freeze.

A text passes if back-translation similarity (LaBSE) >= 0.80 and the language identifier returns
the intended variety (scripts/04_qc.py). Each failing (item, variety) is regenerated once by the
original translator with the original prompt restricted to that variety, then re-checked with the
same QC. The regenerated text is kept only if it passes; otherwise the version with the higher
similarity is kept and flagged (qc_pass = False) for sensitivity analyses.

Usage:
    python scripts/04b_finalize.py
Outputs:
    data/processed/messages_final.csv   scenario_id, category, urgency, emotion_target, variety, text
    data/processed/messages_final_qc.csv, data/processed/epitome_final.csv (with QC columns)
    results/qc_summary.csv
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from langid_darija import identify
from llm import batch, params

ROOT = Path(__file__).resolve().parents[1]
qc = importlib.import_module("04_qc")
bm = importlib.import_module("03_build_messages")
ep = importlib.import_module("06_translate_epitome")
THRESHOLDS = (0.75, 0.80, 0.85)


def regenerate_benchmark(fails):
    """{(scenario_id, variety): text} via 03_build_messages' translation prompt, one variety each."""
    drafts = pd.read_csv(ROOT / "data/processed/messages_draft.csv")
    src = drafts[drafts.is_source].set_index("scenario_id")
    examples = bm.darija_examples()
    reqs = {}
    for sid, v in fails:
        s = src.loc[sid]
        prompt = bm.TRANSLATE_PROMPT.format(
            targets=f"- {v}: {bm.VARIETIES[v]}", examples="\n".join(f"- {e}" for e in examples),
            source_name=bm.VARIETIES[s.variety], text=s.draft, keys=v)
        reqs[f"{sid}|{v}|regen"] = params(bm.DRAFT_MODEL, prompt, max_tokens=4096, effort="medium")
    res = batch(reqs)
    out = {}
    for sid, v in fails:
        try:
            out[(sid, v)] = bm.parse_json(res[f"{sid}|{v}|regen"]["output"], [v])[v].strip()
        except (KeyError, ValueError):
            pass
    return out


def regenerate_epitome(fails):
    """{(epitome_id, variety): {"post", "reply"}} via 06_translate_epitome's prompt, one variety each."""
    sample = pd.read_csv(ROOT / "data/processed/epitome_sample.csv").set_index("epitome_id")
    reqs = {}
    for eid, v in fails:
        r = sample.loc[eid]
        prompt = ep.PROMPT.format(targets=f"- {v}: {bm.VARIETIES[v]}", seeker=r.seeker_post.strip(),
                                  response=r.response_post.strip(), keys=v)
        reqs[f"{eid}|{v}|regen"] = params(ep.MODEL, prompt, max_tokens=6000, effort="low")
    res = batch(reqs)
    out = {}
    for eid, v in fails:
        try:
            obj = bm.parse_json(res[f"{eid}|{v}|regen"]["output"], [v])[v]
            out[(eid, v)] = {"post": obj["post"].strip(), "reply": obj["reply"].strip()}
        except (KeyError, ValueError, TypeError):
            pass
    return out


def check(df, labse):
    """Same QC as 04_qc.py on a long table (id, variety, field, text, source); JSON back-translation."""
    reqs = {f"{r.id}|{r.variety}|{r.field}|regen": params(qc.BT_MODEL, qc.BT_PROMPT_JSON.format(text=r.text),
                                                          max_tokens=2000, effort="low")
            for r in df.itertuples()}
    res = batch(reqs)
    bt = []
    for r in df.itertuples():
        try:
            bt.append(qc.parse_json(res[f"{r.id}|{r.variety}|{r.field}|regen"]["output"],
                                    ["translation"])["translation"].strip())
        except (KeyError, ValueError):
            bt.append("")
    df = df.assign(back_translation=bt)
    emb = lambda col: labse.encode(df[col].fillna("").tolist(), normalize_embeddings=True, batch_size=64)
    df["sim_backtranslation"] = (emb("source") * emb("back_translation")).sum(1).round(3)
    df["lid"] = [identify(t)[0] for t in df.text]
    df["lid_ok"] = df.lid == df.variety
    df["pass"] = (df.sim_backtranslation >= qc.THRESHOLD) & df.lid_ok
    return df


def choose(old, new):
    """Per (id, variety): keep the regenerated text if all its fields pass, else the better version."""
    keys = ["id", "variety"]
    o = old.groupby(keys).agg(ok=("pass", "all"), sim=("sim_backtranslation", "min"))
    n = new.groupby(keys).agg(ok=("pass", "all"), sim=("sim_backtranslation", "min"))
    use_new = n.ok | (~o.reindex(n.index).ok & (n.sim > o.reindex(n.index).sim))
    picked = pd.concat([old.set_index(keys).drop(index=use_new.index[use_new]).reset_index().assign(regenerated=False),
                        new.set_index(keys).loc[use_new.index[use_new]].reset_index().assign(regenerated=True)])
    return picked


def rates(df, label):
    rows = []
    for v, g in [("all", df)] + list(df.groupby("variety")):
        unit = g.groupby(["id", "variety"]).agg(sim=("sim_backtranslation", "min"), lid=("lid_ok", "all"))
        rows.append({"stage": label, "variety": v, "n": len(unit),
                     **{f"pass@{t:.2f}": round(((unit.sim >= t) & unit.lid).mean(), 3) for t in THRESHOLDS}})
    return rows


def main():
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    summary = []

    # Benchmark messages.
    old = pd.read_csv(ROOT / "data/processed/qc_benchmark.csv")
    fails = sorted({(r.id, r.variety) for r in old[~old["pass"]].itertuples()})
    regen = regenerate_benchmark(fails)
    src = old.drop_duplicates("id").set_index("id").source
    new = check(pd.DataFrame([(sid, v, "message", t, src[sid]) for (sid, v), t in regen.items()],
                             columns=["id", "variety", "field", "text", "source"]), labse)
    final = choose(old, new)
    summary += rates(old, "initial") + rates(final, "after_regeneration")
    print(f"benchmark: {len(fails)} failing, {len(regen)} regenerated, "
          f"{final.regenerated.sum()} replaced, {(~final['pass']).sum()} still failing")

    drafts = pd.read_csv(ROOT / "data/processed/messages_draft.csv")
    text = {(r.id, r.variety): r.text for r in final.itertuples()}
    drafts["text"] = [text.get((r.scenario_id, r.variety), r.draft) for r in drafts.itertuples()]
    cols = ["scenario_id", "category", "urgency", "emotion_target", "variety", "text"]
    drafts[cols].to_csv(ROOT / "data/processed/messages_final.csv", index=False)
    final.rename(columns={"id": "scenario_id"}).to_csv(ROOT / "data/processed/messages_final_qc.csv", index=False)

    # EPITOME translations.
    old = pd.read_csv(ROOT / "data/processed/qc_epitome.csv")
    fails = sorted({(r.id, r.variety) for r in old[~old["pass"]].itertuples()})
    regen = regenerate_epitome(fails)
    src = old.drop_duplicates(["id", "field"]).set_index(["id", "field"]).source
    new = check(pd.DataFrame([(eid, v, f, t[f], src[(eid, f)]) for (eid, v), t in regen.items()
                              for f in ("post", "reply")],
                             columns=["id", "variety", "field", "text", "source"]), labse)
    final = choose(old, new)
    summary += [r | {"set": "epitome"} for r in rates(old, "initial") + rates(final, "after_regeneration")]
    print(f"epitome: {len(fails)} failing, {len(regen)} regenerated, "
          f"{final.groupby(['id', 'variety']).regenerated.first().sum()} replaced, "
          f"{(~final.groupby(['id', 'variety'])['pass'].all()).sum()} still failing")

    trans = pd.read_csv(ROOT / "data/processed/epitome_translations.csv")
    wide = final.pivot_table(index=["id", "variety"], columns="field", values="text", aggfunc="first")
    unit = final.groupby(["id", "variety"]).agg(qc_pass=("pass", "all"), regenerated=("regenerated", "first"),
                                                qc_sim=("sim_backtranslation", "min"))
    rows = []
    for r in trans.itertuples(index=False):
        d = r._asdict()
        if r.variety != "en":
            d["post"], d["reply"] = wide.loc[(r.epitome_id, r.variety), ["post", "reply"]]
            d |= unit.loc[(r.epitome_id, r.variety)].to_dict()
        else:
            d |= {"qc_pass": True, "regenerated": False, "qc_sim": 1.0}
        rows.append(d)
    pd.DataFrame(rows).to_csv(ROOT / "data/processed/epitome_final.csv", index=False)

    out = pd.DataFrame(summary)
    out["set"] = out["set"].fillna("benchmark") if "set" in out else "benchmark"
    out.to_csv(ROOT / "results/qc_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
