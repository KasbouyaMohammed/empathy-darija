"""Translate the EPITOME sample (seeker post + response) into fr, msa, ary and ary_latn.

The human empathy labels are carried over to every translation, so judges can be validated
against human gold in each variety (RQ3). The prompt asks for a translation that keeps the
empathy expressed in the response exactly as it is: no softening, no added warmth.

Output: data/processed/epitome_translations.csv (one row per pair x variety, incl. en source)
"""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from llm import batch, params

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"

_spec = importlib.util.spec_from_file_location("bm", Path(__file__).with_name("03_build_messages.py"))
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
VARIETIES, parse_json = _bm.VARIETIES, _bm.parse_json
TARGETS = [v for v in VARIETIES if v != "en"]

PROMPT = """Translate the online support exchange below (a post and a reply to it) into each requested language variety.

This is for a research benchmark on empathy: the reply's level of empathy must stay EXACTLY the same. Do not add, remove, soften or intensify any emotional reaction, interpretation of feelings, or question about the person's experience; keep blunt replies blunt and warm replies warm. Keep the informal online register, as a person from Morocco would write it in that variety. Do not add or remove information.

Requested varieties:
{targets}

Post:
\"\"\"{seeker}\"\"\"

Reply:
\"\"\"{response}\"\"\"

Return a JSON object whose keys are exactly: {keys}. Each value is an object {{"post": "...", "reply": "..."}}. No other text."""


def main():
    sample = pd.read_csv(ROOT / "data/processed/epitome_sample.csv")
    targets = "\n".join(f"- {k}: {VARIETIES[k]}" for k in TARGETS)
    requests = {
        r.epitome_id: params(MODEL, PROMPT.format(targets=targets, seeker=r.seeker_post.strip(),
                                                  response=r.response_post.strip(), keys=", ".join(TARGETS)),
                             max_tokens=6000, effort="low")
        for r in sample.itertuples()}
    results = batch(requests)

    rows, failed = [], []
    for r in sample.itertuples():
        base = {"epitome_id": r.epitome_id, "er": r.er, "ip": r.ip, "ex": r.ex, "total": r.total}
        rows.append(base | {"variety": "en", "post": r.seeker_post, "reply": r.response_post})
        rec = results.get(r.epitome_id)
        try:
            out = parse_json(rec["output"], TARGETS)
            for v in TARGETS:
                rows.append(base | {"variety": v, "post": out[v]["post"].strip(), "reply": out[v]["reply"].strip()})
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as e:
            failed.append((r.epitome_id, rec["stop_reason"] if rec else "missing", str(e)[:80]))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/processed/epitome_translations.csv", index=False)
    print(f"saved {len(out)} rows ({out.epitome_id.nunique()} pairs); failed: {len(failed)} {failed[:5]}")


if __name__ == "__main__":
    main()
