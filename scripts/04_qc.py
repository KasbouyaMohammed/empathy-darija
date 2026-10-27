"""Automatic quality control of translated texts (replaces human post-editing).

For every non-English text:
  1. back-translation to English by a different model than the drafter (Claude Sonnet 5.5, batch);
  2. LaBSE cosine similarity between the English source and the back-translation (meaning kept?)
     and between the English source and the translation itself (cross-lingual);
  3. language-variety identification of the translation (GlotLID + Darija markers, see langid_darija.py).
A text passes if back-translation similarity >= THRESHOLD and the language ID matches.

Usage:
    python scripts/04_qc.py benchmark   # data/processed/messages_draft.csv
    python scripts/04_qc.py epitome     # data/processed/epitome_translations.csv
Output: data/processed/qc_<name>.csv
"""
import importlib
import re
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from langid_darija import identify
from llm import batch, params

ROOT = Path(__file__).resolve().parents[1]
BT_MODEL = "claude-sonnet-5-5"
THRESHOLD = 0.80          # sensitivity reported at 0.75 and 0.85
BT_PROMPT = ("Translate the following text into English as literally and faithfully as possible, "
             "keeping its tone. Output only the translation.\n\n\"\"\"{text}\"\"\"")
# The free-text prompt above sometimes yields commentary ("This looks like Darija...", glosses,
# "Correction -") around the translation, which deflates the similarity. Those cases are re-asked
# once with a JSON-only prompt; the column bt_retry records which rows were.
BT_PROMPT_JSON = ("Translate the following text into English as literally and faithfully as possible, "
                  "keeping its tone. Return only a JSON object {{\"translation\": \"...\"}} with no "
                  "comments, notes or glosses.\n\n\"\"\"{text}\"\"\"")
CONTAMINATED = re.compile(r"(?i)\bnote\b|correction|darija|arabizi|literal|transliterat|appears to be|"
                          r"looks like|moroccan arabic|\*\*|^\s*-\s|\n\s*\n", re.M)
parse_json = importlib.import_module("03_build_messages").parse_json


def load(name):
    """Long table: id, variety, field, text, source (the English text it should mean)."""
    rows = []
    if name == "benchmark":
        d = pd.read_csv(ROOT / "data/processed/messages_draft.csv")
        src = d[d.variety == "en"].set_index("scenario_id")["draft"]
        for r in d[d.variety != "en"].itertuples():
            rows.append((r.scenario_id, r.variety, "message", r.draft, src[r.scenario_id]))
    else:
        d = pd.read_csv(ROOT / "data/processed/epitome_translations.csv")
        src = d[d.variety == "en"].set_index("epitome_id")
        for r in d[d.variety != "en"].itertuples():
            for field in ("post", "reply"):
                rows.append((r.epitome_id, r.variety, field, getattr(r, field), src.loc[r.epitome_id, field]))
    return pd.DataFrame(rows, columns=["id", "variety", "field", "text", "source"])


def main(name):
    df = load(name)
    reqs = {f"{r.id}|{r.variety}|{r.field}": params(BT_MODEL, BT_PROMPT.format(text=r.text),
                                                    max_tokens=2000, effort="low")
            for r in df.itertuples()}
    res = batch(reqs)
    df["back_translation"] = [res.get(f"{r.id}|{r.variety}|{r.field}", {}).get("output", "").strip()
                              for r in df.itertuples()]
    df["bt_retry"] = df.back_translation.eq("") | df.back_translation.str.contains(CONTAMINATED)
    retry = {f"{r.id}|{r.variety}|{r.field}": params(BT_MODEL, BT_PROMPT_JSON.format(text=r.text),
                                                     max_tokens=2000, effort="low")
             for r in df[df.bt_retry].itertuples()}
    res = batch(retry)
    for i in df.index[df.bt_retry]:
        cid = f"{df.at[i, 'id']}|{df.at[i, 'variety']}|{df.at[i, 'field']}"
        try:
            df.at[i, "back_translation"] = parse_json(res[cid]["output"], ["translation"])["translation"].strip()
        except (KeyError, ValueError):
            pass  # keep the first back-translation
    print(f"back-translations re-asked: {df.bt_retry.sum()}, still contaminated: "
          f"{df.back_translation.str.contains(CONTAMINATED).sum()}")

    labse = SentenceTransformer("sentence-transformers/LaBSE")
    emb = lambda col: labse.encode(df[col].fillna("").tolist(), normalize_embeddings=True, batch_size=64)
    e_src, e_bt, e_txt = emb("source"), emb("back_translation"), emb("text")
    df["sim_backtranslation"] = (e_src * e_bt).sum(1).round(3)
    df["sim_crosslingual"] = (e_src * e_txt).sum(1).round(3)

    ids = [identify(t) for t in df["text"]]
    df["lid"] = [v for v, _ in ids]
    df["glotlid"] = [d.get("glotlid") for _, d in ids]
    df["lid_ok"] = df["lid"] == df["variety"]
    df["pass"] = (df["sim_backtranslation"] >= THRESHOLD) & df["lid_ok"]

    out = ROOT / f"data/processed/qc_{name}.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("variety").agg(n=("pass", "size"), pass_rate=("pass", "mean"),
                                        lid_ok=("lid_ok", "mean"), sim_bt=("sim_backtranslation", "median"))
    print(summary.round(3).to_string())
    for t in (0.75, 0.80, 0.85):
        print(f"pass rate at threshold {t}: {((df.sim_backtranslation >= t) & df.lid_ok).mean():.3f}")
    print("most frequent wrong LID labels:", df.loc[~df.lid_ok, "lid"].value_counts().head(5).to_dict())
    print(f"saved -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmark")
