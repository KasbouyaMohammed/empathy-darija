"""Sample EPITOME seeker/response pairs with human empathy labels, for judge validation (RQ3).

EPITOME (Sharma et al., EMNLP 2020) labels each Reddit response on three dimensions
(emotional reactions, interpretations, explorations; levels 0/1/2). We keep pairs labelled on
all three, drop self-harm content and long posts, and stratify by the summed level so that
low-, mid- and high-empathy responses are equally represented.

Output: data/processed/epitome_sample.csv (240 pairs). The text is not redistributed.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N = 240
DIMS = {"emotional-reactions": "er", "interpretations": "ip", "explorations": "ex"}
SELF_HARM = (r"suicid|kill (?:my|him|her)self|killing myself|self[- ]?harm|cutting myself|cut myself|"
             r"end (?:my|it) (?:life|all)|overdose|want to die|wanna die|hang myself|slit")
BINS = [(0, 0), (1, 2), (3, 6)]   # summed level 0-6 -> none / some / high empathy (high is rare: 228)


def main():
    frames = []
    for name, code in DIMS.items():
        d = pd.read_csv(ROOT / f"data/raw/epitome/{name}-reddit.csv")
        frames.append(d[["sp_id", "rp_id", "seeker_post", "response_post", "level"]]
                      .rename(columns={"level": code}))
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f[["sp_id", "rp_id", f.columns[-1]]], on=["sp_id", "rp_id"])
    df = df.drop_duplicates(["sp_id", "rp_id"])

    text = df["seeker_post"].fillna("") + " " + df["response_post"].fillna("")
    df = df[~text.str.contains(SELF_HARM, case=False, regex=True)]
    df = df[df["seeker_post"].str.split().str.len().between(5, 80)
            & df["response_post"].str.split().str.len().between(5, 60)]
    df["total"] = df[list(DIMS.values())].sum(axis=1)
    print("summed-level distribution:", df["total"].value_counts().sort_index().to_dict())

    per_bin = N // len(BINS)
    picked = []
    for lo, hi in BINS:
        b = df[df["total"].between(lo, hi)]
        assert len(b) >= per_bin, (lo, hi, len(b))
        picked.append(b.sample(per_bin, random_state=SEED))
    out = pd.concat(picked).sort_values("total").reset_index(drop=True)
    out.insert(0, "epitome_id", [f"E{i:03d}" for i in range(1, len(out) + 1)])
    out.to_csv(ROOT / "data/processed/epitome_sample.csv", index=False)

    print(f"eligible pairs after filtering: {len(df)}")
    print(out[list(DIMS.values())].apply(lambda s: s.value_counts().sort_index()).fillna(0).astype(int).to_string())
    print(f"saved {len(out)} pairs")


if __name__ == "__main__":
    main()
