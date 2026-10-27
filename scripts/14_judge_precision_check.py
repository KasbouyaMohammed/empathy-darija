"""Does a local judge give the same scores in fp16 as in fp32? (decides the precision of the Gemma-2 judge)

Runs the judge greedily on the same EPITOME items in both precisions and compares the parsed scores.
Usage:
    python scripts/14_judge_precision_check.py --judge gemma2_9b --n 100
Output: results/judge/precision_check_<judge>.csv and a printed summary.
"""
import argparse
import importlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
judge11 = importlib.import_module("11_judge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gemma2_9b")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    prompts, keys = judge11.build_prompts("epitome", args.judge, args.n)
    tmp = ROOT / "results/judge/precision_check_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for dtype in ("fp32", "fp16"):
        raw = judge11.run_open(args.judge, prompts, tmp / f"{dtype}_raw", 32, dtype)
        rows = []
        for k in prompts:
            try:
                obj = judge11.parse_json(raw[k], keys)
                rows.append({"item_key": k, **{f: obj[f] for f in keys}})
            except (KeyError, ValueError, TypeError):
                rows.append({"item_key": k})
        out[dtype] = pd.DataFrame(rows).set_index("item_key")
    shutil.rmtree(tmp)
    m = out["fp32"].join(out["fp16"], lsuffix="_fp32", rsuffix="_fp16")
    m.to_csv(ROOT / f"results/judge/precision_check_{args.judge}.csv")
    parsed = {d: out[d][keys].notna().all(axis=1).mean() for d in out}
    both = m.dropna()
    same = {f: (both[f"{f}_fp32"] == both[f"{f}_fp16"]).mean() for f in keys}
    print(f"{args.judge}: parsed fp32={parsed['fp32']:.3f} fp16={parsed['fp16']:.3f}; "
          f"identical scores on {len(both)} items: {same}")


if __name__ == "__main__":
    main()
