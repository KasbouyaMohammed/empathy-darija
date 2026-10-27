"""LLM judges: rubric scores for every reply (JUDGE_RUBRIC) and EPITOME empathy levels (EPITOME_JUDGE, RQ3).

Judges: Claude Haiku 5.5 (all replies), Claude Sonnet 5.5 (stratified sample of 1,500 replies), and three
local open judges (Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-2-9B-it) on all replies. Claude judges run
through the Batches API with default sampling (the API rejects sampling parameters); open judges decode
greedily so their scores are reproducible. Judges never see which system wrote a reply.

Usage:
    python scripts/11_judge.py --judge haiku --task replies
    python scripts/11_judge.py --judge qwen3_8b --task epitome
Output: results/judge/<judge>_<task>.csv (one row per judged item, parsed fields + raw output)
"""
import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from prompts import EPITOME_JUDGE, JUDGE_KEYS, JUDGE_RUBRIC, checklist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/judge"
CLAUDE = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
OPEN = {"qwen3_8b", "llama31_8b", "gemma2_9b"}
SONNET_SAMPLE = 1500
SEED = 11
parse_json = importlib.import_module("03_build_messages").parse_json


def load_replies():
    files = [ROOT / "data/processed/responses_claude.csv"] + sorted((ROOT / "data/processed").glob("responses_open_*.csv"))
    files = [f for f in files if "pilot" not in f.name and "smoke" not in f.name]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.reply.notna() & (d.reply.str.strip() != "")]
    if "excluded" in d:   # still truncated after one regeneration (08_generate_open.py)
        d = d[d.excluded.fillna(0) != 1]
    d["item_key"] = d.item_id + "|" + d.system + "|" + d.policy
    return d.drop_duplicates("item_key")


def sonnet_sample(d):
    """1,500 replies stratified by system x policy x variety."""
    cells = d.groupby(["system", "policy", "variety"])
    per = max(1, SONNET_SAMPLE // cells.ngroups)
    s = cells.sample(n=per, random_state=SEED, replace=False) if (cells.size() >= per).all() else \
        cells.apply(lambda g: g.sample(min(per, len(g)), random_state=SEED), include_groups=True)
    return s.head(SONNET_SAMPLE)


def build_prompts(task, judge, pilot):
    if task == "replies":
        d = load_replies()
        if judge == "sonnet":
            d = sonnet_sample(d)
        if pilot:
            d = d.sample(pilot, random_state=SEED)
        prompts = {r.item_key: JUDGE_RUBRIC.format(
            actions="\n".join(f"{i + 1}. {a}" for i, a in enumerate(checklist(r.category))),
            message=r.message, reply=r.reply) for r in d.itertuples()}
        keys = JUDGE_KEYS
    else:
        d = pd.read_csv(ROOT / "data/processed/epitome_final.csv")
        d["item_key"] = d.epitome_id.astype(str) + "|" + d.variety
        if pilot:
            d = d.sample(pilot, random_state=SEED)
        prompts = {r.item_key: EPITOME_JUDGE.format(post=r.post, reply=r.reply) for r in d.itertuples()}
        keys = ["er", "ip", "ex"]
    return prompts, keys


def run_claude(model, prompts):
    from llm import batch, params
    res = batch({k: params(model, p, max_tokens=1500, effort="low") for k, p in prompts.items()})
    return {k: res[k]["output"] for k in prompts if k in res}


def run_open(key, prompts, raw_stem, batch_size, dtype="fp32", shard=0, n_shards=1):
    """Greedy decoding with the judge's chat template. Raw outputs are appended to
    <raw_stem>.part<i>of<n>.csv per shard, so a stopped run resumes; a run with --shard 0/1 after all
    shards finished only collects the parts."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gen = importlib.import_module("08_generate_open")
    repo = gen.MODELS[key]
    done = {}
    for f in raw_stem.parent.glob(raw_stem.name + "*.csv"):
        prev = pd.read_csv(f)
        done.update(zip(prev.item_key, prev.raw))
    out_raw = raw_stem.parent / f"{raw_stem.name}.part{shard}of{n_shards}.csv"
    todo = [k for k in list(prompts)[shard::n_shards] if k not in done]
    if todo:
        tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        n_gpu = torch.cuda.device_count()
        size_gib = sum(f.stat().st_size for f in Path(snapshot_download(repo, allow_patterns=["*.safetensors"]))
                       .glob("*.safetensors")) / 2**30
        factor = 2 if dtype == "fp32" else 1
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float32 if dtype == "fp32" else torch.float16, device_map="auto",
            max_memory={i: f"{size_gib * factor / n_gpu + 1:.0f}GiB" for i in range(n_gpu)}).eval()
        extra = {"enable_thinking": False} if key.startswith("qwen3") else {}
        texts = {k: tok.apply_chat_template([{"role": "user", "content": prompts[k]}], tokenize=False,
                                            add_generation_prompt=True, **extra) for k in todo}
        todo.sort(key=lambda k: len(texts[k]))
        bs, i = batch_size, 0
        print(f"{repo}: {len(todo)} items to judge ({len(done)} done)", flush=True)
        while i < len(todo):
            chunk = todo[i:i + bs]
            enc = tok([texts[k] for k in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=400, do_sample=False, temperature=None,
                                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"  OOM: batch size -> {bs}", flush=True)
                continue
            raws = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
            pd.DataFrame({"item_key": chunk, "raw": raws}).to_csv(out_raw, mode="a", header=not out_raw.exists(),
                                                                  index=False)
            done.update(zip(chunk, raws))
            i += len(chunk)
            print(f"  {i}/{len(todo)}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(CLAUDE) + sorted(OPEN))
    ap.add_argument("--task", required=True, choices=["replies", "epitome"])
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--shard", default="0/1", help="i/n: judge every n-th item starting at i")
    args = ap.parse_args()
    # Gemma-2 in fp16 gave scores identical to fp32 on 100 EPITOME items (14_judge_precision_check.py),
    # so all local judges may run in fp16 (deviation 5).
    shard, n_shards = map(int, args.shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.judge}_{args.task}" + ("_pilot" if args.pilot else "")

    prompts, keys = build_prompts(args.task, args.judge, args.pilot)
    if args.judge in CLAUDE:
        raw = run_claude(CLAUDE[args.judge], prompts)
    else:
        raw = run_open(args.judge, prompts, OUT / f"{name}_raw", args.batch_size, args.dtype, shard, n_shards)
        if n_shards > 1:
            return   # parsing happens in a final run with --shard 0/1

    rows, bad = [], 0
    for k in prompts:
        row = {"item_key": k, "judge": args.judge, "raw": raw.get(k)}
        try:
            obj = parse_json(raw[k], keys)
            row |= {f: (json.dumps(obj[f]) if isinstance(obj[f], list) else obj[f]) for f in keys}
            row["parse_ok"] = True
        except (KeyError, ValueError, TypeError):
            row["parse_ok"] = False
            bad += 1
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{args.judge} {args.task}: {len(out)} items, {bad} unparsable -> {OUT / f'{name}.csv'}")


if __name__ == "__main__":
    main()
