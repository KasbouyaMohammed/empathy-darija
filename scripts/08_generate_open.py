"""Replies of the open models, run locally (4x TITAN RTX), under P0 / P1 / P2 or with no system prompt.

Same messages as 07_generate_claude.py (benchmark after QC + natural subset). Each model uses its own
chat template; models without a system role (Gemma-2, Atlas-Chat) get the policy prepended to the user
turn. Weights run in fp32: Turing has no bf16 and Gemma-2-based models are unstable in fp16. Sampling
follows each model's default generation config (Qwen3: the model card's non-thinking settings), with a
fixed seed. Results are appended to the output file as they are produced, so a stopped run resumes.

Usage:
    python scripts/08_generate_open.py --model atlas9b [--pilot 10]
    python scripts/08_generate_open.py --model atlas2b --policies P0 P2
    python scripts/08_generate_open.py --model atlas2b --adapter results/lora/seed0 --policies none --name lora2b_s0
    python scripts/08_generate_open.py --model qwen3_8b --dtype fp16 --regenerate-truncated
Output: data/processed/responses_open_<name>.csv

Truncated replies (no end-of-turn token within MAX_NEW_TOKENS) are regenerated once with seed SEED + 1
(--regenerate-truncated); if the model decodes greedily, regeneration would reproduce the same output, so
they are only flagged. Rows still truncated afterwards get excluded = 1 (frozen plan, Exclusions).
"""
import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts import POLICIES

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "atlas9b": "MBZUAI-Paris/Atlas-Chat-9B",
    "atlas2b": "MBZUAI-Paris/Atlas-Chat-2B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    # Access to the official repo was granted on 2026-10-25. Fallback if it is not accessible: the ungated
    # mirror NousResearch/Meta-Llama-3.1-8B-Instruct (same SHA-256 for weights, config and tokenizer.json).
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2_9b": "google/gemma-2-9b-it",
}
OFFICIAL_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
NO_SYSTEM_ROLE = {"atlas9b", "atlas2b", "gemma2_9b"}
# Qwen3 model card, non-thinking mode; the shipped generation_config holds the thinking-mode values.
OVERRIDES = {"qwen3_8b": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}}
MAX_NEW_TOKENS = 512   # replies are capped at 120 words by the prompt; Arabic script costs more tokens
SEED = 0

load_items = importlib.import_module("07_generate_claude").load_items


def chat_prompt(tok, key, policy, text):
    if policy == "none":
        msgs = [{"role": "user", "content": text}]
    elif key in NO_SYSTEM_ROLE:
        msgs = [{"role": "user", "content": f"{POLICIES[policy]}\n\nCustomer message:\n{text}"}]
    else:
        msgs = [{"role": "system", "content": POLICIES[policy]}, {"role": "user", "content": text}]
    extra = {"enable_thinking": False} if key.startswith("qwen3") else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--adapter", help="LoRA adapter directory (peft)")
    ap.add_argument("--name", help="output name (default: --model)")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                    help="fp16 only for Qwen3 and Llama (Gemma-2-based models are unstable in fp16)")
    ap.add_argument("--shard", default="0/1", help="i/n: generate every n-th job starting at i (parallel GPUs)")
    ap.add_argument("--regenerate-truncated", action="store_true")
    args = ap.parse_args()
    if args.dtype == "fp16" and args.model in NO_SYSTEM_ROLE:
        ap.error("Gemma-2-based models must run in fp32")
    shard, n_shards = map(int, args.shard.split("/"))
    name = (args.name or args.model) + ("_pilot" if args.pilot else "")
    part = f".part{shard}of{n_shards}" if n_shards > 1 else ""
    out_path = ROOT / f"data/processed/responses_open_{name}{part}.csv"

    repo = MODELS[args.model]
    tok_repo = repo
    if args.model == "llama31_8b":
        try:   # official chat template once Meta grants access; the mirror's otherwise
            tok = AutoTokenizer.from_pretrained(OFFICIAL_LLAMA, padding_side="left")
            tok_repo = OFFICIAL_LLAMA
        except OSError:
            tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    else:
        tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Spread the weights evenly over the visible GPUs so each keeps room for the KV cache of large batches.
    n_gpu = torch.cuda.device_count()
    weights = Path(snapshot_download(repo, allow_patterns=["*.safetensors"])).glob("*.safetensors")
    size_gib = sum(f.stat().st_size for f in weights) / 2**30
    factor = 2 if args.dtype == "fp32" else 1          # bf16 checkpoints double in size in fp32
    per_gpu = f"{size_gib * factor / n_gpu + 1:.0f}GiB"
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map="auto",
                                                 max_memory={i: per_gpu for i in range(n_gpu)})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ROOT / args.adapter)
    model.eval()
    gen_cfg = model.generation_config
    for k, v in OVERRIDES.get(args.model, {}).items():
        setattr(gen_cfg, k, v)
    gen_cfg.max_new_tokens = MAX_NEW_TOKENS
    gen_cfg.pad_token_id = tok.pad_token_id
    model_id = repo + (f"+{args.adapter}" if args.adapter else "") + f" [{args.dtype}]" + \
        (f" (tokenizer: {tok_repo})" if tok_repo != repo else "")

    if args.regenerate_truncated:
        return regenerate_truncated(model, tok, gen_cfg, args, name, model_id)

    items = load_items(args.pilot)
    jobs = [(it, pol) for it in items.itertuples() for pol in args.policies][shard::n_shards]
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev.item_id, prev.policy))
    jobs = [(it, pol) for it, pol in jobs if (it.item_id, pol) not in done]
    prompts = [chat_prompt(tok, args.model, pol, it.text) for it, pol in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(prompts[i]))   # less padding per batch
    print(f"{model_id}: {len(jobs)} replies to generate ({len(done)} already done), sampling: "
          f"do_sample={gen_cfg.do_sample} T={gen_cfg.temperature} top_p={gen_cfg.top_p}", flush=True)

    set_seed(SEED)
    eos = set(gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id])
    bs, b = args.batch_size, 0
    while b < len(order):
        idx = order[b:b + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        try:
            with torch.no_grad():
                gen = model.generate(**enc, generation_config=gen_cfg)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  OOM: batch size -> {bs}", flush=True)
            continue
        b += len(idx)
        rows = []
        for j, i in enumerate(idx):
            new = gen[j, enc["input_ids"].shape[1]:].tolist()
            finished = any(t in eos for t in new)
            it, pol = jobs[i]
            rows.append({"set": it.set, "item_id": it.item_id, "scenario_id": it.scenario_id,
                         "category": it.category, "urgency": it.urgency, "variety": it.variety,
                         "message": it.text, "system": args.name or args.model, "policy": pol,
                         "reply": tok.decode(new, skip_special_tokens=True).strip(),
                         "stop_reason": "end_turn" if finished else "max_tokens", "model_id": model_id})
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"  {b}/{len(order)}", flush=True)

    out = pd.read_csv(out_path)
    print(out.groupby("policy")["stop_reason"].value_counts().unstack(fill_value=0).to_string())
    print(f"saved {len(out)} replies -> {out_path}")


def regenerate_truncated(model, tok, gen_cfg, args, name, model_id):
    files = sorted((ROOT / "data/processed").glob(f"responses_open_{name}.csv")) + \
        sorted((ROOT / "data/processed").glob(f"responses_open_{name}.part*.csv"))
    eos = set(gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id])
    set_seed(SEED + 1)
    for f in files:
        d = pd.read_csv(f)
        if "excluded" not in d:
            d["excluded"], d["regenerated"] = 0, 0
        idx = d.index[(d.stop_reason == "max_tokens") & (d.regenerated == 0)]
        if len(idx) and gen_cfg.do_sample:
            prompts = [chat_prompt(tok, args.model, d.at[i, "policy"], d.at[i, "message"]) for i in idx]
            for b in range(0, len(idx), args.batch_size):
                chunk = list(idx[b:b + args.batch_size])
                enc = tok(prompts[b:b + args.batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, generation_config=gen_cfg)
                for j, i in enumerate(chunk):
                    new = gen[j, enc["input_ids"].shape[1]:].tolist()
                    d.at[i, "reply"] = tok.decode(new, skip_special_tokens=True).strip()
                    d.at[i, "stop_reason"] = "end_turn" if any(t in eos for t in new) else "max_tokens"
        d.loc[idx, "regenerated"] = 1
        d["excluded"] = (d.stop_reason == "max_tokens").astype(int)
        d.to_csv(f, index=False)
        print(f"{f.name}: {len(idx)} truncated, {'regenerated' if gen_cfg.do_sample else 'greedy, not regenerated'}; "
              f"still truncated (excluded): {int(d.excluded.sum())}", flush=True)


if __name__ == "__main__":
    main()
