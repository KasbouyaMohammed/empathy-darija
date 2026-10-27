"""Claude API client with an on-disk cache, for single calls and Message Batches.

The key is read from ANTHROPIC_API_KEY, or from a git-ignored `.env` file at the project
root containing a line `ANTHROPIC_API_KEY=...`. Every result is appended to a JSONL cache
keyed by a hash of the request parameters, so reruns never pay twice and the exact raw
outputs are kept for the data release.

Sampling parameters (temperature, top_p, top_k) are not sent: since 2026-10-21 the API
rejects them on every model.
"""
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/claude_calls.jsonl"
_client = None
_cache = None


def _get_client():
    global _client
    if _client is None:
        env = ROOT / ".env"
        if "ANTHROPIC_API_KEY" not in os.environ and env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
        _client = anthropic.Anthropic(max_retries=5)
    return _client


def params(model, user, system=None, max_tokens=2048, effort="low"):
    """Request parameters shared by single calls and batches."""
    p = {"model": model, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": user}],
         "output_config": {"effort": effort}}
    if system:
        p["system"] = system
    return p


def _key(p):
    return hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:   # a line torn by a crash; the call is simply redone
                    continue
                _cache[rec["key"]] = rec
    return _cache


def _store(key, p, message, via_batch=False):
    text = "".join(b.text for b in message.content if b.type == "text")
    rec = {"key": key, "params": p, "model_id": message.model, "stop_reason": message.stop_reason,
           "output": text, "usage": message.usage.model_dump(), "via_batch": via_batch,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)   # several scripts may append at once
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    _load_cache()[key] = rec
    return rec


# Hard budget stop: no new paid request once spent() (an upper bound, about $4.6 above the
# Console figure) reaches this amount.
BUDGET_USD = 60.0


def _check_budget():
    if spent() >= BUDGET_USD:
        raise RuntimeError(f"budget guard: spent() = ${spent():.2f} >= ${BUDGET_USD:.0f}; no new API requests")


def chat(p):
    """One synchronous call; returns the cached record (output, stop_reason, usage, ...)."""
    key = _key(p)
    cache = _load_cache()
    if key not in cache:
        _check_budget()
        _store(key, p, _get_client().messages.create(**p))
    return cache[key]


def batch(requests, poll_seconds=60):
    """Run {custom_id: params} through the Batches API (50% price); returns {custom_id: record}.

    Requests already in the cache are not resubmitted.
    """
    cache = _load_cache()
    keys = {cid: _key(p) for cid, p in requests.items()}
    todo = {cid: p for cid, p in requests.items() if keys[cid] not in cache}
    if todo:
        _check_budget()
        client = _get_client()
        # The API only accepts [a-zA-Z0-9_-]{1,64} as custom_id, so send positional ids.
        wire = {f"r{i}": cid for i, cid in enumerate(todo)}
        b = client.messages.batches.create(requests=[
            Request(custom_id=wid, params=MessageCreateParamsNonStreaming(**todo[cid]))
            for wid, cid in wire.items()])
        print(f"batch {b.id}: {len(todo)} requests submitted")
        while b.processing_status != "ended":
            time.sleep(poll_seconds)
            b = client.messages.batches.retrieve(b.id)
            print(f"  {b.processing_status}: {b.request_counts.processing} processing")
        failed = []
        for r in client.messages.batches.results(b.id):
            cid = wire[r.custom_id]
            if r.result.type == "succeeded":
                _store(keys[cid], todo[cid], r.result.message, via_batch=True)
            else:
                failed.append((cid, r.result.type))
        if failed:
            print(f"  {len(failed)} failed (rerun to retry): {failed[:5]}")
    return {cid: cache[k] for cid, k in keys.items() if k in cache}


def spent():
    """Spend so far from cached usage (USD list prices; batch requests at 50%).

    Records cached before the via_batch flag existed count at full price, so the figure is an upper bound.
    """
    prices = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
              "claude-haiku-5-5": (0.10, 0.50)}
    total = 0.0
    for rec in _load_cache().values():
        pin, pout = prices.get(rec["params"]["model"], (0, 0))
        u = rec["usage"]
        total += (u["input_tokens"] * pin + u["output_tokens"] * pout) / 1e6 * (0.5 if rec.get("via_batch") else 1.0)
    return total
