"""LoRA fine-tuning of Atlas-Chat-2B on calibrated (P2) teacher replies, used with no system prompt (RQ4).

Port of kaggle/train_lora_atlas.ipynb to a rented GPU server (4x TITAN RTX, Turing: no bf16). The model trains in
fp32 because Gemma-2-based models are unstable in fp16. The loss is computed on the reply only.

Data: data/processed/train/{train,val}.jsonl from 09_build_training.py, one object per line:
    {"prompt": [{"role": "user", "content": ...}], "completion": [{"role": "assistant", "content": ...}],
     "scenario_id": ...}
The data-size ablation keeps a random fraction of training *scenarios* (all their varieties), with a
seed-independent draw so that the 25% set is contained in the 50% set.

Usage:
    python scripts/10_train_lora.py --seed 0 [--fraction 0.25] [--smoke]
Output: results/lora/<name>/ (adapter, tokenizer, log_history.json, run_config.json)
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "MBZUAI-Paris/Atlas-Chat-2B"
DATA_DIR = ROOT / "data/processed/train"
MAX_LEN = 1024
ABLATION_SEED = 1234   # fixed, so smaller fractions are nested in larger ones


def subset(ds, fraction):
    if fraction >= 1:
        return ds
    scenarios = sorted(set(ds["scenario_id"]))
    random.Random(ABLATION_SEED).shuffle(scenarios)
    keep = set(scenarios[:round(len(scenarios) * fraction)])
    return ds.filter(lambda r: r["scenario_id"] in keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="5 steps on a toy set, to check the environment")
    args = ap.parse_args()
    name = "smoke" if args.smoke else f"frac{int(args.fraction * 100)}_seed{args.seed}"
    out_dir = ROOT / "results/lora" / name

    if args.smoke:
        demo = [("كارطتي تسرقات ودابا شفت جوج عمليات ما درتهومش، عافاكم عاونوني دغيا!",
                 "فهمت القلق ديالك، وغادي نعاونك دابا. أولا بلوكي الكارط من التطبيق ولا عيط لخدمة الزبناء."),
                ("شنو هوما الوثائق باش نحل حساب جديد؟",
                 "باش تحل حساب خاصك البطاقة الوطنية وشهادة السكنى. تقدر تبدا الطلب من التطبيق ولا من الوكالة.")]
        rows = [{"prompt": [{"role": "user", "content": q}], "completion": [{"role": "assistant", "content": a}],
                 "scenario_id": f"demo{i}"} for i, (q, a) in enumerate(demo)] * 8
        train_ds, val_ds = Dataset.from_list(rows), Dataset.from_list(rows[:4])
    else:
        train_ds = subset(load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train"),
                          args.fraction)
        val_ds = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"), split="train")
    extra = lambda ds: [c for c in ds.column_names if c not in ("prompt", "completion")]
    train_ds, val_ds = train_ds.remove_columns(extra(train_ds)), val_ds.remove_columns(extra(val_ds))
    print(f"{name}: {len(train_ds)} train / {len(val_ds)} val examples", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=str(out_dir), seed=args.seed,
        num_train_epochs=1 if args.smoke else 2, max_steps=5 if args.smoke else -1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.03,
        gradient_checkpointing=True, max_length=MAX_LEN,
        bf16=False, fp16=False,  # trl defaults to bf16, which Turing lacks: train in fp32
        completion_only_loss=True,
        eval_strategy="steps", eval_steps=50, logging_steps=10,
        save_strategy="epoch", save_total_limit=1,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    json.dump(trainer.state.log_history, open(out_dir / "log_history.json", "w"), indent=1)
    json.dump(vars(args) | {"model": MODEL_ID, "n_train": len(train_ds), "n_val": len(val_ds),
                            "lora": {"r": 16, "alpha": 32, "dropout": 0.05}, "lr": 2e-4, "epochs": 2,
                            "effective_batch": 16, "precision": "fp32"},
              open(out_dir / "run_config.json", "w"), indent=1)
    print(f"saved adapter -> {out_dir}")


if __name__ == "__main__":
    main()
