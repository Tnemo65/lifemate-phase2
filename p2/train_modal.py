"""SFT LoRA Qwen3-4B-Instruct-2507 tren Modal.

    python -m modal run train_modal.py --dry-run          # B1: 30 buoc tren mau DAI NHAT (kiem OOM + toc do)
    python -m modal run train_modal.py --tag sft_v1       # B2: train du 2 epoch, merge, luu vao Volume lifemate-ckpt

Truoc khi train: tokenize lai tren Modal va so sha256 voi ban local (artifacts/sft_train.stats.json). Lech -> dung.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
LOCAL_DATA = ROOT.parent / "train+public_test" / "train+public_test"
MODEL_DIR = "/models/Qwen3-4B-Instruct-2507"

models = modal.Volume.from_name("lifemate-models")
ckpt = modal.Volume.from_name("lifemate-ckpt", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==4.57.6", "peft", "accelerate", "numpy")
    .add_local_dir(LOCAL_DATA, "/data", ignore=["**/audio/**", "**/.DS_Store", "**/__pycache__/**",
                                                  "**/*_environment/**"])
    .add_local_dir(ROOT, "/root/p2", ignore=["runs/**", "teacher_runs/**", "**/__pycache__/**", "tokenizer/**", "artifacts/sft_*.jsonl", "*.log"])
)
app = modal.App("lifemate-p2-train", image=image)

HP = dict(r=32, alpha=64, dropout=0.05, lr=1e-4, epochs=2, accum=16, warmup=0.03, seed=42)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@app.function(gpu=["A100-80GB", "H100"], volumes={"/models": models, "/ckpt": ckpt}, timeout=4 * 3600)
def train(tag: str, dry_run: bool, splits: str = "train", mask: float = 0.0, epochs: int = 0, synth: str = ""):
    import random
    import sys

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    sys.path.insert(0, "/root/p2")
    import sft_data

    HP = dict(globals()["HP"], epochs=epochs or globals()["HP"]["epochs"])  # epochs > 0: ghi de (duong cong hoc)
    random.seed(HP["seed"]); np.random.seed(HP["seed"]); torch.manual_seed(HP["seed"])

    # 1) kiem tokenizer tren Modal == local
    names = splits.split(",")
    local_stats = json.load(open(f"/root/p2/artifacts/sft_{sft_data.data_name(names, mask, synth)}.stats.json"))
    rows, stats = sft_data.build(names, MODEL_DIR, "/data", check_replay=False, verbose=False, mask_ratio=mask, synth=synth)
    if stats["sha256"] != local_stats["sha256"]:
        raise RuntimeError(f"Du lieu tokenize tren Modal KHAC local: {stats['sha256']} != {local_stats['sha256']}")
    print("tokenize khop local:", stats["sha256"][:16], "n =", stats["n"])

    if dry_run:
        rows = sorted(rows, key=lambda r: -len(r["input_ids"]))[:30]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, k):
            r = rows[k]
            return {"input_ids": torch.tensor(r["input_ids"]), "labels": torch.tensor(r["labels"]),
                    "attention_mask": torch.ones(len(r["input_ids"]), dtype=torch.long)}

    def collate(b):
        assert len(b) == 1
        return {k: v.unsqueeze(0) for k, v in b[0].items()}

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"],
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out_dir = f"/ckpt/{tag}"
    args = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if dry_run else HP["accum"],
        max_steps=30 if dry_run else -1, num_train_epochs=HP["epochs"],
        learning_rate=HP["lr"], lr_scheduler_type="cosine", warmup_ratio=HP["warmup"],
        bf16=True, logging_steps=1 if dry_run else 5, save_strategy="no", report_to=[],
        seed=HP["seed"], data_seed=HP["seed"], dataloader_num_workers=0, remove_unused_columns=False,
        gradient_checkpointing=False,  # da bat tren model
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    dt = time.time() - t0
    n_tok = sum(len(r["input_ids"]) for r in rows) * (1 if dry_run else HP["epochs"])
    if dry_run:
        n_tok = sum(len(r["input_ids"]) for r in rows)
    info = {"tag": tag, "dry_run": dry_run, "gpu": torch.cuda.get_device_name(0),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "train_s": round(dt, 1), "tokens": n_tok, "tok_per_s": round(n_tok / dt),
            "metrics": res.metrics, "hp": HP, "data_sha": stats["sha256"],
            "loss_log": [(h.get("step"), round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]}
    if not dry_run:
        full_tok = stats["tokens_total"] * HP["epochs"]
        info["eta_note"] = f"{full_tok} token / {info['tok_per_s']} tok/s"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(f"{out_dir}/adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
        tok.save_pretrained(f"{out_dir}/merged")
        json.dump(info, open(f"{out_dir}/train_info.json", "w"), indent=1)
        ckpt.commit()
    else:
        full_tok = stats["tokens_total"] * HP["epochs"]
        info["eta_full_min"] = round(full_tok / info["tok_per_s"] / 60, 1)
    return info


@app.local_entrypoint()
def main(tag: str = "sft_v1", dry_run: bool = False, splits: str = "train", mask: float = 0.0, epochs: int = 0,
         synth: str = ""):
    info = train.remote(tag, dry_run, splits, mask, epochs, synth)
    out = ROOT / "runs" / (f"{tag}_dryrun" if dry_run else tag)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / "train_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
