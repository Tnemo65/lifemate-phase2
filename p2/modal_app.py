"""Chay baseline tren Modal.

    python -m modal run modal_app.py --split dev               # chay ca tap DEV
    python -m modal run modal_app.py --split dev --limit 20    # chay thu 20 mau
    python -m modal run modal_app.py --split final_test        # chi chay 1 lan cuoi!

Model luu tren Volume 'lifemate-models' (tai 1 lan). Du lieu (khong audio) + code gan vao container.
Ket qua ghi ve may: p2/runs/<tag>/ (predictions.jsonl, traces.jsonl, score.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
LOCAL_DATA = ROOT.parent / "train+public_test" / "train+public_test"

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = "/models/Qwen3-4B-Instruct-2507"
MAX_LEN = 16384
MAX_NEW = 512
GPU = ["L4", "A10G", "L40S"]  # thu lan luot neu het cho

models = modal.Volume.from_name("lifemate-models", create_if_missing=True)
ckpt = modal.Volume.from_name("lifemate-ckpt", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
    .add_local_dir(LOCAL_DATA, "/data", ignore=["**/audio/**", "**/.DS_Store", "**/__pycache__/**"])
    .add_local_dir(ROOT, "/root/p2", ignore=["runs/**", "teacher_runs/**", "**/__pycache__/**", "tokenizer/**", "artifacts/sft_*.jsonl", "*.log"])
)

app = modal.App("lifemate-p2", image=image)


@app.function(volumes={"/models": models}, timeout=1800)
def download_model():
    import os
    from huggingface_hub import snapshot_download
    if os.path.exists(f"{MODEL_DIR}/config.json"):
        return "da co san"
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR, token=os.environ.get("HF_TOKEN"))
    models.commit()
    return "tai xong"


@app.function(gpu=GPU, volumes={"/models": models, "/ckpt": ckpt}, timeout=3600)
def run_eval(split: str, limit: int = 0, model: str = "base", prefix_cache: bool = False, max_seqs: int = 8):
    """model = 'base' (Qwen goc) hoac ten run SFT (vd 'sft_v1' -> /ckpt/sft_v1/merged)."""
    import random
    import sys
    model_dir = MODEL_DIR if model == "base" else f"/ckpt/{model}/merged"

    import numpy as np
    sys.path.insert(0, "/data/public_test" if split == "public_test" else "/data/train")  # zaic_tools dung split
    sys.path.insert(0, "/root/p2")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from zaic_tools import ToolEnv

    import agent
    import scorer

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    public = split == "public_test"  # khong co nhan: dung transcript ASR, khong cham
    src = "public_test" if public else "train"
    items = {}
    for line in open(f"/data/{src}/{src}.jsonl", encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it
    ids = sorted(items) if public else json.load(open("/root/p2/artifacts/splits3.json", encoding="utf-8"))[split]
    if limit:
        ids = ids[:limit]
    batch_items = [items[i] for i in ids]
    gh = None if public else json.load(open("/root/p2/artifacts/gold_hashes.json", encoding="utf-8"))
    tools = agent.tool_index(json.load(open(f"/data/{src}/{src}_tools_catalog.json", encoding="utf-8")))
    if public:
        asr = {}
        for line in open("/root/p2/artifacts/transcripts_public_test.jsonl", encoding="utf-8"):
            r = json.loads(line)
            asr[r["path"]] = r["text"]
        texts_fn = lambda it: asr  # noqa: E731  map duong dan audio -> transcript ASR
    else:
        texts_fn = agent.texts_gold

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)  # tokenizer goc: SFT khong doi tokenizer/template
    # Mac dinh: tat prefix cache + toi da 8 chuoi cung luc -> KV cache khong bi ep (lan cham dau tren GPU 24GB
    # voi 209 prompt dong thoi cho output hong).
    llm = LLM(model_dir, tokenizer=MODEL_DIR, max_model_len=MAX_LEN, gpu_memory_utilization=0.90,
              seed=0, dtype="bfloat16", enable_prefix_caching=prefix_cache, max_num_seqs=max_seqs)
    stats_gpu = torch.cuda.get_device_name(0)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, seed=0)
    load_s = time.time() - t0
    stats = {"too_long": 0, "max_prompt_tokens": 0, "gen_calls": 0}

    def generate(batch):
        prompts, idx = [], []
        outs = [None] * len(batch)
        for k, (msgs, tls) in enumerate(batch):
            p = tok.apply_chat_template(msgs, tools=tls, add_generation_prompt=True, tokenize=False)
            n = len(tok(p).input_ids)
            stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], n)
            if n > MAX_LEN - MAX_NEW:
                stats["too_long"] += 1
                continue
            prompts.append(p)
            idx.append(k)
        if prompts:
            stats["gen_calls"] += 1
            res = llm.generate(prompts, sp, use_tqdm=False)
            for k, r in zip(idx, res):
                outs[k] = r.outputs[0].text
        return outs

    t1 = time.time()
    preds = agent.run_batch(batch_items, texts_fn,
                            tools, lambda i: ToolEnv(src, i, root="/data"), generate)
    run_s = time.time() - t1
    if public:
        r = {"n": len(preds), "score": None, "per_sample": {}}
    else:
        r = scorer.score(preds, {i["id"]: i for i in batch_items}, gh, tools)
    return {"model": model, "split": split, "gpu": stats_gpu, "prefix_cache": prefix_cache, "max_seqs": max_seqs,
            "n": r["n"], "score": r["score"], "per_sample": r["per_sample"],
            "preds": {i: {"tool_calls": p["tool_calls"], "state_hash": p["state_hash"]} for i, p in preds.items()},
            "traces": {i: p["trace"] for i, p in preds.items()},
            "load_s": round(load_s, 1), "run_s": round(run_s, 1),
            "avg_s_per_sample_batched": round(run_s / max(len(ids), 1), 2), **stats}


@app.local_entrypoint()
def main(split: str = "dev", limit: int = 0, tag: str = "", model: str = "base",
         gpu: str = "", prefix_cache: bool = False, max_seqs: int = 8):
    if split == "final_test":
        print("CANH BAO: final_test chi chay 1 lan cuoi cung!")
    tag = tag or ("baseline_zeroshot" if model == "base" else model)
    print(download_model.remote())
    fn = run_eval.with_options(gpu=gpu) if gpu else run_eval
    res = fn.remote(split, limit, model, prefix_cache, max_seqs)
    out = ROOT / "runs" / f"{tag}_{split}{'_n' + str(limit) if limit else ''}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, p in res["preds"].items():
            f.write(json.dumps({"id": i, **p}, ensure_ascii=False) + "\n")
    with open(out / "traces.jsonl", "w", encoding="utf-8") as f:
        for i, t in res["traces"].items():
            f.write(json.dumps({"id": i, "score": res["per_sample"].get(i), "trace": t}, ensure_ascii=False) + "\n")
    if split == "public_test":  # file nop portal, dung thu tu cua public_test.jsonl
        order = [json.loads(line)["id"] for line in open(LOCAL_DATA / "public_test" / "public_test.jsonl", encoding="utf-8")]
        with open(out / "submission.jsonl", "w", encoding="utf-8") as f:
            for i in order:
                p = res["preds"][i]
                f.write(json.dumps({"id": i, "tool_calls": p["tool_calls"], "state_hash": p["state_hash"]}, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in res.items() if k not in ("preds", "traces", "per_sample")}
    json.dump(summary, open(out / "score.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    print("ket qua o:", out)
