"""B0: dung du lieu SFT + kiem tra (chay local, CPU, mien phi).

Kiem tra:
  A. len(label.tool_results) == len(label.tool_calls)
  B. replay gold calls qua zaic_tools -> ket qua == label.tool_results  (=> hoi thoai SFT == hoi thoai luc inference)
  C. moi target nam dung sau prompt luc inference (prefix voi add_generation_prompt) va la tien to cua chuoi day du
  D. token cua prompt inference == phan dau token cua chuoi train (khong lech tokenizer o bien)
  E. bien target trung bien token (khong co token cat ngang)
Ket qua: artifacts/sft_<split>.jsonl  ({id, input_ids, labels}) + artifacts/sft_<split>.stats.json

    python sft_data.py --split train
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
sys.path.insert(0, str(DATA / "train"))
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402

MAX_LEN = 10240


def data_name(split_names, mask_ratio=0.0, synth=""):
    return "_".join(split_names) + (f"+{synth}" if synth else "") + (f"_m{int(round(mask_ratio * 100))}" if mask_ratio else "")


def build(split_names, tok_dir, data_dir, check_replay=True, verbose=True, mask_ratio=0.0, synth=""):
    import random as _random

    import mask as _mask
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    items = {}
    for line in open(Path(data_dir) / "train" / "train.jsonl", encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it
    splits = json.load(open(ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    ids = [i for s in split_names for i in splits[s]]
    if synth:  # mau sinh (synth_env.py collect): cung dinh dang mau train, moi truong = mau goc base_id
        for line in open(ROOT / "artifacts" / f"synth_{synth}.jsonl", encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
            ids.append(it["id"])
    tools = agent.tool_index(json.load(open(Path(data_dir) / "train" / "train_tools_catalog.json", encoding="utf-8")))
    if check_replay:
        from zaic_tools import ToolEnv

    generic = _mask.generic_descriptions(tools)
    masked_ids = set(_random.Random(42).sample(ids, round(mask_ratio * len(ids)))) if mask_ratio else set()
    rows, lens, n_tgt, dropped = [], [], [], []
    for i in ids:
        it = items[i]
        lab = it["label"]
        assert len(lab.get("tool_results") or []) == len(lab["tool_calls"]), f"A {i}"
        msgs, n_prefix = agent.sft_messages(it, agent.texts_gold(it))
        tls = agent.tools_for(it, tools)

        if check_replay:  # B
            env = ToolEnv("train", it.get("base_id", i), root=str(data_dir))
            inf = agent.build_messages(it, agent.texts_gold(it))
            for c in lab["tool_calls"]:
                if agent.append_step(inf, c, env.call(c["name"], c["arguments"])):
                    break
            assert inf == msgs[:len(inf)], f"B replay khac label.tool_results: {i}"

        if i in masked_ids:  # Hammer masking (sau kiem tra B, tren ban goc)
            msgs, tls, _ = _mask.mask_sample(msgs, tls, seed=int("".join(i.split("_")[1:])) * 7919 + 42, generic=generic)
            names = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tls}
            for m in msgs[n_prefix:]:
                for tc in m.get("tool_calls") or []:
                    f = tc["function"]
                    assert f["name"] in names, f"M ten tool sau mask khong co trong schema: {i} {f['name']}"
                    assert set(f["arguments"]) <= set(names[f["name"]]), f"M tham so sau mask lech schema: {i}"
                    if f["name"] == agent.ASK_TOOL["function"]["name"]:
                        tn = f["arguments"].get("tool_name")
                        assert tn in names and set(f["arguments"].get("params", [])) <= set(names[tn]), f"M ask: {i}"

        full = tok.apply_chat_template(msgs, tools=tls, tokenize=False)
        spans = []
        for j in range(n_prefix, len(msgs)):
            if msgs[j]["role"] != "assistant":
                continue
            pre = tok.apply_chat_template(msgs[:j], tools=tls, tokenize=False, add_generation_prompt=True)
            end = tok.apply_chat_template(msgs[:j + 1], tools=tls, tokenize=False)
            assert full.startswith(pre) and full.startswith(end) and len(end) > len(pre), f"C {i} j={j}"
            spans.append((len(pre), len(end), pre))

        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        input_ids, offs = enc["input_ids"], enc["offset_mapping"]
        labels = [-100] * len(input_ids)
        for a, b, pre in spans:
            pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
            assert input_ids[:len(pre_ids)] == pre_ids, f"D {i}"  # prompt inference == dau chuoi train
            for k, (s, e) in enumerate(offs):
                if s >= a and e <= b and e > s:
                    labels[k] = input_ids[k]
                elif s < a < e or s < b < e:
                    raise AssertionError(f"E token cat ngang bien target: {i}")
        if len(input_ids) > MAX_LEN:
            dropped.append(i)
            continue
        rows.append({"id": i, "input_ids": input_ids, "labels": labels})
        lens.append(len(input_ids))
        n_tgt.append(sum(x != -100 for x in labels))

    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    stats = {"splits": split_names, "synth": synth, "mask_ratio": mask_ratio, "n_masked": len(masked_ids),
             "n": len(rows), "dropped_too_long": dropped,
             "tokens_total": sum(lens), "len_p50": q(lens, .5), "len_p95": q(lens, .95), "len_max": max(lens),
             "target_tokens_total": sum(n_tgt), "target_p50": q(n_tgt, .5), "target_max": max(n_tgt),
             "mean_len": round(statistics.mean(lens), 1)}
    blob = json.dumps([[r["id"], r["input_ids"], r["labels"]] for r in rows]).encode()
    stats["sha256"] = hashlib.sha256(blob).hexdigest()
    if verbose:
        show = [r for r in rows if r["id"] not in masked_ids][:2] + [r for r in rows if r["id"] in masked_ids][:2]
        for r in show:
            print("[MASKED]" if r["id"] in masked_ids else "[GOC]", end=" ")
            tgt = [t for t, l in zip(r["input_ids"], r["labels"]) if l != -100]
            print(f"--- {r['id']} | user: {items[r['id']]['audio_text']}\nTARGET:\n{tok.decode(tgt)}\n")
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="vd: train  hoac  train,dev")
    ap.add_argument("--mask", type=float, default=0.0, help="ti le mau Hammer masking, vd 0.3")
    ap.add_argument("--synth", default="", help="ten file mau sinh: artifacts/synth_<ten>.jsonl")
    a = ap.parse_args()
    names = a.split.split(",")
    rows, stats = build(names, ROOT / "tokenizer", DATA, mask_ratio=a.mask, synth=a.synth)
    tag = data_name(names, a.mask, a.synth)
    with open(ROOT / "artifacts" / f"sft_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(ROOT / "artifacts" / f"sft_{tag}.stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
