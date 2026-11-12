"""Kiem tra local (khong can GPU): 'model' gia tra ve gold tung buoc -> phai dat 100 diem.
Kiem tra ca: vong ReAct, parse tool_call, scorer, gold hash. Them 1 ca 'model rong' de thay diem giam."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
sys.path.insert(0, str(DATA / "train"))
sys.path.insert(0, str(ROOT))
from zaic_tools import ToolEnv  # noqa: E402
import agent  # noqa: E402
import scorer  # noqa: E402


def load():
    items = {}
    for line in open(DATA / "train" / "train.jsonl", encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it
    splits = json.load(open(ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    gh = json.load(open(ROOT / "artifacts" / "gold_hashes.json", encoding="utf-8"))
    tools = agent.tool_index(json.load(open(DATA / "train" / "train_tools_catalog.json", encoding="utf-8")))
    return items, splits, gh, tools


def main():
    items, splits, gh, tools = load()
    dev = [items[i] for i in splits["dev"]]
    owner = {}
    msgs0 = {}

    def new_env(i):
        return ToolEnv("train", i, root=str(DATA))

    def oracle(batch):
        outs = []
        for msgs, _ in batch:
            i = owner.setdefault(id(msgs), None)
            if i is None:  # first time: identify by system+last user content
                for it in dev:
                    if msgs0.get(it["id"]) is msgs:
                        i = it["id"]
                owner[id(msgs)] = i
            step = sum(1 for m in msgs if m["role"] == "tool") - n_hist_tool[i]
            gold = items[i]["label"]["tool_calls"]
            if step < len(gold):
                outs.append("<tool_call>\n" + json.dumps(gold[step], ensure_ascii=False) + "\n</tool_call>")
            else:
                outs.append("Xong.")
        return outs

    # gan msgs -> id truoc khi chay
    orig_build = agent.build_messages
    n_hist_tool = {}

    def build(it, texts):
        m = orig_build(it, texts)
        msgs0[it["id"]] = m
        n_hist_tool[it["id"]] = sum(1 for x in m if x["role"] == "tool")
        return m
    agent.build_messages = build

    preds = agent.run_batch(dev, agent.texts_gold, tools, new_env, oracle)
    r = scorer.score(preds, {i["id"]: i for i in dev}, gh, tools)
    print("ORACLE dev score:", r["score"], "n=", r["n"])

    empty = {i["id"]: {"tool_calls": [], "state_hash": ""} for i in dev}
    print("EMPTY  dev score:", scorer.score(empty, {i["id"]: i for i in dev}, gh, tools)["score"])

    # do dai prompt (ky tu) de uoc luong token
    lens = sorted(len(json.dumps(msgs0[i["id"]], ensure_ascii=False)) + len(json.dumps(agent.tools_for(i, tools), ensure_ascii=False)) for i in dev)
    print("prompt chars median/max:", lens[len(lens) // 2], lens[-1])
    print("vi du user text:", msgs0[dev[0]["id"]][-1]["content"])


if __name__ == "__main__":
    main()
