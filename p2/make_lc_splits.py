"""Pha 0 (V3_DESIGN.md): them tap con cho duong cong hoc vao artifacts/splits3.json (khong sua cac tap cu).
  train_q50  : 50% TRAIN, phan tang theo (ho du lieu, loai mau), seed 42
  train_s200 : 200 mau TRAIN de cham (100 thuoc train_q50 + 100 ngoai train_q50), phan tang
"""
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test" / "train"


def fam(it):
    s = json.dumps(it["label"], ensure_ascii=False) + json.dumps(it.get("history") or [], ensure_ascii=False)
    ids = re.findall(r'"[a-z]+_(\d+)"', s)
    if not ids:
        return "noid"
    return "tmpl" if max(len(x) for x in ids) <= 2 else "llm"


def kind(it):
    c = it["label"]["tool_calls"]
    return "oos" if not c else ("ask" if any(x["name"] == "sys.ask_missing_info" for x in c) else "exec")


def strat_sample(ids, key, frac, rng):
    g = collections.defaultdict(list)
    for i in ids:
        g[key(i)].append(i)
    out = []
    for k in sorted(g):
        v = sorted(g[k])
        rng.shuffle(v)
        out += v[:round(frac * len(v))]
    return sorted(out)


def main():
    items = {}
    for line in open(DATA / "train.jsonl", encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it
    p = ROOT / "artifacts" / "splits3.json"
    sp = json.load(open(p, encoding="utf-8"))
    key = lambda i: (fam(items[i]), kind(items[i]))  # noqa: E731
    rng = random.Random(42)
    q50 = strat_sample(sp["train"], key, 0.5, rng)
    rest = sorted(set(sp["train"]) - set(q50))
    s200 = strat_sample(q50, key, 100 / len(q50), rng) + strat_sample(rest, key, 100 / len(rest), rng)
    sp["train_q50"], sp["train_s200"] = q50, sorted(s200)
    json.dump(sp, open(p, "w", encoding="utf-8"), indent=1)
    for n in ("train", "train_q50", "train_s200"):
        print(n, len(sp[n]), dict(collections.Counter(key(i) for i in sp[n]).most_common()))
    print("s200 trong q50:", len(set(s200) & set(q50)), "| ngoai q50:", len(set(s200) - set(q50)))


if __name__ == "__main__":
    main()
