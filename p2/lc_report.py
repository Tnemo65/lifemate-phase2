"""Bao cao theo ho du lieu / loai mau cho 1 run bat ky (DEV hoac tap con TRAIN).
    python lc_report.py runs/sft_v2_train_s200 [runs/...]
Neu split la train_s200: tach them 'trong train_q50' / 'ngoai train_q50'.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import scorer  # noqa: E402
from make_lc_splits import DATA, fam, kind  # noqa: E402

items = {}
for line in open(DATA / "train.jsonl", encoding="utf-8"):
    it = json.loads(line)
    items[it["id"]] = it
tools = agent.tool_index(json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8")))
gold = json.load(open(ROOT / "artifacts" / "gold_hashes.json"))
sp = json.load(open(ROOT / "artifacts" / "splits3.json"))
q50 = set(sp.get("train_q50", []))

for run in sys.argv[1:]:
    P = {}
    for line in open(Path(run) / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        P[p["id"]] = p
    per = scorer.score(P, {i: items[i] for i in P}, gold, tools)["per_sample"]
    g = collections.defaultdict(list)
    for i, s in per.items():
        it = items[i]
        g["ALL"].append(s)
        g["ho " + fam(it)].append(s)
        g["loai " + kind(it)].append(s)
        g[f"{fam(it)}/{kind(it)}"].append(s)
        if i in set(sp["train"]):
            part = "TRONG q50" if i in q50 else "NGOAI q50"
            g[part].append(s)
            g[f"{part} / ho {fam(it)}"].append(s)
    print("==", Path(run).name)
    for k in sorted(g):
        print(f"  {k:26}{len(g[k]):5}{100 * sum(g[k]) / len(g[k]):8.1f}")
    t, l = g.get("ho tmpl"), g.get("ho llm")
    if t and l:
        print(f"  quy doi theo public (0.56 tmpl + 0.44 llm): {100 * (0.56 * sum(t) / len(t) + 0.44 * sum(l) / len(l)):.1f}")
