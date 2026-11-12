"""So 2+ run tren cung tap mau, theo ho du lieu, kem thang/thua tung mau va sai so chuan cua hieu.
    python cmp_runs.py [--only dev_clean] runs/A runs/B ...
"""
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import scorer  # noqa: E402
from make_lc_splits import DATA, fam, kind  # noqa: E402

args = sys.argv[1:]
only = None
if args[0] == "--only":
    only, args = args[1], args[2:]
items = {json.loads(line)["id"]: json.loads(line) for line in open(DATA / "train.jsonl", encoding="utf-8")}
tools = agent.tool_index(json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8")))
gold = json.load(open(ROOT / "artifacts" / "gold_hashes.json"))
sp = json.load(open(ROOT / "artifacts" / "splits3.json"))
per = {}
for run in args:
    P = {json.loads(line)["id"]: json.loads(line) for line in open(Path(run) / "predictions.jsonl", encoding="utf-8")}
    per[run] = {i: scorer.sample_score(p["tool_calls"], items[i]["label"]["tool_calls"], p["state_hash"] == gold[i], tools) for i, p in P.items()}
ids = sorted(set.intersection(*(set(v) for v in per.values())))
if only:
    ids = [i for i in ids if i in set(sp[only])]
groups = collections.OrderedDict(ALL=ids)
for f in ("llm", "tmpl", "noid"):
    groups["ho " + f] = [i for i in ids if fam(items[i]) == f]
for k in ("exec", "ask", "oos"):
    groups["loai " + k] = [i for i in ids if kind(items[i]) == k]
print(f"{'nhom':12}{'n':>5}" + "".join(f"{Path(r).name[:22]:>24}" for r in args))
for g, gi in groups.items():
    if gi:
        print(f"{g:12}{len(gi):>5}" + "".join(f"{100 * sum(per[r][i] for i in gi) / len(gi):>24.1f}" for r in args))
t, l = groups["ho tmpl"], groups["ho llm"]
if t and l:
    print(f"{'quy doi*':12}{'':>5}" + "".join(f"{100 * (0.56 * sum(per[r][i] for i in t) / len(t) + 0.44 * sum(per[r][i] for i in l) / len(l)):>24.1f}" for r in args))
    print("  * 0.56 x template + 0.44 x hoi thoai (ti le cua public test)")
if len(args) == 2:
    a, b = per[args[0]], per[args[1]]
    for g in ("ALL", "ho llm", "ho tmpl"):
        d = [b[i] - a[i] for i in groups[g]]
        n = len(d)
        m = sum(d) / n
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
        print(f"  {g}: hieu = {100 * m:+.1f} ± {100 * 1.96 * se:.1f} (95%) | run2 hon {sum(x > 1e-9 for x in d)} mau, kem {sum(x < -1e-9 for x in d)} mau")
