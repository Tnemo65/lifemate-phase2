"""Chia train.jsonl (1545) thanh 3 tap: train / dev / final_test.

- train: giu nguyen tap 'train' cua splits.json cu (1127 mau).
- dev + final_test: 418 mau dev cu (dev_seen 125 + dev_unseen 293) cat doi,
  phan tang theo (nguon dev_seen/dev_unseen, loai nhan exec/ask/oos) de 2 nua tuong duong.
Ket qua: p2/artifacts/splits3.json
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
OLD_SPLITS = Path("E:/zalo/kaggle_stage_code/lifemate/artifacts/splits.json")
OUT = ROOT / "artifacts" / "splits3.json"


def label_kind(item):
    calls = item["label"]["tool_calls"]
    if not calls:
        return "oos"
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return "ask"
    return "exec"


def main():
    items = {}
    with open(DATA / "train" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            items[it["id"]] = it
    old = json.load(open(OLD_SPLITS, encoding="utf-8"))
    assert sum(len(v) for v in old.values()) == len(items)

    groups = defaultdict(list)
    for src in ("dev_seen", "dev_unseen"):
        for i in old[src]:
            groups[(src, label_kind(items[i]))].append(i)

    rng = random.Random(42)
    dev, final = [], []
    flip = 0
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        for i in ids:
            (dev if flip % 2 == 0 else final).append(i)
            flip += 1

    out = {"train": sorted(old["train"]), "dev": sorted(dev), "final_test": sorted(final),
           "unseen_ids": sorted(old["dev_unseen"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    unseen = set(old["dev_unseen"])
    for name in ("train", "dev", "final_test"):
        ids = out[name]
        kinds = Counter(label_kind(items[i]) for i in ids)
        print(f"{name:10s} n={len(ids):4d} unseen={sum(i in unseen for i in ids):3d} {dict(kinds)}")


if __name__ == "__main__":
    main()
