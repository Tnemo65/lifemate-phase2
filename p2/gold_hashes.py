"""Replay gold tool_calls cua 1545 mau train qua zaic_tools -> gold state_hash.
Kiem tra them: state cuoi khop label.final_state (get_state duoc phep tren train).
Ket qua: p2/artifacts/gold_hashes.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
sys.path.insert(0, str(DATA / "train"))
from zaic_tools import ToolEnv  # noqa: E402
from zaic_tools.utils import canon  # noqa: E402


def main():
    out, mismatch = {}, []
    with open(DATA / "train" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            env = ToolEnv("train", it["id"], root=str(DATA))
            for c in it["label"]["tool_calls"]:
                env.call(c["name"], c["arguments"])
            out[it["id"]] = env.get_state_hash()
            fs = it["label"]["final_state"]  # chi chua cac app bi thay doi
            st = env.get_state()
            if canon({k: st.get(k) for k in fs}) != canon(fs):
                mismatch.append(it["id"])
    p = ROOT / "artifacts" / "gold_hashes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=0)
    print(f"{len(out)} hashes, final_state mismatch: {len(mismatch)} {mismatch[:10]}")


if __name__ == "__main__":
    main()
