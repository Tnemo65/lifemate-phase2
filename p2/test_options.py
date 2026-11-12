"""Kiem tra local (CPU, mien phi) cho cac tuy chon moi. Khong can GPU.

    python test_options.py            # tat ca
    python test_options.py compact    # 1 nhom: compact | ctx | data | traps

Moi nhom in so lieu va dung (assert) neu sai.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
sys.path.insert(0, str(DATA / "train"))
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import mask  # noqa: E402
import sft_data  # noqa: E402
from make_lc_splits import fam  # noqa: E402

ITEMS = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "train" / "train.jsonl", encoding="utf-8")}
SP = json.load(open(ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
TOOLS = agent.tool_index(json.load(open(DATA / "train" / "train_tools_catalog.json", encoding="utf-8")))


def _leaves(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield agent._flat(x)


def test_compact():
    """Dang dong khong lam mat khoa hay gia tri nao; do do dai so voi JSON."""
    n = jlen = clen = 0
    for i in SP["train"]:
        it = ITEMS[i]
        res = list(it["label"].get("tool_results") or [])
        for t in it.get("history") or []:
            res += t.get("tool_results") or []
        for r in res:
            c = agent.compact_result(r)
            for leaf in _leaves(r):
                assert leaf in c, (i, leaf, c[:200])
            n += 1
            jlen += len(json.dumps(r, ensure_ascii=False))
            clen += len(c)
    print(f"compact: {n} ket qua tool, khong mat gia tri nao; do dai {clen} / {jlen} ky tu = {100 * clen / jlen:.0f}% cua JSON")


def test_ctx():
    """p2 chen dung 1 lan; nhan [ĐỌC]/[GHI] co tren moi tool cua app va con nguyen sau khi doi ten; mac dinh khong doi."""
    it = ITEMS[SP["train"][0]]
    agent.set_ctx("")
    base_sys = agent.build_messages(it, agent.texts_gold(it))[0]["content"]
    base_tools = agent.tools_for(it, TOOLS)
    assert "[ĐỌC]" not in json.dumps(base_tools, ensure_ascii=False) and "Quy ước của đáp án" not in base_sys
    agent.set_ctx("p2")
    s = agent.build_messages(it, agent.texts_gold(it))[0]["content"]
    assert s.count("Quy ước của đáp án") == 1 and s.replace(agent.P2_RULES, "", 1) == base_sys
    agent.set_ctx("kind")
    tls = agent.tools_for(it, TOOLS)
    lab = [t["function"]["description"][:5] for t in tls if t["function"]["name"] != "sys.ask_missing_info"]
    assert set(lab) <= {"[ĐỌC]", "[GHI]"}, set(lab)
    msgs, _ = agent.sft_messages(it, agent.texts_gold(it))
    _, mt, _ = mask.mask_sample(msgs, tls, seed=1, generic=mask.generic_descriptions(TOOLS))
    assert [t["function"]["description"][:5] for t in mt if t["function"]["name"] != "sys.ask_missing_info"] == lab
    agent.set_ctx("")
    assert agent.tools_for(it, TOOLS) == base_tools
    print(f"ctx: p2 them {len(agent.P2_RULES)} ky tu vao system prompt; {len(lab)} tool co nhan "
          f"({collections.Counter(lab).most_common()}); nhan con nguyen sau khi doi ten; mac dinh khong doi")


def test_data(split="teacher_pilot"):
    """mask_epochs va boost: ban 0 giong het ban mac dinh; dem dong; ti le mau duoc hoc ca ten that."""
    kw = dict(check_replay=False, verbose=False, mask_ratio=0.3)
    r0, s0 = sft_data.build([split], ROOT / "tokenizer", DATA, **kw)
    e = 4
    r1, s1 = sft_data.build([split], ROOT / "tokenizer", DATA, mask_epochs=e, **kw)
    assert r1[:len(r0)] == r0 and len(r1) == e * len(r0), (len(r0), len(r1))
    masked = collections.Counter()
    for k in range(e):
        import random
        ids = SP[split]
        for i in random.Random(42 + k).sample(ids, round(0.3 * len(ids))):
            masked[i] += 1
    always = sum(1 for i in SP[split] if masked[i] == e)
    never = sum(1 for i in SP[split] if masked[i] == 0)
    r2, s2 = sft_data.build([split], ROOT / "tokenizer", DATA, boost="llm:2", **kw)
    n_llm = sum(1 for i in SP[split] if fam(ITEMS[i]) == "llm")
    assert len(r2) == len(r0) + n_llm and s2["sha256"] != s0["sha256"]
    print(f"data ({split}, {len(r0)} mau): mask_epochs={e} -> {len(r1)} dong, ban 0 giong ban mac dinh; "
          f"{always} mau bi doi ten ca {e} ban, {never} mau khong bao gio bi doi ten; "
          f"boost llm:2 -> {len(r2)} dong (+{n_llm} mau ho hoi thoai)")
    print("   ten file:", sft_data.data_name([split], 0.3, "", "p2+kind", e, "llm:2"))


def test_traps():
    """Ti le bay duoc gan tren moi base cua train_q50 so voi muc tieu; 'fix' va 'dict' chi gan khi lam duoc."""
    import synth_env as se
    c, n = collections.Counter(), 0
    for i in SP[se.POOL]:
        it = ITEMS[i]
        types = se.assign_types(i, bool(it.get("history")), fam(it))
        for t in se.assign_traps(i, types, it):
            c[t] += 1
            n += 1
    print(f"traps: {n} cau du kien tren {len(SP[se.POOL])} base")
    for t, (p, _) in se.TRAPS.items():
        print(f"   {t:5} gan {100 * c[t] / n:4.1f}%  (muc tieu toi da {100 * p:.1f}%)")
        assert c[t] / n <= p + 0.02
    print(f"   khong bay: {100 * c[''] / n:.1f}%")


if __name__ == "__main__":
    which = sys.argv[1:] or ["compact", "ctx", "data", "traps"]
    for w in which:
        globals()[f"test_{w}"]()
    print("OK")
