"""Phan bo loi theo "cho lech dau tien" giua du doan va gold, cho 1 run bat ky.
    python errors.py runs/sft_lc50_train_h50
In: (1) mat diem do hash sai vs F1 thieu; (2) bang loai loi theo ho du lieu.
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import scorer  # noqa: E402
from make_lc_splits import DATA, fam, kind  # noqa: E402

ASK = "sys.ask_missing_info"
items = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")}
cat = json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8"))
KIND = {t["name"]: t["kind"] for a in cat["apps"] for t in a["tools"]}
tools = agent.tool_index(cat)
gold = json.load(open(ROOT / "artifacts" / "gold_hashes.json"))


def vtype(name, k, v):
    if k in scorer.fuzzy_keys(name, tools):
        return "câu chữ tự do"
    if isinstance(v, bool):
        return "bật/tắt"
    if isinstance(v, (int, float)):
        return "số"
    if isinstance(v, list):
        return "danh sách"
    if isinstance(v, str):
        if re.match(r"\d{4}-\d\d-\d\d", v):
            return "thời gian"
        if re.match(r"^[a-z]+_\d+", v) or k.endswith("_id") or k == "id":
            return "ID (chọn sai đối tượng)"
        if k == "query":
            return "từ khoá tìm kiếm"
        return "chuỗi / enum"
    return "khác"


def first_div(it, p):
    g, s = it["label"]["tool_calls"], p["tool_calls"]
    if not g:
        return "ĐÁNG RỖNG mà vẫn gọi tool"
    if not s:
        return "TRẢ RỖNG khi phải làm"
    used, bad = set(), None
    for x in s:
        j = next((j for j, y in enumerate(g) if j not in used and scorer.call_match(x, y, tools)), None)
        if j is None:
            bad = x
            break
        used.add(j)
    miss = [g[j] for j in range(len(g)) if j not in used]
    if bad is None:
        if not miss:
            return "đủ call đúng nhưng state sai"
        m = miss[0]
        if m["name"] == ASK:
            return "DỪNG SỚM: thiếu bước hỏi lại"
        return "DỪNG SỚM: thiếu lệnh " + ("ghi" if KIND.get(m["name"]) == "write" else "đọc")
    if bad["name"] == ASK:
        return "HỎI LẠI sai nội dung" if any(y["name"] == ASK for y in miss) else "HỎI THỪA (đáng ra phải làm)"
    same = next((y for y in miss if y["name"] == bad["name"]), None)
    if same is None:
        ga = next((y for y in miss if y["name"] == ASK), None)
        if ga and bad["name"] == ga["arguments"].get("tool_name"):
            return "TỰ ĐOÁN thay vì hỏi lại"
        if not miss:
            return "GỌI THỪA sau khi đã đủ"
        if KIND.get(bad["name"]) == "read":
            return "ĐỌC THỪA / đọc sai tool"
        return "GHI sai tool"
    sa, ga = bad["arguments"], same["arguments"]
    if set(sa) - set(ga):
        return "THỪA tham số tuỳ chọn"
    if set(ga) - set(sa):
        return "THIẾU tham số"
    d = [k for k in ga if not scorer.call_match({"name": bad["name"], "arguments": {k: sa[k]}},
                                               {"name": bad["name"], "arguments": {k: ga[k]}}, tools)]
    return "SAI giá trị: " + vtype(bad["name"], d[0], ga[d[0]]) if d else "khác"


def main(run):
    P = {json.loads(l)["id"]: json.loads(l) for l in open(Path(run) / "predictions.jsonl", encoding="utf-8")}
    ids = sorted(P)
    per = {i: scorer.sample_score(P[i]["tool_calls"], items[i]["label"]["tool_calls"], P[i]["state_hash"] == gold[i], tools) for i in ids}
    n = len(ids)
    hw = [i for i in ids if items[i]["label"]["tool_calls"] and P[i]["state_hash"] != gold[i]]
    part = [i for i in ids if items[i]["label"]["tool_calls"] and P[i]["state_hash"] == gold[i] and per[i] < 1]
    emp = [i for i in ids if not items[i]["label"]["tool_calls"] and per[i] < 1]
    print(f"== {Path(run).name}: {n} mẫu, điểm {100 * sum(per.values()) / n:.1f}; đúng tuyệt đối {sum(v == 1 for v in per.values())} ({100 * sum(v == 1 for v in per.values()) / n:.0f}%)")
    print(f"   mất điểm: state sai (0 điểm) {len(hw)} mẫu = {100 * len(hw) / n:.1f} điểm | state đúng nhưng call lệch {len(part)} mẫu = "
          f"{100 * sum(1 - per[i] for i in part) / n:.1f} điểm | đáng rỗng mà gọi {len(emp)} mẫu = {100 * len(emp) / n:.1f} điểm")
    for f in ("llm", "tmpl"):
        gi = [i for i in ids if fam(items[i]) == f]
        lost, cnt = collections.Counter(), collections.Counter()
        for i in gi:
            if per[i] < 1:
                k = first_div(items[i], P[i])
                lost[k] += 1 - per[i]
                cnt[k] += 1
        tot = sum(lost.values())
        print(f"\n== họ {f}: {len(gi)} mẫu, điểm {100 * sum(per[i] for i in gi) / len(gi):.1f}, sai {sum(cnt.values())} mẫu")
        for k, v in lost.most_common(14):
            print(f"   {k:38}{cnt[k]:4} mẫu   mất {v:5.1f}  ({100 * v / tot:4.1f}% phần mất)")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
