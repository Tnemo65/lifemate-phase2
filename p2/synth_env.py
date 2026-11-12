"""Pha 2 (V3_DESIGN.md): sinh du lieu bang "thay" tren moi truong TRAIN, co kiem cheo va loc bang luat.

Vai NGUOI VIET (biet state, viet yeu cau moi + chay ke hoach du dinh, run 'w'):
    python synth_env.py card  <base_id>                      # de bai cho nguoi viet: boi canh, state, 3 loai cau can viet
    python synth_env.py write <base_id> <k> '<cau noi moi>'  # luu cau thu k (1..3)
    python synth_env.py call  <base_id>-<k> w <tool> '<json>' ; python synth_env.py done <base_id>-<k> w
Vai NGUOI GIAI (KHONG biet state/ke hoach, chi thay cau noi, run 's'):
    python synth_env.py show <base_id>-<k>          # day du (lan dau cua moi base)
    python synth_env.py show <base_id>-<k> brief    # cac cau sau cung base: chi cau moi + vi du
    python synth_env.py call <base_id>-<k> s <tool> '<json>' ; python synth_env.py done <base_id>-<k> s
Nguoi van hanh:
    python synth_env.py plan <n_base> <seed>        # chon base tu train_q50 (chi nua TRAIN nay duoc dung de sinh)
    python synth_env.py collect <ten>               # kiem cheo w == s + loc luat -> artifacts/synth_<ten>.jsonl

Chong overfit:
  - base va vi du truy hoi chi lay tu train_q50; nua TRAIN con lai + DEV + FINAL_TEST khong bao gio duoc dung.
  - cam 7 app dang giu lam "app la" tren DEV.
  - ti le loai cau (lam / hoi lai / rong) lay theo gold TRAIN, do code gan, khong do LLM tu chon.
"""
from __future__ import annotations

import collections
import json
import random
import sys

import teacher_env as te
from make_lc_splits import fam, kind

SYN = te.RUNS / "syn"
HELD_OUT = {"bike_rental", "fitness", "learning", "loyalty", "package_tracking", "podcast_radio", "video_meeting"}
POOL = "train_q50"
ASK = "sys.ask_missing_info"

TYPES = {
    "E1": "LÀM — một việc đơn: 1 thao tác ghi (kèm các bước đọc cần thiết theo quy ước).",
    "E2": "LÀM — nhiều việc trong một câu: 2–3 ý, nên thuộc 2 app khác nhau.",
    "E3": "LÀM — có điều kiện: \"nếu X thì Y (còn không thì Z / thì thôi)\"; X phải kiểm tra được bằng tool đọc trên dữ liệu thật.",
    "E4": "LÀM — chọn theo tiêu chí hoặc nhiều đối tượng: \"cái rẻ nhất\", \"cả hai\", \"mấy cái ở phòng khách\", \"mỗi người một...\"; đáp án phải xác định duy nhất từ dữ liệu thật.",
    "E5": "LÀM — tham chiếu history: dùng \"cái đó\", \"người hồi nãy\", \"cuộc hẹn vừa nói\"... trỏ tới thứ đã xuất hiện ở các lượt trước.",
    "E6": "LÀM — chỉ hỏi thông tin: người dùng hỏi một thông tin phải tra bằng tool đọc (không ghi gì).",
    "A1": "HỎI LẠI — thiếu tham số bắt buộc mà không suy ra được (vd nhắc việc nhưng không nói giờ, chuyển tiền không nói số tiền). Đáp án kết thúc bằng sys.ask_missing_info.",
    "A2": "HỎI LẠI — nhập nhằng thực thể: tool đọc trả về ≥2 kết quả khớp (vd 2 người cùng tên) và câu nói không đủ để chọn. Phải kiểm tra trên dữ liệu thật là đúng có ≥2. Đáp án: đọc rồi sys.ask_missing_info.",
    "O1": "RỖNG — người dùng huỷ/đổi ý ngay trong câu hoặc huỷ việc vừa định làm (\"à thôi khỏi\", \"thôi để sau\"). Không gọi tool nào.",
    "O2": "RỖNG — tán gẫu, cảm ơn, kể chuyện, không yêu cầu gì làm được bằng tool. Không gọi tool nào.",
    "O3": "RỖNG — yêu cầu cần một app/chức năng KHÔNG có trong danh sách tool đã cài. Không gọi tool nào.",
    "O4": "RỖNG — hỏi lại thông tin đã có sẵn trong history (trợ lý vừa tra ở lượt trước). Không gọi tool nào.",
}
KIND_W = {"E": 0.768, "A": 0.146, "O": 0.086}  # ti le gold TRAIN: lam / hoi lai / rong
# so thao tac GHI trong mau LAM cua gold train_q50, theo ho du lieu (dot 1 sinh qua de: template chi 14% co >=2 ghi, gold 46%)
N_WRITE = {"llm": {0: 0.16, 1: 0.57, 2: 0.22, 3: 0.05}, "tmpl": {0: 0.03, 1: 0.51, 2: 0.42, 3: 0.04}}
REF_MIN = 80  # cau mau qua ngan (vd "Ừ đặt đi") thi lay moc 80 ky tu (gold LAM: TB 99-110)
MIN_LEN = {"E": 0.8, "A": 0.5, "O": 0.5}  # do dai cau moi / cau mau cung base (dot 1: trung vi 0.68 -> ngan hon gold)


def assign_types(base_id, has_history, family="llm", n=3):
    """Ma loai: 'A1','O3'... hoac '<E?>w<so ghi>' vd 'E3w2'. Do code gan theo phan bo gold, khong de LLM tu chon."""
    rng = random.Random(f"{base_id}|types2")
    dist = N_WRITE["llm" if family == "llm" else "tmpl"]
    out = []
    for _ in range(n):
        k = rng.choices(list(KIND_W), weights=list(KIND_W.values()))[0]
        if k != "E":
            cand = [t for t in TYPES if t[0] == k and (has_history or t != "O4")]
            cand = [t for t in cand if t not in out] or cand
            out.append(rng.choice(cand))
            continue
        w = rng.choices(list(dist), weights=list(dist.values()))[0]
        if w == 0:
            fl = rng.choice(["E6", "E3"])
        else:
            fl = rng.choice([("E1" if w == 1 else "E2"), "E3", "E4"] + (["E5"] if has_history else []))
        out.append(f"{fl}w{w}")
    return out


def describe(code):
    if "w" not in code:
        return TYPES[code]
    fl, w = code.split("w")
    w = int(w)
    extra = (" Điều kiện phải SAI trên dữ liệu thật, nên đáp án chỉ gồm lệnh đọc rồi dừng." if fl == "E3" and w == 0 else
             f" Đáp án phải có ĐÚNG {w} thao tác GHI (không tính lệnh đọc)." if w else " Đáp án không có thao tác ghi.")
    return TYPES[fl] + extra


def plan(n_base, seed):
    """Chon base tu train_q50: 70% ho hoi thoai (noi model khai quat kem nhat tren nua TRAIN chua hoc: 33 vs 73)."""
    items = te._load_items("train")
    sp = json.load(open(te.ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    rng = random.Random(seed)
    used = {p.name.split("-")[0] for p in SYN.glob("*.req.json")}  # base da dung o dot truoc
    llm = sorted(i for i in sp[POOL] if fam(items[i]) == "llm" and i not in used)
    oth = sorted(i for i in sp[POOL] if fam(items[i]) != "llm" and i not in used)
    rng.shuffle(llm)
    rng.shuffle(oth)
    k = round(0.7 * n_base)
    pick = sorted(llm[:k] + oth[:n_base - k])
    print(" ".join(pick))


def card(base_id):
    items = te._load_items("train")
    it = items[base_id]
    sp = json.load(open(te.ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    assert base_id in set(sp[POOL]), "base phai thuoc train_q50"
    te.show("train", base_id, pool_name=POOL, part="main")
    from zaic_tools import ToolEnv
    st = ToolEnv("train", base_id, root=str(te.DATA)).get_state()
    print("\n=== DỮ LIỆU THẬT TRONG MÁY NGƯỜI DÙNG (chỉ NGƯỜI VIẾT được xem; đã gồm thay đổi từ các lượt trước) ===")
    for app in it["context"]["installed_apps"]:
        if app in st:
            s = json.dumps(st[app], ensure_ascii=False)
            print(f"- {app}: {s[:900]}{'…(còn nữa, dùng tool đọc để xem)' if len(s) > 900 else ''}")
    held = sorted(HELD_OUT & set(it["context"]["installed_apps"]))
    types = assign_types(base_id, bool(it.get("history")), fam(it))
    print("\n=== NHIỆM VỤ CỦA NGƯỜI VIẾT ===")
    print("Lượt user CUỐI ở trên chỉ là mẫu GIỌNG VĂN. Hãy viết 3 câu nói MỚI của chính người dùng này, nối tiếp cùng history, mỗi câu một loại:")
    for k, t in enumerate(types, 1):
        print(f"  Câu {k} — loại {t}: {describe(t)}")
    print("Yêu cầu: giọng văn nói tự nhiên như các lượt user ở trên (xưng hô, phương ngữ, số đọc bằng chữ nếu họ nói vậy); "
          "ý định KHÁC câu mẫu và khác nhau; chỉ dùng dữ liệu có thật ở trên; đáp án phải xác định duy nhất theo quy ước.")
    n = max(len(it["audio_text"]), REF_MIN)
    print(f"Độ dài: lấy mốc {n} ký tự. Câu LÀM phải dài ít nhất {round(MIN_LEN['E'] * n)} ký tự, câu HỎI LẠI / RỖNG ít nhất "
          f"{round(MIN_LEN['A'] * n)} ký tự — thêm bối cảnh, lý do đời thường như cách người này vẫn nói (không thêm ý định mới ngoài loại được giao).")
    if held:
        print("KHÔNG dùng các app: " + ", ".join(held) + ".")


def write(base_id, k, text):
    it = te._load_items("train")[base_id]
    types = assign_types(base_id, bool(it.get("history")), fam(it))
    need = round(MIN_LEN[types[int(k) - 1][0]] * max(len(it["audio_text"]), REF_MIN))
    if len(text.strip()) < need:
        print(f"CHƯA LƯU: câu dài {len(text.strip())} ký tự, cần ít nhất {need}. Viết lại cho giống độ dài/giọng câu mẫu.")
        return
    SYN.mkdir(parents=True, exist_ok=True)
    qid = f"{base_id}-{k}"
    json.dump({"base": base_id, "k": int(k), "type": types[int(k) - 1], "text": text.strip()},
              open(SYN / f"{qid}.req.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"đã lưu {qid} ({types[int(k) - 1]})")


# ---------------------------------------------------------------- kiem cheo + loc
def _catalog():
    cat = json.load(open(te.DATA / "train" / "train_tools_catalog.json", encoding="utf-8"))
    return {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def _producers(T, ref):
    app, coll = ref.split(".", 1)
    return {n for n, t in T.items() if t["app"] == app and t["kind"] == "read" and coll in (t.get("returns") or "")}


def _always_empty_reads(T, items, ids):
    c = collections.defaultdict(collections.Counter)
    for i in ids:
        for x in items[i]["label"]["tool_calls"]:
            if T.get(x["name"], {}).get("kind") == "read":
                c[x["name"]]["empty" if not x["arguments"] else "args"] += 1
    return {n for n, v in c.items() if v["args"] == 0 and v["empty"] >= 5}


def lint(item, calls, results, T, always_empty):
    """Tra ve ly do loai (chuoi) hoac None."""
    installed = set(item["context"]["installed_apps"])
    hist = json.dumps(item.get("history") or [], ensure_ascii=False)
    seen = set()
    for k, (c, r) in enumerate(zip(calls, results)):
        n, a = c["name"], c.get("arguments") or {}
        if isinstance(r, dict) and r.get("ok") is False:
            return f"call bị từ chối: {n}"
        if n == ASK:
            if k != len(calls) - 1:
                return "ask không đứng cuối"
            t = T.get(a.get("tool_name"))
            if not t or t["app"] not in installed or t["app"] in HELD_OUT:
                return "ask: tool_name không hợp lệ"
            if not a.get("params") or not set(a["params"]) <= set(t["parameters"]["properties"]):
                return "ask: params không có trong schema"
            continue
        t = T.get(n)
        if not t or t["app"] not in installed:
            return f"tool không được cài: {n}"
        if t["app"] in HELD_OUT:
            return f"dùng app giữ lại: {t['app']}"
        if n in always_empty and a:
            return f"tool đọc liệt kê có tham số: {n}"
        if t["kind"] == "write":
            for p, v in a.items():
                ref = (t["parameters"]["properties"].get(p) or {}).get("x-ref")
                if ref and isinstance(v, str) and f'"{v}"' not in hist and not (_producers(T, ref) & seen):
                    return f"ID mới không có lệnh đọc trước: {n}.{p}"
        seen.add(n)
    return None


def collect(name):
    import scorer
    from zaic_tools import ToolEnv
    T = _catalog()
    tools = te._tools("train")
    items = te._load_items("train")
    sp = json.load(open(te.ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    always_empty = _always_empty_reads(T, items, sp[POOL])
    rows, why = [], collections.Counter()
    for rp in sorted(SYN.glob("*.req.json")):
        req = json.load(open(rp, encoding="utf-8"))
        qid = f"{req['base']}-{req['k']}"
        logs = {}
        for run in ("w", "s"):
            lp = SYN / f"{qid}.{run}.json"
            if lp.exists():
                lg = json.load(open(lp, encoding="utf-8"))
                if lg.get("done"):
                    logs[run] = lg
        if len(logs) < 2:
            why["chưa đủ 2 lượt (w, s)"] += 1
            continue
        w, s = logs["w"], logs["s"]
        if w["state_hash"] != s["state_hash"]:
            why["lệch trạng thái cuối (w ≠ s)"] += 1
            continue
        if not w["tool_calls"] and not s["tool_calls"]:
            ok = True
        else:
            ok = w["tool_calls"] and s["tool_calls"] and \
                scorer.sample_score(s["tool_calls"], w["tool_calls"], True, tools) == 1.0 and \
                scorer.sample_score(w["tool_calls"], s["tool_calls"], True, tools) == 1.0
        if not ok:
            why["lệch danh sách call (w ≠ s)"] += 1
            continue
        base = items[req["base"]]
        env = ToolEnv("train", req["base"], root=str(te.DATA))
        results = [env.call(c["name"], c["arguments"]) for c in s["tool_calls"]]
        bad = lint(base, s["tool_calls"], results, T, always_empty)
        if bad:
            why["luật: " + bad.split(":")[0]] += 1
            continue
        got = "O" if not s["tool_calls"] else ("A" if s["tool_calls"][-1]["name"] == ASK else "E")
        if got != req["type"][0]:
            why[f"sai loại ({req['type'][0]}→{got})"] += 1
            continue
        if "w" in req["type"]:
            nw = sum(1 for c in s["tool_calls"] if T.get(c["name"], {}).get("kind") == "write")
            if nw != int(req["type"].split("w")[1]):
                why["sai số thao tác ghi so với đề"] += 1
                continue
        it = {k: v for k, v in base.items() if k != "label"}
        it.update(id=f"syn_{req['base'].split('_')[1]}_{req['k']}", base_id=req["base"], audio_text=req["text"],
                  syn_type=req["type"], label={"tool_calls": s["tool_calls"], "tool_results": results})
        rows.append(it)
    out = te.ROOT / "artifacts" / f"synth_{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_req = len(list(SYN.glob("*.req.json")))
    print(f"câu đã viết: {n_req} | giữ: {len(rows)} ({100 * len(rows) / max(n_req, 1):.0f}%) -> {out.name}")
    print("lý do loại:", dict(why.most_common()))
    print("loại câu giữ lại:", dict(collections.Counter(r["syn_type"] for r in rows).most_common()))
    print("họ của base:", dict(collections.Counter(fam(items[r["base_id"]]) for r in rows)))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        plan(int(a[0]), int(a[1]))
    elif cmd == "card":
        card(a[0])
    elif cmd == "write":
        write(a[0], a[1], a[2])
    elif cmd == "show":
        te.show("syn", a[0], brief=len(a) > 1 and a[1] == "brief", part="main")
    elif cmd == "tools":
        te.show("syn" if "-" in a[0] else "train", a[0], part=f"tools:{a[1]}")
    elif cmd == "guide":
        print(open(te.ROOT / f"guide_{a[0]}.txt", encoding="utf-8").read())
    elif cmd == "call":
        te.call("syn", a[0], a[1], a[2], a[3] if len(a) > 3 else "{}")
    elif cmd == "done":
        te.done("syn", a[0], a[1])
    elif cmd == "reset":
        te.reset("syn", a[0], a[1])
    elif cmd == "collect":
        collect(a[0])
