"""Pha 1 (V3_DESIGN.md): moi truong cho "thay" (LLM lon) giai bai trong simulator that.

Thay KHONG duoc doc train.jsonl / *_environment. Chi dung cac lenh duoi:
    python teacher_env.py show  <split> <id>                 # de bai: context, history, tool, quy uoc, vi du gold truy hoi tu TRAIN
    python teacher_env.py call  <split> <id> <run> <tool> '<json args>'   # goi 1 tool, in ket qua (state luu qua cac lan goi)
    python teacher_env.py done  <split> <id> <run>           # ket thuc luot, ghi tool_calls + state_hash
    python teacher_env.py reset <split> <id> <run>           # lam lai tu dau
Ket qua: teacher_runs/<split>/<id>.<run>.json  (log cac call; state duoc tao lai bang replay moi lan goi)
Cham:    python teacher_env.py score <split> <run>  (chi nguoi van hanh dung, so voi gold)
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test"
sys.path.insert(0, str(DATA / "train"))
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402

RUNS = ROOT / "teacher_runs"
K_EXAMPLES = 3

CONVENTIONS = """Quy ước của bộ dữ liệu (rút từ dữ liệu train, bám sát để được chấm đúng):
1. Mỗi lần chỉ gọi 1 tool. Được chấm theo DANH SÁCH call (kể cả call đọc) + trạng thái cuối. Call đọc thừa hoặc thiếu đều bị trừ điểm.
2. Tham số ID (contact_id, device_id, bill_id...) phải lấy từ kết quả tool đọc hoặc từ history. Nếu ID chưa có trong history: gọi tool đọc sinh ra nó trước (vd contacts.search trước sms.send, smart_home.get_status trước light_set, bill_payment.list_bills trước pay, calendar.list_events trước update_event/cancel_event). Nếu ID đã có trong history: KHÔNG đọc lại, gọi thẳng.
3. Trước khi chuyển tiền ngân hàng: banking.get_balance và banking.list_beneficiaries. Đặt món: list_restaurants rồi get_menu.
4. Câu có điều kiện ("nếu ... thì ..."): gọi tool đọc để kiểm tra; điều kiện sai thì dừng, không làm vế sau.
5. Hỏi lại (sys.ask_missing_info với tool_name và đúng tên tham số trong schema) khi: thiếu tham số bắt buộc không suy ra được; hoặc tool đọc trả về ≥2 kết quả mà câu nói không đủ để chọn. Tìm ra đúng 1 kết quả thì KHÔNG hỏi. Hỏi xong thì dừng.
6. Trả rỗng (không gọi tool) khi: người dùng huỷ/đổi ý ("à thôi"), tán gẫu, cảm ơn, app cần thiết chưa cài, hoặc thông tin đã có sẵn trong history.
7. Tham số tuỳ chọn: chỉ điền khi câu nói nêu rõ. Thời gian ISO 8601 +07:00 tính theo current_time. Tiền là số nguyên VND.
7b. Tool ĐỌC dạng liệt kê/trạng thái (list_*, get_status, get_balance, get_points, find_*, my_*, order_history, list_events...): gọi với {} — KHÔNG truyền bộ lọc tuỳ chọn (room, type, near, date_from...) kể cả khi câu nói có nhắc; tự lọc trong kết quả. Chỉ tool tìm kiếm/tra cứu (search, get_price, get_forecast, get_route, convert, get_rate, get_menu, read_*) mới truyền tham số chính.
7c. Chỉ gọi tool đọc khi cần ID cho bước ghi, hoặc cần kiểm tra điều kiện, hoặc người dùng hỏi thông tin. Không đọc thêm "cho chắc". Không search_place nếu tool sau nhận tên địa điểm dạng chữ.
7d. calendar.create_event: nếu người dùng chỉ nói giờ bắt đầu thì end = start + 1 giờ.
8. contacts.search: query viết như tên/quan hệ người dùng nói (vd "Lan", "mẹ"). Nội dung tin nhắn/ghi chú/tiêu đề: ngắn gọn, sát lời người dùng.
9. Không làm lại thao tác đã làm ở lượt trước trong history."""


def _load_items(split):
    p = DATA / split / f"{split}.jsonl"
    return {json.loads(line)["id"]: json.loads(line) for line in open(p, encoding="utf-8")}


def env_key(split, qid):
    """(split, id) cua moi truong that. Mau sinh 'syn': id = '<base>-<k>' -> moi truong cua mau TRAIN goc."""
    return ("train", qid.rsplit("-", 1)[0]) if split == "syn" else (split, qid)


def get_item(split, qid):
    """Mau sinh: giu context + history cua mau goc, thay yeu cau cuoi bang cau moi, bo nhan."""
    if split != "syn":
        return _load_items(split)[qid]
    base = _load_items("train")[env_key(split, qid)[1]]
    req = json.load(open(RUNS / "syn" / f"{qid}.req.json", encoding="utf-8"))
    it = {k: v for k, v in base.items() if k != "label"}
    it["audio_text"] = req["text"]
    return it  # it["id"] = id goc -> retrieve() khong bao gio tra ve chinh mau goc


def _tools(split):
    split = "train" if split == "syn" else split
    return agent.tool_index(json.load(open(DATA / split / f"{split}_tools_catalog.json", encoding="utf-8")))


def _texts(item):
    """Text user: gold audio_text cho train; split khac (public) phai truyen transcript rieng."""
    return agent.texts_gold(item)


# ---------------------------------------------------------------- retrieval tu TRAIN (khong bao gio tra ve chinh no)
_TOK = re.compile(r"\w+", re.U)


def _bag(it):
    return Counter(t.lower() for t in _TOK.findall(it.get("audio_text", "")))


def retrieve(item, k=K_EXAMPLES, pool_name="train"):
    tr = _load_items("train")
    sp = json.load(open(ROOT / "artifacts" / "splits3.json", encoding="utf-8"))
    pool = [i for i in sp[pool_name] if i != item["id"]]
    df = Counter()
    for i in pool:
        df.update(set(_bag(tr[i])))
    n = len(pool)
    idf = lambda w: math.log((n + 1) / (df[w] + 1)) + 1  # noqa: E731
    q = _bag(item)
    qa = set(item["context"]["installed_apps"])
    qn = math.sqrt(sum((c * idf(w)) ** 2 for w, c in q.items())) or 1
    scored = []
    for i in pool:
        b = _bag(tr[i])
        dot = sum(c * b[w] * idf(w) ** 2 for w, c in q.items() if w in b)
        bn = math.sqrt(sum((c * idf(w)) ** 2 for w, c in b.items())) or 1
        used = {c["name"].split(".")[0] for c in tr[i]["label"]["tool_calls"]} - {"sys"}
        app_fit = len(used & qa) / len(used) if used else 0.5  # vi du dung app ma mau nay co cai
        scored.append((dot / (qn * bn) + 0.3 * app_fit, i))
    return [tr[i] for _, i in sorted(scored, reverse=True)[:k]]


def _fmt_example(ex):
    calls = ex["label"]["tool_calls"]
    res = ex["label"].get("tool_results") or []
    lines = [f'Người dùng: "{ex.get("audio_text", "")}"']
    if ex.get("history"):
        lines.append(f"(có {sum(1 for t in ex['history'] if t['role'] == 'user')} lượt trước trong history)")
    if not calls:
        lines.append("→ không gọi tool nào")
    for j, c in enumerate(calls):
        r = json.dumps(res[j], ensure_ascii=False) if j < len(res) else ""
        lines.append(f"→ {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" + (f"   ⇒ {r[:200]}" if r else ""))
    return "\n".join(lines)


PAGE = 11000  # ky tu moi trang schema (dau ra cua 1 lenh bi cat o ~16KB)


def tool_lines(split, it):
    cs = "train" if split == "syn" else split
    cat = {t["name"]: t for a in json.load(open(DATA / cs / f"{cs}_tools_catalog.json", encoding="utf-8"))["apps"] for t in a["tools"]}
    out = []
    for t in agent.tools_for(it, _tools(split)):
        f = t["function"]
        kind = cat.get(f["name"], {}).get("kind", "meta")
        ret = cat.get(f["name"], {}).get("returns", "")
        out.append(f"- {f['name']} [{kind}{', trả về ' + ret if ret else ''}]: {f['description']} | params={json.dumps(f['parameters'], ensure_ascii=False)}")
    return out


def tool_pages(split, it):
    pages, cur = [], []
    for ln in tool_lines(split, it):
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > PAGE:
            pages.append(cur)
            cur = []
        cur.append(ln)
    return pages + ([cur] if cur else [])


def show(split, qid, pool_name=None, brief=False, part="all"):
    """part: 'all' (cu, 1 khoi) | 'main' (khong co schema; schema xem bang tools <trang>) | 'tools:<n>'."""
    it = get_item(split, qid)
    pool_name = pool_name or ("train_q50" if split == "syn" else "train")
    if part.startswith("tools:"):
        pages = tool_pages(split, it)
        n = int(part.split(":")[1])
        print(f"=== TOOL ĐƯỢC DÙNG (schema) — trang {n}/{len(pages)} ===")
        print("\n".join(pages[n - 1]))
        return
    ex = ["=== VÍ DỤ ĐÃ GIẢI ĐÚNG (từ dữ liệu train, bối cảnh khác) ==="]
    for e in retrieve(it, pool_name=pool_name):
        ex += [_fmt_example(e), ""]
    if brief:  # cung boi canh/tool voi lan show day du truoc do; chi in yeu cau moi + vi du
        print("\n".join([f'=== YÊU CẦU MỚI CẦN XỬ LÝ (cùng bối cảnh, history, tool như trên): "{it["audio_text"]}"', ""] + ex))
        return
    msgs = agent.build_messages(it, _texts(it))
    out = ["=== BỐI CẢNH (system) ===", msgs[0]["content"], "", "=== HISTORY + YÊU CẦU ==="]
    for m in msgs[1:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append("assistant gọi: " + "; ".join(f"{t['function']['name']}({json.dumps(t['function']['arguments'], ensure_ascii=False)})" for t in m["tool_calls"]))
        else:
            out.append(f"{m['role']}: {m['content']}")
    out.append("")
    if part == "all":
        out += ["=== TOOL ĐƯỢC DÙNG (schema) ==="] + tool_lines(split, it) + [""]
    else:
        n = len(tool_pages(split, it))
        out += [f"=== TOOL: schema có {n} trang — xem TẤT CẢ bằng lệnh `tools <ID> 1` … `tools <ID> {n}` trước khi làm ===", ""]
    out += ["=== " + CONVENTIONS, ""] + ex
    out.append("=== YÊU CẦU CẦN XỬ LÝ: lượt user cuối cùng ở trên ===")
    print("\n".join(out))


# ---------------------------------------------------------------- chay tool co trang thai (replay log)
def _log_path(split, qid, run):
    p = RUNS / split
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{qid}.{run}.json"


def _env_replay(split, qid, calls):
    from zaic_tools import ToolEnv
    es, eid = env_key(split, qid)
    env = ToolEnv(es, eid, root=str(DATA))
    for c in calls:
        env.call(c["name"], c["arguments"])
    return env


def call(split, qid, run, name, args_json):
    lp = _log_path(split, qid, run)
    log = json.load(open(lp, encoding="utf-8")) if lp.exists() else {"calls": [], "done": False}
    if log["done"]:
        print("Lượt này đã kết thúc (done). Dùng reset để làm lại.")
        return
    args = json.loads(args_json) if args_json.strip() else {}
    env = _env_replay(split, qid, log["calls"])
    res = env.call(name, args)
    log["calls"].append({"name": name, "arguments": args})
    json.dump(log, open(lp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:6000])
    if name == "sys.ask_missing_info":
        print("(Đã hỏi lại người dùng — lượt này nên kết thúc: gọi done.)")


def done(split, qid, run):
    lp = _log_path(split, qid, run)
    log = json.load(open(lp, encoding="utf-8")) if lp.exists() else {"calls": [], "done": False}
    env = _env_replay(split, qid, log["calls"])
    log.update(done=True, tool_calls=env.get_tool_calls(), state_hash=env.get_state_hash())
    json.dump(log, open(lp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"done: {len(log['tool_calls'])} call")


def reset(split, qid, run):
    lp = _log_path(split, qid, run)
    if lp.exists():
        lp.unlink()
    print("reset")


# ---------------------------------------------------------------- cham (nguoi van hanh)
def score(split, run, ids=None):
    import scorer
    items = _load_items("train")
    tools = _tools("train")
    gold = json.load(open(ROOT / "artifacts" / "gold_hashes.json"))
    out = {}
    for lp in sorted((RUNS / split).glob(f"*.{run}.json")):
        qid = lp.name.split(".")[0]
        if ids and qid not in ids:
            continue
        log = json.load(open(lp, encoding="utf-8"))
        if not log.get("done"):
            continue
        it = items[qid]
        out[qid] = scorer.sample_score(log["tool_calls"], it["label"]["tool_calls"], log["state_hash"] == gold[qid], tools)
    return out


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "show":
        show(a[0], a[1], brief=len(a) > 2 and a[2] == "brief")
    elif cmd == "call":
        call(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif cmd == "done":
        done(a[0], a[1], a[2])
    elif cmd == "reset":
        reset(a[0], a[1], a[2])
    elif cmd == "score":
        s = score(a[0], a[1])
        print(json.dumps({"n": len(s), "score": round(100 * sum(s.values()) / max(len(s), 1), 2), "per": s}, indent=1))
