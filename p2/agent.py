"""Baseline zero-shot ReAct agent cho LifeMate phase 2 (text -> tool calls).

Pipeline dung chung cho train va test:
    system prompt + schema app da cai + context + history + text user
      -> LLM (Qwen3 tool-calling goc) -> env.call -> ket qua -> LLM -> ... -> dung

Khong co guard. Chi 2 quy uoc toi thieu cua giao thuc:
  - sau sys.ask_missing_info thi dung (luot nay ket thuc, cho nguoi dung tra loi);
  - toi da MAX_STEPS luot sinh cua model.
Luc inference chi dung env.call / get_tool_calls / get_state_hash (dung luat BTC).
"""
from __future__ import annotations

import json
import re
from typing import Callable

MAX_STEPS = 10
MAX_RESULT_CHARS = 6000

ASK_TOOL = {
    "type": "function",
    "function": {
        "name": "sys.ask_missing_info",
        "description": "Hỏi lại người dùng khi yêu cầu thiếu thông tin bắt buộc để gọi một tool. "
                       "Gọi xong thì dừng, chờ người dùng trả lời.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Tool định gọi, dạng app.tool"},
                "params": {"type": "array", "items": {"type": "string"},
                           "description": "Tên các tham số còn thiếu, đúng tên trong schema của tool"},
            },
            "required": ["tool_name", "params"],
        },
    },
}

SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý giọng nói tiếng Việt điều khiển các app trên thiết bị của người dùng.
Nhiệm vụ: xử lý YÊU CẦU CUỐI CÙNG của người dùng bằng cách gọi tool. Bạn được chấm theo việc bạn LÀM (các tool call và trạng thái app sau cùng), không theo lời bạn nói.

Quy tắc:
- Chỉ gọi các tool được cung cấp (thuộc các app đã cài).
- Thời gian tương đối ("mai", "tối nay", "thứ 6 tuần sau") tính theo current_time trong ngữ cảnh. Thời gian ghi theo ISO 8601 kèm múi giờ, dạng YYYY-MM-DDTHH:MM:SS+07:00.
- ID (device_id, contact_id, event_id, ...) phải lấy từ kết quả của tool đọc dữ liệu, không tự bịa.
- Nếu yêu cầu thiếu thông tin bắt buộc mà không suy ra được, gọi sys.ask_missing_info với tool định gọi và tên các tham số còn thiếu, rồi dừng.
- Nếu không app nào đã cài làm được yêu cầu, trả lời ngắn và KHÔNG gọi tool nào.
- Không gọi tool thừa. Các thao tác ở lượt trước đã được thực hiện rồi, không làm lại.
- Khi đã xong, trả lời ngắn gọn bằng tiếng Việt, không kèm tool call.

Ngữ cảnh người dùng:
{context}

Mốc ngày (tính từ current_time):
{dates}"""

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
SHORT = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
STOP_DONE = "Đã xong."
STOP_OOS = "Không có ứng dụng phù hợp để thực hiện yêu cầu này."


def date_table(current_time: str) -> str:
    """Bang moc ngay: hom nay/mai/mot + cac thu tuan nay va tuan sau (tuan bat dau Thu Hai)."""
    from datetime import datetime, timedelta
    d = datetime.fromisoformat(current_time).date()
    f = lambda x: f"{WEEKDAYS[x.weekday()]} {x.isoformat()}"  # noqa: E731
    mon = d - timedelta(days=d.weekday())
    week = lambda start: ", ".join(f"{SHORT[k]}={(start + timedelta(days=k)).isoformat()}" for k in range(7))  # noqa: E731
    return (f"hôm nay = {f(d)}; mai = {f(d + timedelta(days=1))}; mốt (ngày kia) = {f(d + timedelta(days=2))}\n"
            f"tuần này: {week(mon)}\ntuần sau: {week(mon + timedelta(days=7))}")


# ---------------------------------------------------------------- tools / prompt
def tool_index(catalog: dict) -> dict[str, dict]:
    return {t["name"]: t for a in catalog["apps"] for t in a["tools"]}


def tools_for(item: dict, tools: dict[str, dict]) -> list[dict]:
    out = []
    for app in item["context"]["installed_apps"].values():
        for name in app["tools"]:
            t = tools.get(name)
            if t is None:
                continue
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}})}})
    out.append(ASK_TOOL)
    return out


def _dumps(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + "...(cắt bớt)"


def _tc(call: dict) -> dict:
    return {"type": "function", "function": {"name": call["name"], "arguments": call.get("arguments") or {}}}


def build_messages(item: dict, texts: dict[str, str]) -> list[dict]:
    """texts: map duong dan audio -> text user (gold audio_text hoac transcript ASR)."""
    ctx = {k: v for k, v in item["context"].items() if k != "installed_apps"}
    msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(
        context=json.dumps(ctx, ensure_ascii=False, indent=1), dates=date_table(ctx["current_time"]))}]
    for turn in item.get("history") or []:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": texts.get(turn.get("audio", ""), "")})
            continue
        calls = turn.get("tool_calls") or []
        if calls:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(c) for c in calls]})
            results = turn.get("tool_results") or []
            for i in range(len(calls)):
                msgs.append({"role": "tool", "content": _dumps(results[i]) if i < len(results) else "{}"})
        msgs.append({"role": "assistant", "content": turn.get("text", "")})
    msgs.append({"role": "user", "content": texts.get(item["audio"], "")})
    return msgs


def texts_gold(item: dict) -> dict[str, str]:
    """Train: lay audio_text (golden transcript) cho luot cuoi va cac luot user trong history."""
    t = {item["audio"]: item.get("audio_text", "")}
    for turn in item.get("history") or []:
        if turn["role"] == "user":
            t[turn["audio"]] = turn.get("audio_text", "")
    return t


def append_step(msgs: list, call: dict, result) -> bool:
    """Noi 1 buoc (assistant call + ket qua tool) vao hoi thoai. Dung chung cho inference va SFT.
    Tra True neu phai dung (sau sys.ask_missing_info: khong them ket qua, luot ket thuc)."""
    msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(call)]})
    if call["name"] == "sys.ask_missing_info":
        return True
    msgs.append({"role": "tool", "content": _dumps(result)})
    return False


def sft_messages(item: dict, texts: dict[str, str]) -> tuple[list[dict], int]:
    """Hoi thoai SFT day du cho 1 mau train. Tra (messages, n_prefix): messages[n_prefix:] la luot cuoi;
    cac message assistant trong do la target (tinh loss)."""
    msgs = build_messages(item, texts)
    n_prefix = len(msgs)
    lab = item["label"]
    stopped = False
    for call, res in zip(lab["tool_calls"], lab.get("tool_results") or []):
        stopped = append_step(msgs, call, res)
        if stopped:
            break
    if not stopped:
        msgs.append({"role": "assistant", "content": STOP_DONE if lab["tool_calls"] else STOP_OOS})
    return msgs, n_prefix


# ---------------------------------------------------------------- parse model output
_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[dict]:
    raws = _TC_RE.findall(text)
    if not raws and "<tool_call>" in text:  # thieu </tool_call>: lay JSON dau tien sau <tool_call>
        tail = text.split("<tool_call>", 1)[1].lstrip()
        try:
            obj, _ = json.JSONDecoder().raw_decode(tail)
            raws = [json.dumps(obj, ensure_ascii=False)]
        except json.JSONDecodeError:
            raws = []
    calls = []
    for raw in raws:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


# ---------------------------------------------------------------- batched ReAct loop
def run_batch(items: list[dict], texts_fn: Callable[[dict], dict], tools: dict[str, dict],
              new_env: Callable[[str], object], generate: Callable[[list[tuple[list, list]]], list[str]]):
    """Chay ReAct dong bo theo buoc cho ca lo: moi buoc gom moi mau chua xong -> generate 1 lan.
    generate([(messages, tools), ...]) -> [text, ...]; tra None cho mau qua dai (dung mau do).
    Tra ve {id: {"tool_calls", "state_hash", "trace"}}."""
    st = {}
    for it in items:
        st[it["id"]] = {"env": new_env(it["id"]), "msgs": build_messages(it, texts_fn(it)),
                        "tools": tools_for(it, tools), "done": False, "trace": []}
    for _ in range(MAX_STEPS):
        active = [i for i, s in st.items() if not s["done"]]
        if not active:
            break
        outs = generate([(st[i]["msgs"], st[i]["tools"]) for i in active])
        for i, text in zip(active, outs):
            s = st[i]
            if text is None:
                s["trace"].append({"stop": "prompt_too_long"})
                s["done"] = True
                continue
            calls = parse_tool_calls(text)
            s["trace"].append({"out": text})
            if not calls:
                s["done"] = True
                continue
            c = calls[0]  # moi buoc dung 1 call, khop voi du lieu SFT
            res = s["env"].call(c["name"], c["arguments"])
            s["trace"].append({"call": c, "result": res})
            if append_step(s["msgs"], c, res):
                s["done"] = True
    return {i: {"tool_calls": s["env"].get_tool_calls(), "state_hash": s["env"].get_state_hash(),
                "trace": s["trace"]} for i, s in st.items()}
