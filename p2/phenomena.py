"""Quet hien tuong gay kho trong cau noi / nhan gold va do diem cua model moc (sft_lc50) tren train_h50 (mau gold chua hoc).
    python phenomena.py            # bang tong hop
    python phenomena.py <ten>      # in vi du cua 1 hien tuong
"""
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import scorer  # noqa: E402
from make_lc_splits import DATA, fam, kind  # noqa: E402

items = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")}
cat = json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8"))
KIND = {t["name"]: t["kind"] for a in cat["apps"] for t in a["tools"]}
tools = agent.tool_index(cat)
gold = json.load(open(ROOT / "artifacts" / "gold_hashes.json"))
sp = json.load(open(ROOT / "artifacts" / "splits3.json"))
P = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs" / "sft_lc50_train_h50" / "predictions.jsonl", encoding="utf-8")}
per = {i: scorer.sample_score(p["tool_calls"], items[i]["label"]["tool_calls"], p["state_hash"] == gold[i], tools) for i, p in P.items()}


def txt(it):
    return " " + unicodedata.normalize("NFC", it.get("audio_text", "")).lower() + " "


def R(p):
    rx = re.compile(p)
    return lambda it: bool(rx.search(txt(it)))


def calls(it):
    return it["label"]["tool_calls"]


def nwrite(it):
    return sum(1 for c in calls(it) if KIND.get(c["name"]) == "write")


def n_hist(it):
    return sum(1 for t in it.get("history") or [] if t["role"] == "user")


from traps import B, CALC, CANCEL, DICT, E, FIX  # noqa: E402  (dinh nghia dung chung voi bo sinh du lieu)

COND = B + "nếu" + E + r"|còn không thì|không thì thôi|mà .{0,25} thì "
NUMCMP = B + "(đủ|dưới|trên|hơn|quá|thiếu|ít nhất|tối đa)" + E
PH = collections.OrderedDict([
    ("tự huỷ / đổi ý (thôi khỏi, đừng… nữa)", R(CANCEL)),
    ("tự sửa lời (à nhầm, à không, ý là)", R(FIX)),
    ("  └ có lời huỷ/sửa NHƯNG gold vẫn phải làm", lambda it: (R(CANCEL)(it) or R(FIX)(it)) and bool(calls(it))),
    ("  └ có lời huỷ/sửa VÀ gold rỗng", lambda it: (R(CANCEL)(it) or R(FIX)(it)) and not calls(it)),
    ("điều kiện (nếu… thì)", R(r"\bnếu\b|còn không thì|không thì thôi|mà .{0,25} thì ")),
    ("  └ điều kiện so sánh số (đủ, dưới, trên, hơn, quá)", lambda it: R(COND)(it) and R(NUMCMP)(it)),
    ("  └ điều kiện, gold KHÔNG có thao tác ghi", lambda it: R(r"\bnếu\b")(it) and bool(calls(it)) and nwrite(it) == 0),
    ("phủ định / loại trừ (đừng, trừ, khỏi, không cần)", R(r"\bđừng\b|ngoại trừ|\btrừ\b|\bkhỏi\b|không cần|chưa cần|khoan hãy")),
    ("nhiều đối tượng (cả hai, mỗi, tất cả, hết, mấy)", R(r"cả hai|cả ba|\bmỗi\b|tất cả|\bhết\b|\bmấy\b|\bcác\b|lần lượt|từng ")),
    ("chọn theo tiêu chí (…nhất, đầu tiên, cuối)", R(r"\bnhất\b|đầu tiên|cuối cùng|sớm hơn|muộn hơn|rẻ hơn|đắt hơn|nhiều hơn|ít hơn")),
    ("tham chiếu history (hồi nãy, cái đó, vừa…)", R(r"hồi nãy|lúc nãy|khi nãy|\bnãy\b|vừa rồi|vừa nói|vừa (đặt|tạo|gửi|bật|tắt|hỏi)|cái đó|cái đấy|cái ấy|chỗ đó|người đó|bài đó|\bấy\b|như cũ|lần trước|hôm trước|hôm qua")),
    ("tính giờ (trước/sau N phút, nửa tiếng, rưỡi)", R(r"(trước|sau|sớm hơn|muộn hơn|lùi|dời|thêm) .{0,25}(phút|tiếng|giờ)|nửa tiếng|tiếng nữa|phút nữa|\brưỡi\b|kém")),
    ("ngày tương đối xa (mốt, tuần sau, thứ…)", R(r"\bmốt\b|ngày kia|tuần sau|tuần tới|cuối tuần|thứ (hai|ba|tư|năm|sáu|bảy)|chủ nhật")),
    ("tiền / số đọc bằng chữ", R(r"(nghìn|ngàn|triệu|trăm|chục|\bk\b|đồng|\blít\b|ký|cân)")),
    ("tính toán (chia, mỗi người, một nửa, tổng, gấp)", R(CALC)),
    ("đọc chính tả (email 'a còng', số điện thoại đọc từng số)", R(DICT)),
    ("phương ngữ (răng, rứa, mô, chừ, nghen, hè, tê)", R(r"\brăng\b|\brứa\b|\bmô\b|\bchừ\b|nghen|\bhè\b|\btê\b|\bni\b|\bnớ\b|\bhỉ\b|\bhen\b|\bdùm\b|\bhông\b|\bvô\b|\bbả\b|\bổng\b|\bmạ\b")),
    ("câu dài (> 150 ký tự)", lambda it: len(it.get("audio_text", "")) > 150),
    ("câu rất ngắn (< 40 ký tự)", lambda it: len(it.get("audio_text", "")) < 40),
    ("history dài (≥ 4 lượt user)", lambda it: n_hist(it) >= 4),
    ("không có history", lambda it: n_hist(it) == 0),
    ("nhiều tool (≥ 40 tool được cài)", lambda it: sum(len(a["tools"]) for a in it["context"]["installed_apps"].values()) >= 40),
    ("gold rỗng", lambda it: not calls(it)),
    ("gold hỏi lại", lambda it: kind(it) == "ask"),
    ("  └ hỏi lại SAU khi đã làm việc khác", lambda it: kind(it) == "ask" and nwrite(it) >= 1),
    ("  └ hỏi lại ngay, không đọc gì", lambda it: kind(it) == "ask" and len(calls(it)) == 1),
    ("gold chỉ đọc (không ghi, không hỏi)", lambda it: kind(it) == "exec" and nwrite(it) == 0),
    ("gold ≥ 3 thao tác ghi", lambda it: nwrite(it) >= 3),
    ("gold dùng ≥ 3 app", lambda it: len({c["name"].split(".")[0] for c in calls(it)} - {"sys"}) >= 3),
    ("gold lặp cùng 1 tool ghi ≥ 2 lần", lambda it: any(v >= 2 for k, v in collections.Counter(c["name"] for c in calls(it) if KIND.get(c["name"]) == "write").items())),
    ("gold có ID lấy từ history (ghi mà không đọc trước)", lambda it: nwrite(it) >= 1 and all(KIND.get(c["name"]) != "read" for c in calls(it)) and any(k.endswith("_id") for c in calls(it) for k in c["arguments"])),
])

if len(sys.argv) > 1:
    key = next(k for k in PH if sys.argv[1] in k)
    n = 0
    for i in sp["train"]:
        it = items[i]
        if PH[key](it):
            sc = f"{per[i]:.2f}" if i in per else "  - "
            print(f"[{sc}] {fam(it)} | {it['audio_text'][:170]}\n        -> " + "; ".join(c["name"] for c in calls(it))[:150])
            n += 1
            if n >= int(sys.argv[2]) if len(sys.argv) > 2 else n >= 12:
                break
    sys.exit()

H = [i for i in sp["train_h50"]]
base = 100 * sum(per[i] for i in H) / len(H)
print(f"Mốc: sft_lc50 trên train_h50 = {base:.1f} ({len(H)} mẫu). 'TRAIN' = số mẫu trong 1127 mẫu train; 'đo' = số mẫu train_h50 có hiện tượng.")
print(f"{'hiện tượng':52}{'TRAIN':>6}{'%':>5}{'đo':>5}{'điểm':>7}{'so mốc':>8}")
rows = []
for k, f in PH.items():
    tr = [i for i in sp["train"] if f(items[i])]
    h = [i for i in H if f(items[i])]
    s = 100 * sum(per[i] for i in h) / len(h) if h else float("nan")
    rows.append((k, len(tr), len(h), s))
for k, a, b, s in rows:
    flag = "" if b >= 15 else "  (ít mẫu)"
    print(f"{k:52}{a:>6}{100 * a / len(sp['train']):>5.0f}{b:>5}{s:>7.1f}{s - base:>+8.1f}{flag}")
