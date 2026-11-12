"""Bay trong cau noi: MOT dinh nghia dung chung cho chan doan (phenomena.py) va sinh du lieu (synth_env.py).

Moi bay nhan dien bang regex tren cau noi (va danh sach call khi can). Regex chi de DEM va GAN NHAN,
khong dung de sua dap an. Ranh gioi tu dung B/E thay cho \\b vi \\b cat sai o chu co dau.
"""
import re
import unicodedata

B = r"(?<![a-zà-ỹ])"  # ranh gioi tu (tranh khop "mà không" voi "à không")
E = r"(?![a-zà-ỹ])"
CANCEL = (r"(à |ơ |mà )?thôi(,| )+(khỏi|đừng|không|khoan|bỏ|để|kệ|dẹp)|thôi khỏi|khỏi cần|không cần nữa|bỏ đi|"
          r"đừng .{0,20} nữa|huỷ lệnh|hủy lệnh|thôi nhé|thôi nha|thôi vậy")
FIX = (B + r"(à|ấy|ơ|ủa),? ?(nhầm|không|quên|khoan)" + E + "|" + B +
       r"nhầm(,| rồi)|ý (tôi|anh|chị|em|mình|tớ|cô|chú|bác|con) là|đổi lại|khoan đã|chứ không phải")
DICT = r"a còng|chấm (com|vn|test|net)|không chín|không ba|không tám|không bảy"
CALC = (r"\bchia\b|mỗi người|một nửa|\btổng\b|gấp (đôi|ba)|cộng|\bnhân\b|trung bình|"
        r"bao nhiêu (bước|lít|cây|tiền) .{0,20}thì")
LONG_CHARS = 150

_RX = {k: re.compile(v) for k, v in {"cancel": CANCEL, "fix": FIX, "dict": DICT, "calc": CALC}.items()}


def norm(text) -> str:
    return " " + unicodedata.normalize("NFC", text or "").lower() + " "


def match(name: str, text: str) -> bool:
    return bool(_RX[name].search(norm(text)))


def n_apps(calls) -> int:
    return len({c["name"].split(".")[0] for c in calls} - {"sys"})


def has(trap: str, text: str, calls: list) -> bool:
    """Cau (text) + dap an (calls) co mang bay nay khong. Dung cho ca gold (thong ke) va mau sinh (gan nhan)."""
    if trap == "fix":   # tu sua loi, van phai lam
        return match("fix", text) and bool(calls)
    if trap == "part":  # co loi huy nhung van phai lam phan con lai
        return match("cancel", text) and not match("fix", text) and bool(calls)
    if trap == "dict":  # doc chinh ta email / so dien thoai
        return match("dict", text)
    if trap == "calc":  # phai tinh toan
        return match("calc", text)
    if trap == "app3":  # dap an dung tu 3 app
        return n_apps(calls) >= 3
    if trap == "long":  # cau dai
        return len(text or "") > LONG_CHARS
    raise KeyError(trap)


ALL = ("fix", "part", "dict", "calc", "app3", "long")
