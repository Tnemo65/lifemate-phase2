"""Pha 0c (V3_DESIGN.md): dao quy uoc cua gold tren TRAIN, kiem lai do nhat quan tren DEV (chi dung NHAN, khong dung model).
Ket qua: artifacts/rulebook.json + in tom tat.
  A. tap key tham so cua moi tool (key tuy chon nao hay co)
  B. read-truoc-write theo x-ref: write dung ID cua collection C ma ID chua co trong history -> co goi read sinh ra C truoc khong
  C. read di kem truoc write (kiem tra so du, ...) ngoai x-ref
  D. sys.ask_missing_info: hoi tham so gi, co read truoc khong
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "train+public_test" / "train+public_test" / "train"
ASK = "sys.ask_missing_info"

cat = json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8"))
T = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
items = {}
for line in open(DATA / "train.jsonl", encoding="utf-8"):
    it = json.loads(line)
    items[it["id"]] = it
sp = json.load(open(ROOT / "artifacts" / "splits3.json", encoding="utf-8"))


def producers(ref):
    """x-ref 'contacts.contacts' -> cac read tool cua app do co returns chua 'contacts'."""
    app, coll = ref.split(".", 1)
    return {n for n, t in T.items() if t["app"] == app and t["kind"] == "read" and coll in (t.get("returns") or "")}


def hist_text(it):
    return json.dumps(it.get("history") or [], ensure_ascii=False)


def mine(ids):
    keysets = collections.defaultdict(collections.Counter)
    xref = collections.defaultdict(collections.Counter)      # (write, param) -> {read_truoc, id_trong_history, khong_ca_hai}
    pre = collections.defaultdict(collections.Counter)       # write -> read di truoc
    nw = collections.Counter()
    ask = collections.Counter()
    for i in ids:
        it = items[i]
        g = it["label"]["tool_calls"]
        h = hist_text(it)
        for k, c in enumerate(g):
            n, a = c["name"], c.get("arguments") or {}
            keysets[n][tuple(sorted(a))] += 1
            before = [x["name"] for x in g[:k]]
            if n == ASK:
                tn, ps = a.get("tool_name"), a.get("params") or []
                props = (T.get(tn) or {}).get("parameters", {}).get("properties", {})
                for p in ps:
                    ref = (props.get(p) or {}).get("x-ref")
                    if ref:
                        ask["hoi_ID|" + ("da_read_truoc" if producers(ref) & set(before) else
                                         ("ID_lien_quan_trong_history" if ref.split(".")[0] in h else "khong_read"))] += 1
                    else:
                        ask["hoi_gia_tri|" + ("co_call_truoc" if before else "hoi_ngay")] += 1
                continue
            t = T.get(n)
            if not t or t["kind"] != "write":
                continue
            nw[n] += 1
            for r in set(before):
                if T.get(r, {}).get("kind") == "read":
                    pre[n][r] += 1
            for p, v in a.items():
                ref = (t["parameters"]["properties"].get(p) or {}).get("x-ref")
                if not ref or not isinstance(v, str):
                    continue
                if f'"{v}"' in h:
                    xref[(n, p)]["id_trong_history"] += 1
                elif producers(ref) & set(before):
                    xref[(n, p)]["read_truoc"] += 1
                else:
                    xref[(n, p)]["khong_ca_hai"] += 1
    return keysets, xref, pre, nw, ask


def main():
    tr, dv = mine(sp["train"]), mine(sp["dev"])
    out = {}
    # B
    def xsum(x):
        c = collections.Counter()
        for v in x.values():
            c.update(v)
        return c
    for nm, m in (("TRAIN", tr), ("DEV", dv)):
        c = xsum(m[1])
        need = c["read_truoc"] + c["khong_ca_hai"]
        print(f"B {nm}: ID da co trong history {c['id_trong_history']} | ID moi: {need}, trong do co read sinh ra truoc: "
              f"{c['read_truoc']} ({100 * c['read_truoc'] / max(need, 1):.1f}%)")
    bad = {f"{k[0]}.{k[1]}": dict(v) for k, v in tr[1].items() if v["khong_ca_hai"]}
    print("  ngoai le TRAIN (ID moi ma khong read truoc):", bad)
    out["xref_read_before_write"] = {f"{k[0]}.{k[1]}": dict(v) for k, v in tr[1].items()}
    # C
    print("C. read di kem truoc write (TRAIN, >=80%, >=8 lan; * = ngoai x-ref) -> ti le tren DEV")
    out["precheck"] = {}
    for w, n in sorted(tr[3].items(), key=lambda x: -x[1]):
        refs = set()
        for p in T[w]["parameters"]["properties"].values():
            if p.get("x-ref"):
                refs |= producers(p["x-ref"])
        for r, k in tr[2][w].most_common():
            if n >= 8 and k / n >= 0.8:
                dn = dv[3].get(w, 0)
                out["precheck"][f"{r} -> {w}"] = [k, n]
                print(f"   {'*' if r not in refs else ' '} {r} -> {w}: {k}/{n} = {100 * k / n:.0f}%   DEV: {dv[2][w].get(r, 0)}/{dn}")
    # A
    print("A. tool co >1 kieu tap key (TRAIN, >=10 lan): ti le kieu pho bien nhat")
    out["keysets"] = {}
    mixed = []
    for n, c in tr[0].items():
        tot = sum(c.values())
        out["keysets"][n] = {",".join(k): v for k, v in c.most_common()}
        if tot >= 10 and len(c) > 1:
            mixed.append((c.most_common(1)[0][1] / tot, n, tot, c.most_common(3)))
    for share, n, tot, top in sorted(mixed):
        print(f"   {n} (n={tot}) {100 * share:.0f}%: " + " | ".join(f"{{{','.join(k)}}}x{v}" for k, v in top))
    single = sum(1 for n, c in tr[0].items() if len(c) == 1)
    print(f"   tool chi co 1 kieu tap key: {single}/{len(tr[0])}")
    # D
    print("D. ask TRAIN:", dict(tr[4]), "| DEV:", dict(dv[4]))
    out["ask"] = dict(tr[4])
    json.dump(out, open(ROOT / "artifacts" / "rulebook.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
