"""Cham diem local theo cong thuc BTC.

    gold rong      -> 1 neu bai nop rong, nguoc lai 0
    hash sai       -> 0
    nguoc lai      -> F1 = 2PR/(P+R), P = M/(M+U), R = M/|G|
    Final = 100/N * sum
M = call khop (cung ten, tham so non-fuzzy giong het, fuzzy token-F1 > 0.6), U = call nop khong khop
(ke ca call bi tu choi do sai schema). Khong xet thu tu.
"""
from __future__ import annotations

import unicodedata


def token_f1(a, b) -> float:
    ta = unicodedata.normalize("NFC", str(a)).lower().split()
    tb = unicodedata.normalize("NFC", str(b)).lower().split()
    if not ta or not tb:
        return float(ta == tb)
    rem, common = list(tb), 0
    for t in ta:
        if t in rem:
            rem.remove(t)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def fuzzy_keys(name: str, tools: dict) -> set:
    props = (tools.get(name) or {}).get("parameters", {}).get("properties", {})
    return {k for k, v in props.items() if v.get("x-match") == "fuzzy"}


def call_match(sub: dict, gold: dict, tools: dict) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if set(sa) != set(ga):
        return False
    fk = fuzzy_keys(gold["name"], tools)
    for k, gv in ga.items():
        if k in fk:
            if token_f1(sa[k], gv) <= 0.6:
                return False
        elif sa[k] != gv:
            return False
    return True


def sample_score(sub_calls: list, gold_calls: list, hash_ok: bool, tools: dict) -> float:
    if not gold_calls:
        return 1.0 if not sub_calls else 0.0
    if not hash_ok:
        return 0.0
    used, m = set(), 0
    for s in sub_calls:
        for j, g in enumerate(gold_calls):
            if j not in used and call_match(s, g, tools):
                used.add(j)
                m += 1
                break
    if not m:
        return 0.0
    p, r = m / len(sub_calls), m / len(gold_calls)
    return 2 * p * r / (p + r)


def score(preds: dict, items: dict, gold_hashes: dict, tools: dict) -> dict:
    """preds: {id: {"tool_calls", "state_hash"}}; items: {id: train item}."""
    per = {}
    for i, it in items.items():
        p = preds.get(i, {"tool_calls": [], "state_hash": ""})
        per[i] = sample_score(p["tool_calls"], it["label"]["tool_calls"],
                              p["state_hash"] == gold_hashes[i], tools)
    total = 100.0 * sum(per.values()) / max(len(per), 1)
    return {"score": round(total, 2), "n": len(per), "per_sample": per}
