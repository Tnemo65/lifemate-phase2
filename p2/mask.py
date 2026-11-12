"""Hammer-style function masking (arXiv 2410.04587) cho du lieu SFT.

Voi 1 mau duoc chon: doi ten app, tool va tham so (chi tham so CO description, de model van con thong tin)
thanh chuoi gia-tu ngau nhien, NHAT QUAN giua schema, history, target va sys.ask_missing_info.
Muc dich: model phai doc schema/description thay vi hoc thuoc ten -> tong quat tot hon voi app la.
sys.ask_missing_info giu nguyen ten (tool he thong co dinh), chi doi gia tri tool_name/params ben trong.
"""
from __future__ import annotations

import copy
import random
import re

_C = "bdghklmnprstv"
_V = "aeiou"
ASK = "sys.ask_missing_info"


def _word(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_C) + rng.choice(_V) for _ in range(n))


def generic_descriptions(tools_index: dict) -> set:
    """Description dung chung cho >= 2 ten tham so khac nhau (vd 'So nguyen, ... vi du 2' cho brightness,
    floor, speed...) -> khong mo ta y nghia; tham so mang description nay KHONG doi ten."""
    from collections import defaultdict
    d = defaultdict(set)
    for t in tools_index.values():
        for k, v in (t.get("parameters") or {}).get("properties", {}).items():
            if v.get("description"):
                d[v["description"]].add(k)
    return {desc for desc, ks in d.items() if len(ks) >= 2}


def build_maps(tools: list[dict], rng: random.Random, generic: set = frozenset()):
    used = set()

    def fresh(parts):
        """Ten moi duy nhat trong mau; parts = so am tiet moi phan (vd (2, 2) -> 'bavo_kimu').
        Khong gian ~65^4 / 65^3 nen luon tim duoc; gioi han vong lap de khong bao gio treo."""
        for _ in range(10000):
            w = "_".join(_word(rng, n) for n in parts)
            if w not in used:
                used.add(w)
                return w
        raise RuntimeError("khong sinh duoc ten moi")

    app_map, tool_map, param_map = {}, {}, {}
    for t in tools:
        name = t["function"]["name"]
        if name == ASK:
            continue
        app, fn = name.split(".", 1)
        if app not in app_map:
            app_map[app] = fresh((2, 2))
        tool_map[name] = f"{app_map[app]}.{fresh((2, 2))}"
        pm = {}
        for k, v in (t["function"].get("parameters") or {}).get("properties", {}).items():
            if v.get("description") and v["description"] not in generic:
                pm[k] = fresh((2, 1))  # trong 1 tool ten tham so phai khac nhau; dung chung 'used' cho chac
        param_map[name] = pm
    return app_map, tool_map, param_map


def _sub_text(s: str, tool_map: dict) -> str:
    if not tool_map:
        return s
    pat = re.compile("|".join(re.escape(k) for k in sorted(tool_map, key=len, reverse=True)))
    return pat.sub(lambda m: tool_map[m.group(0)], s)


def mask_tools(tools: list[dict], app_map, tool_map, param_map) -> list[dict]:
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        f = t["function"]
        name = f["name"]
        if name == ASK:
            out.append(t)
            continue
        pm = param_map[name]
        f["name"] = tool_map[name]
        f["description"] = _sub_text(f.get("description", ""), tool_map)
        p = f.get("parameters") or {}
        props = {}
        for k, v in p.get("properties", {}).items():
            v = dict(v)
            if "description" in v:
                v["description"] = _sub_text(v["description"], tool_map)
            if "x-ref" in v:
                a, rest = v["x-ref"].split(".", 1) if "." in v["x-ref"] else (v["x-ref"], "")
                v["x-ref"] = f"{app_map.get(a, a)}.{rest}" if rest else app_map.get(a, a)
            props[pm.get(k, k)] = v
        if "properties" in p:
            p["properties"] = props
        if "required" in p:
            p["required"] = [pm.get(k, k) for k in p["required"]]
        if "x-one-of" in p:
            p["x-one-of"] = [[pm.get(k, k) for k in g] for g in p["x-one-of"]]
        out.append(t)
    return out


def _mask_call(fn: dict, tool_map, param_map) -> dict:
    name, args = fn["name"], fn.get("arguments") or {}
    if name == ASK:
        tn = args.get("tool_name", "")
        pm = param_map.get(tn, {})
        new = dict(args)
        if "tool_name" in args:
            new["tool_name"] = tool_map.get(tn, tn)
        if isinstance(args.get("params"), list):
            new["params"] = [pm.get(x, x) for x in args["params"]]
        return {"name": name, "arguments": new}
    pm = param_map.get(name, {})
    return {"name": tool_map.get(name, name), "arguments": {pm.get(k, k): v for k, v in args.items()}}


def mask_messages(msgs: list[dict], tool_map, param_map) -> list[dict]:
    out = []
    for m in msgs:
        m = copy.deepcopy(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": _mask_call(tc["function"], tool_map, param_map)}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def mask_sample(msgs: list[dict], tools: list[dict], seed: int, generic: set = frozenset()):
    rng = random.Random(seed)
    app_map, tool_map, param_map = build_maps(tools, rng, generic)
    return mask_messages(msgs, tool_map, param_map), mask_tools(tools, app_map, tool_map, param_map), tool_map
