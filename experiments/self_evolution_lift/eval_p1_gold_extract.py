# -*- coding: utf-8 -*-
"""Gold extraction: GLM-5.2 reads a survey's markdown and emits the method-
evolution gold graph (INDEPENDENT of extract_hypergraph — no circularity).

GLM-5.2 is a REASONING model: a single 98k-char prompt exhausts max_tokens on
reasoning and returns empty content. So we split the survey into ~6k-char
windows, extract a LOCAL gold per window (reusing already-known method ids),
then merge + dedup.
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from granular_agent.llm_client import parse_json_response, ENV, _CTX, call_llm, call_paratera

SURVEYS = {
    "ARFM2024": "2024ARFM综述_致密颗粒介质建模进展.md",
    "GRA2008":  "2008综述_致密干颗粒流流动.md",
    "THORNTON2026": "Thornton2026_GranularSegregation.md",
    "SUSP2018": "2018综述-致密颗粒状悬浮液的流变学研究.md",
}
SURVEY_DIR = os.path.join(os.path.dirname(__file__), "pilot_surveys")
OUT_DIR = os.path.join(os.path.dirname(__file__), "gold")
GOLD_MODEL = "GLM-5.2"
WINDOW = 6000   # chars per window (~1500 tokens prompt body)
STRIDE = 4500   # overlap so cross-window relations survive
# 可被 CLI --window/--stride 覆盖 (Thornton2026 dense, reasoning易爆token, 调小)


def _call_gold_llm(prompt, model=GOLD_MODEL, max_tokens=16384, timeout=240):
    # 非reasoning模型: 不爆token, 用于dense综述(Thornton). GLM-4-Flash直接urlopen单次长timeout(180s)
    if model == "GLM-4-Flash":
        import json as _json, urllib.request as u
        key = ENV.get("PARATERA_API_KEY")
        base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
        mt = min(max_tokens, 4000)
        body = _json.dumps({"model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0, "max_tokens": mt}).encode()
        req = u.Request(base + "/chat/completions", data=body,
                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            raw = u.urlopen(req, context=_CTX, timeout=180).read()
            return _json.loads(raw)["choices"][0]["message"].get("content")
        except Exception as e:
            print("  [gold-flash] err: %s" % str(e)[:100], flush=True)
            return None
    if model == "deepseek-chat":
        # deepseek非reasoning不爆token, 直接urlopen单次120s不走fallback(避免fallback到reasoning爆token)
        import json as _json, urllib.request as u
        key = ENV.get("DEEPSEEK_API_KEY")
        if not key:
            return None
        mt = min(max_tokens, 6000)
        body = _json.dumps({"model": "deepseek-chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0, "max_tokens": mt}).encode()
        req = u.Request("https://api.deepseek.com/v1/chat/completions", data=body,
                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        for _ in range(2):
            try:
                raw = u.urlopen(req, context=_CTX, timeout=120).read()
                return _json.loads(raw)["choices"][0]["message"].get("content")
            except Exception as e:
                print("  [gold-ds] err: %s" % str(e)[:100], flush=True)
        return None
    if model == "Kimi-K2.6":
        return call_paratera(prompt, model=model, max_tokens=min(max_tokens, 6000))
    import json as _json, urllib.request as u
    key = ENV.get("PARATERA_API_KEY")
    base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
    if not key:
        return None
    mt = max_tokens
    for attempt in range(2):
        body = _json.dumps({"model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0, "max_tokens": mt}).encode()
        req = u.Request(base + "/chat/completions", data=body,
                        headers={"Authorization": "Bearer " + key,
                                 "Content-Type": "application/json"})
        try:
            raw = u.urlopen(req, context=_CTX, timeout=timeout).read()
            d = _json.loads(raw)
            c = d["choices"][0]["message"].get("content")
            if c:
                return c
            rt = d.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
            fin = d["choices"][0].get("finish_reason")
            print("  [gold] empty content reasoning_tokens=%s finish=%s (mt=%d)" % (
                rt, fin, mt), flush=True)
            if fin == "length":
                mt = min(mt * 2, 28672)   # reasoning ate the budget; give more
        except Exception as e:
            print("  [gold] attempt %d err: %s" % (attempt+1, str(e)[:140]), flush=True)
    return None


LOCAL_PROMPT = """You are a granular-flow physics expert. Extract a LOCAL method-evolution knowledge graph from THIS passage of a scientific review (passage is one window of a larger review). Extract only what THIS passage explicitly states.

Already-known methods (reuse these exact ids if the passage refers to them; otherwise mint new M-numbers continuing from M{next}):
{known}

OUTPUT STRICT JSON (no prose, no fence):
{{
  "methods": [
    {{"id": "M{n}", "name": "<canonical method/model/theory/law name>",
      "aliases": ["<alt names/symbols/abbreviations>"],
      "refs": ["<citation keys as written, e.g. MiDi 2004>"]}}
  ],
  "evolution_edges": [
    {{"src": "<id>", "tgt": "<id>", "type": "extends|improves|replaces|adapts|compares|background",
      "evidence": "<verbatim fragment from THIS passage>"}}
  ],
  "composition_edges": [
    {{"whole": "<id>", "component": "<id>", "evidence": "<verbatim fragment>"}}
  ],
  "law_constraints": [
    {{"law_text": "<equation/law as text>", "constrains": "<id>", "consumer": "<id or null>",
      "evidence": "<verbatim fragment>"}}
  ]
}}

RULES: only explicit relations from THIS passage; each edge needs verbatim evidence; methods are named scientific entities (not generic nouns); give aliases liberally. Output ONLY JSON.

PASSAGE:
<<<
{md}
>>>
"""


def windows(md):
    out = []
    n = len(md)
    if n <= WINDOW:
        return [(0, md)]
    i = 0
    while i < n:
        out.append((i, md[i:i+WINDOW]))
        if i + WINDOW >= n:
            break
        i += STRIDE
    return out


def merge_gold(parts):
    """Merge window-local golds: union methods (dedup by name/alias),
    re-id consistently, remap edges."""
    methods = {}      # norm(name) -> method dict
    next_id = [1]
    def norm(s):
        return re.sub(r"\s+", " ", s.lower().strip())
    def get_or_add(name, aliases=None, refs=None):
        key = norm(name)
        if key in methods:
            m = methods[key]
            if aliases:
                for a in aliases:
                    if a and a not in m["aliases"]:
                        m["aliases"].append(a)
            if refs:
                for r in refs:
                    if r and r not in m["refs"]:
                        m["refs"].append(r)
            return m["id"]
        mid = "M%d" % next_id[0]; next_id[0] += 1
        methods[key] = {"id": mid, "name": name,
                        "aliases": list(aliases or []), "refs": list(refs or [])}
        return mid
    # local id -> global id, per part
    evo, comp, law = [], [], []
    for p in parts:
        lm = p.get("methods", [])
        lmap = {}
        for m in lm:
            nm = m.get("name", "").strip()
            if not nm:
                continue
            gid = get_or_add(nm, m.get("aliases"), m.get("refs"))
            lmap[m.get("id")] = gid
        def remap(x):
            return lmap.get(x)
        for e in p.get("evolution_edges", []):
            s, t = remap(e.get("src")), remap(e.get("tgt"))
            if s and t and s != t:
                evo.append({"src": s, "tgt": t, "type": e.get("type","background"),
                            "evidence": e.get("evidence","")})
        for c in p.get("composition_edges", []):
            w, co = remap(c.get("whole")), remap(c.get("component"))
            if w and co and w != co:
                comp.append({"whole": w, "component": co, "evidence": c.get("evidence","")})
        for l in p.get("law_constraints", []):
            law.append({"law_text": l.get("law_text",""),
                        "constrains": remap(l.get("constrains")),
                        "consumer": remap(l.get("consumer")),
                        "evidence": l.get("evidence","")})
    return {"methods": list(methods.values()), "evolution_edges": evo,
            "composition_edges": comp, "law_constraints": law}


def extract_gold(survey_label, window=None, stride=None, model=None):
    global WINDOW, STRIDE
    if window:
        WINDOW = window
    if stride is None and window:
        stride = int(window * 0.75)
    if stride:
        STRIDE = stride
    global GOLD_MODEL
    if model:
        GOLD_MODEL = model
    md_path = os.path.join(SURVEY_DIR, SURVEYS[survey_label])
    md = open(md_path, encoding="utf-8", errors="replace").read()
    md = "\n".join(l for l in md.splitlines() if not l.lstrip().startswith("!["))
    wins = windows(md)
    print("[%s] %d chars -> %d windows" % (survey_label, len(md), len(wins)), flush=True)
    parts = []
    known_methods = []
    for wi, (off, w) in enumerate(wins):
        next_n = len(known_methods) + 1
        known = "\n".join("- %s: %s" % (m["id"], m["name"]) for m in known_methods) or "(none yet)"
        prompt = LOCAL_PROMPT.format(known=known, next=next_n, n=next_n, md=w)
        raw = _call_gold_llm(prompt, max_tokens=28672, timeout=180)
        if not raw:
            print("  [w%d] empty, skip" % wi, flush=True)
            continue
        p = parse_json_response(raw)
        if not p:
            open(os.path.join(OUT_DIR, "%s_w%d_raw.txt" % (survey_label, wi)), "w", encoding="utf-8").write(raw)
            print("  [w%d] parse fail, raw saved" % wi, flush=True)
            continue
        parts.append(p)
        # register known method names for cross-window reuse
        for m in p.get("methods", []):
            nm = m.get("name", "").strip()
            if nm and not any(norm_eq(k, nm) for k in known_methods):
                known_methods.append({"id": m.get("id","?"), "name": nm})
        print("  [w%d] methods=%d evo=%d comp=%d law=%d (known total %d)" % (
            wi, len(p.get("methods",[])), len(p.get("evolution_edges",[])),
            len(p.get("composition_edges",[])), len(p.get("law_constraints",[])),
            len(known_methods)), flush=True)
    gold = merge_gold(parts)
    print("[%s] MERGED: methods=%d evo=%d comp=%d law=%d" % (
        survey_label, len(gold["methods"]), len(gold["evolution_edges"]),
        len(gold["composition_edges"]), len(gold["law_constraints"])), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(gold, open(os.path.join(OUT_DIR, survey_label + "_gold.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(parts, open(os.path.join(OUT_DIR, survey_label + "_parts.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return gold


def norm_eq(m, name):
    return norm(m["name"]) == norm(name)

def norm(s):
    return re.sub(r"\s+", " ", s.lower().strip())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--window", type=int, default=None, help="覆盖默认window字符数(dense综述调小防爆token)")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--model", default=None, help="抽取模型(默认GLM-5.2 reasoning慢; Thornton dense用GLM-5-Turbo快)")
    a = ap.parse_args()
    extract_gold(a.survey, window=a.window, stride=a.stride, model=a.model)
