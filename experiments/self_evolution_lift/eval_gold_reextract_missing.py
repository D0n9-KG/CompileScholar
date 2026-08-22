# -*- coding: utf-8 -*-
"""补抽 ARFM gold 失败窗口 (deepseek hang 导致的缺失).
复用 gold_extract 逻辑, 只跑指定窗口, deepseek 跑3次都hang换GLM-4-Flash兜底.
成功后追加进 parts.json, 再 merge 成最终 gold.
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gold_extract import LOCAL_PROMPT, windows, merge_gold, _call_gold_llm, SURVEY_DIR, SURVEYS, OUT_DIR
from granular_agent.llm_client import parse_json_response

SURVEY = "ARFM2024"
MISSING = [2, 3, 9, 10, 13, 14, 15, 19, 20, 21, 23, 28, 29]


def call_paratera_flash(prompt, max_tokens=6000):
    """GLM-4-Flash 兜底 (deepseek hang 时)."""
    from granular_agent.llm_client import call_paratera
    return call_paratera(prompt, model="GLM-4-Flash", max_tokens=max_tokens)


def reextract():
    md_path = os.path.join(SURVEY_DIR, SURVEYS[SURVEY])
    md = open(md_path, encoding="utf-8", errors="replace").read()
    md = "\n".join(l for l in md.splitlines() if not l.lstrip().startswith("!"))
    wins = windows(md)
    # 加载已有parts + known_methods
    parts_path = os.path.join(OUT_DIR, SURVEY + "_parts.json")
    parts = json.load(open(parts_path, encoding="utf-8"))
    # 重建 known_methods (从已有parts累积, 按窗口顺序)
    import re as _re
    def norm(s): return _re.sub(r"\s+", " ", s.lower().strip())
    known_methods = []
    seen = set()
    for p in parts:
        for m in p.get("methods", []):
            nm = m.get("name", "").strip()
            k = norm(nm)
            if nm and k not in seen:
                seen.add(k)
                known_methods.append({"id": m.get("id", "?"), "name": nm})
    print(f"已有parts: {len(parts)} 窗口, known {len(known_methods)} 方法")
    print(f"补抽 {len(MISSING)} 失败窗口: {MISSING}")

    new_parts = []
    for wi in MISSING:
        if wi >= len(wins):
            continue
        w = wins[wi][1]
        next_n = len(known_methods) + 1
        known = "\n".join("- %s: %s" % (m["id"], m["name"]) for m in known_methods) or "(none yet)"
        prompt = LOCAL_PROMPT.format(known=known, next=next_n, n=next_n, md=w)
        # deepseek 跑3次
        p = None
        for attempt in range(3):
            raw = _call_gold_llm(prompt, model="deepseek-chat", max_tokens=6000, timeout=120)
            if raw:
                p = parse_json_response(raw)
                if p and (p.get("methods") or p.get("evolution_edges")):
                    break
                p = None
        # GLM-4-Flash 兜底
        if not p:
            print(f"  [w{wi}] deepseek 3次失败, 换GLM-4-Flash", flush=True)
            raw = call_paratera_flash(prompt, max_tokens=4000)
            if raw:
                p = parse_json_response(raw)
        if p and (p.get("methods") or p.get("evolution_edges")):
            new_parts.append(p)
            for m in p.get("methods", []):
                nm = m.get("name", "").strip()
                k = norm(nm)
                if nm and k not in seen:
                    seen.add(k)
                    known_methods.append({"id": m.get("id", "?"), "name": nm})
            print(f"  [w{wi}] OK methods={len(p.get('methods',[]))} evo={len(p.get('evolution_edges',[]))} comp={len(p.get('composition_edges',[]))} law={len(p.get('law_constraints',[]))}", flush=True)
        else:
            print(f"  [w{wi}] FAIL 跳过", flush=True)

    # 追加进 parts, merge
    all_parts = parts + new_parts
    # 去重 (按方法名)
    gold = merge_gold(all_parts)
    print(f"\nMERGED: methods={len(gold['methods'])} evo={len(gold['evolution_edges'])} comp={len(gold['composition_edges'])} law={len(gold['law_constraints'])}")
    # 去重复边 (merge_gold 没去重 evo)
    seen_e = set(); dedup_evo = []
    for e in gold["evolution_edges"]:
        k = (e["src"], e["tgt"], e["type"], e.get("evidence", "")[:60])
        if k not in seen_e:
            seen_e.add(k); dedup_evo.append(e)
    gold["evolution_edges"] = dedup_evo
    print(f"去重后 evo={len(dedup_evo)}")
    json.dump(gold, open(os.path.join(OUT_DIR, SURVEY + "_gold.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(all_parts, open(parts_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved gold + parts")


if __name__ == "__main__":
    reextract()
