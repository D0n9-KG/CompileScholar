# -*- coding: utf-8 -*-
"""跨论文结构五件套·组件3:转述四分类(中性,LLM 只判关系不裁决对错)。

用户裁定(10-01 讨论):
- 四分类 restate/extend/qualify/dispute——转述者站未来视角,不一致≠错,
  不判谁对谁错,双方原文都保留
- 数值冲突不做(gold 仅 0.5% 要数值+原料稀缺+高误报)
- 顺序最后:桥建肥再判

实测天花板(诚实记录):survey_claim 引用形态 1,820 → 引用桥可解析 175
→ 原文论文有主题相关记录 18 对。瓶颈=语料-引文重合度(语料是需求驱动
非引文驱动建库),不是分类器。管线建通,生长循环扩语料后自动变肥。

用法:PYTHONUTF8=1 python classify_paraphrases.py [--dry]
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
sys.path.insert(0, SRC)
from kb_infra.llm import call_paratera  # noqa: E402

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
MODEL = "DeepSeek-V4.1-Flash"

_norm_re = re.compile(r"[^a-z0-9]+")


def toks(s):
    return set(t for t in _norm_re.split(str(s or "").lower()) if len(t) > 2)


def load_records():
    merged = json.load(open(os.path.join(BASE_KB, "records_merged.json"),
                            encoding="utf-8"))
    deep = json.load(open(os.path.join(BASE_KB, "deep_read_records.json"),
                          encoding="utf-8"))
    allrecs = {}
    for pid, payload in merged.items():
        if isinstance(payload, dict):
            allrecs[pid] = payload.get("records") or []
    for pid, payload in deep.items():
        if isinstance(payload, dict):
            base_recs = allrecs.get(pid) or []
            seen = {r.get("id") for r in base_recs}
            allrecs[pid] = base_recs + [
                r for r in payload.get("records", [])
                if r.get("id") not in seen]
    return merged, allrecs


def build_pairs(merged, allrecs):
    pairs = []
    for pid, payload in merged.items():
        if not isinstance(payload, dict):
            continue
        for r in payload.get("records") or []:
            if r.get("kind") != "survey_claim" or not r.get("about_paper_id"):
                continue
            if r.get("paraphrase_rel"):
                continue   # 幂等
            tgt = r["about_paper_id"]
            if tgt not in allrecs or tgt == pid:
                continue
            stoks = toks(r.get("claim")) | toks(r.get("claims_about"))
            scored = []
            for o in allrecs[tgt]:
                if not isinstance(o, dict) or o.get("kind") not in (
                        "finding", "method", "result", "config",
                        "limitation", "absence", "lineage"):
                    continue
                ov = len(stoks & toks(o.get("claim"))) + \
                    len(stoks & toks(o.get("subject")))
                if ov >= 3:
                    scored.append((ov, o))
            scored.sort(key=lambda x: -x[0])
            if scored:
                pairs.append((r, tgt, scored[:3]))
    return pairs


def _classify_one(r, tgt, cands):
    orig_ctx = "\n".join(
        f"- [{o.get('id')}] ({o.get('kind')}) {str(o.get('claim'))[:300]}"
        for _, o in cands)
    prompt = f"""A SURVEY paper paraphrases/reports on work from an ORIGINAL paper. Classify the RELATIONSHIP of the survey's statement to the original paper's own records. Do NOT judge who is right — the survey may have later knowledge; inconsistency is not an error.

SURVEY statement (about the original paper):
  "{str(r.get('claim'))[:400]}"

ORIGINAL paper's own records:
{orig_ctx}

Classes (pick exactly one):
- restate: the survey statement conveys the same content as the original records
- extend: the survey adds information/interpretation beyond the original records (later perspective)
- qualify: the survey adds a condition, evaluation, or third-party judgment about scope/limits
- dispute: the survey statement contradicts the original records' content
- unrelated: the original records are about a different topic (pairing error)

Answer with exactly one label word on the first line, then one short sentence."""
    try:
        out = call_paratera(prompt, model=MODEL, max_tokens=200,
                            temperature=0.0, enable_thinking=False)
    except Exception as e:
        return "ERR", str(e)[:80]
    if not out:
        return "ERR", "no response"
    first = (out.strip().splitlines() or [""])[0].strip().lower()
    for lab in ("restate", "extend", "qualify", "dispute", "unrelated"):
        if first.startswith(lab):
            return lab, " ".join(out.strip().split())[:200]
    return "UNPARSED", " ".join(out.strip().split())[:200]


def main():
    dry = "--dry" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    merged, allrecs = load_records()
    pairs = build_pairs(merged, allrecs)
    print(f"[pair] 可配对转述-原文对: {len(pairs)}")
    if dry:
        for r, tgt, cands in pairs[:5]:
            print(f"  {r['id']} -> {tgt[:40]} "
                  f"(cands={len(cands)}) {str(r.get('claim'))[:60]}")
        print("[dry] 不分类")
        return
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_classify_one, r, tgt, cands)
                for r, tgt, cands in pairs]
        for (r, tgt, cands), fut in zip(pairs, futs):
            lab, why = fut.result()
            results.append((r, tgt, cands[0][1], lab, why))

    from collections import Counter
    dist = Counter(lab for _, _, _, lab, _ in results)
    print(f"[classify] 分布: {dict(dist)}")
    ok = 0
    for r, tgt, o, lab, why in results:
        if lab in ("restate", "extend", "qualify", "dispute"):
            r["paraphrase_rel"] = {
                "label": lab, "orig_record_id": o.get("id"),
                "orig_paper_id": tgt}
            ok += 1
    # 写回 records_merged(r 引用 merged 内对象,就地已改)
    src = os.path.join(BASE_KB, "records_merged.json")
    tmp = src + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    os.replace(tmp, src)
    led = os.path.join(BASE_KB, "paraphrase_rel_ledger.jsonl")
    with open(led, "w", encoding="utf-8") as fh:
        for r, tgt, o, lab, why in results:
            fh.write(json.dumps({
                "record_id": r.get("id"), "survey_paper": r.get("paper_id"),
                "orig_paper": tgt, "orig_record": o.get("id"),
                "label": lab, "why": why}, ensure_ascii=False) + "\n")
    print(f"[write] paraphrase_rel 回填 {ok} 条 + ledger {len(results)} 行")
    for r, tgt, o, lab, why in results[:10]:
        print(f"  {lab:9s} {str(r.get('claim'))[:60]} -> {tgt[:30]}")


if __name__ == "__main__":
    main()
