# -*- coding: utf-8 -*-
"""P0-2（FIX-PLAN v2）：absence 语义修复——缺口生灭链（resolved_by）。

问题：absence 记录是静态的（"X lacks Y"永远为真），但 KB 自己的谱系里
就有 improves/replaces 边证明后来者可能解决了该缺口——缺口没有生命周期。

两段式（规则管结构，LLM 管语义——"复杂语义不用规则"铁律）：
  ① 结构配对（确定性）：absence.subject 与 improves/replaces 边的
     to_name 解析到同一 registry 实体 + 时间合法（缺口宣称年 ≤ 解决
     论文年）→ resolution_candidates 候选链。候选链是事实（两条记录
     存在且时序合法），不作语义断言。
     实测：实体级 227 对，时间约束后 165 对（62 对宣称晚于"解决"被滤）。
  ② LLM 核验（一次性离线，DeepSeek-V4.1-Flash）：每对问"改进工作是否
     针对缺口的具体内容"→ YES 才晋升 resolved_by（保留完整证据链：
     entity/relation/paper_id/quote/verified_at）。UNCERTAIN/NO 留在
     候选层。
     实测动机：纯结构配对抽样 12 对语义精度仅 ~40%（"LLM-大脑同步机制
     未知"配"CoT improves LLM"——改进≠解决该缺口）。

写回 views_cs2.json coverage.absences_extracted（git 在案可回滚）。
幂等：candidates 按 (paper_id, from_name, relation) 去重；已 resolved_by
的不重验。

用法：
  PYTHONUTF8=1 python resolve_absences.py pair     # 仅结构配对（零 API）
  PYTHONUTF8=1 python resolve_absences.py verify   # LLM 核验未验候选
  PYTHONUTF8=1 python resolve_absences.py verify 8 # 并发路数
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
sys.path.insert(0, SRC)

from kb_infra.llm import call_paratera  # noqa: E402

VIEWS = os.path.join(BASE_KB, "views_cs2.json")
VERIFY_MODEL = "DeepSeek-V4.1-Flash"


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _paper_year(pid, man, man_all):
    m = re.match(r"arxiv_(\d{2})\d{2}\.", pid or "")
    if m:
        return 2000 + int(m.group(1))
    for src in (man, man_all):
        if pid in src and src[pid].get("year"):
            try:
                return int(src[pid]["year"])
            except (TypeError, ValueError):
                return None
    return None


def _entity_index(registry):
    s2e = {}
    for surf, eid in (registry.get("surface_index") or {}).items():
        n = _norm(surf)
        if n:
            s2e.setdefault(n, set()).add(str(eid))

    def ent_id(name):
        ids = s2e.get(_norm(name))
        return next(iter(ids)) if ids and len(ids) == 1 else None
    return ent_id


def _atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def pair(verbose=True):
    """① 结构配对 → resolution_candidates 写回 views。"""
    views = json.load(open(VIEWS, encoding="utf-8"))
    cov = views.setdefault("coverage", {})
    absences = cov.get("absences_extracted") or []
    edges = (views.get("genealogy") or {}).get("edges") or []
    registry = json.load(open(os.path.join(BASE_KB, "registry_v2.json"),
                              encoding="utf-8"))
    man = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    man_all = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}
    ent_id = _entity_index(registry)

    ir = [e for e in edges if e.get("relation") in ("improves", "replaces")]
    to_ents = {}
    for e in ir:
        eid = ent_id(e.get("to_name"))
        if eid:
            to_ents.setdefault(eid, []).append(e)

    n_cand = n_new = 0
    for a in absences:
        a.pop("resolved_by", None) if False else None  # keep existing
        eid = ent_id(a.get("subject"))
        if not eid or eid not in to_ents:
            continue
        ay = _paper_year(a.get("paper_id"), man, man_all)
        cands = a.get("resolution_candidates") or []
        have = {(c.get("paper_id"), c.get("entity"), c.get("relation"))
                for c in cands}
        for e in to_ents[eid]:
            ry = _paper_year(e.get("paper_id"), man, man_all)
            if ay is not None and ry is not None and ry < ay:
                continue   # 宣称晚于"解决"——时序不合法
            key = (e.get("paper_id"), e.get("from_name"), e.get("relation"))
            if key in have:
                continue
            cands.append({
                "entity": e.get("from_name"), "relation": e.get("relation"),
                "paper_id": e.get("paper_id"), "year": ry,
                "quote": (e.get("quote") or "")[:200],
                "record_id": e.get("record_id")})
            n_new += 1
        if cands and cands is not (a.get("resolution_candidates") or []):
            a["resolution_candidates"] = cands
            n_cand += 1

    _atomic_write(views, VIEWS)
    if verbose:
        n_resolved = sum(1 for a in absences if a.get("resolved_by"))
        print(f"[pair] absences with candidates: {n_cand} / {len(absences)}"
              f"  new candidate links: {n_new}  already resolved_by: {n_resolved}")
    return views


def _verify_one(a, c):
    prompt = f"""A research knowledge base records a GAP and a later IMPROVEMENT claim. Decide whether the improvement plausibly addresses THE SPECIFIC gap described (not just any improvement to the field).

GAP (recorded by paper {a.get('paper_id')}, year {a.get('year_claimed') or '?'}):
  Subject: {a.get('subject')}
  Missing/lacking: {(a.get('missing') or '')[:400]}
  Context quote: {(a.get('quote') or '')[:400]}

IMPROVEMENT CLAIM (from paper {c.get('paper_id')}, year {c.get('year') or '?'}):
  "{c.get('entity')}" {c.get('relation')} the gap's subject.
  Supporting quote: {c.get('quote') or '(none)'}

Question: does the work on "{c.get('entity')}" plausibly address the SPECIFIC missing capability described in the gap? Consider the actual content of the gap text, not just the field.
Answer with exactly one word on the first line: YES, NO, or UNCERTAIN. Then one short sentence explaining why."""
    try:
        out = call_paratera(prompt, model=VERIFY_MODEL, max_tokens=200,
                            temperature=0.0, enable_thinking=False)
    except Exception as e:
        return "ERR", str(e)[:80]
    if not out:
        return "ERR", "no response"
    first = (out.strip().splitlines() or [""])[0].strip().upper()
    verdict = "YES" if first.startswith("YES") else (
        "NO" if first.startswith("NO") else (
            "UNCERTAIN" if first.startswith("UNCERT") else "UNPARSED"))
    return verdict, " ".join(out.strip().split())[:200]


def verify(n_par=6):
    """② LLM 核验未验候选 → resolved_by。"""
    views = json.load(open(VIEWS, encoding="utf-8"))
    absences = (views.get("coverage") or {}).get("absences_extracted") or []
    jobs = []
    for a in absences:
        if a.get("resolved_by"):
            continue
        for c in (a.get("resolution_candidates") or []):
            if not c.get("verified"):
                jobs.append((a, c))
    if not jobs:
        print("[verify] no unverified candidates")
        return
    print(f"[verify] {len(jobs)} candidate links to verify "
          f"({VERIFY_MODEL}, {n_par} parallel)")
    # 载体：answer 记录缺 year_claimed，给 prompt 用（从 paper_id 推不出
    # sciverse/ext 的年——manifest 查）
    man = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    man_all = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}
    for a, _ in jobs:
        if not a.get("year_claimed"):
            a["year_claimed"] = _paper_year(a.get("paper_id"), man, man_all)

    results = []
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futs = {ex.submit(_verify_one, a, c): (a, c) for a, c in jobs}
        for i, fut in enumerate(futs):
            a, c = futs[fut]
            verdict, why = fut.result()
            c["verified"] = verdict
            c["verify_note"] = why
            results.append((a, c, verdict))
            if (i + 1) % 20 == 0:
                print(f"  ...{i + 1}/{len(jobs)}", flush=True)

    n_yes = sum(1 for _, _, v in results if v == "YES")
    n_err = sum(1 for _, _, v in results if v == "ERR")
    for a, c, v in results:
        if v == "YES":
            # 最早核验通过的为 primary 生灭时刻；其余保留在链上
            rb = a.get("resolved_by") or []
            rb.append({"entity": c["entity"], "relation": c["relation"],
                       "paper_id": c["paper_id"], "year": c.get("year"),
                       "quote": c.get("quote"),
                       "verified_at": "2026-09-30"})
            rb.sort(key=lambda x: x.get("year") or 9999)
            a["resolved_by"] = rb
    _atomic_write(views, VIEWS)
    from collections import Counter
    print(f"[verify] verdicts: {dict(Counter(v for _, _, v in results))}")
    print(f"[verify] resolved_by promoted: {n_yes} links "
          f"({sum(1 for a in absences if a.get('resolved_by'))} absences now resolved)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    if mode == "pair":
        pair()
    elif mode == "verify":
        verify(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        print(__doc__)
