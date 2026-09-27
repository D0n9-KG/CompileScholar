# -*- coding: utf-8 -*-
"""P0-1 修复：cards 视图增量重建（CS2 registry 无 in_corpus_paper_id 的补丁）。

病根（REPAIR-WAVE-0928 P0-1）：build_cards 的 gate 要求
e['in_corpus_paper_id']——Multi 时代的字段（深抽注册实体自带）。CS2 的
registry 从粗抽 mentions 生长（provenance=round2_growth），15100 实体
该字段全 None，等效判据是 mention_papers 非空 → gate 全跳过 → cards
视图空壳 → card() 工具 13 批全域死亡 + F31 A-block 三来源之一死亡。

修法：增量补丁——用 build_cards 的确定性部分（config/result/findings/
lineage 聚合，跳过需要 LLM 的 delta 解析，deltas_explicit 保留记录自带
的）对 views_cs2.json 的 cards 键重建。gate 换成：mention_papers 非空
（CS2 形态）或 in_corpus_paper_id 非空（Multi 形态，向后兼容）。
paper_id/title 取 manifest 里 mention_papers 首篇的近似映射（CS2
mention_papers 是标题列表）。
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.views.compiler import _norm, _ref_name, round_robin_by_paper

CS2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(CS2, "base_kb")


def main():
    views = json.load(open(os.path.join(BASE, "views_cs2.json"),
                           encoding="utf-8"))
    registry = json.load(open(os.path.join(BASE, "registry_v2.json"),
                              encoding="utf-8"))
    manifest = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE, "manifest_all.json"), encoding="utf-8"))}
    records = json.load(open(os.path.join(BASE, "records_merged.json"),
                             encoding="utf-8"))
    deep = json.load(open(os.path.join(BASE, "deep_read_records.json"),
                          encoding="utf-8"))
    for pid, p in deep.items():
        base = (records.get(pid) or {}).get("records") or []
        seen = {r.get("id") for r in base if r.get("id")}
        records[pid] = {"records": base + [
            r for r in p.get("records", []) if r.get("id") not in seen]}

    # title → paper_id 近似映射（mention_papers 是标题列表）
    t2pid = {}
    for pid, m in manifest.items():
        t = re.sub(r"[^a-z0-9]+", " ", str(m.get("title") or "").lower()).strip()
        if t:
            t2pid[t[:50]] = pid

    recs = [r for payload in records.values() for r in payload.get("records", [])]
    print(f"records: {len(recs)}", flush=True)

    # by_ent 聚合（与 build_cards 同逻辑）
    surfaces = {}
    for e in registry.get("entities", []):
        surfaces[_norm(e["canonical"])] = _norm(e["canonical"])
        for a in e.get("aliases", []):
            surfaces[_norm(a)] = _norm(e["canonical"])
    mention_pat = re.compile(r"\b(?:" + "|".join(
        sorted((re.escape(s) for s in surfaces), key=len, reverse=True)) + r")\b") \
        if surfaces else None
    by_ent = defaultdict(list)
    for r in recs:
        attached = False
        for f in ("method_ref", "from_method_ref", "scope_ref_ref"):
            name = _ref_name(r.get(f))
            if name:
                by_ent[_norm(name)].append(r)
                attached = True
                break
        if r.get("kind") == "finding":
            if not attached:
                tname = _ref_name(r.get("target_ref_ref"))
                if tname:
                    by_ent[_norm(tname)].append(r)
            if mention_pat:
                for mm in {_norm(x) for x in
                           mention_pat.findall(str(r.get("claim") or ""))}:
                    canon = surfaces.get(mm)
                    if canon:
                        by_ent[canon].append(r)

    cards = {}
    for e in registry.get("entities", []):
        # gate 修复：CS2 形态（mention_papers）或 Multi 形态（paper_id）
        pid = e.get("in_corpus_paper_id") or None
        if not pid:
            mps = e.get("mention_papers") or []
            if not mps:
                continue   # 无任何提及来源的实体（纯 registry 生长）
            for mp in mps:
                t = re.sub(r"[^a-z0-9]+", " ", str(mp).lower()).strip()
                pid = t2pid.get(t[:50]) or t2pid.get(t[:35])
                if pid:
                    break
            # anchor 解析失败→pid=None：dossier 仍建（记录聚合是主体，
            # 元数据缺失只影响 title/year 展示——mention_papers 常引用
            # 粗抽 corpus 里的论文，其中 62% 不在 manifest 是常态）
        m = (manifest.get(pid) if pid else None) or {}
        rs = by_ent.get(_norm(e["canonical"]), []) + \
            [r for a in e.get("aliases", []) for r in by_ent.get(_norm(a), [])]
        seen_ids, uniq = set(), []
        for r in rs:
            if r.get("id") not in seen_ids:
                seen_ids.add(r.get("id"))
                uniq.append(r)
        if not uniq:
            continue   # 无记录聚合的实体不做空卡片
        configs = [r for r in uniq if r.get("kind") == "config"]
        results = [r for r in uniq if r.get("kind") == "result"]
        findings_all = [r for r in uniq if r.get("kind") == "finding"]
        enames = {_norm(e["canonical"]),
                  *(_norm(a) for a in e.get("aliases", []))}
        direct, rest = [], []
        for r in findings_all:
            cn = _norm(r.get("claim"))
            (direct if any(n and n in cn for n in enames) else rest).append(r)
        findings = round_robin_by_paper(direct) + round_robin_by_paper(rest)
        lin_out = [r for r in uniq if r.get("kind") == "lineage"
                   and _norm(_ref_name(r.get("from_method_ref"))) in enames]
        deltas_explicit = [
            {"record_id": r.get("id"), "delta": r.get("delta"),
             "metric": (r.get("measure") or {}).get("metric"),
             "quote": (r.get("quote") or "")[:120]}
            for r in results if r.get("delta")]
        cards[e["entity_id"]] = {
            "canonical": e["canonical"], "aliases": e.get("aliases", []),
            "paper_id": pid, "title": m.get("title"),
            "year": m.get("year"),
            "configs": [{"item": r.get("item"), "value": r.get("value"),
                         "role": r.get("role"), "record_id": r.get("id"),
                         "quote": (r.get("quote") or "")[:140]}
                        for r in configs[:40]],
            "main_results": [{"subject": (r.get("dims") or {}).get("subject"),
                              "metric": (r.get("measure") or {}).get("metric"),
                              "value": (r.get("measure") or {}).get("value"),
                              "role": r.get("role"),
                              "epistemic": r.get("epistemic"),
                              "record_id": r.get("id")}
                             for r in results
                             if r.get("role") != "ablation"][:40],
            "ablations": [{"variant": (r.get("dims") or {}).get("variant"),
                           "delta": r.get("delta"),
                           "value": (r.get("measure") or {}).get("value"),
                           "record_id": r.get("id")}
                          for r in results
                          if r.get("role") == "ablation"][:30],
            "findings": [{"claim": r.get("claim"),
                          "claim_type": r.get("claim_type"),
                          "strength": r.get("strength"),
                          "epistemic": r.get("epistemic"),
                          "paper_id": r.get("paper_id"),
                          "record_id": r.get("id")}
                         for r in findings[:40]],
            "lineage_out": [{"relation": r.get("relation"),
                             "to": _ref_name(r.get("to_method_ref")),
                             "record_id": r.get("id")} for r in lin_out[:30]],
            "deltas_explicit": deltas_explicit[:30],
            "provenance": "compiled+patched-0928",
        }
    n_anchored = sum(1 for c in cards.values() if c.get("paper_id"))
    print(f"cards rebuilt: {len(cards)} dossiers "
          f"({n_anchored} anchored, {len(cards)-n_anchored} no-anchor)",
          flush=True)
    # 写回（原子）
    views["cards"] = cards
    tmp = os.path.join(BASE, "views_cs2.json.tmp")
    json.dump(views, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(BASE, "views_cs2.json"))
    print(f"views_cs2.json updated ({os.path.getsize(os.path.join(BASE, 'views_cs2.json'))//1024//1024}MB)",
          flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
