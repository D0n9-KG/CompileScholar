# -*- coding: utf-8 -*-
"""领域状态视图 v2（REBUILD-PLAN-1003 §3）：只保留"多篇论文合在一起才成立"、且有消费者的三类对象。

  families  ——方法族/主题节点：来自综述 taxonomy_node 快照（族名 + 族级性质：做法/优劣/适用条件），
               同名节点跨综述合并（规范化名），记录每条性质来自哪篇综述（逐字 quote）。
  limits    ——族级局限与开放问题：综述 challenges 主张 + absence（survey_claimed/explicitly_stated）缺口，
               挂到其 subject 对应的族（名字匹配；挂不上的独立成"未归族"条目，不丢）。
  compares  ——比较关系：综述 comparison 主张 + comparison_table 快照 + lineage（extends/improves/compares/replaces），
               关系词表归一（compares/compares_with→compares）。
每个对象带 evidence 列表：[{paper_id, title, quote}]，检索时作为可引用证据（引用落到原综述/论文的逐字片段）。
输出 base_kb_v2/state.json + 统计。确定性（无 LLM）。
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_kb_v2")
REL = {"compares_with": "compares", "compares": "compares", "extends": "extends", "improves": "improves",
       "replaces": "replaces", "combines": "combines"}


def norm(s):
    s = re.sub(r"\([^)]*\)", " ", (s or "").lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(methods?|approaches?|techniques?|models?|based|the|of|and|for)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _txt(r):
    for k in ("missing_translated", "missing"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main():
    R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
    title = lambda pid: (P.get(pid) or {}).get("title") or pid
    fam = defaultdict(lambda: {"names": set(), "props": [], "limits": [], "compares": [], "sources": set()})

    for pid, p in R.items():
        for r in p["records"]:
            if r.get("snapshot_type") == "taxonomy_node" and r.get("subject"):
                k = norm(r["subject"])
                if not k:
                    continue
                f = fam[k]
                f["names"].add(r["subject"].strip())
                f["sources"].add(pid)
                for c in r.get("claims") or []:
                    if c.get("claim"):
                        f["props"].append({"paper_id": pid, "title": title(pid), "text": c["claim"],
                                           "quote": (r.get("quote") or "")[:600], "record_id": r.get("id")})

    def attach(key_surface, item, slot):
        k = norm(key_surface)
        if k in fam:
            fam[k][slot].append(item)
            return True
        return False

    unattached = {"limits": [], "compares": []}
    for pid, p in R.items():
        for r in p["records"]:
            lab, kind = r.get("semantic_label"), r.get("kind")
            if kind == "survey_claim" and lab == "challenges" and r.get("claim"):
                it = {"paper_id": pid, "title": title(pid), "text": r["claim"], "quote": (r.get("quote") or "")[:600],
                      "about": r.get("claims_about"), "record_id": r.get("id")}
                if not attach(r.get("claims_about") or "", it, "limits"):
                    unattached["limits"].append(it)
            elif kind == "absence" and _txt(r):
                # missing_translated 在部分记录里是布尔标记（"已翻译"）而非文本——只取字符串字段
                it = {"paper_id": pid, "title": title(pid), "text": _txt(r),
                      "quote": (r.get("quote") or "")[:600], "about": r.get("subject"), "gap_type": r.get("absence_type"),
                      "record_id": r.get("id")}
                if not attach(r.get("subject") or "", it, "limits"):
                    unattached["limits"].append(it)
            elif (kind == "survey_claim" and lab == "comparison" and r.get("claim")) or \
                    (kind == "domain_snapshot" and r.get("snapshot_type") == "comparison_table"):
                text = r.get("claim") or "; ".join(c.get("claim", "") for c in (r.get("claims") or [])[:6])
                it = {"paper_id": pid, "title": title(pid), "text": text, "quote": (r.get("quote") or "")[:600],
                      "about": r.get("claims_about") or r.get("subject"), "record_id": r.get("id")}
                if not attach(it["about"] or "", it, "compares"):
                    unattached["compares"].append(it)
            elif kind == "lineage" and REL.get(r.get("relation")):
                fr = r.get("from_method_ref") or {}
                to = r.get("to_method_ref") or {}
                a = fr.get("canonical") or fr.get("surface") if isinstance(fr, dict) else str(fr)
                b = to.get("canonical") or to.get("surface") if isinstance(to, dict) else str(to)
                it = {"paper_id": pid, "title": title(pid), "relation": REL[r["relation"]],
                      "text": f"{a} {REL[r['relation']]} {b}: {r.get('claim') or ''}".strip(),
                      "quote": (r.get("quote") or "")[:600], "about": a, "record_id": r.get("id")}
                if not (attach(a or "", it, "compares") or attach(b or "", it, "compares")):
                    unattached["compares"].append(it)

    families = []
    for k, f in fam.items():
        families.append({"key": k, "names": sorted(f["names"]), "n_sources": len(f["sources"]),
                         "props": f["props"], "limits": f["limits"], "compares": f["compares"]})
    families.sort(key=lambda x: (-x["n_sources"], -len(x["props"]) - len(x["limits"])))
    state = {"families": families, "unattached": unattached}
    json.dump(state, open(os.path.join(V2, "state.json"), "w", encoding="utf-8"), ensure_ascii=False)
    st = {"families": len(families), "multi_source_families": sum(1 for f in families if f["n_sources"] >= 2),
          "props": sum(len(f["props"]) for f in families), "limits_attached": sum(len(f["limits"]) for f in families),
          "compares_attached": sum(len(f["compares"]) for f in families),
          "limits_unattached": len(unattached["limits"]), "compares_unattached": len(unattached["compares"])}
    json.dump(st, open(os.path.join(V2, "state_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
    for f in families[:8]:
        print(f"  {f['names'][:3]} src={f['n_sources']} props={len(f['props'])} limits={len(f['limits'])} cmp={len(f['compares'])}")


if __name__ == "__main__":
    main()
