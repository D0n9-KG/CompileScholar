# -*- coding: utf-8 -*-
"""跨论文结构五件套·组件1:粗抽实体归一(确定性,零LLM)。

实测地基缺陷(2026-10-01 盘上验证):
- 全库实体引用 0% 绑定:records_merged 9,645 个 *_ref 槽位 entity_id 全
  null(resolve_refs 从未在此 CS2 KB 上跑过),deep_read_records 7,715 槽位
  同样全 null(其中仅 24.7% 表面能命中 registry——其余是域外实体,诚实缺失)
- 后果:views.genealogy 边键=表面名(编译器 fid or norm(name)),节点键=
  entity_id——两个键空间互不连通,PPR 种子(entity_id)与邻接表(名)
  永不相交=多跳传播恒空;matrix 按原始表面聚桶,跨论文同方法不归一
- survey_claim 12,712 / absence 2,298 / domain_snapshot 1,580 条记录
  完全没有实体载体(mentions/ref 都没有)

三段(全部幂等;首轮备份 *.bak_pre_normalize.json;ledger 落盘可审计):
  A. 既有 *_ref 槽位 → registry surface_index 绑定(resolve_refs 语义)
  B. mentions(粗抽纯字符串,实测 98.1% 可解析)→ entity_refs 列表
  C. 无载体记录 → subject/claim/claims_about 文本 n-gram 扫描回填
     规则(全部数据驱动,非手挑词):
       - 表面 ≥4 字符(杀 2-3 字母垃圾注册项 're'/'ar'/'is')
       - 词边界匹配(token 级 n-gram,不跨词)
       - 最长优先、命中不重叠
       - 文档频率 ≤25% 论文(预扫一遍;'Artificial Intelligence' 63% 之类
         泛化词自动出局——与被否决的手挑 blocklist 的区别:阈值是统计
         量,不指向任何具体词)
       - 每记录 cap 6(entity_refs 是图桥接用途,不是穷举 NER)

用法:PYTHONUTF8=1 python normalize_entities.py [--dry]
之后:重编译 views(PYTHONPATH=src python -m kb_compiler.views.compiler
  --records base_kb/records_merged.json --registry base_kb/registry_v2.json
  --vocab base_kb/dim_vocab_cs2.json --manifest base_kb/manifest_all.json
  --out base_kb/views_cs2.json) + python backfill_years.py
"""
import json
import os
import re
import shutil
import sys
from collections import defaultdict

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")

REFF = ("method_ref", "from_method_ref", "to_method_ref",
        "scope_ref_ref", "target_ref_ref")
MIN_SURFACE_LEN = 4
MAX_DOC_FREQ = 0.05          # 表面命中 >5% 论文 → 太泛化,不入文本扫描
                            # (实测切点:真实方法实体全部 <3% DF——bert 1.8%
                            #  transformer 2.5% RL 2.0%;3-25% 带被泛词
                            #  model/data/dataset 和大小写陷阱 ComplEx/WILL
                            #  占据。5% 保住 large language model 4.9%/    #  fine-tuning 3.6%,杀 the study/proposed approach 类噪声)
MAX_ENTITY_REFS = 6

_norm_re = re.compile(r"\s+")


def norm(s):
    return _norm_re.sub(" ", str(s or "").strip().lower())


def load(path):
    return json.load(open(path, encoding="utf-8"))


def iter_records(store):
    for pid, payload in store.items():
        if not isinstance(payload, dict):
            continue
        for r in payload.get("records") or []:
            if isinstance(r, dict):
                yield pid, r


def scan_text(text, surface_set, max_n=6):
    """token 级最长优先 n-gram 扫描。返回命中表面列表(不重叠)。"""
    tokens = text.split()
    consumed = [False] * len(tokens)
    hits = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(consumed[i:i + n]):
                continue
            cand = " ".join(tokens[i:i + n])
            if cand in surface_set:
                hits.append(cand)
                for j in range(i, i + n):
                    consumed[j] = True
    return hits


def main():
    dry = "--dry" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    registry = load(os.path.join(BASE_KB, "registry_v2.json"))
    si = registry.get("surface_index", {})
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}

    files = ["records_merged.json", "deep_read_records.json"]
    stores = {f: load(os.path.join(BASE_KB, f)) for f in files}

    # ---- 预扫:文档频率(为 pass C 的泛化过滤)----
    # 统计口径:表面在多少篇论文的记录文本中出现(pass A/B 的表面不受此
    # 限制——mentions/ref 是抽取器显式认定的实体提及,泛化问题只影响
    # 我们自己发明的文本扫描)
    scan_surfaces = {s for s in si
                     if len(s) >= MIN_SURFACE_LEN and si.get(s) in byid}
    paper_hits = defaultdict(set)
    for f in files:
        for pid, r in iter_records(stores[f]):
            text = " ".join(filter(None, [norm(r.get("subject")),
                                          norm(r.get("claim")),
                                          norm(r.get("claims_about"))]))
            if not text:
                continue
            for s in scan_text(text, scan_surfaces):
                paper_hits[s].add(pid)
    n_papers = max(1, len({pid for f in files
                           for pid, _ in iter_records(stores[f])}))
    df_ok = {s for s in scan_surfaces
             if len(paper_hits.get(s, ())) <= MAX_DOC_FREQ * n_papers}
    print(f"[预扫] 论文数={n_papers} 候选表面={len(scan_surfaces)} "
          f"文档频率过滤后={len(df_ok)} "
          f"(>{int(MAX_DOC_FREQ*100)}%论文出局={len(scan_surfaces)-len(df_ok)})")

    ledger = []
    stats = defaultdict(int)

    # ---- pass A/B/C 逐文件处理 ----
    for fname in files:
        store = stores[fname]
        for pid, r in iter_records(store):
            # A: 既有 ref 槽位绑定
            resolved_a = 0
            for fld in REFF:
                ref = r.get(fld)
                if not isinstance(ref, dict):
                    continue
                surf = (ref.get("surface") or "").strip()
                eid = si.get(norm(surf)) if surf else None
                if eid and eid in byid and not ref.get("entity_id"):
                    ref["canonical"] = byid[eid]["canonical"]
                    ref["entity_id"] = eid
                    resolved_a += 1
            if resolved_a:
                stats["passA_refs_bound"] += resolved_a

            # 已有绑定载体(ref 有 entity_id)→ 不做 B/C 补充
            has_bound_ref = any(
                isinstance(r.get(fld), dict) and r.get(fld, {}).get("entity_id")
                for fld in REFF)

            # B: mentions → entity_refs
            added = []
            if not r.get("entity_refs"):
                for m in r.get("mentions") or []:
                    if not isinstance(m, str) or not m.strip():
                        continue
                    eid = si.get(norm(m))
                    if eid and eid in byid:
                        added.append({"surface": m.strip(),
                                      "canonical": byid[eid]["canonical"],
                                      "entity_id": eid})
            # C: 无载体记录文本扫描
            if not added and not has_bound_ref and not r.get("entity_refs"):
                text = " ".join(filter(None, [
                    norm(r.get("subject")), norm(r.get("claim")),
                    norm(r.get("claims_about"))]))
                if text:
                    for s in scan_text(text, df_ok):
                        eid = si.get(s)
                        if eid and eid in byid:
                            added.append({"surface": s,
                                          "canonical": byid[eid]["canonical"],
                                          "entity_id": eid})
                        if len(added) >= MAX_ENTITY_REFS:
                            break
            if added:
                # 去重(同 entity 只留首个表面)
                seen, dedup = set(), []
                for a in added:
                    if a["entity_id"] not in seen:
                        seen.add(a["entity_id"])
                        dedup.append(a)
                r["entity_refs"] = dedup
                stats["passB_mentions" if r.get("mentions") else "passC_textscan"] += 1
                stats["entity_refs_added"] += len(dedup)
                ledger.append({"file": fname, "paper_id": pid,
                               "record_id": r.get("id"), "kind": r.get("kind"),
                               "pass": "B" if r.get("mentions") else "C",
                               "entities": [a["canonical"] for a in dedup]})
        print(f"[{fname}] 处理完成")

    # ---- 汇总 ----
    bound_after = total_refs = refs_records = ents_records = 0
    for f in files:
        for pid, r in iter_records(stores[f]):
            has = False
            for fld in REFF:
                ref = r.get(fld)
                if isinstance(ref, dict) and (ref.get("surface") or ref.get("entity_id")):
                    total_refs += 1
                    has = True
                    if ref.get("entity_id"):
                        bound_after += 1
            if r.get("entity_refs"):
                ents_records += 1
                has = True
            if has:
                refs_records += 1
    n_recs = sum(1 for f in files for _ in iter_records(stores[f]))
    print("\n===== 归一结果 =====")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  ref槽位绑定率: {bound_after}/{total_refs} "
          f"({100*bound_after/max(total_refs,1):.1f}%)")
    print(f"  携带实体载体的记录: {refs_records}/{n_recs} "
          f"({100*refs_records/n_recs:.1f}%)")

    if dry:
        print("[dry] 不写盘")
        return

    for fname in files:
        src = os.path.join(BASE_KB, fname)
        bak = src.replace(".json", ".bak_pre_normalize.json")
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f"[backup] {os.path.basename(bak)}")
        tmp = src + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(stores[fname], fh, ensure_ascii=False)
        os.replace(tmp, src)
        print(f"[write] {fname}")
    led_path = os.path.join(BASE_KB, "entity_norm_ledger.jsonl")
    with open(led_path, "w", encoding="utf-8") as fh:
        for row in ledger:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] entity_norm_ledger.jsonl ({len(ledger)} rows)")


if __name__ == "__main__":
    main()
