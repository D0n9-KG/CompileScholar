# -*- coding: utf-8 -*-
"""KB v2 种子库构建（REBUILD-PLAN-1003 第 2 步）——题目盲、单一论文注册表、可复现、带哈希。

输入（只读，不修改 base_kb/ 下任何文件）：
  base_kb/records_merged.json, deep_read_records.json, manifest_all.json,
  survey_manifest.json, hub_manifest.json, review_1002/main/hub_oai_meta.json（OAI 快照核实的 hub 元数据）
输出：base_kb_v2/
  papers.json         —— 统一论文注册表：paper_id → {title, year, arxiv_id, doi, abstract, layer, source}
  records.json        —— paper_id → {"records": [...], "provenance": ...}（与 KBTools 现有输入形态一致）
  build_report.json   —— 每层保留/剔除计数、剔除理由、截止过滤、文件哈希

层与去留：
  survey（136 篇综述，综述骨干）           保留
  hub（21 篇跨综述共识高引）               保留；元数据以 OAI 快照为准（修 1810.04805 标题错配）
  bulk_category（按 arXiv 类别抽的通域层） 保留
  demand_core（用 CS2 dev 题面逐题检索建的） 剔除（rehearsal_dev100.py；题目盲原则）
  garbage_key（"[ieee …]" 截断标题作 id）   剔除（全属 demand_core 来源：demand_pool 标题截断）
  深读记录：所属论文属于保留层才保留（46/66 篇深读论文属 demand_core 或运行期按题深读，一并剔除）
知识截止：发表年 > 2024（截止当年月份不可判）的论文剔除；无年份的论文保留但计数披露。
"""
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "base_kb")
OUT = os.path.join(HERE, "..", "base_kb_v2")
HUB_OAI = os.path.join(HERE, "..", "..", "..", "..", "review_1002", "main", "hub_oai_meta.json")
CUTOFF_YEAR = 2025  # 只保留 year < 2025（CS2 inserted_before=2025-05；只有年份时截止当年保守排除）


def L(name):
    return json.load(open(os.path.join(BASE, name), encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def yr(v):
    try:
        return int(str(v)[:4])
    except (TypeError, ValueError):
        return None


def main():
    merged, deep = L("records_merged.json"), L("deep_read_records.json")
    man = {r["paper_id"]: r for r in L("manifest_all.json")}
    survey = {r["paper_id"]: r for r in L("survey_manifest.json")}
    hub = {r["paper_id"]: r for r in L("hub_manifest.json")}
    hub_oai = json.load(open(HUB_OAI, encoding="utf-8")) if os.path.exists(HUB_OAI) else {}

    def layer(pid):
        if pid in survey:
            return "survey"
        if pid in hub:
            return "hub"
        if pid.startswith("["):
            return "garbage_key"
        if pid in man:
            return "demand_core" if man[pid].get("primary_category") == "demand_core" else "bulk_category"
        return "unregistered"

    KEEP = {"survey", "hub", "bulk_category"}
    papers, records, report = {}, {}, {"dropped": {}, "kept": {}, "cutoff_dropped": 0, "no_year_kept": 0}

    def register(pid, lay):
        if lay == "survey":
            r = survey[pid]
            return {"title": r.get("title"), "year": yr(r.get("year")), "arxiv_id": r.get("arxiv_id"),
                    "doi": None, "abstract": None, "layer": lay, "source": "survey_manifest"}
        if lay == "hub":
            r = hub[pid]
            o = hub_oai.get(r.get("arxiv_id") or "", {})
            return {"title": o.get("title") or r.get("title"), "year": yr(o.get("date")[-4:] if o.get("date") else r.get("year")) or yr(r.get("year")),
                    "arxiv_id": r.get("arxiv_id"), "doi": None, "abstract": o.get("abstract"),
                    "layer": lay, "source": "hub_manifest+oai" if o else "hub_manifest"}
        r = man[pid]
        return {"title": r.get("title"), "year": yr(r.get("year")), "arxiv_id": r.get("arxiv_id"),
                "doi": r.get("doi"), "abstract": r.get("abstract"), "layer": lay, "source": "manifest_all"}

    for src_name, src in (("merged", merged), ("deep", deep)):
        for pid, p in src.items():
            lay = layer(pid)
            if lay not in KEEP:
                report["dropped"].setdefault(f"{src_name}:{lay}", [0, 0])
                report["dropped"][f"{src_name}:{lay}"][0] += 1
                report["dropped"][f"{src_name}:{lay}"][1] += len(p.get("records") or [])
                continue
            meta = papers.get(pid) or register(pid, lay)
            if meta["year"] is not None and meta["year"] >= CUTOFF_YEAR:
                report["cutoff_dropped"] += 1
                continue
            if meta["year"] is None:
                report["no_year_kept"] += 1
            papers[pid] = meta
            cur = records.setdefault(pid, {"records": [], "provenance": p.get("provenance")})
            seen = {r.get("id") for r in cur["records"]}
            new = [r for r in (p.get("records") or []) if r.get("id") not in seen]
            cur["records"].extend(new)
            k = f"{src_name}:{lay}"
            report["kept"].setdefault(k, [0, 0])
            report["kept"][k][0] += 1
            report["kept"][k][1] += len(new)

    os.makedirs(OUT, exist_ok=True)
    pp, rp = os.path.join(OUT, "papers.json"), os.path.join(OUT, "records.json")
    json.dump(papers, open(pp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(records, open(rp, "w", encoding="utf-8"), ensure_ascii=False)
    report["totals"] = {"papers": len(papers), "records": sum(len(v["records"]) for v in records.values()),
                        "by_layer": {l: sum(1 for m in papers.values() if m["layer"] == l) for l in KEEP}}
    report["inputs_sha16"] = {n: sha(os.path.join(BASE, n)) for n in
                              ("records_merged.json", "deep_read_records.json", "manifest_all.json",
                               "survey_manifest.json", "hub_manifest.json")}
    report["outputs_sha16"] = {"papers.json": sha(pp), "records.json": sha(rp)}
    report["hub_title_fixed"] = [pid for pid, m in papers.items()
                                 if m["layer"] == "hub" and hub[pid]["title"][:25].lower() != (m["title"] or "")[:25].lower()]
    json.dump(report, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("kept", "dropped", "cutoff_dropped", "no_year_kept", "totals", "hub_title_fixed")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
