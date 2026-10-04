# -*- coding: utf-8 -*-
"""BENCHMARK_BLOCKLIST domain-extension proposal generator (prereg §2:
CS 名单 -> 含 bio/photonics 基准名; gold-blind — corpus-derived names only,
question text never touched).

Source: dim_vocab_v1.json subject dimension (评测对象 = datasets/benchmarks/
tasks/environments/specimens) + registry entity aliases that appear as
subjects. LLM classifies each subject entry: is it a NAMED benchmark /
dataset / standard assay / catalog whose name could be mis-emitted as a
metric or leak gold-side vocabulary? -> proposal list with domain + reason.
Output is a PROPOSAL for human review (user gate); nothing is written into
table_semantic.py automatically.

Usage: python propose_blocklist.py [--model local:Qwen3.8-27B]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.common import call_json, par_map  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(BASE, "kb")

PROMPT = """下面是从科学文献语料（跨 cs_nlp/bio/biophysics/photonics/physics 域）收集的「评测对象」名称。判断每个名称是否属于以下类别之一：
- 命名基准/测试集（如 MMLU, GLUE, ImageNet）
- 命名数据集/语料库（如 The Pile, C4, Dolma）
- 标准分析/测定方法名（当它作为专有名称使用时，如 ChIP-seq 不算，但 "TAIR10 参考基因组" 算）
- 参考基因组/目录/数据库专名（如 GENCODE, UniProt, PDB）

这些名称需要进入"基准名黑名单"（防止它们被误当成指标名或方法名输出）。普通实验材料/设备/细胞系/一般性任务描述（如 "HeLa cells", "blood samples", "question answering"）不进名单。

对每个编号输出判定。输出 JSON：
{"proposals": [{"i": 编号, "add": true/false, "domain": "cs|bio|physics|other", "reason": "≤10词"}]}
只输出 JSON。

名称列表：
{lines}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local:Qwen3.8-27B")
    ap.add_argument("--batch", type=int, default=80)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vocab = json.load(open(os.path.join(KB, "dim_vocab_v1.json"),
                           encoding="utf-8"))
    # subject members = evaluation-object surfaces
    names = []
    seen = set()
    for fam in vocab.get("subject", []):
        for m in fam.get("members", []):
            k = m.strip().lower()
            if k and k not in seen:
                seen.add(k)
                names.append(m.strip())
    print(f"subject surfaces: {len(names)} unique")

    batches = [names[i:i + args.batch] for i in range(0, len(names), args.batch)]

    def run(batch):
        lines = "\n".join(f"[{i}] {n}" for i, n in enumerate(batch))
        obj = call_json(PROMPT.replace("{lines}", lines), args.model,
                        max_tokens=8000, retries=3) or {}
        return obj.get("proposals") or []

    proposals = []
    for bi, props in enumerate(par_map(run, batches)):
        got = {p.get("i"): p for p in props if isinstance(p, dict)}
        for i, n in enumerate(batches[bi]):
            p = got.get(i)
            if p and p.get("add") is True:
                proposals.append({"name": n, "domain": p.get("domain", "?"),
                                  "reason": str(p.get("reason", ""))[:60]})
        print(f"  batch {bi}: {len(props)}/{len(batches[bi])} judged, "
              f"{sum(1 for p in props if isinstance(p, dict) and p.get('add') is True)} proposed",
              flush=True)

    out = os.path.join(KB, "blocklist_proposals.json")
    json.dump(proposals, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    by_dom = {}
    for p in proposals:
        by_dom.setdefault(p["domain"], []).append(p["name"])
    print(f"\nproposals: {len(proposals)} -> {out}")
    for d in sorted(by_dom):
        print(f"  [{d}] {len(by_dom[d])}: {by_dom[d][:8]}")


if __name__ == "__main__":
    main()
