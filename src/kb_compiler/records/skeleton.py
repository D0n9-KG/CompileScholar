# -*- coding: utf-8 -*-
"""Stage 1: skeleton pass — one full-text call per paper -> paper_card.

Spec v1.1 §3 Stage 1 (+RC6 affiliation harvest, +RC7 figures/tables census).
The card is Stage 2's routing table and Stage 1.5's registry input.

Domain-agnostic discipline: no domain-specific examples in the prompt; entity
types and dimension names come from the frozen schema, values are harvested
from the text only ("no external knowledge, no guessing" is in the prompt).

Usage:
  python -m kb_compiler.records.skeleton --texts DIR --manifest M.json \
      --out CARDS.json --model DeepSeek-V4-Flash [--only pid1,pid2] [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from .common import (MAX_PAPER_CHARS, call_json, load_corpus, load_json,
                     load_manifest, save_json)

CARD_PROMPT = """你是科学文献知识编译器的第一遍（结构通读）。通读论文全文，输出该论文的"论文卡片"（JSON）。纪律：只输出文中明确写出的内容；不使用外部知识；不猜测；拿不准的字段留空。

论文标题（来自权威元数据）：{title}

输出 JSON（字段定义）：
{{
 "method_identity": {{
   "canonical_name": "本篇提出的方法/系统/框架的常用名（纯评测或批评类论文可为空）",
   "aliases": ["缩写", "全称", "文内变体写法"],
   "entity_type": "method 或 mechanism 或 practice（本篇贡献实体的类型）"
 }},
 "affiliations": ["作者机构名（从文头署名/脚注逐字取，去重）"],
 "contributions": ["论文自述的贡献（每条≤25词，忠于原文措辞）"],
 "experimental_matrix": [
   {{"subject": "被评测/被研究对象（数据集/基准/任务/环境/试样/工况名，逐字）",
     "measures": ["指标名（逐字）"],
     "variants": ["评测过的变体/消融条件/对比基线名（逐字）"],
     "where": "Table N / Section N / Figure N"}}
 ],
 "figures_tables": [
   {{"ref": "Table 1 / Figure 3",
     "kind": "table 或 figure",
     "caption": "caption 逐字（超过40词则截断到40词）"}}
 ],
 "dimension_candidates": {{
   "subject": ["文中出现的评测对象名"],
   "setup": ["协议/环境/工况条件词（测量规约、制备与边界条件、设备条件的定性词，用原文措辞）"],
   "variant": ["方法/模型变体、组件开关名"],
   "hyperparam_item": ["被当作自变量系统扫描的参数名（取值被逐档对比的任何量化参数）"]
 }},
 "related_methods": [
   {{"name": "相关工作实体名（逐字）",
     "entity_type": "method 或 mechanism 或 practice 或 out_of_corpus（文外被引实体选 out_of_corpus）",
     "relation_hint": "本篇原文对它的关系措辞（如 extends/improves/uses/compares 的原文动词短语，逐字）",
     "cited_year": "仅当文内引文逐字给出作者-年份时填（如 2018），否则留空"}}
 ]
}}

只输出 JSON，不要其他文字。

论文全文：
{text}"""

_print_lock = threading.Lock()


def build_card(pid: str, text: str, title: str, model: str):
    prompt = CARD_PROMPT.replace("{title}", title or pid).replace(
        "{text}", text[:MAX_PAPER_CHARS])
    card = call_json(prompt, model, max_tokens=8000, retries=3)
    if not isinstance(card, dict):
        with _print_lock:
            print(f"[{pid}] CARD FAIL (unparseable after retries)", flush=True)
        return pid, None
    card["_paper_id"] = pid
    card["_title_used"] = title
    card["_model"] = model
    with _print_lock:
        mi = card.get("method_identity") or {}
        print(f"[{pid}] card ok | id={mi.get('canonical_name') or '-'} "
              f"| matrix={len(card.get('experimental_matrix') or [])} "
              f"| figtab={len(card.get('figures_tables') or [])} "
              f"| related={len(card.get('related_methods') or [])} "
              f"| aff={card.get('affiliations') or []}", flush=True)
    return pid, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated paper_ids (test runs)")
    args = ap.parse_args()

    corpus = load_corpus(args.texts)
    manifest = load_manifest(args.manifest)
    if args.only:
        keep = set(args.only.split(","))
        corpus = [c for c in corpus if c[0] in keep]

    existing = load_json(args.out, default={}) or {}
    todo = [(pid, text) for pid, text in corpus if pid not in existing]
    print(f"skeleton: {len(todo)} to run ({len(existing)} cached), model={args.model}",
          flush=True)

    cards = dict(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_card, pid, text,
                          (manifest.get(pid) or {}).get("title"), args.model)
                for pid, text in todo]
        for f in futs:
            pid, card = f.result()
            if card is not None:
                cards[pid] = card
                save_json(cards, args.out)  # incremental save (crash-safe)
    n_fail = len(corpus) - sum(1 for pid, _ in corpus if pid in cards)
    print(f"\nsaved {len(cards)} cards -> {args.out} (failed this run: {n_fail})",
          flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
