# -*- coding: utf-8 -*-
"""MDAQA 外部臂公共层：全文获取 + 官方四重叠指标评测。

设计（PREREG-MDAQA-subset.md，官方协议口径）：
  - 全文：arXiv HTML 通道（复用 CS2 的 fetch_arxiv_html——30 题抽样
    63 篇 100% 可得已实测）；落盘 texts/<arxiv_id>.md，全臂共享
    （PaperQA/LightRAG/全文直读/ours 四臂同语料=公平前提）
  - 评测：论文 Table 4 协议——METEOR/ROUGE-L/CIDEr/BERTScore F1 对
    gold answer（GPT-4o 合成参考，弱 gold 披露在案）。官方 repo 只有
    数据生成管线无评测代码，指标按论文描述实现，包用 nltk/rouge_score/
    bert_score（标准实现）。
  - BERTScore 模型：默认 roberta-large（论文未指明——标准默认，披露）。
    本地无 HF 下载通道时降级 bert-base-uncased（记录）。

Usage:
  from mdaqa_common import fetch_support_texts, official_metrics
"""
from __future__ import annotations

import json
import os
import sys
import threading

MDAQA = os.path.dirname(os.path.abspath(__file__))
CS2_BUILD = (r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
             r"\experiments\benchmarks\cs2\base_kb_build")
TEXTS = os.path.join(MDAQA, "texts")
sys.path.insert(0, CS2_BUILD)

_FETCH_SEM = threading.Semaphore(4)   # arXiv HTML 礼貌并发


def fetch_support_texts(subset_path: str, limit: int | None = None) -> dict:
    """子集所有 support 论文全文 → texts/<arxiv_id>.md（幂等，已存在跳过）。

    返回 {arxiv_id: path}。失败清单落 fetch_failures.json（重跑只补漏）。
    """
    os.makedirs(TEXTS, exist_ok=True)
    from fetch_arxiv_html import fetch_text
    subset = json.load(open(subset_path, encoding="utf-8"))
    need = []
    seen = set()
    for q in subset["questions"][:limit]:
        for aid in q["support"]:
            if aid in seen:
                continue
            seen.add(aid)
            p = os.path.join(TEXTS, aid + ".md")
            if not os.path.exists(p) or os.path.getsize(p) < 5000:
                need.append((aid, p))
    print(f"[texts] {len(seen)} unique support papers, "
          f"{len(need)} to fetch", flush=True)
    fails = {}
    fail_path = os.path.join(MDAQA, "fetch_failures.json")
    if os.path.exists(fail_path):
        fails = json.load(open(fail_path, encoding="utf-8"))
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()

    def _one(aid, p):
        with _FETCH_SEM:
            try:
                t = fetch_text(aid)
                if len(t) < 5000:
                    raise RuntimeError(f"too short: {len(t)}")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(t)
                return aid, None
            except Exception as e:
                return aid, str(e)[:120]

    with ThreadPoolExecutor(max_workers=4) as ex:
        for aid, err in ex.map(lambda a: _one(*a), need):
            with lock:
                if err:
                    fails[aid] = err
                else:
                    fails.pop(aid, None)
            if err:
                print(f"  FAIL {aid}: {err}", flush=True)
    json.dump(fails, open(fail_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[texts] done: {len(seen) - len(fails)}/{len(seen)} available, "
          f"{len(fails)} failures -> fetch_failures.json", flush=True)
    return {aid: os.path.join(TEXTS, aid + ".md")
            for aid in seen if aid not in fails}


# ---------------- 官方四重叠指标（论文 Table 4 协议） ----------------

def official_metrics(predictions: list[str], references: list[str]) -> dict:
    """METEOR / ROUGE-L / CIDEr / BERTScore-F1（论文口径）。

    predictions/references 平行列表。CIDEr 官方口径对多参考设计；单参考
    下降级为该参考上的 CIDEr-n 均值（标准 pyt implementation 同源逻辑，
    披露）。全部指标在臂间一致=可比。
    """
    import nltk
    nltk.data.path.insert(0, os.path.expanduser("~/nltk_data"))
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    # tokenize（METEOR 需预分词；ROUGE 内部自做）
    def _tok(s):
        return s.split()

    meteors, rouges = [], []
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for hyp, ref in zip(predictions, references):
        hyp = (hyp or "").strip()
        ref = (ref or "").strip()
        if not hyp:
            meteors.append(0.0)
            rouges.append(0.0)
            continue
        try:
            meteors.append(meteor_score([_tok(ref)], _tok(hyp)))
        except Exception:
            meteors.append(0.0)
        try:
            rouges.append(rs.score(ref, hyp)["rougeL"].fmeasure)
        except Exception:
            rouges.append(0.0)

    # CIDEr（标准实现——句子级 n-gram TF-IDF 共识；单参考时退化为
    # 与参考的 n-gram 重叠加权，与官方多参考版同源）
    cider = _cider(predictions, references)

    # BERTScore（惰性——重模型加载）
    bs = _bertscore_f1(predictions, references)

    return {
        "METEOR": round(sum(meteors) / max(1, len(meteors)), 4),
        "ROUGE-L": round(sum(rouges) / max(1, len(rouges)), 4),
        "CIDEr": round(cider, 4),
        "BERTScore-F1": round(bs, 4),
        "n": len(predictions),
    }


def _cider(hyps, refs) -> float:
    """CIDEr-D 风格单参考版（n=1..4 均值）。全臂同实现=可比，披露。"""
    import math
    from collections import Counter

    def _ngrams(toks, n):
        return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

    # corpus-level document frequency（参考侧）
    doc_freq = Counter()
    ref_ngrams = []
    for ref in refs:
        toks = (ref or "").split()
        grams = set()
        for n in range(1, 5):
            grams.update(_ngrams(toks, n).keys())
        doc_freq.update(grams)
        ref_ngrams.append(toks)
    N = max(1, len(refs))

    def _score(hyp, ref):
        if not (hyp or "").strip() or not (ref or "").strip():
            return 0.0
        htoks, rtoks = hyp.split(), ref.split()
        sims = []
        for n in range(1, 5):
            hg, rg = _ngrams(htoks, n), _ngrams(rtoks, n)
            if not hg or not rg:
                sims.append(0.0)
                continue
            # TF-IDF 余弦（CIDEr-D：clip 计数）。单参考/小语料退化：全
            # idf=log((N+1)/(df+1))≈0 时（实测单参考恒 0）退化为无权重
            # clip-count 余弦（CIDEr-D 的均匀权重极限，臂间一致=可比，
            # 与官方多参考版同源逻辑，披露）
            num = 0.0
            weighted = False
            for g, hc in hg.items():
                if g in rg:
                    idf = math.log((N + 1) / (doc_freq.get(g, 0) + 1))
                    if idf > 0:
                        weighted = True
                    num += min(hc, rg[g]) * idf * idf
            if weighted:
                hh = sum(c * c for c in hg.values())
                rr = sum(c * c for c in rg.values())
                sims.append(num / math.sqrt(hh * rr) if hh and rr else 0.0)
            else:
                # 均匀权重：clip 计数余弦
                shared = set(hg) & set(rg)
                if not shared:
                    sims.append(0.0)
                    continue
                num = float(sum(min(hg[g], rg[g]) for g in shared))
                hh = sum(min(hc, rg.get(g, 0)) ** 2 for g, hc in hg.items())
                rr = sum(min(rc, hg.get(g, 0)) ** 2 for g, rc in rg.items())
                den = math.sqrt(hh * rr)
                sims.append(num / den if den else 0.0)
        return sum(sims) / 4.0

    scores = [_score(h, r) for h, r in zip(hyps, refs)]
    return sum(scores) / max(1, len(scores))


def _bertscore_f1(hyps, refs) -> float:
    try:
        import bert_score
        P, R, F1 = bert_score.score(
            [h or " " for h in hyps], [r or " " for r in refs],
            lang="en", verbose=False, batch_size=32,
            model_type="roberta-large")
        return float(F1.mean())
    except Exception as e:
        print(f"[bertscore] fallback ({str(e)[:80]})", flush=True)
        try:
            import bert_score
            P, R, F1 = bert_score.score(
                [h or " " for h in hyps], [r or " " for r in refs],
                lang="en", verbose=False, batch_size=32)
            return float(F1.mean())
        except Exception as e2:
            print(f"[bertscore] unavailable: {str(e2)[:80]}", flush=True)
            return float("nan")
