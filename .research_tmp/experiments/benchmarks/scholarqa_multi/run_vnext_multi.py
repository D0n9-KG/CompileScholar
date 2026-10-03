# -*- coding: utf-8 -*-
"""Multi-108 × 新答题管线（闭卷固定语料：只用 Multi KB，无外部检索/引文扩展）。

KB = kb/postcheck/records_checked.json（430 篇 union 语料，paper_id=文本 stem）；论文元数据 = corpus/manifest.json。
输出与 baselines/ours/answers_ours.json 同形态（answer_official_all 里引用为官方 [ctx_idx] 形式），
供 multi_judge_incremental.py 直接判分（新臂名 vnext）。
用法：python run_vnext_multi.py --tag v1 [--limit N] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "_shared", "tools")
sys.path.insert(0, TOOLS)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import answer_pipeline as AP  # noqa: E402
from multi_baseline_common import CitationTranslator, load_id_mapping, load_questions  # noqa: E402


class MultiKB(AP.KB):
    """records_checked.json + corpus manifest → 与 KB v2 同接口；向量按需在线建（430 篇，首次 ~5 分钟）。"""

    def __init__(self):
        recs = json.load(open(os.path.join(HERE, "kb", "postcheck", "records_checked.json"), encoding="utf-8"))
        man = json.load(open(os.path.join(HERE, "corpus", "manifest.json"), encoding="utf-8"))
        rows = man if isinstance(man, list) else man.get("papers") or list(man.values())
        self.papers = {}
        for r in rows:
            if isinstance(r, dict) and r.get("paper_id"):
                self.papers[r["paper_id"]] = {"title": r.get("title") or r["paper_id"].replace("_", " "),
                                              "year": r.get("year"), "arxiv_id": r.get("arxiv_id")}
        for pid in recs:
            self.papers.setdefault(pid, {"title": pid.replace("_", " "), "year": None})
        from kb_compiler.retrieve.hybrid import HybridIndex
        from kb_infra.embedding import embed_local
        import numpy as np
        vp = os.path.join(HERE, "kb", "vnext_record_vecs.f32")
        self.index = HybridIndex(recs, self.papers, embed_fn=embed_local)
        n = len(self.index.items)
        if os.path.exists(vp) and os.path.getsize(vp) == n * 4096 * 4:
            self.index.doc_vecs = np.fromfile(vp, dtype="float32").reshape(n, 4096)
            self.index._norm_docs()
        else:
            self.index.ensure_vectors()
            np.asarray(self.index.doc_vecs, dtype="float32").tofile(vp)


def to_official(qid, sections, tr):
    """[k] 引用 → 论文 stem → 官方 [ctx_idx]；多节拼成一段正文（与旧 ours 臂同口径）。"""
    text_parts, stem_cites = [], []
    for s in sections:
        body = s["text"]
        for c in s["citations"]:
            stem = (c.get("metadata") or {}).get("paper_key", "").removeprefix("kb:")
            if stem:
                tag = f"[[{qid}:{len(stem_cites)}]]"
                body = body.replace(c["id"], tag)
                stem_cites.append((tag, stem))
        text_parts.append(body)
    answer = "\n\n".join(text_parts)
    off, recs = tr.translate(answer, stem_cites, policy="all")
    return answer, off, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    qs = load_questions()
    if a.limit:
        qs = qs[:a.limit]
    idmap = load_id_mapping()
    kb = MultiKB()
    out_dir = os.path.join(HERE, "baselines", "vnext")
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, f"answers_vnext_{a.tag}.json")
    done = {r["qid"]: r for r in json.load(open(out_p, encoding="utf-8"))} if os.path.exists(out_p) else {}

    def one(q):
        qid, question = q.get("id") or q.get("qid"), q.get("input") or q.get("question")
        try:
            r = AP.answer(question, kb, use_ext=False, use_cite=False)
            tr = CitationTranslator(qid, idmap)
            raw, off, recs = to_official(qid, r["sections"], tr)
            row = {"qid": qid, "question": question, "raw_answer": raw, "answer_official_all": off,
                   "citations_all": recs, "citation_mapped": tr.mapped, "citation_dropped": tr.dropped,
                   "words": r["trace"]["words"], "trace": {k: r["trace"][k] for k in ("plan", "by_src", "assemble", "elapsed_s")}}
        except Exception as e:
            row = {"qid": qid, "question": question, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {qid} words={row.get('words')} mapped={row.get('citation_mapped')} err={row.get('error')}", flush=True)
        return row

    todo = [q for q in qs if (q.get("id") or q.get("qid")) not in done]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(one, todo):
            done[row["qid"]] = row
            json.dump(list(done.values()), open(out_p, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(done)} -> {out_p}")


if __name__ == "__main__":
    main()
