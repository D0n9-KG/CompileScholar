# -*- coding: utf-8 -*-
"""MDAQA 外部臂 runner：PaperQA2 / LightRAG / 全文直读（社区模式）。

每题一个隔离实例（社区=该题的 support 论文全文 2-4 篇）——对齐
PREREG-MDAQA-subset.md 四臂设计。判分=官方四重叠指标
（mdaqa_common.official_metrics，论文 Table 4 协议）。

资源纪律：与 CS2 判分不抢资源（判分走 Paratera GLM）；本地 GPU 由
本地 LLM/embedding 调用使用，臂间顺序执行（每臂内部并行受
LOCAL_MAX_CONCURRENT 约束——设 8，给 CS2 批次答题留余量）。

Usage:
  python mdaqa_arms.py fulltext   # 全文直读参考臂（最快，先跑）
  python mdaqa_arms.py paperqa    # PaperQA2 社区模式
  python mdaqa_arms.py lightrag   # LightRAG 社区模式
  python mdaqa_arms.py eval       # 对已有答案跑官方指标
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

MDAQA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MDAQA)
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
                   r"\experiments\benchmarks\_shared\tools")

SUBSET = os.path.join(MDAQA, "subset_300.json")
TEXTS = os.path.join(MDAQA, "texts")
ARMS_DIR = os.path.join(MDAQA, "arms")
os.makedirs(ARMS_DIR, exist_ok=True)

os.environ.setdefault("LOCAL_MAX_CONCURRENT", "8")
os.environ.setdefault("LLM_CALL_LOG", os.path.join(
    MDAQA, "ledger_mdaqa.jsonl"))
os.environ.setdefault("LLM_RUN_ID", "mdaqa-external")

for line in open(r"C:\Users\D0n9\Desktop\CompileScholar\.env",
                 encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("LOCAL_MODEL", "Qwen3.8-27B")


def load_subset() -> list[dict]:
    d = json.load(open(SUBSET, encoding="utf-8"))
    return d["questions"]


def q_texts(q: dict) -> list[tuple[str, str]]:
    """[(arxiv_id, full_text)]——support 全文（缺失的跳过并记录）。"""
    out = []
    for aid in q["support"]:
        p = os.path.join(TEXTS, aid + ".md")
        if os.path.exists(p) and os.path.getsize(p) >= 5000:
            out.append((aid, open(p, encoding="utf-8",
                                  errors="replace").read()))
    return out


def _answers_path(arm: str) -> str:
    return os.path.join(ARMS_DIR, f"answers_{arm}.json")


def _load_done(arm: str) -> dict:
    p = _answers_path(arm)
    if os.path.exists(p):
        return {r["qid"]: r for r in json.load(open(p, encoding="utf-8"))}
    return {}


def _save_done(arm: str, done: dict):
    tmp = _answers_path(arm) + ".tmp"
    json.dump(list(done.values()), open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    os.replace(tmp, _answers_path(arm))


def _ask_local(prompt: str, max_tokens: int = 1200) -> str:
    """同步本地 LLM 调用（全文直读臂用）。"""
    from kb_infra.llm import call_local
    return call_local(prompt, MODEL, max_tokens, 0.0, None, False) or ""


# ---------------- 全文直读参考臂（天花板探针） ----------------

FULLTEXT_PROMPT = """You are answering a question about academic papers. Below are the FULL TEXTS of the papers that together contain the answer. Read them and answer the question thoroughly and precisely, citing specific details. Answer in 3-8 sentences.

QUESTION: {question}

PAPERS:
{papers}

ANSWER (direct, factual, no preamble):"""


async def run_fulltext(q_limit: int | None = None):
    questions = load_subset()[:q_limit] if q_limit else load_subset()
    done = _load_done("fulltext")
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[fulltext] {len(todo)} to answer "
          f"({len(done)} resumed)", flush=True)
    for i, q in enumerate(todo):
        docs = q_texts(q)
        if not docs:
            done[q["qid"]] = {"qid": q["qid"], "error": "no support texts"}
            _save_done("fulltext", done)
            continue
        papers = "\n\n".join(
            f"=== PAPER {aid} ===\n{t[:60000]}"
            for aid, t in docs)
        prompt = FULLTEXT_PROMPT.format(question=q["question"],
                                        papers=papers)
        # 128k 窗口：2-4 篇 × 60k chars cap ≈ 最坏 ~60k tokens，安全
        ans = await asyncio.to_thread(_ask_local, prompt, 1000)
        done[q["qid"]] = {"qid": q["qid"], "question": q["question"],
                          "answer": ans, "support": q["support"]}
        _save_done("fulltext", done)
        print(f"[fulltext] {i+1}/{len(todo)} qid={q['qid']} "
              f"{len(ans)}ch", flush=True)


# ---------------- PaperQA2 社区模式 ----------------

async def run_paperqa(q_limit: int | None = None):
    questions = load_subset()[:q_limit] if q_limit else load_subset()
    done = _load_done("paperqa")
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[paperqa] {len(todo)} to answer ({len(done)} resumed)",
          flush=True)
    # PaperQA 每题一个隔离实例（社区语料小，索引秒级）
    import paperqa
    for i, q in enumerate(todo):
        docs = q_texts(q)
        if not docs:
            done[q["qid"]] = {"qid": q["qid"], "error": "no support texts"}
            _save_done("paperqa", done)
            continue
        try:
            # litellm 后端指本地 GPUStack（同模型=公平）
            import litellm
            litellm.api_base = os.environ["LOCAL_BASE_URL"]
            litellm.api_key = os.environ.get("LOCAL_API_KEY", "local")
            docs_obj = paperqa.Docs()
            for aid, text in docs:
                await docs_obj.aadd(
                    path=aid,  # 用 arxiv id 作文件名（citation 溯源用）
                    contents=text,
                    citation=aid + " paper",
                    key=aid,
                    llm_model="openai/" + MODEL,
                    llm_api_base=os.environ["LOCAL_BASE_URL"],
                    llm_api_key=os.environ.get("LOCAL_API_KEY", "local"),
                    embedding_model="openai/qwen3-embedding-8b-local",
                    embedding_api_base=os.environ["LOCAL_BASE_URL"],
                    embedding_api_key=os.environ.get("LOCAL_API_KEY",
                                                     "local"),
                )
            from paperqa import ask
            result = await ask(
                docs=q["question"] and docs_obj,
                llm_model="openai/" + MODEL,
                llm_api_base=os.environ["LOCAL_BASE_URL"],
                llm_api_key=os.environ.get("LOCAL_API_KEY", "local"),
                embedding_model="openai/qwen3-embedding-8b-local",
                embedding_api_base=os.environ["LOCAL_BASE_URL"],
                embedding_api_key=os.environ.get("LOCAL_API_KEY", "local"),
                max_workers=4,
            )
            ans = result.answer or ""
            done[q["qid"]] = {"qid": q["qid"], "question": q["question"],
                              "answer": ans, "support": q["support"]}
        except Exception as e:
            done[q["qid"]] = {"qid": q["qid"],
                              "error": f"{type(e).__name__}: {str(e)[:150]}"}
        _save_done("paperqa", done)
        print(f"[paperqa] {i+1}/{len(todo)} qid={q['qid']} "
              f"{len(done[q['qid']].get('answer') or '')}ch", flush=True)


# ---------------- LightRAG 社区模式 ----------------

async def run_lightrag(q_limit: int | None = None):
    questions = load_subset()[:q_limit] if q_limit else load_subset()
    done = _load_done("lightrag")
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[lightrag] {len(todo)} to answer ({len(done)} resumed)",
          flush=True)
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from kb_infra.embedding import embed_local
    from kb_infra.llm import call_local

    # 每题一个隔离 workdir（社区模式：索引隔离，无跨题泄漏）
    async def llm_local(prompt, system_prompt=None, history_messages=[],
                        keyword_extraction=False, **kwargs):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(
                    call_local, prompt, MODEL,
                    4000 if keyword_extraction else 2000, 0.0, None, False),
                timeout=600)
        except asyncio.TimeoutError:
            return ""
        return out or ""

    async def embed_async(texts):
        return await asyncio.to_thread(embed_local, list(texts))

    import hashlib
    dim = len(embed_local(["probe"])[0])
    print(f"[lightrag] embed dim={dim}", flush=True)

    for i, q in enumerate(todo):
        docs = q_texts(q)
        if not docs:
            done[q["qid"]] = {"qid": q["qid"], "error": "no support texts"}
            _save_done("lightrag", done)
            continue
        wd = os.path.join(ARMS_DIR, "lightrag_work",
                          hashlib.md5(str(q["qid"]).encode()).hexdigest())
        os.makedirs(wd, exist_ok=True)
        try:
            rag = LightRAG(
                working_dir=wd,
                llm_model_func=llm_local,
                embedding_func=EmbeddingFunc(
                    embedding_dim=dim, max_token_size=8192,
                    func=embed_async),
            )
            await rag.initialize_storages()
            # 幂等：workdir 有该社区的完成标记则跳过 ingest
            marker = os.path.join(wd, "_ingested.done")
            if not os.path.exists(marker):
                for aid, text in docs:
                    await rag.ainsert(text[:120000], ids=[aid],
                                      file_paths=[aid])
                open(marker, "w").write("1")
            result = await asyncio.wait_for(
                rag.aquery_llm(q["question"], QueryParam(mode="hybrid")),
                timeout=1200)
            ans = (result.get("llm_response") or {}).get("content") or ""
            done[q["qid"]] = {"qid": q["qid"], "question": q["question"],
                              "answer": ans, "support": q["support"]}
            await rag.finalize_storages()
        except Exception as e:
            done[q["qid"]] = {"qid": q["qid"],
                              "error": f"{type(e).__name__}: {str(e)[:150]}"}
            try:
                await rag.finalize_storages()
            except Exception:
                pass
        _save_done("lightrag", done)
        print(f"[lightrag] {i+1}/{len(todo)} qid={q['qid']} "
              f"{len(done[q['qid']].get('answer') or '')}ch", flush=True)


# ---------------- 官方指标评测 ----------------

def run_eval():
    from mdaqa_common import official_metrics
    questions = {q["qid"]: q for q in load_subset()}
    results = {}
    for arm in ("fulltext", "paperqa", "lightrag"):
        p = _answers_path(arm)
        if not os.path.exists(p):
            print(f"[eval] {arm}: no answers yet, skip")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        # 只评有答案且有 gold 的题
        pairs = [(r.get("answer") or "", questions[r["qid"]]["answer"])
                 for r in rows
                 if r.get("answer") and r["qid"] in questions]
        errs = sum(1 for r in rows if r.get("error"))
        if not pairs:
            print(f"[eval] {arm}: 0 valid answers")
            continue
        m = official_metrics([h for h, _ in pairs], [r for _, r in pairs])
        m["n_answered"] = len(pairs)
        m["n_errors"] = errs
        m["n_total"] = len(rows)
        results[arm] = m
        print(f"[eval] {arm}: {m}", flush=True)
    outp = os.path.join(MDAQA, "official_scores.json")
    json.dump(results, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"saved -> {outp}")


async def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "fulltext"
    ql = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if arm == "fulltext":
        await run_fulltext(ql)
    elif arm == "paperqa":
        await run_paperqa(ql)
    elif arm == "lightrag":
        await run_lightrag(ql)
    elif arm == "eval":
        run_eval()
    else:
        print(__doc__)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
