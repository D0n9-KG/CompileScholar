# -*- coding: utf-8 -*-
"""Multi-108 closed-book LightRAG arm (lightrag-hku 1.5.7).

Template = stageB/rebuild27b_2026-09-11/gate2_lightrag.py (proven PS-16 form),
scaled to the 430-paper union corpus + 108 official questions + citation
translation to official ctx indices:

  - LLM: kb_infra.llm.call_local (Qwen3.8-27B, thinking-off, ledger + allowlist
    gate automatic) -> async wrapper for LightRAG's llm_model_func
  - embed: kb_infra.embedding.embed_local (qwen3-embedding-8b, dim probed at
    startup -- never hardcode)
  - citation: LightRAG 1.5.7 emits its own `### References` section ([n] refs
    into its Reference Document List); aquery_llm's raw_data carries
    data.references[{reference_id, file_path}]. file_path == the stem we
    passed to ainsert(ids=[stem]) -- verified at smoke. Markers are rewritten
    to official [ctx_idx] via id_mapping (dual policy, both recorded).

Resume: per-doc marker files for ingestion; per-qid rows in answers json.
DO NOT launch ingestion while the slot marathon holds the GPU (overnight
plan: LightRAG ingestion is the LLM-heavy step, run after slot).

Usage:
  python multi_baseline_lightrag.py --smoke          # 3 docs + 2 questions
  python multi_baseline_lightrag.py --ingest-only    # 430 docs
  python multi_baseline_lightrag.py --query-only     # 108 questions
  python multi_baseline_lightrag.py                  # both
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MULTI = os.path.join(_HERE, "..", "..", "scholarqa_multi")
_SRC = os.path.join(_MULTI, "..", "..", "..", "..", "src")
WORKDIR = os.path.join(_MULTI, "baselines", "lightrag", "work")
ANSWERS = os.path.join(_MULTI, "baselines", "lightrag", "answers_lightrag.json")
LEDGER = os.path.join(_MULTI, "baselines", "lightrag", "ledger_lightrag.jsonl")

sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)
os.environ["LLM_CALL_LOG"] = LEDGER
os.environ.setdefault("LLM_SOCK_TIMEOUT", "300")
os.environ.setdefault("LLM_WALL_TIMEOUT", "600")

from multi_baseline_common import (load_questions, load_id_mapping,
                                   load_corpus_texts, CitationTranslator,
                                   arm_purity)

MODEL = "Qwen3.8-27B"
CITE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def _patch_atomic_write_retry():
    """WinError-5 class fix (2026-09-23 01:36 incident): lightrag's
    atomic_write does a bare os.replace(tmp, target) with NO retry. On
    Windows, a transient handle on the target (Defender real-time scan of the
    freshly-grown vdb file, indexer, backup) makes the rename fail with
    access denied -> IndexFlushError -> the WHOLE ingestion pipeline aborts
    (199 docs failed in one event, all with empty error strings). Retry with
    backoff absorbs the transient handle window. Patched at every binding
    site because lightrag modules use from-import (patching the defining
    module alone rebinds nothing)."""
    import importlib
    import lightrag.file_atomic as fa
    orig = fa.atomic_write

    def retrying(file_name, write_fn, workspace="_"):
        last = None
        for att in range(6):
            try:
                return orig(file_name, write_fn, workspace)
            except PermissionError as e:
                last = e
                time.sleep(0.5 * (att + 1))
        raise last

    mods = [fa]
    # every lightrag module that binds atomic_write via from-import; some
    # (faiss) carry heavy optional deps — bind what exists, skip what doesn't
    for sub in ("kg.nano_vector_db_impl", "kg.json_kv_impl",
                "kg.json_doc_status_impl", "kg.networkx_impl",
                "kg.faiss_impl", "utils"):
        try:
            mods.append(importlib.import_module(f"lightrag.{sub}"))
        except Exception:
            pass
    n = 0
    for mod in mods:
        if getattr(mod, "atomic_write", None) is orig:
            mod.atomic_write = retrying
            n += 1
    print(f"[lrag] atomic_write retry patch bound at {n} sites", flush=True)


_patch_atomic_write_retry()


def _env_local_concurreny():
    # LightRAG's internal concurrency is modest; the call_local semaphore
    # (LOCAL_MAX_CONCURRENT, runner sets 32) is the real guard.
    os.environ.setdefault("LOCAL_MAX_CONCURRENT", "32")


async def llm_local(prompt, system_prompt=None, history_messages=[],
                    keyword_extraction=False, **kwargs):
    from kb_infra.llm import call_local
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    try:
        out = await asyncio.wait_for(
            asyncio.to_thread(
                call_local, prompt, MODEL,
                8000 if keyword_extraction else 6000, 0.0, None, False),
            timeout=600)
    except asyncio.TimeoutError:
        print("  [lrag-llm-timeout]", flush=True)
        return ""
    return out or ""


async def embed_local_async(texts):
    import numpy as np
    from kb_infra.embedding import embed_local
    embs = await asyncio.to_thread(embed_local, list(texts))
    return np.array(embs) if embs else np.array([])


def probe_embed_dim() -> int:
    from kb_infra.embedding import embed_local
    e = embed_local(["dimension probe"])
    if not e:
        raise RuntimeError("local embedding probe failed")
    return len(e[0])


def extract_citation_markers(answer: str,
                             references: list[dict]) -> list[tuple[str, list[str]]]:
    """[(marker, [stems])] for every [n] / [n,m] marker in the answer.
    LightRAG's reference ids are ints starting at 1; the answer's References
    section uses the same [n]. Multi-number markers map each number to its
    stem (list form -> one official multi-ref marker). Unresolvable numbers
    (LLM drift) contribute no stems; a fully-unresolvable marker yields an
    empty list and is dropped by the translator (recorded, not silent)."""
    ref_by_id = {}
    for r in references or []:
        rid = r.get("reference_id")
        fp = r.get("file_path", "")
        if rid is None:
            continue
        # file_path: the stem we passed as ids=[...]; tolerate path forms
        stem = os.path.basename(str(fp)).rsplit(".", 1)[0]
        ref_by_id[int(rid)] = stem
    out = []
    for m in CITE_RE.finditer(answer):
        inner = m.group(1)
        nums = [int(x.strip()) for x in inner.split(",")]
        stems = [ref_by_id[n] for n in nums if n in ref_by_id]
        out.append((m.group(0), stems))
    return out


async def run(smoke: bool = False, ingest_only: bool = False,
              query_only: bool = False, q_limit: int | None = None):
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc

    os.makedirs(WORKDIR, exist_ok=True)
    dim = probe_embed_dim()
    print(f"[lrag] local embedding dim={dim}", flush=True)

    rag = LightRAG(
        working_dir=WORKDIR,
        llm_model_func=llm_local,
        embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=8192,
                                     func=embed_local_async),
    )
    await rag.initialize_storages()

    corpus = load_corpus_texts()
    if smoke:
        corpus = corpus[:3]
    if not query_only:
        # Ingestion with COMPLETION-verified markers (2026-09-22):
        # the first parallel version wrote markers when ainsert() RETURNED —
        # but LightRAG 1.5.7's ainsert only enqueues ("Request queued"); the
        # user caught 406 false "done" markers while doc_status showed 25
        # processed / 285 pending. Markers are now written only after
        # status polling confirms 'processed' for that doc id.
        # Fanout=1: LightRAG's own pipeline is the concurrency source; the
        # outer fan-out only flooded the queue and lied about completion.
        n = 0
        done_count = sum(1 for s, _ in corpus
                         if os.path.exists(os.path.join(WORKDIR, f"_marker_{s}.done")))
        todo = [(s, t) for s, t in corpus
                if not os.path.exists(os.path.join(WORKDIR, f"_marker_{s}.done"))]
        print(f"ingest: {done_count} done (verified markers), {len(todo)} to go",
              flush=True)

        async def _wait_processed(rag, stem, timeout_s=1800):
            """Poll doc_status until this doc id reports processed (or
            failed/timeout -> no marker, retried on next resume)."""
            import asyncio as _aio
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                try:
                    st = await rag.doc_status.get_docs(ids=[stem], include_content=False)
                    row = st.get(stem) if isinstance(st, dict) else None
                    status = (getattr(row, "status", None) or
                              (row.get("status") if isinstance(row, dict) else None))
                    sstr = str(status).lower() if status else ""
                    if "processed" in sstr or "failed" in sstr:
                        return "processed" in sstr
                except Exception:
                    pass
                await _aio.sleep(5)
            return False

        for stem, text in todo:
            t0 = time.time()
            # smoke-2 fix: ids= only sets the doc key; file_paths= sets the
            # file_path field the query-time reference list is built from
            # (ids-only leaves file_path=unknown_source -> references=[] ->
            # citation bridge dead). Pass BOTH.
            await rag.ainsert(text, ids=[stem], file_paths=[stem])
            ok = await _wait_processed(rag, stem)
            if ok:
                open(os.path.join(WORKDIR, f"_marker_{stem}.done"), "w").write("1")
                n += 1
                print(f"[ingest] {stem[:60]} ({n}/{len(todo)}) "
                      f"{time.time()-t0:.0f}s PROCESSED", flush=True)
            else:
                print(f"[ingest] {stem[:60]} NOT PROCESSED in time "
                      f"({time.time()-t0:.0f}s) — no marker, will retry",
                      flush=True)
        print(f"ingest done (new={n})", flush=True)

    if ingest_only or smoke:
        await rag.finalize_storages()
        if smoke:
            return await run_queries(rag, smoke=True)
        return

    await run_queries(rag, q_limit=q_limit)
    await rag.finalize_storages()


async def run_queries(rag, smoke: bool = False, q_limit: int | None = None):
    from lightrag import QueryParam  # smoke-1 fix: was only imported in run()
    questions = load_questions()
    id_mapping = load_id_mapping()
    if smoke:
        questions = questions[:2]
    if q_limit:
        questions = questions[:q_limit]

    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in json.load(open(ANSWERS, encoding="utf-8"))}

    stats = {"mapped": 0, "dropped": 0}
    # question pool (2026-09-22): questions are independent; fly N at once.
    # Each aquery_llm is itself multi-call internally (keywords+answer), so a
    # modest fan-out (4) keeps total in-flight near the GPU sweet spot.
    todo_qs = [q for q in questions
               if not (q["id"] in done and (done[q["id"]].get("answer_official_all")
                                            or done[q["id"]].get("err")))]
    fanout = int(os.environ.get("LRAG_QUERY_FANOUT", "6"))
    print(f"questions: {len(todo_qs)} to answer, fanout={fanout}", flush=True)
    sem = asyncio.Semaphore(fanout)
    _save_lock = asyncio.Lock()

    async def _answer_one(q):
        qid, text = q["id"], q["input"]
        t0 = time.time()
        row = {"qid": qid, "subject": q.get("subject"), "question": text}
        try:
            async with sem:
                result = await asyncio.wait_for(
                    rag.aquery_llm(text, QueryParam(mode="hybrid")),
                    timeout=1200)
            answer = (result.get("llm_response") or {}).get("content") or ""
            refs = ((result.get("data") or {}).get("references")) or []
            row["raw_answer"] = answer
            row["references"] = refs
            tr = CitationTranslator(qid, id_mapping)
            pairs = extract_citation_markers(answer, refs)
            row["answer_official_all"], row["citations_all"] = tr.translate(
                answer, pairs, policy="all")
            row["answer_official_first"], _ = tr.translate(
                answer, pairs, policy="first")
            row["citation_mapped"] = tr.mapped
            row["citation_dropped"] = tr.dropped
            row["unresolved_markers"] = sum(1 for _, s in pairs if not s)
            stats["mapped"] += tr.mapped
            stats["dropped"] += tr.dropped
        except Exception as e:
            row["err"] = str(e)[:200]
        row["wall_s"] = round(time.time() - t0, 1)
        async with _save_lock:
            done[qid] = row
            os.makedirs(os.path.dirname(ANSWERS), exist_ok=True)
            json.dump(list(done.values()),
                      open(ANSWERS, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        print(f"[q] {qid} {len(row.get('raw_answer') or '')}ch "
              f"cite_map={row.get('citation_mapped')} "
              f"drop={row.get('citation_dropped')} "
              f"{row['wall_s']}s {row.get('err') or ''}", flush=True)

    await asyncio.gather(*(_answer_one(q) for q in todo_qs))
    print(f"[lrag] answers saved: {ANSWERS}", flush=True)
    print(f"[lrag] citation translation: mapped={stats['mapped']} "
          f"dropped={stats['dropped']}", flush=True)
    purity = arm_purity(LEDGER, min_calls=len(todo_qs))
    print(f"[lrag] arm purity: {purity}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--query-only", action="store_true")
    ap.add_argument("--q-limit", type=int, default=None)
    args = ap.parse_args()
    _env_local_concurreny()
    asyncio.run(run(smoke=args.smoke, ingest_only=args.ingest_only,
                    query_only=args.query_only, q_limit=args.q_limit))


if __name__ == "__main__":
    main()
