# -*- coding: utf-8 -*-
"""Multi-108 closed-book PaperQA2 arm (paper-qa 2026.8.12).

Template = PS-era pilot_a6_paperqa.py (proven), re-pointed at the local
GPUStack backend (same model as our build = fairness) and scaled to the
430-paper union corpus + 108 official questions + official citation format:

  - LLM: litellm `openai/Qwen3.8-27B` -> LOCAL_BASE_URL, thinking-off via
    chat_template_kwargs (Qwen-family form, the only verified one).
    Usage logging: litellm success/failure callbacks -> kb_infra-format
    ledger records (arm purity checkable; embedding calls logged with their
    own model name and whitelisted).
  - embed: litellm `openai/qwen3-embedding-8b-local` -> LOCAL_BASE_URL.
  - citation bridge (deterministic, no name guessing): paperqa dockey =
    md5(file bytes); precompute dockey->stem for the 430 corpus files, then
    session contexts' doc.dockey -> stem -> official ctx indices via
    id_mapping. Raw-answer parentheticals `(pqac-xxxx; ...)` are rewritten to
    official `[i,j]` markers (dual one-to-many policy, both recorded).

Resume: answers json per qid (PS-era pattern). Index built once into
baselines/paperqa/index (embedding-heavy; LLM-light since use_doc_details is
off — citation-peek calls only).

Usage:
  python multi_baseline_paperqa.py --smoke      # 3 docs + 2 questions
  python multi_baseline_paperqa.py              # index + 108 questions
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MULTI = os.path.join(_HERE, "..", "..", "scholarqa_multi")
_SRC = os.path.join(_MULTI, "..", "..", "..", "..", "src")
BASE_DIR = os.path.join(_MULTI, "baselines", "paperqa")
DOCS = os.path.join(_MULTI, "corpus", "texts")
INDEX_DIR = os.path.join(BASE_DIR, "index")
ANSWERS = os.path.join(BASE_DIR, "answers_paperqa.json")
LEDGER = os.path.join(BASE_DIR, "ledger_paperqa.jsonl")

sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)
os.environ["LLM_CALL_LOG"] = LEDGER
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "32")

from multi_baseline_common import (load_questions, load_id_mapping,
                                   CitationTranslator, arm_purity)

MODEL = "Qwen3.8-27B"
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-8b-local")
LOCAL_BASE = os.environ.get("LOCAL_BASE_URL", "").rstrip("/")


def _env_from_dotenv():
    """Load .env into os.environ for the keys this harness needs. PaperQA's
    litellm path reads os.environ directly (kb_infra's ENV dict does NOT
    propagate to it — smoke-3 fix: EMBEDDING_MODEL was None -> model
    'openai/None' -> 404)."""
    env_path = os.path.join(_SRC, "..", ".env")
    wanted = ("LOCAL_BASE_URL", "LOCAL_API_KEY", "EMBEDDING_MODEL")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in wanted:
                    os.environ.setdefault(k, v)


def dockey_stem_map(docs_dir: str = DOCS) -> dict:
    """dockey(md5 of file bytes) -> stem, precomputed for the whole corpus."""
    out = {}
    for fn in sorted(os.listdir(docs_dir)):
        if fn.endswith((".md", ".txt")):
            h = hashlib.md5(
                open(os.path.join(docs_dir, fn), "rb").read()).hexdigest()
            out[h] = fn.rsplit(".", 1)[0]
    return out


def install_litellm_ledger_hooks():
    """Log every litellm chat/embedding call into the arm ledger in the
    kb_infra _log_call format (ts/provider/model/ok/usage). Purity check
    then sees exactly the pairs we whitelist.

    Also registers our custom model names in litellm.model_cost (smoke-6):
    lmi's embedding._truncate_if_large reads
    model_cost[name]["max_input_tokens"]; an unknown model name returns
    None from the dict-writer (None * 3 -> TypeError inside the embedding
    TaskGroup), which surfaced as 'unhandled errors in a TaskGroup' and a
    truncated index. Registering a max_input_tokens entry fixes the crash
    AND enables correct truncation for our 4096-dim embedder."""
    import litellm
    from kb_infra.llm import _log_call

    for _n in ("openai/qwen3-embedding-8b-local", "qwen3-embedding-8b-local"):
        # full entry incl. max_input_tokens (smoke-6b: the first fix omitted
        # it -> lmi read model_cost[name]["max_input_tokens"] -> KeyError
        # 'max_input_tokens' — same TaskGroup, third face of the same bug)
        litellm.model_cost[_n] = {
            "max_tokens": 8192, "max_input_tokens": 8192,
            "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
            "litellm_provider": "openai", "mode": "embedding"}

    def _cb(kwargs, completion_response=None, start_time=None, end_time=None):
        try:
            call_type = kwargs.get("litellm_params", {}).get(
                "call_type", "acompletion")
            model = kwargs.get("model") or ""
            usage = None
            ok = True
            if isinstance(completion_response, Exception) or (
                    kwargs.get("exception")):
                ok = False
            elif completion_response is not None:
                usage = getattr(completion_response, "usage", None)
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                if isinstance(usage, dict):
                    usage = {"prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens")}
            prov_model = model.split("/")[-1]
            if call_type in ("acompletion", "completion"):
                _log_call("local", prov_model, ok, 0.0, usage or {}, 0)
            elif call_type in ("aembedding", "embedding"):
                _log_call("local", prov_model, ok, 0.0,
                          {"prompt_tokens": (usage or {}).get("prompt_tokens"),
                           "completion_tokens": 0}, 0)
        except Exception:
            pass

    litellm.success_callback = [_cb]
    litellm.failure_callback = [_cb]


def build_settings():
    from paperqa import Settings
    # smoke-5 fix: read endpoints at CALL time, not import time. The old
    # module-level LOCAL_BASE was '' when .env wasn't in os.environ at import
    # -> litellm treated api_base='' as unset -> hit api.openai.com with the
    # GPUStack key -> AuthenticationError -> every index citation-peek died
    # -> truncated files.zip (misdiagnosed twice as an index/zlib bug).
    base = os.environ.get("LOCAL_BASE_URL", LOCAL_BASE).rstrip("/")
    if not base:
        raise RuntimeError("LOCAL_BASE_URL unresolved — call _env_from_dotenv() first")
    key = os.environ.get("LOCAL_API_KEY", "local")
    qwen_params = {"model": f"openai/{MODEL}", "api_base": base,
                   "api_key": key, "temperature": 0.0, "max_tokens": 6000,
                   "timeout": 300,
                   "extra_body": {"chat_template_kwargs":
                                  {"enable_thinking": False}}}
    emb_model = os.environ.get("EMBEDDING_MODEL", EMBED_MODEL)
    emb_params = {"model": f"openai/{emb_model}", "api_base": base,
                  "api_key": key, "timeout": 120}
    emb_params.pop("dimensions", None) if "dimensions" in emb_params else None
    llm_cfg = {"model_list": [
        {"model_name": f"openai/{MODEL}", "litellm_params": qwen_params},
        {"model_name": MODEL, "litellm_params": qwen_params}]}
    emb_cfg = {"model_list": [
        {"model_name": f"openai/{emb_model}", "litellm_params": emb_params},
        {"model_name": emb_model, "litellm_params": emb_params}]}
    settings = Settings(
        llm=f"openai/{MODEL}", llm_config=llm_cfg,
        summary_llm=f"openai/{MODEL}", summary_llm_config=llm_cfg,
        agent={"agent_llm": f"openai/{MODEL}",
               "agent_llm_config": llm_cfg},
        embedding=f"openai/{emb_model}", embedding_config=emb_cfg,
        verbosity=0, parsing={"use_doc_details": False})
    # home-proven index config (PS-era smoke v1 bug lesson)
    settings.agent.index.paper_directory = DOCS
    settings.agent.index.index_directory = INDEX_DIR
    settings.agent.index.use_absolute_paper_directory = True
    settings.agent.index.sync_with_paper_directory = True
    settings.agent.index.recurse_subdirectories = False
    # smoke-4 fix: indexing 430 files with the default concurrency=5 +
    # batch_size=1 can leave files.zip half-written if the process dies
    # mid-index (measured: 9KB truncated zip -> 107 questions instant-died on
    # 'Error -5 decompressing'). Batch commits make the index durable sooner.
    settings.agent.index.batch_size = 10
    return settings


PAREN_RE = re.compile(r"\(([^()]*pqac-[a-zA-Z0-9]{8}[^()]*)\)")
PQAC_RE = re.compile(r"\bpqac-[a-zA-Z0-9]{8}\b")


def extract_citation_markers(raw_answer: str,
                              id2ctx: dict) -> list[tuple[str, list[str]]]:
    """[(marker, [stems])] for every parenthetical citation group. `marker`
    is the WHOLE group; the translator replaces the group with one official
    multi-ref [i,j] marker (papers cited together in one parenthetical,
    matching the official [2,3] form). Groups with no mappable id produce an
    empty stem list and are dropped (out-of-corpus citation = protocol
    discard)."""
    out = []
    for m in PAREN_RE.finditer(raw_answer):
        ids = PQAC_RE.findall(m.group(1))
        stems = [id2ctx[i] for i in ids if id2ctx.get(i)]
        out.append((m.group(0), stems))
    return out


async def run(smoke: bool = False, q_limit: int | None = None):
    from paperqa import ask
    _env_from_dotenv()
    install_litellm_ledger_hooks()
    settings = build_settings()

    key2stem = dockey_stem_map()
    print(f"[pqa] dockey->stem map: {len(key2stem)} files", flush=True)

    questions = load_questions()
    id_mapping = load_id_mapping()
    if smoke:
        questions = questions[:2]
    if q_limit:
        questions = questions[:q_limit]

    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in json.load(open(ANSWERS, encoding="utf-8"))}

    stats = {"mapped": 0, "dropped": 0, "unmapped_groups": 0}
    for q in questions:
        qid, text = q["id"], q["input"]
        if qid in done and (done[qid].get("answer_official_all")
                            or done[qid].get("err")):
            continue
        t0 = time.time()
        row = {"qid": qid, "subject": q.get("subject"), "question": text}
        try:
            ans = await asyncio.wait_for(
                ask(text, settings=settings), timeout=1800)
            sess = ans.session
            raw = sess.raw_answer or ""
            row["raw_answer"] = raw
            # deterministic dockey bridge: pqac id -> context -> doc.dockey -> stem
            id2ctx = {}
            for c in sess.contexts:
                dk = getattr(c.text.doc, "dockey", None)
                stem = key2stem.get(dk)
                if stem:
                    id2ctx[c.id] = stem
            row["used_contexts"] = sorted(sess.used_contexts)
            row["context_stems"] = [id2ctx.get(i) for i in row["used_contexts"]]
            tr = CitationTranslator(qid, id_mapping)
            pairs = extract_citation_markers(raw, id2ctx)
            # one marker per group: translate drops unmapped groups' text
            row["answer_official_all"], row["citations_all"] = tr.translate(
                raw, pairs, policy="all")
            row["answer_official_first"], _ = tr.translate(
                raw, pairs, policy="first")
            row["citation_mapped"] = tr.mapped
            row["citation_dropped"] = tr.dropped
            stats["mapped"] += tr.mapped
            stats["dropped"] += tr.dropped
            stats["unmapped_groups"] += sum(1 for _, stems in pairs if not stems)
        except Exception as e:
            row["err"] = str(e)[:200]
        row["wall_s"] = round(time.time() - t0, 1)
        done[qid] = row
        os.makedirs(BASE_DIR, exist_ok=True)
        json.dump(list(done.values()),
                  open(ANSWERS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[pqa] {qid} {len(row.get('raw_answer') or '')}ch "
              f"cite_map={row.get('citation_mapped')} "
              f"drop={row.get('citation_dropped')} "
              f"{row['wall_s']}s {row.get('err') or ''}", flush=True)

    print(f"[pqa] answers saved: {ANSWERS}", flush=True)
    print(f"[pqa] citation translation: mapped={stats['mapped']} "
          f"dropped={stats['dropped']} "
          f"unmapped_groups={stats['unmapped_groups']}", flush=True)
    purity = arm_purity(LEDGER)
    print(f"[pqa] arm purity: {purity}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--q-limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(smoke=args.smoke, q_limit=args.q_limit))


if __name__ == "__main__":
    main()
