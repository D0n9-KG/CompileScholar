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
# smoke-10: normalize ALL path constants. The raw joined forms carried
# '..\' segments which leaked into the tantivy index KEYS; paperqa's sync
# then compared them against canonical absolute paths, found 'missing'
# files, and DELETED the whole index (5093 removal lines) on every restart.
_MULTI = os.path.normpath(os.path.join(_HERE, "..", "..", "scholarqa_multi"))
_SRC = os.path.normpath(os.path.join(_MULTI, "..", "..", "..", "..", "src"))
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


def install_lmi_router_shim():
    """smoke-7 fix: paperqa 2026.8.12's make_aviary_tool_selector calls
    get_agent_llm().get_router().acompletion — but fhlmi 1.0.7's LiteLLMModel
    has NO get_router (never had it: verified in 1.0.6/0.48.0 wheels too;
    upstream mismatch). Shim a router whose acompletion routes to the
    module-level litellm.acompletion — the same channel our ledger hooks
    instrument, so purity accounting covers it. Signature per aviary's
    ToolSelector: acompletion(messages=..., tools=..., **kwargs) ->
    ModelResponse with .choices[0].message (content + tool_calls)."""
    from lmi import LiteLLMModel

    if hasattr(LiteLLMModel, "get_router"):
        return  # upstream fixed it — nothing to do

    class _Router:
        @staticmethod
        def _parse_tool_calls(text: str) -> list[dict]:
            """Extract {"tool_calls": [...]} JSON from model text (tolerant:
            fences, prose around the JSON, bare call arrays)."""
            import re as _re
            if not text:
                return []
            # Hermes-style XML fallback FIRST (pure-XML text carries no
            # braces, so the JSON branch below would early-return []):
            # <function>NAME<parameter>k>v</parameter>...</function>
            fx = _re.search(
                r"<function>\s*(\w+)([\s\S]*?)</function>", text)
            if fx:
                args = {}
                for pm in _re.finditer(
                        r"<parameter>\s*(\w+)\s*>\s*([\s\S]*?)\s*</parameter>",
                        fx.group(2)):
                    v = pm.group(2)
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                    args[pm.group(1)] = v
                return [{"name": fx.group(1), "arguments": args}]
            m = _re.search(r"\{[\s\S]*\}", text)  # outermost braces span
            if not m:
                return []
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return []
            if isinstance(obj, dict):
                calls = obj.get("tool_calls")
            elif isinstance(obj, list):
                calls = obj
            else:
                return []
            out = []
            for c in calls or []:
                if isinstance(c, dict) and c.get("name"):
                    out.append({"name": c["name"],
                                "arguments": c.get("arguments") or {}})
            return out

        # aviary binds the model name POSITIONALLY (partial(acompletion,
        # model_name)) then passes messages/tools as kwargs — the old
        # (self, messages, tools=None) signature collided ("got multiple
        # values for argument 'messages'") and every agent tool-selection
        # call died -> every question fell to the "no papers" canned refusal
        # (2026-09-23 root cause, 52 questions wasted before it was caught).
        async def acompletion(self, *args, **kwargs):
            import litellm
            from litellm.types.utils import ModelResponse, Choices, Message
            # positional args come from the partial: (model_name,) — though
            # a bare (messages,) form is tolerated too
            model = kwargs.pop("model", None) or (
                args[0] if args and isinstance(args[0], str) else f"openai/{MODEL}")
            messages = kwargs.pop("messages", None)
            if messages is None and args and not isinstance(args[0], str):
                messages = args[0]
            body = {"model": model, "messages": messages,
                    "temperature": kwargs.pop("temperature", 0.0),
                    # module-level litellm.acompletion bypasses the router
                    # llm_config — inject the local endpoint unless the
                    # caller supplied one (missing creds -> api.openai.com)
                    "api_base": kwargs.pop("api_base", None) or
                                os.environ.get("LOCAL_BASE_URL", "").rstrip("/"),
                    "api_key": kwargs.pop("api_key", None) or
                               os.environ.get("LOCAL_API_KEY", "local")}
            tools = kwargs.pop("tools", None)
            tool_choice = kwargs.pop("tool_choice", None)
            if tools:
                # The GPUStack vLLM server has no --tool-call-parser: ANY
                # native tool-calling is rejected ("tool_choice='required'
                # requires --tool-call-parser"). Emulate tool calling in the
                # prompt: render the schemas, ask for ONE JSON tool call,
                # parse it back into a native tool_calls response. Agent
                # adaptivity (the PaperQA2 method) is fully preserved.
                tool_desc = json.dumps(tools, ensure_ascii=False)
                instr = (
                    "\n\n[TOOL CALLING PROTOCOL] The runtime cannot emit native "
                    "tool_calls. You MUST still select exactly one function, but "
                    "express it as your ENTIRE reply, a single JSON object of "
                    'the form {"tool_calls": [{"name": "<function name>", '
                    '"arguments": {<argument values>}}]} — no prose, no '
                    "markdown fence.\n\nAvailable functions:\n" + tool_desc)
                msgs = [dict(x) if isinstance(x, dict) else dict(x)
                        for x in (messages or [])]
                if msgs and msgs[0].get("role") == "system":
                    msgs[0] = {**msgs[0],
                               "content": str(msgs[0].get("content") or "") + instr}
                else:
                    msgs = [{"role": "system", "content": instr.strip()}] + msgs
                body["messages"] = msgs
                resp = await litellm.acompletion(**body)
                ch = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
                msg = ch.get("message", {}) if isinstance(ch, dict) else ch.message
                text = msg.get("content") if isinstance(msg, dict) else msg.content
                calls = self._parse_tool_calls(text or "")
                if calls:
                    from litellm.types.utils import (ChatCompletionMessageToolCall,
                                                     Function)
                    tcs = [ChatCompletionMessageToolCall(
                        id=f"call_{i}", type="function",
                        function=Function(name=c["name"],
                                          arguments=json.dumps(
                                              c.get("arguments") or {},
                                              ensure_ascii=False)))
                        for i, c in enumerate(calls)]
                    m = Message(role="assistant", content=None, tool_calls=tcs)
                    return ModelResponse(
                        id=resp.get("id", "x") if isinstance(resp, dict) else resp.id,
                        created=0, model=model,
                        choices=[Choices(index=0, finish_reason="tool_calls",
                                         message=m)])
                # no valid tool call parsed -> treat as a plain stop answer
                m = Message(role="assistant", content=text, tool_calls=None)
                return ModelResponse(
                    id=resp.get("id", "x") if isinstance(resp, dict) else resp.id,
                    created=0, model=model,
                    choices=[Choices(index=0, finish_reason="stop", message=m)])
            resp = await litellm.acompletion(**body, **{
                k: v for k, v in kwargs.items()
                if k in ("max_tokens", "timeout", "api_base", "api_key")})
            ch = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
            msg = ch.get("message", {}) if isinstance(ch, dict) else ch.message
            m = Message(role=msg.get("role", "assistant"),
                        content=msg.get("content"),
                        tool_calls=msg.get("tool_calls"))
            return ModelResponse(
                id=resp.get("id", "x") if isinstance(resp, dict) else resp.id,
                created=0, model=model,
                choices=[Choices(index=0,
                                 finish_reason=ch.get("finish_reason", "stop")
                                 if isinstance(ch, dict) else ch.finish_reason,
                                 message=m)])

    LiteLLMModel.get_router = lambda self: _Router()


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
    # lmi drives litellm through the ASYNC path (acompletion/aembedding), and
    # litellm's async handlers read ONLY the _async_* callback lists — a sync
    # callback registered on success_callback alone never fires (measured:
    # zero ledger rows despite thousands of calls). Sync callables in the
    # async list are invoked via customLogger.async_log_event, so the same
    # _cb works on both paths.
    litellm._async_success_callback = [_cb]
    litellm._async_failure_callback = [_cb]


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
    # 2026-09-23 03:25 root fix: use_absolute_paper_directory=True stores
    # ABSOLUTE file_location keys in files.zip while the sync comparison set
    # is ALWAYS relative filenames — structurally every indexed file reads
    # as "extra" and each process restart attempted remove-all (index eroded
    # 430->360 across tonight's restarts, with WinError-5 races on the zip
    # rewrites). Relative keys make the comparison coherent.
    settings.agent.index.use_absolute_paper_directory = False
    # sync=False (default): index is complete and frozen; no add/remove
    # churn. PQA_SYNC=1 (env) enables a one-shot heal pass: re-adds missing
    # files (relative-vs-relative comparison now matches, so nothing is
    # removed). Answering runs must NOT set it.
    settings.agent.index.sync_with_paper_directory = \
        os.environ.get("PQA_SYNC", "0") == "1"
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
    install_lmi_router_shim()
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
    # smoke-8 fix: WAIT for the full index before answering. The first
    # question ran against a half-built index (agent found 41/427 docs and
    # honestly answered 'I cannot answer this question due to having no
    # papers' — correct behavior, wrong precondition). get_directory_index
    # returns a lazy handle; ask() triggers building but does not wait.
    from paperqa.agents.search import get_directory_index
    idx = await get_directory_index(settings=settings, build=True)
    # smoke-11 fix: "index complete" must be a DISK-STABLE verdict, not a
    # volatile in-memory count. The earlier len(files)>=427 check fired on a
    # mid-write read (files.zip hit 430 transiently), released the answer
    # pool against a still-writing index, and every reader collided with the
    # writer (WinError 5) or read half-written zips (zlib -5).
    # 2026-09-23 04:20 fix: the probe used os.listdir(INDEX_DIR)[0] — with
    # sibling dirs (an 'answers' dir, a second pqa_index_* from a settings
    # change) it probed the WRONG index. Resolve the actual runtime index
    # directory from the SearchIndex object itself.
    _IDX_DIR = str((await idx.docs_index_directory).parent)
    _FZIP = os.path.join(_IDX_DIR, "files.zip")
    t0 = time.time()
    stable_rounds = 0
    last_count = -1
    while True:
        files = await idx.index_files
        try:
            mtime = os.path.getmtime(_FZIP)
            import zlib as _z, pickle as _pk
            n_disk = len(_pk.loads(_z.decompress(open(_FZIP, "rb").read())))
        except Exception:
            n_disk, mtime = -1, -1
        if n_disk == last_count and n_disk >= len(key2stem):
            stable_rounds += 1
            if stable_rounds >= 3:  # 3 consecutive reads identical & full
                print(f"[pqa] index STABLE-COMPLETE: {n_disk} files on disk "
                      f"({time.time()-t0:.0f}s wait)", flush=True)
                break
        else:
            stable_rounds = 0
        last_count = n_disk
        if time.time() - t0 > 5400:
            print(f"[pqa] index wait TIMEOUT at {n_disk} stable files — "
                  f"answering anyway (partial index)", flush=True)
            break
        await asyncio.sleep(60)
        print(f"[pqa] index building: disk={n_disk}/{len(key2stem)} "
              f"mem={len(files)} stable={stable_rounds}", flush=True)
    # question pool (2026-09-22): independent questions, N in flight.
    # Each ask() is internally concurrent (agent loop), so fan-out stays modest.
    todo_qs = [q for q in questions
               if not (q["id"] in done and (done[q["id"]].get("answer_official_all")
                                            or done[q["id"]].get("err")))]
    fanout = int(os.environ.get("PQA_QUERY_FANOUT", "3"))
    print(f"[pqa] questions: {len(todo_qs)} to answer, fanout={fanout}", flush=True)
    sem = asyncio.Semaphore(fanout)
    save_lock = asyncio.Lock()

    async def _answer_one(q):
        qid, text = q["id"], q["input"]
        t0 = time.time()
        row = {"qid": qid, "subject": q.get("subject"), "question": text}
        try:
            async with sem:
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
        async with save_lock:
            done[qid] = row
            os.makedirs(BASE_DIR, exist_ok=True)
            json.dump(list(done.values()),
                      open(ANSWERS, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        print(f"[pqa] {qid} {len(row.get('raw_answer') or '')}ch "
              f"cite_map={row.get('citation_mapped')} "
              f"drop={row.get('citation_dropped')} "
              f"{row['wall_s']}s {row.get('err') or ''}", flush=True)

    await asyncio.gather(*(_answer_one(q) for q in todo_qs))
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
