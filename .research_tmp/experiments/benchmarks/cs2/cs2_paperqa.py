# -*- coding: utf-8 -*-
"""CS2 开放集 PaperQA2 臂（paper-qa 2026.8.12）——三范式对照第三臂。

公平性契约（与 ours/harness 同尺）：
  - LLM：本地 GPUStack Qwen3.8-27B（thinking off）——同 ours 答题模型
  - 检索：Sciverse 开放检索（复用 external_tools 的 tiered engine +
    deep_read 全文三通道+缓存）——同 ours 的外部检索通道，注入为
    paperqa 官方扩展点的 SciverseSearch(NamedTool) 工具
  - 判分：同一套 direct_judge（官方 astabench scorer + GLM-5.3），
    出口为 CS2 sections/citations JSON（与 ours 适配器同形态）

与 Multi-108 闭卷臂（multi_baseline_paperqa.py）的区别：
  - 开放检索：paper_directory = 空目录（agent 必须用 sciverse_search 找
    论文，不能预喂语料——与 ours 的开放检索设定对齐）
  - 答案出口：CS2 JSON（无官方 ctx 序号映射——开放集论文按 title/doi
    出引用，scorer 的 citation 判分吃 snippets/texts）

用法：
  python cs2_paperqa.py --smoke      # 1 题冒烟
  python cs2_paperqa.py              # 5 题（批 15 同题）
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "_shared", "tools"))
_CS2 = os.path.normpath(os.path.join(_HERE))
_SRC = os.path.normpath(os.path.join(_CS2, "..", "..", "..", "..", "src"))
BASE_DIR = os.path.join(_CS2, "arm_paperqa")
ANSWERS = os.environ.get("PQA_CS2_ANSWERS") or os.path.join(
    BASE_DIR, "answers_paperqa_cs2.json")
LEDGER = os.path.join(BASE_DIR, "ledger_paperqa_cs2.jsonl")

sys.path.insert(0, _SRC)
sys.path.insert(0, _SHARED)
os.environ["LLM_CALL_LOG"] = LEDGER
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "16")

MODEL = "Qwen3.8-27B"
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-8b-local")


def _env_from_dotenv():
    env_path = os.path.join(_SRC, "..", ".env")
    wanted = ("LOCAL_BASE_URL", "LOCAL_API_KEY", "EMBEDDING_MODEL")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in wanted:
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Sciverse 开放检索工具（paperqa NamedTool 官方扩展点，clinical_trials 同款）
# ---------------------------------------------------------------------------

from paperqa.agents.tools import NamedTool  # noqa: E402


class SciverseSearch(NamedTool):
    """Open scientific literature search via Sciverse (same retrieval
    channel as the compiled-KB arm: meta-search metadata + agentic-search
    full text). Adds retrieved full texts to the session library so the
    standard evidence/answer tools can consume them."""

    TOOL_FN_NAME = "sciverse_search"
    CONCURRENCY_SAFE = True

    # NamedTool 是 pydantic BaseModel——settings 由类级补丁注入；
    # embedding_model 惰性自解析（settings.get_embedding_model()）
    settings: object = None

    @property
    def _emb(self):
        try:
            return self.settings.get_embedding_model()
        except Exception:
            return None

    async def sciverse_search(self, query: str, state) -> str:
        """
        Search the OPEN scientific literature (Sciverse) for papers matching
        the query, then ADD their full texts to the session library.

        Repeat with different phrasings to broaden coverage. Only papers
        whose full text could be resolved are added (abstract-only hits are
        reported as leads). This tool introduces novel papers — invoke it
        when beginning research or when current evidence is insufficient.

        Args:
            query: A search query — specific phrase, complete sentence, or
                keywords, e.g. 'ontology-based text summarization'.

        Returns:
            String describing added papers and current library status.
        """
        ext = _get_ext()
        t0 = time.time()
        try:
            r = await asyncio.to_thread(ext.search_papers, query, 6)
        except Exception as e:
            return f"sciverse_search error: {str(e)[:150]}"
        papers = r.get("papers") or []
        if not papers:
            return ("SciverseSearch: 0 results. Try different keywords or "
                    "a broader phrasing.")
        added, leads = [], []
        for p in papers:
            title = (p.get("title") or "untitled").strip()
            leads.append(title[:60])
            try:
                full = await asyncio.to_thread(_fetch_full, ext, p)
            except Exception:
                full = None
            if not full or len(full) < 2000:
                continue   # 摘要级命中不入库——evidence 工具需要全文
            doc_name = re.sub(r"[^a-z0-9]+", "_", title.lower())[:80]
            try:
                await state.docs.aadd_texts(
                    texts=[full[:120000]],
                    docname=doc_name,
                    doc={"title": title, "year": p.get("year"),
                         "doi": p.get("doi")},
                    settings=self.settings,
                    embedding_model=self._emb,
                )
                added.append(f"{title[:70]} ({p.get('year') or '?'})")
            except Exception:
                continue
        status = state.status
        if not added:
            return (f"SciverseSearch: {len(papers)} leads, 0 full texts "
                    f"resolved (abstract-only). Leads: "
                    + "; ".join(leads[:4])
                    + f"\n{status}")
        return (f"SciverseSearch added {len(added)} papers: "
                + "; ".join(added[:8])
                + f"\n({time.time()-t0:.0f}s)\n{status}")


_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        from external_tools import ExternalTools
        _EXT = ExternalTools({}, {}, "local:Qwen3.8-27B")
    return _EXT


def _fetch_full(ext, paper_row):
    """单篇全文（复用 ExternalTools 的三通道+文本级缓存）。"""
    pid = paper_row.get("paper_id") or paper_row.get("title") or ""
    m = {"title": paper_row.get("title"), "doi": paper_row.get("doi"),
         "doc_id": paper_row.get("doc_id"),
         "arxiv_id": paper_row.get("arxiv_id")}
    try:
        text, _src = ext._fetch_full_text(pid, m)
        return text
    except Exception:
        return None


def install_sciverse_tool():
    """注册进 paperqa 工具体系。两步：
    ① AVAILABLE_TOOL_NAME_TO_CLASS 注册表加 'sciverse_search'；
    ② make_tools 工厂的 elif 链对未知 NamedTool 抛 NotImplementedError
    （tools.py:133-134 硬编码分支）——包一层补丁：在我们的类上挂
    与 ClinicalTrialsSearch 同构的分支（构造实例+make_tool 绑定方法）。
    原工厂保持不动（frozen upstream），补丁只插一个分支。"""
    from paperqa.agents import tools as pqa_tools
    import paperqa.agents.env as pqa_env

    if getattr(SciverseSearch, "_installed", False):
        return
    pqa_tools.AVAILABLE_TOOL_NAME_TO_CLASS[
        "sciverse_search"] = SciverseSearch
    pqa_env.AVAILABLE_TOOL_NAME_TO_CLASS[
        "sciverse_search"] = SciverseSearch

    import paperqa.agents.env as _env_mod
    # make_tools 是 PaperQAEnvironment 的方法（模块级无此函数）——
    # 类级补丁：先跑原方法，tool_names 里的 'sciverse_search' 先摘除
    # （官方工厂的 elif 链不认识会 NotImplementedError），跑完再把
    # 我们的工具插到列表头。
    _orig_make_tools = _env_mod.PaperQAEnvironment.make_tools

    def make_tools_with_sciverse(self):
        settings = self._settings
        saved = None
        if settings.agent.tool_names:
            saved = list(settings.agent.tool_names)
            if "sciverse_search" in saved:
                settings.agent.tool_names = [
                    t for t in saved if t != "sciverse_search"]
        try:
            tools = _orig_make_tools(self)
        finally:
            if saved is not None:
                settings.agent.tool_names = saved
        # 补挂 SciverseSearch（Tool.from_function 与官方 make_tool 同
        # 一构造；embedding model 从 environment 的 make_tools 上下文
        # 拿不到——SciverseSearch 自己解析）
        from aviary.core import Tool
        inst = SciverseSearch(settings=settings)
        tool = Tool.from_function(
            inst.sciverse_search, concurrency_safe=True)
        tools.insert(0, tool)
        return tools

    _env_mod.PaperQAEnvironment.make_tools = make_tools_with_sciverse
    SciverseSearch._installed = True


# ---------------------------------------------------------------------------
# 配置 + 题目 + 主循环
# ---------------------------------------------------------------------------

def build_settings():
    from paperqa.settings import Settings
    _env_from_dotenv()
    base = (os.environ.get("LOCAL_BASE_URL")
            or "http://127.0.0.1:8000/v1").rstrip("/")
    key = os.environ.get("LOCAL_API_KEY", "local")
    qwen_params = {"model": f"openai/{MODEL}", "api_base": base,
                   "api_key": key, "timeout": 300,
                   "extra_body": {"chat_template_kwargs":
                                  {"enable_thinking": False}}}
    emb_params = {"model": f"openai/{EMBED_MODEL}", "api_base": base,
                  "api_key": key, "timeout": 120}
    llm_cfg = {"model_list": [
        {"model_name": f"openai/{MODEL}", "litellm_params": qwen_params},
        {"model_name": MODEL, "litellm_params": qwen_params}]}
    emb_cfg = {"model_list": [
        {"model_name": f"openai/{EMBED_MODEL}", "litellm_params": emb_params},
        {"model_name": EMBED_MODEL, "litellm_params": emb_params}]}
    settings = Settings(
        llm=f"openai/{MODEL}", llm_config=llm_cfg,
        summary_llm=f"openai/{MODEL}", summary_llm_config=llm_cfg,
        agent={"agent_llm": f"openai/{MODEL}",
               "agent_llm_config": llm_cfg,
               "tool_names": ["sciverse_search", "gather_evidence",
                              "gen_answer"],
               "max_timesteps": 30,
               # 开放模式引导（ClinicalTrialsSearch 同款先例：默认 prompt
               # 极简，27B 不会自发先搜文献——冒烟实测直奔 gather_evidence
               # 空库报错）。第一步必须 sciverse_search 建库。
               "agent_prompt": (
                   "Use the tools to answer the question: {question}\n\n"
                   "IMPORTANT: the session library starts EMPTY. Your FIRST "
                   "action must be sciverse_search(query=...) to find and "
                   "add papers from the open literature — 2-3 sciverse_search "
                   "calls with different phrasings cover a question's topic. "
                   "gather_evidence ONLY works on papers already added. "
                   "When the answer looks sufficient, you can terminate by "
                   "calling the {complete_tool_name} tool. If the answer "
                   "does not look sufficient, and you have already tried to "
                   "answer several times with different evidence, terminate "
                   "by calling the {complete_tool_name} tool. The current "
                   "status of evidence/papers/cost is {status}")},
        embedding=f"openai/{EMBED_MODEL}", embedding_config=emb_cfg,
        verbosity=0, parsing={"use_doc_details": False})
    # 开放模式：paper_directory 为空目录——agent 必须用 sciverse_search
    os.makedirs(os.path.join(BASE_DIR, "empty_docs"), exist_ok=True)
    settings.agent.index.paper_directory = \
        os.path.join(BASE_DIR, "empty_docs")
    settings.agent.index.index_directory = \
        os.path.join(BASE_DIR, "index")
    settings.agent.index.use_absolute_paper_directory = False
    settings.agent.index.sync_with_paper_directory = False
    settings.agent.index.recurse_subdirectories = False
    return settings


def load_cs2_questions():
    """批 15 同 5 题（与 ours 双样本同题对照）。"""
    src = os.path.normpath(os.path.join(
        _CS2, "..", "scholarqa_multi", "sqa2_rubrics_v1_recomputed.json"))
    qs = json.load(open(src, encoding="utf-8"))
    return [{"qid": q["case_id"][:24], "question": q["question"]}
            for q in qs[10:15]]


async def run(smoke: bool = False):
    install_sciverse_tool()
    # Multi-108 版验证过的两件套：lmi router shim（paperqa↔fhlmi 版本
    # 错配）+ litellm ledger hooks（臂纯度账本）
    from multi_baseline_paperqa import (install_litellm_ledger_hooks,
                                        install_lmi_router_shim)
    install_litellm_ledger_hooks()
    install_lmi_router_shim()
    from paperqa import ask
    settings = build_settings()
    questions = load_cs2_questions()
    if smoke:
        questions = questions[:1]

    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in
                json.load(open(ANSWERS, encoding="utf-8"))}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[pqa-cs2] {len(todo)} to answer "
          f"({len(done)} resumed)", flush=True)

    sem = asyncio.Semaphore(int(os.environ.get("PQA_CS2_FANOUT", "2")))
    save_lock = asyncio.Lock()

    async def _one(q):
        qid, text = q["qid"], q["question"]
        t0 = time.time()
        row = {"qid": qid, "question": text}
        try:
            async with sem:
                ans = await asyncio.wait_for(
                    ask(text, settings=settings), timeout=1800)
            # AnswerResponse：session=PQASession（contexts/raw_answer 都在
            # session 上；state 在 environment 上不在 session 上——
            # 首版 AttributeError 修正）
            sess = ans.session
            raw = sess.raw_answer or ""
            row["raw_answer"] = raw
            # 引用桥：session contexts → doc 元数据（title/doi/year）
            # → CS2 citations JSON（scorer 吃 snippets=verbatim 上下文）
            cites = []
            docmeta = {}
            for c in sess.contexts:
                d = getattr(c.text.doc, "doc", None) or {}
                dockey = getattr(c.text.doc, "dockey", "")
                if dockey not in docmeta:
                    docmeta[dockey] = {
                        "title": getattr(d, "title", None) or d.get("title")
                        if isinstance(d, dict) else getattr(d, "title", None),
                        "year": getattr(d, "year", None) or (
                            d.get("year") if isinstance(d, dict) else None),
                        "doi": getattr(d, "doi", None) or (
                            d.get("doi") if isinstance(d, dict) else None),
                        "contexts": []}
                docmeta[dockey]["contexts"].append(
                    c.text.text[:400] if hasattr(c.text, "text")
                    else str(c.text)[:400])
            for i, (dk, m) in enumerate(docmeta.items(), 1):
                cites.append({"id": f"[{i}]",
                              "snippets": [t for t in m["contexts"][:3]],
                              "title": m["title"] or f"doc_{dk[:8]}",
                              "metadata": {"year": m["year"],
                                           "doi": m["doi"],
                                           "dockey": dk}})
            # 答案转 CS2 sections（单节——paperqa 的 raw_answer 是连续
            # Markdown，分节交给判分器按段落处理）
            row["sections"] = [{"title": "Answer", "text": raw,
                                "citations": cites}]
            row["n_contexts"] = len(sess.contexts)
            row["used_contexts"] = len(sess.used_contexts)
        except Exception as e:
            row["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["wall_s"] = round(time.time() - t0, 1)
        async with save_lock:
            done[qid] = row
            os.makedirs(BASE_DIR, exist_ok=True)
            json.dump(list(done.values()), open(ANSWERS, "w",
                                                encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        print(f"[pqa-cs2] {qid[:12]} ans={len(row.get('raw_answer') or '')}ch "
              f"cites={len((row.get('sections') or [{}])[0].get('citations', []))} "
              f"{row['wall_s']}s {row.get('err') or ''}", flush=True)

    await asyncio.gather(*(_one(q) for q in todo))
    print(f"[pqa-cs2] answers saved: {ANSWERS}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(smoke=args.smoke))


if __name__ == "__main__":
    main()
