# -*- coding: utf-8 -*-
"""Answer pipeline (moved from benchmarks/_shared/tools/answer_pipeline.py; behaviour pinned byte for byte by
tests/test_characterize_answer_path.py).

One straight line per question:
  1. probe    : search the question itself (external + KB) so planning knows which field / sense of the terms is meant
  2. plan     : 2-6 sections derived from what the question asks, 2-3 queries each
  3. gather   : per section, in parallel — KB hybrid retrieval, field-state channel, external semantic search,
                optional citation expansion (co-citation over the seeds' reference lists)
  4. screen   : drop clearly off-topic excerpts (same word, different meaning)
  5. write    : each section cites evidence ids sentence by sentence
  6. assemble : map ids to CS2 citations deterministically (snippet = the evidence's verbatim text)
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import re
import threading
import time

from ..core.cutoff import allowed as _cut_allowed, raw_cutoff as _cut_raw, set_thread_cutoff as _set_cut
from ..kb.store import KB
from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from ..sources import refgraph
from ..sources.sciverse import SciverseClient
from .prompts import FIELD_BLOCK, PLAN, PROBE_BLOCK, SCREEN, WRITE

# Sciverse 30 req/min account quota: queue for a token rather than fire, get 429 and lose the whole external channel
os.environ.setdefault("SCIVERSE_MAX_WAIT_S", "600")
os.environ.setdefault("SCIVERSE_SHARED_BUCKET", os.path.join(os.path.expanduser("~"), ".sciverse_bucket.json"))

MODEL = os.environ.get("ANSWER_MODEL", "Qwen3.8-27B")


def chat(prompt: str, max_tokens: int = 6000, temperature: float = 0.2) -> str:
    out = call_local(prompt, model=MODEL, max_tokens=max_tokens, temperature=temperature,
                     enable_thinking=False)
    return out or ""


# ---------------------------------------------------------------- 外部检索
_SV = None


def _sciverse():
    """Direct Sciverse semantic-search client (not the tiered SearchService path: its 15 s slot + circuit breaker
    turned quota queueing into "source failed" under concurrent answering — measured 15 of 45 concurrent queries
    empty). Queues on the token bucket (SCIVERSE_MAX_WAIT_S); the cutoff is enforced server-side by year."""
    global _SV
    if _SV is None:
        _SV = SciverseClient(timeout_seconds=60)
    return _SV


def ext_search(q: str, k: int = 8) -> list[dict]:
    last_err = None
    for attempt in range(3):
        try:
            cands = _sciverse().semantic_search(q, limit=max(k, 10))
        except Exception as e:
            last_err = str(e)[:120]
            time.sleep(3 * (attempt + 1))
            continue
        failed = [c for c in cands if c.status != "ready"]
        if failed and not [c for c in cands if c.status == "ready"]:
            last_err = (failed[0].error_summary or "failed")[:120]
            time.sleep(3 * (attempt + 1))
            continue
        out = []
        for c in cands:
            if c.status != "ready" or not c.title:
                continue
            if not _cut_allowed(c.year):
                continue  # 服务端已按年过滤；客户端再兜底（含线程局部截止）
            ab = ((c.raw or {}).get("abstract") or "").strip()
            if len(ab) < 40:
                continue
            out.append({"src": "ext", "paper_key": "ext:" + hashlib.md5(c.title.lower().encode()).hexdigest()[:12],
                        "title": c.title, "year": c.year, "doi": None, "snippet": ab[:1200]})
            if len(out) >= k:
                break
        return out
    return [{"src": "ext", "error": last_err or "unknown"}]


def cite_expand(seed_titles: list[str], k: int = 10, max_seeds: int = 8, budget_s: float = 90.0) -> list[dict]:
    """种子论文的参考文献，按被几篇种子共同引用排序（p8：同候选池召回约为语义检索 2 倍）。

    10-03 改为多来源参考文献图（refgraph：arXiv HTML 主 + Crossref 兜底），不再依赖 S2——无 key 的 S2 / OpenAlex
    按出口 IP 共享免费额度，实测并行即被封（S2 20/20 次 429；OpenAlex 当日额度耗尽）。共引排序逻辑不变；
    参考文献条目只有题录，排序后前 k 条再用 arXiv 精确标题查摘要（查不到摘要的不收：证据必须有可引用原文）。
    budget_s：本次扩展的墙钟上限；命中缓存不耗时。"""
    from collections import Counter
    freq, info = Counter(), {}
    deadline = time.time() + budget_s
    for t in seed_titles[:max_seeds]:
        if time.time() > deadline:
            break
        refs, _src = refgraph.references(t, deadline=deadline)
        seen = set()
        for r in refs:
            key = refgraph.norm(r["title"])
            if len(key.split()) < 2 or key in seen:
                continue
            seen.add(key)
            if (r.get("year") or r.get("date")) and not _cut_allowed(r.get("year"), r.get("date")):
                continue
            freq[key] += 1
            # 同一篇被多个来源列出时，保留带摘要的那条
            if key not in info or (r.get("abstract") and not info[key].get("abstract")):
                info[key] = dict(r)
    top = [info[key] | {"_n": n} for key, n in freq.most_common() if n >= 2][:k * 2]
    if top and time.time() < deadline:
        refgraph.resolve_abstracts([r for r in top if not r.get("abstract")], deadline=deadline)
    out = []
    for r in top:
        if len(out) >= k:
            break
        if not r.get("abstract") or len(r["abstract"]) < 80:
            continue  # 证据必须有可引用原文
        if not _cut_allowed(r.get("year"), r.get("date")):
            continue
        out.append({"src": "cite", "paper_key": "ext:" + hashlib.md5(r["title"].lower().encode()).hexdigest()[:12],
                    "title": r["title"], "year": r.get("year"), "arxiv": (r.get("ids") or {}).get("arxiv"),
                    "snippet": r["abstract"][:1200], "co_cited_by": r["_n"]})
    return out


# ---------------------------------------------------------------- 1. plan
def probe(question: str, kb: KB | None, use_ext: bool = True, k: int = 10) -> list[dict]:
    """先检索再规划：用原题直接检一次（KB+外部），供 plan 判断题目所指的领域/词义。
    10-03 dev 失分分析：盲拆计划时 27B 按先验猜词义（"tree covering"→图论覆盖、"LLM 辅助写 SQL"→泛 text-to-SQL），
    整节检索与写作都跟着跑题——20/74 节 ≥75% 段落被判无关，占全部无关段 61%。"""
    with cf.ThreadPoolExecutor(2, initializer=_set_cut, initargs=(_cut_raw(),)) as ex:
        fk = ex.submit(kb.search, question, k) if kb is not None else None
        fe = ex.submit(ext_search, question, k) if use_ext else None
        # 外部语义检索在前：KB 是 BM25+向量混合，整句题面的 BM25 会被泛词带偏（实测 SQL 题命中红队综述）
        rows = (fe.result() if fe else []) + (fk.result() if fk else [])
    seen, out = set(), []
    for r in rows:
        if r.get("error") or r["title"].lower() in seen:
            continue
        seen.add(r["title"].lower())
        out.append(r)
    return out


class Degradation:
    """Per-question counters of silent fallbacks (W1-10). Every stage below degrades instead of failing when the LLM
    returns nothing usable; these counters make that visible in the trace and in run health summaries."""
    KEYS = ("plan_unparseable", "plan_fallback", "screen_unparseable", "screen_kept_all", "write_empty",
            "ext_query_failed", "cite_section_empty")

    def __init__(self):
        self._lock = threading.Lock()
        self.counts = {k: 0 for k in self.KEYS}

    def add(self, key: str, n: int = 1):
        with self._lock:
            self.counts[key] += n


def plan(question: str, probe_rows: list[dict] | None = None, deg: Degradation | None = None,
         extra: str = "") -> list[dict]:
    """extra: an additional context block appended to the probe block (literature-layer mode passes the field map);
    empty keeps the prompt byte-identical to the legacy path."""
    pb = ""
    if probe_rows:
        pb = PROBE_BLOCK.format(items="\n".join(f"- {r['title'][:150]}: {r['snippet'][:220]}" for r in probe_rows[:16]))
    pb += extra
    for t in (0.0, 0.3):
        obj = parse_json_response(chat(PLAN.format(question=question, probe=pb), max_tokens=1500, temperature=t))
        secs = (obj or {}).get("sections") if isinstance(obj, dict) else None
        if isinstance(secs, list) and secs:
            out = []
            for s in secs[:6]:
                if isinstance(s, dict) and s.get("title") and s.get("queries"):
                    out.append({"title": str(s["title"])[:120], "goal": str(s.get("goal") or "")[:300],
                                "queries": [str(q)[:200] for q in s["queries"][:3]]})
            if out:
                return out
        if deg:
            deg.add("plan_unparseable")
    if deg:
        deg.add("plan_fallback")
    return [{"title": "Overview", "goal": question, "queries": [question]}]


# ---------------------------------------------------------------- 2-3. retrieve + evidence table
def gather(question: str, sections: list[dict], kb: KB | None, use_ext=True, use_cite=True,
           kb_k=10, ext_k=8, cite_k=8, cutoff: str | None = None, use_state: bool = True) -> tuple[list[dict], dict]:
    jobs = []
    # 检索线程池里的线程看不到调用方的线程局部截止——每个工作线程启动时重设
    with cf.ThreadPoolExecutor(6, initializer=_set_cut, initargs=(cutoff,)) as ex:
        for si, s in enumerate(sections):
            for q in s["queries"]:
                if kb is not None:
                    jobs.append((si, "kb", q, ex.submit(kb.search, q, kb_k)))
                    if use_state:
                        jobs.append((si, "state", q, ex.submit(kb.state_search, q)))
                if use_ext:
                    jobs.append((si, "ext", q, ex.submit(ext_search, q, ext_k)))
        raw = [(si, src, q, f.result()) for si, src, q, f in jobs]
    t_ret = time.time()
    if use_cite:
        # 引文扩展（10-03 多来源版）：每节取外部命中前 6 篇作种子；先对全题种子做一次 OpenAlex 批量解析
        # （refgraph.prefetch：6 个标题/次 = 10 credits，参考文献元数据 1 credit/50 篇），各节再按共引排序。
        seeds_by_sec = {si: [r["title"] for (sj, src, _, rows) in raw if sj == si and src == "ext"
                             for r in rows if not r.get("error")][:6] for si in range(len(sections))}
        refgraph.prefetch([t for v in seeds_by_sec.values() for t in v], deadline=time.time() + 120)
        for si, s in enumerate(sections):
            sec_seeds = seeds_by_sec[si]
            raw.append((si, "cite", "cite-expand", cite_expand(sec_seeds, k=cite_k, max_seeds=6) if sec_seeds else []))
    evidence, by_text, trace = [], {}, {"calls": []}
    for si, src, q, rows in raw:
        trace["calls"].append({"section": si, "src": src, "query": q, "n": len(rows),
                               "errors": [r.get("error") for r in rows if r.get("error")]})
        for r in rows:
            if r.get("error"):
                continue
            key = (r["paper_key"], r["snippet"][:160])
            if key in by_text:
                by_text[key]["sections"].add(si)
                continue
            e = {**r, "eid": f"E{len(evidence) + 1}", "sections": {si}}
            by_text[key] = e
            evidence.append(e)
    trace["n_evidence"] = len(evidence)
    trace["t_cite_s"] = round(time.time() - t_ret, 1)
    trace["by_src"] = {s: sum(1 for e in evidence if e["src"] == s) for s in ("kb", "state", "ext", "cite")}
    return evidence, trace


# ---------------------------------------------------------------- 4. write
def _fmt_ev(items: list[dict]) -> str:
    lines = []
    for e in items:
        yr = f", {e['year']}" if e.get("year") else ""
        lines.append(f"[{e['eid']}] ({e['title'][:150]}{yr}) {e['snippet']}")
    return "\n".join(lines)


def screen(question: str, goal: str, items: list[dict], batch: int = 20, min_keep: int = 5,
           deg: Degradation | None = None) -> list[dict]:
    """证据相关性筛选：只剔除"同词异义/无关领域/无内容碎片"（检索召回 ≠ 相关；旧管线无此步）。
    v1 用"保留列表"，实测 27B 过严（fb607 题 245 条全拒）→ 改为"剔除列表 + 存疑不剔"；
    剔除后少于 min_keep 条则回退到不筛（宁可多给写作器材料，也不让一节空掉）。失败时不丢证据。"""
    drop = set()
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        txt = "\n".join(f"[{e['eid']}] ({e['title'][:120]}) {e['snippet'][:400]}" for e in chunk)
        obj = parse_json_response(chat(SCREEN.format(question=question, goal=goal, items=txt),
                                       max_tokens=800, temperature=0.0))
        ids = (obj or {}).get("off_topic") if isinstance(obj, dict) else None
        if isinstance(ids, list):
            drop |= {str(x).strip("[] ") for x in ids}
        elif deg:
            deg.add("screen_unparseable")
    kept = [e for e in items if e["eid"] not in drop or e.get("src") == "target"]
    if len(kept) >= min(min_keep, len(items)):
        return kept
    if deg:
        deg.add("screen_kept_all")
    return items


def _interleave(items: list[dict]) -> list[dict]:
    """按来源轮转排序（来源内保持检索排名）。证据表按"每条检索式：kb→state→ext"插入、引文扩展最后追加，
    原先直接截前 max_ev 条——10-03 实测每节前 40 条里 ext 只占 20%（却是被引最多的来源），引文扩展条目最早在第 54 位、
    从未进入写作上下文（引文扩展消融因此无意义）。"""
    by = {}
    for e in items:
        by.setdefault(e["src"], []).append(e)
    order = [s for s in ("target", "ext", "kb", "cite", "state") if s in by] + [s for s in by if s not in
                                                                               ("target", "ext", "kb", "cite", "state")]
    out, i = [], 0
    while len(out) < len(items):
        for s in order:
            if i < len(by[s]):
                out.append(by[s][i])
        i += 1
    return out


def write_sections(question: str, sections: list[dict], evidence: list[dict], max_ev: int = 60,
                   use_screen: bool = True, word_budget: int | None = None,
                   deg: Degradation | None = None) -> list[str]:
    """word_budget：仅用于长度对照消融（整篇总词数上限，按节均分）；None=无长度上限（默认，prompt 与不传时逐字相同）。"""
    per_sec = max(80, word_budget // max(1, len(sections))) if word_budget else None
    # 跨节去重（每条证据只归一节 + 提纲约束）10-03 试过、判否撤销：1000 词下 AP −0.089 [−0.166,−0.013]——
    # 后节失去主证据后转写邻近话题；跨节复用 14%→4% 但离题段 12%→21%。保持各节独立取证据。

    def one(si):
        s = sections[si]
        items = _interleave([e for e in evidence if si in e["sections"]])
        if use_screen and items:
            items = screen(question, s["goal"], items, deg=deg)
        items = items[:max_ev]
        if not items:
            return ""
        prompt = WRITE.format(question=question, title=s["title"], goal=s["goal"], evidence=_fmt_ev(items))
        if per_sec:
            prompt = prompt.replace("There is no length limit.",
                                    f"Keep this section under {per_sec} words: include the most important points first.")
        text = chat(prompt, max_tokens=5000, temperature=0.2).strip()
        if not text and deg:
            deg.add("write_empty")
        return text
    with cf.ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, range(len(sections))))


# ---------------------------------------------------------------- 5. assemble (CS2 sections)
_EID = re.compile(r"\[(E\d+)\]")


def _identity(e: dict) -> str:
    """One paper, one citation (W1-13): the same paper reached through KB / external search / citation expansion
    carried different paper_keys and got two numbers (v9b test r1: 11/355 sections). Identity = arXiv id (version
    stripped) if known, else the normalized title when it is specific enough (>= 4 words or >= 25 characters),
    else the channel's own paper_key."""
    ax = (e.get("arxiv") or "").strip().lower()
    if ax:
        return "arxiv:" + re.sub(r"v\d+$", "", ax.split("arxiv:")[-1])
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (e.get("title") or "").lower())).strip()
    if len(t.split()) >= 4 or len(t) >= 25:
        return "title:" + t
    return e["paper_key"]


def assemble(sections: list[dict], texts: list[str], evidence: list[dict]) -> tuple[list[dict], dict]:
    ev = {e["eid"]: e for e in evidence}
    out, stats = [], {"cited": 0, "unresolved": 0, "merged_identities": 0}
    for s, text in zip(sections, texts):
        if not text:
            continue
        cites, n_map, keys_by_id = [], {}, {}
        def sub(m):
            eid = m.group(1)
            e = ev.get(eid)
            if e is None:
                stats["unresolved"] += 1
                return ""
            if e["src"] == "target":
                stats["target_refs"] = stats.get("target_refs", 0) + 1
                return ""
            ident = _identity(e)
            if ident not in n_map:
                n_map[ident] = len(n_map) + 1
                keys_by_id[ident] = {e["paper_key"]}
                cites.append({"id": f"[{n_map[ident]}]", "snippets": [], "title": e["title"],
                              "metadata": {"year": e.get("year"), "arxiv": e.get("arxiv"), "doi": e.get("doi"),
                                           "source": e["src"], "paper_key": e["paper_key"]}})
            elif e["paper_key"] not in keys_by_id[ident]:
                keys_by_id[ident].add(e["paper_key"])
                stats["merged_identities"] += 1
            c = cites[n_map[ident] - 1]
            if e["snippet"] not in c["snippets"] and len(c["snippets"]) < 4:
                c["snippets"].append(e["snippet"])
            stats["cited"] += 1
            return f"[{n_map[ident]}]"
        # 无引用句只计数、不删除：删句是迎合判分器（每句当一个 claim），属过拟合纪律禁止项；
        # "每个事实句都要有出处"由写作规则 1 约束，这里只做可审计的统计。
        for para in text.split("\n"):
            sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", para.strip()) if para.strip() else []
            stats["uncited_sentences"] = stats.get("uncited_sentences", 0) + sum(1 for x in sents if not _EID.search(x))
            stats["sentences"] = stats.get("sentences", 0) + len(sents)
        body = _EID.sub(sub, text)
        body = re.sub(r"(\[\d+\])(\s*\1)+", r"\1", body)   # [3][3] → [3]
        body = re.sub(r"[ \t]+([.,;:])", r"\1", body)
        out.append({"title": s["title"], "text": body.strip(), "citations": cites})
    return out, stats


# ---------------------------------------------------------------- literature-layer mode (L6 tools)
def as_of_date(cutoff: str | None) -> str:
    """Pipeline cutoff 'YYYY-MM' (exclusive month, CS2 convention) -> tool as_of 'YYYY-MM-DD' (inclusive, the day
    before). None -> today (no cutoff)."""
    import datetime as _dt
    if not cutoff:
        return _dt.date.today().isoformat()
    y, m = (int(x) for x in cutoff[:7].split("-"))
    return (_dt.date(y, m, 1) - _dt.timedelta(days=1)).isoformat()


def field_block(question: str, as_of: str, tools) -> str:
    fm = tools.field_map(question, as_of)
    lines = []
    for f in fm["families"][:6]:
        name = (f["name_candidates"] or ["(unnamed family)"])[0]
        mem = "; ".join(f"{m['title'][:80]} ({(m['date'] or '')[:4]})" for m in f["members"][:4] if m.get("title"))
        facts = "; ".join(f"{x['text'][:120]} [{x['status']}, {x['n_independent']} papers]" for x in f["facts"][:3])
        lines.append(f"- {name}: {mem}" + (f"\n  field says: {facts}" if facts else ""))
    return FIELD_BLOCK.format(as_of=as_of, items="\n".join(lines)) if lines else ""


def lit_gather(question: str, sections: list[dict], as_of: str, tools, k: int = 8) -> tuple[list[dict], dict]:
    """Evidence from the literature layer only: for each section query, papers (each with its own contribution and how
    the field describes it) and sentence-level evidence. Snippets are verbatim quotes (sentences), never system text."""
    evidence, by_text, trace = [], {}, {"calls": []}

    def add(si, src, r):
        key = (r["paper_key"], r["snippet"][:160])
        if key in by_text:
            by_text[key]["sections"].add(si)
            return
        e = {**r, "src": src, "eid": f"E{len(evidence) + 1}", "sections": {si}}
        by_text[key] = e
        evidence.append(e)

    for si, s in enumerate(sections):
        for q in s["queries"]:
            briefs = tools.search_papers(q, as_of, k)
            ev = tools.find_evidence(q, as_of, k)
            trace["calls"].append({"section": si, "src": "lit", "query": q, "n": len(briefs) + len(ev), "errors": []})
            for b in briefs:
                if not b.get("title"):
                    continue
                card = tools.paper_card(b["id"], as_of)
                # the paper's own words: its contribution sentences (verbatim quotes), else its abstract
                quotes = [c["quote"] for c in card.get("contributions") or []][:2]
                for qt in quotes or [((tools.AsOf(as_of).paper(b["id"][6:]) or {}).get("abstract") or "")[:1200]]:
                    if qt:
                        add(si, "lit_self", {"paper_key": b["id"], "title": b["title"],
                                             "year": (b["date"] or "")[:4] or None, "arxiv": b["id"][6:], "snippet": qt})
            for x in ev:
                by = x.get("about") if x["kind"] == "other" else x["by"]
                if not (by or "").startswith("paper:"):
                    continue
                p = tools.paper_card(by, as_of)
                if p.get("error") or not p.get("title"):
                    continue
                add(si, "lit_" + x["kind"], {"paper_key": by, "title": p["title"], "year": (p["date"] or "")[:4] or None,
                                             "arxiv": by[6:], "snippet": x["quote"]})
    trace["n_evidence"] = len(evidence)
    trace["by_src"] = {s: sum(1 for e in evidence if e["src"] == s) for s in ("lit_self", "lit_other", "lit_passage")}
    return evidence, trace


def answer_lit(question: str, cutoff: str | None = None, use_screen: bool = True, word_budget: int | None = None,
               task_context: dict | None = None, tools=None) -> dict:
    """The built-in consumer of the literature layer: plan with field_map, gather with the L6 tools, then the same
    screen / write / assemble as the legacy path. tools defaults to compilescholar.tools.api."""
    if tools is None:
        from ..tools import api as tools
    t0 = time.time()
    as_of = as_of_date(cutoff)
    deg = Degradation()
    probe_rows = [{"title": b["title"], "snippet": b["self"]} for b in tools.search_papers(question, as_of, 10)
                  if b.get("title")]
    fb = field_block(question, as_of, tools)
    sections = plan(question, probe_rows, deg=deg, extra=fb)
    evidence, trace = lit_gather(question, sections, as_of, tools)
    if task_context and task_context.get("text"):
        evidence.insert(0, {"src": "target", "paper_key": "target:self", "title": task_context.get("title") or "this paper",
                            "year": None, "snippet": task_context["text"][:2000], "eid": "E0",
                            "sections": set(range(len(sections)))})
    texts = write_sections(question, sections, evidence, use_screen=use_screen, word_budget=word_budget, deg=deg)
    out_secs, st = assemble(sections, texts, evidence)
    return {"sections": out_secs,
            "trace": {"mode": "lit", "as_of": as_of, "plan": sections, "field_block": fb, **trace, "assemble": st,
                      "words": sum(len(s["text"].split()) for s in out_secs), "degradation": deg.counts,
                      "elapsed_s": round(time.time() - t0, 1)}}


# ---------------------------------------------------------------- entry
def answer(question: str, kb: KB | None, use_ext=True, use_cite=True, cutoff: str | None = None,
           use_screen: bool = True, use_state: bool = True, task_context: dict | None = None,
           use_probe: bool = True, word_budget: int | None = None) -> dict:
    """task_context：任务本身给定的材料（如 DSB 的目标论文标题+摘要），作为一条 src="target" 证据加入每一节；
    写作时可引用（标为本文），但装配时不生成外部 citation（它不是被引文献）。"""
    t0 = time.time()
    _set_cut(cutoff)
    deg = Degradation()
    probe_rows = probe(question, kb, use_ext=use_ext) if use_probe else []
    sections = plan(question, probe_rows, deg=deg)
    t_plan = time.time() - t0
    evidence, trace = gather(question, sections, kb, use_ext=use_ext, use_cite=use_cite, cutoff=cutoff, use_state=use_state)
    deg.add("ext_query_failed", sum(1 for c in trace["calls"] if c["src"] == "ext" and c["errors"]))
    deg.add("cite_section_empty", sum(1 for c in trace["calls"] if c["src"] == "cite" and c["n"] == 0))
    if task_context and task_context.get("text"):
        evidence.insert(0, {"src": "target", "paper_key": "target:self", "title": task_context.get("title") or "this paper",
                            "year": None, "snippet": task_context["text"][:2000], "eid": "E0",
                            "sections": set(range(len(sections)))})
    t_g = time.time()
    texts = write_sections(question, sections, evidence, use_screen=use_screen, word_budget=word_budget, deg=deg)
    t_write = time.time() - t_g
    out_secs, st = assemble(sections, texts, evidence)
    words = sum(len(s["text"].split()) for s in out_secs)
    return {"sections": out_secs,
            "trace": {"plan": sections, "probe": [r["title"] for r in probe_rows], **trace, "assemble": st, "words": words,
                      "degradation": deg.counts,
                      "used_eids": sorted({m for t in texts for m in _EID.findall(t)}, key=lambda x: int(x[1:])),
                      "evidence": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()} for e in evidence],
                      "t_plan_s": round(t_plan, 1), "t_write_s": round(t_write, 1),
                      "elapsed_s": round(time.time() - t0, 1)}}
