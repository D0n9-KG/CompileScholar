# -*- coding: utf-8 -*-
"""新答题管线 v-next（REBUILD-PLAN-1003 §3）——取代 evidence_gate2r_harness 的 2784 行 ReAct 循环。

一题一条直线，无注入、无停机判据、无"笔记→独立编译"两段式：
  1. plan     ：27B 把题面拆成 3-6 个子问题（每个=一节）+ 每节 2-3 条检索式
  2. retrieve ：每个子问题并行
       KB   —— 记录级混合检索（BM25+向量 RRF，跨论文去冗余；真过滤）
       ext  —— 外部语义检索（Sciverse，服务端强制知识截止）
       cite —— 引文扩展：外部命中论文的参考文献按"被几篇命中论文共同引用"排序（S2，截止过滤）
  3. evidence：去重、按子问题归档；每条证据 = {eid, paper title, year, 逐字 snippet}
  4. write   ：27B 按节写报告，每个事实句用 [eid] 引用证据表；无长度上限
  5. assemble：确定性把 [eid] 映射成 CS2 citations（id 出现在正文、snippet=证据原文）；
               无法解析的引用标记剥离并计数（不删句）
输出 CS2 官方 sections JSON（与 report_adapter 产物同形态），外加完整轨迹（可审计）。
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
from array import array

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "..", "src"))  # CompileScholar/src
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.199.73,localhost,127.0.0.1"
# Sciverse 30 req/min 配额：本管线并发检索，令牌等不到时宁可排队也不要"照发吃 429→熔断→整题外检为 0"
os.environ.setdefault("SCIVERSE_MAX_WAIT_S", "600")

from kb_infra.llm import call_local, parse_json_response  # noqa: E402
from kb_compiler.retrieve.hybrid import HybridIndex  # noqa: E402
from cutoff import allowed as _cut_allowed, raw_cutoff as _cut_raw, set_thread_cutoff as _set_cut  # noqa: E402

MODEL = os.environ.get("ANSWER_MODEL", "Qwen3.8-27B")


def chat(prompt: str, max_tokens: int = 6000, temperature: float = 0.2) -> str:
    out = call_local(prompt, model=MODEL, max_tokens=max_tokens, temperature=temperature,
                     enable_thinking=False)
    return out or ""


# ---------------------------------------------------------------- KB 装载
class KB:
    def __init__(self, kb_dir: str, use_vectors: bool = True):
        self.papers = json.load(open(os.path.join(kb_dir, "papers.json"), encoding="utf-8"))
        records = json.load(open(os.path.join(kb_dir, "records.json"), encoding="utf-8"))
        vecs, embed = None, None
        vp, mp = os.path.join(kb_dir, "record_vecs.f32"), os.path.join(kb_dir, "record_vecs.meta.json")
        if use_vectors and os.path.exists(vp) and os.path.exists(mp):
            import numpy as np
            from kb_infra.embedding import embed_local
            meta = json.load(open(mp))
            flat = np.fromfile(vp, dtype="float32")
            vecs = flat.reshape(meta["n"], meta["dim"])
            embed = embed_local
        self.index = HybridIndex(records, self.papers, embed_fn=embed, doc_vecs=vecs)
        if vecs is not None and len(self.index.items) != vecs.shape[0]:
            raise RuntimeError("record_vecs 与 records.json 不同序/不同量——重跑 embed_kb_v2.py")
        # 领域状态视图（方法族 + 族级性质/局限/比较；base_kb_build/build_state_v2 + merge_state_v2 产物）
        self.families, self._fam_vecs, self._embed = [], None, embed
        sp = os.path.join(kb_dir, "state_merged.json")
        if os.path.exists(sp):
            self.families = [f for f in json.load(open(sp, encoding="utf-8"))["families"]
                             if f["props"] or f["limits"] or f["compares"]]
            if embed is not None and self.families:
                import numpy as np
                M = np.asarray(embed([" / ".join(f["names"][:3]) for f in self.families]), dtype="float32")
                self._fam_vecs = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    def state_search(self, q: str, top_fam: int = 3, per_fam: int = 6, min_sim: float = 0.55) -> list[dict]:
        """状态通道：查询 → 最近的方法族（嵌入余弦）→ 展开族内跨综述的性质 / 局限与开放问题 / 比较关系。
        每条证据仍是来源综述或论文的逐字片段（可引用），但组织单位是"族"而非单条记录——
        平面检索按词面命中单条记录，拿不到"这一类方法的共同局限"这种跨综述聚合。"""
        if self._fam_vecs is None:
            return []
        import numpy as np
        qv = np.asarray(self._embed([q])[0], dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-8)
        sims = self._fam_vecs @ qv
        out = []
        for j in np.argsort(-sims)[:top_fam]:
            if sims[j] < min_sim:
                break
            f = self.families[int(j)]
            fam_name = f["names"][0]
            items = ([("property", x) for x in f["props"]] + [("limitation", x) for x in f["limits"]]
                     + [("comparison", x) for x in f["compares"]])
            seen = set()
            for role, x in items:
                # 同一 taxonomy 节点的多条性质共享一段 quote——按 quote 去重，否则证据表里同一片段重复
                key = (x["paper_id"], (x.get("quote") or str(x.get("text") or ""))[:120])
                if key in seen:
                    continue
                seen.add(key)
                snip = (x.get("quote") or x.get("text") or "").strip()
                if not snip:
                    continue
                m = self.papers.get(x["paper_id"]) or {}
                out.append({"src": "state", "paper_key": f"kb:{x['paper_id']}", "title": m.get("title") or x["title"],
                            "year": m.get("year"), "arxiv": m.get("arxiv_id"),
                            "snippet": f"[{role} of '{fam_name}'] " + snip[:800],
                            "family": fam_name, "role": role, "family_sim": round(float(sims[j]), 3)})
                if sum(1 for o in out if o["family"] == fam_name) >= per_fam:
                    break
        return out

    def search(self, q: str, k: int = 12) -> list[dict]:
        res = self.index.search(q, k=k, per_paper=2)
        out = []
        for h in res["hits"]:
            r, pid = h["record"], h["paper_id"]
            m = self.papers.get(pid) or {}
            snippet = (r.get("quote") or r.get("claim") or "").strip()
            if not snippet:
                continue
            out.append({"src": "kb", "paper_key": f"kb:{pid}", "title": m.get("title") or pid,
                        "year": m.get("year"), "arxiv": m.get("arxiv_id"),
                        "snippet": snippet[:900], "match": res["match"],
                        "record_id": r.get("id"), "kind": r.get("kind")})
        return out


# ---------------------------------------------------------------- 外部检索
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from external_tools import ExternalTools
        _EXT = ExternalTools({}, {}, "local:" + MODEL)
    return _EXT


_SV = None


def _sciverse():
    """直连 Sciverse 语义检索客户端（不走 SearchService 分层扇出：那条路径 15s 档期 + 断路器，
    并发答题时排队等配额会被判成 timeout→熔断 600s→外检整题为 0；实测 45 并发查询 15 个空）。
    这里在令牌桶上排队（SCIVERSE_MAX_WAIT_S），429 退避重试，截止由服务端年份过滤执行。"""
    global _SV
    if _SV is None:
        import re as _re
        from sci_evo_extract.library.sources import SciverseClient
        tok = None
        for line in open(os.path.join(_REPO, "..", ".env"), encoding="utf-8"):
            m = _re.match(r"^\s*SCIVERSE_API_TOKEN\s*=\s*(\S+)", line)
            if m:
                tok = m.group(1)
        if tok:
            os.environ["SCIVERSE_API_TOKEN"] = tok
        _SV = SciverseClient(token=tok, timeout_seconds=60)
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


# ---------------------------------------------------------------- 引文扩展（S2）
_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_CACHE = os.path.join(_HERE, "..", "s2_cache")
os.makedirs(_S2_CACHE, exist_ok=True)
_S2_LAST = [0.0]
_S2_LOCK = __import__("threading").Lock()  # 无 key 的 S2 ~1 req/s：全进程串行，避免多线程同时撞 429 退避


def _s2(path: str, params: dict, tries: int = 4, deadline: float | None = None):
    import random
    import urllib.error
    import urllib.parse
    import urllib.request
    url = _S2_BASE + path + "?" + urllib.parse.urlencode(params)
    fn = os.path.join(_S2_CACHE, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if os.path.exists(fn):
        return json.load(open(fn, encoding="utf-8"))
    delay = 2.0
    for _ in range(tries):
        if deadline is not None and time.time() > deadline:
            return {}
        with _S2_LOCK:
            gap = 1.1 - (time.time() - _S2_LAST[0])
            if gap > 0:
                time.sleep(gap)
            _S2_LAST[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "compilescholar-answer"})
            with urllib.request.urlopen(req, timeout=20) as r:
                out = json.loads(r.read().decode("utf-8"))
            json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 404:
                json.dump({}, open(fn, "w"))
                return {}
        except Exception:
            pass
        time.sleep(delay + random.random())
        delay = min(delay * 1.6, 30)
    return {}


def cite_expand(seed_titles: list[str], k: int = 10, max_seeds: int = 8, budget_s: float = 90.0) -> list[dict]:
    """种子论文的参考文献，按被几篇种子共同引用排序（p8：同候选池召回约为语义检索 2 倍）。
    budget_s：本次扩展的墙钟上限（S2 无 key 限流时宁可少扩，不卡住整题）；命中缓存不耗时。"""
    from collections import Counter
    freq, info = Counter(), {}
    deadline = time.time() + budget_s
    for t in seed_titles[:max_seeds]:
        m = _s2("/paper/search/match", {"query": t[:300], "fields": "paperId,title"}, deadline=deadline)
        pid = ((m.get("data") or [{}])[0] or {}).get("paperId")
        if not pid:
            continue
        refs = _s2(f"/paper/{pid}/references", {"fields": "title,year,abstract,publicationDate", "limit": 200},
                   deadline=deadline)
        for row in refs.get("data") or []:
            c = row.get("citedPaper") or {}
            if not c.get("title") or not c.get("abstract"):
                continue
            if not _cut_allowed(c.get("year"), c.get("publicationDate")):
                continue
            key = c["title"].lower().strip()
            freq[key] += 1
            info[key] = c
    out = []
    for key, n in freq.most_common():
        if n < 2 or len(out) >= k:
            break
        c = info[key]
        out.append({"src": "cite", "paper_key": "ext:" + hashlib.md5(key.encode()).hexdigest()[:12],
                    "title": c["title"], "year": c.get("year"), "snippet": c["abstract"][:1200],
                    "co_cited_by": n})
    return out


# ---------------------------------------------------------------- 1. plan
PLAN = """You are planning a literature-grounded research report that answers the question below.
Break the answer into 3-6 sections. Each section answers one sub-question a domain expert would expect to be covered
(e.g. definitions/background, main approaches and how they differ, empirical findings, limitations and open problems,
applications) — choose what THIS question actually needs, most important first. For each section give 2-3 short
search queries (keyword style, as you would type into a scholarly search engine; include synonyms/acronyms).
{probe}
Question: {question}

Return JSON only: {{"sections": [{{"title": "...", "goal": "one sentence: what this section must establish", "queries": ["...", "..."]}}]}}"""

PROBE_BLOCK = """
An initial search for the question returned the papers below (title: excerpt). Use them to work out which field and
which meaning of the question's terms the question is about, and what the literature on THAT subject actually covers.
If they mix several meanings of a term, follow the one that fits every part of the question. Every section must be
about the question's own subject; do not plan sections on broader or adjacent topics that merely share a term.

{items}
"""


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


def plan(question: str, probe_rows: list[dict] | None = None) -> list[dict]:
    pb = ""
    if probe_rows:
        pb = PROBE_BLOCK.format(items="\n".join(f"- {r['title'][:150]}: {r['snippet'][:220]}" for r in probe_rows[:16]))
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
        # 引文扩展：每节取外部命中前 4 篇作种子（S2 无 key 限速 ~1 req/s，每种子 2 次调用——
        # 5 节×8 种子 曾让单题耗到 600s；4 种子×5 节≈40 次调用≈45s，且被缓存跨题复用）
        for si, s in enumerate(sections):
            sec_seeds = [r["title"] for (sj, src, _, rows) in raw if sj == si and src == "ext" for r in rows][:4]
            raw.append((si, "cite", "cite-expand", cite_expand(sec_seeds, k=cite_k, max_seeds=4) if sec_seeds else []))
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
WRITE = """Write one section of a literature-grounded research report.

Overall question: {question}
Section title: {title}
Section goal: {goal}

EVIDENCE (each item is a verbatim excerpt from a paper; cite by its id):
{evidence}

Rules:
1. EVERY sentence ends with the citation id(s) of the evidence that supports it, e.g. "... [E3]" or "... [E3][E7]".
   This includes topic, transition and summary sentences: if a sentence cannot be tied to specific evidence, delete it.
   Use only ids listed above. Do not cite anything else and do not invent papers.
2. Each sentence must stay within what its cited excerpts actually say. Do not add interpretation, motivation or
   implications the excerpt does not state ("this suggests...", "demonstrating that..."). When combining two papers in
   one sentence, cite both, and make sure each part is supported by its own citation.
3. Synthesize across papers: group related work, compare approaches, state agreements/disagreements, limitations and open
   problems when the evidence supports them. Name methods, datasets and papers concretely.
4. Be comprehensive: use every evidence item that bears on the overall question within this section's goal. Skip items
   that only match the section title but are about a different subject than the overall question. There is no length limit.
5. Write plain academic prose (paragraphs, optionally a short bulleted list). Do not repeat the section title. No references list.

Section text:"""


def _fmt_ev(items: list[dict]) -> str:
    lines = []
    for e in items:
        yr = f", {e['year']}" if e.get("year") else ""
        lines.append(f"[{e['eid']}] ({e['title'][:150]}{yr}) {e['snippet']}")
    return "\n".join(lines)


SCREEN = """You are screening evidence for one section of a research report.

Question: {question}
Section goal: {goal}

Some excerpts below were retrieved only because they share words with the question but are about a DIFFERENT problem
(e.g. another meaning of the same term, an unrelated field, or a table fragment with no usable content). List ONLY those
clearly off-topic excerpts. Keep anything that is plausibly useful background, a related method, a limitation, or a comparison,
even if it does not answer the question directly. When in doubt, do not list it.

Excerpts:
{items}

Return JSON only: {{"off_topic": [<ids of clearly off-topic excerpts>]}}"""


def screen(question: str, goal: str, items: list[dict], batch: int = 20, min_keep: int = 5) -> list[dict]:
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
    kept = [e for e in items if e["eid"] not in drop or e.get("src") == "target"]
    return kept if len(kept) >= min(min_keep, len(items)) else items


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
                   use_screen: bool = True) -> list[str]:
    def one(si):
        s = sections[si]
        items = _interleave([e for e in evidence if si in e["sections"]])
        if use_screen and items:
            items = screen(question, s["goal"], items)
        items = items[:max_ev]
        if not items:
            return ""
        return chat(WRITE.format(question=question, title=s["title"], goal=s["goal"], evidence=_fmt_ev(items)),
                    max_tokens=5000, temperature=0.2).strip()
    with cf.ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, range(len(sections))))


# ---------------------------------------------------------------- 5. assemble (CS2 sections)
_EID = re.compile(r"\[(E\d+)\]")


def assemble(sections: list[dict], texts: list[str], evidence: list[dict]) -> tuple[list[dict], dict]:
    ev = {e["eid"]: e for e in evidence}
    out, stats = [], {"cited": 0, "unresolved": 0}
    for s, text in zip(sections, texts):
        if not text:
            continue
        cites, n_map = [], {}
        def sub(m):
            eid = m.group(1)
            e = ev.get(eid)
            if e is None:
                stats["unresolved"] += 1
                return ""
            if e["src"] == "target":
                stats["target_refs"] = stats.get("target_refs", 0) + 1
                return ""
            if e["paper_key"] not in n_map:
                n_map[e["paper_key"]] = len(n_map) + 1
                cites.append({"id": f"[{n_map[e['paper_key']]}]", "snippets": [], "title": e["title"],
                              "metadata": {"year": e.get("year"), "arxiv": e.get("arxiv"), "doi": e.get("doi"),
                                           "source": e["src"], "paper_key": e["paper_key"]}})
            c = cites[n_map[e["paper_key"]] - 1]
            if e["snippet"] not in c["snippets"] and len(c["snippets"]) < 4:
                c["snippets"].append(e["snippet"])
            stats["cited"] += 1
            return f"[{n_map[e['paper_key']]}]"
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


# ---------------------------------------------------------------- entry
def answer(question: str, kb: KB | None, use_ext=True, use_cite=True, cutoff: str | None = None,
           use_screen: bool = True, use_state: bool = True, task_context: dict | None = None,
           use_probe: bool = True) -> dict:
    """task_context：任务本身给定的材料（如 DSB 的目标论文标题+摘要），作为一条 src="target" 证据加入每一节；
    写作时可引用（标为本文），但装配时不生成外部 citation（它不是被引文献）。"""
    t0 = time.time()
    _set_cut(cutoff)
    probe_rows = probe(question, kb, use_ext=use_ext) if use_probe else []
    sections = plan(question, probe_rows)
    t_plan = time.time() - t0
    evidence, trace = gather(question, sections, kb, use_ext=use_ext, use_cite=use_cite, cutoff=cutoff, use_state=use_state)
    if task_context and task_context.get("text"):
        evidence.insert(0, {"src": "target", "paper_key": "target:self", "title": task_context.get("title") or "this paper",
                            "year": None, "snippet": task_context["text"][:2000], "eid": "E0",
                            "sections": set(range(len(sections)))})
    t_g = time.time()
    texts = write_sections(question, sections, evidence, use_screen=use_screen)
    t_write = time.time() - t_g
    out_secs, st = assemble(sections, texts, evidence)
    words = sum(len(s["text"].split()) for s in out_secs)
    return {"sections": out_secs,
            "trace": {"plan": sections, "probe": [r["title"] for r in probe_rows], **trace, "assemble": st, "words": words,
                      "used_eids": sorted({m for t in texts for m in _EID.findall(t)}, key=lambda x: int(x[1:])),
                      "evidence": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()} for e in evidence],
                      "t_plan_s": round(t_plan, 1), "t_write_s": round(t_write, 1),
                      "elapsed_s": round(time.time() - t0, 1)}}
