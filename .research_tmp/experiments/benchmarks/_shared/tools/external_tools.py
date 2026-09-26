# -*- coding: utf-8 -*-
"""External retrieval tools for the answering loop (broker wiring, 2026-09-25).

Injects four tools onto the KBTools instance used by the Multi answering
harness:

  search_papers(query, k?)      blind external search (shared tiered engine)
  gap_search(query)             gap-driven: absence coordinates -> directed queries
  lineage_walk(entity, direction?)  typed genealogy walk -> frontier -> external continuation
  citation_graph(doi|title, direction?)  external citation neighbors + corpus annotation

Design decisions:
  - injected in multi_ours_run.build_tools_multi (fork-side); the frozen home
    modules (evidence_b2_tools / harness core) stay untouched — the harness
    already dispatches any kb attribute present in TOOL_WHITELIST, which we
    extend at runtime
  - every observation is a self-contained dict with the same provenance
    discipline as typed tools (title/doi/year/abstract per paper), so note
    lines can cite [paper title] or [doi] backrefs — SQA2's citation format
    (LLM attribution over snippets) accepts free-form ids
  - external candidates render TITLE + ABSTRACT (the official asta tool uses
    pd["text"] = pd["abstract"]; we match that shape) — enough for the model
    to judge relevance AND to cite as evidence at SQA2
  - Tier-1 coarse extraction trigger is EXPOSED as a flag on the observation
    (deep_dive: true) — the answering prompt decides when to promote; the
    promotion itself runs coarse_extract (one light call) and is wired here
    as extract_paper(abstract) so the loop can call it as a tool too
"""

from __future__ import annotations

import json
import os
import sys

_SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _ensure_env():
    # .env 是唯一真源（批 8 教训：环境残留的坏 token 会绕过
    # not-get 检查——总是覆盖）
    try:
        for line in open(os.path.join(
                r"C:\Users\D0n9\Desktop\CompileScholar", ".env"),
                encoding="utf-8"):
            if line.strip().startswith("SCIVERSE_API_TOKEN"):
                _, _, v = line.strip().partition("=")
                if v.strip():
                    os.environ["SCIVERSE_API_TOKEN"] = v.strip()
    except Exception:
        pass
    os.environ.setdefault("LOCAL_MODEL", os.environ.get("OURS_MODEL", "Qwen3.8-27B").split(":")[-1])
    os.environ.setdefault("LOCAL_EMBED_MODEL", "qwen3-embedding-8b-local")
    os.environ.setdefault("EMBEDDING_MODEL", "qwen3-embedding-8b-local")


def _cand_row(c) -> dict:
    """One external candidate -> harness observation row (title+abstract
    evidence, official asta shape)."""
    raw = getattr(c, "raw", None) or {}
    abstract = ""
    for k in ("abstract", "summary"):
        v = raw.get(k) if isinstance(raw, dict) else None
        if isinstance(v, str) and v.strip():
            abstract = v
            break
    # OpenAlex inverted-index reconstruction
    inv = raw.get("abstract_inverted_index") if isinstance(raw, dict) else None
    if not abstract and isinstance(inv, dict) and inv:
        slots = {int(p): str(w) for w, ps in inv.items()
                 for p in (ps if isinstance(ps, list) else [])}
        abstract = " ".join(slots[k] for k in sorted(slots))
    return {
        "title": getattr(c, "title", None),
        "year": getattr(c, "year", None),
        "doi": getattr(c, "normalized_doi", None),
        "source": getattr(c, "source_name", None),
        "abstract": (abstract[:900] + "…") if len(abstract) > 900 else abstract,
    }


def _truncate_obs(rows: list[dict], cap: int = 6500) -> list[dict]:
    """Fit observation to the harness OBS_CAP envelope: keep rows until the
    cap, always keep title/doi/source, shrink abstracts first."""
    out = []
    used = 200
    for r in rows:
        row_json = json.dumps(r, ensure_ascii=False)
        if used + len(row_json) <= cap:
            out.append(r)
            used += len(row_json)
            continue
        # shrink abstract to fit
        budget = cap - used - 250
        if budget > 200 and r.get("abstract"):
            r = dict(r, abstract=r["abstract"][:budget] + "…")
            out.append(r)
            used += len(json.dumps(r, ensure_ascii=False))
        break
    return out


class ExternalTools:
    """Lazy holder — the engine stack is built once per answering process."""

    def __init__(self, views: dict, kb_manifest: dict, model: str = "local:Qwen3.8-27B",
                 registry: dict = None, blocklist: list = None,
                 backflow_path: str = None, tier_db: str = None):
        _ensure_env()
        self._model = model
        self._views = views
        self._kb_manifest = kb_manifest
        self._registry = registry
        self._blocklist = blocklist
        self._backflow_path = backflow_path
        self._tier_db = tier_db
        self._tiers = None
        self._svc = None
        self._gap_docs = None
        self._embed = None
        self._citgraph = None
        # deep_read 的记录落点：attach 时注入（=答题进程的 KBTools.records）
        self._records_target = None

    def _tier_store(self):
        """Lazy sci-evo SQLite tier store (batch-3): the system of record
        for coarse/deep extraction state — re-extraction is refused across
        sessions, promotion candidates queue for Tier 2."""
        if self._tiers is None:
            if not self._tier_db:
                self._tiers = False
                return None
            try:
                from sci_evo_extract.library.coarse_store import TierStore
                root = os.path.dirname(self._tier_db)
                self._tiers = TierStore(self._tier_db,
                                        os.path.join(root, "growth_library"))
            except Exception:
                self._tiers = False
        return self._tiers or None

    # ---- lazy singletons -------------------------------------------------

    def _service(self):
        if self._svc is None:
            from sci_evo_extract.library.search_service import SearchService
            self._svc = SearchService(limit=10)
        return self._svc

    def _gap_index(self):
        if self._gap_docs is None:
            from sci_evo_extract.library.gap_search import build_gap_docs
            self._gap_docs = build_gap_docs(self._views)
        return self._gap_docs

    def _embed_fn(self):
        if self._embed is None:
            try:
                from kb_infra.embedding import embed_local
                self._embed = embed_local
            except Exception:
                self._embed = False  # token-overlap fallback in gap_search
        return self._embed or None

    def _citation(self):
        if self._citgraph is None:
            from sci_evo_extract.library.citation_graph import CitationGraphService
            manifest_rows = [
                {"paper_id": pid, "doi": (m or {}).get("doi"),
                 "title": (m or {}).get("title")}
                for pid, m in (self._kb_manifest or {}).items()
            ]
            self._citgraph = CitationGraphService(corpus_manifest=manifest_rows)
        return self._citgraph

    # ---- tool surfaces (harness contract: plain dicts) --------------------

    def search_papers(self, query: str, k: int = 8) -> dict:
        """Blind external search through the shared tiered engine
        (Sciverse semantic leads, keyword fallbacks). Returns title+abstract
        rows usable as evidence."""
        try:
            # agent mode (P14): the loop's query is already refined keyword
            # vocabulary — skip the A3 decomposition, semantic-first single
            # pass with keyword fallback
            r = self._service().search(query, mode="agent", limit=k)
            rows = [_cand_row(c) for c in r.candidates[:k]]
            return {"tool": "search_papers", "n": len(rows), "papers": _truncate_obs(rows),
                    "latency_ms": {k2: v for k2, v in r.latency.items()},
                    "note": "external papers; cite by [title] or [doi] in notes"}
        except Exception as e:
            return {"tool": "search_papers", "n": 0,
                    "error": f"external search failed: {str(e)[:120]}"}

    def gap_search(self, query: str) -> dict:
        """Gap-driven search: matches the query against the knowledge base's
        recorded ABSENCES (what methods cannot do / what has no results),
        then searches externally for papers that FILL the matched gap —
        using vocabulary the question itself does not contain."""
        try:
            from sci_evo_extract.library.gap_search import gap_search as gs
            r = gs(query, gap_docs=self._gap_index(),
                   search_fn=lambda q, k: self._service().search(q, mode="agent", limit=k),
                   embed=self._embed_fn(), limit=8)
            gaps = [{"gap": h.gap_doc["doc"][:140], "score": h.score,
                     "kind": h.gap_doc.get("kind")}
                    for h in r.gaps]
            rows = [_cand_row(c) for c in r.candidates]
            return {"tool": "gap_search", "n": len(rows), "gaps": gaps,
                    "papers": _truncate_obs(rows),
                    "fill_annotations": r.fill_annotations[:8],
                    "note": "candidates annotated as gap fillers where matched"}
        except Exception as e:
            return {"tool": "gap_search", "n": 0,
                    "error": f"gap search failed: {str(e)[:120]}"}

    def lineage_walk(self, entity: str, direction: str = "successors") -> dict:
        """Typed method-genealogy walk: in-corpus extends/improves/replaces
        edges to the frontier, then external continuation for successors.
        Use for 'latest advances in X / what replaced X' questions."""
        try:
            from sci_evo_extract.library.lineage_walk import lineage_walk as lw
            r = lw(entity, self._views, direction=direction, external=True, limit=6)
            chain = [{"entity": n.canonical, "via": n.via_relation,
                      "depth": n.depth} for n in r.in_corpus]
            frontier = [n.canonical for n in r.frontier]
            rows = [_cand_row(c) for c in r.external_candidates]
            return {"tool": "lineage_walk", "entity": entity,
                    "corpus_chain": chain, "frontier": frontier,
                    "attached_external": r.attached_external,
                    "n_external": len(rows), "papers": _truncate_obs(rows),
                    "successor_annotations": r.successor_annotations[:8],
                    "note": "frontier = corpus lineage ends here; papers are external continuations; "
                            "attached_external = papers already coarse-extracted into the library "
                            "(their records are included — no re-search needed)"}
        except Exception as e:
            return {"tool": "lineage_walk", "n_external": 0,
                    "error": f"lineage walk failed: {str(e)[:120]}"}

    def citation_graph(self, doi: str = None, title: str = None,
                       direction: str = "both") -> dict:
        """External citation neighbors of a paper (S2; OpenAlex fallback for
        outgoing). Endpoints matched against our corpus are marked."""
        try:
            g = self._citation().graph(doi=doi, title=title,
                                       direction=direction, max_total=20)
            edges = [{"dir": e.direction, "title": e.title, "year": e.year,
                      "doi": e.doi,
                      "in_corpus": e.in_corpus_paper_id}
                     for e in g.edges[:20]]
            return {"tool": "citation_graph", "anchor": g.anchor,
                    "n_edges": len(edges), "edges": _truncate_obs(edges, 5500),
                    "n_in_corpus": g.n_in_corpus,
                    "note": "in_corpus=paper_id means the neighbor is already in our KB"}
        except Exception as e:
            return {"tool": "citation_graph", "n_edges": 0,
                    "error": f"citation graph failed: {str(e)[:120]}"}

    def extract_paper(self, title: str, abstract: str) -> dict:
        """Tier-1 coarse extraction: digest ONE external paper's abstract into
        structured records (method/finding/limitation + lineage mentions).
        Run this when an external paper looks core to the question; the
        records are abstract-level (provenance=coarse), not full-text."""
        try:
            # 综述识别路由（2026-09-28，修订案 B）：摘要/标题命中综述模式
            # -> 打 is_survey 标记。粗抽照常（摘要级），但 promotion 到
            # 深抽时会走 survey_extract（S1-S4）而非标准管线——综述的
            # 全文价值在谱系/缺口/快照，标准管线对它是结构错配。
            import re as _re
            _survey_pat = _re.compile(
                r"\b(survey|review|we survey|comprehensive overview|"
                r"systematic review|state.of.the.art)\b", _re.I)
            _is_survey = bool(_survey_pat.search(title or "")
                              or _survey_pat.search(abstract or ""))
            from kb_compiler.records.coarse_extract import extract_coarse
            r = extract_coarse(title, abstract, None, self._model)
            recs = [{"kind": x["kind"], "subject": x["subject"],
                     "claim": x["claim"], "quote": x["quote"][:200],
                     "mentions": x["mentions"],
                     # citation handle per record so note lines can backref
                     # the paper (behavior probe: the model invented
                     # [extract_paper] tool-name citations when records
                     # carried no citable id — 4 consecutive reject steps)
                     "title": title}
                    for x in r["records"]]
            obs = {"tool": "extract_paper", "n": len(recs), "records": recs,
                   "title": title,
                   "note": "coarse records from abstract only; cite this paper in notes as [title]"}
            if _is_survey:
                obs["is_survey"] = True
                obs["survey_note"] = (
                    "this paper is a SURVEY — its coarse records are "
                    "abstract-level only; if promoted to deep extraction it "
                    "goes through the survey extractor (lineage edges, "
                    "domain snapshots, gaps, controlled second-hand claims)")
            # mentions backflow (batch-2): the records' mentions match
            # registry entities -> attach this paper to the corpus
            # genealogy as an external continuation node. The library
            # grows; later lineage_walks reach this paper at zero cost.
            try:
                if self._registry is not None and r["records"]:
                    from kb_compiler.records.backflow import build_backflow, apply_backflow
                    bf = build_backflow(
                        r, {"title": title, "year": None},
                        self._registry, self._blocklist)
                    res = apply_backflow(self._views, bf,
                                         persist_path=self._backflow_path)
                    if res.get("attached"):
                        obs["attached_to_lineage"] = res["matched_entities"]
                        obs["library_growth"] = (
                            "paper attached as external continuation of: "
                            + ", ".join(res["matched_entities"]))
            except Exception as e:
                obs["backflow_error"] = str(e)[:120]
            # batch-3: tier state in the sci-evo SQLite registry — the
            # paper is recorded as coarse-extracted (re-extraction refused
            # across sessions); the deterministic promotion rule (limitation
            # signal / multi-lineage anchor) is surfaced to the loop
            try:
                ts = self._tier_store()
                if ts is not None:
                    matched = res["matched_entities"] if res.get("attached") or res.get("matched_entities") else []
                    st = ts.register_coarse(
                        title=title, year=None, abstract=abstract[:6000],
                        records=r["records"], matched_entities=matched)
                    if st.get("reused"):
                        obs["tier_note"] = ("paper already registered "
                                            f"(tier={st['tier']}) — no re-extraction")
                    if st.get("promotion_candidate"):
                        obs["promotion_candidate"] = True
                        obs["promotion_note"] = (
                            "limitation/multi-lineage signal — strong Tier-2 "
                            "candidate (full-text extraction at library-growth time)")
            except Exception as e:
                obs["tier_error"] = str(e)[:120]
            return obs
        except Exception as e:
            return {"tool": "extract_paper", "n": 0,
                    "error": f"coarse extraction failed: {str(e)[:120]}"}

    # ---------------- deep_read: 四级下潜的 L1 定向模式 ----------------
    # （2026-09-28 用户裁定：答题 Agent 决定抽哪部分 + chunk 级缓存 +
    # L2 异步后台。背景：通用 harness 能随时看原文，我们的循环需要同等
    # 深度访问——但经由编译层。批 2-4 实测粗抽摘要级记录答不了"具体数值/
    # 机制细节"类 ingredient；全量深抽 30-60min/篇超答题循环容忍。
    # L1 = 定向抽 2-3 chunks（~3min），L2 = 剩余 chunk 后台补齐（跨题复利），
    # chunk 级缓存持久化（L1 抽过的 L2 不重抽；absence 整扫只在 L2 做）。
    # DocTrace 调研背书：按需构建 -53% 成本 + F1 反升。）
    _DEEP_READ_DONE: set = set()
    _L2_RUNNING: set = set()
    _CACHE_LOCK = __import__("threading").Lock()

    # ---- chunk 级缓存（持久化，跨题/跨批复利） ----

    def _cache_path(self) -> str:
        return getattr(self, "_deep_cache_path", None)

    def _cache_load(self):
        """{paper_id: {"text_hash": h, "card": {...}|None,
        "chunks": {chunk_id: obj}}} — 单进程内存缓存 + JSONL 落盘。"""
        if getattr(self, "_deep_cache", None) is None:
            cache = {}
            p = self._cache_path()
            if p and os.path.exists(p):
                try:
                    for line in open(p, encoding="utf-8"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                        except Exception:
                            continue
                        ent = cache.setdefault(
                            e["paper_id"], {"text_hash": e.get("text_hash"),
                                            "card": None, "chunks": {}})
                        if e.get("text_hash"):
                            ent["text_hash"] = e["text_hash"]
                        if e.get("type") == "card":
                            ent["card"] = e.get("card")
                        elif e.get("chunk_id"):
                            ent["chunks"][e["chunk_id"]] = e.get("obj")
                except Exception:
                    pass
            self._deep_cache = cache
        return self._deep_cache

    def _persist_deep_records(self, paper_id: str, recs: list, replace: bool):
        """终化记录持久化（批10 第三断口修复）：deep_read/L2 的记录若只
        活在答题进程内存，report_adapter 的 EvidenceStore（独立进程）解析
        不了深记录回指→空壳报告。落 deep_read_records.json（合并非替换，
        粗抽记录的回指仍有效）。"""
        p = getattr(self, "_deep_records_path", None)
        if not p or not recs:
            return
        with ExternalTools._CACHE_LOCK:
            try:
                cur = {}
                if os.path.exists(p):
                    cur = json.load(open(p, encoding="utf-8"))
                base_recs = [] if replace else \
                    (cur.get(paper_id) or {}).get("records") or []
                seen = {r.get("id") for r in base_recs if r.get("id")}
                merged = list(base_recs) + [
                    r for r in recs if r.get("id") not in seen]
                cur[paper_id] = {"records": merged}
                json.dump(cur, open(p, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            except Exception as e:
                print(f"[deep_read-persist] {paper_id[:40]}: {str(e)[:80]}",
                      flush=True)

    def _cache_append(self, paper_id: str, text_hash: str, entries: list):
        """entries: [{"type": "card", "card": ...} | {"chunk_id": cid,
        "obj": ...}] — 追加落盘 + 内存同步（线程安全）。"""
        p = self._cache_path()
        if not p:
            return
        with ExternalTools._CACHE_LOCK:
            ent = self._cache_load().setdefault(
                paper_id, {"text_hash": text_hash, "card": None,
                           "chunks": {}})
            if text_hash:
                ent["text_hash"] = text_hash
            lines = []
            for e in entries:
                if e.get("type") == "card":
                    ent["card"] = e["card"]
                else:
                    ent["chunks"][e["chunk_id"]] = e["obj"]
                rec = {"paper_id": paper_id, "text_hash": text_hash}
                rec.update(e)
                lines.append(json.dumps(rec, ensure_ascii=False))
            try:
                with open(p, "a", encoding="utf-8") as f:
                    for ln in lines:
                        f.write(ln + "\n")
            except Exception:
                pass

    # ---- 全文获取（三通道，文本级缓存） ----

    def _fetch_full_text(self, paper_id: str, m: dict):
        """(text, src)。文本缓存优先（Sciverse 分页 20 请求/篇不重付）。"""
        import hashlib as _hl
        cache_dir = getattr(self, "_deep_text_dir", None)
        tp = None
        if cache_dir:
            tp = os.path.join(cache_dir, _hl.md5(paper_id.encode()).hexdigest()
                              + ".txt")
            if os.path.exists(tp):
                try:
                    t = open(tp, encoding="utf-8").read()
                    if len(t) >= 5000:
                        return t, "text-cache"
                except Exception:
                    pass
        text, src = self._fetch_full_text_live(paper_id, m)
        if len(text) >= 5000 and tp:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(tp, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        return text, src

    def _fetch_full_text_live(self, paper_id: str, m: dict):
        import re as _re
        aid = m.get("arxiv_id")
        if not aid:
            mm = _re.match(r"arxiv_(\d{4}\.\d{4,5})", paper_id)
            aid = mm.group(1) if mm else None
        text = ""
        src = None
        if aid:
            _bbp = (r"C:\Users\D0n9\Desktop\CompileScholar"
                    r"\.research_tmp\experiments\benchmarks"
                    r"\cs2\base_kb_build")
            if _bbp not in sys.path:
                sys.path.insert(0, _bbp)
            from fetch_arxiv_html import fetch_text as _fetch_text
            try:
                text = _fetch_text(aid)
                src = f"arxiv-html:{aid}"
            except Exception:
                text = ""
        if len(text) < 5000 and m.get("doc_id"):
            try:
                # 分页拉全（2026-09-28 破局发现）：/content 单次只返回
                # 首页，offset+more 循环才能拿全文——实测 magneto 71,152
                # chars 全文就在 Sciverse 里。
                from sci_evo_extract.library.sources import SciverseClient
                sc = SciverseClient()
                parts, offset = [], 0
                for _ in range(20):
                    resp = sc.read_content(doc_id=m["doc_id"],
                                           offset=offset, limit=10000)
                    t = str(resp.get("text") or "")
                    if not t:
                        break
                    parts.append(t)
                    if not resp.get("more"):
                        break
                    no = resp.get("next_offset")
                    offset = no if no else offset + len(t)
                text = "".join(parts)
                src = f"sciverse-paged:{m['doc_id'][:12]}"
            except Exception:
                pass
        # 通道 3（按需反查）：标题查 arXiv API 拿 id 再拉 HTML——
        # 需求池论文（Sciverse 来的）无 arxiv_id 但多有 arXiv 版本；
        # 批量反查被限流（406），按需只在被 deep_read 时查一次
        if len(text) < 5000:
            _t = _re.sub(r"\s+", " ", (m.get("title") or "")).strip()
            if len(_t) >= 10:
                try:
                    import urllib.request as _ur2
                    import urllib.parse as _up
                    import xml.etree.ElementTree as _ET
                    _url = ("http://export.arxiv.org/api/query?search_query="
                            + _up.quote(f'ti:"{_t[:120]}"') + "&max_results=2")
                    _req = _ur2.Request(_url, headers={
                        "User-Agent": "CompileScholar/0.1"})
                    with _ur2.urlopen(_req, timeout=30) as _r2:
                        _root = _ET.fromstring(_r2.read())
                    _NS = {"a": "http://www.w3.org/2005/Atom"}
                    for _e in _root.findall("a:entry", _NS):
                        _at = _re.sub(r"\s+", " ",
                                      _e.findtext("a:title", "", _NS)
                                      ).strip().lower()
                        if _at[:80] == _t[:80].lower():
                            _aid = (_e.findtext("a:id", "", _NS)
                                    .split("/abs/")[-1].split("v")[0])
                            try:
                                _bbp3 = ("C:/Users/D0n9/Desktop/CompileScholar"
                                         "/.research_tmp/experiments/benchmarks"
                                         "/cs2/base_kb_build")
                                if _bbp3 not in sys.path:
                                    sys.path.insert(0, _bbp3)
                                from fetch_arxiv_html import fetch_text as _ft3
                                text = _ft3(_aid)
                                src = f"arxiv-resolved:{_aid}"
                                # 回填+落盘：同一论文全生命周期只反查一次
                                # （批 6 教训：10 连发反查又触发 406 封禁）
                                try:
                                    (self._kb_manifest or {})[
                                        paper_id]["arxiv_id"] = _aid
                                    _mp = getattr(self, "_manifest_path",
                                                  None)
                                    if _mp:
                                        json.dump(
                                            list(self._kb_manifest.
                                                 values()),
                                            open(_mp, "w",
                                                 encoding="utf-8"),
                                            ensure_ascii=False, indent=1)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            break
                    import time as _time
                    _time.sleep(3.0)   # arXiv 限流纪律
                except Exception:
                    pass
        return text, src

    # ---- L1 chunk 选择 ----

    @staticmethod
    def _chunk_tokens(s: str) -> set:
        import re as _re
        return {t for t in _re.findall(r"[a-z][a-z0-9\-]{2,}", (s or "").lower())}

    def _select_l1_chunks(self, chunks: list, sections, question):
        """Agent 指定 sections 优先（节名/关键词匹配）；缺省=题目 token
        重叠度最高的前 3 chunks；无信号=结果/配置富集节优先（ingredient
        类问题的主力记录类型）。返回 (selected, mode, note)。"""
        import re as _re
        if sections:
            if isinstance(sections, str):   # 模型传了单个字符串——包成列表
                sections = [sections]
            want = [str(s).lower().strip() for s in sections if s and str(s).strip()]
            sel = []
            for ch in chunks:
                title = (ch.get("section") or "").lower()
                head = title + " " + (ch.get("text") or "")[:300].lower()
                thead = self._chunk_tokens(head)
                for w in want:
                    wtoks = self._chunk_tokens(w)
                    if (w in head
                            or (wtoks and thead
                                and len(wtoks & thead) / len(wtoks) >= 0.6)):
                        sel.append(ch)
                        break
            if len(sel) > 5:   # 关键词过泛（如 "results"）：按出现序截断
                sel = sel[:5]
            if sel:
                return sel, "sections", None
            # 指定节名全部未命中——降级到默认选择，并在 note 里给出可用节
            # 名清单让模型重试（不裸失败）
            avail = list(dict.fromkeys(
                (ch.get("section") or "?") for ch in chunks))[:15]
            sel, m2, _ = self._select_l1_chunks(chunks, None, question)
            return sel, "sections-miss", (
                "no section matched "
                + json.dumps([str(s) for s in sections[:5]])
                + f"; fell back to question-relevant chunks; available sections: "
                + json.dumps(avail, ensure_ascii=False))
        if question:
            qtoks = self._chunk_tokens(question)
            scored = []
            for ch in chunks:
                head = (ch.get("section") or "") + " " + (ch.get("text") or "")[:400]
                score = len(qtoks & self._chunk_tokens(head))
                scored.append((score, ch))
            scored.sort(key=lambda x: -x[0])
            top = [ch for s, ch in scored[:3] if s > 0]
            if top:
                return top, "question-top3", None
        # 无题目信号 / 全零：结果与配置富集节优先（ingredient 主力）
        pri = {"result": 0, "config": 1}
        ranked = sorted(
            chunks,
            key=lambda ch: (min(pri.get(k, 9) for k in ch.get("kinds") or ["result"]),
                            ch.get("char_start") or 0))
        return ranked[:3], "result-first", None

    # ---- L2 异步后台（剩余 chunk 增量补齐 + absence 一次） ----

    def _l2_background(self, paper_id, text, card, title, text_hash):
        """daemon 协调线程：L1 未覆盖的 chunk 逐个抽（缓存命中跳过，
        每完成一个立刻落缓存——进程中途死掉已抽部分不丢），全完成后
        跑一次 absence 整扫，再全量 finalize 入账。答题循环不等它。

        线程模型注意（probe 实测教训）：worker 必须是裸 daemon Thread，
        不能用 ThreadPoolExecutor——其 with 块 shutdown(wait=True) 会
        join，且 concurrent.futures 的 atexit 钩子在解释器退出时 join
        全部 worker，进程会挂着等 L2 跑完（违背"当前题不等它"）。
        裸 daemon 线程进程退出即死，已完成 chunk 已在缓存里。"""
        import threading as _th
        if paper_id in ExternalTools._L2_RUNNING:
            return
        ExternalTools._L2_RUNNING.add(paper_id)

        def _run():
            try:
                from kb_compiler.records.deep_extract import (
                    build_paper_tasks as _bpt, _call_chunk, _call_absence,
                    finalize_paper as _fp)
                vocab = {"subject": [], "setup": [], "variant": [],
                         "hyperparam_items": []}
                seed_reg = {"entities": [], "surface_index": {}}
                ctx = _bpt(paper_id, text, card, seed_reg, vocab, title)
                cache = self._cache_load()
                ent = (cache.get(paper_id) or {})
                cached = dict(ent.get("chunks") or {})
                if ent.get("text_hash") and text_hash != ent["text_hash"]:
                    cached = {}   # 文本变了，旧 chunk 缓存作废
                todo = [(ch, p) for ch, p in ctx["chunk_prompts"]
                        if cached.get(ch["chunk_id"]) is None]
                done_evt = _th.Event()
                remaining = [len(todo) + (0 if f"{paper_id}#absence"
                                          in cached else 1)]
                cnt_lock = _th.Lock()
                sem = _th.Semaphore(2)   # 后台低优先：一次 2 路不抢道

                def _tick():
                    with cnt_lock:
                        remaining[0] -= 1
                        if remaining[0] <= 0:
                            done_evt.set()

                def _work(ch, prompt):
                    try:
                        with sem:
                            obj = _call_chunk(prompt, self._model)
                        if isinstance(obj, (dict, list)):
                            self._cache_append(paper_id, text_hash,
                                               [{"chunk_id": ch["chunk_id"],
                                                 "obj": obj}])
                    finally:
                        _tick()

                workers = [_th.Thread(target=_work, args=(ch, p), daemon=True)
                           for ch, p in todo]
                for w in workers:
                    w.start()

                def _absence():
                    try:
                        aobj = _call_absence(ctx["absence_prompt"],
                                             self._model)
                        if isinstance(aobj, dict):
                            self._cache_append(paper_id, text_hash,
                                               [{"chunk_id":
                                                 f"{paper_id}#absence",
                                                 "obj": aobj}])
                    finally:
                        _tick()

                if f"{paper_id}#absence" not in cached:
                    _th.Thread(target=_absence, daemon=True,
                               name=f"deep-l2-abs-{paper_id[:14]}").start()
                else:
                    _tick()

                done_evt.wait(timeout=7200)   # 2h 安全阀
                # 全量 finalize（含 L1 已抽 chunk——确定性合并）并替换入账
                cached2 = dict((self._cache_load().get(paper_id) or {})
                               .get("chunks") or {})
                chunk_objs = {ch["chunk_id"]: cached2.get(ch["chunk_id"])
                              for ch, _ in ctx["chunk_prompts"]}
                absence_obj = cached2.get(f"{paper_id}#absence")
                _p, out = _fp(paper_id, ctx, chunk_objs, absence_obj,
                              seed_reg, card, title)
                recs = out.get("records", [])
                if self._records_target is not None and recs:
                    self._records_target[paper_id] = {"records": recs}
                    try:
                        from evidence_b2_tools import REC_INDEX
                        for r in recs:
                            REC_INDEX[r.get("id")] = r
                    except Exception:
                        pass
                # L2 全量=替换语义（recs 已含 L1 记录，重放 replace 免叠
                # 加膨胀）；落盘给 report_adapter（独立进程）
                self._persist_deep_records(paper_id, recs, replace=True)
                print(f"[deep_read-L2] {paper_id[:50]} background complete: "
                      f"{len(recs)} records", flush=True)
            except Exception as e:
                print(f"[deep_read-L2] {paper_id[:50]} background error: "
                      f"{str(e)[:120]}", flush=True)
            finally:
                ExternalTools._L2_RUNNING.discard(paper_id)

        _th.Thread(target=_run, daemon=True,
                   name=f"deep-l2-{paper_id[:20]}").start()

    def deep_read(self, paper_id: str, sections: list = None) -> dict:
        """L1 定向深读：抽题目相关（或指定）的 2-3 个 section 的 chunks，
        ~3 分钟内出全文级记录（result/config/finding），入账后
        findings(paper_id=...) 即刻可查。剩余 section 在后台异步补齐
        （L2，下一题起免费复利）。sections=节名/关键词列表（如
        ["ablation", "experimental setup"]）；不传=自动选题目最相关的
        前 3 节。"""
        import hashlib as _hl
        import threading as _th
        try:
            if paper_id in ExternalTools._DEEP_READ_DONE:
                return {"tool": "deep_read", "paper_id": paper_id,
                        "note": "already deep-read this session — query its "
                                "records via findings(paper_id=...)"}
            m = (self._kb_manifest or {}).get(paper_id) or {}
            # ① 全文（三通道 + 文本级缓存）
            text, src = self._fetch_full_text(paper_id, m)
            if len(text) < 5000:
                return {"tool": "deep_read", "paper_id": paper_id,
                        "error": "full text unavailable on both channels "
                                 f"(arxiv={'yes' if m.get('arxiv_id') else 'no'}, "
                                 f"sciverse={'yes' if m.get('doc_id') else 'no'})"}
            text_hash = _hl.md5(text.encode("utf-8")).hexdigest()[:16]
            title = m.get("title") or paper_id
            # ② card（缓存命中免 LLM）
            cache = self._cache_load()
            ent = cache.get(paper_id) or {}
            card = ent.get("card") if ent.get("text_hash") == text_hash else None
            if card is None:
                from kb_compiler.records.cards import build_card
                _pid, card = build_card(paper_id, text, title, self._model)
                if card is None:
                    return {"tool": "deep_read", "paper_id": paper_id,
                            "error": "card extraction failed"}
                self._cache_append(paper_id, text_hash,
                                   [{"type": "card", "card": card}])
            # ③ 全 chunk 划分 + L1 选择（agent sections / 题目重叠）
            from kb_compiler.records.deep_extract import (
                build_paper_tasks as _bpt, _call_chunk, finalize_paper as _fp)
            vocab = {"subject": [], "setup": [], "variant": [],
                     "hyperparam_items": []}
            seed_reg = {"entities": [], "surface_index": {}}
            ctx = _bpt(paper_id, text, card, seed_reg, vocab, title)
            chunks_all = [ch for ch, _ in ctx["chunk_prompts"]]
            cached = dict((ent.get("chunks") or {})
                          if ent.get("text_hash") == text_hash else {})
            sel, mode, miss_note = self._select_l1_chunks(
                chunks_all, sections, getattr(self, "_current_question", None))
            if not sel:
                return {"tool": "deep_read", "paper_id": paper_id,
                        "error": "no selectable chunks (text parse produced "
                                 "zero sections)"}
            # ④ L1 抽取：缓存命中跳过，未命中的并行抽（独立车道）
            todo = [(ch, p) for ch, p in ctx["chunk_prompts"]
                    if ch["chunk_id"] in {c["chunk_id"] for c in sel}
                    and cached.get(ch["chunk_id"]) is None]
            n_cached = len(sel) - len(todo)
            if todo:
                from concurrent.futures import ThreadPoolExecutor as _TP
                with _TP(max_workers=min(3, len(todo))) as ex:
                    objs = list(ex.map(
                        lambda pr: _call_chunk(pr[1], self._model), todo))
                new_entries = [{"chunk_id": ch["chunk_id"], "obj": obj}
                               for (ch, _), obj in zip(todo, objs)
                               if isinstance(obj, (dict, list))]
                self._cache_append(paper_id, text_hash, new_entries)
                cached = dict(self._cache_load()
                              .get(paper_id, {}).get("chunks") or {})
            # ⑤ L1 finalize（选定 chunk 子集；absence 留给 L2）
            sel_ids = {c["chunk_id"] for c in sel}
            ctx_l1 = {"inj": ctx["inj"],
                      "chunks": [c for c in ctx["chunks"]
                                 if c["chunk_id"] in sel_ids],
                      "chunk_prompts": [(ch, p) for ch, p in
                                        ctx["chunk_prompts"]
                                        if ch["chunk_id"] in sel_ids],
                      "absence_prompt": ctx["absence_prompt"]}
            chunk_objs = {ch["chunk_id"]: cached.get(ch["chunk_id"])
                          for ch, _ in ctx_l1["chunk_prompts"]}
            _p, out = _fp(paper_id, ctx_l1, chunk_objs, None,
                          seed_reg, card, title)
            recs = out.get("records", [])
            # ⑥ 会话 KB 入账（typed tools 数据源）+ REC_INDEX 同步
            #    + 持久化（report_adapter 的 EvidenceStore 是独立进程——
            #    批10 第三断口：不落盘=深记录回指解析不了→空壳报告）
            if self._records_target is not None and recs:
                self._records_target.setdefault(paper_id, {"records": recs})
                try:
                    from evidence_b2_tools import REC_INDEX
                    for r in recs:
                        REC_INDEX[r.get("id")] = r
                except Exception:
                    pass
            self._persist_deep_records(paper_id, recs, replace=False)
            ExternalTools._DEEP_READ_DONE.add(paper_id)
            # ⑦ L2 异步后台：剩余 chunk 增量补齐（缓存命中跳过）+ absence
            n_rest = len(chunks_all) - len(sel)
            if n_rest > 0:
                self._l2_background(paper_id, text, card, title, text_hash)
            from collections import Counter as _C
            kinds = dict(_C(r.get("kind") for r in recs))
            secs = list(dict.fromkeys(
                (c.get("section") or "?") for c in sel))[:6]
            obs = {"tool": "deep_read", "paper_id": paper_id,
                   "mode": f"L1/{mode}", "n_records": len(recs),
                   "kinds": kinds, "sections_read": secs,
                   "n_chunks_read": len(sel), "total_chunks": len(chunks_all),
                   "n_cache_hits": n_cached, "src": src,
                   "note": (f"L1 directed extraction complete — {len(recs)} "
                            f"full-text records from sections "
                            f"{json.dumps(secs, ensure_ascii=False)} now "
                            f"queryable via findings(paper_id=\"{paper_id}\") "
                            f"and card(entity=...); remaining {n_rest} "
                            "sections are extracting in the background and "
                            "will be queryable in later questions")}
            if miss_note:
                obs["section_miss"] = miss_note[:400]
            return obs
        except Exception as e:
            return {"tool": "deep_read", "paper_id": paper_id,
                    "error": f"deep read failed: {str(e)[:120]}"}


def attach_external_tools(kb, views: dict, kb_manifest: dict,
                          model: str = "local:Qwen3.8-27B",
                          registry: dict = None, blocklist: list = None,
                          backflow_path: str = None,
                          tier_db: str = None,
                          manifest_path: str = None,
                          deep_cache_path: str = None,
                          deep_text_dir: str = None,
                          deep_records_path: str = None) -> ExternalTools:
    """Inject the external tools onto a KBTools instance and extend the
    harness tool whitelist/catalog at runtime. `registry` enables mentions
    backflow (extract_paper -> genealogy attachment); `backflow_path`
    persists + replays library growth across sessions; `tier_db` puts
    coarse/deep extraction state into the sci-evo SQLite registry.
    `deep_cache_path`/`deep_text_dir` persist deep_read chunk records and
    fetched full texts (L1/L2 incremental — 抽过的 chunk 永不重付)。
    `records_target` = kb.records（deep_read 记录入账 typed tools 数据源）。

    批 5-9 复盘（2026-09-28）：deep_read 曾只进了 TOOL_SIGS 没进
    TOOL_WHITELIST 也没挂 kb——模型 40+ 次调用全部撞 'unknown tool'
    （248 字符 obs），Sciverse 分页修复根本没被触达。本修复三线齐挂。"""
    ext = ExternalTools(views, kb_manifest, model, registry=registry,
                        blocklist=blocklist, backflow_path=backflow_path,
                        tier_db=tier_db)
    ext._manifest_path = manifest_path
    ext._deep_cache_path = deep_cache_path
    ext._deep_text_dir = deep_text_dir
    ext._deep_records_path = deep_records_path
    ext._records_target = kb.records   # deep_read 记录直落入账
    if backflow_path:
        try:
            from kb_compiler.records.backflow import load_backflow
            n = load_backflow(views, backflow_path)
            if n:
                print(f"[backflow] replayed {n} external papers into "
                      f"genealogy", flush=True)
        except Exception:
            pass
    kb.search_papers = ext.search_papers
    kb.gap_search = ext.gap_search
    kb.lineage_walk_ext = ext.lineage_walk
    kb.citation_graph = ext.citation_graph
    kb.extract_paper = ext.extract_paper
    kb.deep_read = ext.deep_read
    kb._ext_tools = ext   # harness run_question 钩子：注入当前题目
    # whitelist/catalog extension (runtime, fork-side; frozen home untouched)
    try:
        from evidence_b2_tools import TOOL_WHITELIST, TOOL_SIGS
        TOOL_WHITELIST.update({"search_papers", "gap_search",
                               "lineage_walk_ext", "citation_graph",
                               "extract_paper", "deep_read"})
        TOOL_SIGS.update({
            "search_papers": "(query,k?)",
            "gap_search": "(query)",
            "lineage_walk_ext": "(entity,direction?)",
            "citation_graph": "(doi?,title?,direction?)",
            "extract_paper": "(title,abstract)",
            "deep_read": "(paper_id,sections?)",
        })
    except Exception:
        pass
    return ext
