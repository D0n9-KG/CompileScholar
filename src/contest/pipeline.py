# -*- coding: utf-8 -*-
"""SPARBench practice runner (2026-08-27) — large-gold regime (mean 11.1/query,
gold = expert-verified SEMANTICALLY RELEVANT papers, not citation lists).

Recall-oriented: query rewrite (multi-view sub-queries + synonyms) -> OpenAlex
keyword+semantic parallel recall (top-40 each) -> LLM relevance filter with
graded output (highly/partially) -> return top ranked list (up to 15).

Usage: python .research_tmp/spar_runner.py --limit 10 [--offset 0]
"""
import argparse, json, os, re, sys, time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# path bootstrap only — NO chdir (a module-level chdir hijacks the importer's
# cwd: importing contest.pipeline from a runner in another directory silently
# broke the runner's relative paths — the R9 FileNotFoundError root cause).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from granular_agent.llm_client import call_paratera, parse_json_response
from contest.cost_ledger import ledger

BASE_URL = "http://localhost:8000"
QS = "_spar_tmp/benchmark/spar_bench.jsonl"


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def rewrite_query(query: str) -> list[str]:
    """Multi-view rewrite: broad 2-3 token variants + survey/review variants +
    specific facet queries. S2 bulk is AND-token matching — long 6-token queries
    over-constrain (measured: gold found by 'synthetic data generation survey',
    missed by 'synthetic data generation fine-tuning generalization')."""
    p = ("A researcher's academic search request:\n"
         f"«{query}»\n\n"
         "Generate 6-8 SHORT keyword search queries for a paper search engine that "
         "AND-matches all tokens. Mix THREE kinds: (a) 2-3 broad core-topic queries "
         "(e.g. 'synthetic data generation', 'domain generalization'); (b) ONE "
         "'<core topic> survey' or '<core topic> review' variant (experts often "
         "want representative/survey papers); (c) 2-3 specific facet queries "
         "(technique+task, key properties). Keep named methods verbatim. No "
         "question phrasing, no stopwords.\n"
         'Output JSON: {"queries": ["...", ...]}')
    raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=400, enable_thinking=False)
    obj = parse_json_response(raw) or {}
    qs = [q for q in (obj.get("queries") or []) if q][:8]
    qs = qs or [query]
    # provider-side nondeterminism (temp 0 still varies) → pool and F1 vary run-to-run
    # (measured q0: 0.16 vs 0.40). Cache the rewrite so runs are comparable.
    try:
        import hashlib
        key = hashlib.md5(query.encode()).hexdigest()[:12]
        cpath = f".research_tmp/contest_survey/_rewrite_cache/{key}.json"
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        if os.path.exists(cpath):
            return json.loads(open(cpath, encoding="utf-8").read())
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False)
    except Exception:
        pass
    return qs


_CACHE_DIR = ".research_tmp/contest_survey/_oa_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _search_one(q: str, mode: str, k: int):
    """OpenAlex search via sci-evo, with a disk cache keyed by (query, mode).
    Cache-first: a repeated practice run never re-hits the API (429 lesson).
    v2 key: candidate rows now carry citation_count (authority signal) — the
    v1 cache predates the field, so bump to force one refetch."""
    import hashlib, urllib.request
    key = hashlib.md5(f"crossref|{q}|{mode}|{k}|v2".encode()).hexdigest()[:16]
    cpath = os.path.join(_CACHE_DIR, key + ".json")
    if os.path.exists(cpath):
        try:
            ledger._record("http", source="s2bulk-cache", dt=0.0, ok=True)
            return json.loads(open(cpath, encoding="utf-8").read())
        except Exception:
            pass
    body = json.dumps({"query": q, "sources": ["crossref"], "mode": mode,
                       "limit": k, "dry_run": True}).encode()
    req = urllib.request.Request(BASE_URL + "/api/library/discovery/search",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            cands = json.loads(r.read()).get("source_candidates", []) or []
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False)
        return cands
    except Exception:
        return []


def recall(query: str) -> list[dict]:
    """Parallel recall: each rewritten query x {keyword, semantic}, merged,
    PLUS S2 bulk search (the only arXiv-covering source up today — crossref
    gold coverage measured at 4%: arXiv-only golds are invisible to it).
    Candidates carry _q (their recall query) for per-query trimming later.
    Graph-expanded rewrites (hypergraph concept variants, form A) are appended
    when the query matches graph concepts — the graph participates in recall,
    not just reranking."""
    queries = rewrite_query(query)
    try:
        from granular_agent.query_expander import (load_concept_dict,
                                                   expanded_rewrites)
        _CD = getattr(recall, "_cd", None)
        if _CD is None:
            _CD = recall._cd = load_concept_dict()
        queries = expanded_rewrites(query, queries, _CD)[:10]
    except Exception:
        pass
    # ①③ schema intent expansion (GOAL 2026-08-29; GO + holdout-GO,
    # goal_log_0829.md): frozen-schema patterns -> entity-vocabulary queries
    # + one representative-narrowing variant per pattern. Blind (query+schema
    # only). Validated: dev +30.4pt / holdout +16.7pt pool coverage.
    try:
        from contest.intent_recall import schema_expansion_queries
        queries = (queries + schema_expansion_queries(query))[:16]
    except Exception:
        pass
    # term-of-art field disambiguation (GOAL 2026-08-29 深夜, env-gated,
    # default OFF). Diagnosed on AutoScholar q1: context-free queries hide
    # the gold's FIELD ("reconstruction-based techniques" = anomaly
    # detection jargon, not image reconstruction); verbatim S2 AND Google
    # both land in the wrong field. One LLM call enumerates field readings
    # with each field's own title vocabulary.
    if os.environ.get("CONTEST_DISAMBIG") == "on":
        try:
            from contest.disambig_recall import disambiguation_queries
            queries = (queries + disambiguation_queries(query))[:20]
        except Exception:
            pass
    jobs = [(q, m) for q in queries for m in ("keyword", "semantic")]
    out, seen = [], {}
    # Google source (SerpAPI; GOAL 2026-08-29 晚, env-gated, default OFF —
    # budget ~249 searches/month makes every run count). GA-baseline evidence:
    # bare site:arxiv.org Google = 0.2451 SPARBench F1 (SPAR paper table),
    # strongest public proxy for the expert-picked-representative gold culture.
    # Appends BEFORE the S2/crossref merge loop so existing-pool papers keep
    # a google:* _q group (<=20 candidates) that survives per-query trim as a
    # whole group — Google's ranking signal passes intact into the funnel.
    if os.environ.get("CONTEST_GOOGLE") in ("on", "plain"):
        try:
            from contest.google_recall import google_recall
            for c in google_recall(query, queries,
                                   plain=os.environ["CONTEST_GOOGLE"] == "plain"):
                key = _norm_title(c.get("title", ""))
                if key and key not in seen:
                    seen[key] = c
                    out.append(c)
        except Exception:
            pass

    def _merge(cands, q):
        for c in cands:
            key = _norm_title(c.get("title", ""))
            if key and key not in seen:
                seen[key] = c
                c["_q"] = q
                out.append(c)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(_search_one, j[0], j[1], 40) for j in jobs]
        futs += [ex.submit(_s2_bulk_one, q) for q in queries]
        for f, j in zip(futs, [(q, m) for q in queries for m in ("keyword", "semantic")] + [(q, "") for q in queries]):
            _merge(f.result(), j[0])
    # ③b slot-fill direct recall (GOAL 2026-08-29): matched patterns ->
    # graph hyperedges -> provenance papers straight into the pool. No extra
    # LLM (intent_parse is disk-cached; graph lookup is offline). Dev-set
    # gold∩graph was 0/23 — the value grows with the explore graph; wired
    # because the design says so, measured honestly in the arm numbers.
    try:
        from contest.intent_recall import graph_direct_hits, intent_parse
        matches = intent_parse(query)
        if matches:
            papers_all = []
            for h in graph_direct_hits([m["pattern_id"] for m in matches],
                                       max_edges=20):
                for p in h.get("papers", []):
                    papers_all.append((h['pattern_type'], p))
            # resolve REAL citation counts (GOAL 2026-08-30 fix): graph papers
            # carried citationCount=0 into a grader prompt that reads
            # "REPRESENTATIVE high-citation works are H" — systematically
            # suppressed. S2 batch resolve (same resolver as google_recall).
            try:
                from contest.google_recall import _s2_batch_resolve
                # no arxiv ids on graph papers — resolve by title via match
                # endpoint is 1-call-each; cap at 12 (graph hits are few)
                from contest.google_recall import _s2_match_title
                for _, p in papers_all[:12]:
                    m = _s2_match_title(p.get("title", ""))
                    if m:
                        p["citationCount"] = m.get("citationCount", 0)
                        p["year"] = m.get("year") or p.get("year", "")
                        p["abstract"] = m.get("abstract", "")
            except Exception:
                pass
            for ptype, p in papers_all:
                key = _norm_title(p.get("title", ""))
                if key and key not in seen:
                    seen[key] = {"title": p["title"],
                                 "year": p.get("year", ""),
                                 "citationCount": p.get("citationCount", 0),
                                 "_q": f"graph:{ptype}"}
                    out.append(seen[key])
    except Exception:
        pass
    # ② schema-gated citation-intent exploration (GOAL 2026-08-29), env-gated
    # (single code path, CONTEST_ORDER precedent): ingest top pool papers
    # cites-only into the 'explore' graph, mine their classified references,
    # resolve mined titles via S2 bulk back into the pool. Default OFF — the
    # +①③ ablation arm must stay clean; set CONTEST_EXPLORE=on to enable.
    if os.environ.get("CONTEST_EXPLORE") == "on":
        try:
            from contest.explore_recall import explore
            k = int(os.environ.get("CONTEST_EXPLORE_K", "2"))
            new, _stats = explore(query, out, k_seeds=k)
            for c in new:
                key = _norm_title(c.get("title", ""))
                if key and key not in seen:
                    seen[key] = c
                    out.append(c)
        except Exception:
            pass
    # memory recall (GOAL 2026-08-30 下午): semantic match of the query
    # against ALL titles ever mined by ② across previous queries (cached
    # citreports — 4k+ titles, growing). Compensates the one-hop structural
    # ceiling: 5/10 boost1 misses sat in titles mined by earlier queries.
    # Zero new LLM cost; own _q group so MECH_BOOST trim quota applies.
    if os.environ.get("CONTEST_MEMORY") == "on":
        try:
            from contest.memory_recall import memory_recall, hub_recall
            for c in memory_recall(query):
                key = _norm_title(c.get("title", ""))
                if key and key not in seen:
                    seen[key] = c
                    out.append(c)
            # hub recall — minimal 2nd hop (GOAL 2026-08-30): papers cited
            # by >=3 distinct ingested seeds = representative-paper detector
            # (cross-seed co-citation); pure cache query, zero ingestion
            for c in hub_recall(query):
                key = _norm_title(c.get("title", ""))
                if key and key not in seen:
                    seen[key] = c
                    out.append(c)
        except Exception:
            pass
    return out


_S2_CACHE_DIR = ".research_tmp/contest_survey/_s2_cache"
os.makedirs(_S2_CACHE_DIR, exist_ok=True)


def _s2_bulk_one(q: str, max_pages: int = 2) -> list[dict]:
    """S2 /paper/search/bulk (AND-token matching over title+abstract) — returns
    candidates shaped like the sci-evo rows (title/year/venue/citation_count),
    disk-cached. Bulk endpoint is not behind the普通search 429 wall."""
    import hashlib, urllib.request
    # v2: early runs cached entries without citationCount (field added later);
    # citation presort silently buried them. Version-bump forces one clean refetch.
    key = hashlib.md5(f"s2bulk|{q}|{max_pages}|v3-abstract".encode()).hexdigest()[:16]
    cpath = os.path.join(_S2_CACHE_DIR, key + ".json")
    if os.path.exists(cpath):
        try:
            ledger._record("http", source="s2bulk-cache", dt=0.0, ok=True)
            return json.loads(open(cpath, encoding="utf-8").read())
        except Exception:
            pass
    out, token, n_err = [], None, 0
    for _ in range(max_pages):
        params = {"query": q, "fields": "title,year,citationCount,venue,externalIds,abstract"}
        if token:
            params["token"] = token
        url = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(url, headers={"User-Agent": "LogicKG-research/0.1"})
        try:
            d = ledger.http("s2bulk", url, timeout=40)
            if d is None:
                raise RuntimeError("s2bulk fetch failed")
        except Exception:
            n_err += 1
            if n_err >= 2:
                break
            time.sleep(3)
            continue
        out.extend(d.get("data") or [])
        token = d.get("token")
        if not token:
            break
        time.sleep(1.2)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def embedding_rerank(query: str, cands: list[dict], top_k: int = 150) -> list[dict]:
    """Semantic rerank between recall and grading (2026-08-28, the 'ranked
    recall' layer). Measured: S2 bulk is an unranked firehose (PPO, cited 30k,
    sampled out of page 1); qwen3-embedding:8b cosine floats gold to top-10
    of a mixed 85-pool (q8 injection test: ranks 3/7/10). This replaces the
    citation-count presort as the grading-window selector — citation count
    stays in the grader prompt (authority-in-prompt, the paired-A/B verdict).
    Falls back to citation-count order if embeddings fail (honest degrade)."""
    try:
        from granular_agent.hypergraph_evolution import _embed_texts_robust
        texts = [query] + [(c.get("title") or "") + ". " + (c.get("abstract") or "")[:400]
                           for c in cands[:400]]
        embs = _embed_texts_robust(texts)
        if not embs or len(embs) != len(texts):
            raise RuntimeError("embedding unavailable")
        import math
        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0
        scored = sorted(zip(cands[:400], embs[1:]),
                        key=lambda t: -_cos(embs[0], t[1]))
        # carry the semantic score on the candidate — the output ranker
        # (rerank_authority) reuses it instead of re-embedding
        for c, e in scored:
            c["_sem"] = round(_cos(embs[0], e), 4)
        return [c for c, _ in scored[:top_k]]
    except Exception:
        # degrade: citation-count order (the old behavior)
        return sorted(cands, key=lambda c: -(c.get("citation_count")
                                              or c.get("citationCount") or 0))[:top_k]


def grade_and_rank(query: str, cands: list[dict], max_out: int = 15,
                   rank: str = "authority", semantic_trim: bool = True,
                   # GOAL 2026-08-29 晚: window reverted to 150 (pre-registered
                   # decision). The 300 default was reverted because the
                   # funnel-replay needed to justify it costs ~10h under
                   # evening embedding throttling (4.3 texts/s measured) for
                   # an expected +-0.002, and mechanism reasoning says
                   # embed-rank 151-300 gold scores too low in hybrid ordering
                   # to crack the top-20 cap (corroborated by the cap30
                   # negative: marginal precision 0.012 < half the average).
                   grade_window: int = 150, grade_chunk: int = 50) -> list[dict]:
    """LLM grades candidates highly/somewhat/not on the query's facets; then
    rank. rank='authority' fuses the grade with a citation-count signal
    (SPARBench diagnosis: gold = expert-picked REPRESENTATIVE papers; pure
    relevance ranking returns niche recent works instead). rank='llm' keeps
    the old grade-order output as the baseline arm.

    semantic_trim/grade_window/grade_chunk default to the SPAR-measured
    configuration. The demo endpoint passes a faster config (citation trim +
    one global embedding pass + a single grading chunk) — it changes the demo
    latency, NOT the SPARM arm behavior (spar_runner calls with defaults)."""
    if not cands:
        return []
    # pools can reach 4000+ with S2 bulk (unranked firehose): before the grading
    # window, sort by citation count so the representative end of the pool is what
    # the LLM sees (gold diagnosis: expert-picked representative papers).
    # Measured on q0: global citation sort puts facet-matched gold (cited 12-65)
    # at ranks 128-254 — so first cap PER QUERY SOURCE (top 20 by citation per
    # recall query, preserving facet diversity), then global sort.
    by_query: dict[str, list[dict]] = {}
    for c in cands:
        by_query.setdefault(c.get("_q") or "", []).append(c)
    trimmed = []
    # GOAL 2026-08-30: mechanism groups (explore:*/graph:*) carry pool golds
    # that die OUTSIDE the grading window (measured q21-22 boost run: in-pool
    # golds incl. Visual Instruction Tuning all crowded out of the 150 window
    # by 13k-pool wide queries). Their titles came from REAL reference lists
    # (title-precise by construction) — they get a HIGHER per-group trim
    # quota than generic search groups. Env-gated so the running arms' module
    # state (already loaded) is unaffected; RSQ/pre-registered runs only.
    _mech_boost = os.environ.get("CONTEST_MECH_BOOST") == "on"
    for ql, lst in by_query.items():
        is_mech = str(ql).startswith(("explore:", "graph:"))
        cap_q = 40 if (is_mech and _mech_boost) else 20
        if semantic_trim:
            # per-query trim SEMANTIC (embedding cosine to the query), citation
            # only as tiebreak — measured: global citation sort buried facet gold
            # at ranks 128-254; embedding rerank floats injected gold to top-10/85.
            ranked = embedding_rerank(query, lst, top_k=cap_q)
        else:
            # fast (demo) trim: citation order per recall query, no per-group
            # embedding — keeps facet diversity, one global embed happens below.
            ranked = sorted(lst, key=lambda c: -(c.get("citation_count")
                                                 or c.get("citationCount") or 0))[:cap_q]
        trimmed.extend(ranked)
    if len(trimmed) < 100:            # small pools: fall back to global sort
        trimmed = cands
    cands = embedding_rerank(query, trimmed, top_k=grade_window)
    # chunked grading: a 150-item monolithic listing measurably dilutes attention
    # (q2: 3 window-gold present, 0 graded; isolated or 50-chunked: 3/3). Grade in
    # 50-item chunks and merge — chunk-local H inflation is handled by the rank cap.
    # (2026-08-28) abstract-assisted grading: title-only grading was systemically
    # blind to niche-venue gold ('Research on...' generic titles — q2: 7/12 in
    # pool, 0 graded). Abstracts come free with the S2 bulk v3 cache (82%
    # coverage); the first ~180 chars disambiguate generic titles at modest
    # token cost. Judge by title+abstract, year, citations as before.
    high, some = [], []
    for ci in range(0, len(cands), grade_chunk):
        chunk = cands[ci:ci + grade_chunk]
        listing = "\n".join(
            f"{i}: {c.get('title','')} ({c.get('year','')}, cited {c.get('citation_count') or c.get('citationCount') or 0})"
            + (f" — {(c.get('abstract') or '')[:180]}" if c.get("abstract") else "")
            for i, c in enumerate(chunk))
        p = ("A researcher's academic search request:\n"
             f"«{query}»\n\n"
             "Candidate papers (index: title (year, citation count) — abstract excerpt):\n"
             + listing + "\n\n"
             "Grade each candidate: H = directly about the request's specific "
             "technique/task/point; S = relevant to the broader request but not the "
             "specific point; N = not relevant. Judge by title AND abstract. When "
             "the request asks for representative/top-tier/influential works or "
             "wants to 'expand ideas'/'get started', REPRESENTATIVE high-citation "
             "works directly on the topic are H — recent niche variants of the same "
             "topic stay S. Be GENEROUS with S — the request wants comprehensive "
             "coverage.\n"
             'Output JSON: {"H": [indices], "S": [indices]}')
        if os.environ.get("CONTEST_PRECISION") == "on":
            # precision mode (GOAL 2026-08-29 深夜): small-gold regime — the
            # request is a SPECIFIC question typically answered by 1-5 papers
            # (AutoScholar: mean gold 2.4/query). Generous-S grading drowns the
            # exact match in near-neighbors (q2: gold in pool, out of window).
            # Strict prompt: only precise matches are H/S; output is H-first
            # with a small cap (rerank_authority honors CONTEST_CAP).
            p = ("A researcher's SPECIFIC academic search request:\n"
                 f"«{query}»\n\n"
                 "Candidate papers (index: title (year, citation count) — abstract excerpt):\n"
                 + listing + "\n\n"
                 "This kind of request is typically answered by only 1-5 papers. "
                 "Grade each candidate: H = the paper IS one of the specific "
                 "studies the request asks about (its core contribution matches "
                 "the named technique/task/point); S = closely adjacent work on "
                 "the same specific point but not itself an answer; N = merely "
                 "the same broad topic. Judge by title AND abstract. Be "
                 "CONSERVATIVE: when in doubt between S and N, choose N — the "
                 "requester wants the exact studies, not the neighborhood.\n"
                 'Output JSON: {"H": [indices], "S": [indices]}')
        raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=600, enable_thinking=False)
        if raw is None:
            # grading is the conversion bottleneck — a transient provider
            # failure must not silently zero a query (GOAL 2026-08-29: q1 of
            # the final run pred=0 from a rate-limit burst). One bounded retry.
            time.sleep(8)
            raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=600, enable_thinking=False)
        obj = parse_json_response(raw) or {}
        for x in (obj.get("H") or []):
            try:
                i = int(x)
                if 0 <= i < len(chunk):
                    high.append(chunk[i])
            except (ValueError, TypeError):
                pass
        for x in (obj.get("S") or []):
            try:
                i = int(x)
                if 0 <= i < len(chunk):
                    some.append(chunk[i])
            except (ValueError, TypeError):
                pass
    # archive the graded pool for paired offline A/B (ranker comparison without
    # re-grading — grader variance between runs measured at q2: 0.0 vs 0.148).
    # cands is ALREADY windowed here (embedding_rerank top_k=grade_window) —
    # the old cands[:150] slice silently truncated window-300 runs and made the
    # offline order-replay unfaithful (GOAL 2026-08-29 fix).
    try:
        _archive_grade(query, cands, high, some)
    except Exception:
        pass
    if rank == "llm":
        return (high + some)[:max_out]
    return rerank_authority(high, some, None, query=query)


_GRADE_ARCHIVE = ".research_tmp/contest_survey/_grade_archive.jsonl"


def _slot_token_sets(query: str) -> list[set]:
    """⑤ helper: token sets of the parsed query's slot-filler phrases (each
    set must be fully contained in a candidate title to count). Disk-cached
    per query (intent_parse is cached; this is a pure transform)."""
    import hashlib
    cache = getattr(_slot_token_sets, "_c", None)
    if cache is None:
        cache = _slot_token_sets._c = {}
    k = hashlib.md5(query.encode()).hexdigest()[:12]
    if k in cache:
        return cache[k]
    sets = []
    try:
        from contest.intent_recall import intent_parse
        for m in intent_parse(query):
            for v in (m.get("slots") or {}).values():
                toks = {t for t in re.sub(r"[^a-z0-9 ]", " ",
                                          (v or "").lower()).split()
                        if len(t) > 3}
                if toks:
                    sets.append(toks)
    except Exception:
        pass
    cache[k] = sets
    return sets


def _archive_grade(query, cands, high, some):
    """Append {query, titles, H, S} once per (query) — the paired-A/B material."""
    import hashlib
    qid = hashlib.md5(query.encode()).hexdigest()[:12]
    line = {"qid": qid, "query": query,
            "titles": [c.get("title", "") for c in cands],
            "cited": [c.get("citation_count") or c.get("citationCount") or 0 for c in cands],
            "H": [c.get("title", "") for c in high],
            "S": [c.get("title", "") for c in some]}
    with open(_GRADE_ARCHIVE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def rerank_authority(high: list[dict], some: list[dict], max_out: int | None = None,
                     query: str = "") -> list[dict]:
    """R10: adaptive output + hybrid ordering. Two measured failure modes
    fixed at once:
      R8 (citation-only order): specific-query gold (low-cited niche papers)
      buried under fashionable non-gold — q4 stuck at 0.091 for six rounds.
      R9 (semantic-only order): broad/survey queries collapsed (q1 0.136→0.045)
      — when every on-topic paper scores similar, citation authority was the
      only thing separating 'expert-picked' from 'merely relevant'.
    So: semantic 60% + authority 40% within grade (hybrid), AND the output cap
    grows with the H-grade size (max(12, len(H)+3)) — a 13-H query outputs 16,
    a 2-H query outputs 12; capacity follows the grader's own confidence."""
    import math
    if max_out is None:
        # fixed moderate cap. R10 lesson: keying output size to len(H) trusts
        # an inflated signal (abstract grading pushes H to 34-109 -> output
        # exploded 37-112/query, precision collapsed). GOAL 2026-08-29 cap
        # experiment 20->30: NEGATIVE — micro-F1 0.032 vs baseline 0.0347
        # (+6 tp but P diluted 0.027->0.022 across 1500 pred slots; R rose
        # 0.049->0.060). Uniform cap expansion recovers fewer graded-gold
        # than the precision it costs. Reverted to 20; the remaining
        # conversion lever is the grading WINDOW (goal_log_0829.md).
        # CONTEST_CAP env: per-benchmark output discipline (small-gold
        # benchmarks want 5, e.g. AutoScholar mean gold 2.4/query — cap 20
        # caps F1 at 0.21 even with perfect recall). Env knob, single code
        # path (stale-copy lesson).
        max_out = int(os.environ.get("CONTEST_CAP", "20"))
    def _ck(c):
        try:
            return math.log10(1 + int(c.get("citation_count") or c.get("citationCount") or 0))
        except (TypeError, ValueError):
            return 0.0
    pool = high + some
    pool_max = max((_ck(c) for c in pool), default=1.0) or 1.0

    # order-mode switch (50-query 3-arm experiment; env-gated, single code path —
    # the stale-copy lesson). hybrid = semantic-majority + citation-minority;
    # semantic/citation = pure arms.
    _mode = os.environ.get("CONTEST_ORDER", "hybrid")
    # ⑤ schema-match rerank (GOAL 2026-08-29): candidates whose title covers
    # a parsed slot-filler phrase of the query intent get a small boost.
    # Lexical (no LLM here), cached per query, blind to gold by construction
    # (slot fillers come from the query text via the frozen schema).
    _slot_sets = _slot_token_sets(query)
    def _slot_hit(c):
        if not _slot_sets:
            return False
        tt = set(re.findall(r"[a-z0-9]{4,}", (c.get("title") or "").lower()))
        return any(ts <= tt for ts in _slot_sets)
    def _score(c, grade_w):
        sem = c.get("_sem") or 0.0          # embedding cosine to the query
        auth = _ck(c) / pool_max            # normalized log citations
        if _mode == "semantic":
            s = grade_w * (0.3 + 0.7 * sem)
        elif _mode == "citation":
            s = grade_w * (0.5 + 0.5 * auth)
        else:
            s = grade_w * (0.3 + 0.6 * sem + 0.2 + 0.2 * auth)  # hybrid
        return s * (1.15 if _slot_hit(c) else 1.0)

    ranked = sorted(
        [(_score(c, 1.0), -i, c) for i, c in enumerate(high)]
        + [(_score(c, 0.55), -i, c) for i, c in enumerate(some)],
        key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in ranked[:max_out]]


def score(pred_titles, gold_titles):
    pred = {_norm_title(t) for t in pred_titles if t}
    gold = {_norm_title(t) for t in gold_titles if t}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "n_pred": len(pred), "n_gold": len(gold),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def run(limit, offset, out_path, rank: str = "authority"):
    lines = open(QS, encoding="utf-8").read().strip().split("\n")[offset:offset + limit]
    results = []
    t0 = time.time()
    for n, line in enumerate(lines):
        d = json.loads(line)
        q, gold = d["question"], d["answer"]
        try:
            cands = recall(q)
            kept = grade_and_rank(q, cands, rank=rank)
            pred = [c.get("title", "") for c in kept]
        except Exception as e:
            pred, cands, kept = [], [], []
            print(f"  [err] {e!r}", flush=True)
        s = score(pred, gold)
        results.append({"question": q, "gold": gold, "pred": pred,
                        "n_recall": len(cands), "score": s})
        print(f"[{n+1}/{len(lines)}] recall={len(cands)} pred={len(pred)} "
              f"P={s['precision']} R={s['recall']} F1={s['f1']}", flush=True)
    tp = sum(r["score"]["tp"] for r in results)
    np_ = sum(r["score"]["n_pred"] for r in results)
    ng = sum(r["score"]["n_gold"] for r in results)
    mp = tp / np_ if np_ else 0
    mr = tp / ng if ng else 0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0
    macro = sum(r["score"]["f1"] for r in results) / len(results) if results else 0
    print(f"\n[spar-practice rank={rank}] {len(results)} queries {time.time()-t0:.0f}s | "
          f"micro-F1={mf1:.3f} (P={mp:.3f} R={mr:.3f}) | macro-F1={macro:.3f}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"results": results,
                   "summary": {"micro_f1": round(mf1, 4), "micro_p": round(mp, 4),
                               "micro_r": round(mr, 4), "macro_f1": round(macro, 4),
                               "n": len(results), "rank": rank}},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rank", choices=["authority", "llm"], default="authority")
    a = ap.parse_args()
    run(a.limit, a.offset,
        f".research_tmp/contest_survey/spar_practice_{a.rank}_{a.offset}_{a.limit}.json",
        rank=a.rank)
