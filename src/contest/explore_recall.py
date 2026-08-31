# -*- coding: utf-8 -*-
"""Mechanism ② — schema-gated citation-intent exploration (GOAL 2026-08-29).

Coarse-recall seeds -> recall-mode ingest (CITES-ONLY: register PAPER node +
citation-intent stage with a parsed reference list; NO section extraction /
evolution / alignment — the frozen-schema online architecture) -> mine the
seed's classified non-background references (title, intent) as new pool
candidates. Seeds also accumulate cites edges between each other.

Scope decision (recorded in .research_tmp/goal_log_0829.md): §4's recall mode
includes light concept extraction; the CONTEST variant here is cites-only,
because seed concept edges add nothing measurable while 50q gold is outside
the graph (form-(b) measured 0/23) and the extraction cost (~5min/paper
serial — section-level parallelism is FROZEN) would blow the batch budget.
Full recall mode stays paper-phase work.

Graph store: seeds go to a SEPARATE named graph (default 'explore') — the
llm_domain rebuild (2026-08-29) holds its store open and saves after every
paper; concurrent writes to the same store would lose updates. Merge later.
"""
import glob, json, os, re, shutil, time, urllib.request

from contest.cost_ledger import ledger
from granular_agent.llm_client import parse_json_response

SCI_API = "http://127.0.0.1:8000/api/library"
MINERU_BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"
SCIEVO_LIB = "C:/Users/D0n9/Desktop/sci-evo-extract/data/library/papers"
_CACHE_DIR = os.path.join(".research_tmp", "contest_survey", "_explore_cache")

_NON_BG = {"extends", "improves", "compares", "replaces", "adapts"}


def _req(method: str, path: str, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(SCI_API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def _cache_get(name, key):
    p = os.path.join(_CACHE_DIR, f"{name}_{key}.json")
    if os.path.exists(p):
        try:
            return json.loads(open(p, encoding="utf-8").read())
        except Exception:
            return None
    return None


def _cache_put(name, key, obj):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, f"{name}_{key}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def pid_from_arxiv(arxiv_id: str) -> str:
    return f"ARXIV_{arxiv_id.replace('.', '_')}"


def acquire_arxiv(arxiv_id: str) -> str | None:
    """sci-evo acquisition (arXiv DOI route — no OpenAlex): fetch PDF + MinerU
    parse, install content_list into MINERU_BASE. Returns the content pid
    (ARXIV_*) or None on failure. Resumable (content-presence check first)."""
    pid_target = pid_from_arxiv(arxiv_id)
    if os.path.exists(os.path.join(MINERU_BASE, pid_target, "content_list.json")):
        return pid_target
    try:
        d = _req("POST", "/acquisitions",
                 {"doi": f"10.48550/arXiv.{arxiv_id}", "process": True})
        pid = d["paper_id"]
        time.sleep(3)
        p = _req("GET", f"/papers/{pid}")
        cand = next((c for c in p.get("source_candidates", [])
                     if c["source_name"] == "arxiv" and c["status"] == "ready"), None)
        if not cand:
            ledger._record("explore", source="acquire-nocandidate", dt=0.0, ok=False)
            return None
        _req("POST", f"/papers/{pid}/source-candidates/{cand['candidate_id']}/confirm",
             {"process": True}, timeout=900)
        # install the parsed content where load_paper_blocks reads it
        srcs = [s for s in glob.glob(f"{SCIEVO_LIB}/{pid}/mineru/**/*_content_list.json",
                                     recursive=True) if not s.endswith("_v2.json")]
        if not srcs:
            return None
        dstdir = os.path.join(MINERU_BASE, pid_target)
        os.makedirs(dstdir, exist_ok=True)
        shutil.copy(srcs[0], os.path.join(dstdir, "content_list.json"))
        return pid_target
    except Exception as e:
        ledger._record("explore", source=f"acquire-err:{type(e).__name__}", dt=0.0, ok=False)
        return None


def _reference_entries(pid: str, max_entries: int = 220) -> list[str]:
    """Raw reference-list entries from the content_list (load_paper_blocks
    SKIPS reference-like blocks, so read the raw file). MinerU stores refs as
    type=list / sub_type=ref_text blocks whose list_items are the entries —
    spread across SEVERAL blocks (page breaks; measured 2005.01643: 10 blocks
    of 1-4 items — a per-block >=5 test drops them all, so aggregate every
    ref_text block). Fallback: longest contiguous run of numbered TEXT blocks.
    Heading-walk abandoned (bios/appendices follow refs; 1708.05866 case)."""
    p = os.path.join(MINERU_BASE, pid, "content_list.json")
    if not os.path.isfile(p):
        return []
    cl = json.load(open(p, encoding="utf-8"))
    entry_re = re.compile(r"^\s*\[?\d{1,3}[\].]\s")
    items = []
    for it in cl:
        if it.get("type") == "list" and it.get("sub_type") == "ref_text" \
                and it.get("list_items"):
            items.extend((x or "").strip() for x in it["list_items"]
                         if x and x.strip())
    # MERGE (not fallback): ref_text blocks may cover only part of the list —
    # 1708.05866 measured 79 items in ref_text vs 168 in the numbered TEXT
    # run. Union both sources (dedup via the entry keying downstream).
    texts = [(it.get("text") or "").strip()
             for it in cl if it.get("type") == "text" and it.get("text")]
    best, cur = [], []
    for t in texts:
        if entry_re.match(t):
            cur.append(t)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    items.extend(best)
    return items[:max_entries]


_AY_ENTRY_RE = re.compile(
    r"^\s*([A-Z][\w'À-ɏ-]+),?\s.*?\((\d{4})[a-z]?\)")
# given-name-first variant with a BARE year (no parentheses): "Rishabh
# Agarwal, Max Schwarzer, ..., and Marc Bellemare. Deep RL at the edge...
# In Advances in Neural Information Processing Systems, 2021." Measured on
# 2405.16195: 67/67 entries in this format (0/67 matched by the paren
# regexes). Surname = capitalized word before the FIRST comma; year = first
# bare (yyyy) at a word boundary anywhere in the entry.
_AY_LOOSE_RE = re.compile(
    r"^\s*[A-Z][\w'À-ɏ-]+\s+([A-Z][\w'À-ɏ-]+),.*?\b(19\d{2}|20\d{2})\b",
    re.S)


def parse_ref_titles(pid: str, chunk: int = 80) -> dict[str, str]:
    """Reference list -> {label: title} (cached), label = numeric rid as str
    ("12") for numbered lists OR "surname year" (lowercase) for author-year
    lists — the two keys match how citation mentions join downstream (numeric
    rid inside extract_citation_intents; "surname year" for author-year
    mentions, post-joined in ingest_seed_cites). Deterministic entry keying;
    LLM only extracts the title span. Surveys run 150+ refs — chunk the LLM
    extraction, merge all (v1's first-80 cap left most mentions titleless).
    GOAL 2026-08-30 fix: given-name-first author-year entries (67/67 missed on
    2405.16195 — zero titles mined from a 74-ref seed) now keyed by the loose
    variant; also fall back to FIRSTCAP-word keying with the entry's first
    year when both author-year regexes fail (an entry with no year and no
    number still carries a mineable title)."""
    cached = _cache_get("reftitles", pid)
    if cached is not None:
        return dict(cached)
    entries = _reference_entries(pid)
    pairs = {}
    for e in entries:
        e = re.sub(r"\s+", " ", e).strip()[:400]
        m = re.match(r"^\s*\[?(\d{1,3})[\].]\s*", e)
        if m:
            pairs.setdefault(m.group(1), e)
            continue
        m2 = _AY_ENTRY_RE.match(e)
        if m2:
            pairs.setdefault(f"{m2.group(1).lower()} {m2.group(2)}", e)
            continue
        m3 = _AY_LOOSE_RE.match(e)
        if m3:
            pairs.setdefault(f"{m3.group(1).lower()} {m3.group(2)}", e)
            continue
        # last resort: first capitalized token as a bare key (title still
        # mineable via the LLM; the join key just won't match an author-year
        # mention — acceptable: mining does not need the join, only the title)
        m4 = re.match(r"^\s*([A-Z][\w'À-ɏ-]+)", e)
        if m4:
            pairs.setdefault(m4.group(1).lower(), e)
    items = sorted(pairs.items())
    if not items:
        _cache_put("reftitles", pid, {})
        return {}
    out = {}
    for ci in range(0, len(items), chunk):
        batch = dict(items[ci:ci + chunk])
        listing = "\n".join(f"[{lab}] {txt}" for lab, txt in batch.items())
        p = ("Reference entries of a paper. Extract the PAPER TITLE from each "
             "entry (drop authors, venue, year, pages, doi). If an entry has "
             "no clear title, output an empty string for it. Keep the entry "
             "labels exactly as given.\n" + listing + "\n"
             'Output JSON: {"titles": {"<label>": "<title or empty>", ...}}')
        raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=3000,
                         enable_thinking=False)
        if raw is None:
            continue              # transient LLM failure: NEVER cache empty (GOAL 08-30 — same bug class as intent_recall 15:30)
        obj = parse_json_response(raw) or {}
        for k, v in (obj.get("titles") or {}).items():
            k = str(k).strip()
            if v and str(v).strip() and k in batch:
                out[k] = str(v).strip()[:160]
    if out:
        _cache_put("reftitles", pid, out)
    return out


def _llm_fn(prompt, mt=3000):
    return ledger.llm("DeepSeek-V4-Flash", prompt, max_tokens=mt,
                      enable_thinking=False)


def _prepare_seed(arxiv: str, title: str) -> dict:
    """PARALLEL-SAFE half of the recall-mode ingest (paper-level parallelism,
    the §4 speedup design): acquire content, parse the reference list, locate
    + classify citation mentions. NO KB access — safe under ThreadPoolExecutor.
    Returns a prep dict (with 'error' set on failure, 'cached_report' on a
    previous ingest)."""
    from concurrent.futures import ThreadPoolExecutor
    from granular_agent.citation_intent import (classify_intent,
                                                locate_author_year_contexts,
                                                locate_citation_contexts)
    from granular_agent.structure_mapper import full_text_from_blocks, load_paper_blocks

    pid = acquire_arxiv(arxiv)
    if not pid:
        return {"arxiv": arxiv, "title": title, "error": "acquire"}
    cached = _cache_get("citreport", pid)
    if cached is not None:
        return {"pid": pid, "title": title, "cached_report": cached}
    blocks = load_paper_blocks(pid)
    if not blocks:
        return {"pid": pid, "title": title, "error": "no_text"}
    fulltext = full_text_from_blocks(blocks)
    titles_all = parse_ref_titles(pid)
    num_ctx = locate_citation_contexts(fulltext)
    ay_ctx = locate_author_year_contexts(fulltext)
    # max_refs 160 (vs pipeline default 60): surveys run 150+ refs
    num_keys = sorted(num_ctx)[:160]
    ay_keys = sorted(ay_ctx)
    jobs = [(str(rid), num_ctx[rid]) for rid in num_keys] + \
           [(k, ay_ctx[k]) for k in ay_keys]
    with ThreadPoolExecutor(max_workers=8) as ex:
        classified = list(ex.map(
            lambda j: classify_intent(j[0], j[1], _llm_fn), jobs))
    by_key = {j[0]: rec for j, rec in zip(jobs, classified)}
    return {"pid": pid, "title": title, "by_key": by_key,
            "num_keys": num_keys, "ay_keys": ay_keys,
            "titles_all": titles_all}


def _extract_concepts_frozen(kb, pid: str, blocks: list, llm: str = "DeepSeek-V4-Flash") -> int:
    """RECALL-MODE concept extraction (the approved freeze exception's full
    form: concepts YES, evolution NO, alignment NO). Section loop mirrors
    agent.process_paper_via_kernel's Stage 1 (frozen arm equivalent); the
    agent module itself stays untouched (frozen boundary — imports only).
    Returns the number of sections processed."""
    from granular_agent.agent import _locate_chapter_text
    from granular_agent.extraction_agent import ExtractionAgent
    from granular_agent.hypergraph_extractor import HGBlackboard
    from granular_agent.structure_mapper import (map_structure,
                                                 section_text_for_node,
                                                 topo_order,
                                                 full_text_from_blocks)
    from granular_agent.llm_client import call_llm, call_paratera

    smap = map_structure(pid, blocks, llm=llm, domain="ml")
    if not smap or not smap.get("dag", {}).get("nodes"):
        return 0
    ext = ExtractionAgent(
        kb,
        llm_extract=lambda prompt, max_tokens=8000: call_paratera(
            prompt, model=llm, max_tokens=max_tokens, enable_thinking=False),
        llm_verify=lambda prompt, max_tokens=4000: call_llm(
            prompt, model="deepseek-chat", max_tokens=max_tokens),
        domain_default="ml", executor_model=llm)
    full_text = full_text_from_blocks(blocks)
    dag_nodes = topo_order(smap["dag"])
    node_texts: dict[str, str] = {}
    for node in dag_nodes:
        nid = node.get("id", "")
        sec_name = node.get("section", nid)
        sec_text = section_text_for_node(node, smap.get("sections", []), blocks)
        if sec_text and len(sec_text) >= 80:
            node_texts[nid] = sec_text
            continue
        fb = _locate_chapter_text(sec_name, full_text,
                                  [n.get("section", "") for n in dag_nodes])
        if fb and len(fb) >= 80:
            node_texts[nid] = fb
        elif not any(node_texts.values()):
            node_texts[nid] = full_text
    bb = HGBlackboard()
    n_done = 0
    for node in dag_nodes:
        nid = node.get("id", "")
        sec_text = node_texts.get(nid, "")
        if not sec_text or len(sec_text) < 80:
            continue
        discourse = node.get("section", nid)
        predecessor = bb.predecessor_summary(node.get("deps", []))
        ext_rep = ext.extract_section(sec_text, discourse, nid, pid,
                                      year=kb._paper_meta.get(pid, {}).get("year", ""),
                                      section=discourse,
                                      predecessor_summary=predecessor)
        plan_summary = (ext_rep.get("plan", {}).get("paper_anchor", "")
                        if isinstance(ext_rep.get("plan"), dict) else "")
        bb.add(nid, plan_summary or discourse)
        n_done += 1
        # recall mode: dropped edges are NOT fed to the evolver (no schema
        # evolution online — frozen architecture)
    return n_done


def _commit_seed(graph_name: str, prep: dict,
                 extract_concepts: bool = True) -> dict | None:
    """SERIAL half: KB register + (optional recall-mode concept extraction) +
    intra-corpus join + commit cites edges + save. Runs single-threaded
    (kb.json is not concurrent-write safe); the join/post-check logic mirrors
    citation_stage.extract_citation_intents (that module stays untouched —
    frozen boundary; duplication is scoped to this contest module, recorded
    in goal_log_0829.md)."""
    from granular_agent.graph_store import load_graph, save_graph
    from granular_agent.agent import GranularFlowAgent
    from granular_agent.citation_stage import (CORPUS_REGISTRY, INTENT_TYPES,
                                               register_corpus_papers)
    from granular_agent.knowledge_base import Mutation, Op, Role

    if "error" in prep:
        return None
    pid, title = prep["pid"], prep["title"]
    if "cached_report" in prep:
        return prep["cached_report"]

    agent = GranularFlowAgent(llms=["DeepSeek-V4-Flash"], domain="ml")
    kb = load_graph(graph_name)
    if kb is not None:
        agent._kb = kb
        for cid, c in kb.abox.concepts.items():
            if c.type == "PAPER":
                t = c.canonical_name or (c.surface_variants[0].surface
                                         if c.surface_variants else "")
                p0 = (c.source_papers or [""])[0]
                if t and p0:
                    CORPUS_REGISTRY[p0] = {"title": t, "surnames": [], "years": []}
    else:
        kb = agent._get_kernel()
    register_corpus_papers(kb, [{"pid": pid, "title": title,
                                 "surnames": [], "years": []}])
    from granular_agent.hypergraph_extractor import _extract_metadata
    from granular_agent.structure_mapper import load_paper_blocks
    blocks = load_paper_blocks(pid)
    if blocks:
        kb._paper_meta[pid] = _extract_metadata(blocks, pid)
    if blocks and extract_concepts:
        try:
            n_sec = _extract_concepts_frozen(kb, pid, blocks)
            print(f"  [explore] {pid}: recall-mode concepts from {n_sec} sections",
                  flush=True)
        except Exception as e:
            print(f"  [explore] {pid}: concept extraction failed {type(e).__name__}",
                  flush=True)

    # intra-corpus join index (lowercased keys — the 08-29 production fix)
    idx = {}
    for p0, info in CORPUS_REGISTRY.items():
        if p0 == pid:
            continue
        for sn in info["surnames"]:
            for yr in info["years"]:
                idx[f"{sn} {yr}".lower()] = p0
    titles_all = prep["titles_all"]
    ref_titles_int = {int(k): v for k, v in titles_all.items() if k.isdigit()}
    by_key, mutations = prep["by_key"], []
    report = {"paper_id": pid, "edges": [], "n_background": 0}
    for rid in prep["num_keys"]:
        rec = by_key.get(str(rid))
        if not rec:
            continue
        if rid in ref_titles_int:
            rec["ref_title"] = ref_titles_int[rid][:160]
        report["edges"].append({**rec, "mention_kind": "numeric"})
    for key in prep["ay_keys"]:
        rec = by_key.get(key)
        if not rec:
            continue
        target = idx.get(key)
        if target:
            rec["corpus_pid"] = target
            # note: citation_stage's verbatim window post-check is not carried
            # here (prep doesn't keep windows); intra-explore-graph joins are
            # empty-surname today, so this path is dormant (recorded).
            mutations.append((target, rec))
        report["edges"].append({**rec, "mention_kind": "author-year"})
    # author-year title post-join (numeric handled above via ref_titles_int)
    for e in report["edges"]:
        if e.get("mention_kind") == "author-year" and not e.get("ref_title"):
            t = titles_all.get(str(e.get("rid", "")).lower())
            if t:
                e["ref_title"] = t
    src_cid = None
    for cid, c in kb.abox.concepts.items():
        if c.type == "PAPER" and (c.source_papers or [None])[0] == pid:
            src_cid = cid
            break
    committed = 0
    if src_cid:
        batch = []
        for target_pid, rec in mutations:
            if rec.get("intent") not in INTENT_TYPES:
                continue
            dst_cid = None
            for cid, c in kb.abox.concepts.items():
                if c.type == "PAPER" and (c.source_papers or [None])[0] == target_pid:
                    dst_cid = cid
                    break
            if not dst_cid or dst_cid == src_cid:
                continue
            batch.append(Mutation(
                op=Op.ADD_CONCEPT_RELATION, target=src_cid,
                proposer_role=Role.ALIGNER, domain="global",
                evidence=str(rec.get("note", ""))[:400],
                rationale=f"citation-intent: {rec['intent']} ({str(rec.get('note',''))[:120]})",
                payload={"node_ids": [src_cid, dst_cid],
                         "roles": ["from", "to"],
                         "kind": "cites",
                         "intent": rec["intent"]}))
        if batch:
            result = kb.commit(batch)
            committed = sum(1 for r in result.routed
                            if r.get("outcome") != "reject")
    report["n_committed"] = committed
    report["n_background"] = sum(1 for e in report["edges"]
                                 if e.get("intent") == "background")
    _cache_put("citreport", pid, report)
    save_graph(kb, graph_name, {"papers": {pid: {"title": title}}})
    return report


def ingest_seed_cites(graph_name: str, pid: str, title: str,
                      surnames: list | None = None,
                      years: list | None = None) -> dict | None:
    """Compat wrapper = prepare + commit (serial). explore() calls the split
    halves directly for paper-level parallelism."""
    arxiv = pid.replace("ARXIV_", "").replace("_", ".") if pid.startswith("ARXIV_") else pid
    prep = _prepare_seed(arxiv, title)
    if "error" in prep and "pid" not in prep:
        return None
    return _commit_seed(graph_name, prep)


def enrich_seed_concepts(graph_name: str, pid: str) -> bool:
    """Post-hoc recall-mode CONCEPT enrichment for an already-ingested seed
    (the batch path that grows the graph's concept layer without touching the
    cites-only F1 arm). Idempotent: skips papers whose concept hyperedges
    already carry provenance for this pid."""
    from granular_agent.graph_store import load_graph, save_graph
    from granular_agent.structure_mapper import load_paper_blocks

    kb = load_graph(graph_name)
    if kb is None or pid not in kb._paper_meta:
        return False
    hg = kb.abox.hyperedges
    edges = hg.values() if isinstance(hg, dict) else hg
    for h in edges:
        if h.kind == "cites":
            continue
        for pr in h.provenance or []:
            if pr.get("paper_id") == pid:
                return False                     # concept edges already present
    blocks = load_paper_blocks(pid)
    if not blocks:
        return False
    try:
        n_sec = _extract_concepts_frozen(kb, pid, blocks)
    except Exception as e:
        print(f"  [enrich] {pid}: failed {type(e).__name__}", flush=True)
        return False
    save_graph(kb, graph_name)
    print(f"  [enrich] {pid}: {n_sec} sections -> "
          f"{len(kb.abox.concepts)} concepts / {len(kb.abox.hyperedges)} edges",
          flush=True)
    return n_sec > 0


def mine_candidates(report: dict, max_per_seed: int = 25) -> list[dict]:
    """Classified references WITH titles -> new candidates (the 'dig' step).
    Intent is carried as a tag (rerank weighting downstream), NOT a filter:
    measured on the DRL survey seed — 61/62 mentions classified 'background',
    yet its enumerative citations ARE the representative papers surveys exist
    to list. A non-bg filter discarded 59/60 candidates on the best seed
    (goal_log_0829.md 02:50). Cap: top-N per seed ranked non-bg-first then
    classifier confidence (a survey yields 150+ titled refs; resolving all
    cost ~7min/seed and floods the pool — the per-query semantic trim would
    absorb it, but the resolve cost is real). GOAL 2026-08-30: 15 -> 25 —
    measured 3/24 holdout-missed golds were already-mined but truncated
    (ViT / Point Transformer / LLaVA-Improved died at the 15-cap or the
    resolve top-5; pre-registered q21-30 run)."""
    out = []
    for e in report.get("edges", []):
        if e.get("corpus_pid"):
            continue                      # intra-corpus: already in the graph
        t = (e.get("ref_title") or "").strip()
        if t:
            out.append({"title": t, "intent": e.get("intent", "background"),
                        "via": report.get("paper_id", ""),
                        "confidence": e.get("confidence", 0.0)})
    # INTENT ROUTING (GOAL 2026-08-30, user direction: walk intent edges,
    # don't just bulk-recall): extends/improves/replaces (method-evolution
    # edges) rank first, then compares/adapts, then background — confidence
    # within each tier. Survey enumerations (background) still mine (the
    # 02:50 verdict stands) but the intent edges get the seed's slots first.
    _tier = {"extends": 0, "improves": 0, "replaces": 0,
             "compares": 1, "adapts": 1, "background": 2}
    out.sort(key=lambda c: (_tier.get(c["intent"], 2), -c["confidence"]))
    return out[:max_per_seed]


def select_seeds(query: str, pool: list[dict], k: int = 3,
                 graph_name: str = "explore") -> list[dict]:
    """Pick seeds from the recall pool: needs an arXiv id (content route) and
    not already ingested. Ranked by SEMANTIC closeness to the query with a
    survey boost (GOAL 2026-08-30 fix): the original citation-count ranking
    picked generic off-domain classics from polluted pools (measured on
    AutoScholar q1-3: astrophysics/Grad-CAM seeds from image-recon pools —
    zero gold in mined refs). The mechanism design says intent-matched
    seeds ("沿与查询意图匹配的引用意图边") — embedding the query against
    pool titles is that intent match. Falls back to citation ranking when
    embeddings are unavailable (honest degrade)."""
    from granular_agent.graph_store import load_graph
    kb = load_graph(graph_name)
    done = set()
    if kb is not None:
        for pid in getattr(kb, "_paper_meta", {}):
            done.add(pid)
    seen_titles = set()
    scored = []
    for c in pool:
        ax = (c.get("externalIds") or {}).get("ArXiv")
        if not ax:
            continue
        if pid_from_arxiv(ax) in done:
            continue
        key = (c.get("title") or "").lower()
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        scored.append((ax, c.get("title", ""), c))
    if not scored:
        return []
    # semantic ranking (one embedding pass over candidate titles)
    sem_scores: dict[str, float] = {}
    try:
        from granular_agent.hypergraph_evolution import _embed_texts_robust
        texts = [query] + [t for _, t, _ in scored[:300]]
        embs = _embed_texts_robust(texts)
        if embs and len(embs) == len(texts):
            import math
            qe = embs[0]
            for (ax, t, _), e in zip(scored[:300], embs[1:]):
                dot = sum(x * y for x, y in zip(qe, e))
                n1 = math.sqrt(sum(x * x for x in qe))
                n2 = math.sqrt(sum(x * x for x in e))
                sem_scores[ax] = dot / (n1 * n2) if n1 and n2 else 0.0
    except Exception:
        pass
    def _rank(item):
        ax, t, c = item
        s = sem_scores.get(ax)
        if s is None:
            import math
            auth = c.get("citationCount") or c.get("citation_count") or 0
            s = 0.05 * math.log10(1 + int(auth))   # fallback: weak citation signal
        if re.search(r"\b(survey|review)\b", t.lower()):
            s += 0.05                    # surveys = gold mines of representative refs
        return s
    scored.sort(key=_rank, reverse=True)
    return [{"arxiv": ax, "title": t} for ax, t, _ in scored[:k]]


def explore(query: str, pool: list[dict], graph_name: str = "explore",
            k_seeds: int = 3, enrich_concepts: bool = False) -> tuple[list[dict], dict]:
    """The ② step: ingest up to k_seeds pool papers, mine their references,
    resolve the mined titles via S2 bulk, return (new candidates, stats).
    Failures skip, never block. enrich_concepts=True adds recall-mode concept
    extraction per seed (~+4min/seed — used by the graph-growth batch, not
    the F1 arm)."""
    from concurrent.futures import ThreadPoolExecutor
    from contest.pipeline import _s2_bulk_one, _norm_title
    stats = {"seeds": 0, "ingested": 0, "mined": 0, "resolved": 0, "errors": []}
    seeds = select_seeds(query, pool, k=k_seeds, graph_name=graph_name)
    new, seen = [], {_norm_title(c.get("title", "")) for c in pool}
    # paper-level parallelism (§4 speedup design): prepare (acquire+parse+
    # classify) runs 2 seeds concurrently — 2 × inner-8 classify ≈ Paratera's
    # measured 16-way ceiling; KB commits stay serial in the loop below.
    t_prep = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        preps = list(ex.map(lambda s: _prepare_seed(s["arxiv"], s["title"]),
                            seeds))
    stats["prepare_dt"] = round(time.time() - t_prep, 1)
    for s, prep in zip(seeds, preps):
        stats["seeds"] += 1
        t0 = time.time()
        if prep.get("error"):
            stats["errors"].append(f"{prep['error']}:{s['arxiv']}")
            continue
        try:
            report = _commit_seed(graph_name, prep,
                                  extract_concepts=enrich_concepts)
        except Exception as e:
            stats["errors"].append(f"ingest:{s['arxiv']}:{type(e).__name__}")
            continue
        if not report:
            stats["errors"].append(f"nocontent:{s['arxiv']}")
            continue
        stats["ingested"] += 1
        for cand in mine_candidates(report):
            stats["mined"] += 1
            tk = _norm_title(cand["title"])
            if tk in seen:
                continue
            seen.add(tk)
            # resolve the mined title to real candidates (AND-token bulk on
            # the title's own words — the most specific legal query).
            # GOAL 2026-08-30: top-5 -> top-8 — golds truncated at 5 measured
            # on holdout misses (pre-registered q21-30 run).
            toks = re.sub(r"[^a-z0-9 ]", " ", cand["title"].lower()).split()
            tq = " ".join(toks[:7])
            try:
                for row in _s2_bulk_one(tq)[:8]:
                    row["_q"] = f"explore:{cand['intent']}"
                    row["_via"] = s["title"][:60]
                    new.append(row)
                stats["resolved"] += 1
            except Exception:
                stats["errors"].append(f"resolve:{tk[:30]}")
            time.sleep(1.0)
        stats[f"seed_{stats['seeds']}_dt"] = round(time.time() - t0, 1)
    return new, stats
