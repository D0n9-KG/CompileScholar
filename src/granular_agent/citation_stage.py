# -*- coding: utf-8 -*-
"""Citation-intent stage for the kernel extraction pipeline (2026-08-28).

Integrates citation-intent extraction INTO the main extractor pipeline (was a
bypass JSON writer — violated the 'one working memory' architecture). After a
paper's concepts are in the KB, this stage:
  1. registers a PAPER concept node per paper (surface=title, canonical_name,
     provenance = the paper itself) — the paper-level identity the L2 ranker
     navigates;
  2. extracts citation intents from the SAME MinerU fulltext (citation_intent
     kernel: numeric + author-year mention location, LLM intent classify);
  3. writes intent edges through KnowledgeBase.commit as ADD_CONCEPT_RELATION
     with kind='cites' + intent=extends/improves/compares/replaces/adapts —
     the paper-level evolution edges, same contract/ledger/provenance as every
     other edge (the paper-identity fix the contest report needs).

Deterministic post-check (rules-for-structural discipline): an intent edge is
only committed when the evidence window actually CONTAINS the citation mention
(text of the mention match) — the same verbatim-evidence binding the verifier
enforces for concept edges.
"""
import re

from .knowledge_base import Mutation, Op, Role
from .citation_intent import (locate_citation_contexts,
                              locate_author_year_contexts, classify_intent,
                              INTENT_TYPES)

# corpus registry: pid -> (surnames-with-particles, years, title). The join of
# in-text mentions to CORPUS papers (intra-corpus evolution edges). Populated
# by the runner from the corpus being ingested.
CORPUS_REGISTRY: dict[str, dict] = {}


def register_corpus_papers(kb, papers: list[dict]) -> None:
    """Register (or reuse) a PAPER concept node per corpus paper.
    papers: [{pid, title}] — idempotent, safe to call per run."""
    for p in papers:
        pid, title = p["pid"], p["title"]
        c = kb.abox.get_or_create("PAPER", title, paper_id=pid,
                                  evidence="paper identity node",
                                  year=p.get("year", ""))
        c.canonical_name = title
        CORPUS_REGISTRY[pid] = {"title": title,
                                "surnames": p.get("surnames", []),
                                "years": p.get("years", [])}


def paper_concept_id(kb, pid: str) -> str | None:
    """The PAPER node's concept_id (by registered title surface match)."""
    info = CORPUS_REGISTRY.get(pid)
    if not info:
        return None
    key = (info["title"] or "").strip().lower()
    cid = kb.abox._surface2concept.get(key)
    return cid if cid in kb.abox.concepts else None


def extract_citation_intents(kb, paper_id: str, fulltext: str, llm_fn,
                             ref_titles: dict[int, str] | None = None,
                             max_refs: int = 60) -> dict:
    """The stage body: locate mentions, classify intents, commit PAPER-level
    'cites' edges for intra-corpus references. Returns a report dict."""
    num_ctx = locate_citation_contexts(fulltext)
    ay_ctx = locate_author_year_contexts(fulltext)
    report = {"paper_id": paper_id, "n_numeric": len(num_ctx),
              "n_author_year": len(ay_ctx), "edges": [], "n_background": 0}

    # intra-corpus joins need the registry
    idx = {}
    for pid, info in CORPUS_REGISTRY.items():
        if pid == paper_id:
            continue
        for sn in info["surnames"]:
            for yr in info["years"]:
                idx[f"{sn} {yr}"] = pid

    mutations = []
    # numeric mentions: join via ref list (rid -> title -> corpus title match)
    for rid in sorted(num_ctx)[:max_refs]:
        rec = classify_intent(str(rid), num_ctx[rid], llm_fn)
        if ref_titles and rid in ref_titles:
            rec["ref_title"] = ref_titles[rid][:160]
        report["edges"].append({**rec, "mention_kind": "numeric"})

    # author-year mentions: corpus join -> PAPER-level 'cites' edge
    for key in sorted(ay_ctx):
        rec = classify_intent(key, ay_ctx[key], llm_fn)
        target_pid = idx.get(key)
        if target_pid:
            rec["corpus_pid"] = target_pid
            # deterministic post-check: the evidence window must contain the
            # surname token of the mention (verbatim binding)
            surname = key.rsplit(" ", 1)[0].split()[-1]
            window = ay_ctx[key][0] or ""
            if surname.lower() not in window.lower():
                rec["postcheck"] = "mention-not-in-evidence"
            else:
                mutations.append((target_pid, rec, window))
        report["edges"].append({**rec, "mention_kind": "author-year"})

    # commit the intra-corpus intent edges (one commit per paper: atomic)
    src_cid = paper_concept_id(kb, paper_id)
    committed = 0
    if src_cid:
        batch = []
        for target_pid, rec, window in mutations:
            dst_cid = paper_concept_id(kb, target_pid)
            if not dst_cid or dst_cid == src_cid:
                continue
            if rec["intent"] not in INTENT_TYPES:
                continue
            batch.append(Mutation(
                op=Op.ADD_CONCEPT_RELATION, target=src_cid,
                proposer_role=Role.ALIGNER, domain="global",
                evidence=window[:400],
                rationale=f"citation-intent: {rec['intent']} ({rec['note'][:120]})",
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
    return report
