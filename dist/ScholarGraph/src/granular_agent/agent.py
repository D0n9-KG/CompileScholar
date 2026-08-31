"""GranularFlow-Bench Agent: self-evolving schema extraction agent.

Main entry point. Single pipeline (B+ rebuild, 2026-08-25): every write goes
through the KnowledgeBase via the four builder agents —
Extraction → Evolution → Alignment → Maintenance (see process_paper_via_kernel).

Old atom/hypergraph pipelines are physically archived under legacy/ (see
legacy/README.md); 主代码不 import legacy.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granular_agent.structure_mapper import load_paper_blocks, full_text_from_blocks, map_structure
from granular_agent.llm_client import call_llm
from granular_agent.hypergraph_schema import seed_meta_hypergraph, MetaHypergraph


def _locate_chapter_text(sec_name: str, full_text: str, all_headings: list) -> str:
    """Fallback when map_structure cut sections but missed a DAG node's section:
    locate the chapter heading in the full text + slice to the NEXT heading.
    Deterministic rule (string locate), not LLM. Returns "" if the heading
    isn't found (caller then falls back to full_text). Honest: this is a
    band-aid for map_structure's section切分 failure — the real fix is upstream
    in map_structure's LLM, but this keeps the paper from extracting 0 concepts."""
    if not sec_name or not full_text:
        return ""
    import re as _re
    # try to find the heading as a line/phrase in the text (case-insensitive)
    # common heading forms: "1. Introduction", "Introduction", "1 Introduction"
    pat = r"(?:^|\n)\s*(?:\d+\.?\s*)?" + _re.escape(sec_name) + r"\s*(?:\n|\.|$)"
    m = _re.search(pat, full_text, _re.IGNORECASE | _re.MULTILINE)
    if not m:
        # looser: just find the name anywhere
        m = _re.search(_re.escape(sec_name), full_text, _re.IGNORECASE)
    if not m:
        return ""
    start = m.start()
    # find the NEXT heading after this one (slice to it)
    end = len(full_text)
    for h in all_headings:
        if h == sec_name or not h:
            continue
        hm = _re.search(r"(?:^|\n)\s*(?:\d+\.?\s*)?" + _re.escape(h) + r"\s*(?:\n|\.|$)",
                        full_text[start + 1:], _re.IGNORECASE | _re.MULTILINE)
        if hm:
            end = min(end, start + 1 + hm.start())
    return full_text[start:end].strip()


class GranularFlowAgent:
    """Self-evolving schema extraction agent (single kernel pipeline).

    Every write routes through the KnowledgeBase (transactional: ledger /
    validate / two-phase commit) via the four builder agents. The T-box
    (meta_hg) and A-box (concept_graph) persist across papers so schema
    evolution is cross-paper.
    """

    def __init__(self,
                 llms: list[str] = None,
                 domain: str = "granular flow physics",
                 corpus_dir: str = ""):
        self.llms = llms or ["DeepSeek-V4-Flash"]
        # domain (NOT a paper-specific special case — a parameter callers pass;
        # gates domain-specific cleanup regexes so non-granular inputs aren't
        # overfit-cleaned). Default is granular-flow (the current pilot domain).
        self.domain = domain
        # corpus_dir: caller-provided md directory for load_paper_blocks fallback
        # (NOT guessed from paper_id format — that was overfit). Empty = use
        # MINERU_BASE only (no md fallback).
        self.corpus_dir = corpus_dir

        # State
        self.hg_results = []

        # T-box: cross-paper meta-hypergraph (schema). Shared across papers so
        # patterns added by earlier papers are available to later ones.
        self.meta_hg = seed_meta_hypergraph()
        # seed pattern ids at init — _save_kernel_bundle reports only patterns
        # added AFTER these as new (evolution-added). Callers that replace
        # meta_hg (e.g. injection probes) must reset this themselves.
        self._initial_seed_pats = set(self.meta_hg.patterns.keys())
        # A-box: cross-paper concept hypergraph (n-ary, accumulated across
        # papers with LLM semantic alignment) — see concept_graph.py.
        from granular_agent.concept_graph import ConceptGraph
        self.concept_graph = ConceptGraph()

    # ===================================================================
    # process_paper_via_kernel — the KB-backed (底座读写者) pipeline.
    # SINGLE pipeline (B+ rebuild 2026-08-25; legacy paths archived under
    # legacy/). Routes every write through the KnowledgeBase via the four
    # builder agents: ExtractionAgent (add_edge → KB validate) →
    # EvolutionAgent (evolver Mutation → KB commit, schema并进) →
    # AlignmentAgent (align_concept_merge → KB, 真合并) → MaintenanceAgent
    # (rich topology + utility prune → KB retire). The KB shares
    # self.meta_hg / self.concept_graph (same objects, transactional:
    # ledger / validate / replay).
    # ===================================================================
    def _get_kernel(self):
        """Lazy-build the KnowledgeBase wrapping self.meta_hg / self.concept_graph.
        Same objects (not copies) so the KB's commits mutate the agent's evolving
        schema/A-box directly. Caches on self._kb."""
        if getattr(self, "_kb", None) is None:
            from granular_agent.knowledge_base import KnowledgeBase
            self._kb = KnowledgeBase(
                tbox=self.meta_hg, abox=self.concept_graph,
                domain_ns={"global": True, "granular": True, "ml": True,
                           "molecular": True})
        return self._kb

    def _kernel_llm_extract(self, prompt, max_tokens=8000):
        # extractor LLM = self.llms[0] (default DeepSeek-V4-Flash via Paratera,
        # thinking disabled). (MA2 fix: removed a dead `if llm == "deepseek"`
        # branch that never triggered since llms holds model names like
        # "DeepSeek-V4-Flash", not the literal "deepseek".)
        from granular_agent.llm_client import call_paratera
        llm = self.llms[0] if self.llms else "DeepSeek-V4-Flash"
        return call_paratera(prompt, model=llm, max_tokens=max_tokens, enable_thinking=False)

    def _kernel_llm_verify(self, prompt, max_tokens=4000):
        # verifier ≠ extraction model (avoid self-endorsement). Use deepseek-chat
        # for verify even when extract uses V4-Flash.
        return call_llm(prompt, model="deepseek-chat", max_tokens=max_tokens)

    def process_paper_via_kernel(self, paper_id: str, arm: str = "full") -> dict:
        """KB-backed pipeline: run the four builder agents on each
        section of a paper, writing through KnowledgeBase.commit. Returns a
        report per section + aggregate. arm: full/add_only/no_intra_dag/frozen
        — frozen skips SCHEMA evolution only; A-box alignment runs in every
        arm (2026-08-28 fix — see the Stage 3 note below)."""
        from granular_agent.extraction_agent import ExtractionAgent
        from granular_agent.evolution_agent import EvolutionAgent, TriggerSource
        from granular_agent.alignment_agent import AlignmentAgent, AlignmentReport
        from granular_agent.maintenance_agent import MaintenanceAgent
        from granular_agent.structure_mapper import section_text_for_node

        kb = self._get_kernel()
        llm = self.llms[0] if self.llms else "DeepSeek-V4-Flash"
        blocks = load_paper_blocks(paper_id, corpus_dir=self.corpus_dir)
        if not blocks:
            return {"paper_id": paper_id, "error": "no_text", "n_nodes": 0, "n_hyperedges": 0}
        # 真漏 fix #6: extract paper-level metadata (title/authors/doi/year/venue)
        # into the KB so every hyperedge is可溯源 to the paper identity (edge ->
        # evidence offset -> paper_id -> title/authors/year). The legacy pipeline
        # did this via _extract_metadata; the kernel path dropped it, breaking
        # paper-level provenance. Delegates to the verified _extract_metadata内核.
        import re as _re
        _ym = _re.search(r'(\d{4})', paper_id)
        from granular_agent.hypergraph_extractor import _extract_metadata
        kb._paper_meta[paper_id] = _extract_metadata(blocks, paper_id)
        if _ym and not kb._paper_meta[paper_id].get("year"):
            kb._paper_meta[paper_id]["year"] = _ym.group(1)
        smap = map_structure(paper_id, blocks, llm=llm, domain=self.domain)
        if not smap or not smap.get("dag", {}).get("nodes"):
            return {"paper_id": paper_id, "error": "structure_map_failed"}

        ext = ExtractionAgent(kb, llm_extract=self._kernel_llm_extract,
                              llm_verify=self._kernel_llm_verify, domain_default=self.domain,
                              executor_model=llm)
        evo = EvolutionAgent(kb, llm=self._kernel_llm_verify, domain_default=self.domain)
        align = AlignmentAgent(kb, llm_define=self._kernel_llm_verify,
                               llm_judge=self._kernel_llm_verify, domain_default=self.domain)
        maint = MaintenanceAgent(kb, llm=None, domain_default=self.domain)

        pre_v = kb.version
        section_reports = []
        if arm not in ("full", "add_only", "no_intra_dag", "frozen"):
            raise ValueError(f"unknown arm {arm!r} — valid: full/add_only/no_intra_dag/frozen")
        deferred_failures: list[tuple[str, dict]] = []   # no_intra_dag batch mode
        # 真漏 fix #1+#2: process DAG in TOPOLOGICAL order (deps before dependents)
        # + pass predecessor_summary (a section sees its dependency sections'
        # summaries — cross-section context: Results can see Method's concepts).
        # Legacy did this via topo_order(dag) + HGBlackboard; the kernel path
        # iterated smap["dag"]["nodes"] in raw order with no predecessor context,
        # breaking multi-section协同. Delegates to the verified topo_order +
        # HGBlackboard内核 (surgical).
        from granular_agent.structure_mapper import topo_order
        from granular_agent.hypergraph_extractor import HGBlackboard
        bb = HGBlackboard()
        # Fallback for map_structure section切分失败: if the DAG has nodes whose
        # section names don't all map to a sliced section (map_structure's LLM cut
        # only 1 section but identified N chapter headings), section_text_for_node
        # returns "" for the missing ones -> whole chapters dropped (BIO: 7 DAG
        # nodes, 1 section -> 0 concepts). Fallback: synthesize per-node text by
        # locating the chapter HEADING in the full text + slicing to the next
        # heading (deterministic rule, not LLM). If even that fails, fall back to
        # the full text as ONE section (like the 1-section case) so the paper
        # still extracts (no silent 0-concept result).
        full_text = full_text_from_blocks(blocks)
        dag_nodes = topo_order(smap["dag"])
        # build a name->text map for every DAG node's section, filling missing ones
        node_texts: dict[str, str] = {}
        for node in dag_nodes:
            nid = node.get("id", "")
            sec_name = node.get("section", nid)
            sec_text = section_text_for_node(node, smap.get("sections", []), blocks)
            if sec_text and len(sec_text) >= 80:
                node_texts[nid] = sec_text
                continue
            # fallback A: locate the chapter heading in the full text, slice to next heading
            fb = _locate_chapter_text(sec_name, full_text, [n.get("section","") for n in dag_nodes])
            if fb and len(fb) >= 80:
                node_texts[nid] = fb
            else:
                # fallback B (last resort): full text (no chapter split) — only use
                # for the FIRST missing node so we don't extract the full text N times.
                if not any(node_texts.values()):
                    node_texts[nid] = full_text
        # iterate the resolved texts (skip nodes with no text resolved)
        for node in dag_nodes:
            nid = node.get("id", "")
            sec_text = node_texts.get(nid, "")
            if not sec_text or len(sec_text) < 80:
                continue
            discourse = node.get("section", nid)
            # predecessor context: this section's dependency sections' summaries
            # (deps resolved by topo_order first). Lets Results see Method's terms.
            predecessor = bb.predecessor_summary(node.get("deps", []))
            # Stage 1: extract (plan-execute-verify-fix-commit → KB)
            ext_rep = ext.extract_section(sec_text, discourse, nid, paper_id,
                                          year="", section=discourse,
                                          predecessor_summary=predecessor)
            # record this section's summary on the blackboard for dependents
            plan_summary = (ext_rep.get("plan", {}).get("paper_anchor", "")
                            if isinstance(ext_rep.get("plan"), dict) else "")
            if not plan_summary:
                plan_summary = discourse
            bb.add(nid, plan_summary)
            # Stage 2: evolve — feed this section's validation failures (edges the
            # verifier dropped or that didn't match a pattern) to the evolver.
            # arm semantics (D2 fix — the arms used to be no-ops):
            #   full/add_only: per-section in-loop drain (schema evolves DURING
            #     the paper; later sections see the evolved schema)
            #   no_intra_dag: proposals accumulate, drained ONCE at paper end
            #     (batch evolution — the in-loop timing axis's control arm)
            #   frozen: no evolution at all (static schema)
            if arm in ("full", "add_only"):
                # MA1 fix: feed the REAL dropped edges (from the extractor's
                # fixer) to the evolver as schema-gap triggers — NOT a synthetic
                # Hyperedge. The dropped_edges carry pattern_type + evidence +
                # node_ids/roles + reason (verbatim/type/role). A recurring gap
                # (same pattern_type dropped across >=2 sections) accumulates
                # cross_node and the conservative gate accepts it -> schema并进
                # actually fires. The previous合成 trigger (pattern_type=
                # "_unknown", node_ids=["a","b"], evidence=sec_text[:120]) was a
                # fake signal that cross_node gate correctly rejected (cross=1)
                # so schema evolution never happened in single-paper runs.
                # feed the REAL dropped_edge dicts (carry pattern_type + evidence
                # + node_ids/roles + node_surfaces + reason) to the evolver. The
                # evolver builds an instance from node_surfaces so evolution_probe
                # sees real surfaces (P0 audit fix: instance=None → 0 proposals).
                failing = list(ext_rep.get("dropped_edges", []))
                if failing:
                    evo.propose_validate_failures(failing, node_id=nid,
                                                  paper_id=paper_id, domain=self.domain)
                # SOFT SCHEMA ROUTING: committed _novel_type edges feed the
                # induction channel (recurrence-tracked promotion, gated by
                # governance). The kept-edge records don't carry the _novel_type
                # flag explicitly, so detect novelty by tbox membership.
                for ke in ext_rep.get("kept_edges", []):
                    pt = ke.get("pattern_type", "")
                    if pt and pt not in kb.tbox.patterns:
                        evo.propose_novel_type(
                            {"pattern_type": pt, "roles": ke.get("roles", []),
                             "surfaces": [], "evidence": ke.get("evidence", "")},
                            node_id=nid, paper_id=paper_id, domain=self.domain)
                evo_rep = evo.drain()
            elif arm == "no_intra_dag":
                deferred_failures.extend(
                    (nid, de) for de in ext_rep.get("dropped_edges", []))
                evo_rep = None
            else:
                evo_rep = None
            section_reports.append({"node_id": nid, "extract": ext_rep,
                                    "evolve": evo_rep.__dict__ if evo_rep else None})

        # no_intra_dag: single batch evolution at paper end (control arm for
        # the in-loop timing axis — schema does NOT evolve during the paper)
        if arm == "no_intra_dag" and deferred_failures:
            for nid, de in deferred_failures:
                evo.propose_validate_failures([de], node_id=nid,
                                              paper_id=paper_id, domain=self.domain)
            evo.drain()

        # Stage 3: align — Define + Canonicalize on the new concepts ingested.
        # Runs in EVERY arm including frozen (2026-08-28 fix): frozen skips
        # SCHEMA evolution (T-box writes), not A-box alignment — cross-paper
        # concept merges are the extractor's job, not evolution. Measured cost
        # of the old gate: frozen runs left the variant dictionary empty
        # (420/428 single-surface concepts), killing the online side of the
        # 'offline-evolve + online-frozen' architecture (query expansion,
        # graph vocabulary) — the A-box islanded per paper.
        try:
            align_rep = align.align_new(domain=self.domain)
        except Exception as e:
            align_rep = AlignmentReport()
            print(f"  [kernel] align failed: {e!r}", flush=True)

        # Stage 3.5: citation intent — paper-level 'cites' edges (extends/
        # improves/compares/...) into the SAME graph, committed through the
        # kernel contract (2026-08-28: was a bypass JSON writer; integrating it
        # is the 'one working memory' architecture requirement). Runs in every
        # arm incl. frozen (intent is extraction, not evolution). Needs corpus
        # papers registered as PAPER nodes first (caller does via
        # citation_stage.register_corpus_papers; without registration this
        # stage only classifies mentions, no intra-corpus edges).
        cit_rep = None
        try:
            from granular_agent.citation_stage import extract_citation_intents
            cit_rep = extract_citation_intents(
                kb, paper_id, full_text, self._kernel_llm_extract)
        except Exception as e:
            print(f"  [kernel] citation stage failed: {e!r}", flush=True)

        # 优化: build_abox_instance ONCE, share across rich_topology + active_repair
        # + T-box拓扑. Must be BEFORE all consumers.
        inst_abox = None
        try:
            inst_abox = maint._build_abox_instance(paper_id)
        except Exception as e:
            # GBK-safe print: repr(e) may contain unicode (−, μ...) that crashes
            # the Windows GBK console and kills the whole pipeline.
            print("  [kernel] build_abox_instance failed: " + repr(e).encode("ascii", "replace").decode(), flush=True)
        # rich topology (最稳卖点) — share inst_abox
        rt_edges = []
        try:
            rt_edges = maint.infer_rich_topology_for_abox(paper_id=paper_id, inst=inst_abox)
        except Exception as e:
            print("  [kernel] rich topology failed: " + repr(e).encode("ascii", "replace").decode(), flush=True)
        rt_by_kind = {}
        for e in rt_edges:
            k = e.get("kind", "")
            rt_by_kind[k] = rt_by_kind.get(k, 0) + 1
        # active repair detect (split/merge/rename) — share inst_abox.
        # add_only skips repair (evolve-only arm: patterns may accumulate
        # duplicates that full's repair would merge — that's the arm's point)
        active_repair = {"split": 0, "merge": 0, "rename": 0}
        if arm in ("full", "no_intra_dag") and inst_abox is not None:
            try:
                from granular_agent.hypergraph_evolution import (
                    detect_split_triggers, detect_merge_triggers, detect_rename_triggers)
                triggers = []
                for t in detect_split_triggers(kb.tbox, inst_abox, llm=llm):
                    triggers.append(TriggerSource(kind="self_split", domain=self.domain,
                        payload=t, paper_id=paper_id))
                for t in detect_merge_triggers(kb.tbox):
                    triggers.append(TriggerSource(kind="self_merge", domain=self.domain,
                        payload=t, paper_id=paper_id))
                for t in detect_rename_triggers(kb.tbox):
                    triggers.append(TriggerSource(kind="self_rename", domain=self.domain,
                        payload=t, paper_id=paper_id))
                for ts in triggers:
                    evo.propose_trigger(ts)
                if triggers:
                    rep = evo.drain()
                    active_repair = {"split": sum(1 for a in rep.accepted if a["op"] == "split"),
                                     "merge": sum(1 for a in rep.accepted if a["op"] == "merge"),
                                     "rename": sum(1 for a in rep.accepted if a["op"] == "rename")}
            except Exception as e:
                print(f"  [kernel] active repair detect failed: {e!r}", flush=True)
        # T-box富拓扑 + violations (复用inst_abox, 不重复_build)
        tbox_topo = {"dependencies": 0, "constraints": 0, "compositions": 0, "violations": 0}
        if inst_abox is not None:
            try:
                from granular_agent.hypergraph_evolution import (
                    infer_pattern_dependencies, infer_pattern_constraints,
                    infer_pattern_compositions)
                tbox_topo["dependencies"] = len(infer_pattern_dependencies(kb.tbox, inst_abox, paper_id=paper_id))
                tbox_topo["constraints"] = len(infer_pattern_constraints(kb.tbox, inst_abox, paper_id=paper_id))
                tbox_topo["compositions"] = len(infer_pattern_compositions(kb.tbox, inst_abox, paper_id=paper_id))
                tbox_topo["violations"] = len(kb.tbox.detect_constraint_violations(inst_abox, domain=self.domain))
            except Exception as e:
                print(f"  [kernel] T-box topology failed: {e!r}", flush=True)
        n_concepts = len(kb.abox.concepts)
        n_hyperedges = len(kb.abox.hyperedges)
        result = {
            "paper_id": paper_id,
            "pipeline": "kernel",
            "paper_meta": kb._paper_meta.get(paper_id, {}),   # 可溯源论文级
            "n_sections": len(section_reports),
            "n_concepts": n_concepts,
            "n_hyperedges": n_hyperedges,
            "version_before": pre_v,
            "version_after": kb.version,
            "section_reports": section_reports,
            "align": (align_rep.__dict__ if align_rep else None),
            "total_patterns_after": len(kb.tbox.patterns),
            "ledger_entries": len(kb.ledger),
            "rich_topology": {"total": len(rt_edges), "by_kind": rt_by_kind},
            "active_repair": active_repair,   # 真漏 fix #9: split/merge/rename detect
            "tbox_topology": tbox_topo,        # 真漏 fix #10/#11: T-box富拓扑+violations
            "citation_intents": ({"n_mentions": cit_rep["n_numeric"] + cit_rep["n_author_year"],
                                  "n_committed": cit_rep["n_committed"]}
                                 if cit_rep else None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.hg_results.append(result)
        print(f"  [kernel] {paper_id}: {len(section_reports)} sections | "
              f"{n_concepts} concepts / {n_hyperedges} hyperedges | "
              f"v{pre_v}->{kb.version} ({len(kb.tbox.patterns)} patterns) | "
              f"ledger={len(kb.ledger)}", flush=True)
        # Save full bundle — 超图 + schema + 事务日志 + dropped(带verdict) +
        # 富拓扑. 跑完自动存, 不靠临时脚本. 后面断点重续/工程化也用这些.
        # 这是知识超图抽取+schema演化项目的基本工程: 跑完超图和schema必须存.
        try:
            self._save_kernel_bundle(paper_id, result, kb)
        except Exception as e:
            print(f"  [kernel] bundle save failed: {e!r}", flush=True)
        return result

    def _save_kernel_bundle(self, paper_id: str, result: dict, kb) -> str:
        """Save the full extraction + evolution result to disk so it can be
        inspected/audited/resumed without re-running. Path:
        .research_tmp/runs/kernel_v2/{paper_id}/.
        Stores: kept_edges / dropped_edges(with verifier verdict) / new_patterns /
        ledger / concept_graph(A-box) / meta_snapshot(T-box evolved schema) /
        rich_topology / result(summary+fix_stats+paper_meta+tbox_topology)."""
        import json as _json
        out_dir = os.path.join(".research_tmp", "runs", "kernel_v2", paper_id)
        os.makedirs(out_dir, exist_ok=True)
        kept = [ke for s in result["section_reports"] for ke in s.get("extract", {}).get("kept_edges", [])]
        dropped = [de for s in result["section_reports"] for de in s.get("extract", {}).get("dropped_edges", [])]
        seed_pats = getattr(self, "_initial_seed_pats", set())
        new_pats = {pid: {"desc": p.description, "boundary": p.semantic_boundary,
                          "family": p.family, "role_slots": [dict(s) for s in p.role_slots]}
                    for pid, p in kb.tbox.patterns.items() if pid not in seed_pats}
        _w = lambda name, data: open(os.path.join(out_dir, name), "w", encoding="utf-8").write(
            _json.dumps(data, ensure_ascii=False, indent=2, default=str))
        _w("kept_edges.json", kept)
        _w("dropped_edges.json", dropped)
        _w("new_patterns.json", new_pats)
        _w("ledger.json", kb.ledger)
        _w("concept_graph.json", kb.abox.to_dict())
        _w("meta_snapshot.json", kb.tbox.to_dict())
        _w("rich_topology.json", result.get("rich_topology", {}))
        _w("result.json", {k: v for k, v in result.items() if k != "section_reports"})
        print(f"  [kernel] bundle saved -> {out_dir}/", flush=True)
        return out_dir

    def process_batch_via_kernel(self, paper_ids: list[str]) -> list[dict]:
        """KB-backed batch: run process_paper_via_kernel per paper (meta+A-box
        persist via the shared KB), then a final maintenance prune (utility)
        + rich-topology snapshot. Returns per-paper results."""
        results = [self.process_paper_via_kernel(pid) for pid in paper_ids]
        from granular_agent.maintenance_agent import MaintenanceAgent
        kb = self._get_kernel()
        maint = MaintenanceAgent(kb, llm=None, domain_default=self.domain)
        # final utility prune: retire patterns with zero A-box usage (freq < 1).
        # honest: frequency-only (citation/evolution contribution reserved).
        prune = maint.prune_by_utility(domain=self.domain, min_frequency=1)
        print(f"  [kernel-batch] final prune: retired {prune.n_retired} unused patterns", flush=True)
        return results

    def save_meta(self, path: str):
        """Save JUST the evolved meta-hypergraph (full-field, round-trippable)
        to `path`. Use for incremental-evolution checkpoints across sessions."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.meta_hg.to_dict(), f, ensure_ascii=False, indent=2)

    def load_meta(self, path: str) -> bool:
        """Load an evolved meta-hypergraph from a save_meta output, replacing
        the current seed meta. Closes the 'schema gone after the run' gap:
        an evolved schema can be restored and incrementally evolved further
        across sessions (the self-evolution asset is now durable). Returns
        True on success."""
        try:
            with open(path, encoding="utf-8") as f:
                self.meta_hg = MetaHypergraph.from_dict(json.load(f))
            print(f"  [hypergraph] loaded meta v{self.meta_hg.version} "
                  f"({len(self.meta_hg.patterns)} patterns) from {path}", flush=True)
            return True
        except Exception as e:
            print(f"  [hypergraph] load_meta failed ({e}); keeping seed", flush=True)
            return False
