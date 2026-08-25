"""GranularFlow-Bench Agent: self-evolving schema extraction agent.

Main entry point. Orchestrates the full pipeline:
  Extract → Fuse → GapDiscovery → Validate → ExtendSchema → QAGenerate

Uses Pydantic AI for the agent framework (capabilities + hooks),
but the core flow is deterministic with event-triggered branches.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granular_agent.schema_manager import SchemaManager
from granular_agent.extractor import Extractor
from granular_agent.structure_mapper import load_paper_blocks, full_text_from_blocks, map_structure
from granular_agent.chained_extractor import extract_chained
from granular_agent import grounding as grounding_mod
from granular_agent.gap_discovery import active_gap_scan, validate_gap
from granular_agent.qa_generator import generate_qa
from granular_agent.llm_client import call_llm, parse_json_response
from granular_agent.hypergraph_schema import seed_meta_hypergraph, MetaHypergraph
from granular_agent.hypergraph_extractor import extract_hypergraph
from granular_agent.hypergraph_evolution import (
    EvolutionTrigger, run_split, run_merge, run_retire, run_rename,
    infer_pattern_dependencies, infer_pattern_constraints, infer_pattern_compositions,
    infer_rich_topology_direct, consolidate_instance)


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
    """Self-evolving schema extraction agent for granular flow literature.

    Architecture: single-agent + capabilities + hooks (not multi-agent, not autonomous).
    Deterministic flow with event-triggered schema evolution branches.
    """

    def __init__(self, worktree: str = "C:/Users/D0n9/Desktop/LogicKG-benchmark",
                 llms: list[str] = None,
                 self_evolution_enabled: bool = True,
                 domain: str = "granular flow physics",
                 corpus_dir: str = ""):
        self.worktree = worktree
        self.schema_manager = SchemaManager(worktree)
        self.llms = llms or ["DeepSeek-V4-Flash"]
        self.extractor = Extractor(self.schema_manager, llms=self.llms)
        self.self_evolution_enabled = self_evolution_enabled
        # domain (NOT a paper-specific special case — a parameter callers pass;
        # gates domain-specific cleanup regexes so non-granular inputs aren't
        # overfit-cleaned). Default is granular-flow (the current pilot domain).
        self.domain = domain
        # corpus_dir: caller-provided md directory for load_paper_blocks fallback
        # (NOT guessed from paper_id format — that was overfit). Empty = use
        # MINERU_BASE only (no md fallback).
        self.corpus_dir = corpus_dir

        # Hook registry (event-driven triggers)
        self.hooks = {
            "on_extraction_complete": [],
            "on_gap_found": [],
            "on_batch_complete": [],
            "on_schema_extended": [],
            "on_qa_generated": [],
        }

        # State
        self.extraction_results = []
        self.schema_evolution_log = []
        self.qa_pairs = []

        # Hypergraph (deep self-evolution) path state. Shared across papers so
        # the meta-hypergraph + trigger persist (cross-paper evolution). This
        # path is parallel to the old atom pipeline; see process_paper_hypergraph.
        self.meta_hg = seed_meta_hypergraph()
        self.hg_trigger = EvolutionTrigger()
        self.hg_results = []
        # cross-paper instance corpus (Path C stage-3): accumulates each
        # paper's InstanceHypergraph + merges nodes by surface so cross-paper
        # downstream (QA/retrieval/conflict) works — the shared meta alone
        # was only a schema bridge, this is the instance bridge.
        # A-box: cross-paper concept hypergraph (n-ary, accumulated across
        # papers with LLM semantic alignment). Replaces InstanceCorpus
        # surface-only merge — see concept_graph.py + DESIGN_full.md.
        from granular_agent.concept_graph import ConceptGraph
        self.concept_graph = ConceptGraph()
        # per-paper instances (kept for save)
        self.hg_instances: dict[str, Any] = {}

    def register_hook(self, event: str, handler):
        """Register an event hook."""
        if event in self.hooks:
            self.hooks[event].append(handler)

    def _fire_hooks(self, event: str, data: dict):
        """Fire all hooks for an event."""
        for handler in self.hooks.get(event, []):
            handler(data)

    def _extract_adaptive(self, paper_id: str, intra_dag_evolution: bool = False) -> dict:
        """Three-phase extraction: structure map → chained DAG → grounding.

        Replaces the truncating Extractor.extract() for full-text coverage.
        Falls back to the old extractor only if structure mapping fails.
        When intra_dag_evolution=True, schema evolves DURING Phase 1 (v2 design).
        """
        blocks = load_paper_blocks(paper_id, corpus_dir=self.corpus_dir)
        if not blocks:
            return {"atoms": [], "gaps": [], "error": "no_text", "n_calls": 0}

        llm = self.llms[0] if self.llms else "DeepSeek-V4-Flash"

        # Phase 0: structure mapping (1 call, full text in context)
        smap = map_structure(paper_id, blocks, llm=llm, domain=self.domain)
        if not smap or not smap.get("dag", {}).get("nodes"):
            # Fallback to old truncated extractor if structure mapping fails
            print(f"    [fallback] structure mapping failed → old extractor", flush=True)
            return self.extractor.extract(paper_id)

        full_text = full_text_from_blocks(blocks)

        # Phase 1: chained extraction over DAG (4-8 calls)
        ext = extract_chained(smap, blocks, self.schema_manager, llm=llm,
                             intra_dag_evolution=intra_dag_evolution,
                             full_text=full_text, paper_id=paper_id)
        atoms = ext["atoms"]
        n_calls = ext["n_calls"]
        intra_evolutions = ext.get("schema_evolutions", [])
        intra_gap_count = len(ext.get("gap_records", []))
        if intra_dag_evolution:
            print(f"    Phase1 [intra-DAG]: {intra_gap_count} gaps detected across nodes, "
                  f"{len(intra_evolutions)} extensions accepted", flush=True)
        print(f"    Phase1: {len(atoms)} atoms in {n_calls} calls ({ext['fissions']} fissions)", flush=True)

        # Phase 2: grounding + rebind + lookup (deterministic + 0-1 LLM call)
        atoms = grounding_mod.attach_discourse_roles(atoms, smap)
        atoms = grounding_mod.ground_atoms(atoms, full_text)
        rebind_candidates = grounding_mod.find_rebind_candidates(atoms)
        pre = grounding_mod.summary(atoms)
        print(f"    Phase2-pre: in_text {pre['n_in_text']}/{pre['n_atoms']} "
              f"({pre['n_in_text']/pre['n_atoms']:.3f} compliance) | "
              f"supported {pre['n_supported']}/{pre['n_atoms']} "
              f"({pre['n_supported']/pre['n_atoms']:.3f} REAL support)", flush=True)
        lookup_needed = pre["n_atoms"] - pre["n_grounded"] > 0 or pre["n_low_conf"] > 0
        atoms = grounding_mod.lookup(atoms, smap, blocks, llm=llm)
        if lookup_needed:
            n_calls += 1  # one lookup call ran
        atoms = grounding_mod.filter_grounded(atoms)
        post = grounding_mod.summary(atoms)
        print(f"    Phase2: grounded {pre['n_grounded']}/{pre['n_atoms']} → kept {post['n_atoms']} "
              f"({len(rebind_candidates)} rebind candidates)", flush=True)

        # Detect schema gaps on the grounded atoms (reuse extractor's logic)
        gaps = self.extractor._detect_gaps(atoms)
        return {
            "atoms": atoms,
            "gaps": gaps,
            "n_calls": n_calls + 1,  # +1 for Phase 0
            "rebind_candidates": rebind_candidates,
            "grounding": {"pre": pre, "post": post},
            "intra_evolutions": intra_evolutions,
            "intra_gap_count": intra_gap_count,
        }

    def process_paper(self, paper_id: str, intra_dag_evolution: bool = False) -> dict:
        """Process a single paper through the full pipeline.

        Returns: {paper_id, atoms, gaps, qa_pairs, schema_changes, schema_version}
        """
        print(f"  Processing {paper_id} (schema v{self.schema_manager.current_version})...", flush=True)

        # Step 1: Adaptive three-phase extraction (full text, no truncation)
        result = self._extract_adaptive(paper_id, intra_dag_evolution=intra_dag_evolution)
        atoms = result.get("atoms", [])
        gaps = result.get("gaps", [])
        n_calls = result.get("n_calls", 0)
        print(f"    Extracted {len(atoms)} atoms, {len(gaps)} gaps detected ({n_calls} calls)", flush=True)

        # Hook: on_extraction_complete
        self._fire_hooks("on_extraction_complete", {
            "paper_id": paper_id, "atoms": atoms, "gaps": gaps
        })

        # Step 2: Schema evolution
        # Intra-DAG evolution (arm C) already ran inside _extract_adaptive —
        # merge those and skip the post-hoc loop. Post-hoc evolution (arm B)
        # runs here when self_evolution_enabled and not intra_dag.
        schema_changes = list(result.get("intra_evolutions", []))
        if self.self_evolution_enabled and not intra_dag_evolution and gaps:
            for gap in gaps:
                # Hook: on_gap_found
                self._fire_hooks("on_gap_found", {"paper_id": paper_id, "gap": gap})

                # Step 3: Validate (evidence-linked)
                validated = validate_gap(gap, paper_id, self.schema_manager)
                if not validated or not validated.get("valid"):
                    continue

                # Step 4: Extend schema
                gap_type = gap.get("type", "")
                value = gap.get("value", "")
                evidence = validated.get("evidence", "")

                new_version = None
                if "entity_type" in gap_type:
                    new_version = self.schema_manager.extend_entity_type(value, evidence, paper_id)
                elif "subtype" in gap_type:
                    new_version = self.schema_manager.extend_contribution_subtype(value, evidence, paper_id)
                elif "relation" in gap_type:
                    new_version = self.schema_manager.extend_relation_type(value, evidence, paper_id)

                if new_version:
                    change = {
                        "version": new_version,
                        "gap_type": gap_type,
                        "value": value,
                        "evidence": evidence,
                        "paper_id": paper_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    schema_changes.append(change)
                    self.schema_evolution_log.append(change)

                    # Hook: on_schema_extended
                    self._fire_hooks("on_schema_extended", change)
                    print(f"    Schema evolved: v{new_version} +{value} ({gap_type})", flush=True)

        # Step 5: QA generation
        qa_pairs = generate_qa(paper_id, atoms)
        self.qa_pairs.extend(qa_pairs)

        # Hook: on_qa_generated
        self._fire_hooks("on_qa_generated", {"paper_id": paper_id, "qa_pairs": qa_pairs})

        result = {
            "paper_id": paper_id,
            "atoms": atoms,
            "n_atoms": len(atoms),
            "n_calls": n_calls,
            "gaps": gaps,
            "schema_changes": schema_changes,
            "schema_version": self.schema_manager.current_version,
            "qa_pairs": qa_pairs,
            "intra_gap_count": result.get("intra_gap_count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.extraction_results.append(result)
        return result

    def process_batch(self, paper_ids: list[str]) -> list[dict]:
        """Process a batch of papers with active gap discovery."""
        results = []
        for pid in paper_ids:
            result = self.process_paper(pid)
            results.append(result)

        # Active gap discovery: scan across all papers
        if self.self_evolution_enabled:
            print(f"  Active gap scan across {len(results)} papers...", flush=True)
            candidates = active_gap_scan(results, self.schema_manager, min_recurrence=3)
            print(f"    Found {len(candidates)} recurring gap candidates", flush=True)

            for candidate in candidates:
                # Hook: on_gap_found
                self._fire_hooks("on_gap_found", {"paper_id": "batch", "gap": candidate})

                # Validate
                validated = validate_gap(candidate, candidate.get("papers", ["unknown"])[0], self.schema_manager)
                if not validated or not validated.get("valid"):
                    continue

                # Extend schema
                gap_type = candidate.get("gap_type", "")
                value = candidate.get("value", "")
                evidence = validated.get("evidence", "")

                new_version = None
                if gap_type == "entity_type":
                    new_version = self.schema_manager.extend_entity_type(value, evidence, "batch")
                elif gap_type == "contribution_subtype":
                    new_version = self.schema_manager.extend_contribution_subtype(value, evidence, "batch")
                elif gap_type == "relation_type":
                    new_version = self.schema_manager.extend_relation_type(value, evidence, "batch")

                if new_version:
                    change = {
                        "version": new_version,
                        "gap_type": gap_type,
                        "value": value,
                        "evidence": evidence,
                        "paper_id": "batch",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self.schema_evolution_log.append(change)
                    self._fire_hooks("on_schema_extended", change)
                    print(f"    Schema evolved (batch): v{new_version} +{value} ({gap_type})", flush=True)

        # Hook: on_batch_complete
        self._fire_hooks("on_batch_complete", {"results": results})

        return results

    def process_paper_hypergraph(self, paper_id: str, arm: str = "full") -> dict:
        """Process one paper through the DEEP self-evolution hypergraph path.

        Uses the shared meta-hypergraph + trigger (cross-paper evolution):
        patterns added by earlier papers are available to this one, and
        structural-mismatch recurrence accumulates across papers. The
        meta-hypergraph is mutated in place.

        arm (ablation): 'full' (default, evolve+propagate+repair),
        'add_only' (evolve+propagate, skip repair), 'no_intra_dag'
        (evolve, no propagation, repair), 'frozen' (no evolve).
        Mirrors .research_tmp/corpus_driver.py so ablation arms run through
        the agent main flow, not a shadow.
        """
        llm = self.llms[0] if self.llms else "DeepSeek-V4-Flash"
        evolve = arm in ("full", "add_only", "no_intra_dag")
        propagate = arm in ("full", "add_only")
        blocks = load_paper_blocks(paper_id, corpus_dir=self.corpus_dir)
        if not blocks:
            return {"paper_id": paper_id, "error": "no_text", "n_nodes": 0, "n_hyperedges": 0}
        smap = map_structure(paper_id, blocks, llm=llm, domain=self.domain)
        if not smap or not smap.get("dag", {}).get("nodes"):
            print(f"  [hypergraph] structure mapping failed for {paper_id}", flush=True)
            return {"paper_id": paper_id, "error": "structure_map_failed"}
        pre_v = self.meta_hg.version
        pre_patterns = set(self.meta_hg.patterns_ids())
        res = extract_hypergraph(smap, blocks, self.meta_hg, llm=llm, paper_id=paper_id,
                                 trigger=self.hg_trigger, evolve=evolve,
                                 propagate_intra_dag=propagate,
                                 domain=self.domain)
        acc = [e for e in res["evolutions"] if not e.get("rejected")]
        rej = [e for e in res["evolutions"] if e.get("rejected")]
        # pattern-level repair: split over-wide patterns (deterministic trigger,
        # LLM names only), merge near-dup (DIAL-KG op, cited), retire orphans.
        # These mutate the shared meta_hg + re-attribute this paper's instance.
        # Only full + no_intra_dag run repair (add_only skips; frozen has no
        # evolution so nothing to repair) — mirrors corpus_driver arm logic.
        inst = res["instance"]
        pre_repair_v = self.meta_hg.version
        if arm in ("full", "no_intra_dag"):
            splits = run_split(self.meta_hg, inst, paper_id=paper_id, llm=llm)
            merges = run_merge(self.meta_hg, inst, paper_id=paper_id, llm=llm)
            retires = run_retire(self.meta_hg, inst, paper_id=paper_id)
            renames = run_rename(self.meta_hg, llm=llm)
        else:
            # add_only/frozen: skip repair (split/merge/retire/rename are
            # pattern-maintenance ops, frozen arm runs NO evolution ops).
            splits, merges, retires, renames = [], [], [], []
        deps = infer_pattern_dependencies(self.meta_hg, inst, paper_id=paper_id)
        cons = infer_pattern_constraints(self.meta_hg, inst, paper_id=paper_id)
        comp = infer_pattern_compositions(self.meta_hg, inst, paper_id=paper_id)
        violations = self.meta_hg.detect_constraint_violations(inst, domain=self.domain)
        cons2 = consolidate_instance(inst, domain=self.domain)
        rich_topo = infer_rich_topology_direct(inst, paper_id=paper_id)
        result = {
            "paper_id": paper_id,
            "n_nodes": res["n_nodes"],
            "n_hyperedges": res["n_hyperedges"],
            "validation_failures": res["validation_failures"],
            "failed_edges": res.get("failed_edges", []),  # full rejected-edge detail
            "evolutions_detail": res.get("evolutions", []),  # full proposal list w/ accept/reject+reason
            "n_calls": res["n_calls"] + 1,  # +1 structure map
            "n_acc": len(acc),
            "n_rej": len(rej),
            "consolidate": cons2,
            "cross_node": sorted(set(e.get("cross_node", 1) for e in acc)),
            "new_patterns": [e.get("type_id") for e in acc
                             if e.get("op") in ("add_pattern", "add_meta_node", "add_subclass", "split_meta_node")],
            "version_before": pre_v,
            "version_after": self.meta_hg.version,
            "total_patterns_after": len(self.meta_hg.patterns),
            "topology": {
                "dependencies": len(deps),
                "constraints": len(cons),
                "compositions": len(comp),
                "violations": len(violations),
                "dependency_edges": [{"dependent": d["dependent"], "depends_on": d["depends_on"],
                                     "via_node": d["via_node"]} for d in deps],
                "constraint_edges": [{"authority": c["authority"], "constrains": c["constrains"],
                                     "via_node": c["via_node"]} for c in cons],
                "composition_edges": [{"whole": c["whole"], "composes": c["composes"],
                                      "via_node": c["via_node"]} for c in comp],
                "violations_detail": violations,
                "rich_topology_direct": {
                    "total": len(rich_topo),
                    "by_kind": {k: sum(1 for e in rich_topo if e["kind"] == k) for k in
                                ["law_parameter", "method_parameter", "method_phenomenon",
                                 "method_regime", "composition", "definition", "nary", "evolution"]},
                    "edges": rich_topo[:30],
                },
            },
            "repair": {
                "splits": [{"pattern": s.get("pattern_id"),
                            "sub_patterns": s.get("sub_patterns"),
                            "method": s.get("method"),
                            "cluster_sizes": s.get("cluster_sizes")}
                           for s in splits if not s.get("skipped")],
                "merges": [{"merged": m.get("merged"), "into": m.get("into"),
                            "cosine": m.get("cosine")}
                           for m in merges if not m.get("skipped")],
                "retires": [r.get("pattern_id") for r in retires],
                "version_before_repair": pre_repair_v,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.hg_results.append(result)
        # accumulate into the A-box concept hypergraph (n-ary, with provenance).
        # Replaces InstanceCorpus surface-only merge. year parsed from paper_id
        # for P2 time-order emergence signals.
        import re as _re
        _ym = _re.search(r'(\d{4})', paper_id)
        self.hg_instances[paper_id] = inst  # keep for save (InstanceCorpus removed)
        self.concept_graph.ingest_instance(inst, year=_ym.group(1) if _ym else "",
                                          domain=self.domain)
        print(f"  [hypergraph] {paper_id}: {res['n_nodes']} nodes / {res['n_hyperedges']} he / "
              f"{len(acc)} acc / {len(rej)} rej | cross_node={result['cross_node']} | "
              f"v{pre_v}->{self.meta_hg.version} ({len(self.meta_hg.patterns)} patterns) | "
              f"split={len([s for s in splits if not s.get('skipped')])} "
              f"merge={len([m for m in merges if not m.get('skipped')])} "
              f"retire={len(retires)} | topo: dep={len(deps)} cons={len(cons)} comp={len(comp)} viol={len(violations)}", flush=True)
        return result

    def process_batch_hypergraph(self, paper_ids: list[str]) -> list[dict]:
        """Run the hypergraph path over a batch. Meta + trigger persist across
        papers (cross-paper deep evolution). After all papers ingested into
        the A-box, run one LLM semantic alignment pass (merge near-synonym
        concepts surface-string missed). Returns per-paper results."""
        results = [self.process_paper_hypergraph(pid) for pid in paper_ids]
        # one LLM alignment pass over the accumulated concept graph
        try:
            llm = self.llms[0] if self.llms else "DeepSeek-V4-Flash"
            from granular_agent.llm_client import call_paratera
            n_merged = self.concept_graph.align_concepts(
                lambda p: call_paratera(p, model="GLM-5-Turbo", max_tokens=3000, temperature=0.0))
            print(f"  [concept-graph] LLM alignment: merged {n_merged} near-synonym concepts", flush=True)
        except Exception as e:
            print(f"  [concept-graph] alignment failed: {e!r}", flush=True)
        return results

    # ===================================================================
    # Step 7: process_paper_via_kernel — the KB-backed (底座读写者) pipeline.
    # NEW path (parallel to process_paper_hypergraph above). Routes every write
    # through the KnowledgeBase (Step 1) via the four builder agents (Step 2-5):
    # ExtractionAgent (add_edge → KB validate) → EvolutionAgent (evolver Mutation
    # → KB commit, schema并进) → AlignmentAgent (align_concept_merge → KB,真合并)
    # → MaintenanceAgent (rich topology + utility prune → KB retire). The KB
    # shares self.meta_hg / self.concept_graph (same objects) so this path and
    # the legacy path operate on the same evolving schema/A-box — but THIS path
    # is transactional (ledger / validate / replay), the legacy path is not.
    # The legacy process_paper_hypergraph is KEPT for ablation对照 + eval
    # continuity (surgical: don't break the running pipeline). (DECISION: 重组
    # 为底座读写者, not a second parallel pipeline — same KB objects, transactional.)
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
        """KB-backed pipeline (Step 7): run the four builder agents on each
        section of a paper, writing through KnowledgeBase.commit. Returns a
        report per section + aggregate. arm mirrors process_paper_hypergraph
        (full/add_only/no_intra_dag/frozen) — frozen skips evolution+align."""
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
                              llm_verify=self._kernel_llm_verify, domain_default=self.domain)
        evo = EvolutionAgent(kb, llm=self._kernel_llm_verify, domain_default=self.domain)
        align = AlignmentAgent(kb, llm_define=self._kernel_llm_verify,
                               llm_judge=self._kernel_llm_verify, domain_default=self.domain)
        maint = MaintenanceAgent(kb, llm=None, domain_default=self.domain)

        pre_v = kb.version
        section_reports = []
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
            # The extractor already dropped bad edges at commit (kernel validate),
            # so here we surface the verifier's dropped + reextract edges as
            # potential schema gaps (cross-node recurrence accumulates across
            # sections — a real gap recurs at >1 node and gets accepted).
            if arm != "frozen":
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
                evo_rep = evo.drain()
            else:
                evo_rep = None
            section_reports.append({"node_id": nid, "extract": ext_rep,
                                    "evolve": evo_rep.__dict__ if evo_rep else None})

        # Stage 3: align — Define + Canonicalize on the new concepts ingested
        if arm != "frozen":
            try:
                align_rep = align.align_new(domain=self.domain)
            except Exception as e:
                align_rep = AlignmentReport()
                print(f"  [kernel] align failed: {e!r}", flush=True)
        else:
            align_rep = None

        # Stage 4: maintenance — rich topology (the most-stable selling point,
        # memory:富拓扑直读 win最稳). P0 audit fix: the kernel path never called
        # infer_rich_topology_direct, so富拓扑 kinds were LOST. Run it on the
        # A-box here (per-paper) via MaintenanceAgent.infer_rich_topology_for_abox
        # (adapts ConceptGraph→InstanceHypergraph, delegates to verified内核).
        # Results recorded in report; the A-box hyperedges keep their raw
        # pattern_type (for prune_by_utility) and the富拓扑 view is on-demand
        # via snapshot_rich_topology.
        rt_edges = []
        try:
            rt_edges = maint.infer_rich_topology_for_abox(paper_id=paper_id)
        except Exception as e:
            print(f"  [kernel] rich topology failed: {e!r}", flush=True)
        rt_by_kind = {}
        for e in rt_edges:
            k = e.get("kind", "")
            rt_by_kind[k] = rt_by_kind.get(k, 0) + 1
        # 真漏 fix #9: active repair — detect split/merge/rename triggers on the
        # current A-box (as a synthetic InstanceHypergraph) + feed them to the
        # evolver's self_* path. Legacy did this via run_split/run_merge/run_rename
        # proactively; the kernel path only reacted to validate-failures (missed
        # pattern-over-wide detection). Delegates to detect_*_triggers内核.
        active_repair = {"split": 0, "merge": 0, "rename": 0}
        if arm != "frozen":
            try:
                inst_for_repair = maint._build_abox_instance(paper_id)
                from granular_agent.hypergraph_evolution import (
                    detect_split_triggers, detect_merge_triggers, detect_rename_triggers)
                triggers = []
                for t in detect_split_triggers(kb.tbox, inst_for_repair, llm=llm):
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
        # 真漏 fix #10/#11: schema-layer富拓扑 (T-box dependencies/constraints/
        # compositions) + constraint-violation detection. Legacy computed these
        # per-paper; the kernel path dropped them (memory: a DIAL-KG-can't-do
        # differentiator — schema constraint violations detectable). Delegates to
        # the verified infer_*内核. Used for the 'rich_topology' T-box section +
        # a schema-quality signal (violations = undefined params referenced).
        tbox_topo = {"dependencies": 0, "constraints": 0, "compositions": 0, "violations": 0}
        try:
            inst_t = maint._build_abox_instance(paper_id)
            from granular_agent.hypergraph_evolution import (
                infer_pattern_dependencies, infer_pattern_constraints,
                infer_pattern_compositions)
            tbox_topo["dependencies"] = len(infer_pattern_dependencies(kb.tbox, inst_t, paper_id=paper_id))
            tbox_topo["constraints"] = len(infer_pattern_constraints(kb.tbox, inst_t, paper_id=paper_id))
            tbox_topo["compositions"] = len(infer_pattern_compositions(kb.tbox, inst_t, paper_id=paper_id))
            tbox_topo["violations"] = len(kb.tbox.detect_constraint_violations(inst_t, domain=self.domain))
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.hg_results.append(result)
        print(f"  [kernel] {paper_id}: {len(section_reports)} sections | "
              f"{n_concepts} concepts / {n_hyperedges} hyperedges | "
              f"v{pre_v}->{kb.version} ({len(kb.tbox.patterns)} patterns) | "
              f"ledger={len(kb.ledger)}", flush=True)
        return result

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

    def save_hypergraph_results(self, output_dir: str):
        """Save hypergraph instances + evolved meta-hypergraph (FULL field,
        round-trippable) + the cross-paper instance corpus."""
        os.makedirs(output_dir, exist_ok=True)
        # full-field meta (replaces the lossy summary that dropped
        # is_abstract/family/deprecated/family_roots — load_meta restores all).
        with open(os.path.join(output_dir, "meta_hypergraph.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta_hg.to_dict(), f, ensure_ascii=False, indent=2)
        # per-paper instances (each a full InstanceHypergraph.to_dict)
        for pid, inst in self.hg_instances.items():
            with open(os.path.join(output_dir, f"instance_{pid}.json"), "w", encoding="utf-8") as f:
                json.dump(inst.to_dict(), f, ensure_ascii=False, indent=2)
        # A-box: concept hypergraph (n-ary, cross-paper aligned)
        with open(os.path.join(output_dir, "concept_graph.json"), "w", encoding="utf-8") as f:
            json.dump(self.concept_graph.to_dict(), f, ensure_ascii=False, indent=2)
        corpus_summary = {
            "n_papers": len(self.hg_instances),
            "n_concepts": len(self.concept_graph.concepts),
            "n_hyperedges": len(self.concept_graph.hyperedges),
            "per_paper": self.hg_results,
        }
        with open(os.path.join(output_dir, "corpus_index.json"), "w", encoding="utf-8") as f:
            json.dump(corpus_summary, f, ensure_ascii=False, indent=2)
        print(f"  [hypergraph] saved meta (full-field) + {len(self.hg_instances)} instances "
              f"+ concept-graph ({len(self.concept_graph.concepts)} concepts, "
              f"{len(self.concept_graph.hyperedges)} hyperedges) to {output_dir}", flush=True)

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

    def get_schema_evolution_summary(self) -> dict:
        """Get summary of schema evolution."""
        changelog = self.schema_manager.get_changelog()
        return {
            "initial_version": "4.0",
            "current_version": self.schema_manager.current_version,
            "total_evolutions": len(changelog),
            "changelog": changelog,
            "entity_types": self.schema_manager.get_entity_types(),
            "contribution_subtypes": self.schema_manager.get_contribution_subtypes(),
            "relation_types": self.schema_manager.get_relation_types(),
        }

    def save_results(self, output_dir: str):
        """Save all results to output directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Extraction results
        with open(os.path.join(output_dir, "extraction_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.extraction_results, f, ensure_ascii=False, indent=2)

        # QA pairs
        with open(os.path.join(output_dir, "qa_pairs.json"), "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)

        # Schema evolution summary
        with open(os.path.join(output_dir, "schema_evolution.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_schema_evolution_summary(), f, ensure_ascii=False, indent=2)

        print(f"  Results saved to {output_dir}", flush=True)
