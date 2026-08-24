"""Rich-topology + maintenance agent (Step 5): direct-read rich topology (the
verified deterministic win) + utility pruning (SEDM) + honest decay降级, all
through the KnowledgeBase kernel.

Design (see .research_tmp/plan_total_design_v2.md "Builder 群 / 富拓扑推断 + 维护 agent"
+ 断点 12/13):

Rich topology: the verified `infer_rich_topology_direct` reads 7 kinds of rich-
topology edges DIRECTLY from instance hyperedges (law_parameter / method_parameter
/ method_phenomenon / composition / definition / nary / method_regime) — NOT co-
occurrence guessing. This is the n-ary红利 (memory: 735->595 noise down, the
most stable selling point). Step 5 delegates this内核 (surgical) and routes the
resulting edges through kb.commit (add_edge for rich edges, OR they're produced
as a read-only view — see below).

Ambiguous rule (断点 12, honest narrowing): the direct read classifies by node-
TYPE combination, which is mostly MUTUALLY EXCLUSIVE per kind. True ambiguity
(>1 kind applicable, role overlap >0.5, same-sentence evidence) is RARE in direct
read. So Step 5 does NOT add an LLM ambiguous gate to the direct read (that
would be over-engineering a mostly-deterministic path). Instead: if a hyperedge
matches multiple kinds, it's recorded under the most-specific kind + flagged
`ambiguous_kinds` for audit. Honest scope — the design's ">1 类上 LLM" was
intended for co-occurrence inference; direct read doesn't need it.

Utility pruning (SEDM): retire low-utility patterns/edges by frequency /
citation / evolution contribution. Routes through kb.commit(retire Mutation) —
maintainer write contract. Soft-delete (deprecated flag), ledger-recoverable.

Decay reaper (断点 13, honest降级 FIRST): we do NOT implement an active background
decay reaper (MemoryBank forgetting-curve) — honest降级. retire = soft-delete +
ledger-recoverable; nothing is hard-deleted or auto-decayed. The reaper is a
reserved interface (decay_reaper hook) to add later if utility pruning alone
proves insufficient. This avoids over-claiming a MemoryBank feature we don't have.

Consumer working memory (DocTrace): snapshot_rich_topology(domain, concept_ids)
returns the on-demand subgraph for a consumer's query — NOT全量预计算. A consumer
fetches the rich-topology edges touching its query concepts.

Honest scope: the agent delegates the direct-read内核 (surgical); it adds the
maintenance (utility prune) + the on-demand view, both through the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from granular_agent.knowledge_base import KnowledgeBase, Mutation, Op, Role
from granular_agent.hypergraph_schema import InstanceHypergraph
import granular_agent.hypergraph_evolution as hev


@dataclass
class RichTopologyView:
    """An on-demand rich-topology subgraph for a consumer query (DocTrace
    working memory). NOT全量预计算 — only the edges touching the queried
    concepts. Carries the 7 kinds + ambiguous flags."""
    concept_ids: list[str]
    edges: list[dict] = field(default_factory=list)   # [{kind, node_ids, evidence, ambiguous_kinds}]
    n_by_kind: dict = field(default_factory=dict)


@dataclass
class MaintenanceReport:
    """Result of a maintenance pass. Lets the caller SEE what was pruned."""
    n_retired: int = 0
    retired: list[dict] = field(default_factory=list)   # [{target, reason, utility}]
    n_ambiguous_flagged: int = 0


class MaintenanceAgent:
    """Rich-topology direct read + utility pruning + honest decay降级, through
    the KnowledgeBase kernel.

    llm: reserved for the ambiguous LLM gate (currently unused — direct read is
    deterministic; honest scope). Kept in the constructor so adding the gate
    later doesn't change the interface.
    """

    def __init__(self, kb: KnowledgeBase,
                 llm: Callable[[str, int], str | None] | None = None,
                 domain_default: str = "global"):
        self.kb = kb
        self.llm = llm
        self.domain_default = domain_default

    # ===================================================================
    # Rich topology: direct read (delegated verified内核) + ambiguous flag
    # ===================================================================

    def infer_rich_topology(self, instance: InstanceHypergraph,
                            paper_id: str = "", include_other: bool = False
                            ) -> list[dict]:
        """Direct-read rich topology (the verified deterministic win). Delegates
        to infer_rich_topology_direct (surgical). Adds an `ambiguous_kinds` flag
        when a hyperedge matches >1 kind (断点 12 honest narrowing: direct read
        classifies by node-type combination, mostly mutually exclusive; the rare
        multi-match is flagged, NOT sent to LLM — that was intended for co-
        occurrence inference which we don't do)."""
        edges = hev.infer_rich_topology_direct(instance, paper_id=paper_id,
                                               include_other=include_other)
        # flag ambiguous (a hyperedge matching >1 kind is rare in direct read;
        # flag for audit rather than LLM-disambiguate)
        for e in edges:
            kinds = self._kinds_for_edge(e)
            if len(kinds) > 1:
                e["ambiguous_kinds"] = kinds
        return edges

    def _kinds_for_edge(self, edge: dict) -> list[str]:
        """Which rich-topology kinds this edge could be classified as (for
        ambiguous flagging). Direct read picks one; this checks if others
        also fit (role overlap).

        Honest scope (MAJOR fix): only the method_* kinds can overlap (a
        METHOD+PARAMETER+PHENOMENON edge is both method_parameter and
        method_phenomenon). The other kinds are determined by pattern_type and
        are mutually exclusive (law_parameter only for constitutive_law,
        composition only for composed_of, definition only for defines, nary for
        >=3 distinct types) — they can't co-occur as 'ambiguous' in a meaningful
        way. So we check the 3 method_* overlaps; law/composition/definition/
        nary are pattern_type-determined (no ambiguity to flag)."""
        # the direct read already set 'kind'; we check if the node-type combo
        # also fits other kinds (e.g. a METHOD+PARAMETER+PHENOMENON edge is both
        # method_parameter and method_phenomenon).
        labels = set()
        for nd in edge.get("nodes", []):
            labels.update(nd.get("labels", []))
        kinds = []
        if "METHOD" in labels and "PARAMETER" in labels:
            kinds.append("method_parameter")
        if "METHOD" in labels and "PHENOMENON" in labels:
            kinds.append("method_phenomenon")
        if "METHOD" in labels and "REGIME" in labels:
            kinds.append("method_regime")
        return kinds

    # ===================================================================
    # Consumer working memory: on-demand subgraph (DocTrace)
    # ===================================================================

    def snapshot_rich_topology(self, concept_ids: list[str]) -> RichTopologyView:
        """On-demand rich-topology subgraph for a consumer query (DocTrace
        working memory). Returns only the A-box hyperedges touching the queried
        concepts — NOT全量预计算. A consumer calls this per-query."""
        view = RichTopologyView(concept_ids=list(concept_ids))
        cid_set = set(concept_ids)
        for he in self.kb.abox.hyperedges:
            touched = set(he.node_ids) & cid_set
            if not touched:
                continue
            kinds = self._kinds_for_edge({"nodes": [
                {"labels": [self.kb.abox.concepts[cid].type]}
                for cid in he.node_ids if cid in self.kb.abox.concepts]})
            view.edges.append({
                "kind": he.kind, "node_ids": list(he.node_ids),
                "evidence": (he.provenance[0].get("evidence", "") if he.provenance else ""),
                "ambiguous_kinds": kinds if len(kinds) > 1 else []})
        for e in view.edges:
            k = e["kind"]
            view.n_by_kind[k] = view.n_by_kind.get(k, 0) + 1
        return view

    # ===================================================================
    # Utility pruning (SEDM): retire low-utility, through the kernel
    # ===================================================================

    def prune_by_utility(self, domain: str = "",
                         min_frequency: int = 0,
                         min_citations: int = 0) -> MaintenanceReport:
        """Utility pruning (SEDM): retire patterns whose utility (frequency /
        citation / evolution contribution) falls below thresholds. Routes
        through kb.commit(retire Mutation) — maintainer write contract, soft-
        delete (deprecated flag), ledger-recoverable. NOTHING hard-deleted.

        Utility here = how many A-box hyperedges use the pattern (frequency).
        Citation/evolution contribution would need cross-paper signals (reserved
        for when the consumer群 is wired, Step 6). Honest scope: frequency-only
        for now; the interface accepts min_citations for later."""
        domain = domain or self.domain_default
        report = MaintenanceReport()
        # utility = number of A-box hyperedges referencing this T-box pattern,
        # counted by he.pattern_type (the raw T-box ref) — NOT he.kind (the
        # rich-topology classification). (BLOCKER fix: previously used he.kind
        # which is a rich-topology label like method_parameter/law_parameter,
        # mismatching T-box pattern_id like constitutive_law/defines — this
        # wrongly retired in-use patterns. pattern_type is the T-box reference.)
        usage: dict[str, int] = {}
        for he in self.kb.abox.hyperedges:
            pt = he.pattern_type or he.kind  # fall back to kind only if no
            # pattern_type recorded (legacy edges); honest: kind is a poor proxy
            # but better than skipping the edge entirely.
            if pt:
                usage[pt] = usage.get(pt, 0) + 1
        for pid, pat in list(self.kb.tbox.patterns.items()):
            if pat.deprecated:
                continue
            freq = usage.get(pid, 0)
            if freq < min_frequency:
                # retire: maintainer op, soft-delete, ledger-recoverable
                result = self.kb.commit([Mutation(
                    op=Op.RETIRE, target=pid, proposer_role=Role.MAINTAINER,
                    domain=domain, evidence=f"utility prune: frequency={freq} < {min_frequency}",
                    rationale="SEDM utility pruning", payload={})])
                if result.ok:
                    report.n_retired += 1
                    report.retired.append({"target": pid, "reason": "low-frequency",
                                           "utility": {"frequency": freq}})
        return report

    # ===================================================================
    # Decay reaper (断点 13, honest降级 FIRST)
    # ===================================================================

    def decay_reaper(self, *args, **kwargs):
        """Honest降级 (断点 13): NO active background decay reaper is implemented.
        retire = soft-delete + ledger-recoverable; nothing auto-decays. This
        avoids over-claiming a MemoryBank forgetting-curve feature we don't
        have. The hook is reserved (a no-op) so adding a real reaper later
        doesn't change the interface — but we honestly do NOT have one now."""
        return {"decayed": 0, "note": "no active reaper — retire is soft-delete, ledger-recoverable"}
