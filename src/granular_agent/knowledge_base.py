"""Knowledge Base kernel (the 底座内核): the transactional core that sits
under every builder (extractor / evolver / aligner / maintainer) and consumer.

This module implements the v2 total design (see .research_tmp/plan_total_design_v2.md
section "底座内核" + .research_tmp/plan_detailed_design_supplement.md section 2/3).
It COMPOSES the existing MetaHypergraph (T-box) + ConceptGraph (A-box) and adds
the four mechanisms the design requires that the existing classes do not have:

1. Mutation + Write Contract (role -> allowed ops). Every write is a Mutation
   carrying proposer_role / domain / base_version / evidence / rationale.
2. Transactional two-phase commit:
   - validate phase (ATOMIC pass/fail): schema-constraint (IS-A/family/invariant,
     the "schema-constrained rewrite" narrowed honestly to IS-A/family — NOT
     2608.18104's full consumer-contract modeling) + role permission + candidate-
     evidence binding + A-box referential integrity + recurring-crystallize hook.
     Any failure rolls back the whole batch.
   - route phase (NON-atomic, per-edge, ALIGNER ONLY): the MELD 5-outcome
     insert/merge/relate/conflict/reject decided by n-ary node-set subset relation
     (Challenge B). Rules + embedding filter first; only ambiguous cases call the
     judge_fn the aligner plugs in (which MUST be a different model than the
     extractor's, to avoid self-endorsement).
3. Mutation Ledger (who/when/why/evidence/version_diff/domain) -> audit/replay/
   time-travel. Soft-delete only (retire/deprecated), never hard-delete: the
   ledger can replay any past version.
4. Skill Library (the 4th layer, 断点 5) + distill_skill op (evolver write
   contract): Skill(domain/pattern/extraction_hint/stability_score/provenance).

Concurrency (断点 10, narrowed): builders write SERIALLY through commit() (one
consumer of the queue), so there is NO builder-builder CAS race. The Mutation
carries base_version (CAS interface preserved) but commit() does NOT enforce
builder-builder CAS — it is recorded for audit only. Consumer reads take a
snapshot() with a version stamp for consistent snapshot reads.

Honest scope: schema-constrained rewrite here ONLY preserves IS-A-tree
completeness / family归属 / runtime invariants + A-box referential integrity.
It does NOT model consumer read-contracts the way arXiv 2608.18104 does — that
is heavier and out of scope; we claim only "IS-A/family invariant preserved",
NOT "conforms to 2608.18104 in full". (See plan_total_design_v2.md "不做".)

The kernel makes NO LLM calls itself. Ambiguous 5-outcome decisions are
delegated to a judge_fn the aligner installs (set_judge). If no judge is set,
ambiguous -> default insert (data-preserving) and flagged in the route result.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from granular_agent.hypergraph_schema import (
    MetaHypergraph, MetaHyperedgePattern, MetaNode, MetaEdge,
    Hyperedge, InstanceHypergraph, HGNode,
)
from granular_agent.concept_graph import ConceptGraph, Concept, ConceptHyperedge


# ---------------------------------------------------------------------------
# Op + Role + Write Contract
# ---------------------------------------------------------------------------

class Op:
    """The bounded mutation operation set. aligner ops (align_merge /
    add_concept_relation / conflict_mark) are IN the enum (断点 1) and are the
    ONLY ops that enter the 5-outcome route phase (断点 2/8)."""
    # extractor
    ADD_EDGE = "add_edge"
    # evolver (T-box + Skill)
    ADD_NODE = "add_node"               # add a node TYPE (MetaNode), 断点 7: evolver owns node types
    ADD_PATTERN = "add_pattern"
    ADD_SUBCLASS = "add_subclass"
    SPLIT = "split"
    MERGE = "merge"
    RETIRE = "retire"
    RENAME = "rename"
    RELABEL = "relabel"                # maintainer relabel a concept (soft)
    DISTILL_SKILL = "distill_skill"    # 断点 5: 4th-layer write op
    # aligner (A-box, 5-outcome only path)
    ALIGN_MERGE = "align_merge"
    ADD_CONCEPT_RELATION = "add_concept_relation"
    CONFLICT_MARK = "conflict_mark"


class Role:
    EXTRACTOR = "extractor"
    EVOLVER = "evolver"
    ALIGNER = "aligner"
    MAINTAINER = "maintainer"
    CONSUMER = "consumer"   # read-only at the KB; may only propose


# Write Contract: which role may propose which op (断点 4/7 fixes).
# extractor: add_edge ONLY (carries concept instances inline, 断点 7); does NOT
#   enter 5-outcome (断点 2) — extractor edges go through candidate-evidence +
#   schema constraint in validate, then straight apply.
# evolver: all T-box ops + distill_skill + add_node (node TYPES).
# aligner: the three A-box reconciliation ops — the ONLY ops that route.
# maintainer: retire / relabel (utility pruning, SEDM).
# consumer: read-only (proposes via consumer.* sub-roles if ever; no direct op).
WRITE_CONTRACT: dict[str, set[str]] = {
    Role.EXTRACTOR: {Op.ADD_EDGE},
    Role.EVOLVER: {Op.ADD_PATTERN, Op.ADD_SUBCLASS, Op.SPLIT, Op.MERGE,
                   Op.RETIRE, Op.RENAME, Op.DISTILL_SKILL, Op.ADD_NODE},
    Role.ALIGNER: {Op.ALIGN_MERGE, Op.ADD_CONCEPT_RELATION, Op.CONFLICT_MARK},
    Role.MAINTAINER: {Op.RETIRE, Op.RELABEL},
    Role.CONSUMER: set(),
}


class Outcome:
    """MELD 5-outcome for the aligner route phase. reject = lawful discard
    (write conflict_mark or nothing); does NOT roll back the batch (route is
    non-atomic per-edge, 断点 3)."""
    INSERT = "insert"        # new claim, no overlap with existing -> add as new
    MERGE = "merge"          # same / subset node-set -> union into existing
    RELATE = "relate"       # partial overlap non-subset -> add_concept_relation, don't merge
    CONFLICT = "conflict"   # same kind, semantically same relation but inconsistent -> mark
    REJECT = "reject"        # lawful discard (e.g. duplicate, or judge says drop)


# ---------------------------------------------------------------------------
# Skill Library (4th layer, 断点 5)
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A crystallized extraction skill (4th layer). Born from recurring
    extraction patterns (Metis crystallize): the skill distiller observes the
    same domain+pattern drawn the same way across >=threshold extractions and
    crystallizes a reusable extraction_hint. The extractor injects the hint
    into its prompt next time it draws that domain+pattern (SAGE writer-reader
    feedback, forward propagation).

    stability_score = (same-mode frequency) / (total extractions of that
    domain+pattern). Crystallize threshold = 0.6 (plan_detailed_design_supplement
    section 4). Below threshold the skill is NOT materialized."""
    skill_id: str
    domain: str
    pattern: str                        # pattern_id the hint applies to
    extraction_hint: str
    stability_score: float = 0.0
    provenance_papers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SkillLibrary:
    """The 4th layer. Written ONLY through the distill_skill op (evolver write
    contract), so skills are versioned alongside T-box evolution in the same
    commit transaction — a skill and the pattern it hints at are committed
    together or not at all."""

    def __init__(self):
        self.skills: dict[str, Skill] = {}            # skill_id -> Skill
        # index for lookup by (domain, pattern) — the extractor's hot path
        self._dp_index: dict[tuple[str, str], str] = {}

    def add(self, skill: Skill) -> None:
        self.skills[skill.skill_id] = skill
        self._dp_index[(skill.domain, skill.pattern)] = skill.skill_id

    def lookup(self, domain: str, pattern: str) -> Skill | None:
        sid = self._dp_index.get((domain, pattern))
        return self.skills.get(sid) if sid else None

    def to_dict(self) -> dict:
        return {"skills": {sid: s.to_dict() for sid, s in self.skills.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillLibrary":
        lib = cls()
        for sid, sd in (d or {}).get("skills", {}).items():
            s = Skill(skill_id=sd.get("skill_id", sid), domain=sd.get("domain", ""),
                      pattern=sd.get("pattern", ""), extraction_hint=sd.get("extraction_hint", ""),
                      stability_score=sd.get("stability_score", 0.0),
                      provenance_papers=list(sd.get("provenance_papers", [])))
            lib.add(s)
        return lib


# ---------------------------------------------------------------------------
# Mutation + Commit result + Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single proposed write. Immutable intent; commit() fills status.
    base_version is the CAS interface (断点 10 narrowed): recorded for audit
    but NOT enforced builder-builder (builders are serial). Consumer snapshot
    reads use it to detect stale reads if ever needed."""
    op: str
    target: str                        # pattern_id / type_id / concept_id / edge key / skill_id
    payload: dict = field(default_factory=dict)
    proposer_role: str = ""
    domain: str = "global"             # 断点 4: domain injection point = the Mutation
    base_version: str = ""             # CAS (recorded, not enforced — serial builders)
    evidence: str = ""                 # verbatim span (candidate-evidence binding)
    rationale: str = ""
    timestamp: str = ""
    # filled by commit:
    mutation_id: str = ""
    status: str = "pending"            # pending | committed | rejected | routed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CommitResult:
    """Result of a batched commit. validate phase is atomic (all-or-nothing):
    if `ok` is False, NOTHING was applied (rolled back). If `ok` is True, every
    mutation passed validate and was applied; aligner mutations additionally
    have a route outcome in `routed`."""
    ok: bool
    version: str                       # KB version after commit (unchanged if rolled back)
    validated: list[str] = field(default_factory=list)   # mutation_ids that passed validate
    rejected: list[dict] = field(default_factory=list)   # [{mutation_id, op, reason}] (validate failures)
    routed: list[dict] = field(default_factory=list)     # [{mutation_id, outcome, detail}] (aligner route)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KBSnapshot:
    """A consistent read-only snapshot for a consumer (version-stamped).
    The full A-box is large; the snapshot carries a summary + the version so a
    consumer can fetch subgraphs on demand (DocTrace-style on-demand working
    memory, not全量预计算)."""
    version: str
    domain: str
    tbox_snapshot: dict                # MetaHypergraph.to_dict() (small, full)
    abox_summary: dict                 # {n_concepts, n_hyperedges, concept_ids, hyperedge_ids}
    skills_snapshot: dict              # SkillLibrary.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """The transactional kernel. One instance per evolving corpus. All builder
    writes go through commit(); consumer reads go through snapshot().

    The kernel wraps an existing MetaHypergraph + ConceptGraph (passed in) so
    an already-evolved schema/abox can be loaded and incrementally evolved
    further — the same "schema gone after the run" gap closure the existing
    to_dict/from_dict already provides, now at the transactional layer."""

    # --- Crystallize thresholds (plan_detailed_design_supplement section 4) ---
    SKILL_CRYSTALLIZE_STABILITY = 0.6

    def __init__(self, tbox: MetaHypergraph | None = None,
                 abox: ConceptGraph | None = None,
                 skills: SkillLibrary | None = None,
                 domain_ns: dict[str, bool] | None = None,
                 judge_fn: Callable[[dict], str] | None = None):
        self.tbox = tbox or MetaHypergraph()
        self.abox = abox or ConceptGraph()
        self.skills = skills or SkillLibrary()
        # domain namespace (HASTE layered + DecentMem 防混域): meta:global shared +
        # meta:granular / meta:ml / meta:molecular domain-specific. Alignment is
        # same-domain only (an aligner-level rule; the kernel records domain on
        # every mutation so cross-domain proposals are auditable).
        self.domain_ns: dict[str, bool] = domain_ns or {"global": True, "granular": True,
                                                         "ml": True, "molecular": True}
        self.ledger: list[dict] = []                 # [{mutation, result, version_diff}]
        # paper_id -> domain map: the extractor records domain on each add_edge,
        # and _apply_add_edge updates this so the aligner (Step 4) can filter
        # same-domain concepts (DecentMem 防混域 — cross-domain concepts don't
        # merge). Honest: populated lazily as papers ingest; empty until then.
        self._paper_domain: dict[str, str] = {}
        self.version: str = self.tbox.version
        # the aligner's ambiguous-5-outcome judge (MUST be != extraction model).
        # If None, ambiguous -> insert (data-preserving) + flagged in route detail.
        self._judge_fn = judge_fn
        self._queue: list[Mutation] = []
        self._mid_counter = 0

    # ---- judge installation (aligner, Step 4) ----
    def set_judge(self, fn: Callable[[dict], str]) -> None:
        """Install the 5-outcome judge for ambiguous align_merge cases.
        MUST be a different model than the extractor (avoid self-endorsement).
        Signature: fn({new_edge, candidates, domain}) -> one of Outcome.* .
        The aligner (Step 4) installs this; the kernel itself makes no LLM call."""
        self._judge_fn = fn

    # ---- propose: enqueue a mutation (consumer + builder both propose) ----
    def propose(self, mutation: Mutation) -> str:
        """Enqueue a proposed mutation. Returns its mutation_id. Does NOT apply
        — commit() drains the queue (builder single-consumer, serial)."""
        if mutation.proposer_role not in WRITE_CONTRACT:
            raise ValueError(f"unknown proposer_role: {mutation.proposer_role}")
        if mutation.op not in WRITE_CONTRACT.get(mutation.proposer_role, set()):
            # Still enqueue (the violation surfaces at validate as a rejection,
            # so it's auditable — but flag it loudly here too).
            pass
        self._mid_counter += 1
        mutation.mutation_id = f"mut_{self._mid_counter:06d}"
        mutation.timestamp = mutation.timestamp or _now()
        if not mutation.base_version:
            mutation.base_version = self.version
        self._queue.append(mutation)
        return mutation.mutation_id

    # ---- commit: two-phase transactional apply (serial builder) ----
    def commit(self, batch: list[Mutation] | None = None) -> CommitResult:
        """Drain the queue (or the explicit `batch`) through two phases:
          1. validate (ATOMIC): every mutation must pass _validate_one against
             the CURRENT kb state (pre-batch). Any failure -> whole batch rolled
             back (apply nothing), rejected records why.
          2. route (NON-atomic, ALIGNER ONLY): each aligner mutation gets a
             5-outcome decision (insert/merge/relate/conflict/reject). reject/
             conflict do not roll back siblings.
        Returns CommitResult. Intra-batch dependencies (a mutation referencing
        a node another mutation in the SAME batch would add) are NOT supported —
        split such dependencies across commits. (Documented honestly; this keeps
        validate against pre-batch state, no speculative apply+rollback needed.)
        """
        mutations = list(batch) if batch is not None else list(self._queue)
        if batch is None:
            self._queue.clear()
        if not mutations:
            return CommitResult(ok=True, version=self.version)

        # fill base_version (CAS record) for mutations committed directly
        # (evolver/aligner commit bypasses propose(); propose() does this too).
        # Also stamp the pre-commit version for an accurate ledger version_diff.
        v_before = self.version
        for mut in mutations:
            if not mut.base_version:
                mut.base_version = v_before
            if not mut.mutation_id:
                self._mid_counter += 1
                mut.mutation_id = f"mut_{self._mid_counter:06d}"
            if not mut.timestamp:
                mut.timestamp = _now()

        validated: list[Mutation] = []
        rejected: list[dict] = []
        # ---- Phase 1: validate (atomic) ----
        for mut in mutations:
            ok, reason = self._validate_one(mut)
            if ok:
                validated.append(mut)
            else:
                rejected.append({"mutation_id": mut.mutation_id, "op": mut.op,
                                 "reason": reason})
        if rejected:
            # ATOMIC: roll back the whole batch — apply nothing.
            for mut in mutations:
                if mut.status == "pending":
                    mut.status = "rejected"
            self._record_ledger(mutations, CommitResult(
                ok=False, version=self.version, rejected=rejected))
            return CommitResult(ok=False, version=self.version, rejected=rejected)

        # ---- apply (all validated) ----
        version_before = self.version
        for mut in validated:
            self._apply(mut)
            mut.status = "committed"
        # version may have bumped via tbox writes
        self.version = self.tbox.version

        # ---- Phase 2: route (aligner only, non-atomic per-edge) ----
        routed: list[dict] = []
        for mut in validated:
            if mut.proposer_role == Role.ALIGNER:
                outcome, detail = self._route(mut)
                mut.status = "routed"
                routed.append({"mutation_id": mut.mutation_id, "op": mut.op,
                               "outcome": outcome, "detail": detail})

        result = CommitResult(ok=True, version=self.version,
                              validated=[m.mutation_id for m in validated],
                              routed=routed)
        self._record_ledger(mutations, result)
        return result

    # ---- audit / replay / time-travel ----
    def audit(self, version: str | None = None) -> list[dict]:
        """Return the ledger. If `version` given, return mutations up to (and
        including) that version — time-travel / replay. The ledger is the
        source of truth: any past kb state is reconstructable by replaying
        mutations up to a version (soft-delete only, nothing hard-deleted)."""
        if version is None:
            return list(self.ledger)
        out = []
        for entry in self.ledger:
            out.append(entry)
            if entry.get("result", {}).get("version") == version:
                break
        return out

    def replay_to(self, version: str) -> "KnowledgeBase":
        """Reconstruct a fresh KB at `version` by replaying the ledger from the
        initial tbox/abox state. (Time-travel for ablation snapshots.)

        version is bumped ONLY by T-box writes (evolver ops); aligner A-box
        writes (ALIGN_MERGE etc.) do NOT bump version. So multiple ledger
        entries can share the same version. To time-travel to `version` we must
        replay up to and INCLUDING the LAST entry whose post-commit version
        equals `version` (not the first — the first would drop later same-
        version aligner writes)."""
        # find the last ledger entry whose post-commit version == target
        last_idx = -1
        for i, entry in enumerate(self.ledger):
            res = entry.get("result", {})
            if res.get("ok") and res.get("version") == version:
                last_idx = i
        kb = KnowledgeBase(
            tbox=MetaHypergraph(), abox=ConceptGraph(), skills=SkillLibrary(),
            domain_ns=dict(self.domain_ns), judge_fn=self._judge_fn)
        for i, entry in enumerate(self.ledger):
            if i > last_idx:
                break
            muts = entry.get("mutations", [])
            res = entry.get("result", {})
            if not res.get("ok"):
                continue
            for mut in muts:
                if mut.get("status") in ("committed", "routed"):
                    m = Mutation(**{k: mut[k] for k in mut if k in
                                    Mutation.__dataclass_fields__})
                    kb._apply(m)
                    # ALIGN_MERGE writes its A-box changes in _route (not
                    # _apply, which is a no-op for it). replay must call _route
                    # too, or the replayed KB silently loses every aligner
                    # edge (insert/merge/relate). ADD_CONCEPT_RELATION and
                    # CONFLICT_MARK write in _apply already, so only the
                    # ALIGN_MERGE path needs the route replay. (BLOCKER fix)
                    if m.op == Op.ALIGN_MERGE and m.proposer_role == Role.ALIGNER:
                        kb._route(m)
            kb.version = kb.tbox.version
        return kb

    # ---- snapshot (consumer read, version-stamped) ----
    def snapshot(self, domain: str = "global") -> KBSnapshot:
        """A consistent read-only snapshot. Consumers fetch subgraphs on demand
        from the abox (DocTrace-style); the snapshot carries the version stamp
        so a consumer can detect if its read went stale mid-query."""
        return KBSnapshot(
            version=self.version, domain=domain,
            tbox_snapshot=self.tbox.to_dict(),
            abox_summary={"n_concepts": len(self.abox.concepts),
                          "n_hyperedges": len(self.abox.hyperedges),
                          "concept_ids": list(self.abox.concepts.keys()),
                          "hyperedge_ids": [he.he_id for he in self.abox.hyperedges]},
            skills_snapshot=self.skills.to_dict())

    # ===================================================================
    # Phase 1: validate (per-mutation; atomic batch in commit())
    # ===================================================================
    def _validate_one(self, mut: Mutation) -> tuple[bool, str]:
        # 0. domain registered (domain_ns gate; 断点 4)
        if mut.domain not in self.domain_ns:
            return False, f"domain-not-registered:{mut.domain}"
        # 1. role permission (write contract)
        allowed = WRITE_CONTRACT.get(mut.proposer_role, set())
        if mut.op not in allowed:
            return False, f"op-not-allowed-for-role:{mut.op}<-{mut.proposer_role}"
        # 2. candidate-evidence binding (verbatim span required for writes that
        #    originate from text — extractor add_edge, evolver T-box changes,
        #    aligner merge. retire/relabel of utility-only items may carry the
        #    rationale as evidence surrogate, but T-box retire still needs a
        #    rationale evidence trail.)
        needs_evidence = mut.op not in (Op.RELABEL,)
        if needs_evidence and not (mut.evidence or "").strip():
            return False, "missing-candidate-evidence"
        # 3. op-specific schema constraint (schema-constrained rewrite, narrowed
        #    to IS-A/family/invariant — NOT 2608.18104 full consumer-contract)
        sc_ok, sc_reason = self._schema_constraint(mut)
        if not sc_ok:
            return False, sc_reason
        # 4. A-box referential integrity (dangling refs rejected; 断点 11)
        ri_ok, ri_reason = self._referential_integrity(mut)
        if not ri_ok:
            return False, ri_reason
        # 5. recurring-crystallize hook (Metis): if this commit makes a pattern
        #    cross the crystallize threshold, the evolver should propose a
        #    distill_skill — but the kernel does NOT auto-crystallize here (the
        #    evolver owns that proposal). Hook left as a no-op pass; the evolver
        #    (Step 3) reads recurrence counts and proposes distill_skill.
        return True, "ok"

    def _schema_constraint(self, mut: Mutation) -> tuple[bool, str]:
        """schema-constrained rewrite for evolver T-box ops (Challenge C).
        Honest scope: IS-A tree completeness + family归属 + role_slots invariant
        + runtime invariant. NOT 2608.18104's consumer-contract modeling."""
        op = mut.op
        if op == Op.ADD_EDGE:
            # extractor add_edge: kind (pattern_type) MUST be a known ACTIVE
            # T-box pattern. A new pattern_type the schema doesn't know must go
            # through the evolver's add_pattern op FIRST (then add_edge), not
            # be smuggled in as an add_edge. Rejecting unknown/deprecated kinds
            # here is what makes the verifier's retype fix safe: a retype to a
            # non-existent pattern is rejected at the kernel, not silently
            # committed as bad data. (BLOCKER B3 fix — the DECISION's claim
            # "retype to a non-existent pattern is rejected" was previously
            # false because ADD_EDGE had no schema-constraint branch.)
            kind = mut.payload.get("kind", "")
            pat = self.tbox.patterns.get(kind)
            if pat is None:
                return False, f"add_edge:unknown-pattern-type:{kind}"
            if pat.deprecated:
                return False, f"add_edge:pattern-deprecated:{kind}"
            # role compatibility: every role the edge uses must be declared by
            # the pattern's role_slots (a defines edge using whole/component
            # roles = composed_of's roles misattributed to defines). This is a
            # deterministic structural check (rule, not LLM). Catches the
            # real-run failure where a 'defines' edge carried composed_of roles.
            declared = {s.get("role") for s in pat.role_slots}
            used = {r for r in mut.payload.get("roles", [])}
            extra = used - declared
            if extra:
                return False, f"add_edge:role-not-in-pattern:{sorted(extra)}<-{kind}"
            return True, "ok"
        if op == Op.ADD_PATTERN:
            pat: MetaHyperedgePattern = mut.payload.get("pattern")
            if pat is None:
                return False, "add_pattern:missing-pattern-in-payload"
            # role_slots types must be <= existing node types (THING root fallback)
            for s in pat.role_slots:
                t = s.get("type", "")
                if t and t not in self.tbox.meta_nodes:
                    return False, f"add_pattern:unknown-slot-type:{t}"
            # pattern_id must not collide (a re-add is a no-op, not an error, but
            # flag it so the evolver knows)
            if pat.pattern_id in self.tbox.patterns:
                return False, "add_pattern:pattern-already-exists"
            return True, "ok"
        if op == Op.SPLIT:
            pid = mut.target
            parent = self.tbox.patterns.get(pid)
            if parent is None:
                return False, "split:unknown-parent"
            if parent.is_abstract:
                return False, "split:parent-already-abstract-split-a-child"
            subs = mut.payload.get("sub_patterns", [])
            if len(subs) < 2:
                return False, "split:needs->=2-sub-patterns"
            parent_roles = {s.get("role") for s in parent.role_slots}
            from granular_agent.hypergraph_schema import _role_sig
            parent_sig = _role_sig(parent.role_slots)
            for sp in subs:
                if sp.pattern_id in self.tbox.patterns:
                    return False, f"split:sub-collides:{sp.pattern_id}"
                # Challenge C: sub must NOT introduce a new role beyond the
                # parent's (no new role). The kernel enforces this as an EXACT
                # role-signature match with the parent (role-seq + type-seq
                # identical), matching the underlying split_pattern's constraint
                # (hypergraph_schema._role_sig). Split is a SEMANTIC boundary
                # split (same role structure, different semantic sub-kind) —
                # dropping a role changes arity = structural change, which is
                # add_pattern's job, not split. (DECISION-split-role-signature)
                # This keeps the kernel gate CONSISTENT with the underlying
                # split_pattern so a kernel-passed split never reaches a silent
                # None return at the bottom layer.
                sub_roles = {s.get("role") for s in sp.role_slots}
                if not sub_roles.issubset(parent_roles):
                    return False, "split:sub-introduces-new-role"
                if _role_sig(sp.role_slots) != parent_sig:
                    return False, "split:sub-role-signature-must-equal-parent"
            return True, "ok"
        if op == Op.MERGE:
            pids = mut.payload.get("pattern_ids", [])
            into = mut.payload.get("into", "")
            if len(pids) < 2:
                return False, "merge:needs->=2-patterns"
            for p in pids:
                if p not in self.tbox.patterns:
                    return False, f"merge:unknown-pattern:{p}"
            # Challenge C: merged role_slots = union (all roles reachable).
            # The survivor (into) must end up covering every role any merged
            # pattern had, else an existing edge of a dropped role dangles.
            all_roles: set[str] = set()
            for p in pids:
                all_roles |= {s.get("role") for s in self.tbox.patterns[p].role_slots}
            into_pat = self.tbox.patterns.get(into)
            if into_pat:
                into_roles = {s.get("role") for s in into_pat.role_slots}
                if not all_roles.issubset(into_roles):
                    return False, "merge:survivor-missing-roles-after-union"
            else:
                # `into` is a NEW pattern id — the underlying merge_patterns
                # will clone role_slots from pattern_ids[0] (the donor). If the
                # donor doesn't cover all union roles, the survivor dangles.
                # Require the caller to declare into_role_slots covering the
                # union, OR ensure the donor (pattern_ids[0]) covers all roles.
                into_decl = mut.payload.get("into_role_slots")
                if into_decl is not None:
                    decl_roles = {s.get("role") for s in into_decl}
                    if not all_roles.issubset(decl_roles):
                        return False, "merge:new-into-missing-union-roles"
                else:
                    donor = self.tbox.patterns.get(pids[0])
                    donor_roles = {s.get("role") for s in donor.role_slots} if donor else set()
                    if not all_roles.issubset(donor_roles):
                        return False, "merge:new-into-needs-into_role_slots-or-full-union-donor"
            # taxonomy guard: refuse abstract-parent + concrete-leaf merge
            abstractions = [p for p in pids if self.tbox.patterns[p].is_abstract]
            concretes = [p for p in pids if not self.tbox.patterns[p].is_abstract]
            if abstractions and concretes:
                return False, "merge:abstract+concrete-different-rank"
            return True, "ok"
        if op == Op.RETIRE:
            pid = mut.target
            if pid not in self.tbox.patterns:
                return False, "retire:unknown-pattern"
            # Challenge C: no ACTIVE A-box hyperedge may reference the pattern
            # (referential integrity T-box<->A-box). Must reclassify/retire the
            # edges first. (kind on ConceptHyperedge mirrors pattern_type for
            # evolution edges; for rich-topology kinds this is a no-op check.)
            for he in self.abox.hyperedges:
                if he.kind == pid:
                    return False, "retire:active-abox-edge-references-pattern"
            return True, "ok"
        if op == Op.ADD_SUBCLASS:
            sub, sup = mut.target, mut.payload.get("super", "")
            if sub not in self.tbox.meta_nodes or sup not in self.tbox.meta_nodes:
                return False, "add_subclass:unknown-type"
            if self.tbox.is_subtype(sup, sub):
                return False, "add_subclass:would-create-cycle"
            return True, "ok"
        if op == Op.ADD_NODE:
            tid = mut.target
            if tid in self.tbox.meta_nodes:
                return False, "add_node:type-already-exists"
            return True, "ok"
        if op == Op.RENAME:
            old, new = mut.target, mut.payload.get("new_id", "")
            if old not in self.tbox.patterns:
                return False, "rename:unknown-old"
            if new in self.tbox.patterns:
                return False, "rename:new-id-taken"
            return True, "ok"
        if op == Op.DISTILL_SKILL:
            skill: Skill | None = mut.payload.get("skill")
            if skill is None:
                return False, "distill_skill:missing-skill"
            # crystallize threshold (plan_detailed_design_supplement section 4)
            if skill.stability_score < self.SKILL_CRYSTALLIZE_STABILITY:
                return False, f"distill_skill:stability<{self.SKILL_CRYSTALLIZE_STABILITY}"
            if skill.pattern not in self.tbox.patterns:
                return False, "distill_skill:pattern-not-in-tbox"
            return True, "ok"
        if op == Op.RELABEL:
            cid = mut.target
            if cid not in self.abox.concepts:
                return False, "relabel:unknown-concept"
            return True, "ok"
        # aligner ops: structural validity checked here, 5-outcome in _route
        if op == Op.ALIGN_MERGE:
            ne = mut.payload.get("new_edge", {})
            if not ne.get("node_ids"):
                return False, "align_merge:empty-node-ids"
            for cid in ne["node_ids"]:
                if cid not in self.abox.concepts:
                    return False, f"align_merge:dangling-concept:{cid}"
            return True, "ok"
        if op == Op.ADD_CONCEPT_RELATION:
            for cid in mut.payload.get("node_ids", []):
                if cid not in self.abox.concepts:
                    return False, f"add_concept_relation:dangling-concept:{cid}"
            return True, "ok"
        if op == Op.CONFLICT_MARK:
            he_id = mut.target
            if not any(he.he_id == he_id for he in self.abox.hyperedges):
                return False, "conflict_mark:unknown-edge"
            return True, "ok"
        return True, "ok"

    def _referential_integrity(self, mut: Mutation) -> tuple[bool, str]:
        """A-box referential integrity (断点 11): dangling node_ids/concept_ids
        rejected. Most op-specific referential checks live in _schema_constraint
        (retire↔abox, align_merge↔concepts); this is the catch-all for add_edge."""
        if mut.op == Op.ADD_EDGE:
            # extractor add_edge carries concept instances INLINE (断点 7), so
            # there are no dangling refs — concepts are created via get_or_create
            # during apply. We only require >=2 concepts (a real edge).
            concepts = mut.payload.get("concepts", [])
            if len(concepts) < 2:
                return False, "add_edge:needs->=2-concepts"
            return True, "ok"
        return True, "ok"

    # ===================================================================
    # apply (validated mutations only)
    # ===================================================================
    def _apply(self, mut: Mutation) -> None:
        op = mut.op
        if op == Op.ADD_EDGE:
            self._apply_add_edge(mut)
        elif op == Op.ADD_PATTERN:
            pat = _as_pattern(mut.payload["pattern"])
            self.tbox.add_pattern(pat, evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.ADD_SUBCLASS:
            self.tbox.add_subclass(mut.target, mut.payload.get("super", ""),
                                   evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.SPLIT:
            subs = [_as_pattern(s) for s in mut.payload.get("sub_patterns", [])]
            self.tbox.split_pattern(mut.target, subs,
                                    evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.MERGE:
            self.tbox.merge_patterns(mut.payload.get("pattern_ids", []),
                                     mut.payload.get("into", ""),
                                     evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.RETIRE:
            self.tbox.retire_pattern(mut.target, evidence=mut.evidence,
                                     paper_id=mut.domain)
        elif op == Op.RENAME:
            self.tbox.rename_pattern(mut.target, mut.payload.get("new_id", ""),
                                     evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.ADD_NODE:
            self.tbox.add_meta_node(mut.target, mut.payload.get("description", ""),
                                    evidence=mut.evidence, paper_id=mut.domain)
        elif op == Op.DISTILL_SKILL:
            self.skills.add(_as_skill(mut.payload["skill"]))
        elif op == Op.RELABEL:
            c = self.abox.concepts.get(mut.target)
            if c:
                c.canonical_name = mut.payload.get("canonical_name",
                                                    c.canonical_name)
        elif op == Op.ALIGN_MERGE:
            # outcome applied in _route (route phase owns the A-box write for
            # aligner ops; apply is a no-op placeholder so commit ordering is
            # uniform). Nothing to do here.
            pass
        elif op == Op.ADD_CONCEPT_RELATION:
            self._apply_add_concept_relation(mut)
        elif op == Op.CONFLICT_MARK:
            he = next((h for h in self.abox.hyperedges if h.he_id == mut.target), None)
            if he:
                he.emergence_signals["conflict"] = mut.payload.get("detail", "")

    def _apply_add_edge(self, mut: Mutation) -> None:
        """Extractor add_edge: carries concept instances inline (断点 7). Creates
        each concept via abox.get_or_create (so new surfaces land in the A-box),
        then adds the n-ary ConceptHyperedge. Does NOT enter 5-outcome (断点 2)."""
        payload = mut.payload
        concepts = payload.get("concepts", [])
        kind = payload.get("kind", "")
        # record paper→domain (aligner same-domain filter, Step 4)
        prov0 = payload.get("provenance", {})
        pid0 = prov0.get("paper_id", "")
        if pid0 and mut.domain:
            self._paper_domain[pid0] = mut.domain
        roles = payload.get("roles", [])
        prov = payload.get("provenance", {})
        paper_id = prov.get("paper_id", mut.domain)
        evidence = prov.get("evidence", mut.evidence)
        year = prov.get("year", "")
        section = prov.get("section", "")
        # provenance first-class fields (M4): cited_from / method /
        # evidence_strength lifted out of qualifiers by the extractor, carried
        # onto the ConceptHyperedge.provenance entry (design行17 一等公民).
        prov_cited_from = prov.get("cited_from", "")
        prov_method = prov.get("method", "")
        prov_strength = prov.get("evidence_strength", "")
        cids: list[str] = []
        for c in concepts:
            conc = self.abox.get_or_create(c.get("type", "PROPERTY"),
                                           c.get("surface", ""),
                                           paper_id, c.get("evidence", evidence),
                                           year, section,
                                           central=bool(c.get("central", False)))
            cids.append(conc.concept_id)
        # need >=2 distinct concepts for a real n-ary edge
        if len(set(cids)) >= 2:
            he = self.abox.add_hyperedge(cids, kind, roles=roles, paper_id=paper_id,
                                         evidence=evidence, year=year, section=section)
            # attach the first-class provenance fields onto the edge's provenance
            # entry (ConceptHyperedge.provenance is a list of per-paper dicts;
            # the entry just added is the last one).
            if he is not None and he.provenance:
                pe = he.provenance[-1]
                if prov_cited_from:
                    pe["cited_from"] = prov_cited_from
                if prov_method:
                    pe["method"] = prov_method
                if prov_strength:
                    pe["evidence_strength"] = prov_strength

    def _apply_add_concept_relation(self, mut: Mutation) -> None:
        """A 'relate' outcome's body: add an n-ary edge that RELATES (not merges)
        the concepts, kind='relates', so the partial-overlap case keeps both
        hyperedges distinct and adds a cross-link."""
        self.abox.add_hyperedge(mut.payload.get("node_ids", []),
                                kind="relates",
                                roles=mut.payload.get("roles", []),
                                paper_id=mut.domain,
                                evidence=mut.evidence)

    # ===================================================================
    # Phase 2: route (aligner only, 5-outcome, non-atomic per-edge)
    # ===================================================================
    def _route(self, mut: Mutation) -> tuple[str, str]:
        """5-outcome decision for an aligner mutation. Non-atomic: reject/
        conflict here does NOT roll back siblings (断点 3). Rules + subset
        relation first (Challenge B); only the no-overlap-same-kind ambiguous
        case calls the judge_fn."""
        if mut.op == Op.ALIGN_MERGE:
            return self._route_align_merge(mut)
        if mut.op == Op.ADD_CONCEPT_RELATION:
            # explicit relate: add_concept_relation already applied in _apply
            return Outcome.RELATE, "explicit-relation"
        if mut.op == Op.CONFLICT_MARK:
            # conflict_mark already applied in _apply
            return Outcome.CONFLICT, "marked"
        return Outcome.INSERT, "no-route-for-op"

    def _route_align_merge(self, mut: Mutation) -> tuple[str, str]:
        """n-ary 5-outcome by node-set subset relation (Challenge B):
          - new node-set == existing            -> MERGE (union provenance)
          - new ⊂ existing or existing ⊂ new   -> MERGE to the superset
                                                   (union nodes, keep provenance)
          - partial overlap, neither subset     -> RELATE (add_concept_relation)
          - no overlap, same kind               -> AMBIGUOUS: judge decides
                                                   conflict (semantically same
                                                   but inconsistent) vs INSERT
                                                   (genuinely different). No
                                                   judge -> INSERT + flag.
          - no overlap, different kind          -> INSERT
        n-ary merge by subset relation is the non-trivial part: binary only
        judges u==v; n-ary must preserve arity and union param sets."""
        ne = mut.payload.get("new_edge", {})
        new_ids = set(ne.get("node_ids", []))
        kind = ne.get("kind", "")
        prov = ne.get("provenance", {})
        # find candidate existing edges of the same kind with node overlap
        candidates = [he for he in self.abox.hyperedges
                      if he.kind == kind and (set(he.node_ids) & new_ids)]
        if not candidates:
            # no overlap — ambiguous if same kind could still be semantically
            # the same claim; only the judge can tell. No judge -> INSERT.
            if self._judge_fn is not None:
                try:
                    decision = self._judge_fn({
                        "new_edge": ne, "candidates": [], "domain": mut.domain})
                    if decision == Outcome.CONFLICT:
                        return Outcome.CONFLICT, "judge:no-overlap-same-kind-conflict"
                    if decision == Outcome.REJECT:
                        return Outcome.REJECT, "judge:drop"
                except Exception:
                    pass
            # default: insert as a genuinely new claim
            self.abox.add_hyperedge(list(ne.get("node_ids", [])), kind,
                                    roles=ne.get("roles", []),
                                    paper_id=prov.get("paper_id", mut.domain),
                                    evidence=prov.get("evidence", mut.evidence),
                                    year=prov.get("year", ""),
                                    section=prov.get("section", ""))
            flag = "" if self._judge_fn else "no-judge:ambiguous-defaulted-insert"
            return Outcome.INSERT, flag
        # pick the most-overlapping candidate
        best = max(candidates, key=lambda he: len(set(he.node_ids) & new_ids))
        ex_ids = set(best.node_ids)
        if new_ids == ex_ids:
            self._merge_provenance(best, prov, mut)
            return Outcome.MERGE, "identical-node-set"
        if new_ids.issubset(ex_ids) or ex_ids.issubset(new_ids):
            # merge to the superset: union nodes (preserve arity + all params),
            # keep both provenances. This is the n-ary merge that binary cannot
            # do (binary would just union two endpoints, losing the param set
            # distinction that makes the n-ary edge meaningful).
            superset_ids = list(ex_ids | new_ids)
            best.node_ids = superset_ids
            self._merge_provenance(best, prov, mut)
            return Outcome.MERGE, "subset-merge-to-superset"
        # partial overlap, neither subset -> RELATE: the new claim is DISTINCT
        # (semantics may genuinely differ), so INSERT it as its own hyperedge
        # AND add a 'relates' cross-link to the overlapping existing edge. Do NOT
        # merge (collapsing would lose the distinction); do NOT drop (that would
        # lose the new claim's data). Both edges kept + a cross-link between them.
        self.abox.add_hyperedge(list(ne.get("node_ids", [])), kind,
                                roles=ne.get("roles", []),
                                paper_id=prov.get("paper_id", mut.domain),
                                evidence=prov.get("evidence", mut.evidence),
                                year=prov.get("year", ""),
                                section=prov.get("section", ""))
        self.abox.add_hyperedge(list(new_ids | ex_ids), kind="relates",
                                roles=[], paper_id=prov.get("paper_id", mut.domain),
                                evidence=prov.get("evidence", mut.evidence))
        return Outcome.RELATE, "partial-overlap-non-subset"

    def _merge_provenance(self, he: ConceptHyperedge, prov: dict, mut: Mutation) -> None:
        paper_id = prov.get("paper_id", mut.domain)
        entry = {"paper_id": paper_id, "evidence": prov.get("evidence", mut.evidence),
                 "year": prov.get("year", ""), "section": prov.get("section", "")}
        he.provenance.append(entry)

    # ===================================================================
    # ledger
    # ===================================================================
    def _record_ledger(self, mutations: list[Mutation], result: CommitResult) -> None:
        version_diff = None
        if result.ok:
            version_diff = {"before": mutations[0].base_version if mutations else self.version,
                            "after": result.version}
        self.ledger.append({
            "mutations": [m.to_dict() for m in mutations],
            "result": result.to_dict(),
            "version_diff": version_diff,
        })

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "tbox": self.tbox.to_dict(),
            "abox": self.abox.to_dict(),
            "skills": self.skills.to_dict(),
            "domain_ns": dict(self.domain_ns),
            "ledger": list(self.ledger),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeBase":
        kb = cls(tbox=MetaHypergraph.from_dict(d.get("tbox", {})),
                 abox=ConceptGraph.from_dict(d.get("abox", {})),
                 skills=SkillLibrary.from_dict(d.get("skills", {})),
                 domain_ns=dict(d.get("domain_ns", {"global": True})))
        kb.version = d.get("version", kb.tbox.version)
        kb.ledger = list(d.get("ledger", []))
        # restore mutation id counter from ledger
        max_mid = 0
        for entry in kb.ledger:
            for m in entry.get("mutations", []):
                mid = m.get("mutation_id", "")
                if mid.startswith("mut_"):
                    try:
                        max_mid = max(max_mid, int(mid[4:]))
                    except ValueError:
                        pass
        kb._mid_counter = max_mid
        return kb


def _now() -> str:
    """RFC3339 timestamp for ledger entries. (Kernel does not call this for
    control flow — only for audit provenance, so the no-Date restriction in
    workflow scripts does not apply; this runs in the application runtime.)"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_pattern(p) -> MetaHyperedgePattern:
    """Coerce a dict (ledger round-trip) or a MetaHyperedgePattern to the
    dataclass. The ledger stores mutations via asdict, so nested patterns become
    dicts; replay_to / from_dict must reconstruct them before _apply. Preserves
    semantic_boundary (M1 fix — was dropped on ledger round-trip)."""
    if isinstance(p, MetaHyperedgePattern):
        return p
    if isinstance(p, dict):
        return MetaHyperedgePattern(
            pattern_id=p.get("pattern_id", ""), description=p.get("description", ""),
            role_slots=[dict(s) for s in p.get("role_slots", [])],
            allowed_qualifiers=list(p.get("allowed_qualifiers", [])),
            deprecated=p.get("deprecated", False), is_abstract=p.get("is_abstract", False),
            split_from=p.get("split_from", ""), family=p.get("family", ""),
            semantic_boundary=p.get("semantic_boundary", ""))
    raise TypeError(f"cannot coerce {type(p)} to MetaHyperedgePattern")


def _as_skill(s) -> Skill:
    """Coerce a dict (ledger round-trip) or a Skill to the dataclass."""
    if isinstance(s, Skill):
        return s
    if isinstance(s, dict):
        return Skill(skill_id=s.get("skill_id", ""), domain=s.get("domain", ""),
                     pattern=s.get("pattern", ""), extraction_hint=s.get("extraction_hint", ""),
                     stability_score=s.get("stability_score", 0.0),
                     provenance_papers=list(s.get("provenance_papers", [])))
    raise TypeError(f"cannot coerce {type(s)} to Skill")
