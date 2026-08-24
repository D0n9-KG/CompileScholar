"""Smoke +断点 test for alignment_agent.py (Step 4: EDC + 5-outcome aligner).

Verifies without real LLM:
- Define stage: fills Concept.definition (the field Step 2 added).
- Canonicalize: candidate groups found via _llm_align_batch (mocked), routed
  through kb.commit(align_merge) → 5-outcome (NOT direct merge_concepts).
- Same-domain filter (DecentMem): cross-domain concepts don't merge.
- Incremental: new_concept_ids aligns only the new ones.
- judge ≠ extraction model (separate llm_define / llm_judge).
- Every alignment write enters the ledger (kernel transaction).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from granular_agent.alignment_agent import AlignmentAgent, AlignmentReport
from granular_agent.knowledge_base import KnowledgeBase, Op, Role, Outcome
from granular_agent.hypergraph_schema import seed_meta_hypergraph
from granular_agent.concept_graph import Concept, _llm_align_batch

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


def _kb():
    return KnowledgeBase(tbox=seed_meta_hypergraph())


def _seed_concept(kb, cid, type_, surface, paper_id="p1", domain="granular"):
    c = Concept(concept_id=cid, type=type_)
    c.add_variant(surface, paper_id, "evidence", "2020", "")
    kb.abox.concepts[cid] = c
    kb.abox._paper_domain = getattr(kb.abox, "_paper_domain", {})  # not used; kb has map
    kb._paper_domain[paper_id] = domain
    return c


# mock _llm_align_batch to return a known group (so we don't hit LLM)
import granular_agent.alignment_agent as aa_mod
def _mock_align_batch(items, type_label, llm_fn):
    # group the first two as same-concept
    if len(items) >= 2:
        return [[items[0][0], items[1][0]]] + [[items[i][0]] for i in range(2, len(items))]
    return [[it[0]] for it in items]
aa_mod._llm_align_batch = _mock_align_batch


def _define_llm(prompt, max_tokens):
    # return a definition per concept_id in the prompt
    import re, json
    # extract the Concepts JSON array (between "Concepts:\n" and the next blank line)
    m = re.search(r'Concepts:\s*\n(\[.*?\])', prompt, re.DOTALL)
    items = []
    if m:
        try:
            items = json.loads(m.group(1))
        except Exception:
            pass
    defs = [{"concept_id": it.get("concept_id", ""),
             "definition": f"a {it.get('surface','')}: defined for alignment"} for it in items]
    return json.dumps({"definitions": defs})

def _judge_llm(prompt, max_tokens):
    return '{"groups": [[0,1]]}'


# ===========================================================================
# 1. Define stage fills Concept.definition
# ===========================================================================
kb = _kb()
_seed_concept(kb, "M1", "METHOD", "μ(I) rheology")
_seed_concept(kb, "M2", "METHOD", "local rheology")
agent = AlignmentAgent(kb, llm_define=_define_llm, llm_judge=_judge_llm,
                       domain_default="granular")
n = agent.define()
check("Define: fills Concept.definition (EDC's D)", n == 2)
check("Define: definition content correct",
      "μ(I) rheology" in kb.abox.concepts["M1"].definition)
# incremental: re-define skips already-defined
n2 = agent.define()
check("Define: incremental (skips already-defined)", n2 == 0)

# ===========================================================================
# 2. Canonicalize routes via align_merge 5-outcome (NOT direct merge)
# ===========================================================================
kb2 = _kb()
_seed_concept(kb2, "M1", "METHOD", "μ(I) rheology", "p1", "granular")
_seed_concept(kb2, "M2", "METHOD", "mu-I-rheology", "p2", "granular")  # same concept diff surface
agent2 = AlignmentAgent(kb2, llm_define=_define_llm, llm_judge=_judge_llm,
                        domain_default="granular")
report = agent2.align()
check("Canonicalize: candidate group found (mock returns [M1,M2])", report.n_candidates >= 1)
# the group routes through align_merge -> 5-outcome. same concept-set (no existing
# same_concept edge) -> INSERT (first time) or MERGE if identical node-set.
# Both M1+M2 as a new same_concept edge -> INSERT (no existing edge to merge into).
check("Canonicalize: routed via 5-outcome (insert/merge/relate/conflict)",
      report.n_insert + report.n_merge + report.n_relate + report.n_conflict >= 1)
# the alignment write must be in the ledger (kernel transaction, not direct merge)
check("Canonicalize: align_merge entry in ledger (kernel transaction)",
      any(m.get("op") == Op.ALIGN_MERGE for e in kb2.ledger for m in e.get("mutations", [])))

# ===========================================================================
# 3. Same-domain filter (DecentMem): cross-domain concepts don't merge
# ===========================================================================
kb3 = _kb()
_seed_concept(kb3, "M1g", "METHOD", "μ(I) rheology", "pg", "granular")
_seed_concept(kb3, "M1m", "METHOD", "μ(I) rheology", "pm", "ml")  # same surface, diff domain
agent3 = AlignmentAgent(kb3, llm_define=_define_llm, llm_judge=_judge_llm,
                         domain_default="granular")
# align in granular domain -> M1m (ml) should be filtered out
report3 = agent3.align(domain="granular")
# M1g alone (no same-domain peer) -> no group of >=2 -> no merge
check("Same-domain filter: cross-domain concept excluded from granular align",
      all("M1m" not in m.get("merged", []) for m in report3.merges))
# align in ml domain -> M1m alone
report3b = agent3.align(domain="ml")
check("Same-domain filter: granular concept excluded from ml align",
      all("M1g" not in m.get("merged", []) for m in report3b.merges))

# ===========================================================================
# 4. judge ≠ extraction model (separate fns)
# ===========================================================================
check("judge separated from define (≠ model contract)",
      agent.llm_define is _define_llm and agent.llm_judge is _judge_llm
      and _define_llm is not _judge_llm)

# ===========================================================================
# 5. Incremental: new_concept_ids aligns only the new ones
# ===========================================================================
kb5 = _kb()
_seed_concept(kb5, "M1", "METHOD", "μ(I) rheology", "p1", "granular")  # existing
_seed_concept(kb5, "M2", "METHOD", "mu-I-rheology", "p2", "granular")  # existing
_seed_concept(kb5, "M3", "METHOD", "nonlocal model", "p3", "granular")  # NEW
agent5 = AlignmentAgent(kb5, llm_define=_define_llm, llm_judge=_judge_llm,
                         domain_default="granular")
# align only M3 (new) -> only M3 in the candidate pool, no group of >=2 -> no merge
report5 = agent5.align(domain="granular", new_concept_ids=["M3"])
check("Incremental: align only new concept (M3) — no false merge with old pool",
      all("M1" not in m.get("merged", []) and "M2" not in m.get("merged", [])
          for m in report5.merges))

# ===========================================================================
# 6. align_new: full EDC (Define + Canonicalize) on new concepts
# ===========================================================================
kb6 = _kb()
_seed_concept(kb6, "M1", "METHOD", "μ(I) rheology", "p1", "granular")
_seed_concept(kb6, "M2", "METHOD", "mu-I-rheology", "p2", "granular")
agent6 = AlignmentAgent(kb6, llm_define=_define_llm, llm_judge=_judge_llm,
                       domain_default="granular")
report6 = agent6.align_new(domain="granular")
check("align_new: Define ran (definitions filled)", report6.n_defined >= 1)
check("align_new: Canonicalize ran (routed via 5-outcome)",
      report6.n_insert + report6.n_merge + report6.n_relate + report6.n_conflict >= 1)
check("align_new: report has merges detail (see content)",
      isinstance(report6.merges, list))

# ===========================================================================
# 7. ledger replayability (align_merge writes are in ledger)
# ===========================================================================
check("ledger has align_merge entries (replayable)",
      any(m.get("op") == Op.ALIGN_MERGE for e in kb6.ledger for m in e.get("mutations", [])))

print()
print(f"{'ALL PASS' if not fail else 'FAILURES: ' + str(fail)}  ({len(fail)} fail)")
sys.exit(1 if fail else 0)
