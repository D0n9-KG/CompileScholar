# Topology Generalization Log (探索-修改-测试循环)

Goal: make self-evolving rich-topology schema hypergraph trigger rich-topology on
GENERAL (non-physics) domains, then validate downstream value on ResearchQA.

## Round 1 — diagnose why shared_nodes=0 (the upstream blocker)

**Explored**: ran topo_experiment.py --refresh (4 ResearchQA domains, extract_hypergraph
= our real method). Old result: shared_nodes 0/0/2/0, V0 topo 0/0/2/0. Fresh re-run:
shared 0/0/0/0, all variants 0.

**Inspected content (discipline 2)**: ML extracted 39 nodes but only 1 instance edge —
38 orphan nodes. Diagnostic direct-call: LLM emitted 11 rich n-ary edges (evidence
verbatim, semantically correct — attention O(L²) complexity, ProbSparse, Distilling,
Informer>DeepAR/ARIMA/Prophet) but ALL 11 rejected by validate() as
`no-matching-meta-pattern` → n_he=0 → center entities orphaned → 0 shared → 0 topology.

**Root cause = self-inflicted contradictions between EXTRACT_HG_PROMPT and the seed schema**
(see DECISION-validation-rejects-all-general-edges.md):
1. seed role_slots too narrow (influences=[source,target]) vs prompt encouraging
   output/parameter → edges with extra roles rejected by _match_variadic_set.
2. general-seed type system: ENTITY/METHOD/RESULT/NUMERIC independent, no common
   supertype → cross-type n-ary edge (METHOD whole + ENTITY components) fails
   is_subtype.
3. (found mid-fix) _match_variadic_set required min=1 for repeatable slots → adding
   optional roles made matching STRICTER not looser.
4. (found mid-fix) applies_in_regime enum = physics regimes → ML tag 'LSTF' rejected
   (12 rejections).
5. (found mid-fix) LLM uses `condition` role (biology) not in any seed → 10 rejections.

**Modified** (5 incremental fixes, each committed):
- expand general-seed role_slots to full prompt vocabulary (output/parameter/coefficient/
  cause/effect/object/condition).
- _match_variadic_set: repeatable slots OPTIONAL (min=0).
- THING root supertype; slot types=THING.
- applies_in_regime enum → free_text.
- influences/claim_relation gain `condition` role; prompt DIVERSE section strengthened
  with mis-classification examples + >70%-same-pattern self-check.

**Tested** (final):
| domain          | nodes | edges | shared | V0 topo | V1 topo |
|-----------------|-------|-------|--------|---------|---------|
| machine_learning| 34    | 10    | 2      | 2       | 4       |
| biology         | 42    | 15    | 3      | 2       | 6       |
| mathematics     | 41    | 16    | 5      | 7       | 10      |
| economics       | 32    | 12    | 3      | 3       | 6       |
| (was, all)      | —     | —     | 0,0,0,0| 0,0,0,0 | 0,0,0,0 |

**Content inspected (discipline 2) — PASS**:
- ML: captures Informer's full contribution — ProbSparse constitutive_law (u=c·lnLQ),
  Distilling composed_of (Conv1d/ELU/MaxPool), complexity O(L²)→O(LlogL), Informer
  outperforms DeepAR/ARIMA/Prophet with specific MSE%. Center entities (ProbSparse,
  Distilling, Informer) reused across influences/constitutive_law/composed_of/claim.
- biology: ASD reused across influences+defines; genes/genomic regions across
  influences+composed_of; MRI studies across measures+influences. Covers heritability,
  genetics, MRI, treatment adverse events, metformin.
- math: Coulomb matrix (composed_of+influences), extremely randomized trees+adaptive
  boosting (composed_of+influences, 0.12 eV/atom target), transparency
  (defines+composed_of). Rich cross-pattern.
- economics: corrugated structure, angles→drag, variance→success rate, random forests
  (measures+composed_of), actions/rewards (influences+composed_of).
- Evidence spans verbatim. Roles sensible. No garbage. 10-16 edges/paper (not
  over-triggered, <50).

**Granular regression — PASS**: smoke 53/53; single physics paper (PPR_0BFD9133C81F)
30 edges / 6 topology / 13 patterns. Physics seed role_slots untouched; global changes
(min=0, regime free-text) are loosening-only. 8-paper aggregate 33 not re-run (cost);
single-paper 6 is proportionally consistent.

**Judgment**: hypothesis confirmed — the blocker was validation rejecting all general-domain
n-ary edges, NOT a fundamental "rich-topology is physics-only" property. Once the seed
schema is made consistent with its own extraction prompt, general papers trigger
rich-topology with semantically meaningful shared central entities.

## Next phase — ResearchQA downstream validation
Topology now triggers on general domains. Per goal final-target: build hypergraph
retrieval for ResearchQA multi_hop QA, survey SOTA baselines (ResearchQA paper's 8
leading models + recent structured RAG + bare-LLM floor), compare section_coverage /
citation accuracy. Two ResearchQA benchmarks exist: 2607.11074 (citation-grounded,
6211 QA, 10 domains — our main) and 2509.00496 (survey-distilled, 21K, 75 fields).
eval_dataset.jsonl (6211 QA) already local.

## Phase 0 — full-paper extraction pipeline (done)
- S3 PDF fetch from assets.openpaper.ai: SSL cert hostname mismatch → disable verify.
- pymupdf parse + section split on header lines + per-section DAG → extract_hypergraph.
- slice_blocks is [start,end), so block_range must be [i, i+1].
- Result on 2 ML papers: 147 edges/topo83, 57 edges/topo24. Rich-topology triggers
  on FULL papers (not just 12k fragments). Content inspection GeoAI paper: captures
  LLMs(GPT-2/BERT) measures geometry, 73% accuracy claim, numeric estimation
  challenges — center entities reused across patterns. See RESEARCHQA-DOWNSTREAM-PLAN.md.

## Phase 1 — 3-arm retrieval harness (in progress)
Arms A (bare-LLM) / B (naive chunk RAG, bge-m3 top-k) / C (hypergraph retrieval).
Scoring = ResearchQA deterministic metrics (citation_precision, section_coverage,
citation_accuracy) with token-overlap matching (strict substring misses truncated
verbatim copies — a citation copied then truncated is not a substring of the alt).

### Phase 1b findings (honest, in progress)
- Initial C arm (Q-similarity top-k edges) gave CITE_PRECISION=0 with the OLD
  bge load failing (FlagEmbedding not installed) — fixed via sentence-transformers.
  After fix, C arm hit expected references on several Qs.
- **C2 topology-expand via instance-level SHARED-NODE did NOT help** (sometimes
  hurt cite_acc — shared-node pulls in same-section noise, not cross-section).
  This was the wrong substrate: instance shared-node ≠ schema rich-topology.
- **Fix**: build_hypergraph now persists schema_topo (depends_on/constrains/composes
  edges between PATTERN_TYPES) + pat_edges map; C2 walks schema-topology adjacency
  (pattern A's seed edge pulls in pattern B's edges when schema has A—dep/con/comp—>B).
  Cross-PATTERN adjacency is what spans sections. Testing now.

### Pending
- C1 vs C2 (schema-topology-expand) aggregate on multi_hop section_coverage.
- If C2 > C1 → rich-topology is doing real retrieval work (the contribution).
- If not → reconsider whether hypergraph retrieval's value is precision (cite_acc)
  not coverage, or whether multi_hop needs a different retrieval formulation.

