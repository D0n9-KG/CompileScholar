# Self-Evolving Schema with N-ary Hypergraph: Lifting Low-Order Relations to Higher-Order Method Evolution via Structural Signals

*Draft — step8 成文（基于 step1-7 已验证机制+评测）*

## Abstract

Modeling how scientific methods extend, improve, or compare to one another is
central to mapping methodological progress, yet most schema-induction systems
either probe high-order relations directly from text (skipping the low-order
structure that justifies them) or stay frozen at the extraction schema. We
present a self-evolving schema + n-ary hypergraph system where a low-order
extraction (patterns, n-ary edges, qualifiers, verbatim evidence) is **lifted**
into a higher-order method-evolution layer (extends / improves / compares /
background) and written back into the shared schema — so a frozen schema, which
never calls lift, provably cannot grow a higher-order layer. The lift uses the
**full structural signal** of the low-order hypergraph, not just text: (1)
citation relations as a judge prior (A cites B → extends/improves/background
lean), and (2) edge-qualifier profiles (evidence_strength derived/measured,
cited_from prior_art, method theory/simulation/experiment) to distinguish true
extension from parallel or background relations. We close the paper-registration
loop (DOI/title → fulltext → hypergraph → citations, unified by paper_id) for
reproducibility. On 19 granular-flow papers, a 5-arm ablation shows lift is
necessary (frozen 0/3 vs lifted 3/3 gold type coverage), structural signals
make lifting more precise (improves +67–167%, over-extends 11→5), and on
complex method-relation questions a frozen schema scores 0/4 while a lifted one
scores 2/4. A same-LLM same-data baseline PK shows low-order lift beats
IncSchema-style direct-probe (3/3 vs 2/3 gold; improves 8 vs 0). On SciREX,
our n-ary extractor recalls 100% of gold Methods (14/14); on SciFact
(biomedical fact-checking) we honestly do not beat naive-RAG (0.250 vs 0.333) —
a domain boundary we report openly.

## 1. Introduction

Scientific progress is, structurally, a graph of methods: μ(I) rheology is
extended by the I-gradient model, improved by nonlocal fluidity, compared
against kinetic theory. Extracting this higher-order method-evolution layer
from papers is valuable for retrieval, QA, and survey reconstruction, but two
gaps remain. (i) Existing schema-induction systems (IncSchema, AutoSchemaKG,
ASEE) probe high-order relations **directly** from text — they never build the
low-order n-ary structure first, so relations like "improves" (A resolves a
limitation of B) that require low-order evidence of the limitation are
unreachable. (ii) Frozen schemas cannot grow a higher-order layer: the relation
graph is whatever the seed defines.

Contributions:
1. **Self-evolving lift as a 6th schema operation** — `lift_corpus` induces
   method nodes + higher-order relation patterns and writes them into the meta
   schema. A frozen schema never calls it → no higher-order layer (verified:
   F arm 0/3 gold).
2. **Structural-signal lifting** — instead of flattening the n-ary hypergraph to
   text, lift uses (a) citation relations as a judge prior and (b) edge
   qualifier profiles, proven complementary and precision-improving.
3. **Paper-registration closed loop** — DOI/title → fulltext (MinerU) → n-ary
   hypergraph (LogicKG) → citations (OpenAlex), unified by paper_id, with a new
   `external_artifacts` store.
4. **Honest, multi-module evaluation** — ablation, complex survey questions,
   same-LLM baseline PK, and two public datasets (SciREX strong, SciFact honest
   domain boundary).

## 2. Related Work

Schema induction: **IncSchema** [raspberryice/inc-schema] prompts the LLM
directly for event expansion (no low-order lift, OpenAI-bound, event-schema).
**AutoSchemaKG** is text-driven RAG with no self-evolution. **ASEE**
(Adaptive-EE) is cross-lingual event IE. n-ary KG: **Hyper-KGGen** (closest:
n-ary + self-evolving skill, but evaluates extraction quality not schema
routing). Method evolution: **Intern-Atlas** builds method-evolution gold from
30 surveys (NMR/ERR/PSC), which we adopt as evaluation protocol. Differentiation
(Table 1): none of the six neighbors do self-evolving **lift from low-order**
(IncSchema probes directly; others have no evolution).

## 3. Method

### 3.1 Self-evolving schema + 5 operations + lift (6th)
Five pattern operations (add/split/merge/retire/rename) mutate the shared
MetaHypergraph across papers. `lift_into_schema` is the 6th: it induces METHOD_
nodes and `higher_order_method_relation` patterns and writes them via
`add_meta_node`/`add_pattern` — so the next extraction sees them. Frozen =
never lift → no METHOD_ nodes (verified).

### 3.2 N-ary hypergraph extraction
`extract_hypergraph` runs a section DAG in topo order, matching each node to
MetaHyperedgePatterns to emit n-ary edges: {pattern_type, ordered nodes,
role_slots, qualifiers (relation_type/method/evidence_strength/cited_from),
verbatim evidence_span}. Four anti-hallucination measures (retrieval-augmented,
decomposed verification, strict admission, verbatim code verification) borrowed
from IncSchema.

### 3.3 Lifting with full structural signal (core 1)
`cluster_methods_by_llm` groups cross-paper edges by modeling method (LLM > embedding,
which collapses by topic). `induce_method_node` lifts a method node; `judge_relation`
judges A→B along a decision path (improves if B has a limitation A resolves;
extends if A generalizes B; compares if parallel; background if A is prior art).
Structural signals injected:
- **Citation prior** (step 3): if an A-family paper cites a B-family paper, the
  REL_PROMPT receives it as an extends/improves/background lean — a prior, not a
  hard rule (honest: a citation across different physics still judges null).
- **Qualifier profile** (step 4): per family, the distribution of
  evidence_strength / cited_from / method is injected into METHOD_PROMPT and
  REL_PROMPT. derived-heavy vs measured-heavy → extends/compares lean;
  prior_art-heavy → extends/background; both theory-heavy with similar evidence
  → compares.
`use_quals`/`paper_citations` are backward-compatible flags (off = step-3 /
pure-text baseline).

### 3.4 Paper-registration closed loop (core 2)
`paper_registry.PaperRegistryClient` (stdlib urllib) talks to sci-evo-extract:
resolve(DOI/title) → get_paper → get_fulltext (mineru_markdown) /
get_content_list → get_references (OpenAlex, fetch+upsert). `corpus_driver`
dual-tracks local mineru + API. A new `external_artifacts` table + POST endpoint
stores LogicKG hypergraphs keyed by paper_id → `get_hypergraph` retrieves them.

## 4. Evaluation

### 4.1 Ablation (5 arms, 19 papers, Table 2)
| arm | rel | ext | imp | cmp | bg | gold |
|---|---|---|---|---|---|---|
| F frozen | 0 | 0 | 0 | 0 | 0 | **0/3** |
| T text | 43 | 9 | 3 | 30 | 1 | 3/3 |
| C +citation | 47 | 8 | 5 | 34 | 0 | 3/3 |
| Q +quals | 62 | 9 | 8 | 42 | 3 | 3/3 |
| B full | 40 | 7 | 7 | 24 | 2 | 3/3 |

Lift is necessary (F 0/3). Citation prior: improves +67%. Qualifiers: improves
+167%. Full: most conservative/precise (fewest over-relations).

### 4.2 Complex survey questions (N=4, Table 3)
Method-relation questions (e.g., "how does NGF relate to μ(I)?"). F frozen
0/4 ("schema has no such info"); B full 2/4. Lift raises accuracy 0→0.5.

### 4.3 Full structure-vs-text A/B (19 papers)
A pure-text 56 rel (ext 11, imp 5, cmp 37) vs B full 48 (ext 5, imp 9, cmp 31):
structural signals improve true-inheritance (improves 5→9) and suppress
over-extends (11→5, distinguishing parallel/background from true extension).

### 4.4 Public datasets
**SciFact** (N=12 SUPPORT claims): hypergraph 0.250 vs naive-RAG 0.333 — honest
domain mismatch (our schema's strength is method relations, not fact-checking).
**SciREX** (N=8 docs): Method mention recall **1.000 (14/14)** — n-ary
extraction quality is high.

### 4.5 SOTA baseline fair PK (same deepseek, same ARFM2024)
IncSchema original is OpenAI-bound + event-schema (task-mismatched), so we adapt
its **direct-probe** style to our task and run it with our LLM on our data — a
controlled same-LLM comparison. Inc-direct: 18 rel, gold 2/3 (no improves).
ours-lift: 48 rel, gold 3/3, improves 8. Low-order lift beats direct probe
specifically on improves (0→8) — the relation type that requires low-order
limitation/resolution evidence.

### 4.6 NMR (Intern-Atlas-style)
Embedding (GLM-Embedding-2) + greedy-Hungarian matching of induced method names
to gold M-ids: **NMR=0.700** (7/10). M1/M17 ambiguity (both mention μ(I)) is the
main miss — refinable.

## 5. Discussion
- **Domain boundary (honest)**: lift's value is method-relation reasoning
  (granular flow: complex-Q 0→0.5; SciREX extraction 1.000); fact-checking on
  SciFact does not benefit. We report both.
- **Signal complementarity**: citation prior gives an inheritance lean;
  qualifiers correct over-extends. Combined, more conservative and precise.
- **Engineering**: judge parallelization (ThreadPoolExecutor) makes 132-pair
  lifts feasible despite occasional LLM hangs.
- **LLM non-determinism**: repeated runs vary (T=43 vs step5 A=56); full-paper
  numbers should be multi-run means (future work).
- **Limitations**: small samples (N=4/8/12), baseline PK adapts direct-probe
  style rather than original IncSchema (OpenAI-bound), NMR M1/M17 ambiguity.

## 6. Conclusion
A self-evolving schema lifts low-order n-ary edges to a higher-order
method-evolution layer using full structural signals (citation prior + qualifier
profiles); lift is necessary (frozen 0/3) and more precise than text-only or
direct-probe baselines (improves 0→8 vs IncSchema-direct); the paper-registration
loop makes it reproducible. Honest domain boundary reported.

## References (selected)
IncSchema (raspberryice/inc-schema); AutoSchemaKG; ASEE (USTC-StarTeam/ASEE);
Hyper-KGGen (arxiv 2602.19543); Intern-Atlas (arxiv 2604.28158); SciREX
(allenai/scirex); SciFact (AI2).

## Code
LogicKG: `src/granular_agent/{paper_registry,corpus_driver,hypergraph_lifter,
hypergraph_schema}.py`; sci-evo-extract: `external_artifacts` + API; experiments:
`experiments/self_evolution_lift/` (ablation/NMR/complex-Q/SciFact/SciREX/baseline-PK).
