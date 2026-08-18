# DECISION: validation rejects ~100% of general-domain edges (root cause of 0 rich-topology)

**Date**: 2026-08-14
**Status**: change decided, pre-change numbers recorded below

## Pre-change numbers (the disease)
topo_experiment.py --refresh, 4 ResearchQA domains, extract_hypergraph (our real
method, NOT bare LLM), seed_meta_hypergraph_general:

| domain          | nodes | instance edges | shared nodes | V0 topo | V1 topo |
|-----------------|-------|----------------|--------------|---------|---------|
| machine_learning| 39    | 1              | 0            | 0       | 0       |
| biology         | 53    | 2              | 0            | 0       | 0       |
| mathematics     | 35    | 5              | 0            | 0       | 0       |
| economics       | 30    | 8              | 0            | 0       | 0       |

ML diagnostic (direct call): LLM emitted 11 rich n-ary edges, evidence verbatim and
semantically correct (attention O(L^2) complexity, ProbSparse, Self-attention
Distilling, Informer outperforms DeepAR/ARIMA/Prophet). **ALL 11 rejected by
validate() with reason `no-matching-meta-pattern`.** n_he=0. The center entities
(Informer, ProbSparse, the metrics) ARE extracted as nodes but orphaned because no
edge survives → 0 shared nodes → 0 rich-topology.

## Root cause (two self-inflicted contradictions between prompt and seed schema)

### Disease 1: seed role_slots too narrow vs the prompt's encouraged role vocabulary
The EXTRACT_HG_PROMPT explicitly tells the LLM to use a rich role set:
`output/input, cause/effect, subject/object, from/to, whole/component,
source/target, instrument/object, exponent/parameter/coefficient`.
But each seed pattern declares only 2 roles:
- `influences` → [source, target]    (LLM uses [source, target, output] → reject)
- `constitutive_law` → [output, input] (LLM uses [output, input, parameter] → reject)
- `claim_relation` → [from, to]        (LLM uses [from, to, parameter] → reject)
- `defines` → [subject, definition]    (LLM uses [subject, object] → reject)
`_match_variadic_set` (hypergraph_schema.py:447) rejects any role not declared in
the pattern → the whole edge is dropped.

### Disease 2: general-seed type system has 4 independent top-level types, no root
seed_meta_hypergraph_general declares ENTITY, METHOD, RESULT, NUMERIC as FOUR
INDEPENDENT types (no subclass_of relations). An n-ary edge mixing a METHOD whole
with ENTITY components (e.g. "Self-attention Distilling"[METHOD] composed_of
[Conv1d, ELU, MaxPool]) fails `is_subtype(METHOD, ENTITY)` (line 580) → reject.
General papers routinely mix method+entity+result in one relation; physics papers
don't (single-typed quantities), so the physics seed+domain escapes this.

### Why the evolution loop doesn't self-heal
run_evolution_loop fires on the 11 failures and DOES add new patterns (3
subclass_of IS-A edges: complexity_scaling, performance_comparison,
architecture_component). But new patterns inherit the parent's rigid role_slots
(see `_role_sig` constraint in split), so re-validation still rejects → n_he stays 0.
The evolution loop spins uselessly.

## The change (this commit)
Make the seed schema CONSISTENT with its own prompt:
1. **Expand seed role_slots** (both seed_meta_hypergraph and
   seed_meta_hypergraph_general) to declare the full role vocabulary the prompt
   encourages per family (output/parameter/coefficient/exponent on laws,
   output/cause/effect on influences, object on defines, parameter on claims).
   All repeatable where multi-node. This cures disease 1 at the source.
2. **Add a THING root supertype** in seed_meta_hypergraph_general and make
   ENTITY/METHOD/RESULT/NUMERIC subclass_of THING; set general-seed slot types to
   THING. A METHOD whole with ENTITY components now passes is_subtype. Cures disease 2
   without losing type info (types still on nodes; schema just permits cross-type
   n-ary edges). Physics seed untouched (it works).

## Why not relax _match_variadic_set to ignore unknown roles (alternative)
Considered. Would tolerate garbage roles too and weakens the structural-mismatch →
evolution-trigger signal. First try making the seed declare what the prompt asks
for (the consistent fix). If residual rejections remain, revisit per-edge.

## What "success" looks like after the change
- ML/biology/economics instance edges jump from {1,2,8} toward the 5-12 the prompt
  targets (the LLM IS emitting them; they're just being dropped).
- shared_nodes > 0 on >=3/4 domains (central entities now wired into surviving
  edges → reused across pattern_types).
- V0/V1 rich-topology > 0 on >=3/4 domains.
- Content inspection (discipline 2): surviving edges semantically correct, central
  entities reused meaningfully (not accidental name collision).
- granular regression: physics seed+domain still 33 topo edges, smoke 53/53.
