# Decision: pattern_constraint trigger scope (narrowed after real-paper inspection)

## Context
Initial `infer_pattern_constraints` (commit above) used:
- AUTHORITY_FAMILIES = (definition, constitutive_law)
- CONSUMER_FAMILIES = (measure, claim)

## Real-paper finding (PPR_00180B90C8D8, PPR_08CE5406F5B2 — both granular-flow theory)
- dependency edges: fire correctly (11, then 30+)
- composition edges: fire correctly (4, then 6)
- **constraint edges: ZERO** on both papers.

Root cause (NOT a bug — honest): these are theory papers. The extractor
produces `influences` / `defines_*` / `constitutive_law` / `composed_of` edges.
It produces NO `measures` or `claim_relation` edges. So consumer-family edges
are absent → constraint never fires.

## Problem with the broad fallback
Broadening CONSUMER to include `dependency` would make constraint fire on the
SAME (definition, dependency) co-occurrence pairs as `depends_on` (reverse
direction) — redundant: two edges on one pair saying the same co-occurrence.
That adds noise, not information.

## Decision
Narrow constraint to a genuinely DISTINCT relation: a LAW constrains a relation
it governs. Definitions are already handled by depends_on (B depends_on the
entity A defines); constraint must NOT overlap that.

- AUTHORITY_FAMILIES = (constitutive_law,)   # only laws constrain (definitions just name)
- CONSUMER_FAMILIES  = (dependency, measure, claim)  # relations bounded by a law

Distinct signal: "constitutive_law X constrains dependency Y" — X fixes the
functional form among quantities that Y says depend on each other. This is NOT
the reverse of depends_on (depends_on is about DEFINITIONS, not laws). On paper
2 this fires: `constitutive_law_contact_force_theory` (a law) shares nodes
(stress/shear_rate) with `influences` (dependency) edges → the law constrains
those dependencies. Genuinely useful + non-redundant.

## Verification target
Re-run the 3 cached papers; confirm constraint edges now fire on law↔dependency
co-occurrence, and that no (definition, dependency) pair carries BOTH a
depends_on and a constrains edge (non-overlap check).
