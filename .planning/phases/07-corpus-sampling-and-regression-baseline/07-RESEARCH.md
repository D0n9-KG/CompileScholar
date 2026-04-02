# Phase 7: Corpus Sampling And Regression Baseline - Research

**Researched:** 2026-04-03
**Domain:** filesystem-first corpus sampling, reproducible batch manifests, and corpus-health separation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 7 must treat the shared corpus filesystem inventory as the source of truth for sampling eligibility.
- Neo4j ingestion status must be tracked as metadata on sampled papers, not used as the primary eligibility gate.
- Every iteration cycle must contain one fixed regression set of `10` papers and one random exploration set of `5` papers.
- Phase 7 must stay file-based and auditable, consistent with the repo's existing replay / package / export bundle style.
- Phase 7 must separate corpus-health failures from model-quality failures and from downstream environment gaps.
- Phase 7 stops at the sampling baseline and reporting loop. It does not perform `L2` fixes and does not assemble bounded multi-paper packets.
- `L3/L4` remain topic-bounded multi-paper work for later phases; Phase 7 samples only single-paper items.

### the agent's Discretion
- Exact module and CLI names for the Phase 7 implementation
- Exact local bundle filenames as long as manifest / summary / inspection style stays consistent
- The documented heuristic for the first fixed regression set, as long as it is intentionally stable and auditable
- Whether Neo4j enrichment is mandatory or best-effort at runtime, as long as the batch artifact records the outcome explicitly

### Deferred Ideas (OUT OF SCOPE)
- Running the full `1000+` corpus end to end in Phase 7
- Letting arbitrary random samples flow directly into `L3/L4`
- Building a frontend corpus-batch management UI
- Combining corpus-health cleanup and `L2` quality repair into one phase
</user_constraints>

<research_summary>
## Summary

Phase 7 should create a dedicated sampling layer above the shared corpus filesystem, not reuse Neo4j paper listing as the batch gate. The most important findings are:

1. The shared corpus root really is the only complete candidate pool available right now.
2. The corpus layout is heterogeneous but patterned enough to inventory deterministically.
3. Recursive scans over the UNC root already surface real `DirectoryNotFound` failures, so corpus-health reporting is not hypothetical.
4. Existing repo workflows already prefer `bundle_manifest.json` plus `summary` and `inspection` payloads written from backend CLIs.

The observed corpus shape supports a stable Phase 7 design:

- top-level directories are named like `1746_Random Packings of Frictionless Particles`
- markdown derivatives often live under nested directories such as `.../1746_Random_Packings_of_Frictionless_Particles/1746_Random_Packings_of_Frictionless_Particles.md`
- many plain-text files also exist under a shared `output/txt/` directory
- recursive enumeration hits broken nested paths on the share, which means the inventory walker must capture traversal errors instead of treating them as invisible drops

That leads to the recommended architecture:

1. Build a typed filesystem inventory from the shared corpus root.
2. Derive a stable `corpus_paper_id` from the leading numeric prefix when present, with a path-hash fallback.
3. Preserve both `md_path` and `txt_path` when they exist, and choose a deterministic `preferred_source_path` for downstream work.
4. Record unreadable paths, missing nested directories, and other traversal failures as explicit corpus-health artifacts.
5. Enrich entries with Neo4j metadata only after the filesystem inventory exists, and never let graph absence remove a file-backed paper from eligibility.
6. Write one local sampling bundle containing the fixed set, the random set, the health failures, and the exact seed / filters used.
7. Keep committed docs free of machine-specific UNC paths by storing only `corpus_paper_id` and corpus-relative references in repo artifacts.

**Primary recommendation:** implement a new `backend/app/research_logic/corpus_sampling.py` module plus Phase 7 bundle writers in `backend/app/research_logic/replay_io.py`, then expose the workflow through a standalone CLI such as `backend/scripts/run_corpus_sampling_baseline.py`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/replay_io.py` | Existing | Bundle manifest / summary / inspection writer pattern | Best existing fit for Phase 7 file-based output contracts |
| `backend/scripts/run_replay_pilot.py` | Existing | Backend CLI structure with JSON summary output | Good pattern for a new Phase 7 sampling CLI |
| `backend/scripts/run_route_state_package.py` | Existing | Dedicated single-purpose compilation CLI | Reinforces the repo convention of one clear workflow per backend script |
| `backend/scripts/export_decision_episode_pilot.py` | Existing | Separate audited export CLI with dedicated bundle directory | Shows how to keep new workflow surfaces explicit instead of overloading old scripts |
| `backend/app/ingest/paper_identity.py` | Existing | Stable `paper_id` derivation from DOI or `md_path` | Important reference for aligning Phase 7 item identity with later ingest |
| `backend/app/graph/neo4j_client.py` | Existing | Current `Paper` fields include `paper_id`, `paper_source`, `source_md_path`, and `ingested` | Gives Phase 7 the exact graph metadata fields to mirror without using Neo4j as the gate |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `backend/tests/test_replay_io.py` | Existing | Regression boundary for manifest / summary / inspection bundle shapes | Extend when adding Phase 7 bundle writers |
| `docs/replay/pilot_packets/README.md` | Existing | Rule that committed docs keep machine-specific corpus paths out of the repo | Use to decide what belongs in committed fixed-set docs versus local runtime bundles |
| `eval_quality.py` | Existing | Batch-style quality-reporting precedent | Useful inspiration for aggregate counts, while keeping Phase 7 narrower and more auditable |
| `.planning/codebase/TESTING.md` | Existing | Testing guidance and current replay-on-real-corpus gap | Confirms Phase 7 should add focused backend tests plus one real corpus pilot |
| shared corpus root | External | Real candidate pool for filesystem inventory | Use as the primary sample source and the only true eligibility boundary |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| filesystem-first inventory | Neo4j `MATCH (p:Paper)` as the sample pool | Simpler if ingestion were complete, but wrong for the current milestone because many corpus items are not yet in the graph |
| dedicated Phase 7 CLI | extending `eval_quality.py` or `run_replay_pilot.py` | Faster short-term, but mixes sampling-baseline work with later quality or replay logic |
| committed fixed-set manifest with relative refs | committed UNC absolute paths | Easier locally, but violates the repo's existing rule against hardcoding machine-specific corpus locations |
| explicit corpus-health artifacts | silently skipping traversal failures | Cleaner-looking counts, but destroys the main signal Phase 7 is supposed to surface |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Filesystem Inventory First, Graph Metadata Second
**What:** Build eligibility from the corpus root itself, then optionally enrich entries with Neo4j metadata such as `paper_id`, `source_md_path`, and `ingested`.
**When to use:** When the file share is larger and messier than the graph, and you need sampling to reflect reality rather than current ingest coverage.
**Why recommended:** It matches the user's decision and preserves papers that exist on disk but have not yet been ingested.

### Pattern 2: Stable Corpus Item Identity From Numeric Prefix Plus Relative Ref
**What:** Use the leading integer prefix before the first underscore as the primary `corpus_paper_id` when it exists, and fall back to a hash of the corpus-relative path when it does not.
**When to use:** When the share contains both nested markdown directories and flat `txt` files that refer to the same paper family.
**Why recommended:** The observed corpus naming pattern is stable enough to unify `.md` and `.txt` siblings without forcing Phase 7 to fully parse paper metadata.

### Pattern 3: Dual Path Preservation With Deterministic Preferred Source
**What:** Store `md_path`, `txt_path`, and `preferred_source_path`, where markdown wins if present and readable, and text is the fallback.
**When to use:** When downstream phases need a stable source pointer but the corpus contains multiple path variants per paper.
**Why recommended:** It preserves auditability and keeps Phase 8 free to use richer markdown when available without losing plain-text fallback.

### Pattern 4: Committed Fixed Set, Local Runtime Bundle
**What:** Commit the intentional fixed regression set as repo docs using ids and corpus-relative refs only, while runtime sampling bundles under `tmp/` may record absolute source paths and environment-specific metadata.
**When to use:** When the workflow needs both auditability in git and runnable local provenance on a specific machine.
**Why recommended:** It matches the repo's `docs/replay/pilot_packets/` design rule and avoids hardcoding the UNC share into committed assets.

### Pattern 5: Corpus-Health As A Separate Output Surface
**What:** Write missing directories, traversal errors, unreadable files, and orphaned path variants into a dedicated `corpus_health_failures.json` plus summary counts.
**When to use:** Whenever the shared corpus itself is unstable enough that path failures could masquerade as downstream extraction failures.
**Why recommended:** The real UNC scans already produced `DirectoryNotFound` errors, so this separation is required, not optional.

### Pattern 6: Best-Effort Neo4j Enrichment
**What:** Query Neo4j by `source_md_path` for markdown-backed entries, populate graph metadata when available, and record `neo4j_lookup_status` in the bundle even if the graph is unavailable.
**When to use:** When graph coverage is useful for planning later phases but cannot be a blocker for Phase 7 sampling.
**Why recommended:** It preserves the user's choice that Neo4j is metadata only, while still making graph readiness visible for later `L2` and packet work.

### Anti-Patterns To Avoid
- Sampling only from `MATCH (p:Paper) WHERE coalesce(p.ingested, false) = true`.
- Writing committed docs that include `\\192.168.199.138\...` absolute corpus paths.
- Treating recursive scan failures as harmless noise and excluding them from reports.
- Mixing corpus-health failures into `L2` quality counts or replay failure buckets.
- Letting the random batch reselect ids already locked into the fixed regression set.
</architecture_patterns>

<validation_architecture>
## Validation Architecture

Phase 7 should validate at four levels:

1. **inventory and batch-selection contract tests**
   - prove `.md` and `.txt` variants can collapse into one stable corpus item,
   - prove markdown is preferred when both formats are readable,
   - prove unreadable or broken entries are reported as corpus-health failures,
   - prove the random batch excludes fixed-set ids and uses a reproducible seed.

2. **bundle writer contract tests**
   - prove Phase 7 writes `bundle_manifest.json`, `sampling_summary.json`, `sampling_inspection.json`, `outputs/fixed_regression_batch.json`, `outputs/random_exploration_batch.json`, and `outputs/corpus_health_failures.json`,
   - prove the manifest stores counts, seed, mode, and file map in the same auditable style as replay / review / export bundles.

3. **CLI smoke tests**
   - prove the Phase 7 CLI can scan a fixture corpus, load the committed fixed set, and emit a local bundle,
   - prove missing fixed ids, empty eligible pools, and Neo4j lookup failures produce explicit results instead of hidden drops.

4. **real shared-corpus pilot audit**
   - run the CLI against the actual UNC root,
   - confirm the fixed set size is `10`, random set size is `5`,
   - confirm the bundle surfaces real corpus-health failures separately,
   - confirm committed docs stay relative while the runtime bundle keeps absolute local provenance.

The phase does not need to prove `L2` quality yet. It only needs to prove the sampling baseline is reproducible, auditable, and honest about corpus-health messiness.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 7 output contract | a free-form pile of JSON files with no manifest | bundle manifest plus summary / inspection pattern in `replay_io.py` | Keeps sampling artifacts compatible with the repo's existing workflow style |
| corpus item identity | ad hoc title matching or display-name parsing only | numeric-prefix-derived `corpus_paper_id` plus corpus-relative refs | More stable under punctuation, spaces, and Chinese titles |
| graph readiness signal | a hard Neo4j dependency that blocks sampling | best-effort enrichment recorded in bundle metadata | Matches the user's decision that Neo4j is metadata only |
| fixed regression documentation | a local-only text file outside the repo | committed JSON + notes under `docs/replay/corpus_sampling/` | Makes the fixed set intentional and reviewable |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: The corpus walker loses failures during traversal
**What goes wrong:** Missing nested directories never make it into the output, so the operator thinks the corpus is cleaner than it really is.
**Why it happens:** Simple recursion helpers often stop or silently skip when the UNC share contains broken subpaths.
**How to avoid:** Use a traversal method that captures errors into explicit corpus-health records.
**Warning signs:** The summary shows zero health failures even though ad hoc scans already produced `DirectoryNotFound`.

### Pitfall 2: `.md` and `.txt` variants become separate papers
**What goes wrong:** The same paper can be sampled twice or excluded inconsistently because markdown and text were inventoried independently.
**Why it happens:** The share mixes nested markdown paths with a flat `txt/` directory.
**How to avoid:** Normalize to one `corpus_paper_id` and preserve both path variants on the same inventory entry.
**Warning signs:** Fixed and random sets contain visually identical titles with different path kinds.

### Pitfall 3: Absolute share paths leak into committed docs
**What goes wrong:** Git-tracked docs capture the machine-specific UNC root and make the repo less portable.
**Why it happens:** Runtime provenance is useful, so it is tempting to copy it directly into committed manifests.
**How to avoid:** Keep committed manifests on ids and corpus-relative refs only, and reserve absolute paths for local bundles in `tmp/`.
**Warning signs:** `docs/replay/corpus_sampling/*.json` contains `\\192.168.199.138`.

### Pitfall 4: Neo4j availability distorts eligibility
**What goes wrong:** Papers that exist on the share disappear from the sample pool because the graph is incomplete or temporarily unavailable.
**Why it happens:** Existing graph helpers often filter on `coalesce(p.ingested, false) = true`.
**How to avoid:** Perform graph enrichment after inventory creation and mark lookup status explicitly.
**Warning signs:** The eligible count suddenly collapses to the number of currently ingested papers.

### Pitfall 5: Corpus-health and downstream-unavailable get merged
**What goes wrong:** A missing directory, an unreadable markdown file, and a paper that simply lacks graph metadata all show up under the same failure bucket.
**Why it happens:** All three look like "not ready" if the report is too coarse.
**How to avoid:** Separate `corpus_health_failures`, `eligible_and_selected`, and `selected_but_downstream_unavailable`.
**Warning signs:** Operators cannot tell whether to fix the share, the ingest layer, or the evaluator.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should the fixed regression set optimize for topical continuity or filename/path diversity?**
   - What we know: Phase 7 is about the sampling baseline, not `L2` quality yet.
   - Recommendation: choose a deliberate mix that includes both stable jamming references and a few path-shape edge cases.

2. **Should markdown-less papers still be eligible for the random batch?**
   - What we know: the user chose filesystem-first eligibility, and many items exist only as `.txt`.
   - Recommendation: yes, but mark `preferred_source_kind = txt` and keep downstream-unavailable reasons explicit.

3. **Should Neo4j lookup failure fail the CLI?**
   - What we know: Neo4j is metadata only in Phase 7.
   - Recommendation: no. The CLI should continue and record `neo4j_lookup_status = unavailable`.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md`
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md`
- `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md`
- `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md`
- `backend/app/research_logic/replay_io.py`
- `backend/app/graph/neo4j_client.py`
- `backend/app/ingest/paper_identity.py`
- `backend/scripts/run_replay_pilot.py`
- `backend/scripts/run_route_state_package.py`
- `backend/scripts/export_decision_episode_pilot.py`
- `backend/tests/test_replay_io.py`
- `docs/replay/pilot_packets/README.md`

### Observed Runtime Evidence (HIGH confidence)
- Shared corpus directories include names such as `1746_Random Packings of Frictionless Particles`
- Markdown derivatives exist under nested directories like `.../1746_Random_Packings_of_Frictionless_Particles/1746_Random_Packings_of_Frictionless_Particles.md`
- Text derivatives exist under `output/txt/*.txt`
- Recursive UNC scans currently emit `DirectoryNotFound` errors for some nested subpaths

### Secondary (MEDIUM confidence)
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/TESTING.md`
- `.planning/codebase/INTEGRATIONS.md`
- `eval_quality.py`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: filesystem-backed corpus inventory and reproducible sampling bundles
- Ecosystem: `research_logic`, backend CLIs, Neo4j metadata enrichment, local `tmp/` artifact bundles
- Patterns: manifest + summary + inspection outputs, committed-relative docs plus local absolute provenance, corpus-health separation
- Pitfalls: broken UNC subpaths, duplicated `.md` / `.txt` identities, graph-over-gate mistakes, path leakage into git

**Confidence breakdown:**
- Inventory source-of-truth choice: HIGH
- Bundle pattern reuse: HIGH
- Corpus shape observations: HIGH
- First fixed-set curation heuristic: MEDIUM
- Optional Neo4j enrichment approach: HIGH
</metadata>
