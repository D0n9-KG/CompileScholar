# Phase 5: Multi-Route Prior Induction - Research

**Researched:** 2026-04-02
**Domain:** bounded `L4` prior / anti-pattern induction from grouped `RouteState` bundles
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 5 is a bounded, auditable `L4` induction pilot rather than a claim that the `L4` layer is mature.
- The induction input is multiple `RouteState` objects; priors must not be generated directly from raw traces, single papers, or a single packet.
- `L1` and `L2` remain grounding constraints and evidence provenance, not the direct prior-generation layer.
- `AntiPatternCard` must ship in the same phase as `DecisionPriorCard` and remain a first-class typed object.
- `green` acceptance must still require support cluster evidence, counterexample search, held-out consistency, and reviewer metadata.
- Review stays file/CLI/report based for now; do not build a frontend review UI.
- Phase 5 must stay compatible with the replay / episode path, but must not overclaim training-ready exports.

### the agent's Discretion
- Exact clustering features and bucket heuristics for the first-pass multi-route induction path.
- Whether prior and anti-pattern generation live in one module or split builders.
- Exact JSON and Markdown artifact layout for the candidate registry and review summary.
- Whether replay consumes accepted cards directly in Phase 5 or only through an explicit pilot path.

### Deferred Ideas (OUT OF SCOPE)
- Declaring `L4` production-ready or cross-topic generalized.
- Emitting training-ready `DecisionEpisode` datasets.
- Reworking `L1` / `L2` / `L3` contracts just to make induction easier.
- Building a database-backed or frontend-heavy review product.
</user_constraints>

<research_summary>
## Summary

Phase 5 should add a candidate-induction layer above the existing replay / route-state pipeline, not replace the pipeline. The safest architecture is:

1. reuse the Phase 3 / Phase 4 `support / alternative / held_out` route-state package as the input boundary;
2. cluster grouped `RouteState` objects into reusable support sets;
3. turn each support cluster into one or more `DecisionPriorCard` candidates and `AntiPatternCard` candidates;
4. run held-out, counterexample, and review gates on those candidates;
5. export only auditable candidate / acceptance artifacts that later replay or episode builders can consume.

The current codebase already has the hard parts of the downstream contract: typed `DecisionPriorCard`, typed `AntiPatternCard`, strict green validation in `models.py`, route-state package grouping, replay failure records, and episode selection from prior / anti-pattern lists. The missing piece is batch induction.

Right now the system builds one prior card inside `HistoricalReplayCompiler` from one primary route plus support routes. That is a good Phase 4 pilot, but it is too narrow for `L4-01` because it conflates three roles:

- discovering clusters,
- generating candidate cards,
- selecting a single replay-time prior.

Phase 5 should separate those roles. A dedicated induction module can emit a candidate registry from grouped route states while keeping the existing single-card builder as the per-cluster primitive. That preserves test coverage and keeps Phase 5 bounded.

Anti-patterns should be induced from the same route-state neighborhood, but with different evidence emphasis: repeated blocking / not-now signals, replay failure ownership, weak comparison signals, and repeated held-out or counterexample contradictions. The project already has the right upstream surfaces for this in `RouteState.not_now_features`, `RouteComparisonCase`, and Phase 4 failure records. The first anti-pattern builder does not need novel data collection; it needs explicit object assembly.

The acceptance path should stay conservative. `green` still means:

- support cluster is sufficiently large,
- counterexample search was performed,
- held-out checks were run,
- reviewer metadata exists.

Everything else should remain `candidate` or `yellow`. That supports the user's intent: run the cross-layer loop end to end first, then learn where upstream layers are still weak.

**Primary recommendation:** implement a typed candidate-registry workflow that consumes grouped `RouteState` inputs, emits prior and anti-pattern candidates plus acceptance metadata, and only optionally feeds accepted cards back into replay / episode compilation.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/route_state_package.py` | Existing | Stable grouped `RouteState` input boundary | Already defines `support`, `alternative`, and `held_out` roles that Phase 5 wants to reuse directly |
| `backend/app/research_logic/decision_prior_builder.py` | Existing | Per-cluster prior builder with held-out and review logic | Best primitive to reuse instead of rewriting prior assembly |
| `backend/app/research_logic/models.py` | Existing | Typed `DecisionPriorCard` / `AntiPatternCard` / `DecisionEpisode` contracts | Already enforces green-gate rules and prevents schema drift |
| `backend/app/research_logic/historical_replay_compiler.py` | Existing | Current replay-time prior selection path | Shows the compatibility boundary Phase 5 must preserve |
| `backend/app/research_logic/replay_io.py` | Existing | JSON artifact and inspection writer | Best place to expose candidate registries and acceptance summaries without inventing a second I/O pattern |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `backend/app/research_logic/decision_episode_builder.py` | Existing | Downstream consumer of prior / anti-pattern lists | Use when accepted cards should influence episode quality and selection |
| `backend/scripts/run_route_state_package.py` | Existing | CLI for compiling grouped route-state bundles | Reuse for upstream package preparation during the pilot |
| `backend/scripts/run_replay_pilot.py` | Existing | CLI for replay bundle generation | Extend only if accepted Phase 5 cards should feed the replay path or emit reviewable candidate artifacts |
| `backend/tests/test_decision_prior_builder.py` | Existing | Prior-card regression boundary | Use for per-cluster induction and green-gate rules |
| `backend/tests/test_decision_episode_builder.py` | Existing | Downstream prior / anti-pattern consumption boundary | Use when accepted cards change episode behavior |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Replay integration boundary | Use when replay consumes accepted cards or candidate metadata |
| `backend/tests/test_route_state_package.py` | Existing | Grouped package contract boundary | Use when Phase 5 adds package-to-induction helpers |
| `backend/tests/test_research_logic_models.py` | Existing | Typed card contract validation | Use for anti-pattern and acceptance schema regressions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| grouped `RouteState` input | direct trace- or paper-level induction | Simpler short-term, but violates the locked "`L4` mainly from `L3`" decision |
| typed candidate registry | ad hoc Markdown-only notes | Faster to sketch, but weak for downstream reuse and test coverage |
| file/CLI review workflow | new frontend review UI | Better ergonomics later, but scope creep now |
| optional replay integration | immediate full dataset export | Jumps ahead to Phase 6 and overclaims maturity |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Candidate Registry Above The Existing Single-Card Builder
**What:** Add a `prior_induction` layer that groups route states and emits many candidates, while reusing `DecisionPriorBuilder` per cluster.
**When to use:** When Phase 5 needs multiple priors and anti-patterns rather than a single replay-time card.
**Why recommended:** It preserves existing tests and keeps the single-card builder as a stable primitive.

### Pattern 2: Route-State Package Roles As Induction Semantics
**What:** Treat `support`, `alternative`, and `held_out` package roles as the first induction contract, not just replay inputs.
**When to use:** When building clusters, counterexample searches, and held-out consistency checks.
**Why recommended:** The repo already validated that role split in Phase 3 / 4.

### Pattern 3: Anti-Pattern Induction From Failure And Not-Now Signals
**What:** Build anti-pattern candidates from repeated warning signals across `not_now_features`, replay failure ownership, route comparisons, and counterexamples.
**When to use:** When negative patterns are consistent enough to deserve a reusable warning card.
**Why recommended:** It keeps anti-patterns grounded in existing route-state and replay surfaces instead of free-text intuition.

### Pattern 4: Acceptance Metadata Separate From Discovery
**What:** Store support size, held-out results, counterexample search status, reviewer metadata, and quality flags as acceptance fields on candidates.
**When to use:** When deciding whether a candidate stays `draft`, `candidate`, `reviewed`, or `green`.
**Why recommended:** Discovery can stay exploratory while acceptance remains conservative and auditable.

### Anti-Patterns To Avoid
- Deriving priors directly from raw papers or traces without passing through `RouteState`.
- Treating one replay-generated prior as equivalent to multi-route induction.
- Hiding anti-patterns inside reviewer notes or untyped report text.
- Promoting candidate cards to `green` without held-out or reviewer metadata.
- Building a UI review surface before a stable file-based workflow exists.
</architecture_patterns>

<validation_architecture>
## Validation Architecture

Phase 5 should validate at three levels:

1. **contract-level unit tests**
   - prove candidate priors and anti-patterns satisfy typed model expectations,
   - keep green-gate rules strict for support, held-out, and review metadata.
2. **package-to-induction integration tests**
   - prove grouped `support / alternative / held_out` route-state bundles feed the inducer directly,
   - prevent the phase from inventing a second input format.
3. **replay / episode compatibility tests**
   - prove accepted cards can still feed replay and episode compilation,
   - keep non-green cards from silently looking training-ready.

The real-world audit should stay bounded to one existing route-state package slice plus one candidate review report. Cross-topic validation remains a later milestone because the immediate goal is to prove the acceptance loop and artifact contract, not generalization.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cluster input contract | a new bespoke YAML or notebook format | `RouteStatePackageCompilation.grouped_route_states()` | Phase 3 already standardized the grouped input boundary |
| Prior acceptance rules | a second quality schema | existing `HeldOutConsistency`, `ReviewMetadata`, and `CardQuality` | The contract already captures the required gates |
| Anti-pattern storage | free-form strings in reports | typed `AntiPatternCard` objects plus JSON registry | Keeps Phase 5 reusable by replay and episode builders |
| Review workflow | a custom DB or frontend | JSON registry + Markdown summary + CLI output | Matches the user's lightweight-review constraint |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: A cluster is really just one packet restated
**What goes wrong:** The code groups routes trivially and still emits one candidate per replay run.
**Why it happens:** It is easy to wrap the existing builder without changing the induction semantics.
**How to avoid:** Make the registry record cluster membership and require at least two distinct support-route ids for a reusable candidate.
**Warning signs:** Candidate ids differ, but every candidate draws from the same one-route support set.

### Pitfall 2: Anti-patterns become an afterthought
**What goes wrong:** Priors are typed and tested, but anti-patterns remain loose notes.
**Why it happens:** The current codebase has a prior builder but no anti-pattern builder.
**How to avoid:** Give anti-pattern induction its own explicit output path and schema regressions in Phase 5.
**Warning signs:** `DecisionEpisodeBuilder` accepts anti-patterns, but the pipeline never creates any.

### Pitfall 3: Green means "looks plausible"
**What goes wrong:** Candidate priors are promoted without enough held-out or review evidence.
**Why it happens:** The first successful cluster feels like proof of generality.
**How to avoid:** Reuse the spec rule that missing support, held-out, or counterexample search blocks `green`.
**Warning signs:** `review_status` stays `candidate`, but `quality_tier` becomes `green`.

### Pitfall 4: Review workflow drifts into product work
**What goes wrong:** The team spends the phase on interface design rather than acceptance mechanics.
**Why it happens:** Human review is real work, and UI seems like the natural next step.
**How to avoid:** Keep the first workflow to JSON + Markdown + CLI output only.
**Warning signs:** New frontend components appear before a stable candidate registry exists.

### Pitfall 5: Phase 5 accidentally becomes Phase 6
**What goes wrong:** Plans promise training-ready exports or cross-topic generalization.
**Why it happens:** The downstream `DecisionEpisode` contract already exists and is tempting to fill.
**How to avoid:** Limit Phase 5 to candidate induction and acceptance; let Phase 6 own audited export packaging.
**Warning signs:** Plan text starts promising "training-ready" samples or dataset production.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should prior and anti-pattern induction live in one module or two?**
   - What we know: both consume grouped route-state inputs but emphasize different evidence.
   - What's unclear: whether shared clustering logic plus separate builders is cleaner than one combined inducer.
   - Recommendation: use a shared clustering / registry module plus a dedicated anti-pattern assembly helper.

2. **Should replay consume accepted cards in Phase 5 or only the pilot path do so?**
   - What we know: `DecisionEpisodeBuilder` already accepts prior and anti-pattern lists.
   - What's unclear: whether `HistoricalReplayCompiler` should switch from building one fresh prior to selecting accepted candidates.
   - Recommendation: keep replay backward compatible, but add an optional accepted-card injection path.

3. **Should current runtime route-state assets be promoted before the first induction pilot?**
   - What we know: grouped runtime assets already exist under `tmp/`.
   - What's unclear: whether lack of committed assets is a blocker for Phase 5.
   - Recommendation: do not block the pilot on asset promotion; record the risk and reuse the existing bounded slice first.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`
- `backend/app/research_logic/models.py`
- `backend/app/research_logic/decision_prior_builder.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/route_state_package.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/decision_episode_builder.py`

### Secondary (MEDIUM confidence)
- `backend/scripts/run_route_state_package.py`
- `backend/scripts/run_replay_pilot.py`
- `backend/tests/test_decision_prior_builder.py`
- `backend/tests/test_decision_episode_builder.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_route_state_package.py`
- `backend/tests/test_research_logic_models.py`
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md`
- `docs/replay/reports/phase4-l2-surgical-delta.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: multi-route prior and anti-pattern induction
- Ecosystem: `research_logic`, route-state packages, replay bundles, file-based review artifacts
- Patterns: candidate registry, shared clustering, conservative acceptance gates
- Pitfalls: single-route restatement, anti-pattern omission, premature green promotion, Phase 6 scope creep

**Confidence breakdown:**
- Standard stack: HIGH
- Architecture: HIGH
- Acceptance workflow: MEDIUM
- Cross-topic generalization: LOW

**Research date:** 2026-04-02
**Valid until:** 2026-05-02
</metadata>

---

*Phase: 05-multi-route-prior-induction*
*Research completed: 2026-04-02*
*Ready for planning: yes*
