# Decisions

One line per decision, newest first. Full records live in `.research_tmp/docs_decisions/` (tracked since 2026-10-04);
those files are not rewritten — a later decision supersedes an earlier one here.

| Date | Decision | Why | Supersedes | Record |
|---|---|---|---|---|
| 2026-10-04 | Pre-register before any upgraded result: official CS2 judge primary, one main hypothesis per benchmark, mechanism-only void rules, dev 25–99 for confirmation, test once | the test was touched; dev 10–24 is selection-contaminated | — | `upgrade-1004/PREREG.md` |
| 2026-10-04 | W6 first: package the answer path behind byte-identical characterization tests, then fix evaluation (W1), then the cross-paper upgrade (W2–W5) | numbers must be reproducible before they are improved | — | `upgrade-1004/DESIGN-W1-W6.md` |
| 2026-10-04 | Upgrade plan: cross-paper skeleton → field-state compiler in the answer KB → consensus/contested → FieldQA decisive experiment; no general results matrix; stub papers outside the KB allowed (disclosed) | the state layer contributes Δ≈0 to CS2 for structural reasons (no paper resolution, families copied from survey taxonomies) | — | `upgrade-1004/UPGRADE-PLAN.md`, `DESIGN-CROSSPAPER.md` |
| 2026-10-04 | Rewrite only the unpushed history (strip >95 MB blobs and two credentials), fast-forward push; keep keys (never pushed) but remove them from code | 215 commits existed on one disk only; GitHub rejects >100 MB files | — | `upgrade-1004/AUDIT-WORKSPACE.md` |
| 2026-10-04 | Held-out survey test reported as the mean of two compiler runs | "state" and "state_v2" were the same system run twice | earlier per-column best | `PAPER-DRAFT-EXPERIMENTS-1003.md` §6.5 |
| 2026-10-03 | Freeze v9b for CS2 test (multi-source citation expansion, ~1,000-word budget) | dev 0.837, +0.094 over the same-model harness | v8 freeze | `cs2/FREEZE_CS2_TEST_1003_v9b.json` |
| 2026-10-03 | CS2 default answer length ~1,000 words | official scoring has no length term; AP drops with length | "no length limit" | handoff 10-03 |
| 2026-10-03 | Narrative v8: compile field state, not content; evidence = direct field-level test + attributable ablations | v7 (coverage) judged weaker; v6 claims untested | v6, v7 | `NARRATIVE-V8-1003.md` |
| 2026-10-03 | Warm-start KB must be question-blind (demand_core layer built from dev questions deleted) | building the KB from benchmark questions is leakage | — | `NARRATIVE-V8-1003.md` §4 |
| 2026-09-28 | Judge DeepSeek-V4.1-Flash instead of GLM-5.3 (all earlier GLM scores void) | GLM inflated scores and was 20–30× slower | GLM judge | `docs_decisions/EXPERIMENT-DESIGN-REV-0928.md` |

Earlier history (directions closed before September, schema / hypergraph era): see `MEMORY` index and
`.research_tmp/docs_decisions/` by date; `docs/archive/` holds the retired top-level docs.
