# legacy/

Code superseded by `src/compilescholar` (upgrade W6, 2026-10-04). Moved here with `git mv`, so history is intact.
Nothing in this folder is imported by the package or by `tests/`, and it is not packaged.

**Reproducing a reported number:** check out the tag the number was produced at and run the old scripts from their
original paths (they use paths relative to `.research_tmp/experiments/benchmarks/`, which only exist at those tags):

| Result | Tag | Entry point at that tag |
|---|---|---|
| CS2 test, frozen v9b (ours 0.829; archived arms; harness 0.744; GPTR 0.736) | `freeze-cs2-v9b` (code) / `cs2-test-v9b-final` (code + scores + stats) | `cs2/run_test_final.sh`, `cs2/paired_stats.py` |
| CS2 dev ablations (same 15 questions, offset 10–24) | `cs2-test-v9b-final` | `cs2/run_vnext.py --tag dev10_*`, `cs2/cmp15.py` |
| DSB (ours 0.310 / harness_v2 0.241 / B0 0.369) | `cs2-test-v9b-final` | `cs2/run_vnext_dsb.py`, `cs2/dsb_harness.py`, `review_1002/p6/judge_nuggets.py` |
| Multi-108 v2 (pre-registered / strict F1) | `cs2-test-v9b-final` | `scholarqa_multi/run_vnext_multi.py`, `score_multi_f1.py`, `score_multi_strict.py` |
| Held-out survey field test (18 surveys) | `cs2-test-v9b-final` | `cs2/base_kb_build/heldout_field_eval.py`, `summarize_field_eval.py` |

The new package reproduces the answer path, scoring and the moved components byte for byte; see
`tests/test_characterize_*.py` (goldens generated from this code before the move).

## What moved where

| Old file (under `.research_tmp/experiments/benchmarks/`) | Replacement in `src/compilescholar/` | Status here |
|---|---|---|
| `_shared/tools/answer_pipeline.py` | `answer/pipeline.py`, `answer/prompts/*.txt`, `kb/store.py` | superseded |
| `_shared/tools/refgraph.py` | `sources/refgraph.py` | superseded |
| `_shared/tools/cutoff.py` | `core/cutoff.py` | superseded |
| `_shared/mcp/cc_compat_proxy.py`, `retrieval_mcp.py` | `baselines/harness/proxy.py`, `mcp_server.py` | superseded |
| `cs2/harness_arm_run.py`, `dsb_harness.py`, `harness_to_judge.py` | `baselines/harness/runner.py` | superseded |
| `cs2/direct_judge.py` | `eval/cs2/judge.py` (JudgeAdapter) | superseded |
| `cs2/cs2_scoring.py`, `paired_stats.py` | `eval/cs2/scoring.py`, `eval/stats.py` | superseded |
| `cs2/memorized_adapter.py` | `baselines/memorized.py` | superseded |
| `cs2/cs2_gptr.py` (adapter part) | `baselines/gptr.py` | runner part still only here (see below) |
| `cs2/base_kb_build/heldout_field_eval.py` | `eval/field.py` | superseded |
| `src/kb_compiler/views/field_state.py` (not moved; still in src) | `compile/state/field_state.py` | old copy kept in src until kb_compiler is retired |
| `review_1002/p6/judge_nuggets.py`, `analyze.py` (not moved; outside benchmarks) | `eval/dsb.py` | old copies kept in place (judged/ outputs live next to them) |
| `cs2/run_vnext.py`, `run_vnext_dsb.py`, `run_test_final.sh` | `compilescholar answer/judge/score/verify` (`cli.py`) + `configs/` | superseded |
| `scholarqa_multi/run_vnext_multi.py` | not yet ported (crashes on re-run: MultiKB skips the parent constructor; W1-11) | here only |

## Only here (not ported)

- **Baseline runners whose behaviour depends on legacy retrieval**: `cs2/cs2_gptr.py` (runner), `cs2/cs2_storm.py`,
  `cs2/dsb_storm.py`, `cs2/cs2_paperqa.py`, and `_shared/tools/external_tools.py` (their retrieval path: tiered
  SearchService with a 15 s slot and a 600 s circuit breaker). Porting them onto `compilescholar.sources` would change
  the arms; W1-8 decides whether they are re-run on the unified search path.
- **KB v2 build chain** (`cs2/base_kb_build/build_kb_v2.py`, `embed_kb_v2.py`, `build_state_v2.py`, `merge_state_v2.py`
  and the in-place rewrite scripts in `cs2/`: `normalize_entities.py`, `resolve_citations.py`, `translate_absences_v2.py`,
  `classify_paraphrases.py`, `recover_deep_records.py`, `backfill_*.py`): per decision 9 the KB is published as a frozen
  data artifact with checksums; these scripts document how it was built. Order (from the build reports and file
  timestamps; the rewrites were run by hand): records_merged.json ← normalize_entities → resolve_citations →
  translate_absences_v2 → classify_paraphrases; deep_read_records.json ← recover_deep_records; then build_kb_v2 →
  embed_kb_v2 → build_state_v2 → merge_state_v2.
- **Multi-108 support** (`scholarqa_multi/*.py`, `_shared/tools/multi_*`): produced the reported Multi-108 numbers.
- **Superseded experiments**: the ReAct harness `_shared/tools/evidence_gate2r_harness.py` (2,784 lines) and its
  report adapters; PaperScope-era scripts (`ps53r_run.py`, `judge_answer.py`, `judge_kb.py`, `behavior_probe.py`);
  one-off probes, audits and patches in `cs2/` (`probe_l1*.py`, `audit_s5_*.py`, `patch_*.py`, `verify_cit_bug.py`,
  `growth_*.py`, ...); older judge drivers (`judge_run.py`, `judge_dev20.py`, `judge_batch_task.py`,
  `gen_judge_task.py`, `judge_runs_ours*/`). `score_compare.py` uses a wrong aggregate ((IR+AP+F1)/3) and must not
  be used.
- `tests/`: `test_backref_and_cite_repair.py`, `test_external_tools_backflow.py` — tests of the ReAct harness and
  external_tools above.

## 10-06: the old packages left `src/` (tag `pre-integration-20261005`)

`src/kb_compiler/`, `src/kb_infra/`, `src/retrieval/` (the sci-evo library copy) and `src/sci_evo_extract.py` were removed
from `src/` after their capabilities moved (INTEGRATED-SYSTEM-1005 v2 §5–§6; module verdicts in
`docs/design/review/FITNESS-*.md`). They are not copied here: `git checkout pre-integration-20261005 -- src/<pkg>`
restores any of them. The sci-evo project itself is untouched at its original location.

| Old capability | Now | Notes |
|---|---|---|
| `kb_infra.llm` (call_local / call_paratera / call_cst / vision / purity / ledger / backoff) | `llm/client.py` | rewritten: streamed httpx, cancellable wall, lanes from config, response cache |
| `kb_infra.embedding` | `llm/embedding.py` | local model only, model label returned; no cross-model fallback |
| `kb_compiler.records.common.salvage_json_records`, `cards._repair_json_escapes` | `llm/jsonparse.py` | byte-identical on the old tests |
| `kb_compiler.records.table_channel` | `documents/tables.py` | |
| `kb_compiler.records.coarse_extract` | `compile/state/coarse.py` (field_state's input) → superseded by `extract/self_pass.py` T1 | |
| `kb_compiler.views.field_state` | `compile/state/field_state.py` | evaluation comparison arm (design v2 §6) |
| `kb_compiler.retrieve.hybrid` | `kb/index.py` (frozen v9b) | |
| `kb_compiler.records.deep_extract` v1.4, `postcheck`, `canary`, `table_semantic`, `notation_harvest` | to be rebuilt in phase C (T2 full-text, unified final check, canary on schema v2) | concepts kept; FITNESS-EXTRACTION §3 |
| `kb_compiler` registry / registry_growth / embed_block / backflow / views / arbitration / chunk_retry / figure_channel / survey_extract | retired | FITNESS-EXTRACTION §3, INTERNAL-AUDIT |
| `retrieval.circuit` | `sources/circuit.py` | directly usable |
| `retrieval.registry` | to be rebuilt in phase B as `library/` (DOI-first identity, merge queue, dates with precision, bulk import) | FITNESS-LIBRARY-ACQUIRE §1 |
| `retrieval.sources` (OpenAlex / S2 / Crossref / Sciverse / arXiv / Sci-Hub) | `sources/` (one client each, phase B); Sciverse already in `sources/sciverse.py` | |
| `retrieval.acquisition_chain`, `_see_upstream` MinerU `/file_parse` client | `acquire/`, `sources/mineru.py` (phase B) | the file_parse client is reused (9602, 18 lanes) |
| `retrieval.evidence_units`, `search_service`, `query_understanding`, `pdf_resolver`, `search_resolve`, `process_workflow`, `_see_*` | retired | |
| sci-evo `api/` + `frontend/` | `service/` + `web/` (phase E) | |
| Tests of retired code (`test_backflow`, `test_card_redirect`, `test_coarse_extract`, `test_compare_vocab_hint`, `test_deep_extract_*`, `test_embed_block`, `test_inline_pipeline`, `test_notation_harvest`, `test_postcheck_lenient`, `test_registry_*`, `test_table_semantic`, `test_views_derived`, `test_kb_compiler_import_gate`, `test_llm_ledger_fields`) | removed with the code | at the tag |
| `experiments/dfc/first_field.py` | `legacy/experiments/first_field_NOT_QUESTION_BLIND.py` | **not question-blind** (used ScholarCatalyst gold as build targets); kept only as a record, never to be run for results |
| root `build.py`, `build_manifest.json`, `conf/` (incl. `capacity.yaml`), `runs/_legacy_build_manifests/` | `compilescholar build` + `configs/` (capacity numbers now commented in `configs/base.yaml`) | removed 10-06; at the tag |
| `artifacts/MANIFEST.tsv` + README (10-04 pins of large files under `.research_tmp`) | `legacy/ARTIFACTS-*-20261004.*`; current pins in `data/MANIFEST.tsv` | the v9b answer KB copied to `data/benchmarks/cs2/kb_v2/` matches these sha256 byte for byte |
