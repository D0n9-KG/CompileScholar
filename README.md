# CompileScholar

Compile a research field into a **field state** — method families with their members, family-level properties and
limitations, and cross-paper open problems, each backed by verbatim support from several papers — and answer
literature questions from that state plus open retrieval, citing evidence sentence by sentence.

**Status (2026-10-04):** frozen pre-upgrade system evaluated on CS2 test (ours 0.829 vs SciSpace 0.837, Elicit 0.798,
same-model Claude Code harness 0.744; `results/`). Upgrade in progress: cross-paper skeleton (citations resolved to
papers, proposer extraction), the field-state compiler in the answer KB, consensus/contested states, and a
field-level QA benchmark. Plan: `.research_tmp/docs_decisions/upgrade-1004/UPGRADE-PLAN.md`; pre-registration:
`.research_tmp/docs_decisions/upgrade-1004/PREREG.md`.

## Quick start

```bash
python -m pip install -e ".[eval,dev]"        # Python >= 3.13
cp .env.example .env                           # fill in endpoints and keys (never committed)
python -m pytest -q                            # characterization + unit tests, no network

# answer CS2 questions with a config, judge, compare
compilescholar answer --config configs/bench/cs2_dev_confirm.yaml --run-id cs2-dev-<tag>
compilescholar judge  --run-id cs2-dev-<tag>            # DeepSeek-V4.1-Flash, official scorers (PREREG amendment 1)
compilescholar verify --run-id cs2-dev-<tag>            # recompute code / input hashes from the run manifest
```

Every run writes `runs/<run_id>/` (config, manifest with code / prompt / input hashes, answers, judge input, scores).

## Repository map

| Path | What |
|---|---|
| `src/compilescholar/` | the package: `answer/` (pipeline + prompts), `kb/` (KB + hybrid index), `compile/state/` (field-state compiler), `sources/` (Sciverse, multi-source reference graph), `eval/` (CS2 judge adapter + scoring, DSB, field test, stats), `baselines/` (harness, archived systems, GPT-Researcher adapter), `core/` (config, paths, secrets, cutoff, manifests), `cli.py` |
| `src/kb_compiler/`, `src/kb_infra/` | KB build pipeline (records, registry, views) used to build the existing KBs; to be folded into `compilescholar.compile` |
| `configs/` | run configurations (`base.yaml` < `local.yaml` < experiment config < `--set`) |
| `tests/` | characterization tests (goldens from the pre-move code) and unit tests |
| `results/` | frozen runs behind reported numbers (compressed, hash-pinned) |
| `artifacts/MANIFEST.tsv` | sha256 of large data the code depends on (answer KB, untracked intermediates) |
| `third_party/` | pinned upstream evaluators (asta-bench, deepscholar-bench) and local patches |
| `legacy/` | superseded code, kept with history; `legacy/INDEX.md` maps every reported number to a tag + entry point |
| `docs/` | `ARCHITECTURE.md`, `EXPERIMENTS.md`, `RESULTS.md`, `DECISIONS.md` |
| `.research_tmp/` | working area: experiment data, KBs, decision records, paper drafts (mostly untracked data) |

## Reproducing reported numbers

See `docs/RESULTS.md`: every number lists the run, the tag that produced it and how to recompute it.
