# CHANGELOG

All notable changes to CompileScholar. Format loosely follows Keep a
Changelog; dates are local. Historical era (LogicKG, granular-agent,
contest) lives in `archive/` and git history — not repeated here.

## 2026-10-06 — integrated system, phase A (foundations)

Design: `docs/design/INTEGRATED-SYSTEM-1005.md` v2.1 (rewritten after four module fitness reviews,
`docs/design/review/`). Tag `pre-integration-20261005` marks the state before.

### Changed
- Stage contract (`dfc/store.py`) rewritten: import-closure code hashes, config-level `stale` vs data-level `behind`,
  item-level `_work` bookkeeping (only ok is done; retries; sweep), atomic manifests, stage locks, `--rebuild`,
  breaker; per-thread read connections. All stages ported; the eight contract probes are regression tests.
- One LLM client (`llm/client.py`): streamed and cancellable, lanes / rate limits from config, ledger always on,
  response cache, purity audit, vision, `call_json`. Measured 791 tok/s at 48 lanes.
- `core/ids.py` and `core/asof.py`: one identifier normalisation and one visibility rule (dates with precision);
  tools filter cited objects by visibility (version leak found in the real store).
- Paths: `data/{library,derived,benchmarks,external}`, `cache/`, `results/`, `third_party/`; machine resources only in
  `configs/local.yaml`; `data/MANIFEST.tsv` pins the benchmark inputs (`tools/data_manifest.py`).

### Removed
- `src/kb_compiler`, `src/kb_infra`, `src/retrieval`, `src/sci_evo_extract.py` (capabilities moved; `legacy/INDEX.md`),
  root `build.py` / `build_manifest.json` / `conf/` / `artifacts/`, tracking of `.research_tmp/` (kept on disk);
  workspace clutter cold-stored on the NAS (`CompileScholar-cold-20261006`, every file sha256-verified).

## 2026-09-22 — pipeline engineering day (A/B series)

### Added
- `conf/` layered config: `base.yaml` (git) + `local.yaml` (gitignored) +
  30-line deep-merge loader + env projection (`kb_compiler.config`). The
  structural fix for the env-name-mismatch bug (run_stage wrote
  LLM_SOCK_TIMEOUT, kb_infra read LOCAL_SOCK_TIMEOUT — the 600 never took
  effect and 62 marathon chunks died at the 300s default).
- `build.py`: single pipeline entry. Stage registry (Kedro Node idea),
  fixed-path outputs + skip-if-fresh (Snakemake make semantics), LLM-stage
  protect() (output-present = skip, rerun needs --force), per-run manifests
  `runs/manifest-*.json` (DVC-lock/ACM idea), `--list/--dry-run/--dot`.
  Design researched from primary docs (Snakemake/Kedro/Dagster/DVC/Hydra/uv);
  over-engineered team-scale features deliberately not adopted.
- `pyproject.toml` + editable install + `kb` CLI entry
  (`kb build|list|test`). Package name compile-scholar 0.4.0.
- `CHANGELOG.md` (this file).
- chunk-level checkpoint WAL (`records_slot.json.progress.jsonl`):
  write-ahead log of completed chunks; resume replays it and only calls
  missing chunks. `--repair` seeds the WAL from existing results so hole
  re-fills touch ONLY the holes (measured: 52 papers, 1153 chunks seeded,
  402 hole tasks called — 74% call savings vs whole-paper rerun).

### Changed
- **Global chunk pool** replaces workers × chunk-threads (which floated
  9–33 in-flight and starved the marathon tail): one fixed pool (default
  30, `conf/base.yaml slot.pool_size`), per-paper finalize fires as soon
  as all its tasks land. Output byte-comparable to the serial path (B2
  invariant, 5 new tests).
- **Renames (module = stage = product, codenames retired at the code
  surface)**: slot.py → `deep_extract.py`, skeleton.py → `cards.py`,
  registry_round2.py → `registry_growth.py`.
- **Table channel merged**: `table_extract.py` single stage entry drives
  the two battle-tested phases (deterministic parse + LLM semantic);
  historical F24/F35 codenames no longer appear at the stage surface.
- `kb_infra.llm.load_env` default path now repo-relative (was hardcoded
  absolute). `verification/consistency.py` __main__ path likewise.
- `LOCAL_SOCK_TIMEOUT` default 300 → 900 (big-output chunk calls
  physically need ~600s; 12 retries × 300s were burned per stuck chunk).

### Fixed
- Marathon tail crawl diagnosed and fixed (two causes: chunk-serial
  per-paper execution → global pool; 300s socket wall → 900s).
- 62 lost marathon chunks: 61 refilled by the repair run (proof they were
  the timeout bug, not content). 1 genuine content-triggered straggler
  remains, disclosed: `Improving_text_embeddings#c23` (appendix
  instruction-list chunk; output exceeds the 9k-token budget).
- LightRAG baseline harness: QueryParam import scope bug; ainsert needs
  `ids=` AND `file_paths=` (ids-only leaves file_path=unknown_source →
  empty reference list → dead citation bridge).

## 2026-09-21 — Multi-108 scaling (registry/vocab/slot marathon)

See git log 2026-09-21/22 and ASSET-STATE.md §0–§4 for the campaign
record: embedding-block registry merge (τ=0.84), vocab families, 430-paper
corpus, slot marathon (61.57M tokens, purity PURE), baseline harnesses.

## 2026-09-20 — rename LogicKG → CompileScholar

Repo renamed; system's thesis stated in the name: compile-time
understanding vs OpenScholar's runtime retrieval.
