# Architecture

CompileScholar is a literature layer for scientific agents: it finds papers, reads them, retrieves verbatim evidence,
and — its distinctive part — compiles **diachronic field cognition**: how the field has described each work up to any
date T (families, lineage, what a work is used as, the limitations the field states, how that changed). Every
capability is exposed as a tool that takes `as_of`.

Design documents: `docs/design/INTEGRATED-SYSTEM-1005.md` (the current system design; module verdicts in
`docs/design/review/FITNESS-*.md`) and `docs/design/system-vision-1004/` (NARRATIVE-V9-1005, EVAL-PLAN-1005,
DESIGN-UPGRADE-1005, DESIGN-LITERATURE-LAYER-1005 and the research reports behind them).

**State (2026-10-06).** Phase A (foundations) is done: the stage contract, the one LLM client, identifiers and time,
configuration and paths, a single package and a clean workspace. Phase B (the base) is built: the library with
DOI-first identity and a merge queue judged by two models, acquisition with identity checks, the two document tiers
(GROBID + PDF links for every paper, MinerU for the deep subset) and version awareness. The derived stages (documents,
citations, extract, index) still key on arXiv ids; phase C moves them onto the library (paper_id, text version and
its date). The data flow below marks what each stage is today and what it becomes.

## The library (phase B)

```
sources/   http (paced per source across processes, Retry-After, circuit, transient failures never cached),
           arxiv_oai (format 2: every version's date), arxiv_snapshot, scihub (local archive, internal use only)
library/   registry.sqlite — papers (paper_id: doi:/arxiv:/title:, never reassigned), identifiers (one value names
           one paper), dates (per source, kind, version; interval + precision), records, authors, members (a
           benchmark's candidate corpus: membership only, never questions or labels), aliases, merge_queue,
           assets, attempts, parses, imports
           import_arxiv · import_benchmarks · import_scievo · import_crossref: bulk, idempotent
           identity: resolve-then-write, nothing merged at write time; adjudicate: two models (local + Paratera)
           must both say "same work"; two different arXiv ids never merge automatically
acquire/   channels (arXiv per-version NAS mirror -> GCS bucket; OpenAlex OA -> Sci-Hub), identity check (own id
           stamp, title + surnames, LLM for the unsure), assets as pointers, mismatches to quarantine/
documents/ parse (fast: GROBID 0.9.1-full + hyperref links, every asset; careful: MinerU task router, the deep
           subset), tei (units, sentences, entries, citation pairs with page + bbox), mineru (sections, paragraphs,
           table HTML, formulas, captions, footnotes, references), assemble (one paper version + its text date),
           versions (v1 in full, the latest version's delta)
```

## Data flow

```
papers      arXiv metadata, v1 date to the day                    -> phase B: library/ registry, DOI-first identity,
                                                                     dates with precision per version and source
documents   full texts -> units (section / paragraph / table)     -> phase B: fast tier (PDF text layer, every paper)
                                                                     + careful tier (MinerU, deep-extraction subset)
citations   citation sentences -> bibliography entries -> paper   (zero LLM)
extract     statements: self pass (T1 / T2), result pass, other   -> phase C: schema v2, T2 over the full text,
            pass (what citing sentences say about the cited work)    unified final check, time-stratified sampling
index       BM25 (FTS5) + optional dense vectors                  -> phase D: tantivy + binary-code vectors
cognition   computed on read through AsOf(T)                      -> phase D: materialised (event-time tables,
                                                                     family snapshots, LLM verdicts with effective dates)
tools       14 tools, all with as_of; MCP server fixes the cutoff server-side
grow        typed gap diagnosis -> acquisition -> the same stages (offline, question-blind)
```

## Foundations (phase A)

- **Stage contract** (`dfc/store.py`). One SQLite file per stage under `data/derived/`, a manifest per stage
  (`data/derived/manifests/`), a file lock per stage. Two levels of change: *config* (params, the import closure of the
  stage's code computed from the AST, an upstream reconfiguration) makes a stage `stale`; *data* (an upstream stage
  gained or changed items) makes it `behind`. Each unit of work is a row in the stage's `_work` table keyed by
  (pass, item) and its input fingerprint: only `ok` counts as done, a failure is retried (3 attempts, then `gave_up`),
  items that left the set are swept with their outputs, so incremental builds equal a clean build. `--rebuild` writes a
  new file and swaps it in. A build whose block raises writes no manifest; consecutive failures trip a breaker.
  `compilescholar build status` shows stale / behind / complete per stage.
- **LLM client** (`llm/client.py`, `llm/embedding.py`, `llm/jsonparse.py`). One client for local / Paratera / CST and
  vision: streamed httpx (a deadline closes the connection, which stops server-side generation), lanes and rate limits
  per provider from `configs/` (`llm:`), retries with backoff and Retry-After, a breaker (`ChannelDead`), an
  always-on ledger (`runs/<run_id>/llm_calls.jsonl`), a temperature-0 response cache (`cache/llm/`, reproducible
  rebuilds), allowlist and arm-purity audit, `call_json` with validation retries and truncation salvage. Embeddings
  return their model label and dimension.
- **Identifiers and time** (`core/ids.py`, `core/asof.py`). One normalisation for DOIs (arXiv DOIs map to arXiv ids),
  arXiv ids (old style, versions) and titles (Unicode). Dates are intervals with a precision (day / month / year); a
  thing is visible at T iff its interval ends on or before T — a year-only date is never visible before Dec 31.
  `AsOf.before_month` gives the CS2 convention; `legacy_month_rule` keeps the frozen v9b rule.
- **Configuration and paths** (`core/config.py`, `core/paths.py`). `configs/base.yaml` < `configs/local.yaml` <
  experiment config < `--set`; sections `answer`, `runtime`, `bench`, `build`, `llm`. Every location comes from
  `core/paths` (`data/{library,derived,benchmarks,external}`, `cache/`, `runs/`, `results/`, `third_party/`);
  machine resources (NAS snapshot and PDF mirror, Sci-Hub index, MinerU, harness cwd) are named only in
  `configs/local.yaml` `paths:`. `tests/test_path_gate.py` keeps the package free of scratch paths and machine paths.

## Identity and the statement (current; phase B/C change)

- A paper is `paper:<arxiv_id>`; an unresolved cited work is `stub:<normalized title>`. Phase B replaces this with the
  registry's `paper_id` (DOI-first, never reassigned, aliases on merge).
- **Statement** (`extract/schema.py`) is the atomic unit: speaker, date, kind (self / other), about, role (one relation
  vocabulary), facet, text, verbatim quote, and the (pass, item) that produced it. Everything from `cognition` on is a
  function of statements dated <= T. Phase C adds facets config / absence / definition and the fields epistemic,
  condition, loc, schema_version.

## Module map (`src/compilescholar/`)

| Module | Role |
|---|---|
| `core/` | config, paths, secrets, ids, asof, run manifests, frozen v9b cutoff |
| `llm/` | the one LLM client, embedding, JSON parsing |
| `dfc/store.py` | the stage contract |
| `sources/` | the http layer; arXiv (OAI-PMH, snapshot, HTML), Sci-Hub local archive, Sciverse, refgraph, circuit breaker |
| `library/` | the registry (identity, identifiers, dates, assets, parses), imports, merge queue and its LLM verdicts |
| `acquire/` | full-text channels, identity check, quarantine |
| `corpus/papers.py` | papers stage (arXiv-keyed, until phase C moves the derived stages onto the library); title resolution |
| `documents/` | the two parse tiers (GROBID TEI, MinerU), version deltas, assembly by paper version; the documents stage; deterministic table parsing |
| `citations/` | citation sentences, entry -> paper |
| `extract/` | the schema and the passes |
| `index/` | multi-granularity search with as_of |
| `cognition/` | AsOf view and the compiled objects |
| `tools/` | the tool layer and its MCP server |
| `answer/` | built-in consumer: `answer_lit` (tools) and the frozen v9b path |
| `grow/` | gap diagnosis and acquisition |
| `eval/`, `baselines/` | benchmarks, judges, comparison systems |
| `kb/`, `compile/` | frozen KB v2 path (v9b record, flat-retrieval arm) and the field-state comparison arm |

## Workspace

`src/compilescholar/` (the only package) · `tests/` · `configs/` · `experiments/<topic>/` (one-off scripts, write to
`runs/`) · `runs/` (ignored) · `results/<frozen run>/` (tracked, with MANIFEST.tsv) · `data/` (ignored, pinned by
`data/MANIFEST.tsv`, `tools/data_manifest.py check`) · `cache/` · `third_party/` (pinned checkouts) · `docs/{design,
archive}/` · `legacy/` (INDEX of where every old capability went; the removed packages are at tag
`pre-integration-20261005`) · `.research_tmp/` (scratch only, untracked, never read by the package).

## Invariants (tested)

- every statement's quote is a substring of its source; result units are valid only inside their own paper (no
  cross-paper numeric alignment);
- `as_of` monotone: no tool returns anything dated after `as_of`, including references that a later text version added
  (`test_reference_newer_than_citing_v1_is_not_returned`); the MCP server fixes `as_of` itself;
- every id a tool returns is a paper visible at `as_of` or a flagged stub;
- a stage never runs on a stale or behind upstream; a failed item is never done; two builders of one stage cannot run
  at once (`tests/test_dfc_store.py`, the eight contract probes);
- the frozen v9b answer path reproduces its goldens byte for byte (`tests/test_characterize_*.py`), and the frozen CS2
  test numbers recompute from `results/` (`results/README.md`).
