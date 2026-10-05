# Architecture

CompileScholar is a literature layer for scientific agents: it finds papers, reads them, retrieves verbatim evidence,
and — its distinctive part — compiles **diachronic field cognition**: how the field has described each work up to any
date T (families, lineage, what a work is used as, the limitations the field states, how that changed). Every
capability is exposed as a tool that takes `as_of`.

Design documents: `.research_tmp/docs_decisions/system-vision-1004/` (DESIGN-UPGRADE-1005, DESIGN-LITERATURE-LAYER-1005,
NARRATIVE-V9-1005, EVAL-PLAN-1005).

## Data flow (one store, one identity, one schema)

```
papers      arXiv metadata, first-version date to the day          data/dfc/papers.sqlite
documents   full texts -> units (section / paragraph / table / caption)
citations   citation sentences -> bibliography entries -> paper ids  (zero LLM)
extract     statements: self pass (T1 abstract, T2 full text + method/experiment call), result pass (tables,
            deterministic), other pass (what citing sentences say about the cited work)
index       BM25 (FTS5) + optional dense vectors over papers / passages / statements
cognition   computed on read through AsOf(T): identity, families, lineage, profiles, facts, shifts, comparisons
tools       14 tools, all with as_of; MCP server fixes the cutoff server-side
grow        typed gap diagnosis -> acquisition -> back through the same stages (offline, question-blind)
```

- **Identity.** A paper is `paper:<arxiv_id>`; a cited work we cannot resolve is `stub:<normalized title>`. Dates are
  the arXiv v1 date. Every benchmark adapter maps to this identity.
- **Statement** (`extract/schema.py`) is the atomic unit: speaker, date, kind (self / other), about, role (one relation
  vocabulary: extends / improves / replaces / adapts / combines / uses / compares / proposes / background / criticizes /
  describes), facet, text, verbatim quote. Everything from `cognition` on is a function of statements dated <= T.
- **Stages** (`dfc/store.py`): one SQLite file per stage under `data/dfc/`, each with a manifest (params, upstream
  digests, code hashes). A stage refuses to run on a stale upstream; changing upstream data or a stage's code makes
  everything downstream stale. `compilescholar build status` shows the chain.

## Extraction tiers (kept from the old system's coarse/deep split)

| Tier | Scope | Input | Cost |
|---|---|---|---|
| T0 | every paper | metadata | 0 |
| T1 | papers in scope (primary category + start date) | title + abstract | one call |
| T2 | top-n in-scope papers with full text, by in-corpus citation count | abstract + introduction, method + experiment sections, tables | two calls + deterministic tables |
| other | every in-scope paper that is cited | <= 30 citation sentences, spread over citing months | ~24 pairs per call |

Selection uses corpus statistics only, never benchmark annotations.

## Module map (`src/compilescholar/`)

| Module | Role |
|---|---|
| `sources/arxiv_oai.py`, `arxiv_snapshot.py`, `arxiv_html.py` | metadata (OAI-PMH arXivRaw for v1 dates), title index, HTML cache (15 s pacing) |
| `corpus/papers.py` | papers stage; title resolution |
| `documents/units.py`, `tables.py`, `build.py` | full-text units; deterministic table parsing (moved from the old table channel) |
| `citations/markdown.py`, `html.py`, `resolve.py`, `build.py` | citation sentences, entry -> paper |
| `extract/schema.py`, `self_pass.py`, `result_pass.py`, `other_pass.py`, `build.py` | the one schema and three passes |
| `index/build.py` | multi-granularity search with as_of |
| `cognition/*` | AsOf view and the compiled objects |
| `tools/api.py`, `tools/mcp_server.py` | the tool layer |
| `answer/pipeline.py` | built-in consumer: `answer_lit` (tools) and the frozen v9b path `answer` |
| `grow/plan.py` | gap diagnosis and acquisition |
| `eval/*`, `baselines/*` | benchmarks, judges, comparison systems |
| `kb/`, `compile/state/` | frozen KB v2 path (v9b record; the "flat retrieval" comparison arm) |

## Invariants (tested)

- every statement's quote is a substring of its source; result units are valid only inside their own paper (no
  cross-paper numeric alignment);
- `as_of` monotone: no tool returns anything dated after `as_of`; the MCP server fixes `as_of` itself;
- every id a tool returns is a paper visible at `as_of` or a flagged stub;
- a stage never runs on a stale upstream;
- the frozen v9b answer path reproduces its goldens byte for byte (`tests/test_characterize_*.py`).
