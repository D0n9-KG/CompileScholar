# Architecture

## Two phases

**Compile (offline, question-blind).** Papers → typed records with verbatim quotes → field-level objects that no
single paper states. Two state builders exist:

- `kb_compiler` + `.research_tmp/.../base_kb_build` (legacy chain; produced `cs2/base_kb_v2`, the KB behind every
  reported CS2/DSB number). Its "state" is method families merged from survey taxonomies (`state_merged.json`).
- `compilescholar.compile.state.field_state` (new): per-paper coarse records → family induction (propose in batches,
  merge) → family-level properties / limitations with support sets and `n_papers` → area-level open problems.
  Evaluated so far only on the held-out survey test; wiring it into the answer KB is upgrade step W3.

**Answer (per question, read-only).** `compilescholar.answer.pipeline.answer()`:

1. probe — search the question itself (external + KB) to fix the field / sense of the terms;
2. plan — 2–6 sections derived from the question, 2–3 queries each (`prompts/plan.txt`, `probe_block.txt`);
3. gather — per query in parallel: KB hybrid retrieval (BM25 + dense, RRF, per-paper cap), field-state channel,
   Sciverse semantic search (server-side cutoff), optional citation expansion (co-citation over seed references via
   `sources.refgraph`: OpenAlex batch → S2 → Crossref);
4. screen — drop clearly off-topic excerpts (`prompts/screen.txt`);
5. write — each section cites evidence ids sentence by sentence, optional word budget (`prompts/write.txt`);
6. assemble — deterministic id → citation mapping; snippets are the evidence excerpts.

## Package layout (`src/compilescholar/`)

| Module | Role | Notes |
|---|---|---|
| `core/config.py` | layered YAML → typed `RunConfig` | unknown keys rejected |
| `core/paths.py` | every repo-relative location | `CS_ROOT`/`CS_DATA`/`CS_CACHE`/`CS_RUNS` overrides |
| `core/secrets.py` | the only `.env` reader | process env wins over `.env` |
| `core/cutoff.py` | knowledge-cutoff rule shared by every arm | per-thread value for per-question cutoffs |
| `core/manifest.py` | run manifests, `verify()` | LF-normalised hashes |
| `llm/` | local server + Paratera clients, embeddings, lenient JSON | TLS verification on; lanes created on first call |
| `sources/sciverse.py` | external semantic search | cross-process token bucket (30 req/min account limit) |
| `sources/refgraph.py` | multi-source reference graph | per-source pacing in `~/.pace_<src>.json`; OpenAlex credit guard |
| `kb/` | KB loader, hybrid index, state channel | |
| `answer/` | pipeline + prompt files | prompt sha256 in every manifest |
| `compile/state/` | field-state compiler, coarse extraction | |
| `eval/cs2/` | official scorers driven directly (`JudgeAdapter`), four-facet scoring | each judge deviation switchable and recorded per row |
| `eval/dsb.py`, `eval/field.py`, `eval/stats.py` | DSB nuggets, held-out survey test, paired statistics | missing answers score 0 everywhere |
| `baselines/` | Claude Code harness (runner, MCP server, compat proxy), archived-system adapters, GPT-Researcher adapter | harness isolation: project-only settings, strict MCP config, neutral cwd |
| `cli.py` | `answer` / `judge` / `score` / `verify` | |

## Invariants (tested)

- The moved answer path, scoring and components reproduce the pre-move code byte for byte
  (`tests/test_characterize_*.py`, goldens from tag `cs2-test-v9b-final`).
- Importing the package does not modify `sys.path` and pulls in no legacy module.
- A run directory refuses to resume under a different resolved configuration; judge-error rows are never scored as 0.
