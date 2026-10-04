# Experiments

All self-run arms use the same local model (Qwen3.8-27B, thinking off) and the same knowledge cutoff, enforced
server-side in every retrieval tool. Missing answers and system failures score 0; judge errors are re-judged until 0.

## Benchmarks

| Benchmark | Data | Split use | Cutoff | Metric | Entry point |
|---|---|---|---|---|---|
| ScholarQA-CS2 (AstaBench) | `.research_tmp/experiments/benchmarks/scholarqa_multi/sqa2_rubrics_v{1,2}_recomputed.json` (dev = v1, test = v2, as in astabench task.py) | dev offset 10–24 regression only; dev 25–99 confirmation; test once per frozen system | 2025-05 | four official facets, equal weight | `compilescholar answer/judge` with `configs/bench/cs2_*.yaml` |
| DeepScholar-Bench | `third_party` deepscholar-bench @ c95413b + `.research_tmp/review_1002/p6/oracle_inputs.json` (48 unique questions) | 16 dev / 32 held-out (PREREG §3) | target paper's publication month | nugget coverage (strict, official assignment prompt) | `compilescholar.eval.dsb` |
| ScholarQA-Multi-108 | closed corpus of 430 papers | all 108 (98 after excluding 10 mismatched questions) | n/a | citation F1 (pre-registered and strict) | legacy (`legacy/benchmarks/scholarqa_multi/`) |
| Held-out survey field test | 20 surveys (18 with reference lists), gold from survey text | all | n/a | family / property / limitation recall, LLM-judged (`MATCH` prompt) | `python -m compilescholar.eval.field` |

## Judges

- CS2: official astabench scorers (`score_sqa`, `score_precision`, `score_citation`, official arguments), judge model
  DeepSeek-V4.1-Flash via Paratera for every arm. The official model `google/gemini-3-flash-preview` is region-blocked
  from our network (HTTP 403 on every OpenRouter route); PREREG amendment 1 makes DeepSeek the primary measure.
  Legacy runs used three local deviations (`third_party/README.md`); new runs keep only the network fix.
- DSB: Nuggetizer, DeepSeek-V4.1-Flash via Paratera.
- Field test: DeepSeek-V4.1-Flash; human spot-check pending.

## Known pitfalls (measured)

- **Rate limits.** Sciverse 30 req/min per account (shared token bucket across processes); S2 without a key is an
  IP-shared pool (~1 req/s, IP-level 429 when several processes hit it); OpenAlex 10k credits/day (title search 10,
  id batch 1); arXiv export ≤ 1 req / 3 s, arxiv.org/html Crawl-delay 15. Never run two S2-using jobs at once.
- **Harness serving layer.** The local endpoint returns tool calls without `input` and non-standard thinking deltas;
  the compat proxy fixes both without touching model content. Without `--setting-sources project` and a neutral cwd,
  Claude Code injects the user's own settings and CLAUDE.md.
- **Archived arms.** Elicit responses are keyed by question text (exact match after normalisation); its citation
  titles sit in `metadata.paper.title` and must be lifted, or the scorer drops both citation facets.
- **Long jobs on Windows.** Background shells are killed after 2 h and `until` loops survive `TaskStop`; long runs are
  launched as detached processes and checked by command line.
- **Length.** The official CS2 configuration has no length term and rubric recall grows with length; all CS2 runs use a
  ~1,000-word budget and report word counts.
