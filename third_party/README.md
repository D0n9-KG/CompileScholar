# Third-party code

Both evaluators are used as pinned upstream checkouts plus the patches in `patches/`. Nothing here is vendored:
the checkouts live outside the package and are located through environment variables.

| Dependency | Upstream | Pinned commit | Used for | Location (env override) | Local patches |
|---|---|---|---|---|---|
| asta-bench | https://github.com/allenai/asta-bench | 9bf087a (2026-09-26) | CS2 scorers (`astabench.evals.sqa`) | `CS_ASTABENCH` (default: `.research_tmp/scratch/ai2_baseline_2026-09-08/asta/asta-bench-main`, editable install) | none in the checkout; runtime adapter in `src/compilescholar/eval/cs2/judge.py` (`JudgeAdapter`, every deviation switchable and recorded per row) |
| deepscholar-bench | https://github.com/guestrin-lab/deepscholar-bench | c95413b (2026-03-28) | DSB nugget judge (`nuggetizer`) + gold nuggets | `CS_DSB` (default: `.research_tmp/experiments/benchmarks/deepscholar/dsb`) | `patches/deepscholar-bench-c95413b-nuggetizer-base-url.patch` (lets the OpenAI client use an OpenAI-compatible `OPENAI_API_BASE`; no effect when unset) |

Reproduce a checkout:

```bash
git clone https://github.com/allenai/asta-bench && git -C asta-bench checkout 9bf087a && pip install -e asta-bench
git clone https://github.com/guestrin-lab/deepscholar-bench && git -C deepscholar-bench checkout c95413b
git -C deepscholar-bench apply ../third_party/patches/deepscholar-bench-c95413b-nuggetizer-base-url.patch
```

Judge-side deviations from the official CS2 pipeline (disclosed in the paper's evaluation appendix):

| Switch | Effect | Legacy runs (v9b and earlier) | Official-judge runs |
|---|---|---|---|
| connection_close | `Connection: close` on httpx (keep-alive hang on the Paratera gateway) | on | on |
| max_retries_4 | generate_with_retry max_retries 20 → 4, base_delay 1.0 | on | off |
| idx_zero_to_one | shift 0-based criteria_idx to 1-based | on | off |
| judge model | DeepSeek-V4.1-Flash (Paratera) instead of google/gemini-3-flash-preview | DeepSeek | gemini-3-flash-preview (OpenRouter) |
