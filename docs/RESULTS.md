# Results

Only frozen numbers. Each row: what, n, the number with a 95% CI where paired, the run / tag that produced it, and how
to recompute it. Working numbers live in run directories, not here.

## CS2 test, frozen v9b (pre-upgrade record) — `results/cs2-test100-v9b-20261004/`

Judge: DeepSeek-V4.1-Flash, legacy adapter. n = 100, missing / failed = 0. Ours = mean of two runs (0.828 / 0.830).
Paired: percentile bootstrap 10,000 + sign-flip permutation, Holm over the five global comparisons.

| System | G | IR | AP | CR | CP | ours − system |
|---|---|---|---|---|---|---|
| Ours (v9b) | 0.829 | .743 | .785 | .837 | .953 | — |
| SciSpace (archived) | 0.837 | .852 | .875 | .739 | .883 | −0.008 [−0.024, +0.009] |
| Elicit (archived) | 0.798 | .744 | .919 | .690 | .839 | +0.031 [+0.011, +0.052] |
| Claude Code + 27B harness | 0.744 | .764 | .981 | .418 | .811 | +0.086 [+0.066, +0.107] |
| GPT-Researcher + 27B | 0.736 | .506 | .758 | .777 | .903 | +0.094 [+0.074, +0.113] |
| OpenAI Deep Research (archived) | 0.684 | .922 | .900 | .226 | .686 | +0.146 [+0.129, +0.162] |

Recompute: `results/README.md`. Code: tag `freeze-cs2-v9b`. Open caveat: citation snippets are longest for ours
(median 1,142 characters vs 239 for the harness); a format-aligned re-judgment is pending (W1-6).

## DeepScholar-Bench, 48 questions (earlier configuration: no citation expansion, no length budget)

| System | Nugget coverage | Words (median) |
|---|---|---|
| Ours (vnext_v3_nocite) | 0.310 | 2,130 |
| Claude Code + 27B harness (per-question cutoff; 3 failures = 0) | 0.241 | 668 |
| Oracle references, direct writing (B0) | 0.369 | — |

Ours − harness +0.069 [+0.034, +0.104]; not length-matched. Recompute: `compilescholar.eval.dsb.summarize/paired` over
`.research_tmp/review_1002/p6/judged/`. Tag `cs2-test-v9b-final`.

## Held-out survey field test (18 surveys) — protocol fixes pending (W1-4), numbers provisional

Compiled state (mean of two runs) vs one-call direct writing: families +0.121 [+0.007, +0.239], family properties
+0.035 [+0.007, +0.073]; limitations n.s. Known issues: 250-candidate judge cap, unequal input truncation, single
LLM judge. Source: `.research_tmp/docs_decisions/PAPER-DRAFT-EXPERIMENTS-1003.md` §6.5.
