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

Recompute: `results/README.md`. Code: tag `freeze-cs2-v9b`. BCa intervals differ from the percentile ones by ≤ 0.001.

**Length (W1-7).** score ~ question fixed effects + system + log(words), question-cluster bootstrap
(`results/cs2-test100-v9b-20261004/length_regression.json`): slope +0.052 per log-word [+0.016, +0.065]; length-adjusted
ours − harness +0.063 [+0.044, +0.084] (unadjusted +0.086), SciSpace − harness +0.051 [+0.027, +0.082]. Length explains
part of the gap, not all of it.

**Citation format (W1-6).** Our snippets are the longest (median 1,142 characters vs 239 for the harness); the
format-aligned re-judgment (snippets trimmed to 240 characters; titles only) is running — reading rule in PREREG §4.

## DeepScholar-Bench, 48 questions (earlier configuration: no citation expansion, no length budget)

| System | Nugget coverage | Words (median) |
|---|---|---|
| Ours (vnext_v3_nocite) | 0.310 | 2,130 |
| Claude Code + 27B harness (per-question cutoff; 3 failures = 0) | 0.241 | 668 |
| Oracle references, direct writing (B0) | 0.369 | — |

Ours − harness +0.069 [+0.034, +0.104] at original lengths. **Length-matched (W1-7): ours cut per question to the
harness answer's length (median 614 vs 668 words, keeping leading paragraphs) scores 0.179; ours − harness −0.062
[−0.097, −0.028] (8 wins / 13 ties / 27 losses).** The DSB advantage at original length comes from length; the cut is a
lower bound (median 2 of 4 sections survive), and a generation-time ~650-word budget is the fair control (pending).
Recompute: `compilescholar.eval.dsb.summarize/paired` over `.research_tmp/review_1002/p6/judged/` and
`runs/w1-7-dsb-length-matched-20261004/`. Tag `cs2-test-v9b-final`.

## Held-out survey field test (18 surveys) — protocol fixes pending (W1-4), numbers provisional

Compiled state (mean of two runs) vs one-call direct writing: families +0.121 [+0.007, +0.239], family properties
+0.035 [+0.007, +0.073]; limitations n.s. Known issues: 250-candidate judge cap, unequal input truncation, single
LLM judge. Source: `.research_tmp/docs_decisions/PAPER-DRAFT-EXPERIMENTS-1003.md` §6.5.
