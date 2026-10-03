# 论文草稿：实验章（v8，2026-10-03）

> 已有数据的部分用真实数字；未出结果的标 [TBD]。所有数字可由 `.research_tmp/review_1002/` 与 `experiments/benchmarks/` 下脚本复现。
> 纪律：CS2 终数只来自 test；dev 数字只用于开发与分析章。

---

## 6 Experiments

### 6.1 Setup

**Model and judge.** All self-run arms use the same local model (Qwen3.8-27B). All arms — including archived answers of
external systems — are judged by the same judge (DeepSeek-V4.1-Flash) in place of the official judge; we re-judge 30 test
questions with a second judge to check ranking stability [TBD].

**Knowledge cutoff.** Enforced server-side in every retrieval tool of every arm (year filter pushed to the search API, plus a
client-side check). CS2: before 2025-05; DSB: before the target paper's publication month.

**Scoring.** CS2: official four facets with equal weight (ingredient recall, answer precision, citation recall, citation
precision); unanswered or failed questions score 0. DSB: nugget coverage (official assignment prompt), 48 unique questions.
Multi-108: citation F1 (pre-registered) and a strict variant (§6.4).

### 6.2 Main results [TBD — test split]

| System | CS2 test (G / IR / AP / CR / CP) | DSB nugget | Multi F1 (pre-reg / strict) |
|---|---|---|---|
| Ours (full) | [TBD] | [TBD] | [TBD] |
| Claude Code + 27B + retrieval MCP (harness) | [TBD] | [TBD] | — |
| GPT-Researcher + 27B | [TBD] | — | — |
| OpenAI Deep Research (archived answers, re-judged) | [TBD] | — | — |
| Elicit (archived, re-judged) | [TBD] | — | — |
| STORM + 27B | — | [TBD] | — |
| LightRAG / PaperQA2 (closed corpus) | — | — | 0.515 / 0.461 ; 0.449 / 0.431 |

### 6.2b Answer length (dev, 15 questions; reported because the official CS2 configuration has no length term)

| Arm | words (median) | G | IR | AP | CR | CP |
|---|---|---|---|---|---|---|
| Ours, ~1,000-word budget (default) | 1,024 | 0.842 | .744 | .883 | .822 | .921 |
| Ours, no length limit | 2,832 | 0.812 | .802 | .715 | .795 | .936 |
| Claude Code + 27B harness | 843 | 0.743 | .764 | .979 | .419 | .810 |

Removing the budget raises ingredient recall (−0.059 with the budget, CI [−0.105, −0.010]) but lowers answer precision more
(+0.169 with the budget, [+0.120, +0.208]); net +0.031 [+0.006, +0.050] for the budgeted arm. At matched length the gap to the
harness is +0.099 [+0.049, +0.149] and comes entirely from citation recall (+0.40) and precision (+0.11); ingredient recall
ties (−0.02, n.s.). We use the ~1,000-word budget for all CS2 runs.

### 6.3 Ablations (CS2 dev, same 15 questions, paired bootstrap; DSB [TBD])

Each row changes one component against the configuration it was developed on (the pipeline evolved during development, so
the reference arm differs by row; all deltas are paired on the same 15 questions).

| Change | Reference arm (G) | Δ G [95% CI] | Facet that moved |
|---|---|---|---|
| − state retrieval | no-limit, pre-probe (0.803) | +0.000 [−0.037, +0.034] | AP −0.05, CR +0.03, CP +0.03 (cancel) |
| − probe (retrieve before planning) | no-limit (0.806) | −0.003 [−0.046, +0.040] | off-topic paragraphs 37% → 45% |
| − citation expansion (multi-source) | final, ~1,000 words (0.837) | −0.010 [−0.040, +0.019] | AP −0.08 (S2 variant, sig.) |
| citation expansion via S2 instead of multi-source | final (0.837) | +0.005 n.s. | — |
| − length budget (no limit) | ~1,000 words, S2 expansion (0.842) | −0.031 [−0.050, −0.006] | IR +0.06, AP −0.17 (§6.2b) |
| + cross-section deduplication | ~1,000 words, S2 expansion (0.842) | −0.010 [−0.048, +0.027] | AP −0.089 [−0.166, −0.013] |
| + screening (Multi-108, strict F1) | without screening (0.320) | +0.104 [+0.086, +0.122] | — |

No component of the retrieval/organization stack other than the length budget and per-sentence evidence citation (§6.2b)
moves the CS2 score beyond noise at n = 15. In particular, retrieving from the compiled state contributes nothing to CS2
single-question scores; its value is measured where the task asks for field-level judgments (§6.5).
[TBD: − knowledge base (open retrieval only); DSB deltas.]

### 6.4 Multi-108: metric definition matters (已有数据)

The pre-registered citation F1 maps each cited paper to the question's own context set and **drops** citations outside it.
Our closed corpus is the union of all 108 questions' contexts (430 papers), so a system can cite related papers that belong to
other questions. A strict variant counts such out-of-question citations as false positives.

| System | Pre-registered F1 | Strict F1 | Share of citations inside the question's set (median) |
|---|---|---|---|
| Ours (new pipeline, v1, no screening) | 0.852 | 0.320 | 0.31 |
| Ours (new pipeline, v2, with evidence screening) | 0.846 | 0.424 | — |
| Ours (previous system) | 0.650 | 0.532 | 0.65 |
| LightRAG | 0.515 | 0.461 | — |
| PaperQA2 | 0.449 | 0.431 | — |

The ranking flips under the strict variant. We therefore report both and do not claim the pre-registered gain as an
improvement. Evidence screening raises strict F1 by +0.104 (paired bootstrap 95% CI [+0.086, +0.122]) at no lenient cost
(−0.006 [−0.016, +0.003]); under the strict variant v2 ties LightRAG (−0.037 [−0.083, +0.007]) and PaperQA2
(−0.007 [−0.058, +0.044]) and stays below the previous system (−0.108 [−0.159, −0.054]).

**Data defect.** For 10 questions (bohao_cs_1–10) the gold answer and gold contexts are about optical microcavity sensing while
the question asks about NLP/HCI reading tools (question–answer TF-IDF cosine 0.000–0.029). No system can retrieve the gold
contexts for them. On the remaining 98 questions: ours v1 0.935 / 0.352, ours v2 0.929 / 0.467, previous system 0.717 / 0.586,
LightRAG 0.567 / 0.508, PaperQA2 0.495 / 0.475 (pre-registered / strict).

### 6.5 Field-layer evaluation

20 held-out surveys (4 each from cs.AI/CL/CV/IR/LG), gold = the surveys' own taxonomy nodes, node properties and stated
limitations/gaps extracted verbatim. Systems see only each survey's cited papers, never the knowledge base (the 20 surveys
are in the answer KB, so any future variant that compiles over the KB must exclude them and their records). Inputs are
title + abstract ( 18/20 surveys had a resolvable reference list; cited papers with an abstract, median 88 per survey). All systems use the same 27B and
the same inputs; a different model (DeepSeek-V4.1-Flash) judges whether each gold statement is expressed by some system
statement. Systems: **state** (compiled field state: per-paper records → family induction → family-level facts with
verbatim support → cross-paper aggregated open problems), **direct** (one call: all papers → write the taxonomy, family
properties and open problems), **flat** (per-paper extraction only, no cross-paper organization), **memory** (survey
title only; measures what the 27B already knows). Gold items were tagged self-contained or not by the judge model before
any system output was inspected ("clean": 316/460 limitations, 380/528 properties). Recall@K takes each system's first K
items, K = number of gold items.

**Run-to-run variance (must be read before the table).** The compiler was run twice per survey (run 1 / run 2: same
per-paper records, re-induced families and facts; on 4 surveys run 1 predates the open-problem stage, so run 1 has no
open problems there). The two runs differ by +0.088 [+0.027, +0.157] on family recall and −0.080 [−0.240, +0.071] on
limitations — as large as the effects below. We therefore report the mean of the two runs as "state" and never pick the
better run per column. [TBD: a third run; seeds/temperature logged.]

| System | Families recall | Properties recall (clean) | Limitations recall (clean) | Items (fam / prop / lim) |
|---|---|---|---|---|
| state (mean of 2 runs) | 0.586 (run 1 .630 / run 2 .542) | 0.099 (.109 / .089) | 0.313 (.273 / .353) | 10 / 70 / 65 |
| direct | 0.465 | 0.064 | 0.263 | 8 / 32 / 31 |
| flat | 0.596 | 0.033 | 0.346 | 188 / 471 / 70 |
| memory | 0.317 | 0.050 | 0.153 | 5 / 20 / 25 |

Paired bootstrap over surveys (state = 2-run mean; n = 18, properties n = 17):
- Families: state − direct +0.121 [+0.007, +0.239]; state − memory +0.268 [+0.110, +0.414]; state − flat −0.011 n.s.
  (flat lists 19× more items).
- Family properties: state − direct +0.035 [+0.007, +0.073]; state − flat +0.067 [+0.018, +0.123]; state − memory +0.050
  [−0.012, +0.110] n.s.
- Limitations: state − direct +0.050 n.s.; state − flat −0.034 n.s.; state − memory +0.160 [+0.046, +0.271].

Reading. Compiling across papers recovers the surveys' families and family-level properties somewhat better than writing
the skeleton in one call from the same papers; the margins are small and the property gain over memory-only is not
significant. For limitations, organization adds nothing over the unorganized per-paper limitations. Absolute property
recall is ≈0.1 for every system.

Known protocol issues to fix before this section is final: (i) the judge sees only the first 250 candidates, which
truncates flat on 14/18 surveys for properties (3/18 for families) and biases flat downward; (ii) direct's input
truncates abstracts to 600 characters while the compiler reads the full abstract; (iii) recall@K is not reported because
flat's items have no ranking; (iv) single LLM judge; [TBD: human spot-check of 50 matched / 50 unmatched judgments, second
judge].

## 7 Analysis

### 7.1 Synthesis vs. coverage on DeepScholar-Bench (已有数据)

With the gold reference set given (oracle, 43 common questions), writing directly from abstracts scores 0.379 nugget coverage;
imposing abstract organization axes hurts (target-induced design dimensions 0.269, −0.110; complete category enumeration 0.309,
−0.070), topic clustering ties (0.381, +0.002), section-end positioning ties (+0.001), and the official DeepScholar-Base
two-stage procedure scores 0.325 (−0.054). A human-verified error analysis of a retrieval agent's missed nuggets (154
nuggets) attributes 60.6% to retrieval (the cited paper was never found), 2.9% to reading depth, 15.9% to selection and 10.1%
to cross-paper organization. The retrieval gap is not specific to that agent: in open retrieval our pipeline's citations
overlapped the gold reference lists in 5 of 55 references on three pilot questions; querying the search API with gold
reference titles verbatim recovered 3 of 10, topic queries 1 of 20, and co-citation expansion from relevant seeds 1 of 7;
only 5.2% of gold-reference names appear in the target abstracts (847 references, 48 questions), so entity-driven
querying from the task input cannot close it. [TBD: full-48 numbers]

### 7.2 Evaluation validity on CS2 (已有数据，dev)

- **Length.** The official configuration disables the length penalty (simplified rubric scoring); rubric recall correlates with
  answer length: within the harness arm on dev, Pearson r = 0.755 (rank correlation 0.52, n = 93; word count of the judged
  section text, median 1,385 words).
- **Cutoff leakage.** Without server-side enforcement, an unconstrained agent cited post-cutoff papers in 42 of 93 answers;
  those answers scored 0.774 vs 0.609 (confounded by topic).
- **Adapter sensitivity.** Converting GPT-Researcher's markdown citations so that citation ids appear in the text (as the
  official format requires) changes its dev score from 0.763 to 0.557 (citation recall 0.845 → 0.463).
- **Archived-answer pairing.** Elicit's released responses are keyed by the question text itself (200 of 201 match a
  dev/test question exactly after normalization); a token-overlap matcher we used earlier paired 48 of 148 responses with the
  wrong question. All archived systems are re-paired by exact match.
- **Serving-layer compatibility.** With a self-hosted model behind an Anthropic-compatible endpoint, the agent harness failed
  after one turn whenever the model called a no-argument tool (the endpoint returned the tool call without an `input`
  field; streaming thinking deltas were also non-standard) — 2 of the first 3 questions of the cutoff rerun. A normalizing
  proxy that changes no model-generated content fixes both; we rerun the harness arm through it and count any remaining
  failure as 0. [TBD: whether earlier harness failures (12/100 dev, 7 timeouts) share this cause]
- **Judge noise.** Re-judging identical inputs: mean |Δ global| 0.043 per question (5 questions).
- **Statistical power.** With per-question paired SD ≈ 0.24, detecting Δ = 0.05 at 80% power needs ≈ 176 paired questions;
  CS2 test (100 questions) supports non-inferiority claims at ≈ ±0.05 but not small differences.
