# 论文草稿：Introduction + Related Work 骨架（v8，2026-10-03）

> 对应 NARRATIVE-V8-1003.md。数字留空（[TBD]），test 冻结后填。所有引用均为一手核过的 ID。

---

## 1 Introduction

**P1 — 任务与张力。** LLM research agents are increasingly asked field-level questions: *which approaches exist for X and
how do they differ, what are their shared limitations, what remains open*. Answering such questions requires evidence that
no single paper contains. Current systems retrieve passages or per-paper summaries at question time and leave the
cross-paper synthesis to the model.

**P2 — 现有编译路线与其缺口。** A recent line of work compiles literature ahead of time — reasoning graphs over tens of
millions of claims (LKM, 2609.27297), layered research assets with domain-level organization (ScholarStack, 2609.23735),
typed knowledge graphs with lineage and limitation entities (Agents-K1, 2606.13669). These systems build field-level views,
but the views themselves are evaluated only indirectly: LKM states that its landscape views "are not yet directly evaluated";
ScholarStack reports downstream task gains without assessing the domain layer. It remains unclear (a) whether the
field-level objects are *correct*, and (b) whether they — rather than better retrieval or a stronger writer — are what
improves answers.

**P3 — 我们做什么。** We compile a field state from a question-blind seed corpus: method families merged across surveys,
with family-level properties, limitations, open problems and comparisons, each grounded in verbatim quotes. A fixed
answering pipeline retrieves from this state alongside record-level, open-web and citation-expanded evidence, screens
off-topic evidence, and writes with sentence-level citations that are verifiable by construction.

**P4 — 证据。** On three external benchmarks with the same 27B model, the same knowledge cutoff and the same judge for
all self-run arms — ScholarQA-CS2 (AstaBench), DeepScholar-Bench, ScholarQA-Multi — we report [TBD]. Ablations isolate the
state layer: removing state retrieval changes [TBD]; removing the knowledge base entirely (a strong open-retrieval agent on
the same pipeline) changes [TBD]. We also evaluate the field layer directly by reconstructing held-out surveys' method
families and stated limitations from their cited papers [TBD].

**P5 — 评测发现（作为贡献之一，写克制）。** Along the way we find that common evaluation setups for literature QA are
sensitive to factors unrelated to answer quality: rubric recall grows with answer length when the official length
penalty is disabled; unenforced knowledge cutoffs let agents cite post-cutoff papers ([TBD]% of answers for an
unconstrained agent); adapter choices change a baseline's citation score from 0.763 to 0.557 on CS2 dev; and the official
closed-corpus citation metric silently drops out-of-question citations, which can turn a large gain into a large loss.
We release adapters and a strict variant.

**Contributions.**
1. A question-blind pipeline that compiles a field state (families, family-level limitations and comparisons) and answers
   from it with verifiable citations.
2. Same-model, same-cutoff, same-judge comparisons on three benchmarks, with ablations attributing gains to the state layer.
3. A direct evaluation protocol for field-level objects via held-out survey reconstruction.
4. An analysis of evaluation validity for literature QA (length, cutoff leakage, adapter sensitivity, metric definition).

---

## 2 Related Work（骨架：每段 = 一类 + 我们的位置）

**Compiled scientific knowledge.** LKM (2609.27297), ScholarStack (2609.23735), Agents-K1 (2606.13669), ASKS (2608.29612),
AskChem (2607.28618). They compile claims, reasoning chains and domain organization at scale; their field-level views are
evaluated through downstream tasks or not at all. We evaluate the field layer directly and attribute downstream effect to it.

**Cross-paper claim relations and taxonomies.** ClaimFlow (2603.16073) annotates support/extend/qualify/refute relations;
TaxoBench (2601.12369) and TaxoBench-CS (2509.19125) reconstruct survey taxonomies from cited papers; SciTraj (2606.22342)
types paper-to-paper relations. These score labels or trees; we score family-level judgments (shared limitations, open
problems) and use them for answering.

**Literature QA and report generation.** PaperQA2 (2409.13740), OpenScholar, STORM (2402.14207), DeepScholar-Base
(2508.20033), GPT-Researcher; deep-research agents (OpenAI DR). Query-time retrieval and synthesis; we compare under the
same model and cutoff.

**Citation expansion and coverage.** PaSa (2501.10120), SPAR (2507.15245), Sahu et al. (2605.29234), PaperQA2's citation
traversal. We use co-citation-ranked expansion as one retrieval channel; it is not our contribution.

**Evaluation of literature QA.** AstaBench / ScholarQA-CS2, DeepScholar-Bench (2508.20033), RWGBench (2606.24894), the AI2
meta-evaluation "Deep Research, Shallow Evaluation" (2603.06942), length–quality conflation (2511.07685), temporal leakage
in date-filtered retrieval (2602.00758). Our validity analysis adds adapter sensitivity and closed-corpus metric definition.
