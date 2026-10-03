# 论文草稿：方法章（v8，2026-10-03）

> 对应叙事 NARRATIVE-V8-1003.md。只写已实现、可复现的部分；数字一律留空待 test 冻结后填。
> 实现位置：`src/kb_compiler/retrieve/hybrid.py`、`.research_tmp/experiments/benchmarks/_shared/tools/answer_pipeline.py`、
> `cs2/base_kb_build/{build_kb_v2,embed_kb_v2,build_state_v2,merge_state_v2}.py`。

---

## 3 Method

### 3.1 Overview

Given a research question, the system answers from a compiled **field state** plus open retrieval. Compilation happens
offline and is question-blind; answering is a fixed five-stage pipeline with no agentic loop. The design separates
*what is known about a field* (compiled once, reused across questions) from *what a question needs* (planned per question).

### 3.2 Compiling the field state (offline, question-blind)

**Seed corpus.** The warm-start knowledge base consists of three layers chosen without access to benchmark questions:
(i) 136 survey papers (one per CS sub-area, selected by type=review, citation count and a pre-cutoff date);
(ii) 21 "hub" papers co-cited by ≥3 surveys; (iii) 1,105 papers sampled per arXiv primary category.
A paper registry records title, year, identifiers and layer for every paper; metadata of hub papers is verified against
the arXiv OAI snapshot. Everything published after the benchmark knowledge cutoff is excluded.

**Records.** Each paper is compiled into typed records (finding, method, result, limitation, lineage, absence,
survey claim, domain snapshot), each with a verbatim quote and location in the source text. Surveys get a dedicated
extractor that emits taxonomy nodes, comparison tables, cross-method lineage and stated gaps.

**Field-level objects.** On top of records we build objects that no single paper states:

- *Method families.* Taxonomy nodes from different surveys are merged into families when their names are near-duplicates
  in embedding space (cosine ≥ τ_merge) **and** share a non-generic content word; union-find merges transitively but
  generic names (no content word) are excluded as bridges.
- *Family-level limitations and open problems.* Survey "challenges" claims and recorded absences (survey-claimed or
  explicitly stated gaps) are attached to the nearest family (cosine ≥ τ_attach with a shared content word).
- *Comparison relations.* Survey comparison claims, comparison tables and normalized lineage relations
  (extends / improves / compares / replaces / combines) are attached to families the same way.

Every field-level object keeps the verbatim quotes of its sources, so it remains citable evidence.

### 3.3 Answering (per question, read-only)

1. **Plan.** The LLM decomposes the question into 3–6 sections, each with a goal and 2–3 keyword queries.
2. **Retrieve.** For every query, in parallel:
   - *Record retrieval*: hybrid BM25 + dense retrieval over records with reciprocal rank fusion and a per-paper cap,
     so that no single paper dominates;
   - *State retrieval*: the query is matched to the nearest method families; their properties, limitations and comparisons
     are expanded as evidence;
   - *Open retrieval*: semantic scholarly search with a server-side date filter (knowledge cutoff);
   - *Citation expansion*: references of the top open-retrieval hits, ranked by how many hits co-cite them.
3. **Evidence table.** Results are deduplicated by (paper, excerpt) and assigned stable ids per section.
4. **Screen.** For each section, the LLM lists excerpts that are clearly off-topic (same words, different problem);
   ambiguous items are kept, and screening is skipped if it would leave fewer than five items.
5. **Write and assemble.** Each section is written from its evidence table; every sentence must carry evidence ids and stay
   within what the cited excerpts state. A deterministic assembler maps ids to citations whose snippets are the excerpts
   themselves, so every citation is verifiable by construction.

### 3.4 Growth (question-blind)

The same retrieval and extraction modules grow the knowledge base offline: the next acquisition targets are papers that
are frequently cited by papers already in the base but absent from it (citation frontier), plus queries built from
recorded gaps. Growth is frozen (hashed snapshot) before any evaluation; answering never writes to the base.

---

## 4 Experimental setup（要点，写作时展开）

- **Same model for all self-run arms**: Qwen3.8-27B (local); judge DeepSeek-V4.1-Flash replacing the official judge,
  identical for all arms; 30-question second-judge check.
- **Same knowledge cutoff for all arms**, enforced server-side in every retrieval tool (CS2: 2025-05; DSB: the target paper's
  publication month).
- **CS2**: development on dev; final numbers on the untouched test split; official four-facet score, missing answers count 0.
- **Multi-108**: closed corpus; report both the pre-registered citation F1 (out-of-question citations dropped) and a strict
  variant (out-of-question citations counted as false positives); disclose 10 questions whose gold answers are off-topic
  relative to the question text, with a 98-question sensitivity analysis.
- **DSB**: 48 unique questions (the released 66 gold directories contain duplicates); nugget coverage.

## 5 Ablations（预注册，判读规则见叙事档 §3.2）

| Arm | Change |
|---|---|
| Full | as above |
| − state | no state retrieval (record retrieval only) |
| − KB | no records or state; open retrieval + citation expansion only |
| − citation expansion | |
| − screening | |
| − open retrieval | KB only |
