# ResearchQA Downstream Validation — SOTA Baseline Survey + Plan

## The benchmark (ResearchQA, arXiv 2607.11074v1, Khoj Inc / OpenPaper)
Citation-grounded QA on scientific papers. Local data: .research_tmp/researchqa/data/eval_dataset.jsonl
- **6,211 QA across 10 domains**; qtypes: lookup 1999 / comprehension 1999 / adversarial 1221 /
  **multi_hop 992** (our target — cross-section reasoning, where rich-topology should help).
- **multi_hop spans 471 unique papers** (PDF on assets.openpaper.ai S3; DOI given).
- Each row: paper_id, paper_doi, paper_s3_url, domain, question_type, question, expected_answer,
  expected_references, judge_rubric. **metadata_source_text is EMPTY for multi_hop** — full
  paper text must be acquired (S3 PDF or DOI → OA).
- expected_references: list of {section_label, alternatives:[verbatim text snippets]} (2-3 per Q).

## Scoring metrics (deterministic + LLM-eval, measured independently)
**Deterministic (citation-grounding):**
- **citation_precision** = fraction of model's citations matching some alternative in a required section (useful cites).
- **section_coverage** = fractional satisfaction of required sections (AND across sections, OR within alternatives) — did the model touch EVERY required section. ← rich-topology's leverage point (multi_hop = cross-section).
- **citation_accuracy** = fraction of model's citations that are verifiable substrings of paper raw text (after normalization) — did it fabricate quotes.
- **refusal_correctness** (adversarial only).

**LLM-eval (1–5 scale, anchored):** factual_accuracy, completeness. (These largely saturate; the
deterministic citation metrics discriminate — so we focus there.)

## Community SOTA baselines (Table 1, 100-row sample, citation-grounding scores)
| Model | Cite.Prec | Sect.Cov | Cite.Acc | Refusal | Factual | Compl | Lat(s) |
|-------|-----------|----------|----------|---------|---------|-------|--------|
| gemini-3.1-pro-preview | 0.723 | **0.955** | 0.817 | 0.870 | 5.00 | 4.94 | 18.3 |
| gpt-5.4 | 0.647 | 0.902 | 0.834 | 0.783 | 4.97 | 4.87 | 12.3 |
| zai-glm-4.7 | **0.744** | 0.776 | 0.840 | 0.783 | 4.91 | 4.88 | 5.6 |
| gpt-4.1 | 0.776 | 0.755 | 0.786 | 0.870 | 4.91 | 4.73 | 10.9 |
| claude-opus-4-7 | 0.569 | 0.876 | 0.809 | 0.739 | 4.98 | 4.98 | 30.4 |
| claude-haiku-4-5 | 0.716 | 0.843 | **0.845** | 0.739 | 4.80 | 4.66 | 9.8 |
| gemini-3-flash | 0.491 | 0.913 | 0.835 | 0.652 | 4.99 | 4.94 | 14.8 |
| gpt-oss-120b | 0.660 | 0.636 | 0.700 | 0.478 | 4.75 | 4.57 | 2.9 |

**Discrimination is in citation metrics (evaluator saturates).** Largest spread:
section_coverage 0.636–0.955 (32pt), refusal 0.478–0.870, cite_prec 0.491–0.776, cite_acc 0.700–0.845.

## Our comparison layering (per goal: not just beat bare LLM)
1. **Bare-LLM floor** (deepseek, no retrieval) — the "no schema" lower bound.
2. **Naive chunk RAG** (deepseek + top-k source chunks) — standard RAG, no structure.
3. **Our method** = extract_hypergraph over full paper → hypergraph retrieval → deepseek answers
   with citations from retrieved hyperedges. The rich-topology (cross-pattern shared entities,
   dependency/constraint/composition edges) should help multi_hop cross-section questions find
   the right sections → higher section_coverage.
4. **Anchors**: the Table 1 frontier models (gemini-3.1-pro 0.955 sect_cov etc.) are the published
   SOTA ceiling; we are deepseek-only (much smaller), so absolute numbers won't match frontier —
   the claim is RELATIVE: our-hypergraph > naive-RAG > bare-LLM on section_coverage for multi_hop,
   holding the LLM fixed. Beating a frontier model is not the bar; demonstrating the retrieval
   structure's contribution (controlled by the same LLM) is.

## Key design decision: control the LLM
Since we're deepseek-only (memory constraint), all arms use deepseek-chat. The variable is the
RETRIEVAL/context: (a) none, (b) flat chunks, (c) hypergraph. This isolates the hypergraph's
contribution — the honest, defensible comparison. We cite the Table 1 frontier as external
reference (what SOTA looks like) but our contribution is the controlled A/B/C.

## Feasibility blockers to resolve first
1. **Paper acquisition**: 471 multi_hop papers. metadata_source_text empty for multi_hop. Need
   full text. Options: (a) S3 PDF fetch (assets.openpaper.ai — curl got no response, may need
   proper headers / Playwright), (b) DOI → unpaywall/crossref OA (we have unpaywall infra per
   pdf-acquisition-channels). Must resolve before any retrieval arm.
2. **Cost**: 471 papers × extract_hypergraph (multi-chunk LLM) is large. Pilot on ~20 papers /
   ~40 multi_hop Qs first, prove the retrieval contribution, then scale.
3. **Citation grounding**: model must emit verbatim paper-text citations (citation_accuracy is
   substring match). Our hyperedge evidence_span is verbatim — natural fit. Retrieval returns
   hyperedges; answer cites their evidence_span.

## Phased plan
- **Phase 0** (now): acquire a small multi_hop paper subset's full text (S3 or DOI-OA), parse.
  Prove extract_hypergraph runs on a full ResearchQA paper (not just fragments).
- **Phase 1**: build 3-arm harness (bare-LLM / naive-RAG / hypergraph) on ~20 papers / ~40 Qs,
  all deepseek. Score citation_precision / section_coverage / citation_accuracy.
- **Phase 2**: if hypergraph shows section_coverage lift on multi_hop, scale to more papers.
- **Phase 3**: write up with Table 1 as external SOTA reference + our controlled A/B/C.

## Honest scoping
- Deepseek-only → we won't claim absolute SOTA. Contribution = retrieval structure's lift on
  multi_hop section_coverage (controlled). This is a methodology contribution, not a leaderboard win.
- If S3 + DOI-OA both fail for too many papers, fallback to the lookup/comprehension Qs (which
  DO have metadata_source_text) for a smaller but grounded pilot — but multi_hop is the real test.
