# Self-Evolving Lift Experiments (step3-7)

Evaluation scripts for the self-evolving schema + n-ary hypergraph + low-order
lifts-to-higher-order system. These produce the empirical results for the
top-conference paper.

## Scripts (run from LogicKG root)

| script | what | result |
|---|---|---|
| `build_arfm2024_citations.py` | build ARFM2024 in-corpus citation edges via OpenAlex (side-A API) | 32 edges across 19 papers |
| `eval_step7_ablation.py` | 5-arm ablation: F frozen / T text / C +citation / Q +quals / B +full | F 0/3 gold; citation improves +67%; quals +167% |
| `eval_step5_full_ab.py` | (in .research_tmp) full 19-paper A/B pure-text vs full-signal | improves 5→9, extends 11→5 |
| `eval_step7_nmr.py` | Intern-Atlas-style NMR (node match ratio) via embedding+Hungarian | NMR=0.700 |
| `eval_step7_complex_q.py` | module-1 survey complex questions, frozen vs full | frozen 0/4, full 2/4 |
| `eval_step7_scifact.py` | module-2 SciFact fact-checking, hypergraph vs naive-RAG | hypergraph 0.250 vs RAG 0.333 (domain mismatch, honest) |

## Key findings (honest)

- **Lift is necessary** (F frozen 0/3 gold type coverage vs lifted 3/3).
- **Structural signals make lifting more precise**: citation prior improves
  identification of `improves` (+67%), quals evidence-strength corrects
  over-`extends` (extends 11→5, distinguishing parallel/background from true
  extension). Both signals complementary.
- **Domain boundary (honest)**: in granular-flow domain (method-relation
  reasoning), lift raises complex-question accuracy 0→0.5; in SciFact
  (biomedical fact claims), hypergraph arm (0.250) does not beat naive-RAG
  (0.333) — schema strength is method relations, not fact checking.
- **NMR=0.700**: semantic matching of LLM-named induced methods to gold M-ids
  is feasible (M1/M17 ambiguity to be refined).

Data deps: ARFM2024 runs in `.research_tmp/runs/ARFM2024/`, citations in
`.research_tmp/runs/ARFM2024/citations.json`, SciFact in `.research_tmp/data/`.
