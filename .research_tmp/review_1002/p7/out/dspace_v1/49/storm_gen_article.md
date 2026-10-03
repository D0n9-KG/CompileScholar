## Related Work

**Dense Retrieval and Pre-training**
A substantial body of research has focused on optimizing the architecture and training paradigms of dense retrieval models to maximize downstream retrieval effectiveness. Early foundational work demonstrated that dual-encoder frameworks could effectively implement retrieval for open-domain question answering, outperforming sparse baselines [2]. Subsequent efforts have addressed the discrepancy between training and testing data distributions by introducing approximate nearest neighbor indices to select realistic negative instances [1]. To reduce the reliance on heavy data engineering and large batch sizes, unsupervised corpus-aware pre-training methods have been proposed to warm up passage embeddings via contrastive losses [3]. Furthermore, retrieval-oriented pre-training paradigms based on Masked Auto-Encoders with asymmetric masking have been introduced to further improve dense retrieval performance [4]. While these methods share the objective of maximizing ranking quality, they primarily focus on the retrieval model itself rather than the generation of training data.

**Synthetic Data Generation for Retrieval**
To address the scarcity of labeled query-document pairs, recent work has leveraged Large Language Models (LLMs) to generate synthetic queries for data augmentation. Several approaches aim to maximize downstream retrieval effectiveness by using LLMs to generate few-shot queries to train task-specific dense retrievers [11] or to generate synthetic query-document pairs for general information retrieval augmentation [9, 10]. Document expansion techniques, such as Doc2Query and its variants, also utilize query generation to improve retrieval, with some methods specifically designed for efficiency or specific improvements over original approaches [6, 8]. More recently, methods have been proposed to expand documents with predicted queries using sequence-to-sequence models, which improves retrieval effectiveness, particularly when combined with a re-ranking component [5]. Although these works utilize synthetic data to enhance retrieval, they typically treat the generated queries as static inputs or rely on external signals for quality control, rather than optimizing the generation process itself against ranking objectives.

**Optimization of Query Generation and Data Quality**
The optimization of query generation models has evolved from maximizing the likelihood of ground-truth queries to incorporating more complex feedback signals. Some approaches focus on maximizing query-document semantic similarity by distilling knowledge from LLMs through the generation and refinement of synthetic paired data [12]. Other methods employ reinforcement learning with reward models, such as Token-level Proximal Policy Optimization (TPPO), to improve LLM query generation using reinforcement learning from AI feedback [13, 14]. In contrast to these methods that rely on semantic similarity or general reinforcement learning rewards, existing filtering-based approaches often remove noisy query-document pairs based on signals from an external re-ranker [5]. However, no prior work integrates Direct Preference Optimization (DPO) directly into the query generation training loop to iteratively refine noisy synthetic pairs via preference learning, thereby internalizing the ranking objective within the generator rather than relying on post-hoc filtering or separate reward models.

## References

[1] Approximate Nearest Neighbor Negative Contrastive Learning for Dense
  Text Retrieval
[2] Dense Passage Retrieval for Open-Domain Question Answering
[3] Unsupervised Corpus Aware Language Model Pre-training for Dense Passage
  Retrieval
[4] RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked
  Auto-Encoder
[5] Document Expansion by Query Prediction
[6] From doc2query to docTTTTTquery
[7] Attention Is All You Need
[8] Doc2Query-: When Less is More
[9] InPars: Data Augmentation for Information Retrieval using Large Language
  Models
[10] {InPars-v2: Large Language Models as Efficient Dataset Generators for
               Information Retrieval}
[11] Promptagator: Few-shot Dense Retrieval From 8 Examples
[12] Gecko: Versatile Text Embeddings Distilled from Large Language Models
[13] Token-level Proximal Policy Optimization for Query Generation
[14] Proximal Policy Optimization Algorithms