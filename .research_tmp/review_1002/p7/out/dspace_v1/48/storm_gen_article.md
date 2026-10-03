## Related Work

### Model Specificity and Language Adaptation
Prior research in information retrieval has predominantly relied on general-purpose multilingual models or language-agnostic sparse methods, often overlooking the specific linguistic challenges of low-resource languages. A significant body of work focuses on multilingual general-purpose models, such as Multilingual BERT (mBERT) and XLM-R, which aim to capture cross-lingual transfer capabilities [5, 7, 8, 9, 10, 12, 13, 14]. These models are frequently evaluated on their ability to handle diverse languages, with studies analyzing the impact of tokenization on monolingual performance [12] and the viability of such models for low-resourced languages with limited data [10]. In contrast, traditional approaches utilize language-agnostic sparse retrieval techniques, including probabilistic frameworks like BM25 and context-aware term weighting, which do not leverage deep semantic representations [1, 2, 3, 4]. While some works explore cross-lingual transfer via unsupervised dense retrieval [11] or monolingual general-purpose dense models [6], these often fail to account for the morphological richness and data scarcity specific to languages like Amharic. Our work stands distinct by introducing language-specific dense retrieval models tailored exclusively for Amharic, addressing the suboptimal tokenization and data scarcity issues that limit the effectiveness of general multilingual baselines [15, 17, 18].

### Backbone Architecture
The choice of backbone architecture significantly influences retrieval performance, particularly in low-resource settings where pre-training data is scarce. Most prior studies employ general multilingual transformers, such as XLM-R and mBERT, as the foundation for dense retrieval models, assuming that broad pre-training across hundreds of languages provides sufficient linguistic coverage [5, 7, 8, 9, 10, 12, 13, 14]. Alternatively, some approaches rely on sparse statistical models like TF-IDF and BM25 [1, 4] or neural sparse models such as SPLADE, which learns sparse lexical representations with document expansion [2, 3]. Other works utilize monolingual transformers [6] or generic neural networks for dense retrieval [11], but these often lack the specialized linguistic features required for morphologically rich languages. In contrast to these generic or sparse backbones, our approach utilizes Amharic-specific pre-trained transformers (BERT and RoBERTa) as the backbone. This choice directly addresses the tokenization and data scarcity gaps identified in the abstract, allowing the model to better capture the unique morphological structures of Amharic, a design choice shared only with recent efforts to develop pre-trained embedding models specifically for this language [15].

### Interaction Mechanism
The mechanism used to compute similarity between queries and documents is a critical design dimension in retrieval systems. The majority of prior work employs dense vector similarity via bi-encoder architectures, where queries and documents are encoded into fixed-length vectors and compared using dot product or cosine similarity [5, 6, 11, 13, 14, 15]. Other studies rely on sparse term matching, where relevance is determined by the overlap of terms between the query and document, often enhanced by learned weights or expansions [1, 2, 3, 4]. While these two paradigms dominate the literature, there is a notable gap in the application of late interaction mechanisms for low-resource languages like Amharic. No cited prior work utilizes a ColBERT-based late interaction model for this specific context. Our paper introduces a ColBERT-based late interaction retrieval model, which computes relevance by matching query and document tokens individually before aggregating the scores. This approach allows for finer-grained semantic matching, enabling our model to achieve the highest MRR@10 score among all evaluated models, thereby demonstrating the superiority of late interaction over standard dense or sparse methods in this low-resource setting.

### Model Efficiency and Scale
Efficiency and model scale are increasingly important considerations for deploying retrieval systems, especially in resource-constrained environments. Prior work has largely focused on full-size monolingual models [5, 6] or large-scale multilingual models with hundreds of millions of parameters, such as Arctic Embed 2.0 (568M parameters) and other variants [7, 8]. Some recent efforts have explored efficiency through techniques like Matryoshka Representation Learning [13] or by offering variable model sizes (Small/Base/Large) to allow for trade-offs between performance and computational cost [14]. However, there is a gap in the literature regarding compact, parameter-efficient models that specifically outperform larger multilingual baselines in low-resource contexts. Our work addresses this by proposing compact variants, such as RoBERTa-Base-Amharic-Embed (110M parameters) and RoBERTa-Medium-Amharic-Embed (42M parameters). These models are over 13x smaller than the strongest multilingual baseline yet achieve significant improvements in MRR@10 and Recall@10, highlighting the importance of language-specific adaptation for achieving both high performance and deployment efficiency.

## References

[1] The Probabilistic Relevance Framework: {BM25} and Beyond
[2] SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking
[3] SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
[4] Context-aware term weighting for first stage passage retrieval
[5] Dense Passage Retrieval for Open-Domain Question Answering
[6] Approximate Nearest Neighbor Negative Contrastive Learning for Dense
  Text Retrieval
[7] How multilingual is Multilingual BERT?
[8] Unsupervised Cross-lingual Representation Learning at Scale
[9] SERENGETI: Massively Multilingual Language Models for Africa
[10] Small Data? {No} Problem! {Exploring} the Viability of Pretrained Multilingual Language Models for Low-resourced Languages
[11] Unsupervised Dense Information Retrieval with Contrastive Learning
[12] How Good is Your Tokenizer? On the Monolingual Performance of
  Multilingual Language Models
[13] Arctic-Embed 2.0: Multilingual Retrieval Without Compromise
[14] Multilingual E5 Text Embeddings: A Technical Report
[15] The Development of Pre-processing Tools and Pre-trained Embedding Models for {A}mharic
[16] Morphologically annotated {Amharic} text corpora
[17] {2AIRTC}: The {Amharic} Adhoc Information Retrieval Test Collection
[18] An Amharic News Text classification Dataset