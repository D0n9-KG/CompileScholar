## Related Work

**Theoretical Analysis of Generative Retrieval Limitations**
Prior research on generative retrieval has predominantly relied on empirical benchmarking to evaluate performance on standard datasets [5, 6, 9, 10, 11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 34] or has focused on heuristic algorithmic improvements to enhance efficiency and accuracy [3, 4, 7, 28, 29, 31, 32, 33]. While a few studies have provided systematic surveys [1] or conceptual syntheses of system design [2], and others have offered theoretical properties alongside empirical validation [27], the majority of work lacks rigorous error bounds. A smaller subset of literature has begun to address theoretical derivations of error bounds [12, 15, 25, 26, 30], yet these often focus on generalization error under specific verification or inference constraints rather than the intrinsic mechanics of constrained decoding. In contrast, this paper provides a theoretical derivation of error bounds specifically for the constrained auto-regressive decoding paradigm, isolating the impact of decoding constraints from model capacity limitations.

**Model Assumptions and Distributional Properties**
Most existing generative retrieval frameworks assume a finite-capacity neural network with inherent approximation error [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 29, 31, 32], while others operate under the assumption of pre-trained LLMs with fixed weights [2, 17, 27, 28, 33, 34]. Some approaches incorporate general machine learning models with verifier modules [25]. However, the assumption of a Bayes-optimal model that exactly captures the underlying relevance distribution is rare, with only [30] sharing this specific design choice. By adopting a Bayes-optimal setting, this paper distinguishes itself from prior work that conflates model approximation errors with decoding limitations, thereby allowing for a clearer analysis of how constraints affect retrieval performance independent of model capacity.

**Focus on Decoding Mechanics and Constraints**
The majority of prior work focuses on external factors or system-level challenges, such as the scalability of index construction [3, 15, 29, 31], out-of-distribution robustness [19, 20], or the definition and semantic quality of document identifiers [4, 9, 10]. Other studies address specific limitations like the learning gap between generation and ranking [6], data distribution mismatches [7], binary relevance data [11], forced hierarchical structures [13], or the difficulty of updating models with new documents [17, 18, 23, 24]. While some works examine the disconnection between retrieval and generative tasks [33, 34] or the limited information capacity of parametric models [21, 22], few explicitly target the inherent constraints of auto-regressive decoding mechanics, such as the interplay between constraints and beam search. This paper aligns with a small group of studies [27, 28, 30, 32] that focus on these inherent decoding constraints, but it uniquely derives lower bounds on error specifically for the constrained auto-regressive generation process.

**Generalization to Out-of-Distribution Corpora**
Existing literature largely focuses on in-distribution performance optimization [3, 4, 5, 6, 10, 11, 13, 15, 16, 29, 30, 31, 32, 33, 34] or specific generalization scenarios such as cross-lingual transfer [7], zero-shot retrieval [14, 21], and continual learning over dynamic corpora [18, 23, 24]. While [19, 20] have analyzed out-of-distribution robustness, they primarily do so through empirical studies on query variations and corpus expansion. This paper shares the focus on out-of-distribution corpora via corpus-specific constraints with [19, 20], but it advances the field by providing a theoretical analysis of how adding these constraints affects generalization, rather than relying solely on empirical observations of robustness.

## References

[1] From Matching to Generation: A Survey on Generative Information
  Retrieval
[2] Rethinking Search: Making Domain Experts out of Dilettantes
[3] Autoregressive Entity Retrieval
[4] Learning to Tokenize for Generative Retrieval
[5] A Neural Corpus Indexer for Document Retrieval
[6] Learning to Rank in Generative Retrieval
[7] Bridging the Gap Between Indexing and Retrieval for Differentiable
  Search Index with Query Generation
[8] Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines
[9] Auto Search Indexer for End-to-End Document Retrieval
[10] Semantic-Enhanced Differentiable Search Index Inspired by Learning
  Strategies
[11] Generative Retrieval Meets Multi-Graded Relevance
[12] Generative Retrieval as Multi-Vector Dense Retrieval
[13] Autoregressive Search Engines: Generating Substrings as Document
  Identifiers
[14] Transformer Memory as a Differentiable Search Index
[15] Generative Retrieval as Dense Retrieval
[16] Scalable and Effective Generative Information Retrieval
[17] Generative Retrieval with Few-shot Indexing
[18] Continual Learning for Generative Retrieval over Dynamic Corpora
[19] On the Robustness of Generative Information Retrieval Models
[20] On the Robustness of Generative Retrieval Models: An Out-of-Distribution Perspective
[21] Nonparametric Decoding for Generative Retrieval
[22] Generative Dense Retrieval: Memory Can Be a Burden
[23] IncDSI: Incrementally Updatable Document Retrieval
[24] {DSI}++: Updating Transformer Memory with New Documents
[25] Generalization Analysis on Learning with a Concurrent Verifier
[26] Understanding the impact of introducing constraints at inference time on generalization error
[27] Guaranteed Generation from Large Language Models
[28] Tractable Control for Autoregressive Language Generation
[29] Constructing Tree-based Index for Efficient and Effective Dense
  Retrieval
[30] Learning Optimal Tree Models Under Beam Search
[31] Joint Optimization of Tree-based Index and Deep Model for Recommender
  Systems
[32] Planning Ahead in Generative Retrieval: Guiding Autoregressive
  Generation through Simultaneous Decoding
[33] CorpusLM: Towards a Unified Language Model on Corpus for
  Knowledge-Intensive Tasks
[34] UniGen: A Unified Generative Framework for Retrieval and Question
  Answering with Large Language Models