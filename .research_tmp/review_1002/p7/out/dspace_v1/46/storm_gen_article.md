## Related Work

**Mechanism of Cross-Domain Integration**
Existing approaches to cross-domain sequential recommendation (CDSR) primarily enhance performance by augmenting standard transformer architectures with explicit external components. A significant body of work focuses on adding domain-aware module blocks to capture specific cross-domain interactions; for instance, DTCDR employs a dual-target framework [14], while C^2DSR utilizes contrastive learning with GNNs and attentive encoders [16], and other methods integrate specific domain-specific modules [21, 22]. Alternatively, some methods apply post-hoc domain alignment losses to transfer user embeddings across domains [15], or employ cooperative learning strategies that utilize adaptive loss weighting and mutual information maximization to manage gradient flow [17]. In contrast, our work argues that these additional components often overlook the inherent potential of the core self-attention mechanism. We propose enhancing the self-attention module directly via Pareto-optimal multi-objective optimization, a strategy not previously explored in the cited literature, thereby leveraging the transformer’s native ability to learn behavioral correlations without relying on external auxiliary blocks.

**Knowledge Transfer Strategy**
Prior research on CDSR employs various strategies to manage the transfer of knowledge between domains, often relying on static assumptions or explicit feature sharing. Some methods utilize explicit feature sharing without dynamic gating mechanisms [15, 16], while others adopt adaptive weighting based on estimated negative transfer and mutual information maximization [17]. Other approaches focus on dynamic selection of cross-domain information based on user preferences via mixed information flow [21], or fuse local and global attention patterns using item similarity and group-prototype attention [22]. Unlike these methods, which may require manual tuning or static configurations to determine which domains are helpful, our approach automates knowledge transfer. We formulate cross-domain learning as a multi-objective problem that dynamically minimizes cross-domain attention scores to mitigate negative transfer and encourage complementary exchange, a mechanism distinct from the static or heuristic-based transfer strategies found in prior work.

**Model Complexity and Deployment Cost**
The computational efficiency and deployment complexity of CDSR models vary significantly across existing literature. Several studies focus on reducing the computational cost of the attention mechanism itself, such as replacing quadratic self-attention with linear scaling [9], employing sparse-interest and interest aggregation modules [10], or utilizing low-rank decomposition of self-attention [11]. However, other CDSR-specific frameworks introduce substantial complexity through multi-stage training pipelines [16] or by requiring the training of both single-domain and cross-domain models alongside asymmetric cooperative networks [17]. Our proposal differs by offering a plug-and-play module that integrates into existing recommenders with little extra computational overhead and without the need for heavy hyper-parameter tuning. This lightweight design contrasts with the complex multi-stage or multi-model training requirements of prior CDSR frameworks, making our approach more practical for deployment.

**Base Model Architecture**
The architectural foundations of sequential recommendation models have evolved from recurrent and matrix factorization techniques to transformer-based architectures. Early works relied on recurrent neural networks for session-based recommendations [1, 2] or attention/memory-based models [3], while others utilized hybrid encoders with attention [4] or matrix factorization and kNN approaches [5]. More recent advancements include sequential models with multi-interest embeddings [10], generative pretrained language models [12], and collective matrix factorization [13]. While some CDSR methods employ hybrid architectures with non-attention backbones [16] or claim model-agnostic applicability [22], our work is specifically designed to enhance existing Transformer-based recommenders such as SASRec [7] and Bert4Rec [8]. We build upon the standard transformer architecture [6] and its variants [9, 11], positioning our contribution as a direct enhancement to the self-attention component of these widely adopted models rather than a replacement of the underlying backbone.

## References

[1] Recurrent recommender networks
[2] Session-based Recommendations with Recurrent Neural Networks
[3] STAMP: short-term attention/memory priority model for session-based recommendation
[4] Neural Attentive Session-based Recommendation
[5] BPR: Bayesian Personalized Ranking from Implicit Feedback
[6] Attention Is All You Need
[7] Self-Attentive Sequential Recommendation
[8] BERT4Rec: Sequential Recommendation with Bidirectional Encoder
  Representations from Transformer
[9] Longformer: The Long-Document Transformer
[10] Sparse-Interest Network for Sequential Recommendation
[11] Lighter and better: low-rank decomposed self-attention networks for next-item recommendation
[12] M6-rec: Generative pretrained language models are open-ended recommender systems
[13] Relational learning via collective matrix factorization
[14] Dtcdr: A framework for dual-target cross-domain recommendation
[15] One for all, all for one: Learning and transferring user embeddings for cross-domain recommendation
[16] Contrastive Cross-Domain Sequential Recommendation
[17] Pacer and Runner: Cooperative Learning Framework between Single- and
  Cross-Domain Sequential Recommendation
[18] Multi-Domain Sequential Recommendation via Domain Space Learning
[19] MDMTRec: An Adaptive Multi-Task Multi-Domain Recommendation Framework
[20] $\pi$-net: A parallel information-sharing network for shared-account cross-domain sequential recommendations
[21] Mixed Information Flow for Cross-domain Sequential Recommendations
[22] Mixed Attention Network for Cross-domain Sequential Recommendation