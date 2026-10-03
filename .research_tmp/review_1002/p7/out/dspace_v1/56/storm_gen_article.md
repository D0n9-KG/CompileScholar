## Related Work

### Cross-Domain Linkage Mechanisms
Existing cross-domain sequential recommendation (CDSR) methods primarily address the challenge of transferring knowledge across domains through various structural and statistical mechanisms. A significant portion of prior work relies on shared user identifiers, assuming that users interact with items in all domains to learn cross-domain relationships [9]. To mitigate the reliance on strict overlap, other approaches employ complex sequence modeling or graph-based techniques. For instance, Tri-CDR utilizes triple sequence correlation to jointly model source, target, and mixed behavior sequences [10], while C^2DSR leverages joint learning of intra- and inter-sequence relationships via contrastive infomax [6]. Additionally, graph-based methods have been introduced to handle bias and density issues; EA-GCL employs graph contrastive learning with self-supervised learning for bias removal [7], and HAMUR utilizes dynamic parameter generation via a hyper-network to adapt to domain-specific distributions [3]. However, these methods generally depend on explicit behavioral overlaps or complex structural alignments that may not capture semantic similarities between items. In contrast, our work introduces LLM-based semantic bridging through unified representation and hierarchical profiling, which decouples cross-domain learning from the requirement of overlapping users or items by leveraging the semantic view of user preferences.

### Backbone Architectures
The choice of backbone architecture fundamentally determines the capacity for representation and reasoning in recommendation systems. Recent advancements have seen a shift towards Large Language Models (LLMs) as core engines, with numerous studies exploring their integration into recommendation pipelines [11, 12, 13, 14, 15, 16, 17, 18, 19]. These LLM-based approaches often employ trainable adapters or fine-tuning strategies to align general language capabilities with specific recommendation tasks. In contrast, traditional CDSR methods typically rely on specialized neural architectures tailored for sequential or graph data. For example, C^2DSR utilizes GNNs and sequential attentive encoders [6], while EA-GCL combines a graph encoder with an External Attention sequence encoder [7]. Other non-LLM approaches include parallel information-sharing networks for shared-account scenarios [9] and standard sequential recommendation models such as SASRec or BERT4Rec [10]. While LLMs have shown promise in general recommendation, their specific application to the cross-domain sequential setting, particularly with mechanisms to bridge semantic gaps without heavy reliance on behavioral overlap, remains underexplored. Our model adopts LLMs with trainable adapters as the backbone, distinguishing it from both traditional sequential encoders and general LLM applications by specifically targeting the cross-domain semantic linkage.

### User Preference Modeling Granularity
Modeling user preferences at an appropriate granularity is critical for capturing the complex transition patterns in mixed behavior sequences. Prior work has approached this from various angles, ranging from multi-granularity fusion to domain-specific adaptation. For instance, some methods model preferences at multiple levels, such as inter-sequence to intra-sequence [1] or joint single- and cross-domain representations [6]. Others focus on global behavioral patterns using external attention mechanisms [7] or triple sequence modeling to capture global and target preferences [10]. In the context of LLMs, recent works have explored different granularities of preference extraction, such as Graph of Thoughts reasoning over short-term, long-term, and collaborative interests [17], fine-tuned LLM internal representations [18], or ranking-based preference modeling with negative samples [19]. However, these approaches often process sequences sequentially or rely on implicit representations that may not explicitly summarize the cross-domain nature of user behavior. Our work addresses the "transition complexity" problem by introducing hierarchical LLMs profiling, which explicitly summarizes and structures cross-domain preferences, providing a more coherent semantic view of user intent than existing granular modeling techniques.

### Integration Strategies
The integration strategy determines how effectively semantic information from LLMs or other advanced models is adapted to the specific cross-domain recommendation task. A common approach in LLM-enhanced recommendation is the two-stage pipeline, where features are extracted first and then used to train a sequential recommendation model [14, 15, 22]. Alternatively, some methods opt for end-to-end fine-tuning of the LLM to align it directly with the recommendation objective [18, 19]. In non-LLM CDSR frameworks, integration often involves specialized loss functions or modular designs, such as pluggable modules with hyper-networks [3], contrastive cross-domain infomax objectives [6], multi-task learning with self-supervised learning and external attention [7], or triple contrastive learning strategies [10]. While these methods have shown effectiveness in their respective domains, they do not fully exploit the reasoning capabilities of LLMs within a unified cross-domain framework. Our proposed tri-thread framework with contrastive regularization offers a novel integration strategy that seamlessly combines LLM-based semantic representations with cross-domain sequential modeling, bridging the gap between semantic understanding and behavioral prediction.

## References

[1] A Survey on Cross-Domain Sequential Recommendation
[2] Gromov-wasserstein guided representation learning for cross-domain recommendation
[3] HAMUR: Hyper Adapter for Multi-Domain Recommendation
[4] AutoTransfer: Instance transfer for cross-domain recommendations
[5] Graph neural networks in recommender systems: a survey
[6] Contrastive Cross-Domain Sequential Recommendation
[7] Unbiased and Robust: External Attention-enhanced Graph Contrastive
  Learning for Cross-domain Sequential Recommendation
[8] A multi-view graph contrastive learning framework for cross-domain sequential recommendation
[9] $\pi$-net: A parallel information-sharing network for shared-account cross-domain sequential recommendations
[10] Triple Sequence Learning for Cross-domain Recommendation
[11] How Can Recommender Systems Benefit from Large Language Models: A Survey
[12] Large language models for recommendation: Past, present, and future
[13] Towards Next-Generation LLM-based Recommender Systems: A Survey and
  Beyond
[14] LLMEmb: Large Language Model Can Be a Good Embedding Generator for
  Sequential Recommendation
[15] LLMSeR: Enhancing Sequential Recommendation via LLM-based Data
  Augmentation
[16] Large Language Models: A Survey
[17] GOT4Rec: Graph of Thoughts for Sequential Recommendation
[18] TALLRec: An Effective and Efficient Tuning Framework to Align Large
  Language Model with Recommendation
[19] On Softmax Direct Preference Optimization for Recommendation
[20] Large language model enhanced recommender systems: Taxonomy, trend, application and future
[21] Enhancing sequential recommendation via llm-based semantic embedding learning
[22] A Practice-Friendly Two-Stage LLM-Enhanced Paradigm in Sequential Recommendation
[23] Llm-esr: Large language models enhancement for long-tailed sequential recommendation