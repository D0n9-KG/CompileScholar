## Related Work

**Information Loss Mitigation in Binarized Representations**
Existing approaches to efficient recommendation via binarization generally fall into two categories regarding the handling of quantization-induced information loss. One group focuses on standard binarization or low-loss quantization techniques without explicit mechanisms to recover lost semantic information, such as binary graph neural networks [20] and low-loss 1-bit quantization methods [18]. Another group attempts to address this through optimization strategies or post-hoc corrections, including the use of continuation methods for learning binary hash codes [11], re-ranking candidates with real-valued models after binary retrieval [15], and multi-faceted quantization reinforcement [21]. In contrast, our work explicitly mitigates information loss at various stages of embedding binarization by leveraging supervisory signals from pseudo-positive samples, a strategy that differs from both standard quantization and reinforcement-based approaches by directly addressing the discrepancy between real and latent representations.

**Sources of Supervisory Signals for Binarization**
The source of supervisory signals used to guide the binarization process varies significantly across prior literature. Many foundational collaborative filtering and graph-based methods rely exclusively on real positive and negative interaction pairs to train their models [3, 4, 6, 7, 8]. Other approaches utilize latent embedding distillation, where the supervision is derived solely from the alignment of student and teacher embeddings [25, 26, 29]. Distinct from these, some methods employ alternative signals such as imbalanced similarity data [11], preference-preserving binary embedding learning [15], ensemble predictions [22], or teacher GCN distributions [24]. Our proposed framework, BiGeaR++, distinguishes itself by utilizing a hybrid source of supervision: pseudo-positive samples that combine both real item data and synthesized latent embedding samples, thereby providing a more robust signal for mitigating binarization errors than methods relying on either real data or latent distillation alone.

**Granularity of Knowledge Distillation**
Knowledge distillation has been applied to graph neural networks with varying levels of granularity. A significant portion of early graph collaborative filtering and hashing works does not employ distillation at all [3, 4, 5, 6, 7, 8]. Among those that do, most focus on feature-level distillation, aligning node embeddings or local structures between teacher and student models [24, 25, 26, 29]. Some methods employ coarse-grained output distillation, such as compressing ensemble predictions into a single student model [22]. Our work introduces a fine-grained inference distillation mechanism, which operates at a more detailed level than feature-level or output-level distillation, specifically targeting the inference process to enhance the quality of binarized representations.

**Embedding Sample Synthesis**
The synthesis of embedding samples is a relatively underexplored dimension in prior graph collaborative filtering and hashing literature. Most existing methods, including standard graph collaborative filtering approaches [3, 4, 5, 6, 7, 8] and various knowledge distillation frameworks for GNNs [25, 26, 29], do not involve the synthesis of embedding samples for supervisory purposes. While some hashing and retrieval methods utilize transformed samples for self-distillation [31] or hierarchical message aggregation [32], they do not specifically address the synthesis of latent embedding samples to enhance binarization in collaborative filtering. Our framework incorporates an effective embedding sample synthesis approach, enabling the generation of latent samples that serve as critical supervisory signals, a capability not present in the majority of prior works cited.

## References

[1] Deep neural networks for youtube recommendations
[2] Graph convolutional neural networks for web-scale recommender systems
[3] BPR: Bayesian Personalized Ranking from Implicit Feedback
[4] Neural Collaborative Filtering
[5] Semi-Supervised Classification with Graph Convolutional Networks
[6] Graph Convolutional Matrix Completion
[7] Neural Graph Collaborative Filtering
[8] Disentangled Graph Collaborative Filtering
[9] LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation
[10] Similarity search in high dimensions via hashing
[11] HashNet: Deep Learning to Hash by Continuation
[12] Discrete collaborative filtering
[13] Discrete personalized ranking for fast collaborative filtering from implicit feedback
[14] Learning binary codes with neural collaborative filtering for efficient recommendation systems
[15] Candidate Generation with Binary Codes for Large-Scale Top-N
  Recommendation
[16] Learning to hash with GNNs for recommender systems
[17] A Comprehensive Survey on Graph Neural Networks
[18] Towards low-loss 1-bit quantization of user-item representations for top-k recommendation
[19] Bi-gcn: Binary graph convolutional network
[20] Binary Graph Neural Networks
[21] Learning Binarized Graph Representations with Multi-faceted Quantization
  Reinforcement for Top-K Recommendation
[22] Distilling the Knowledge in a Neural Network
[23] Knowledge Distillation on Graphs: A Survey
[24] Distilling Knowledge from Graph Convolutional Networks
[25] On Representation Knowledge Distillation for Graph Neural Networks
[26] Distilling Holistic Knowledge with Graph Neural Networks
[27] Knowledge distillation improves graph structure augmentation for graph neural networks
[28] Fine-grained learning behavior-oriented knowledge distillation for graph neural networks
[29] FreeKD: Free-direction Knowledge Distillation for Graph Neural Networks
[30] Semi-supervised knowledge distillation for cross-modal hashing
[31] Deep Hash Distillation for Image Retrieval
[32] Teacher-student learning: Efficient hierarchical message aggregation hashing for cross-modal retrieval