## Related Work

**Sequence Processing Backbone**
The architectural backbone for sequential recommendation has evolved through distinct paradigms, primarily categorized into recurrent, attention-based, and state-space models. Early approaches relied on Recurrent Neural Networks (RNNs) to model sequential dependencies, such as session-based recommendation systems that utilize RNNs to capture short-term user behaviors [11] and hierarchical RNNs that transfer latent states across sessions for personalization [6]. Subsequently, Transformer-based architectures became dominant, leveraging self-attention to capture long-range dependencies; this includes foundational models like SASRec [8] and BERT4Rec [9], as well as variants designed for specific challenges, such as personalized transformers [3], sparse transformers [4], and multi-behavior transformers [5]. To address the quadratic complexity of standard attention, linear attention mechanisms have been proposed to improve efficiency for long-term sequences [10]. More recently, State-Space Models (SSMs) have emerged as a promising alternative, offering linear complexity and efficient variable-length sequence processing. This shift is exemplified by the introduction of Mamba, a selective SSM that outperforms Transformers in various modalities [12], and its systematic surveying as a Transformer alternative [13]. In the recommendation domain, SSMs have been applied to address efficiency issues in sequential recommendation [19], handle lifelong sequences with reduced memory costs [15], and enhance pattern capture through bidirectional gated architectures [18] and spectral filtering [16]. Additionally, SSMs have been adapted for knowledge tracing to improve efficiency and interpretability [17]. While prior work has extensively explored RNNs, Transformers, and SSMs individually, STAR-Rec distinguishes itself by adopting an SSM backbone specifically to synergize with attention mechanisms for handling length variance and pattern diversity.

**Pattern Diversity Handling Mechanism**
A critical challenge in sequential recommendation is capturing the diverse behavioral patterns of users, ranging from focused category-specific browsing to broad exploratory behaviors. The majority of prior works employ a single unified model architecture without explicit routing mechanisms to adapt to these varying patterns. This includes Transformer-based models such as SASRec [8], BERT4Rec [9], and personalized or sparse variants [3, 4, 5], as well as RNN-based approaches [11]. Similarly, recent SSM-based methods, including Mamba [12], its application to lifelong recommendation [15], and bidirectional or spectral-filtered variants [16, 18], typically rely on a single unified model structure. While some works incorporate dynamic memory to segment sequences [7] or use frequency-enhanced attention [1], they do not explicitly route different behavioral patterns to specialized experts at the sequence level. In contrast, STAR-Rec introduces a sequence-level Mixture-of-Experts (MoE) framework that adaptively routes different behavioral patterns to specialized experts, a design choice not present in the cited prior works, thereby explicitly addressing pattern diversity at the sequence granularity.

**Attention Mechanism Design**
The design of attention mechanisms in sequential recommendation varies significantly, with some approaches omitting attention entirely in favor of pure SSMs or RNNs. For instance, RNN-based models [6, 11] and several SSM-based approaches, including Mamba [12], its lifelong application [15], bidirectional variants [16, 18], and knowledge tracing models [17], do not incorporate an explicit attention mechanism. Among models that do use attention, standard self-attention is prevalent in foundational Transformer models like SASRec [8] and BERT4Rec [9]. Other works have modified attention to capture specific characteristics: LinRec proposes a linear attention mechanism to reduce computational complexity [10], DMAN utilizes dynamic memory-based attention to capture long-term interests from segmented sub-sequences [7], and FEARec enhances self-attention with frequency-domain processing to capture periodicity [1]. However, no cited prior work employs a preference-aware attention mechanism specifically designed to capture both inherently similar item relationships and diverse preferences in synergy with SSMs. STAR-Rec fills this gap by introducing preference-aware attention that complements the temporal dynamics captured by the SSM, modeling nuanced item relationships that standard or linear attention mechanisms may miss.

**Theoretical Unification Approach**
The integration of different sequence modeling components in recommendation systems has often been empirical rather than theoretically grounded. For example, FEARec combines frequency-domain processing with attention [1], and SASRec utilizes self-attention for sequential modeling [8], but these works generally present empirical combinations without a unified theoretical framework connecting distinct architectural families. In contrast, recent theoretical advances have sought to bridge these gaps. Specifically, the work on State Space Duality establishes a theoretical framework connecting Transformers and State Space Models, leading to the design of Mamba-2 [14]. This theoretical unification provides a principled basis for understanding how SSMs and attention mechanisms relate. STAR-Rec builds upon this theoretical foundation, explicitly demonstrating how SSMs and attention can be naturally unified in recommendation scenarios, where SSMs capture temporal dynamics through state compression and attention models item relationships, positioning the architecture as a principled approach rather than a mere empirical stacking of components.

## References

[1] Frequency Enhanced Hybrid Attention Network for Sequential
  Recommendation
[2] Deep learning based recommender system: A survey and new perspectives
[3] SSE-PT: Sequential recommendation via personalized transformer
[4] STRec: Sparse Transformer for Sequential Recommendations
[5] Multi-behavior sequential transformer recommender
[6] Personalizing Session-based Recommendations with Hierarchical Recurrent
  Neural Networks
[7] Dynamic Memory based Attention Network for Sequential Recommendation
[8] Self-Attentive Sequential Recommendation
[9] BERT4Rec: Sequential Recommendation with Bidirectional Encoder
  Representations from Transformer
[10] LinRec: Linear Attention Mechanism for Long-term Sequential Recommender
  Systems
[11] Session-based Recommendations with Recurrent Neural Networks
[12] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[13] A Survey of Mamba
[14] Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality
[15] Uncovering Selective State Space Model's Capabilities in Lifelong
  Sequential Recommendation
[16] EchoMamba4Rec: Harmonizing Bidirectional State Space Models with
  Spectral Filtering for Advanced Sequential Recommendation
[17] Mamba4KT:An Efficient and Effective Mamba-based Knowledge Tracing Model
[18] Bidirectional gated mamba for sequential recommendation
[19] Mamba4Rec: Towards Efficient Sequential Recommendation with Selective
  State Space Models