## Related Work

**User Simulation Mechanism**
Prior research on user simulation in conversational recommendation systems (CRSs) has primarily relied on direct human-in-the-loop interaction to capture preferences, as seen in frameworks that model deep interaction between conversational and recommender components [2, 3]. In contrast, recent advancements have shifted toward automated simulation using reward models; specifically, discriminative reward models have been employed to provide scalar scores for candidate selection [10], while generative reward models (GRMs) have emerged to unify instruction-based feedback mechanisms such as scoring and critique [11, 12]. Although [11, 12] share the use of generative reward models, this paper distinguishes itself by designing a unified simulated user that integrates both coarse-grained generative item scoring and fine-grained attribute-based item critique within a single instruction-tuned framework, thereby enabling more nuanced preference capture than scalar-only or single-modality approaches.

**Interaction Search Strategy**
The management of the multi-turn interaction space varies significantly across existing methods. Some approaches utilize graph-based path reasoning to navigate the recommendation process [2], while others employ Thompson Sampling to handle exploration-exploitation trade-offs in cold-start scenarios [3]. In the domain of reward-guided selection, prior work has largely adopted Best-of-N strategies, where the highest-ranked candidate is selected from multiple generations based on verifier scores [10, 12]. However, no cited prior work employs beam search guided by generative rewards for the interaction process. This paper introduces this specific search strategy to balance effectiveness and efficiency, drawing inspiration from reward-guided search paradigms in complex reasoning tasks to optimize the multi-turn dialogue trajectory.

**Feedback Granularity**
The granularity of feedback provided during interaction is a critical factor in understanding complex user preferences. Existing methods have utilized attribute-level feedback to explicitly model user interests [2], implicit bandit feedback for automatic preference learning [3], or scalar reward scores for simple ranking [10]. Other approaches have focused on synthetic preference labels [11] or verification rationales to guide model behavior [12]. Unlike these single-modality or implicit feedback mechanisms, this paper proposes a dual-level feedback structure that unifies coarse-grained generative item scoring with fine-grained attribute-based item critique. This combination allows the simulated user to provide both holistic evaluations and specific attribute-based justifications, addressing the limitations of methods that rely on a single feedback modality.

**Training Data Source**
The source of training data significantly impacts the scalability and transferability of CRS models. Several studies have leveraged real-world human-annotated dialogue datasets to train their models [2], while others have utilized pre-trained general-purpose large language models without specific CRS tuning [8]. In the context of simulation and reward modeling, recent works have increasingly relied on synthesized data for instruction tuning, including frameworks that generate balanced datasets to alleviate long-tail problems [6] and those that train generative reward models on self-generated reasoning traces or synthetic preference labels [11, 12]. This paper aligns with the latter group by utilizing synthesized data for instruction tuning, which facilitates the development of a scalable simulated user without the high costs and variability associated with collecting real multi-turn conversational recommendation data.

## References

[1] Estimation-action-reflection: Towards deep interaction between conversational and recommender systems
[2] Interactive Path Reasoning on Graph for Conversational Recommendation
[3] Seamlessly Unifying Attributes and Items: Conversational Recommendation
  for Cold-Start Users
[4] Towards unified conversational recommender systems via knowledge-enhanced prompt learning
[5] Improving conversational recommendation systems via counterfactual data simulation
[6] Alleviating the Long-Tail Problem in Conversational Recommender Systems
[7] Broadening the view: Demonstration-augmented prompt learning for conversational recommendation
[8] Large language models as zero-shot conversational recommenders
[9] Unleashing the Retrieval Potential of Large Language Models in Conversational Recommender Systems
[10] Training Verifiers to Solve Math Word Problems
[11] Generative Reward Models
[12] Generative Verifiers: Reward Modeling as Next-Token Prediction