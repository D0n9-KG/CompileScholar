## Related Work

**Federated Online Learning to Rank**
Research in Federated Online Learning to Rank (FOLTR) has primarily focused on developing novel algorithms to handle decentralized, implicit feedback data while preserving privacy. Several studies have proposed specific methodologies to improve ranking performance in this setting, such as using evolution strategies [1, 3] or analyzing the impact of non-IID data distributions on system stability [5]. Other works have addressed security concerns within FOLTR, specifically analyzing untargeted poisoning attacks and corresponding defense mechanisms [6], or investigating how to effectively remove clients from the global model, often measured via the impact of poisoning attacks on the ranker [14]. While these works establish the foundational infrastructure and security challenges of FOLTR, they do not systematically evaluate the efficacy of unlearning strategies using comprehensive metrics for both under- and over-unlearning. In contrast, this paper focuses on a comparative reproducibility study of existing unlearning strategies within the FOLTR domain, aiming to provide verifiable insights into their performance rather than proposing new ranking algorithms.

**Unlearning in Federated Learning**
The development of unlearning methods in federated settings has largely been driven by the need to satisfy "right to be forgotten" regulations, with most prior work proposing novel algorithms for efficient data removal. A significant body of literature addresses unlearning in general federated learning contexts, such as image classification, through techniques like rapid retraining with formal convergence analysis [10], knowledge distillation to restore model performance [9], or local unlearning followed by few rounds of federated learning [11]. Other approaches target specific domains, including federated recommendation systems using log-based rollback [12] and federated knowledge graph embedding using mutual knowledge distillation [13]. Additionally, some works focus on centralized online learning to rank, introducing differentiable unbiased methods [4] or online neural ranking models [7], but do not address the federated unlearning aspect. Unlike these studies, which typically introduce new unlearning mechanisms, this paper conducts a comparative reproducibility study of five existing strategies, distinguishing itself by rigorously verifying their capabilities rather than proposing new theoretical constructs.

**Evaluation Metrics and Verification**
A critical limitation in prior unlearning research is the reliance on single or narrow evaluation metrics, which often fail to capture the full spectrum of unlearning effectiveness. Many studies evaluate unlearning through specific security or performance lenses, such as backdoor attack success rates [9], poisoning attack impact [14], or model utility and efficiency [10]. Others focus on theoretical aspects, such as regret bounds and empirical comparisons against baselines [7], or formal convergence and complexity analyses [10]. In the context of FOLTR, evaluations have often been limited to measuring the impact of client removal via poisoning attacks [14] or analyzing performance under specific data distribution settings [5]. While some works, such as [11], compare unlearning methods against a gold standard retraining baseline, and [13] uses link prediction and knowledge forgetting metrics, the majority of prior work does not employ a multi-metric assessment that simultaneously covers both under-unlearning and over-unlearning. This paper addresses this gap by adopting a systematic analysis of unlearning capabilities in a verifiable manner, utilizing adapted and newly proposed metrics to provide a comprehensive assessment that prior single-metric approaches have lacked.

## References

[1] Federated online learning to rank with evolution strategies
[2] Federated online learning to rank with evolution strategies: a reproducibility study
[3] Effective and privacy-preserving federated online learning to rank
[4] Differentiable Unbiased Online Learning to Rank
[5] Is Non-IID Data a Threat in Federated Online Learning to Rank?
[6] An Analysis of Untargeted Poisoning Attack and Defense Methods for
  Federated Online Learning to Rank Systems
[7] Learning Neural Ranking Models Online from Implicit User Feedback
[8] Federaser: Enabling efficient client-level data removal from federated learning models
[9] Federated Unlearning with Knowledge Distillation
[10] The Right to be Forgotten in Federated Learning: An Efficient
  Realization with Rapid Retraining
[11] Federated Unlearning: How to Efficiently Erase a Client in FL?
[12] Federated Unlearning for On-Device Recommendation
[13] Heterogeneous Federated Knowledge Graph Embedding Learning and
  Unlearning
[14] How to Forget Clients in Federated Online Learning to Rank?
[15] Manipulating the Byzantine: Optimizing Model Poisoning Attacks and
                  Defenses for Federated Learning