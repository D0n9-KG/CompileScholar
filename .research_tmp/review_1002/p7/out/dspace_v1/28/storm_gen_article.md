## Related Work

**Heterogeneity Handling in Federated Active Learning**
A significant portion of early active learning research assumes homogeneous data distributions, employing standard uncertainty sampling strategies that ignore client-specific heterogeneity [2, 3]. While foundational federated learning frameworks like FedAvg address non-IID data through iterative model averaging [13, 14], they do not incorporate heterogeneity-aware mechanisms into the data selection process itself. Other approaches have attempted to mitigate heterogeneity through distribution regularization [18], experimental robustness analysis [19], or client selection based on data variability [22], yet these methods often decouple the selection strategy from the specific dynamics of parameter fluctuations. More recent work has begun to explicitly account for client data distribution heterogeneity and resulting parameter fluctuations in data selection [20, 21]. However, unlike these methods, CHASe specifically targets the oscillation of epistemic variations around decision boundaries, providing a more granular response to the instability caused by heterogeneous data distributions.

**Uncertainty Metrics for Data Selection**
Traditional active learning methods rely on standard predictive uncertainty metrics, such as entropy or margin, to identify informative samples [2, 3]. To capture deeper model uncertainty, some studies utilize Bayesian posterior variance via weight uncertainty [5] or dropout-based Bayesian approximation [6]. Alternative metrics include temporal output discrepancy [7], core-set selection based on data geometry [10], and predicted loss [11]. In the federated context, collaborative strategies have been proposed to select informative instances [17], while others leverage discrepancies between local and global models [20] or inter-class diversity [21]. Despite these advancements, no prior work has employed Epistemic Variations (EVs) derived from inference inconsistencies across training epochs as the primary selection criterion. CHASe distinguishes itself by using this temporal inconsistency metric to identify samples that notably oscillate around decision boundaries, a feature not captured by static or single-epoch uncertainty measures.

**Model Calibration Mechanisms**
Most existing active learning and federated learning approaches utilize raw model outputs directly for selection without explicit calibration mechanisms [2, 3]. This lack of calibration can lead to unreliable uncertainty estimates, particularly when local models are inaccurate due to data heterogeneity. While some federated methods employ knowledge-compensatory updates to address model discrepancies [20], they do not introduce a dedicated loss function to calibrate decision boundaries. CHASe addresses this gap by proposing a new alignment loss specifically designed to calibrate the decision boundaries of inaccurate local models. This mechanism ensures that the epistemic variations used for selection are reliable, even in the presence of significant parameter fluctuations, a capability absent in prior cited works.

**Data Selection Efficiency Mechanisms**
Efficiency in active learning is often addressed through sequential sampling [2] or the strategic selection of high-value examples [3]. Other methods improve efficiency via matrix partitioning [8], core-set selection [10], or loss prediction modules [11]. In federated settings, client-level active sampling has been proposed to maximize training efficiency [22]. However, these approaches typically require repeated full evaluations or complex computational overheads. CHASe introduces a data freeze and awaken mechanism with subset sampling, which enhances data selection efficiency by avoiding redundant computations on stable samples. This mechanism is distinct from prior efficiency strategies, as it dynamically manages the active set based on the stability of epistemic variations, offering a novel balance between effectiveness and computational cost.

## References

[1] Heterogeneous uncertainty sampling for supervised learning
[2] A Sequential Algorithm for Training Text Classifiers
[3] Active Learning for Multi-class Image Classification
[4] Active learning: Synthesis lectures on artificial intelligence and machine learning
[5] Weight Uncertainty in Neural Networks
[6] Dropout as a Bayesian Approximation: Representing Model Uncertainty in
  Deep Learning
[7] Semi-Supervised Active Learning with Temporal Output Discrepancy
[8] Active instance sampling via matrix partition.
[9] Active learning using pre-clustering
[10] Active Learning for Convolutional Neural Networks: A Core-Set Approach
[11] Learning Loss for Active Learning
[12] Agreement-Discrepancy-Selection: Active learning with progressive distribution alignment
[13] Federated Optimization:Distributed Optimization Beyond the Datacenter
[14] Communication-Efficient Learning of Deep Networks from Decentralized
  Data
[15] Active learning based federated learning for waste and natural disaster image classification
[16] {FLARE:} Federated active learning assisted by naming for responding
               to emergencies
[17] Federated Active Learning (F-AL): an Efficient Annotation Strategy for
  Federated Learning
[18] Distribution-Regularized Federated Learning on Non-IID Data
[19] Robustness analytics to data heterogeneity in edge computing
[20] Knowledge-Aware Federated Active Learning with Non-IID Data
[21] Re-thinking Federated Active Learning based on Inter-class Diversity
[22] Active Federated Learning