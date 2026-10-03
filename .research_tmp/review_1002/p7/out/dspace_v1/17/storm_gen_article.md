## Related Work

**Outlier Detection Mechanisms**
A predominant line of research in semi-supervised learning (SSL) relies on confidence scores or entropy from a single model to distinguish inliers from outliers, assuming that high-confidence predictions correspond to known classes [1, 2, 3, 6, 7, 9, 10, 22]. While effective under ideal distributional assumptions, this approach often fails when labeled data is insufficient, as the single model may assign high confidence to unknown concepts. To mitigate this, several methods have explored alternative signals for outlier detection. Some leverage consistency between student and teacher models, such as weight-averaged targets [4], or consensus predictions from temporal ensembles of the same model over time [5]. Other approaches utilize distribution alignment and augmentation anchoring [8], graph-based contrastive learning with embedding smoothness constraints [12], or point mutual information (PMI) in representation space to filter unknown categories [16]. More recent works have introduced specific architectural or optimization-based signals, including OOD memory queues for selecting reliable samples [15], joint optimization frameworks that output an OOD probability score [18], cross-modal matching scores [19], semantic overlap via aliasing and soft orthogonality [20], unified open-set classification targets from multi-binary classifiers [23], pseudo-negative mining with non-linear feature separation [24], and transferability metrics relative to in-distribution data [26]. However, none of these prior methods exploit prediction disagreements among multiple differently biased models within a single framework, which is the core mechanism proposed in this work to ensure robustness against label scarcity.

**Model Architecture and Training Strategy**
Most existing SSL and open-set SSL methods employ a single-head model architecture, either with uncertainty estimation [1, 2, 3, 9, 10] or data augmentation-based consistency [6, 7, 8]. Alternative architectural strategies include student-teacher frameworks with weight averaging [4], self-ensembling of a single model over time [5], and joint learning of class probabilities and low-dimensional embeddings [12]. Specific to open-set settings, architectures have been designed to simultaneously handle close-set and open-set self-training with OOD memory queues [15], perform weight-aware distillation from unsupervised contrastive representations [16], or utilize multi-task curriculum learning with joint optimization [18]. Other single-model approaches incorporate warm-up pretext tasks with cross-modal matching [19], aliasing matching modules with soft orthogonality regularization [20], one-vs-all (OVA) classifiers with open-set soft-consistency [22], multi-binary classifiers alongside standard closed-set classifiers [23], non-linear transformations with pseudo-negative mining [24], or adversarial domain adaptation [26]. In contrast to these single-model or dual-model (student-teacher) paradigms, our approach constructs a collection of differently biased models through a single training process using multiple divergent heads. This multi-head ensemble strategy allows us to achieve the benefits of ensemble diversity without the computational overhead of training multiple independent networks, a design choice not present in the cited prior works.

**Bias Induction and Robustness to Label Scarcity**
The majority of prior SSL methods, including those based on pseudo-labeling [1, 2, 3, 6, 7, 9, 10], consistency regularization [4, 5, 8], and graph-based methods [12], do not explicitly induce bias towards the unlabeled distribution, relying instead on natural variance in model predictions. Some open-set SSL methods attempt to address distribution mismatch by treating OOD samples as an additional class to refine the decision boundary [15] or by using adaptive weights based on PMI to filter unknown categories [16]. While methods such as [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 23] aim for robust performance even when labeled data is underspecified, they often still suffer from performance degradation when the labeled set is too small to accurately model the known class distribution, a limitation highlighted by studies showing that naive SSL can be worse than no SSL in such scenarios [13, 14]. Our method explicitly encourages divergent heads to be differently biased towards the unlabeled distribution while maintaining consistency for inliers. This deliberate bias induction ensures that the disagreement signal is specifically sensitive to outliers rather than general uncertainty, providing a robustness advantage over prior methods that rely on implicit or natural variance, particularly in regimes where labeled data is severely insufficient.

## References

[1] Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks
[2] Semi-supervised Learning by Entropy Minimization
[3] Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised
  Learning
[4] Mean teachers are better role models: Weight-averaged consistency
  targets improve semi-supervised deep learning results
[5] Temporal Ensembling for Semi-Supervised Learning
[6] MixMatch: A Holistic Approach to Semi-Supervised Learning
[7] FixMatch: Simplifying Semi-Supervised Learning with Consistency and
  Confidence
[8] ReMixMatch: Semi-Supervised Learning with Distribution Alignment and
  Augmentation Anchoring
[9] FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo
  Labeling
[10] FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning
[11] Graph-Based Semi-Supervised Learning: {A} Comprehensive Review
[12] CoMatch: Semi-supervised Learning with Contrastive Graph Regularization
[13] Realistic Evaluation of Deep Semi-Supervised Learning Algorithms
[14] Semi-Supervised Learning under Class Distribution Mismatch
[15] SCOMatch: Alleviating Overtrusting in Open-set Semi-supervised Learning
[16] Semi-Supervised Learning via Weight-aware Distillation under Class
  Distribution Mismatch
[17] {SAFER-STUDENT} for Safe Deep Semi-Supervised Learning With Unseen-Class
                  Unlabeled Data
[18] Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning
[19] Trash to Treasure: Harvesting OOD Data with Cross-Modal Matching for
  Open-Set Semi-Supervised Learning
[20] Out-of-Distributed Semantic Pruning for Robust Semi-Supervised Learning
[21] Unknown-Aware Graph Regularization for Robust Semi-supervised Learning
                  from Uncurated Data
[22] OpenMatch: Open-set Consistency Regularization for Semi-supervised
  Learning with Outliers
[23] IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint
  Inliers and Outliers Utilization
[24] SSB: Simple but Strong Baseline for Boosting Performance of Open-Set
  Semi-Supervised Learning
[25] Binary Decomposition: A Problem Transformation Perspective for Open-Set Semi-Supervised Learning
[26] They are Not Completely Useless: Towards Recycling Transferable
  Unlabeled Data for Class-Mismatched Semi-Supervised Learning