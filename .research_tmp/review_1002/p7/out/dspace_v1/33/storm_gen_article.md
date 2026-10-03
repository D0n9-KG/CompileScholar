## Related Work

**Rehearsal Strategies in Continual Learning**
A significant body of literature addresses catastrophic forgetting through rehearsal mechanisms, which can be broadly categorized into concurrent, generative, and non-rehearsal approaches. Concurrent rehearsal, the standard paradigm in recent empirical studies, involves training on a subset of past data alongside new data in the current task; this strategy is central to methods such as iCaRL [7] and Gradient Episodic Memory (GEM) [8], as well as in theoretical analyses of replay-based learning [17]. In contrast, generative rehearsal reduces memory requirements by synthesizing samples from previous tasks using deep generative models, as proposed in Deep Generative Replay [5]. A distinct line of work avoids data storage entirely, relying on regularization or gradient constraints to preserve prior knowledge; examples include Elastic Weight Consolidation (EWC) [2], Gradient Projection Memory (GPM) [3], Trust Region Gradient Projection (TRGP) [4], A-GEM [6], and Progressive Neural Networks [1]. While these methods have demonstrated empirical success, prior work has predominantly assumed that concurrent rehearsal is the optimal or standard strategy for replay-based methods [7, 8, 17]. This paper challenges that assumption by introducing and analyzing Sequential Rehearsal, where past data is revisited sequentially after new data training, and a Hybrid Rehearsal method that dynamically selects between concurrent and sequential strategies based on task similarity, a dimension unoccupied by prior cited works.

**Theoretical Analysis of Forgetting and Generalization**
Theoretical understanding of continual learning has evolved from empirical benchmarking to rigorous mathematical characterization. Early and many recent works, including those on EWC [2], GPM [3], TRGP [4], Deep Generative Replay [5], iCaRL [7], and GEM [8], primarily rely on empirical evaluations to demonstrate performance improvements without providing closed-form theoretical guarantees. In contrast, a growing body of literature provides theoretical analyses to explain the mechanisms behind forgetting and generalization. These theoretical studies include works deriving generalization bounds in the Neural Tangent Kernel (NTK) regime for Orthogonal Gradient Descent [9], analyzing catastrophic forgetting via the NTK overlap matrix [10], formulating regularization-based learning through Taylor approximations [11], and studying linear classification on separable data [12]. More recently, several papers have focused on overparameterized linear models to derive explicit expressions for forgetting and generalization error, such as [13, 14, 15, 16, 17]. While these studies share the goal of theoretical characterization, this paper provides the first comprehensive theoretical analysis specifically comparing concurrent versus sequential rehearsal strategies, explicitly characterizing how the choice of rehearsal order impacts forgetting and generalization in overparameterized settings.

**Model Assumptions and Regimes**
The choice of model assumption significantly influences the tractability and insights of theoretical analyses in continual learning. Many foundational studies rely on deep neural networks, often treating them as black boxes or using empirical approximations, as seen in [1, 2, 3, 4, 5, 7, 8]. To enable rigorous analysis, other works adopt specific theoretical regimes: some operate within the Neural Tangent Kernel (NTK) framework [9, 10], while others use second-order Taylor approximations for general neural networks [11]. A distinct group focuses on linear models, with some analyzing standard linear regression [12] or both underparameterized and overparameterized regimes [15]. The most relevant prior theoretical works, including [13, 14, 16, 17], specifically analyze overparameterized linear models, a regime that allows for closed-form solutions to optimization problems. This paper aligns with the latter group by employing overparameterized linear models to derive its theoretical results, ensuring that the insights regarding sequential and hybrid rehearsal are grounded in a mathematically tractable framework that captures the effects of overparameterization on catastrophic forgetting.

**Handling Task Similarity**
The treatment of task similarity varies across continual learning methods, ranging from task-agnostic approaches to those that explicitly leverage similarity metrics. Many standard methods apply a uniform strategy regardless of the relationship between tasks, such as EWC [2], GPM [3], Deep Generative Replay [5], iCaRL [7], GEM [8], and linear classification analyses [12, 13, 15]. Other approaches handle similarity implicitly through regularization [11] or by using random orthogonal transformations to model task dissimilarity [16]. Some methods utilize proxies for task similarity, such as NTK overlap [9, 10], or employ task-specific architectures like Progressive Neural Networks [1]. A subset of recent theoretical and empirical works explicitly conditions their strategies on task similarity, including TRGP [4], which uses trust regions to select related tasks, and theoretical analyses in [14, 17] that examine how task relationships affect forgetting. This paper extends this line of work by deriving that the optimal rehearsal strategy—concurrent versus sequential—is explicitly conditioned on task similarity. Based on these theoretical insights, we propose a Hybrid Rehearsal method that dynamically switches strategies, training similar tasks concurrently and revisiting dissimilar tasks sequentially, thereby optimizing performance across diverse task distributions.

## References

[1] Progressive Neural Networks
[2] Overcoming catastrophic forgetting in neural networks
[3] Gradient Projection Memory for Continual Learning
[4] TRGP: Trust Region Gradient Projection for Continual Learning
[5] Continual Learning with Deep Generative Replay
[6] Efficient Lifelong Learning with A-GEM
[7] iCaRL: Incremental Classifier and Representation Learning
[8] Gradient Episodic Memory for Continual Learning
[9] Generalisation Guarantees for Continual Learning with Orthogonal
  Gradient Descent
[10] A Theoretical Analysis of Catastrophic Forgetting through the NTK
  Overlap Matrix
[11] Optimization and Generalization of Regularization-Based Continual
  Learning: a Loss Approximation Viewpoint
[12] Continual Learning in Linear Classification on Separable Data
[13] How catastrophic can catastrophic forgetting be in linear regression?
[14] Theory on Forgetting and Generalization of Continual Learning
[15] Understanding Forgetting in Continual Learning with Linear Regression
[16] Analysis of Catastrophic Forgetting for Random Orthogonal Transformation
  Tasks in the Overparameterized Regime
[17] Theoretical Insights into Overparameterized Models in Multi-Task and
  Replay-Based Continual Learning