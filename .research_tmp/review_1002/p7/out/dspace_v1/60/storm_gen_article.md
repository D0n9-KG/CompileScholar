## Related Work

**Calibrator Functionality**
Prior research on prediction calibration has largely relied on predefined transformation functions with limited expressiveness to adjust model outputs. A significant body of work employs fixed parametric functions, such as logistic or sigmoid transformations, to map raw scores to probabilities [5, 6]. Other approaches utilize non-parametric or semi-parametric techniques, including isotonic regression [10] and ensembles of basis functions [12]. More recent methods have introduced feature-aware or context-aware mechanisms, such as tree-based binning with linear functions [1], confidence-aware multi-field calibration [7], and field-aware neural networks for post-hoc adjustment [9]. Additionally, some studies constrain the calibration mapping to specific functional forms, such as functions of logit subtraction [14]. In contrast, our work departs from these rigid or fixed-form transformations by proposing an Unconstrained Monotonic Neural Network (UMNN) that learns arbitrary monotonic functions, thereby offering superior flexibility and modeling power for complex calibration scenarios.

**Constraint Mechanism**
Ensuring that calibration does not distort the relative order of predictions is critical for ranking systems, yet prior methods employ varying mechanisms to handle this constraint. Some approaches, such as feature-aware binning [1], lack explicit monotonicity constraints, risking order distortion. Others rely on post-hoc adjustments or field-adaptive mechanisms guided by posterior statistics [9, 10]. Several frameworks address the tension between calibration and ranking through loss function design, including decoupled multi-objective losses [11], value and shape decomposition [12], contextualized hybrid objectives [14], regression-compatible ranking objectives [15], and calibration-compatible listwise distillation [16]. Unlike these methods that enforce constraints via loss functions or post-hoc processing, our approach guarantees order preservation through an architectural monotonicity constraint inherent to the UMNN structure, ensuring that the relative ranking of predictions is strictly maintained during calibration.

**Optimization Objective**
The optimization of calibration models has traditionally focused on minimizing generic error metrics or balancing competing objectives. Common objectives include multi-view calibration losses [1], unbiased empirical risk minimization [6], and standard metrics such as NLL, Brier score, and AUC [9]. In industrial settings, objectives often involve guiding calibration via posterior statistics [10], combining pointwise calibration losses with ranking losses [11], or optimizing multi-field calibration accuracy [12]. Other works propose joint optimization strategies, such as contextualized hybrid discriminative-generative objectives [14], combined regression and ranking objectives to improve the Pareto frontier [15], listwise distillation with calibration compatibility [16], and Bayesian approaches for joint ranking and calibration [17]. To address the specific challenge of optimizing a highly flexible neural calibrator, we introduce the Smooth Calibration Loss (SCLoss), which targets the necessary conditions for achieving an ideal calibration state rather than relying on generic error minimization or decoupled objectives.

**Deployment Context**
The validation of calibration methods varies significantly across deployment contexts, ranging from academic benchmarks to large-scale industrial systems. Several studies evaluate their methods exclusively on offline academic benchmarks [1, 4]. Others focus on specific industrial domains, such as online advertising [9, 10, 12], personalized ranking on real-world datasets [6], industrial online advertising ranking systems [11], CTR prediction with online A/B testing [14], YouTube Search production systems [15], and CTR prediction with privileged features [16]. While some works demonstrate effectiveness in large-scale online video ranking systems, such as those at LinkedIn [8] and for advertisement ranking [7], our method is specifically validated in Kuaishou’s large-scale online video ranking system. This deployment context allows us to demonstrate that the proposed calibration improvements translate directly into enhanced business metrics in a high-stakes, real-world environment.

## References

[1] MBCT: Tree-Based Feature-Aware Binning for Individual Uncertainty
  Calibration
[2] Transforming classifier scores into accurate multiclass probability estimates
[3] Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods
[4] Revisiting the Calibration of Modern Neural Networks
[5] Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers
[6] Obtaining Calibrated Probabilities with Personalized Ranking Models
[7] Confidence-Aware Multi-Field Model Calibration
[8] LiRank: Industrial Large Scale Ranking Models at LinkedIn
[9] Field-aware Calibration: A Simple and Empirically Strong Method for
  Reliable Probabilistic Predictions
[10] Posterior Probability Matters: Doubly-Adaptive Calibration for Neural
  Predictions in Online Advertising
[11] A Self-boosted Framework for Calibrated Ranking
[12] Deep Ensemble Shape Calibration: Multi-Field Post-hoc Calibration in
  Online Advertising
[13] Scale Calibration of Deep Ranking Models
[14] Joint Optimization of Ranking and Calibration with Contextualized Hybrid
  Model
[15] Regression Compatible Listwise Objectives for Calibrated Ranking with
  Binary Relevance
[16] Calibration-compatible Listwise Distillation of Privileged Features for
  CTR Prediction
[17] Beyond Binary Preference: Leveraging Bayesian Approaches for Joint Optimization of Ranking and Calibration