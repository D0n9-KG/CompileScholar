## Related Work

**Theoretical Framework**
Prior research on the accuracy-fairness trade-off in re-ranking has employed diverse theoretical lenses to characterize the underlying mechanisms. A significant body of work frames the problem through game-theoretic equilibrium [5, 17] or specific fairness axioms such as max-min fairness [4]. Other approaches utilize stochastic policy optimization [9] or draw from economic concepts such as utility and exposure distribution [10], generalized Gini welfare functions for inequality measurement [11], and two-sided marketplace optimization [12, 13]. Additionally, some studies model the problem as the fair allocation of indivisible goods [14]. While these frameworks provide valuable insights, they often treat the trade-off as a static conflict or a constraint satisfaction problem. In contrast, this paper adopts an economic elasticity framework, specifically a tax transfer analogy, to explain the dynamic mechanism by which item-side fairness constraints propagate to user-side accuracy loss, a perspective shared only with [8] among the cited works.

**Evaluation Metric**
The evaluation of fair re-ranking algorithms has traditionally relied on standardized benchmarks and specific fairness metrics. For instance, [3] introduces a comprehensive toolkit for standardized evaluation across diverse metrics, while [5] and [14] focus on allocation-based metrics such as Maximin Share (MMS) and Envy-Free up to One item (EF1). Other works evaluate performance using utility metrics like NDCG under fairness constraints [9], analyze individual and group-level trade-offs [10], or measure exposure inequality via the Gini index [11]. Metrics for consumer and producer fairness [12] and two-sided fairness with personalization levels [13] have also been proposed, alongside axiomatic properties such as envy-freeness and Pareto optimality [17]. Notably, [8] evaluates continuity and controllability over accuracy loss. However, no cited prior work employs a dynamic curve-based framework that assesses performance across varying elasticity levels. This paper addresses this gap by introducing the Elastic Fairness Curve (EF-Curve), which facilitates comparative analysis of algorithm performance under different sensitivity levels.

**Algorithmic Mechanism**
Algorithmic approaches to fair re-ranking are predominantly categorized into post-hoc re-ranking via constrained optimization and other specialized techniques. A large group of studies, including [4, 5, 8, 12, 17], utilizes post-hoc re-ranking methods that integrate fairness constraints into an optimization objective. Other mechanisms include toolkits integrating pre-, in-, and post-processing techniques [3], policy-gradient search over stochastic ranking policies [9], vertical allocation-based fair exposure amortizing [10], non-smooth optimization with projection operators [11], two-sided fairness-aware recommendation models [13], and fair allocation-based algorithms [14]. While these methods effectively incorporate fairness, they typically operate within linear or standard constraint-based frameworks. This paper distinguishes itself by proposing ElasticRank, an algorithm that employs elasticity calculations to adjust inter-item distances within a curved space, a mechanism not observed in the cited prior work.

**Causal Interpretation**
The interpretation of the relationship between fairness and accuracy varies across the literature. Many studies view fairness as a hard constraint that inherently conflicts with accuracy [5, 9, 12, 14], or as an axiomatic allocation principle independent of relevance [17]. Others describe the relationship as a balance between ranking relevance and fairness [10], a normative criterion for equality [11], or a trade-off between quality and two-sided fairness [13]. In contrast, this paper provides a specific causal narrative where item-side fairness acts as a "tax" that transfers to user-side accuracy loss. This mechanistic interpretation, which explains *why* accuracy drops rather than just observing the trade-off, aligns with the taxation perspective of [8] but offers a deeper analysis through the lens of elasticity.

## References

[1] Fairness in Recommendation: Foundations, Methods and Applications
[2] A Survey of Research on Fair Recommender Systems
[3] FairDiverse: A Comprehensive Toolkit for Fair and Diverse Information
  Retrieval Algorithms
[4] P-MMF: Provider Max-min Fairness Re-ranking in Recommender System
[5] FairRec: Two-Sided Fairness for Personalized Recommendations in
  Two-Sided Platforms
[6] Multistakeholder Recommendation: Survey and Research Directions
[7] Multi-stakeholder Recommendation and its Connection to Multi-sided
  Fairness
[8] A Taxation Perspective for Fair Re-ranking
[9] Policy Learning for Fairness in Ranking
[10] Vertical Allocation-based Fair Exposure Amortizing in Ranking
[11] Optimizing generalized Gini indices for fairness in rankings
[12] CPFair: Personalized Consumer and Producer Fairness Re-ranking for
  Recommender Systems
[13] TFROM: A Two-sided Fairness-Aware Recommendation Model for Both
  Customers and Providers
[14] Towards Fair Recommendation in Two-Sided Platforms
[15] Fairness Constraints: A Flexible Approach for Fair Classification
[16] The Distribution and Redistribution of Income
[17] Fair Ranking as Fair Division: Impact-Based Individual Fairness in
  Ranking