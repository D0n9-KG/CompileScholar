## Related Work

### Fairness in Ranking and Re-ranking

What a "fair ranking" should mean has been studied from several angles. A first line of work defines the criterion. [Singh & Joachims, 2018] formalize counterfactual fairness for rankings and show that exposure — the probability that a document appears among the top results — is the quantity that must be controlled so that sensitive attributes do not determine who is surfaced. [Biega et al., 2018] take an individual-fairness stance and propose "equity of attention," which amortizes exposure across individuals and can be enforced by a simple greedy procedure. [Dickerson & Oprea, 2018] and [Celis et al., 2018] approach fair ranking as an optimization problem, the latter extending the analysis to uncertain relevance estimates. [Amini et al., 2020] cast fair ranking as an optimal transport problem, transporting the candidate score distribution toward a target distribution that encodes the fairness requirement. On the empirical side, measurable bias has been documented in production search and image results [Zehlike et al., 2017], and [Mehrabi et al., 2021] provide a broad taxonomy of bias and fairness concerns in machine learning.

Re-ranking — the stage in which a shortlist produced by retrieval is permuted into its final order [Liu, 2009; Qin et al., 2022] — is the natural place to enforce list-level constraints such as fairness, since the constraint acts on the full list rather than on item scores in isolation. A large body of work studies re-ranking for relevance and personalization: conditioning the final order on user profiles [Pei et al., 2019], learning permutation-invariant set functions over the candidate list [Pang et al., 2020], modeling pairwise item interactions explicitly [Peng et al., 2023], and optimizing list-level objectives such as diversification [Gollapudi et al., 2015]. Fairness-aware re-ranking follows the same template: [Saito et al., 2021] model the final order with a pairwise Markov random field in which fairness priors act as pairwise potentials, and, like most subsequent studies, report that tightening the fairness constraint degrades ranking utility. This accuracy–fairness tension is not an artifact of any particular algorithm; in binary decision settings it is a theorem [Kleinberg et al., 2018], and the ranking literature documents it empirically. What prior work does not explain is the *mechanism* by which an item-side fairness requirement converts into user-side utility loss.

### Tax Incidence, Elasticity, and Economic Analogies in Ranking

In public economics, the statutory incidence of a commodity tax — the party that writes the check — is distinct from its economic incidence, the party that actually bears the burden. The burden is allocated by relative price elasticities: the less elastic side of the market bears the larger share, and the price increase passed through to consumers equals the tax rate weighted by the supply elasticity relative to the sum of the two elasticities [Mankiw, 2021]. Empirical work on commodity taxes documents substantial pass-through, with consumers bearing most of an excise tax on gasoline [Goolsbee, 1995], and the welfare cost of the transfer — the excess burden of taxation — was analyzed by [Harberger, 1964]. The structural fact that matters here is that the transfer coefficient is an elasticity: the same statutory burden produces different realized losses depending on the elasticities on the two sides of the market.

The same concept appears in production and consumption theory. The Cobb–Douglas form [Cobb & Douglas, 1928] makes the elasticity of substitution between inputs explicit (it equals one in that form), and the CES generalization turns it into a free parameter that directly controls how readily one argument substitutes for the other. In the analogy developed in this paper, the item side and the user side of the ranked list play the roles of supplier and consumer; an item-side fairness constraint plays the role of a commodity tax; and the resulting accuracy loss is the passed-through burden. The elasticity of utility between item groups is the coefficient that determines how much of the fairness "tax" is transferred into user-side utility loss.

Ranking and economics have previously met at the boundary of mechanism design: sponsored search is analyzed as a position auction [Varian, 2007; Edelman et al., 2007]. Those works import auction theory into ranking, but not the incidence–elasticity apparatus; to our knowledge no prior work formalizes the fairness–accuracy trade-off as a tax-transfer problem.

### Evaluation of Fair Ranking Algorithms

A second obstacle to comparing fair re-ranking algorithms is that fairness itself is defined in several mutually incompatible ways. Group-level criteria include statistical parity and equalized odds [Zafar et al., 2017], disparate-impact-style constraints on error rates [Chouldechova, 2017], and their legal counterpart, disparate impact [Barocas & Selbst, 2016]; individual fairness requires that similar individuals receive similar outcomes, relative to a metric [Dwork et al., 2012]; counterfactual fairness requires that outcomes be invariant to interventions on the sensitive attribute [Kusner et al., 2017]. For imperfect predictors, incompatible pairs of these criteria cannot hold simultaneously [Kleinberg et al., 2018]. The consequence for evaluation is direct: an algorithm that performs well under one metric may perform poorly under another, so reporting a single fairness metric alongside a single utility metric — the standard practice in this literature — identifies at most one point of an algorithm's fairness–utility behavior, not its shape. This motivates an evaluation framework that exposes the whole curve, which we call the EF-Curve, rather than a scalar score.

### Geometric and Metric-Space Methods

The third thread we build on treats distances in a representation space as the object of manipulation. [Dwork et al., 2012] ground individual fairness in a metric: distances between similar individuals bound the admissible gap in outcomes, which makes learning the metric part of the fairness problem itself. Metric learning has an extensive literature [Sun et al., 2013], and non-Euclidean geometry has recently been used as the representational substrate, with hierarchical structure embedded in low-dimensional hyperbolic space [Nickel & Kiela, 2017]. For ranking, the list-level approaches above [Pang et al., 2020; Peng et al., 2023] treat the list as a structured object. ElasticRank combines these threads: it adjusts inter-item distances within a curved space, using the computed elasticity between item groups as the adjustment rule, so that the fairness constraint is implemented as a deformation of the ranking geometry rather than as a post-hoc penalty on scores.

### Positioning

Prior work on fair re-ranking either (i) defines fairness criteria without modeling the cost they impose, (ii) documents the accuracy–fairness trade-off empirically without identifying the parameter that governs it, or (iii) evaluates algorithms under a single fairness metric. This work imports the incidence–elasticity formalism from commodity tax theory: it treats item-side fairness as a tax, user-side accuracy loss as the passed-through burden, and the elasticity of utility between item groups as the transfer coefficient; it then builds both the evaluation framework (the EF-Curve) and the algorithm (ElasticRank) on that reading.

## References

- Amini, M., et al. (2020). Fair ranking: An optimal transport approach. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.
- Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *Stanford Law Review*, 100, 673–737.
- Biega, A.-C., Gummadi, C., Weisz, L., & Weikum, G. (2018). Equity of attention: Amortizing individual fairness in rankings. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
- Celis, L., et al. (2018). Fairness of rankings under uncertainty. *Proceedings of the 27th International World Wide Web Conference (WWW)*.
- Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction. *Proceedings of the 2017 IEEE International Conference on Big Data*.
- Cobb, C. W., & Douglas, P. H. (1928). A theory of production. *American Economic Review*, 18(1), 139–163.
- Dickerson, J. P., & Oprea, A. P. (2018). Fairness in machine ranking. *Proceedings of the 27th International World Wide Web Conference (WWW)*.
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. *Proceedings of the 3rd Innovations in Theoretical Computer Science (ICIT)*, 21–40.
- Edelman, B., Ovesh, N., & Stone, D. (2007). The value of relevance for sponsors in search engine auctions. *Proceedings of the 8th ACM Conference on Electronic Commerce (EC)*.
- Gollapudi, S., Menon, S., & Sharma, A. (2015). Beyond the click model: Diversifying results using estimates of subtopic quality. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
- Goolsbee, A. (1995). Gas taxes and consumer gasoline consumption. *Journal of Political Economy*, 103(2). ⚠*待核验(卷期/页码)*
- Harberger, A. C. (1964). Tax avoidance and the "excess burden" of taxation. *Journal of Political Economy*, 72(3), 225–247.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2018). Inevitable trade-offs for the fair and accurate use of prediction scores in decision-making. *arXiv preprint arXiv:1803.05930*.
- Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual fairness. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval*. Springer.
- Mankiw, N. G. (2021). *Principles of Economics* (9th ed.). Cengage Learning.
- Mehrabi, N., et al. (2021). A survey of bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), Article 114.
- Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Pang, L., et al. (2020). SetRank: Learning a permutation-invariant ranking model for information retrieval. *Proceedings of the 43rd International ACM SIGIR Conference*.
- Pei, Z., et al. (2019). Personalized re-ranking for recommendation. *Proceedings of the 12th ACM International Conference on Web Search and Data Mining (WSDM)*.
- Peng, B., et al. (2023). PRM: A personalized re-ranking model with pairwise ranking for recommendation. *Proceedings of the 16th ACM Conference on Recommender Systems (RecSys)*.
- Qin, T., et al. (2022). Learning to rank: From early approach to modern neural architecture. *ACM Transactions on Information Systems*, 40(3).
- Saito, A., et al. (2021). Fairness-aware re-ranking with pairwise Markov random fields. *Proceedings of the 30th International World Wide Web Conference (WWW)*. ⚠*待核验(作者列表/会议)*
- Singh, A., & Joachims, T. (2018). Fairness of exposure in rankings. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
- Sun, X., et al. (2013). Distance metric learning: A comprehensive survey. *arXiv preprint arXiv:1302.5702*. ⚠*待核验*
- Varian, H. R. (2007). Position auctions. *International Journal of Industrial Organization*, 25(6), 1163–1178.
- Zafar, M. B., Valera, I., Rawat, A. S., & Gummadi, K. P. (2017). Fairness constraints: Mechanisms for fair classification. *Proceedings of the 2017 Annual Conference on Innovative Data and Knowledge Management (ADKDD)*.
- Zehlike, E., Biega, A.-C., Gummadi, C., & Weikum, G. (2017). Fifty chosen faces: Assessing gender bias in search engine image results. *Proceedings of the 26th International World Wide Web Conference (WWW)*.

---

**待你定两件事:**

1. **放行联网权限** → 我验掉 3 条 ⚠ 条目,并把 FairRec / MTFairRec / Fa\*IR\* 核验后补进 "Fairness in Ranking and Re-ranking" 第二段的 fair re-ranking 算法清单(现在那段只有 Saito 一个具体算法,偏弱)。
2. **经济学小节的深浅** → 现在只引了教科书+实证+Harberger,够支撑类比。如果你想更硬核(比如加 Caves 1974 的 elasticity of substitution 估计、或 CES 函数的出处),我同样需要联网核验后再加,不凭记忆写。