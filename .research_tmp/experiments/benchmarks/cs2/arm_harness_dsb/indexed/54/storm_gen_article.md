## Related Work

### Federated Learning and Learning-to-Rank

Federated learning (FL) is a paradigm for training machine-learning models collaboratively across a set of decentralized data holders, so that raw data never leaves its local source while the shared model still benefits from the aggregate signal [McMahan et al., 2017]. Since its introduction, FL has matured into a broad research area, with substantial work cataloguing its challenges and methods [Li et al., 2020; Kairouz et al., 2021]. A central motivation in the search and recommendation literature is to bring this paradigm to learning-to-rank (LTR), where ranking models are traditionally trained on large pools of user-interaction logs. By distributing training across users' devices or organizations, federated LTR aims to improve search-ranking quality from implicit feedback — such as clicks — while keeping each user's interaction history private.

The underlying LTR task is itself well studied. The field is commonly organized around pairwise methods, exemplified by RankNet [Burges et al., 2005], and listwise methods, exemplified by ListNet [Cao et al., 2007], with the LambdaRank family combining listwise objectives with efficient gradient computation [Burges, 2010] and treated comprehensively in [Liu, 2009]. Because search ranking is driven by implicit click feedback rather than explicit labels, a large body of work addresses learning-to-rank from biased, partially observed feedback, including counterfactual and unbiased estimators [Swaminathan & Joachims, 2015; Joachims et al., 2017]. Federated online LTR (FOLTR) as studied here sits at the intersection of these threads: it couples the privacy-preserving, decentralized training of FL with the online, click-feedback-driven setting of LTR [REF: FOLTR].

### The Right to be Forgotten and Machine Unlearning

Regulatory developments have added a further requirement on such systems. Legislation including the EU General Data Protection Regulation [European Union, 2018] and the California Consumer Privacy Act [California, 2018] enshrines a right to erasure — the so-called "right to be forgotten" — under which an individual may request that their personal data be removed. For machine-learning services this extends to the models themselves: a user should be able to have their contribution to a model's training excised, not merely their raw record. This has motivated the area of machine unlearning, which seeks to remove the influence of a specified data item from an already-trained model. Early work on unlearning [Cao & Yang, 2015] and subsequent formalizations [Bourtoule et al., 2021] established the problem and a range of solution strategies. Broadly, unlearning methods fall on a spectrum from the "golden" baseline of retraining the model from scratch without the target data — accurate but expensive at scale — to more efficient approaches that avoid full retraining, such as parameter-level or model-editing schemes [Bourtoule et al., 2021] and influence- or data-attribution-based methods built on influence functions [Koh & Liang, 2017] [REF: UNLEARN-SURVEY].

### Federated Unlearning

Unlearning in a federated setting introduces challenges with no direct analogue in centralized training. In FL, the global model is a composition of many clients' updates, and the central server never observes any single client's raw data. Removing a particular user's contribution therefore cannot be achieved by simply retraining on a filtered local dataset at the server; it requires reasoning about how that client's updates have been folded into the shared parameters. Federated unlearning studies how to provide a verifiable "forget" capability in this distributed setting while limiting the impact on other clients' data and on overall model utility [REF: FED-UNLEARN]. This setting is the substrate for the present work, which targets unlearning strategies specifically within FOLTR.

### Evaluating Unlearning: Under- and Over-Unlearning

A persistent difficulty in unlearning research is evaluation: determining whether a "forget" operation actually succeeded. Relying on a single metric is increasingly seen as insufficient, because unlearning involves a tension between two failure modes. *Under-unlearning* leaves residual influence of the removed data in the model, creating a privacy risk; *over-unlearning* removes too much and degrades the model's utility on retained data. A comprehensive assessment therefore needs to measure both the extent of forgetting and the preservation of utility, typically benchmarked against a retrained reference model. Prior proposals of unlearning methods, however, have often been reported against a single evaluation metric [REF: UNLEARN-EVAL]. The present study addresses this gap: it assesses five unlearning strategies in FOLTR using both adapted and newly proposed metrics, and explicitly characterizes each strategy's behaviour across under- and over-unlearning scenarios, in order to inform the privacy–performance trade-off in federated unlearning.

---

## References

*(These are from my model knowledge and are high-confidence, but please double-check author lists, venues, and years before submission — I could not run live verification this session.)*

- Bourtoule, L., et al. (2021). "Machine Unlearning." *IEEE Symposium on Security and Privacy (S&P)*.
- Burges, C. J. C. (2010). "From RankNet to LambdaRank to LambdaMART: An Overview." *Microsoft Research Technical Report*.
- Burges, C. J. C., Shaked, T., Renshaw, E., Lazier, A., Deeds, J., Hamilton, T., & Hullender, S. (2005). "Learning to Rank using Gradient Descent." *ICML*.
- Cao, J., & Yang, J. (2015). *[TITLE TO VERIFY — foundational machine unlearning]*.
- Cao, Y., Qin, T., Liu, T.-Y., Tsai, M.-F., & Li, H. (2007). "Learning to Rank: From Pairwise Approach to Listwise Approach." *ICML*.
- California. (2018). *California Consumer Privacy Act (CCPA)*.
- European Union. (2018). *General Data Protection Regulation (GDPR), Regulation (EU) 2016/679*.
- Joachims, T., Swaminathan, A., & de Rijke, M. (2017). *Unbiased Learning-to-Rank with Biased Feedback*. MIT Press.
- Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in Machine Learning*.
- Koh, P. W., & Liang, P. (2017). "Understanding Black-box Predictions via Influence Functions." *ICML*.
- Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). "Federated Learning: Challenges, Methods, and Future Directions." *IEEE Signal Processing Magazine*, 37(3).
- Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval*. Springer.
- McMahan, B., Moore, E., Ramage, D., Hampson, B., & y Arcas, B. A. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*.
- Swaminathan, A., & Joachims, T. (2015). "Counterfactual Risk Minimization: Learning from Logged Bandit Feedback." *ICML*.

## Citations to verify / complete

These are the slots I deliberately left unfilled rather than guess. Each maps to a `[REF: …]` marker in the text:

- **[REF: FOLTR]** — the direct prior work on *federated learning to rank* / *federated online LTR* (the closest predecessors to this paper's setting). Search: `"federated learning to rank"`, `"federated online learning to rank"`, `FOLTR`.
- **[REF: FED-UNLEARN]** — specific *federated unlearning* methods/frameworks. Search: `"federated unlearning"`, `"unlearning in federated learning"`.
- **[REF: UNLEARN-EVAL]** — unlearning evaluation/benchmarking, especially work discussing *under-* vs *over-unlearning*. Search: `"machine unlearning benchmark"`, `"evaluating machine unlearning"`, `"under-unlearning over-unlearning"`.
- **[REF: UNLEARN-SURVEY]** — a recent machine-unlearning survey to anchor the taxonomy. Search: `"machine unlearning survey"`.
- **[Cao & Yang, 2015]** — my least-confident single citation; verify the exact title and confirm it's the canonical first unlearning paper.

---

**要你说人话的部分 / 需要你定的：**

1. **为什么留了占位符**：联网（搜索/抓取）这个会话没开权限，我没法逐条对 arXiv/DBLP 核验。按你"假引文靠一手直查揪出"的纪律，我宁可留空 `[REF: …]` 也不编。
2. **最该先核的一条**：`Cao & Yang 2015`（unlearning 起源）——年份/作者我有把握，但**确切标题记不准**，这条最可能出错，建议第一个核。
3. **我没读到你的 repo**：所以正文里"five unlearning strategies"是泛指的，具体是哪五个、各对应哪篇来源论文，你比我清楚——补 `[REF: FOLTR]` 和 `[REF: FED-UNLEARN]` 时顺便对一下。
4. **下一步你选一个**：
   - **给我开联网权限**（WebSearch/WebFetch）→ 我把上面 5 个占位符逐条查成真实引文（走一手 ID 核验），并顺手核掉 `Cao & Yang` 那条；
   - 或者你直接把你们草稿里已有的这几篇 FOLTR/联邦遗忘文献发我，我填进占位符并统一格式。

要我开权限直接去核验并补全吗？