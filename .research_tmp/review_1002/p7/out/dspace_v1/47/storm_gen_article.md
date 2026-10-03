## Related Work

**Domain Selection Strategy**
Existing multi-domain recommendation (MDR) approaches generally struggle with the Negative Transfer Problem (NTP) due to rigid or unselective strategies for leveraging cross-domain knowledge. A significant portion of prior work adopts an unselective transfer strategy, where information is indiscriminately shared from all available domains to the target, often leading to performance degradation when domain conflicts exist [9, 11]. Alternatively, some methods employ a static, global selection mechanism that identifies a fixed set of source domains for all targets, failing to account for the varying needs of individual domains [8]. Other approaches explore task grouping based on cooperation and competition analyses to determine which tasks should be learned together [10]. In contrast, this paper introduces a dynamic, per-domain selection principle (SDSP) that explicitly measures domain-level gaps to dynamically select similar domains for each specific target, a strategy not previously explored in the cited literature.

**Domain Relationship Modeling**
The modeling of relationships between domains varies significantly across existing literature, ranging from implicit alignment to explicit metric-based analysis. Many MDR methods rely on implicit modeling through shared embedding spaces without defining explicit distance metrics, assuming that proximity in the latent space suffices for transfer [2, 7, 9]. Other works attempt to quantify these relationships more directly; for instance, some employ explicit measurement of domain-level gaps using distance measures such as Gromov-Wasserstein distances or embedding alignment via random walks [4, 8]. Additionally, distinct approaches include the explicit analysis of task cooperation and competition [10] or the performance-based estimation of negative transfer degrees [11]. This paper aligns with the latter group by proposing a novel prototype-based distance measure to explicitly model the complexity of relationships between domains, providing a quantitative basis for the selection mechanism.

**Transfer Structure**
The architectural structure used to facilitate knowledge transfer is a critical design dimension in MDR. The majority of existing methods adopt a single, fixed structure for transferring knowledge across all domains, such as star topologies with shared central parameters [9], multi-view disentangled frameworks with gated decoders [7], or cooperative learning frameworks with adaptive loss weighting [11, 8]. While some recent works have moved toward partitioned network structures based on task affinity to optimize time-accuracy trade-offs [10], they still rely on static partitions. This paper distinguishes itself by proposing a dynamic, variable transfer relationship where the set of source domains changes per target domain, allowing the model to adapt its structural dependencies based on the specific characteristics of each domain rather than enforcing a uniform transfer architecture.

**Integration Methodology**
From an implementation perspective, prior work can be categorized by the computational overhead and architectural changes required for integration. Most MDR methods require end-to-end training of new, monolithic multi-domain models, which can be computationally expensive and difficult to retrofit into existing systems [7, 8, 9, 11]. In contrast, a few approaches, such as PEPNet, offer lightweight, plug-and-play modules that can be incorporated into existing recommendation frameworks with minimal overhead [2]. This paper shares this lightweight integration philosophy, positioning SDSP as a simple and dynamic module that can be easily incorporated with existing MDR methods to improve performance without introducing excessive time overheads, thereby emphasizing practicality and ease of adoption.

## References

[1] MAMDR: A model agnostic learning framework for multi-domain recommendation
[2] PEPNet: Parameter and Embedding Personalized Network for Infusing with
  Personalized Prior Information
[3] A unified framework for multi-domain ctr prediction via large language models
[4] Gromov-wasserstein guided representation learning for cross-domain recommendation
[5] AutoTransfer: Instance transfer for cross-domain recommendations
[6] Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations
[7] MDAP: A Multi-view Disentangled and Adaptive Preference Learning
  Framework for Cross-Domain Recommendation
[8] Multi-domain Recommendation with Embedding Disentangling and Domain
  Alignment
[9] One Model to Serve All: Star Topology Adaptive Recommender for
  Multi-Domain CTR Prediction
[10] Which Tasks Should Be Learned Together in Multi-task Learning?
[11] Pacer and Runner: Cooperative Learning Framework between Single- and
  Cross-Domain Sequential Recommendation