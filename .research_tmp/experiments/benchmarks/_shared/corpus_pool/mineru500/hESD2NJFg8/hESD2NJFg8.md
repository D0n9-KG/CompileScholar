# LABEL-FREE NODE CLASSIFICATION ON GRAPHS WITH LARGE LANGUAGE MODELS (LLMs)

Zhikai Chen $^{1}$ , Haitao Mao $^{1}$ , Hongzhi Wen $^{1}$ , Haoyu Han $^{1}$ , Wei Jin $^{2}$ , Haiyang Zhang $^{3}$ , Hui Liu $^{1}$ , Jiliang Tang $^{1}$

$^{1}$ Michigan State University $^{2}$ Emory University $^{3}$ Amazon.com {chenzh85, haitaoma, wenhongz, hanhaoy1, liuhui7, tangjili}@msu.edu, wei.jin@emory.edu, hhaiz@amazon.com

# ABSTRACT

In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN on text-attributed graphs from various domains. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset OGBN-PRODUCTS with a cost less than 1 dollar. Our code is available from https://github.com/CurryTang/LLMGNN.

# 1 INTRODUCTION

Graphs are prevalent across multiple disciplines with diverse applications (Ma & Tang, 2021). A graph is composed of nodes and edges, and nodes often with certain attributes, especially text attributes, representing properties of nodes. For example, in the OGBN-PRODUCTS dataset (Hu et al., 2020b), each node represents a product, and its corresponding textual description corresponds to the node's attribute. Node classification is a critical task for graphs which aims to assign labels to unlabeled nodes based on a part of labeled nodes, node attributes, and graph structures. In recent years, Graph Neural Networks (GNNs) have achieved superior performance in node classification (Kipf & Welling, 2016; Hamilton et al., 2017; Veličković et al., 2017). Despite the effectiveness of GNNs, they always assume the ready availability of ground truth labels as a prerequisite. Particularly, such assumption often neglects the pivotal challenge of procuring high-quality labels for graph-structured data: (1) given the diverse and complex nature of graph-structured data, human labeling is inherently hard; (2) given the sheer scale of real-world graphs, such as OGBN-PRODUCTS (Hu et al., 2020b) with millions of nodes, the process of annotating a significant portion of the nodes becomes both time-consuming and resource-intensive.

Compared to GNNs which require adequate high-quality labels, Large Language Models (LLMs) with massive knowledge have showcased impressive zero-shot and few-shot capabilities, especially for the node classification task on text-attributed graphs (TAGs) (Guo et al., 2023; Chen et al., 2023; He et al., 2023a). Such evidence suggests that LLMs can achieve promising performance with-

out the requirement for any labeled data. However, unlike GNNs, LLMs cannot naturally capture and understand informative graph structural patterns (Wang et al., 2023a). Moreover, LLMs can not be well-tuned since they can only utilize limited labels due to the limitation of input context length (Dong et al., 2022). Thus, though LLMs can achieve promising performance in zero-shot or few-shot scenarios, there may still be a performance gap between LLMs and GNNs trained with abundant labeled nodes (Chen et al., 2023). Furthermore, the prediction cost of LLMs is much higher than that of GNNs, making it less scalable for large datasets such as OGBN-ARXIV and OGBN-PRODUCTS (Hu et al., 2020b).

In summary, we make two primary observations: (1) Given adequate annotations with high quality, GNNs excel in utilizing graph structures to provide predictions both efficiently and effectively. Nonetheless, limitations can be found when adequate high-quality annotations are absent. (2) In contrast, LLMs can achieve satisfying performance without high-quality annotations while being costly. Considering these insights, it becomes evident that GNNs and LLMs possess complementary strengths. This leads us to an intriguing question: Can we harness the strengths of both while addressing their inherent weaknesses?

In this paper, we provide an affirmative answer to the above question by investigating the potential of harnessing the zero-shot learning capabilities of LLMs to alleviate the substantial training data demands of GNNs, a scenario we refer to as label-free node classification. Notably, unlike the common assumption that ground truth labels are always available, noisy labels can be found when annotations are generated from LLMs. We thus confront a unique challenge: How can we ensure the high quality of the annotation without sacrifice diversity and representativeness? On one hand, we are required to consider the design of appropriate prompts to enable LLMs to produce more accurate annotations. On the other hand, we need to strategically choose a set of training nodes that not only possess high-quality annotations but also exhibit informativeness and representativeness, as prior research has shown a correlation between these attributes and the performance of the trained model (Huang et al., 2010).

To overcome these challenges, we propose a label-free node classification on graphs with LLMs pipeline, LLM-GNN. Different from traditional graph active node selection (Wu et al., 2019; Cai et al., 2017), LLM-GNN considers the node annotation difficulty by LLMs to actively select nodes. Then, it utilizes LLMs to generate confidence-aware annotations and leverages the confidence score to further refine the quality of annotations as post-filtering. By seamlessly blending annotation quality with active selection, LLM-GNN achieves impressive results at a minimal cost, eliminating the necessity for ground truth labels. Our main contributions can be summarized as follows:

1. We introduce a new label-free pipeline LLM-GNN to leverage LLMs for annotation, providing training signals on GNN for further prediction.   
2. We adopt LLMs to generate annotations with calibrated confidence, and introduce difficulty-aware active selection with post filtering to get training nodes with a proper trade-off between annotation quality and traditional graph active selection criteria.   
3. On the massive-scale OGBN-PRODUCTS dataset, LLM-GNN can achieve 74.9% accuracy without the need for human annotations. This performance is comparable to manually annotating 400 randomly selected nodes, while the cost of the annotation process via LLMs is under 1 dollar.

# 2 PRELIMINARIES

In this section, we introduce text-attributed graphs and notation utilized in our study. We then review two primary pipelines on node classification. The first pipeline is the default node classification pipeline to evaluate the performance of GNNs (Kipf & Welling, 2016), while it totally ignores the data selection process. The second pipeline further emphasizes the node selection process, trying to identify the most informative nodes as training sets to maximize the model performance within a given budget.

Our study focuses on Text-Attributed Graph (TAG), represented as $\mathcal{G}_{T} = (\mathcal{V}, \mathbf{A}, \mathbf{T}, \mathbf{X})$ . $V = \{v_{1}, \cdots, v_{n}\}$ is the set of n nodes paired with raw attributes $T = \{t_{1}, t_{2}, \ldots, t_{n}\}$ . Each text attributes can then be encoded as sentence embedding $X = \{x_{1}, x_{2}, \ldots, x_{n}\}$ with the help of SentenceBERT (Reimers & Gurevych, 2019). The adjacency matrix $A \in \{0, 1\}^{n \times n}$ represents graph connectivity where $A[i, j] = 1$ indicates an edge between nodes i and j. Although our study puts more emphasis on TAGs, it has the potential to be extended to more types of graphs through methods like Liu et al. (2023) and Zhao et al. (2023).

Traditional GNN-based node classification pipeline. assumes a fixed training set with ground truth labels $y_{V_{train}}$ for the training set $V_{train}$ . The GNN is trained on those graph truth labels. The well-trained GNN predicts labels of the rest unlabeled nodes $V \setminus V_{train}$ in the test stage.

Traditional graph active learning-based node classification. aims to select a group of nodes $\mathcal{V}_{\mathrm{act}} = \mathcal{S}(\mathbf{A}, \mathbf{X})$ from the pool V so that the performance of GNN models trained on those graphs with labels $y_{V_{act}}$ can be maximized.

Limitations of the current pipelines. Both pipelines above assume they can always obtain ground truth labels (Zhang et al., 2021c; Wu et al., 2019) while overlooking the intricacies of the annotation process. Nonetheless, annotations can be both expensive and error-prone in practice, even for seemingly straightforward tasks (Wei et al., 2021). For example, the accuracy of human annotations for the CIFAR-10 dataset is approximately 82%, which only involves the categorization of daily objects. Annotation on graphs meets its unique challenge. Recent evidence (Zhu et al., 2021a) shows that the human annotation on graphs is easily biased, focusing nodes sharing some characteristics within a small subgraph. Moreover, it can be even harder when taking graph active learning into consideration, the improved annotation diversity inevitably increase the difficulty of ensuring the annotation quality. For instance, it is much easier to annotate focusing on a few small communities than annotate across all the communities in a social network. Considering these limitations of existing pipelines, a pertinent question arises: Can we design a pipeline that can leverage LLMs to automatically generate high-quality annotations and utilize them to train a GNN model with promising node classification performance?

# 3 METHOD

To overcome the limitations of current pipelines for node classifications, we propose a new pipeline Label-free Node Classification on Graphs with LLMs, short for LLM-GNN. It (1) adopts LLMs that demonstrate promising zero-shot performance on various node classification datasets (Chen et al., 2023; He et al., 2023a) as the annotators; and (2) introduces the (difficulty-aware) active selection and optional filtering strategy to get training nodes with high annotation quality, representativeness, and diversity simultaneously.

# 3.1 AN OVERVIEW OF LLM-GNN

The proposed LLM-GNN pipeline is designed with four flexible components as shown in Figure 1: (difficulty-aware) active node selection, (confidence-aware annotations), optional post-filtering, and GNN model training and prediction. Compared with the original pipelines with ground truth label, the annotation quality of LLM provides a unique new challenge. (1) The active node selection phase is to find a candidate node set for LLM annotation. Despite only considering the diversity and representativeness (Zhang et al., 2021c) as the original baseline, we pay additional attention to the influence on annotation quality. Specifically, we incorporate a difficulty-aware heuristic that correlates the annotation quality with the feature density. (2) With the selected node set, we then utilize the strong zero-shot ability of LLMs to annotate those nodes with confidence-aware prompts. The confidence score associated with annotations is essential, as LLM annotations (Chen et al., 2023), akin to human annotations, can exhibit a certain degree of label noise. This confidence score can help to identify the annotation quality and help us filter high-quality labels from noisy ones. (3) The optional post-filtering stage is a unique step in our pipeline, which aims to remove low-quality annotations. Building upon the annotation confidence, we further refine the quality of annotations with LLMs' confidence scores and remove those nodes with lower confidence from the previously selected set and (4) With the filtered high-quality annotation set, we then train GNN models on

![](images/86e06fc9725c7e54a14da4cf8897df3384c37cc3a96790156506c337bc6f59cc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Document"] --> B["Document"]
    B --> C["Document"]
    C --> D["Document"]
    D --> E["Document"]
    E --> F["Document"]
    F --> G["Document"]
    G --> H["Document"]
    H --> I["Document"]
    I --> J["Document"]
    J --> K["Document"]
    K --> L["Document"]
    L --> M["Document"]
    M --> N["Document"]
    N --> O["Document"]
    O --> P["Document"]
    P --> Q["Document"]
    Q --> R["Document"]
    R --> S["Document"]
    S --> T["Document"]
    T --> U["Document"]
    U --> V["Document"]
    V --> W["Document"]
    W --> X["Document"]
    X --> Y["Document"]
    Y --> Z["Document"]
    Z --> AA["Document"]
    AA --> AB["Document"]
    AB --> AC["Document"]
    AC --> AD["Document"]
    AD --> AE["Document"]
    AE --> AF["Document"]
    AF --> AG["Document"]
    AG --> AH["Document"]
    AH --> AI["Document"]
    AI --> AJ["Document"]
    AJ --> AK["Document"]
    AK --> AL["Document"]
    AL --> AM["Document"]
    AM --> AN["Document"]
    AN --> AO["Document"]
    AO --> AP["Document"]
    AP --> AQ["Document"]
    AQ --> AR["Document"]
    AR --> AS["Document"]
    AS --> AT["Document"]
    AT --> AU["Document"]
    AU --> AV["Document"]
    AV --> AW["Document"]
    AW --> AX["Document"]
    AX --> AY["Document"]
    AY --> AZ["Document"]
    AZ --> BA["Document"]
    BA --> BB["Document"]
    BB --> BC["Document"]
    BC --> BD["Document"]
    BD --> BE["Document"]
    BE --> BF["Document"]
    BF --> BG["Document"]
    BG --> BH["Document"]
    BH --> BI["Document"]
    BI --> BJ["Document"]
    BJ --> BK["Document"]
    BK --> BL["Document"]
    BL --> BM["Document"]
    BM --> BN["Document"]
    BN --> BO["Document"]
    BO --> BP["Document"]
    BP --> BQ["Document"]
    BQ --> BR["Document"]
    BR --> BS["Document"]
    BS --> BT["Document"]
    BT --> BU["Document"]
    BU --> BV["Document"]
    BV --> BW["Document"]
    BW --> BX["Document"]
    BX --> BY["Document"]
    BY --> BZ["Document"]
    BZ --> CA["Document"]
    CA --> CB["Document"]
    CB --> CC["Document"]
    CC --> CD["Document"]
    CD --> CE["Document"]
    CE --> CF["Document"]
    CF --> CG["Document"]
    CG --> CH["Document"]
    CH --> CI["Document"]
    CI --> CJ["Document"]
    CJ --> CK["Document"]
    CK --> CR["Document"]
    CR --> CS["Document"]
    CS --> CT["Document"]
    CT --> CU["Document"]
    CU --> CV["Document"]
    CV --> CW["Document"]
    CW --> CX["Document"]
    CX --> CY["Document"]
    CY --> CZ["Document"]
    CZ --> DA["Document"]
    DA --> DB["Document"]
    DB --> DC["Document"]
    DC --> DD["Document"]
    DD --> DE["Document"]
    DE --> DF["Document"]
    DF --> DG["Document"]
    DG --> DH["Document"]
    DH --> DI["Document"]
    DI --> DJ["Document"]
    DJ --> DK["Document"]
    DK --> DL["Document"]
    DL --> DJ
```
</details>

Figure 1: Our proposed pipeline LLM-GNN for label-free node classification on graphs. It is designed with four components: (1)(difficulty-aware) active node selection to select suitable nodes which are easy to annotate. (2) (confidence-aware annotations) generate the annotation on the node set and confidence score reflecting the label quality. (3) optional post-filtering removes low quality annotation via confidence score (4) GNN model training and prediction.

selected nodes and their annotations. Ultimately, the well-trained GNN model is then utilized to perform predictions. It should be noted that the framework we propose is very flexible, with different designs possible for each part. For example, for the part of active node selection, we can use conventional active learning methods combined with post-filtering to improve the overall quality of the labeling. We then detail each component.

# 3.2 DIFFICULTY-AWARE ACTIVE NODE SELECTION

Node selection aims to select a node candidate set, which will be annotated by LLMs, and then learned on GNN. Notably, the selected node set is generally small to ensure a controllable money budget. Unlike traditional graph active learning which mainly takes diversity and representativeness into consideration, label quality should also be included since LLM can produce noisy labels with large variances across different groups of nodes.

In the difficulty-aware active selection stage, we have no knowledge of how LLMs would respond to those nodes. Consequently, we are required to identify some heuristics building connections to the difficulty of annotating different nodes. A preliminary investigation of LLMs' annotations brings us inspiration on how to infer the annotation difficulty through node features, where we find that the accuracy of annotations generated by LLMs is closely related to the clustering density of nodes.

To demonstrate this correlation, we employ k-means clustering on the original feature space, setting the number of clusters equal to the distinct class count. 1000 nodes are sampled from the whole dataset and then annotated by LLMs. They are subsequently sorted and divided into ten equally sized groups based on their distance to the nearest cluster center. As shown in Figure 2, we observe a consistent pattern: nodes closer to cluster centers typically exhibit better annotation quality, which indicates lower annotation difficulty. Full results are included in Appendix F.2. We then adopt this distance as a heuristic to approximate the annotation reliability. Since the cluster number equals the distinct class count, we denote this heuristic as C-Density, calculated as C-Density $(v_{i}) = \frac{1}{1 + \|x_{v_{i}} - x_{\mathrm{CC}v_{i}}\|}$ , where for any node $v_{i}$ and its closest cluster center $\mathrm{CC}_{v_i}$ , and $x_{v_i}$ represents the feature of node $v_{i}$ . We demonstrate the effectiveness of this method through theoretical explanation in Appendix K.

![](images/ca80c9fbeead65fbeed73830f7388880731b8fac22a9511312668e0f7977540a.jpg)

<details>
<summary>bar</summary>

| Region Index | Average Accuracy |
| ------------ | ---------------- |
| 0            | 0.85             |
| 1            | 0.82             |
| 2            | 0.72             |
| 3            | 0.68             |
| 4            | 0.65             |
| 5            | 0.75             |
| 6            | 0.79             |
| 7            | 0.69             |
| 8            | 0.67             |
| 9            | 0.72             |
</details>

(a) CORA

![](images/9ee6fa32ebd0e22028ecac68f287169cfb48ba7a44ed211f582d65f0e27561dd.jpg)

<details>
<summary>bar</summary>

| Region Index | Average Accuracy |
| ------------ | ---------------- |
| 0            | 0.85             |
| 1            | 0.82             |
| 2            | 0.76             |
| 3            | 0.71             |
| 4            | 0.71             |
| 5            | 0.75             |
| 6            | 0.71             |
| 7            | 0.69             |
| 8            | 0.81             |
| 9            | 0.64             |
</details>

(b) CITESEER   
Figure 2: The annotation accuracy by LLMs vs. the distance to the nearest clustering center. The bars represent the average accuracy within each selected group, while the blue line indicates the cumulative average accuracy. At group i, the blue line denotes the average accuracy of all nodes in the preceding i groups.

We then incorporate this annotation difficulty heuristic in traditional active selection. For traditional active node selection, we select B nodes from the unlabeled pools with top scores $f_{\mathrm{act}}(v_i)$ , where $f_{\mathrm{act}}()$ is a score function (we defer the detailed introduction of $f_{\mathrm{act}}()$ to Appendix A). To benefit the performance of the trained models, the selected nodes should have a trade-off between annotation difficulty and traditional active selection criteria (e.g., representativeness (Cai et al., 2017) and diversity (Zhang et al., 2021c)). In traditional graph active learning, selection criteria can usually be denoted as a score function, such as PageRank centrality $f_{\mathrm{pg}}(v_i)$ for measuring the structural diversity. Then, one feasible way to integrate difficulty heuristic into traditional graph active learning is ranking aggregation. Compared to directly combining several scores via summation or multiplication, ranking aggregation is more robust to scale differences since it is scale-invariant and considers only the relative ordering of items. Considering the original score function for graph active learning as $f_{\mathrm{act}}(v_i)$ , we denote $r_{f_{\mathrm{act}}}(v_i)$ as the high-to-low ranking percentage. Then, we incorporate difficulty heuristic by first transforming C-Density( $v_i$ ) into a rank $r_{C-Density}(v_i)$ . Then we combine these two scores: $f_{\mathrm{DA-act}}(v_i) = \alpha_0 \times r_{f_{\mathrm{act}}}(v_i) + \alpha_1 \times r_{C-Density}(v_i)$ . “DA” stands for difficulty aware. Hyperparameters $\alpha_0$ and $\alpha_1$ are introduced to balance annotation difficulty and traditional graph active learning criteria such as representativeness, and diversity. Finally, nodes $v_i$ with larger $f_{\mathrm{DA-act}}(v_i)$ are selected for LLMs to annotate, which is denoted as $V_{anno}$ .

Table 1: Accuracy of annotations. Yellow denotes the best and Green denotes the second best result. The cost is determined by comparing the token consumption to that of zero-shot prompts. 

<table><tr><td rowspan="2">Prompt Strategy</td><td colspan="2">CORA</td><td colspan="2">OGBN-PRODUCTS</td><td colspan="2">WIKICS</td></tr><tr><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td></tr><tr><td>Vanilla (zero-shot)</td><td> $68.33 \pm 6.55$ </td><td>1</td><td> $75.33 \pm 4.99$ </td><td>1</td><td> $68.33 \pm 1.89$ </td><td>1</td></tr><tr><td>Vanilla (one-shot)</td><td> $69.67 \pm 7.72$ </td><td>2.2</td><td> $78.67 \pm 4.50$ </td><td>1.8</td><td> $72.00 \pm 3.56$ </td><td>2.4</td></tr><tr><td>TopK (zero-shot)</td><td> $68.00 \pm 6.38$ </td><td>1.1</td><td> $74.00 \pm 5.10$ </td><td>1.2</td><td> $72.00 \pm 2.16$ </td><td>1.1</td></tr><tr><td>Most Voting (zero-shot)</td><td> $68.00 \pm 7.35$ </td><td>1.1</td><td> $75.33 \pm 4.99$ </td><td>1.1</td><td> $69.00 \pm 2.16$ </td><td>1.1</td></tr><tr><td>Hybrid (zero-shot)</td><td> $67.33 \pm 6.80$ </td><td>1.5</td><td> $73.67 \pm 5.25$ </td><td>1.4</td><td> $71.00 \pm 2.83$ </td><td>1.4</td></tr><tr><td>Hybrid (one-shot)</td><td> $70.33 \pm 6.24$ </td><td>2.9</td><td> $75.67 \pm 6.13$ </td><td>2.3</td><td> $73.67 \pm 2.62$ </td><td>2.9</td></tr></table>

![](images/e1b6071b8005c13cba3c77d300fa315c9f919dd456e913ce7498809b562d91e6.jpg)

<details>
<summary>line</summary>

| k   | One-Shot | Moist Voting | Top-K | Hybrid(One-Shot) | Zero-Shot | Hybrid(Zero-Shot) |
| --- | -------- | ------------ | ----- | ----------------- | --------- | ------------------ |
| 50  | 0.76     | 0.88         | 0.88  | 0.86              | 0.80      | 0.91               |
| 100 | 0.77     | 0.90         | 0.84  | 0.84              | 0.81      | 0.91               |
| 150 | 0.79     | 0.84         | 0.84  | 0.84              | 0.83      | 0.87               |
| 200 | 0.80     | 0.81         | 0.81  | 0.81              | 0.82      | 0.83               |
| 250 | 0.79     | 0.79         | 0.79  | 0.79              | 0.81      | 0.76               |
| 300 | 0.73     | 0.73         | 0.73  | 0.73              | 0.73      | 0.69               |
</details>

Figure 3: An illustration of the relation between accuracy and confidence on WIKICS.

# 3.3 CONFIDENCE-AWARE ANNOTATIONS

After obtaining the candidate set $V_{anno}$ through active selection, we use LLMs to generate annotations for nodes in the set. Despite we select nodes easily to annotate with difficulty-aware active node selection, we are not aware of how the LLM responds to the nodes at that stage. It leads to the potential of remaining low-quality nodes. To figure out the high-quality annotations, we need some guidance on their reliability, such as the confidence scores. Inspired by recent literature on generating calibrated confidence from LLMs (Xiong et al., 2023; Tian et al., 2023; Wang et al., 2022), we investigate the following strategies: (1) directly asking for confidence (Tian et al., 2023), denoted as “Vanilla (zero-shot)”; (2) reasoning-based prompts to generate annotations, including chain-of-thought and multi-step reasoning (Wei et al., 2022; Xiong et al., 2023); (3) TopK prompt, which asks LLMs to generate the top K possible answers and select the most probable one as the answer (Xiong et al., 2023); (4) Consistency-based prompt (Wang et al., 2022), which queries LLMs multiple times and selects the most common output as the answer, denoted as “Most voting”; (5) Hybrid prompt (Wang et al., 2023a), which combines both TopK prompt and consistency-based prompt. In addition to the prompt strategy, few-shot samples have also been demonstrated critical to the performance of LLMs (Chen et al., 2023). We thus also investigate incorporating few-shot samples into prompts. In this work, we try 1-shot sample to avoid time and money cost. Detailed descriptions and full prompt examples are shown in Appendix D.

Then, we do a comparative study to identify the effective prompts in terms of accuracy, calibration of confidence, and costs. (1) For accuracy and cost evaluation, we adopt popular node classification benchmarks CORA, OGBN-PRODUCTS, and WIKICS. We randomly sample 100 nodes from each dataset and repeat with three different seeds to reduce sampling bias, and then we compare the generated annotations with the ground truth labels offered by these datasets. The cost is estimated by the number of tokens in input prompts and output contents. (2) It is important to note that the role of confidence is to assist us in identifying label reliability. Therefore, To validate the quality of the confidence produced by LLMs is to examine how the confidence can reflect the quality of the corresponding annotation. Therefore, we check how the annotation accuracy changes with the confidence. Specifically, we randomly select 300 nodes and sort them in descending order based on their confidence. Subsequently, we calculate the annotation accuracy for the top k nodes where K is varied in $\{50, 100, 150, 200, 250, 300\}$ . For each K, a higher accuracy indicates a better quality of the generated confidence. Empirically, we find that reasoning-based prompts will generate outputs that don't follow the format requirement, and greatly increase the query time and costs. Therefore, we don't further consider them in this work. For other prompts, we find that for a small portion of inputs, the outputs of LLMs do not follow the format requirements (for example, outputting an annotation out of the valid label names). For those invalid outputs, we design a self-correction prompt and set a larger temperature to review previous outputs and regenerate annotations.

The evaluation of performance and cost is shown in Table 1, and the evaluation of confidence is shown in Figure 3. The full results are included in Appendix E. From the experimental results, we make the following observations: First, LLMs present promising zero-shot prediction performance on all datasets, which suggests that LLMs are potentially good annotators. Second, compared to zero-shot prompts, prompts with few-shot demonstrations could slightly increase performance with double costs. Third, zero-shot hybrid strategies present the most effective approach to extract high-quality annotations since the confidence can greatly indicate the quality of annotation. We thus adopt zero-shot hybrid prompt in the following studies and leave the evaluation of other prompts as one future work.

# 3.4 POST-FILTERING

After achieving annotations together with the confidence scores, we may further refine the set of annotated nodes since we have access to confidence scores generated by LLMs, which can be used to filter high-quality labels. However, directly filtering out low-confidence nodes may result in a label distribution shift and degrade the diversity of the selected nodes, which will degrade the performance of subsequently trained models. Unlike traditional graph active learning methods which try to model diversity in the selection stage with criteria such as feature dissimilarity (Ren et al., 2022), in the post-filtering stage, label distribution is readily available. As a result, we can directly consider the label diversity of selected nodes. To measure the change of diversity, we propose a simple score function change of entropy (COE) to measure the entropy change of labels when removing a node from the selected set. Assuming that the current selected set of nodes is $\mathcal{V}_{\mathrm{sel}}$ , then COE can be computed as: $\mathrm{COE}(v_i) = H(\tilde{y}_{\mathcal{V}_{\mathrm{sel}} - \{v_i\}}) - H(\tilde{y}_{\mathcal{V}_{\mathrm{sel}}})$ where $H()$ is the Shannon entropy function (Shannon, 1948), and $\tilde{y}$ denotes the annotations generated by LLMs. It should be noted that the value of COE may possibly be positive or negative, and a small $\mathrm{COE}(v_i)$ value indicates that removing this node could adversely affect the diversity of the selected set, potentially compromising the performance of trained models. When a node is removed from the selected set, the entropy adjusts accordingly, necessitating a re-computation of COE. However, it introduces negligible computation overhead since the size of the selected set $\mathcal{V}_{\mathrm{anno}}$ is usually much smaller than the whole dataset. COE can be further combined with confidence $f_{\mathrm{conf}}(v_i)$ to balance diversity and annotation quality, in a ranking aggregation manner. It should be noted that $r_{\mathrm{C-Density}}(v_i)$ is also available in the post filtering phase. So, the final filtering score function $f_{\mathrm{filter}}$ can be stated as: $f_{\mathrm{filter}}(v_i) = \beta_0 \times r_{f_{\mathrm{conf}}(v_i)} + \beta_1 \times r_{\mathrm{COE}(v_i)} + \beta_2 \times r_{\mathrm{C-Density}}(v_i)$ . Hyper-parameters $\beta_0, \beta_1,$ and $\beta_2$ are introduced to balance label diversity and annotation quality. $r_{f_{\mathrm{conf}}}$ is the high-to-low ranking percentage of the confidence score $f_{\mathrm{conf}}$ . To conduct post-filtering, each time we remove the node with the smallest $f_{\mathrm{filter}}$ value until a pre-defined maximal number is reached.

# 3.5 GNN TRAINING AND PREDICTION

After obtaining the training labels, we further train a GNN. Our framework supports a variety of GNNs, and we select GCN, the most popular model, as our primary subject of study. Additionally, another critical component during the training process is the loss function. Traditional GNN-based pipelines mainly adopt cross-entropy loss; however, due to the noisy labels generated by the LLMs, we may also utilize a weighted cross-entropy loss in this part. Specifically, we can use the confidence scores from the previous section as the corresponding weights.

# 4 EXPERIMENT

In this section, we present experiments to evaluate the performance of our proposed pipeline LLM-GNN. We begin by detailing the experimental settings. Next, we investigate the following research questions: RQ1. How do active selection, post-filtering, and loss function affect the performance of LLM-GNN? How to come up with an effective pipeline implementation? RQ2. How does the performance and cost of LLM-GNN compare to other label-free node classification methods? RQ3. How do different budgets affect the performance of the pipelines? RQ4. How do LLMs' annotations compare to ground truth labels?

# 4.1 EXPERIMENTAL SETTINGS

In this paper, we adopt the following TAG datasets widely adopted for node classification: CORA (McCallum et al., 2000), CITESEER (Giles et al., 1998), PUBMED (Sen et al., 2008), OGBN-ARXIV, OGBN-PRODUCTS (Hu et al., 2020b), and WIKICS (Mernyei & Cangea, 2020). Statistics and descriptions of these datasets are in Appendix C.

In terms of the settings for each component in the pipeline, we adopt gpt-3.5-turbo-0613 $^{1}$ to generate annotations. In terms of the prompt strategy for generating annotations, we choose the “zero-shot hybrid strategy” considering its effectiveness in generating calibrated confidence. We leave the evaluation of other prompts as future works considering the massive costs. For the budget of the active selection, we refer to the popular semi-supervised learning setting for node classifications (Yang et al., 2016) and set the budget equal to 20 multiplied by the number of classes. For GNNs, we adopt one of the most popular models GCN (Kipf & Welling, 2016). The major goal of this evaluation is to show the potential of the LLM-GNN pipeline. Therefore, we do not tune the hyper-parameters in both difficulty-aware active selection and post-filtering but simply setting them with the same value.

Table 2: Impact of different active selection strategies. We show the top three performance in each dataset with pink, green, and yellow, respectively. To compare with traditional graph active selections, we underline their combinations with our selection strategies if the combinations outperform their corresponding traditional graph active selection. OOT means that this method can not scale to large-scale graphs because of long execution time. 

<table><tr><td></td><td>CORA</td><td>CITESEER</td><td>PUBMED</td><td>WIKICS</td><td>OGBN-ARXIV</td><td>OGBN-PRODUCTS</td></tr><tr><td>Random</td><td> $70.48 \pm 0.73$ </td><td> $65.11 \pm 1.12$ </td><td> $72.98 \pm 2.15$ </td><td> $60.69 \pm 1.73$ </td><td> $64.59 \pm 0.16$ </td><td> $70.40 \pm 0.60$ </td></tr><tr><td>Random-W</td><td> $71.77 \pm 0.75$ </td><td> $65.92 \pm 1.05$ </td><td> $73.92 \pm 1.75$ </td><td> $61.42 \pm 1.54$ </td><td> $64.95 \pm 0.19$ </td><td> $71.96 \pm 0.59$ </td></tr><tr><td>C-Density</td><td> $42.22 \pm 1.59$ </td><td> $66.44 \pm 0.34$ </td><td> $74.43 \pm 0.28$ </td><td> $57.77 \pm 0.85$ </td><td> $44.08 \pm 0.39$ </td><td> $8.29 \pm 0.00$ </td></tr><tr><td>PS-Random-W</td><td> $72.38 \pm 0.72$ </td><td> $67.18 \pm 0.92$ </td><td> $73.31 \pm 1.65$ </td><td> $62.60 \pm 0.94$ </td><td> $65.22 \pm 0.15$ </td><td> $71.62 \pm 0.54$ </td></tr><tr><td>Density</td><td> $72.40 \pm 0.35$ </td><td> $61.06 \pm 0.95$ </td><td> $74.43 \pm 0.28$ </td><td> $64.96 \pm 0.53$ </td><td> $51.77 \pm 0.24$ </td><td> $20.22 \pm 0.11$ </td></tr><tr><td>Density-W</td><td> $72.39 \pm 0.34$ </td><td> $59.88 \pm 0.97$ </td><td> $73.00 \pm 0.19$ </td><td> $63.80 \pm 0.69$ </td><td> $51.03 \pm 0.27$ </td><td> $20.97 \pm 0.15$ </td></tr><tr><td>DA-Density</td><td> $70.73 \pm 0.32$ </td><td> $62.92 \pm 1.05$ </td><td> $74.43 \pm 0.28$ </td><td> $63.08 \pm 0.45$ </td><td> $51.33 \pm 0.29$ </td><td> $8.50 \pm 0.32$ </td></tr><tr><td>PS-Density-W</td><td> $74.61 \pm 0.13$ </td><td> $61.00 \pm 0.55$ </td><td> $74.50 \pm 0.23$ </td><td> $65.57 \pm 0.45$ </td><td> $51.73 \pm 0.29$ </td><td> $19.15 \pm 0.18$ </td></tr><tr><td>DA-Density-W</td><td> $67.29 \pm 0.96$ </td><td> $62.98 \pm 0.77$ </td><td> $73.39 \pm 0.35$ </td><td> $63.26 \pm 0.62$ </td><td> $51.36 \pm 0.39$ </td><td> $8.52 \pm 0.11$ </td></tr><tr><td>AGE</td><td> $69.15 \pm 0.38$ </td><td> $54.25 \pm 0.31$ </td><td> $74.55 \pm 0.54$ </td><td> $55.51 \pm 0.12$ </td><td> $46.68 \pm 0.30$ </td><td> $65.63 \pm 0.15$ </td></tr><tr><td>AGE-W</td><td> $69.70 \pm 0.45$ </td><td> $57.60 \pm 0.35$ </td><td> $64.30 \pm 0.49$ </td><td> $55.15 \pm 0.14$ </td><td> $47.84 \pm 0.35$ </td><td> $64.92 \pm 0.19$ </td></tr><tr><td>DA-AGE</td><td> $74.38 \pm 0.24$ </td><td> $59.92 \pm 0.42$ </td><td> $74.20 \pm 0.51$ </td><td> $59.39 \pm 0.21$ </td><td> $48.21 \pm 0.35$ </td><td> $60.03 \pm 0.11$ </td></tr><tr><td>PS-AGE-W</td><td> $72.61 \pm 0.39$ </td><td> $57.44 \pm 0.49$ </td><td> $64.00 \pm 0.44$ </td><td> $56.13 \pm 0.11$ </td><td> $47.12 \pm 0.39$ </td><td> $68.62 \pm 0.15$ </td></tr><tr><td>DA-AGE-W</td><td> $74.96 \pm 0.22$ </td><td> $58.41 \pm 0.45$ </td><td> $65.85 \pm 0.67$ </td><td> $59.19 \pm 0.24$ </td><td> $47.79 \pm 0.32$ </td><td> $59.95 \pm 0.23$ </td></tr><tr><td>RIM</td><td> $69.86 \pm 0.38$ </td><td> $63.44 \pm 0.42$ </td><td> $76.22 \pm 0.16$ </td><td> $66.72 \pm 0.16$ </td><td>OOT</td><td>OOT</td></tr><tr><td>DA-RIM</td><td> $73.99 \pm 0.44$ </td><td> $60.33 \pm 0.40$ </td><td> $79.17 \pm 0.11$ </td><td> $67.82 \pm 0.32$ </td><td>OOT</td><td>OOT</td></tr><tr><td>PS-RIM-W</td><td> $73.19 \pm 0.45$ </td><td> $62.85 \pm 0.49$ </td><td> $74.52 \pm 0.19$ </td><td> $69.84 \pm 0.19$ </td><td>OOT</td><td>OOT</td></tr><tr><td>DA-RIM-W</td><td> $74.73 \pm 0.41$ </td><td> $60.80 \pm 0.57$ </td><td> $77.94 \pm 0.24$ </td><td> $68.22 \pm 0.25$ </td><td>OOT</td><td>OOT</td></tr><tr><td>GraphPart</td><td> $68.57 \pm 2.18$ </td><td> $66.59 \pm 1.34$ </td><td> $77.50 \pm 1.23$ </td><td> $67.28 \pm 0.87$ </td><td>OOT</td><td>OOT</td></tr><tr><td>GraphPart-W</td><td> $69.90 \pm 2.03$ </td><td> $68.20 \pm 1.42$ </td><td> $78.91 \pm 1.04$ </td><td> $68.43 \pm 0.92$ </td><td>OOT</td><td>OOT</td></tr><tr><td>DA-GraphPart</td><td> $69.35 \pm 1.92$ </td><td> $69.37 \pm 1.27$ </td><td> $79.49 \pm 0.85$ </td><td> $68.72 \pm 1.01$ </td><td>OOT</td><td>OOT</td></tr><tr><td>PS-GraphPart-W</td><td> $69.92 \pm 1.75$ </td><td> $69.06 \pm 1.19$ </td><td> $78.84 \pm 1.05$ </td><td> $66.90 \pm 1.05$ </td><td>OOT</td><td>OOT</td></tr><tr><td>DA-GraphPart-W</td><td> $68.61 \pm 1.32$ </td><td> $68.82 \pm 1.17$ </td><td> $79.89 \pm 0.79$ </td><td> $67.13 \pm 1.23$ </td><td>OOT</td><td>OOT</td></tr><tr><td>FeatProp</td><td> $72.82 \pm 0.08$ </td><td> $66.61 \pm 0.55$ </td><td> $73.90 \pm 0.15$ </td><td> $64.08 \pm 0.12$ </td><td> $66.06 \pm 0.07$ </td><td> $74.04 \pm 0.15$ </td></tr><tr><td>FeatProp-W</td><td> $73.56 \pm 0.13$ </td><td> $68.04 \pm 0.69$ </td><td> $76.90 \pm 0.19$ </td><td> $63.80 \pm 0.21$ </td><td> $66.32 \pm 0.15$ </td><td> $74.32 \pm 0.14$ </td></tr><tr><td>PS-FeatProp</td><td> $75.54 \pm 0.34$ </td><td> $69.06 \pm 0.32$ </td><td> $74.98 \pm 0.35$ </td><td> $66.09 \pm 0.35$ </td><td> $66.14 \pm 0.27$ </td><td> $74.91 \pm 0.17$ </td></tr><tr><td>PS-FeatProp-W</td><td> $76.23 \pm 0.07$ </td><td> $68.64 \pm 0.71$ </td><td> $78.84 \pm 1.05$ </td><td> $64.72 \pm 0.19$ </td><td> $65.84 \pm 0.19$ </td><td> $74.54 \pm 0.24$ </td></tr></table>

In terms of evaluation, we compare the generated prediction of GNNs with the ground truth labels offered in the original datasets and adopt accuracy as the metric. Similar to (Ma et al., 2022), we adopt a setting where there's no validation set, and models trained on selected nodes will be further tested based on the rest unlabeled nodes. All experiments will be repeated for 3 times with different seeds. For hyper-parameters of the experiment, we adopt a fixed setting commonly used by previous papers or benchmarks Kipf & Welling (2016); Hamilton et al. (2017); Hu et al. (2020b). One point that should be strengthened is the number of training epochs. Since there's no validation set and the labels are noisy, models may suffer from overfitting (Song et al., 2022). However, we find that most models work well across all datasets by setting a small fixed number of training epochs, such as 30 epochs for small and medium-scale datasets (CORA, CITESEER, PUBMED, and WIKICS), and 50 epochs for the rest large-scale datasets. This setting, which can be viewed as a simpler alternative to early stop trick (without validation set) (Bai et al., 2021) for training on noisy labels so that we can compare different methods more fairly and conveniently.

# 4.2 RQ1. IMPACT OF DIFFERENT ACTIVE SELECTION STRATEGIES

We conduct a comprehensive evaluation of different active selection strategies, which is the key component of our pipeline. Specifically, we examine how effectiveness of (1) difficulty-aware active node selection before LLM annotation (2) post-filtering after LLM annotation and how they combine with traditional active learning algorithms (3) loss functions. For selection strategies, we consider (1) Traditional graph active selection: Random selection, Density-based selection (Ma et al., 2022), GraphPart (Ma et al., 2022), FeatProp (Wu et al., 2019), Degree-based selection (Ma et al., 2022), Pagerank centrality-based selection (Ma et al., 2022), AGE (Cai et al., 2017), and RIM (Zhang et al., 2021b). (2) Difficulty-aware active node selection: C-Density-based selection, and traditional graph active selections combined with C-Density. To denote these selections, we add the prefix “DA-”. For example, “DA-AGE” means combining the original AGE method with our proposed C-Density. (3) Post Filtering: Traditional graph active selection combined with confidence and COE-based selections, we add the prefix “PS-”. For FeatProp, as it selects candidate nodes directly using the

K-Medoids algorithm (Wu et al., 2019), integrating it with difficulty-aware active selections is not feasible. For loss functions, we consider both cross entropy loss and weighted cross entropy loss, where we add a “-W” postfix for the latter. Detailed introductions of these methods are shown in Appendix A. The results for GCN are shown in Table 2. In terms of space limits, we move part of the results and more ablation studies to Appendix J.

From the experimental results, we make the following observations:

1. The proposed post-filtering strategy presents promising effectiveness. Combined with traditional graph active learning methods like GraphPart, RIM, and Featprop, it can consistently outperform. Combined with FeatProp, it can achieve both promising accuracy and better scalability.   
2. Although C-Density-based selection can achieve superior annotation quality, merely using this metric will make the trained model achieve poor performance. To better understand this phenomenon, we check the labels of the selected nodes. We find that the problem lies in label imbalance brought by active selection. For example, we check the selected nodes for PUBMED, and find that all annotations belong to one class. We further find that tuning the number of clustering centers for C-Density can trade off between diversity and annotation quality, where a larger K can mitigate the class imbalance problem. However, it proposes a challenge to find a proper K for massive-scale datasets like OGBN-PRODUCTS, where weighted loss and post-filtering are more effective.   
3. Comparing normal cross entropy loss to weighted cross entropy loss, weighted cross entropy loss further enhance the performance for most of the cases.   
4. In a nutshell, we summarize the following empirical rules of thumbs: (1) Featprop-based methods can consistently achieve promising performance across different datasets efficiently; (2) Comparing DA and PS, DA costs less since we don't need LLMs to generate the confidence and we may use a simpler prompt. PS can usually get better performance. On large-scale datasets, PS usually get much better results.

# 4.3 (RQ2.) COMPARISON WITH OTHER LABEL-FREE NODE CLASSIFICATION METHODS

To demonstrate the effectiveness and novelty of our proposed pipeline, we further conduct a comparison with other label-free node classification pipelines, which include: (1) Zero-shot node classification method: SES, TAG-Z (Li & Hooi, 2023); (2) Zero-shot classification models for texts: BART-large-MNLI (Lewis et al., 2019); and (3) Directly using LLMs for predictions: LLMs-as-Predictors (Chen et al., 2023). Detailed introductions of these models can be found in Appendix A. We compare both performance and costs of these models, and the results are shown in Table 3.

Table 3: Comparison of label-free node classification methods. The cost is computed in dollars. The performance of methods with \* are taken from Li & Hooi (2023). Notably, the time cost of LLMs is proportional to the expenses. 

<table><tr><td></td><td colspan="2">OGBN-ARXIV</td><td colspan="2">OGBN-PRODUCTS</td></tr><tr><td>Methods</td><td>Acc</td><td>Cost</td><td>Acc</td><td>Cost</td></tr><tr><td>SES(*)</td><td>13.08</td><td>N/A</td><td>6.67</td><td>N/A</td></tr><tr><td>TAG-Z(*)</td><td>37.08</td><td>N/A</td><td>47.08</td><td>N/A</td></tr><tr><td>BART-large-MNLI</td><td>13.2</td><td>N/A</td><td>28.8</td><td>N/A</td></tr><tr><td>LLMs-as-Predictors</td><td>73.33</td><td>79</td><td>75.33</td><td>1572</td></tr><tr><td>LLM-GNN</td><td>66.32</td><td>0.63</td><td>74.91</td><td>0.74</td></tr></table>

From the experimental results in the table, we can see that (1) our proposed pipeline LLM-GNN can significantly outperform SES, TAG-Z and BART-large-MNLI. (2) Despite LLMs-as-Predictors has better performance than LLM-GNN, its cost is much higher than LLM-GNN. For example, the cost of LLMs-as-Predictors in OGBN-PRODUCTS is 2, 124× that of LLM-GNN. Besides, the promising performance of LLMs-as-Predictors on OGBN-ARXIV may be an exception, relevant to the specific prompts leveraging the memorization of LLMs (Chen et al., 2023).

# 4.4 (RQ3.) HOW DO DIFFERENT BUDGETS AFFECT THE PERFORMANCE OF OUR PIPELINES?

We conduct a comprehensive evaluation on different budgets rather than the fixed budget in previous experiments. It aims to examine how effective our algorithm is when confronting different real-world scenarios with different to meet different cost and performance requirements. Experiments are typically conducted on the CORA dataset by setting the budget as $\{35, 70, 105, 140, 175, 280, 560, 1,120\}$ . We choose both random selections and those methods that perform well in Table 2. We can have the following observations from Figure 4. (1) with the increase in the budget, the performance tends to increase gradually. (2) unlike using ground truth, the performance growth is relatively limited as the budget increases. It suggests that there exists a trade-off between the performance and the cost in the real-world scenario.

# 4.5 (RQ4.) CHARACTERISTICS OF LLMs' ANNOTATIONS

Although LLMs' annotations are noisy labels, we find that they are more benign than the ground truth labels injected with synthetic noise adopted in (Zhang et al., 2021b). Specifically, assuming that the quality of LLMs' annotations is $q\%$ , we randomly select $(1 - q)\%$ ground truth labels and flip them into other classes uniformly to generate synthetic noisy labels. We then train GNN models on LLMs' annotations, synthetic noisy labels, and LLMs' annotations with all incorrect labels removed, respectively. The results are demonstrated in Figure 5, from which we observe that: LLMs' annotations present totally different training dynamics from synthetic noisy labels. The extent of over-fitting for LLMs' annotations is much less than that for synthetic noisy labels.

![](images/f1c779e0d207f7b1decc50e5dae495346c84a9ac6d33cbc10057c4513c8dec50.jpg)

<details>
<summary>line</summary>

| Budget | Random CE | Random WE | PS Random | Featprop | PS Featprop |
| ------ | --------- | --------- | --------- | -------- | ----------- |
| 35     | 58        | 59        | 56        | 65       | 68          |
| 70     | 67        | 69        | 67        | 70       | 70          |
| 105    | 69        | 70        | 70        | 74       | 76          |
| 140    | 70        | 71        | 71        | 73       | 76          |
| 175    | 70        | 71        | 71        | 73       | 76          |
| 280    | 71        | 72        | 73        | 71       | 74          |
| 560    | 72        | 73        | 74        | 73       | 73          |
| 1120   | 73        | 73        | 72        | 74       | 73          |
</details>

Figure 4: Investigation on how different budgets affect the performance of LLM-GNN. Methods achieving top performance in Table 2 and random selection-based methods are compared in the figure. CE and WE means normal cross entropy loss and weighted loss, respectively.

![](images/d1ecca0db3f07e20e23689a2b94fbccb27cac4f14095bfe691552f51ca055c9f.jpg)

<details>
<summary>line</summary>

| Epochs | Ran Anno | Ran Filtered | Ran GT | Ran Syn |
| ------ | -------- | ------------ | ------ | ------- |
| 0      | 0.5      | 0.5          | 0.5    | 0.5     |
| 25     | 0.7      | 0.8          | 0.85   | 0.7     |
| 50     | 0.7      | 0.8          | 0.85   | 0.65    |
| 75     | 0.7      | 0.8          | 0.85   | 0.62    |
| 100    | 0.7      | 0.8          | 0.85   | 0.61    |
| 125    | 0.7      | 0.8          | 0.85   | 0.61    |
| 150    | 0.7      | 0.8          | 0.85   | 0.61    |
</details>

Figure 5: Comparisons among LLMs' annotations, ground truth labels, and synthetic noisy labels. “Ran” represents random selection, “GT” indicates the ground truth labels. “Filtered” means replacing all wrong annotations with ground truth labels.

# 5 RELATED WORKS

Graph active learning. Graph active learning (Cai et al., 2017) aims to maximize the test performance with nodes actively selected under a limited query budget. To achieve this goal, algorithms are developed to maximize the informativeness and diversity of the selected group of nodes (Zhang et al., 2021c). These algorithms are designed based on assumptions. In Ma et al. (2022), diversity is assumed to be related to the partition of nodes, and thus samples are actively selected from different communities. In Zhang et al. (2021c;b;a), representativeness is assumed to be related to the influence of nodes, and thus nodes with a larger influence score are first selected. Another line of work directly sets the accuracy of trained models as the objective (Gao et al., 2018; Hu et al., 2020a; Zhang et al., 2022), and then adopts reinforcement learning to do the optimization.

LLMs for graphs. Recent progress on applying LLMs for graphs (He et al., 2023a; Guo et al., 2023) aims to utilize the power of LLMs and further boost the performance of graph-related tasks. LLMs are either adopted as the predictor (Chen et al., 2023; Wang et al., 2023a; Ye et al., 2023), which directly generates the solutions or as the enhancer (He et al., 2023a), which takes the capability of LLMs to boost the performance of a smaller model with better efficiency. In this paper, we adopt LLMs as annotators, which combine the advantages of these two lines to train an efficient model with promising performance and good efficiency, without the requirement of any ground truth labels.

# 6 CONCLUSION

In this paper, we revisit the long-term ignorance of the data annotation process in existing node classification methods and propose the pipeline label-free node classification on graphs with LLMs to solve this problem. The key design of our pipelines involves LLMs to generate confidence-aware annotations, and using difficulty-aware selections and confidence-based post-filtering to further enhance the annotation quality. Comprehensive experiments validate the effectiveness of our pipeline.

# 7 ACKNOWLEDGEMENTS

This research is supported by the National Science Foundation (NSF) under grant numbers CNS 2246050, IIS1845081, IIS2212032, IIS2212144, IOS2107215, DUE 2234015, DRL 2025244 and IOS2035472, the Army Research Office (ARO) under grant number W911NF-21-1-0198, the Home Depot, Cisco Systems Inc, Amazon Faculty Award, Johnson&Johnson, JP Morgan Faculty Award and SNAP.

# REFERENCES

Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu. Understanding and improving early stopping for learning with noisy labels. Advances in Neural Information Processing Systems, 34:24392–24403, 2021.   
Parikshit Bansal and Amit Sharma. Large language models as annotators: Enhancing generalization of nlp models at minimal cost. arXiv preprint arXiv:2306.15766, 2023.   
Hongyun Cai, Vincent W Zheng, and Kevin Chen-Chuan Chang. Active learning for graph embedding. arXiv preprint arXiv:1705.05085, 2017.   
Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, et al. Exploring the potential of large language models (llms) in learning on graphs. arXiv preprint arXiv:2307.03393, 2023.   
Bosheng Ding, Chengwei Qin, Linlin Liu, Lidong Bing, Shafiq Joty, and Boyang Li. Is gpt-3 a good data annotator? arXiv preprint arXiv:2212.10450, 2022.   
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022.   
Li Gao, Hong Yang, Chuan Zhou, Jia Wu, Shirui Pan, and Yue Hu. Active discriminative network representation learning. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18, pp. 2142–2148. International Joint Conferences on Artificial Intelligence Organization, 7 2018. doi: 10.24963/ijcai.2018/296. URL https://doi.org/10.24963/ijcai.2018/296.   
Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. Chatgpt outperforms crowd-workers for text-annotation tasks. arXiv preprint arXiv:2303.15056, 2023.   
C. Lee Giles, Kurt D. Bollacker, and Steve Lawrence. Citeseer: An automatic citation indexing system. In Proceedings of the Third ACM Conference on Digital Libraries, DL 98, pp. 89–98, New York, NY, USA, 1998. ACM. ISBN 0-89791-965-3. doi: 10.1145/276675.276685. URL http://doi.acm.org/10.1145/276675.276685.   
Jiayan Guo, Lun Du, and Hengyu Liu. Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking. arXiv preprint arXiv:2305.15066, 2023.   
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.   
Haibo He and Edwardo A. Garcia. Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9):1263–1284, 2009. doi: 10.1109/TKDE.2008.239.   
Xiaoxin He, Xavier Bresson, Thomas Laurent, and Bryan Hooi. Explanations as features: Llm-based features for text-attributed graphs. arXiv preprint arXiv:2305.19523, 2023a.   
Xingwei He, Zhenghao Lin, Yeyun Gong, Alex Jin, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, Weizhu Chen, et al. Annollm: Making large language models to be better crowd-sourced annotators. arXiv preprint arXiv:2303.16854, 2023b.

Shengding Hu, Zheng Xiong, Meng Qu, Xingdi Yuan, Marc-Alexandre Côté, Zhiyuan Liu, and Jian Tang. Graph policy network for transferable active learning on graphs. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 10174–10185. Curran Associates, Inc., 2020a. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/73740ea85c4ec25f00f9acbd859f861d-Paper.pdf.   
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118–22133, 2020b.   
Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. Can llms effectively leverage graph structural information: When and why. arXiv preprint arXiv:2309.16595, 2023.   
Sheng-jun Huang, Rong Jin, and Zhi-Hua Zhou. Active learning by querying informative and representative examples. In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta (eds.), Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper\_files/paper/2010/file/5487315b1286f907165907aa8fc96619-Paper.pdf.   
Wei Ju, Yifang Qin, Siyu Yi, Zhengyang Mao, Kangjie Zheng, Luchen Liu, Xiao Luo, and Ming Zhang. Zero-shot node classification with graph contrastive embedding network. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=8wGXnjRLSy.   
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.   
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. CoRR, abs/1910.13461, 2019. URL http://arxiv.org/abs/1910.13461.   
Yuexin Li and Bryan Hooi. Prompt-based zero-and few-shot node classification: A multimodal approach. arXiv preprint arXiv:2307.11572, 2023.   
Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. One for all: Towards training one graph model for all classification tasks. arXiv preprint arXiv:2310.00149, 2023.   
Jiaqi Ma, Ziqiao Ma, Joyce Chai, and Qiaozhu Mei. Partition-based active learning for graph neural networks. arXiv preprint arXiv:2201.09391, 2022.   
Yao Ma and Jiliang Tang. Deep learning on graphs. Cambridge University Press, 2021.   
Andrew McCallum, Kamal Nigam, Jason D. M. Rennie, and Kristie Seymore. Automating the construction of internet portals with machine learning. Information Retrieval, 3:127–163, 2000.   
Péter Mernyei and Cătălina Cangea. Wiki-cs: A wikipedia-based benchmark for graph neural networks. arXiv preprint arXiv:2007.02901, 2020.   
Nicholas Pangakis, Samuel Wolken, and Neil Fasching. Automated annotation with generative ai requires validation. arXiv preprint arXiv:2306.00176, 2023.   
Xiaohuan Pei, Yanxi Li, and Chang Xu. Gpt self-supervision for a better data annotator. arXiv preprint arXiv:2306.04349, 2023.   
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Conference on Empirical Methods in Natural Language Processing, 2019. URL https://api.semanticscholar.org/CorpusID:201646309.   
Zhicheng Ren, Yifu Yuan, Yuxin Wu, Xiaxuan Gao, Yewen Wang, and Yizhou Sun. Dissimilar nodes improve graph active learning. arXiv preprint arXiv:2212.01968, 2022.

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad. Collective classification in network data. AI Magazine, 29(3):93, Sep. 2008. doi:10.1609/aimag.v29i3.2157. URL https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157.   
C. E. Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379–423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x.   
Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2022.   
Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv preprint arXiv:2305.14975, 2023.   
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.   
Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? arXiv preprint arXiv:2305.10037, 2023a.   
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432, 2023b.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.   
Zheng Wang, Jialong Wang, Yuchen Guo, and Zhiguo Gong. Zero-shot node classification with decomposed graph prototype network. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1769–1779, 2021.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.   
Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning with noisy labels revisited: A study using real-world human annotations. arXiv preprint arXiv:2110.12088, 2021.   
Yuexin Wu, Yichong Xu, Aarti Singh, Yiming Yang, and Artur Dubrawski. Active learning for graph neural networks via node feature propagation. arXiv preprint arXiv:1910.07567, 2019.   
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv preprint arXiv:2306.13063, 2023.   
Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. ArXiv, abs/1603.08861, 2016. URL https://api.semanticscholar.org/CorpusID:7008752.   
Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. Natural language is all a graph needs. arXiv preprint arXiv:2308.07134, 2023.   
Qin Yue, Jiye Liang, Junbiao Cui, and Liang Bai. Dual bidirectional graph convolutional networks for zero-shot node classification. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '22, pp. 2408–2417, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393850. doi: 10.1145/3534678.3539316. URL https://doi.org/10.1145/3534678.3539316.

Wentao Zhang, Yu Shen, Yang Li, Lei Chen, Zhi Yang, and Bin Cui. Alg: Fast and accurate active learning framework for graph convolutional networks. In Proceedings of the 2021 International Conference on Management of Data, SIGMOD '21, pp. 2366–2374, New York, NY, USA, 2021a. Association for Computing Machinery. ISBN 9781450383431. doi: 10.1145/3448016.3457325. URL https://doi.org/10.1145/3448016.3457325.   
Wentao Zhang, Yexin Wang, Zhenbang You, Meng Cao, Ping Huang, Jiulong Shan, Zhi Yang, and Bin Cui. Rim: Reliable influence-based active learning on graphs. Advances in Neural Information Processing Systems, 34:27978–27990, 2021b.   
Wentao Zhang, Zhi Yang, Yexin Wang, Yu Shen, Yang Li, Liang Wang, and Bin Cui. Grain: Improving data efficiency of graph neural networks via diversified influence maximization. arXiv preprint arXiv:2108.00219, 2021c.   
Yuheng Zhang, Hanghang Tong, Yinglong Xia, Yan Zhu, Yuejie Chi, and Lei Ying. Batch active learning with graph neural networks via multi-agent deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 9118–9126, 2022.   
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations (ICLR 2023), 2023.   
Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and Jian Tang. Graphtext: Graph reasoning in text space. arXiv preprint arXiv:2310.01089, 2023.   
Qi Zhu, Natalia Ponomareva, Jiawei Han, and Bryan Perozzi. Shift-robust gnns: Overcoming the limitations of localized graph training data. Advances in Neural Information Processing Systems, 34:27965–27977, 2021a.   
Zhaowei Zhu, Yiwen Song, and Yang Liu. Clusterability as an alternative to anchor points when learning with noisy labels. In International Conference on Machine Learning, pp. 12912–12923. PMLR, 2021b.

# A DETAILED INTRODUCTIONS OF BASELINE MODELS

In this part, we present a more detailed introduction to related works, especially those baseline models adopted in this paper.

# A.1 GRAPH ACTIVE LEARNING METHODS

1. Density-based selection (Ma et al., 2022): We apply KMeans clustering in the feature space with the number of clusters set to the number of budgets. Then, nodes with closest distances to their clustering centers are selected as the training nodes. $f_{act}()$ is calculated as the reciprocal of the distance to the clustering centers.   
2. GraphPart (Ma et al., 2022): A graph partition method is first applied to ensure the selection diversity. Then inside each region, nodes with high aggregated density are selected based on an approximate K-medoids algorithm. $f_{act}()$ is calculated as the reciprocal of the distance to the local clustering centers after partition.   
3. FeatProp (Wu et al., 2019): Node features are first aggregated based on the adjacency matrix. Then, K-Medoids is applied to select the training nodes. $f_{act}()$ is calculated using K-Medoids.   
4. Degree-based selection (Ma et al., 2022): Nodes with the highest degrees are selected as the training nodes. $f_{act}()$ is calculated as the degree od nodes.   
5. Pagerank-based selection (Ma et al., 2022): Nodes with the highest PageRank centrality are selected as the training nodes. $f_{act}()$ is calculated as the PageRank score of nodes.   
6. AGE (Cai et al., 2017): Nodes are selected based on a score function that considers both the density of features and PageRank centrality. We ignore the uncertainty measurement since we adopt a one-step selection setting. $f_{act}()$ is calculated as the weighted aggregation of the density of aggregated features together with the PageRank score.   
7. RIM (Zhang et al., 2021b): Nodes with the highest reliable social influence are selected as the training nodes. Reliability is measured based on a pre-defined oracle accuracy and similarity between the aggregated node feature matrices. $f_{act}()$ is based on the density of aggregated features.

# A.2 ZERO-SHOT CLASSIFICATION BASELINE MODELS

1. SES (Li & Hooi, 2023): An unsupervised method to do classification. Both node features and class names are projected into the same semantic space, and the class with the smallest distance is selected as the label.   
2. TAG-Z (Li & Hooi, 2023): Adopting prompts and graph topology to produce preliminary logits, which are readily applicable for zero-shot node classification.   
3. BART-Large-MNLI (Lewis et al., 2019): A zero-shot text classification model fine-tuned on the MNLI dataset.

# B MORE RELATED WORKS

LLMs as Annotators. Curating human-annotated data is both labor-intensive and expensive, especially for intricate tasks or niche domains where data might be scarce. Due to the exceptional zero-shot inference capabilities of LLMs, recent literature has embraced their use for generating pseudo-annotated data (Bansal & Sharma, 2023; Ding et al., 2022; Gilardi et al., 2023; He et al., 2023b; Pangakis et al., 2023; Pei et al., 2023). Gilardi et al. (2023); Ding et al. (2022) evaluate LLMs' efficacy as annotators, showcasing superior annotation quality and cost-efficiency compared to human annotators, particularly in tasks like text classification. Furthermore, Pei et al. (2023); He et al. (2023b); Bansal & Sharma (2023) delve into enhancing the efficacy of LLM-generated annotations: Pei et al. (2023); He et al. (2023b) explore prompt strategies, while Bansal & Sharma (2023) investigates sample selection. However, while acknowledging their effectiveness, Pangakis et al. (2023) highlights certain shortcomings in LLM-produced annotations, underscoring the need for caution when leveraging LLMs in this capacity.

Zero-shot Node Classification. Another issue related to our work is zero-shot node classification. The goal of zero-shot node classification is to train a model on existing training data and then generalize it to new categories. It is important to note that this is different from the concept of label-

free node classification that we propose, as in our problem there are no labels available from the start. Wang et al. (2021) transfers the knowledge from existing labels to unseen labels by semantic descriptions of labels. Yue et al. (2022) adopts a label consistency module to transfer the knowledge from the original label domain to the unseen label domain. Ju et al. (2023) adopts a two-level contrastive learning to learn the node embeddings and class assignments in an end-to-end manner, which can thus transfer the knowledge to unseen classes.

# C DATASETS

In this paper, we use the following popular datasets commonly adopted for node classifications: CORA, CITESEER, PUBMED, WIKICS, OGBN-ARXIV, and OGBN-PRODUCTS. Since LLMs can only understand the raw text attributes, for CORA, CITESEER, PUBMED, OGBN-ARXIV, and OGBN-PRODUCTS, we adopt the text attributed graph version from Chen et al. (2023). For WIKICS, we get the raw attributes from https://github.com/pmernyei/wiki-cs-dataset. Then, we give a detailed description of each dataset in Table 4.

Table 4: Dataset descriptions 

<table><tr><td>Dataset Name</td><td>#Nodes</td><td>#Edges</td><td>Task Description</td><td>Classes</td></tr><tr><td>CORA</td><td>2708</td><td>5429</td><td>Given the title and abstract, predict the category of this paper</td><td>Rule Learning, Neural Networks, Case Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods</td></tr><tr><td>CITESEER</td><td>3186</td><td>4277</td><td>Given the title and abstract, predict the category of this paper</td><td>Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence</td></tr><tr><td>PUBMED</td><td>19717</td><td>44335</td><td>Given the title and abstract, predict the category of this paper</td><td>Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2</td></tr><tr><td>WIKICS</td><td>11701</td><td>215863</td><td>Given the contents of the Wikipedia article, predict the category of this article</td><td>Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, Programming language topics</td></tr><tr><td>OGBN-ARXIV</td><td>169343</td><td>1166243</td><td>Given the title and abstract, predict the category of this paper</td><td>40 classes from https://arxiv.org/archive/cs</td></tr><tr><td>OGBN-PRODUCTS</td><td>2449029</td><td>61859140</td><td>Given the product description, predict the category of this product</td><td>47 classes from Amazon, including Home &amp; Kitchen, Health &amp; Personal Care...</td></tr></table>

# D PROMPTS

In this part, we show the prompts designed for annotations. We require LLMs to output a Python dictionary-like object so that it's easier to extract the results from the output texts. We find that LLMs sometimes generate information without following the users' instructions. As a result, we design the self-correction prompt to fix those invalid outputs (Table 5).

Table 5: Prompt used for self-correction 

<table><tr><td>Previous prompt: (Previous input)Your previous output doesn’t follow the format, please correct it old output: (Previous output)Your previous answer (Previous answer) is not a valid class.Your should only output categories from the following list:(Lists of label names)New output here:</td></tr></table>

We then demonstrate two concrete examples of our prompts. The first prompt is based on the zero-shot consistency strategy (see Table 6). The second prompt is based on the few-shot consistency strategy (see Table 7). It should be noted that the latter prompt is only used to be compared with zero-shot prompts. We don't use it in Section 4. For one-shot prompts, we leverage large language models to automatically generate both the top K predictions and confidence scores according to the philosophy of Zhang et al. (2023).

Table 6: Full prompt example for zero-shot annotation with consistency strategy on the PUBMED dataset 

<table><tr><td>Input: Question: (Contents) Paper: Title: Heritability of pancreatic beta-cell function among nondiabetic members of Caucasian familial type 2 diabetic kindreds. Abstract: Both defective insulin secretion and insulin resistance have been reported in relatives of type 2 diabetic subjects. We tested 120 members of 26 families with a type 2 diabetic sibling pair with a tolbutamide-modified, frequently sampled i.v....</td></tr><tr><td>Task:There are following categories:[diabetes mellitus experimental, diabetes mellitus type 1, diabetes mellitus type 2]What&#x27;s the category of this paper?Provide your 3 best guesses and a confidence number that each is correct (0 to 100) for the following question from most probable to least. The sum of all confidence should be 100. For example, [{&quot;answer&quot;: &lt;your_first_answer&gt;, &quot;confidence&quot;: &lt;confidence_for_first_answer&gt;}, ... ]Output:</td></tr></table>

Table 7: Full prompt example for 1-shot annotation with TopK strategy on the CITESEER dataset 

<table><tr><td>Input: I will first give you an example and you should complete task following the example. Question: (Contents for in-context learning samples) Argument in Multi-Agent Systems Multi-agent systems ...(Skipped for clarity) Computational societies are developed for two primary reasons: Mode...Task:There are following categories:[agents, machine learning, information retrieval, database, human computer interaction, artificial intelligence]What&#x27;s the category of this paper?Provide your 3 best guesses and a confidence number that each is correct (0 to 100) for the following question from most probable to least. The sum of all confidence should be 100. For example, [{&quot;answer&quot;:,, &quot;confidence&quot;:}, ... ]Output:[{&quot;answer&quot;: &quot;agents&quot;, &quot;confidence&quot;: 60}, {&quot;answer&quot;: &quot;artificial intelligence&quot;, &quot;confidence&quot;: 30}, {&quot;answer&quot;: &quot;human computer interaction&quot;, &quot;confidence&quot;: 10}]Question: Decomposition in Data Mining: An Industrial Case Study Data (Contents for data to be annotated)Task:There are following categories:[agents, machine learning, information retrieval, database, human computer interaction, artificial intelligence] What&#x27;s the category of this paper?Provide your 3 best guesses (The same instruction, skipped for clarity) ...Output:</td></tr></table>

# E COMPLETE RESULTS FOR THE PRELIMINARY STUDY ON EFFECTIVENESS OF CONFIDENCE-AWARE PROMPTS

The complete results for the preliminary study on confidence-aware prompts are shown in Table 8, Figure 6, and Figure 7.

Table 8: Accuracy of annotations generated by LLMs after applying various kinds of prompt strategies. We use yellow to denote the best and green to denote the second best result. The cost is determined by comparing the token consumption to that of zero-shot prompts. 

<table><tr><td rowspan="2">Prompt Strategy</td><td colspan="2">CORA</td><td colspan="2">CITESEER</td><td colspan="2">PUBMED</td><td colspan="2">OGBN-ARXIV</td><td colspan="2">OGBN-PRODUCTS</td><td colspan="2">WIKICS</td></tr><tr><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td><td>Acc (%)</td><td>Cost</td></tr><tr><td>Zero-shot</td><td> $68.33 \pm 6.55$ </td><td>1</td><td> $64.00 \pm 7.79$ </td><td>1</td><td> $88.67 \pm 2.62$ </td><td>1</td><td> $73.67 \pm 4.19$ </td><td>1</td><td> $75.33 \pm 4.99$ </td><td>1</td><td> $68.33 \pm 1.89$ </td><td>1</td></tr><tr><td>One-shot</td><td> $69.67 \pm 7.72$ </td><td>2.2</td><td> $65.67 \pm 7.13$ </td><td>2.1</td><td> $90.67 \pm 1.25$ </td><td>2.0</td><td> $69.67 \pm 3.77$ </td><td>1.9</td><td> $78.67 \pm 4.50$ </td><td>1.8</td><td> $72.00 \pm 3.56$ </td><td>2.4</td></tr><tr><td>TopK</td><td> $68.00 \pm 6.38$ </td><td>1.1</td><td> $66.00 \pm 5.35$ </td><td>1.1</td><td> $89.67 \pm 1.25$ </td><td>1.1</td><td> $72.67 \pm 4.50$ </td><td>1</td><td> $74.00 \pm 5.10$ </td><td>1.2</td><td> $72.00 \pm 2.16$ </td><td>1.1</td></tr><tr><td>Most Voting</td><td> $68.00 \pm 7.35$ </td><td>1.1</td><td> $65.33 \pm 7.41$ </td><td>1.1</td><td> $88.33 \pm 2.49$ </td><td>1.4</td><td> $73.33 \pm 4.64$ </td><td>1.1</td><td> $75.33 \pm 4.99$ </td><td>1.1</td><td> $69.00 \pm 2.16$ </td><td>1.1</td></tr><tr><td>Hybrid (Zero-shot)</td><td> $67.33 \pm 6.80$ </td><td>1.5</td><td> $66.33 \pm 6.24$ </td><td>1.5</td><td> $87.33 \pm 0.94$ </td><td>1.7</td><td> $72.33 \pm 4.19$ </td><td>1.2</td><td> $73.67 \pm 5.25$ </td><td>1.4</td><td> $71.00 \pm 2.83$ </td><td>1.4</td></tr><tr><td>Hybrid (One-shot)</td><td> $70.33 \pm 6.24$ </td><td>2.9</td><td> $67.67 \pm 7.41$ </td><td>2.7</td><td> $90.33 \pm 0.47$ </td><td>2.2</td><td> $70.00 \pm 2.16$ </td><td>2.1</td><td> $75.67 \pm 6.13$ </td><td>2.3</td><td> $73.67 \pm 2.62$ </td><td>2.9</td></tr></table>

![](images/d3052d11ad03af6aaf66ae834f63f613614630029e5f34cbd4db58875e7b4bcc.jpg)

<details>
<summary>line</summary>

| k    | Cora One-Shot | Cora Most Voting | Cora Top-K | Cora Hybrid(One-Shot) | Cora Zero-Shot | Cora Hybrid(Zero-Shot) | WikiCS One-Shot | WikiCS Most Voting | WikiCS Top-K | WikiCS Hybrid(One-Shot) | WikiCS Zero-Shot | WikiCS Hybrid(Zero-Shot) |
|------|---------------|------------------|----------|------------------------|----------------|-------------------------|-----------------|--------------------|--------------|-------------------------|------------------|--------------------------|
| 50   | 0.84          | 0.82             | 0.91     | 0.84                   | 0.79           | 0.95                    | 0.82            | 0.83               | 0.92         | 0.92                    | 0.85             | 0.96                     |
| 100  | 0.82          | 0.81             | 0.91     | 0.81                   | 0.77           | 0.91                    | 0.75            | 0.84               | 0.93         | 0.84                    | 0.89             | 0.91                     |
| 150  | 0.81          | 0.80             | 0.87     | 0.79                   | 0.77           | 0.90                    | 0.79            | 0.83               | 0.88         | 0.83                    | 0.89             | 0.90                     |
| 200  | 0.79          | 0.79             | 0.82     | 0.77                   | 0.76           | 0.86                    | 0.81            | 0.82               | 0.83         | 0.82                    | 0.88             | 0.86                     |
| 250  | 0.77          | 0.78             | 0.79     | 0.75                   | 0.75           | 0.84                    | 0.79            | 0.81               | 0.81         | 0.81                    | 0.85             | 0.84                     |
| 300  | 0.75          | 0.76             | 0.72     | 0.73                   | 0.73           | 0.76                    | 0.76            | 0.71               | 0.73         | 0.76                    | 0.75             | 0.76                     |
</details>

Figure 6: Relationship between the group accuracies sorted by confidence generated by LLMs. k refers to the number of nodes selected in the groups.

# F PRELIMINARY OBSERVATIONS OF LLM'S ANNOTATIONS

# F.1 LABEL-LEVEL OBSERVATIONS

We begin by examining the label-level characteristics of annotations generated by LLMs. Label distribution is critical as it can influence model training. For example, imbalanced training labels may make models overfit to the majority classes (He & Garcia, 2009). Specifically, we juxtapose the distribution of LLM annotations against the ground truths. To facilitate this analysis, we employ noise transition matrices, a tool frequently used to study both real-world noisy labels (Wei et al., 2021) and synthetic ones (Zhu et al., 2021b). Considering a classification problem with N classes, the noise transition matrix T will be an $N \times N$ matrix where each entry $T_{ij}$ represents the probability that an instance from the true class i is given the label j. If i = j, then $T_{ij}$ is the probability that a true class i is correctly labeled as i. If $i \neq j$ , then $T_{ij}$ is the probability that a true class i is mislabeled as j.

To generate the noise transition matrices, we sample 1000 nodes randomly for each dataset. Figure 8 demonstrates the noise transition matrix for WIKICS, CORA, and CITESEER, three datasets with proper number of classes for visualization. We demonstrate the results of adopting the “zero-shot hybrid” prompting strategy in this figure. We illustrate the results for few-shot demonstration strategies and synthetic noisy labels in Appendix F.1. Since the number of samples in each class is imbalanced, we also show the label distribution for both ground truth and LLM’s annotations in Figure 11.

An examination of Figure 8 reveals intriguing patterns. FIRST, the quality of annotations, defined as the proportion of annotations aligned with the ground truth, exhibits significant variation across different classes. For instance, in the WIKICS dataset, LLMs produce perfect annotations for samples where the ground truth class is $c_{0}$ (referring to “Computational linguistics”). In contrast, only 31% of annotations are accurate for samples belonging to $c_{7}$ , referring to “distributed computing architecture”. Similar phenomena can be observed in the CORA and CITESEER datasets. SECOND, for those classes with low annotation quality, incorrect annotations will not be distributed to other classes with a uniform probability. Instead, they are more likely to fall into one or two specific categories.

For example, for WIKICS, $c_{7}$ (“distributed computing architecture”) tends to flip into $c_{8}$ (“web technology”). The reason may be that these two classes present similar meanings, and for some samples, categorizing them into any of these two classes is reasonable. Moreover, we find that such

![](images/dc975474735c527d9ad365e5a67687f7597aba7e7fee95ffab9a8f51188f782c.jpg)  
Figure 7: Relationship between the group accuracies sorted by confidence generated by LLMs. k refers to the number of nodes selected in the groups.

kind of flipping can be asymmetric. Although $c_{7}$ in WIKICS tends to flip into $c_{8}$ , samples from $c_{8}$ never flip into $c_{7}$ . These properties make LLMs' annotations highly different from the synthetic noisy labels commonly adopted in previous literature (Song et al., 2022), which are much more uniform among different classes.

![](images/daa1d1fb2f1581cf2f0463103d2fd542205d324ece6842c84be6a422ab0fd81f.jpg)  
(a) WIKICS

![](images/706d60e499603fa710249c1f888ffe920b871c48a1ec60520c0b92721f9344b8.jpg)  
(b) CORA

![](images/dfd0a4b348e9e27ee7002f965d599d35d6e89bfeed597c4b2776accfd348f4d8.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.52 | 0.10 | 0.05 | 0.01 | 0.10 | 0.23 |
| 1 | 0.04 | 0.77 | 0.06 | | 0.01 | 0.12 |
| 2 | 0.02 | 0.21 | 0.70 | 0.02 | 0.05 | 0.01 |
| 3 | 0.02 | 0.05 | 0.18 | 0.64 | 0.01 | 0.10 |
| 4 | 0.04 | 0.10 | 0.12 | 0.02 | 0.69 | 0.03 |
| 5 | 0.08 | 0.31 | 0.11 | | 0.07 | 0.43 |
</details>

(c) CITESEER

Figure 8: Noise transition matrix plot for WIKICS, CORA, and CITESEER with annotations generated by LLMs   
![](images/ddeabdacf435d01dcf4f80ea2111122ce80daf3231efe4490b7d5ee87e9bfef4.jpg)  
(a) WIKICS

![](images/070eb8e4b1788b1bf58b3df861ab0ed1938b8ee56731252a1dda353f46a770c9.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.62 | 0.07 | 0.05 | 0.03 | 0.07 | 0.03 | 0.12 |
| 1 | 0.01 | 0.63 | 0.03 | 0.02 | 0.12 | 0.00 | 0.17 |
| 2 | 0.07 | 0.02 | 0.64 | 0.01 | 0.21 | 0.03 | 0.02 |
| 3 | 0.02 | 0.04 | 0.01 | 0.83 | 0.06 | 0.04 | 0.01 |
| 4 | 0.23 | 0.09 | 0.06 | 0.02 | 0.38 | 0.07 | 0.16 |
| 5 | - | 0.08 | 0.01 | 0.05 | 0.04 | 0.76 | 0.05 |
| 6 | 0.01 | 0.03 | 0.03 | 0.01 | 0.09 | 0.01 | 0.83 |
</details>

(b) CORA

![](images/6a314f278e16cb83b81dc762c9be71c7b081e512602620cbc9f94e0c5655ff11.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.82 | 0.06 | 0.03 | 0.01 | 0.06 | 0.03 |
| 1 | 0.02 | 0.81 | 0.08 | 0.01 | 0.01 | 0.07 |
| 2 | 0.01 | 0.18 | 0.75 | 0.03 | 0.02 | 0.00 |
| 3 | 0.02 | 0.04 | 0.23 | 0.65 | 0.01 | 0.05 |
| 4 | 0.03 | 0.08 | 0.15 | 0.04 | 0.69 | 0.02 |
| 5 | 0.18 | 0.43 | 0.10 | 0.02 | 0.05 | 0.22 |
</details>

(c) CITESEER   
Figure 9: Noise transition matrix plot for WIKICS, CORA, and CITESEER with annotations generated by LLMs using prompts with few-shot demonstrations

Referring to Figure 11, we observe a noticeable divergence between the annotation distribution generated by LLMs and the original ground truth distribution across various datasets. Notably, in the WIKICS dataset, what is originally a minority class, $c_{8}$ , emerges as one of the majority classes. A similar shift is evident with class $c_{6}$ in the CORA dataset. This trend can be largely attributed to the asymmetry inherent in the noise transition matrix.

![](images/786c555c4bdca114f6d3a28edc25242e39f62065231dbb2292fc8f711e72810e.jpg)  
(a) WIKICS

![](images/8f344f3a68cfafd4cb43df9bf6c87aac255b07b2b07f5fe257c96ec5c0b3329f.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.61 | 0.05 | 0.05 | 0.05 | 0.12 | 0.07 | 0.05 |
| 1 | 0.05 | 0.65 | 0.07 | 0.05 | 0.07 | 0.05 | 0.05 |
| 2 | 0.07 | 0.07 | 0.62 | 0.06 | 0.03 | 0.07 | 0.08 |
| 3 | 0.05 | 0.08 | 0.05 | 0.66 | 0.06 | 0.06 | 0.04 |
| 4 | 0.07 | 0.03 | 0.08 | 0.04 | 0.69 | 0.05 | 0.05 |
| 5 | 0.04 | 0.01 | 0.03 | 0.07 | 0.07 | 0.73 | 0.05 |
| 6 | 0.03 | 0.10 | 0.03 | 0.03 | 0.06 | 0.06 | 0.68 |
</details>

(b) CORA

![](images/ae7601ce904d61effa1f12fc40db77d2c413679e1073a4783c0e976fa20d5826.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.66 | 0.05 | 0.05 | 0.04 | 0.11 | 0.08 |
| 1 | 0.06 | 0.62 | 0.10 | 0.07 | 0.07 | 0.09 |
| 2 | 0.05 | 0.09 | 0.66 | 0.07 | 0.07 | 0.05 |
| 3 | 0.08 | 0.06 | 0.06 | 0.66 | 0.06 | 0.08 |
| 4 | 0.08 | 0.05 | 0.06 | 0.08 | 0.67 | 0.06 |
| 5 | 0.06 | 0.11 | 0.07 | 0.06 | 0.12 | 0.58 |
</details>

(c) CITESEER

Figure 10: Noise transition matrix plot for WIKICS, CORA, and CITESEER with annotations generated by injecting random noise into the ground truth.   
![](images/21988a926eded83530491f4e7d395d2f1b4e0fcdd98c6c6451974470c87257f5.jpg)

<details>
<summary>bar</summary>

| Labels | Ground truth | Annotations |
| ------ | ------------ | ----------- |
| 0      | 80           | 20          |
| 1      | 50           | 50          |
| 2      | 160          | 180         |
| 3      | 130          | 190         |
| 4      | 220          | 230         |
| 5      | 70           | 60          |
| 6      | 40           | 50          |
| 7      | 30           | 80          |
| 8      | 140          | 40          |
| 9      | 100          | 120         |
</details>

(a) WIKICS

![](images/33dd30ec0a2dfda8a4d3e211fa6cbeb3f1c46f4e3d524510cd2aed9a78208207.jpg)

<details>
<summary>bar</summary>

| Labels | Ground truth | Annotations |
|---|---|---|
| 0 | 60 | 60 |
| 1 | 250 | 320 |
| 2 | 110 | 120 |
| 3 | 120 | 145 |
| 4 | 85 | 135 |
| 5 | 105 | 75 |
| 6 | 270 | 145 |
</details>

(b) CORA   
Figure 11: Label distributions for ground truth labels and LLM's annotations

# F.2 RELATIONSHIP BETWEEN ANNOTATION QUALITY AND C-DENSITY

As shown in Figure 12, the phenomenon on PUBMED is not as evident as on the other datasets. One possible reason is that the annotation quality of LLM on PUBMED is exceptionally high, making the differences between different groups less distinct. Another possible explanation, as pointed out in (Chen et al., 2023), is that LLM might utilize some shortcuts present in the text attributes during its annotation process, and thus the correlation between features and annotation qualities is weakened.

![](images/1c764be02ab6215eecd268d6c19224d30eea38158921e202313aa1d5553314e7.jpg)

<details>
<summary>bar</summary>

| Region Index | Average Accuracy |
| ------------ | ---------------- |
| 0            | 0.95             |
| 1            | 0.86             |
| 2            | 0.90             |
| 3            | 0.91             |
| 4            | 0.85             |
| 5            | 0.87             |
| 6            | 0.86             |
| 7            | 0.84             |
| 8            | 0.84             |
| 9            | 0.78             |
</details>

(a) WIKICS

![](images/bc401d6e163a0a4876ffa0ca895c674d6e1a3b58f9957d1144ac45489e7777d3.jpg)

<details>
<summary>bar</summary>

| Region Index | Average Accuracy |
| ------------ | ---------------- |
| 0            | 0.94             |
| 1            | 0.91             |
| 2            | 0.87             |
| 3            | 0.82             |
| 4            | 0.88             |
| 5            | 0.91             |
| 6            | 0.85             |
| 7            | 0.88             |
| 8            | 0.92             |
| 9            | 0.94             |
</details>

(b) PUBMED   
Figure 12: Relationship between the group accuracies and distances to the clustering centers. The bar shows the average accuracy inside the selected group. The line shows the accumulated average accuracy.

# G EFFECTIVENESS OF CONFIDENCE GENERATED BY LLMs

In this section, we demonstrate the effectiveness of confidence generated by LLMs with a case study on the Cora dataset. We plot the confidence calibration plot with three prompt strategies: zero-shot, TopK, and hybrid. From the results, we can see that hybrid prompt can generate accurate and diverse confidence scores, which is more effective.

![](images/c4c971fc01cd01fbb0f85be8b85ae3ab4581b23f5c301ce360d8d198d9b62bb4.jpg)

<details>
<summary>bar</summary>

| Confidence | Accuracy |
| ---------- | -------- |
| 0.0        | 0.35     |
| 0.7        | 0.25     |
| 0.8        | 0.35     |
| 0.9        | 0.8      |
| 1.0        | 0.35     |
</details>

(a) Zero-shot

![](images/556d21e6455dbbf363e8a076f85145e059a99e032d5f6e135ae1507cd3b8bccc.jpg)

<details>
<summary>bar</summary>

| Confidence | Accuracy |
| ---------- | -------- |
| 0.7        | 0.35     |
| 0.8        | 0.6      |
| 0.9        | 0.85     |
| 1.0        | 1.0      |
</details>

(b) TopK

![](images/afb9b2e4ab7b7c1cb3b62d285b024aa88cddcb37a902dd97ecd9ca8c626154a4.jpg)

<details>
<summary>histogram</summary>

| Confidence Range | Accuracy |
| ---------------- | -------- |
| 0.0 - 0.1        | 0.0      |
| 0.1 - 0.2        | 0.0      |
| 0.2 - 0.3        | 0.3      |
| 0.3 - 0.4        | 0.2      |
| 0.4 - 0.5        | 0.4      |
| 0.5 - 0.6        | 0.3      |
| 0.6 - 0.7        | 0.5      |
| 0.7 - 0.8        | 0.6      |
| 0.8 - 0.9        | 0.8      |
| 0.9 - 1.0        | 1.0      |
</details>

(c) Hybrid   
Figure 13: Comparison of three prompt strategies

# H HYPERPARAMTERS

We now demonstrate the hyper-parameters adopted in this paper, which is inspired by Hu et al. (2020b):

1. For small-scale datasets including CORA, CITESEER, PUBMED, and WIKICS, we set: learning rate to 0.01, weight decay to $5e^{-4}$ , hidden dimension to 64, dropout to 0.5.   
2. For large-scale datasets including OGBN-ARXIV and OGBN-PRODUCTS, we set: learning rate to 0.01, weight decay to $5e^{-4}$ , hidden dimension to 256, dropout to 0.5.

# I COMPARING LLMs ANNOTATIONS TO SYNTHETIC NOISY LABELS

![](images/edc2470ef2824d14d0b33c4a93d18d1b8f070ff1d447fb85b734640c58178f09.jpg)

<details>
<summary>line</summary>

| Epochs | Accuracy (Red) | Accuracy (Blue) | Accuracy (Green) | Accuracy (Yellow) |
| ------ | -------------- | --------------- | ---------------- | ----------------- |
| 0      | 0.6            | 0.6             | 0.6              | 0.6               |
| 25     | 0.83           | 0.85            | 0.7              | 0.78              |
| 50     | 0.83           | 0.95            | 0.65             | 0.78              |
| 75     | 0.83           | 0.98            | 0.6              | 0.78              |
| 100    | 0.83           | 0.99            | 0.6              | 0.78              |
| 125    | 0.83           | 0.99            | 0.6              | 0.78              |
| 150    | 0.83           | 0.99            | 0.6              | 0.78              |
</details>

(a) CORA

![](images/937cea5bcedd8af4a0a8a641e7d17af83d784a48c805cbc99042176fbb614641.jpg)

<details>
<summary>line</summary>

| Epochs | Accuracy (Line 1) | Accuracy (Line 2) | Accuracy (Line 3) | Accuracy (Line 4) | Accuracy (Line 5) |
| ------ | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0      | 0.5               | 0.5               | 0.5               | 0.5               | 0.5               |
| 25     | 0.95              | 0.9               | 0.75              | 0.65              | 0.55              |
| 50     | 1.0               | 1.0               | 0.7               | 0.65              | 0.5               |
| 75     | 1.0               | 1.0               | 0.7               | 0.65              | 0.5               |
| 100    | 1.0               | 1.0               | 0.7               | 0.65              | 0.5               |
| 125    | 1.0               | 1.0               | 0.7               | 0.65              | 0.5               |
| 150    | 1.0               | 1.0               | 0.7               | 0.65              | 0.5               |
</details>

(b) CITESEER   
Figure 14: The red curve represents the performance of models trained with ground truth labels. The yellow curve represents the performance of models trained with LLMs' annotations with all wrong labels fixed. The blue curve represents the models trained by LLMs' annotations. The green curve represents the models trained by synthetic noisy labels with the same annotation quality as LLMs' annotations. The solid line represents the performance on the test set, while the dashed line represents the performance on the training set.

# J EXTRA RESULTS FOR THE COMPARATIVE STUDY IN TABLE 2

In Table 10, we further demonstrate the results for combing weighted loss, difficulty-aware selection, and post-filtering to degree-based selection and pagerank-based selection. In Table 9, we explore the effectiveness of applying PS and DA together. We find that applying them simultaneously usually won't get good performance, which means it's probably not the proper way to integrate LLMs' confidence into selection. As a comparison, we find that using weighted cross-entropy loss can enhance the performance most of the time. Even though it can not surpass the baselines, the gap is very small, which shows its effectiveness.

Table 9: Ablation study for the effectiveness of different combinations. 

<table><tr><td></td><td>Cora</td><td>CiteSeer</td></tr><tr><td>AGE</td><td>69.15 ± 0.38</td><td>54.25 ± 0.31</td></tr><tr><td>DA-AGE</td><td>74.38 ± 0.24</td><td>59.92 ± 0.42</td></tr><tr><td>DA-AGE-W</td><td>74.96 ± 0.22</td><td>58.41 ± 0.45</td></tr><tr><td>PS-DA-AGE</td><td>71.53 ± 0.19</td><td>56.38 ± 0.14</td></tr><tr><td>RIM</td><td>69.86 ± 0.38</td><td>63.44 ± 0.42</td></tr><tr><td>DA-RIM</td><td>73.99 ± 0.44</td><td>60.33 ± 0.40</td></tr><tr><td>DA-RIM-W</td><td>74.73 ± 0.41</td><td>60.80 ± 0.57</td></tr><tr><td>PS-DA-RIM</td><td>72.34 ± 0.19</td><td>60.33 ± 0.40</td></tr></table>

Table 10: Extra results for degree-based selection and pagerank-based selection 

<table><tr><td></td><td>CORA</td><td>CITESEER</td><td>PUBMED</td><td>WIKICS</td><td>OGBN-ARXIV</td><td>OGBN-PRODUCTS</td></tr><tr><td>Degree</td><td>68.67 ± 0.30</td><td>60.23 ± 0.54</td><td>67.77 ± 0.07</td><td>65.38 ± 0.35</td><td>54.98 ± 0.37</td><td>71.22 ± 0.21</td></tr><tr><td>Degree-W</td><td>69.86 ± 0.35</td><td>60.47 ± 0.49</td><td>68.24 ± 0.09</td><td>65.61 ± 0.31</td><td>55.69 ± 0.24</td><td>71.96 ± 0.23</td></tr><tr><td>DA-Degree</td><td>72.86 ± 0.27</td><td>60.23 ± 0.54</td><td>74.51 ± 0.04</td><td>63.40 ± 0.51</td><td>55.32 ± 0.33</td><td>44.41 ± 0.53</td></tr><tr><td>PS-Degree-W</td><td>70.92 ± 0.28</td><td>62.36 ± 0.69</td><td>74.83 ± 0.05</td><td>67.21 ± 0.29</td><td>55.89 ± 0.39</td><td>71.57± 0.25</td></tr><tr><td>DA-Degree-W</td><td>73.01 ± 0.24</td><td>61.29 ± 0.47</td><td>74.11 ± 0.04</td><td>63.14 ± 0.55</td><td>55.35 ± 0.32</td><td>47.90 ± 0.45</td></tr><tr><td>Pagerank</td><td>70.31 ± 0.42</td><td>61.21 ± 0.11</td><td>68.58 ± 0.14</td><td>67.13 ± 0.46</td><td>59.52 ± 0.03</td><td>69.20 ± 0.32</td></tr><tr><td>Pagerank-W</td><td>71.50 ± 0.44</td><td>61.97 ± 0.19</td><td>68.86 ± 0.19</td><td>69.61 ± 0.34</td><td>59.60 ± 0.04</td><td>69.75 ± 0.29</td></tr><tr><td>DA-Pagerank</td><td>74.34 ± 0.41</td><td>60.44 ± 0.40</td><td>72.84 ± 0.15</td><td>67.15 ± 0.44</td><td>58.82 ± 0.52</td><td>54.77 ± 0.36</td></tr><tr><td>PS-Pagerank-W</td><td>74.81 ± 0.37</td><td>63.27 ± 0.34</td><td>68.23 ± 0.17</td><td>69.86 ± 0.29</td><td>58.84 ± 0.14</td><td>69.69 ± 0.45</td></tr><tr><td>DA-Pagerank-W</td><td>75.62 ± 0.39</td><td>61.25 ± 0.45</td><td>73.60 ± 0.22</td><td>68.19 ± 0.32</td><td>59.40 ± 0.26</td><td>55.57 ± 0.24</td></tr></table>

# K THEORETICAL MOTIVATION

In this section, we further analyze the theoretical motivation for difficulty-aware selection. In a nutshell, our objective is to show why C-Density is a useful metric to select nodes with high annotation quality.

Given the parameter distribution of LLMs Q, the parameter distribution of encoder P (SBERT), we assume ground truth $Y_{L} \in R^{N \times M}$ , pseudo label $Y \in R^{N \times M}$ , and node features encoded by the encoder $X \in R^{N \times d}$ . Here, M denotes the number of classes and d denotes the hidden dimension. Our objective is to maximize the accuracy of annotations, which is thus to minimize the discrepancies between Y and $Y_{L}$ .

$$
\min _ {(n _ {1}, \dots , n _ {k})} \mathbb {E} _ {\theta \sim Q} f (Y) = \ell (Y, Y _ {L}) = \sum_ {i = 1} ^ {k} \ell (y ^ {n _ {i}}, y _ {L} ^ {n _ {i}})
$$

where $(n_{1},\ldots,n_{k})$ is the index of the k selected nodes, and $f(Y)$ is defined as follows: $x^{n_{i}}$ represents the feature of node $n_{i}$ , while $x_{L}^{n_{i}}$ represents the unknown latent embedding of node $n_{i}$ .

$$
f (Y) = \ell (Y, Y _ {L}) = \sum_ {i = 1} ^ {k} \ell (y ^ {n _ {i}}, y _ {L} ^ {n _ {i}}) = \ln (1 + \sum_ {i = 1} ^ {k} \| x ^ {n _ {i}} - x _ {L} ^ {n _ {i}} \| ^ {2}) = \ln (1 + \| X - X _ {L} \| ^ {2})
$$

Since we only have access to the node features $X$ generated by encoder $\theta$ , we need to make a connection between $\mathcal{Q}$ and $\mathcal{P}$ .

Lemma 1 For any annotation $y$ generated by LLMs, $\mathbb{E}_{\theta \sim \mathcal{Q}}f(y) \leq \log \mathbb{E}_{\theta' \sim \mathcal{P}}\exp(f(y)) + KL(\mathcal{Q}||\mathcal{P})$ .

Proof.

$$
\mathbb {E} _ {\theta^ {\prime} \sim \mathcal {P}} f (y) = \int f (y) p (y) d y = \int f (y) \frac {p (y)}{q (y)} q (y) d y = \mathbb {E} _ {\theta \sim \mathcal {Q}} f (y) \frac {p (y)}{q (y)}
$$

Since $KL(\mathcal{Q}||\mathcal{P}) = \mathbb{E}_{\theta \sim Q}\log \frac{q(y)}{p(y)},$

$$
\log \mathbb {E} _ {\theta^ {\prime} \sim \mathcal {P}} f (y) = \log \mathbb {E} _ {\theta \sim \mathcal {Q}} f (y) \frac {p (y)}{q (y)} \geq \mathbb {E} _ {\theta \sim \mathcal {Q}} \log f (y) \frac {p (y)}{q (y)} = \mathbb {E} _ {\theta \sim \mathcal {Q}} \log f (y) - K L (\mathcal {Q} \| \mathcal {P})
$$

Then, we transform the objective into

$$
\min _ {(n _ {1}, \dots , n _ {k})} \mathbb {E} _ {\theta^ {\prime} \sim \mathcal {P}} \exp (f (Y))
$$

Assuming the label distribution $\mathcal{H}$ , we further have

$$
\min _ {(n _ {1}, \dots , n _ {k})} \mathbb {E} _ {Y \sim \mathcal {H}} \mathbb {E} _ {\theta^ {\prime} \sim \mathcal {P}} \exp (f (Y)) = \min _ {(n _ {1}, \dots , n _ {k})} \mathbb {E} _ {Y \sim \mathcal {H}} \mathbb {E} _ {\theta \sim \mathcal {P}} \| X - X _ {L} \| ^ {2}
$$

Assuming X and $X_{L}$ follows gaussian distribution, where $X \sim N(\mu_{i}, \sigma_{i}); X_{L} \sim N(\mu_{j}, \sigma_{j})$ , then

$$
\begin{array}{l} \operatorname{E} \left(\| X - X _ {L} \| ^ {2}\right) = \operatorname{E} \left(\| (X - \mu_ {i}) - (X _ {L} - \mu_ {j}) + (\mu_ {i} - \mu_ {j}) \| ^ {2}\right) \\ = \mathrm{E} \left(\| X - \mu_ {i} \| ^ {2}\right) + \mathrm{E} \left(\| X _ {L} - \mu_ {j} \| ^ {2}\right) + \| \mu_ {i} - \mu_ {j} \| ^ {2} \\ = n \sigma_ {i} ^ {2} + n \sigma_ {j} ^ {2} + \left\| \mu_ {i} - \mu_ {j} \right\| ^ {2} \\ \end{array}
$$

Since $(\mu_{j},\sigma_{j})$ is unknown, $\mu_{i}$ is fixed, thus we want to minimize $\sigma_{i}$ . Given an arbitrary node, $\sigma_{i}$ can be viewed as the distance to the clustering centers. The smaller $\sigma_{i}$ is, the smaller the corresponding $\operatorname{E}\left(\|X-X_{L}\|^{2}\right)$ also becomes, indicating that the minimum value of $\mathbb{E}_{\theta\sim\mathcal{Q}}f(y)$ is attained when $\sigma_{i}$ is at its smallest. This demonstrates why nodes closer to the clustering centers are “preferred” by LLMs and thus achieve better annotation quality.

# L DESIGN PHILOSOPHY BEHIND LLMGNN

Regarding LLMGNN, we have the following two key designs:

1. In the annotation process, we do not consider structural information, hence we have designed a structure-free prompt. 2. In the choice of a LLM, we do not use the more advanced GPT-4, but instead, we choose the more cost-effective GPT-3.5-turbo. In practice, we find that these two designs are currently the most appropriate because: 1. We discover that using a structure-aware prompt does not effectively improve annotation quality without introducing ground truth labels, due to the limited structural understanding capabilities of LLMs. 2. We find that compared to GPT-3.5-turbo, the improvement brought by GPT-4 is very limited. After exploration, we realize that this is related to the ambiguity in the ground truth labels of the node classification task. Currently, GNN is the best tool capable of utilizing structural information to handle ambiguity.

To begin with, assigning correct labels for node classification is hard for LLMs. We show the annotation quality (the ratio of annotations matching the ground truth labels) of GPT3.5 and GPT4 in Table 11 by randomly selecting the annotated samples (140 nodes for Cora, and 120 nodes for CiteSeer, and we control the seed to make sure we select the same set of nodes). We can see that GPT4 doesn't give much better annotations (in terms of accuracy) than GPT3.5. The overall performance (less than $70\%$ ) indicates that node classification is not an easy task for LLMs.

If we further check the annotation results given by these two different models, we find that one common bottleneck preventing these models from getting better results is the annotation bias (also can be viewed as "label ambiguity") of the labels. We showcase one example in Table 12.

Table 11: Comparison of GPT-3.5 and GPT-4 on the annotation task 

<table><tr><td></td><td>CORA</td><td>CITESEER</td></tr><tr><td>GPT3.5</td><td>67.62 ± 2.05</td><td>66.39 ± 8.62</td></tr><tr><td>GPT4</td><td>68.81 ± 1.87</td><td>65.56 ± 9.29</td></tr></table>

Table 12: An example from the CITESEER dataset

Input: Attribute of one node from Citeseer: 'Discovering Web Access Patterns and Trends by Applying OLAP and Data Mining Technology on Web Logs As a confluence of data mining and WWW technologies, it is now possible to perform data mining on web log records collected from the Internet web page access history...

The ground truth label for this node is “Database”, while the prediction of both two LLMs is “Information Retrieval”. From human beings’ understanding, both “Database” and “Information retrieval” are somewhat reasonable categories of this node. However, because of the single label setting of node classification, only one of them is correct. This somehow demonstrates that to get high accuracy on node classification, the models need to capture such kinds of dataset-specific bias in the annotation (the information in LLMs is more like “commonsense knowledge”, so there can be a mismatch). Structural information may help us find such kind of bias. For example, if the category of neighboring nodes is more easier to predict, and most of them should be related to “Database”, then this node is more likely to come from “Database”. This phenomenon has also been observed in Chen et al. (2023).

So, why LLMs with structure-aware prompts can not improve their performance by introducing that structural information (shown in Table 13)? It's mainly because of LLMs' poor understanding capability of structural information. Huang et al. (2023) and Wang et al. (2023a) try various kinds of structure-aware prompts, and find that LLMs present limited structure reasoning abilities with current prompt designs. Chen et al. (2023) and Huang et al. (2023) further show that incorporating neighboring ground truth labels can improve the performance. However, ground truth labels are not available in the label-free settings. As a comparison, we find that merely summarizing the neighboring contents may not effectively improve the performance. Under the current structure-aware prompt design, LLMs can only utilize the structural information in a very limited manner (like using neighboring labels). As a comparison, message-passing GNNs can capture complicated structural information effectively and efficiently. For example, we try replacing GNN with MLP in LLMGNN in the Table 14, and we observe a huge performance gap.

Table 13: Comparison of using structure-aware and structure-free prompts 

<table><tr><td></td><td>CORA</td><td>CORA</td><td>CITESEER</td><td>CITESEER</td></tr><tr><td></td><td>No Struct</td><td>Struct</td><td>No Struct</td><td>Struct</td></tr><tr><td>FeatProp</td><td> $72.82 \pm 0.08$ </td><td> $69.08 \pm 0.39$ </td><td> $66.61 \pm 0.55$ </td><td> $65.92 \pm 0.43$ </td></tr><tr><td>FeatProp + PS</td><td> $75.54 \pm 0.34$ </td><td> $67.55 \pm 0.70$ </td><td> $69.06 \pm 0.32$ </td><td> $67.80 \pm 0.45$ </td></tr></table>

Based on the above reasons, we only use textual information in the labeling process, while in the training process of GNN, we utilize structural information. At the current stage, we believe this to be a more effective paradigm. This highlights the motivation for us to design LLMGNN which can enjoy the advantages of both LLMs and GNNs while mitigating their limitations.

Designing an effective structure-aware prompt is a valuable future direction, and methods like Agent-based prompting (Wang et al., 2023b) (multi-round prompt) may help to further improve the performance. The main focus of our paper is to propose a flexible framework that supports various kinds of prompt designs, and new prompt designs can also enhance the effectiveness of our framework.

Table 14: Comparisons of LLMs-as-Predictors, LLMGNN, and LLMMLP, which demonstrates the superiority of GNN 

<table><tr><td></td><td>LLMs-as-Predictors</td><td>LLMGNN</td><td>LLMMLP</td></tr><tr><td>CORA</td><td>68.33</td><td>76.23</td><td>67.06</td></tr></table>

The importance of the prompt is reflected in the fact that if we can improve the quality of annotations, we can further enhance the effectiveness of LLMGNN. For example, if we correct a portion of incorrect annotations to the right ones, we may improve the performance of LLMGNN. The “original quality” means the original annotation given by LLMs. “+k%” means that we randomly turn “k%” wrong annotations into correct ones. The results are shown in Table 15.

Table 15: The performance-changing trend of Random selection and PS-FeatProp-W selection when fix a portion of the wrong annotations 

<table><tr><td></td><td>original quality</td><td>+7%</td><td>+12%</td><td>+17%</td></tr><tr><td>Random</td><td>70.48</td><td>72.66</td><td>75.92</td><td>78.58</td></tr><tr><td>PS-FeatProp-W</td><td>76.23</td><td>77.95</td><td>80.25</td><td>81.96</td></tr></table>

Moreover, we want to show that the annotation quality will influence the scaling behavior of LLMGNN when we increase the budget. From Table 16 (the first row means the budget equal to $k \times$ number of classes), we can see that (1) When the budget is low, the difference between LLMGNN trained by high-quality and low-quality annotations is smaller (< 20); when the budget is high, this difference becomes larger(> 20); (2) the performance of LLMGNN is closely related to the annotation quality (the performance of LLMGNN on PubMed is better than one on Cora). With better annotation quality, LLMGNN may potentially achieve a higher performance upper bound when we gradually increase the budgets.

Table 16: The scaling behavior of LLMGNN on CORA and PUBMED when we increase the budgets 

<table><tr><td></td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td><td>40</td><td>80</td><td>160</td></tr><tr><td>CORA</td><td>57.35</td><td>66.45</td><td>68.42</td><td>70.17</td><td>69.64</td><td>70.68</td><td>72.07</td><td>72.73</td></tr><tr><td>PUBMED</td><td>62.49</td><td>64.56</td><td>72.44</td><td>73.16</td><td>75.92</td><td>77.8</td><td>82.38</td><td>82.5</td></tr></table>