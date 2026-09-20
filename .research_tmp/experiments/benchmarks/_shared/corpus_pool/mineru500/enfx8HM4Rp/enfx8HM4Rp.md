# Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks

Jun Yin\*

Central South University
yinjun2000@csu.edu.cn

Chaozhuo Li\*

Microsoft Research Asia
cli@microsoft.com

Hao Yan

Central South University
CSUyh1999@csu.edu.cn

Jianxun Lian

Microsoft Research Asia
jianxun.lian@microsoft.com

Senzhang Wang $^{†}$

Central South University
szwang@csu.edu.cn

# Abstract

Intrinsic interpretable graph neural networks aim to provide transparent predictions by identifying the influential fraction of the input graph that guides the model prediction, i.e., the explanatory subgraph. However, current interpretable GNNs mostly are dataset-specific and hard to generalize to different graphs. A more generalizable GNN interpretation model which can effectively distill the universal structural patterns of different graphs is until-now unexplored. Motivated by the great success of recent pre-training techniques, we for the first time propose the Pre-training Interpretable Graph Neural Network ( $\pi-GNN^{3}$ ) to distill the universal interpretability of GNNs by pre-training over synthetic graphs with ground-truth explanations. Specifically, we introduce a structural pattern learning module to extract diverse universal structure patterns and integrate them together to comprehensively represent the graphs of different types. Next, a hypergraph refining module is proposed to identify the explanatory subgraph by incorporating the universal structure patterns with local edge interactions. Finally, the task-specific predictor is cascaded with the pre-trained $\pi-GNN$ model and fine-tuned over downstream tasks. Extensive experiments demonstrate that $\pi-GNN$ significantly surpasses the leading interpretable GNN baselines with up to 9.98% interpretation improvement and 16.06% classification accuracy improvement. Meanwhile, $\pi-GNN$ pre-trained on graph classification task also achieves the top-tier interpretation performance on node classification task, which further verifies its promising generalization performance among different downstream tasks.

# 1 Introduction

Although graph neural networks (GNNs) [1, 2, 3, 4, 5] have achieved remarkable success in various applications [6, 7, 8, 9], their black-box nature prevents humans from understanding the inner decision-making mechanism [10, 11]. This issue calls for the development of intrinsic interpretable GNNs [12, 13, 14], which can reveal the mystery of "Which fraction of the input graph is the most vital and leads to the model prediction?" Intrinsic interpretable GNNs aim to identify an influential subgraph of the input graph, i.e., the explanation, and make final predictions under the guidance of the explanatory subgraph [12, 13, 14]. Constructing interpretable GNNs makes it possible to investigate

the decision-making mechanism and justify model predictions, which is critically important to develop trustworthy artificial intelligence.

However, a major issue of the existing interpretable GNNs $[12, 13, 14]$ is that they are mostly dataset-specific, which means an interpretable GNN model trained on a certain graph dataset (e.g., the semantic networks $[15, 16, 17]$ ) usually does not work well on another graph dataset (e.g., the molecular graphs $[18]$ ). It is also difficult for an interpretable GNN trained on a certain task (e.g., graph classification $[7]$ ) to generalize to another task (e.g., node classification $[6]$ ). Compared to the text or image data $[19, 20]$ , graphs with non-Eucildean structure are often associated with substantial node features of various domains, making the pre-training method in NLP and CV hard to be directly applied $[21]$ . Recently graph representation learning based pre-training methods have been investigated $[22, 21, 23, 24]$ , however, they target at downstream prediction tasks but ignore the interpretability of the GNNs. It is widely acknowledged that the topological structure of various graphs generally follows some universal structural patterns or properties which are transferable $[22]$ , such as the scale-free property $[25]$ , the motif distribution $[26]$ , and the core-periphery structure $[27]$ . Therefore, in this paper we argue that the intrinsic GNN interpretation contains some universal structural patterns, which are independent of the downstream tasks and generalizable to different types of graphs. Motivated by the great success of pre-training technique $[19, 20, 22, 21]$ , we for the first time study: whether we can and how to construct a pre-training interpretable GNN that is general enough to work well on different types of graphs and downstream tasks?

The challenges of designing a pre-training interpretable GNN are three-fold. First, labeling ground-truth explanation in the real-world graphs is extremely resource- and time-comsuming $[28]$ . The lacking of ground-truth explanation makes it hard to distill the universal interpretability in the pre-training phase. Second, multiple structural patterns usually co-exist in one graph dataset, such as the scale-free pattern and the motif distribution pattern in chemical molecule graphs $[29]$ . How to extract and integrate multiple structural patterns for a more comprehensive and general graph representation during pre-training is also challenging. Third, the local structural interactions such as the neighbor edges interaction $[30]$ should be considered when identifying explanations. However, due to the structural diversity of different local neighborhoods, it is non-trivial to incorporate the global structural patterns with the local structural interactions.

In this papaer, we propose a Pre-training Interpretable Graph Neural Network model ( $\pi$ -GNN for short), which is first pre-trained over a large synthetic graph dataset with ground-truth explanations and then fine-tuned on different downstream datasets and tasks. Specifically, we first construct a synthetic graph dataset named PT-Motifs that contains various structural patterns and ground truth explanations, over which the $\pi$ -GNN model is pre-trained. Considering the co-existence of multiple structural patterns, a structural pattern learning module is introduced to extract and integrate multiple structural patterns to make them generalizable to diverse graph datasets. To better identify the explanatory subgraph, a hypergraph refining module is also proposed to capture the local structural interaction and incorporate it with the universal structural patterns. It is also proved the structural representation ability of $\pi$ -GNN can approach the theoretical upper bound through the hypergraph refining process. Our main contributions are summarized as follows.

- We for the first time propose a pre-training interpretable GNN model $\pi$ -GNN, which can be generalized to different graph datasets and diverse downstream tasks.   
- To extract the universal interpretability of $\pi$ -GNN, we construct a synthetic graph classification dataset PT-Motifs with ground-truth explanations for pre-training.   
- Two innovative modules, i.e., the structural pattern learning module and the hypergraph refining module, are designed and integrated into $\pi$ -GNN. The former captures and integrates multiple universal structural patterns for generalizable graph representation. The latter incorporates the universal patterns with local structural interactions to identify the explanation.   
- Compared with the SOTA baselines, $\pi$ -GNN achieves up to $9.98\%$ ROC-AUC improvement in interpretation and $16.06\%$ accuracy improvement in prediction. Moreover, $\pi$ -GNN pre-trained on graph classification dataset is able to achieve comparable performance with the leading baselines on the node classification task.

# 2 Background

In this section, we briefly introduce the graph neural networks and the GNN explanation methods $^{4}$ . The key notations are summarized in Appendix A for clarity.

Graph Neural Networks. Graph structure data can be denoted as $G = (\mathcal{V}, \mathcal{E})$ with the node set $\mathcal{V}$ and the edge set $\mathcal{E}$ . The node features are represented as the matrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times d}$ and the edge features (if exist) are represented as $\mathbf{X}_E \in \mathbb{R}^{|\mathcal{E}| \times d_E}$ . The topological structure is usually represented as an adjacency matrix $\mathbf{A} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ , where the element $A_{ij} = 1$ indicates the edge $(i, j)$ exists and $A_{ij} = 0$ otherwise. Graph neural networks (GNNs) aim to learn expressive representation on graphs for the downstream tasks [1, 2, 3, 4, 32, 33], such as graph classification [7], node classification [6, 34], and link prediction [35, 36]. Typically, to learn the representation of node $v_i$ , GNNs aggregate the information from its neighborhood $\mathcal{N}(v_i)$ and then combine it with $v_i$ 's own features. For example, the operation of the $k$ -th GCNs layer can be formulated as follows [1],

$$
\mathbf {X} _ {\mathbf {k} + \mathbf {1}} = F \left(\mathbf {D} ^ {- \frac {1}{2}} \hat {\mathbf {A}} \mathbf {D} ^ {- \frac {1}{2}} \mathbf {X} _ {\mathbf {k}} \mathbf {W} _ {\mathbf {k}}\right), \tag {1}
$$

where $X_{k}$ and $X_{k+1}$ are the input and output of the k-th layer, $\hat{A} = A + I$ is the adjacent matrix with self-loops, and D is a diagonal matrix whose element $D_{i,i}$ represents the degree of $v_{i}$ . $W_{k}$ is a trainable matrix and $F(\cdot)$ is a non-linear activation function.

GNN Explanation Methods. GNN explanation methods can be categorized into the post-hoc explanation methods $[10, 11, 37, 38, 39]$ and the intrinsic interpretable methods $[3, 40, 12, 13, 14]$ . The post-hoc methods target at explaining the black-box models which are fixed or unaccessible. The interpretable methods devote to making transparent prediction from scratch, including not only the predicted label but also the influential subgraph that guides the prediction $[14]$ . In both the post-hoc $[10, 11, 37, 39]$ and the interpretable methods $[12, 13, 14]$ , the GNN explainer learns a contribution function h which maps each feature of the input graph into the contribution score to the predicted label. One insight in GNN explanation methods is that, the edge contribution function is more essential to GNN explanation compared with that of the node $[11, 13, 14]$ . For example, when some nodes are selected, it is non-trivial to identify the explanatory subgraph. On the contrary, when the important edges are selected, the correlated endpoints are naturally selected as well. We can naturally identify the explanatory subgraph or further explore the important subset of node features. In this work, we follow the previous works $[11, 39, 13, 14, 3, 40]$ and focus on the contribution of structure features (i.e., edges). Formally, we learn the contribution function h in terms of each edge in graph $G = (\mathcal{V}, \mathcal{E})$ as follows,

$$
\hat {\rho} = h (G), \tag {2}
$$

where $\hat{\rho} \in \mathbb{R}^{|\mathcal{E}|}$ and each element in $\hat{\rho}$ is the contribution score of the edge to the task label. Next, a selection module $S$ as follows is employed to select the edges of the explanatory subgraph $g$ , such as the top- $k$ selector [11, 13], the threshold selector [10], and the probabilistic selector [30, 14],

$$
g = \mathcal {S} (G, \hat {\rho}). \tag {3}
$$

# 3 Methodology

The framework of the proposed $\pi$ -GNN is shown in Figure 1, which contains an explainer pre-training phase and a conjoint fine-tuning phase. In the explainer pre-training phase, we pre-train the $\pi$ -GNN explainer over the synthetic dataset PT-Motifs with ground-truth explanations, by taking the binary edge classification as the pretext task. Afterwards, the pre-trained $\pi$ -GNN explainer is incorporated with task-specific predictor to identify explanatory subgraphs and provide transparent predictions on different tasks. During the fine-tuning phase, the explainer and the predictor are conjointly optimized. Next, we introduce the $\pi$ -GNN model in detail.

# 3.1 Explainer Pre-training Phase

Due to the lack of ground-truth explanation in real-world graph datasets, we first construct a large synthetic dataset with ground-truth explanation called PT-Motifs to support the explainer pre-training.

![](images/f68e8c104c0c2a461473e70913d3c004029be2e967ba3e8a2456f6ca3766d30f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Synthetic graphs"] --> B["Explainer pre-training phase"]
    B --> C["Structural Pattern Learning Module"]
    B --> D["Hypergraph Refining Module"]
    C --> E["Ground-truth explanation"]
    D --> E
    E --> F["Predicted explanation"]
    
    G["Real-world graphs"] --> H["Conjoint fine-tuning phase"]
    H --> I["Explainer"]
    I --> J["Structural Pattern Learning Module"]
    I --> K["Hypergraph Refining Module"]
    J --> L["Explanation"]
    K --> L
    L --> M["Task-Specific Predictor"]
    M --> N["Predict Loss"]
    
    style A fill:#f9f,stroke:#333
    style G fill:#bbf,stroke:#333
    style H fill:#bfb,stroke:#333
    style I fill:#ffb,stroke:#333
    style J fill:#fbb,stroke:#333
    style K fill:#fbb,stroke:#333
    style L fill:#fff,stroke:#333
    style M fill:#666,stroke:#333
    style N fill:#666,stroke:#333
```
</details>

Figure 1: Framework of the $\pi$ -GNN model. In the explainer pre-training phase, we use the synthetic graphs with ground-truth explanations to pretrain the $\pi$ -GNN explainer. In the fine-tuning phase towards downstream tasks in real-world graphs, the pre-trained explainer and the task-specific predictor are conjointly fine-tuned for transparent and accurate predictions.

Following previous works on generating synthetic graphs [10, 11, 13], each graph G in PT-Motifs dataset consists of one base subgraph $G_{b}$ and one explanation subgraph $G_{e}$ (also known as the motif) and the ground-truth task label y which is determined by $G_{e}$ solely [10, 13]. The shapes of the explanatory subgraphs in PT-Motifs include Diamond, House, Crane, Cycle, and Star and the basic shapes are Clique, Tree, Wheel, Ladder, and the Barabási–Albert Net [25]. Each combination of base and motif has the same number of samples in PT-Motifs. As shown in Figure 4(e) of Appendix B.2, the degree distribution of PT-Motifs follows the power law function $y = 10^{6} \times x^{-1.501}$ . See Appendix B.2 for more detailed structural patterns of PT-Motifs. The proposed $\pi$ -GNN model distills interpretability from the PT-Motifs dataset during the pre-training phase.

Structural Pattern Learning Module. To capture the multiple structural patterns, we propose to parallelize multi-thread of basic pattern-learner $B = \{B_{i} | i = 1, 2, \cdots, N\}$ . Then, an integrated pattern-learner aggregates them for a more comprehensive and general representation of various graph structural patterns. Specifically, each basic learner $B_{i}$ identifies a vectorized representation of the structural patterns (such as the degree distribution) and the integrated learner $\Phi$ provides a combination of each individual representation. We formally define the basic pattern-learner as follows.

Definition 1 (Basic Pattern-Learner). Consider a graph $G$ with $n$ nodes, whose adjacency matrix is $\mathbf{A} \in \mathbb{R}^{n \times n}$ . The basic pattern-learner $B_i$ projects the adjacency matrix $\mathbf{A}$ into a low-dimensional pattern matrix $\mathbf{Z}_i \in \mathbb{R}^{n \times d}, d < n$ , as follows,

$$
\mathbf {Z} _ {i} = B _ {i} (\mathbf {A}). \tag {4}
$$

The basic pattern-learner $B_{i}$ approaches the low-dimensional pattern matrix $Z_{i}$ by maximizing the likelihood of preserving the graph topological structure.

As a widely adopted technique which provides global views of a graph [41, 42], node embedding can serve as a simple yet effective basic pattern-learner. Inspired by the multi-head attention mechanism for improving the expressive power [43], we parallelize N-thread basic pattern-learners to achieve the structural pattern tensor $Z = [Z_{1}, Z_{2}, \cdots, Z_{N}] \in R^{N \times v \times d}$ which contains multiple universal structural patterns. For each basic pattern-learner, the adjacency matrix A is randomly permuted [42, 44]. Afterwards, the integrated learner $\Phi$ defined as follows aggregates the pattern tensor Z for a more expressive and generalizable pattern representation $Z_{Int}$ [45].

Definition 2 (Integrated Pattern-Learner). Given the structural pattern tensor $\mathcal{Z} \in \mathbb{R}^{N \times v \times d}$ , the integrated pattern-learner $\Phi$ moves forward to a convex combination $\mathbf{Z}_{\mathrm{Int}} \in \mathbb{R}^{v \times d_{\mathrm{Int}}}$ of each

![](images/4c3a29c60c21bc798d8d5cc930c8d44be04dfeabee54d2872880178606a1ef3b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Graph H"] -->|G = (V, ε)| B["Graph G"]
    B --> C["Graph H"]
    subgraph Graph_H
        D1["1"] --> E1["2"]
        E1 --> F1["3"]
        F1 --> G1["4"]
        G1 --> H1["5"]
        H1 --> I1["6"]
        I1 --> J1["7"]
        J1 --> K1["8"]
        K1 --> L1["9"]
        L1 --> M1["10"]
        M1 --> N1["11"]
        N1 --> O1["12"]
        O1 --> P1["13"]
        P1 --> Q1["14"]
        Q1 --> R1["15"]
        R1 --> S1["16"]
        S1 --> T1["17"]
        T1 --> U1["18"]
        U1 --> V1["19"]
        V1 --> W1["20"]
        W1 --> X1["21"]
        X1 --> Y1["22"]
        Y1 --> Z1["23"]
        Z1 --> AA1["24"]
        AA1 --> AB1["25"]
        AB1 --> AC1["26"]
        AC1 --> AD1["27"]
        AD1 --> AE1["28"]
        AE1 --> AF1["29"]
        AF1 --> AG1["30"]
        AG1 --> AH1["31"]
        AH1 --> AI1["32"]
        AI1 --> AJ1["33"]
        AJ1 --> AK1["34"]
        AK1 --> AL1["35"]
        AL1 --> AM1["36"]
        AM1 --> AN1["37"]
        AN1 --> AO1["38"]
        AO1 --> AP1["39"]
        AP1 --> AQ1["40"]
        AQ1 --> AR1["41"]
        AR1 --> AS1["42"]
        AS1 --> AT1["43"]
        AT1 --> AU["44"]
        AU --> AV["45"]
        AV --> AW["46"]
        AW --> AX["47"]
        AX --> AY["48"]
        AY --> AZ["49"]
        AZ --> BA["50"]
        BA --> BB["51"]
        BB --> BC["52"]
        BC --> BD["53"]
        BD --> BE["54"]
        BE --> BF["55"]
        BF --> BG["56"]
        BG --> BH["57"]
        BH --> BI["58"]
        BI --> BJ["59"]
        BJ --> BK["60"]
        BK --> BL["61"]
        BL --> BM["62"]
        BM --> BN["63"]
        BN --> BO["64"]
        BO --> BP["65"]
        BP --> BQ["66"]
        BQ --> BR["67"]
        BR --> BS["68"]
        BS --> BT["69"]
        BT --> BU["70"]
        BU --> BV["71"]
        BV --> BW["72"]
        BW --> BX["73"]
        BX --> BY["74"]
        BY --> BZ["75"]
        BZ --> CA["76"]
        CA --> CB["77"]
        CB --> CC["78"]
        CC --> CD["79"]
        CD --> CE["80"]
    end
    subgraph Graph_G fill:#f9f,stroke:#333
    end
    note right of A: V = {A,B,C,D}
    note right of B: V_h = {1,2,3,4,5}
    note right of BC: E_h = {A,B,C,D}
```
</details>

Figure 2: An example of the graph-hypergraph transformation. Considering the 3-degree node $A$ in graph $G$ , its corresponding hyper-view is the hyperedge $A$ which connects the hypernodes $\{1,4,5\}$ .

individual pattern matrix $Z_{i}$ as

$$
\mathbf {Z} _ {\text { Int }} = \Phi (\mathcal {Z}) = \sum_ {i = 1} ^ {N} \omega_ {i} \mathbf {Z} _ {i}, \tag {5}
$$

where $\Phi$ is an integration function to comprise appropriate structural patterns for diverse graphs.

Subsequently, $Z_{Int}$ is fed into the hypergraph refining module for computing the edge contribution score and exploring explanatory subgraph.

Hypergraph Refining Module. With the universal structural patterns extracted from the graphs, we next incorporate it with the edge interactions to identify the explanatory subgraph. Specifically, we first learn the edge structural representation $Z_{E}$ from the integrated pattern matrix $Z_{Int}$ . Taking the edge $e = (i, j)$ as an example, the corresponding structural representation $Z_{E}^{e}$ can be approached from the endpoint representations in $Z_{Int}$ as follows,

$$
\mathbf {Z} _ {E} ^ {e} = f ^ {(2)} \left(\mathbf {Z} _ {\text {Int}} ^ {i}, \mathbf {Z} _ {\text {Int}} ^ {j}\right), \tag {6}
$$

where $f^{(2)}$ is a 2-variable function and $Z_{Int}^{i}, Z_{Int}^{j}$ are the representations of nodes $v_{i}, v_{j}$ , respectively. Since the edge representation $Z_{E}$ directly determines the edge contribution score in the explanation, it should express as much information as possible. Theoretically, the expressive power of $Z_{E}$ is guaranteed by the following Theorem 1.

Theorem 1. Let $\Sigma_{n}$ be the set of all adjacency matrix $\mathbf{A}$ with $n$ nodes. Given a graph $G = (\mathcal{V},\mathcal{E})\in \Sigma_n,n\geq 2$ , let $\Gamma^{*}(S,\mathbf{A})$ be a most-expressive structural representation of nodes set $S\subseteq \mathcal{V}$ in $G$ . $\forall \mathbf{A}\in \overline{\Sigma}_n$ , there exists a most-expressive node representation $\mathbf{Z}^*\mid \mathbf{A}$ satisfies the relationship as follows,

$$
\Gamma^ {*} (S, \mathbf {A}) = \mathbb {E} _ {\mathbf {Z} ^ {*}} [ f ^ {(| S |)} ((\mathbf {Z} _ {v} ^ {*}) _ {v \in S}) | \mathbf {A} ], \forall S \subseteq \mathcal {V}, \tag {7}
$$

for an appropriate k-variable function $f^{(k)}(\cdot)$ .

Theorem 1 defines the upper bound of $\mathbf{Z}_E$ representation ability [44]. The upper bound can be approached according to the Theorem 2 as follows.

Theorem 2. The structural representation of edge $e = (v_i, v_j)$ can be learnt by simply approaching a function $f^{(2)}$ which satisfies $\Gamma(e, \mathbf{A}) = f^{(2)}(\mathbb{E}[(\mathbf{Z}_v)_{v \in \{i,j\}}|\mathbf{A}])$ .

In the hypergraph refining module, we adopt the integrated pattern matrix $Z_{Agg}$ as an approximation of the expectation embedding matrix $\mathbb{E}[(\mathbf{Z}_{v})_{v\in\{i,j\}}|\mathbf{A}]$ and a 2-layer MLP to fit the transition function $f^{(2)}$ during pre-training phase [44]. See Appendix G for the proofs of Theorems 1 and 2.

Based on the edge structural representation $Z_{E}$ , a straightforward way is to directly estimate the contribution score of each individual edge [13, 14]. However, such procedure ignores the incorporation of universal patterns with local structural interactions, leading to the missing of the dependencies among the edges. The edges in the explanation are supposed to interact with each other [46], form the coalition, and guide the downstream prediction task better than individuals. We next introduce how to capture the local edge interactions and incorporate them with the universal structural patterns.

Given the graph $G = (\mathcal{V}, \mathcal{E})$ , we define its corresponding hypergraph as $G_h = (\mathcal{V}_h, \mathcal{E}_h)$ . For a $k$ -degree node $v \in \mathcal{V}$ , the corresponding hyper-view is a hyperedge connecting $k$ hypernodes.

Similarly, the edge $e \in E$ becomes a hypernode in the corresponding hypergraph. An example of graph-hypergraph transformation is shown in Figure 2. See Appendix H for the detailed algorithm of graph-hypergraph transformation. During the graph-hypergraph transformation, the edge structural representation is naturally converted to the hypernode structural representation. Therefore, we can capture the local structural interaction by conducting hyperedge message passing. In our implementation, a 2-layer hypergraph convolutional network is employed to capture the edge interaction information and further map the edge structural representations to the edge contribution scores $\hat{\rho}$ . Finally, the hypergraph refining is formally represented as,

$$
\mathbf {Z} _ {E} ^ {(1)} = \operatorname{Tanh} (\text { HyperConv } (\mathbf {Z} _ {E})), \tag {8}
$$

$$
\hat {\rho} = \sigma (\text { HyperConv } (\mathbf {Z} _ {E} ^ {(1)})), \tag {9}
$$

where the $\operatorname{Tanh}(\cdot)$ function is used for a zero-mean activate value and the $\sigma(\cdot)$ function normalizes the output to a probability value within the range of [0, 1]. The normalized contribution score in $\hat{\rho}$ measures the probability of each edge belongs to the explanatory subgraph.

Under the supervision of the ground-truth explanation in PT-Motifs, $\pi$ -GNN distills the universal interpretability during the pre-training phase. Specifically, for each synthetic graph $G = (\mathcal{V}, \mathcal{E})$ , the edges in $\mathcal{E}_P$ that belong to the explanation subgraph are assigned with positive labels and the complementary part $\mathcal{E}_N$ receives a negative label [10, 11]. We denote the ground-truth explanation as $\rho \in \{0, 1\}^{|\mathcal{E}|}$ . $\pi$ -GNN takes $G$ as input and outputs the predicted explanation $\hat{\rho} \in [0, 1]^{|\mathcal{E}|}$ , which is optimized by the binary cross-entropy loss as follows,

$$
L (\hat {\rho}, \rho) = - \sum_ {i = 1} ^ {| \mathcal {E} |} [ \rho_ {i} \cdot \log \hat {\rho} _ {i} + (1 - \rho_ {i}) \cdot \log (1 - \hat {\rho} _ {i}) ]. \tag {10}
$$

# 3.2 Conjoint Fine-tuning Phase

During the conjoint fine-tuning phase, we combine the pre-trained $\pi$ -GNN explainer with task-specific predictors to identify the explanatory subgraph and make final prediction simultaneously for real-world datasets. Note that the pre-trained $\pi$ -GNN explainer is orthogonal to the post-positional predictor. That is, we can implement a predictor with arbitrary architecture as long as it can deal with the graph structure data. Even though the pre-training dataset PT-Motifs belongs to graph classification task, the pre-trained $\pi$ -GNN model can be easily generalized to other tasks, such as nodel classification, by simply implementing a node classifier.

Following existing works [11, 13, 14], given the input graph G and the corresponding task label y, we introduce a probabilistic sampler S to comprise the explanatory subgraph g according to the predicted edge probability $\hat{\rho}$ as follows,

$$
g = \mathcal {S} (G, \hat {\rho}). \tag {11}
$$

Going beyond the probabilistic sampling procedure, the post-positional predictor takes the explanatory subgraph g as input and fits the mapping function to the predicted label $\hat{y}$ , by optimizing a task-specific loss function $L_{\mathrm{task}}(\hat{y}, y)$ . In addition, we introduce an entropy regularizer [11, 14] in the fine-tuning phase to amplify the gap of each value in $\hat{\rho}$ for sparser explanations as follows,

$$
L (\hat {\rho}) = - \sum_ {i = 1} ^ {| \mathcal {E} |} [ \hat {\rho} _ {i} \cdot \log \hat {\rho} _ {i} + (1 - \hat {\rho} _ {i}) \cdot \log (1 - \hat {\rho} _ {i}) ] + | | \hat {\rho} | | _ {1}. \tag {12}
$$

The overall objective of the fine-tuning phase is to jointly optimize the task-specific term and the entropy regularizer as follows,

$$
L (\hat {y}, \hat {\rho}, y) = L _ {\text { task }} (\hat {y}, y) + L (\hat {\rho}). \tag {13}
$$

# 4 Experiment

In this section, we conduct extensive experiments to evaluate the performance of $\pi$ -GNN by answering the following two questions.

- RQ1: How effective is $\pi$ -GNN when it is generalized to different graph datasets?   
- RQ2: How effective is $\pi$ -GNN when it is generalized to different graph tasks?

Table 1: Interpretation Performance (ROC-AUC) Comparison. The underlined results highlight the best baselines. The bold results mean the $\pi$ -GNN or $\pi$ -GNN $_{DFT}$ outperform the best baselines. 

<table><tr><td>Model</td><td>BA-2Motifs</td><td>Mutag</td><td>MNIST-75sp</td><td>b=0.5</td><td>Spurious-Motif b=0.7</td><td>b=0.9</td></tr><tr><td>GNNExplainer</td><td>67.35 ± 3.29</td><td>61.98 ± 5.45</td><td>59.01 ± 2.04</td><td>62.62 ± 1.35</td><td>62.25 ± 3.61</td><td>58.86 ± 1.93</td></tr><tr><td>PGExplainer</td><td>84.59 ± 9.09</td><td>60.91 ± 17.10</td><td>69.34 ± 4.32</td><td>69.54 ± 5.64</td><td>72.33 ± 9.18</td><td>72.34 ± 2.91</td></tr><tr><td>GraphMask</td><td>92.54 ± 8.07</td><td>62.23 ± 9.01</td><td>73.10 ± 6.41</td><td>72.06 ±5.58</td><td>73.06 ± 4.91</td><td>66.68 ± 6.96</td></tr><tr><td>IB-Subgraph</td><td>86.06 ± 28.37</td><td>91.04 ± 6.59</td><td>51.20 ± 5.12</td><td>57.29 ± 14.35</td><td>62.89 ± 15.59</td><td>47.29 ± 13.39</td></tr><tr><td>DIR</td><td>82.78 ± 10.97</td><td>64.44 ± 28.81</td><td>32.35 ± 9.39</td><td>78.15 ± 1.32</td><td>77.68 ± 1.22</td><td>49.08 ± 3.66</td></tr><tr><td>GIN-GSAT</td><td>98.74 ± 0.55</td><td>99.60 ± 0.51</td><td>83.36 ± 1.02</td><td>78.45 ± 3.12</td><td>74.07 ± 5.28</td><td>71.97 ± 4.41</td></tr><tr><td>PNA-GSAT</td><td>93.77 ± 3.90</td><td>99.07 ± 0.50</td><td>84.68 ± 1.06</td><td>83.34 ± 2.17</td><td>86.94 ± 4.05</td><td>88.66 ± 2.44</td></tr><tr><td>π-GNN</td><td>99.33 ± 0.63</td><td>99.81 ± 0.17</td><td>92.77 ± 0.80</td><td>93.24 ± 0.72</td><td>96.92 ± 0.85</td><td>96.39 ± 0.92</td></tr><tr><td>π-GNNDFT</td><td>93.19 ± 1.48</td><td>95.29 ± 0.67</td><td>85.18 ± 1.08</td><td>86.29 ± 2.22</td><td>87.43 ± 2.47</td><td>89.64 ± 2.26</td></tr></table>

# 4.1 Experimental Settings

In the experiment, we use two popular synthetic datasets $[10, 11, 13, 14]$ and four real-world datasets of graph classification tasks. The details of dataset characteristics and statistics are summarized in Appendix B. A brief introduction to the datasets is as follows.

- Synthetic Datasets. BA-2Motifs [10] and Spurious-Motif [13] are two widely-used synthetic datasets to evaluate the interpretation performance of the GNN explanation methods.   
- Real-world Datasets. We use four real-world datasets, the superpixel graph dataset MNIST-75sp [40], the sentiment analysis dataset Graph-SST2 [16], and two chemical molecule datasets Mutag [18] and Ogbg-Molhiv [47]. Note that, in the MNIST-75sp dataset, the subgraph with nonzero pixel values is regarded as the ground-truth explanation [14]; in the Mutag dataset, $-\mathrm{NO}_2$ and $-\mathrm{NH}_2$ functional groups in mutagen graphs are labelled as the ground-truth explanation [28]. Hence, we use the MNIST-75sp and Mutag datasets for both the interpretation and the prediction evaluations.

We extensively compare $\pi$ -GNN with the following two types of baselines:

- Interpretation Baselines. We compare the interpretation performance with both the post-hoc explanation methods including GNNExplainer [10], PGExplainer [11], and GraphMask [39] and the intrinsic interpretable methods including DIR [13], IB-subgraph [12], GIN-GSAT, and PNA-GSAT [14]. Following the standard setting, the evaluation metric is the explanation ROC-AUC [13, 14].   
- Prediction Baselines. We compare the prediction performance with the powerful GNN models including GIN [4] and PNA [32] and the intrinsic interpretable methods including DIR, IB-subgraph, GIN-GSAT, and PNA-GSAT. For the OGBG-Molhiv dataset, we use the classification ROC-AUC as the prediction metric [47]. For all the other dataset, we report the classification accuracy [13].

Additionally, we report the performance of $\pi$ -GNN which directly fine-tunes on downstream datasets without pre-training (denoted as the $\pi$ -GNN $_{DFT}$ ), to investigate the effectiveness of pre-training phase. All the results are averaged over 10-times evaluation with different random seeds. Architecture of the downstream GNN predictors used in $\pi$ -GNN and $\pi$ -GNN $_{DFT}$ are reported in Appendix C.

# 4.2 Main Results (RQ1)

To investigate the effectiveness of $\pi$ -GNN, we compare the interpretation and the prediction performance with the SOTA interpretation and prediction baselines. See Appendix C for the pre-training and fine-tuning details. The overall interpretation performance and prediction performance are summarized in Table 1 and Table 2, respectively. We conclude the following observations:

\- $\pi$ -GNN significantly outperforms the leading GNN explanation methods. Specifically, for the Spurious-Motif dataset, $\pi$ -GNN surpasses DIR by $27.21\%$ on average and by $47.31\%$ at most. Compared with the best baselines, i.e., the GSAT with a 4-layer PNA encoder, $\pi$ -GNN improves the interpretation performance by $9.20\%$ on average. However, we merely employ the combination of a truncated SVD embedding and a 2-layer MLP as the encoder, which convincingly demonstrates the effectiveness of our proposed pre-training phase. Moreover, as the degree of spurious correlation in

Table 2: Prediction Performance (Acc) Comparison. The underlined results highlight the best baselines. The bold results mean the $\pi$ -GNN or $\pi$ -GNN $_{DFT}$ outperform the best baselines. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Molhiv(AUC)</td><td rowspan="2">Graph-SST2</td><td rowspan="2">MNIST-75sp</td><td colspan="3">Spurious-Motif</td></tr><tr><td>b=0.5</td><td>b=0.7</td><td>b=0.9</td></tr><tr><td>GIN</td><td>76.69 ± 1.25</td><td>82.73 ± 0.77</td><td>95.74 ± 0.36</td><td>39.87 ± 1.30</td><td>39.04 ± 1.62</td><td>38.57 ± 2.31</td></tr><tr><td>PNA</td><td>78.91 ± 1.04</td><td>79.87 ± 1.02</td><td>87.20 ± 5.61</td><td>68.15 ± 2.39</td><td>66.35 ± 3.34</td><td>61.40 ± 3.56</td></tr><tr><td>IB-Subgraph</td><td>76.43 ± 2.65</td><td>82.99 ± 0.67</td><td>93.10 ± 1.32</td><td>54.36 ± 7.09</td><td>48.51 ± 5.76</td><td>46.19 ± 5.63</td></tr><tr><td>DIR</td><td>76.34 ± 1.01</td><td>82.32 ± 0.85</td><td>88.51 ± 2.57</td><td>45.49 ± 3.81</td><td>41.13 ± 2.62</td><td>37.61 ± 2.02</td></tr><tr><td>GIN-GSAT</td><td>76.47 ± 1.53</td><td>82.95 ± 0.58</td><td>96.24 ± 0.17</td><td>52.74 ± 4.08</td><td>49.12 ± 3.29</td><td>44.22 ± 5.57</td></tr><tr><td>PNA-GSAT</td><td>80.24 ± 0.73</td><td>80.92 ± 0.66</td><td>93.96 ± 0.92</td><td>68.74 ± 2.24</td><td>64.38 ± 3.20</td><td>57.01 ± 2.95</td></tr><tr><td>π-GNN</td><td>80.86 ± 0.61</td><td>88.05 ± 0.43</td><td>96.89 ± 0.20</td><td>74.67 ± 0.63</td><td>77.52 ± 0.77</td><td>77.46 ± 0.96</td></tr><tr><td>π-GNN $_{DFT}$ </td><td>79.71 ± 1.08</td><td>83.48 ± 1.20</td><td>92.89 ± 0.95</td><td>70.78 ± 1.63</td><td>71.02 ± 1.43</td><td>72.61 ± 1.75</td></tr></table>

Spurious-Motif (i.e., the parameter $b$ ) increasing, the performance of $\pi$ -GNN does not decrease as most of the baselines. The post-hoc explainers, including GNNExplainer, PGExplainer, and GraphMask seems to be robust in terms of the spurious correlation, but their interpretability is limited by the fixed prediction model. Although DIR and GSAT introduce complex mechanism for mitigating the spurious correlation, $\pi$ -GNN still performs better. We ascribe this superiority to the generalizable knowledge which is distilled from the pre-training phase over large dataset with ground-truth explanations. For the BA-2Motifs and the Mutag datasets, $\pi$ -GNN using a more simpler encoder architecture also achieves comparable performance. For the MNIST-75sp dataset, $\pi$ -GNN surpasses the best baseline PNA-GSAT by 4.50%. Such top-tier performance strongly validates the effectiveness of $\pi$ -GNN explainer.

\- $\pi$ -GNN also achieves better prediction performance than the baselines. Overall, $\pi$ -GNN outperforms all the prediction baselines consistently by a significant margin. Specifically, for the Spurious-Motif dataset, $\pi$ -GNN outperforms the best baselines by $11.05\%$ on average, which is a significant improvement. If we restrict the baselines into interpretable GNNs, the performance boost is much more significant (by $13.17\%$ on average and up to $20.45\%$ ). For the Graph-SST2 dataset, $\pi$ -GNN outperforms the black-box predictor (i.e., GIN and PNA), as well as the SOTA interpretable methods IB-Subgraph, which optimizes the explanatory subgraph based on the information bottleneck principle. Note that, we only implement a 2-layer GCN with global mean pooling function as the Graph-SST2 predictor. We credit such outperformance to the pre-training phase, from which the universal patterns of the graphs are potentially distilled. For the MNIST-75sp dataset, $\pi$ -GNN achieves comparable prediction accuracy with the GSAT methods, but our interpretation ROC-AUC exceeds GSAT by $8.09\%$ . Additionally, one can notice that the black-box predictors are insensitive to the spurious correlation while the interpretable baselines deteriorate obviously. This phenomena may indicate that insufficient interpretability conflicts with the chase of prediction accuracy, but a powerful interpreter is quite helpful to the subsequent predictor.

\- The pre-training phase advances the interpreter in terms of both interpretation and perdiction performance. As shown in Table 1 and Table 2, the $\pi$ -GNN with pre-training phase significantly improves both the interpretation and the prediction performance, compared with the reduced variant $\pi$ - $\mathrm{GNN}_{\mathrm{DFT}}$ . For the interpretation ROC-AUC, $\pi$ -GNN outperforms the reduced variant by $6.91\%$ on average. This suggests that the universal structural patterns distilled in the pre-training phase can indeed generalize to various downstream tasks and improve the interpretability. Compared with the reduced variant, $\pi$ -GNN consistently provides much stabler interpretation with smaller variance. Additionally, one can observe that the reduced variant $\pi$ - $\mathrm{GNN}_{\mathrm{DFT}}$ still surpasses the best baselines on some real-world datasets, such as the interpretation performance on MNIST-75sp dataset and the prediction performance on Graph-SST2 dataset. This may imply that the edge interaction captured by $\pi$ - $\mathrm{GNN}_{\mathrm{DFT}}$ is able to identify the influential subgraphs more accurately.

Moreover, we present the explanatory visualization, the investigate of different pre-training datasets and the hyper-parameter analysis in Appendix D, E and F, respectively.

# 4.3 Inter-Task Generalization Performance (RQ2)

To further investigate whether the universal structural patterns behind different tasks is common, we next study the generalization ability across different tasks. Specifically, we evaluate the $\pi$ -GNN model that is pre-trained over graph classification dataset PT-Motifs on the explanation task of node

Table 3: Inter-Task Interpretation Performance (ROC-AUC) Comparison. The bold font highlights the best results and the underlined results highlight the second best method. 

<table><tr><td>Model</td><td>BA-Shapes</td><td>BA-Community</td><td>Tree-Cycles</td><td>Tree-Grid</td></tr><tr><td>GRAD</td><td>88.20</td><td>75.00</td><td>90.50</td><td>61.20</td></tr><tr><td>Attention</td><td>81.50</td><td>73.90</td><td>82.40</td><td>66.70</td></tr><tr><td>GNNExplainer</td><td>92.50</td><td>83.60</td><td>94.80</td><td>87.50</td></tr><tr><td>PGExplainer</td><td> $\underline{96.30} \pm 1.10$ </td><td> $\underline{94.50} \pm 1.90$ </td><td> $\underline{98.70} \pm 0.70$ </td><td> $\underline{90.70} \pm 1.40$ </td></tr><tr><td> $\pi$ -GNN</td><td> $\underline{94.78} \pm 0.32$ </td><td> $\underline{94.67} \pm 1.50$ </td><td> $\underline{95.19} \pm 0.88$ </td><td> $\underline{90.11} \pm 1.10$ </td></tr><tr><td> $\pi$ - $GNN_{DFT}$ </td><td> $\underline{93.17} \pm 0.40$ </td><td> $\underline{92.15} \pm 1.61$ </td><td> $\underline{92.53} \pm 1.81$ </td><td> $\underline{88.62} \pm 1.87$ </td></tr></table>

classification datasets. The pre-trained $\pi$ -GNN model and the reduced variant are both equipped with a node classifier. Following existing works, we use four widely-used synthetic node classification datasets [10], namely BA-Shapes, BA-Community, Tree-Cycles and Tree-Grid, whose detailed statistics are presented in Appendix B. The main result of inter-task explanation in Table 3 shows that $\pi$ -GNN achieves comparable performance with SOTA node classification explainers. This evidence is accordant with our basic premise that the universal structural patterns can generalize across datasets of different tasks. For the BA-Community dataset, $\pi$ -GNN even supasses PGExplainer with smaller variance. This may be because the community structure in the BA-Community graph conforms more to the universal patterns embedded in $\pi$ -GNN.

# 4.4 Ablation Study

As shown in Figure 3, we conduct ablation study on $\pi$ -GNN, by evaluate the performance of its three variants. First, $\pi$ -GNN-SPL substitutes the structural pattern learning module with N-thread GNN encoders. Second, $\pi$ -GNN-HPR removes the hypergraph refining module and directly calculates the edge contribution score. Third, we simultaneously conduct the two ablations above and denote it as $\pi$ -GNN-ALL. Additionally, we report the performance of the variant without pre-training phase.

Specifically, when removing the structural pattern learning module, the interpretation performance on Mutag decreases by 3.85% on average and the prediction performance on Graph-SST2 decreases by 1.08% on average. For the variant $\pi$ -GNN-HPR, the interpretation and prediction performance decreases by 6.56% and 1.84%, respectively. When we remove the two modules simultaneously, the interpretation and prediction performance decreases by 7.56% and 2.16%, respectively. The ablation study on the two modules demonstrates their effectiveness in capturing universal structural patterns and identifying the explanatory subgraphs. Moreover, for all the variants, $\pi$ -GNN consistently outperforms the variant without pre-training phase by 5.59% in terms of ROC-AUC on Mutag and 2.84% in terms of accuracy on Graph-SST2.

# 5 Related Work

Intrinsic interpretable GNNs. The leading interpretable GNNs $[12, 13, 14]$ usually consist of an explainer module and a predictor module. The prepositional explainer takes the raw graph as input and outputs the explanation subgraph. The subsequent predictor calculates the prediction strictly relying on the explanation. Graph neural networks with attention mechanism are regarded as the initial interpretable GNNs $[13, 14]$ , such as graph attention network $[3]$ , self-attention graph pooling $[40]$ , where the learned weights can be interpreted as the importance of certain features. Recently, invariant learning is introduced to construct intrinsic interpretable GNNs $[48, 49]$ , such as DIR $[13]$ . It argues that augmenting training data with causal intervention may assist explainer to distinguish the causal and non-causal parts. Besides, interpretable GNNs based on the information bottleneck principle $[50]$ , such as IB-Subgraph $[12]$ and GSAT $[14]$ , are proposed to constraint the information flow from the input graph to the prediction, where the label-relevant graph components will be kept while the label-irrelevant ones are reduced.

Pre-training on Graphs. The research focus of current graph pre-training is the graph representation learning problem, whose objective is to learn a generic encoder $f(\mathbf{A}, \mathbf{X})$ for various downstream tasks [51, 52]. The development of graph pre-training can be broadly divided into pre-trained graph embeddings [41, 53, 42] and pre-trained graph encoders [23, 22, 24, 21]. Pre-trained graph embedding models aim to provide good graph embeddings for various tasks, while the models themselves are no

![](images/6bad4b9fd006eb855b25072f91014b3a864f8089932c75f695f1ca4b7a336698.jpg)

<details>
<summary>bar</summary>

| Pre-training Status | π-GNN | π-GNN-SPL | π-GNN-HPR | π-GNN-ALL |
| ------------------- | ----- | --------- | --------- | --------- |
| With Pre-training    | 99.8  | 96.5      | 94.2      | 93.0      |
| Without Pre-training| 95.3  | 91.0      | 87.8      | 87.2      |
</details>

(a) Interpretation performance on Mutag

![](images/ca3f8647fd20b4b8a68cb03cb23ed19ef7c531220cb20e58c78d720778ff426a.jpg)

<details>
<summary>bar</summary>

| Model | With Pre-training | Without Pre-training |
| :--- | :--- | :--- |
| π-GNN | 93.2 | 86.3 |
| π-GNN-SPL | 84.1 | 83.1 |
| π-GNN-HPR | 87.0 | 84.5 |
| π-GNN-ALL | 78.7 | 74.2 |
</details>

(b) Interpretation performance on Spurious-Motif

![](images/615405cc9e016df1822a1f62a8c706a0fa74fd1e9ead1aaa96eba1c06a20dcef.jpg)

<details>
<summary>bar</summary>

| Pre-training | π-GNN | π-GNN-SPL | π-GNN-HPR | π-GNN-ALL |
| ------------ | ----- | --------- | --------- | --------- |
| With Pre-training | 88.0 | 86.7 | 84.9 | 84.1 |
| Without Pre-training | 83.5 | 82.7 | 83.0 | 83.2 |
</details>

(c) Prediction performance on Graph-SST2

![](images/023545942fa76d34929aaeb801b23662944f4cb1832be09849812c59fabeabe6.jpg)

<details>
<summary>bar</summary>

| Pre-training | π-GNN | π-GNN-SPL | π-GNN-HPR | π-GNN-ALL |
| ------------ | ----- | --------- | --------- | --------- |
| With Pre-training | 74.5 | 72.0 | 71.8 | 68.8 |
| Without Pre-training | 70.8 | 69.7 | 70.0 | 66.4 |
</details>

(d) Prediction performance on Spurious-Motif   
Figure 3: The ablation study of $\pi$ -GNN on Mutag, Graph-SST2, and Spurious-Motif datasets.

longer needed to the tasks. DeepWalk [41] explores the graph embeddings by conducting random walks over graphs to generate node sequences which contain the co-occurrence relationship. Further, Node2vec [42] defines a flexible node neighborhood and proposes a biasd random walk process. The goal of pre-trained graph encoder models is a generic encoder model which can deal with different tasks. Gpt-GNN [23] pre-trains a 5-layer GIN encoder across the graph-level and node-level tasks. GCC [22] introduces the contrastive learning framework to pre-train the encoder over subgraph discrimination task. However, the precursor graph pre-training works are not designed to the graph explanation problem and can not be directly applied to the pre-training interpretable GNNs.

# 6 Conclusion

In this work, we for the first time investigated the universal interpretation problem in graph data and proposed the Pre-training Interpretable Graph Neural Network named $\pi$ -GNN. $\pi$ -GNN is able to work well on different types of graphs and downstream tasks. $\pi$ -GNN was pre-trained over a constructed large synthetic graph dataset with ground-truth explanations to distill the generalizable interpretability. Then, $\pi$ -GNN was fine-tuned on downstream tasks. Technically, we proposed an intergrated embedding module to capture and integrate multiple graph structural patterns for more generalizable representations. A hypergraph refining module was aslo proposed to incorporate the universal patterns with local interaction for more faithful explanatory subgraphs identification. Extensive experiments on different datasets and tasks demonstrated the promising generalizable interpretability as well as prediction performance of $\pi$ -GNN.

# Acknowledgement

This research was funded by the National Science Foundation of China (No. 62172443), Open Project of Xiangjiang Laboratory (22XJ03010, 22XJ03005), the Science and Technology Major Project of Changsha (No. kh2202004), Hunan Provincial Natural Science Foundation of China (No. 2022JJ30053), and the High Performance Computing Center of Central South University.

# References

[1] Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of ICLR, 2017.   
[2] William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive Representation Learning on Large Graphs. In Proceedings of NIPS, 2017.   
[3] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph Attention Networks. In Proceedings of ICLR, 2018.   
[4] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural Networks? In Proceedings of ICLR, 2019.   
[5] Hao Chen, Yue Xu, Feiran Huang, Zengde Deng, Wenbing Huang, Senzhang Wang, Peng He, and Zhoujun Li. Label-aware graph convolutional networks. In Proceedings of KDD, 2020.   
[6] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard S. Zemel. Gated Graph Sequence Neural Networks. In Proceedings of ICLR, 2016.   
[7] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec. Hierarchical Graph Representation Learning with Differentiable Pooling. In Proceedings of NeurIPS, 2018.   
[8] Yue Xu, Hao Chen, Zefan Wang, Jianwen Yin, Qijie Shen, Dimin Wang, Feiran Huang, Lixiang Lai, Tao Zhuang, Junfeng Ge, and Xia Hu. Multi-factor sequential re-ranking with perception-aware diversification. In Proceedings of KDD, 2023.   
[9] Senzhang Wang, Hao Yan, Jinlong Du, Jun Yin, Junxing Zhu, Chaozhuo Li, and Jianxin Wang. Adversarial hard negative generation for complementary graph contrastive learning. In Proceedings of SDM, 2023.   
[10] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. GNNExplainer: Generating Explanations for Graph Neural Networks. In Proceedings of NeurIPS, 2019.   
[11] Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang. Parameterized Explainer for Graph Neural Network. In Proceedings of NeurIPS, 2020.   
[12] Junchi Yu, Tingyang Xu, Yu Rong, Yatao Bian, Junzhou Huang, and Ran He. Graph Information Bottleneck for Subgraph Recognition. In Proceedings of ICLR, 2021.   
[13] Yingxin Wu, Xiang Wang, An Zhang, Xiangnan He, and Tat-Seng Chua. Discovering Invariant Rationales for Graph Neural Networks. In Proceedings of ICLR, 2022.   
[14] Siqi Miao, Mia Liu, and Pan Li. Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism. In Proceedings of ICML, 2022.   
[15] Nils Chr Stenseth, Wilhelm Falck, Ottar N Bjørnstad, , and Charles J Krebs. Population regulation in snowshoe hare and canadian lynx: asymmetric food web configurations between hare and lynx. In Proceedings of NAS, 1997.   
[16] Yuan Luo Liang Yao, Chengsheng Mao. Graph Convolutional Networks for Text Classification. In Proceedings of AAAI, 2019.

[17] Zhoujin Tian, Chaozhuo Li, Zhiqiang Zuo, Zengxuan Wen, Lichao Sun, Xinyue Hu, Wen Zhang, Haizhen Huang, Senzhang Wang, Weiwei Deng, et al. Pass: Personalized advertiser-aware sponsored search. In Proceedings of KDD, 2023.   
[18] Jeroen Kazius, Ross McGuire, and Roberta Bursi. Derivation and validation of toxicophores for mutagenicity prediction. Journal of Medicinal Chemistry, 48, 2005.   
[19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT, 2019.   
[20] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of CVPR, 2020.   
[21] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph Contrastive Learning with Augmentations. In Proceedings of NeurIPS, 2020.   
[22] Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, and Jie Tang. GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training. In Proceedings of KDD, 2020.   
[23] Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. GPT-GNN: Generative Pre-Training of Graph Neural Networks. In Proceedings of KDD, 2020.   
[24] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-Supervised Graph Transformer on Large-Scale Molecular Data. In Proceedings of NeurIPS, 2020.   
[25] Réka Albert and Albert-László Barabási. Statistical mechanics of complex networks. Reviews of modern physics, 74, 2002.   
[26] Ron Milo, Shalev Itzkovitz, Nadav Kashtan, Reuven Levitt, Shai Shen-Orr, Inbal Ayzenshtat, Michal Sheffer, and Uri Alon. Superfamilies of Evolved and Designed Networks. Science, 303(5663), 2004.   
[27] Stephen P Borgatti and Martin G Everett. Models of core/periphery structures. Social networks, 21, 2000.   
[28] Juntao Tan, Shijie Geng, Zuohui Fu, Yingqiang Ge, Shuyuan Xu, Yunqi Li, and Yongfeng Zhang. Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning. In Proceedings of WebConf, 2022.   
[29] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee. Motif-based Graph Self-Supervised Learning for Molecular Property Prediction. In Proceedings of NeurIPS, 2021.   
[30] Xiang Wang, Ying-Xin Wu, An Zhang, Xiangnan He, and Tat-Seng Chua. Towards Multi-Grained Explainability for Graph Neural Networks. In Proceedings of NeurIPS, 2021.   
[31] Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in Graph Neural Networks: A Taxonomic Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45, 2023.   
[32] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Velickovic. Principal Neighbourhood Aggregation for Graph Nets. In Proceedings of NeurIPS, 2020.   
[33] Peiyan Zhang, Yuchen Yan, Chaozhuo Li, Senzhang Wang, Xing Xie, Guojie Song, and Sunghun Kim. Continual learning on dynamic graphs via parameter isolation. In Proceedings of SIGIR, 2023.   
[34] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. Learning on large-scale text-attributed graphs via variational inference. In Proceedings of ICLR, 2022.   
[35] Muhan Zhang and Yixin Chen. Link Prediction Based on Graph Neural Networks. In Proceedings of NeurIPS, 2018.

[36] Zhongyu Huang, Yingheng Wang, Chaozhuo Li, and Huiguang He. Going deeper into permutation-sensitive graph neural networks. In Proceedings of ICML, pages 9377–9409, 2022.   
[37] Xiaoqi Wang and Han-Wei Shen. GNNInterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks. In Proceedings of ICLR, 2023.   
[38] Yifei Liu, Chao Chen, Yazheng Liu, Xi Zhang, and Sihong Xie. Multi-objective Explanations of GNN Predictions. In Proceedings of ICDM, 2021.   
[39] Michael Sejr Schlichtkrull, Nicola De Cao, and Ivan Titov. Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking. In Proceedings of ICLR, 2021.   
[40] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-Attention Graph Pooling. In Proceedings of ICML, 2019.   
[41] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. DeepWalk: online learning of social representations. In Proceedings of KDD, 2014.   
[42] Aditya Grover and Jure Leskovec. node2vec: Scalable Feature Learning for Networks. In Proceedings of KDD, 2016.   
[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In Proceedings of NIPS, 2017.   
[44] Balasubramaniam Srinivasan and Bruno Ribeiro. On the Equivalence between Positional Node Embeddings and Structural Graph Representations. In Proceedings of ICLR, 2020.   
[45] Thomas G. Dietterich. Ensemble Methods in Machine Learning. In Proceedings of MCS, 2000.   
[46] Christopher Frye, Damien de Mijolla, Tom Begley, Laurence Cowton, Megan Stanley, and Ilya Feige. Shapley explainability on the data manifold. In Proceedings of ICLR, 2021.   
[47] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open Graph Benchmark: Datasets for Machine Learning on Graphs. In Proceedings of NeurIPS, 2020.   
[48] Shiyu Chang, Yang Zhang, Mo Yu, , and Tommi S. Jaakkola. Invariant Rationalization. In Proceedings of ICML, 2020.   
[49] David Krueger, Ethan Caballero, Jörn-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Rémi Le Priol, and Aaron C. Courville. Out-of-Distribution Generalization via Risk Extrapolation (REx). In Proceedings of ICML, 2021.   
[50] Tailin Wu, Hongyu Ren, Pan Li, and Jure Leskovec. Graph Information Bottleneck. In Proceedings of NeurIPS, 2020.   
[51] Jun Xia, Yanqiao Zhu, Yuanqi Du, and Stan Z. Li. A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications. In Proceedings of IJCAI, 2022.   
[52] Hao Yan, Senzhang Wang, Jun Yin, Chaozhuo Li, Junxing Zhu, and Jianxin Wang. Going deeper into permutation-sensitive graph neural networks. In Proceedings of ECML/PKDD, 2023.   
[53] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. LINE: Large-scale Information Network Embedding. In Proceedings of WWW, 2015.

# A Notations

In Table 4, we give the mathematical symbols and their descriptions used in this paper for clarity.

Table 4: Symbols and their descriptions. 

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $G, \mathcal{V}, \mathcal{E}$ </td><td>Graph instance, node set, edge set</td></tr><tr><td> $\mathbf{X}$ </td><td>Node feature matrix</td></tr><tr><td> $\mathbf{X}_{E}$ </td><td>Edge feature matrix</td></tr><tr><td> $\mathbf{A}, A_{ij}$ </td><td>Adjacency matrix, element at the  $i$ -th row,  $j$ -th column of  $\mathbf{A}$ </td></tr><tr><td> $v, v_{i}; e, e_{i}$ </td><td>Node instance; edge instance</td></tr><tr><td> $\mathcal{N}(v)$ </td><td>Neighborhood of node  $v$ </td></tr><tr><td> $\mathbf{W}$ </td><td>Trainable parameter matrix</td></tr><tr><td> $F(\cdot), \sigma(\cdot)$ </td><td>Non-linear activation function, Sigmoid function</td></tr><tr><td> $h$ </td><td>Edge contribution function</td></tr><tr><td> $\rho, \hat{\rho}$ </td><td>Ground-truth explanation, predicted explanation</td></tr><tr><td> $\mathcal{S}$ </td><td>Selection module</td></tr><tr><td> $g$ </td><td>Explanatory subgraph</td></tr><tr><td> $\mathcal{B}, B_{i}$ </td><td>Set of basic pattern-learners, basic pattern-learner</td></tr><tr><td> $N$ </td><td>Number of basic pattern-learners</td></tr><tr><td> $\mathbf{Z}_{i}, \mathcal{Z}$ </td><td>Pattern matrix, pattern tensor</td></tr><tr><td> $\Phi$ </td><td>Integrated pattern-learner</td></tr><tr><td> $\mathbf{Z}_{\text{Int}}$ </td><td>Integrated pattern representation</td></tr><tr><td> $\mathbf{Z}_{E}$ </td><td>Edge structural representation</td></tr><tr><td> $S$ </td><td>Subset of node</td></tr><tr><td> $f^{(2)}$ </td><td>Node to edge transition function</td></tr><tr><td> $G_{h}, \mathcal{V}_{h}, \mathcal{E}_{h}$ </td><td>Hypergraph instance, hypernode set, hyperedge set</td></tr><tr><td> $\mathcal{E}_{P}, \mathcal{E}_{N}$ </td><td>Positive (Negative) edge set</td></tr><tr><td> $b$ </td><td>Degree of spurious correlation in Spurious-Motif dataset</td></tr></table>

# B Datasets

We introduce the datasets used in the main experiment as follows.

- BA-2Motifs [10] is a synthetic dataset with binary graph classes for evaluating the interpretation performance. House motifs and cycle motifs decide the graph labels and serve as the ground-truth explanations of the two classes, respectively.   
- Spurious-Motif [13] is a synthetic dataset with three graph classes for evaluating both the interpretation and the prediction performance. Each class corresponds to a particular motif which is regarded as the ground-truth explanation. During the synthesis process, spurious correlations, including the unbalanced base sampling and the scale drift, are injected into the training dataset. A hyper-parameter $b$ controls the degree of spurious correlation, usually set as $\{0.5, 0.7, 0.9\}$ .   
- Mutag [18] is a chemical molecule dataset with binary graph classes which represents mutagenic property. Benefiting from domain knowledge, $-\mathrm{NO}_2$ and $-\mathrm{NH}_2$ functional groups in mutagen graphs are labelled as the ground-truth explanation. Therefore, we employ the Mutag dataset for the evaluation of interpretation and prediction performance.   
- MNIST-75sp [40] converts the MNIST image dataset into a superpixel graph dataset with ten graph classes. The nodes are superpixels, while edges are the spatial distance between the related endpoints. The subgraph with nonzero pixel values is regarded as the ground-truth explanation. Hence, the Mnist-75sp is used for both the interpretation and the prediction evaluations.   
- Graph-SST2 [16] is a sentiment analysis dataset with binary graph labels, where each sequence in SST2 dataset is transformed to a graph. Since no ground-truth explanations are available, we evaluate the prediction performance only.

\- OGBG-Molhiv [47] is a molecular dataset with binary graph labels according to the inhibition effect on HIV virus replication, where nodes are atoms, and edges are chemical bonds. As there are no ground-truth explanations, we merely evaluate the prediction performance.

# B.1 Statistical Characteristics

We show the detailed statistics of the datasets in Table 5.

Table 5: Detailed statistics of graph classification datasets in main experiment. 

<table><tr><td rowspan="2"></td><td colspan="3">BA-2Motifs</td><td colspan="3">Spurious-Motif</td><td colspan="3">Mutag</td></tr><tr><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>#Classes</td><td></td><td>2</td><td></td><td></td><td>3</td><td></td><td></td><td>2</td><td></td></tr><tr><td>#Graphs</td><td>400</td><td>400</td><td>200</td><td>9,000</td><td>3,000</td><td>6,000</td><td>1000</td><td>1000</td><td>951</td></tr><tr><td>Avg. #Nodes</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.4</td><td>26.1</td><td>88.7</td><td>30.1</td><td>30.1</td><td>30.2</td></tr><tr><td>Avg. #Edges</td><td>50.9</td><td>51.0</td><td>50.9</td><td>35.4</td><td>36.2</td><td>131.1</td><td>61.3</td><td>60.2</td><td>61.2</td></tr><tr><td rowspan="2"></td><td colspan="3">MNIST-75sp</td><td colspan="3">Graph-SST2</td><td colspan="3">OGBG-Molhiv</td></tr><tr><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>#Classes</td><td></td><td>10</td><td></td><td></td><td>2</td><td></td><td></td><td>2</td><td></td></tr><tr><td>#Graphs</td><td>20,000</td><td>5,000</td><td>10,000</td><td>28,237</td><td>3,147</td><td>12,305</td><td>32,901</td><td>4,113</td><td>4,113</td></tr><tr><td>Avg. #Nodes</td><td>66.8</td><td>67.3</td><td>67.0</td><td>17.7</td><td>17.3</td><td>3.45</td><td>25.3</td><td>27.8</td><td>25.3</td></tr><tr><td>Avg. #Edges</td><td>539.3</td><td>545.9</td><td>540.9</td><td>33.3</td><td>33.5</td><td>4.89</td><td>54.1</td><td>61.1</td><td>55.6</td></tr></table>

We show the detailed statistics of datasets that used in the inter-task evaluation in Table 6.

Table 6: Detailed statistics of node classification datasets in inter-task experiment. 

<table><tr><td></td><td>BA-Shapes</td><td>BA-Community</td><td>Tree-Cycles</td><td>Tree-Grid</td></tr><tr><td>#Classes</td><td>4</td><td>8</td><td>2</td><td>2</td></tr><tr><td>#Nodes</td><td>700</td><td>1,400</td><td>871</td><td>1,231</td></tr><tr><td>#Edges</td><td>4,110</td><td>8,920</td><td>1,950</td><td>3,410</td></tr></table>

# B.2 Structural Patterns

Degree Distribution. As shown in Figure 4, we visualize the degree distribution of the four real-world datasets, i.e., Mutag, MNNIST-75sp, Ogbg-Molhiv, and Graph-SST2, and our synthetic scale-free dataset PT-Motifs. The power-law like degree function $D(x) = cx^{-\alpha}$ of each real-world dataset is formally represented as follows, optimized by the ordinary least square error. Here $D(x)$ represents the number of x-degree nodes and $R^{2}$ is the coefficient of determination.

- Mutag: $D(x) = 54447x^{-1.537}, R^2 = 0.9015$ .   
- MNIST-75sp: $D(x) = 10^{7}x^{-5.913}, R^{2} = 0.6713$ .   
- Ogbg-Molhiv: $D(x) = 3 \times 10^{6} x^{-6.039}, R^{2} = 0.6828$ .   
- Graph-SST2: $D(x) = 8 \times 10^{6} x^{-4.988}, R^{2} = 0.8491$ .   
- PT-Motifs: $D(x) = 10^{6}x^{-1.501}, R^{2} = 0.9677$ .

In Table 7, we report the average value of the four structural patterns over each individual graphs in the datasets, including

\- Transitivity is defined as the fraction of all possible triangles present in the given graph $G$ . It measures the tendency of connections or relationships between nodes to form triangles or triplets and captures the clustering characteristics in a given graph. Let triads be the structure that two edges with a shared node, then the transitivity can be formally represented as follows,

$$
\text { Transitivity } = 3 \frac {\# \text { Triangles }}{\# \text { Triads }}, \tag {14}
$$

where #Triangles and #Triads are the number of triangles and triads in G, respectively.

![](images/d8ed08c372c5bdc64bce51213a7e191b8800588f6904a3dbc88f905b9a2a7e98.jpg)

<details>
<summary>bar</summary>

| Category | #Nodes |
|---|---|
| 0 | 100 |
| 1 | 45000 |
| 2 | 5000 |
| 3 | 27000 |
| 4 | 11000 |
</details>

(a) Degree distribution of Mutag

![](images/45a38dc405f62187e98499ce55ea3d16dd1e2e921fb30406234ce9f306e61aa5.jpg)

<details>
<summary>bar</summary>

| Index | #Nodes |
|---|---|
| 1 | 0 |
| 2 | 10000 |
| 3 | 120000 |
| 4 | 200000 |
| 5 | 330000 |
| 6 | 520000 |
| 7 | 870000 |
| 8 | 540000 |
| 9 | 560000 |
| 10 | 810000 |
| 11 | 520000 |
| 12 | 180000 |
| 13 | 40000 |
| 14 | 15000 |
| 15 | 0 |
| 16 | 0 |
| 17 | 0 |
</details>

(b) Degree distribution of MNIST-75sp

![](images/cef1de94d48776dde1d7bead71d0ed03b92cc44e7cfcd7c64138599ebc984383.jpg)

<details>
<summary>bar</summary>

| Category | #Nodes |
|---|---|
| 0 | 0 |
| 1 | 210000 |
| 2 | 495000 |
| 3 | 325000 |
| 4 | 25000 |
| 5 | 0 |
| 6 | 0 |
| 7 | 0 |
| 8 | 0 |
| 9 | 0 |
| 10 | 0 |
</details>

(c) Degree distribution of Ogbg-Molhiv

![](images/e71b50a5414c9d82f2545bca668709da9e1391a343deac2acef999d2e3bc2829.jpg)

<details>
<summary>bar</summary>

| Index | #Nodes |
|---|---|
| 1 | 400000 |
| 2 | 160000 |
| 3 | 85000 |
| 4 | 40000 |
| 5 | 20000 |
| 6 | 10000 |
| 7 | 5000 |
| 8 | 2000 |
| 9 | 1000 |
| 10 | 500 |
| 11 | 200 |
| 12 | 100 |
| 13 | 50 |
</details>

(d) Degree distribution of Graph-SST2

![](images/e944a97e6639452ffd94c1126a8a3c7119d7545865f43eb7a4941b669c098a52.jpg)

<details>
<summary>bar</summary>

| Index | #Nodes |
|---|---|
| 0 | 800000 |
| 1 | 1300000 |
| 2 | 650000 |
| 3 | 400000 |
| 4 | 120000 |
| 5 | 60000 |
| 6 | 65000 |
| 7 | 55000 |
| 8 | 55000 |
| 9 | 45000 |
| 10 | 45000 |
| 11 | 45000 |
| 12 | 45000 |
| 13 | 45000 |
| 14 | 45000 |
| 15 | 45000 |
| 16 | 45000 |
| 17 | 35000 |
| 18 | 35000 |
| 19 | 35000 |
| 20 | 35000 |
| 21 | 35000 |
| 22 | 35000 |
| 23 | 35000 |
| 24 | 35000 |
| 25 | 35000 |
| 26 | 35000 |
| 27 | 35000 |
| 28 | 35000 |
| 29 | 35000 |
| 30 | 35000 |
</details>

(e) Degree distribution of PT-Motifs   
Figure 4: The degree distribution of four real-world graph datasets (Mutag, MNIST-75sp, Ogbg-Molhiv, and Graph-SST2) and our synthetic dataset PT-Motifs. The x axis is the node degree.

- Assortativity is defined as the similarity of connections in the given graph $G$ with respect to the node degree. It measures the preference of nodes to connect with nodes of similar or different characteristics. Assortativity helps uncover the underlying patterns of connectivity and provides insights into the organization and functioning of complex networks. The average assortativity of the given graph $G$ is the mean value over all nodes.   
- Efficiency of a pair of nodes is defined as the multiplicative inverse of the shortest path distance between the endpoints in the given graph $G$ . The average global efficiency of a graph is the average efficiency of all pairs of nodes.

\- Clustering of the node $v$ is defined as the fraction of possible triangles through that node. The corresponding clustering coefficient $c_{u}$ can be formulated as,

$$
c _ {v} = 2 \frac {\# \text { Triangles } _ {u}}{\deg_ {u} (\deg_ {u} - 1)}, \tag {15}
$$

where $\#$ Triangles $_{u}$ is the number of triangles through node u and $deg_{u}$ is degree of node u. The average clustering coefficient C of the given graph G with n nodes is defined as follows,

$$
C = \frac {1}{n} \sum_ {v \in G} c _ {v}. \tag {16}
$$

Table 7: Structural patterns of the PT-Motifs dataset and four real-world datasets. 

<table><tr><td></td><td>PT-Motifs</td><td>Mutag</td><td>MNIST-75sp</td><td>Molhiv</td><td>Graph-SST2</td></tr><tr><td>Transitivity</td><td>0.3620</td><td>0.0013</td><td>0.5005</td><td>0.0021</td><td>0</td></tr><tr><td>Avg Assortativity</td><td>0.1525</td><td>-0.4196</td><td>0.3235</td><td>-0.2678</td><td>-0.6269</td></tr><tr><td>Avg Efficiency</td><td>0.5044</td><td>0.3261</td><td>0.3977</td><td>0.3243</td><td>0.5491</td></tr><tr><td>Avg Clustering</td><td>0.4185</td><td>0.0010</td><td>0.5408</td><td>0.0020</td><td>0</td></tr></table>

For an intuitive understanding of the graph structural patterns, we visualize the distribution of node pair efficiency and node assortativity in Figure 5 and Figure 6, respectively. One can notice that the efficiency distributions of PT-Motifs, Mutag, MNIST-75sp, and Molhiv all follow a bell-shaped curve, while that of Graph-SST2 is a little different. Similarly, we found that the node assortativity distributions of Mutag, MNIST-75sp, Molhiv, and PT-Motifs all have one main peak which largely surpasses the adjacent values, while the node assortativity of Graph-SST2 dataset has four peaks.

Overall, the structural patterns above, including the degree distribution, the transitivity, the assortativity, the efficiency, and the clustering, are universal and generalizable across different datasets. However, the expression degree of different structural patterns may differ largely in different datasets. As shown above, the Graph-SST2 dataset strictly follows the power law shaped degree distribution, but its node pair efficiency and node assortativity distributions are different from the other three real-world datasets. Therefore, we propose the structural pattern learning module in $\pi$ -GNN, to capture multiple structural patterns and integrate them for a more universal and generalizable representation.

# C Experimental Details

We first present the details of the pre-training phase over the synthetic PT-Motifs dataset. The PT-Motifs dataset is split into the training set of 50,000 graphs, the validation set of 10,000 graphs, and the testing set of 20,000 graphs. Each graph class has equal number of instances in the three sets. During the pre-training phase, the batchsize is set as $\{32, 64, 128, 256\}$ and the learning rate is set as $\{10^{-3}, 5 \times 10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}\}$ . The pre-training epoch is set as $\{20, 40, 60, 80\}$ . We select the pre-trained $\pi$ -GNN model according to the validation performance. During the pre-training and the fine-tuning phases, we use the Adam optimizer.

As shown in Tables 8 and 9, we present both the downstream predictor architecture and the fine-tuning details of the graph classification datasets and the node classification datasets, respectively. For all the graph classification datasets, we use the global mean pooling as the pooling function. Specially, in the Molhiv predictor, we introduce the virtual node and weighted loss tricks to mitigate the class-imbalance issue (39,684 negative examples and only 1,443 positive examples). All experiments are conducted on a single NVIDIA GeForce 3090 GPU (24GB).

# D Explanatory Visualization

For an intuitive understanding on the $\pi$ -GNN explanation, we present a few visualized explanation results of the Graph-SST2 dataset in Figure 7. Overall, $\pi$ -GNN has the ability to highlight the influential phrases that directly express the positive or negative sentiments in the sentence. Specifically, $\pi$ -GNN correctly allocates large weights to the positive words "a legend" in Figure 7(a), as well as "bewilderingly brilliant" and "entertaining" in Figure 7(c). Furthermore, the negative word, such

![](images/c28728b50535ff71b178f1174002177a2202febae4c87443257d801a5dfea3ca.jpg)

<details>
<summary>line</summary>

| x     | #Nodes |
|-------|--------|
| 0.00  | 0      |
| 0.05  | 0      |
| 0.10  | 0      |
| 0.15  | 0      |
| 0.20  | 30     |
| 0.25  | 90     |
| 0.30  | 180    |
| 0.35  | 120    |
| 0.40  | 80     |
| 0.45  | 20     |
| 0.50  | 40     |
| 0.55  | 10     |
| 0.60  | 0      |
| 0.65  | 0      |
| 0.70  | 0      |
</details>

(a) Node pair efficiency distribution of Mutag

![](images/60ae16f869b5d881c0dd8eeca07ec2cb26db09fca56201674e5adcf515019d1e.jpg)

<details>
<summary>line</summary>

| x     | #Nodes |
|-------|--------|
| 0.35  | 0      |
| 0.36  | 0      |
| 0.37  | 0      |
| 0.38  | 20000  |
| 0.39  | 150000 |
| 0.40  | 400000 |
| 0.41  | 80000  |
| 0.42  | 0      |
</details>

(b) Node pair efficiency distribution of MNIST-75sp

![](images/bdfdc5c70c3443134a9fefc8c1cfba2e3742d53fd606b395abada3855d6a235c.jpg)

<details>
<summary>line</summary>

| x     | #Nodes |
|-------|--------|
| 0.05  | 0      |
| 0.10  | 100    |
| 0.15  | 300    |
| 0.20  | 800    |
| 0.25  | 1500   |
| 0.30  | 2500   |
| 0.35  | 2400   |
| 0.40  | 1800   |
| 0.45  | 1200   |
| 0.50  | 600    |
| 0.55  | 300    |
| 0.60  | 150    |
| 0.65  | 100    |
| 0.70  | 50     |
| 0.75  | 25     |
| 0.80  | 10     |
| 0.85  | 5      |
| 0.90  | 2      |
| 0.95  | 1      |
| 1.00  | 0      |
</details>

(c) Node pair efficiency distribution of Ogbg-Molhiv

![](images/df3366ec46a3f8d4b8f37efdd151b21179218a07efe8ca9377370cdb7ec524f8.jpg)

<details>
<summary>line</summary>

| x     | #Nodes |
|-------|--------|
| 0.00  | 4000   |
| 0.20  | 0      |
| 0.30  | 1000   |
| 0.40  | 1200   |
| 0.50  | 1100   |
| 0.60  | 1500   |
| 0.70  | 3000   |
| 0.80  | 7000   |
| 1.00  | 7000   |
</details>

(d) Node pair efficiency distribution of Graph-SST2

![](images/c65a4e7846e350d34591f381a96b79c33865ad1548da9f84804c4867c3789896.jpg)

<details>
<summary>line</summary>

| x      | #Nodes |
| ------ | ------ |
| 0.10   | 800    |
| 0.15   | 3200   |
| 0.16   | 1600   |
| 0.17   | 1400   |
| 0.18   | 1200   |
| 0.19   | 2000   |
| 0.20   | 800    |
| 0.25   | 200    |
| 0.26   | 800    |
| 0.27   | 1600   |
| 0.28   | 200    |
| 0.29   | 200    |
| 0.30   | 200    |
| 0.31   | 400    |
| 0.32   | 600    |
| 0.33   | 1400   |
| 0.34   | 1600   |
| 0.35   | 2000   |
| 0.36   | 1200   |
| 0.37   | 1400   |
| 0.38   | 1200   |
| 0.39   | 1400   |
| 0.40   | 200    |
| 0.41   | 200    |
| 0.42   | 200    |
| 0.43   | 200    |
| 0.44   | 200    |
| 0.45   | 200    |
| 0.46   | 200    |
| 0.47   | 600    |
| 0.48   | 850    |
| 0.49   | 12000  |
| 0.50   | 2800   |
| 0.51   | 600    |
| 0.52   | 400    |
| 0.53   | 280    |
| 0.54   | 420    |
| 0.55   | 220    |
| 0.56   | 60     |
| 0.57   | 48     |
| 0.58   | 28     |
| 0.59   | 28     |
| 0.60   | 28     |
| 0.61   | 28     |
| 0.62   | 28     |
| 0.63   | 28     |
| 0.64   | 28     |
| 0.65   | 28     |
| 0.66   | 48     |
| 0.67   | 68     |
| 0.68   | 68     |
| 0.69   | 68     |
| 0.70   | 68     |
| 0.71   | 88     |
| 0.72   | 148     |
| 0.73   | 188     |
| 0.74   | 168     |
| 0.75   | 128     |
| 0.76   | 188     |
| 0.77   | 48     |
| 0.78   | 128     |
| 0.79   | 128     |
| 0.80   | 128     |
| 0.81   | 128     |
| 0.82   | 128     |
| 0.83   | 128     |
| 0.84   | 128     |
| 0.85   | 128     |
| 0.86   | 128     |
| 0.87   | 128     |
| 0.88   | 128     |
| 0.89   | 168     |
| 0.90   | 328     |
| 0.91   | 248     |
| 0.92   | 48     |
| 0.93   | 248     |
| 0.94   | 48     |
| 0.95   | 248     |
| 0.96   | 48     |
| 0.97   | 248     |
| 0.98   | 48     |
| 0.99   | 248     |
| 1.00   | 48     |
</details>

(e) Node pair efficiency distribution of PT-Motifs

Figure 5: The node pair efficiency distribution of four real-world graph datasets (Mutag, MNIST-75sp, Ogbg-Molhiv, and Graph-SST2) and the synthetic dataset PT-Motifs.   
Table 8: Predictor architecture and fine-tuning details of graph classification datasets. 

<table><tr><td></td><td>BA-2Motifs</td><td>Spurious-Motif</td><td>Mutag</td><td>MNIST-75sp</td><td>Molhiv</td><td>Graph-SST2</td></tr><tr><td>Backbone</td><td>GCN</td><td>GCN</td><td>GCN</td><td>GIN</td><td>GIN</td><td>GCN</td></tr><tr><td>Layers</td><td>1</td><td>1</td><td>1</td><td>2</td><td>2</td><td>1</td></tr><tr><td>Batchsize</td><td>256</td><td>128</td><td>64</td><td>256</td><td>128</td><td>32</td></tr><tr><td>Learning rate</td><td> $8 \times 10^{-4}$ </td><td> $4 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-3}$ </td><td> $1 \times 10^{-3}$ </td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Epochs</td><td>30</td><td>30</td><td>30</td><td>50</td><td>40</td><td>40</td></tr></table>

as "absurd lengths" and "skip dreck" in Figure 7(b) and Figure 7(d), are identified by $\pi$ -GNN for a transparent sentiment prediction. The visualized explanation again demonstrates the effectiveness of $\pi$ -GNN: (1) the pre-trained $\pi$ -GNN interpreter can faithfully extract the most vital subgraph that contains the label-relevant information; and (2) the pre-trained $\pi$ -GNN interpreter is able to cooperate with specific downstream predictor for both accuracy and interpretability. See Figure 7 for more visualized results of the explanations.

# E Supplement Experiment

To further investigate the impact of the pre-training dataset on the final results, we conduct empirical studies on the size and the imbalance degree of the pre-training dataset.

![](images/1b3e19e844092d4d8147fda46103ac2cb7c98da8768cb2b1ee04a9058d3fe058.jpg)

<details>
<summary>line</summary>

| x      | #Nodes |
| ------ | ------ |
| -1.00  | 0      |
| -0.95  | 0      |
| -0.90  | 0      |
| -0.85  | 0      |
| -0.80  | 0      |
| -0.75  | 0      |
| -0.70  | 0      |
| -0.65  | 0      |
| -0.60  | 0      |
| -0.55  | 0      |
| -0.50  | 0      |
| -0.45  | 0      |
| -0.40  | 0      |
| -0.35  | 0      |
| -0.30  | 0      |
| -0.25  | 0      |
| -0.20  | 0      |
| -0.15  | 0      |
| -0.10  | 0      |
| -0.05  | 0      |
| 0.00   | 0      |
| 0.05   | 0      |
| 0.10   | 0      |
| 0.15   | 0      |
| 0.20   | 0      |
| 0.25   | 0      |
| 0.30   | 0      |
| 0.35   | 0      |
| 0.40   | 0      |
| 0.45   | 0      |
| 0.50   | 0      |
| 0.55   | 0      |
| 0.60   | 0      |
| 0.65   | 0      |
| 0.70   | 0      |
| 0.75   | 0      |
| 0.80   | 0      |
| 0.85   | 0      |
| 0.90   | 0      |
| 0.95   | 0      |
| 1.00   | 0      |
</details>

(a) Node assortativity distribution of Mutag

![](images/0ce42b0e4794d0297c100277edfea8b7e5eb2236557f853ac8075b4336b4b725.jpg)

<details>
<summary>line</summary>

| x      | #Nodes |
| ------ | ------ |
| -0.20  | 0      |
| -0.15  | 0      |
| -0.10  | 0      |
| -0.05  | 0      |
| 0.00   | 500    |
| 0.05   | 750    |
| 0.10   | 500    |
| 0.15   | 250    |
| 0.20   | 500    |
| 0.25   | 1000   |
| 0.30   | 1750   |
| 0.35   | 2500   |
| 0.40   | 3000   |
| 0.45   | 2500   |
| 0.50   | 1500   |
| 0.55   | 500    |
| 0.60   | 0      |
</details>

(b) Node assortativity distribution of MNIST-75sp

![](images/564269e786de161bc332dd0269004dcad5bb8dfe8faed5f3f031502984027cbf.jpg)

<details>
<summary>line</summary>

| x      | #Nodes |
| ------ | ------ |
| -1     | 0      |
| -0.9   | 10     |
| -0.8   | 20     |
| -0.7   | 30     |
| -0.6   | 40     |
| -0.5   | 50     |
| -0.4   | 60     |
| -0.3   | 70     |
| -0.2   | 80     |
| -0.1   | 90     |
| 0      | 100    |
| 0.1    | 110    |
| 0.2    | 120    |
| 0.3    | 130    |
| 0.4    | 140    |
| 0.5    | 150    |
| 0.6    | 160    |
| 0.7    | 170    |
| 0.8    | 180    |
| 0.9    | 190    |
| 1      | 200    |
| 1.1    | 210    |
| 1.2    | 220    |
| 1.3    | 230    |
| 1.4    | 240    |
| 1.5    | 250    |
| 1.6    | 260    |
| 1.7    | 270    |
| 1.8    | 280    |
| 1.9    | 290    |
| 2      | 300    |
| 2.1    | 310    |
| 2.2    | 320    |
| 2.3    | 330    |
| 2.4    | 340    |
| 2.5    | 350    |
| 2.6    | 360    |
| 2.7    | 370    |
| 2.8    | 380    |
| 2.9    | 390    |
| 3      | 400    |
| 3.1    | 410    |
| 3.2    | 420    |
| 3.3    | 430    |
| 3.4    | 440    |
| 3.5    | 450    |
| 3.6    | 460    |
| 3.7    | 470    |
| 3.8    | 480    |
| 3.9    | 490    |
| 4      | 500    |
</details>

(c) Node assortativity distribution of Ogbg-Molhiv

![](images/600144788218dac8f1ea3a898b718deaa70093d3ea7fc67c6303154f3e45d8f6.jpg)

<details>
<summary>line</summary>

| x       | #Nodes |
| ------- | ------ |
| -1.0    | 100000 |
| -0.9    | 0      |
| -0.8    | 0      |
| -0.7    | 0      |
| -0.6    | 0      |
| -0.5    | 0      |
| -0.4    | 0      |
| -0.3    | 0      |
| -0.2    | 0      |
| -0.1    | 0      |
| 0.0     | 0      |
| 0.1     | 0      |
| 0.2     | 0      |
| 0.3     | 0      |
| 0.4     | 0      |
| 0.5     | 0      |
| 0.6     | 0      |
| 0.7     | 0      |
| 0.8     | 0      |
| 0.9     | 0      |
| 1.0     | 0      |
</details>

(d) Node assortativity distribution of Graph-SST2

![](images/2cc40d75104a96b6c48ec1e44967528386531796e87e41e6a6792bc2a8550c95.jpg)

<details>
<summary>line</summary>

| x_value | #Nodes |
| ------- | ------ |
| -0.75   | 200    |
| -0.70   | 500    |
| -0.65   | 100    |
| -0.60   | 600    |
| -0.55   | 500    |
| -0.50   | 700    |
| -0.45   | 1200   |
| -0.40   | 200    |
| -0.35   | 800    |
| -0.30   | 4300   |
| -0.25   | 2800   |
| -0.20   | 1700   |
| -0.15   | 500    |
| -0.10   | 1400   |
| -0.05   | 1300   |
| 0.00    | 800    |
| 0.05    | 400    |
| 0.10    | 300    |
| 0.15    | 100    |
| 0.20    | 200    |
| 0.25    | 150    |
| 0.30    | 600    |
| 0.35    | 1100   |
| 0.40    | 10     |
| 0.45    | 500    |
| 0.50    | 700    |
| 0.55    | 30     |
| 0.60    | 10     |
| 0.65    | 20     |
| 0.70    | 700    |
| 0.75    | 1100   |
</details>

(e) Node assortativity distribution of PT-Motifs   
Figure 6: The node assortativity distribution of four real-world graph datasets (Mutag, MNIST-75sp, Ogbg-Molhiv, and Graph-SST2) and our synthetic dataset PT-Motifs.

Table 9: Predictor architecture and fine-tuning details of node classification datasets. 

<table><tr><td></td><td>BA-Shapes</td><td>BA-Community</td><td>Tree-Cycles</td><td>Tree-Grid</td></tr><tr><td>Backbone</td><td>GCN</td><td>GCN</td><td>GCN</td><td>GCN</td></tr><tr><td>Layers</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-1}$ </td><td> $1 \times 10^{-1}$ </td><td> $1 \times 10^{-2}$ </td><td> $5 \times 10^{-2}$ </td></tr><tr><td>Epochs</td><td>20</td><td>20</td><td>20</td><td>20</td></tr></table>

First, compared with PT-Motifs (80,000 graphs) in the main experiment, we generate another two datasets with different sizes. In detail, PT-Motifs-M with 50,000 graphs and PT-Motifs-S with 10,000 graphs represent the middle-level and the small-level pre-training datasets, respectively. The split ratio of the training, validation, and testing set is $\{0.7, 0.1, 0.2\}$ in PT-Motifs-M and PT-Motifs-S. As shown in Table 10, the results show that even the PT-Motifs-S is able to outperform the model without pre-training. Moreover, we can notice that a large pre-training dataset can indeed improve the performance more significantly than that with small size.

Furthermore, it is possible that if the synthetic pre-training dataset is imbalanced, the performance improvement on downstream tasks will degrade, in terms of both the interpretation and prediction. To further investigate this issue, we have added supplement experiments on imbalanced pre-training dataset. Following existing works $[13, 11]$ , to generate the imbalanced datasets, we sample the

![](images/87786521945f680af1d2c896616d15a4fd5cffe093a36032b7a77a877da3f5f0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["starts"] --> B["with"]
    B --> C["a"]
    C --> D["legend"]
    D --> C
```
</details>

(a) Explanation subgraph: positive sentiment

![](images/36964a1ad21cc84a58af89477e300c5d7f32426ed73c62925023c280a69222c0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["goes"] --> B["to"]
    B --> C["absurd"]
    B --> D["lengths"]
```
</details>

(b) Explanation subgraph: negative sentiment

![](images/a07226c09813bcd78da4006eeec5c58e0cf99be0df83567bfe44a5130c0f593a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["bewilderingly"] --> B["brilliant"]
    B --> C["entertaining"]
    C -->|and| B
```
</details>

(c) Explanation subgraph: positive sentiment

![](images/f918b121d1218fb7531316bc8e31f47c350e3163c2798cd0816d13593e853c3f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["skip"] --> B["dreck"]
    B --> C
    C --> A
    style A fill:#cce5ff,stroke:#333
    style B fill:#cce5ff,stroke:#333
    style C fill:#cce5ff,stroke:#333
    linkStyle 0 stroke:#000,stroke-width:2px
    linkStyle 1 stroke:#000,stroke-width:2px
    linkStyle 2 stroke:#000,stroke-width:2px
    linkStyle 3 stroke:#000,stroke-width:2px
    linkStyle 4 stroke:#000,stroke-width:2px
    linkStyle 5 stroke:#000,stroke-width:2px
    linkStyle 6 stroke:#000,stroke-width:2px
    linkStyle 7 stroke:#000,stroke-width:2px
    linkStyle 8 stroke:#000,stroke-width:2px
    linkStyle 9 stroke:#000,stroke-width:2px
    linkStyle 10 stroke:#000,stroke-width:2px
```
</details>

(d) Explanation subgraph: negative sentiment

![](images/db19788138c7ba1249b94a05084052a33a25c31c0fe111b383c788c0b1ebe860.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["with"] --> B["leonine"]
    B --> C["power"]
```
</details>

(e) Explanation subgraph: positive sentiment

![](images/9b977daed397f15077f912ad2feab697b21044e364dbc31e1d780d3f50aefc70.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["plodding"] --> B["picture"]
```
</details>

(f) Explanation subgraph: negative sentiment

![](images/4358843fe8d3d1dca404d09cc5b1e8c99276745c7b86a855948b010410d99b24.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["very"] --> B["funny"]
    A --> C["joke"]
    B --> C
```
</details>

(g) Explanation subgraph: positive sentiment

![](images/7929c28edd32dba735db10b0417f3ce6da05fc7a3e413862ed7c8c5f74e24e5a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    i --> hate
    hate --> it
```
</details>

(h) Explanation subgraph: negative sentiment

![](images/bf3e8f8a5e8e2eec0d8eaf7fa9dcd64554b38addd650e0657d3f496359c2a64f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["to"] --> B["genuinely"]
    B --> C["satisfying"]
    C --> A
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
```
</details>

(i) Explanation subgraph: positive sentiment

![](images/e26a41d3b41aa109ef003bac2107834e385b447e38540fbf1a1bae71cbdc160f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["dime"] --> B["store"]
    B --> C["ruminations"]
    C --> A
```
</details>

(j) Explanation subgraph: negative sentiment

![](images/d4e72f19065de0b585a8e598b493dc2297af2aec08184d555f5121e816069868.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["perfect"] --> B["star"]
    B --> C["vehicle"]
```
</details>

(k) Explanation subgraph: positive sentiment

![](images/2229d7e3ddacc6e044013561fbe21d118f03ee4016a72004246d30b6fd016cbc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["action"] --> B["is"]
    B --> C["stilted"]
    style B fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
```
</details>

(1) Explanation subgraph: negative sentiment

Figure 7: Visualization of $\pi$ -GNN explanation on the Graph-SST2 dataset. Each graph represents a comment, where the explanations are highlighted by bule boxes and bolder lines.   
Table 10: The impact of pre-training datasets with different sizes. 

<table><tr><td>Task</td><td>Dataset</td><td>PT-Motifs</td><td>W/O Pre-train</td><td>PT-Motifs-M</td><td>PT-Motifs-S</td></tr><tr><td rowspan="2">Interpretation</td><td>BA-2Motifs</td><td>99.33</td><td>93.19</td><td>98.91</td><td>97.07</td></tr><tr><td>Mutag</td><td>99.81</td><td>95.29</td><td>99.06</td><td>96.28</td></tr><tr><td rowspan="2">Prediction</td><td>Ogbg-Molhiv</td><td>80.86</td><td>79.71</td><td>80.77</td><td>79.82</td></tr><tr><td>Graph-SST2</td><td>88.05</td><td>83.48</td><td>87.59</td><td>85.23</td></tr></table>

explanatory $G_{e}$ from uniform distribution, while the base $G_{b}$ is determined by the following formula,

$$
P (G _ {b}) = b \times I (G _ {b} = G _ {e}) + \frac {1 - b}{4} \times I (G _ {b} \neq G _ {e}). \tag {17}
$$

Therefore, we can manipulate b to control the imbalance degree and the imbalance degree is defined as $i = 4b/(1 - b)$ . The corresponding result is reported in Table 11. The results demonstrate that when pretraining on an imbalanced dataset, the performance improvement is less significant than that on the balanced one, but still better than that without pretraining. Moreover, an overly imbalanced pre-training dataset ( $b = 0.7, i \approx 9.3$ ) may cause negative transfer issue (on Muatg, Molhiv, and Graph-SST2). Therefore, the pre-training dataset ought to be balanced and thus can mitigate the negative transfer issue to some extents.

Table 11: The impact of pre-training datasets with different sizes. 

<table><tr><td>Task</td><td>Dataset</td><td>PT-Motifs</td><td>W/O Pre-train</td><td>PT-Motifs-M</td><td>PT-Motifs-S</td></tr><tr><td rowspan="2">Interpretation</td><td>BA-2Motifs</td><td>99.33</td><td>93.19</td><td>98.91</td><td>97.07</td></tr><tr><td>Mutag</td><td>99.81</td><td>95.29</td><td>99.06</td><td>96.28</td></tr><tr><td rowspan="2">Prediction</td><td>Ogbg-Molhiv</td><td>80.86</td><td>79.71</td><td>80.77</td><td>79.82</td></tr><tr><td>Graph-SST2</td><td>88.05</td><td>83.48</td><td>87.59</td><td>85.23</td></tr></table>

# F Hyper-parameter Analysis

In Figure 8, we conduct analysis on the number of basic pattern-learners, to evaluate the tendency of $\pi$ -GNN performance with the increase of the pattern-learner number. Additionally, we report the performance of the variant without pre-training phase, which is marked by the subscript "DFT". One can observe that as the number of basic learners increasing, both the interpretation and prediction performance generally improves, which demonstrates the co-existence of multiple structural patterns. For the Mutag dataset, when we increase the number to 2, the performance improvement (5.76%) is the most significant. But when the number increases to 16 and 32, the performance is inferior to that of 8 basic pattern-learners. We ascribe this degradation to the difficulty of integrating more and more patterns when the basic learner number increasing. For the Graph-SST2 dataset, the prediction performance consistently improves along with the number of basic learners. This may indicate that the structural patterns in Graph-SST2 are more intricate than those in Mutag dataset. Moreover, the induced variant without the pre-training phase is superior to the $\pi$ -GNN model consistently, which again verifies the effectiveness of the explainer pre-training phase, by 3.32% ROC-AUC score on Mutag dataset and 4.08% accuracy on Graph-STT2 dataset on average.

# G Derivation

First, we restate and prove Theorem 1 [44].

Theorem 1. Let $\Sigma_{n}$ be the set of all adjacency matrix $\mathbf{A}$ with $n$ nodes. Given a graph $G = (\mathcal{V},\mathcal{E})\in \Sigma_n,n\geq 2$ , let $\Gamma^{*}(S,\mathbf{A})$ be a most-expressive structural representation of nodes set $S\subseteq \mathcal{V}$ in $G$ . $\forall \mathbf{A}\in \Sigma_n$ , there exists a most-expressive node representation $\mathbf{Z}^{\ast}|\mathbf{A}$ satisfies the relationship as follows,

$$
\Gamma^ {*} (S, \mathbf {A}) = \mathbb {E} _ {\mathbf {Z} ^ {*}} [ f ^ {(| S |)} ((\mathbf {Z} _ {v} ^ {*}) _ {v \in S}) | \mathbf {A} ], \forall S \subseteq \mathcal {V}, \tag {18}
$$

for an appropriate k-variable function $f^{(k)}(\cdot)$ .

Proof. Given graph $G = (\mathcal{V}, \mathcal{E})$ with n nodes, we construct an equivalent set of the most-expressive structural representation $\Gamma^{*}(S, \mathbf{A})$ , with permutations on node indices:

$$
\Pi (\mathbf {A}) = \left\{\Gamma^ {*} (v, \mathbf {A}, \pi (1, 2, \dots , n)) _ {\forall v \in \mathcal {V}} \mid \pi \in \Pi_ {n} \right\} \tag {19}
$$

Define $Z^{*}|A$ as the random variable with a uniform measure over $\Pi(\mathbf{A})$ . Assume the node subset S has no other joint isomorphic set $S'$ . Then, for any such S and any element $\gamma_{\pi} \in \Pi(\mathbf{A})$ as follow,

$$
\gamma_ {\pi} = \Gamma^ {*} (v, \mathbf {A}, \pi (1, 2, \dots , n)) _ {\forall v \in \mathcal {V}}, \tag {20}
$$

there exists a bijective measurable map between the nodes in $S$ and their positions in the representation vector $\gamma_{\pi}$ . Next, we consider the representation set $\mathcal{O}_S(\mathbf{A})$ restricted to the node subset $S$ as follows,

$$
\mathcal {O} _ {S} (\mathbf {A}) := \left\{\Gamma^ {*} (v, \mathbf {A}, \pi (1, 2, \dots , n)) _ {\forall v \in S} \mid \pi \in \Pi_ {n} \right\} = \left\{\left((\mathbf {Z} _ {v} ^ {*}) _ {v \in S} | \mathbf {A}\right) \right\}, \tag {21}
$$

and prove that there exists an surjection between $\mathcal{O}_{S}(\mathbf{A})$ and $\Gamma^{*}(S,\mathbf{A})$ . The surjection exists if, $\forall$ non-isomorphic node subset $S_{1}, S_{2}$ , it implies $\mathcal{O}_{S_{1}}(\mathbf{A}) \neq \mathcal{O}_{S_{2}}(\mathbf{A})$ . This condition naturally holds if $|S_{1}| \neq |S_{2}|$ . When $|S_{1}| = |S_{2}|$ , we prove by contradiction and assume $\mathcal{O}_{S_{1}}(\mathbf{A}) = \mathcal{O}_{S_{2}}(\mathbf{A})$ . Since the node indices is unique and $\Gamma^{*}$ is most-expressive, the representation $\Gamma^{*}(v, \mathbf{A}, \pi(1, 2, \cdots, n))$ of node v and permutation $\pi$ is unique too. As $S_{1}$ is non-isomorphic to $S_{2}$ , there must exist at least one node $u \in S_{1}$ that has no isomorphic equivalent element in $S_{2}$ . Hence, $\exists \pi \in \Pi_{n}$ that provides a representation $\Gamma^{*}(u, \mathbf{A}, \pi'(1, 2, \cdots, n))$ and $\not\exists \pi' \in \Pi_{n}, v \in S_{2}$ that the corresponding representation

![](images/a4e51c78ae206ab65ac52a01de4ec8181ecd93863188b6188a6079006cfeecec.jpg)

<details>
<summary>line</summary>

| Number of Basic Pattern-Learners | π-GNN | π-GNN_DFT |
| --------------------------------- | ----- | --------- |
| 1                                 | 88.7  | 85.5      |
| 2                                 | 94.5  | 92.5      |
| 4                                 | 96.5  | 93.2      |
| 8                                 | 99.8  | 95.3      |
| 16                                | 98.5  | 95.8      |
| 32                                | 99.5  | 96.0      |
</details>

(a) Interpretation performance on Mutag

![](images/aa544977ef3253b1d56b327f01cb01e41ffaa06d7a9a83a6862a8a3ebe41469d.jpg)

<details>
<summary>line</summary>

| Number of Basic Pattern-Learners | π-GNN  | π-GNN_DFT |
| -------------------------------- | ------ | --------- |
| 1                                | 88.5   | 80.0      |
| 2                                | 89.8   | 84.8      |
| 4                                | 91.2   | 86.7      |
| 8                                | 93.2   | 86.3      |
| 16                               | 94.0   | 87.8      |
| 32                               | 94.2   | 87.9      |
</details>

(b) Interpretation performance on Spurious-Motif

![](images/8566aff55d047eb88ac5a38ff086bdf4acaba552884748451e3b79430111a9ae.jpg)

<details>
<summary>line</summary>

| Number of Basic Pattern-Learners | π-GNN  | π-GNN_DFT |
| --------------------------------- | ------ | --------- |
| 1                                 | 85.4   | 82.0      |
| 2                                 | 85.8   | 82.3      |
| 4                                 | 87.0   | 83.0      |
| 8                                 | 88.0   | 83.5      |
| 16                                | 90.8   | 86.2      |
| 32                                | 92.0   | 86.9      |
</details>

(c) Prediction performance on Graph-SST2

![](images/7a827daf79a49ce18f5d0c4666c9d290e7d68eeb66c3c80afd42f1056a47d2e9.jpg)

<details>
<summary>line</summary>

| Number of Basic Pattern-Learners | π-GNN  | π-GNN_DFT |
| --------------------------------- | ------ | --------- |
| 1                                 | 64.0   | 63.5      |
| 2                                 | 66.5   | 64.5      |
| 4                                 | 69.0   | 67.0      |
| 8                                 | 74.5   | 70.5      |
| 16                                | 75.5   | 72.0      |
| 32                                | 75.8   | 72.8      |
</details>

(d) Prediction performance on Spurious-Motif   
Figure 8: The hyper-parameter analysis on the number of the basic pattern-learners.

$\Gamma^{*}(u,\mathbf{A},\pi'(1,2,\cdots,n))$ matches. Finally, we conclude a contradiction from the original assumption $\mathcal{O}_{S_{1}}(\mathbf{A})=\mathcal{O}_{S_{2}}(\mathbf{A})$ and we know the surjection between $\mathcal{O}_{S}(\mathbf{A})$ and $\Gamma^{*}(S,\mathbf{A})$ does exist.

Furthermore, it has been proved that for finite multisets with real number elements, a most-expressive multiset function can be defined as the expectation of a function $f^{(|S|)}$ over the multiset. Therefore, there exists some subjective function $f^{(|S|)}$ whose expectation over $\mathcal{O}_{S}(\mathbf{A})$ gives $\Gamma^{*}(S,\mathbf{A})$ .

Next, we restate and prove Theorem 2.

Theorem 2. The structural representation of edge $e = (v_i, v_j)$ can be learnt by simply approaching a function $f^{(2)}$ which satisfies $\Gamma(e, \mathbf{A}) = f^{(2)}(\mathbb{E}[(\mathbf{Z}_v)_{v \in \{i,j\}}|\mathbf{A}])$ .

Proof. According to Theorem 1, $(\mathbf{Z}_{v})_{v\in\{i.j\}}|\mathbf{A}$ can be represented as follows,

$$
\left(\mathbf {Z} _ {v}\right) _ {v \in \{i, j \}} | \mathbf {A} = \varphi \left(\Gamma (v, \mathbf {A}) _ {v \in \{i, j \}}, \epsilon_ {\{i, j \}}\right), \tag {22}
$$

where the noise $\epsilon_{\{i,j\}}$ is marginalized from an independent noise distribution. With an assumption of $f^{(2)}$ that is able to capture the structural dependencies within the adjacent matrix A, we can compute the expectation of $(\mathbf{Z}_{v})_{v\in\{i,j\}}|\mathbf{A}$ and eliminate the noise.

# H Algorithm

We present the algorithm of graph-hypergraph transformation as following.

Algorithm 1 Graph-Hypergraph Transformation   
Input: Edge index $\Omega \in \mathbb{R}^{2\times |\mathcal{E}|}$ of raw graph $G$ Output: Hyperedge index $\Omega_h$ of hypergraph $G_h$ 1: initialize hyperedge set $\mathcal{E}_h \Leftarrow \emptyset$ 2: initialize hyperedge index $\Omega_h \Leftarrow \emptyset$ 3: for edge $e_i = (u, v)$ in $\Omega$ do
4: $\mathcal{E}_h^u \Leftarrow \mathcal{E}_h^u \cup \{i\}, \mathcal{E}_h^v \Leftarrow \mathcal{E}_h^v \cup \{i\}$ // Record the endpoints of each edge
5: end for
6: for the $p$ -th hyperedge $\mathcal{E}_h^p$ in $\mathcal{E}_h$ do
7:    for item $q$ in $\mathcal{E}_h^p$ do
8: $\Omega_h \Leftarrow \Omega_h \cup \{(p, q)\}$ // Allocate the same index $p$ for each individual hypernode $q$ 9:    end for
10: end for
11: return $\Omega_h$

We also provide an illustrative example of the transformation process in Figure 2 of Section 3.1.

# I Limitations

At last, we provide open discussion about the limitations of the $\pi$ -GNN model.

Efficiency. If the number of basic pattern-learners need to be increased to a large amount for some intricate graph datasets, the computational efficiency of $\pi$ -GNN will become the bottleneck. Although the efficiency can be improved by introducing multi-thread computation, the time complexity of $\pi$ -GNN with multiple basic pattern-learners is higher than the current interpretable GNNs.

Feature Dimension. The feature dimensions of the downstream graph datasets are usually different from that of the PT-Motifs dataset in pre-training phase. Therefore, we have to additionally introduce a linear layer to align the dimension differences between PT-Motifs and the downstream datasets. Though this linear layer can be merged into the hypergraph refining module, it indeed increases the optimization difficulty when fine-tuning.

Node Feature. $\pi$ -GNN focuses on identifying the influential subgraphs by computing the edge contribution score, but it is unable to select the important fraction of the node features that leads to the model prediction. As a future direction, we consider to systematically extend $\pi$ -GNN to the interpretation problem in node classification and link prediction task, where the node features are more informative and influential than those in the graph classification task.

Unseen Structural Patterns. As shown in Appendix B.2, several structural patterns are universal and generalize to the synthetic PT-Motifs dataset and the real-world datasets (e.g., Mutag, MNIST-75sp, Molhiv, Graph-SST2). During the pre-training phase over PT-Motifs dataset, $\pi$ -GNN extracts these universal structural patterns and combines them with the local structural interactions to achieve generalizable interpretation. But for some unseen structural patterns that does not exist in PT-Motifs, $\pi$ -GNN is unable to capture and then employ them to identify explanations.