# EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions

Huayu Deng $^{1}$ Xiangming Zhu $^{1}$ Yunbo Wang $^{1}$ Xiaokang Yang $^{1}$

# Abstract

Graph neural networks have been a powerful tool for mesh-based physical simulation. To efficiently model large-scale systems, existing methods mainly employ hierarchical graph structures to capture multi-scale node relations. However, these graph hierarchies are typically manually designed and fixed, limiting their ability to adapt to the evolving dynamics of complex physical systems. We propose EvoMesh, a fully differentiable framework that jointly learns graph hierarchies and physical dynamics, adaptively guided by physical inputs. EvoMesh introduces anisotropic message passing, which enables direction-specific aggregation of dynamic features between nodes within each hierarchy, while simultaneously learning node selection probabilities for the next hierarchical level based on physical context. This design creates more flexible message shortcuts and enhances the model's capacity to capture long-range dependencies. Extensive experiments on five benchmark physical simulation datasets show that EvoMesh outperforms recent fixed-hierarchy message passing networks by large margins. The project page is available at https://hbell99.github.io/evo-mesh/.

# 1. Introduction

Simulating physical systems with deep neural networks has achieved remarkable success due to their efficiency compared with traditional numerical solvers. Graph Neural Networks (GNNs) have been validated as a powerful tool for mesh-based simulation, such as for fluids and rigid collisions (Wu et al., 2020). The primary mechanism driving the GNN-based models is message passing, where time-varying physical quantities are encoded within the mesh structure and are temporally updated by aggregating information

$^{1}$ MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University. Correspondence to: Yunbo Wang <yunbow@sjtu.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

Table 1. Comparison of mesh-based physical simulation models. Dynamic hierarchy refers to hierarchical graph structures that evolve over time. Adaptive indicates that the graph structures are determined by physical inputs. Prop. denotes feature propagation. 

<table><tr><td>Model</td><td>Dynamic Hierarchy</td><td>Adaptive Hierarchy</td><td>Anisotropic Intra-level Prop</td><td>Learnable Inter-level Prop</td></tr><tr><td>MGN (2021)</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Lino et al. (2022)</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>BSMS (2023)</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Eagle (2023)</td><td>✕</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>HCMT (2024)</td><td>✕</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>EvoMesh</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

broadcast from neighboring nodes (Sanchez-Gonzalez et al., 2020; Pfaff et al., 2021; Allen et al., 2023). Existing methods generally rely on repeated local message passing to propagate influence over long distances, which becomes extremely costly for large-scale mesh graphs. A common solution involves using multi-scale graph structures to create direct information shortcuts between distant nodes. (Lino et al., 2022; Cao et al., 2023; Yu et al., 2024; Han et al., 2022; Fortunato et al., 2022).

However, as shown in Table 1, previous methods commonly rely on heuristic node selection to create predefined (data-independent) coarser message passing graphs (Cao et al., 2023; Yu et al., 2024). These predefined graphs limit the model's adaptation ability in two key ways. First, the fixed graph hierarchies, applied to the entire input sequence, do not account for the variety of physical contexts. In practical systems like turbulence, even with identical boundary conditions, small changes in initial conditions can lead to significant differences in subsequent dynamics. Second, since the spatial correlations in a physical process can evolve over time, static graph hierarchies are insufficient for capturing the time-varying node interactions.

To tackle this challenge, we propose a novel neural network approach named EvoMesh, which constructs data-adaptive and time-evolving graph hierarchies based on the input physical quantities. The key insight is to develop a differentiable node selection method that allows for flexible correlation of long-range, dynamic node interactions. This is technically supported by an anisotropic message passing (AMP) mechanism, which (i) aggregates neighboring features with non-uniform, learnable importance weights within each hi-

erarchical level, (ii) predicts the probabilities of the node being retained for the next hierarchy based on physical context, and (iii) adaptively learns cross-hierarchy interactions to optimize information flow across scales. To enable differentiability in the node selection process, we approximate the discrete downsampling decisions using Gumbel-Softmax.

Another advantage of the AMP mechanism is its ability to enable features to transfer between nodes with varying importance, aligning with the directionally non-uniform nature of the dynamic patterns, as observed in scenarios such as CylinderFlow, AirFoil, and Flying Flag simulations. It applies to both intra-level and inter-level feature propagation. In contrast, as shown in Table 1, most previous GNN-based mesh simulation methods perform isotropic feature aggregation within the intra-level transition and rely on unlearnable importance weights to transfer inter-level information across hierarchical levels, assuming equal contributions from neighboring nodes.

Overall, our contributions are summarized as follows:

- We present EvoMesh, which generates dynamic graph hierarchies through differentiable node selection, enabling adaptive modeling of multi-scale physical relations.   
- EvoMesh employs anisotropic message passing to enable directionally varied feature propagation both within and across graph hierarchies.   
- On average, EvoMesh outperforms fixed-hierarchy models by around 20% across a range of standard benchmarks. It also demonstrates strong generalization to test cases with time-varying mesh structures, novel resolutions, and out-of-distribution dynamics.

# 2. Preliminaries

Message passing. We consider simulating mesh-based physical systems, where the task is to predict the dynamic quantities of the mesh at future timesteps given the current mesh configuration. A mesh-based system is represented as a bi-directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})^{1}$ , where V and E denote the set of nodes and edges, respectively. Message passing neural networks (MPNNs) compute the node representations by stacking multiple message passing layers of the form:

$$
\text { Edge   update: } \hat {\mathbf {e}} _ {i j} = \phi^ {e} (\mathbf {e} _ {i j}, \mathbf {v} _ {i}, \mathbf {v} _ {j}); \tag {1}
$$

Node update: $\hat{\mathbf{v}}_i = \phi^v (\mathbf{v}_i,\psi (\{\hat{\mathbf{e}}_{ij}\mid \forall j,e_{ij}\in \mathcal{E}\}))$ , (2)

where $v_{i}$ is the feature of node $v_{i} \in V$ and $\psi$ denotes a non-parametric aggregation function. The function $\phi^{e}$ updates the features of edges based on the endpoints, while $\phi^{v}$ updates the node states with aggregated messages from its neighbors. In existing GNN-based mesh simulation methods, multi-layer perceptrons (MLPs) with residual connections are commonly employed for $\phi^{e}(\cdot)$ and $\phi^{v}(\cdot)$ , with the non-parametric aggregation function $\psi(\cdot)$ being defined as the sum of edge features. Notably, since the aggregation function treats all neighbors equally, the contributions from neighboring nodes may be averaged out, and the repeated message-passing process can further dilute distinctive node features. This issue is exacerbated in dynamic physical systems, where transferring directed patterns is crucial. Attention-based methods address this issue by reweighting neighbor features, either locally or globally (Veličković et al., 2018; Yu et al., 2024; Han et al., 2022; Yun et al., 2019). While effective for directional aggregation, most weighting remains limited to intra-level features and does not support dynamic graph hierarchy construction.

Hierarchical MPNNs. To facilitate long-range modeling, hierarchical MPNNs process information at L scales by creating a graph for each level and propagating information between them (Lino et al., 2022; Fortunato et al., 2022; Cao et al., 2023; Yu et al., 2024). Let $\mathcal{G}_{1} = (\mathcal{V}_{1}, \mathcal{E}_{1})$ represent the graph structure at the finest level, i.e., the input mesh. The lower-resolution graphs $G_{2}, G_{3}, \ldots, G_{L}$ , with $|V_{1}| > |V_{2}| > \ldots > |V_{L}|$ , contain fewer nodes and edges, which allows for more efficient feature propagation over longer physical distances with certain propagation steps. The typical process for constructing multi-scale structures primarily involves downsampling and upsampling between adjacent graph hierarchies. Downsampling reduces the number of nodes while upsampling transfers information from a lower-resolution graph to a higher-resolution one. The downsampling operation includes two steps:

\- SELECT: Nodes are selected from the current graph structure $\mathcal{G}_l$ to create a new, coarser graph $\mathcal{G}_{l+1}$ . Various strategies have been proposed to construct $\mathcal{V}_{l+1}$ , including hand-crafted designs (Lino et al., 2022; Cao et al., 2023; Yu et al., 2024), geometric clustering (Han et al., 2022; Janny et al., 2023), and differentiable pooling methods that predict cluster assignments or select top-ranked informative nodes (Ying et al., 2018; Gao & Ji, 2019; Lee et al., 2019; Ranjan et al., 2020). The edges $\mathcal{E}_{l+1}$ in $\mathcal{G}_{l+1}$ are constructed by connecting the selected nodes based on the original edges $\mathcal{E}_l$ . However, this process can sometimes lead to loss of connectivity and introduce partitions (Gao & Ji, 2019; Lee et al., 2019; Cao et al., 2023). To mitigate this, connectivity in $\mathcal{E}_{l+1}$ can be strengthened by adding $K$ -hop edges.

\- REDUCE: The features of the nodes in $\mathcal{V}_{l+1}$ are aggregated from their corresponding neighborhood features in the finer graph $\mathcal{G}_l$ .

The upsampling process is represented by EXPAND, which

![](images/aabc092905ab2972a947ebb16ad19fa7f01d48f1ab1f3a2f733e46d19ef5c6a8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input
        G1[" G₁ "] -->|1xAMP| G1a[" G₁ "]
        G1a -->|DiffSELECT REDUCE| G2a[" G₂ "]
        G2a -->|1xAMP| G2b[" G₂ "]
        G2b -->|DiffSELECT REDUCE| GLL[" Gₗ "]
        GLL -->|1xAMP| GLLb[" Gₗ "]
        G1a -->|EXPAND| G1b
        G2a -->|EXPAND| G2b
    end
    subgraph Output
        G1a -->|1xAMP| G1b
        G2a -->|EXPAND| G2b
        G2b -->|EXPAND| GLLb[" Gₗ "]
        GLLb -->|Forward Gradient| Output
    end
    style Input fill:#f9f,stroke:#333
    style Output fill:#bbf,stroke:#333
```
</details>

![](images/6d0f0b167bde69b36094db1b653c7af442f7fde30c6e19663aa906e96582b465.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Node"] -->|0.2| B["v_i"]
    A -->|0.5| C["v_j ∈ N_{vi}"]
    B -->|0.1| D["Node"]
    B -->|0.2| E["Node"]
    C -->|0.5| F["Node"]
    D -->|0.05| G["Node"]
    E -->|0.2| H["Node"]
    F -->|0.5| I["Node"]
    style A fill:#ccc,stroke:#333
    style B fill:#fff,stroke:#333
    style C fill:#fff,stroke:#333
    style D fill:#fff,stroke:#333
    style E fill:#fff,stroke:#333
    style F fill:#fff,stroke:#333
    style G fill:#fff,stroke:#333
    style H fill:#fff,stroke:#333
    style I fill:#fff,stroke:#333
    note bottom of B: v'_i = φ^v(v_i, Σ_{j∈N_i} α_ij e'_ij)
    note bottom of C: v_j ∈ N_{vi}
```
</details>

![](images/f0e6dfc9f1a3af2e2aec01ddea0b34da03205326d89300a7f7db3d1c2988ed59.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A[" "] --> B[" "]
    B --> C[" "]
    C --> D[" "]
    D --> E[" "]
    E --> F[" "]
    F --> G[" "]
    G --> H[" "]
    H --> I[" "]
    I --> J[" "]
    J --> K[" "]
    K --> L[" "]
    L --> M[" "]
    M --> N[" "]
    N --> O[" "]
    O --> P[" "]
    P --> Q[" "]
    Q --> R[" "]
    R --> S[" "]
    S --> T[" "]
    T --> U[" "]
    U --> V[" "]
    V --> W[" "]
    W --> X[" "]
    X --> Y[" "]
    Y --> Z[" "]
    Z --> A
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
    style R fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style T fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style V fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333
```
</details>

Figure 1. The architecture of EvoMesh. Physical dynamics is modeled on multiple graph resolutions with adaptive structures, $G_{1}, G_{2}, \ldots, G_{L}$ , and are processed using their respective AMP layers. The DiffSELECT operation performs differentiable pooling to create coarser graphs with learnable downsampling probabilities. REDUCE and EXPAND integrate inter-level information using learned feature aggregation weights over the neighboring nodes. EvoMesh is trained end-to-end with one-step supervision.

is the inverse of the REDUCE function and transfers information from the coarser level back to the finer level. Most previous work generates coarser graphs either by using numerical software or by downsampling the input mesh through heuristic pooling strategies (Cao et al., 2023; Lino et al., 2022; Yu et al., 2024; Janny et al., 2023). This process is performed during the data preprocessing stage. The preprocessed hierarchy with the same input mesh topology is reused across different initial conditions and time steps.

# 3. Method

In this section, we introduce EvoMesh, a fully differentiable model that adaptively generates time-evolving graph hierarchies over the sequence, while simultaneously simulating the physical system over these learned hierarchical graphs. Figure 1 demonstrates an overview of the proposed model, which operates in an encode-process-decode pipeline. The encoder first maps the input field to a latent feature space $V_{1} = \{v_{i} | v_{i} \in V_{1}\}$ at the original mesh resolution. Subsequently, we model the physical dynamics across the learned multi-scale graph hierarchies with adaptive graph structures.

In Section 3.1, we present the details of the AMP layer. In Section 3.2, we discuss the approach for learning context-aware graph hierarchies. In Section 3.3, we describe the inter-level downsampling and upsampling processes that incorporate AMP-based feature propagation. Finally, in Section 3.4, we outline the implementation details.

# 3.1. Anisotropic Message Passing

We introduce the AMP layer, which facilitates information propagation both within and between graph hierarchies, enabling EvoMesh to effectively capture local and long-range dependencies simultaneously.

As shown in Eq. (2), a common non-parametric aggregation in GNN-based mesh simulation is to use the summation for node update: $\hat{\mathbf{v}}_{i} = \phi^{v}\left(\mathbf{v}_{i}, \sum_{v_{j} \in \mathcal{N}_{v_{i}}} \hat{\mathbf{e}}_{ij}\right)$ , where $v_{j} \in N_{v_{i}}$ denotes a neighboring node of $v_{i}$ in the graph.

To differentiate the contributions of neighboring nodes, the AMP layer employs learnable parameters $\phi^{w}$ to predict the anisotropic importance weight of edge feature $\hat{e}_{ij}$ with respect to node $v_{i}$ . These weights are then normalized across the neighborhood of $v_{i}$ using a softmax function:

$$
w _ {i j} = \phi^ {w} (\mathbf {e} _ {i j}, \mathbf {v} _ {i}, \mathbf {v} _ {j}), \alpha_ {i j} = \frac {\exp (w _ {i j})}{\sum_ {k \in \mathcal {N} _ {i}} \exp (w _ {i k})}. \tag {3}
$$

The normalized coefficients are used to compute a linear combination of the corresponding edge features. This linear combination serves as the final input for the node update function $\phi^{v}$ given node feature $v_{i}$ :

$$
\hat {\mathbf {v}} _ {i} = \phi^ {v} \left(\mathbf {v} _ {i}, \sum_ {v _ {j} \in \mathcal {N} _ {v _ {i}}} \alpha_ {i j} \hat {\mathbf {e}} _ {i j}\right). \tag {4}
$$

The proposed AMP layer enables the implicit assignment of varying contribution weights to the updated edge features within the same neighborhood. Analyzing the learned direction-specific weights in AMP further enhances interpretability. We adopt an MLP implementation for $\phi^{w}$ , while alternative designs, such as graph attention (Veličković et al., 2018) or cross-attention (Vaswani et al., 2017), are also feasible. A detailed comparison is provided in Appendix C.2.

# 3.2. Differentiable Multi-Scale Graph Construction

With the AMP layer functioning within each graph level, local dependencies are effectively propagated throughout the high-resolution graphs, guiding the selection of nodes to be discarded in the next hierarchy for improved long-range modeling. We now delve into the details of the differentiable node selection method (DiffSELECT) for hierarchical graph construction.

In the DiffSELECT operation, we train the node update module $\phi^{v}$ based on anisotropic aggregated edge features to produce a 2-dimensional probability vector $\pi_{i}^{l}$ for each node $v_{i}$ . This vector $\pi_{i}^{l} = (\pi_{i,0}^{l}, \pi_{i,1}^{l})$ represents the probabilities of discarding or retaining node $v_{i}$ in the next-level coarser graph $G_{l+1}$ . We rewrite Eq. (4) as follows:

$$
\hat {\mathbf {v}} _ {i} ^ {l}, \boldsymbol {\pi} _ {i} ^ {l} = \phi^ {v} \left(\mathbf {v} _ {i} ^ {l}, \sum_ {v _ {j} \in \mathcal {N} _ {v _ {i}}} \alpha_ {i j} ^ {l} \hat {\mathbf {e}} _ {i j} ^ {l}\right). \tag {5}
$$

In the next step, we apply Gumbel-Softmax sampling (Jang et al., 2017) independently to each node, using the log-probabilities $(\log \pi_{i,0}^{l},\log \pi_{i,1}^{l})$ as logits. This produces a soft one-hot vector $\mathbf{z}_i^l = (z_{i,0}^l,z_{i,1}^l)$ for each node:

$$
\begin{array}{l} z _ {i, k} ^ {l} = \text { Gumbel - Softmax } \left(\log \pi_ {i, 0} ^ {l}, \log \pi_ {i, 1} ^ {l}\right) \\ = \frac {\exp \left((\log \pi_ {i , k} ^ {l} + g _ {i , k} ^ {l}) / \tau\right)}{\sum_ {k ^ {\prime} = 0} ^ {1} \exp \left((\log \pi_ {i , k ^ {\prime}} ^ {l} + g _ {i , k ^ {\prime}} ^ {l}) / \tau\right)}, \tag {6} \\ \end{array}
$$

where $g_{i,k}^{l}$ is Gumbel noise sampled independently for each node, and $\tau$ is the temperature parameter controlling the smoothness of the sampling. In this way, the node set $V_{l+1}$ is adaptively constructed based on node features from the finer graph level. The straight-through Gumbel-Softmax estimator provides a differentiable approximation to hard sampling, thereby facilitating end-to-end training. We implement the Gumbel-Softmax with temperature annealing to stabilize training, initially encouraging the exploration of hierarchies and gradually refining the selection process.

The edges $\mathcal{E}_{l + 1}$ in the coarser graph $\mathcal{G}_{l + 1}$ are constructed by connecting the selected nodes using the original graph's edges $\mathcal{E}_l$ . However, this process may result in disconnected partitions (see Appendix Figure 6). To address this issue, we enhance the connectivity in $\mathcal{E}_{l + 1}$ by incorporating the $K$ -hop edges during edge selection, defined as follows:

$$
\begin{array}{l} \widetilde {\mathcal {E}} _ {l} ^ {(K)} = \mathcal {E} _ {l} \cup \left\{e _ {i j} \mid \exists v _ {k _ {1}}, v _ {k _ {2}}, \dots , v _ {k _ {K - 1}} \in \mathcal {V} _ {l} \right. \\ \left. \text { s.t. } e _ {i, k _ {1}}, e _ {k _ {1}, k _ {2}}, \dots , e _ {k _ {K - 1}, j} \in \mathcal {E} _ {l} \right\}. \tag {7} \\ \end{array}
$$

In essence, $e_{ij} \in \widetilde{E}_{l}^{K}$ if there exists a sequence of intermediate nodes $\{v_{k_{1}}, v_{k_{2}}, \ldots, v_{k_{K-1}}\}$ consecutively connected by edges in $E_{l}$ or $e_{ij} \in E_{l}$ . The edges in $E_{l+1}$ are defined as:

$$
\mathcal {E} _ {l + 1} = \left\{e _ {i j} \mid \exists v _ {i}, v _ {j} \in \mathcal {V} _ {l + 1} \text {   s.t.   } e _ {i j} \in \widetilde {\mathcal {E}} _ {l} ^ {(K)} \right\}. \tag {8}
$$

$E_{l+1}$ consists of edges from the enhanced edge set $\widetilde{\mathcal{E}}_{l}^{(K)}$ that connect nodes in $V_{l+1}$ . As K increases, nodes in $\widetilde{\mathcal{E}}_{l}^{(K)}$ can be connected through additional intermediate nodes, thereby improving long-range connectivity. In practice, the most effective value of K is found to be 2. We include further discussions in Appendix C.3.

The graph construction process is fully differentiable, allowing for seamless integration into differentiable physical simulators. By flexibly adapting graph hierarchies based on simulation states, it paves the way for more accurate predictions of the spatiotemporal patterns in complex systems.

# 3.3. Inter-Level Feature Propagation with AMP

During the downsampling process from $\mathcal{G}_l$ to the generated coarser graph $\mathcal{G}_{l + 1}$ , as illustrated in Figure 1, the REDUCE operation aggregates information to each node in $\mathcal{V}_{l + 1}$ from its corresponding neighbors in $\mathcal{V}_l$ . Conversely, the EXPAND operation unpools the reduced graph back to a finer resolution, delivering the information of the pooled nodes to their neighbors at the finer level.

Prior work employed non-parametric aggregation in inter-level propagation, convolving features based on the normalized node degree. It simplifies intricate relationships between nodes and neglects the directional aspects of information flow. To address this, EvoMesh is designed to learn inter-level aggregation weights that are both data-specific and time-varying. Specifically, the importance weight $\alpha_{ij}^{l}$ computed by the AMP layer inherently captures the relevance of node $v_{j}$ 's features to node $v_{i}$ at the graph level l. These weights can be directly reused for the REDUCE and EXPAND operations in the downsampling and upsampling processes. We provide details of these operations as follows:

- REDUCE: Let $v_i$ be the node at the coarser graph level. The downsampling process aggregates the information of the current neighbors $\mathcal{N}_i$ by reusing the weight $\alpha_{ij}^l$ : $\mathbf{v}_i^{l+1} \leftarrow \text{REDUCE}(\{\mathbf{v}_j^l, \alpha_{ij}^l\}_{j \in \mathcal{N}_i}) := \sum_{j \in \mathcal{N}_i} \alpha_{ij}^l \mathbf{v}_j^l$ .   
- EXPAND: We first unpool the node features from the coarser graph $\mathcal{G}_{l+1}$ back to the finer level $\mathcal{G}_l$ . To achieve

Table 2. Quantitative comparison of the one-step and long-term prediction errors. We report the mean results over 3 random seeds, with corresponding standard deviations detailed in Appendix C.7. Promotion denotes the improvement over the second-best model. 

<table><tr><td rowspan="2">Model</td><td colspan="4">RMSE-1 ( $\times 10^{-2}$ )</td><td colspan="4">RMSE-All ( $\times 10^{-2}$ )</td></tr><tr><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td></tr><tr><td>MGN (2021)</td><td>0.3046</td><td>77.38</td><td> $\underline{0.3459}$ </td><td>0.0579</td><td>59.78</td><td>2816</td><td>115.3</td><td>3.982</td></tr><tr><td>Lino et al. (2022)</td><td>3.9352</td><td>85.66</td><td>0.9993</td><td> $\underline{0.0291}$ </td><td>27.60</td><td> $\underline{2080}$ </td><td>118.2</td><td>2.090</td></tr><tr><td>BSMS-GNN (2023)</td><td>0.2263</td><td>71.69</td><td>0.5080</td><td>0.0632</td><td> $\underline{16.98}$ </td><td>2493</td><td>168.1</td><td> $\underline{1.811}$ </td></tr><tr><td>Eagle (2023)</td><td> $\underline{0.1733}$ </td><td>51.55</td><td>0.3805</td><td>0.0392</td><td>20.05</td><td>2344</td><td>127.7</td><td>7.797</td></tr><tr><td>HCMT (2024)</td><td>0.9190</td><td> $\underline{48.62}$ </td><td>0.4013</td><td>0.0295</td><td>23.59</td><td>3238</td><td> $\underline{90.32}$ </td><td>2.468</td></tr><tr><td>EvoMesh</td><td>0.1568</td><td> $\underline{41.41}$ </td><td>0.3049</td><td>0.0282</td><td>6.571</td><td>2002</td><td> $\underline{76.16}$ </td><td>1.296</td></tr><tr><td>Promotion</td><td>9.53%</td><td>14.8%</td><td>11.9%</td><td>3.10%</td><td>61.3%</td><td>3.75%</td><td>15.7%</td><td>28.5%</td></tr></table>

this, we record the nodes selected during the downsampling process and use this information to place the nodes back in their original positions in the graph. Then, we reuse the previously computed importance weights $\alpha_{ij}^{l}$ to assign weighted features from $G_{l+1}$ back to $G_{l}$ . The EXPAND operation is formally defined as: $\tilde{\mathbf{v}}_{i}^{l} \leftarrow \text{EXPAND}(\{\mathbf{v}_{j}^{l+1}, \alpha_{ij}^{l}\}_{j \in \mathcal{N}_{i}}) := \sum_{j \in \mathcal{N}_{i}} \mathbf{v}_{j}^{l+1} \alpha_{ij}^{l}$ .

\- FeatureMixing: While the EXPAND operation up-samples coarser-level features of $G_{l+1}$ to match the resolution of the current level $G_l$ , naively upsampled features may suffer from artifacts or misalignment. To mitigate this, we introduce FeatureMixing to refine and fuse coarse-level features with intra-level information at the current resolution. Specifically, we apply an additional anisotropic message passing step to the upsampled features $\tilde{\mathbf{v}}_i^l$ and then integrate these features with the original intra-level features $\mathbf{v}_l$ in $\mathcal{G}_l$ (prior to downsampling) using a skip connection: $\bar{\mathbf{v}}_i^l \leftarrow \text{FeatureMixing}(\tilde{\mathbf{v}}_i^l, \mathbf{v}_i^l, \{\mathbf{e}_{ij}^l\}_{j \in \mathcal{N}_i}) := \mathbf{v}_i^l + \text{AMP}(\tilde{\mathbf{v}}_i^l, \{\tilde{\mathbf{v}}_j^l\}_{j \in \mathcal{N}_i}, \{\mathbf{e}_{ij}^l\}_{j \in \mathcal{N}_i})$ .

# 3.4. Implementation Details

We train EvoMesh using the one-step supervision that measures the $L_{2}$ loss between the ground truth and the next-step predictions. We include detailed descriptions of the implementation of encoder, decoder, node update function and edge update function in Appendix B.

# 4. Experiments

In this section, we present the key results of the proposed method. Additional analyses can be found in Appendix C.

# 4.1. Experimental Setup

We evaluate EvoMesh on five mesh-based benchmarks from previous work (Pfaff et al., 2021; Cao et al., 2023; Wu et al., 2023; Narain et al., 2012). For detailed descriptions, including the input physical quantities, please refer to Appendix A.

- CylinderFlow: Simulation of incompressible flow around a cylinder based on 2D Eulerian meshes.   
- Airfoil: Aerodynamic simulation around airfoil cross-sections based on 2D Eulerian meshes.   
- FlyingFlag: Simulation of flag dynamics in the wind based on Lagrangian meshes with fixed topology.   
- DeformingPlate: Deformation of hyper-elastic plates based on Lagrange tetrahedral meshes.   
- FoldingPaper: Deformation of paper sheets with evolving meshes driven by varying forces at the four corners.

We mainly compare EvoMesh with the following methods:

- MGN (Pfaff et al., 2021), which performs multiple times of message passing on the original graph.   
- Lino et al. (Lino et al., 2022), which also trains MPNNs on manually-set multi-scale mesh graphs.   
- BSMS-GNN (Cao et al., 2023), which generates static hierarchies using bi-stride pooling and performs message passing on predefined meshes.   
- Eagle (Janny et al., 2023), which constructs a two-scale hierarchy using precomputed geometric clustering and performs message passing at both levels.   
- HCMT (Yu et al., 2024), which generates static hierarchies by applying Delaunay triangulation to the bi-stride pooled nodes, and enables directed feature propagation with the attention mechanism.

All models are trained using the Adam optimizer for $1M$ steps, with an exponential learning rate decay from $10^{-4}$ to $10^{-6}$ over the first $500K$ steps. We provide further details on the architecture and hyperparameters of the compared models in Appendix D.

# 4.2. Main Results

Standard benchmarks. Table 2 presents the root mean squared error (RMSE) of one-step prediction (RMSE-1) and

![](images/99b08ddeb028757e62d9db4bcea4684392606c651fcf08114ba87bb920e2be87.jpg)

Figure 2. Prediction showcases over 400 future steps on CylinderFlow. From the displayed error maps, it is evident that EvoMesh effectively captures long-term dynamics, providing predictions that closely align with the ground truth.   
![](images/04bba120c5e49218a72be4bb557923e69afe667d0066fc7d464faa44ad1fd569.jpg)

<details>
<summary>natural_image</summary>

Abstract thermal or fluid simulation visualization with red-orange gradient and a central blue circular feature, labeled 't=100' in top-left corner (no other text or symbols)
</details>

![](images/38f43562eb1697751e762e4c31e7629489e8846bf7ce1361cba1a9c67056fcb4.jpg)

<details>
<summary>natural_image</summary>

Thermal or fluid simulation image showing a blue circular object with red-orange gradient background, labeled 't=400' in top-left corner (no other text or symbols)
</details>

![](images/9d9f9a71e07fb011290bfabe939e9a2ce962c019f759ce42cd91193d07946ffb.jpg)

<details>
<summary>natural_image</summary>

Abstract network visualization with interconnected nodes and a central red-orange gradient (no text or symbols)
</details>

![](images/82f28ce4b3213d5bc36b62e45db6b256dcceb47d811d436b44d3e77e3b8acca0.jpg)

<details>
<summary>natural_image</summary>

Abstract network diagram with interconnected nodes and a central circular element, no text or symbols present
</details>

Figure 3. A demonstration of how the learned hierarchies adapt to evolving physical dynamics. Top: the velocity field from the true data. Bottom: the temporal difference of the velocity fields between adjacent time steps alongside the constructed coarser-level mesh graph ( $G_{l=4}$ ). The highlighted areas demonstrate a notable experimental phenomenon: the mesh dynamically evolves with the data context and aligns with the critical areas of change in the data.

long-term rollouts for 100–600 future time steps (RMSE-all). EvoMesh consistently outperforms the compared models across all benchmarks. This demonstrates the effectiveness of building context-aware, time-evolving hierarchies with learnable, directionally non-uniform feature propagation both within and across graph levels. Figure 2 presents long-term predictions on CylinderFlow, based solely on the system's initial conditions at the first step. As we can see, EvoMesh captures the complex, time-varying fluid flow around the cylinder obstacle more successfully, with its predictions closely matching the ground truth evolution. More results are shown in Appendix C.9.

Can the learned hierarchies adapt to evolving data dynamics? In Figure 3, we visualize the time-evolving hierarchies constructed by EvoMesh at different time steps, where coarser-level nodes tend to concentrate in regions highlighted by the temporal differences in the true data. We have two observations here: First, the constructed hierarchy evolves as the data context changes. Second, the time-evolving graph structures align with the high-intensity regions, either in the velocity fields (top) or in their temporal variations (bottom). These results highlight the effectiveness of our approach in capturing significant dynamic patterns.

Paper simulation with changing meshes. We evaluate EvoMesh in a more challenging setting with time-varying meshes for paper folding simulation, generated using the ARCSim solver (Narain et al., 2012; Wu et al., 2023), and compare EvoMesh with MGN (Pfaff et al., 2021). Lino et al., BSMS-GNN, Eagle, and HCMT rely on pre-computed hierarchies during preprocessing, which limits their applicability in scenarios with dynamically changing mesh topologies. Therefore, we do not include them in this evaluation.

Table 3. Simulation results of 2D paper folding with time-varying input meshes. We here compare EvoMesh with MGN, as BSMS and HCMT require pre-computed hierarchies, which are unsuitable for scenarios involving continuously changing meshes. 

<table><tr><td>Model</td><td>RMSE-1 ( $\times 10^{-2}$ )</td><td>RMSE-All ( $\times 10^{-2}$ )</td></tr><tr><td>MGN (2021)</td><td>0.0618</td><td>24.08</td></tr><tr><td>EvoMesh</td><td>0.0544</td><td>7.412</td></tr><tr><td>Promotion</td><td>12.0%</td><td>69.2%</td></tr></table>

We assess the models using ground-truth remeshing nodes provided by the ARCSim Adaptive Remeshing component, following the setup from (Pfaff et al., 2021). As shown in Table 3, EvoMesh achieves superior short-term and long-term accuracy compared to MGN, indicating that the time-evolving graph hierarchies in our approach can better fit physical systems with significant geometric variations, as represented by the time-varying input mesh structures.

Model stability under variable graph structures. Due to the stochasticity of Gumbel-Softmax sampling in DiffSELECT, we evaluate the stability of trained EvoMesh by conducting three independent runs on the test set. The mean and standard deviations of the prediction errors reveal minimal discrepancies across different runs, as shown in Table 11 in Appendix C.6. These findings demonstrate that once trained, EvoMesh generates consistent graph hierarchies based on the same inputs.

# 4.3. Ablation Studies

EvoMesh has three key components: (i) evolving graph hierarchy, (ii) anisotropic intra-level propagation, (iii) learnable inter-level propagation. To evaluate the contribution of each component, we implement several ablated variants of EvoMesh, including: Static(Bi-stride)-Anisotropic-Unlearnable (M1), Static(Bi-stride)-Anisotropic-Learnable (M2), Uniform-Anisotropic-Learnable (M3), and Dynamic-Anisotropic-Unlearnable (M4), and compare them against the BSMS-GNN baseline, which uses static hierarchies, isotropic intra-level summation, and unlearnable inter-level propagation. Both M1 and M2 adopt the same static bi-stride hierarchy via preprocessing as BSMS-GNN. M3 constructs the hierarchy via uniform node sampling in each hierarchy, while M4 applies dynamic hierarchy construction without learnable inter-level updates.

Figure 4 demonstrates the effectiveness of direction-aware message propagation at both intra- and inter-levels, as well as the benefit of learning dynamic graph hierarchies. Comparing M1 with BSMS-GNN shows that integrating AMP improves performance even under a static hierarchy. Furthermore, the comparison between EvoMesh and M4, as well as between M1 with M2, highlights the importance of learnable inter-level propagation in capturing hierarchical signal flow. In addition, although M3 benefits from learnable propagation, its use of uniform node sampling results in suboptimal performance, suggesting the necessity of adaptive and data-aware hierarchy construction.

Furthermore, in Figure 5, we visualize the variance of predicted anisotropic edge weights and compare it with areas where physical quantities present substantial variations over time. The results reveal a strong correlation between the anisotropic learning mechanism and the rapidly changing dynamics of the physical system.

# 4.4. Generalization Analyses

Generalization to out-of-distribution mesh resolutions. Nearly all existing machine learning models for mesh-based simulations are not resolution-free and may fail when evaluated on unseen mesh resolutions. We assess the generalization performance of EvoMesh by training it on low-resolution meshes and testing it on high-resolution meshes. The average number of nodes in the test data is twice that of the training data, and the number of edges is three times greater. Table 4 reports one-step and 50-step rollout errors on out-of-distribution (OOD) mesh resolutions. EvoMesh shows strong generalization on the CylinderFlow and Airfoil datasets, highlighting its zero-shot capability to handle refined mesh structures. This performance gain is largely attributed to EvoMesh's ability to construct hierarchical graphs adaptively, allowing it to scale effectively with increased resolution. On the other hand, MGN achieves lower errors on the FlyingFlag and DeformingPlate datasets, indicating that mesh-based architectures remain effective for systems with more regular structures and smoother deformation dynamics. While our method does not yet achieve full generalization across arbitrary resolutions, truly resolution-free modeling remains an open challenge that calls for more advanced architectural design. Nevertheless, this holds significant value in practical applications and has the potential to greatly reduce the time overhead of numerical simulation processes for preparing the large-scale mesh data required for model training.

Generalization to physical variations. We evaluate EvoMesh under strong distribution shifts in the input physical quantities. Table 5 presents data statistics and the RMSE results on the CylinderFlow and Airfoil datasets. EvoMesh consistently outperforms the compared models in both short-term and long-term simulations. This advantage mainly comes from its ability to learn evolving hierarchies and model intra-level and inter-level interactions based on physical context. When the fluid dynamics in the test set become more complex—characterized by increased variance in the velocity field over time—the dynamics patterns propagate more rapidly in space. EvoMesh adaptively constructs graph hierarchies and more effectively captures long-range node interactions in response to evolving physical contexts.

![](images/e6a7bcd22fbf74b72c485dbc4969bb00c86b0e901d8e07653106ba7a22a11229.jpg)

Figure 4. Ablation studies. We provide analyses of time-evolving hierarchies, anisotropic intra-level propagation, and learnable inter-level feature propagation. The red dashed lines represent results from BSMS-GNN (Cao et al., 2023). Lower values indicate better performance.   
![](images/cf733cdeba3583d6801811adedcbb3cb9bc03bfc310f32537da7238afd6ec51d.jpg)

<details>
<summary>text_image</summary>

Importance Weight Variance
</details>

![](images/420c5e4f841fd8a90072decb9f9aa77db7f9f7880b8b09d9408e225eb67b32bd.jpg)

<details>
<summary>natural_image</summary>

Finite element mesh simulation with a highlighted circular region containing red and blue dots (no text or symbols)
</details>

![](images/f2505b06a5e7e78364b6c4b7f5b97b289fe3bfaaf213ff5d61ba8d05f40bc5a9.jpg)

<details>
<summary>text_image</summary>

Physical Quantity Variance
</details>

![](images/df421ff0f292b364aa97ed0a721f5cdc2d81a1ff42db496673103e3f5512ea06.jpg)

<details>
<summary>natural_image</summary>

Finite element mesh simulation with a highlighted rectangular region containing red dots and blue triangular boundaries (no text or symbols)
</details>

Figure 5. A demonstration of how the predicted anisotropic edge weights respond to dramatic changes in physical quantities over time. Top: Visualizations of the variance in the generated anisotropic weights, calculated on adjacent edges. Bottom: Variance in physical quantities over time. The strong correlation between them highlights the AMP's ability to detect significant patterns in data.

# 5. Related Work

# Learning-based and GNN-based physical simulation.

Recent advances have demonstrated that learning-based approaches can efficiently tackle complex and high-dimensional physical simulation tasks, including fluid dynamics (Zhu et al., 2024), structural analysis (Kavvas et al., 2018; Thai, 2022), and climate modeling (Kurth et al., 2018; Rasp et al., 2018; Rolnick et al., 2022; Lam et al., 2023). These methods can be broadly categorized by their data representations: partial differential equations (Raissi et al., 2017; 2019; Lu et al., 2019; Li et al., 2021; Wang et al., 2021), particle-based systems (Li et al., 2019; Sanchez-Gonzalez et al., 2020; Ummenhofer et al., 2020; Prantl et al., 2022), and mesh-based systems (Pfaff et al., 2021; Lino et al., 2022; Fortunato et al., 2022; Cao et al., 2023). The rapid inference and differentiable nature of these models have facilitated a range of downstream applications, such as inverse design (Wang & Zhang, 2021; Goodrich et al., 2021; Allen et al., 2022; Janny et al., 2023). In particular, Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling physical systems across various domains, including articulated bodies (Sanchez-Gonzalez et al., 2018), soft-body deformation and fluids (Li et al., 2019; Mrowca et al., 2018; Sanchez-Gonzalez et al., 2020; Rubanova et al., 2022; Wu et al., 2023), rigid body dynamics (Battaglia et al., 2016; Li et al., 2019; Mrowca et al., 2018; Bear et al., 2021; Rubanova et al., 2022), and aerodynamics (Belbute-Peres et al., 2020; Hines & Bekemeyer, 2023; Pfaff et al., 2021; Fortunato et al., 2022; Cao et al., 2023). Among these, MeshGraphNets (Pfaff et al., 2021) serves as a representative, introducing a general scheme for representing meshes as graphs and learning mesh-based dynamics, inspiring subsequent works that focus on improving modeling capacity and computational efficiency.

Hierarchical GNNs for physical simulation. Hierarchical GNNs leverage multi-scale graph structures (Lino et al., 2022; Han et al., 2022; Fortunato et al., 2022; Allen et al., 2023; Janny et al., 2023; Cao et al., 2023; Yu et al., 2024) to reduce computational overhead by operating on coarser representations and to facilitate long-range information propagation. For example, GMR-Transformer-GMUS (Han et al., 2022) employs uniform sampling for pooling, while Eagle (Janny et al., 2023) adopts a two-scale message passing scheme with geometric clustering by preprocessing the

Table 4. One-step (RMSE-1) and 50-step (RMSE-50) rollout prediction errors on out-of-distribution (OOD) mesh resolutions. The average number of nodes in the test data is twice that of the training data, and the number of edges is three times greater. 

<table><tr><td rowspan="2">Model</td><td colspan="4">RMSE-1 ( $\times 10^{-2}$ )</td><td colspan="4">RMSE-50 ( $\times 10^{-2}$ )</td></tr><tr><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td></tr><tr><td>MGN (2021)</td><td>1.0596</td><td>169.6</td><td>0.4215</td><td>0.0359</td><td>7.833</td><td>1829</td><td>55.96</td><td>0.2467</td></tr><tr><td>Lino et al. (2022)</td><td>25.893</td><td>144.4</td><td>0.8906</td><td>0.0475</td><td>65.21</td><td>1391</td><td>93.68</td><td>3.9845</td></tr><tr><td>BSMS-GNN (2023)</td><td>0.9177</td><td>202.3</td><td>0.6486</td><td>0.0474</td><td>2.097</td><td>1677</td><td>59.18</td><td>0.2554</td></tr><tr><td>HCMT (2024)</td><td>1.3864</td><td>205.5</td><td>1.0634</td><td>0.0354</td><td>7.541</td><td>2569</td><td>86.87</td><td>0.2957</td></tr><tr><td>EvoMesh</td><td>0.4855</td><td>126.7</td><td>0.5536</td><td>0.0368</td><td>1.077</td><td>812.5</td><td>58.29</td><td>0.3780</td></tr></table>

Table 5. Generalization results across domains with various scales of input velocities. The domain gap is presented by the variance and norm of data in training/test splits. Increase denotes the relative increase of the test data compared to the training data. 

<table><tr><td rowspan="2">Split</td><td colspan="2">Cylinder</td><td colspan="2">Airfoil</td></tr><tr><td>Var</td><td>Norm</td><td>Var</td><td>Norm</td></tr><tr><td>Train</td><td>7.92</td><td>579.6</td><td>288.3</td><td>173.4</td></tr><tr><td>Test</td><td>13.43</td><td>826.3</td><td>827.4</td><td>180.6</td></tr><tr><td>Increase</td><td>64.5%</td><td>42.5%</td><td>186.9%</td><td>4.20%</td></tr><tr><td rowspan="2">Model</td><td colspan="2">Cylinder</td><td colspan="2">Airfoil</td></tr><tr><td>RMSE-1</td><td>RMSE-All</td><td>RMSE-1</td><td>RMSE-All</td></tr><tr><td>MGN</td><td> $4.99 \times 10^{-3}$ </td><td>1.020</td><td>1.193</td><td>88.23</td></tr><tr><td>Lino et al.</td><td> $5.60 \times 10^{-3}$ </td><td>1.415</td><td>3.226</td><td>410.5</td></tr><tr><td>BSMS-GNN</td><td> $2.58 \times 10^{-3}$ </td><td>0.251</td><td>1.035</td><td>30.32</td></tr><tr><td>Eagle</td><td> $2.49 \times 10^{-3}$ </td><td>0.273</td><td>0.931</td><td>51.83</td></tr><tr><td>HCMT</td><td> $7.35 \times 10^{-3}$ </td><td>1.047</td><td>1.697</td><td>63.18</td></tr><tr><td>EvoMesh</td><td> $2.14 \times 10^{-3}$ </td><td>0.091</td><td>0.665</td><td>22.57</td></tr><tr><td>Promotion</td><td>14.1%</td><td>63.7%</td><td>28.6%</td><td>25.6%</td></tr></table>

mesh structure. LayersNet (Shao et al., 2023) introduces a static, patch-based hierarchy for garment animation, simplifying interaction modeling via particle patches. More recent works (Lino et al., 2022; Cao et al., 2023; Yu et al., 2024; Garnier et al., 2024; Hy & Kondor, 2023) integrate hierarchical GNNs with U-Net architectures (Ronneberger et al., 2015), using static multi-level structures for message passing. For instance, Lino et al. (2022) relies on manually defined grid resolutions, BSMS-GNN (Cao et al., 2023) proposes a bi-stride pooling strategy with enhanced edge connectivity, and HCMT (Yu et al., 2024) further refines the hierarchy using Delaunay triangulation and replaces message passing with graph attention. However, these methods typically employ precomputed, static hierarchies and uniform feature aggregation, limiting their adaptability to dynamic physical environments. In contrast, our approach constructs context-aware, temporally evolving graph hierarchies with learnable anisotropic feature propagation, enabling robust adaptation to diverse initial conditions and rapidly changing dynamics.

Differentiable graph pooling. A variety of differentiable and learnable graph pooling methods have been proposed to enable end-to-end training of hierarchical graph representations, such as DiffPool (Ying et al., 2018), TopKPool (Gao & Ji, 2019), SAGPool (Lee et al., 2019), and ASAPooling (Ranjan et al., 2020). These methods construct hierarchical representations of graphs by either learning soft cluster assignments or selecting top-ranked nodes based on learned importance scores. However, most of these methods are primarily designed for static graphs and focus on global graph-level tasks, where unpooling or reconstruction of the original graph structure is not required. In contrast, mesh-based physical simulation demands fine-grained local information and temporally-evolving hierarchies for accurate modeling of physical dynamics. Our approach differs by constructing context-aware, time-varying graph hierarchies tailored for mesh-based simulation, enabling flexible integration of global and local features and supporting dynamic adaptation to changing physical conditions.

# 6. Conclusions and Limitations

In this paper, we introduced EvoMesh, a neural network that significantly advances the state-of-the-art in mesh-based simulation. Our key innovation is adaptively creating the temporally-evolving and context-aware graph structures of hierarchical GNNs through a differentiable node selection process. To this end, we proposed an anisotropic message passing mechanism to enhance the propagation of long-term dependencies between distant nodes, aligning with the directed nature of significant dynamic patterns. Extensive experiments show that EvoMesh outperforms existing models, especially those with fixed graph hierarchies, in both short-term and long-term predictions.

A potential limitation of this work is the need to improve the interpretability of the learned hierarchies. Additionally, we would consider incorporating physical priors into EvoMesh to enhance the model's robustness and generalizability, particularly in resolution-free problem settings, which have been less explored in existing mesh-based approaches.

# Acknowledgments

This work was supported by the National Natural Science Foundation of China (62250062), the Smart Grid National Science and Technology Major Project (2024ZD0801200),

the Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), and the Fundamental Research Funds for the Central Universities.

# Impact Statement

In this work, we adhere to the highest ethical standards throughout all stages of research. No human subjects were involved, and no personal data was used, ensuring full compliance with privacy and security protocols. All datasets employed are publicly available, minimizing concerns regarding sensitive information exposure.

We recognize that while physical simulation models can drive significant advancements, they also have the potential for misuse if applied irresponsibly. Thus, we emphasize the importance of thoughtful consideration regarding the context and ethical implications when deploying these models, particularly in high-stakes applications such as healthcare, engineering, and environmental management.

# References

Allen, K. R., Lopez-Guavara, T., Stachenfeld, K., Sanchez-Gonzalez, A., Battaglia, P., Hamrick, J. B., and Pfaff, T. Inverse design for fluid-structure interactions using graph network simulators. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=HaZuqj0Gvp2.   
Allen, K. R., Guevara, T. L., Rubanova, Y., Stachenfeld, K., Sanchez-Gonzalez, A., Battaglia, P., and Pfaff, T. Graph network simulators can learn discontinuous, rigid contact dynamics. In Conference on Robot Learning, pp. 1157–1167. PMLR, 2023.   
Battaglia, P., Pascanu, R., Lai, M., Jimenez Rezende, D., et al. Interaction networks for learning about objects, relations and physics. Advances in neural information processing systems, 29, 2016.   
Bear, D. M., Wang, E., Mrowca, D., Binder, F. J., Tung, H.-Y. F., Pramod, R., Holdaway, C., Tao, S., Smith, K., Sun, F.-Y., et al. Physion: Evaluating physical prediction from vision in humans and machines. In NeurIPS Datasets and Benchmarks, 2021.   
Belbute-Peres, F. D. A., Economon, T., and Kolter, Z. Combining differentiable pde solvers and graph neural networks for fluid flow prediction. In international conference on machine learning, pp. 2402–2411. PMLR, 2020.   
Cao, Y., Chai, M., Li, M., and Jiang, C. Efficient learning of mesh-based physical simulation with bi-stride multi-scale graph neural network. In International Conference on Machine Learning, pp. 3541–3558. PMLR, 2023.

Fortunato, M., Pfaff, T., Wirnsberger, P., Pritzel, A., and Battaglia, P. Multiscale meshgraphnets. In ICML 2022 2nd AI for Science Workshop, 2022. URL https://openreview.net/forum?id=G3TRIsmMhhf.

Gao, H. and Ji, S. Graph u-nets. In international conference on machine learning, pp. 2083–2092. PMLR, 2019.

Garnier, P., Viquerat, J., and Hachem, E. Multi-grid graph neural networks with self-attention for computational mechanics. arXiv preprint arXiv:2409.11899, 2024.

Goodrich, C. P., King, E. M., Schoenholz, S. S., Cubuk, E. D., and Brenner, M. P. Designing self-assembling kinetics with differentiable statistical physics models. Proceedings of the National Academy of Sciences, 118(10): e2024083118, 2021.

Han, X., Gao, H., Pfaff, T., Wang, J.-X., and Liu, L. Predicting physics in mesh-reduced space with temporal attention. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=XctLdNfCmP.

Hines, D. and Bekemeyer, P. Graph neural networks for the prediction of aircraft surface pressure distributions. Aerospace Science and Technology, 137:108268, 2023.

Hy, T. S. and Kondor, R. Multiresolution equivariant graph variational autoencoder. Machine Learning: Science and Technology, 4(1):015031, 2023.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=rkE3y85ee.

Janny, S., Bénéteau, A., Nadri, M., Digne, J., Thome, N., and Wolf, C. EAGLE: Large-scale learning of turbulent fluid dynamics with mesh transformers. In ICLR, 2023. URL https://openreview.net/forum?id=mfIX4QpsARJ.

Kavvas, E. S., Catoiu, E., Mih, N., Yurkovich, J. T., Seif, Y., Dillon, N., Heckmann, D., Anand, A., Yang, L., Nizet, V., et al. Machine learning and structural analysis of mycobacterium tuberculosis pan-genome identifies genetic signatures of antibiotic resistance. Nature communications, 9(1):4306, 2018.

Kurth, T., Treichler, S., Romero, J., Mudigonda, M., Luehr, N., Phillips, E., Mahesh, A., Matheson, M., Deslippe, J., Fatica, M., et al. Exascale deep learning for climate analytics. In SC18: International conference for high performance computing, networking, storage and analysis, pp. 649–660. IEEE, 2018.

Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421, 2023. doi: 10.1126/science.adi2336. URL https://www.science.org/doi/abs/10.1126/science.adi2336.   
Lee, J., Lee, I., and Kang, J. Self-attention graph pooling. In International conference on machine learning, pp. 3734–3743. PMLR, 2019.   
Li, Y., Wu, J., Tedrake, R., Tenenbaum, J. B., and Torralba, A. Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids. In ICLR, 2019.   
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. In ICLR, 2021.   
Lino, M., Fotiadis, S., Bharath, A. A., and Cantwell, C. D. Multi-scale rotation-equivariant graph neural networks for unsteady eulerian fluid dynamics. Physics of Fluids, 34(8), 2022.   
Lu, L., Jin, P., and Karniadakis, G. E. Deeponet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. arXiv preprint arXiv:1910.03193, 2019.   
Mrowca, D., Zhuang, C., Wang, E., Haber, N., Fei-Fei, L. F., Tenenbaum, J., and Yamins, D. L. Flexible neural representation for physics prediction. Advances in neural information processing systems, 31, 2018.   
Narain, R., Samii, A., and O'brien, J. F. Adaptive anisotropic remeshing for cloth simulation. ACM transactions on graphics (TOG), 31(6):1–10, 2012.   
Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., and Battaglia, P. Learning mesh-based simulation with graph networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=roNqYL0\_XP.   
Prantl, L., Ummenhofer, B., Koltun, V., and Thuerey, N. Guaranteed conservation of momentum for learning particle-based fluid dynamics. In NeurIPS, 2022.   
Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations. arXiv preprint arXiv:1711.10561, 2017.

Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378:686–707, 2019.   
Ranjan, E., Sanyal, S., and Talukdar, P. Asap: Adaptive structure aware pooling for learning hierarchical graph representations. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 5470–5477, 2020.   
Rasp, S., Pritchard, M. S., and Gentine, P. Deep learning to represent subgrid processes in climate models. Proceedings of the National Academy of Sciences, 115(39):9684–9689, 2018.   
Rolnick, D., Donti, P. L., Kaack, L. H., Kochanski, K., Lacoste, A., Sankaran, K., Ross, A. S., Milojevic-Dupont, N., Jaques, N., Waldman-Brown, A., et al. Tackling climate change with machine learning. ACM Computing Surveys (CSUR), 55(2):1–96, 2022.   
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.   
Rubanova, Y., Sanchez-Gonzalez, A., Pfaff, T., and Battaglia, P. Constraint-based graph network simulator. In ICML, 2022.   
Sanchez-Gonzalez, A., Heess, N., Springenberg, J. T., Merel, J., Riedmiller, M., Hadsell, R., and Battaglia, P. Graph networks as learnable physics engines for inference and control. In International conference on machine learning, pp. 4470–4479. PMLR, 2018.   
Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., and Battaglia, P. Learning to simulate complex physics with graph networks. In ICML, pp. 8459–8468, 2020.   
Shao, Y., Loy, C. C., and Dai, B. Towards multi-layered 3d garments animation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14361–14370, 2023.   
Thai, H.-T. Machine learning for structural engineering: A state-of-the-art review. In Structures, volume 38, pp. 448–491. Elsevier, 2022.   
Ummenhofer, B., Prantl, L., Thuerey, N., and Koltun, V. Lagrangian fluid simulation with continuous convolutions. In ICLR, 2020.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. Graph attention networks. In International Conference on Learning Representations, 2018.   
Wang, Q. and Zhang, L. Inverse design of glass structure with deep graph neural networks. Nature communications, 12(1):5359, 2021.   
Wang, S., Wang, H., and Perdikaris, P. Learning the solution operator of parametric partial differential equations with physics-informed deeponets. Science advances, 7(40): eabi8605, 2021.   
Wu, T., Maruyama, T., Zhao, Q., Wetzstein, G., and Leskovec, J. Learning controllable adaptive simulation for multi-resolution physics. In ICLR, 2023. URL https://openreview.net/forum?id=PbfgkZ2HdbE.   
Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Philip, S. Y. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4–24, 2020.   
Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., and Leskovec, J. Hierarchical graph representation learning with differentiable pooling. Advances in neural information processing systems, 31, 2018.   
Yu, Y.-Y., Choi, J., Cho, W., Lee, K., Kim, N., Chang, K., Woo, C., Kim, I., Lee, S., Yang, J. Y., et al. Learning flexible body collision dynamics with hierarchical contact mesh transformer. In ICLR, 2024. URL https://openreview.net/forum?id=90yw2uM6J5.   
Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H. J. Graph transformer networks. Advances in neural information processing systems, 32, 2019.   
Zhu, X., Deng, H., Yuan, H., Wang, Y., and Yang, X. Latent intuitive physics: Learning to transfer hidden physics from a 3d video. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=WZu4gUGN13.

# Appendix

# A. Datasets

We employ four established datasets from MGN (Pfaff et al., 2021): CylinderFlow, Airfoil, Flag, and DeformingPlate.

- The CylinderFlow case examines the transient incompressible flow field around a fixed cylinder positioned at different locations, with varying inflow velocities.   
- The Airfoil case explores the transient compressible flow field at varying Mach numbers around the airfoil, with different angles of attack.   
- The Flag case involves a flag blowing in the wind on a fixed Lagrangian mesh.   
- The DeformingPlate case involves hyperelastic plates being compressed by moving obstacles.

The CylinderFlow, Airfoil, and Flag datasets are each split into 1,000 training sequences, 100 validation sequences, and 100 testing sequences. The DeformingPlate dataset is split into 500 training sequences, 100 validation sequences, and 100 testing sequences.

We also consider a more challenging dataset, FoldingPaper, where varying forces at the four corners deform paper with time-varying Lagrangian mesh graphs, generated using the ARCSim solver (Narain et al., 2012; Wu et al., 2023). This dataset is divided into 500 training sequences, 100 validation sequences, and 100 testing sequences.

We present statistical details of all five datasets in Table 6 and the input physical quantities in Table 7.

Table 6. Statistics of the CylinderFlow, Airfoil, Flag, DeformingPlate, and FoldingPaper datasets. 

<table><tr><td>Dataset</td><td>Average # nodes</td><td>Average # edges</td><td>Mesh type</td><td># Hierarchies</td><td># Steps</td></tr><tr><td>CylinderFlow</td><td>1886</td><td>5424</td><td>triangle, 2D</td><td>7</td><td>600</td></tr><tr><td>Airfoil</td><td>5233</td><td>15449</td><td>triangle, 2D</td><td>7</td><td>100</td></tr><tr><td>Flag</td><td>1579</td><td>9212</td><td>triangle, 2D</td><td>7</td><td>400</td></tr><tr><td>DeformingPlate</td><td>1271</td><td>4611</td><td>tetrahedron, 3D</td><td>6</td><td>400</td></tr><tr><td>FoldingPaper</td><td>110</td><td>724</td><td>triangle, 2D</td><td>3</td><td>325</td></tr></table>

Table 7. Comparisons of the edge offsets and node inputs of different physical systems. 

<table><tr><td>Dataset</td><td>Type</td><td>Edge offset  $\mathbf{e}_{ij}$ </td><td>Node Input  $\mathbf{v}_i$ </td><td>Outputs</td><td>Noise Scale</td></tr><tr><td>CylinderFlow</td><td>Eulerian</td><td> $X_{ij}, |X_{ij}|$ </td><td> $v_i, n_i$ </td><td> $\dot{v}_i$ </td><td> $v_i: 2e-2$ </td></tr><tr><td>Airfoil</td><td>Eulerian</td><td> $X_{ij}, |X_{ij}|$ </td><td> $\rho_i, v_i, n_i$ </td><td> $\dot{v}_i, \dot{\rho}_i, P_i$ </td><td> $v_i: 2e-2, \rho_i: 1e1$ </td></tr><tr><td>Flag</td><td>Lagrangian</td><td> $X_{ij}, |X_{ij}|, x_{ij}, |x_{ij}|$ </td><td> $\dot{x}_i, n_i$ </td><td> $\dot{x}_i$ </td><td> $x_i: 3e-3$ </td></tr><tr><td>DeformingPlate</td><td>Lagrangian</td><td> $X_{ij}, |X_{ij}|, x_{ij}, |x_{ij}|$ </td><td> $\dot{x}_i, n_i$ </td><td> $\dot{x}_i$ </td><td> $x_i: 3e-3$ </td></tr><tr><td>FoldingPaper</td><td>Lagrangian</td><td> $X_{ij}, |X_{ij}|, x_{ij}, |x_{ij}|$ </td><td> $\dot{x}_i, n_i$ </td><td> $\dot{x}_i$ </td><td> $x_i: 3e-3$ </td></tr></table>

# B. Model Implementation

We present model configurations of different physical systems below:

- Edge offsets. $X$ and $x$ stand for the mesh-space and world-space position. For an Eulerian system, only mesh position is used for $\mathbf{e}_{ij}$ , while for a Lagrangian system, both mesh-space and world-space positions are used. The edge offsets are directly used as low-dimensional input to the edge update function $\phi^e$ . These features are concatenated and fed into $\phi^e$ without any transformation through an MLP or other encoding processes to generate a higher-dimensional representation.   
- Input and target of the physical term of node $v_{i}$ . $v$ is the velocity, $\rho$ is the density, $P$ is the absolute pressure, and the dot $\dot{a} = a_{t+1} - a_t$ stands for temporal change for a variable $a$ . $n$ stands for the node type of $v_{i}$ . Random Gaussian noise is added to the node input features to enhance robustness during training (Pfaff et al., 2021; Sanchez-Gonzalez et al., 2020; Cao et al., 2023). All the variables involved are normalized to zero-mean and unit variance via preprocessing. The preprocessed physical term is fed to the encoder to transform it into a high-dimensional representation.

The encoder, decoder, node update function $\phi^v$ , and edge update function $\phi^e$ all utilize two-layer MLPs with ReLU activation and a hidden size of 128. Similarly, the importance weight network $\phi^w$ in AMP is implemented using a two-layer MLP.

Table 8. Qualitative results of model variants of EvoMesh and the baseline model. 

<table><tr><td rowspan="2">Model</td><td colspan="2">RMSE-1 ( $\times 10^{-2}$ )</td><td colspan="2">RMSE-All ( $\times 10^{-2}$ )</td></tr><tr><td>Cylinder</td><td>Flag</td><td>Cylinder</td><td>Flag</td></tr><tr><td>BSMS-GNN (Cao et al., 2023)</td><td>0.2263</td><td>0.5080</td><td>16.98</td><td>168.1</td></tr><tr><td>Static(Bi-stride)-Anisotropic-Unlearnable (M1)</td><td>0.1995</td><td>0.4804</td><td>9.621</td><td>121.1</td></tr><tr><td>Static(Bi-stride)-Anisotropic-Learnable (M2)</td><td>0.1695</td><td>0.4666</td><td>8.317</td><td>109.9</td></tr><tr><td>Uniform-Anisotropic-Learnable (M3)</td><td>0.2019</td><td>0.3999</td><td>9.357</td><td>145.27</td></tr><tr><td>Dynamic-Anisotropic-Unlearnable (M4)</td><td>0.1631</td><td>0.3538</td><td>7.793</td><td>82.65</td></tr><tr><td>EvoMesh</td><td>0.1568</td><td>0.3049</td><td>6.571</td><td>76.16</td></tr></table>

![](images/3eb7075199dd0814da0a0e51e730cd9ec17842920fa60162dd43fd6936d16523.jpg)

<details>
<summary>natural_image</summary>

Geometric pattern composed of interconnected triangles forming a tessellated grid (no text or symbols)
</details>

![](images/68067c966ce487309fdc5d762b992460692d70b82ae4f100e51a63456b8c54b6.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric pattern composed of interconnected triangles (no text or symbols)
</details>

![](images/26ffd75646619d35beee3dedcb89babf17370102e38281003b587b8ba053eecc.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of interlocking lines forming a spiral or spiral structure (no text or symbols)
</details>

Figure 6. Mesh visualization on Flag Dataset. Original mesh (left), sub-level graph after differentiable node selection with $K$ -hop enhancement with $K = 2$ (middle), and sub-level graph after node selection without $K$ -hop enhancement (right).

LayerNorm is applied to the MLP outputs, except for the decoder and the importance weight network. We set K = 2 for edge enhancement, which is aligned with the setting of BSMS-GNN (Cao et al., 2023). In the Gumbel-Softmax for differentiable node selection, temperature annealing decreases the temperature from 5 to 0.1 using a decay factor of $\gamma = 0.999$ , encouraging exploration of hierarchies while gradually refining their selection to ensure stability. EvoMesh is trained with Adam optimizer, using an exponential learning rate decay from $10^{-4}$ to $10^{-6}$ . All experiments are conducted using 4 Nvidia RTX 3090. We mainly build EvoMesh based on the released code of BSMS-GNN (Cao et al., 2023).

# C. Additional Results

# C.1. Ablation Study

In Sec. 4.3, we compare different variants of our EvoMesh model against the BSMS-GNN baseline, to evaluate the effectiveness of (i) dynamic hierarchy construction based on the input mesh topology and physical quantities, (ii) anisotropic intra-level feature propagation, (iii) learnable inter-level feature propagation. The variants we investigate include:

• Static(Bi-stride)-Anisotropic-Unlearnable (M1),   
• Static(Bi-stride)-Anisotropic-Learnable (M2),   
• Uniform-Anisotropic-Learnable (M3),   
• Dynamic-Anisotropic-Unlearnable (M4).

In this ablation study, we utilize a static graph hierarchy preprocessed using bi-stride pooling as described in the BSMS-GNN paper (Cao et al., 2023) for static hierarchy variants, along with a non-parametric intra-level aggregation function from previous works (Pfaff et al., 2021; Cao et al., 2023). Additionally, BSMS-GNN employs unlearnable node degree metrics to generate inter-level aggregation weights, which convolve features based on the normalized node degree for inter-level propagation. We show the quantitative RMSE values of Figure 4 in Table 8.

# C.2. Alternative Implementation of AMP

We evaluate the effectiveness of the AMP module by comparing it with alternative attention-based designs (Veličković et al., 2018; Vaswani et al., 2017).

Unlike standard attention mechanisms such as Graph Attention Networks(GAT) (Veličković et al., 2018), AMP enables differentiable dynamic hierarchy construction—an essential capability for modeling evolving physical systems. While

Table 9. Results of different implementations of AMP module. 

<table><tr><td rowspan="2">AMP Implementation</td><td colspan="2">Cylinder</td><td colspan="2">Flag</td></tr><tr><td>RMSE-1 ( $\times 10^{-2}$ )</td><td>RMSE-All ( $\times 10^{-2}$ )</td><td>RMSE-1( $\times 10^{-2}$ )</td><td>RMSE-All ( $\times 10^{-2}$ )</td></tr><tr><td>EvoMesh-GATConv</td><td>0.2025</td><td>10.253</td><td>0.3009</td><td>106.2</td></tr><tr><td>EvoMesh-CrossAttention</td><td>0.1881</td><td>8.356</td><td>0.3650</td><td>75.12</td></tr><tr><td>EvoMesh</td><td>0.1568</td><td>6.571</td><td>0.3049</td><td>76.16</td></tr></table>

AMP and GAT both compute importance weights, their usage differs substantially: GAT uses these weights to aggregate node features, whereas AMP directly applies the predicted weights to edge features and reuses them for inter-level feature propagation. This dual mechanism allows AMP to jointly support dynamic hierarchy learning and physics modeling via end-to-end training. To assess the benefits of AMP over GAT, we replace AMP with GATConv (using a single attention head) in EvoMesh. As shown in Table 9, the AMP-based model consistently outperforms the GAT-based variant across datasets, highlighting the importance of dynamic hierarchy construction.

We further explore the flexibility of AMP by implementing it with a cross-attention mechanism (Vaswani et al., 2017) in place of the default MLP. Specifically, edge features $\hat{e}_{ij}$ are refined by attending to node features $v_{i}$ through scaled dot-product attention:

$$
\hat {\mathbf {e}} _ {\text { aggr }} = \text { CrossAttention } (Q = \mathbf {v} _ {i},, K = \hat {\mathbf {e}} _ {i j},, V = \hat {\mathbf {e}} _ {i j}), \tag {9}
$$

and the resulting attention scores are reused for inter-level propagation. Experimental results in Table 9 show that while the cross-attention implementation is competitive, the original MLP-based AMP achieves better overall accuracy. This suggests that the edge features $\hat{e}_{ij}$ already encode relevant node information, making the explicit inclusion of $v_{i}$ through attention redundant and potentially less efficient.

# C.3. Edge Enhancement

When constructing the lower-level graph $G_{l+1}$ based on the selected nodes, the edges $E_{l+1}$ are formed by connecting these nodes using the original edges $E_{l}$ from the previous graph. However, this approach may lead to disconnected partitions, as observed in previous work (Lee et al., 2019; Cao et al., 2023; Gao & Ji, 2019), and illustrated in Figure 6. To address this issue, we enhance the connectivity of $E_{l+1}$ by incorporating K-hop edges during the edge construction process. We investigate the impact of different K values, specifically K = 2, 3, 4, on the Flag dataset. The results are presented in Table 10, along with comparisons of the computational efficiency.

Notably, K = 2 yields the lowest RMSE across all conditions (RMSE-1, RMSE-50, and RMSE-all), indicating superior performance compared to higher K values. Despite the performance decline observed with K = 3 and K = 4, they still outperform the baseline results, indicating the effectiveness of dynamic hierarchical modeling and anisotropy message passing.

Table 10. Results for different values of K in edge enhancement. Here, K = 1 denotes directly using edges of selected nodes from previous graph levels. Training time and memory usage are measured with a batch size of 32, while inference time and memory are evaluated with a batch size of 1. 

<table><tr><td rowspan="2"></td><td rowspan="2">RMSE-1 ( $\times 10^{-2}$ )</td><td rowspan="2">RMSE-All ( $\times 10^{-2}$ )</td><td colspan="2">Training</td><td colspan="2">Infer</td></tr><tr><td>Time (ms)</td><td>vRAM (GBs)</td><td>Time (ms)</td><td>vRAM (GBs)</td></tr><tr><td> $K = 1$ </td><td> $\underline{0.3296}$ </td><td>100.1</td><td> $\underline{31.57}$ </td><td> $\underline{14.75}$ </td><td> $\underline{23.60}$ </td><td> $\underline{1.17}$ </td></tr><tr><td> $K = 2$ </td><td> $\underline{0.3049}$ </td><td> $\underline{76.16}$ </td><td> $\underline{33.67}$ </td><td> $\underline{16.53}$ </td><td> $\underline{26.33}$ </td><td> $\underline{1.24}$ </td></tr><tr><td> $K = 3$ </td><td>0.3380</td><td> $\underline{86.84}$ </td><td> $\underline{34.67}$ </td><td> $\underline{18.49}$ </td><td> $\underline{33.21}$ </td><td> $\underline{1.25}$ </td></tr><tr><td> $K = 4$ </td><td>0.3510</td><td> $\underline{105.4}$ </td><td>35.27</td><td>18.76</td><td>32.25</td><td>1.28</td></tr></table>

# C.4. Hyperparameter Analyses on Number of Hierarchies

We conduct an ablation study to assess the impact of varying numbers of hierarchies on model performance. The results from the CylinderFlow dataset, illustrated in Figure 7, demonstrate that EvoMesh consistently outperforms BSMS-GNN across all tested numbers of graph hierarchies. Both models show improved performance with increased hierarchy depth up to 7, indicating that deeper levels help capture more complex interactions and thus enhance accuracy. However, a slight

![](images/9185752761e221b3bc62e8505ba6eaf05a54ee46dee743b9b4489b1ab82d86ab.jpg)

Figure 7. Model comparisons on different numbers of graph hierarchies on CylinderFlow dataset.   
![](images/bb46b59f27435c8e791c104fc7a7a29efbc6af0b1b9b98b237f8e2c5ed69ad13.jpg)

<details>
<summary>text_image</summary>

BSMS-GNN
</details>

(a) Error distribution for BSMS-GNN and EvoMesh   
![](images/6b718c0859a12373c88693a4894f6b89cad420451365bcd6dd93102766763cb6.jpg)

<details>
<summary>text_image</summary>

EvoMesh
</details>

![](images/f099fb24dc6c9e043dc2a4073db343d2feb8910a297d68f5eb6887455be31ebf.jpg)

<details>
<summary>bar</summary>

| Version | Bi-stride Pooling | DiffSELECT |
| ------- | ----------------- | ---------- |
| V4      | 9%                | 50%        |
| V5      | 8%                | 50%        |
| V6      | 9%                | 48%        |
| V7      | 9%                | 47%        |
</details>

(b) Ratios of challenging nodes   
Figure 8. (a) Error maps, where nodes with the top 10% of errors in each model's predictions are marked in yellow and referred to as “challenging nodes”. (b) EvoMesh retains more challenging nodes in coarser graph hierarchies to capture multi-scale dependencies.

performance decline is observed at level 9, which may suggest the onset of overfitting. Overall, the dynamically learned hierarchies in EvoMesh are shown to be more effective compared to the predefined static hierarchies used in BSMS-GNN.

# C.5. Effectiveness of Evolving Hierarchies

From Figure 4, by comparing EvoMesh vs. M2 and M4 vs. M1, we observe the advantages of learning adaptive and temporally evolving hierarchical graph structures. These results highlight the significance of adaptively modeling interactions in context-dependent graphs. To better understand how EvoMesh constructs dynamic hierarchies, we visualize the distribution of nodes with the top 10% prediction errors in Figure 8(a). Accordingly in Figure 8(b), we observe that EvoMesh retains a higher proportion of “challenging” nodes in the coarser message passing levels, enabling our model to capture multi-scale dependencies more effectively, especially in areas where finer message passing levels struggle. In contrast, the predefined static hierarchies in the Bi-stride pooling baseline are data-independent and may inevitably overlook modeling long-range relations surrounding these pivotal nodes, even though they typically present higher errors than those in EvoMesh.

# C.6. Stability Analysis

Given the inherent randomness introduced by the Gumbel-Softmax sampling process in DiffSELECT, we evaluated the stability of EvoMesh by running the trained model on the test set in three independent trials. We report the mean and standard deviation of the prediction errors in Table 11. Despite the stochastic nature of the node selection process, the results show a very small standard deviation, demonstrating that EvoMesh reliably constructs stable and consistent dynamic hierarchies. This stability can be attributed to the DiffSELECT operation, where the node update module $\phi^v$ generates probabilities for retaining nodes in the next-level graph based on anisotropic aggregated edge features. The Gumbel-Softmax

Table 11. Evaluation of EvoMesh with three independent tests. 

<table><tr><td></td><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td></tr><tr><td>RMSE-1 ( $\times 10^{-2}$ )</td><td> $0.1506 \pm 3.6E-4$ </td><td> $36.27 \pm 5.7E-4$ </td><td> $0.2741 \pm 2.4E-2$ </td><td> $0.0263 \pm 5.6E-6$ </td></tr><tr><td>RMSE-All ( $\times 10^{-2}$ )</td><td> $6.317 \pm 0.33$ </td><td> $2018 \pm 130$ </td><td> $68.66 \pm 2.9$ </td><td> $1.327 \pm 0.002$ </td></tr></table>

Table 12. Full quantitative results over three training seeds. 

<table><tr><td></td><td>Cylinder</td><td>Airfoil</td><td>Flag</td><td>Plate</td></tr><tr><td colspan="5">RMSE-1 ( $\times 10^{-2}$ )</td></tr><tr><td>MGN (2021)</td><td> $0.3046 \pm 1.08E-2$ </td><td> $77.38 \pm 1.34E+1$ </td><td> $0.3459 \pm 6.34E-2$ </td><td> $0.0579 \pm 2.64E-3$ </td></tr><tr><td>Lino et al. (2022)</td><td> $3.9352 \pm 11.3E-2$ </td><td> $85.66 \pm 0.35E+1$ </td><td> $0.9993 \pm 2.44E-2$ </td><td> $0.0291 \pm 0.19E-3$ </td></tr><tr><td>BSMS-GNN (2023)</td><td> $0.2263 \pm 4.39E-2$ </td><td> $71.69 \pm 1.41E+1$ </td><td> $0.5080 \pm 0.48E-2$ </td><td> $0.0632 \pm 14.3E-3$ </td></tr><tr><td>Eagle (2023)</td><td> $0.1733 \pm 3.02E-2$ </td><td> $0.385 \pm 1.03E+1$ </td><td> $51.55 \pm 0.87E-2$ </td><td> $0.0392 \pm 0.52E-3$ </td></tr><tr><td>HCMT (2024)</td><td> $0.9190 \pm 61.2E-2$ </td><td> $48.62 \pm 0.51E+1$ </td><td> $0.4013 \pm 1.76E-2$ </td><td> $0.0295 \pm 3.45E-3$ </td></tr><tr><td>EvoMesh</td><td> $\textbf{0.1568} \pm 0.94\textbf{E-2}$ </td><td> $\textbf{41.41} \pm 0.66\textbf{E+1}$ </td><td> $\textbf{0.3049} \pm 6.34\textbf{E-2}$ </td><td> $\textbf{0.0282} \pm 2.65\textbf{E-3}$ </td></tr><tr><td colspan="5">RMSE-All ( $\times 10^{-2}$ )</td></tr><tr><td>MGN (2021)</td><td> $59.78 \pm 2.00\textbf{E+1}$ </td><td> $2816 \pm 1.99\textbf{E+2}$ </td><td> $115.3 \pm 1.30\textbf{E+1}$ </td><td> $3.982 \pm 1.14\textbf{E-2}$ </td></tr><tr><td>Lino et al. (2022)</td><td> $27.60 \pm 0.86\textbf{E+1}$ </td><td> $2080 \pm 0.39\textbf{E+2}$ </td><td> $118.2 \pm 0.58\textbf{E+1}$ </td><td> $2.090 \pm 13.2\textbf{E-2}$ </td></tr><tr><td>BSMS-GNN (2023)</td><td> $16.98 \pm 0.12\textbf{E+1}$ </td><td> $2493 \pm 1.70\textbf{E+2}$ </td><td> $168.1 \pm 0.65\textbf{E+1}$ </td><td> $1.811 \pm 0.42\textbf{E-2}$ </td></tr><tr><td>Eagle (2023)</td><td> $20.05 \pm 0.67\textbf{E+1}$ </td><td> $2344 \pm 2.11\textbf{E+2}$ </td><td> $127.7 \pm 0.88\textbf{E+1}$ </td><td> $7.797 \pm 2.35\textbf{E-2}$ </td></tr><tr><td>HCMT (2024)</td><td> $23.59 \pm 1.38\textbf{E+1}$ </td><td> $3238 \pm 3.62\textbf{E+2}$ </td><td> $90.32 \pm 0.50\textbf{E+1}$ </td><td> $2.468 \pm 42.4\textbf{E-2}$ </td></tr><tr><td>EvoMesh</td><td> $\textbf{6.571} \pm 0.06\textbf{E+1}$ </td><td> $\textbf{2002} \pm 1.02\textbf{E+2}$ </td><td> $\textbf{76.16} \pm 1.30\textbf{E+1}$ </td><td> $\textbf{1.296} \pm 1.14\textbf{E-2}$ </td></tr></table>

technique, coupled with temperature annealing, enables differentiable and stable node selection across hierarchy levels. As a result, the dynamic hierarchies are constructed in a manner that is not only consistent but also optimized for long-range dependencies. Moreover, the prediction errors from EvoMesh are significantly smaller than those of the baseline models, underscoring the robustness and reliability of the model, even with its dynamic node selection mechanism.

# C.7. Full Results over Multiple Training Seeds

In Table 2 in the main manuscript, we report the mean results calculated over three random seeds. In Table 12, we provide full comparisons between our model and the baseline models, including standard deviations.

# C.8. Computation Efficiency

We evaluate computational efficiency based on four criteria: training hours, inference time per step, and the total number of model parameters. The results are presented in Table 13.

# C.9. Rollout Showcases

Figures 9 showcase rollout error maps for the Airfoil, Flag, and DeformingPlate datasets. EvoMesh exhibits much lower rollout errors than the baseline models.

# C.10. Constructed Dynamic Hierarchies

We visualize the constructed context-aware and temporally evolving hierarchies in Figure 10. We can see that the constructed hierarchies evolve as the input context changes and the evolving graph structures align with high-intensity regions. More visualizations of the evolution of the graph structure across the sequence are included in the supplementary material.

# D. Baseline Details

We compare EvoMesh with following competitive baselines: (1) MGN (Pfaff et al., 2021) which performs multiple message passing on the input high-resolution mesh topology; (2) Lino et al. (Lino et al., 2022), which uses manually set grid resolutions and spatial proximity for graph pooling; (3) BSMS-GNN (Cao et al., 2023), which uses predefined bi-stride pooling prior as preprocessing to generate static hierarchies on same mesh topology; (4) Eagle (Yu et al., 2024), which uses

Table 13. The detailed measurements of computation efficiency for EvoMesh and baseline models. 

<table><tr><td>Measurements</td><td>Dataset</td><td>BSMS-GNN</td><td>HCMT</td><td>EvoMesh</td></tr><tr><td rowspan="4">Training cost (hrs)</td><td>Cylinder</td><td>37.11</td><td>80.60</td><td>35.96</td></tr><tr><td>Airfoil</td><td>79.09</td><td>114.32</td><td>75.45</td></tr><tr><td>Flag</td><td>18.27</td><td>66.70</td><td>17.14</td></tr><tr><td>Plate</td><td>39.80</td><td>99.84</td><td>41.85</td></tr><tr><td rowspan="4">Infer time/step (ms)</td><td>Cylinder</td><td>16.55</td><td>79.52</td><td>21.79</td></tr><tr><td>Airfoil</td><td>38.04</td><td>106.34</td><td>58.84</td></tr><tr><td>Flag</td><td>17.18</td><td>85.87</td><td>26.33</td></tr><tr><td>Plate</td><td>28.44</td><td>100.78</td><td>47.45</td></tr><tr><td rowspan="4"># Parameter</td><td>Cylinder</td><td>2.05M</td><td>2.03M</td><td>2.66M</td></tr><tr><td>Airfoil</td><td>2.58M</td><td>2.03M</td><td>2.27M</td></tr><tr><td>Flag</td><td>2.06M</td><td>2.03M</td><td>2.67M</td></tr><tr><td>Plate</td><td>2.87M</td><td>2.03M</td><td>3.20M</td></tr></table>

a two-scale hierarchical message-passing approach, downscaling mesh resolution via geometric clustering of mesh positions; (5) HCMT (Yu et al., 2024), which uses Delauny triangulation based on bi-stride nodes and adopt attention mechanism to enable non-uniform feature propagation. The architecture details of the compared models are as follows:

- MGN. In MGN, we use 15 message passing steps in all datasets. The encoder, decoder, node update function, and edge update function are configured in the same way as in our model.   
- Lino et al. We use the four-scale GNN structure proposed in the work of Lino et al. (2022). The edge length of the smallest cell for each dataset is $1/10$ of the average scene size, with each lower scale doubling in size. We follow its original paper to use 4 message passing steps at the top and bottom levels and two for the others.   
- BSMS-GNN. We use the same number of graph hierarchies in EvoMesh and as in BSMS-GNN. We use the minimum average distance as the seeding heuristic for the BFS search recommended in its original paper. The multi-level building is processed in one pass. The inter-level propagation uses the normalized node degree to convolve features from neighbors to central nodes. The encoder, decoder, node update function, and edge update function are set up the same way as in our model. We perform one message passing step at each graph level.   
- Eagle. We use the same-size KMeans algorithm in the preprocessing step with a cluster size of 10. The encoder consists of 4 stacked GNN layers. The graph pooling module comprises a single-layer gated recurrent unit followed by a single-layer MLP. The hidden dimension is set to 128 across all datasets. The attention module includes 4 sequential attention blocks, each with a single attention head. A final layer normalization is applied after the last attention block. The decoder shares the same architecture as that used in EvoMesh.   
- HCMT. The hidden dimension and the number of attention heads in the HCMT block are set to 128 and 4, respectively. We use the same number of hierarchies as in EvoMesh. For the Cylinder and Airfoil datasets, due to the presence of hollow sections in the mesh, we do not apply Delaunay triangulation for remeshing. Instead, we use edge connections generated through bi-stride pooling. Like in EvoMesh, we use a single message passing step at each graph level.

Notably, the node encoder, decoder, node update function, and edge update function of MGN, Lino et al., BSMS-GNN, HCMT and Eagle have the same network architecture as those in EvoMesh. To reduce the number of network parameters, we avoid separately encoding the edge offset $e_{ij}$ . Instead, we concatenate it with the node latents and use this combined input for the edge update function to compute $\hat{e}_{ij}$ .

All models are trained using the Adam optimizer for 1M steps, with an exponential learning rate decay from $10^{-4}$ to $10^{-6}$ and a decay rate of $\gamma = 0.79$ from the first 500K steps. The batch size is set to 32.

![](images/f0df84d8142be4c20a0460f7e4d03289f19e0fa9a8b901ff5ef19ded13c2a23e.jpg)  
Figure 9. Showcases of rollout prediction error maps on Airfoil, Flag and DeformingPlate dataset.

![](images/b1ea9212b7f4891380cc7595d7755e7bc160271b33ef41dd1f7fbb96f5f68575.jpg)  
Figure 10. Row 1: The velocity field from the true data on the CylinderFlow dataset. Row 2-6: The temporal difference of the velocity fields between adjacent time steps alongside the constructed coarser-level graphs.