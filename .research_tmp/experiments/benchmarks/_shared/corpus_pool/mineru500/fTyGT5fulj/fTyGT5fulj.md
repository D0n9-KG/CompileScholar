# Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First

Zheng Zhang $^{\dagger}$ Junxiang Wang $^{\diamond}$ Liang Zhao $^{\dagger}$

$^{\dagger}$ Emory University, Atlanta, GA $\diamondsuit$ NEC Labs America, Princeton, NJ

{zheng.zhang,liang.zhao}@emory.edu

{junxiang.wang}@alumni.emory.edu

# Abstract

Graph Neural Networks (GNNs) have achieved great success in representing data with dependencies by recursively propagating and aggregating messages along the edges. However, edges in real-world graphs often have varying degrees of difficulty, and some edges may even be noisy to the downstream tasks. Therefore, existing GNNs may lead to suboptimal learned representations because they usually treat every edge in the graph equally. On the other hand, Curriculum Learning (CL), which mimics the human learning principle of learning data samples in a meaningful order, has been shown to be effective in improving the generalization ability and robustness of representation learners by gradually proceeding from easy to more difficult samples during training. Unfortunately, existing CL strategies are designed for independent data samples and cannot trivially generalize to handle data dependencies. To address these issues, we propose a novel CL strategy to gradually incorporate more edges into training according to their difficulty from easy to hard, where the degree of difficulty is measured by how well the edges are expected given the model training status. We demonstrate the strength of our proposed method in improving the generalization ability and robustness of learned representations through extensive experiments on nine synthetic datasets and nine real-world datasets. The code for our proposed method is available at https://github.com/rollingstonezz/Curriculum\_learning\_for\_GNNs.

# 1 Introduction

Inspired by cognitive science studies $[8, 33]$ that humans can benefit from the sequence of learning basic (easy) concepts first and advanced (hard) concepts later, curriculum learning (CL) $[2]$ suggests training a machine learning model with easy data samples first and then gradually introducing more hard samples into the model according to a designed pace, where the difficulty of samples can usually be measured by their training loss $[25]$ . Many previous studies have shown that this easy-to-hard learning strategy can effectively improve the generalization ability of the model $[2, 19, 15, 11, 35, 46]$ , and some studies $[19, 15, 11]$ have shown that CL strategies can also increase the robustness of the learned model against noisy training samples. An intuitive explanation is that in CL settings noisy data samples correspond to harder samples, and CL learner spends less time with the harder (noisy) samples to achieve better generalization performance and robustness.

Although CL strategies have achieved great success in many fields such as computer vision and natural language processing, existing methods are designed for independent data (such as images) while designing effective CL methods for data with dependencies has been largely underexplored. For example, in a citation network, two researchers with highly related research topics (e.g. machine learning and data mining) are more likely to collaborate with each other, while the reason behind a collaboration of two researchers with less related research topics (e.g. computer architecture and social science) might be more difficult to understand. Prediction on one sample impacts that of another, forming a graph structure that encompasses all samples connected by their dependencies. There are

many machine learning techniques for such graph-structured data, ranging from traditional models like conditional random field [36], graph kernels [37], to modern deep models like GNNs [29, 30, 52, 38, 49, 12, 53, 42]. However, traditional CL strategies are not designed to handle the curriculum of the dependencies between nodes in graph data, which are insufficient. Handling graph-structured data require not only considering the difficulty in individual samples, but also the difficulty of their dependencies to determine how to gradually composite correlated samples for learning.

As previous CL strategies indicated that an easy-to-hard learning sequence on data samples can improve the generalization and robustness performance, an intuitive question is whether a similar strategy on data dependencies that iteratively involves easy-to-hard edges in learning can also benefit. Unfortunately, there exists no trivial way to directly generalize existing CL strategies on independent data to handle data dependencies due to several unique challenges: (1) Difficulty in quantifying edge selection criteria. Existing CL studies on independent data often use supervised computable metrics (e.g. training loss) to quantify sample difficulty, but how to quantify the difficulties of understanding the dependencies between data samples which has no supervision is challenging. (2) Difficulty in designing an appropriate curriculum to gradually involve edges. Similar to the human learning process, the model should ideally retain a certain degree of freedom to adjust the pacing of including edges according to its own learning status. As existing CL methods for graph data typically use fixed pacing function to involve samples, they can not provide this flexibility. Designing an adaptive pacing function for handling graph data is difficult since it requires joint optimization of both supervised learning tasks on nodes and the number of chosen edges. (3) Difficulty in ensuring convergence and a numerical steady process for CL in graphs. Discrete changes in the number of edges can cause drift in the optimal model parameters between training iterations. How to guarantee a numerically stable learning process for CL on edges is challenging.

In order to address the aforementioned challenges, in this paper, we propose a novel CL algorithm named Relational Curriculum Learning (RCL) to improve the generalization ability and robustness of representation learners on data with dependencies. To address the first challenge, we propose an approach to select the edges by quantifying their corresponding difficulties in a self-supervised learning manner. Specifically, for each training iteration, we choose K easiest edges whose corresponding relations are most well-expected by the current model. Second, to design an appropriate learning pace for gradually involving more edges in training, we present the learning process as a concise optimization model, which automatically lets the model gradually increase the number K to involve more edges in training according to its own status. Third, to ensure convergence of optimizing the model, we propose an alternative optimization algorithm with a theoretical convergence guarantee and an edge reweighting scheme to smooth the graph structure transition. Finally, we demonstrate the superior performance of RCL compared to state-of-the-art comparison methods through extensive experiments on both synthetic and real-world datasets.

# 2 Related Works

Curriculum Learning (CL). Bengio et al.[2] pioneered the concept of Curriculum Learning (CL) within the machine learning domain, aiming to improve model performance by gradually including easy to hard samples in training the model. Self-paced learning [25] measures the difficulty of samples by their training loss, which addressed the issue in previous works that difficulties of samples are generated by prior heuristic rules. Therefore, the model can adjust the curriculum of samples according to its own training status. Following works [18, 17, 55] further proposed many supervised measurement metrics for determining curriculums, for example, the diversity of samples [17] or the consistency of model predictions [55]. Meanwhile, many empirical and theoretical studies were proposed to explain why CL could lead to generalization improvement from different perspectives. For example, studies such as MentorNet [19] and Co-teaching [15] empirically found that utilizing CL strategy can achieve better generalization performance when the given training data is noisy. [11] provided theoretical explanations on the denoising mechanism that CL learners waste less time with the noisy samples as they are considered harder samples. Some studies [2, 35, 46, 13, 24] also realized that CL can help accelerate the optimization process of non-convex objectives and improve the speed of convergence in the early stages of training.

Despite great success, most of the existing designed CL strategies are for independent data such as images, and there is little work on generalizing CL strategies to handle samples with dependencies. Few existing attempts on graph-structured data $[26, 21, 28]$ , such as $[44, 5, 45, 28]$ , simply treat

![](images/b9c9665589924c67bdffe14997e7ceb788bdd172459323a83d24327b06eea8a7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Incremental Edge Selection (IES)"] --> B["GNN"]
    B --> C["Latent embedding"]
    C --> D["Decoder"]
    D --> E["Residual graph"]
    E --> F["Top K expected"]
    F --> G["Refined structure"]
    H["IES"] --> I["Training process T"]
    J["0"] --> K["1"]
    L["..."] --> M["t"]
    N["..."] --> O["T"]
    P["..."] --> Q["T"]
    R["y"] --> S["y"]
    T["ŷ"] --> U["y"]
```
</details>

Figure 1: The overall framework of RCL. (a) The Incremental Edge Selection module first extracts the latent node embedding by the GNN model given the current training structure, then jointly learns the node prediction label y and reconstructs the input structure by a decoder. A small residual error on an edge indicates the corresponding dependency is well expected and thus can be added to the refined structure for the next iteration. (b) The iterative learning process of RCL. The model starts with an empty structure and gradually includes more edges until the training structure converges to the input structure.

nodes as independent samples and then apply CL strategies on independent data, which ignore the fundamental and unique dependency information carried by the structure in data, and thus can not well handle the correlation between data samples. Furthermore, these models are mostly based on heuristic-based sample selection strategies $[5, 45, 28]$ , which largely limit the generalizability of these methods.

Graph structure learning. Another stream of existing studies that are related to our work is graph structure learning. Recent studies have shown that GNN models are vulnerable to adversarial attacks on graph structure $[7, 48]$ . In order to address this issue, studies in graph structure learning usually aim to jointly learn an optimized graph structure and corresponding graph representations. Existing works $[9, 4, 20, 54, 31]$ typically consider the hypothesis that the intrinsic graph structure should be sparse or low rank from the original input graph by pruning “irrelevant” edges. Thus, they typically use pre-deterministic methods $[7, 56, 9]$ to preprocess graph structure such as singular value decomposition (SVD), or dynamically remove “redundant” edges according to the downstream task performance on the current sparsified structure $[4, 20, 31]$ . However, modifying the graph topology will inevitably lose potentially useful information lying in the removed edges. More importantly, the modified graph structure is usually optimized for maximizing the performance on the training set, which can easily lead to overfitting issues.

# 3 Preliminaries

Graph neural networks (GNNs) are a class of methods that have shown promising progress in representing structured data in which data samples are correlated with each other. Typically, the data samples are treated as nodes while their dependencies are treated as edges in the constructed graph. Formally, we denote a graph as $G = (\mathcal{V}, \mathcal{E})$ , where $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ is a set of nodes that $N = |\mathcal{V}|$ denotes the number of nodes in the graph and $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of edges. We also let $\mathbf{X} \in \mathbb{R}^{N \times b}$ denote the node attribute matrix and let $\mathbf{A} \in \mathbb{R}^{N \times N}$ represent the adjacency matrix. Specifically, $A_{ij} = 1$ denotes that there is an edge connecting nodes $v_i$ and $v_j \in \mathcal{V}$ , otherwise $A_{ij} = 0$ . A GNN model $f$ maps the node feature matrix $\mathbf{X}$ associated with the adjacency matrix $\mathbf{A}$ to the model predictions $\hat{\mathbf{y}} = f(\mathbf{X}, \mathbf{A})$ , and get the loss $L_{\mathrm{GNN}} = L(\hat{\mathbf{y}}, \mathbf{y})$ , where $L$ is the objective function and $\mathbf{y}$ is the ground-truth label of nodes. The loss on one node $v_i$ is denoted as $l_i = L(\hat{y}_i, y_i)$ .

# 4 Methodology

As previous CL methods have shown that an easy-to-hard learning sequence of independent data samples can improve the generalization ability and robustness of the representation learner, the goal of this paper is to develop an effective CL method on data with dependencies, which is extremely difficult due to several unique challenges: (1) Difficulty in designing a feasible principle to select

edges by properly quantifying their difficulties. (2) Difficulty in designing an appropriate pace of curriculum to gradually involve more edges in training based on model status. (3) Difficulty in ensuring convergence and a numerical steady process for optimizing the CL model.

In order to address the above challenges, we propose a novel CL method named Relational Curriculum Learning (RCL). The sequence, which gradually includes edges from easy to hard, is called curriculum and learned in different grown-up stages of training. In order to address the first challenge, we propose a self-supervised module Incremental Edge Selection (IES), which is shown in Figure 1(a), to select the K easiest edges at each training iteration that are mostly expected by the current model. The details are elaborated in Section 4.1. To address the second challenge, we present a joint optimization framework to automatically increase the number of selected edges K given its own training status. The framework is elaborated in Figure 1(b) and details can be found in Section 4.2. Finally, to ensure convergence of optimization and steady the numerical process, we propose an EM-style alternative optimization algorithm with a theoretical convergence guarantee in Section 4.2 Algorithm 1 and an edge reweighting scheme to smooth the discrete edge incrementing process in Section 4.3.

# 4.1 Incremental Edge Selection by Quantifying Difficulties of Sample Dependencies

Here we propose a novel way to select edges by first quantifying their difficulty levels. Existing works on independent data typically use supervised metrics such as training loss of samples to quantify their difficulty level, but there exists no supervised metrics on edges. To address this issue, we propose a self-supervised module Incremental Edge Selection (IES). We first quantify the difficulty of edges by measuring how well the edges are expected from the currently learned embeddings of their connected nodes. Then the most well-expected edges are selected as the easiest edges for the next iteration of training. As shown in Figure 1(a), given the currently selected edges at iteration t, we first feed them to the GNN model to extract the latent node embeddings. Then we restore the latent node embeddings to the original graph structure through a decoder, which is called the reconstruction of the original graph structure. The residual graph R, which is defined as the degree of mismatch between the original adjacency matrix A and the reconstructed adjacency matrix $\mathbf{A}^{(t)}$ , can be considered a strong indicator for describing how well the edges are expected by the current model. Specifically, a smaller residual error indicates a higher probability of being a well-expected edge.

With the developed self-supervised method to measure the difficulties of edges, here we formulate the key learning paradigm of selecting the top $K$ easiest edges. To obtain the training adjacency matrix $\mathbf{A}^{(t)}$ that will be fed into the GNN model $f^{(t)}$ , we introduce a learnable binary mask matrix $\mathbf{S}$ with each element $\mathbf{S}_{ij} \in \{0,1\}$ . Thus, the training adjacency matrix at iteration $t$ can be represented as $\mathbf{A}^{(t)} = \mathbf{S}^{(t)} \odot \mathbf{A}$ . To filter out the edges with $K$ smallest residual error, we penalize the summarized residual errors over the selected edges, which can be represented as $\sum_{i,j} \mathbf{S}_{ij} \mathbf{R}_{ij}$ . Therefore, the learning objective can be presented as follows:

$$
\min _ {\mathbf {w}} L _ {\mathrm{GNN}} + \beta \sum_ {i, j} \mathbf {S} _ {i j} \mathbf {R} _ {i j}, \tag {1}
$$

$$
s. t. \left\| \mathbf {S} \right\| _ {1} \geq K,
$$

where the first term $L_{\mathrm{GNN}} = L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ is the node-level predictive loss, e.g. cross-entropy loss for the node classification task. The second term $\sum_{i,j} S_{ij} R_{ij}$ aims at penalizing the residual errors over the edges selected by the mask matrix S. $\beta$ is a hyperparameter to tune the balance between terms. The constraint is to guarantee only the most K well-expected edges are selected.

More concretely, the value of a residual edge $\tilde{\mathbf{A}}_{ij}^{(t)}\in [0,1]$ can be computed by a non-parametric kernel function $\kappa (\mathbf{z}_i^{(t)},\mathbf{z}_j^{(t)})$ , e.g. the inner product kernel. Then the residual error $\mathbf{R}_{ij}$ between the input structure and the reconstructed structure can be defined as $\left\| \tilde{\mathbf{A}}_{ij}^{(t)} - \mathbf{A}_{ij}\right\|$ , where $\| \cdot \|$ is commonly chosen to be the squared $\ell_2$ -norm.

# 4.2 Automatically Control the Pace of Increasing Edges

In order to dynamically include more edges into training, an intuitive way is to iteratively increase the value of K in Equation 1 to allow more edges to be selected. However, it is difficult to determine an appropriate value of K with respect to the training status of the model. Besides, directly solving

Algorithm 1 Alternating Minimization Algorithm for Optimizing Equation 2   
Input: Node features X, adjacency matrix A, stepsize $\mu$ and hyperparameter $\gamma$ Output: Parameters w of GNN model f
1: Initialize $\mathbf{w}^{(0)}$ , $\mathbf{S}^{(0)}$ , $\lambda$ 2: while Not converged do
3: $\mathbf{w}^{(t)} = \arg\min_{\mathbf{w}} L(f(\mathbf{X}, \mathbf{A}^{(t-1)}; \mathbf{w}), \mathbf{y}) + \beta \sum_{i,j} \mathbf{S}_{ij} \left\| \tilde{\mathbf{A}}_{ij}^{(t-1)} - \mathbf{A}_{ij} \right\| + \frac{\gamma}{2} \left\| \mathbf{w} - \mathbf{w}^{(t-1)} \right\|$ 4: Given $\mathbf{w}^{(t)}$ , extract latent nodes embedding $\mathbf{Z}^{(t)}$ from GNN model f
5: Calculate reconstructed structure $\tilde{\mathbf{A}}_{ij}^{(t)} = \kappa(\mathbf{z}_{i}^{(t)}, \mathbf{z}_{j}^{(t)})$ for all pairs of i, j
6: $\mathbf{S}^{(t)} = \arg\min_{\mathbf{S}} \beta \sum_{i,j} \mathbf{S}_{ij} \left\| \mathbf{A}_{ij} - \tilde{\mathbf{A}}_{ij}^{(t)} \right\| + g(\mathbf{S}; \lambda) + \frac{\gamma}{2} \left\| \mathbf{S} - \mathbf{S}^{(t-1)} \right\|$ 7: Compute $\mathbf{A}^{(t)} = \mathbf{S}^{(t)} \odot \mathbf{A}$ 8: if $\mathbf{A}^{(t)} \neq \mathbf{A}$ then
9: Increase $\lambda$ by stepsize $\mu$ 10: end if
11: end while

Equation 1 is difficult since $\mathbf{S}$ is a binary matrix where each element $\mathbf{S}_{ij} \in \{0,1\}$ , optimizing $\mathbf{S}$ would require solving a discrete constraint program at each iteration. To address this issue, we first relax the problem into continuous optimization so that each $\mathbf{S}_{ij}$ can be allowed to take any value in the interval $[0,1]$ . Note that the inequality $||\mathbf{S}||_1 \geq K$ in Eqn. 1 is equivalent to the equality $||\mathbf{S}||_1 = K$ . This is because the second term in the loss function would always encourage fewer selected edges by the mask matrix $\mathbf{S}$ , as all values in the residual error matrix $\mathbf{R}$ and mask matrix $\mathbf{S}$ are nonnegative. Given this, we can incorporate the equality constraint as a Lagrange multiplier and rewrite the loss function as $\mathcal{L} = L_{GNN} + \beta \sum_{i,j} \mathbf{S}_{ij} \mathbf{R}_{ij} - \lambda (||\mathbf{S}||_1 - K)$ . Considering that $K$ remains constant, the optimization of the loss function can be equivalently framed by substituting the given constraint with a regularization term denoted as $g(\mathbf{S};\lambda)$ . As such, the overall loss function can be reformulated as:

$$
\min _ {\mathbf {w}, \mathbf {S}} L _ {\mathrm{GNN}} + \beta \sum_ {i, j} \mathbf {S} _ {i j} \mathbf {R} _ {i j} + g (\mathbf {S}; \lambda), \tag {2}
$$

where $g(\mathbf{S};\lambda)=\lambda\|\mathbf{S}-\mathbf{A}\|$ and $\|\cdot\|$ is commonly chosen to be the squared $\ell_{2}$ -norm. Since the training adjacency matrix $\mathbf{A}^{(t)}=\mathbf{S}^{(t)}\odot\mathbf{A}$ , as $\lambda\to\infty$ , more edges in the input structure are included until the training adjacency matrix $\mathbf{A}^{(t)}$ converges to the input adjacency matrix A. Specifically, the regularization term $g(\mathbf{S};\lambda)$ controls the learning scheme by the age parameter $\lambda$ , where $\lambda=\lambda(t)$ grows with the number of iterations. By monotonously increasing the value of $\lambda$ , the regularization term $g(\mathbf{S};\lambda)$ will push the mask matrix gradually converge to the input adjacency matrix A, resulting in more edges automatically involved in the training structure.

Optimization of learning objective. In optimizing the objective function in Equation 2, we need to jointly optimize parameter w for GNN model f and the mask matrix S. To tackle this, we introduce an EM-style optimization scheme (detailed in Algorithm 1) that iteratively updates both. The algorithm uses the node feature matrix X, the original adjacency matrix A, a step size $\mu$ to control the age parameter $\lambda$ increase rate, and a hyperparameter $\gamma$ for regularization adjustments. Post initialization of w and S, it alternates between: optimizing GNN model f (Step 3), extracting latent node embeddings and reconstructing the adjacency matrix (Steps 4 & 5), refining the mask matrix using the reconstructed matrix and regularization, and results in more edges are gradually involved (Step 6), updating the training adjacency matrix (Step 7), and incrementing $\lambda$ when the training matrix $\mathbf{A}^{(t)}$ differs from input matrix A, incorporating more edges in the next iteration.

# Theorem 4.1. We have the following convergence guarantees for Algorithm 1:

- Avoidance of Saddle Points. If the second derivatives of $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ are continuous, then for sufficiently large $\gamma$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 with random initializations will not converge to a strict saddle point of $F$ almost surely.   
- Second Order Convergence. If the second derivatives of $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ are continuous, and $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ satisfy the Kurdyka-Lojasiewicz (KL) property [41], then for sufficiently large $\gamma$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 with random initialization will almost surely converge to a second-order stationary point of $F$ .

The detailed proof can be found in Appendix B.

# 4.3 Smooth Structure Transition by Edge Reweighting

Note that in the Algorithm 1, the optimization process requires iteratively updating the parameters w of the GNN model f and current adjacency matrix $\mathbf{A}^{(t)}$ , where $\mathbf{A}^{(t)}$ varies discretely between iterations. However, GNN models mostly work in a message-passing fashion, which computes node representations by iteratively aggregating information along edges from neighboring nodes. Discretely modifying the number of edges will result in a great drift of the optimal model parameters between iterations. In Appendix Figure, we demonstrate that a shift in the optimal parameters of the GNN results in a spike in the training loss. Therefore, it can increase the difficulty of finding optimal parameters and even hurt the generalization ability of the model in some cases. Besides the numerical problem caused by discretely increasing the number of edges, another issue raised by the CL strategy in Section 4.1 is the trustworthiness of the estimated edge difficulty, which is inferred by the residual error on the edges. Although the residual error can reflect how well edges are expected in the ideal case, the quality of the learned latent node embeddings may affect the validity of this metric and compromise the quality of the designed curriculum by the CL strategy.

To address both issues, we propose a novel edge reweighting scheme to (1) smooth the transition of the training structure between iterations, and (2) reduce the weight of edges that connect nodes with low-confidence latent embeddings. Formally, we use a smoothed version of structure $\bar{\mathbf{A}}^{(t)}$ to substitute $\mathbf{A}^{(t)}$ for training the GNN model $f$ in step 3 of Algorithm 1, where the mapping from $\mathbf{A}^{(t)}$ to $\bar{\mathbf{A}}^{(t)}$ can be represented as:

$$
\bar {\mathbf {A}} _ {i j} ^ {(t)} = \pi_ {i j} ^ {(t)} \mathbf {A} _ {i j} ^ {(t)}, \tag {3}
$$

where $\pi_{ij}^{(t)}$ is the weight imposed on edge $e_{ij}$ at iteration t. $\pi_{ij}^{(t)}$ is calculated by considering the counted occurrences of edge $e_{ij}$ until the iteration t and the confidence of the latent embedding for the connected pair of nodes $v_{i}$ and $v_{j}$ :

$$
\pi_ {i j} ^ {(t)} = \psi (e _ {i j}) \rho (v _ {i}) \rho (v _ {j}), \tag {4}
$$

where $\psi$ is a function that reflects the number of edge occurrences and $\rho$ is a function to reflect the degree of confidence for the learned latent node embedding. The details of these two functions are described as follow.

Smooth the transition of the training structure between iterations. In order to obtain a smooth transition of the training structure between iterations, we take the learned curriculum of selected edges into consideration. Formally, we model $\psi$ by a smooth function of the edge selected occurrences compared to the model iteration occurrences before the current iteration:

$$
\psi (e _ {i j}) = t (e _ {i j}) / t, \tag {5}
$$

where t is the number of current iterations and $t(e_{ij})$ represents the counting number of selecting edge $e_{ij}$ . Therefore, we transform the original discretely changing training structure into a smoothly changing one by taking the historical edge selection curriculum into consideration.

Reduce the influence of nodes with low confidence latent embeddings. As introduced in our Algorithm 1 line 6, the estimated structure $\tilde{A}$ is inferred from the latent embedding $\mathbf{Z}$ , which is extracted from the trained GNN model $f$ . Such estimated latent embedding may possibly differ from the true underlying embedding, which results in the inaccurately reconstructed structure around the node. In order to alleviate this issue, we model the function $\rho$ by the training loss on nodes, which indicates the confidence of their learned latent embeddings. This idea is similar to previous CL strategies on inferring the difficulty of data samples by their supervised training loss. Specifically, a larger training loss indicates a low confident latent node embedding. Mathematically, the weights $\rho(v_i)$ on node $v_i$ can be represented as a distribution of their training loss:

$$
\rho \left(v _ {i}\right) \sim e ^ {- l _ {i}} \tag {6}
$$

where $l_{i}$ is the training loss on node $v_{i}$ . Therefore, a node with a larger training loss will result in a smaller value of $\rho(v_{i})$ , which reduces the weight of its connecting edges.

# 5 Experiments

In this section, the experimental settings are introduced first in Section 5.1, then the performance of the proposed method on both synthetic and real-world datasets are presented in Section 5.2. We further present the robustness test on our CL method against topological structure noise in Section 5.3.

Table 1: Node classification accuracy on synthetic datasets (%). The best-performing method on each backbone GNN model is highlighted in bold, while the second-best method is underlined. In situations where RCL's performance is not strictly the best among all methods, we can see that almost all methods can achieve a near-perfect performance and RCL is still close to the best methods. 

<table><tr><td>Homo ratio</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td></tr><tr><td>GCN</td><td> $50.84 \pm 1.03$ </td><td> $56.50 \pm 0.50$ </td><td> $65.17 \pm 0.48$ </td><td> $77.94 \pm 0.54$ </td><td> $87.15 \pm 0.44$ </td><td> $93.27 \pm 0.24$ </td><td> $97.48 \pm 0.25$ </td><td> $99.10 \pm 0.17$ </td><td> $99.93 \pm 0.03$ </td></tr><tr><td>GNNSVD</td><td> $54.96 \pm 0.76$ </td><td> $58.45 \pm 0.56$ </td><td> $63.06 \pm 0.63$ </td><td> $70.23 \pm 0.61$ </td><td> $80.51 \pm 0.41$ </td><td> $85.02 \pm 0.46$ </td><td> $90.31 \pm 0.27$ </td><td> $94.23 \pm 0.22$ </td><td> $96.74 \pm 0.23$ </td></tr><tr><td>ProGNN</td><td> $47.87 \pm 0.87$ </td><td> $54.59 \pm 0.55$ </td><td> $65.39 \pm 0.44$ </td><td> $76.96 \pm 0.49$ </td><td> $87.76 \pm 0.51$ </td><td> $93.16 \pm 0.34$ </td><td> $97.60 \pm 0.31$ </td><td> $99.04 \pm 0.19$ </td><td> $99.94 \pm 0.03$ </td></tr><tr><td>NeuralSparse</td><td> $51.42 \pm 1.35$ </td><td> $57.99 \pm 0.69$ </td><td> $65.10 \pm 0.43$ </td><td> $75.37 \pm 0.34$ </td><td> $87.40 \pm 0.29$ </td><td> $93.54 \pm 0.28$ </td><td> $97.16 \pm 0.15$ </td><td> $99.01 \pm 0.22$ </td><td> $99.83 \pm 0.07$ </td></tr><tr><td>PTDNet</td><td> $48.21 \pm 1.98$ </td><td> $55.52 \pm 2.82$ </td><td> $65.82 \pm 0.94$ </td><td> $79.37 \pm 0.45$ </td><td> $89.17 \pm 0.39$ </td><td> $94.19 \pm 0.18$ </td><td> $98.61 \pm 0.12$ </td><td> $99.51 \pm 0.09$ </td><td> $99.81 \pm 0.05$ </td></tr><tr><td>CLNodes</td><td> $50.37 \pm 0.73$ </td><td> $56.64 \pm 0.56$ </td><td> $65.04 \pm 0.66$ </td><td> $77.52 \pm 0.48$ </td><td> $86.85 \pm 0.44$ </td><td> $93.10 \pm 0.47$ </td><td> $97.34 \pm 0.25$ </td><td> $99.02 \pm 0.18$ </td><td> $99.88 \pm 0.04$ </td></tr><tr><td>RCL</td><td> $57.57 \pm 0.43$ </td><td> $62.06 \pm 0.28$ </td><td> $73.98 \pm 0.55$ </td><td> $84.54 \pm 0.75$ </td><td> $92.69 \pm 0.09$ </td><td> $97.42 \pm 0.17$ </td><td> $99.62 \pm 0.05$ </td><td> $99.89 \pm 0.02$ </td><td> $99.93 \pm 0.06$ </td></tr><tr><td>GIN</td><td> $48.33 \pm 1.89$ </td><td> $53.62 \pm 1.39$ </td><td> $64.08 \pm 0.99$ </td><td> $77.55 \pm 1.10$ </td><td> $85.31 \pm 0.75$ </td><td> $90.57 \pm 0.36$ </td><td> $97.82 \pm 0.18$ </td><td> $99.59 \pm 0.11$ </td><td> $99.91 \pm 0.02$ </td></tr><tr><td>GNNSVD</td><td> $43.21 \pm 1.60$ </td><td> $45.68 \pm 1.66$ </td><td> $54.90 \pm 1.16$ </td><td> $68.29 \pm 0.79$ </td><td> $79.76 \pm 0.52$ </td><td> $85.63 \pm 0.44$ </td><td> $93.65 \pm 0.39$ </td><td> $97.22 \pm 0.17$ </td><td> $98.94 \pm 0.17$ </td></tr><tr><td>ProGNN</td><td> $45.76 \pm 1.40$ </td><td> $52.96 \pm 1.01$ </td><td> $64.12 \pm 1.07$ </td><td> $76.95 \pm 0.87$ </td><td> $85.13 \pm 0.71$ </td><td> $89.96 \pm 0.55$ </td><td> $96.54 \pm 0.48$ </td><td> $99.51 \pm 0.12$ </td><td> $99.78 \pm 0.05$ </td></tr><tr><td>NeuralSparse</td><td> $50.23 \pm 2.05$ </td><td> $54.12 \pm 1.52$ </td><td> $62.81 \pm 0.75$ </td><td> $76.98 \pm 1.17$ </td><td> $85.14 \pm 0.94$ </td><td> $92.57 \pm 0.44$ </td><td> $98.02 \pm 0.20$ </td><td> $99.61 \pm 0.12$ </td><td> $99.91 \pm 0.05$ </td></tr><tr><td>PTDNet</td><td> $53.23 \pm 2.76$ </td><td> $56.12 \pm 2.03$ </td><td> $65.81 \pm 1.38$ </td><td> $77.81 \pm 1.02$ </td><td> $86.14 \pm 0.65$ </td><td> $93.21 \pm 0.74$ </td><td> $97.08 \pm 0.41$ </td><td> $99.51 \pm 0.18$ </td><td> $99.91 \pm 0.03$ </td></tr><tr><td>CLNodes</td><td> $45.36 \pm 1.42$ </td><td> $51.10 \pm 1.15$ </td><td> $62.53 \pm 0.88$ </td><td> $75.83 \pm 1.07$ </td><td> $87.76 \pm 0.90$ </td><td> $94.25 \pm 0.44$ </td><td> $98.30 \pm 0.26$ </td><td> $99.60 \pm 0.09$ </td><td> $99.92 \pm 0.03$ </td></tr><tr><td>RCL</td><td> $57.63 \pm 0.66$ </td><td> $62.08 \pm 1.17$ </td><td> $71.02 \pm 0.61$ </td><td> $80.61 \pm 0.69$ </td><td> $88.62 \pm 0.43$ </td><td> $94.88 \pm 0.36$ </td><td> $98.19 \pm 0.19$ </td><td> $99.32 \pm 0.08$ </td><td> $99.89 \pm 0.04$ </td></tr><tr><td>GraphSAGE</td><td> $62.57 \pm 0.55$ </td><td> $67.33 \pm 0.64$ </td><td> $71.06 \pm 0.74$ </td><td> $80.88 \pm 0.54$ </td><td> $85.88 \pm 0.51$ </td><td> $91.42 \pm 0.37$ </td><td> $95.26 \pm 0.33$ </td><td> $97.78 \pm 0.16$ </td><td> $99.52 \pm 0.13$ </td></tr><tr><td>GNNSVD</td><td> $64.42 \pm 0.80$ </td><td> $65.71 \pm 0.39$ </td><td> $67.12 \pm 0.58$ </td><td> $68.47 \pm 0.50$ </td><td> $77.70 \pm 0.65$ </td><td> $82.86 \pm 0.50$ </td><td> $87.81 \pm 0.71$ </td><td> $91.61 \pm 0.55$ </td><td> $95.01 \pm 0.50$ </td></tr><tr><td>ProGNN</td><td> $58.57 \pm 2.09$ </td><td> $66.75 \pm 0.91$ </td><td> $72.14 \pm 0.64$ </td><td> $81.27 \pm 0.44$ </td><td> $86.89 \pm 0.47$ </td><td> $92.10 \pm 0.39$ </td><td> $95.21 \pm 0.30$ </td><td> $97.51 \pm 0.23$ </td><td> $99.50 \pm 0.11$ </td></tr><tr><td>NeuralSparse</td><td> $61.70 \pm 0.77$ </td><td> $66.65 \pm 0.66$ </td><td> $70.60 \pm 0.79$ </td><td> $79.65 \pm 0.45$ </td><td> $84.19 \pm 0.91$ </td><td> $91.31 \pm 0.54$ </td><td> $94.86 \pm 0.53$ </td><td> $97.16 \pm 0.23$ </td><td> $99.55 \pm 0.19$ </td></tr><tr><td>PTDNet</td><td> $65.72 \pm 1.08$ </td><td> $69.25 \pm 0.92$ </td><td> $72.60 \pm 0.77$ </td><td> $79.65 \pm 0.45$ </td><td> $86.54 \pm 0.56$ </td><td> $91.79 \pm 0.53$ </td><td> $96.10 \pm 0.58$ </td><td> $97.98 \pm 0.13$ </td><td> $99.78 \pm 0.08$ </td></tr><tr><td>CLNodes</td><td> $69.41 \pm 0.66$ </td><td> $70.83 \pm 0.58$ </td><td> $75.51 \pm 0.36$ </td><td> $82.65 \pm 0.43$ </td><td> $87.08 \pm 0.56$ </td><td> $91.58 \pm 0.41$ </td><td> $95.91 \pm 0.38$ </td><td> $98.33 \pm 0.26$ </td><td> $99.57 \pm 0.14$ </td></tr><tr><td>RCL</td><td> $68.03 \pm 0.37$ </td><td> $71.39 \pm 0.51$ </td><td> $76.99 \pm 0.99$ </td><td> $83.76 \pm 0.55$ </td><td> $88.24 \pm 0.30$ </td><td> $93.34 \pm 0.56$ </td><td> $97.66 \pm 0.52$ </td><td> $98.86 \pm 0.28$ </td><td> $99.64 \pm 0.08$ </td></tr></table>

We verify the effectiveness of framework components through ablation studies in Section 5.4. Intuitive visualizations of the edge selection curriculum are shown in Section 5.5. In addition, we measure the parameter sensitivity in Appendix A.2 and running time analysis in Appendix A.5 due to the space limit.

# 5.1 Experimental Settings

Synthetic datasets. To evaluate the effectiveness of our proposed method on datasets with ground-truth difficulty labels on edges, we follow previous studies $[22, 1]$ to generate a set of synthetic datasets, where the formation probability of an edge is designed to reflect its likelihood to positively contribute to the node classification job, which indicates its ground-truth difficulty level. Specifically, the nodes in a generated graph are divided into 10 equally sized node classes $1, 2, \ldots, 10$ , and the node features are sampled from overlapping multi-Gaussian distributions. Each generated graph is associated with a homophily coefficient (homo) which indicates the probability of a node forming an edge to another node with the same label. For the rest edges that are formed between nodes with different labels, the probability of forming an edge is inversely proportional to the distances between their labels. Nodes with close classes are more likely to be connected since the formation probability decreases with the distance of the node label, and connections from nodes with close classes can increase the likelihood of accurately classifying a node due to the homophily property of the designed node classification task. Therefore, an edge with a high formation probability indicates a higher chance to positively contribute to the node classification task because it connects a node with a close class, and thus can be considered an easy edge. We vary the value of homo to generate nine graphs in total. More details and visualization about the synthetic dataset can be found in Appendix A.1.

Real-world datasets. To further evaluate the performance of our proposed method in real-world scenarios, nine benchmark real-world attributed network datasets, including four citation network datasets Cora, Citeseer, Pubmed $[51]$ and ogbn-arxiv $[16]$ , two coauthor network datasets CS and Physics $[34]$ , two Amazon co-purchase network datasets Photo and Computers $[34]$ , and one protein interaction network ogbn-proteins $[16]$ . We follow the data splits from $[3]$ on citation networks and use a 5-fold cross-validation setting on coauthor and Amazon co-purchase networks. All datasets are publicly available from Pytorch-geometric library $[10]$ and Open Graph Benchmark (OGB) $[16]$ , where basic statistics are reported in Table 2.

Comparison methods. We incorporate three commonly used GNN models, including GCN [23], GraphSAGE [14], and GIN [50], as the baseline model and also the backbone model for RCL. In addition to evaluating our proposed method against the baseline GNNs, we further leverage two categories of state-of-the-art comparison methods in the experiments: (1) We incorporate four graph structure learning methods GNNSVD [9], ProGNN [20], NeuralSparse [54], and PTDNet [31]; (2) We further compare with a curriculum learning method named CLNode [45] which gradually select nodes in the order of the difficulties defined by a heuristic-based strategy. More details about the comparison methods can be found in Appendix A.1.

Table 2: Node classification results on real-world datasets (%). The best-performing method on each backbone is highlighted in bold and second-best is underlined. (OOM) shorts for out-of-memory. 

<table><tr><td></td><td>Cora</td><td>Citeseer</td><td>Pubmed</td><td>CS</td><td>Physics</td><td>Photo</td><td>Computers</td><td>ogbn-arxiv</td><td>ogbn-proteins</td></tr><tr><td># nodes</td><td>2,708</td><td>3,327</td><td>19,717</td><td>18,333</td><td>34,493</td><td>7,650</td><td>13,752</td><td>169,343</td><td>132,534</td></tr><tr><td># edges</td><td>10,556</td><td>9,104</td><td>88,648</td><td>163,788</td><td>495,924</td><td>238,162</td><td>491,722</td><td>1,166,243</td><td>39,561,252</td></tr><tr><td># features</td><td>1,433</td><td>3,703</td><td>500</td><td>6,805</td><td>8,415</td><td>745</td><td>767</td><td>100</td><td>8</td></tr><tr><td>GCN</td><td> $85.74 \pm 0.42$ </td><td> $78.93 \pm 0.32$ </td><td> $87.91 \pm 0.09$ </td><td> $93.03 \pm 0.32$ </td><td> $96.55 \pm 0.15$ </td><td> $93.25 \pm 0.70$ </td><td> $88.09 \pm 0.40$ </td><td> $71.74 \pm 0.29$ </td><td> $72.51 \pm 0.35$ </td></tr><tr><td>GNNSVD</td><td> $83.24 \pm 1.03$ </td><td> $74.80 \pm 0.87$ </td><td> $88.81 \pm 0.38$ </td><td> $93.79 \pm 0.11$ </td><td> $96.11 \pm 0.13$ </td><td> $89.63 \pm 0.73$ </td><td> $86.49 \pm 0.77$ </td><td> $67.44 \pm 0.51$ </td><td> $66.92 \pm 0.64$ </td></tr><tr><td>ProGNN</td><td> $85.66 \pm 0.61$ </td><td> $74.78 \pm 0.55$ </td><td> $87.22 \pm 0.33$ </td><td> $94.04 \pm 0.19$ </td><td> $96.75 \pm 0.26$ </td><td> $92.07 \pm 0.67$ </td><td> $88.72 \pm 0.59$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>NeuralSparse</td><td> $\underline{85.95 \pm 0.98}$ </td><td> $76.24 \pm 0.48$ </td><td> $86.83 \pm 0.40$ </td><td> $\underline{92.31 \pm 0.47}$ </td><td> $\underline{95.56 \pm 0.30}$ </td><td> $90.57 \pm 0.90$ </td><td> $88.62 \pm 0.83$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>PTDNet</td><td> $83.84 \pm 0.95$ </td><td> $77.54 \pm 0.42$ </td><td> $87.89 \pm 0.08$ </td><td> $93.60 \pm 0.43$ </td><td> $96.56 \pm 0.09$ </td><td> $88.92 \pm 0.87$ </td><td> $87.52 \pm 0.70$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>CLNode</td><td> $85.67 \pm 0.33$ </td><td> $78.99 \pm 0.57$ </td><td> $89.50 \pm 0.28$ </td><td> $93.83 \pm 0.24$ </td><td> $95.76 \pm 0.16$ </td><td> $\underline{93.39 \pm 0.83}$ </td><td> $89.28 \pm 0.38$ </td><td> $70.95 \pm 0.18$ </td><td> $71.40 \pm 0.32$ </td></tr><tr><td>RCL</td><td> $\underline{87.15 \pm 0.44}$ </td><td> $\underline{79.79 \pm 0.55}$ </td><td> $\underline{89.79 \pm 0.12}$ </td><td> $\underline{94.66 \pm 0.32}$ </td><td> $\underline{97.02 \pm 0.23}$ </td><td> $\underline{94.41 \pm 0.76}$ </td><td> $\underline{90.23 \pm 0.23}$ </td><td> $74.08 \pm 0.33$ </td><td> $75.19 \pm 0.26$ </td></tr><tr><td>GIN</td><td> $84.43 \pm 0.65$ </td><td> $74.87 \pm 0.20$ </td><td> $85.72 \pm 0.40$ </td><td> $91.48 \pm 0.36$ </td><td> $95.62 \pm 0.30$ </td><td> $93.02 \pm 0.91$ </td><td> $86.94 \pm 1.58$ </td><td> $69.26 \pm 0.34$ </td><td> $74.51 \pm 0.32$ </td></tr><tr><td>GNNSVD</td><td> $82.23 \pm 0.65$ </td><td> $72.11 \pm 0.70$ </td><td> $88.31 \pm 0.15$ </td><td> $91.40 \pm 0.87$ </td><td> $95.30 \pm 0.29$ </td><td> $89.49 \pm 1.11$ </td><td> $82.66 \pm 2.26$ </td><td> $67.79 \pm 0.41$ </td><td> $\underline{70.65 \pm 0.53}$ </td></tr><tr><td>ProGNN</td><td> $85.02 \pm 0.41$ </td><td> $\underline{78.12 \pm 0.93}$ </td><td> $87.82 \pm 0.51$ </td><td>(OOM)</td><td>(OOM)</td><td> $92.23 \pm 0.67$ </td><td> $83.54 \pm 1.48$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>NeuralSparse</td><td> $\underline{84.92 \pm 0.58}$ </td><td> $75.44 \pm 0.87$ </td><td> $86.11 \pm 0.49$ </td><td> $89.66 \pm 0.82$ </td><td> $95.05 \pm 0.57$ </td><td> $\underline{93.28 \pm 0.83}$ </td><td> $87.22 \pm 0.54$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>PTDNet</td><td> $83.02 \pm 1.01$ </td><td> $75.00 \pm 0.74$ </td><td> $88.04 \pm 0.29$ </td><td> $91.01 \pm 0.21$ </td><td> $95.57 \pm 0.40$ </td><td> $90.70 \pm 0.76$ </td><td> $87.08 \pm 0.65$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>CLNode</td><td> $83.52 \pm 0.77$ </td><td> $75.82 \pm 0.58$ </td><td> $86.92 \pm 0.61$ </td><td> $91.71 \pm 0.41$ </td><td> $95.75 \pm 0.46$ </td><td> $92.78 \pm 0.90$ </td><td> $85.93 \pm 1.53$ </td><td> $70.58 \pm 0.17$ </td><td> $73.97 \pm 0.31$ </td></tr><tr><td>RCL</td><td> $\underline{86.64 \pm 0.39}$ </td><td> $77.60 \pm 0.18$ </td><td> $\underline{89.17 \pm 0.29}$ </td><td> $\underline{93.92 \pm 0.27}$ </td><td> $\underline{96.75 \pm 0.17}$ </td><td> $\underline{93.88 \pm 0.51}$ </td><td> $\underline{89.76 \pm 0.19}$ </td><td> $\underline{72.55 \pm 0.15}$ </td><td> $\underline{78.76 \pm 0.22}$ </td></tr><tr><td>GraphSAGE</td><td> $86.22 \pm 0.27$ </td><td> $77.27 \pm 0.23$ </td><td> $88.50 \pm 0.16$ </td><td> $94.22 \pm 0.18$ </td><td> $96.26 \pm 0.34$ </td><td> $93.82 \pm 0.51$ </td><td> $88.62 \pm 0.21$ </td><td> $71.49 \pm 0.27$ </td><td> $77.68 \pm 0.20$ </td></tr><tr><td>GNNSVD</td><td> $83.11 \pm 0.82$ </td><td> $\underline{73.19 \pm 0.49}$ </td><td> $88.42 \pm 0.38$ </td><td> $\underline{93.86 \pm 0.36}$ </td><td> $95.96 \pm 0.12$ </td><td> $89.31 \pm 0.53$ </td><td> $81.46 \pm 1.15$ </td><td> $69.82 \pm 0.34$ </td><td> $71.82 \pm 0.39$ </td></tr><tr><td>ProGNN</td><td> $86.23 \pm 0.42$ </td><td> $74.45 \pm 0.83$ </td><td> $88.52 \pm 0.45$ </td><td>(OOM)</td><td>(OOM)</td><td> $90.89 \pm 0.69$ </td><td> $89.34 \pm 0.54$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>NeuralSparse</td><td> $84.60 \pm 0.52$ </td><td> $76.32 \pm 0.55$ </td><td> $89.02 \pm 0.39$ </td><td> $93.89 \pm 0.58$ </td><td> $96.67 \pm 0.20$ </td><td> $90.78 \pm 1.06$ </td><td> $88.37 \pm 0.37$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>PTDNet</td><td> $86.03 \pm 0.60$ </td><td> $76.07 \pm 0.58$ </td><td> $\underline{86.78 \pm 0.45}$ </td><td> $93.78 \pm 0.43$ </td><td> $95.32 \pm 0.31$ </td><td> $92.96 \pm 0.87$ </td><td> $84.89 \pm 1.47$ </td><td>(OOM)</td><td>(OOM)</td></tr><tr><td>CLNode</td><td> $86.60 \pm 0.64$ </td><td> $77.23 \pm 0.54$ </td><td> $88.76 \pm 0.57$ </td><td> $94.13 \pm 0.34$ </td><td> $96.87 \pm 0.45$ </td><td> $93.90 \pm 0.42$ </td><td> $89.57 \pm 0.62$ </td><td> $71.54 \pm 0.20$ </td><td> $78.40 \pm 0.41$ </td></tr><tr><td>RCL</td><td> $\underline{86.90 \pm 0.39}$ </td><td> $78.95 \pm 0.18$ </td><td> $90.14 \pm 0.43$ </td><td> $95.05 \pm 0.23$ </td><td> $\underline{96.88 \pm 0.19}$ </td><td> $95.06 \pm 0.52$ </td><td> $\underline{90.47 \pm 0.38}$ </td><td> $\underline{73.13 \pm 0.14}$ </td><td> $\underline{79.89 \pm 0.35}$ </td></tr></table>

Initializing graph structure by a pre-trained model. It is worth noting that the model needs an initial training graph structure $\mathbf{A}^{(0)}$ in the initial stage of training. An intuitive way is that we can initialize the model to work in a purely data-driven scenario that starts only with isolated nodes where no edges exist. However, an instructive initial structure can greatly reduce the search cost and computational burden. Inspired by many previous CL works [46, 13, 19, 55] that incorporate prior knowledge of a pre-trained model into designing curriculum for the current model, we initialize the training structure $\mathbf{A}^{(0)}$ by a pre-trained vanilla GNN model $f^{*}$ . Specifically, we follow the same steps from line 4 to line 7 in the algorithm 1 to obtain the initial training structure $\mathbf{A}^{(0)}$ but the latent node embedding is extracted from the pre-trained model $f^{*}$ .

Implementation details. We use the baseline model (GCN, GIN, GraphSAGE) as the backbone model for both our RCL method and all comparison methods. For a fair comparison, we require all models follow the same GNN architecture with two convolution layers. For each split, we run each model 10 times to reduce the variance in particular data splits. Test results are according to the best validation results. General training hyperparameters (such as learning rate or the number of training epochs) are equal for all models.

# 5.2 Effectiveness Results

Table 1 presents the node classification results of the synthetic datasets. We report the average accuracy and standard deviation for each model against the homo of generated graphs. From the table, we observe that our proposed method RCL consistently achieves the best or most competitive performance to all the comparison methods over three backbone GNN architectures. Specifically, RCL outperforms the second best method on average by $4.17\%$ , $2.60\%$ , and $1.06\%$ on GCN, GIN, and GraphSAGE backbones, respectively. More importantly, the proposed RCL method performs significantly better than the second best model when the homo of generated graphs is low ( $\leq 0.5$ ), on average by $6.55\%$ on GCN, $4.17\%$ on GIN, and $2.93\%$ on GraphSAGE backbones. These demonstrate that our proposed RCL method significantly improves the model's capability of learning an effective representation to downstream tasks especially when the edge difficulties vary largely in the data.

We report the experimental results of the real-world datasets in Table 2. The results demonstrate the strength of our proposed method by consistently achieving the best results in all 9 datasets by GCN backbone architecture, all 9 datasets by GraphSAGE backbone architecture, and 8 out of 9 datasets by GIN backbone architecture. Specifically, our proposed method improved the performance of baseline models on average by 1.86%, 2.83%, and 1.62% over GCN, GIN, and GraphSAGE, and outperformed the second best models model on average by 1.37%, 2.49%, and 1.22% over the three backbone models, respectively. The results demonstrate that the proposed RCL method consistently improves the performance of GNN models in real-world scenarios.

Our experimental results are statically sound. In 43 out of 48 tasks our method outperforms the second-best performing model with strong statistical significance. Specifically, we have in 30 out of 43 cases with a significance p < 0.001, in 8 out of 43 cases with a significance p < 0.01, and in 5

Figure 2: Node classification accuracy (%) on Cora and Citeseer under random structure attack. The attack edge ratio is computed versus the original number of edges, where 100% means that the number of inserted edges is equal to the number of original edges.   
![](images/d873baf86e174b69ce2e82d3fb1bcd026445cd1c2f3f22bf849dcbbe2deaa5d7.jpg)

Table 3: Ablation study. Here “Full” represents the original method without removing any component. The best-performing method on each dataset is highlighted in bold. 

<table><tr><td></td><td>Synthetic1</td><td>Synthetic2</td><td>Citeseer</td><td>CS</td><td>Computers</td></tr><tr><td>Full</td><td>73.98±0.55</td><td>97.42±0.17</td><td>79.79±0.55</td><td>94.66±0.22</td><td>90.23±0.23</td></tr><tr><td>Curriculum-linear</td><td>70.93±0.54</td><td>95.19±0.19</td><td>79.04±0.38</td><td>94.14±0.26</td><td>89.28±0.21</td></tr><tr><td>Curriculum-root</td><td>70.13±0.72</td><td>95.50±0.18</td><td>78.27±0.54</td><td>94.47±0.34</td><td>89.27±0.15</td></tr><tr><td>Random-linear</td><td>58.76±0.46</td><td>89.78±0.11</td><td>77.43±0.49</td><td>92.76±0.14</td><td>88.76±0.18</td></tr><tr><td>Random-root</td><td>61.04±0.20</td><td>91.04±0.09</td><td>76.81±0.35</td><td>92.92±0.15</td><td>88.81±0.28</td></tr><tr><td>w/o edge appearance</td><td>70.70±0.43</td><td>95.77±0.16</td><td>77.77±0.65</td><td>94.39±0.21</td><td>89.56±0.30</td></tr><tr><td>w/o node confidence</td><td>72.38±0.41</td><td>96.86±0.17</td><td>78.72±0.72</td><td>94.34±0.13</td><td>90.03±0.62</td></tr><tr><td>w/o pre-trained model</td><td>72.56±0.69</td><td>93.89±0.14</td><td>78.28±0.77</td><td>94.50±0.14</td><td>89.80±0.55</td></tr></table>

out of 43 cases with a significance $p < 0.05$ . Such statistical significance results can demonstrate that our proposed method can consistently perform better than the baseline models in both scenarios.

# 5.3 Robustness Analysis Against Topological Noise

To further examine the robustness of the RCL method on extracting powerful representation from correlated data samples, we follow previous works $[20, 31]$ to randomly inject fake edges into real-world graphs. This adversarial attack can be viewed as adding random noise to the topological structure of graphs. Specifically, we randomly connect M pairs of previously unlinked nodes in the real-world datasets, where the value of M varies from 10% to 100% of the original edges. We then train RCL and all the comparison methods on the attacked graph and evaluate the node classification performance. The results are shown in Figure 2, we can observe that RCL shows strong robustness to adversarial structural attacks by consistently outperforming all compared methods on all datasets. Especially, when the proportion of added noisy edges is large ( $>50\%$ ), the improvement becomes more significant. For instance, under the extremely noisy ratio at 100%, RCL outperforms the second best model by 4.43% and 2.83% on Cora dataset, and by 6.13%, 3.47% on Citeseer dataset, with GCN and GIN backbone models, respectively.

# 5.4 Ablation Study

To investigate the effectiveness of our proposed model with some simpler heuristics, we deploy a series of abalation analysis. We first train the model with node classification task purely and select the top K expected edges as suggested by the reviewer. Specifically, we follow previous works $[43, 45]$ using two classical selection pacing functions as follows:

$$
\text { Linear: } K _ {\text { linear }} (t) = \frac {t}{T} | E |; \quad \text { Root: } K _ {\text { root }} (t) = \sqrt {\frac {t}{T}} | E |,
$$

where t is the number of current iterations and T is the number of total iterations, and $|E|$ is the number of total edges. We name these two variants Curriculum-linear and Curriculum-root, respectively. In addition, we also remove the edge difficulty measurement module and use random selection instead. Specifically, we gradually incorporate more edges into training in random order to verify the effectiveness of the learned curriculum. We name two variants as Random-linear and Random-root with the above two mentioned pacing functions, respectively.

In order to further investigate the impact of the proposed components of RCL. We also first consider variants of removing the edge smoothing components mentioned in Section 4.3. Specifically, we

![](images/d29c1c27a84169e434a0e6048f521a7eccde7861f20215b86f7176198f509ffc.jpg)

<details>
<summary>area</summary>

| Model    | Easy  | Medium | Hard  |
| -------- | ----- | ------ | ----- |
| GCN      | 0.2   | 0.3    | 0.4   |
| GIN      | 0.2   | 0.3    | 0.4   |
| GraphSage| 0.2   | 0.3    | 0.4   |
</details>

Figure 3: Visualization of edge selection process during training.

consider two variants w/o EC and w/o NC, which remove the smoothing function of the edge occurrence ratio and the component to reflect the degree of confidence for the latent node embedding in RCL, respectively. In addition to examining the effectiveness of edge smoothing components, we further consider a variant w/o pre-trained model that avoids using a pre-trained model to initialize model, which is mentioned in Section 5.1, to initialize the training structure by a pre-trained model and instead starts with inferred structure from isolated nodes with no connections.

We present the results of two synthetic datasets (homophily coefficient=0.3, 0.6) and three real-world datasets in Table 3. We summarize our findings from the above table as below: (i) Our full model consistently outperforms the two variants Curriculum-linear and Curriculum-root by an average of 1.59% on all datasets, suggesting that our pacing module can benefit model training. It is worth noting that these two variants also outperform the baseline vanilla GNN model Vanilla by an average of 1.92%, which supports the assumption that even a simple curriculum learning strategy can still improve model performance. (ii) We observe that the performance of the two variants Random-linear and Random-root on all datasets drops by 3.86% on average compared to the variants Curriculum-linear and Curriculum-root. Such behavior demonstrates the effectiveness of our proposed edge difficulty quantification module by showing that randomly involving edges into training cannot benefit model performance. (iii) We can observe a significant performance drop consistently for all variants that remove the structural smoothing techniques and initialization components. The results validate that all structural smoothing and initialization components can benefit the performance of RCL on the downstream tasks.

# 5.5 Visualization of Learned Edge Selection Curriculum

Besides the effectiveness and robustness of the RCL method on downstream classification results, it is also interesting to verify whether the learned edge selection curriculum satisfies the rule from easy to hard. Since real-world datasets do not have ground-truth labels of difficulty on edges, we conduct visualization experiments on synthetic datasets, where the difficulty of each edge can be indicated by its formation probability. Specifically, we classify edges into three balanced categories according to their difficulty: easy, medium, and hard. Here, we define all homogenous edges that connect nodes with the same class as easy, edges connecting nodes with adjacent classes as medium, and the remaining edges connecting nodes with far away classes as hard. We report the proportion of edges selected for each category during training in Figure 3. We can observe that RCL can effectively select most of the easy edges at the early stage of training, then more easy edges and most medium edges are gradually included during training, and most hard edges are left unselected until the end stage of training. Such edge selection behavior is highly consistent with the core idea of designing a curriculum for edge selection, which verifies that our proposed method can effectively design curriculums to select edges according to their difficulty from easy to hard.

# 6 Conclusion

This paper focuses on developing a novel CL method to improve the generalization ability and robustness of GNN models on learning representations of data samples with dependencies. The proposed method Relational Curriculum Learning (RCL) effectively addresses the unique challenges in designing CL strategy for handling dependencies. First, a self-supervised learning module is developed to select appropriate edges that are expected by the model. Then an optimization model is presented to iteratively increment the edges according to the model training status and a theoretical guarantee of the convergence on the optimization algorithm is given. Finally, an edge reweighting scheme is proposed to steady the numerical process by smoothing the training structure transition. Extensive experiments on synthetic and real-world datasets demonstrate the strength of RCL in improving the generalization ability and robustness.

# Acknowledgement

This work was supported by the National Science Foundation (NSF) Grant No. 1755850, No. 1841520, No. 2007716, No. 2007976, No. 1942594, No. 1907805, a Jeffress Memorial Trust Award, Amazon Research Award, NVIDIA GPU Grant, and Design Knowledge Company (subcontract number: 10827.002.120.04). The authors acknowledge Emory Computer Science department for providing computational resources and technical support that have contributed to the experimental results reported within this paper.

# References

[1] Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. Mixhop: Higher-order graph convolutional architectures via sparsified neighborhood mixing. In international conference on machine learning, pages 21–29. PMLR, 2019.   
[2] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.   
[3] Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: Fast learning with graph convolutional networks via importance sampling. In International Conference on Learning Representations, 2018.   
[4] Yu Chen, Lingfei Wu, and Mohammed Zaki. Iterative deep graph learning for graph neural networks: Better and robust node embeddings. Advances in neural information processing systems, 33:19314–19326, 2020.   
[5] Guanyi Chu, Xiao Wang, Chuan Shi, and Xunqiang Jiang. Cuco: Graph representation with curriculum contrastive learning. In IJCAI, pages 2300–2306, 2021.   
[6] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. Principal neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems, 33:13260–13271, 2020.   
[7] Hanjun Dai, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song. Adversarial attack on graph structured data. In International conference on machine learning, pages 1115–1124. PMLR, 2018.   
[8] Jeffrey L Elman. Learning and development in neural networks: The importance of starting small. Cognition, 48(1):71–99, 1993.   
[9] Negin Entezari, Saba A Al-Sayouri, Amirali Darvishzadeh, and Evangelos E Papalexakis. All you need is low (rank) defending against adversarial attacks on graphs. In Proceedings of the 13th International Conference on Web Search and Data Mining, pages 169–177, 2020.   
[10] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.   
[11] Tieliang Gong, Qian Zhao, Deyu Meng, and Zongben Xu. Why curriculum learning & self-paced learning work in big/noisy data: A theoretical perspective. Big Data & Information Analytics, 1(1):111, 2016.   
[12] Xiaojie Guo, Shiyu Wang, and Liang Zhao. Graph neural networks: Graph transformation. Graph Neural Networks: Foundations, Frontiers, and Applications, pages 251–275, 2022.   
[13] Guy Hacohen and Daphna Weinshall. On the power of curriculum learning in training deep networks. In International Conference on Machine Learning, pages 2535–2544. PMLR, 2019.   
[14] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.   
[15] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31, 2018.   
[16] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv preprint arXiv:2005.00687, 2020.

[17] Lu Jiang, Deyu Meng, Shoou-I Yu, Zhenzhong Lan, Shiguang Shan, and Alexander Hauptmann. Self-paced learning with diversity. Advances in neural information processing systems, 27, 2014.   
[18] Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander G Hauptmann. Self-paced curriculum learning. In Twenty-ninth AAAI conference on artificial intelligence, 2015.   
[19] Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International conference on machine learning, pages 2304–2313. PMLR, 2018.   
[20] Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, and Jiliang Tang. Graph structure learning for robust graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 66–74, 2020.   
[21] Wei Ju, Zheng Fang, Yiyang Gu, Zequn Liu, Qingqing Long, Ziyue Qiao, Yifang Qin, Jianhao Shen, Fang Sun, Zhiping Xiao, et al. A comprehensive survey on deep graph representation learning. arXiv preprint arXiv:2304.05055, 2023.   
[22] Fariba Karimi, Mathieu Génois, Claudia Wagner, Philipp Singer, and Markus Strohmaier. Homophily influences ranking of minorities in social networks. Scientific reports, 8(1):1–12, 2018.   
[23] Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of the 5th International Conference on Learning Representations, 2017.   
[24] Yajing Kong, Liu Liu, Jun Wang, and Dacheng Tao. Adaptive curriculum learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5067-5076, 2021.   
[25] M Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. Advances in neural information processing systems, 23, 2010.   
[26] Haoyang Li, Xin Wang, and Wenwu Zhu. Curriculum graph machine learning: A survey. arXiv preprint arXiv:2302.02926, 2023.   
[27] Qiuwei Li, Zhihui Zhu, and Gongguo Tang. Alternating minimizations converge to second-order optimal solutions. In International Conference on Machine Learning, pages 3935-3943. PMLR, 2019.   
[28] Xiaohe Li, Lijie Wen, Yawen Deng, Fuli Feng, Xuming Hu, Lei Wang, and Zide Fan. Graph neural network with curriculum learning for imbalanced node classification. arXiv preprint arXiv:2202.02529, 2022.   
[29] Chen Ling, Junji Jiang, Junxiang Wang, and Zhao Liang. Source localization of graph diffusion via variational autoencoders for graph inverse problems. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 1010–1020, 2022.   
[30] Chen Ling, Junji Jiang, Junxiang Wang, My T Thai, Renhao Xue, James Song, Meikang Qiu, and Liang Zhao. Deep graph representation learning and optimization for influence maximization. In International Conference on Machine Learning, pages 21350–21361. PMLR, 2023.   
[31] Dongsheng Luo, Wei Cheng, Wenchao Yu, Bo Zong, Jingchao Ni, Haifeng Chen, and Xiang Zhang. Learning to drop: Robust graph neural network via topological denoising. In Proceedings of the 14th ACM international conference on web search and data mining, pages 779–787, 2021.   
[32] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.   
[33] Douglas LT Rohde and David C Plaut. Language acquisition in the absence of explicit negative evidence: How important is starting small? Cognition, 72(1):67–109, 1999.   
[34] Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Günnemann. Pitfalls of graph neural network evaluation. arXiv preprint arXiv:1811.05868, 2018.

[35] Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769, 2016.   
[36] Charles Sutton, Andrew McCallum, et al. An introduction to conditional random fields. Foundations and Trends® in Machine Learning, 4(4):267–373, 2012.   
[37] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11:1201–1242, 2010.   
[38] Junxiang Wang, Junji Jiang, and Liang Zhao. An invertible graph diffusion neural network for source localization. In Proceedings of the ACM Web Conference 2022, pages 1058-1069, 2022.   
[39] Junxiang Wang, Hongyi Li, Zheng Chai, Yongchao Wang, Yue Cheng, and Liang Zhao. Toward quantized model parallelism for graph-augmented mlps based on gradient-free admm framework. IEEE Transactions on Neural Networks and Learning Systems, 2022.   
[40] Junxiang Wang, Hongyi Li, and Liang Zhao. Accelerated gradient-free neural network training by multi-convex alternating optimization. Neurocomputing, 487:130–143, 2022.   
[41] Junxiang Wang, Fuxun Yu, Xiang Chen, and Liang Zhao. Admm for efficient deep learning with global convergence. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 111–119, 2019.   
[42] Shiyu Wang, Xiaojie Guo, and Liang Zhao. Deep generative model for periodic graphs. Advances in Neural Information Processing Systems, 35, 2022.   
[43] Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):4555–4576, 2021.   
[44] Yiwei Wang, Wei Wang, Yuxuan Liang, Yujun Cai, and Bryan Hooi. Curgraph: Curriculum learning for graph classification. In Proceedings of the Web Conference 2021, pages 1238–1248, 2021.   
[45] Xiaowen Wei, Weiwei Liu, Yibing Zhan, Du Bo, and Wenbin Hu. Clnode: Curriculum learning for node classification. arXiv preprint arXiv:2206.07258, 2022.   
[46] Daphna Weinshall, Gad Cohen, and Dan Amir. Curriculum learning by transfer learning: Theory and experiments with deep networks. In International Conference on Machine Learning, pages 5238–5246. PMLR, 2018.   
[47] Richard Lee Wheeden and Antoni Zygmund. Measure and integral, volume 26. Dekker New York, 1977.   
[48] Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu, and Liming Zhu. Adversarial examples for graph data: deep insights into attack and defense. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, pages 4816–4823, 2019.   
[49] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4–24, 2020.   
[50] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2018.   
[51] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In International conference on machine learning, pages 40–48. PMLR, 2016.   
[52] Zheng Zhang and Liang Zhao. Representation learning on spatial networks. Advances in Neural Information Processing Systems, 34:2303–2318, 2021.   
[53] Zheng Zhang and Liang Zhao. Unsupervised deep subgraph anomaly detection. In 2022 IEEE International Conference on Data Mining (ICDM), pages 753–762. IEEE, 2022.   
[54] Cheng Zheng, Bo Zong, Wei Cheng, Dongjin Song, Jingchao Ni, Wenchao Yu, Haifeng Chen, and Wei Wang. Robust graph representation learning via neural sparsification. In International Conference on Machine Learning, pages 11458–11468. PMLR, 2020.   
[55] Tianyi Zhou, Shengjie Wang, and Jeff Bilmes. Robust curriculum learning: from clean label detection to noisy label self-correction. In International Conference on Learning Representations, 2020.

[56] Daniel Zügner, Amir Akbarnejad, and Stephan Günnemann. Adversarial attacks on neural networks for graph data. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2847–2856, 2018.

# A Additional Experimental Settings and Results

# A.1 Additional Experimental Settings

![](images/76419ee72aaca760f9d9c07c12e8a53d69930a777d307dc7e46fef729533cf8c.jpg)

<details>
<summary>scatter</summary>

| x | y | color |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | (data not extractable) |
</details>

Figure 4: Visualization of synthetic datasets. Each color represents a class of nodes. Node attributes are sampled from overlapping multi-Gaussian distributions, where the attributes of nodes with close labels are likely to have short distances. Homogeneous edges represent edges that connect nodes of the same class (with the same color). The probability of connecting two nodes of different classes decreases with the distance between the center points of their class distribution. Therefore, the formation probability of a node denotes the edge difficulty, since edges between nodes with close classes are more likely to positively contribute to the prediction under the homogeneous assumption.

Synthetic datasets. To evaluate the effectiveness of our proposed method on datasets with ground-truth difficulty labels on structure, we first follow previous studies [22, 1] to generate a set of synthetic datasets, where the difficulty of edges in generated graphs are indicated by their formation probability. Specifically, as shown in Figure 4, each generated graph is with 5,000 nodes, which are divided into 10 equally sized node classes 1, 2, ..., 10. The node features are sampled from overlapping multi-Gaussian distributions. Each generated graph is associated with a homophily coefficient (homo) which indicates the likelihood of a node forming a connection to another node with the same label (same color in Figure 4). For example, a generated graph with homo = 0.5 will have on average half of the edges formed between nodes with the same label. For the rest edges that are formed between nodes with different labels (different colors in Figure 4), the probability of forming an edge is inversely proportional to the distances between their labels. Mathematically, the probability of forming an edge between node u and node v follows $p_{u \to v} \propto e^{-|c_u - c_v|}$ , where the distances between labels $|c_u - c_v|$ means shortest distance of two classes on a circle. Therefore, the probability of forming an edge in the synthetic graph can reflect how well this edge is expected. Specifically, edges with a higher formation probability, e.g. connecting nodes with the same label or close labels, meaning that there is a higher chance that this connection will positively contribute to the prediction (less chance to be a noisy edge). Conversely, edges with a lower formation probability, e.g., connecting nodes with faraway labels, mean that there is a higher chance that this connection will negatively contribute to the prediction (higher chance to be a noisy edge). We vary the value of homo from 0.1, 0.2, ..., 0.9 to generate nine graphs in total. Similar to previous works [22, 1], we randomly partition each synthetic graph into equal-sized train, validation, and test node splits.

Implementation Details. We use the baseline model (GCN, GIN, GraphSage) as the backbone model for both our RCL method and all comparison methods. For a fair comparison, we require all models follow the same GNN architecture with two convolution layers. For each split, we run each model 10 times to reduce the variance in particular data splits. Test results are according to the best validation results. General training hyperparameters (such as learning rate or the number of training

epochs) are equal for all models. For the pre-trained model to initialize the training structure, we utilize the same model as the backbone model utilized by our method. For example, if we use GCN as the backbone model for RCL, the pre-trained model to initialize is also GCN. All experiments are conducted on a 64-bit machine with four NVIDIA Quadro RTX 8000 GPUs. The proposed method is implemented with Pytorch deep learning framework [32].

The following describes the details of our comparison models.

Graph Neural Networks (GNNs). We first introduce three baseline GNN models as follows.

(i) GCN. Graph Convolutional Networks (GCN) [23] is a commonly used GNN, which introduces a first-order approximation architecture of the Chebyshev spectral convolution operator;   
(ii) GIN. Graph Isomorphism Networks (GIN) [50] is a variant of GNN, which has provably powerful discriminating power among the class of 1-order GNNs;   
(iii) GraphSage. GraphSage [14] is a GNN method that computes the hidden representation of the root node by aggregating the hidden node representations hierarchically from bottom to top.

Graph structure learning. We then introduce four state-of-the-art methods for jointly learning the optimal graph structure and downstream tasks.

(i) GNNSVD. GNNSVD [9] first apply singular value decomposition (SVD) on the graph adjacency matrix to obtain a low-rank graph structure and apply GNN on the obtained low-rank structure;   
(ii) ProGNN. ProGNN [20] is a method to defend against graph adversarial attacks by obtaining a sparse and low-rank graph structure from the input structure;   
(iii) NeuralSparse. NeuralSparse [54] is a method to learn robust graph representations by iteratively sampling k-neighbor subgraphs for each node and sparsing the graph according to the performance on the node classification;   
(iv) PTDNet. PTDNet [31] learns a sparsified graph by pruning task-irrelevant edges, where sparsity is controlled by regulating the number of edges.

Curriculum learning on graph data. We introduce a recent curriculum learning work on node classification as follows.

(i) CLNode. CLNode [45] regards nodes as data samples and gradually incorporates more nodes into training according to their difficulty. They apply a heuristic-based strategy to measure the difficulty of nodes, where the nodes that connect neighboring nodes with different classes are considered difficult.

# Searching space for hyperparameters.

Number of epochs trained: {150, 500};

Learning rate for model: $\{1e-2,5e-3,1e-3\}$ ;

Number of GNN layers: {2};

Dimension of hidden state: {64};

Age parameter $\lambda : \{1, 2, 3, 4, 5\}$ (A larger value indicates faster pacing for adding edges, where 1 denotes the training structure will converge to the input structure at the final iteration).

# A.2 Additional Effectiveness Experiments on Heterophilic Datasets

<table><tr><td>Dataset</td><td>Edge homo ratio</td><td>GCN</td><td>GCN-RCL</td><td>GIN</td><td>GIN-RCL</td></tr><tr><td>Texas</td><td>0.11</td><td>0.5645</td><td>0.6006</td><td>0.5885</td><td>0.6156</td></tr><tr><td>Cornell</td><td>0.30</td><td>0.4084</td><td>0.5045</td><td>0.4234</td><td>0.4925</td></tr><tr><td>Wisconsin</td><td>0.21</td><td>0.4923</td><td>0.5294</td><td>0.5141</td><td>0.5599</td></tr><tr><td>Actor</td><td>0.22</td><td>0.2868</td><td>0.3186</td><td>0.2678</td><td>0.3006</td></tr><tr><td>Squirrel</td><td>0.22</td><td>0.2743</td><td>0.2999</td><td>0.2347</td><td>0.2519</td></tr><tr><td>Chameleon</td><td>0.23</td><td>0.3625</td><td>0.4385</td><td>0.3233</td><td>0.4033</td></tr></table>

Table 4: Node classification results for six real-world heterophilic datasets, where the best performance of each model category in one dataset is highlighted.

In order to further verify the effectiveness of our proposed strategy on heterophilic graph datasets, we have included new experiments on six real-world heterophilic datasets. As shown in Table 4, our method consistently improve performance of backbone GNN models on these heterophilic datasets. Specifically, RCL outperforms the second best method on average by 5.04%, and 4.55%, on GCN and GIN backbones, respectively. The results can demonstrate our method is not limited to homophily graphs.

Although the inner product decoder utilized in experiments might imply an underlying homophily assumption, our method can still benefit from leveraging the edge curriculum present within the input datasets. A reasonable explanation is that standard GNN models are usually struggled with the heterophily edges, while our methodology designs a curriculum allowing more focus on homophily edges, which potentially leads to the observed performance boost.

# A.3 Additional Effectiveness Experiments on PNA Backbone Model.

<table><tr><td>Dataset</td><td>PNA</td><td>PNA-RCL</td><td>PNA-linear</td><td>PNA-root</td><td>GCN</td><td>GCN-RCL</td><td>GCN-linear</td><td>GCN-root</td></tr><tr><td>Synthetic-0.3</td><td>0.6982</td><td>0.7667</td><td>0.7463</td><td>0.7445</td><td>0.6517</td><td>0.7398</td><td>0.6641</td><td>0.6533</td></tr><tr><td>Synthetic-0.5</td><td>0.8742</td><td>0.9016</td><td>0.8476</td><td>0.8704</td><td>0.8715</td><td>0.9269</td><td>0.8494</td><td>0.8854</td></tr><tr><td>Synthetic-0.7</td><td>0.9658</td><td>0.9821</td><td>0.9514</td><td>0.9766</td><td>0.9748</td><td>0.9962</td><td>0.9712</td><td>0.9796</td></tr><tr><td>Cora</td><td>0.8310</td><td>0.8521</td><td>0.8145</td><td>0.8254</td><td>0.8574</td><td>0.8715</td><td>0.8327</td><td>0.8553</td></tr><tr><td>Citeseer</td><td>0.7478</td><td>0.7652</td><td>0.7482</td><td>0.7505</td><td>0.7893</td><td>0.7979</td><td>0.7723</td><td>0.7814</td></tr><tr><td>Computers</td><td>0.8989</td><td>0.9096</td><td>0.8866</td><td>0.8975</td><td>0.8809</td><td>0.9023</td><td>0.8713</td><td>0.8985</td></tr><tr><td>ogbn-arxiv</td><td>0.7175</td><td>0.7441</td><td>0.6980</td><td>0.7242</td><td>0.7174</td><td>0.7408</td><td>0.7288</td><td>0.7359</td></tr></table>

Table 5: Node classification results for our method and traditional CL methods using PNA and GCN as backbone. Here ‘-RCL’ denotes our method, while ‘-linear’ and ‘-root’ denotes two traditional CL methods with different pacing functions.

In Table 5, new experiments that adopt modern GNN architecture - PNA model [6] have been added. From the table we can observe that our proposed method improves the performance of PNA backbone by $2.54\%$ on average, which further verified the effectiveness of our method under different choices of backbone GNN model.

In addition, in Table 5 we further include two traditional CL methods for independent data as additional baselines, following classical works [2, 25]. We employed the supervised training loss of a pretrained GNN model as the difficulty metric, and selected two well-established pacing functions for curriculum design: linear and root pacing, defined as follows:

$$
\text { Linear: } K _ {\text { linear }} (t) = \frac {t}{T} | V |; \text { Root: } K _ {\text { root }} (t) = \sqrt {\frac {t}{T}} | V |,
$$

where $t$ is the number of current iterations and $T$ is the number of total iterations, and $|V|$ is the number of nodes.

We utilized GCN and PNA as backbone architectures, identified by the suffixes '-linear' and '-root'. Across all datasets, the results consistently demonstrate that our proposed method outperforms traditional CL approaches.

# A.4 Additional Robustness Experiments on PNA Backbone Model.

We present further robustness test against random noisy edges by using the PNA backbone model. The results are shown in Table 6, which further proves that our curriculum learning approach improves the robustness against edge noise with the advanced PNA model as the backbone.

# A.5 Time Complexity Analysis

Here we consider GCN as the backbone. First, the time complexity of an L-layer GCN is $O(L|\mathcal{E}|b + L|\mathcal{V}|b^{2})$ , where b is the number of node attributes. Second, the time complexity of measuring the

<table><tr><td>Dataset</td><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td></tr><tr><td>Cora</td><td>PNA</td><td>0.8310</td><td>0.7911</td><td>0.7621</td><td>0.7402</td><td>0.7331</td><td>0.7210</td><td>0.6894</td><td>0.7042</td><td>0.6792</td><td>0.6617</td></tr><tr><td>Cora</td><td>PNA-RCL</td><td>0.8521</td><td>0.8315</td><td>0.8162</td><td>0.7969</td><td>0.7992</td><td>0.7951</td><td>0.7571</td><td>0.7642</td><td>0.7457</td><td>0.7371</td></tr><tr><td>Citeseer</td><td>PNA</td><td>0.7478</td><td>0.7195</td><td>0.7184</td><td>0.6934</td><td>0.6952</td><td>0.6920</td><td>0.6852</td><td>0.6552</td><td>0.6481</td><td>0.6327</td></tr><tr><td>Citeseer</td><td>PNA-RCL</td><td>0.7652</td><td>0.7422</td><td>0.7222</td><td>0.7254</td><td>0.7041</td><td>0.7012</td><td>0.6953</td><td>0.6921</td><td>0.6884</td><td>0.6794</td></tr></table>

Table 6: Further robustness test using PNA as backbone model. Here the percentage denotes the ratio of number of added random edges to the original edges.

<table><tr><td></td><td>Synthetic</td><td>Citeseer</td><td>Computers</td><td>ogbn-arxiv</td><td>ogbn-proteins</td></tr><tr><td>Vanilla</td><td>7.32s</td><td>3.90s</td><td>16.88s</td><td>55.22s</td><td>1438.23s</td></tr><tr><td>GNNSVD</td><td>11.49s</td><td>3.82s</td><td>35.96s</td><td>135.72s</td><td>2632.42s</td></tr><tr><td>CLNode</td><td>6.29s</td><td>3.96s</td><td>17.02s</td><td>58.53s</td><td>1545.53s</td></tr><tr><td>ProGNN</td><td>220.25s</td><td>72.42s</td><td>1953.23s</td><td>(-)</td><td>(-)</td></tr><tr><td>NeuralSparse</td><td>310.02s</td><td>88.91s</td><td>6553.34s</td><td>(-)</td><td>(-)</td></tr><tr><td>PTDNet</td><td>153.43s</td><td>48.42s</td><td>2942.02s</td><td>(-)</td><td>(-)</td></tr><tr><td>Ours</td><td>4.07s</td><td>2.42s</td><td>14.62s</td><td>71.49s</td><td>2239.05s</td></tr></table>

Table 7: Running time of our method and comparison methods. Here (-) denotes an out-of-memory error and Vanilla denotes the standard GNN model.

difficulty levels of edges by reconstruction is $O(|\mathcal{E}|d)$ where d is the number of latent embedding dimensions. Third, the time complexity of selecting the edges to add is $O(|\mathcal{E}|)$ . Therefore, the total time complexity of our algorithm is $O(|\mathcal{E}|(Lb+d)+L|\mathcal{V}|b^{2})$ .

In addition, we compare the total running time of our method and all comparison methods in the Table 7. We can observe that the running time of our proposed method is comparable to that of standard GNN models in all datasets. Notably, our method is even faster than standard GNN models in some datasets. One possible reason is that at the beginning of training, the graphs in our model have much fewer edges than those in standard GNN models. Therefore, the computational cost of the GNN model is also reduced.

# A.6 Parameter Sensitivity Analysis

Recall that RCL learns a curriculum to gradually add edges in a given input graph structure to the training process until all edges are included. An interesting question is how the speed of adding edges will affect the performance of the model. Here we conduct experiments to explore the impact of age parameter $\lambda$ which controls the speed of adding edges to the model performance. Here a larger value of $\lambda$ means that the training structure will converge to the input structure earlier. For example, $\lambda = 1$ means that the training structure will probably not converge to the input structure until the last iteration, and $\lambda = 5$ means that the training structure will converge to the input structure around half of the iterations are complete, and then the model will be trained with the full input structure for the remaining iterations. We present the results on two synthetic datasets (homophily coefficient=0.3, 0.6) and two real-world datasets in Figure 5. As can be seen from the figure, the classification results are steady that the average standard deviation is only 0.41%. It is also worth noting that the peak values for all datasets consistently appear around $\lambda = 3$ , which indicates that the best performance is when the training structure converges to the full input structure around two-thirds of the iterations are completed.

![](images/b9f71b45e15859366292bcd82d946a1a3ea2ef751ff7850490d8828ee3d4c1f5.jpg)  
Figure 5: Parameter sensitivity analysis on four datasets. Here a larger value of $\lambda$ means the training structure will converge to the original structure at an earlier training stage.

# A.7 Visualization of Importance on Smoothing Component

![](images/67fb88e9c124f1619b891604ebf100589ae0c3a57916f5b64624a00186d92a83.jpg)

<details>
<summary>line</summary>

| # Epochs | Ours  | w/o smoothing |
| -------- | ----- | ------------- |
| 0        | 2.0   | 2.0           |
| 20       | 1.5   | 1.5           |
| 40       | 0.5   | 0.5           |
| 60       | 0.2   | 0.2           |
| 80       | 0.1   | 0.1           |
| 100      | 0.05  | 0.1           |
| 120      | 0.03  | 0.08          |
| 140      | 0.02  | 0.05          |
</details>

Figure 6: The comparison between our full model and the version without smoothing technique on the training loss trend.

Our experimental results demonstrated the importance of applying our smoothing component in stabilizing the optimization process of training. Figure 6 shows that without the smoothing technique, the training loss spiked that reflects the GNN parameter shifts, which was caused by the number of edges discretely changed. However, after adding the smoothing technique, the training loss can smoothly converge, hence, the smoothing technique plays an important role in stabilizing the training process.

# B Mathematical Proof

Theorem 1. We have the following convergence guarantees for Algorithm 1:

- Avoidance of Saddle Points If the second derivatives of $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ are continuous, then for sufficiently large $\gamma$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 with random initializations will not converge to a strict saddle point of $F$ almost surely.   
- Second Order Convergence If the second derivatives of $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ are continuous, and $L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}), \mathbf{y})$ and $g(\mathbf{S}; \lambda)$ satisfy the Kurdyka-Lojasiewicz (KL) property [40], then for sufficiently large $\gamma$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 with random initialization will almost surely converges to a second-order stationary point of $F$ .

Proof. We prove this theorem by Theorem 10 and Corollary 3 from [27].

[Avoidance of Saddle Points] Because the sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ is bounded, and the second derivatives of L and g are continuous, then they are bounded. In other words, we have $\max\{\|\nabla_{\mathbf{w}}^{2}L(f(\mathbf{X}, \mathbf{A}^{(t)}; \mathbf{w}^{(t)}), \mathbf{y})\|, \|\nabla_{\mathbf{S}}^{2}g(S^{(t)}; \lambda)\|\}\leq p$ , where p>0 is a constant. Similarly, it is easy to check that the second derivative of the term $\sum_{i,j}S_{ij}\left\|\tilde{\mathbf{A}}_{ij}^{(t)}-\mathbf{A}_{ij}\right\|_{2}^{2}$ is bounded, i.e., $\max\{\left\|\nabla_{\mathbf{w}}^{2}\sum_{i,j}\mathbf{S}_{ij}\left\|\tilde{\mathbf{A}}_{ij}^{(t)}-\mathbf{A}_{ij}\right\|_{2}^{2}\right\|,\left\|\nabla_{\mathbf{S}}^{2}\sum_{i,j}\mathbf{S}_{ij}\left\|\tilde{\mathbf{A}}_{ij}^{(t)}-\mathbf{A}_{ij}\right\|_{2}^{2}\right\}\leq q$ , where q>0 is constant and $\tilde{A}$ is a function of w. Therefore, it means that the objective F is bi-smooth, i.e. $\max\{\|\nabla_{\mathbf{w}}^{2}F\|\}, \|\nabla_{\mathbf{S}}^{2}F\|\}\leq p+q$ . In other words, F satisfies Assumption 4 from [27]. Moreover, the second derivative of F is continuous. For any $\gamma>p+q$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 will not converge to a strict saddle of F almost surely by Theorem 10 from [27].

[Second Order Convergence] From the above proof of avoidance of saddle points, we know that $F$ satisfies Assumption 4 from [27]. Moreover, because $L$ and $g$ satisfy the KL property, and the term $\sum_{i,j} \mathbf{S}_{ij} \left\| \tilde{\mathbf{A}}_{ij}^{(t)} - \mathbf{A}_{ij} \right\|_2^2$ satisfies the KL property, we conclude that $F$ satisfy the KL property as

well. From the proof above, we also know that the second derivative of F is continuous. Because continuous differentiability implies Lipschitz continuity [47], it infers that the first derivative of F is Lipschitz continuous. As a result, F satisfies Assumption 1 from [27]. Because F satisfies Assumptions 1 and 4, then for any $\gamma > p + q$ , any bounded sequence $(\mathbf{w}^{(t)}, \mathbf{S}^{(t)})$ generated by Algorithm 1 will almost surely converges to a second-order stationary point of F by Corollary 3 from [27].

While the convergence of Algorithm 1 entails the second-order optimality conditions of $f$ and $g$ , some commonly used $f$ such as the GNN with sigmoid or tanh activations and some commonly used $g$ such as the squared $\ell_2$ norm satisfy the KL property [39, 40], and Algorithm 1 is guaranteed to avoid a strict saddle point and converges to a second-order stationary point.