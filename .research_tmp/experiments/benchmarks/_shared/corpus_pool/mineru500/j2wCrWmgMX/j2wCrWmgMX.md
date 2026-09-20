# Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities

Alexander Nikitin $^{1}$ Jannik Kossen $^{2}$ Yarin Gal $^{2}$ Pekka Marttinen $^{1}$

$^{1}$ Department of Computer Science, Aalto University $^{2}$ OATML, Department of Computer Science, University of Oxford
alexander.nikitin@aalto.fi

# Abstract

Uncertainty quantification in Large Language Models (LLMs) is crucial for applications where safety and reliability are important. In particular, uncertainty can be used to improve the trustworthiness of LLMs by detecting factually incorrect model responses, commonly called hallucinations. Critically, one should seek to capture the model's semantic uncertainty, i.e., the uncertainty over the meanings of LLM outputs, rather than uncertainty over lexical or syntactic variations that do not affect answer correctness. To address this problem, we propose Kernel Language Entropy (KLE), a novel method for uncertainty estimation in white- and black-box LLMs. KLE defines positive semidefinite unit trace kernels to encode the semantic similarities of LLM outputs and quantifies uncertainty using the von Neumann entropy. It considers pairwise semantic dependencies between answers (or semantic clusters), providing more fine-grained uncertainty estimates than previous methods based on hard clustering of answers. We theoretically prove that KLE generalizes the previous state-of-the-art method called semantic entropy and empirically demonstrate that it improves uncertainty quantification performance across multiple natural language generation datasets and LLM architectures.

# 1 Introduction

Large Language Models (LLMs) have demonstrated exceptional capabilities across a wide array of natural language processing tasks $[57, 65, 68]$ . This has led to their application in many domains, including medicine $[11]$ , education $[32]$ , and software development $[40]$ . Unfortunately, LLM generations suffer from so-called hallucinations, commonly defined as responses that are “nonsensical or unfaithful to the provided source content” $[26, 18, 51]$ . Hallucinations pose significant risks when LLMs are deployed to high-stakes applications, and methods that reliably detect them are sorely needed.

A promising direction to improve the reliability of LLMs is estimating the uncertainty of model generations $[36, 13, 50, 44, 23]$ . As LLM predictions tend to be well-calibrated $[57, 30]$ , high predictive uncertainty is indicative of model errors or hallucinations in settings such as answering multiple-choice questions. This allows us to prevent harmful outcomes by abstaining from prediction or by consulting human experts. However, the best means of estimating uncertainty for free-form natural language generation remains an active research question. The unique properties of LLMs and natural language preclude the application of established methods for uncertainty quantification $[20, 39, 45, 58, 54]$ .

A particular challenge is that language outputs can contain multiple types of uncertainty, including lexical (which word is used), syntactic (how the words are ordered), and semantic (what a text means). For many problems, semantic uncertainty is the desired quantity, as it pertains directly to the accuracy of the meaning of the generated response. However, measuring the uncertainty of the generation via token likelihoods conflates all types of uncertainty. To address this, Kuhn et al. [36] have recently introduced semantic entropy (SE), which estimates uncertainty as the predictive entropy of generated texts with respect to clusters of identical semantic meaning (we discuss this in more detail in Sec. 2).

![](images/17faf7435f2be81c2017101b72c56646977806bda35b9db5e3941483f69478a1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["LLM₁"] --> B["LLM₁ (inp)"]
    C["LLM₂"] --> D["LLM₂ (inp)"]
    B --> E["C₁: Laplace"]
    D --> F["C₁': Laplace"]
    E --> G["C₂: Einstein"]
    F --> H["C₂': Kolemogorov and Laplace"]
    G --> I["C₃: McCartney"]
    H --> J["C₃': Kolemogorov and Laplace"]
    I --> K["C₄: Einstein"]
    J --> L["C₄': Kolemogorov and Laplace"]
    K --> M["Semantic Kernels"]
    L --> M
    M --> N["Entropy"]
    N --> O["Semantic Similarity"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#fcc,stroke:#333
```
</details>

Figure 1: Illustration of Kernel Language Entropy (KLE). We here show a version of KLE called KLE-c, which operates on semantic clusters. Given an input query and two different LLMs, we sample 10 answers from each model $a_1, \ldots, a_{10}$ and $a_1', \ldots, a_{10}'$ and cluster them by semantic equivalence into clusters $C_1, \ldots, C_3$ and $C_1', \ldots, C_3'$ . For the sake of the example, we assume that the numbers and sizes of clusters, as well as individual cluster probabilities, are all equal $p(C_i|\text{inp}) = p(C_i'|\text{inp})$ for all $i$ . Then, semantic entropy would yield identical uncertainties for both LLMs. However, uncertainty should be lower for $\text{LLM}_2$ because semantic “similarity” between the generations is much higher; i.e., the model is fairly confident that “Kolmogorov” and “Laplace” are good answers. KLE, explicitly accounts for the semantic similarity between texts using a kernel-based approach and correctly identifies that $\text{LLM}_2$ 's generations should be assigned lower uncertainty (see right).

A critical limitation of SE is that it captures semantic relations between the generated texts only through equivalence relations. This does not capture a distance metric in the semantic space, which would allow one to account for more nuanced semantic similarity between generations. For instance, it separates “apple” as equally strongly from “house” as it will “apple” from “granny smith” even though the latter pair is more closely related. In this paper, we address this problem by incorporating a distance in the semantic space of generated answers into the uncertainty estimation.

We propose Kernel Language Entropy (KLE). KLE leverages semantic similarities by using a distance measure in the space of the generated answers, encoded by unit trace positive semidefinite kernels. We quantify uncertainty by measuring the von Neumann entropy of these kernels. This approach allows us to incorporate a metric between generated answers or, alternatively, semantic clusters into the uncertainty estimation. Our approach uses kernels to describe semantic spaces, making KLE more general and better at capturing the semantics of generated texts than the previous methods. We theoretically prove that our method is more expressive than semantic entropy, meaning there are cases where KLE, but not SE, can distinguish the uncertainty of generations. Importantly, our approach does not rely on token likelihood and works for both white-box and black-box LLMs.

Our work makes the following contributions towards better uncertainty quantification in LLMs:

- We propose Kernel Language Entropy, a novel method for uncertainty quantification in natural language generation (Sec. 3),   
- We propose concrete design choices for our method that are effective in practice, for instance, graph kernels and weight functions (Sec. 3.1),   
- We prove that our method is a generalization of semantic entropy (Thm. 3.5),   
- We empirically compare our approach against baselines methods across several tasks and LLMs with up to 70B parameters (60 scenarios total), achieving SoTA results (Sec. 5).

We release the code and instructions for reproducing our results at https://github.com/AlexanderVNikitin/kernel-language-entropy.

# 2 Background

Uncertainty Estimation. Information theory [48] offers a principled framework for quantifying the uncertainty of predictions as the predictive entropy of the output distribution:

$$
\mathrm{PE} (x) = H (Y \mid x) = - \int p (y \mid x) \log p (y \mid x) d y, \tag {1}
$$

where $Y$ is the output random variable, $x$ is the input, and $H(Y|x)$ is a conditional entropy which represents average uncertainty about $Y$ when $x$ is given. Uncertainty is often categorized into

aleatoric (data) and epistemic (knowledge) uncertainty. Following previous work on uncertainty quantification in LLMs, we assume that LLMs capture both types of uncertainty $[30]$ and do not attempt to disambiguate them, as both epistemic and aleatoric uncertainty contribute to model errors.

UQ in sequential models. Let $S \in T^{N}$ be a sequence of length N, consisting of tokens, $s_{i} \in T$ , where the set T denotes a vocabulary of tokens. The probability of S is then the joint probability of the tokens, obtained as the product of conditional token probabilities:

$$
p (S \mid x) = \prod_ {i} p (s _ {i} | s _ {<   i}, x). \tag {2}
$$

Instead of Eq. (2), the geometric mean of token probabilities has proven to be successful in practice [49]. Using Eq. (1) and (2), we can define the predictive entropy of a sequential model.

Definition 2.1. The predictive entropy for a random output sequence $S$ and input $x$ is

$$
U (x) = H (S \mid x) = - \sum_ {s} p (s \mid x) \log (p (s \mid x)), \tag {3}
$$

where the sum is taken over all possible output sequences s.

A downside of naive predictive entropy for Natural Language Generation (NLG) is that it measures uncertainty in the space of tokens while the uncertainty of interest lies in semantic space. As an illustrative example, consider two sets of n answers, $S_{i}$ and $S_{i}^{\prime}$ sampled from two LLMs with equivalent token likelihood $p(S_{i}|x) = p(S_{i}^{\prime}|x)$ as a response to the question “What is the capital of France?” [36]. Suppose the answers from the first LLM are various random cities (“Paris”, “Rome”, etc.), and those from the second LLM are paraphrases of the correct answer “It is Paris”. Naive predictive entropy computation can give similar values, even though the second LLM is not uncertain about the meaning of its answer. Kuhn et al. [36] have proposed semantic entropy to address this problem.

We first define the concept of semantic clustering. Semantic clusters are equivalence classes obtained using a semantic equivalence relation, $E(\cdot,\cdot)$ , which is reflexive, symmetric, and transitive and should capture semantic equivalence between input texts. In practice, E is computed using bidirectional entailment predictions from a Natural Language Inference (NLI) model, such as DeBERTa [22] or a prompted LLM, that classifies relations between pairs of texts as “entailment,” “neutral,” or “contradiction”. Two texts are semantically equivalent if they entail each other bi-directionally. Semantic clusters are obtained by greedily aggregating generations into clusters of equivalent meaning. We can now define semantic entropy.

Definition 2.2. For an input $x$ and semantic clusters $C \in \Omega$ , where $\Omega$ is a set of all semantic clusters, Semantic Entropy (SE) is defined as

$$
\operatorname{SE} (x) = - \sum_ {C \in \Omega} p (C \mid x) \log p (C \mid x) = - \sum_ {C \in \Omega} \left(\left(\sum_ {s \in c} p (s \mid x)\right) \log \left[ \sum_ {s \in C} p (s \mid x) \right]\right). \tag {4}
$$

In practice, it is not possible to calculate $\sum_{C} p(C \mid x) \log p(C \mid x)$ because of the intractable number of semantic clusters. Instead, SE uses a Rao-Blackwellized Monte Carlo estimator

$$
\mathrm{SE} (x) \approx - \sum_ {i = 1} ^ {M} p ^ {\prime} (C _ {i} | x) \log p ^ {\prime} (C _ {i} | x), \tag {5}
$$

where $C_{i}$ are M clusters extracted from the generations and $p'(C_{i} \mid x)$ is a normalized semantic probability, $p'(C_{i} \mid x) = p(C_{i}|x)/\sum_{i} p(C_{i}|x)$ , which we refer to as $p(C_{i}|x)$ in the following for simplicity. SE can be extended to cases where token likelihoods are not available by approximating $p(C_{i}|x)$ with the fraction of generated texts in each cluster, $p(C_{i}|x) \approx \sum_{i=1}^{N} \mathbb{I}(S_{i} \in C_{i})/N$ . We refer to this variant as Discrete Semantic Entropy [16].

# 3 Kernel Language Entropy

This section introduces Kernel Language Entropy (KLE), our novel approach to computing semantic uncertainty that accounts for fine-grained similarities between generations for better uncertainty quantification. We introduce two variants of KLE: the first, simply called KLE, operates directly on the generated texts, and the second, KLE-c operates on the space of semantic clusters.

Motivating Example. Figure 1 illustrates the advantages of KLE (to be precise, the KLE-c variant) over other methods such as SE. Imagine querying two LLMs such that the outputs of $\mathrm{LLM}_1$ are

all semantically different and those of $\mathrm{LLM}_2$ are semantically similar but not equivalent. For simplicity, we assume an equal amount of clusters between LLMs and equal likelihoods of clusters $p(C_i|\mathrm{inp}) = p(C_i'|\mathrm{inp})$ . SE would not distinguish between those cases and, thus, would misleadingly predict equal uncertainty. KLE on the other hand, will correctly assign lower uncertainty to the outputs of $\mathrm{LLM}_2$ , its kernels accounting for the fact that $\mathrm{LLM}_2$ produces semantically similar outputs.

Before introducing KLE, we recall the definition of a positive semidefinite (PSD) kernel.

Definition 3.1. For a set $\mathcal{X} \neq \emptyset$ , a symmetric function $K: \mathcal{X} \times \mathcal{X} \to \mathrm{R}$ is called a PSD kernel if for all $n > 0$ , $x_i \in \mathcal{X}$ , $\alpha_i \in \mathbb{R}$

$$
\sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} \alpha_ {i} \alpha_ {j} K (x _ {i}, x _ {j}) \geq 0. \tag {6}
$$

For a finite set $\mathcal{X}$ , a PSD kernel is a PSD matrix of the size $|\mathcal{X}|$ .

Next, we define semantic kernels, denoted $K_{sem}$ , as unit trace $^{1}$ positive semidefinite kernels over the finite domain of generated texts. Unit trace PSD matrices are also called density matrices. These kernels should, informally speaking, capture the semantic similarity $^{2}$ between the texts such that $K(s_{1}, t_{1}) > K(s_{2}, t_{2})$ if and only if texts $s_{1}$ and $t_{1}$ are more semantically related than texts $s_{2}$ and $t_{2}$ . Analogously, we define semantic kernels over semantic clusters of texts, in which case the kernel should capture the semantic similarity between the clusters. In practice there are multiple ways to concretely specify a proper semantic kernel, and some options are described in Section 3.1.

The von Neumann Entropy. We propose to use the von Neumann entropy (VNE) to evaluate the uncertainty associated with a semantic kernel.

Definition 3.2 (Von Neumann Entropy). For a unit trace positive semidefinite matrix $A \in \mathbb{R}^{n \times n}$ , the von Neumann entropy (VNE) [71] is defined as

$$
\operatorname{VNE} (A) = - \operatorname{Tr} [ A \log A ]. \tag {7}
$$

It can be shown that $\mathrm{VNE}(A)=\sum_{i}^{n}-\lambda_{i}\log\lambda_{i}$ where $\lambda_{i},1\leq i\leq n$ are the eigenvalues of A. Within this definition, we assume $0\log0=0$ . This reformulation shows that VNE is, in fact, the Shannon entropy over the eigenvalues of a kernel. We can now define Kernel Language Entropy.

Definition 3.3 (Kernel Language Entropy). Given a set of LLM generations $S_{1}, \ldots, S_{N}$ , an input $x$ , and semantic kernel $K_{sem}$ over these generations and input, we define Kernel Language Entropy (KLE) as the von Neumann entropy of a semantic kernel $K_{sem}$ :

$$
\mathrm{KLE} (x) = \mathrm{VNE} (K _ {s e m}). \tag {8}
$$

The von Neumann entropy has the following properties, which is aligned with the overarching goal of measuring the uncertainty of a set of generations.

Proposition 3.4 (Properties of the von Neumann Entropy [5]). The VNE of a unit trace positive semidefinite kernel has the following properties:

1. The VNE of a kernel with only one non-zero element is equal to 0.   
2. The VNE is invariant under changes of basis $U$ : $\mathrm{VNE}(K) = \mathrm{VNE}(UKU^{\top})$ .   
3. The VNE is concave. For a set of positive coefficients $\alpha_{i}$ , $\sum_{i=1}^{k} \alpha_{i} = 1$ , and density matrices $K_{i}$ , it holds that $\mathrm{VNE}\left(\sum_{i=1}^{k} \alpha_{i} K_{i}\right) \geq \sum_{i=1}^{k} \alpha_{i} \mathrm{VNE}(K_{i})$ .

Let us briefly discuss the practical implications of these properties. Property 1 states that if an LLM outputs a single answer (for KLE) or a semantic cluster (for KLE-c), the VNE is zero, indicating high certainty. Property 2 is significant as it allows the VNE to be calculated in practice as the Shannon entropy of the diagonal elements of an orthogonalized kernel, which can be interpreted as a disentangled representation of a semantic kernel. Property 3 states that entropy is concave, meaning that the entropy of a combined system is greater than or equal to the entropy of its individual parts, a common requirement for entropy metrics. The intuition behind our use of VNE for LLMs also relates to its origins in quantum information.

The VNE in Quantum Information Theory. In quantum information theory, the states of a quantum system (or pure states) are defined as unit vectors in $C^{N}$ . However, experiments often result in statistical mixtures of pure quantum states, represented as density matrices. The VNE is used to evaluate the entropy of the mixed states. Analogously, we can think of KLE as considering each answer as a mixture of pure “semantic meanings”, measuring the entropy of this mixture. We refer the reader to Aaronson [1] for further background reading on the VNE and quantum information theory.

KLE-c. Instead of defining semantic kernels directly over individual model generations, we

can also apply KLE to clusters of semantic equivalence. We call this variant of our method KLE-c. Although KLE is more general than KLE-c for non-trivial clusterings, KLE-c can provide practical value as it is cheaper to compute and more interpretable due to its smaller kernel sizes.

Algorithm. Algorithm 1 provides a generic description of the steps required to compute KLE. We describe the practical details for defining and combining semantic kernels later in Sec. 3.1.

Computational Complexity. The computational complexity of KLE is approximately identical to SE which requires sampling from an LLM N times and running the entailment model $O(N^{2})$ times. Additionally, KLE requires $O(N^{3})$ elementary operations for kernel and VNE calculation. The actual cost of this is negligible in comparison to the forward passes through the LLM or entailment model.

# 3.1 Semantic Graph Kernels

This section describes a practical approach for constructing semantic kernels over LLM generations or semantic clusters. Concretely, we apply NLI models to construct semantic graphs over the LLM outputs and then borrow from graph kernel theory to construct kernels from these graphs.

Graph Theory Preliminaries. First, let us recall the basics of graph theory. A graph is a pair of two sets $G = (V, E)$ , where $V = \{1, \ldots, n\}$ is a set of n vertices and $E \subseteq V \times V$ is a set of edges. A graph is called weighted when a weight is assigned to each edge, and the weight matrix $W_{ij}$ contains weights between nodes i and j. For unweighted graphs, we can use a binary adjacency matrix to encode edges between nodes. The degree matrix D is a diagonal $|V| \times |V|$ matrix with $D_{ii} = \sum_{j=1}^{|V|} W_{ij}$ . The graph Laplacian is defined as L = D - W. L is a positive semidefinite matrix, and eigenvalues of L are often used to study the structure of graphs [10, 70].

Semantic Graph. We define semantic graphs as graphs over LLM generations ( $G_{sem}$ ) or semantic clusters ( $G_{sem-c}$ ). For $G_{sem}$ , edges can be defined as a function of NLI predictions in both directions: $W_{ij} = f(\text{NLI}(S_i, S_j), \text{NLI}(S_j, S_i))$ , where NLI are the predicted probabilities for entailment, neutral, and contradiction for $S_i$ and $S_j$ . For example, f could be the weighted sum over the predicted probabilities for entailment and neutral classes. For $G_{semc-c}$ , the weights between the clusters are computed by summing the entailment predictions over the generations assigned to the clusters, $W_{ij} = \sum_{s \in C_i} \sum_{t \in C_j} f(\text{NLI}(s, t), \text{NLI}(t, s))$ .

Graph Kernels. When a semantic graph is obtained, KLE calculates graph kernels over semantic graph nodes to compute a distance measure. Since graphs are discrete and finite, any positive semidefinite matrix would be a kernel over the graph. However, we seek kernels that exploit knowledge about the graph structure. We therefore adopt Partial Differential Equation (PDE) and Stochastic Partial Differential Equation (SPDE) approaches to graph kernels $[34, 6, 56]$ . If $u \in R^{n}$ is a signal over the nodes of a graph, the heat kernel is a solution to the partial differential equation $\frac{\partial u}{\partial t} + Lu = 0$ and the Matérn kernel is a solution to the stochastic differential equation, $(2\nu/\kappa^{2} + L)^{\frac{\nu}{2}}u = w$ , where w is white noise over the graph nodes and L is the graph Laplacian defined above. The corresponding solutions to these equations are:

$$
K _ {t} = e ^ {- t L} \quad [ \text {HEAT} ] \quad K _ {\nu \kappa} = \left(^ {2 \nu} / _ {\kappa} ^ {2} I + L\right) ^ {- \nu} \quad [ \text {MATÉRN} ]. \tag {9}
$$

These kernels allow for the incorporation of a distance measure that reflects the graph's locality properties (right part of Fig. 2). For example, the Taylor series of the heat kernel can be shown

Algorithm 1 Kernel Language Entropy   
Require: LLM, Input $x \in T^{L}$ , Number of samples n, Boolean kle-c indicating variant, Semantic kernels $K_{i}$ 1: Initialize a multiset of answers $O \leftarrow \emptyset$ 2: for $k \leftarrow 1$ to n do ▷ Sampling n answers

3: Add LLM(x) to O

4: end for

5: if kle-c then

6: Update $O \leftarrow cluster(O)$ ▷ as in [36]

7: end if

8: Combine $K_{i}(O, O)$ in $K_{sem}$ ▷ see Sec. 3.1

9: Return VNE( $K_{sem}$ ) ▷ Eq. (8)

![](images/3cc96d08a41067474d68b1f3f9fe852d8772ce8eb0cee4d7bc7082d524cc241d.jpg)

<details>
<summary>heatmap</summary>

| Number of Edges | VNE | t     |
| --------------- | --- | ----- |
| 0               | 1.0 | -0.50 |
| 50              | 0.8 | -0.25 |
| 100             | 0.6 | -0.10 |
| 200             | 0.4 | 0.0   |
| 500             | 0.2 | 0.25  |
| 1000            | 0.1 | 0.50  |
</details>

Figure 2: Entropy Convergence Plots for heat kernels. For graphs of various sizes $|V|$ , we grow the number of edges and examine the VNE. For large lengthscales t, corresponding to darker colored curves, the VNE quickly converges to zero. We can use these plots to determine kernel hyperparameters without validation sets. The VNE is scaled to start at 1 for visualization purposes.

to be equal to a sum of powers of random walk matrices. Both kernels have hyperparameters: lengthscales $t$ in the heat kernel and $\kappa$ in Matérn kernels, and $\nu$ in the Matérn kernel, often interpreted as smoothness. The scaled eigenvalues of the Matérn kernel converge to the eigenvalues of the heat kernel [6] when $\nu$ goes to infinity. Matérn kernels provide more flexibility at the cost of the additional parameter. Note that any kernel can be normalized into a unit trace kernel via $K(x,y) \leftarrow K(x,y)(K(x,x)K(y,y))^{-1/2} / N$ , where $N$ is the size of $K$ . We refer to [34, 56, 6] for further background reading on graph kernels.

Kernel Hyperparameters. We propose two ways to select the hyperparameters of the heat and Matérn kernels: either by maximizing the validation set performance or by selecting parameters from what we call Entropy Convergence Plots, illustrated in Fig. 2. We obtain these plots by defining a set of progressively denser graphs $G_{1} \prec \ldots \prec G_{K}$ . These can be obtained by starting from a graph without edges and a fixed number of vertices and adding new edges either randomly or by filling in the adjacencies of each node sequentially. We then plot the VNE against the number of edges in the graphs $G_{i}$ . We analyze the von Neumann entropy over these plots to avoid pathologies connected to the fact that for large lengthscales, the VNE converges rather quickly and such behavior should generally be avoided. For all remaining values, we can either choose hyperparameters randomly from the range of non-collapsing hyperparameters or rely on prior domain knowledge.

Kernel Combination. KLE offers the additional flexibility of combining kernels from various methods (e.g., multiple NLI models, different graph kernels, or other methods). For example, we can combine multiple kernels using convex combinations, $K = \sum_{i=1}^{P} \alpha_{i} K_{i}$ , where $\sum_{i=1}^{P} \alpha_{i} = 1$ .

# 3.2 Kernel Language Entropy Generalizes Semantic Entropy

The semantic kernels used in KLE are more informative than the semantic equivalence relations used in SE [36]. The next theorem shows that KLE can recover SE for any semantic clustering.

Theorem 3.5 (KLE and KLE-c generalize SE). For any semantic clustering, there exists a semantic kernel over texts $K_{sem}(s, s')$ such that the VNE of this kernel is equal to semantic entropy (computed as in Eq. (5)). Moreover, there exists a semantic kernel over clusters $K_{sem}(c, c')$ such that the VNE of this kernel is equal to SE.

Proof Sketch. For any semantic clustering, we consider a kernel with a block diagonal structure. Each block corresponds to a semantic cluster and cluster likelihoods are normalized by the size of the cluster, $p(C_{i}|x)/m_{i}$ . This is a valid semantic kernel and the KLE for this kernel equals the SE. Thm. B.1 and Thm. B.2 in the Appendix contain the detailed proofs. □

The proof of Thm. 3.5 shows that the block diagonal semantic kernels used with KLE can recover semantic entropy for any clustering. However, there are other kernels available that allow KLE to be more expressive than SE. Comparing KLE and KLE-c, we find that KLE is more general than KLE-c for any non-trivial clustering.

Table 1: Detailed experimental results for Llama 2 70B Chat and Falcon 40B Instruct. 

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">BioASQ [35]</td><td colspan="2">NQ [38]</td><td colspan="2">SQuAD [61]</td><td colspan="2">SVAMP [59]</td><td colspan="2">Trivia QA [29]</td></tr><tr><td>AUROC</td><td>AUARC</td><td>AUROC</td><td>AUARC</td><td>AUROC</td><td>AUARC</td><td>AUROC</td><td>AUARC</td><td>AUROC</td><td>AUARC</td></tr><tr><td rowspan="7">Llama 270B Chat</td><td>SE [36]</td><td> $0.74 \pm 0.04$ </td><td> $0.90 \pm 0.01$ </td><td> $0.71 \pm 0.03$ </td><td> $0.47 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.62 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.77 \pm 0.03$ </td><td> $0.79 \pm 0.02$ </td></tr><tr><td>DSE [36]</td><td> $0.75 \pm 0.04$ </td><td> $0.90 \pm 0.01$ </td><td> $0.71 \pm 0.03$ </td><td> $0.46 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.77 \pm 0.03$ </td><td> $0.79 \pm 0.02$ </td></tr><tr><td>PE [49]</td><td> $0.69 \pm 0.04$ </td><td> $0.90 \pm 0.01$ </td><td> $0.67 \pm 0.03$ </td><td> $0.44 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.59 \pm 0.03$ </td><td> $0.58 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.73 \pm 0.03$ </td></tr><tr><td>P(True) [30]</td><td> $0.86 \pm 0.03$ </td><td> $\mathbf{0.92} \pm 0.01$ </td><td> $\mathbf{0.78} \pm 0.03$ </td><td> $0.50 \pm 0.03$ </td><td> $0.69 \pm 0.03$ </td><td> $\mathbf{0.68} \pm 0.03$ </td><td> $0.74 \pm 0.02$ </td><td> $0.68 \pm 0.03$ </td><td> $0.76 \pm 0.03$ </td><td> $0.79 \pm 0.02$ </td></tr><tr><td>ER</td><td> $0.70 \pm 0.05$ </td><td> $0.89 \pm 0.01$ </td><td> $0.58 \pm 0.03$ </td><td> $0.39 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.64 \pm 0.03$ </td><td> $0.68 \pm 0.03$ </td><td> $0.64 \pm 0.03$ </td><td> $0.76 \pm 0.03$ </td><td> $0.79 \pm 0.02$ </td></tr><tr><td>KLE( $K_{HEAT}$ )</td><td> $0.87 \pm 0.03$ </td><td> $\mathbf{0.92} \pm 0.01$ </td><td> $\mathbf{0.78} \pm 0.02$ </td><td> $\mathbf{0.51} \pm 0.03$ </td><td> $\mathbf{0.71} \pm 0.03$ </td><td> $\mathbf{0.68} \pm 0.03$ </td><td> $\mathbf{0.76} \pm 0.02$ </td><td> $\mathbf{0.69} \pm 0.03$ </td><td> $\mathbf{0.84} \pm 0.03$ </td><td> $\mathbf{0.82} \pm 0.02$ </td></tr><tr><td>KLE( $K_{FULL}$ )</td><td> $\mathbf{0.88} \pm 0.03$ </td><td> $\mathbf{0.92} \pm 0.01$ </td><td> $0.77 \pm 0.02$ </td><td> $0.50 \pm 0.03$ </td><td> $0.70 \pm 0.03$ </td><td> $\mathbf{0.68} \pm 0.03$ </td><td> $0.70 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.80 \pm 0.03$ </td><td> $0.81 \pm 0.02$ </td></tr><tr><td rowspan="7">Falcon 40B Instr</td><td>SE [36]</td><td> $0.85 \pm 0.02$ </td><td> $0.90 \pm 0.01$ </td><td> $\mathbf{0.78} \pm 0.03$ </td><td> $\mathbf{0.43} \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.79 \pm 0.03$ </td><td> $0.72 \pm 0.03$ </td></tr><tr><td>DSE [36]</td><td> $0.85 \pm 0.02$ </td><td> $0.89 \pm 0.01$ </td><td> $0.77 \pm 0.03$ </td><td> $0.40 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.62 \pm 0.03$ </td><td> $0.67 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.79 \pm 0.03$ </td><td> $0.71 \pm 0.03$ </td></tr><tr><td>PE [49]</td><td> $0.75 \pm 0.03$ </td><td> $0.87 \pm 0.01$ </td><td> $0.71 \pm 0.03$ </td><td> $0.38 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.60 \pm 0.03$ </td><td> $0.59 \pm 0.03$ </td><td> $0.57 \pm 0.03$ </td><td> $0.68 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td></tr><tr><td>P(True) [30]</td><td> $0.87 \pm 0.03$ </td><td> $0.89 \pm 0.01$ </td><td> $0.71 \pm 0.03$ </td><td> $0.37 \pm 0.03$ </td><td> $0.66 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.73 \pm 0.03$ </td><td> $0.67 \pm 0.03$ </td><td> $0.72 \pm 0.03$ </td><td> $0.69 \pm 0.03$ </td></tr><tr><td>ER</td><td> $0.74 \pm 0.04$ </td><td> $0.85 \pm 0.02$ </td><td> $0.73 \pm 0.03$ </td><td> $0.39 \pm 0.03$ </td><td> $0.63 \pm 0.03$ </td><td> $0.61 \pm 0.03$ </td><td> $0.75 \pm 0.02$ </td><td> $\mathbf{0.68} \pm 0.03$ </td><td> $0.76 \pm 0.03$ </td><td> $0.69 \pm 0.03$ </td></tr><tr><td>KLE( $K_{HEAT}$ )</td><td> $\mathbf{0.92} \pm 0.01$ </td><td> $\mathbf{0.91} \pm 0.01$ </td><td> $0.76 \pm 0.03$ </td><td> $0.42 \pm 0.03$ </td><td> $\mathbf{0.70} \pm 0.03$ </td><td> $\mathbf{0.66} \pm 0.03$ </td><td> $\mathbf{0.77} \pm 0.02$ </td><td> $\mathbf{0.68} \pm 0.03$ </td><td> $\mathbf{0.80} \pm 0.02$ </td><td> $\mathbf{0.74} \pm 0.03$ </td></tr><tr><td>KLE( $K_{FULL}$ )</td><td> $0.90 \pm 0.02$ </td><td> $\mathbf{0.91} \pm 0.01$ </td><td> $\mathbf{0.78} \pm 0.03$ </td><td> $\mathbf{0.43} \pm 0.03$ </td><td> $0.69 \pm 0.03$ </td><td> $0.65 \pm 0.03$ </td><td> $0.69 \pm 0.03$ </td><td> $0.64 \pm 0.03$ </td><td> $\mathbf{0.80} \pm 0.03$ </td><td> $0.73 \pm 0.03$ </td></tr></table>

# 4 Related Work

In the context of machine learning, the VNE has been studied theoretically[4], applied to GAN regularization [33], and the exponential of the VNE has been used for effective rank and sample diversity analysis [63, 19].

The first attempts at estimating the entropy of language date back to the 1950s [64], and, today, techniques for uncertainty quantification are widely used in natural language processing. For instance, Desai and Durrett [15] and Jiang et al. [28] presented calibration techniques for classification tasks. Xiao and Wang [72] empirically showed that, for various tasks including sentiment analysis and named entity recognition, measuring model uncertainty can be used to improve performance. Calibration techniques have also been applied in machine translation tasks to improve accuracy [37].

Malinin and Gales [49] discussed the challenges of estimating uncertainty in sequential models. Several previous works have queried LLMs to elicit statements about uncertainty, either via fine-tuning or by directly including previous LLM generations in the prompt [30, 9, 53, 43, 52, 21, 62, 67, 12, 73, 36]. Zhang et al. [75] studied UQ for long text generation. Quach et al. [60] used conformal predictions to quantify LLM uncertainty, which is orthogonal to the approach we pursue here. Yang et al. [74] have shown that Bayesian modeling of LLMs using low-rank Laplace approximations improves calibration in small-scale multiple-choice settings. Lin et al. [44] applied spectral graph analysis to graphs of answers for black-box LLMs. Aichberger et al. [2] proposed a new method for sampling diverse answers from LLMs; more diverse sampling strategies could improve KLE as well.

There are a variety of ways besides model uncertainty to detect hallucinations in LLMs such as querying external knowledge bases $[17, 42, 69]$ , hidden state interventions $[76, 24, 46]$ , using probes $[8, 41, 47]$ , or applying fine-tuning $[31, 66]$ . KLE is complementary to many of these directions and focuses on estimating more fine-grained semantic uncertainty. It can either be used to improve these approaches or be combined with them sequentially.

# 5 Experiments

Datasets and Models. Our experiments span over 60 dataset-model pairs. We evaluate on the following tasks covering different domains of natural language generation: general knowledge (TriviaQA [29] and SQuAD [61]), biology and medicine (BioASQ [35]), general domain questions from Google search (Natural Questions, NQ [38]), and natural language math problems (SVAMP [59]). We generally discard the context associated with each input for all datasets except SVAMP, as the tasks become too easy for the current generation of models when context is provided. We use the following LLMs: Llama-2 7B, 13B, and 70B [68], Falcon 7B and 40B [3], and Mistral 7B [27], using both standard and instruction-tuned versions of these models. As the NLI model for defining semantic graphs or semantic clusters, we use DeBERTa-Large-MNLI [22].

Baselines. As baseline methods, we compare KLE with semantic entropy $[36]$ , discrete semantic entropy $[16, 36]$ , token predictive entropy $[49]$ , embedding regression $[16]$ , and P(True) $[30]$ . For embedding regression, we train a logistic regression model on the last layer hidden states to predict whether a given LLM answer is correct.

KLE Kernels. We propose to use the following two semantic kernels with KLE: $K_{HEAT}$ and $K_{FULL}$ . Both are obtained from the weighted graph $W_{ij} = w \text{ NLI}'(S_i, S_j) + w \text{ NLI}'(S_j, S_i)$ , where

![](images/dde21f4091031fa457b9aad4febc47f945c50e673fcdfdc56d4d2bd215235949.jpg)

<details>
<summary>heatmap</summary>

| | KLE(K_full) | KLE(K_HEAT) | SE | DSE | P(True) | PE | ER |
|---|---|---|---|---|---|---|---|
| KLE(K_FULL) | 0.77 | 0.77 | 0.77 | 0.77 | 0.78 | 0.90 | 0.88 |
| KLE(K_HEAT) | 0.72 | 0.77 | 0.77 | 0.77 | 0.83 | 0.90 | 0.88 |
| SE | 0.23 | 0.28 | 0.30 | 0.30 | 0.57 | 0.82 | 0.80 |
| DSE | 0.23 | 0.23 | 0.70 | 0.60 | 0.60 | 0.87 | 0.85 |
| P(True) | 0.22 | 0.17 | 0.43 | 0.40 | 0.55 | 0.70 | |
| PE | 0.10 | 0.10 | 0.18 | 0.13 | 0.45 | 0.67 | |
| ER | 0.12 | 0.12 | 0.20 | 0.15 | 0.30 | 0.33 | |
</details>

(a) Win rate measured with AUROC

![](images/77fd6c3c3c6bf2af8bbda4a8aee08901c76373bf3b6880c7d5866d53f72f7e45.jpg)  
(b) Win rate measured with AUARC   
Figure 3: Summary of 60 experimental scenarios. Each cell contains the fraction of experiments where a method from a row outperforms a method from a column. Our methods are labeled $\mathrm{KLE}(\cdot)$ . Values larger than or equal to 0.62 correspond to the significance level $p < 0.05$ according to the binomial statistical significance test.

$w = (1, 0.5, 0)^{\top}$ is a weight vector. Here, we assume that NLI' returns a one-hot prediction over (entailment, neutral class, contradiction). $K_{HEAT}$ is a heat kernel over this graph. We further propose $K_{FULL} = \alpha K_{HEAT} + (1 - \alpha)K_{SE}$ , where $\alpha \in [0, 1]$ and $K_{SE}$ is a semantic entropy kernel. We ablate these kernel choices in our experiments below.

Evaluation metrics. Following previous work, we evaluate uncertainty methods by measuring their ability to predict the correctness of model responses, calculating the Area under the Receiver Operating Curve (AUROC). Further, uncertainty metrics can be used to refuse answering when uncertainty is high, increasing model accuracy on the subset of questions with uncertainty below a threshold. We measure this with the Area Under the Accuracy-Rejection Curve (AUARC), [55]. The rejection accuracy at a given uncertainty threshold is the accuracy of the model on the subset of inputs for which uncertainty is lower than the threshold; the AUARC score computes the area under the rejection accuracy curve for all possible thresholds.

Sampling. We sample 10 answers per input via top-K sampling with K = 50 and nucleus sampling with p = 0.9 at temperature T = 1. To assess model accuracy, we draw an additional low-temperature sample (T = 0.1) and ask an additional LLM (Llama 3 8B Instruct) to compare the model response to the ground truth answer provided by the datasets.

Statistical significance. We assess statistical significance in two ways. First, we run a large number of experimental scenarios (60 model-dataset pairs), and second, for each experimental scenario, we also obtain confidence intervals over 1000 bootstrap resamples. We note that standard errors in each scenario are more representative of the LLM and the dataset rather than the method. Therefore, our main criterion for comparing the methods is based on the fraction of experimental cases where our method outperforms baselines (assessed with a binomial statistical significance test).

KLE outperforms previous methods. We compare the performance of UQ methods over 60 scenarios (12 models, five datasets). Figure 3 shows the heatmaps of pairwise win rates. We observe that both our methods, $\mathrm{KLE}(K_{\mathrm{HEAT}})$ and $\mathrm{KLE}(K_{\mathrm{FULL}})$ , are superior to the baselines. Furthermore, Table 1 shows the detailed results for the two largest models from our experiments, Llama 2 70B Chat and Falcon 40B Instruct. The results show that for largest models our method consistently achieves best results compared to baselines. In Fig. D.3 and Fig. D.4, we show the experimental results for all considered models. Importantly, our best method, $\mathrm{KLE}(K_{\mathrm{HEAT}})$ , does not require token-level probabilities from a model, and works in black-box scenarios.

KLE hyperparameters can be selected without validation sets. We compare the strategies of hyperparameter selection from Sec. 3.1: entropy convergence plots and validation sets (100 samples per dataset except for SVAMP, where we used default hyperparameters). We observe that default hyperparameters achieve similar results as selecting hyperparameters from validation sets and conclude that choosing default hyperparameters from entropy convergence plots is a good way to select hyperparameters in practice. In Fig. 4, we compare the two strategies for selecting

hyperparameters, and see that the ranking of the methods remains stable and the pairwise win-rates are similar for both methods.

Many design choices outperform existing methods, the best is $\mathrm{KLE}(K_{\mathrm{HEAT}})$ . Next, in Fig. 4, we compare several design choices for KLE: choosing a kernel (heat or Matérn), using KLE-c, combining kernels via a weighted sum or product, and using the probabilities returned by DeBERTa for edge weights. The superscript indicates the type of a graph: no superscript indicates a weighted graph as described above, DB means weights are assigned using probabilities from DeBERTa, and C means a weighted graph over clusters (KLE-c). The subscript indicates the semantic kernels: SE stands for a diagonal kernel with semantic probabilities, HEAT and MATÉRN for the type of kernel, and $\star$ for the best of Heat and Matérn kernels. We observe that even though all design choices outperform SE, the heat kernel over a weighted semantic graph, $\mathrm{KLE}(K_{\mathrm{HEAT}})$ , was overall the best. Additionally, we notice that the methods based on token likelihoods are performing better for non-instruction tuned models, and we can practically recommend including semantic probabilities (e.g., use variations of $K_{FULL}$ ) if KLE is used in non-instruction tuned scenarios (see Fig. D.5).

![](images/61052ccb7603c9cdef31bcaa78cc23e0b5c1f104dc90488710f85fb97077dd1a.jpg)

<details>
<summary>other</summary>

| Method           | Validation | Default |
| ---------------- | ---------- | ------- |
| K_HEAT           | 0.7        | 0.6     |
| K_MATÉRN         | 0.65       | 0.6     |
| K*_DB            | 0.6        | 0.6     |
| K_HEAT^C        | 0.55       | 0.5     |
| K*_C·K_SE        | 0.5        | 0.5     |
| K_FULL^C         | 0.45       | 0.45    |
| K*_C + K_SE      | 0.3        | 0.3     |
| K_FULL^DB        | 0.25       | 0.25    |
| SE               | 0.2        | 0.2     |
</details>

Figure 4: Comparison of various design choices for semantic graph kernels. ★ represents the best hyperparameters and □ – defaults. Error bars are twice the standard error. Summary of 48 experiments.

KLE is better in practice because it captures more fine-grained semantic relations than SE. The performance of KLE improves over SE because in complex free-form language generation scenarios, such as those studied here, LLMs can generate similar but not strictly equal answers. SE assigns these to separate clusters, predicting high entropy. By contrast, our method can account for semantic similarities using the kernel metric in the space of meanings over generated texts, and predict reduced uncertainty if necessary. We give a detailed illustrative example for which KLE provides better uncertainty estimates than SE from the NQ Open dataset in Fig. C.2.

# 6 Discussion

Measuring semantic uncertainty in LLMs is a challenging and important problem. It requires navigating the semantic space of the answers, and we have suggested a method, KLE, that encodes a similarity measure in this space via semantic kernels. KLE allows for fine-grained estimation of uncertainty and is an expressive generalization of semantic entropy. We provided several specific design choices by defining NLI-based semantic graphs and kernels, and studying kernel hyperparameters. We have evaluated KLE across various domains of natural language generation, and it has demonstrated superior performance compared to the previous methods. Our method works both for white- and black-box settings, enabling its application to a wide variety of practical scenarios. We hope to inspire more work that moves from semantic equivalence to semantic similarity for estimating semantic uncertainty in LLMs.

Broader Impact. Our work advances the progress toward safer and more reliable uses of LLMs. KLE can positively impact areas that involve using LLMs by providing more accurate uncertainty estimates, which can filter out a proportion of erroneous outputs.

Limitations. One limitation of the proposed method is that it requires multiple samples from an LLM, which generally increases the generation cost. However, in safety-critical tasks, the potential cost of hallucination should outweigh the cost of sampling multiple answers, so reliable uncertainty quantification via KLE should always be worthwhile. Additionally, we study semantic kernels derived from NLI-based semantic graphs, but other semantic kernels warrant investigation, such as kernels on embeddings. Lastly, the NLG landscape is highly diverse, and the method should be carefully evaluated for other potential applications of LLM, such as code generation.

# Acknowledgments and Disclosure of Funding

This work was supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI, and grants 352986, 358246) and EU (H2020 grant 101016775 and NextGenerationEU).

# References

[1] S. Aaronson. Introduction to quantum information science II lecture notes, 2022.   
[2] L. Aichberger, K. Schweighofer, M. Ielanskyi, and S. Hochreiter. How many opinions does your llm have? improving uncertainty estimation in nlg. In ICLR 2024 Workshop on Secure and Trustworthy Large Language Models, 2024.   
[3] E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojocaru, M. Debbah, É. Goffinet, D. Hesslow, J. Launay, Q. Malartic, et al. The falcon series of open language models. arXiv preprint arXiv:2311.16867, 2023.   
[4] F. Bach. Information theory with kernel methods. IEEE Transactions on Information Theory, 69(2):752-775, 2022.   
[5] I. Bengtsson and K. Życzkowski. Geometry of quantum states: an introduction to quantum entanglement. Cambridge university press, 2017.   
[6] V. Borovitskiy, I. Azangulov, A. Terenin, P. Mostowsky, M. Deisenroth, and N. Durrande. Matérn Gaussian processes on graphs. In International Conference on Artificial Intelligence and Statistics, pages 2593–2601. PMLR, 2021.   
[7] A. Budanitsky and G. Hirst. Evaluating wordnet-based measures of lexical semantic relatedness. Computational linguistics, 32(1):13–47, 2006.   
[8] C. Burns, H. Ye, D. Klein, and J. Steinhardt. Discovering latent knowledge in language models without supervision, 2022.   
[9] J. Chen and J. Mueller. Quantifying uncertainty in answers from any language model via intrinsic and extrinsic confidence assessment. arXiv preprint arXiv:2308.16175, 2023.   
[10] F. R. Chung. Spectral graph theory, volume 92. American Mathematical Soc., 1997.   
[11] J. Clusmann, F. R. Kolbinger, H. S. Muti, Z. I. Carrero, J.-N. Eckardt, N. G. Laleh, C. M. L. Löffler, S.-C. Schwarzkopf, M. Unger, G. P. Veldhuizen, et al. The future landscape of large language models in medicine. Communications medicine, 3(1):141, 2023.   
[12] R. Cohen, M. Hamri, M. Geva, and A. Globerson. LM vs LM: Detecting factual errors via cross examination. arXiv preprint arXiv:2305.13281, 2023.   
[13] J. R. Cole, M. J. Zhang, D. Gillick, J. M. Eisenschlos, B. Dhingra, and J. Eisenstein. Selectively answering ambiguous questions. Conference on Empirical Methods in Natural Language Processing, 2023.   
[14] D. Crystal. The Cambridge encyclopedia of the English language. Cambridge university press, 2018.   
[15] S. Desai and G. Durrett. Calibration of pre-trained transformers. arXiv preprint arXiv:2003.07892, 2020.   
[16] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal. Personal Communication, 2024.   
[17] P. Feldman, J. R. Foulds, and S. Pan. Trapping LLM hallucinations using tagged context prompts. arXiv preprint arXiv:2306.06085, 2023.   
[18] K. Filippova. Controlled hallucinations: Learning to generate faithfully from noisy data. arXiv preprint arXiv:2010.05873, 2020.   
[19] D. Friedman and A. B. Dieng. The Vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023.   
[20] Y. Gal and Z. Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, pages 1050–1059. PMLR, 2016.   
[21] D. Ganguli, A. Askell, N. Schiefer, T. I. Liao, K. Lukošiūtė, A. Chen, A. Goldie, A. Mirhoseini, C. Olsson, D. Hernandez, et al. The capacity for moral self-correction in large language models. arXiv preprint arXiv:2302.07459, 2023.   
[22] P. He, X. Liu, J. Gao, and W. Chen. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654, 2020.   
[23] D. Hendrycks, N. Carlini, J. Schulman, and J. Steinhardt. Unsolved problems in ML safety. arXiv preprint arXiv:2109.13916, 2021.   
[24] E. Hernandez, B. Z. Li, and J. Andreas. Measuring and manipulating knowledge representations in language models. arXiv preprint arXiv:2304.00740, 2023.

[25] R. A. Horn and C. R. Johnson. Matrix analysis. Cambridge university press, 2012.   
[26] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.   
[27] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
[28] Z. Jiang, J. Araki, H. Ding, and G. Neubig. How can we know when language models know? on the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9:962–977, 2021.   
[29] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
[30] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.   
[31] K. Kang, E. Wallace, C. Tomlin, A. Kumar, and S. Levine. Unfamiliar finetuning examples control how language models hallucinate. arXiv preprint arXiv:2403.05612, 2024.   
[32] E. Kasneci, K. Seßler, S. Küchemann, M. Bannert, D. Dementieva, F. Fischer, U. Gasser, G. Groh, S. Günnemann, E. Hüllermeier, et al. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274, 2023.   
[33] J. Kim, S. Kang, D. Hwang, J. Shin, and W. Rhee. Vne: An effective method for improving deep representation by manipulating eigenvalue distribution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3799–3810, 2023.   
[34] R. I. Kondor and J. Lafferty. Diffusion kernels on graphs and other discrete structures. In Proceedings of the 19th international conference on machine learning, volume 2002, pages 315-322, 2002.   
[35] A. Krithara, A. Nentidis, K. Bougiatiotis, and G. Paliouras. BioASQ-QA: A manually curated corpus for biomedical question answering. Scientific Data, 10(1):170, 2023.   
[36] L. Kuhn, Y. Gal, and S. Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664, 2023.   
[37] A. Kumar and S. Sarawagi. Calibration of encoder decoder models for neural machine translation. arXiv preprint arXiv:1903.00802, 2019.   
[38] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.   
[39] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.   
[40] H. Le, Y. Wang, A. D. Gotmare, S. Savarese, and S. C. H. Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.   
[41] K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36, 2024.   
[42] X. Li, R. Zhao, Y. K. Chia, B. Ding, S. Joty, S. Poria, and L. Bing. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources. In The Twelfth International Conference on Learning Representations, 2023.   
[43] S. Lin, J. Hilton, and O. Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.   
[44] Z. Lin, S. Trivedi, and J. Sun. Generating with confidence: Uncertainty quantification for black-box large language models. arXiv preprint arXiv:2305.19187, 2023.   
[45] J. Liu, Z. Lin, S. Padhy, D. Tran, T. Bedrax Weiss, and B. Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in neural information processing systems, 33:7498–7512, 2020.   
[46] S. Liu, L. Xing, and J. Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. arXiv preprint arXiv:2311.06668, 2023.   
[47] M. MacDiarmid, T. Maxwell, N. Schiefer, J. Mu, J. Kaplan, D. Duvenaud, S. Bowman, A. Tamkin, E. Perez, M. Sharma, C. Denison, and E. Hubinger. Simple probes can catch sleeper agents, 2024. URL https://www.anthropic.com/news/probes-catch-sleeper-agents.   
[48] D. J. MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.

[49] A. Malinin and M. Gales. Uncertainty estimation in autoregressive structured prediction. arXiv preprint arXiv:2002.07650, 2020.   
[50] P. Manakul, A. Liusie, and M. J. Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Conference on Empirical Methods in Natural Language Processing, 2023.   
[51] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald. On faithfulness and factuality in abstractive summarization. arXiv preprint arXiv:2005.00661, 2020.   
[52] S. J. Mielke, A. Szlam, Y.-L. Boureau, and E. Dinan. Linguistic calibration through metacognition: aligning dialogue agent responses with expected correctness. arXiv preprint arXiv:2012.14983, 11, 2020.   
[53] S. J. Mielke, A. Szlam, E. Dinan, and Y.-L. Boureau. Reducing conversational agents' overconfidence through linguistic calibration. Transactions of the Association for Computational Linguistics, 10:857–872, 2022.   
[54] J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. Torr, and Y. Gal. Deep deterministic uncertainty: A new simple baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24384–24394, 2023.   
[55] M. S. A. Nadeem, J.-D. Zucker, and B. Hanczar. Accuracy-rejection curves (ARCs) for comparing classification methods with a reject option. In Machine Learning in Systems Biology, pages 65–81. PMLR, 2009.   
[56] A. V. Nikitin, S. John, A. Solin, and S. Kaski. Non-separable spatio-temporal graph kernels via SPDEs. In International Conference on Artificial Intelligence and Statistics, pages 10640–10660. PMLR, 2022.   
[57] OpenAI. GPT-4 technical report. 2023.   
[58] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and J. Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. Advances in neural information processing systems, 32, 2019.   
[59] A. Patel, S. Bhattamishra, and N. Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.168. URL https://aclanthology.org/2021.naacl-main.168.   
[60] V. Quach, A. Fisch, T. Schuster, A. Yala, J. H. Sohn, T. S. Jaakkola, and R. Barzilay. Conformal language modeling. arXiv preprint arXiv:2306.10193, 2023.   
[61] P. Rajpurkar, R. Jia, and P. Liang. Know what you don't know: Unanswerable questions for SQuAD. arXiv preprint arXiv:1806.03822, 2018.   
[62] J. Ren, Y. Zhao, T. Vu, P. J. Liu, and B. Lakshminarayanan. Self-evaluation improves selective generation in large language models. arXiv preprint arXiv:2312.09300, 2023.   
[63] O. Roy and M. Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pages 606–610. IEEE, 2007.   
[64] C. E. Shannon. Prediction and entropy of printed english. Bell system technical journal, 30(1):50–64, 1951.   
[65] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: a family of highly capable multimodal models. 2023.   
[66] K. Tian, E. Mitchell, H. Yao, C. D. Manning, and C. Finn. Fine-tuning language models for factuality. arXiv, 2023.   
[67] K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, and C. D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv preprint arXiv:2305.14975, 2023.   
[68] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
[69] N. Varshney, W. Yao, H. Zhang, J. Chen, and D. Yu. A stitch in time saves nine: Detecting and mitigating hallucinations of llms by validating low-confidence generation. arXiv preprint arXiv:2307.03987, 2023.   
[70] U. Von Luxburg. A tutorial on spectral clustering. Statistics and computing, 17:395–416, 2007.   
[71] J. Von Neumann. Mathematical foundations of quantum mechanics: New edition, volume 53. Princeton university press, 2018.   
[72] Y. Xiao and W. Y. Wang. Quantifying uncertainties in natural language processing tasks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 7322-7329, 2019.

[73] Y. Xiao and W. Y. Wang. On hallucination and predictive uncertainty in conditional language generation. arXiv preprint arXiv:2103.15025, 2021.   
[74] A. X. Yang, M. Robeyns, X. Wang, and L. Aitchison. Bayesian low-rank adaptation for large language models. arXiv preprint arXiv:2308.13111, 2023.   
[75] C. Zhang, F. Liu, M. Basaldella, and N. Collier. Luq: Long-text uncertainty quantification for llms. arXiv preprint arXiv:2403.20279, 2024.   
[76] A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

# Supplementary Material:

# Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities

# A Background

# A.1 Linear Algebra

Definition A.1. For a set $\mathcal{X} \neq \emptyset$ , a symmetric function $K: \mathcal{X} \times \mathcal{X} \to \mathrm{R}$ is called a positive-semidefinite kernel if for all $n > 0$ , $x_i \in \mathcal{X}$ , $\alpha_i \in \mathbb{R}$

$$
\sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} \alpha_ {i} \alpha_ {j} K (x _ {i}, x _ {j}) \geq 0. \tag {A.1}
$$

For a finite set $\mathcal{X}$ , a positive semidefinite kernel is a positive semidefinite matrix of the size $|\mathcal{X}|$ .

Lemma A.2. For a block diagonal matrix

$$
A = \left( \begin{array}{c c c c c} A _ {1 1} & 0 & 0 & \ldots & 0 \\ 0 & A _ {2 2} & 0 & \ldots & 0 \\ 0 & 0 & A _ {3 3} & \ldots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & A _ {n n} \end{array} \right)
$$

eigenvalues are all eigenvalues of the blocks $A_{ii}$ combined, or equivalently $\det (A - \lambda I) = 0 \Leftrightarrow \prod_{i=1}^{n} \det(A_{ii} - \lambda I) = 0$

Proof. Notice, that a block diagonal matrix can be decomposed into the following product:

$$
A = \left( \begin{array}{c c c c} A _ {1 1} & 0 & \ldots & 0 \\ 0 & I _ {2 2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & I _ {n n} \end{array} \right) \left( \begin{array}{c c c c} I _ {1 1} & 0 & \ldots & 0 \\ 0 & A _ {2 2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & I _ {n n} \end{array} \right) \dots \left( \begin{array}{c c c c} I _ {1 1} & 0 & \ldots & 0 \\ 0 & I _ {2 2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & A _ {n n} \end{array} \right),
$$

where $I_{ii}$ are the identity matrices of the same size as $A_{ii}$ .

By using the product rule for determinants, we obtain $\det(A-\lambda I)=0\Leftrightarrow\prod_{i=1}^{n}\det(A_{ii}-\lambda I)=0.$

Lemma A.3 (Horn and Johnson [25]). An all-ones matrix $J$ of size $n$ has eigenvalues $\{n, \underbrace{0, \ldots, 0}_{n-1}\}$ .

# A.2 Discrete Mathematics

Throughout the text, we often refer to the notion of equivalence relation. We remind readers of the definition of equivalence relation here.

Definition A.4. Equivalence relation is a binary relation $E(\cdot, \cdot)$ on a set $\mathcal{X}$ , that is for any $x, y, z \in \mathcal{X}$ , this relation is

1. reflexive $E(x, x)$ ,   
2. symmetric $E(x,y) \Longleftrightarrow E(y,x)$ ,   
3. transitive if $E(x,y)$ and $E(y,z)$ then $E(x,z)$ .

# B Theoretical Results and Proofs

In this section, we prove Thm. 3.5, for convenience we separate it into two theorems for KLE and KLE-c.

Theorem B.1 (KLE is a generalization of SE). For any semantic clustering, there exists a semantic kernel over texts $K_{sem}(s, s')$ such that the VNE of this kernel is equal to semantic entropy (computed as in Eq. (5)).

Proof. Let us fix an arbitrary semantic clustering over M clusters $C = \{C_{1}, \ldots, C_{M}\}$ , with the size of each cluster $m_{i}$ . Now, we will construct a kernel K for an input x such that the von Neumann entropy with this kernel will be equal to the semantic entropy of the texts $\mathrm{VNE}(K) = \mathrm{SE}(x, \mathcal{C})$ . Let us consider a block-diagonal kernel K. We will denote blocks of K as $K_{1}, \ldots, K_{M}$ :

$$
K = \left( \begin{array}{c c c c c} K _ {1} & 0 & 0 & \dots & 0 \\ 0 & K _ {2} & 0 & \dots & 0 \\ 0 & 0 & K _ {3} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & K _ {M} \end{array} \right) \tag {B.1}
$$

where $M$ corresponds to the number of semantic clusters. The size of each block $K_{i}$ is $m_{i} \times m_{i}$ . Note that because $K$ is block-diagonal, it follows that $\mathrm{VNE}(K) = \sum_{i=1}^{M} \mathrm{VNE}(K_{i})$ . Consequently, if

1. VNE( $K_{i}$ ) = -p( $C_{i}|x$ ) log p( $C_{i}|x$ ),   
2. the sum of eigenvalues of $K_{i}$ is equal to $p(C_i|x)$ ,   
3. $K$ is positive semidefinite and unit trace,

then VNE(K) = SE(s|x).

Let us define each block as $K_{i} = \frac{p(C_{i}|x)}{m_{i}} J_{m_{i}}$ where $J_{m_i}$ is an all-ones matrix of size $m_{i} \times m_{i}$ .

Next, we prove that the desired properties from the list above hold. Indeed, the eigenvalues of $K_{i}$ are $p(C_{i}|x)$ with multiplicity one and 0 with multiplicity $m_{i}-1$ . So, $\mathrm{VNE}(K_{i})=-p(C_{i}|x)\log p(C_{i}|x)$ (recall that for calculating VN entropy, we assume $0\log0=0$ ), and Properties 1 and 2 are fulfilled. K is also symmetric and has non-negative eigenvalues. Thus, Property 3 is fulfilled as well.

Because K satisfies all properties, we have proven that $\mathrm{VNE}(K(s,x))=\mathrm{SE}(s|x)$ .

![](images/f574e1a9a60d11f31c7cc4b38058e35c2e9da0eaeead00c412e41c50d350865d.jpg)

Theorem B.2 (KLE-c is more general than SE). For any semantic clustering, there exists a kernel over semantic clusters $K_{s}(c,c')$ such that the VNE of this kernel is equal to semantic entropy (computed as in Eq. (5)).

Proof. Analogously to Thm. B.1 but with the blocks of size one.

![](images/b2c14c52a2feb65aa659b1c04eb32082223752df8f02c085d6c9b50691303c33.jpg)

The theorems not only show that KLE generalizes SE but also provide an explicit form for a semantic kernel that can be used with KLE to recover SE.

# C Kernel Hyperparameters

Following the discussion about kernel hyperparameters selection from Sec. 3.1, we visualize entropy convergence plots for Heat kernels in Fig. 2 and visualize heat and Matérn kernels on 2-d grid in Fig. C.4. Next, we expand on the question of parameter sensitivity, in Fig. C.3, and whether it is necessary to use a validation set for selecting kernel hyperparameters. We observe that both with reasonable default choices (t = 0.3, $\alpha = 0.5$ , $\nu = 1$ , and $\kappa = 1$ ) and by selecting hyperparameters on a separate set of answers, we outperform the existing methods. When choosing hyperparameters, we also have included a boolean flag whether the graph Laplacian should be normalized, as, generally

![](images/b19779c9b11053427585618cf3c6c7c74ab28cfd8e18e14a06fb32f5d667b700.jpg)  
Figure C.1: Matérn Entropy Convergence Plots.

speaking, both the normalized and the standard graph Laplacians can be used with heat and Matérn kernels

$$
L _ {\mathrm{n}} = \left(D ^ {+}\right) ^ {1 / 2} L \left(D ^ {+}\right) ^ {1 / 2}, \tag {C.1}
$$

where $D^{+}$ is the Moore-Penrose inverse of the degree matrix $D$ . We observe similar results when analyzing other semantic kernels.

Prompts. We prompt the models to generate full sentences as answers with the following prompt: Answer the following question in a single brief but complete sentence..

Also, we have used the following prompt to check the accuracy of the responses:

We are assessing the quality of answers to the following question:

{question} \n The expected answer is: {correct\_answer}. \n The proposed answer is: {predicted\_answer} \n Within the context of the question, does the proposed answer mean the same as the expected answer? \n Respond only with yes or no.\n Response:

Here we mark placeholders with the orange color. Or, if several correct answers were provided, we have used the following prompt:

![](images/0a261b5133358a5615ef0ec86c68384e79a01e625b0e058ffc0e59a169a258fd.jpg)

<details>
<summary>text_image</summary>

Question
who is responsible for introducing the principle of uniformitarianism?
Answer
James Hutton is credited with introducing the principle of uniformitarianism in geology.
0
James Hutton is credited with introducing the principle of uniformitarianism in geology.
5
Scottish geologist James Hutton is credited with introducing the principle of uniformitarianism in 1785.
1
James Hutton, an 18th-century Scottish geologist and physician, is credited with introducing the principle of uniformitarianism, which states that the geological processes observed in the present are the same processes that have shaped the Earth throughout its history.
6
James Hutton is credited with introducing the principle of uniformitarianism, which states that geological phenomena have been produced by the same causes and processes that operate today, in his 1785 book "The Theory of the Earth".
2
James Hutton is credited with introducing the principle of uniformitarianism in geology.
7
James Hutton, a Scottish geologist and physician, is credited with introducing the principle of uniformitarianism in geology.
3
James Hutton.
8
The principle of uniformitarianism was introduced by James Hutton, a Scottish geologist, in the 18th century.
4
James Hutton, an 18th-century Scottish physician and geologist, is credited with introducing the principle of uniformitarianism in geology.
8
James Hutton is credited with introducing the principle of uniformitarianism, which states that geological processes occurring today are the same processes that have shaped the Earth throughout history.
</details>

![](images/e0832c0058f10a172cf126eb99220d626ce36a0c1baeba37f2790626e11ced86.jpg)

Figure C.2: Example from NQ.   
![](images/7642b5411f5a1f113a58cff142c125dd3930fa99b86f94af4222e1e8652597fa.jpg)  
(a) AUROC: Default hyperparameters

![](images/76d542d2935e08a3f709524ef5136494ee5273c9c60c9a2e483795e0ee841ef2.jpg)  
(b) AUROC: Validation set hyperparameters

![](images/710da506357d4c92007e98e164372bfa6e70ebae24c1a7adba5e4a8b708db4bc.jpg)  
(c) AUARC: Default hyperparameters

![](images/e22d18777d9b103a940e16ab57f0718acab6377138ed538bbde926bfcd8bf8ab.jpg)  
(d) AUARC: Validation set hyperparameters   
Figure C.3: Summary of 60 experimental scenarios. Comparing hyperparameters selection strategies. Our methods are labeled KLE( $\cdot$ ).

We are assessing the quality of answers to the following question: {question} \n The following are expected answers to this question: {correct\_answers}. \n The proposed answer is: {predicted\_answer} \n Within

the context of the question, does the proposed answer mean the same as any of the expected answers? \n Respond only with yes or no.\n Response:

Example. We visualize an example from the NQ dataset in Fig. C.2; we have used Llama-2 70B Chat for this example. In order to analyze cases where SE and KLE are inconsistent, we ranked all the answers separately by KLE and SE and found those cases where the difference between indices in the list ranked by KLE and ranked by SE is high. In Fig. C.2, a model provides the correct answer. However, SE estimates the uncertainty to be high because it can detect only two answers as equal and thus considers the majority of the answers as semantically distinct. Instead, our method considers more fine-grained relations between the answers and provides better uncertainty estimates (i.e., orange and red cells in the weight matrix). It is an illustrative example of the cases we analyzed. It indicates that the longer and more nuanced the answers are, the more KLE would outperform SE.

![](images/ddefb569ed9d5d827436af10d2b29f8f3a58f8b39a1a2e2e89c9043d28834479.jpg)

<details>
<summary>text_image</summary>

Heat Kernel (t = 1)
</details>

![](images/4740883cfbe9a9a0996d2c680a852a32614d8931d2ca7004080ebbc09d2c2eeb.jpg)

<details>
<summary>chemical</summary>

Matérn Kernel diagram showing lattice structure with ν=5/2, κ=3
</details>

![](images/8b393d6b44cf745ec653c61c8567d83e3e025f3470fa7cbc829139d7d025b077.jpg)  
Figure C.4: Heat and Matérn kernels visualized on 2-d grid.

# D Additional Experimental Details

In this section, we provide additional experimental results.

Hardware and Resources. We ran Llama 2 70B models on two NVIDIA A100 80GB GPUs, and the rest of the models on a single NVIDIA A100 80GB. The generation process took from one to seven hours depending on a model for each experimental scenario, and the evaluation additionally took roughly four hours per scenario which can be further optimized by reducing the number of hyperparameters. The project spent more resources due to other experiments. Our experimental pipeline first generates the answers for all the datasets and then computes various uncertainty measures. We did not recompute generations, but in each experimental run we only evaluated uncertainty measures.

Licenses. We release our code under a clear BSD-3-Clause-Clear. The datasets used in this paper are released under CC BY 2.5 (BioASQ [35]), Apache 2.0 (TriviaQA [29]), CC BY-SA 4.0 (SQuAD [61]), MIT (SVAMP, [59]), and CC BY-SA 3.0 (NQ [38]).

# D.1 Models and datasets

In Fig. D.1, we show samples from each dataset we used in the experimental evaluation of our method.

![](images/e53b63eafb02dfda5e60497971770caf8ab3c434ca4116065b7251b2f15da5c4.jpg)

<details>
<summary>text_image</summary>

Trivia QA
Question: Correct Answer:
What city, Chile's second largest,
suffered an 8.8 earthquake in
2010?
Concepción
NQ
Question: Correct Answer:
Who played the gorilla
in the cadbury advert?
Garon Michael
SQuAD
Question: Correct Answer:
In what year of 20th century, did
Harvard release an important
document about education in
America?
1945
BioASQ
Question: Correct Answer:
What is the purpose of
Macropinocytosis?
Macropinocytosis is an endocytic
process, which involves the engulfment
of extra-cellular content in vesicles
known as macropinosomes.
Context
Steven has 14 peaches.
Jake has 6 fewer peaches than
Steven and 3 more peaches
than Jill.
SVAMP
Question: Correct Answer:
How many
peaches does Jill have?
5
</details>

Figure D.1: Samples from datasets we use: Trivia QA, NQ, SQuAD, BioASQ, and SVAMP.

Additionally, we demonstrate the accuracy of the models used in the experiments on each dataset in Fig. D.2. As can be seen, we evaluate our method on a diverse set of models with a varying level of

![](images/72af177627a4bfca354703fa5ab1c6e0c19bba26efc1452b2243cafa7f691d23.jpg)

<details>
<summary>bar</summary>

| Model | TrivialQA | NQ Open | SVAMP | SquAD | BioASQ |
|---|---|---|---|---|---|
| Materal7B | 0.88 | 0.65 | 0.55 | 0.59 | 0.79 |
| Materal7B testect | 0.62 | 0.33 | 0.54 | 0.20 | 0.45 |
| Falcon7B | 0.68 | 0.38 | 0.21 | 0.24 | 0.50 |
| Falcon7B testect | 0.59 | 0.28 | 0.26 | 0.15 | 0.43 |
| Falcon40B | 0.84 | 0.57 | 0.51 | 0.28 | 0.57 |
| Falcon40B testect | 0.82 | 0.53 | 0.58 | 0.28 | 0.54 |
| Llama 27B | 0.78 | 0.43 | 0.41 | 0.22 | 0.45 |
| Llama 2 Chat7B | 0.79 | 0.48 | 0.51 | 0.27 | 0.51 |
| Llama 213B | 0.83 | 0.46 | 0.49 | 0.33 | 0.57 |
| Llama 2 Chat13B | 0.83 | 0.53 | 0.62 | 0.31 | 0.46 |
| Llama 270B | 0.89 | 0.54 | 0.62 | 0.28 | 0.59 |
| Llama 2 Chat73B | 0.88 | 0.59 | 0.69 | 0.35 | 0.56 |
</details>

Figure D.2: Accuracy of the models

accuracy across the tasks at hand. This is especially important for UQ, because UQ methods should perform well for all the models regardless of their downstream effectiveness.

Real-world applications often involve deploying models with varying degrees of performance, and a robust UQ method should provide reliable uncertainty estimates for all of them. By demonstrating the efficacy of our method across a wide variety of models, we validate its applicability in diverse scenarios. This highlights that our approach can be confidently used in practical settings where model performance can fluctuate.

# D.2 Instruction-tuned and non-instruction tuned models

Furthermore, we investigate the performance of UQ methods by splitting the set of experimental scenarios into instruction-tuned and non-instruction tuned models. We visualize the splits in Fig. D.5. Interestingly, our approach significantly outperforms the existing methods when evaluated with instruction-tuned models, and only marginally outperforms when evaluated on non-instruction-tuned models. We can hypothesize that non-instruction tuned models are better calibrated and thus methods based on token-likelihoods make sense whereas with instruction tuning worsens calibration. This hypothesis is also supported by comparison of SE and DSE (DSE significantly outperforms SE on an instruction-tuned split, when AUROC is measured).

# D.3 Detailed results of UQ

We provide a detailed comparison of our method with previous uncertainty quantification measures. In Fig. D.3 and Fig. D.4, we show the results for a wide range of models across five datasets for non-instruction tuned and instruction-tuned models, respectively. We want to note that ER has failed for Llama 2 13B (non-instruction tuned version) for all datasets except BioASQ because training datasets for ER contained samples of only one class. We have assigned zero score to the failed cases.

# E Additional Notes

# E.1 Lexical, semantic, and syntactic variability

Table 2: Examples of semantic, syntactic, and lexical variability of a sentence “Paris is the capital of France.” 

<table><tr><td></td><td>Semantic Variability</td><td>Syntactic Variability</td><td>Lexical Variability</td></tr><tr><td>Paris is the capital of France.</td><td>Rome is capital of FranceParis is the capital of Italy.</td><td>The capital of France is Paris.</td><td>France’s capital is situated in Paris.France’s capital city is Paris.</td></tr></table>

We resort to the 6-level model of the structure for text analysis proposed in [14] to extensively describe aspects of language beyond semantics. This model distinguishes four basic notions for text analysis: medium of transmission, grammar, semantics, and pragmatics. Medium of transmission is irrelevant to the study of language model outputs (however, it becomes relevant for multimodal foundation models that can, for instance, answer a request either with a text or an image); grammar is further divided into the syntax and morphology of the text and semantics into semantics and

![](images/96a1429e005d95f096d3347994f1a3a30624ee0c7031e6c9177eea74a11244c7.jpg)  
Figure D.3: Full results of non-instruction tuned models

discourse. Another dimension is pragmatics, or how the text is used. In this work, we focus only on the semantics of the text. However, the method can be extended to other aspects of text analysis. For instance, one can design syntactic or pragmatic kernels. We leave the study of other kernel modalities to future works.

![](images/366069c265e9ec439c611fc2b62a69144f6aa6504a7cfb12e4b36a9260697bb0.jpg)  
Figure D.4: Full results of instruction-tuned models

![](images/55de8bc1d538d14a605a2f0409ea2bba5e5f7bf27002bdea8fd4fc81587000fd.jpg)

<details>
<summary>heatmap</summary>

| | KLE(K_FULL) | KLE(K_HEAT) | SE | DSE | P(True) | PE | ER |
|---|---|---|---|---|---|---|---|
| KLE(K_FULL) | 0.20 | 0.93 | 0.90 | 0.77 | 0.93 | 0.90 | |
| KLE(K_HEAT) | 0.80 | 0.93 | 0.90 | 0.90 | 0.97 | 0.93 | |
| SE | 0.07 | 0.07 | 0.07 | 0.43 | 0.80 | 0.70 | |
| DSE | 0.10 | 0.10 | 0.93 | 0.50 | 0.87 | 0.80 | |
| P(True) | 0.23 | 0.10 | 0.57 | 0.50 | 0.70 | 0.63 | |
| PE | 0.07 | 0.03 | 0.20 | 0.13 | 0.30 | 0.50 | |
| ER | 0.10 | 0.07 | 0.30 | 0.20 | 0.37 | 0.50 | |
</details>

(a) AUROC instruction-tuned models

![](images/f8efcb092f97504d98e009b1141ef0655f6b2730afe951e3a459a29961d33f78.jpg)  
(b) AUROC: non-instruction tuned models

![](images/4162051cf3b967a87b67d3dcb8003dcb7b9ad0c312a2f9d5923661bd961625ce.jpg)  
(c) AUARC: instruction-tuned models

![](images/da3fd5c64677621544b3def081c01a9a374a910494029c74323e6f23a6684a83.jpg)  
(d) AUARC: non-instruction tuned models   
Figure D.5: Summary of 60 experimental scenarios. Comparing the results on instruction-tuned and non-instruction tuned models. Our methods are labeled $\mathrm{KLE}(\cdot)$ .