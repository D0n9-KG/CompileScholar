# Disentangling and Integrating Relational and Sensory Information in Transformer Architectures

Awni Altabaa $^{1}$ John Lafferty $^{2}$

# Abstract

Relational reasoning is a central component of generally intelligent systems, enabling robust and data-efficient inductive generalization. Recent empirical evidence shows that many existing neural architectures, including Transformers, struggle with tasks requiring relational reasoning. In this work, we distinguish between two types of information: sensory information about the properties of individual objects, and relational information about the relationships between objects. While neural attention provides a powerful mechanism for controlling the flow of sensory information between objects, the Transformer lacks an explicit computational mechanism for routing and processing relational information. To address this limitation, we propose an architectural extension of the Transformer framework that we call the Dual Attention Transformer (DAT), featuring two distinct attention mechanisms: sensory attention for directing the flow of sensory information, and a novel relational attention mechanism for directing the flow of relational information. We empirically evaluate DAT on a diverse set of tasks ranging from synthetic relational benchmarks to complex real-world tasks such as language modeling and visual processing. Our results demonstrate that integrating explicit relational computational mechanisms into the Transformer architecture leads to significant performance gains in terms of data efficiency and parameter efficiency.

$^{1}$ Department of Statistics and Data Science, Yale University $^{2}$ Department of Statistics and Data Science, Wu Tsai Institute, Yale University. Correspondence to: Awni Altabaa <awni.altabaa@yale.edu>, John Lafferty <john.lafferty@yale.edu>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# 1. Introduction

A central goal of machine learning research is to develop a universal architecture capable of learning and reasoning across a wide range of tasks and data modalities. Theoretical frameworks for understanding human and animal intelligence seek to explain intelligent behavior through a small set of fundamental principles $[1]$ . However, in machine intelligence, there is a tension between the objective of developing a general architecture and the need to incorporate inductive biases that are beneficial for specific tasks $[2, 3]$ . When faced with finite training data and numerous solutions to empirical risk minimization, inductive biases steer the learning algorithm towards solutions with desirable properties, enhancing data efficiency and generalization. A core scientific challenge of machine learning is to identify a complete and broadly applicable set of inductive biases that promote robust, flexible, and data-efficient learning across a diverse set of problems.

Relational reasoning is a central component of generally intelligent systems and is believed to underlie human abilities for abstraction and systematic generalization $[4–7]$ . The power of relational reasoning lies in its capacity to generate inferences and generalizations in systematic and novel ways, even across instances which are superficially very different but have similarities on an abstract structural level. This grants humans a capacity for out-of-distribution generalization, data efficiency, and continual learning, which modern machine learning is yet to match $[8–10]$ . Replicating this ability in machine intelligence can ultimately lead to universal inductive generalization from a finite set of observations to an infinite set of novel instances $[11]$ .

The ability of artificial intelligence systems to perform relational reasoning has long been an area of active study across various approaches to AI. In symbolic modeling frameworks, the relationships between symbols are explicitly defined in the language of logic and mathematics $[12–14]$ . However, such approaches often require hand-crafted representations and suffer from the symbol grounding problem, limiting their ability to generalize to variations in the task or input outside a certain domain. By contrast, deep learning approaches build data-dependent representations that are, in principle, capable of generalizing across diverse conditions.

However, recent work exploring the ability of deep learning models to learn relational tasks finds that seemingly simple relational inferences can be remarkably difficult for powerful neural network architectures $[15–23]$ . An emerging hypothesis, which we explore further here, explains this through the lens of inductive biases, arguing that neural networks struggle with relational reasoning because they overemphasize individual object representations—sensory information—while lacking explicit mechanisms for encoding and processing relational information $[24]$ .

In this work, we explore this idea in the context of the Transformer architecture $[25]$ , which offers a promising starting point for building a versatile, general-purpose neural architecture. Although Transformers, like other neural architectures, struggle to learn abstract relational representations $[19–23]$ , there is encouraging evidence that Transformer-based foundation models acquire some relational reasoning ability $[26–32]$ through large-scale training on large amounts of data $[33–36]$ . This presents an opportunity to explore Transformer-based architectures with built-in neural mechanisms and inductive biases for explicit and enhanced relational reasoning capabilities, seeking to imbue the Transformer architecture with a greater capacity for efficient induction of abstractions.

To introduce our proposal, we highlight a distinction between two types of information which are encoded in the internal representation of Transformer models: sensory information, which represents features of individual objects, and relational information, representing comparisons and relationships between objects. In the standard attention mechanism of Transformers, relational information is entangled with sensory information, limiting the model's ability to learn to explicitly represent and reason about the relationships between objects. In particular, in standard attention, relational information directs the flow of information (i.e., attention scores encoding a selection criterion), while the values routed are representations of the sensory attributes of individual objects. In this work, we explore a new type of attention mechanism we call relational attention, in which the values being routed are themselves representations of relationships between the source and the target, computed explicitly as a series of comparisons under learned feature maps. This equips the model with a mechanism for explicitly routing and processing relational information, decoupling it from sensory features.

We integrate relational attention with the standard attention mechanism of Transformers, yielding a variant of multi-head attention that processes both sensory and relational information in parallel. The resulting Dual Attention Transformer (DAT) architecture disentangles the two types of information during the information retrieval stage and integrates them during the local processing stage of each layer. We empirically evaluate this architecture on a diverse set of tasks, ranging from synthetic benchmarks on relational reasoning to complex real-world tasks such as language modeling and visual processing. Our results demonstrate that integrating explicit relational mechanisms into the Transformer architecture leads to significant performance gains in terms of data efficiency and parameter efficiency.

# 2. Disentangling Attention over Sensory and Relational Information

# 2.1. Standard Attention: Attention over Sensory Information

The attention mechanism of standard Transformers can be understood as a differentiable computational mechanism for dynamically routing sensory information between different elements in the input. An object emits a query that is compared against the keys of each object in its context via an inner product. A “match” occurs when the inner product is large, causing an encoding of the features of the attended object to be retrieved and added to the residual stream of the receiver. Formally, attention between an object $x \in R^{d}$ and a context $\boldsymbol{y} = (y_{1}, \ldots, y_{n}) \in \mathbb{R}^{n \times d}$ takes the form

$$
\operatorname{Attention} (x, (y _ {1}, \dots , y _ {n}))
$$

$$
= \sum_ {i = 1} ^ {n} \alpha_ {i} (x, \boldsymbol {y}) \phi_ {v} (y _ {i}), \text { where }, \tag {1}
$$

$$
\alpha (x, \boldsymbol {y}) = \operatorname{Softmax} \bigl (\left[ \left\langle \phi_ {q} ^ {\mathrm{attn}} (x), \phi_ {k} ^ {\mathrm{attn}} (y _ {i}) \right\rangle \right] _ {i = 1} ^ {n} \bigr),
$$

where $\phi_{q}^{attn}, \phi_{k}^{attn}$ are learnable query and key maps controlling the selection criterion, and $\phi_{v}$ is a learnable value map controlling what information about $y_{i}$ is sent. The attention scores $\alpha(x, \mathbf{y})$ are used to retrieve a convex combination of the values, where $\alpha_{i}(x, \mathbf{y})$ denotes the i-th component.

Here, the retrieved information is sensory, comprising the features and attributes of individual objects in the context. For this reason, we refer to standard neural attention as “sensory attention”.

# 2.2. Relational Attention: Attention over Relational Information

Standard neural attention does not explicitly capture information about the relationship between the source/sender and the target/receiver, making relational learning in standard Transformers inefficient $[15, 16, 18–24, 37]$ . We propose relational attention, an attention mechanism for dynamically routing relational information between objects.

Mirroring standard attention, this operation begins with each object emitting a query and a key, which are compared via an inner product to compute attention scores determining which objects to attend to. Next, instead of retrieving the

![](images/54c050b6f4f19549b5f1c1d5731c42aeb076f15196adc40d59249fa458fe6b3e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    x["x"] -->|α₁| v1["v₁"]
    x -->|α₂| v2["v₂"]
    x -->|αₙ| vn["vₙ"]
    v1 --> y1["y₁"]
    v2 --> y2["y₂"]
    vn --> yn["yₙ"]
    style x fill:#f9f,stroke:#333
    style v1 fill:#ccf,stroke:#333
    style v2 fill:#ccf,stroke:#333
    style vn fill:#ccf,stroke:#333
```
</details>

(a) Attention $(x, (y_{1}, \ldots, y_{n}))$

![](images/d39e55353ebaaa93e76bc2dc39821c641e69a6b8a6da3bae4b3c97f3ee313681.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x"] -->|α₁| B["r(x,y₁) + s₁"]
    A -->|α₂| C["r(x,y₂) + s₂"]
    A -->|αₙ| D["r(x,yₙ) + sₙ"]
    B --> E["y₁"]
    C --> F["y₂"]
    D --> G["yₙ"]
    style A fill:#ff0000,stroke:#333
    style B fill:#66ccff,stroke:#333
    style C fill:#66ccff,stroke:#333
    style D fill:#66ccff,stroke:#333
    style E fill:#fff,stroke:#333
    style F fill:#fff,stroke:#333
    style G fill:#fff,stroke:#333
```
</details>

(b) RelationalAttention $(x,(y_1,\dots ,y_n))$   
Figure 1. Standard self-attention retrieves sensory information $v_{i}$ about the attributes of individual objects while relational attention retrieves relational information $r(x, y_{i})$ about the relationship between the objects in the context and the target. Each relation is tagged with a symbol $s_{i}$ which acts as an abstract variable identifying the source. In both cases, information is aggregated according to the attention scores $\alpha_{i}$ , which are computed by a softmax over inner products of queries and keys.

sensory features of the selected object, relational attention retrieves the relation between the two objects—defined as a series of comparisons between the two objects under different feature subspaces. In addition, a symbolic identifier is sent to indicate the identity of the sender to the receiver. Formally, this operation is defined as follows.

RelationalAttention $(x,(y_{1},\dots ,y_{n}))$

$$
= \sum_ {i = 1} ^ {n} \alpha_ {i} (x, \boldsymbol {y}) \big (r (x, y _ {i}) W _ {r} + s _ {i} W _ {s} \big), \text {   where },
$$

$$
\alpha (x, \boldsymbol {y}) = \operatorname{Softmax} \big (\left[ \left\langle \phi_ {q} ^ {\mathrm{attn}} (x), \phi_ {k} ^ {\mathrm{attn}} (y _ {i}) \right\rangle \right] _ {i = 1} ^ {n} \big),
$$

$$
r (x, y _ {i}) = \big (\left<   \phi_ {q, \ell} ^ {\mathrm{rel}} (x), \phi_ {k, \ell} ^ {\mathrm{rel}} (y _ {i}) \right> \big) _ {\ell \in [ d _ {r} ]},
$$

$$
\left(s _ {1}, \dots , s _ {n}\right) = \text { SymbolRetriever } (\boldsymbol {y}; S _ {\mathrm{lib}}) \tag {2}
$$

Thus, relational attention between the object $x$ and the context $\boldsymbol{y} = (y_1, \ldots, y_n)$ retrieves a convex combination of the relation vectors $\{r(x, y_i)\}_{i=1}^n$ , representing $x$ 's relationship with each object in the context. Relational attention also retrieves a symbol vector $s_i$ that encodes the identity information of the attended object. The role and implementation of the symbols will be discussed in the next subsection. As with standard attention, $\phi_q^{\text{attn}}$ , $\phi_k^{\text{attn}}$ are learned feature maps that govern the attention selection criterion. A separate set of query and key feature maps, $\phi_{q,\ell}^{\text{rel}}$ , $\phi_{k,\ell}^{\text{rel}}$ , $\ell \in [d_r]$ , are learned to represent the relation between the sender and the receiver. For each $\ell \in [d_r]$ , the feature maps $\phi_{q,\ell}^{\text{rel}}$ , $\phi_{k,\ell}^{\text{rel}}$ extract specific attributes from the object pair, which are compared by an inner product. This produces a $d_r$ -dimensional relation vector representing a fine-grained series of comparisons $(\langle \phi_{q,\ell}^{\text{rel}}(x), \phi_{k,\ell}^{\text{rel}}(y_i) \rangle)_{\ell \in [d_r]}$ across different feature subspaces.

In certain tasks [21-23], a useful inductive bias on the relations function $r(\cdot, \cdot)$ is symmetry; i.e., $r(x, y) = r(y, x)$ , $\forall x, y$ . This corresponds to using the same feature filter for the query and key maps, $\phi_q^{\mathrm{rel}} = \phi_k^{\mathrm{rel}}$ . This adds structure to the relation function, transforming it into a positive semi-definite kernel that defines a pseudometric on the object space, along with a corresponding geometry.

# 2.3. Symbol Assignment Mechanisms

To process relational information effectively, the receiver must have two pieces of information: 1) its relationship to the objects in its context, and 2) the identity of the object associated with each relation. In relational attention, the former is captured by $r(x, y_{i})$ and the latter by $s_{i}$ . The symbols $s_{i}$ are used to tag each relation with the identity information of the sender.

The symbol $s_{i}$ identifies or points to the object $y_{i}$ , but, importantly, is designed to not fully encode the features of the object. Instead, the symbols $s_{i}$ function as abstract references to objects, perhaps viewed as a connectionist analog of pointers in traditional symbolic systems. In particular, by drawing symbol vectors from a finite library $S_{lib}$ , relational attention maintains a relation-centric representation. This separation between sensory and relational information is key to decoupling relational attention disentangled from sensory features, enabling generalization across relations.

The notion of the “identity” of an object can vary depending on context. In this work, we consider modeling three types of identifiers: 1) position, 2) relative position, or 3) an equivalence class over features. For each type of identifier, we model a corresponding symbol assignment mechanism [22]. We find that different symbol assignment mechanisms are more effective in different domains.

Positional Symbols. In some applications, it is sufficient to identify objects through their position in the input sequence. We maintain a library of symbols $S_{\mathrm{lib}} = (s_1, \ldots, s_{\max\_len}) \in \mathbb{R}^{\max\_len \times d}$ and assign $s_i$ to the i-th object in the sequence. These are essentially learned positional embeddings.

Position-Relative Symbols. Often, the relative position with respect to the receiver is a more useful identifier than absolute position. This can be implemented with position-relative embeddings. We learn a symbol library $S_{\mathrm{lib}} = (s_{-\Delta}, \ldots, s_{-1}, s_0, s_1, \ldots, s_\Delta) \in \mathbb{R}^{(2\Delta+1) \times d}$ , where $\Delta$ is the maximum relative position, and relational attention becomes $\sum_j \alpha_{ij}(r(x_i, x_j) W_r + s_{j-i} W_s)$ .

Symbolic Attention. In certain domains, some information about the objects' features is necessary to identify them for the purposes of relational processing. Yet, to maintain a relational inductive bias, we would like to avoid a full encoding of object-level features. In symbolic attention, we learn a set of symbol vectors, $S_{\mathrm{lib}} = (s_1, \ldots, s_{n_s}) \in \mathbb{R}^{n_s \times d}$ and a matching set of feature templates $F_{\mathrm{lib}} = (f_1, \ldots, f_{n_s})$ . We retrieve a symbol for each object by an attention operation that matches the input vectors $x_i$ against the feature templates $f_j$ and retrieves symbols $s_j$ .

$$
\text { SymbolicAttention } (\boldsymbol {x}) = \operatorname{Softmax} \left(\left(\boldsymbol {x} W _ {q}\right) F _ {\mathrm{lib}} ^ {\top}\right) S _ {\mathrm{lib}}. \tag {3}
$$

Here, $S_{lib}$ , $F_{lib}$ , $W_{q}$ are learned parameters. This can be thought of as implementing a learned differentiable “equivalence class map” over feature embeddings. Crucially, the number of symbols (i.e., feature equivalence classes) is finite, which enables relational attention to still produce a relation-centric representation while tagging the relations with the necessary identifier.

# 2.4. What Class of Functions can Relational Attention Compute?

To give some intuition about the type of computation that relational attention can perform, we present the following expressivity result. The following theorem states that relational attention can approximate any function on $\mathcal{X} \times \mathcal{Y}^n$ that 1) selects an element in $(y_1, \ldots, y_n)$ , then 2) computes a relation with it. Both the selection criterion and the relation function are arbitrary, and the selection criterion can be query-dependent. The formal statement and proof are given in Appendix A.

Theorem 1 (Informal). Let Select: $\mathcal{X} \times \mathcal{Y}^n \to \mathcal{Y}$ be an arbitrary preference selection function, which selects an element among $(y_1, \ldots, y_n)$ based on a query-dependent preorder relation $\{\preccurlyeq_x\}_{x \in \mathcal{X}}$ . Let $\operatorname{Rel}: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^{d_r}$ be an arbitrary continuous relation function on $\mathcal{X} \times \mathcal{Y}$ . There exists a relational attention module that approximates the function $\operatorname{Rel}(x, \operatorname{Select}(x, \boldsymbol{y}))$ to arbitrary precision.

Algorithm 1: Dual Attention   
$\begin{array}{rl} & \textbf{Input: } \boldsymbol{x} = (x_1, \ldots, x_n) \in \mathbb{R}^{n \times d} \\ & \textbf{Compute self-attention heads} \\ & \boldsymbol{\alpha}^{(h)} \leftarrow \text{Softmax}\big((\boldsymbol{x}W_{q,h}^{\text{attn}})(\boldsymbol{x}W_{k,h}^{\text{attn}})^{\intercal}\big), \quad h \in [n_h^{sa}] \\ & e_i^{(h)} \leftarrow \sum_j \alpha_{ij}^{(h)}x_j W_v^h, \quad i \in [n], h \in [n_h^{sa}] \\ & e_i \leftarrow \text{concat}\big(e_i^{(1)}, \ldots, e_i^{(n_h^{sa})}\big)W_o^{sa}, \quad i \in [n] \\ & \textbf{Assign symbols:} \\ & \boldsymbol{s} = (s_1, \ldots, s_n) \leftarrow \text{SymbolRetriever}(\boldsymbol{x}; S_{\text{lib}}) \\ & \textbf{Compute relational attention heads} \\ & \boldsymbol{\alpha}^{(h)} \leftarrow \text{Softmax}\big((\boldsymbol{x}W_{q,h}^{\text{attn}})(\boldsymbol{x}W_{k,h}^{\text{attn}})^{\intercal}\big), \quad h \in [n_h^{ra}] \\ & \boldsymbol{r}_{ij} \leftarrow \big(\langle x_iW_{q,\ell}^{\text{rel}}, x_jW_{k,\ell}^{\text{rel}}\rangle\big)_{\ell \in [d_r]} \quad i, j \in [n] \\ & a_i^{(h)} \leftarrow \sum_j \alpha_{ij}^{(h)}\big(\boldsymbol{r}_{ij}W_r^h + s_jW_s^h\big), \quad i \in [n], h \in [n_h^{ra}] \\ & a_i \leftarrow \text{concat}\big(a_i^{(1)}, \ldots, a_i^{(n_h^{ra})}\big)W_o^{ra}, \quad i \in [n] \\ & \textbf{Output:}\big(\text{concat}(e_i, a_i)\big)_{i=1}^n \end{array}$

# 3. Integrating Attention over Sensory and Relational Information

# 3.1. Dual Attention

One of the keys to the success of the Transformer architecture is the use of so-called multi-head attention. This involves computing multiple attention operations in parallel at each layer and concatenating the output, enabling the model to learn multiple useful criteria for routing information between objects. However, in standard Transformers, these attention heads focus solely on routing sensory information, lacking explicit support for routing relational information between objects.

We posit that both sensory and relational information are crucial for robust and flexible learning over sequences or collections of objects. To this end, we propose an extension of multi-head attention comprising two distinct types of attention heads: sensory attention (i.e., standard self-attention), and relational attention. This yields a powerful mechanism for dynamically routing both sensory and relational information in parallel. Our hypothesis is that by having access to both computational mechanism, the model can learn to select between them based on the current task or context, as well as compose them to create highly-expressive and flexible computational circuits.

Algorithm 1 describes the proposed module, referred to as dual attention. The number of sensory attention heads $n_{h}^{sa}$

Algorithm 2: Dual Attention Encoder Block 

<table><tr><td> $\overline{\text{Input}}: \boldsymbol{x} \in \mathbb{R}^{n \times d}$ </td></tr><tr><td> $\boldsymbol{x} \leftarrow \text{Norm}(\boldsymbol{x} + \text{DualAttn}(\boldsymbol{x}))$ </td></tr><tr><td> $\boldsymbol{x} \leftarrow \text{Norm}(\boldsymbol{x} + \text{MLP}(\boldsymbol{x}))$ </td></tr></table>

# Output: x

and number of relational attention heads $n_{h}^{ra}$ are hyperparameters. The sensory attention heads attend to sensory information while the relational attention heads attend to relational information. The combined $n_{h} := n_{h}^{sa} + n_{h}^{ra}$ heads are then concatenated to produce the output. The result is a representation of contextual information with integrated sensory and relational components. Appendix B provides further discussion on the details of the architecture and its implementation.

Attention Masks & Causality. Any type of attention mask (e.g., causal mask for autoregressive language modeling) can be implemented in relational attention in the same way as for standard self-attention (i.e., mask is added to $\alpha_{ij}^{h}$ pre-softmax).

Positional Encoding. There exists different methods in the literature for encoding positional information in the Transformer architecture. For example, $[25]$ propose adding positional embeddings to the input, $[38]$ propose adding relative-positional embeddings at each attention operation, and $[39]$ propose rotary positional embeddings (RoPE) which apply a position-dependent map to the queries and keys pre-softmax. These methods are compatible with dual attention and are configurable options in our public implementation.

Computational complexity. The computational complexity of relational attention scales similarly to standard self-attention with a $O(n^{2})$ dependence on sequence length. Like standard attention, relational attention can be computed in parallel via efficient matrix multiplication operations.

Symmetric relations. A symmetry constraint can be injected into the relations $r_{ij}$ by imposing that $W_{q}^{rel} = W_{k}^{rel}$ , which is a useful inductive bias when the task-relevant relations are inherently symmetric.

# 3.2. The Dual Attention Transformer Architecture

The standard Transformer architecture is composed of repeated blocks of attention (information retrieval) followed by an MLP (local processing). Our proposed Dual Attention Transformer follows this same structure, but replaces multi-head self-attention with dual attention (Algorithm 1). At each layer, dual attention dynamically retrieves both sensory and relational information from the previous level of computation, which is then processed locally by an MLP. Al

Algorithm 3: Dual Attention Decoder Block 

<table><tr><td>Input: x, y ∈ Rn×d</td></tr><tr><td>x ← Norm(x + DualAttn(x))</td></tr><tr><td>x ← Norm(x + CrossAttn(x, y))</td></tr><tr><td>x ← Norm(x + MLP(x))</td></tr><tr><td>Output: x</td></tr></table>

gorithms 2 and 3 in Appendix B define encoder and decoder blocks with dual attention. Composing these blocks yields the Dual Attention Transformer architecture.

The Dual Attention Transformer framework supports all architectural variants of the standard Transformer, making it applicable to a wide range of task paradigms. An encoder-decoder architecture with causal dual-head attention in the decoder can be applied to sequence-to-sequence tasks, as in the original Transformer paper $[25]$ . An encoder-only architecture can be used for a BERT-style language embedding model $[40]$ or a ViT-style vision model $[41]$ . A decoder-only architecture with causal dual-head attention can be used for autoregressive language modeling.

# 4. Empirical evaluation

We empirically evaluate the Dual Attention Transformer (abbreviated, DAT) architecture on a range of tasks spanning different domains and modalities. Our goal is to assess the impact of integrating relational inductive biases into the Transformer architecture. We begin with a synthetic relational learning benchmark to evaluate DAT's relational computational mechanisms in a more controlled setting. We then proceed to evaluate the proposed architecture on more complex real-world tasks, including mathematical problem-solving, image recognition, and language modeling. These experiments cover multiple task paradigms and architectural variants, including: discriminative (encoder-only architecture), sequence-to-sequence (encoder-decoder), autoregressive language modeling (decoder-only), and vision (ViT-style architecture) tasks. For each experiment, we compare a DAT model that incorporates both sensory and relational heads against a standard Transformer where all heads are ordinary sensory attention heads. The difference in performance highlights the impact of integrating both types of attention heads, enabling a richer representation of sensory and relational information. We summarize the experimental results below and defer certain experimental details to Appendix C.

# 4.1. Sample-Efficient Relational Reasoning: Relational Games

We begin our empirical evaluation with the “Relational Games” benchmark for visual relational reasoning proposed

![](images/a93384c046ed96187557bfd6644ee01f66846e166f27c2d4e76aeb0fdd83c825.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Line 1) | Generalization Accuracy (Line 2) |
| ----------------- | -------------------------------- | -------------------------------- |
| 0                 | 0.6                              | 0.6                              |
| 500               | 0.8                              | 0.7                              |
| 1000              | 0.9                              | 0.85                             |
| 1500              | 0.92                             | 0.88                             |
| 2000              | 0.93                             | 0.9                              |
| 2500              | 0.94                             | 0.91                             |
</details>

![](images/e773f176c7621917418d75e5a0469900337e2825bf808f7d8019d1112f01c165.jpg)

<details>
<summary>line</summary>

| Training Set Size | Blue Line | Orange Line |
| ----------------- | --------- | ----------- |
| 500               | 0.6       | 0.55        |
| 1000              | 0.9       | 0.7         |
| 1500              | 0.9       | 0.8         |
| 2000              | 0.9       | 0.9         |
| 2500              | 0.9       | 0.9         |
</details>

![](images/ef9a572206f2a08281000a0476d7644f7187dbff0c2222ae8e121e5866e99a75.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 |
| ----------------- | ------ | ------ |
| 500               | 0.55   | 0.54   |
| 1000              | 0.70   | 0.65   |
| 1500              | 0.78   | 0.72   |
| 2000              | 0.85   | 0.78   |
| 2500              | 0.90   | 0.85   |
</details>

![](images/066bd462f3482ef9b8a5f7b06a0cb2c40921a3934535c2216753dd51c1c6c4e1.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy |
| ----------------- | ------------------------ |
| 0                 | 0.6                      |
| 500               | 0.7                      |
| 1000              | 0.8                      |
| 1500              | 0.85                     |
| 2000              | 0.9                      |
| 2500              | 0.92                     |
</details>

![](images/28b8e0554353a3982da5017597ba273e403fd87d8f070e4c6c64c28181f24264.jpg)

<details>
<summary>line</summary>

| Training Set Size | Blue Line | Purple Line | Orange Line |
| ----------------- | --------- | ----------- | ----------- |
| 5000              | 0.55      | 0.55        | 0.55        |
| 10000             | 0.75      | 0.65        | 0.58        |
| 15000             | 0.85      | 0.75        | 0.60        |
| 20000             | 0.90      | 0.85        | 0.65        |
| 25000             | 0.92      | 0.88        | 0.68        |
</details>

$$
\begin{array}{rl}
& \text{Model} \\
& \text{—} DAT(n_h^{sa}=0,n_h^{ra}=2)[421K] \\
& \text{—} DAT(n_h^{sa}=1,n_h^{ra}=1)[404K] \\
& \text{—} T(n_h^{sa}=8,n_h^{ra}=0)[481K] \\
& \text{—} T(n_h^{sa}=2,n_h^{ra}=0)[481K] \\
& \text{—} T(n_h^{sa}=8,n_h^{ra}=0)[386K] \\
& \text{—} T(n_h^{sa}=2,n_h^{ra}=0)[386K]
\end{array}
$$

Figure 2. Learning curves on the relational games benchmark. Each subplot corresponds to a different task. Numbers in square brackets in legend labels indicate parameter counts. Solid lines indicate the mean over 5 trials with different random seeds and the shaded regions indicate bootstrap 95% confidence intervals. DAT is more data-efficient at relational learning compared to a Transformer.

by Shanahan et al. [20]. The dataset consists of a family of binary classification tasks, each testing a model's ability to identify a particular visual relationship among a series of objects (see Figure 6 for examples). The input is an RGB image depicting a grid of objects, and the target is a binary classification indicating whether the particular relationship holds for this input. This forms a controlled synthetic setting for evaluating DAT's effectiveness in relational learning.

Our goal in this section is to explore how the relational computational mechanisms of DAT affect data-efficiency in relational learning—that is, how much data is necessary to learn a given task. We evaluate learning curves by varying the size of the training set, training each model until convergence, and evaluating on a hold-out validation set. We test two configurations of DAT: one with only relational attention heads, and one with a combination both sensory and relational heads. We include several Transformer baselines, varying the number of attention heads and the model dimension, controlling for parameter count. The results are depicted in Figure 2.

We find that DAT is significantly more sample-efficient, particularly at more difficult tasks. Both configurations of DAT are consistently more sample-efficient compared to the standard Transformer. The effect is particularly dramatic on the 'match pattern' task which is the most difficult and requires identifying a second-order relation (i.e., a relation between relations). We note that these tasks are purely relational in the sense that pairwise same/different relations between objects are a sufficient statistic for predicting the target. This suggests that relational attention is sufficient for solving the task. Indeed, the DAT variant with only relational heads performs slightly better than the variant with a combination of both sensory and relational heads. Notably, however, the difference is only marginal, suggesting that the model is able to learn to select the computational mechanisms that are most relevant to the given task. We provide further discussion in Appendix C.1, including results comparing against previously-proposed relational architectures with stricter inductive biases.

# 4.2. Relational Inductive Biases for Symbolic Reasoning in Sequence-to-Sequence tasks: Mathematical Problem Solving

Next, we evaluate $DAT$ on a set of mathematical problem-solving tasks based on the benchmark contributed by Saxton et al. [42]. Mathematical problem-solving is an interesting test for neural models because it requires more than statistical pattern recognition—it requires inferring laws, axioms, and symbol manipulation rules. The benchmark consists of a suite of mathematical problem-solving tasks, with each task's dataset consisting of a set of question-answer pairs. The tasks range across several topics including solving equations, adding polynomials, expanding polynomials, differentiating functions, predicting the next term in a sequence, etc. An example of a question in the “polynomials\_\_expand” task is “Expand $(5x - 3)(2x + 1)$ ” with the target answer “ $10x^{2} - x - 3$ ”. This is modeled as a sequence-to-sequence task with character-level encoding.

We compare DAT against Transformers using an encoder-decoder architecture. The encoder processes the question, and the decoder autoregressively generates the answer while cross-attending to the encoder. We explore how performance scales with model size by varying the number of layers. In the Transformer, all attention heads are standard self-attention with $n_{h}^{sa} = 8$ , while in DAT we have a combination of both types of attention heads with $n_{h}^{sa} = n_{h}^{ra} = 4$ .

![](images/362a9a7e05e0aadf024c55c57f869c80dc51f777c6849b4fad8cbb2a626ada1f.jpg)

<details>
<summary>line</summary>

| Model Size (Parameter Count) | Accuracy (Blue Line) | Accuracy (Red Line) |
| ----------------------------- | -------------------- | ------------------- |
| 750K                          | 0.68                 | 0.64                |
| 1M                            | 0.75                 | 0.63                |
| 1.25M                         | 0.76                 | 0.60                |
| 1.5M                          | 0.75                 | 0.57                |
| 1.75M                         | 0.74                 | 0.54                |
</details>

![](images/8946ebc3d58ccdb8bb4aec2e1e709891ea3bea6ac50c42d6872a01e7203641da.jpg)

<details>
<summary>line</summary>

| Model Size (Parameter Count) | Blue Line Value | Red Line Value |
| ----------------------------- | --------------- | -------------- |
| 750K                          | 0.92            | 0.91           |
| 1M                            | 0.96            | 0.93           |
| 1.25M                         | 0.97            | 0.96           |
| 1.5M                          | 0.98            | 0.95           |
| 1.75M                         | 0.97            | 0.94           |
</details>

![](images/1c6f070ff7f724517517a58855bc09c31b6486a6d92978172511f1abd6273c22.jpg)

<details>
<summary>line</summary>

| Model Size (Parameter Count) | Value  |
| ---------------------------- | ------ |
| 750K                         | 0.999  |
| 1M                           | 0.999  |
| 1.25M                        | 0.999  |
| 1.5M                         | 0.999  |
| 1.75M                        | 0.998  |
</details>

![](images/b38149a3c3ded8c1aefde7e7100c2fffe9d504c10c2366e99f5dd1069a15d7dd.jpg)

<details>
<summary>line</summary>

| Model Size (Parameter Count) | Accuracy (Blue Line) | Accuracy (Red Line) |
| ----------------------------- | -------------------- | ------------------- |
| 750K                          | 0.78                 | 0.76                |
| 1M                            | 0.85                 | 0.78                |
| 1.25M                         | 0.88                 | 0.82                |
| 1.5M                          | 0.90                 | 0.85                |
| 1.75M                         | 0.92                 | 0.90                |
</details>

![](images/dae30619f39bd4c56e89237bd057a0b04fba1858433a40ca568da2c7bc7ee605.jpg)

<details>
<summary>line</summary>

| Model Size (Parameter Count) | Blue Line Value | Red Line Value |
| ---------------------------- | --------------- | -------------- |
| 750K                         | 0.86            | 0.83           |
| 1M                           | 0.88            | 0.85           |
| 1.25M                        | 0.89            | 0.86           |
| 1.5M                         | 0.89            | 0.87           |
| 1.75M                        | 0.89            | 0.88           |
</details>

Model
Transformer
DAT

Figure 3. Average character-level accuracy on different mathematical problem-solving tasks measured at different model sizes. Error bars indicate bootstrap 95% confidence intervals over 5 trials. DAT outperforms a standard Transformer across model sizes, suggesting that relational computational mechanisms confer benefits on sequence-to-sequence tasks that involve symbolic computation.

Figure 3 depicts the character-level accuracy for DAT and Transformers across varying model sizes. We find that the DAT model outperforms the standard Transformer at all model scales and across all tested tasks. This suggests that the relational computational mechanisms of DAT are beneficial for the type of symbolic processing involved in solving mathematical problems.

# 4.3. Visual Processing with Relational Inductive Biases

As a general sequence model, the Transformer architecture can be applied to visual inputs by dividing an image into patches that are then flattened, linearly embedded into vectors, and passed in as a sequence. Through a series of attention and MLP operations, the visual input is processed for the downstream visual task. This architecture is referred to as a Vision Transformer (ViT) [41]. Although Transformers lack the explicit spatial inductive biases found in models like convolutional networks, recent work has demonstrated its effectiveness at scale [43], demonstrating the versatility of attention as a computational mechanism across several data modalities.

In this section, we explore how the relational computational mechanisms introduced in DAT—namely, relational attention—impact visual processing tasks. We hypothesize that visual processing benefits from attending to both sensory and relational information. That is, when processing a local region of a visual input (e.g., a patch, object, or object part), it is useful consider not only the sensory features of other regions but also the relationships between these regions. For example, this captures information about similar objects occurring in multiple places in the scene, or objects which are similar across some attributes (e.g., texture) but different across others (e.g., color). In particular, the relations in relational attention can be interpreted as taking the source patch as a filter and comparing it against each patch in the image under different transformations.

We evaluate a ViT-style DAT architecture (ViDAT), and compare it against a standard ViT on the CIFAR image recognition benchmarks $[44]$ . We train directly on CIFAR-10 and CIFAR-100, respectively, without pretraining on larger datasets. During training, we use random cropping, MixUp $[45]$ , and CutMix $[46]$ as data augmentation techniques. We evaluate 8-layer models with $d_{model} = d_{ff} = 384$ . The ViT model has $n_{h}^{sa} = 12$ standard self-attention heads, while the DAT model uses both sensory and relational heads, with an even split of $n_{h}^{sa} = n_{h}^{ra} = 6$ . We use symmetric relations $r_{ij}$ based on the intuition that visual processing involves symmetric attribute-similarity relations. In Appendix C.3, we present ablations, additional results, visualizations of learn relations, and further discussion.

Table 1 reports the classification accuracy of each model. ViDAT outperforms ViT on both datasets, suggesting that relational computational mechanisms enhance visual processing. These experiments show that relational inductive biases can be useful for image recognition. We hypothesize that relational processing is even more important in visual tasks requiring complex scene parsing, where reasoning about the relationships between constituent objects is essential. Recent work on scene understanding in large vision-language models supports this view $[47–49]$ . We leave the exploration of such tasks for future work.

# 4.4. Relational Inductive Biases in Language Modeling

Language understanding involves processing and organizing relational information, such as syntactic structures, semantic roles, and contextual dependencies, to extract meaning from words and their connections within sentences. Transformers have been remarkably effective at language modeling, with neural scaling laws demonstrating that increasing model size and dataset size result in predictable improvements in performance across a range of language tasks $[34, 35]$ . While the standard attention mechanism of Transformers is able to capture simple positional and syntactic relations in its attention scores, this is only used to control the flow of information between tokens rather than explicitly encoding relational information in the latent embeddings themselves. The relational attention mechanism of DAT enables explicitly learning relational contextual information that is directly encoded in each token's latent embedding.

In this section, we evaluate DAT on causal language modeling, exploring the impact of its relational computational mechanisms in the domain of language. We use a decoder-only architecture, where the model receives a sequence of tokens as input and is trained to causally predict the next token at each position. We train on 10 billion tokens of the FineWeb-Edu dataset $[50]$ , which is a curated dataset of high-quality educational text data from CommonCrawl. We train models at multiple parameter scales, up to 1.3 billion parameters, to study the scaling properties of DAT on language modeling with respect to both model size and data size. Details of training and architectural hyperparameters are given in Appendix C.4, together with further discussion of the results.

Figure 4 depicts the scaling properties of DAT's language modeling performance with respect to model size and data size, compared to a standard Transformer. We observe that DAT demonstrates greater data and parameter efficiency, achieving improved performance across model and data scales. This suggests that DAT's relational computational mechanisms confers benefits in language processing.

<table><tr><td>Dataset</td><td>Model</td><td>Params</td><td>Accuracy</td></tr><tr><td rowspan="2">CIFAR-10</td><td>ViT</td><td>7.1M</td><td>86.4 ± 0.1%</td></tr><tr><td>ViDAT</td><td>6.0M</td><td>89.7 ± 0.1%</td></tr><tr><td rowspan="2">CIFAR-100</td><td>ViT</td><td>7.2M</td><td>68.8 ± 0.2%</td></tr><tr><td>ViDAT</td><td>6.1M</td><td>70.5 ± 0.1%</td></tr></table>

Table 1. Classification accuracy on image recognition with the CIFAR-10 and CIFAR-100 datasets. Each training configuration is repeated 10 times with different random seeds; we report the mean accuracy ± the standard error of mean. DAT outperforms a standard Vision Transformer, suggesting that relational computational mechanisms are useful for visual processing tasks.

![](images/a3d9073cb17d5413fb6787f2b7222b5701c24237fc595dc16527a5cabd1e8ceb.jpg)

<details>
<summary>line</summary>

| Param Count | Perplexity 15.0 | Perplexity 18.0 | Perplexity 21.0 | Perplexity 24.0 | Perplexity 27.0 | Perplexity 30.0 | Model DAT | Model Transformer |
| ----------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------- | ----------------- |
| 350M        | ~6B             | ~5B             | ~4B             | ~3B             | ~2B             | ~1B             | ~7B       | ~6B               |
| 750M        | ~4B             | ~3B             | ~2B             | ~1.5B           | ~1B             | ~0.5B           | ~9B       | ~8B               |
| 1.3B        | ~3B             | ~2.5B           | ~1.5B           | ~1B             | ~0.5B           | ~0.25B          | ~7B       | ~6.5B             |
</details>

Figure 4. Performance scaling of DAT compared to a standard Transfomer on language modeling: DAT is more data-efficient and more parameter-efficient.

Beyond performance improvements, we also find evidence that relational attention encodes human-interpretable semantic relations. Figure 5 depicts a visualization of the relations $r_{ij}$ learned by a DAT language model. We observe that the relations learned by relational attention tend to encode semantic relations, rather than syntactic relations. That is, relational activations $r_{ij} \in R^{d_r}$ are large between tokens with related meanings. We believe that further exploration of this phenomenon from a mechanistic interpretability perspective could offer an exciting avenue for future research. Such an exploration would be complementary to interpretability efforts seeking to understand the attention scores of standard Transformers, which have been found to attend e.g., based on position, syntax, and punctuation [51–53].

# 5. Conclusion

Summary. The standard attention mechanism of Transformers provides a versatile mechanism for retrieval of sensory information from a given context, but does not explicitly support retrieval of relational information. In this work, we presented an extension of the Transformer architecture that disentangles and integrates sensory and relational information through a variant of multi-head attention with two distinct types of attention heads: standard self-attention for sensory information and a novel relational attention mechanism for relational information. We empirically evaluate this architecture and find that it yields performance improvements across a range of tasks and modalities.

Limitations & Future Work. The proposed architecture introduces several hyperparameters and possible configurations. Although we carried out ablations on the major configuration choices (e.g., composition of head types, symmetry, symbol assignment mechanisms), an expanded empirical investigation would help develop an improved understanding of the behavior of this architecture under different configurations. We also note that our implementation of the Dual-Attention Transformer currently lacks the hardware-aware optimizations available for standard Transformers (e.g., Flash-Attention [54]), which makes it slower to train

![](images/de3a6e28ac93f02df77f6f7a105ec5a52d7e537e51986bbfa474ccd727ddb98c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Finite"] --> B["machine"]
    B --> C["FS"]
    B --> D["M"]
    B --> E["or"]
    B --> F["finite"]
    F --> G["state"]
    G --> H["automaton"]
    H --> I["F"]
    H --> J["SA"]
    J --> K["plural"]
    K --> L["automata"]
    L --> M["."]
    L --> N["finite"]
    N --> O["automaton"]
    O --> P["'or simply a state machine'"]
    P --> Q["'is a mathematical model of computation'"]
    
    A --> R["Finite"]
    R --> S["machine"]
    S --> T["FS"]
    S --> U["M"]
    S --> V["or"]
    S --> W["finite"]
    W --> X["state"]
    X --> Y["automaton"]
    Y --> Z["F"]
    Y --> AA["SA"]
    AA --> AB["plural"]
    AB --> AC["automata"]
    AC --> AD["'or simply a state machine'"]
    AD --> AE["'is a mathematical model of computation'"]
    
    A --> AF["Finite"]
    AF --> AG["machine"]
    AG --> AH["FS"]
    AG --> AI["M"]
    AG --> AJ["or"]
    AG --> AK["finite"]
    AK --> AL["state"]
    AL --> AM["automaton"]
    AM --> AN["F"]
    AL --> AO["SA"]
    AO --> AP["'plural'"]
    AP --> AQ["automata"]
    AQ --> AR["'or simply a state machine'"]
    
    A --> AS["Finite"]
    AS --> AT["machine"]
    AT --> AU["FS"]
    AT --> AV["M"]
    AT --> AW["or"]
    AT --> AX["finite"]
    AX --> AY["-"]
    
    A --> AZ["Finite"]
    AZ --> BA["machine"]
    BA --> BB["FS"]
    BA --> BC["M"]
    BA --> BD["or"]
    BA --> BE["finite"]
    BE --> BF["-"]
    
    A --> BG["Finite"]
    BG --> BH["machine"]
    BH --> BI["FS"]
    BH --> BJ["M"]
    BH --> BK["or"]
    BH --> BL["finite"]
    BL --> BM["-"]
```
</details>

Figure 5. Relational attention in DAT language models encodes human-interpretable semantic relations. A visualization of the relations $r_{ij}$ learned by a 24-layer 343M-parameter DAT language model. Top. Visualization of one relation dimension in the first layer, focusing on the token 'model', which has high activation with the tokens 'state', 'machine', and 'mathematical'. Bottom. Visualization of one relation dimension in the twelfth layer, focusing on the token 'state', which has high activation with the tokens 'mathematical', 'model', and 'computation'.

overall (though we expect similar optimizations to be possible). An important direction for future work is the mechanistic interpretability $[53, 55, 56]$ of DAT models, focusing on identifying specific circuits that perform key computations, to better understand the performance improvements observed in complex domains like language modeling.

# Code and Reproducibility

Our implementation of the Dual Attention Transformer architecture is open-sourced at https://github.com/Awni00/dual-attention and published as a Python package. Pre-trained model weights, including the 1.3B-parameter DAT language model, are made publicly available and can be loaded directly using the package. Additionally, we provide code for running the experiments described in this paper, along with instructions for reproducing our results and access to the experimental logs.

# Acknowledgment

This research was supported by the funds provided by the National Science Foundation and by DoD OUSD (R&E) under Cooperative Agreement PHY-2229929 (The NSF AI Institute for Artificial and Natural Intelligence).

# Impact Statement

This work develops a variant of the Transformer architecture, called the Dual Attention Transformer, that integrates explicit relational computational mechanisms to enhance data-efficiency and generalization in learning relational processing. This advancement has the potential to improve performance in applications requiring structured reasoning. Like other developments in Transformer architectures, this research shares the potential societal consequences associated with the broader field.

# References

[1] Gary F Marcus. “The algebraic mind: Integrating connectionism and cognitive science”. MIT press, 2003 (cited on page 1).   
[2] David H Wolpert, William G Macready, et al. “No free lunch theorems for search”. Tech. rep. Citeseer, 1995 (cited on page 1).   
[3] Jonathan Baxter. “A model of inductive bias learning”. In: Journal of artificial intelligence research (2000) (cited on page 1).   
[4] Dedre Gentner. “Structure-mapping: A theoretical framework for analogy”. In: Cognitive science (1983) (cited on page 1).   
[5] Richard E Snow, Patrick C Kyllonen, Brachia Marshalek, et al. “The topography of ability and learning correlations”. In: Advances in the psychology of human intelligence (1984) (cited on page 1).   
[6] Charles Kemp and Joshua B Tenenbaum. “The discovery of structural form”. In: Proceedings of the National Academy of Sciences (2008) (cited on page 1).   
[7] Keith J Holyoak. “Analogy and relational reasoning”. In: The Oxford handbook of thinking and reasoning (2012) (cited on page 1).   
[8] Brenden M Lake et al. “Building machines that learn and think like people”. In: Behavioral and brain sciences (2017) (cited on page 1).   
[9] Brian Cantwell Smith. “The promise of artificial intelligence: reckoning and judgment”. Mit Press, 2019 (cited on page 1).   
[10] Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. "Investigating the Limitations of Transformers with Simple Arithmetic Tasks". 2021. arXiv: 2102. 13019 [cs.CL] (cited on page 1).   
[11] Anirudh Goyal and Yoshua Bengio. “Inductive biases for deep learning of higher-level cognition”. In: Proceedings of the Royal Society A (2022) (cited on page 1).   
[12] Allen Newell. “Physical symbol systems”. In: Cognitive science (1980) (cited on page 1).   
[13] Stevan Harnad. “The symbol grounding problem”. In: Physica D: Nonlinear Phenomena (1990) (cited on page 1).   
[14] Jerry A Fodor and Zenon W Pylyshyn. “Connectionism and cognitive architecture: A critical analysis”. In: Cognition (1988) (cited on page 1).   
[15] Brenden Lake and Marco Baroni. “Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks”. In: International conference on machine learning. PMLR. 2018 (cited on page 2).

[16] David Barrett et al. “Measuring abstract reasoning in neural networks”. In: International conference on machine learning. PMLR. 2018 (cited on page 2).   
[17] Felix Hill et al. “Learning to Make Analogies by Contrasting Abstract Relational Structure”. In: International Conference on Learning Representations. 2019 (cited on page 2).   
[18] Taylor W. Webb, Ishan Sinha, and Jonathan D. Cohen. “Emergent Symbols through Binding in External Memory”. Mar. 2021. arXiv: 2012.14601 [cs] (cited on page 2).   
[19] Adam Santoro et al. “A Simple Neural Network Module for Relational Reasoning”. June 2017. arXiv:1706.01427 [cs] (cited on page 2).   
[20] Murray Shanahan et al. “An Explicitly Relational Neural Network Architecture”. In: Proceedings of the 37th International Conference on Machine Learning. 2020 (cited on pages 2, 6, 19).   
[21] Giancarlo Kerg et al. “On Neural Architecture Inductive Biases for Relational Tasks”. June 2022. arXiv:2206.05056 [cs] (cited on pages 2, 3, 19).   
[22] Awni Altabaa et al. “Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers”. In: The Twelfth International Conference on Learning Representations. 2024 (cited on pages 2, 3, 19, 25–27).   
[23] Awni Altabaa and John Lafferty. “Learning Hierarchical Relational Representations through Relational Convolutions”. Feb. 2024. arXiv: 2310.03240 [cs] (cited on pages 2, 3).   
[24] Taylor W. Webb et al. “The Relational Bottleneck as an Inductive Bias for Efficient Abstraction”. In: Trends in Cognitive Sciences (May 2024) (cited on page 2).   
[25] Ashish Vaswani et al. “Attention Is All You Need”. In: Advances in neural information processing systems (2017) (cited on pages 2, 5, 17).   
[26] Jason Wei et al. “Chain-of-thought prompting elicits reasoning in large language models”. In: Advances in neural information processing systems (2022) (cited on page 2).   
[27] Takeshi Kojima et al. “Large language models are zero-shot reasoners”. In: Advances in neural information processing systems (2022) (cited on page 2).   
[28] Karthik Valmeekam et al. “Large Language Models Still Can’t Plan (A Benchmark for LLMs on Planning and Reasoning about Change)”. In: NeurIPS 2022 Foundation Models for Decision Making Workshop. 2022 (cited on page 2).

[29] Jie Huang and Kevin Chen-Chuan Chang. “Towards Reasoning in Large Language Models: A Survey”. 2023. arXiv: 2212.10403 [cs.CL] (cited on page 2).   
[30] Taylor Webb, Keith J Holyoak, and Hongjing Lu. "Emergent analogical reasoning in large language models". In: Nature Human Behaviour (2023) (cited on page 2).   
[31] Jie Huang et al. “Large Language Models Cannot Self-Correct Reasoning Yet”. 2024. arXiv: 2310. 01798 [cs.CL] (cited on page 2).   
[32] Iman Mirzadeh et al. “GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models”. 2024. arXiv: 2410.05229 [cs.LG] (cited on page 2).   
[33] Tom B. Brown et al. “Language Models are Few-Shot Learners”. 2020. arXiv: 2005.14165 [cs.CL] (cited on page 2).   
[34] Jared Kaplan et al. “Scaling Laws for Neural Language Models”. 2020. arXiv: 2001. 08361 [cs.LG] (cited on pages 2, 8, 23).   
[35] Jordan Hoffmann et al. “Training Compute-Optimal Large Language Models”. 2022. arXiv: 2203.15556 [cs.CL] (cited on pages 2, 8).   
[36] Aaron Grattafiori et al. “The Llama 3 Herd of Models”. 2024. arXiv: 2407.21783 [cs.AI] (cited on page 2).   
[37] Adam Santoro et al. “Relational Recurrent Neural Networks”. In: Advances in Neural Information Processing Systems. Curran Associates, Inc., 2018 (cited on page 2).   
[38] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. "Self-Attention with Relative Position Representations". In: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). Ed. by Marilyn Walker, Heng Ji, and Amanda Stent. New Orleans, Louisiana: Association for Computational Linguistics, June 2018 (cited on page 5).   
[39] Jianlin Su et al. “RoFormer: Enhanced Transformer with Rotary Position Embedding”. Nov. 2023. arXiv:2104.09864 [cs] (cited on pages 5, 17).   
[40] Jacob Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics. Association for Computational Linguistics, June 2019 (cited on page 5).   
[41] Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition

at Scale". In: International Conference on Learning Representations. 2021 (cited on pages 5, 7, 20).   
[42] David Saxton et al. “Analysing Mathematical Reasoning Abilities of Neural Models”. In: International Conference on Learning Representations. 2019 (cited on pages 6, 19).   
[43] Xiaohua Zhai et al. “Scaling vision transformers”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022 (cited on page 7).   
[44] Alex Krizhevsky. “Learning multiple layers of features from tiny images”. Tech. rep. 2009 (cited on pages 7, 20).   
[45] Hongyi Zhang et al. “mixup: Beyond Empirical Risk Minimization”. In: International Conference on Learning Representations. 2018 (cited on pages 7, 22).   
[46] Sangdoo Yun et al. “Cutmix: Regularization strategy to train strong classifiers with localizable features”. In: Proceedings of the IEEE/CVF international conference on computer vision. 2019 (cited on pages 7, 22).   
[47] Justin Johnson et al. “CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning”. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017 (cited on page 7).   
[48] Aimen Zerroug et al. “A benchmark for compositional visual reasoning”. In: Advances in neural information processing systems (2022) (cited on page 7).   
[49] Bingchen Zhao et al. “Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning”. In: arXiv preprint arXiv:2406.12742 (2024) (cited on page 7).   
[50] Anton Lozhkov et al. “FineWeb-Edu”. May 2024 (cited on pages 8, 23).   
[51] Kevin Clark et al. “What Does BERT Look At? An Analysis of BERT’s Attention”. 2019. arXiv: 1906. 04341 [cs.CL] (cited on page 8).   
[52] Phu Mon Htut et al. “Do Attention Heads in BERT Track Syntactic Dependencies?” 2019. arXiv: 1911. 12246 [cs.CL] (cited on page 8).   
[53] Nelson Elhage et al. “A Mathematical Framework for Transformer Circuits”. In: Transformer Circuits Thread (2021). https://transformercircuits.pub/2021/framework/index.html (cited on pages 8, 9).   
[54] Tri Dao et al. “Flashattention: Fast and memory-efficient exact attention with IO-awareness”. In: Advances in Neural Information Processing Systems (2022) (cited on page 8).

[55] Catherine Olsson et al. “In-context Learning and Induction Heads”. In: Transformer Circuits Thread (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (cited on page 9).   
[56] Kevin Ro Wang et al. “Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small”. In: The Eleventh International Conference on Learning Representations. 2023 (cited on page 9).   
[57] Gerard Debreu et al. “Representation of a preference ordering by a numerical function”. In: Decision processes (1954) (cited on page 13).   
[58] Awni Altabaa and John Lafferty. “Approximation of Relation Functions and Attention Mechanisms”. Feb. 2024. arXiv: 2402.08856 [cs, stat] (cited on page 14).   
[59] Noam Shazeer. “GLU Variants Improve Transformer”. Feb. 2020. arXiv: 2002. 05202 [cs, stat] (cited on page 17).   
[60] Yann N Dauphin et al. “Language modeling with gated convolutional networks”. In: International conference on machine learning. PMLR. 2017 (cited on page 17).   
[61] Dan Hendrycks and Kevin Gimpel. “Gaussian Error Linear Units (GELUs)”. 2016. arXiv: 1606.08415 [cs.LG] (cited on page 17).   
[62] Hugo Touvron et al. “Llama 2: Open Foundation and Fine-Tuned Chat Models”. July 2023. arXiv: 2307. 09288 [cs] (cited on page 17).   
[63] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. “Layer Normalization”. 2016. arXiv: 1607. 06450 [stat.ML] (cited on page 17).   
[64] Biao Zhang and Rico Sennrich. “Root mean square layer normalization”. In: Advances in Neural Information Processing Systems (2019) (cited on page 17).   
[65] Ruibin Xiong et al. “On layer normalization in the transformer architecture”. In: International Conference on Machine Learning. PMLR. 2020 (cited on page 17).   
[66] Francesco Locatello et al. “Object-Centric Learning with Slot Attention”. Oct. 2020. arXiv: 2006.15055 [cs, stat] (cited on page 18).   
[67] Joshua Ainslie et al. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023 (cited on page 22).   
[68] Ekin D. Cubuk et al. “AutoAugment: Learning Augmentation Policies from Data”. 2019. arXiv: 1805.09501 [cs.CV] (cited on page 22).

[69] Guilherme Penedo et al. “The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale”. 2024. arXiv: 2406.17557 [cs.CL] (cited on page 23).   
[70] Alec Radford et al. “Language models are unsupervised multitask learners”. In: OpenAI blog (2019) (cited on page 23).   
[71] Ronen Eldan and Yuanzhi Li. “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” May 2023. arXiv: 2305.07759 [cs] (cited on page 26).

# A. Function Class of Relational Attention: a universal approximation result

To gain a better understanding of the types of functions that can be computed by relational attention, we presented a simple approximation result (Theorem 1) in Section 2.4. Here, we will provide a formal statement of the result and prove it.

Recall that relational attention is a mapping on $R^{d} \times R^{n \times d} \to R^{d_{out}}$ , where d is the dimension of the input objects and $d_{out}$ is the output dimension. For convenience, we denote the “query space” by X and the “key space” by Y, though both are $R^{d}$ in this setting. Relational attention takes as input a query $x \in X$ and a collection of objects $\boldsymbol{y} = (y_{1}, \ldots, y_{n}) \in \mathcal{Y}^{n}$ and computes the following

$$
\mathrm{RA} (x, \boldsymbol {y}) = \sum_ {i = 1} ^ {n} \alpha_ {i} (x; \boldsymbol {y}) \big (r (x, y _ {i}) W _ {r} + s _ {i} W _ {s} \big), \tag {4}
$$

$$
\alpha (x; \boldsymbol {y}) = \operatorname{Softmax} \left(\left[ \left\langle \phi_ {q} ^ {\text { attn }} (x), \phi_ {k} ^ {\text { attn }} (y _ {i}) \right\rangle \right] _ {i = 1} ^ {n}\right) \in \Delta^ {n}, \tag {5}
$$

$$
r (x, y _ {i}) = \left(\left\langle \phi_ {q, \ell} ^ {\mathrm{rel}} (x), \phi_ {k, \ell} ^ {\mathrm{rel}} (y _ {i}) \right\rangle\right) _ {\ell \in [ d _ {r} ]} \in \mathbb {R} ^ {d _ {r}}, \tag {6}
$$

$$
\left(s _ {1}, \dots , s _ {n}\right) = \text { SymbolRetriever } \left(\boldsymbol {y}; S _ {\text { lib }}\right) \in \mathbb {R} ^ {n \times d _ {\text { out }}}, \tag {7}
$$

where $\phi_{q}^{attn},\phi_{k}^{attn},\phi_{q,\ell}^{rel},\phi_{k,\ell}^{rel}:R^{d}\to R^{d_{k}}$ are the feature maps defining the attention mechanism and the relation, respectively. For this section, these are multi-layer perceptrons. Note that in Algorithm 1 these are linear maps, but they are preceded by multi-layer perceptron in Algorithms 2 and 3, which makes the overall function class the same. Moreover, for this analysis we will take $W_{r}=I,d_{out}=d_{r}$ and $W_{s}=0$ . We will later discuss how the role of symbols fits within the message of the result.

The following result states that relational attention can approximate any function of the form: 1) select an object in $(y_{1},\ldots ,y_{n})$ by an arbitrary query-dependent selection criterion, and 2) compute an arbitrary relation $r:\mathcal{X}\times \mathcal{Y}\to \mathbb{R}^{d_r}$ with the selected object. This is formalized below.

To formalize (1), we adopt an abstract and very general formulation of a “selection criterion” in terms of a family of preference preorders, $\{\preccurlyeq_{x}\}_{x}$ : for each possible query x, the preorder $\preccurlyeq_{x}$ defines a preference over objects in Y to be selected. Intuitively, “ $y_{1} \preccurlyeq_{x} y_{2}$ ” means that $y_{2}$ is more relevant to the query x than $y_{1}$ .

More precisely, for each query $x \in X$ , $\preccurlyeq_{x}$ is a complete (for each $y_{1}, y_{2} \in Y$ , either $y_{1} \preccurlyeq y_{2}$ or $y_{2} \preccurlyeq_{x} y_{1}$ ), reflexive ( $y \preccurlyeq_{x} y$ for all $y \in Y$ ), and transitive ( $y_{1} \preccurlyeq_{x} y_{2}$ and $y_{2} \preccurlyeq_{x} y_{3}$ implies $y_{1} \preccurlyeq_{x} y_{3}$ ) relation. For each $x \in X$ , $\preccurlyeq_{x}$ induces a preordered space ( $Y, \preccurlyeq_{x}$ ). This implicitly defines two additional relations: $\prec_{x}$ and $\sim_{x}$ . We will write $y_{1} \prec_{x} y_{2}$ if “ $y_{1} \preccurlyeq_{x} y_{2}$ and not $y_{2} \preccurlyeq_{x} y_{1}$ ”, and $y_{1} \sim y_{2}$ if “ $y_{1} \preccurlyeq_{x} y_{2}$ and $y_{2} \preccurlyeq_{x} y_{1}$ ”.

For a collection of objects $\boldsymbol{y}=(y_{1},\ldots,y_{n})\in\mathcal{Y}^{n}$ and a query $x\in X$ , the preorder $\preccurlyeq_{x}$ defines a selection function

$$
\operatorname{Select} \left(x, \left(y _ {1}, \dots , y _ {n}\right)\right) := \max \left(\left(y _ {1}, \dots , y _ {n}\right), \text { key } = \preccurlyeq_ {x}\right). \tag {8}
$$

That is, $\operatorname{Select}(x, \mathbf{y})$ returns the most relevant element with respect to the query $x$ . In particular, it returns $y_{i}$ when $y_{i} \succ_{x} y_{j}$ , $\forall j \neq i$ (and may return an arbitrary element if no unique maximal element exists in $(y_{1}, \ldots, y_{n})$ ).

We will assume some regularity conditions on the family of preorders $\{\preccurlyeq_x\}_x$ which essentially stipulate that: 1) nearby elements in $\mathcal{V}$ have a similar preference with respect to each $x$ , and 2) nearby queries in $\mathcal{X}$ induce similar preference preorders.

Assumption 1 (Selection criterion is query-continuous and key-continuous). The family of preorder relations $\{\preccurlyeq_x\}_{x\in \mathcal{X}}$ satisfies the following:

1. Key-continuity. For each $x \in X$ , $\preccurlyeq_{x}$ is continuous. That is, for any sequence $(y_{i})_{i}$ such that $y_{i} \preccurlyeq_{x} z$ and $y_{i} \to y_{\infty}$ , we have $y_{\infty} \preccurlyeq_{x} z$ . Equivalently, for any $y \in Y$ , $\{z \in Y : z \preccurlyeq_{x} y\}$ and $\{z \in Y : y \preccurlyeq_{x} z\}$ are closed sets in Y.   
2. Query-continuity. Under key-continuity, Debreu et al. [57] shows that for each $x \in X$ , there exists a continuous in utility function $u_x : Y \to R$ for $\preccurlyeq_x$ such that $y_1 \preccurlyeq_x y_2 \iff u_x(y_1) \leq u_x(y_2)$ . For query-continuity, we make the further assumption that there exists a family of utility functions $\{u_x : Y \to R\}_{x \in X}$ such that $u(x, y) := u_x(y)$ is also continuous in its first argument.

For technical reasons, for Equation (8) to make sense, we must assume that there exists a unique element to be selected. We formulate this in terms of an assumption on the data distribution of the space $X \times Y^{n}$ . This is a technical assumption, and different forms of such an assumption would be possible (e.g., instead condition on this event).

Assumption 2 (Selection is unique almost always). Let $(x,y) \sim \mathbb{P}_{x,y}$ . For each $\varepsilon > 0$ , there exists $\eta_{\varepsilon} > 0$ such that $\min_{j \neq i} |u_x(y_i) - u_x(y_j)| > \eta_{\varepsilon}$ with probability at least $1 - \varepsilon$ .

Theorem (Function class of relational attention). Let $\mathcal{X},\mathcal{Y}$ be compact Euclidean spaces. Let $\{\preccurlyeq_x\}_{x\in \mathcal{X}}$ be an arbitrary family of relevance preorders on $\mathcal{Y}$ which are query-continuous and key-continuous (Assumption 1). Let $\operatorname {Select}(x,(y_1,\dots ,y_n)) = \max ((y_1,\dots ,y_n),\mathrm{key} = \preccurlyeq_x)$ be the selection function associated with $\{\preccurlyeq_x\}_x$ . Let $R:\mathcal{X}\times \mathcal{Y}\to$ $\mathbb{R}^{d_r}$ be an arbitrary continuous relation function. Suppose $x,\pmb {y}\sim \mathbb{P}_{x,\pmb {y}}$ and that Assumption 2 holds (i.e., the data distribution is such that there exists a unique most-relevant element w.h.p). For any $\varepsilon >0$ , there exists multi-layer perceptrons $\phi_q^{\mathrm{attn}},\phi_k^{\mathrm{attn}},\phi_q^{\mathrm{rel}},\phi_k^{\mathrm{rel}}$ and a choice of symbols such that,

$$
\left\| \mathrm{RA} (x, (y _ {1}, \dots , y _ {n})) - R (x, \operatorname{Select} (x, (y _ {1}, \dots , y _ {n}))) \right\| _ {\infty} <   \varepsilon
$$

Proof. Condition on the event $\mathcal{E} := \{(x, \boldsymbol{y}) \in \mathcal{X} \times \mathcal{Y}^n : \min_{j \neq i} |u_x(y_i) - u_x(y_j)| > \eta_\varepsilon\}$ . Let $i^* = \arg \max((y_1, \ldots, y_n), \text{key} = \preccurlyeq_x) = \arg \max(u_x(y_1), \ldots, u_x(y_n))$ . By [58, Theorem 5.1], for any $\varepsilon_1 > 0$ , there exists MLPs $\phi_q^{\text{attn}}, \phi_k^{\text{attn}}$ such that $\alpha_{i^*}(x, \boldsymbol{y}) > 1 - \varepsilon_1$ for any $(x, \boldsymbol{y}) \in \mathcal{E}$ . That is, the attention score is nearly 1 for the $\preccurlyeq_x$ -selected element uniformly over inputs in $\mathcal{E}$ .

Similarly, by [58, Theorem 3.1], for any $\varepsilon_{2} > 0$ , there exists MLPs $(\phi_{q,\ell}^{\mathrm{rel}},\phi_{k,\ell}^{\mathrm{rel}})_{\ell \in [d_r ]}$ such that $r(x,y) := (\langle \phi_{q,\ell}^{\mathrm{rel}}(x),\phi_{k,\ell}^{\mathrm{rel}}(y)\rangle)_{\ell \in [d_r ]}$ approximates the target relation $R$ uniformly within an error of $\varepsilon_{2}$ ,

$$
\left\| R (x, y) - r (x, y) \right\| _ {\infty} <   \varepsilon_ {2}, \quad \text { Lebesgue   almost   every } (x, y) \in \mathcal {X} \times \mathcal {Y}.
$$

Thus, we have

$$
\begin{array}{l} \left\| \mathrm{RA} (x, (y _ {1}, \dots , y _ {n})) - R (x, \operatorname{Select} (x, (y _ {1}, \dots , y _ {n}))) \right\| _ {\infty} \\ = \left\| \sum_ {i = 1} ^ {n} \alpha_ {i} (x; \boldsymbol {y}) r (x, y _ {i}) - R (x, y _ {i ^ {*}}) \right\| _ {\infty} \\ \leq \sum_ {i = 1} ^ {n} \| \alpha_ {i} (x; \boldsymbol {y}) r (x, y _ {i}) - R (x, y _ {i ^ {*}}) \| _ {\infty} \\ \leq \alpha_ {i ^ {*}} (x, \boldsymbol {y}) \| r (x, y _ {i ^ {*}}) - R (x, y _ {i ^ {*}}) \| _ {\infty} + \sum_ {j \neq i ^ {*}} \alpha_ {i} (x; \boldsymbol {y}) \| r (x, y _ {i}) - R (x, y _ {i ^ {*}}) \| _ {\infty} \\ \leq (1 - \varepsilon_ {1}) \varepsilon_ {2} + \varepsilon_ {1} \max _ {x, y, y ^ {*}} \| r (x, y) - R (x, y ^ {*}) \| _ {\infty}. \\ \end{array}
$$

Note that $\max_{x,y,y^{*}}\|r(x,y)-R(x,y^{*})\|_{\infty}$ is finite since X, Y are compact and r, R are continuous. Letting $\varepsilon_{1},\varepsilon_{2}$ be small enough completes the proof.

To summarize the analysis in this section, we showed that relational attention can approximate any computation composed of first selecting an object from a collection then computing a relation with that object. We can approximate any well-behaved selection criterion by formulating it in terms of an abstract preference preorder, and approximating the corresponding utility function (given by a Debreu representation theorem) by inner products of query and key feature maps. We can then approximate the target relation function similarly by inner products of a different set of query and key feature maps.

In the analysis above, we set aside the role of the symbols. Note that the function class this approximation result proves involves retrieving a relation from a selected object, but does not explicitly encode the identity of the selected object. Informally, the receiver knows that it has a particular relation with one of the objects in its context, and knows that this relation is with an object that was selected according to a particular selection criterion, but does not know the identity of the object beyond that. This is the purpose of adding symbols to relational attention—the retrieved relation is tagged with a symbol identifying the source.

# B. Architecture & Implementation Details

In this section, we briefly discuss some details of implementation that may be of interest to some readers. Our code is publicly available through the project git repository and includes detailed instructions for reproducing our experimental results. We also provide links to experimental logs. Our code uses the PyTorch framework.

# B.1. Relational Attention and Dual-Head Attention

The relational attention operation is defined as part of dual-head attention in Algorithm 1. We briefly mention some details of the implementation.

Learnable parameters. Let $n_{h} := n_{h}^{sa} + n_{h}^{ra}$ be the total number of sensory and relational heads. The learnable parameters are

\- Sensory attention heads. For each head $h \in [n_h^{sa}]$ :

○ Attention query/key projections: $W_{q,h}^{attn}$ , $W_{k,h}^{attn} \in R^{d_{model} \times d_{key}}$ ,   
○ Value projections: $W_{v}^{h} \in R^{d_{model} \times d_{h}}$ ,   
○ Output projection: $W_{o}^{sa} \in R^{d_{model} \times d_{model}}$ .

\- Relational attention heads. For each head $h \in [n_h^{ra}]$ and each relation $\ell \in [d_r]$ :

○ Attention query/key projections: $W_{q,h}^{attn}$ , $W_{k,h}^{attn} \in R^{d_{model} \times d_{key}}$ ,   
○ Relation query/key projections: $W_{q,\ell}^{rel}, W_{k,\ell}^{rel} \in R^{d_{model} \times d_{proj}}$ ,   
○ Symbol projection: $W_{s}^{h} \in R^{d_{model} \times d_{h}}$ ,   
○ Relation projection: $W_{r}^{h} \in R^{d_{r} \times d_{h}}$ ,   
○ Output projection: $W_{o}^{ra} \in R^{d_{model} \times d_{model}}$ .

We let $d_{key}, d_{h} = d_{model}/n_{h}$ to maintain the same dimension for the input and output objects. Similarly, we let $d_{proj} = d_{h} \cdot n_{h}^{ra}/d_{r}$ so that the number of parameters is fixed as $d_{r}$ varies. That is, we scale $d_{proj}$ down as $d_{r}$ increases; $d_{proj}$ has the interpretation of being the dimensionality of the subspace on which we are computing comparisons. So, having a larger number of relations corresponds to a more fine-grained comparison between the two objects.

To model symmetric relations, we let $W_{q,\ell}^{rel} = W_{k,\ell}^{rel}$ . Recall that this has the interpretation of computing a comparison between the same attributes in the pair of objects.

Note that the same $d_{r}$ -dimensional relation is used for all $n_{h}^{ra}$ attention heads, with a different learned linear map $W_{r}^{h}$ for each head extracting the relevant aspects of the relation for that attention head and controlling the placement in the residual stream. This allows for useful computations to be shared across all heads. Note also that the head dimension $d_{h} = d_{model}/n_{h}$ is defined in terms of the total number of attention heads and is the same for both sensory attention and relational attention. The output of each head is a $d_{h}$ -dimensional vector. This means that after concatenating all the heads, the proportion in the final $d_{model}$ -dimensional output that corresponds to each attention head type is proportional to the number of heads of that type. For example, if $n_{h}^{sa} = 6$ , $n_{h}^{ra} = 2$ , then 75% of the $d_{model}$ -dimensional output is composed of the output of sensory attention heads and 25% is composed of the output of relational attention heads. This enables tuning the relative importance of each head type for the task.

Code. We briefly discuss the code implementing relational attention. We use einsum operations heavily in our implementation due to the flexibility they offer for implementing general tensor contractions. From Algorithm 1, recall that relational attention takes the form:

$$
a _ {i} ^ {(h)} \leftarrow \sum_ {j} \alpha_ {i j} ^ {(h)} \big (\boldsymbol {r} _ {i j} W _ {r} ^ {h} + s _ {j} W _ {s} ^ {h} \big), \tag {9}
$$

where $\alpha_{ij}^{(h)}$ are the softmax attention scores for head $h\in [n_h^{ra}]$ , $\pmb{r}_{ij}\in \mathbb{R}^{d_r}$ are relation vectors, $s_j\in \mathbb{R}^{d_\mathrm{model}}$ is the symbol associated with the $j$ -th input, and $W_r^h,W_s^h$ map $\pmb{r}_{ij}$ and $s_j$ , respectively, to $d_h$ -dimensional vectors. We assume those are already computed and focus on a particular portion of the computation of relational attention. We break up the computation as follows:

$$
\sum_ {j} \alpha_ {i j} ^ {(h)} \left(\boldsymbol {r} _ {i j} W _ {r} ^ {h} + s _ {j} W _ {s} ^ {h}\right) = \sum_ {j} \left(\alpha_ {i j} ^ {(h)} s _ {j} W _ {s} ^ {h}\right) + \left(\sum_ {j} \alpha_ {i j} ^ {(h)} \boldsymbol {r} _ {i j}\right) W _ {r} ^ {h}. \tag {10}
$$

Note that we factor out the $W_r^h$ linear map and apply it after computing $\sum_j \alpha_{ij}^{(h)} r_{ij}$ . This is intentional, as will be explained below.

This can be computed in PyTorch via einsum operations as follows.

```python
# sv: (b, n, n_h, d_h)
# attn_scores: (b, n_h, n, n)
# relations: (b, n, n, d_r)
# self.wr: (n_h, d_h, d_r)

attended_symbols = torch.einsum('bhij,bjhd->bihd', attn_scores, sv)
# shape: (b, n, n_h, d_h)

attended_relations = torch.einsum('bhij,bijr->bihr', attn_scores, relations)
# shape: (b, n, n_h, d_r)

attended_relations = torch.einsum('bihr,hdr->bihd', attended_relations, self.wr)
# shape: (b, n, n_h, d_h)

output = attended_symbols + attended_relations
# shape: (b, n, n_h, d_h) 
```

Here, we assume sv, attn\_scores, and relations are already computed, and focus on a particular part of the computation. sv[:, :, h, :] = s W $_{s}^{h}$ , corresponds to the symbols of each object in the context, attn\_scores[:, h, :, :] = $\alpha^{h}$ are the softmax attention scores, and relations[:, i, j, :] = $r_{ij}$ are the relations, which can all be computed with simple matrix multiplication operations, very similar to the standard implementations of multi-head attention.

The first line corresponds to computing $\sum_{j}\alpha_{ij}^{h}s_{j}W_{s}^{h}$ . The second line corresponds to computing $\sum_{j}\alpha_{ij}^{h}r_{ij}$ . The third line corresponds to applying the linear map $W_{r}^{h}$ to the retrieved relations at each head. The reason we apply the map $W_{r}^{h}$ after attending to the relations is for memory efficiency reasons. If we were to apply $W_{r}^{h}$ first, we would need to manifest a tensor of dimension $b\times n\times n\times n_{h}^{ra}\times d_{h}$ , which is of order $\mathcal{O}(b\cdot n^{2}\cdot d_{\mathrm{model}})$ . Instead, by factoring out $W_{r}^{h}$ and applying it after computing attention, we only need to manifest a tensor of dimension $b\times n\times n\times d_{r}$ , which is much smaller since $d_{r}\ll d_{model}$ . This tensor is contracted to a dimension $b\times n\times d_{r}$ first, then mapped up to $b\times n\times n_{h}^{ra}\times d_{h}$ . This makes the memory footprint of relational attention of the same order as standard (sensory) attention when $d_{r}\asymp n_{h}$ .

When using position-relative symbols, the implementation is adjusted since we need to compute

$$
\sum_ {j} \alpha_ {i j} ^ {(h)} \left(\boldsymbol {r} _ {i j} W _ {r} ^ {h} + s _ {j - i} W _ {s} ^ {h}\right) \tag {11}
$$

instead, where the symbol $s_{j-i}$ sent now depends on both the source j and the target i. Thus, we now compute a symbols tensor which is indexed by both the source j and target i: $sv[i,j,h,:]=s_{j-i}W_{s}^{h}$ . Then, the implementation is adjusted by replacing the first line in the code above with

```python
attended_symbols = torch.einsum('bhij, ijhd->bihd', attn_scores, sv) 
```

The full implementation is made available through the project's github repository.

Composing relational attention to learn hierarchical relations. We remark that composing relational attention modules can be interpreted as representing hierarchical or higher-order relations. That is, relations between relations. An example of this is the relation tested in the match pattern task in the relational games benchmark. After one iteration of relational attention, an object's representation is updated with the relations it has with its context. A second iteration of relational attention now computes a representation of the relation between an object's relations and the relations of the objects in its context.

# B.2. Encoder and Decoder Blocks

We briefly mention a few configurations in our implementation that appear in our experiments. We aimed to make our implementation configurable to allow for various tweaks and optimizations that have been found in the literature for training Transformer models.

Symbol assignment. A shared symbol assignment module is used for all layers in the model. We explore three types of symbol assignment mechanisms: positional symbols, position-relative symbols, and symbolic attention. Different symbol assignment mechanisms are more well-suited to different tasks. We discuss ablation experiments we carried out on the effect of the symbol assignment mechanism in Appendix C.

MLP block. The MLP block uses a 2-layer feedforward network with a configurable activation function. The intermediate layer size is $d_{ff} = 4 \cdot d_{model}$ by default. We also use the SwiGLU “activation function” [59] in some of our experiments. SwiGLU is not merely an activation function, but is rather a neural network layer defined as the component-wise product of two linear transformations of the input. It is a type of gated linear unit [60] with the sigmoid activation replaced with a Swish activation [61], $\text{SwiGLU}(x) = \text{Swish}(xW + b) \otimes (xV + c)$ . This is used in the Llama series of models and was found to be a useful modification [62].

Normalization. Either LayerNorm $[63]$ or RMSNorm $[64]$ can be used. Normalization can be performed post-attention, like in the original Transformer paper $[25]$ , or pre-attention as in $[65]$ .

Positional encoding. Our experiments use either learned positional embeddings or RoPE [39].

![](images/514a7bca1ed062e16e77fedf63a866592a99ca2e9356bef7d80c5bb559095564.jpg)

<details>
<summary>natural_image</summary>

Two-panel image showing pixelated geometric shapes on black background, no text or symbols present
</details>

![](images/b62561ee7c93da9c272f51230af6313af008a6dcabbfd1abf4b53f3ba7d70caf.jpg)

<details>
<summary>text_image</summary>

occurs
</details>

![](images/d136eb5e893e490eda19fd8be029487f473a7bf29c3ae96f488b475c81f0ad4a.jpg)

<details>
<summary>text_image</summary>

xoccurs
</details>

![](images/55a996637dcb6cef20688637c22fa4c24c892a0083866a7ec8182b801b55beee.jpg)

<details>
<summary>text_image</summary>

between
</details>

![](images/94e061791bc6965b057304aa16ecf851a305ba867f1cd65618764d4f214ee22c.jpg)

<details>
<summary>text_image</summary>

match patt
</details>

Figure 6. Examples of different tasks in the Relational Games benchmark. Each column corresponds to a different task in the benchmark. The top row is an example of a positive instance and the bottom row is an example of a negative instance.

# C. Experimental Details & Further Discussion

# C.1. Relational Games (Section 4.1)

# EXPERIMENTAL DETAILS

Dataset details. The Relational Games benchmark datasets consists of $36 \times 36 \times 3$ RGB images depicting a $3 \times 3$ grid of objects which satisfy a particular visual relationship. The task is to identify whether a given relationship holds or not. The set of objects consists of simple geometric shapes. Examples of each task are presented in Figure 6. For example, in the occurs task, one object is present in the top row and three in the bottom row, and the task is to determine whether the object in the top row occurs (i.e., is among) the objects in the bottom row. The most difficult task in the benchmark is the match pattern task, where the grid contains a triplet of objects in the top row and another triplet of objects in the bottom row. Each triplet satisfies some relationship (e.g., ABC, ABA, ABB, or AAB), and the task is to determine whether the relation in the first triplet is the same as the relation in the second triplet. The difficulty in solving this task is that it requires parsing a second-order relation (a relation between relations). We remark that composing relational attention modules naturally captures this kind of hierarchical relations: the first relational attention operation produces objects representing relational information and the second would compute relations between those relations (i.e., second-order relations).

Model architectures. We use a Vision-Transformer-type architecture where the input image is split up into patches, flattened, and passed through the sequence model with added learned positional embeddings. We use average pooling at the end and pass through an MLP to produce the final prediction. We use a patch size of $12 \times 12$ which separates objects according to the grid structure. We note that in more general visual relational reasoning tasks where there isn't this type of grid structure, it would be appropriate to combine our approach with an object-discovery module such as Slot Attention [66].

We use 2-layer models. The DAT models use $d_{model} = 128$ , $d_{ff} = 256$ . One set of Transformer baselines uses the same, while another is larger with $d_{model} = 144$ , $d_{ff} = 288$ . All models use SwiGLU “activation”, dropout rate = 0.1, and pre-LayerNormalization. For the DAT models, we use positional symbols as the symbol assignment mechanism. The composition of sensory and relational attention heads are depicted in the figure. In Figure 2, we use symmetric relations (i.e., imposing that $W_{q}^{rel} = W_{k}^{rel}$ ). Below, we also explore the effect of this inductive bias, evaluating variants without the symmetry constraint.

Training details. For each task and model, we evaluated learning curves by varying the training set size and training the model until convergence, then evaluating on a hold-out test set. For four out of five of the tasks, we evaluate learning curves within the range of 250 to 2,500 samples, in increments of 250. For the more difficult match pattern, the range is from 5,000 to 25,000 in increments of 5,000. The ranges were chosen based on the difficulty of the different tasks in order to identify the right “resolution”. When evaluating learning curves, each training set is sampled randomly from the full dataset. For each task, model, and training set size, we repeat the experiment 5 times with different random seeds to compute approximate confidence intervals (accounting for randomness in sampling the dataset and random initialization). We use an Adam optimizer with a learning rate of 0.001, $\beta_{1}=0.9$ , $\beta_{2}=0.99$ , and a batch size of 512. We train for 50 epochs.

# FURTHER DISCUSSION, EXPLORATION, & ABLATIONS

Comparison to previous relational architectures. Previous research has explored relational learning in synthetic settings, proposing various architectures with relational inductive biases. Here, we compare DAT to three such architectures: PrediNet $[20]$ , CoRelNet $[21]$ , and Abstractor $[22]$ . Unlike DAT, these architectures use subtractive rather than additive relational inductive biases, imposing constraints on the types of learnable representations to improve relational learning efficiency. As a result, they are not general-purpose architectures and cannot be applied to broader domains such as language modeling. Nonetheless, it is useful to compare DAT against those architectures to explore the trade-offs of strong inductive biases and evaluate DAT in comparison to alternative approaches to relational learning. Figure 7 shows learning curves comparing DAT against those baselines. DAT performs competitively with previous relational architectures, generally outperforming PrediNet and Abstractor, while performing marginally worse than CoRelNet. It is relevant to note that CoRelNet incorporates strong task-specific inductive biases, and was partially designed with this benchmark in mind.

![](images/8f742c1f3a1ff3144d2fe36e2ebe2c7acffb95fdc24932eb31ed6205ade3c6dc.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Line 1) | Generalization Accuracy (Line 2) | Generalization Accuracy (Line 3) | Generalization Accuracy (Line 4) | Generalization Accuracy (Line 5) |
| ----------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| 0                 | 0.6                              | 0.7                              | 0.8                              | 0.9                              | 0.95                             |
| 500               | 0.7                              | 0.8                              | 0.9                              | 0.95                             | 0.98                             |
| 1000              | 0.8                              | 0.9                              | 0.95                             | 0.98                             | 0.99                             |
| 1500              | 0.85                             | 0.92                             | 0.96                             | 0.99                             | 0.995                            |
| 2000              | 0.88                             | 0.93                             | 0.97                             | 0.995                            | 0.998                            |
| 2500              | 0.9                              | 0.94                             | 0.98                             | 0.998                            | 0.999                            |
</details>

![](images/f7254f8d3cba78592080ad1fb0d56c52e287e9a12715fab9c645ae90b1a85ee3.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ----------------- | ------ | ------ | ------ | ------ | ------ |
| 500               | 0.7    | 0.6    | 0.55   | 0.5    | 0.45   |
| 1000              | 0.85   | 0.7    | 0.6    | 0.6    | 0.55   |
| 1500              | 0.9    | 0.8    | 0.7    | 0.7    | 0.65   |
| 2000              | 0.92   | 0.85   | 0.8    | 0.8    | 0.75   |
| 2500              | 0.95   | 0.9    | 0.9    | 0.9    | 0.85   |
</details>

![](images/85b6ef923407d35115d1c6b687a6ee03db8b5ca428f7e9aa2430b37c21a66161.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ----------------- | ------ | ------ | ------ | ------ | ------ |
| 500               | 0.6    | 0.55   | 0.55   | 0.55   | 0.55   |
| 1000              | 0.75   | 0.65   | 0.6    | 0.6    | 0.6    |
| 1500              | 0.8    | 0.75   | 0.65   | 0.65   | 0.65   |
| 2000              | 0.85   | 0.8    | 0.7    | 0.7    | 0.7    |
| 2500              | 0.9    | 0.85   | 0.75   | 0.75   | 0.75   |
</details>

![](images/70eedb7384346de2837574dae3b528a13acc6ed2ae180227b877a7374243b8c5.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Line 1) | Generalization Accuracy (Line 2) | Generalization Accuracy (Line 3) | Generalization Accuracy (Line 4) | Generalization Accuracy (Line 5) | Generalization Accuracy (Line 6) |
| ----------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| 0                 | 0.75                             | 0.60                             | 0.58                             | 0.57                             | 0.56                             | 0.55                             |
| 500               | 0.85                             | 0.75                             | 0.70                             | 0.68                             | 0.65                             | 0.62                             |
| 1000              | 0.90                             | 0.85                             | 0.80                             | 0.78                             | 0.75                             | 0.72                             |
| 1500              | 0.92                             | 0.90                             | 0.85                             | 0.83                             | 0.80                             | 0.78                             |
| 2000              | 0.93                             | 0.92                             | 0.88                             | 0.86                             | 0.84                             | 0.82                             |
| 2500              | 0.94                             | 0.93                             | 0.90                             | 0.88                             | 0.86                             | 0.84                             |
</details>

![](images/fd87be5c30d561e345aa7f2e6d6c2f27c4ba42c186d652dba127937826d96965.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ----------------- | ------ | ------ | ------ | ------ | ------ |
| 5000              | 0.55   | 0.55   | 0.55   | 0.55   | 0.55   |
| 10000             | 0.75   | 0.80   | 0.60   | 0.65   | 0.65   |
| 15000             | 0.90   | 0.90   | 0.65   | 0.65   | 0.65   |
| 20000             | 0.95   | 0.95   | 0.65   | 0.65   | 0.65   |
| 25000             | 0.95   | 0.95   | 0.65   | 0.65   | 0.65   |
</details>

Model
— DAT ( $n_{h}^{sa} = 0, n_{h}^{ra} = 2$ ) [421K]

$T\left(n_{h}^{sa}=8,n_{h}^{ra}=0\right)$ [481K]

$T\left(n_{h}^{sa}=2,n_{h}^{ra}=0\right)$ [481K]

$T\left(n_{h}^{sa}=2,n_{h}^{ra}=0\right)$ [386K]

PrediNet [376K]

CoRelNet [215K]

— Abstractor [469K]

Figure 7. Learning curves on the Relational Games benchmark, comparing DAT against previously-proposed relational architectures. DAT performs competitively with previous relational architectures.

Ablation over symmetry. We performed an ablation over the symmetry inductive bias in the relations computed in relational attention. Our implementation exposes an argument which controls whether the relation $r(x,y)=\left(\langle W_{q,\ell}^{\mathrm{rel}},W_{k,\ell}^{\mathrm{rel}}\rangle\right)_{\ell\in[d_{r}]\in\mathbb{R}^{d_{r}}}$ modeled in relational attention is constrained to be symmetric by setting $W_{q,\ell}^{rel}=W_{k,\ell}^{rel}$ . Indeed, we find symmetry to be a useful inductive bias in this task. Figure 8 depicts learning curves for the two configurations of DAT comparing symmetric RA against asymmetric RA. We find that symmetry results in faster learning curves for both configurations.

# C.2. Mathematical Problem-Solving (Section 4.2)

# EXPERIMENTAL DETAILS

Dataset details. Saxton et al. [42] propose a benchmark to assess neural models' ability to perform mathematical reasoning. The dataset consists of a suite of tasks in free-form textual input/output format. The tasks cover several topics in mathematics, including arithmetic, algebra, and calculus. For each task, the authors programmatically generate $2 \times 10^{6}$ training examples and $10^{4}$ validation examples. Questions have a maximum length of 160 characters and answers have a maximum length of 30 characters.

Model architectures. We use an encoder-decoder architecture for this experiment, treating it as a sequence-to-sequence task. We use character-level encoding with a common alphabet of size 85 containing small and upper case letters, digits 0-9, and symbols (e.g., \*, /, +, -). We vary the number of layers to explore how performance scales with model size

![](images/5da12ab1af1480b37405e8ce4d3976117f7dae2a1aef7bd70c3dff199253b247.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy |
| ----------------- | ------------------------ |
| 0                 | 0.63                     |
| 500               | 0.75                     |
| 1000              | 0.88                     |
| 1500              | 0.91                     |
| 2000              | 0.93                     |
| 2500              | 0.94                     |
</details>

![](images/e3c15f445b9bd1945e602e001eff3e2e0ba8dec354feba89cfe819491dad55a0.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 |
| ----------------- | ------ | ------ | ------ |
| 0                 | 0.6    | 0.6    | 0.6    |
| 500               | 0.8    | 0.7    | 0.65   |
| 1000              | 0.85   | 0.8    | 0.75   |
| 1500              | 0.88   | 0.85   | 0.8    |
| 2000              | 0.9    | 0.88   | 0.85   |
| 2500              | 0.92   | 0.9    | 0.88   |
</details>

![](images/3ac5e0ef2e789cb501b161f1d760bd63623ff64aec600813eb48232ada09bc5d.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 |
| ----------------- | ------ | ------ | ------ |
| 500               | 0.55   | 0.55   | 0.55   |
| 1000              | 0.65   | 0.60   | 0.58   |
| 1500              | 0.75   | 0.70   | 0.60   |
| 2000              | 0.85   | 0.80   | 0.75   |
| 2500              | 0.90   | 0.85   | 0.80   |
</details>

![](images/b684d1e6fae09cd722fda23edf19b1e6a2e0bb12b5dbac14e2d370fef683e9bc.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Solid Line) | Generalization Accuracy (Dashed Line) |
| ----------------- | ------------------------------------ | -------------------------------------- |
| 500               | 0.6                                  | 0.6                                    |
| 1000              | 0.8                                  | 0.75                                   |
| 1500              | 0.85                                 | 0.8                                    |
| 2000              | 0.9                                  | 0.85                                   |
| 2500              | 0.95                                 | 0.9                                    |
</details>

![](images/2758e27eac0896298e13aca6672fc41c921bd174fb0778b08e00560e4ad2cb7a.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 |
| ----------------- | ------ | ------ | ------ |
| 5000              | 0.5    | 0.5    | 0.5    |
| 10000             | 0.8    | 0.6    | 0.5    |
| 15000             | 0.9    | 0.7    | 0.6    |
| 20000             | 0.95   | 0.8    | 0.7    |
| 25000             | 0.98   | 0.9    | 0.8    |
</details>

![](images/ac3f557b90e1d25ea3e9e968fdad3aee83a90e58ab2149741ff41cd3384c559f.jpg)

<details>
<summary>text_image</summary>

Model
— Dual-Attn Transformer (n_h^sa = 1, n_h^ra = 1)
— Dual-Attn Transformer (n_h^sa = 0, n_h^ra = 2)
— Symmetric RA
— Asymmetric RA
</details>

Figure 8. An ablation of the effect of symmetry in relational attention in the relational games experiments.

in DAT compared to standard Transformers. Each encode/decoder block uses ReLU activation, dropout rate = 0.1, and post-normalization. We use $d_{model} = 128$ , $d_{ff} = 256$ for the DAT models and $d_{model} = 144$ , $d_{ff} = 288$ in the Transformer models to control for parameter count and give the Transformer an advantage in the evaluation. Sinusoidal positional embeddings are used as the positional encoding method. For all models, the total number of attention heads (across self-attention and relational attention) is 8. For the Transformer model, there are only self-attention heads: $n_{h}^{sa} = 8$ for both the encoder and decoder. For DAT, we evaluated two configurations for the composition of head types, one with $n_{h}^{sa} = n_{h}^{ra} = 4$ in the encoder and $n_{h}^{sa} = 8$ , $n_{h}^{ra} = 0$ in the decoder (i.e., standard Transformer Decoder), and one with $n_{h}^{sa} = 4 = n_{h}^{ra} = 4$ in the encoder and $n_{h}^{sa} = 4 = n_{h}^{ra} = 4$ in the decoder. The number of cross-attention heads in the decoder is 8 in all cases. No symmetry constraint is made on relational attention. Position-relative symbols are used as the symbol assignment mechanism, and the symbol library is shared across all layers in both the encoder and decoder.

Training Details. Each model is trained on each task for 50 epochs. We use the Adam optimizer with $\beta_{1}=0.9$ , $\beta_{2}=0.995$ , a learning rate of $6\times10^{-4}$ , and a batch size of 128. We evaluate and track the per-character accuracy over the course of training. We repeat this process 5 times for each combination of model and task with different random seeds to compute approximate confidence intervals.

# FURTHER DISCUSSION, EXPLORATION, & ABLATIONS

Table 2 reports the full set of results obtained for this experiment, including certain configurations omitted from the figure in the main text.

# C.3. Visual Processing (Section 4.3)

# EXPERIMENTAL DETAILS

Dataset details. In this set of experiments, we use the CIFAR-10 and CIFAR-100 datasets $[44]$ which are datasets of labeled small images. The CIFAR-10 dataset consists of 60,000 $32 \times 32$ RGB images, evenly split across 10 classes. The CIFAR-100 dataset consists of 60,000 RGB images of the same size, evenly split across 100 classes.

Model architectures. We use a ViT-style architecture $[41]$ . RGB images are divided into $4 \times 4$ patches, flattened, linearly embedded into a vector, and fed through an Encoder. We use average pooling followed by an MLP to produce the final prediction. We evaluate 8-layer models with $d_{model} = d_{ff} = 384$ , GeLU activation, Pre-LayerNormalization, and no

<table><tr><td>Task</td><td>Model</td><td>Parameter Count</td><td># Layers</td><td> $d_{model}$ </td><td>Encoder  $n_h^{sa}$ </td><td>Encoder  $n_h^{ra}$ </td><td>Decoder  $n_h^{sa}$ </td><td>Decoder  $n_h^{ra}$ </td><td>Accuracy</td></tr><tr><td rowspan="7">algebra_linear_1</td><td>Transformer</td><td>692K</td><td>2</td><td>128</td><td>8</td><td>0</td><td>8</td><td>0</td><td>62.5 ± 1.1%</td></tr><tr><td>DAT</td><td>783K</td><td>2</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>66.5 ± 1.0%</td></tr><tr><td>Transformer</td><td>871K</td><td>2</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>64.0 ± 1.5%</td></tr><tr><td>DAT</td><td>1.09M</td><td>3</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>68.1 ± 6.5%</td></tr><tr><td>Transformer</td><td>1.3M</td><td>3</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>57.0 ± 2.3%</td></tr><tr><td>DAT</td><td>1.43M</td><td>4</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>73.1 ± 1.1%</td></tr><tr><td>Transformer</td><td>1.7M</td><td>4</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>53.2 ± 1.1%</td></tr><tr><td rowspan="7">algebra_sequence_next_term</td><td>Transformer</td><td>692K</td><td>2</td><td>128</td><td>8</td><td>0</td><td>8</td><td>0</td><td>91.1 ± 0.2%</td></tr><tr><td>DAT</td><td>783K</td><td>2</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>91.6 ± 0.6%</td></tr><tr><td>Transformer</td><td>871K</td><td>2</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>91.4 ± 0.2%</td></tr><tr><td>DAT</td><td>1.09M</td><td>3</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>97.0 ± 0.5%</td></tr><tr><td>Transformer</td><td>1.3M</td><td>3</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>96.1 ± 0.5%</td></tr><tr><td>DAT</td><td>1.43M</td><td>4</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>-</td></tr><tr><td>Transformer</td><td>1.7M</td><td>4</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>93.4 ± 2.0%</td></tr><tr><td rowspan="7">calculus_differentiate</td><td>Transformer</td><td>692K</td><td>2</td><td>128</td><td>8</td><td>0</td><td>8</td><td>0</td><td>99.9 ± 0.0%</td></tr><tr><td>DAT</td><td>783K</td><td>2</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>100.0 ± 0.0%</td></tr><tr><td>Transformer</td><td>871K</td><td>2</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>99.9 ± 0.0%</td></tr><tr><td>DAT</td><td>1.09M</td><td>3</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>-</td></tr><tr><td>Transformer</td><td>1.3M</td><td>3</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>99.9 ± 0.0%</td></tr><tr><td>DAT</td><td>1.43M</td><td>4</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>100.0 ± 0.0%</td></tr><tr><td>Transformer</td><td>1.7M</td><td>4</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>99.9 ± 0.0%</td></tr><tr><td rowspan="7">polynomials_add</td><td>Transformer</td><td>692K</td><td>2</td><td>128</td><td>8</td><td>0</td><td>8</td><td>0</td><td>83.3 ± 0.1%</td></tr><tr><td>DAT</td><td>783K</td><td>2</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>85.6 ± 0.0%</td></tr><tr><td>Transformer</td><td>871K</td><td>2</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>84.5 ± 0.3%</td></tr><tr><td>DAT</td><td>1.09M</td><td>3</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>87.8 ± 0.1%</td></tr><tr><td>Transformer</td><td>1.3M</td><td>3</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>86.4 ± 0.3%</td></tr><tr><td>DAT</td><td>1.43M</td><td>4</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>88.7 ± 0.0%</td></tr><tr><td>Transformer</td><td>1.7M</td><td>4</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>87.6 ± 0.2%</td></tr><tr><td rowspan="7">polynomials_expand</td><td>Transformer</td><td>692K</td><td>2</td><td>128</td><td>8</td><td>0</td><td>8</td><td>0</td><td>74.0 ± 0.7%</td></tr><tr><td>DAT</td><td>783K</td><td>2</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>77.8 ± 0.1%</td></tr><tr><td>Transformer</td><td>871K</td><td>2</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>74.1 ± 0.6%</td></tr><tr><td>DAT</td><td>1.09M</td><td>3</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>-</td></tr><tr><td>Transformer</td><td>1.3M</td><td>3</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>81.0 ± 1.2%</td></tr><tr><td>DAT</td><td>1.43M</td><td>4</td><td>128</td><td>4</td><td>4</td><td>8</td><td>0</td><td>91.4 ± 0.9%</td></tr><tr><td>Transformer</td><td>1.7M</td><td>4</td><td>144</td><td>8</td><td>0</td><td>8</td><td>0</td><td>89.2 ± 0.5%</td></tr><tr><td>Dataset</td><td>Model</td><td>Parameter Count</td><td># Layers</td><td> $d_{model}$ </td><td> $n_h^{sa}$ </td><td> $n_h^{ra}$ </td><td>Symmetric  $r_{ij}$ </td><td>Accuracy</td></tr><tr><td rowspan="3">CIFAR-10</td><td>ViT</td><td>7.1M</td><td>8</td><td>384</td><td>12</td><td>0</td><td>NA</td><td>86.4 ± 0.1%</td></tr><tr><td rowspan="2">ViDAT</td><td>6.0M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>Yes</td><td colspan="2">89.7 ± 0.1%</td></tr><tr><td>6.6M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>No</td><td colspan="2">89.5 ± 0.1%</td></tr><tr><td rowspan="3">CIFAR-100</td><td>ViT</td><td>7.2M</td><td>8</td><td>384</td><td>12</td><td>0</td><td>NA</td><td>68.8 ± 0.2%</td></tr><tr><td rowspan="2">ViDAT</td><td>6.1M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>Yes</td><td colspan="2">70.5 ± 0.1%</td></tr><tr><td>6.7M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>No</td><td colspan="2">70.5 ± 0.1%</td></tr></table>

Table 2. Full results of mathematical problem-solving experiments. For each task, this table shows the mean test character-level accuracy $\pm$ the standard error of mean for each model configuration.

Table 3. Ablation over symmetry of $r_{ij}$ in relational attention for image recognition experiments.

<table><tr><td>Dataset</td><td>Model</td><td>Parameter Count</td><td># Layers</td><td> $d_{model}$ </td><td> $n_h^{sa}$ </td><td> $n_h^{ra}$ </td><td>Accuracy</td></tr><tr><td rowspan="2">CIFAR-10</td><td>ViT</td><td>7.1M</td><td>8</td><td>384</td><td>12</td><td>0</td><td>89.5 ± 0.1%</td></tr><tr><td>ViDAT</td><td>6.0M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>91.7 ± 0.1%</td></tr><tr><td rowspan="2">CIFAR-100</td><td>ViT</td><td>7.2M</td><td>8</td><td>384</td><td>12</td><td>0</td><td>68.2 ± 0.1%</td></tr><tr><td>ViDAT</td><td>6.1M</td><td>8</td><td>384</td><td>6</td><td>6</td><td>70.9 ± 0.1%</td></tr></table>

Table 4. Classification accuracy on CIFAR-10 and CIFAR-100 with AutoAugment data augmentation during training. Each training configuration is repeated 10 times with different random seeds; we report the mean accuracy $\pm$ the standard error of mean. DAT continues to outperform the standard Vision Transformer.

dropout. The ViT model has $n_{h}^{sa} = 12$ standard self-attention heads, while the DAT model uses both sensory and relational heads, with an even split $n_{h}^{sa} = n_{h}^{ra} = 6$ . In the main text, we use symmetric relations $r_{ij}$ with the intuition that visual processing involves symmetric attribute-similarity relations. We also carried out experiments with asymmetric relations and discuss the results below. In DAT, we use position-relative symbols as the symbol assignment mechanism. Further, we use Grouped Query Attention [67] in DAT to reduce the parameter count to account for the added parameters in relational attention.

Training Details. We train for 100 epochs. We use the Adam optimizer with a learning rate schedule consisting of a gradual warmup to $10^{-3}$ in the first 5 epochs, followed by a cosine rate decay down to $10^{-5}$ . We use the hyperparameters $\beta_{1} = 0.9$ , $\beta_{2} = 0.999$ , and weight decay of $5 \cdot 10^{-5}$ . We normalize the images channel-wise such that pixels have mean zero and unit standard deviation. In the results reported in Table 1 in the main text, we use random cropping, MixUp [45], and CutMix [46] as data augmentation techniques during training. We also report results using AutoAugment [68] below.

# FURTHER DISCUSSION, EXPLORATION, & ABLATIONS

Effect of symmetry in $r_{ij}$ . In the main text, Table 1 reports DAT results with symmetric relations $r_{ij}$ by imposing $W_{q}^{rel} = W_{k}^{rel}$ . Here, we explore the effect of this choice. Table 3 compares DAT models with and without the symmetry constraint. We find no significant difference in performance. Though, we note the smaller parameter count in the symmetric variant.

Interpretability Visualization. Figure 9 depicts a visualization of the relations learned by a ViDAT model trained on the CIFAR dataset. The relations $r_{ij}$ in relational attention can be interpreted as applying the source patch i as a filter, comparing it against each patch j. The different components of $r_{ij}$ can be seen as analogous to the channels in a convolution operation. We see that some relations in the ViDAT model appear to capture a human-interpretable notion of visual similarity across patches.

Alternative data augmentation. In the main text, we use random cropping, MixUp, and CutMix data augmentation during training. Here, we report results on an alternative data augmentation technique: AutoAugment $[68]$ . AutoAugment is an optimized set of data augmentation policies, found through a data-dependent automatic search procedure. At each mini-batch, a random sub-policy is chosen which consists of image processing operations such as translation, rotation, or shearing. Table 4 reports results using this data augmentation procedure. We continue to find that ViDAT outperforms the standard ViT model.

![](images/f72b5ff35c3a42e7c94a2beab44acc1d729e6b3ec5fb59b96bca0c139d8440e9.jpg)

<details>
<summary>natural_image</summary>

Grid of grayscale pixelated images with a red square highlighting a small dark region (no text or symbols)
</details>

(a) Original Image

![](images/9fc48a8bf3783bb0c4ba4839c64e2463d7f5962479a1e4458e89bed27209d9c0.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 grayscale pixelated images with a red square highlighting a dark region in the center (no text or symbols)
</details>

(b) A Relation in the First Layer

![](images/c9120295fbd549f8f7e4251eb5061b9ee4a462ccfd77051014913dd9d7d0c077.jpg)

<details>
<summary>natural_image</summary>

Grid of pixelated grayscale images with a red square highlighting a dark region in the center (no text or symbols)
</details>

(c) A Relation in the Fifth Layer   
Figure 9. A visualization of the relations $r_{ij}[\ell]$ learned by an 8-layer 6M-parameter ViDAT model trained on CIFAR. The left panel depicts the original image input. The middle and right panels depict the first relation in the relation vector $r_{ij}$ between the source patch i (red outline) and every other patch j. The relation activation is normalized with the tanh function for visualization. The normalized value of the relation is represented by a tint in each patch j—a green hue for large positive activations and a red tint for negative activations.

# C.4. Language Modeling (Section 4.4)

# EXPERIMENTAL DETAILS

Dataset details. The FineWeb-Edu $[50]$ dataset is a curated dataset of text data. It is generated by filtering the large-scale FineWeb dataset for LLM pre-training $[69]$ using an educational quality classifier trained on annotations generated by Llama3-70B-instruct. FineWeb-Edu has been shown to outperform FineWeb on several benchmarks, demonstrating the importance of data quality. We train our language models on a random subset of 10 billion tokens of FineWeb-Edu.

Model Architectures. We use a Decoder-only architecture, with causal attention for autoregressive language modeling. We vary model size to explore the scaling properties of DAT with respect to both model size and data size, comparing to the scaling properties of standard Transformers. Our architectural hyperparameters follow common choices at different model scales, based on scaling analyses performed for Transformers $[69]$ . We explore 3 model scales: 350M ( $d_{model} = 1024$ , $n_{h} = 16$ , L = 24), 750M ( $d_{model} = 1536$ , $n_{h} = 24$ , L = 24), and 1.3B ( $d_{model} = 2048$ , $n_{h} = 32$ , L = 24) parameters. We use $d_{ff} = 4 \cdot d_{model}$ , GeLU activation, RoPE positional encoding, no bias, no dropout, and Pre-LayerNormalization. We use the GPT2 tokenizer $[70]$ . We use symbolic attention as the symbol assignment mechanism, with the number of symbols in the symbol library scaling with model size: 1024 symbols and 8 heads for the 350M and 750M scale models, and 2048 symbols with 16 heads for the 1.3B scale model. We also increase the relation dimension with model size. We don't impose a symmetry constraint, with the intuition that linguistic relations can be asymmetric. We use Grouped Query Attention in the DAT models to reduce parameter count to account for the added parameters in relational attention, making them smaller overall compared to the Transformer baselines at each parameter scale.

Training Details. We train for 10B Tokens, with each batch containing 524, 288 tokens, split into context windows of 1,024 tokens. We use gradient accumulation to fit micro-batches into memory. We use the AdamW optimizer with a maximum learning rate of $6 \times 10^{-4}$ and minimum learning rate of $6 \times 10^{-5}$ , first linearly warming up over the first 715 steps, then decaying back down with a cosine schedule. We use $\beta_{1} = 0.9$ , $\beta_{2} = 0.95$ and a weight decay of 0.1. We also use gradient clipping to unit norm.

# FURTHER DISCUSSION, EXPLORATION, & ABLATIONS

Figure 4 in the main text depicts the scaling properties of a DAT language model with respect to model size and data size compared to a standard Transformer. Here, we provide a few additional representations of the results. Table 5 reports the end-of-training validation perplexity of the different models.

Figure 10 depicts training curves for the different model scales. We observe a power law scaling of the validation loss with respect to number of training tokens. This matches the neural scaling laws [34], which suggest that validation loss ought to scale roughly as $d^{-\alpha}$ where $d$ is the amount of training data and the exponent $\alpha$ is a constant that depends on model

architecture, training details, etc.

Table 5. End-of-training validation perplexity in language modeling on FineWeb-Edu dataset. 

<table><tr><td>Model</td><td>Param count</td><td># Tokens</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td> $n_h^{sa}$ </td><td> $n_h^{ra}$ </td><td> $d_r$ </td><td> $n_{kv}^h$ </td><td>Perplexity ↓</td></tr><tr><td>Transformer</td><td>353M</td><td>10B</td><td>1024</td><td>24</td><td>16</td><td>-</td><td>-</td><td>-</td><td>16.94</td></tr><tr><td>DAT</td><td>343M</td><td>10B</td><td>1024</td><td>24</td><td>8</td><td>8</td><td>64</td><td>4</td><td>16.09</td></tr><tr><td>Transformer</td><td>757M</td><td>10B</td><td>1536</td><td>24</td><td>24</td><td>-</td><td>-</td><td>-</td><td>14.65</td></tr><tr><td>DAT</td><td>734M</td><td>10B</td><td>1536</td><td>24</td><td>12</td><td>12</td><td>64</td><td>6</td><td>14.31</td></tr><tr><td>Transformer</td><td>1.31B</td><td>10B</td><td>2048</td><td>24</td><td>32</td><td>-</td><td>-</td><td>-</td><td>13.63</td></tr><tr><td>DAT</td><td>1.27B</td><td>10B</td><td>2048</td><td>24</td><td>16</td><td>16</td><td>128</td><td>8</td><td>13.43</td></tr></table>

![](images/60fe52a36c36c8e05a0dc89cd66c4d07d2d730eee2d98f9b3e79c2d1a361ac42.jpg)

<details>
<summary>line</summary>

| Tokens | Transformer - 353M | DAT - 343M |
| ------ | ------------------ | ---------- |
| 1B     | 3.45               | 3.40       |
| 10B    | 2.85               | 2.78       |
</details>

![](images/2fe9b1c79064a28c7d33e036eb9a2d09c142f6632c1f4f6d01d56aac8c82d8d2.jpg)

<details>
<summary>line</summary>

| Tokens | Transformer - 757M | DAT - 734M |
| ------ | ------------------ | ---------- |
| 1B     | 3.3                | 3.3        |
| 10B    | 2.7                | 2.65       |
</details>

![](images/3b8093543390a3dca556288c6acb8bd374912906f55fdeec9d1cbbbced6b5427.jpg)

<details>
<summary>line</summary>

| Tokens | Transformer | DAT   |
| ------ | ----------- | ----- |
| 1B     | 3.31B       | 3.27B |
| 10B    | 2.60B       | 2.60B |
</details>

Figure 10. Validation loss on a logarithmic scale to examine data scaling laws. Dual Attention Transformer language models obey similar scaling laws as standard Transformers with respect to the amount of training data, while consistently achieving smaller loss at multiple model scales.

# D. Comparison to Altabaa et al. [22]: Abstractors and Relational Cross-Attention

A closely related work is Altabaa et al. $[22]$ , which proposes a Transformer-based module called the “Abstractor” with relational inductive biases. The core operation in the Abstractor is a variant of attention dubbed “relational cross-attention” (RCA). In this section, we will discuss the relation between the Dual Attention Transformer and the Abstractor.

# D.1. Comparison between RA (this work) and RCA [22]

Altabaa et al. [22] propose a variant of attention called relational cross-attention which shares some characteristics with our proposal of what we're calling “relational attention” in this work. In this discussion, we will use the acronyms RCA and RA, respectively to distinguish between the two.

RCA processes a sequence of objects $\boldsymbol{x} = (x_{1}, \ldots, x_{n})$ and produces a sequence of objects $\boldsymbol{x}' = (x_{1}', \ldots, x_{n}')$ via the following operation

$$
\boldsymbol {x} ^ {\prime} \leftarrow \sigma_ {\mathrm{rel}} \left(\phi_ {q} (\boldsymbol {x}) \phi_ {k} (\boldsymbol {x}) ^ {\intercal}\right) \boldsymbol {s},
$$

$$
\boldsymbol {s} = \operatorname{SymbolRetriever} (\boldsymbol {x})
$$

where $\phi_{q}, \phi_{k}$ are query and key transformations, and the symbols s take the same role as in this work. $\sigma_{rel}$ is referred to as a “relation activation”. It may be either softmax or an element-wise activation (e.g., tanh, sigmoid, or linear). For the purposes of this discussion, let us consider $\sigma_{rel} = Softmax$ , which was used in the majority of the experiments in [22].

To facilitate the discussion, let us write RA and RCA side-by-side using a common notation.

RA (this work)

$$
(x _ {1} ^ {\prime}, \dots , x _ {n} ^ {\prime}) \leftarrow \operatorname{RA} (\boldsymbol {x}; S _ {\mathrm{lib}}),
$$

$$
x _ {i} ^ {\prime} = \sum_ {j = 1} ^ {n} \alpha_ {i j} \left(r (x _ {i}, x _ {j}) W _ {r} + s _ {j} W _ {s}\right),
$$

$$
\boldsymbol {\alpha} = \operatorname{Softmax} \left(\phi_ {q} (\boldsymbol {x}) \phi_ {k} (\boldsymbol {x}) ^ {\intercal}\right),
$$

$$
r (x, y) = \big (\left<   \phi_ {q, \ell} ^ {\mathrm{rel}} (x), \phi_ {k, \ell} ^ {\mathrm{rel}} (y) \right> \big) _ {\ell \in [ d _ {r} ]},
$$

$$
(s _ {1}, \dots , s _ {n}) = \text { SymbolRetriever } (\boldsymbol {x}; S _ {\text { lib }})
$$

RCA [22]

$$
\left(x _ {1} ^ {\prime}, \dots , x _ {n} ^ {\prime}\right) \leftarrow \operatorname{RCA} (\boldsymbol {x}; S _ {\text { lib }})
$$

$$
x _ {i} ^ {\prime} = \sum_ {j = 1} ^ {n} \alpha_ {i j} s _ {j},
$$

$$
\boldsymbol {\alpha} = \operatorname{Softmax} \left(\phi_ {q} (\boldsymbol {x}) \phi_ {k} (\boldsymbol {x}) ^ {\intercal}\right),
$$

$$
(s _ {1}, \dots , s _ {n}) = \text { SymbolRetriever } (\boldsymbol {x}; S _ {\text { lib }})
$$

RCA can be understood as self-attention, but the values are replaced with symbols (i.e., Attention $(Q \leftarrow x, K \leftarrow x, V \leftarrow s)$ ). By viewing the attention scores $\alpha_{ij}$ as relations, this has the effect of producing a relation-centric representation. The rationale is that in standard self-attention, the attention scores form a type of relation, but these relations are only used as an intermediate processing step in an information-retrieval operation. The relations encoded in the attention scores are entangled with the object-level features, which have much greater variability. This thinking also motivates the design of RA in the present work.

RCA can be understood as computing a pairwise relation $\langle\phi_{q}^{\mathrm{attn}}(x_{i}),\phi_{k}^{\mathrm{attn}}(x_{j})\rangle$ between $x_{i}$ and each $x_{j}$ in the context, and retrieving the symbol $s_{j}$ associated with the object $x_{j}$ with which the relation is strongest. That is, RCA treats the relations and the attention scores as the same thing. By contrast, the attention operation and computation of relations are separate in RA. The attention component is modeled by one set of query/key maps $\phi_{q}^{attn},\phi_{k}^{attn}$ and the relation component is modeled by another set of query/key maps $(\phi_{q,\ell}^{\mathrm{rel}},\phi_{k,\ell}^{\mathrm{rel}})_{\ell\in[d_{r}]}.$

The intuitive reason for this choice is that, for many tasks, the optimal “selection criterion” will be different from the task-relevant relation. For example, in a language modeling task, you may want to attend to objects on the basis of proximity and/or syntax while being interested in a relation based on semantics. Similarly, in a vision task, you may want to attend to objects on the basis of proximity, while computing a relation across a certain visual attribute. Thus, the relational attention mechanism proposed in this work offers greater flexibility and expressivity compared to RCA.

In RA, the symbols maintain the role of identifying the source. But they are now explicitly attached to a separately parameterized relation vector.

# D.2. Comparison between DAT and the Abstractor

We now briefly discuss the differences in the corresponding model architectures. Altabaa et al. $[22]$ propose an encoder-like module called the Abstractor which consists of essentially replacing self-attention in an Encoder with relational cross-attention. That is, it consists of iteratively performing RCA followed by an MLP. The paper proposes several ways to incorporate this into the broader Transformer architecture. For example, some of the experiments use a Encoder $\rightarrow$ Abstractor $\rightarrow$ Decoder architecture to perform a sequence-to-sequence task. Here, the output of a standard Transformer Encoder is fed into an Abstractor, and the Decoder cross-attends to the output of the Abstractor. In another sequence-to-sequence experiment, Altabaa et al. $[22]$ use an architecture where the Decoder cross-attends to both the Encoder and the Abstractor, making use of both sensory and relational information. In particular, the standard encoder and decoder blocks are the same (focusing on sensory information), but an additional module is inserted in between with a relational inductive bias.

By contrast, our approach in this paper is to propose novel encoder and decoder architectures imbued with two distinct types of attention heads, one with an inductive bias for sensory information and the other with an inductive bias for relational information. This has several potential advantages. The first is versatility and generality. The Abstractor architectures that were explored in $[22]$ only explicitly support sequence-to-sequence or discriminative tasks. For example, they do not support autoregressive models like modern decoder-only language models (e.g., of the form we experiment with in Section 4.4). Moreover, even in sequence-to-sequence tasks, Abstractor architectures only support relational processing over the input sequence, but they do not support relational processing over the target sequence (since the decoder does not have RCA). Another potential advantage of DAT is simplicity. The Abstractor paper proposes several architectures and configurations for the Encoder/Abstractor/Decoder modules, introducing several hyperparameters that are not trivial to choose. Moreover, it is unclear how to interpret this kind of architecture as the number of layers increases, and the original paper does not experiment with scaling up the number of layers. The final potential advantage is increased expressivity. In DAT, the two types of attention heads exist side by side in each layer. This allows relational attention heads to attend to the output of the self-attention heads at the previous layer, and vice-versa. This yields broader representational capacity, and potentially more interesting behavior as we scale the number of layers.

# D.3. How would RCA perform in an DAT-style dual head-type architecture?

One question one might ask is: how would an DAT-style dual head-type architecture perform if we used Altabaa et al. [22]'s RCA instead of the RA head-type proposed in this work? We carried out a few ablation experiments to answer this question.

Figure 11 compares learning curves on the relational games benchmark between standard DAT (with RA-heads) and a version of DAT with Altabaa et al. [22]'s RCA heads. We find that the two models perform similarly, with most differences small enough to be within the margin of error. This figure depicts the configuration with asymmetric RA and positional symbols.

Figure 12 depicts the validation loss curves on a small-scale language modeling experiment based on the Tiny Stories dataset $[71]$ , comparing standard DAT against a version with RCA heads. Here, we find that our relational attention heads yield better-performing models, with the RCA-head variant of DAT performing no better than a standard Transformer with a matching total number of heads.

![](images/9e6daf3e628caa3f6eea1e13a0e60505517c3ddec5343c203ddf987846be12b7.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Line 1) | Generalization Accuracy (Line 2) |
| ----------------- | -------------------------------- | -------------------------------- |
| 0                 | 0.6                              | 0.6                              |
| 500               | 0.7                              | 0.65                             |
| 1000              | 0.8                              | 0.75                             |
| 1500              | 0.85                             | 0.8                              |
| 2000              | 0.9                              | 0.85                             |
| 2500              | 0.9                              | 0.9                              |
</details>

![](images/310d5550c966516cd474b7335d7e7f71b8088ad705cdee9267fa5e66b5bc3245.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 |
| ----------------- | ------ | ------ |
| 500               | 0.6    | 0.6    |
| 1000              | 0.7    | 0.75   |
| 1500              | 0.8    | 0.85   |
| 2000              | 0.85   | 0.9    |
| 2500              | 0.9    | 0.95   |
</details>

![](images/b2e87e2ec348aafe0afd6ef89a027d5a7fb3f50539d176d423dd708b7e8e5fba.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 |
| ----------------- | ------ | ------ | ------ |
| 500               | 0.55   | 0.55   | 0.55   |
| 1000              | 0.60   | 0.60   | 0.60   |
| 1500              | 0.70   | 0.65   | 0.65   |
| 2000              | 0.85   | 0.80   | 0.80   |
| 2500              | 0.90   | 0.85   | 0.85   |
</details>

![](images/0d5d420fe5c6bad6cacaa2e084e41c3be55a7d5112e6b452bcaa81d5e105c6a1.jpg)

<details>
<summary>line</summary>

| Training Set Size | Generalization Accuracy (Solid Line) | Generalization Accuracy (Dashed Line) |
| ----------------- | ------------------------------------ | -------------------------------------- |
| 500               | 0.6                                  | 0.6                                    |
| 1000              | 0.8                                  | 0.75                                   |
| 1500              | 0.9                                  | 0.85                                   |
| 2000              | 0.95                                 | 0.9                                    |
| 2500              | 1.0                                  | 0.95                                   |
</details>

![](images/70e243b77dfe1ee177d565c9b9c1a31d8122f7d4909645057160b7cf81b393d2.jpg)

<details>
<summary>line</summary>

| Training Set Size | Line 1 | Line 2 | Line 3 |
| ----------------- | ------ | ------ | ------ |
| 5000              | 0.5    | 0.5    | 0.5    |
| 10000             | 0.55   | 0.55   | 0.55   |
| 15000             | 0.6    | 0.6    | 0.6    |
| 20000             | 0.8    | 0.7    | 0.65   |
| 25000             | 1.0    | 0.9    | 0.8    |
</details>

![](images/f51b56a777eff9349a410564a50ae4facf0067a6d914501fee2120fd6213b59c.jpg)

<details>
<summary>text_image</summary>

Model
— Dual-Attn Transformer (n^s_a = 1, n^r_a = 1)
— Dual-Attn Transformer (n^s_a = 0, n^r_a = 2)
— Relational Attention
— — Abstractor's RCA
</details>

Figure 11. Learning curves for DAT with RA compared with DAT with RCA on the relational games benchmark. The performance is similar, with most differences within the margin of error.

![](images/a6d48e2dcb41051fafc0d5291d0063c549ed6a4fe92768844c9c77845ff43dae.jpg)

<details>
<summary>line</summary>

| Tokens (x10^10) | Dual-Attn Transformer (n_h^sa = 6, n_h^ra = 2) - Disentangled Relational Attention | Dual-Attn Transformer (n_h^sa = 6, n_h^ra = 2) - Abstractor's RCA | Dual-Attn Transformer (n_h^sa = 4, n_h^ra = 4) - Disentangled Relational Attention | Dual-Attn Transformer (n_h^sa = 4, n_h^ra = 4) - Abstractor's RCA |
| --------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 0.0             | 2.25                                                                               | 2.25                                                                 | 2.25                                                                               | 2.25                                                                 |
| 0.2             | 1.85                                                                               | 1.85                                                                 | 1.85                                                                               | 1.85                                                                 |
| 0.4             | 1.80                                                                               | 1.80                                                                 | 1.80                                                                               | 1.80                                                                 |
| 0.6             | 1.78                                                                               | 1.78                                                                 | 1.78                                                                               | 1.78                                                                 |
| 0.8             | 1.77                                                                               | 1.77                                                                 | 1.77                                                                               | 1.77                                                                 |
| 1.0             | 1.76                                                                               | 1.76                                                                 | 1.76                                                                               | 1.76                                                                 |
| 1.2             | 1.75                                                                               | 1.75                                                                 | 1.75                                                                               | 1.75                                                                 |
| 1.4             | 1.74                                                                               | 1.74                                                                 | 1.74                                                                               | 1.74                                                                 |
</details>

Figure 12. Ablation of relational attention type. The solid line depicts the form of relational attention proposed in this work. The dotted line depicts RCA as proposed by Altabaa et al. [22]. We find that our relational attention mechanism performs better, whereas RCA performs no better than a Transformer.