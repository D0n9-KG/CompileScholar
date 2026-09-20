# On the Similarities of Embeddings in Contrastive Learning

Chungpa Lee $^{1}$ Sehee Lim $^{1}$ Kibok Lee $^{1}$ Jy-yong Sohn $^{1}$

# Abstract

Contrastive learning operates on a simple yet effective principle: Embeddings of positive pairs are pulled together, while those of negative pairs are pushed apart. In this paper, we propose a unified framework for understanding contrastive learning through the lens of cosine similarity, and present two key theoretical insights derived from this framework. First, in full-batch settings, we show that perfect alignment of positive pairs is unattainable when negative-pair similarities fall below a threshold, and this misalignment can be mitigated by incorporating within-view negative pairs into the objective. Second, in mini-batch settings, smaller batch sizes induce stronger separation among negative pairs in the embedding space, i.e., higher variance in their similarities, which in turn degrades the quality of learned representations compared to full-batch settings. To address this, we propose an auxiliary loss that reduces the variance of negative-pair similarities in mini-batch settings. Empirical results show that incorporating the proposed loss improves performance in small-batch settings.

# 1. Introduction

Contrastive learning (CL) has emerged as a powerful approach to representation learning (Chen & He, 2021; Chen et al., 2020; He et al., 2020; Radford et al., 2021; Zhai et al., 2023; Khosla et al., 2020). In CL, an embedding model is trained to produce similar representations for two different views of the same data instance—referred to as a positive pair—and dissimilar representations for views from different data instances—referred to as negative pairs. Recent empirical studies have shown that representations learned by these objectives, i.e., aligning positive pairs while separat-

$^{1}$ Department of Statistics and Data Science, Yonsei University, Seoul, Korea. Correspondence to: Jy-yong Sohn <jysohn1108@yonsei.ac.kr>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

ing negative pairs, achieve remarkable performance across various downstream tasks (Wang & Isola, 2020).

In parallel, several theoretical studies have analyzed the embeddings obtained by CL in full-batch settings. Lu & Steinerberger (2022) showed that the widely used InfoNCE loss (Oord et al., 2018) achieves its minimum value when positive pairs are perfectly aligned, and negative pairs are uniformly separated with the cosine similarity of $-\frac{1}{n-1}$ , where n is the size of the training dataset. Lee et al. (2024) further extended this optimality analysis to other contrastive losses, including the SigLIP loss (Zhai et al., 2023).

Due to computational constraints, CL methods are typically implemented using mini-batches rather than full batches in practical scenarios. This has motivated recent studies on understanding whether the property of the optimal state observed in full-batch settings, i.e., perfect alignment of positive pairs and uniform separation of negative pairs with the similarity of $-\frac{1}{n-1}$ , also holds under mini-batch settings. Cho et al. (2024) partially addressed this by showing that, for the InfoNCE loss, the optimal embeddings of mini-batch settings are identical to those of full-batch settings, only when the sum of all possible mini-batch losses is minimized. Koromilas et al. (2024) explored embeddings learned through kernel-based contrastive losses (Li et al., 2021; Waida et al., 2023) in mini-batch settings. While these studies provide valuable insights into understanding CL, their analyses are limited to specific forms of contrastive losses.

To address this limitation, we propose a unified framework for analyzing the embeddings obtained by both full-batch and mini-batch settings. Our analysis centers on the cosine similarity of embeddings of both positive and negative pairs. By characterizing the statistical properties of similarities of these embeddings, we reveal how different contrastive losses influence the structure of the learned embedding space under various training conditions.

Our key contributions are summarized as follows:

\- In full-batch settings, we identify a fundamental trade-off between the alignment of positive pairs and the separation of negative pairs. Specifically, we show that the perfect alignment of positive pairs is not feasible when the average similarity of negative pairs falls below the certain threshold $-\frac{1}{n-1}$ . We demonstrate that such misalignment

arises in a class of existing contrastive losses, which can be mitigated by incorporating within-view negative pairs into the contrastive loss.

- In mini-batch settings, we demonstrate that negative pairs within the same batch exhibit stronger separation compared to those from different batches. As a result, we show that smaller batch sizes induce a higher variance in the similarities of negative pairs in the embedding space. We identify this increased variance as a distinctive feature of mini-batch settings that is absent in full-batch settings.   
- Motivated by prior studies that full-batch settings often outperform their mini-batch counterparts, we hypothesize that the increased variance may underlie this performance gap. To explore this, we propose an auxiliary loss that can be integrated into contrastive losses to reduce this variance. Empirical results show that incorporating the proposed term improves the performance of CL methods, especially in small-batch settings.

# 2. Related Work

Contrastive Loss. The InfoNCE loss (Gutmann & Hyvärinen, 2010; Oord et al., 2018) is a widely adopted contrastive loss that has been applied to various tasks (Wu et al., 2018; Hjelm et al., 2019; Bachman et al., 2019; Chi et al., 2021; Gao et al., 2021; Qian et al., 2021). SimCLR (Chen et al., 2020) modifies the InfoNCE loss to improve robustness to augmentations by treating different augmented views of the same instance as positives and all other augmented instances in the batch as negatives. However, SimCLR simultaneously optimizes both positive and negative pairs in the normalization, which can introduce conflicts during optimization. To address this issue, Decoupled Contrastive Loss (DCL) (Yeh et al., 2022) modifies the normalization so that the selection of negative pairs is restricted. Building on DCL, Decoupled Hyperspherical Energy Loss (DHEL) (Koromilas et al., 2024) further refines the selection by focusing on negative pairs between augmented views of the same instance, which are more challenging than those between different instances. Figure 5 visualizes which pairs are included as positives and negatives for each method.

Meanwhile, Zhai et al. (2023) raised concerns about the softmax function in the InfoNCE loss, noting that it causes all instances to be dependent on each other through normalization. To address this limitation, they propose replacing the softmax with a sigmoid function, which allows each instance to be processed independently in an additive manner.

Understanding CL Through Embedding Structures. Several studies have examined how optimal embeddings should be structured to minimize contrastive loss (Lee et al., 2025). Lu & Steinerberger (2022) showed that the optimal embeddings that minimize the InfoNCE loss form a simplex Equiangular Tight Frame (ETF) (Papyan et al., 2020; Sustik et al., 2007), where each positive pair is perfectly aligned and negative pairs are equally separated at the same angle, resulting in maximal separation among representations. Lee et al. (2024) further showed that the sigmoid-based contrastive loss, a variant of the softmax-based InfoNCE loss, also achieves the same simplex ETF optimum when the temperature parameter of the loss is sufficiently large. This optimal simplex ETF structure still holds even in mini-batch settings, provided that optimization is performed over all possible mini-batch combinations, rather than a single batch at a time (Cho et al., 2024). Building on these findings, we introduce a unified theoretical analysis of the similarities of embedding pairs.

Effect of Batch Size in CL. CL shows outstanding performance, particularly when trained with large batch sizes (Chen et al., 2020; Radford et al., 2021; Pham et al., 2023; Tian et al., 2020b; Jia et al., 2021). However, large batch sizes require substantial memory resources, which poses practical challenges and often necessitates the use of smaller batches. This compromise in batch size typically leads to performance degradation, motivating several theoretical studies to investigate its causes (Cho et al., 2024; Koromilas et al., 2024). For example, Yuan et al. (2022) demonstrate that the optimization error in SimCLR (Chen et al., 2020) is upper bounded by a function inversely proportional to the batch size, indicating that smaller batches yield larger optimization errors. Additionally, Chen et al. (2022) show that contrastive losses exhibit increasing discrepancies between the true gradients and those estimated during training as the batch size decreases. While previous studies have primarily focused on optimization error and gradient estimation, we prove that training with small batch sizes leads to increased variance in the similarities of negative pairs in learned embeddings.

# 3. Problem Setup

Let $(\boldsymbol{x}, \boldsymbol{y})$ denote a pair of data points used for model training, where x and y correspond to two distinct views of instances. This formulation provides a unified framework for CL in both unimodal (Chen et al., 2020) and multimodal (Radford et al., 2021) settings. In the unimodal case, x and y are two randomly augmented views. In the multimodal case, x and y are views from different modalities. For clarity, we present our analysis in the unimodal setting, but the findings also apply to the multimodal case.

In CL, an encoder $f(\cdot) \in \mathbb{R}^{d}$ is trained to map inputs into d-dimensional embedding vectors, thereby representing the data. The encoder is assumed to produce normalized embeddings such that $\|u\|_{2} = 1$ for all embeddings u. This

normalization is commonly adopted in related works for mathematical simplicity (Wang et al., 2017; Wu et al., 2018; Tian et al., 2020a; Wang & Isola, 2020; Zimmermann et al., 2021; Cho et al., 2024; Lee et al., 2024), and is widely used in practice, as experimental results consistently demonstrate its effectiveness in improving performance (Chen et al., 2020; Chen & He, 2021; Xue et al., 2024).

The encoder produces outputs $\boldsymbol{u} = f(\boldsymbol{x})$ and $\boldsymbol{v} = f(\boldsymbol{y})$ , which together form an embedding pair $(\boldsymbol{u}, \boldsymbol{v})$ . When the embedding pair is generated from augmented views of the same instance, it is called a positive embedding pair and is encouraged to be similar. In contrast, a negative embedding pair, where each embedding comes from different instances, is encouraged to be dissimilar. For simplicity, we refer to positive embedding pairs as positive pairs and similarly to negative pairs, when there is no risk of confusion.

Formally, $p_{\mathrm{pos}}(\boldsymbol{x}, \boldsymbol{y})$ denotes the distribution of positive pairs, and $p_{\mathrm{neg}}(\boldsymbol{x}, \boldsymbol{y})$ represents the distribution of negative pairs. As in prior work (Wang & Isola, 2020), the marginal distribution of augmented view x is denoted by $p_x$ , and similarly use $p_y$ to denote the marginal distribution for y. These marginals satisfy the following conditions: $p_x(\boldsymbol{x}) = \int p_{\mathrm{pos}}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y} = \int p_{\mathrm{neg}}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y}$ for all x, and $p_y(\boldsymbol{y}) = \int p_{\mathrm{pos}}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x} = \int p_{\mathrm{neg}}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x}$ for all y. Note that the randomness in these distributions arises from the data augmentation process used to generate different views.

Let n be the size of the training dataset. For any positive integers a and b with $a \leq b \leq n$ , define the index sets $[a : b] := \{a, a + 1, \cdots, b\}$ and $[a] := [1 : a]$ , where index $i \in [n]$ refers to the i-th instance in the dataset. For $i \in [n]$ , let $\hat{p}_{\mathrm{pos}}^{i}(\mathbf{x}, \mathbf{y})$ denote the empirical distribution of positive pairs derived from the i-th instance. We assume that the supports of $\hat{p}_{pos}^{i}$ and $\hat{p}_{pos}^{j}$ are disjoint for $i \neq j$ , as each instance is distinct. Moreover, we assume that each instance is used equally for training. Therefore, the empirical distribution of all positive pairs, denoted as $\hat{p}_{\mathrm{pos}}(\mathbf{x}, \mathbf{y})$ , can be written as $\hat{p}_{\mathrm{pos}}(\mathbf{x}, \mathbf{y}) = \frac{1}{n} \sum_{i \in [n]} \hat{p}_{\mathrm{pos}}^{i}(\mathbf{x}, \mathbf{y})$ . Similarly, the empirical distribution of all negative pairs is denoted by $\hat{p}_{\mathrm{neg}}(\mathbf{x}, \mathbf{y})$ .

Following the notations introduced in Koromilas et al. (2024), we denote the element-wise pushforward measures induced by the encoder $f$ as $f_{\sharp}\hat{p}_x$ , $f_{\sharp}\hat{p}_y$ , $f_{\sharp}\hat{p}_{\mathrm{pos}}$ , and $f_{\sharp}\hat{p}_{\mathrm{neg}}$ . For example, $f_{\sharp}\hat{p}_{\mathrm{neg}}(\mathbf{u},\mathbf{v})$ represents the empirical distribution of negative embedding pairs, i.e., the distribution of $(\mathbf{u},\mathbf{v}) = (f(\boldsymbol {x}),f(\boldsymbol {y}))$ where $(\boldsymbol {x},\boldsymbol {y})$ comes from $\hat{p}_{\mathrm{neg}}$ .

Contrastive Loss. For any $a \leq b$ with $a, b \in [n]$ , let $(\boldsymbol{U}_{[a:b]}, \boldsymbol{V}_{[a:b]}) := \{(\boldsymbol{u}_i, \boldsymbol{v}_i) : i \in [a : b]\}$ be the set of embedding pairs for instances whose indices range from a to b. The notation $(\boldsymbol{U}_{[a:b]}, \boldsymbol{V}_{[a:b]}) \sim f_{\sharp}\hat{p}_{\mathrm{pos}}^{[a:b]}$ indicates that each positive pair $(\boldsymbol{u}_i, \boldsymbol{v}_i)$ is sampled from $f_{\sharp}\hat{p}_{\mathrm{pos}}^i$ for all $i \in [a : b]$ . Note that $u_i$ and $v_i$ are random variables, as they are the encoder outputs of randomly augmented views. For simplicity, the subscript is omitted for the set of all n pairs, i.e., $(\boldsymbol{U}, \boldsymbol{V}) := (\boldsymbol{U}_{[n]}, \boldsymbol{V}_{[n]})$ .

Let $f^{\star}$ be the optimal encoder that minimizes the expected contrastive loss, given by

$$
f ^ {\star} := \underset {f} {\arg \min} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\text { pos }} ^ {[ n ]}} \left[ \mathcal {L} (\boldsymbol {U}, \boldsymbol {V}) \right], \tag {1}
$$

where $\mathcal{L}(U,V)$ denotes the contrastive loss for a given sample $(U,V)$ . In this work, we focus on two specific forms of contrastive losses used in practice:

Definition 3.1 (InfoNCE-Based Contrastive Loss). For a given index set $I \subseteq [n]$ , the contrastive loss $\mathcal{L}_{\text{info-sym}}(U_I, V_I)$ is defined in its symmetric form as

$$
\mathcal {L} _ {\text { info - sym }} (\boldsymbol {U} _ {\boldsymbol {I}}, \boldsymbol {V} _ {\boldsymbol {I}}) := \frac {1}{2} \mathcal {L} _ {\text { info }} (\boldsymbol {U} _ {\boldsymbol {I}}, \boldsymbol {V} _ {\boldsymbol {I}}) + \frac {1}{2} \mathcal {L} _ {\text { info }} (\boldsymbol {V} _ {\boldsymbol {I}}, \boldsymbol {U} _ {\boldsymbol {I}}), \tag {2}
$$

where the asymmetric component $\mathcal{L}_{\mathrm{info}}(U_I, V_I)$ is

$$
\begin{array}{l} \mathcal {L} _ {\text { info }} \left(\boldsymbol {U} _ {\boldsymbol {I}}, \boldsymbol {V} _ {\boldsymbol {I}}\right) := \frac {1}{| \boldsymbol {I} |} \sum_ {i \in \boldsymbol {I}} \psi \left(c _ {1} \sum_ {j \in \boldsymbol {I} \backslash \{i \}} \phi \left(\left(\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i}\right) \right. \\ \left. + c _ {2} \sum_ {j \in \boldsymbol {I} \backslash \{i \}} \phi \left(\left(\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i}\right)\right), \\ \end{array}
$$

for some constants $(c_{1}, c_{2}) \in \{(0,1), (1,0), (1,1)\}$ and some convex and increasing functions $\phi, \psi : \mathbb{R} \to \mathbb{R}$ .

Definition 3.2 (Independently Additive Contrastive Loss). For a given index set $I \subseteq [n]$ , the contrastive loss $\mathcal{L}_{\mathrm{ind - add}}(U_I, V_I)$ is defined as

$$
\begin{array}{l} \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U} _ {\boldsymbol {I}}, \boldsymbol {V} _ {\boldsymbol {I}}) := - \frac {1}{| \boldsymbol {I} |} \sum_ {i \in \boldsymbol {I}} \phi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}) \tag {3} \\ + \frac {c _ {1}}{| \boldsymbol {I} | (| \boldsymbol {I} | - 1)} \sum_ {i \neq j \in \boldsymbol {I}} \psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}) \\ + \frac {c _ {2}}{2 | \boldsymbol {I} | (| \boldsymbol {I} | - 1)} \sum_ {i \neq j \in \boldsymbol {I}} \left(\psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j}) + \psi (\boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j})\right) \\ \end{array}
$$

for some constants $(c_{1}, c_{2}) \in \{(0, 1), (1, 0), (1, 1)\}$ , where $\phi : R \to R$ is a differentiable, concave, and increasing function, and $\psi : R \to R$ is a differentiable, convex, and increasing function. Here, $i \neq j \in I$ is a simplified notation representing $i \in I$ and $j \in I \setminus \{i\}$ .

The constants $(c_{1}, c_{2}) \in \{(0, 1), (1, 0), (1, 1)\}$ in both loss formulations determine which types of negative pairs are included in the contrastive loss. The cross-view negatives refer to pairs of embeddings from different views of different instances, i.e., $(\boldsymbol{u}_{i}, \boldsymbol{v}_{j})$ with $i \neq j$ , whereas within-view negatives are pairs from the same view but different instances, i.e., $(\boldsymbol{u}_{i}, \boldsymbol{u}_{j})$ or $(\boldsymbol{v}_{i}, \boldsymbol{v}_{j})$ with $i \neq j$ (Shen et al., 2016). Setting $c_{1} = 1$ includes cross-view negatives, while $c_{2} = 1$

![](images/542c5f0ecdab79e5df733e382be944db350512ccb597a206807523f953014385.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁
neg.
u₂
pos.
...
un
neg.
v₁
neg.
v₂
pos.
...
vn
</details>

(a) $(c_{1}, c_{2}) = (1, 0)$

![](images/ee614d9362a42e72641efe047de4da0563af7fd981db8a2b39c26b7aff8e3e56.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁ neg.
u₂ pos.
...
uₙ neg.
v₁ neg.
v₂ pos.
...
vₙ neg.
...
</details>

(b) $(c_{1}, c_{2}) = (0, 1)$

![](images/bfd283760845d08f1602793c8672e4fc09d91f830325fb4169383bf21a27aa44.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁ neg. neg.
u₂ pos.
...
neg. neg.
uₙ neg. neg.
v₁ neg. neg.
v₂ pos. neg.
...
neg. neg.
vₙ neg. neg.
</details>

(c) $(c_{1},c_{2}) = (1,1)$   
Figure 1. Illustration of negative pair considered in the loss formulations defined in Def. 3.1 and Def. 3.2, which depends on the choice of $(c_{1}, c_{2})$ . Each grid shows all possible pairs of embeddings in $U_{[n]}$ and $V_{[n]}$ , and each cell represents one pair. Green regions represent positive pairs, and blue-striped regions indicate which negative pairs are included in the loss.

includes within-view negatives. Figure 1 summarizes which types of negatives are incorporated for each configuration of $(c_{1}, c_{2})$ . Further discussion on the distinction between cross-view and within-view negatives in the single-modal case are provided in Appendix B, emphasizing that their key difference lies in how they are structurally incorporated into the loss, not in how they are generated.

Remark 3.3. The first form of losses in Def. 3.1 encompasses a variety of contrastive losses such as InfoNCE (Oord et al., 2018; Radford et al., 2021), SimCLR (Chen et al., 2020), DCL (Yeh et al., 2022), and DHEL (Koromilas et al., 2024), see Appendix A.1. The second form in Def. 3.2 includes contrastive losses such as SigLIP (Zhai et al., 2023) and Spectral CL (HaoChen et al., 2021), see Appendix A.2.

The difference between the two forms of losses lies in computational efficiency. The first form in Def. 3.1 necessitates simultaneous computation based on pairwise similarities across the entire set of embeddings due to the need for normalization, which becomes impractical for extremely large batch sizes. In contrast, the second form in Def. 3.2 is independently additive, allowing it to compute components of each pairwise similarity individually and aggregate them, making it applicable to larger datasets.

# 4. Similarities of Embedding Pairs

In CL, the encoder is trained to bring positive embedding pairs closer together while pushing negative embedding pairs further apart. Accordingly, the cosine similarities of embedding pairs provide a straightforward way to evaluate how well representation achieves its objective. These similarities are formally defined as follows.

Definition 4.1 (Similarities of Positive/Negative Pairs). The similarity of embeddings of a positive pair is defined as

$$
\boldsymbol {s} (f; \hat {p} _ {\mathrm{pos}}) := f (\boldsymbol {x}) ^ {\top} f (\boldsymbol {y}) \quad \text { for } \quad (\boldsymbol {x}, \boldsymbol {y}) \sim \hat {p} _ {\mathrm{pos}},
$$

dubbed as the positive-pair similarity for the encoder f. Similarly, the similarity of embeddings of a negative pair is defined as

$$
\boldsymbol {s} (f; \hat {p} _ {\text { neg }}) := f (\boldsymbol {x}) ^ {\top} f (\boldsymbol {y}) \quad \text {   for   } \quad (\boldsymbol {x}, \boldsymbol {y}) \sim \hat {p} _ {\text { neg }},
$$

dubbed as the negative-pair similarity for the encoder f. We call both similarities as embedding similarities.

Note that the similarities of embeddings in Def. 4.1, denoted by $s(f; \hat{p}_{\mathrm{pos}})$ and $s(f; \hat{p}_{\mathrm{neg}})$ , are random variables, where randomness arises from data sampling and augmentation. Specifically, a positive pair is generated by selecting an instance from a dataset of size n and applying two random augmentations. Similarly, a negative pair is generated by randomly selecting an instance pair from the $n(n - 1)$ possible combinations and independently applying random augmentations to each instance.

Expectation & Variance of Negative-pair Similarities. Recall that the negative-pair similarity $s(f;\hat{p}_{\mathrm{neg}})$ is a random variable. Here we investigate how the expectation and the variance of $s(f;\hat{p}_{\mathrm{neg}})$ affect the learned embeddings. First, as the expectation $\mathbb{E}\left[s(f;\hat{p}_{\mathrm{neg}})\right]$ increases, the negative pairs are less separated, which typically degrades the representation quality. Second, the high variance $\operatorname{Var}\left[s(f;\hat{p}_{\mathrm{neg}})\right]$ implies that some negative pairs are mapped unusually close, while others are mapped much farther apart. Figure 2 shows the effect of the variance $\operatorname{Var}\left[s(f;\hat{p}_{\mathrm{neg}})\right]$ of the negative-pair similarities. Here, we have three different cases of learned embeddings $\{u_{i},v_{i}\}_{i\in[4]}$ in three dimensional space. While all three cases have same expectation $\mathbb{E}\left[s(f;\hat{p}_{\mathrm{neg}})\right]$ , the geometry

Mini-batch loss: $\arg \min \left\{\mathcal{L}\left(U_{[1:2]},V_{[1:2]}\right) + \mathcal{L}\left(U_{[3:4]},V_{[3:4]}\right)\right\}$

Full-batch loss: $\arg \min \left\{\mathcal{L}\left(U_{[1:4]},V_{[1:4]}\right)\right\}$

![](images/7dc0e9a5cd4ebc5239924c71e07f591ac833152af815ccb90bda43526e06ba78.jpg)  
Figure 2. Visualization of three different cases of eight embeddings on the three-dimensional unit sphere. Positive embedding pairs are represented in the same color and share the same subscript, while negative pairs refer to any two embeddings with different subscripts. In (a) and (b), embeddings minimize the fixed mini-batch contrastive loss described in Theorem 5.5, with the batches partitioned as $\{\pmb{u}_1,\pmb{v}_1,\pmb{u}_2,\pmb{v}_2\}$ and $\{\pmb{u}_3,\pmb{v}_3,\pmb{u}_4,\pmb{v}_4\}$ . In (c), the embeddings minimize the full-batch contrastive loss described in Theorem 5.1. In all cases, the expectation of negative-pair similarities, $\mathbb{E}_{i\neq j\in [4]}[\pmb{u}_i^\top \pmb{v}_j]$ , remains the same. However, the variance of negative-pair similarities, $\mathrm{Var}_{i\neq j\in [4}[\pmb{u}_i^\top \pmb{v}_j]$ , increases in mini-batch settings, indicating that some negative pairs are much more similar to each other while others are more dissimilar.

of embeddings significantly changes depending on the variance $\operatorname{Var}\left[s(f;\hat{p}_{\mathrm{neg}})\right]$ . One can confirm that the rightmost case having equi-distant embeddings $\{u_{i}\}_{i\in[4]}$ achieves the zero variance of negative-pair similarities.

Comparison with Existing Metrics. The similarities of embeddings in Def. 4.1 are related with metrics proposed in previous work. For example, the alignment metric used in (Wang & Isola, 2020) can be represented as

$$
\mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \| \boldsymbol {u} - \boldsymbol {v} \| _ {2} ^ {2} \right] = - 2 \mathbb {E} \left[ \boldsymbol {s} (f; \hat {p} _ {\mathrm{pos}}) \right] + 2, (4)
$$

which is related with the positive-pair similarity $s(f; \hat{p}_{\mathrm{pos}})$ . Here, the higher alignment value indicates that representations are largely invariant to random noise factors introduced by augmentations.

On the other hand, the uniformity metric used in (Wang & Isola, 2020) is related with the negative-pair similarity $s(f; \hat{p}_{\mathrm{neg}})$ , since it is defined by the logarithm of the Gaussian potential function (Cohn & Kumar, 2007) as

$$
\begin{array}{l} \log \mathbb {E} _ {\substack {\boldsymbol {u} \sim f _ {\sharp} \hat {p} _ {x} \\ \boldsymbol {v} \sim f _ {\sharp} \hat {p} _ {y}}} \left[ \exp \left(- \| \boldsymbol {u} - \boldsymbol {v} \| _ {2} ^ {2}\right) \right] (5) \\ \approx 2 \left(\mathbb {E} \left[ \boldsymbol {s} (f; \hat {p} _ {\mathrm{neg}}) \right] + \operatorname{Var} \left[ \boldsymbol {s} (f; \hat {p} _ {\mathrm{neg}}) \right] - 1\right), (6) \\ \end{array}
$$

where the approximation in (6) is detailed in Appendix C.1. The uniformity metric measures how representations are uniformly distributed on the unit hypersphere, and thus a lower uniformity value indicates that the representations preserve more information from the data.

# 5. Behavior of Learned Embeddings

In this section, we interpret the behavior of embeddings trained by contrastive learning, through the lens of similarities of positive/negative pairs, denoted by $s(f; \hat{p}_{\mathrm{pos}})$ and $s(f; \hat{p}_{\mathrm{neg}})$ , defined in Sec. 4. We begin by examining the full-batch CL and subsequently extend our analysis to the mini-batch CL. Proofs for all statements in Sec. 5.1 and Sec. 5.2 are provided in Appendix C.3 and C.4, respectively.

# 5.1. Full-Batch Contrastive Learning

Recall that we consider various contrastive losses which can be categorized into two parts: the InfoNCE-based contrastive loss in Def. 3.1 and the independently additive contrastive loss in Def. 3.2. Our first main result below provides the similarities of positive/negative pairs for the two types of contrastive losses, when the encoder is full-batch trained.

Theorem 5.1. Suppose $d \geq n - 1$ . Let the contrastive loss $\mathcal{L}(\boldsymbol{U}, \boldsymbol{V})$ be one of the following forms:

$$
i. \mathcal {L} _ {\text { info - sym }} (\boldsymbol {U}, \boldsymbol {V}) \text {   in   Def.   3.1. }
$$

$$
i i. \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U}, \boldsymbol {V}) \text {   in   Def.   3.2,   where   } (c _ {1}, c _ {2}) \in \{(0, 1), (1, 1) \}.
$$

$$
i i i. \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U}, \boldsymbol {V}) \text {   in   Def.   3.2,   where   } (c _ {1}, c _ {2}) = (1, 0) \text {   and   }
$$

$$
\phi^ {\prime} (1) > \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(- \frac {1}{n - 1}\right).
$$

Then, in full-batch settings, the embedding similarities for the optimal encoder $f^{\star}$ in (1) are

$$
\boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

According to Theorem 5.1, all positive pairs achieve the perfect alignment, while all negative pairs are uniformly separated with the cosine similarity of $-\frac{1}{n-1}$ . This is a generalized version of previous studies (Lu & Steinerberger, 2022; Cho et al., 2024; Lee et al., 2024; Koromilas et al., 2024), where we extend in two directions. First, the CL loss formulations (Def. 3.1 and Def. 3.2) we considered are a much broader class of losses compared with existing work. Second, we consider the randomness of $n$ embedding pairs in the optimization, where this randomness is introduced through augmentations. Now we analyze the behavior of embeddings for arbitrary encoder $f$ , that is not necessarily in the optimal status $f^{\star}$ . The below result provides the relationship between the positive-pair similarity and the negative-pair similarity.

Theorem 5.2. For any normalized encoder f,

$$
\mathbb {E} \left[ \boldsymbol {s} (f; \hat {p} _ {\mathrm{pos}}) \right] \leq 1 + \left(\mathbb {E} \left[ \boldsymbol {s} (f; \hat {p} _ {\mathrm{neg}}) \right] + \frac {1}{n - 1}\right), \tag {7}
$$

where the equality in (7) holds if and only if $\operatorname{tr}\left(\mathrm{Var}_{(\boldsymbol{u},\boldsymbol{v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}[\boldsymbol{u} - \boldsymbol{v}])} = 0$ and $\mathbb{E}_{\substack{\boldsymbol{u}\sim f_{\sharp}\hat{p}_x[\boldsymbol {u} + \boldsymbol {v}] = \boldsymbol {0}.\\ \boldsymbol {v}\sim f_{\sharp}\hat{p}_y}}$

Theorem 5.2 highlights the relationship between the expectation of positive-pair similarities, $\mathbb{E}\left[s\left(f;\hat{p}_{\mathrm{pos}}\right)\right]$ , and that of negative-pair similarities, $\mathbb{E}\left[s\left(f;\hat{p}_{\mathrm{neg}}\right)\right]$ . When the average of negative-pair similarities drops below $-\frac{1}{n-1}$ , the positive pairs cannot be fully aligned, which is not desired. We call such phenomenon as the excessive separation of negative pairs in full-batch CL, since the average of negative-pair similarities drops below $-\frac{1}{n-1}$ when negative pairs are more separated compared with the optimal status specified in Theorem 5.1. This issue may arise when certain losses are used, as shown in the following theorem.

Theorem 5.3 (Excessive Separation in Full-Batch CL). Suppose that $d \geq n$ . Consider the contrastive loss $\mathcal{L}(\boldsymbol{U}, \boldsymbol{V})$ in the form of $\mathcal{L}_{\text{ind-add}}(\boldsymbol{U}, \boldsymbol{V})$ in Def. 3.2, where $(c_1, c_2) = (1, 0)$ and

$$
\phi^ {\prime} (1) <   \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(- \frac {1}{n - 1}\right). \tag {8}
$$

Then, under full-batch settings, the embedding similarities for the optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

The inequality condition on $\phi$ and $\psi$ in (8) explains why the loss $\mathcal{L}_{\mathrm{ind - add}}(\boldsymbol {U},\boldsymbol {V})$ in Def. 3.2 causes the excessive separation of negative pairs. Specifically, $\mathcal{L}_{\mathrm{ind - add}}(\boldsymbol {U},\boldsymbol {V})$ is formulated as the sum of $\psi (\cdot)$ over negative pairs and $-\phi (\cdot)$ over positive pairs. Therefore, $\psi^{\prime}(-\frac{1}{n - 1})$ indicates how much the loss decreases if the negative-pair similarity falls below $-\frac{1}{n - 1}$ , while $\phi '(1)$ measures how much the loss increases if the positive-pair similarity drops below 1. When the condition in (8) is satisfied, ignoring the scaling factor, the loss reduction from separating negative pairs beyond the similarity threshold of $-\frac{1}{n-1}$ can be greater than the loss increase from reducing positive-pair similarity below 1. As a result, the optimization process favors pushing negative pairs even further apart, leading to the excessive separation. We provide a specific loss where this issue arises:

Example 5.4. Consider the sigmoid contrastive loss $\mathcal{L}_{\mathrm{sig}}(U,V)$ (Zhai et al., 2023), defined as

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{sig}} (\boldsymbol {U}, \boldsymbol {V}) := \frac {1}{n} \sum_ {i \in [ n ]} \log \left(1 + \exp \left(- t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \cdot \exp (b)\right) \\ + \frac {1}{n} \sum_ {i \neq j \in [ n ]} \log \left(1 + \exp \left(t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) \cdot \exp (- b)\right), \tag {9} \\ \end{array}
$$

where t > 0 and $b \in R$ are hyperparameters. This loss follows the form of $\mathcal{L}_{\text{ind-add}}(\boldsymbol{U}, \boldsymbol{V})$ in Def. 3.2, where $(c_{1}, c_{2}) = (1, 0)$ , $\phi(x) = -\log(1 + \exp(-tx + b))$ , and $\psi(x) = (n - 1) \cdot \log(1 + \exp(tx - b))$ . If hyperparameters t and b are chosen such that

$$
\frac {1 + \exp \left(\frac {t}{n - 1} + b\right)}{1 + \exp (t - b)} <   \frac {n - 2}{2}, \tag {10}
$$

then embedding similarities for the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

Note that (10) is just a rephrase of (8) by plugging in $\phi$ and $\psi$ for the sigmoid contrastive loss. When the hyperparameter b of the sigmoid contrastive loss is sufficiently small, the condition in (10) is satisfied, thus the learned embedding suffers from the excessive separation issue. This can be also explained by the sigmoid contrastive loss formula. In (9), the relative weight of second term (compared to the first term) increases as b decreases. In such case, minimizing the negative-pair similarity becomes more important. Consequently, decreasing b induces the excessive separation of negative pairs.

Mitigating Excessive Separation. A natural question that arises is whether the excessive separation of negative pairs in full-batch CL can be mitigated. According to Theorem 5.1 and Theorem 5.3, this issue depends on the specific form of the contrastive loss. In particular, under the independently additive loss $\mathcal{L}_{\mathrm{ind-add}}(\boldsymbol{U},\boldsymbol{V})$ , the optimal embeddings do not suffer from excessive separation when $c_{2}=1$ (i.e., case (ii) of Theorem 5.1), whereas the issue arises when $c_{2}=0$ and condition (8) holds.

This observation suggests two potential solutions to the excessive separation problem. The first is to set $c_{2} = 1$ in the loss, effectively incorporating within-view negative pairs. The second is to tune hyperparameters such that condition (8) does not hold, i.e., case (iii) in Theorem 5.1. While both approaches are theoretically valid, the former offers a more principled and practical remedy. In contrast, the latter lacks clear guidance for selecting suitable hyperparameters and may incur significant computational overhead. Therefore, we advocate including within-view negative pairs as a straightforward and effective strategy to avoid excessive separation in full-batch CL.

# 5.2. Mini-Batch Contrastive Learning

In Sec. 5.1, we show that when a proper contrastive loss is chosen for full-batch settings, the learned embedding satisfies the following behavior: all negative pairs have the cosine similarity of $-\frac{1}{n-1}$ and all positive pairs are perfectly aligned. What about the practical scenarios when we use mini-batches for training? Suppose n training samples are partitioned into b mini-batches where each batch contains m := n/b samples. Consider training the embeddings by under the fixed mini-batch configuration, where k-th mini-batch contains the samples with indices in $I_{k} = \{m(k-1) + 1, m(k-1) + 2, \cdots, mk\}$ . Under this scenario, the below theorem analyzes the behavior of embeddings learned by mini-batch training.

Theorem 5.5 (Excessive Separation in Mini-Batch CL). Suppose $d \geq m - 1$ . Let the contrastive loss $\mathcal{L}(\mathbf{U},\mathbf{V})$ be one of the forms in Theorem 5.1. Define $f_{\mathrm{batch}}^{\star}$ as the optimal encoder that minimizes the fixed mini-batch loss, given by

$$
f _ {\text {batch}} ^ {\star} := \arg \min _ {f} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\text {pos}} ^ {[ n ]}} \left[ \sum_ {k \in [ b ]} \mathcal {L} \left(\boldsymbol {U} _ {\boldsymbol {I} _ {k}}, \boldsymbol {V} _ {\boldsymbol {I} _ {k}}\right) \right],
$$

where $I_k := [m(k-1)+1:mk]$ for $k \in [b]$ . Then, embedding similarities for the optimal encoder $f_{\text{batch}}^{\star}$ satisfy

$$
\pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1,
$$

$$
\mathbb {E} \left[ \pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{neg}}) \right] = - \frac {1}{n - 1},
$$

$$
\operatorname{Var} \left[ \boldsymbol {s} (f _ {\text { batch }} ^ {\star}; \hat {p} _ {\text { neg }}) \right] \in \left[ \frac {n - m}{(m - 1) (n - 1) ^ {2}}, \frac {n (n - m)}{(m - 1) (n - 1) ^ {2}} \right]. \tag {11}
$$

A necessary condition for attaining the minimum variance of negative-pair similarities in (11) is $d \geq b(m - 1)$ .

According to Theorem 5.5, the embeddings learned by mini-batch settings have the following behaviors. First, the positive-pair similarity is equal to 1, i.e., all positive pairs are fully aligned. Second, the expectation of negative-pair similarities is equal to $-\frac{1}{n-1}$ , which happens for full-batch settings as well. Third, unlike full-batch settings, the negative-pair similarity is not uniform across the pairs, i.e., the variance is positive when the mini-batch size m is strictly less than the sample size n. Thus, the effect of using mini-batch (compared with using full-batch) is in the increased variance of negative-pair similarities. Throughout the paper, we call such phenomenon as the excessive separation of negative pairs in mini-batch CL.

Figure 2 visualizes the effect of using mini-batches, compared with full-batch settings, when n = 4 and m = 2. For full-batch settings, shown in (c) of Figure 2, the variance of negative-pair similarities is zero, indicating that all negative pairs are equi-distant. In contrast, (a) and (b) of Figure 2 illustrate the embeddings learned by mini-batch settings, where the fixed mini-batches are specified as $\left(\boldsymbol{U}_{[1:2]}, \boldsymbol{V}_{[1:2]}\right)$ and $\left(\boldsymbol{U}_{[3:4]}, \boldsymbol{V}_{[3:4]}\right)$ . For both (a) and (b), the variance of negative-pair similarities is positive, where (a) represents the case that achieves the highest variance in (11), and (b) corresponds to the case that achieves the lowest variance.

Effect of Batch Size. Note that the variance of negative-pair similarities in (11) depends on the batch size m. For example, in the full-batch case where m = n, the upper bound in (11) is zero, which is consistent with Theorem 5.1. One can confirm that the upper and lower bounds on the variance is a monotonically decreasing function of m, which implies that smaller batch sizes inherently exacerbate the excessive separation of negative pairs in mini-batch settings.

The below theorem analyzes the effect of batch size on the training dynamics, when a popular CL loss is used:

Theorem 5.6. Consider the InfoNCE loss $\mathcal{L}_{\mathrm{InfoNCE}}(U,V)$ (Oord et al., 2018), which corresponds to the loss $\mathcal{L}_{\mathrm{info-sym}}(U,V)$ in Def. 3.1 where $\phi(x) = \exp(x/t)$ for some $t > 0$ , $\psi(x) = \log(1 + x)$ , and $(c_1,c_2) = (1,0)$ . For any two integers $m_1,m_2\in [n]$ satisfying $m_1\leq m_2$ , the gradient of the InfoNCE loss with respect to the negative-pair similarity satisfies

$$
0 \leq \frac {\partial}{\partial \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right)} \mathcal {L} _ {\text { InfoNCE }} \left(\boldsymbol {U} _ {[ m _ {2} ]}, \boldsymbol {V} _ {[ m _ {2} ]}\right)
$$

$$
\leq \frac {\partial}{\partial \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right)} \mathcal {L} _ {\text { InfoNCE }} \left(\boldsymbol {U} _ {[ m _ {1} ]}, \boldsymbol {V} _ {[ m _ {1} ]}\right) \tag {12}
$$

for any distinct indices $i \neq j \in [m]$ . Moreover, the equality in (12) holds if and only if $m_1 = m_2$ .

According to Theorem 5.6, the gradient of the loss with respect to the negative-pair similarity is always non-negative, implying that gradient descent decreases the similarities of negative pairs, thereby pushing them further apart. Notably, the magnitude of this gradient increases as the batch size gets smaller. As a result, negative pairs within each batch exhibit greater separation the batch size gets smaller.

Proposed Variance Reduction Method. Prior empirical studies on CL have shown that mini-batch settings often underperform compared to full-batch settings (Chen et al., 2020; Radford et al., 2021). This naturally raises a question: What is the main factor contributing to this performance degradation in mini-batch settings, and how can it be addressed? According to Theorem 5.5, from the perspective of cosine similarity of embeddings, the key difference introduced by mini-batch settings lies in the increased variance of negative-pair similarities. Motivated by this theoretical insight, we propose an approach to improve mini-batch contrastive learning by introducing an auxiliary loss term, $\mathcal{L}_{\mathrm{VRNS}}(\mathbf{U},\mathbf{V})$ , which explicitly reduces the variance of negative-pair similarities:

Definition 5.7 (Reducing Variance of Negative-Pair Similarities). Let $m$ be the mini-batch size. Define

$$
\mathcal {L} _ {\mathrm{VRNS}} \left(\mathbf {U} _ {[ m ]}, \mathbf {V} _ {[ m ]}\right) := \frac {1}{m (m - 1)} \sum_ {i \neq j \in [ m ]} \left(\mathbf {u} _ {i} ^ {\top} \mathbf {v} _ {j} + \frac {1}{n - 1}\right) ^ {2}
$$

as the auxiliary loss for reducing the variance of negative-pair similarities.

One can combine arbitrary conventional mini-batch loss $\mathcal{L}\left(\boldsymbol{U}_{[m]},\boldsymbol{V}_{[m]}\right)$ with the proposed auxiliary loss to get the modified loss, given by

$$
\mathcal {L} \left(\boldsymbol {U} _ {[ m ]}, \boldsymbol {V} _ {[ m ]}\right) + \lambda \cdot \mathcal {L} _ {\text { VRNS }} \left(\boldsymbol {U} _ {[ m ]}, \boldsymbol {V} _ {[ m ]}\right),
$$

where $\lambda > 0$ is a hyperparameter. By including the proposed term into the mini-batch loss, we encourage all negative-pair similarities to be close to $-\frac{1}{n-1}$ , the ideal value achieved in full-batch settings in Theorem 5.1. As a result, the proposed loss controls the variance of negative-pair similarities.

# 6. Empirical Validation

In this section, we empirically validate the impact of our theoretical results discussed in Sec. 5, especially for the practical scenarios of mini-batch settings. First, we empirically observe that the excessive separation of negative pairs (proven in Theorem 5.5) actually happens in experiments on benchmark datasets. Second, we empirically confirm that such excessive separation issue can be mitigated by using the proposed loss in Def. 5.7 which reduces the variance of the negative-pair similarities. Third, we observe such variance reduction improves the quality of learned representations in various real-world experiments.

# 6.1. Excessive Separation of Negative Pairs

To investigate the excessive separation of negative pairs in CL, we evaluate the variance of the negative-pair similarities of embeddings learned by real-world experiments. Following prior works on contrastive learning (Chen et al., 2020; Table 1. Variance of the similarities of embeddings of negative pairs, obtained from models trained with different batch sizes. Each model is trained using either the SimCLR loss alone or jointly with our auxiliary loss in Def. 5.7, which is proposed to reduce this variance. One can confirm that the variance is effectively reduced by using the proposed auxiliary loss.

<table><tr><td rowspan="2">Batch size</td><td colspan="2">Variance of negative-pair similarities</td></tr><tr><td>SimCLR</td><td>SimCLR + Ours</td></tr><tr><td>32</td><td>0.1649</td><td>0.1008</td></tr><tr><td>64</td><td>0.1505</td><td>0.0952</td></tr><tr><td>128</td><td>0.1444</td><td>0.0929</td></tr><tr><td>256</td><td>0.1404</td><td>0.0921</td></tr><tr><td>512</td><td>0.1396</td><td>0.0917</td></tr></table>

Koromilas et al., 2024), we use a ResNet-18 encoder (He et al., 2016) followed by a two-layer projection head. The models are pretrained on CIFAR-100 (Krizhevsky et al., 2009) by minimizing the SimCLR loss with the temperature parameter of t = 0.2. Five models are trained with mini-batches sampled uniformly at random, with batch sizes of 32, 64, 128, 256, and 512, respectively. Additional details of the experimental setup are provided in Appendix D.

Based on these pretrained models, we generate 5,000 positive embedding pairs by applying random augmentations to the training data and extracting the corresponding outputs from the projection head. We then compute the cosine similarities of negative pairs, and report the variance of these similarities in Table 1. As shown in the table, training with smaller batch sizes leads to higher variance in negative-pair similarities, which aligns with the result in Theorem 5.5.

To evaluate the effectiveness of our proposed auxiliary loss, we train five models for each batch size by minimizing the SimCLR loss combined with the auxiliary loss $\mathcal{L}_{\mathrm{VRNS}}(\mathbf{U},\mathbf{V})$ in Def. 5.7 with the hyperparameter of $\lambda=30$ . The variances of negative-pair similarities from these additional models are shown in the last column of Table 1, and are consistently reduced across all batch sizes. This indicates that our proposed loss effectively mitigates excessive separation of negative pairs in mini-batch settings.

# 6.2. Effect of Variance Reduction on Performance

We further investigate whether reducing the variance of negative-pair similarities improves the quality of learned representations in terms of the downstream performances.

Experimental Setup. We pretrain models on CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009), and ImageNet (Deng et al., 2009) using various contrastive losses that follow the formulation in Def. 3.1, including SimCLR, DCL, and DHEL. For all methods, we compare models trained with and without incorporating the auxiliary loss $\mathcal{L}_{\mathrm{VRNS}}(\mathbf{U},\mathbf{V})$

![](images/20697ae8dc8b8056167da09c915798250530139839343a7aeb0e777e45e8d189.jpg)

<details>
<summary>line</summary>

| temperature parameter | SimCLR + Ours (CIFAR 10) | SimCLR (CIFAR 10) | SimCLR + Ours (CIFAR 100) | SimCLR (CIFAR 100) |
| --------------------- | ------------------------ | ----------------- | ------------------------- | ------------------ |
| 0.07                  | 84.0                     | 84.0              | 57.0                      | 56.0               |
| 0.10                  | 86.0                     | 86.0              | 59.0                      | 58.0               |
| 0.25                  | 88.0                     | 88.0              | 60.0                      | 59.0               |
| 0.50                  | 89.0                     | 89.0              | 60.0                      | 59.0               |
| 1.00                  | 88.0                     | 87.0              | 57.0                      | 46.0               |
| 2.00                  | 86.0                     | 81.0              | 54.0                      | 37.0               |
</details>

Figure 3. Classification accuracy on CIFAR datasets. Models are trained by minimizing the SimCLR loss with and without the auxiliary loss proposed in Def. 5.7, using various temperature parameters in the SimCLR loss.

in Def. 5.7. The hyperparameter $\lambda$ for the proposed loss is tuned over $\{0.1, 0.3, 1, 3, 10, 30, 100\}$ . Unless otherwise specified, the other settings follow those in Sec. 6.1.

Performance Gains from Variance Reduction. The quality of the representations learned through CL is known to be sensitive to the choice of the temperature parameter in contrastive losses, as it influences the distribution of similarities among embeddings (Wang & Liu, 2021). To investigate this sensitivity, we train models using the SimCLR loss with temperature values ranging from 0.07 to 2.00. We compare the standard SimCLR loss against our proposed variant, which incorporates the auxiliary loss introduced in Def. 5.7 to reduce the variance of negative-pair similarities. As shown in Figure 3, incorporating the proposed term leads to a consistently higher and more stable classification accuracy across all temperature settings, alleviating the need for careful temperature tuning.

We further evaluate the auxiliary loss in Def. 5.7 on existing CL methods, including DCL and DHEL. Experiments are conducted on the CIFAR datasets with various batch sizes. As shown in Figure 4, incorporating the proposed term leads to improved classification accuracy, with the effect being more pronounced at smaller batch sizes. Additional results on the ImageNet dataset are presented in Appendix E.

Caveats of Variance Reduction. While the auxiliary loss proposed in Def. 5.7 effectively reduces the variance of negative-pair similarities, this reduction can influence both desirable and undesirable sources of variance. On the positive side, it helps reduce variance introduced by mini-batch sampling, which can degrade the representation quality. However, it may also suppress the variance that captures meaningful structure in the data. Further discussion of the limitations of the proposed loss is provided in Appendix F.

![](images/5b6f0f11eb3d1669f075b41121f255beb4344ff58db2e316ebc68d47299b5e4f.jpg)

<details>
<summary>bar</summary>

| dataset   | batch size | SimCLR | DCL  | DHEL | + Ours |
| --------- | ---------- | ------ | ---- | ---- | ------ |
| CIFAR-10  | 32         | 86.4   | 87.0 | 86.0 | 87.1   |
| CIFAR-10  | 64         | 87.7   | 87.9 | 87.5 | 88.2   |
| CIFAR-10  | 128        | 88.0   | 88.4 | 88.4 | 88.6   |
| CIFAR-10  | 256        | 88.2   | 88.7 | 88.5 | 88.8   |
| CIFAR-10  | 512        | 88.5   | 88.6 | 88.6 | 89.0   |
| CIFAR-100 | 32         | 55.7   | 59.0 | 56.3 | 57.0   |
| CIFAR-100 | 64         | 58.1   | 60.7 | 59.7 | 60.3   |
| CIFAR-100 | 128        | 58.6   | 61.7 | 61.2 | 61.9   |
| CIFAR-100 | 256        | 59.7   | 61.4 | 61.7 | 61.7   |
| CIFAR-100 | 512        | 59.4   | 61.0 | 60.9 | 61.4   |
</details>

Figure 4. Effect of the auxiliary loss proposed in Def. 5.7 on top-1 classification accuracy when combined with various baseline methods (SimCLR, DCL, and DHEL) on CIFAR datasets. The gray bars highlight the performance gains achieved by incorporating the proposed term across various batch sizes. The proposed auxiliary loss consistently improves the model performance.

# 7. Conclusion

To understand contrastive learning (CL), we mathematically analyze the distributions of similarities of embeddings, measured for positive pairs and negative pairs. Our theoretical results in full-batch settings demonstrate that misalignment of positive pairs becomes inevitable when the average similarity of negative pairs falls below its optimal value, a situation that can arise with existing contrastive losses. In mini-batch settings, we prove that the variance of negative-pair similarities increases as the batch size decreases—a distinctive characteristic absent in full-batch settings and a potential contributor to the performance degradation observed in mini-batch settings. To address this, we propose an auxiliary loss that explicitly reduces the variance of negative-pair similarities. Empirical results show that incorporating the proposed loss improves the performance of CL methods, especially in small-batch settings.

Promising directions for future work include extending this work in two directions. First, disentangling the variance of negative-pair similarities that reflects intrinsic data structure from that caused by mini-batching could enable targeting only the variance that degrades representation quality. Second, analyzing the behavior of embedding similarities not only during pretraining but also during fine-tuning may provide deeper insights into its dynamics throughout different stages of training.

# Acknowledgements

This work was partially supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT) of the Korean government (RS-2024-00341749, RS-2024-00345351, RS-2024-00408003), under the ICT Challenge and Advanced Network of HRD (ICAN) support program (RS-2023-00259934, RS-2025-02283048), supervised by the Institute for Information & Communications Technology Planning & Evaluation (IITP). This research was also supported by the Yonsei University Research Fund (2025-22-0025).

We sincerely thank the anonymous reviewers for their critical reading and constructive feedback enhancing this paper.

# Impact Statement

This paper presents work whose goal is to theoretically understand CL through embedding similarities. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Bachman, P., Hjelm, R. D., and Buchwalter, W. Learning representations by maximizing mutual information across views. In Advances in Neural Information Processing Systems, volume 32, 2019.   
Chen, C., Zhang, J., Xu, Y., Chen, L., Duan, J., Chen, Y., Tran, S. D., Zeng, B., and Chilimbi, T. Why do we need large batchsizes in contrastive learning? a gradient-bias perspective. In Advances in Neural Information Processing Systems, 2022.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PMLR, 2020.   
Chen, X. and He, K. Exploring simple siamese representation learning. In IEEE/CVF conference on Computer Vision and Pattern Recognition, pp. 15750–15758, 2021.   
Chi, Z., Dong, L., Wei, F., Yang, N., Singhal, S., Wang, W., Song, X., Mao, X.-L., Huang, H., and Zhou, M. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In Conference of the North American Chapter of the Association for Computational Linguistics, June 2021.   
Cho, J., Sreenivasan, K., Lee, K., Mun, K., Yi, S., Lee, J.-G., Lee, A., yong Sohn, J., Papailiopoulos, D., and Lee, K. Mini-batch optimization of contrastive loss. Transactions on Machine Learning Research, 2024.

Cohn, H. and Kumar, A. Universally optimal distribution of points on spheres. Journal of the American Mathematical Society, 20(1):99–148, 2007.

da Costa, V. G. T., Fini, E., Nabi, M., Sebe, N., and Ricci, E. solo-learn: A library of self-supervised methods for visual representation learning. Journal of Machine Learning Research, 23(56):1–6, 2022.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 248–255. Ieee, 2009.

Gao, T., Yao, X., and Chen, D. SimCSE: Simple contrastive learning of sentence embeddings. In Conference on Empirical Methods in Natural Language Processing, 2021.

Gutmann, M. and Hyvärinen, A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In International Conference on Artificial Intelligence and Statistics, pp. 297–304, 2010.

HaoChen, J. Z., Wei, C., Gaidon, A., and Ma, T. Provable guarantees for self-supervised deep learning with spectral contrastive loss. In Advances in Neural Information Processing Systems, volume 34, pp. 5000–5011, 2021.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729–9738, 2020.

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., and Bengio, Y. Learning deep representations by mutual information estimation and maximization. In International Conference on Learning Representations, 2019.

Horn, R. A. and Johnson, C. R. Matrix analysis. Cambridge university press, 2012.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 2021.

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., and Krishnan, D. Supervised contrastive learning. In Advances in Neural Information Processing Systems, volume 33, pp. 18661–18673, 2020.

Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better? In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.   
Koromilas, P., Bouritsas, G., Giannakopoulos, T., Nicolaou, M., and Panagakis, Y. Bridging mini-batch and asymptotic analysis in contrastive learning: From InfoNCE to kernel-based losses. In International Conference on Machine Learning, 2024.   
Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.   
Lee, C., Chang, J., and Sohn, J.-y. Analysis of using sigmoid loss for contrastive learning. In International Conference on Artificial Intelligence and Statistics, 2024.   
Lee, C., Oh, J., Lee, K., and Sohn, J.-y. A theoretical framework for preventing class collapse in supervised contrastive learning. In International Conference on Artificial Intelligence and Statistics, 2025.   
Lee, H., Lee, K., Lee, K., Lee, H., and Shin, J. Improving transferability of representations via augmentation-aware self-supervision. In Advances in Neural Information Processing Systems, volume 34, pp. 17710–17722, 2021.   
Li, Y., Pogodin, R., Sutherland, D. J., and Gretton, A. Self-supervised learning with kernel dependence maximization. In Advances in Neural Information Processing Systems, 2021.   
Lu, J. and Steinerberger, S. Neural collapse under cross-entropy loss. Applied and Computational Harmonic Analysis, 59:224–241, 2022.   
Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.   
Papyan, V., Han, X., and Donoho, D. L. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.   
Pham, H., Dai, Z., Ghiasi, G., Kawaguchi, K., Liu, H., Yu, A. W., Yu, J., Chen, Y.-T., Luong, M.-T., Wu, Y., et al. Combined scaling for zero-shot transfer learning. Neurocomputing, 555:126658, 2023.   
Qian, R., Meng, T., Gong, B., Yang, M.-H., Wang, H., Belongie, S., and Cui, Y. Spatiotemporal contrastive video representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.,

et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Shen, X., Sun, Q.-S., and Yuan, Y.-H. Semi-paired hashing for cross-view retrieval. Neurocomputing, 213:14–23, 2016.   
Sustik, M. A., Tropp, J. A., Dhillon, I. S., and Heath Jr, R. W. On the existence of equiangular tight frames. Linear Algebra and its applications, 426(2-3):619–635, 2007.   
Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In European Conference on Computer Vision, 2020a.   
Tian, Y., Sun, C., Poole, B., Krishnan, D., Schmid, C., and Isola, P. What makes for good views for contrastive learning? In Advances in Neural Information Processing Systems, volume 33, pp. 6827–6839, 2020b.   
Waida, H., Wada, Y., Andéol, L., Nakagawa, T., Zhang, Y., and Kanamori, T. Towards understanding the mechanism of contrastive learning via similarity structure: A theoretical analysis. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2023.   
Wang, F. and Liu, H. Understanding the behaviour of contrastive loss. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2495–2504, 2021.   
Wang, F., Xiang, X., Cheng, J., and Yuille, A. L. Normface: L2 hypersphere embedding for face verification. In ACM international conference on Multimedia, 2017.   
Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pp. 9929–9939. PMLR, 2020.   
Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3733–3742, 2018.   
Xue, Y., Gan, E., Ni, J., Joshi, S., and Mirzasoleiman, B. Investigating the benefits of projection head for representation learning. In International Conference on Learning Representations, 2024.   
Yeh, C.-H., Hong, C.-Y., Hsu, Y.-C., Liu, T.-L., Chen, Y., and LeCun, Y. Decoupled contrastive learning. In European Conference on Computer Vision, 2022.   
Yuan, Z., Wu, Y., Qiu, Z.-H., Du, X., Zhang, L., Zhou, D., and Yang, T. Provable stochastic optimization for global contrastive learning: Small batch does not harm performance. In International Conference on Machine Learning, pp. 25760–25782. PMLR, 2022.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision, 2023.   
Zimmermann, R. S., Sharma, Y., Schneider, S., Bethge, M., and Brendel, W. Contrastive learning inverts the data generating process. In International Conference on Machine Learning, pp. 12979–12990. PMLR, 2021.

# A. Contrastive Losses

We outline how various losses commonly used in CL can be instantiated by the general formulation provided in Def. 3.1 and Def. 3.2. For each case, we specify the corresponding choices of functions and parameters.

A.1. Contrastive Losses Following Def. 3.1   
![](images/f94c0c2e679468d348c7ca2f2e75ad2a2212fb800104ca7fce714e29675930da.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁
u₂
⋮
uₙ
v₁ pos
v₂
⋮
vₙ
</details>

InfoNCE

![](images/0102467451899b780f7007201d62f20b87e06a7afb1df269706b539aca997da9.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁
u₂
⋮
uₙ
v₁ pos
v₂
⋮
vₙ
</details>

SimCLR

![](images/50bf111df3dc263cce69a0870713e460cad01c3bcda83b39e8fc91147877d02d.jpg)

<details>
<summary>text_image</summary>

u₁ u₂ ... uₙ v₁ v₂ ... vₙ
u₁
u₂
⋮
uₙ
v₁ pos
v₂
⋮
vₙ
</details>

DCL

![](images/ddb14160b557aca7c654e07fc5a95c98d60edd9863beb9a4487e5f43263a454e.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| u1 | pos |
| v1 | pos |
| v2 | pos |
| ... | ... |
| u_n | ... |
| v_1 | ... |
| v_2 | ... |
| ... | ... |
| v_n | ... |
</details>

DHEL   
Figure 5. Illustration comparing four different contrastive losses, all following the form of Def. 3.1. The green area represents the positive pair, while the blue-striped regions indicate the negative pairs that are normalized together with the positive pair for each loss.

Table 2. Function and parameter selections in Def. 3.1 that correspond to contrastive losses. 

<table><tr><td></td><td> $\phi(x)$ </td><td> $\psi(x)$ </td><td> $c_1$ </td><td> $c_2$ </td></tr><tr><td>InfoNCE (Oord et al., 2018)</td><td> $\exp(x/t)$ </td><td> $\log(1+x)$ </td><td>1</td><td>0</td></tr><tr><td>SimCLR (Chen et al., 2020)</td><td> $\exp(x/t)$ </td><td> $\log(1+x)$ </td><td>1</td><td>1</td></tr><tr><td>DCL (Yeh et al., 2022)</td><td> $\exp(x/t)$ </td><td> $\log(x)$ </td><td>1</td><td>1</td></tr><tr><td>DHEL (Koromilas et al., 2024)</td><td> $\exp(x/t)$ </td><td> $\log(x)$ </td><td>0</td><td>1</td></tr></table>

1. InfoNCE (Oord et al., 2018), CLIP (Radford et al., 2021):

$$
\begin{array}{l} \mathcal {L} _ {\text {InfoNCE}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ]} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t\right)}\right) - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ]} \exp \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} / t\right)}\right) \tag {13} \\ = \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(1 + \sum_ {j \in [ n ] \backslash \{i \}} \exp \left((\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}) ^ {\top} \boldsymbol {u} _ {i} / t\right)\right) \\ + \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(1 + \sum_ {j \in [ n ] \backslash \{i \}} \exp \left((\boldsymbol {u} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right)\right), \\ \end{array}
$$

where t > 0 is the temperature parameter.

2. SimCLR (Chen et al., 2020):

$$
\begin{array}{l} \mathcal {L} _ {\text {SimCLR}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ]} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t\right) + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} / t\right)}\right) \\ - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ]} \exp \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} / t\right) + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i} / t\right)}\right) \\ = \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(1 + \sum_ {j \in [ n ] \backslash \{i \}} \exp \left(\left(\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i} / t\right) + \sum_ {j \in [ n ] \backslash \{i \}} \exp \left(\left(\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i} / t\right)\right) \\ + \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(1 + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left((\boldsymbol {u} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right) + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left((\boldsymbol {v} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right)\right), \\ \end{array}
$$

where t > 0 is the temperature parameter.

3. DCL (Yeh et al., 2022):

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DCL}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t\right) + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} / t\right)}\right) \\ - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} / t\right) + \sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} / t\right)}\right) \\ = \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\sum_ {j \in [ n ] \backslash \{i \}} \left(\exp \left((\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}) ^ {\top} \boldsymbol {u} _ {i} / t\right) + \exp \left((\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}) ^ {\top} \boldsymbol {u} _ {i} / t\right)\right)\right) \\ + \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\sum_ {j \in [ n ] \backslash \{i \}} \left(\exp \left((\boldsymbol {u} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right) + \exp \left((\boldsymbol {v} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right)\right)\right), \\ \end{array}
$$

where t > 0 is the temperature parameter.

4. DHEL (Koromilas et al., 2024):

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DHEL}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} / t\right)}\right) - \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\frac {\exp \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} / t\right)}{\sum_ {j \in [ n ] \setminus \{i \}} \exp \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} / t\right)}\right) \\ = \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\sum_ {j \in [ n ] \setminus \{i \}} \exp \left((\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}) ^ {\top} \boldsymbol {u} _ {i} / t\right)\right) + \frac {1}{2 n} \sum_ {i \in [ n ]} \log \left(\sum_ {j \in [ n ] \setminus \{i \}} \exp \left((\boldsymbol {v} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t\right)\right), \\ \end{array}
$$

where t > 0 is the temperature parameter.

# A.2. Contrastive Losses Following Def. 3.2

Table 3. Function and parameter selections in Def. 3.2 that correspond to contrastive losses. 

<table><tr><td></td><td> $\phi(x)$ </td><td> $\psi(x)$ </td><td> $c_1$ </td><td> $c_2$ </td></tr><tr><td>SigLIP (Zhai et al., 2023)</td><td> $-\log(1+\exp(-tx+b))$ </td><td> $(n-1)\cdot\log(1+\exp(tx-b))$ </td><td>1</td><td>0</td></tr><tr><td>Spectral Contrastive Loss (HaoChen et al., 2021)</td><td>x</td><td> $x^2$ </td><td>1</td><td>0</td></tr></table>

1. SigLIP (Zhai et al., 2023):

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{SigLIP}} (\boldsymbol {U}, \boldsymbol {V}) = \frac {1}{n} \sum_ {i \in [ n ]} \log \left(1 + \exp \left(- t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} + b\right)\right) + \frac {1}{n} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \log \left(1 + \exp \left(t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} - b\right)\right) \\ = - \frac {1}{n} \sum_ {i \in [ n ]} \left(- \log \left(1 + \exp \left(- t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} + b\right)\right)\right) + \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} (n - 1) \cdot \log \left(1 + \exp \left(t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} - b\right)\right), \\ \end{array}
$$

where $t > 0$ is the temperature parameter and $b \in \mathbb{R}$ is the bias term.

2. Spectral Contrastive Loss (HaoChen et al., 2021):

$$
\mathcal {L} _ {\text {Spectral}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} + \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) ^ {2}.
$$

# B. Distinction Between Cross-View and Within-View Negative Pairs

![](images/6ceb9b69a4c8b90eefd570b6ef0603029807c47d6ec7da6b21438914480d01ad.jpg)

— cross-view
negative pair   
— within-view
negative pair

Figure 6. Graphs illustrating different loss configurations for n = 3.

The key distinction between cross-view and within-view negative pairs in our analysis lies in their structural incorporation within the contrastive loss, rather than in the manner of their generation. To demonstrate that cross-view and within-view negatives are not equivalent, we present a graph-based representation in Figure 6, which reframes Figure 1. In these graphs, each node corresponds to an embedding, and each edge indicates a negative pair considered in the loss.

In the unimodal CL, the distinction between the two views, $u_{i}$ and $v_{i}$ for $i \in [3]$ , is not semantically meaningful. Thus, $u_{i}$ and $v_{i}$ may be interchanged without affecting the results. This implies that the four graphs depicted in Figure 6a are equivalent under permutation of views, and the same reasoning applies to Figure 6b. Nevertheless, the overall graph structures in Figure 6a and Figure 6b remain fundamentally different. One can confirm that cross-view graphs are fully connected bipartite, whereas within-view graphs consist of disconnected subgraphs. This topological difference highlights their non-equivalence.

# C. Proofs

# C.1. Approximation of Uniformity Metric

Under the normality assumption on $u^{\top}v$ , Proposition C.1 gives

$$
\log \mathbb {E} \left[ \exp \left(2 \boldsymbol {u} ^ {\top} \boldsymbol {v}\right) \right] = 2 \left(\mathbb {E} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \operatorname{Var} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right]\right).
$$

Accordingly, the uniformity metric can be approximated as

$$
\begin{array}{l} \log \mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} \hat {p} _ {x}}   \boldsymbol {v} \sim f _ {\sharp} \hat {p} _ {y} \left[ \exp \left(- \| \boldsymbol {u} - \boldsymbol {v} \| _ {2} ^ {2}\right) \right] \approx \log \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \exp \left(- \| \boldsymbol {u} - \boldsymbol {v} \| _ {2} ^ {2}\right) \right] \\ \approx 2 \left(\mathbb {E} \left[ \boldsymbol {s} (f; \hat {p} _ {\text { neg }}) \right] + \operatorname{Var} \left[ \boldsymbol {s} (f; \hat {p} _ {\text { neg }}) \right] - 1\right), \\ \end{array}
$$

by Proposition C.2, as $n$ goes to infinity.

Proposition C.1. Assume that the random variable $u^{\top}v$ follows the normal distribution. Then,

$$
\log \mathbb {E} \left[ \exp \left(2 \boldsymbol {u} ^ {\top} \boldsymbol {v}\right) \right] = 2 \left(\mathbb {E} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \operatorname{Var} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right]\right)
$$

Proof. Let $X = u^{\top}v$ . Since X follows the normal distribution, we define $\mu := E[X]$ and $\sigma^{2} := Var[X]$ .

Note that the moment generating function of normal distribution is given by

$$
\mathbb {E} \left[ \exp (t X) \right] = \exp \left(\mu t + \frac {\sigma^ {2} t ^ {2}}{2}\right).
$$

Substituting $t = 2$ , we have

$$
\mathbb {E} \left[ \exp (2 X) \right] = \exp \left(2 \mu + 2 \sigma^ {2}\right) = \exp \left(2 \mathbb {E} [ X ] + 2 \operatorname{Var} [ X ]\right),
$$

which is equal to

$$
\log \mathbb {E} [ \exp (2 X) ] = 2 (\mathbb {E} [ X ] + \operatorname{Var} [ X ]).
$$

Since $X = \pmb{u}^{\top}\pmb{v}$ , we conclude

$$
\log \mathbb {E} \left[ \exp \left(2 \boldsymbol {u} ^ {\top} \boldsymbol {v}\right) \right] = 2 \left(\mathbb {E} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \operatorname{Var} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right]\right).
$$

# C.2. Proofs for Relation Between Positive and Negative Pairs

Proposition C.2. The distribution of negative pairs satisfies $p_{\mathrm{neg}}(\boldsymbol{x}, \boldsymbol{y}) = p_x(\boldsymbol{x}) p_y(\boldsymbol{y})$ for all x and y. However, for a training dataset of size n, the empirical distribution of negative pairs is given by

$$
\hat {p} _ {\text { neg }} (\boldsymbol {x}, \boldsymbol {y}) = \frac {n}{n - 1} \cdot \hat {p} _ {x} (\boldsymbol {x}) \hat {p} _ {y} (\boldsymbol {y}) - \frac {1}{n - 1} \cdot \hat {p} _ {\text { pos }} (\boldsymbol {x}, \boldsymbol {y}),
$$

for all $\pmb{x}$ and $\pmb{y}$ .

Proof. The empirical distribution of positive pairs, $\hat{p}_{pos}$ , is defined under the assumption that all instances are equally weighted with the probability $\Pr\{I=i\}=\frac{1}{n}$ for all $i\in[n]$ , where I is a random variable representing the index. Under this assumption, the probability that a randomly selected pair is positive is

$$
\operatorname * {P r} \{\text { pos } \} = \operatorname * {P r} \{\boldsymbol {i} = \boldsymbol {i} ^ {\prime} \} = \sum_ {i \in [ n ]} \operatorname * {P r} \{\boldsymbol {i} = i \} \operatorname * {P r} \{\boldsymbol {i} = i \} = n \cdot \frac {1}{n ^ {2}} = \frac {1}{n}.
$$

Then, the empirical distribution of negative pairs, $\hat{p}_{neg}$ , is subsequently derived as

$$
\begin{array}{l} \hat {p} _ {x} (\boldsymbol {x}) \hat {p} _ {y} (\boldsymbol {y}) = \hat {p} _ {\text { pos }} (\boldsymbol {x}, \boldsymbol {y}) \operatorname * {P r} \{\text { pos } \} + \hat {p} _ {\text { neg }} (\boldsymbol {x}, \boldsymbol {y}) (1 - \operatorname * {P r} \{\text { pos } \}) \\ = \hat {p} _ {\text { pos }} (\boldsymbol {x}, \boldsymbol {y}) \cdot \frac {1}{n} + \hat {p} _ {\text { neg }} (\boldsymbol {x}, \boldsymbol {y}) \cdot \frac {n - 1}{n}, \tag {14} \\ \end{array}
$$

which leads to

$$
\hat {p} _ {\mathrm{neg}} (\pmb {x}, \pmb {y}) = \frac {n}{n - 1} \cdot \hat {p} _ {x} (\pmb {x}) \hat {p} _ {y} (\pmb {y}) - \frac {1}{n - 1} \cdot \hat {p} _ {\mathrm{pos}} (\pmb {x}, \pmb {y}).
$$

Moreover, as $n \to \infty$ , the above result implies that $p_{\text{neg}}(\boldsymbol{x}, \boldsymbol{y}) = p_x(\boldsymbol{x}) p_y(\boldsymbol{y})$ .

Lemma C.3. Assume that the encoder $f(\cdot)$ satisfies $\|f(\boldsymbol{x})\|_{2}^{2}=1$ for all x. For any distribution p, the following holds.

$$
1 - \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \underset {\boldsymbol {v} \sim f _ {\sharp} p _ {y}} {\mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} p _ {x}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] = \frac {1}{2} \operatorname{tr} \left(\operatorname{Var} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]\right) + \frac {1}{2} \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2}.
$$

Proof. Note that

$$
\begin{array}{l} \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ] \right\| _ {2} ^ {2} - \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2} = - 4 \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} ] ^ {\top} \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {v} ] \\ = - 4 \mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} p _ {x}} [ \boldsymbol {u} ] ^ {\top} \mathbb {E} _ {\boldsymbol {v} \sim f _ {\sharp} p _ {y}} [ \boldsymbol {v} ], \tag {15} \\ \end{array}
$$

where the equality in (15) follows from the assumption of matching marginals.

From the definition of variance, we have

$$
\begin{array}{l} \left. \right. \operatorname{tr} \left(\operatorname{Var} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]\right) = \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ \operatorname{tr} \left(\left((\boldsymbol {u} - \boldsymbol {v}) - \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]\right) ((\boldsymbol {u} - \boldsymbol {v}) - \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]) ^ {\top}\right)\right] \\ = \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ \left\| (\boldsymbol {u} - \boldsymbol {v}) - \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ] \right\| _ {2} ^ {2} \right] \\ = \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ \| \boldsymbol {u} - \boldsymbol {v} \| _ {2} ^ {2} \right] - \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ] \right\| _ {2} ^ {2} \\ = \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ 2 - 2 \cdot \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] - \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2} + 4 \mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} p _ {x}} [ \boldsymbol {u} ] ^ {\top} \mathbb {E} _ {\boldsymbol {v} \sim f _ {\sharp} p _ {y}} [ \boldsymbol {v} ] \\ = 2 - 2 \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\text {pos}}} [ \boldsymbol {u} ^ {\top} \boldsymbol {v} ] - \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\text {pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2} + 4 \mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} p _ {x}} [ \boldsymbol {u} ] ^ {\top} \mathbb {E} _ {\boldsymbol {v} \sim f _ {\sharp} p _ {y}} [ \boldsymbol {v} ]. \tag {16} \\ \end{array}
$$

By rearranging (16) and dividing by 2, we have

$$
\begin{array}{l} \frac {1}{2} \operatorname{tr} \left(\operatorname{Var} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]\right) + \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2} = 1 - \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} p _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \mathbb {E} _ {\boldsymbol {u} \sim f _ {\sharp} p _ {x}} [ \boldsymbol {u} ] ^ {\top} \mathbb {E} _ {\boldsymbol {v} \sim f _ {\sharp} p _ {y}} [ \boldsymbol {v} ] \\ = 1 - \mathbb{E}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}p_{\mathrm{pos}}}\left[\boldsymbol{u}^{\top}\boldsymbol {v}\right] + \mathbb{E}_{\substack{\boldsymbol {u}\sim f_{\sharp}p_{x}\\ \boldsymbol {v}\sim f_{\sharp}p_{y}}}\left[\boldsymbol{u}^{\top}\boldsymbol {v}\right] \\ \end{array}
$$

Lemma C.4. Assume that the encoder $f(\cdot)$ satisfies $\|f(\boldsymbol{x})\|_{2}^{2}=1$ for all x. For any empirical distribution $\hat{p}$ with a sample size of n, the following holds.

$$
1 - \frac {n - 1}{n} \mathbb {E} _ {(\pmb {u}, \pmb {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \pmb {u} ^ {\top} \pmb {v} \right] + \frac {n - 1}{n} \mathbb {E} _ {(\pmb {u}, \pmb {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \pmb {u} ^ {\top} \pmb {v} \right] = \frac {1}{2} \operatorname{tr} \left(\operatorname{Var} _ {(\pmb {u}, \pmb {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} [ \pmb {u} - \pmb {v} ]\right) + \frac {1}{2} \left\| \mathbb {E} _ {(\pmb {u}, \pmb {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} [ \pmb {u} + \pmb {v} ] \right\| _ {2} ^ {2}.
$$

Proof. Using Proposition C.2, the expectation over the empirical distribution can be decomposed into the expectations of positive and negative pairs as follows.

$$
\mathop{\mathbb{E}}_{\substack{\boldsymbol {u}\sim f_{\sharp}\hat{p}_{x}\\ \boldsymbol {v}\sim f_{\sharp}\hat{p}_{y}}}\left[ \boldsymbol{u}^{\top}\boldsymbol {v}\right] = \frac{1}{n}\mathop{\mathbb{E}}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}}\left[ \boldsymbol{u}^{\top}\boldsymbol {v}\right] + \frac{n - 1}{n}\mathop{\mathbb{E}}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}\hat{p}_{\mathrm{neg}}}\left[ \boldsymbol{u}^{\top}\boldsymbol {v}\right].
$$

Applying Lemma C.3 to the empirical distribution $\hat{p}$ , we have

$$
\begin{array}{l} \frac{1}{2}\operatorname{tr}\left(\operatorname{Var}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}\hat{p}_{\text{pos}}}\left[\boldsymbol {u} - \boldsymbol {v}\right]\right) + \left\| \mathbb{E}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}\hat{p}_{\text{pos}}}\left[\boldsymbol {u} + \boldsymbol {v}\right]\right\|_{2}^{2} = 1 - \mathbb{E}_{(\boldsymbol {u},\boldsymbol {v})\sim f_{\sharp}\hat{p}_{\text{pos}}}\left[\boldsymbol{u}^{\top}\boldsymbol{v}\right] + \mathbb{E}_{\substack{\boldsymbol {u}\sim f_{\sharp}\hat{p}_{x}\\ \boldsymbol {v}\sim f_{\sharp}p_{y}}}\left[\boldsymbol{u}^{\top}\boldsymbol{v}\right] \\ = 1 - \frac {n - 1}{n} \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \frac {n - 1}{n} \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right]. \\ \end{array}
$$

Theorem C.5. Assume that the encoder $f(\cdot)$ satisfies $\| f(\boldsymbol{x})\| _2^2 = 1$ for all $\boldsymbol{x}$ . For any empirical distribution $\hat{p}$ with a sample size of $n$ , the following inequality holds.

$$
\mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] \leq 1 + \left(\mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \frac {1}{n - 1}\right),
$$

where equality holds if and only if $\mathrm{tr}\left(\mathrm{Var}_{(\pmb {u},\pmb {v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}}\left[\pmb {u} - \pmb {v}\right]\right) = 0$ and $\mathbb{E}_{\pmb {u}\sim f_{\sharp}\hat{p}_x}[\pmb {u}] + \mathbb{E}_{\pmb {v}\sim f_{\sharp}\hat{p}_y}[\pmb {v}] = \mathbf{0}$ .

Proof. From Lemma C.4, and variance and norm are non-negative, we have

$$
\begin{array}{l} \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] = \frac {n}{n - 1} + \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] \\ - \frac {n}{2 (n - 1)} \operatorname{tr} \left(\operatorname{Var} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} [ \boldsymbol {u} - \boldsymbol {v} ]\right) - \frac {n}{2 (n - 1)} \left\| \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} [ \boldsymbol {u} + \boldsymbol {v} ] \right\| _ {2} ^ {2} \\ \leq \frac {n}{n - 1} + \mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right], \tag {17} \\ \end{array}
$$

where equality in (17) holds if and only if $\operatorname{tr}\left(\operatorname{Var}_{(\boldsymbol{u},\boldsymbol{v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}[\boldsymbol{u}-\boldsymbol{v}])}=0\right.$ and $\left\|\mathbb{E}_{(\boldsymbol{u},\boldsymbol{v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}[\boldsymbol{u}+\boldsymbol{v}]]\right\|_2^2=0$ . Moreover, the condition of $\left\|\mathbb{E}_{(\boldsymbol{u},\boldsymbol{v})\sim f_{\sharp}\hat{p}_{\mathrm{pos}}[\boldsymbol{u}+\boldsymbol{v}]]\right\|_2^2=0$ is equal to $\mathbb{E}_{\boldsymbol{u}\sim f_{\sharp}\hat{p}_x[\boldsymbol{u}]+\mathbb{E}_{\boldsymbol{v}\sim f_{\sharp}\hat{p}_y[\boldsymbol{v}]=0}$ . As a result, the following holds:

$$
\mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] \leq 1 + \left(\mathbb {E} _ {(\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\mathrm{neg}}} \left[ \boldsymbol {u} ^ {\top} \boldsymbol {v} \right] + \frac {1}{n - 1}\right).
$$

# C.3. Proofs for Full-Batch CL

Lemma C.6 (Restatement of Lemma 1 in Lee et al. (2024)). Let $u_{1}, v_{1}, u_{2}, v_{2}, \cdots, u_{n}, v_{n}$ be 2n vectors, satisfying $u_{i}^{\top}u_{i} = v_{i}^{\top}v_{i} = 1$ for all $i \in [n]$ . Then, the following inequality holds.

$$
\frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} \geq \frac {n - 2}{2 n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {n}{2 (n - 1)}, \tag {18}
$$

where the equality conditions are

$$
\left\{ \begin{array}{l} \boldsymbol {u} _ {i} - \boldsymbol {v} _ {i} = \boldsymbol {c} \text {   for   all   } i \in [ n ], \text {   for   some   constant   vector   } \boldsymbol {c}, \\ \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} + \sum_ {i \in [ n ]} \boldsymbol {v} _ {i} = \boldsymbol {0}. \end{array} \right.
$$

Proof. By using Jensen's inequality, we have

$$
\begin{array}{l} \frac {1}{n} \sum_ {i \in [ n ]} \| \boldsymbol {u} _ {i} - \boldsymbol {v} _ {i} \| _ {2} ^ {2} \geq \left\| \frac {1}{n} \sum_ {i \in [ n ]} (\boldsymbol {u} _ {i} - \boldsymbol {v} _ {i}) \right\| _ {2} ^ {2} \\ = \left\| \frac {1}{n} \sum_ {i \in [ n ]} (\boldsymbol {u} _ {i} + \boldsymbol {v} _ {i}) \right\| _ {2} ^ {2} - 4 \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) \\ \geq - 4 \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right), \\ \end{array}
$$

where the equality conditions are

$$
\left\{ \begin{array}{l} \boldsymbol {u} _ {i} - \boldsymbol {v} _ {i} = \boldsymbol {c} \text {   for   all   } i \in [ n ], \text {   for   some   constant   vector   } \boldsymbol {c}, \\ \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} + \sum_ {i \in [ n ]} \boldsymbol {v} _ {i} = \boldsymbol {0}. \end{array} \right. \tag {19}
$$

From the above, it follows that

$$
\left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) \geq - \frac {1}{4 n} \sum_ {i \in [ n ]} \| \boldsymbol {u} _ {i} - \boldsymbol {v} _ {i} \| _ {2} ^ {2} \tag {20}
$$

$$
= - \frac {1}{2} + \frac {1}{2 n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}, \tag {21}
$$

where the last equality uses $\|u_{i}-v_{i}\|_{2}^{2}=2-2u_{i}^{\top}v_{i}$ for all $i\in[n]$ .

Note that the inner product of centroids is

$$
\left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) = \frac {1}{n ^ {2}} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} + \frac {1}{n ^ {2}} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}.
$$

Combining this with (21), we have

$$
\frac {1}{n ^ {2}} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} = \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) - \frac {1}{n ^ {2}} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}
$$

$$
\geq - \frac {1}{2} + \frac {1}{2 n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {1}{n ^ {2}} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}
$$

$$
= \frac {n - 2}{2 n ^ {2}} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {1}{2}.
$$

The inequality follows from $(20)$ , with equality achieved under the conditions specified in $(19)$ .

Lemma C.7. Let $u_{1}, v_{1}, u_{2}, v_{2}, \cdots, u_{n}, v_{n}$ be 2n vectors, satisfying $u_{i}^{\top}u_{i} = v_{i}^{\top}v_{i} = 1$ for all $i \in [n]$ . Then, the following inequality holds.

$$
\frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} + 2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}) \geq - \frac {2}{n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {2}{n - 1}, \tag {22}
$$

where equality holds if and only if $\sum_{i\in[n]}(\boldsymbol{u}_{i}+\boldsymbol{v}_{i})=\boldsymbol{0}$ .

Proof. Note that

$$
\begin{array}{l} \left\| \sum_ {i \in [ n ]} \left(\boldsymbol {u} _ {i} + \boldsymbol {v} _ {i}\right) \right\| _ {2} ^ {2} = \left(\sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) + \left(\sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) ^ {\top} \left(\sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) + 2 \left(\sum_ {i \in [ n ]} \boldsymbol {u} _ {i}\right) ^ {\top} \left(\sum_ {i \in [ n ]} \boldsymbol {v} _ {i}\right) \\ = n + \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + n + \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} + 2 \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} + 2 \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} \\ = 2 n + \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} + 2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) + 2 \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}. \\ \end{array}
$$

Rearranging terms, we have

$$
\frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} + 2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) = \frac {1}{n (n - 1)} \left\| \sum_ {i \in [ n ]} \left(\boldsymbol {u} _ {i} + \boldsymbol {v} _ {i}\right) \right\| _ {2} ^ {2} - \frac {2}{n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {2}{n - 1}
$$

$$
\geq - \frac {2}{n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {2}{n - 1},
$$

where the equality condition is $\sum_{i\in [n]}(\boldsymbol{u}_i + \boldsymbol{v}_i) = \mathbf{0}$ .

![](images/2d5f04de84b678ea7cca09203903c887b6d1621834ec1adf0aac5ac5fc0521c3.jpg)

Lemma C.8. Let $u_{1}, v_{1}, u_{2}, v_{2}, \cdots, u_{n}, v_{n}$ be 2n vectors, satisfying $u_{i}^{\top}u_{i} = v_{i}^{\top}v_{i} = 1$ for all $i \in [n]$ . Then, the following inequality holds.

$$
\frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j}) \geq - \frac {2}{n - 1}, \tag {23}
$$

where equality holds if and only if $\sum_{i\in[n]}u_{i}=\sum_{i\in[n]}v_{i}=0$ .

Proof. Since $u_{i}^{\top}u_{i}=v_{i}^{\top}v_{i}=1$ for all $i\in[n]$ , we have

$$
\begin{array}{l} \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) = \frac {1}{n (n - 1)} \left(\sum_ {i \in [ n ]} \sum_ {j \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} - \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {i} + \sum_ {i \in [ n ]} \sum_ {j \in [ n ]} \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} - \sum_ {i \in [ n ]} \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \\ = \frac {1}{n (n - 1)} \left(\left\| \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} \right\| _ {2} ^ {2} + \left\| \sum_ {j \in [ n ]} \boldsymbol {v} _ {j} \right\| _ {2} ^ {2}\right) - \frac {2}{n - 1} \\ \geq - \frac {2}{n - 1}, \\ \end{array}
$$

where the equality condition is $\sum_{i\in [n]}\pmb{u}_i = \sum_{i\in [n]}\pmb{v}_i = \mathbf{0}$ .

![](images/a709a685f985cf2b51db51775a67b59e5ed26d71bd914ec7a3c4f7cc2342a45b.jpg)

Theorem C.9. Suppose that $d \geq n - 1$ . Let the contrastive loss $\mathcal{L}(U, V)$ be one of the following forms.

i. $\mathcal{L}_{\mathrm{info - sym}}(U,V)$ in Def. 3.1.

ii. $\mathcal{L}_{\text{ind-add}}(\boldsymbol{U}, \boldsymbol{V})$ in Def. 3.2, where $(c_{1}, c_{2}) \in \{(0,1), (1,1)\}$ .

iii. $\mathcal{L}_{\mathrm{ind - add}}(U,V)$ in Def. 3.2, where $(c_{1},c_{2}) = (1,0)$ and $\phi^{\prime}(1) > \frac{n - 2}{2(n - 1)}\cdot \psi^{\prime}\left(-\frac{1}{n - 1}\right)$ .

Then, the embedding similarities for the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

Theorem C.10. Suppose that $d \geq n$ . Let the contrastive loss $\mathcal{L}(\mathbf{U},\mathbf{V})$ be the form of $\mathcal{L}_{\mathrm{ind - add}}(\mathbf{U},\mathbf{V})$ in Def. 3.2, where $(c_{1},c_{2}) = (1,0)$ and $\phi^{\prime}(1) < \frac{n - 2}{2(n - 1)}\cdot \psi^{\prime}\left(-\frac{1}{n - 1}\right)$ . Then, embedding similarities for the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

Proof of Theorem. C.9 and Theorem. C.10. We prove for each category of loss, $\mathcal{L}_{\mathrm{info - sym}}(U,V)$ in Def. 3.1 and $\mathcal{L}_{\mathrm{ind - add}}(U,V)$ in Def. 3.2, separately.

First, consider the case of $\mathcal{L}_{\mathrm{info - sym}}(U,V)$ in Def. 3.1. By using Jensen's inequality, we have

$$
\begin{array}{l} \mathcal {L} _ {\text { info }} (\boldsymbol {U}, \boldsymbol {V}) = \frac {1}{n} \sum_ {i \in [ n ]} \psi \left(c _ {1} \sum_ {j \in [ n ] \backslash \{i \}} \phi \left(\left(\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i}\right) + c _ {2} \sum_ {j \in [ n ] \backslash \{i \}} \phi \left(\left(\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i}\right)\right) \\ \left. \right. \geq \frac {1}{n} \sum_ {i \in [ n ]} \psi \left((c _ {1} + c _ {2}) (n - 1) \cdot \phi \left(\frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \backslash \{i \}} \left(c _ {1} \left(\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i} + c _ {2} \left(\boldsymbol {u} _ {j} - \boldsymbol {v} _ {i}\right) ^ {\top} \boldsymbol {u} _ {i}\right)\right)\right) \tag {24} \\ = \frac {1}{n} \sum_ {i \in [ n ]} \psi \left((c _ {1} + c _ {2}) (n - 1) \cdot \phi \left(\frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \setminus \{i \}} \left(c _ {1} \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {u} _ {i} + c _ {2} \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} - (c _ {1} + c _ {2}) \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {u} _ {i}\right)\right)\right), \\ \end{array}
$$

where equality in (24) holds if the argument of $\phi$ is constant for all $j\in [n]\setminus \{i\}$ .

Let define $h_{(c_{1},c_{2})}(x):=\psi((c_{1}+c_{2})(n-1)\phi(x))$ , which is a convex and increasing function. Then, we have

$$
\begin{array}{l} \mathcal {L} _ {\text { info - sym }} (\boldsymbol {U}, \boldsymbol {V}) = \frac {1}{2} \mathcal {L} _ {\text { info }} (\boldsymbol {U}, \boldsymbol {V}) + \frac {1}{2} \mathcal {L} _ {\text { info }} (\boldsymbol {V}, \boldsymbol {U}) \\ \geq \frac {1}{2 n} \sum_ {i \in [ n ]} \psi \left((c _ {1} + c _ {2}) (n - 1) \cdot \phi \left(\frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \setminus \{i \}} \left(c _ {1} \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {u} _ {i} + c _ {2} \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} - (c _ {1} + c _ {2}) \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {u} _ {i}\right)\right)\right) \\ + \frac {1}{2 n} \sum_ {i \in [ n ]} \psi \left((c _ {1} + c _ {2}) (n - 1) \cdot \phi \left(\frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \setminus \{i \}} \left(c _ {1} \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i} - (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right)\right)\right) \\ \geq \psi \left((c _ {1} + c _ {2}) (n - 1) \cdot \phi \left(\frac {1}{2 n} \sum_ {i \in [ n ]} \cdot \frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \backslash \{i \}} \left(c _ {1} \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {u} _ {i} + c _ {2} \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} - (c _ {1} + c _ {2}) \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {u} _ {i}\right) \right. \right. \\ \left. + \frac {1}{2 n} \sum_ {i \in [ n ]} \cdot \frac {1}{(c _ {1} + c _ {2}) (n - 1)} \sum_ {j \in [ n ] \backslash \{i \}} \left(c _ {1} \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i} - (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right)\right) \tag {25} \\ = h _ {(c _ {1}, c _ {2})} \left(\frac {1}{2 (c _ {1} + c _ {2}) n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \left(c _ {1} 2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} (\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i}) - 2 (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right)\right) \\ \end{array}
$$

where the inequality in (25) holds for Jensen's inequality. Equality in (25) holds if the arguments of $h_{(c_1, c_2)}$ are constant for all $i \in [n]$ . Moreover, from Jensen's inequality, we have

$$
\begin{array}{l} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \mathcal {L} _ {\mathrm{info-sym}} (\boldsymbol {U}, \boldsymbol {V}) \right] \\ \geq \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ h _ {(c _ {1}, c _ {2})} \left(\frac {1}{2 (c _ {1} + c _ {2}) n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \left(c _ {1} 2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} (\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i}) - 2 (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right)\right) \right] \\ \geq h _ {(c _ {1}, c _ {2})} \left(\frac {1}{2 (c _ {1} + c _ {2})} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \big (c _ {1} 2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} (\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i}) - 2 (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}) \right]\right). \\ \end{array}
$$

Therefore, we only have to minimize

$$
\mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(c _ {1} 2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + c _ {2} (\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i}) - 2 (c _ {1} + c _ {2}) \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \right].
$$

Now, consider each case of $(c_{1}, c_{2}) \in \{(0, 1), (1, 0), (1, 1)\}$ .

For the case of $(c_{1}, c_{2}) = (1, 1)$ , by using Lemma C.7, we have

$$
\mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} + \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i} - 4 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \right]
$$

$$
\geq \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ - \frac {2}{n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {2}{n - 1} - \frac {4}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} \right].
$$

For the case of $(c_{1}, c_{2}) = (0, 1)$ , by using Lemma C.8, we have

$$
\mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \left(\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {u} _ {i} + \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {v} _ {i} - 2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \right] \geq \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ - \frac {2}{n - 1} - \frac {2}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} \right].
$$

For the case of $(c_{1}, c_{2}) = (1, 0)$ , by using Lemma C.6, we have

$$
\begin{array}{l} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {1}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \setminus \{i \}} \left(2 \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {v} _ {i} - 2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \right] \\ \geq \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \frac {n - 2}{2 n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {n}{2 (n - 1)} - \frac {2}{n (n - 1)} \sum_ {i \in [ n ]} \sum_ {j \in [ n ] \backslash \{i \}} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} \right] \\ = - \frac {2}{n (n - 1)} + \frac {- n ^ {3} + n ^ {2} + n - 2}{2 n (n - 1)} \cdot \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} \right]. \\ \end{array}
$$

Therefore, for every cases of $(c_{1}, c_{2})$ , $u_{i}^{\top}v_{i}=1$ holds for all positive pairs $(\boldsymbol{u}_{i},\boldsymbol{v}_{i})\sim f_{\sharp}^{\star}\hat{p}_{\mathrm{pos}}^{i}$ and $i\in[n]$ . To achieve the all equality conditions, $u^{\top}v=-\frac{1}{n-1}$ must hold for all negative pairs $(\boldsymbol{u},\boldsymbol{v})\sim f_{\sharp}^{\star}\hat{p}_{\mathrm{neg}}$ . Therefore, embedding similarities for the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

Second, consider the case of $\mathcal{L}_{\mathrm{ind - add}}(U,V)$ in Def. 3.2. From Jensen's inequality, we have

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{ind-add}} (\boldsymbol {U}, \boldsymbol {V}) = - \frac {1}{n} \sum_ {i \in [ n ]} \phi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}) + \frac {c _ {1}}{n (n - 1)} \sum_ {i \neq j \in [ n ]} \psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}) + \frac {c _ {2}}{2 n (n - 1)} \sum_ {i \neq j \in [ n ]} \left(\psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j}) + \psi (\boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j})\right) \\ \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \frac {c _ {1}}{n (n - 1)} \sum_ {i \neq j \in [ n ]} \psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}) + \frac {c _ {2}}{2 n (n - 1)} \sum_ {i \neq j \in [ n ]} \left(\psi (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j}) + \psi (\boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j})\right) \\ \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(\frac {1}{(2 c _ {1} + c _ {2}) n (n - 1)} \sum_ {i \neq j \in [ n ]} (2 c _ {1} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} + c _ {2} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + c _ {2} \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j})\right). \\ \end{array}
$$

Equality conditions for both inequality is $\phi$ and $\psi$ are applied to a constant argument.

Now, consider each case of $(c_{1}, c_{2}) \in \{(0, 1), (1, 0), (1, 1)\}$ .

For the case of $(c_{1}, c_{2}) = (1, 1)$ , by using Lemma C.7, we have

$$
\begin{array}{l} \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U}, \boldsymbol {V}) \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(\frac {1}{3 n (n - 1)} \sum_ {i \neq j \in [ n ]} (2 \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j})\right) \\ \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(- \frac {2}{3 n (n - 1)} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {2}{3 (n - 1)}\right). \\ \end{array}
$$

Note that $\phi$ and $\psi$ are increasing functions. Therefore, by using the similar manner in the proof of $\mathcal{L}_{\mathrm{info - sym}}(U,V)$ case above, embedding similarities in Def. 4.1 of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

For the case of $(c_{1}, c_{2}) = (0, 1)$ , by using Lemma C.8, we have

$$
\begin{array}{l} \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U}, \boldsymbol {V}) \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(\frac {1}{n (n - 1)} \sum_ {i \neq j \in [ n ]} \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {u} _ {j} + \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right)\right) \\ \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(- \frac {2}{n - 1}\right) \\ \end{array}
$$

Note that $\phi$ and $\psi$ are increasing functions. Therefore, by using the similar manner in the case of $\mathcal{L}_{\mathrm{info - sym}}(U,V)$ in Def. 3.1, embedding similarities in Def. 4.1 of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

For the case of $(c_{1}, c_{2}) = (1, 0)$ , by using Lemma C.6, we have

$$
\begin{array}{l} \mathcal {L} _ {\text { ind - add }} (\boldsymbol {U}, \boldsymbol {V}) \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(\frac {1}{n (n - 1)} \sum_ {i \neq j \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) \\ \geq - \phi \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) + \psi \left(\frac {n - 2}{2 n (n - 1)} \sum_ {i = 1} ^ {n} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i} - \frac {n}{2 (n - 1)}\right) \\ = h \left(\frac {1}{n} \sum_ {i \in [ n ]} \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \\ \end{array}
$$

where the function $h(\cdot)$ is defined as

$$
h (x) := - \phi (x) + \psi \left(\frac {n - 2}{2 (n - 1)} \cdot x - \frac {n}{2 (n - 1)}\right).
$$

Note that both $-\phi(\cdot)$ and $\psi(\cdot)$ are differentiable and convex functions, therefore $h(\cdot)$ is also a differentiable and convex function. Therefore, to attain the minimum at x=1, $h'(1)<0$ holds. Then,

$$
0 > h ^ {\prime} (1) = - \phi^ {\prime} (1) + \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(\frac {n - 2}{2 (n - 1)} - \frac {n}{2 (n - 1)}\right) = - \phi^ {\prime} (1) + \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(- \frac {1}{n - 1}\right),
$$

which is equal to

$$
\phi^ {\prime} (1) > \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(- \frac {1}{n - 1}\right).
$$

Therefore, by using the similar manner in the case of $\mathcal{L}_{\mathrm{info-sym}}(U,V)$ in Def. 3.1, embedding similarities in Def. 4.1 of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1, \qquad \boldsymbol {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) = - \frac {1}{n - 1}.
$$

if $\phi'(1) > \frac{n-2}{2(n-1)} \cdot \psi' \left(-\frac{1}{n-1}\right)$ . On the other hand, if $\phi'(1) < \frac{n-2}{2(n-1)} \cdot \psi' \left(-\frac{1}{n-1}\right)$ , embedding similarities in Def. 4.1 of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

Moreover, the existence of the embedding for $d \geq n$ can be shown in Proposition 1 in Lee et al. (2024)

Example C.11. Consider the sigmoid contrastive loss $\mathcal{L}_{\mathrm{sig}}(\boldsymbol{U},\boldsymbol{V})$ (Zhai et al., 2023), defined as

$$
\mathcal {L} _ {\mathrm{sig}} (\boldsymbol {U}, \boldsymbol {V}) := \frac {1}{n} \sum_ {i \in [ n ]} \log \left(1 + \exp \left(- t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {i}\right) \cdot \exp (b)\right) + \frac {1}{n} \sum_ {i \neq j \in [ n ]} \log \left(1 + \exp \left(t \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right) \cdot \exp (- b)\right),
$$

where $t > 0$ and $b \in \mathbb{R}$ are hyperparameters. This loss follows the form of $\mathcal{L}_{\mathrm{ind - add}}(\mathbf{U},\mathbf{V})$ in Def. 3.2, where $(c_1,c_2) = (1,0),\phi (x) = -\log (1 + \exp (-tx + b)),$ and $\psi (x) = (n - 1)\cdot \log (1 + \exp (tx - b))$ .

If hyperparameters $t$ and $b$ are chosen such that

$$
\frac {1 + \exp \left(\frac {t}{n - 1} + b\right)}{1 + \exp (t - b)} <   \frac {n - 2}{2},
$$

embedding similarities of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

Proof. The sigmoid contrastive loss $\mathcal{L}_{\mathrm{sig}}(\boldsymbol{U},\boldsymbol{V})$ follows the loss form in Def. 3.2, see Appendix A.2. Consequently, by Theorem C.10, it suffices to verify the condition of

$$
\phi^ {\prime} (1) <   \frac {n - 2}{2 (n - 1)} \cdot \psi^ {\prime} \left(- \frac {1}{n - 1}\right), \tag {26}
$$

where $\phi(x) = -\log(1 + \exp(-tx + b))$ and $\psi(x) = (n - 1)\log(1 + \exp(tx - b))$ . Taking the derivative of $\phi(x)$ , we obtain

$$
\phi^ {\prime} (x) = \frac {t \exp (- t x + b)}{1 + \exp (- t x + b)} = \frac {t}{1 + \exp (t x - b)},
$$

and differentiating $\psi (x)$ yields

$$
\psi^ {\prime} (x) = (n - 1) \cdot \frac {t \exp (t x - b)}{1 + \exp (t x - b)}.
$$

By plugging the derivative values into (26), we get

$$
\frac {t}{1 + \exp (t - b)} <   \frac {n - 2}{2 (n - 1)} \cdot \frac {(n - 1) t \exp \left(- \frac {t}{n - 1} - b\right)}{1 + \exp \left(- \frac {t}{n - 1} - b\right)},
$$

which simplifies to

$$
{\frac {1}{1 + \exp (t - b)}} <   {\frac {n - 2}{2 (n - 1)}} \cdot {\frac {(n - 1) \exp \left(- {\frac {t}{n - 1}} - b\right)}{1 + \exp \left(- {\frac {t}{n - 1}} - b\right)}} = {\frac {n - 2}{2}} \cdot {\frac {1}{\exp \left({\frac {t}{n - 1}} + b\right) + 1}}.
$$

Therefore, if hyperparameters t and b satisfy

$$
\frac {1 + \exp \left(\frac {t}{n - 1} + b\right)}{1 + \exp (t - b)} <   \frac {n - 2}{2}, \tag {27}
$$

following from Theorem C.10, the similarities of the full-batch optimal encoder $f^{\star}$ in (1) satisfy

$$
\pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{pos}}) <   1, \qquad \pmb {s} (f ^ {\star}; \hat {p} _ {\mathrm{neg}}) <   - \frac {1}{n - 1}.
$$

# C.4. Proofs for Mini-Batch CL

Definition C.12 (Simplex ETF). A set of $n$ vectors $U$ on the $d$ -dimensional unit sphere is called $(n - 1)$ -simplex ETF, if

$$
\| \boldsymbol {u} \| _ {2} ^ {2} = 1 \text {   and   } \boldsymbol {u} ^ {\top} \boldsymbol {v} = - \frac {1}{n - 1}, \quad \forall \boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {U}.
$$

Note that $(n - 1)$ -simplex ETF exists when $d \geq n - 1$ .

Lemma C.13. Let a set of $n$ vectors $U$ be a $(n - 1)$ -simplex ETF on the $d$ -dimensional unit sphere with $d \geq n - 1$ . Then, the following holds:

$$
\sum_ {\boldsymbol {u} \in \boldsymbol {U}} \boldsymbol {u} = \boldsymbol {0}
$$

Proof. By the definition of a $(n-1)$ -simplex ETF, each vector $u \in U$ satisfies $\|u\|^{2} = 1$ and the pairwise inner product for any $u, v \in U$ with $u \neq v$ is $u^{\top}v = -\frac{1}{n-1}$ . Then,

$$
\left\| \sum_ {\boldsymbol {u} \in U} \boldsymbol {u} \right\| _ {2} ^ {2} = \left(\sum_ {\boldsymbol {u} \in U} \boldsymbol {u}\right) ^ {\top} \left(\sum_ {\boldsymbol {u} \in U} \boldsymbol {u}\right) = \sum_ {\boldsymbol {u} \in U} \boldsymbol {u} ^ {\top} \boldsymbol {u} + \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in U} \boldsymbol {u} ^ {\top} \boldsymbol {v} = n \cdot 1 + n (n - 1) \cdot \left(- \frac {1}{n - 1}\right) = 0.
$$

Since the squared norm is zero, we conclude $\sum_{u\in U} u = 0$ .

Lemma C.14. Let a set of n vectors U be a $(n-1)$ -simplex ETF, and a set of m vectors V be a $(m-1)$ -simplex ETF, where all vectors are on the d-dimensional unit sphere with $d \geq \max(n,m) - 1$ . Then, the following holds:

$$
\frac {1}{| \boldsymbol {U} \cup \boldsymbol {V} | (| \boldsymbol {U} \cup \boldsymbol {V} | - 1)} \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {U} \cup \boldsymbol {V}} \boldsymbol {u} ^ {\top} \boldsymbol {v} = - \frac {1}{n + m - 1}.
$$

Here, $U \cup V$ denotes the concatenation of the two sets, representing the full collection of all $n + m$ vectors from U and V.

Proof. From Def C.12, we have

$$
\frac {1}{n (n - 1)} \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {U}} \boldsymbol {u} ^ {\top} \boldsymbol {v} = - \frac {1}{n - 1}, \quad \frac {1}{m (m - 1)} \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {V}} \boldsymbol {u} ^ {\top} \boldsymbol {v} = - \frac {1}{m - 1}.
$$

Moreover, Lemma C.13 implies that

$$
\sum_ {\boldsymbol {u} \in U} \sum_ {\boldsymbol {v} \in V} \boldsymbol {u} ^ {\top} \boldsymbol {v} = \left(\sum_ {\boldsymbol {u} \in U} \boldsymbol {u}\right) ^ {\top} \left(\sum_ {\boldsymbol {v} \in V} \boldsymbol {v}\right) = \boldsymbol {0} ^ {\top} \boldsymbol {0} = 0.
$$

The total sum of pairwise inner products for the combined set $U \cup V$ can be written as

$$
\begin{array}{l} \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in U \cup V} \boldsymbol {u} ^ {\top} \boldsymbol {v} = \sum_ {\boldsymbol {u} \in U \cup V} \sum_ {\boldsymbol {v} \in U \cup V \setminus \{\boldsymbol {u} \}} \boldsymbol {u} ^ {\top} \boldsymbol {v} \\ = \sum_ {\boldsymbol {u} \in U} \sum_ {\boldsymbol {v} \in U \cup V \backslash \{\boldsymbol {u} \}} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \in V} \sum_ {\boldsymbol {v} \in U \cup V \backslash \{\boldsymbol {u} \}} \boldsymbol {u} ^ {\top} \boldsymbol {v} \\ = \sum_ {\boldsymbol {u} \in U} \sum_ {\boldsymbol {v} \in U \backslash \{\boldsymbol {u} \}} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \in U} \sum_ {\boldsymbol {v} \in V} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \in V} \sum_ {\boldsymbol {v} \in U} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \in V} \sum_ {\boldsymbol {v} \in V \backslash \{\boldsymbol {u} \}} \boldsymbol {u} ^ {\top} \boldsymbol {v} \\ = \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in U} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \in U, \boldsymbol {v} \in V} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {v} \in V, \boldsymbol {u} \in U} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in V} \boldsymbol {u} ^ {\top} \boldsymbol {v} \\ = \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {U}} \boldsymbol {u} ^ {\top} \boldsymbol {v} + \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {V}} \boldsymbol {u} ^ {\top} \boldsymbol {v} \tag {28} \\ = n (n - 1) \cdot \left(- \frac {1}{n - 1}\right) + m (m - 1) \cdot \left(- \frac {1}{m - 1}\right) \\ = - n - m, \\ \end{array}
$$

where the equality in (28) holds because the total pairwise inner product sums within U and within V are zero.

Therefore, we obtain

$$
\frac {1}{| \boldsymbol {U} \cup \boldsymbol {V} | (| \boldsymbol {U} \cup \boldsymbol {V} | - 1)} \sum_ {\boldsymbol {u} \neq \boldsymbol {v} \in \boldsymbol {U} \cup \boldsymbol {V}} \boldsymbol {u} ^ {\top} \boldsymbol {v} = \frac {1}{(n + m) (n + m - 1)} (- n - m) = - \frac {1}{n + m - 1}.
$$

Lemma C.15. Suppose $d \geq n - 1$ . Let V and W be $d \times n$ matrices whose columns form $(n - 1)$ -simplex ETF on the d-dimensional unit sphere. Then, there exist an orthogonal matrix $P \in R^{d \times d}$ such that V = PW.

Proof. From the definition of simplex ETF in Def. C.12, the Gram matrices satisfy

$$
\boldsymbol {V} ^ {\top} \boldsymbol {V} = \boldsymbol {W} ^ {\top} \boldsymbol {W},
$$

where each diagonal entry is 1 and each off-diagonal entry is $-\frac{1}{n-1}$ . Then, from Theorem 7.3.11 in Horn & Johnson (2012) there exist an orthogonal matrix $P \in \mathbb{R}^{d \times d}$ such that $V = PW$ .

Theorem C.16. Suppose $d \geq m - 1$ . Let the contrastive loss $\mathcal{L}(\mathbf{U}, \mathbf{V})$ be one of the forms in Theorem 5.1. Define $f_{\mathrm{batch}}^{\star}$ as the optimal encoder that minimizes the fixed mini-batch loss, given by

$$
f _ {\mathrm{batch}} ^ {\star} := \underset {f} {\arg \min} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \sum_ {k \in [ b ]} \mathcal {L} (\boldsymbol {U} _ {\boldsymbol {I} _ {k}}, \boldsymbol {V} _ {\boldsymbol {I} _ {k}}) \right],
$$

where $I_{k} := [m(k-1) + 1 : mk]$ for $k \in [b]$ .

Then, embedding similarities for the mini-batch optimal encoder $f_{batch}^{\star}$ satisfy

$$
\pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1,
$$

$$
\mathbb {E} \left[ \pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{neg}}) \right] = - \frac {1}{n - 1},
$$

$$
\operatorname{Var} \left[ \boldsymbol {s} (f _ {\text { batch }} ^ {\star}; \hat {p} _ {\text { neg }}) \right] \in \left[ \frac {n - m}{(m - 1) (n - 1) ^ {2}}, \frac {n (n - m)}{(m - 1) (n - 1) ^ {2}} \right]. \tag {29}
$$

A necessary condition for attaining the minimum variance of negative-pair similarities in (11) is $d \geq b(m - 1)$ .

Proof. Note that

$$
\mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {[ n ]}} \left[ \sum_ {k \in [ b ]} \mathcal {L} \left(\boldsymbol {U} _ {\boldsymbol {I} _ {k}}, \boldsymbol {V} _ {\boldsymbol {I} _ {k}}\right) \right] = \sum_ {k \in [ b ]} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {\boldsymbol {I} _ {k}}} \left[ \mathcal {L} \left(\boldsymbol {U} _ {\boldsymbol {I} _ {k}}, \boldsymbol {V} _ {\boldsymbol {I} _ {k}}\right) \right],
$$

by applying Theorem 5.1 to each batch, $m$ random vectors in each batch are degenerated to construct the $(m - 1)$ -simplex ETF in Def. C.12. Therefore, for $k \in [b]$ , we have:

$$
\boldsymbol {u} ^ {\top} \boldsymbol {v} = 1 \quad \forall (\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\text { pos }} ^ {\boldsymbol {I} _ {k}},
$$

$$
\boldsymbol {u} ^ {\top} \boldsymbol {v} = - \frac {1}{m - 1} \quad \forall (\boldsymbol {u}, \boldsymbol {v}) \sim f _ {\sharp} \hat {p} _ {\text { neg }} ^ {\boldsymbol {I} _ {k}}. \tag {30}
$$

In what follows, for all positive pairs, we have

$$
\pmb {u} ^ {\top} \pmb {v} = 1 \qquad \forall (\pmb {u}, \pmb {v}) \sim f _ {\sharp} ^ {\star} \hat {p} _ {\mathrm{pos}},
$$

which is equal to

$$
\pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{pos}}) = 1.
$$

For $k \in [b]$ , let $\boldsymbol{U}^{(k)}$ and $\boldsymbol{V}^{(k)}$ denote $d \times m$ random matrices, where the columns represent the vectors in the k-th batch. Additionally, define U and V as $d \times mb$ random matrices formed by concatenation of all corresponding batch matrices, i.e., $\boldsymbol{U} := [\boldsymbol{U}^{(1)}, \boldsymbol{U}^{(2)}, \cdots, \boldsymbol{U}^{(b)}]$ and $\boldsymbol{V} := [\boldsymbol{V}^{(1)}, \boldsymbol{V}^{(2)}, \cdots, \boldsymbol{V}^{(b)}]$ .

Let W be a $d \times m$ matrix whose columns form $(m - 1)$ -simplex ETF in Def. C.12. From Lemma C.15, for $k \in [b]$ , there exist orthogonal matrices $P^{(k)} \in \mathbb{R}^{d \times d}$ such that $V^{(k)} = P^{(k)}W$ . Moreover, based on the singular value decomposition, let $W = W_{1}\Sigma W_{2}^{\top}$ , where $W_{1}$ is a $d \times d$ orthogonal matrix, $W_{2}$ is an $m \times m$ orthogonal matrix, and $\Sigma$ is a $d \times m$ rectangular diagonal matrix with non-negative values of $\sigma_{1}, \sigma_{2}, \cdots, \sigma_{m}$ on the diagonal.

For all $k\in [b]$ , we have

$$
\left\| \left(\boldsymbol {U} ^ {(k)}\right) ^ {\top} \boldsymbol {V} ^ {(k)} \right\| _ {F} ^ {2} = \left\| \left(\boldsymbol {V} ^ {(k)}\right) ^ {\top} \boldsymbol {V} ^ {(k)} \right\| _ {F} ^ {2} = \left\| \boldsymbol {W} ^ {\top} \boldsymbol {W} \right\| _ {F} ^ {2} = m \cdot 1 + m (m - 1) \cdot \left(- \frac {1}{m - 1}\right) ^ {2} = \frac {m ^ {2}}{m - 1}.
$$

For all $k_{1} \neq k_{2} \in [b]$ , we have

$$
\left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} = \left\| \left(\boldsymbol {V} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} = \left\| \boldsymbol {W} ^ {\top} \left(\boldsymbol {P} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {P} ^ {(k _ {2})} \boldsymbol {W} \right\| _ {F} ^ {2} \geq 0,
$$

where the minimum value of zero is achieved if $\left(\boldsymbol{P}^{(k_1)}\right)^\top \boldsymbol{P}^{(k_2)}$ is the $d\times d$ consisting entirely of zero elements.

On the other hand,

$$
\begin{array}{l} \left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} = \left\| \left(\boldsymbol {V} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} \\ = \left\| \boldsymbol {W} ^ {\top} \left(\boldsymbol {P} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {P} ^ {(k _ {2})} \boldsymbol {W} \right\| _ {F} ^ {2} \\ = \left\| \boldsymbol {W} _ {2} \boldsymbol {\Sigma} \boldsymbol {W} _ {1} ^ {\top} \left(\boldsymbol {P} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {P} ^ {(k _ {2})} \boldsymbol {W} _ {1} \boldsymbol {\Sigma} \boldsymbol {W} _ {2} ^ {\top} \right\| _ {F} ^ {2} \\ = \left\| \boldsymbol {\Sigma} \boldsymbol {W} _ {1} ^ {\top} \left(\boldsymbol {P} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {P} ^ {(k _ {2})} \boldsymbol {W} _ {1} \boldsymbol {\Sigma} \right\| _ {F} ^ {2} \\ = \left\| \pmb {\Sigma} \pmb {P} _ {1} ^ {\top} \pmb {P} _ {2} \pmb {\Sigma} \right\| _ {F} ^ {2}, \\ \end{array}
$$

where $P_{1} := P^{(k_{1})}W_{1}$ and $P_{2} := P^{(k_{2})}W_{1}$ are $m \times m$ orthogonal matrices, and each $P_{1i}$ and $P_{2i}$ is a column vector of $P_{1}$ and $P_{2}$ , respectively. Since $P_{1}$ and $P_{2}$ are orthogonal matrices, their columns are orthonormal vectors, respectively. Then,

$$
\left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} = \left\| \boldsymbol {\Sigma} \boldsymbol {P} _ {1} ^ {\top} \boldsymbol {P} _ {2} \boldsymbol {\Sigma} \right\| _ {F} ^ {2} = \sum_ {i \in [ m ]} \sigma_ {i} ^ {2} P _ {1 i} ^ {\top} P _ {2 i} \geq \sum_ {i \in [ m ]} \sigma_ {i} ^ {2} = \| \boldsymbol {\Sigma} \boldsymbol {\Sigma} \| _ {F} ^ {2} = \left\| \boldsymbol {W} ^ {\top} \boldsymbol {W} \right\| _ {F} ^ {2} = \frac {m ^ {2}}{m - 1},
$$

where the maximum value of m is attained if $P_{1i} = P_{2i}$ for all $i \in [m]$ , i.e., $P_{1} = P_{2}$ .

From Lemma C.14, the expectation of the negative-pair similarity is

$$
\mathbb {E} \left[ \pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{neg}}) \right] = \mathbb {E} _ {(\pmb {u}, \pmb {v}) \sim f _ {\mathrm {batch\sharp}} ^ {\star} \hat {p} _ {\mathrm{neg}}} \left[ \pmb {u} ^ {\top} \pmb {v} \right] = - \frac {1}{n - 1}.
$$

Note that

$$
\begin{array}{l} \mathbb {E} \left[ \pmb {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{neg}}) ^ {2} \right] = \mathbb {E} _ {(\pmb {u}, \pmb {v}) \sim f _ {\mathrm{batch} \sharp} ^ {\star} \hat {p} _ {\mathrm{neg}}} \left[ \left(\pmb {u} ^ {\top} \pmb {v}\right) ^ {2} \right] \\ = \frac {1}{n ^ {2} - n} \left(\left\| \pmb {U} ^ {\top} \pmb {V} \right\| _ {F} ^ {2} - n\right) \\ = \frac {1}{n ^ {2} - n} \sum_ {k _ {1}, k _ {2} \in [ b ]} \left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} - \frac {1}{n - 1} \\ = \frac {1}{n ^ {2} - n} \sum_ {k _ {1} \neq k _ {2} \in [ b ]} \left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} + \frac {1}{n ^ {2} - n} \sum_ {k \in [ b ]} \left\| \left(\boldsymbol {U} ^ {(k)}\right) ^ {\top} \boldsymbol {V} ^ {(k)} \right\| _ {F} ^ {2} - \frac {1}{n - 1} \\ \end{array}
$$

$$
\begin{array}{l} = \frac {1}{n ^ {2} - n} \sum_ {k _ {1} \neq k _ {2} \in [ b ]} \left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} + \frac {1}{n ^ {2} - n} \cdot b \cdot \frac {m ^ {2}}{m - 1} - \frac {1}{n - 1} \\ = \frac {1}{n ^ {2} - n} \sum_ {k _ {1} \neq k _ {2} \in [ b ]} \left\| \left(\boldsymbol {U} ^ {(k _ {1})}\right) ^ {\top} \boldsymbol {V} ^ {(k _ {2})} \right\| _ {F} ^ {2} + \frac {1}{(m - 1) (n - 1)} \\ \in \left[ 0 + \frac {1}{(m - 1) (n - 1)}, \frac {1}{n ^ {2} - n} \cdot b (b - 1) \cdot \frac {m ^ {2}}{m - 1} + \frac {1}{(m - 1) (n - 1)} \right], \\ \end{array}
$$

where the range is equal to $\left[\frac{1}{(m - 1)(n - 1)},\frac{m(b - 1) + 1}{(m - 1)(n - 1)}\right]$ .

Therefore, the variance of the negative-pair similarity is

$$
\begin{array}{l} \operatorname{Var} \left[ \boldsymbol {s} (f _ {\text {batch}} ^ {\star}; \hat {p} _ {\text {neg}}) \right] = \mathbb {E} \left[ \boldsymbol {s} (f _ {\text {batch}} ^ {\star}; \hat {p} _ {\text {neg}}) ^ {2} \right] - \mathbb {E} \left[ \boldsymbol {s} (f _ {\text {batch}} ^ {\star}; \hat {p} _ {\text {neg}}) \right] ^ {2} \\ = \mathbb {E} \left[ \boldsymbol {s} (f _ {\mathrm{batch}} ^ {\star}; \hat {p} _ {\mathrm{neg}}) ^ {2} \right] - \frac {1}{(n - 1) ^ {2}} \\ \in \left[ \frac {n - m}{(m - 1) (n - 1) ^ {2}}, \frac {n (n - m)}{(m - 1) (n - 1) ^ {2}} \right]. \\ \end{array}
$$

![](images/0cf86b293f5bd3bcb6462c834903d556194e68836671dc23d9dd17bd8edac9a3.jpg)

Lemma C.17. Let $m_1$ and $m_2$ be natural numbers with $m_1 \leq m_2$ . For any positive values $\{c_i > 0\}_{i \in [m_2]}$ , the following inequality holds:

$$
\frac {m _ {1} \sum_ {i \in [ m _ {1} ]} c _ {i}}{m _ {2} \sum_ {i \in [ m _ {2} ]} c _ {i}} \leq 1,
$$

where the equality condition is $m_{1} = m_{2}$ .

Proof. Since $m_2 > 0$ and $m_2 - m_1 \geq 0$ , we have

$$
m _ {2} \sum_ {i \in [ m _ {2} ]} c _ {i} - m _ {1} \sum_ {i \in [ m _ {1} ]} c _ {i} = m _ {2} \sum_ {i \in [ m _ {2} ] \setminus [ m _ {1} ]} c _ {i} + (m _ {2} - m _ {1}) \sum_ {i \in [ m _ {1} ]} c _ {i} \geq 0,
$$

where the equality condition is $m_{1} = m_{2}$ . Note that $m_{1} \sum_{i \in [m_{1}]} c_{i} > 0$ . Rearranging the above yields

$$
\frac {m _ {1} \sum_ {i \in [ m _ {1} ]} c _ {i}}{m _ {2} \sum_ {i \in [ m _ {2} ]} c _ {i}} \leq 1.
$$

![](images/6200c82d2d4d3aed8cc8615edc537c88f3f602f858c29db4fdb91c3ce6633449.jpg)

Theorem C.18. Consider the InfoNCE loss $\mathcal{L}_{\mathrm{InfoNCE}}(U,V)$ (Oord et al., 2018), which corresponds to the loss $\mathcal{L}_{\mathrm{info-sym}}(U,V)$ in Def. 3.1 where $\phi(x)=\exp(x/t)$ for some t>0, $\psi(x)=\log(1+x)$ , and $(c_{1},c_{2})=(1,0)$ .

For any two integers $m_{1}, m_{2} \in [n]$ such that $m_{1} \leq m_{2}$ , the gradient of the InfoNCE loss with respect to a negative-pair similarity satisfies the following inequalities for any distinct indices $i \neq j \in [m]$ .

$$
\mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} ^ {1: m _ {1}}} \left[ - \frac {\partial}{\partial (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j})} \mathcal {L} _ {\mathrm{InfoNCE}} (\boldsymbol {U}, \boldsymbol {V}) \right] \leq \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} ^ {1: m _ {2}}} \left[ - \frac {\partial}{\partial (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j})} \mathcal {L} _ {\mathrm{InfoNCE}} (\boldsymbol {U}, \boldsymbol {V}) \right] \leq 0.
$$

Moreover, the equality condition of the first inequality is $m_{1} = m_{2}$ .

Proof. Without loss of generality, suppose there are m positive pairs in $(U, V)$ . Then, the InfoNCE loss is given as follows.

$$
\mathcal {L} (\boldsymbol {U}, \boldsymbol {V}) = \frac {1}{m} \sum_ {i \in [ m ]} \log \left(1 + \sum_ {j \in [ m ] \setminus \{i \}} \exp (\boldsymbol {u} _ {i} ^ {\top} (\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}) / t)\right) + \frac {1}{m} \sum_ {i \in [ m ]} \log \left(1 + \sum_ {j \in [ m ] \setminus \{i \}} \exp ((\boldsymbol {u} _ {j} - \boldsymbol {u} _ {i}) ^ {\top} \boldsymbol {v} _ {i} / t)\right).
$$

Following the gradients analysis in Wang & Liu (2021), the partial derivatives of the loss with respect to negative pair are derived as follows. In particular, for all $i \neq j \in [m]$ , we have

$$
\begin{array}{l} \frac {\partial}{\partial \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right)} \mathcal {L} (\boldsymbol {U}, \boldsymbol {V}) = \frac {1}{m} \cdot \frac {\partial}{\partial \left(\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j}\right)} \log \left(1 + \sum_ {j ^ {\prime} \in [ m ] \backslash \{i \}} \exp (\boldsymbol {u} _ {i} ^ {\top} (\boldsymbol {v} _ {j ^ {\prime}} - \boldsymbol {v} _ {i}) / t)\right) \\ + \frac {1}{m} \cdot \frac {\partial}{\partial (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j})} \log \left(1 + \sum_ {i ^ {\prime} \in [ m ] \backslash \{i \}} \exp ((\boldsymbol {u} _ {i ^ {\prime}} - \boldsymbol {u} _ {j}) ^ {\top} \boldsymbol {v} _ {j} / t)\right) \\ = \frac {1}{m} \cdot \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} (\boldsymbol {v} _ {j} - \boldsymbol {v} _ {i}) / t) / t}{1 + \sum_ {j ^ {\prime} \in [ m ] \backslash \{i \}} \exp (\boldsymbol {u} _ {i} ^ {\top} (\boldsymbol {v} _ {j ^ {\prime}} - \boldsymbol {v} _ {i}) / t)} + \frac {1}{m} \cdot \frac {\exp ((\boldsymbol {u} _ {i} - \boldsymbol {u} _ {j}) ^ {\top} \boldsymbol {v} _ {j} / t) / t}{1 + \sum_ {i ^ {\prime} \in [ m ] \backslash \{i \}} \exp ((\boldsymbol {u} _ {i ^ {\prime}} - \boldsymbol {u} _ {j}) ^ {\top} \boldsymbol {v} _ {j} / t)} \\ = \frac {1}{m t} \cdot \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j ^ {\prime} \in [ m ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j ^ {\prime}} / t)} + \frac {1}{m t} \cdot \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ^ {\prime} \in [ m ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)} \\ = \frac {1}{m t} \left(\frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j \in [ m ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} + \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ^ {\prime} \in [ m ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)}\right) \\ \geq 0. \\ \end{array}
$$

Then, for any $m_{1} \leq m_{2} \leq n$ and $i \neq j \in [m_{1}]$ , the following inequality holds:

$$
\begin{array}{l} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {\partial}{\partial (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j})} \mathcal {L} (\boldsymbol {U}, \boldsymbol {V}) \right] = \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {1}{m _ {2} t} \left(\frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} + \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ^ {\prime} \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)}\right) \right] \\ = \frac {1}{m _ {1} t} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} \cdot \frac {m _ {1} \sum_ {j \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{m _ {2} \sum_ {j \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} \right] \\ + \frac {1}{m _ {1} t} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ^ {\prime} \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)} \cdot \frac {m _ {1} \sum_ {i ^ {\prime} \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)}{m _ {2} \sum_ {i ^ {\prime} \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)} \right] \\ \leq \frac {1}{m _ {1} t} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} \cdot 1 \right] \\ + \frac {1}{m _ {1} t} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {2}}} \left[ \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ^ {\prime} \in [ m _ {2} ]} \exp (\boldsymbol {u} _ {i ^ {\prime}} ^ {\top} \boldsymbol {v} _ {i} / t)} \cdot 1 \right] (31) \\ = \frac {1}{m _ {1} t} \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\text {pos}} ^ {m _ {1}}} \left[ \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {j \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)} + \frac {\exp (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j} / t)}{\sum_ {i ' \in [ m _ {1} ]} \exp (\boldsymbol {u} _ {i '} ^ {\top} \boldsymbol {v} _ {i} / t)} \right] (32) \\ = \mathbb {E} _ {(\boldsymbol {U}, \boldsymbol {V}) \sim f _ {\sharp} \hat {p} _ {\mathrm{pos}} ^ {m _ {1}}} \left[ \frac {\partial}{\partial (\boldsymbol {u} _ {i} ^ {\top} \boldsymbol {v} _ {j})} \mathcal {L} (\boldsymbol {U}, \boldsymbol {V}) \right], \\ \end{array}
$$

where the inequality in (31) follows from Lemma C.17, since the exponential function is strictly positive. The equality condition in (31) is $m_1 = m_2$ .

Moreover, equality in (32), where $f_{\sharp}\hat{p}_{pos}^{m_{2}}$ is replaced with $f_{\sharp}\hat{p}_{pos}^{m_{1}}$ , holds because the expectation involves only embeddings $(\boldsymbol{U},\boldsymbol{V})$ with indices in $[m_{1}]$ . This concludes the proof. □

# D. Experiment Details

In this section, we provide the details of the experiment setup mentioned in Sec. 6. Our implementation is based on the open-source library solo-learn (da Costa et al., 2022) for self-supervised learning. The source code is available at https://github.com/leechungpa/embedding-similarity-cl/.

# D.1. Architecture and Training Details

For all experiments, we use modified ResNet-18 (He et al., 2016; Chen et al., 2020) as the backbone for CIFAR datasets and ResNet-50 for ImageNet-100. For CIFAR datasets, we modify ResNet-18 by replacing the first convolutional layer with a 3×3 kernel at a stride of 1 and removing the initial max pooling step. In contrast, we use the standard ResNet-50 architecture for ImageNet-100 without any modifications. Regardless of the backbone used, we attach a 2-layer MLP as the projection head, which projects representations to a 128-dimensional latent space. Batch normalization is applied to the fully connected layers, with the hidden layer dimension set to 2048 for ImageNet-100 and 512 for the CIFAR datasets.

We follow the data augmentation strategy used in SimCLR (Chen et al., 2020). Specifically, we apply random resized cropping, horizontal flipping, color jittering, and Gaussian blurring. For the CIFAR datasets, the crop size is set to 32, while for ImageNet-100, we use a crop size of 224. These augmentations are applied consistently across all experiments.

For the optimizer, we use stochastic gradient descent (SGD) for 200 epochs. The learning rate is scaled linearly with the batch size as lr × BatchSize/256, where the base learning rate is set to 0.3 for the CIFAR datasets and 0.1 for ImageNet-100. A cosine decay schedule is applied, with a weight decay of 0.0001 and SGD momentum set to 0.9. Additionally, we use linear warmup for the first 10 epochs.

We tune the temperature parameter for baseline methods, SimCLR, DCL, and DHEL, by performing a grid search over the range of 0.1 to 0.5 in increments of 0.1 and selecting the temperature value that yielded the best performance for each method. For tuning the proposed loss $\mathcal{L}_{\mathrm{VRNS}}(\mathbf{U},\mathbf{V})$ in Def. 5.7, we conducted a grid search for $\lambda$ from the set $\{0.1, 0.3, 1, 3, 10, 30, 100\}$ .

All experiments were conducted using a single NVIDIA RTX 4090 GPU.

# D.2. Evaluation Details

For the linear evaluation protocol, we remove the projector head and using the pretrained encoder for downstream classification tasks. Specifically, we extract the encoder outputs from the trained model without applying any augmentations. These feature vectors are then normalized and used to train a linear classifier. Following prior works (Kornblith et al., 2019; Lee et al., 2021; Koromilas et al., 2024), we report top 1 accuracy on the downstream dataset. We use SGD, setting the learning rate to 0.1 without weight decay. The classifier is trained for 200 epochs with a batch size of 256.

# E. Additional Experiments on the ImageNet Dataset

Table 4. Effect of our proposed loss (when combined with SimCLR) on the top-1 and top-5 classification accuracies (%). Bold entries indicate the highest accuracy. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Temperature</td><td colspan="2">SimCLR</td><td colspan="2">SimCLR + Ours</td></tr><tr><td>Top 1</td><td>Top 5</td><td>Top 1</td><td>Top 5</td></tr><tr><td rowspan="5">ImageNet-100</td><td>t = 0.1</td><td>70.14</td><td>91.14</td><td>69.50</td><td>90.80</td></tr><tr><td>t = 0.2</td><td>73.80</td><td>93.14</td><td>73.94</td><td>93.08</td></tr><tr><td>t = 0.3</td><td>72.30</td><td>92.94</td><td>74.04</td><td>93.24</td></tr><tr><td>t = 0.4</td><td>69.92</td><td>92.10</td><td>73.12</td><td>92.98</td></tr><tr><td>t = 0.5</td><td>68.60</td><td>90.90</td><td>72.88</td><td>92.84</td></tr></table>

We further validate the effectiveness of the proposed loss through experiments on the ImageNet dataset. Table 4 reports the top-1 and top-5 classification accuracies on ImageNet-100 for various temperature values. For each temperature, we compare SimCLR with and without the proposed loss. The results demonstrate that incorporating our loss generally improves

Table 5. Top-1 accuracy (%) on the full ImageNet dataset. 

<table><tr><td>Method</td><td>Top-1 accuracy (%)</td></tr><tr><td>SimCLR</td><td>51.79</td></tr><tr><td>SimCLR + Ours</td><td>52.48</td></tr></table>

performance across a range of temperatures, with the largest gain observed at t = 0.3.

We also report results from 100-epoch training on the full ImageNet dataset. Using the same experimental setup as for ImageNet-100, we compare SimCLR with our method (SimCLR + the proposed loss) using t = 0.2 and $\lambda = 40$ . As shown in Table 5, our method outperforms the baseline.

# F. Discussion on the Proposed Loss for Variance Reduction

The auxiliary loss $\mathcal{L}_{\mathrm{VRNS}}(\mathbf{U},\mathbf{V})$ proposed in Def. 5.7 shows effectiveness in the following scenarios:

- Small Batch Training: As established in Theorem 5.5, the variance of negative-pair similarities increases as batch size decreases. The proposed loss in Def. 5.7 directly penalizes this variance, making it particularly advantageous when training with small batch sizes. This effect is empirically validated in Figure 4.   
- Temperature Robustness: The performance of CL methods is often sensitive to the choice of the temperature parameter, which affects the distribution of similarities among embedding pairs (Wang & Liu, 2021). By explicitly encouraging negative-pair similarities towards the optimal value of $-1/(n - 1)$ , the proposed loss reduces this sensitivity and stabilizes performance across a wide range of temperature settings, as shown in Figure 3.

Despite its advantages, the proposed loss also presents several limitations:

- Suppression of Semantically Meaningful Variance: In some cases, variance in negative-pair similarities may capture meaningful semantic differences between instances. Enforcing uniform similarity can potentially suppress this informative structure, adversely affecting representation quality.   
- Reduced Impact with Large Batch Sizes: The variance-reducing effect of proposed loss diminishes as batch size increases, since the variance of negative-pair similarities naturally decreases in larger batches.   
- Hyperparameter Sensitivity: The proposed loss introduces an additional hyperparameter, $\lambda$ , which necessitates careful tuning to achieve optimal performance.