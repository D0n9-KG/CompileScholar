# Bidirectional Logits Tree: Pursuing Granularity Reconcilement in Fine-Grained Classification

Zhiguang Lu $^{1,2}$ , Qianqian Xu $^{1*}$ , Shilong Bao $^{2}$ , Zhiyong Yang $^{2}$ , Qingming Huang $^{1,2,3*}$

$^{1}$ Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences $^{2}$ School of Computer Science and Technology, University of Chinese Academy of Sciences $^{3}$ Key Laboratory of Big Data Mining and Knowledge Management, University of Chinese Academy of Sciences  
{luzhiguang23s, xuqianqian}@ict.ac.cn, {baoshilong, yangzhiyong21, qmhuang}@ucas.ac.cn

# Abstract

This paper addresses the challenge of Granularity Competition in fine-grained classification tasks, which arises due to the semantic gap between multi-granularity labels. Existing approaches typically develop independent hierarchy-aware models based on shared features extracted from a common base encoder. However, because coarse-grained levels are inherently easier to learn than finer ones, the base encoder tends to prioritize coarse feature abstractions, which impedes the learning of fine-grained features. To overcome this challenge, we propose a novel framework called the Bidirectional Logits Tree (BiLT) for Granularity Reconciliation. The key idea is to develop classifiers sequentially from the finest to the coarsest granularities, rather than parallelly constructing a set of classifiers based on the same input features. In this setup, the outputs of finer-grained classifiers serve as inputs for coarser-grained ones, facilitating the flow of hierarchical semantic information across different granularities. On top of this, we further introduce an Adaptive Intra-Granularity Difference Learning (AIGDL) approach to uncover subtle semantic differences between classes within the same granularity. Extensive experiments demonstrate the effectiveness of our proposed method.

Code — https://github.com/ZhiguangLuu/BiLT

# Introduction

Fine-grained visual classification (FGVC) has been a longstanding and challenging research focus in the deep learning community (Xu et al. 2023; Jiang et al. 2024; Pu et al. 2024). Unlike traditional image classification tasks, FGVC demands models to capture subtle distinctions between categories for accurate predictions. Recently, several studies (Zhang et al. 2024; Chen et al. 2022; Wang et al. 2021) have recognized that the hierarchical label structure (HLS) inherent in class names (Deng et al. 2009; Krizhevsky, Hinton et al. 2009; Zhao, Li, and Xing 2011; Maji et al. 2013) can significantly enhance the understanding and performance of FGVC tasks. Consequently, substantial efforts have been made to explore methods for effectively integrating hierarchical information into FGVC tasks (Jain, Karthik, and Gandhi 2024; Wang et al. 2023). Generally, once hierarchical semantic information is incorporated, the primary goal is to ensure that the model's predictions consistently align with the inherent hierarchical structure of the data. In other words, these methods aim to make precise predictions or at least keep the results as close as possible to semantically similar categories even when errors occur, thereby minimizing the severity of mistake (Bertinetto et al. 2020). For example, suppose a model misclassifies a “Husky”. In that case, it is more acceptable within the hierarchy for the output to be another dog breed, such as a “Corgi”, rather than an unrelated category like a “Siamese Cat”.

In pursuit of this goal, one effective paradigm (Silla and Freitas 2011; Zhang and Zhou 2013) usually adopts a common shared encoder to extract the semantic features of images, and then constructs hierarchy-aware models independently for the classification tasks at different levels. Typically, (Bertinetto et al. 2020) introduces a multi-task learning method that employs hierarchical cross-entropy loss and a soft label technique to train these hierarchy-aware models. On top of this, (Karthik et al. 2021) develops a post hoc approach using Conditional Risk Minimization (CRM) to calibrate likelihoods during the test phase. Besides, borrowing the idea from neural collapse (Papyan, Han, and Donoho 2020), (Liang and Davis 2023) utilizes an Equiangular Tight Frame to achieve feature alignment across all classes.

Despite significant progress, a critical challenge remains: the current FGVC paradigm struggles with the issue of Granularity competition in feature learning due to the semantic gap between multi-granularity labels (as shown in Fig.2). Specifically, coarse labels are generally much easier to distinguish than finer ones. As a result, if we pay equal attention to all levels, the learning process of the underlying encoder is likely to be dominated by the coarse branches, leading to insufficient learning of detailed information at finer granularities. To address this issue, (Chang et al. 2021) disentangles coarse-level features from fine-grained ones via level-specific classification heads, and also uses the finer-grained features to enhance the performance of coarser-grained predictions. (Garg, Sani, and Anand 2022) proposes a soft balance strategy to align the learning process of multi-granularity classifications. However, these methods primarily emphasize the negative impacts of coarse-grained learning on finer details, while overlooking the crucial fact

that coarse-grained information can also enhance fine-grained learning. Consequently, the issue of granularity competition remains inadequately considered.

In this paper, we propose a generic framework called Bidirectional Logits Tree (BiLT), designed to fully utilize hierarchical label information. Specifically, BiLT builds hierarchy-aware models in a sequential manner rather than a parallel one, where the inputs to the coarse classifier depend on the outputs of the preceding finer-grained model. This paradigm improves the priority of learning finer features and also facilitates semantic information flow among multi-granularity labels during training. Meanwhile, classification errors at coarser levels could also serve as an auxiliary supervision signal for upstream finer models' updates, reconciling the granularity competition issue end-to-end. Last but not least, we recognize that accurately identifying subtle semantic differences between sub-classes within the same granularity is also crucial for achieving promising performance. To this end, an Adaptive Intra-Granularity Difference Learning (AIGDL) approach is developed to serve our purpose better.

In summary, the contributions of this paper are as follows:

- We propose a novel paradigm called BiLT to construct hierarchy-aware predictors for fine-grained classifications, which can effectively alleviate the granularity competition issue by facilitating the flow of hierarchical semantic information across all granularities.   
- An Adaptive Intra-Granularity Difference Learning (AIGDL) method is further developed to serve our strategy better. It empowers BiLT to learn the fine-grained semantic differences of classes within the same granularity, boosting the final performance sharply.   
- Empirical studies over three widely used benchmark datasets consistently demonstrate the effectiveness of our proposed approach.

# Related Work

Hierarchical Architecture Methods Hierarchical architecture methods aim to design a hierarchy-aware model architecture that leverages the hierarchical relationship. (Silla and Freitas 2011) first introduced hierarchy-aware classifiers that incorporated a well-defined hierarchy, empirically outperforming flat classifiers across various application scenarios. Building on this idea, (Wu et al. 2016; Bertinetto et al. 2020) proposed connecting classifiers at all levels to a shared feature vector, framing the overall optimization as a multi-task learning problem. However, (Chang et al. 2021) later highlighted a potential issue with this method that sharing the same feature vector across multiple granularities can cause granularity competition problems, where coarse-level predictions impede fine-grained feature learning, whereas fine-grained feature learning can facilitate coarse-grained classification. To avoid this issue, (Liang and Davis 2023) from the perspective of neural collapse, aligned features with class distances by fixing classifier weights to a pre-computed Equiangular Tight Frame based on the distance matrix. However, this method is limited by the requirement that the feature dimension must equal the number of classes, making it difficult to apply to large datasets and certain downstream tasks. More recently, (Jain, Karthik, and Gandhi 2024) proposed a multi-granularity ensemble method, training multiple neural networks at different levels separately, and adjusting fine-grained outputs based on coarse-grained outputs during testing. This approach improved Top-1 Accuracy and reduced Mistake Severity in fine-grained classification. However, this approach requires training multiple networks separately, which is computationally expensive and difficult to scale to datasets with many levels.

Hierarchical Loss/Cost Methods Hierarchical loss methods aim to design loss functions that can effectively leverage the hierarchical relationships between labels. (Bertinetto et al. 2020) proposed a hierarchical cross-entropy (HXE) loss, a conditional probabilistic loss that conditions class probabilities on those of their ancestors. This approach jointly optimizes the HXE loss across all granularities, framing it as a multi-task learning problem. In contrast, (Karthik et al. 2021) introduced a cost-sensitive method based on conditional risk minimization, calibrating outputs according to the class-relationship matrix during the test phase to minimize the risk. As noted earlier, using hard labels for coarse-grained classification can impede fine-grained feature learning. To address this, (Garg, Sani, and Anand 2022) proposed summing the soft labels of subclasses and aligning them with coarse-grained hard labels using Jensen-Shannon divergence, while also aligning subclass features with those of their superclass using a geometric consistency loss.

Label Embedding Methods This approach focuses on uncovering relationships in a unified label semantic space and therefore describes differences between labels by the distance in this space. For instance, (Frome et al. 2013; Xian et al. 2016) started early attempts to explore a generic algorithm for learning a joint embedding space for images and labels simultaneously. Inspired by the strengths of hyperbolic space in modeling hierarchical structures, (Liu et al. 2020) proposed learning a hyperbolic label space for fine-grained classifications, leading to promising performance. In addition, to learn a favorable space, (Bertinetto et al. 2020) first initialized relationships between labels based on their lowest common ancestor (LCA) height, and proposed a soft label method for precise fine-tuning. Taking a step further, (Zhang et al. 2021) developed an online label smoothing strategy to adaptively learn the label embedding during training, while (Collins, Bhatt, and Weller 2022) proposed a crowdsourced soft label method for individual images. Additionally, label smoothing plays a significant role in various tasks, including but not limited to knowledge distillation(Yuan et al. 2020; Park et al. 2023; Han et al. 2024), image generation(Zhang et al. 2023), weakly supervised learning(Gong, Bisht, and Xu 2024; Wei et al. 2022), and trustworthy machine learning(Qin et al. 2021). Overall, label smoothing is a widely used technique to enhance model generalizability and performance(Müller, Kornblith, and Hinton 2019). Nevertheless, these methods often fail to utilize prior knowledge of inter-label distances. They heavily rely on predefined class relationships, or even solely depend on heuristic strategies for label embedding adjustment, which results

in suboptimal performance and slow convergence.

# Problem Definition

Let X and Y be the input space and label space, respectively. Under the context of fine-grained visual classifications (FGVC) (Liu et al. 2024; Wang et al. 2024; Du et al. 2021), the label space Y could be formulated as a hierarchy label tree with $H + 1$ levels, where each node corresponds to a specific class, and each edge contains high-level semantic relationships between classes. Typically, for $h \in \{0, 1, 2, \ldots, H\}$ , the root node (h = 0) represents the superclass of all classes, and the label becomes finer and finer as the level h increases in the tree. Subsequently, let $C^{h}$ be the number of classes at the h-th level, the label space could be expressed as $Y = \bigcup_{h=0}^{H} Y_{h}$ , where $Y_{h} = \{0, 1, \ldots, C^{h} - 1\}$ and we have $C^{0} = 1 < C^{1} \leq C^{2} \leq \cdots \leq C^{H}$ . Finally, suppose that there are N training samples, denoted as $\mathcal{D} = \{(\mathbf{x}_{i}, \{y_{i}^{h}\}_{h=1}^{H})\}_{i=1}^{N}$ , where $x_{i} \in X$ is an input image and $y_{i}^{h} \in Y_{h}$ corresponds to its ground-truth label at the h-th level. The primary concern of this paper is to train a well-performed classifier that can accurately predict the fine-grained class for each image while ensuring that the predictions are close to the ground truth when it makes mistake.

# Methodology

In this section, we will first reveal the fundamental limitations of current fine-grained classification methods, i.e., the granularity competition issue. To address this, we propose a generic framework called Bidirectional Logits Tree (BiLT). Fig. 1 presents the overall pipeline of our proposed BiLT. On top of this, a novel Adaptive Intra-Granularity Difference Learning is developed to exploit the fine-grained semantics among all classes sufficiently. Further details on BiLT will be discussed in the following.

# Granularity Competition Impedes Fine-grained Learning

We start with our discussions by reviewing the traditional FGVC paradigms (Wang et al. 2015) using the hierarchy label tree. Briefly speaking, given an image x with its label sets $\{y^{h}\}_{h=1}^{H}$ , conventional FGVC methods using the hierarchy label tree first adopt a base encoder $\Phi$ to extract the semantic features of each sample, i.e., $\Phi(x)$ . On this basis, an independent classifier (denoted as $f^{h}$ ) will be constructed for each label level h whose goal is to make an accurate prediction $\hat{y}^{h}$ as much as possible at its own level. In order to learn these classifiers effectively, current studies (Wang et al. 2024; Garg, Sani, and Anand 2022; Chang et al. 2021) usually consider the following multi-task optimization problem:

$$
\min _ {\Phi , f ^ {1}, \dots f ^ {H}} \mathbb {E} _ {\mathcal {D}} \left[ \sum_ {h = 1} ^ {H} \lambda_ {h} \cdot \mathcal {L} ^ {h} (\mathcal {Y} ^ {h}, f ^ {h} (\Phi (x))) \right], \tag {1}
$$

where $\lambda_{h}$ is a tunable parameter, and $L^{h}$ represents the classification error (such as cross-entropy loss) at h-th level in the hierarchy label tree. Fig. 3 presents a toy example $(H = 3)$ to instantiate the mainstream learning paradigm of FGVC.

Despite great success, in this paper, we argue that current FGVC paradigms using the hierarchy label tree, suffer from Granularity competition issues, leading to limited performance. According to Eq.1, it is apparent that coarse-grained levels are inherently simpler to learn than fine-grains, so the learning process of the base encoder $\Phi$ will be almost dominated by the shallow level (i.e., coarse labels). Fig. 2 examines this phenomenon on the real-world benchmark dataset FGVC-Aircraft, showing that methods sharing the same features at fine-grained levels converge more slowly and yield lower accuracy compared to only fine-grained level trained exclusively with standard cross-entropy loss (Only CE). As a result, the unique features extracted by $\Phi$ inevitably collapse, where the detailed information for fine-grained feature learns insufficiently (Chang et al. 2021). In addition, current paradigms have not fully explored the semantic relationships between classes at different granularities, which further impedes learning fine-grained features for accurate predictions.

# Bidirectional Logits Tree for Granularity Reconcilement

According to the above discussions, the fundamental limitations of previous FGVC literature (Bertinetto et al. 2020; Chang et al. 2021; Garg, Sani, and Anand 2022) come from the fact that all classifiers at different granularities built on unique base encoders. We propose a novel generic framework called Bidirectional Logits Tree (BiLT) to address this. The principle of BiLT is to develop coarse-level classifiers on top of its previous finer classifier outputs instead of constructing a set of classifiers separately, as shown in Fig. 1.

Specifically, let $\phi_{i} := \Phi(x_{i})$ be the feature vector for simplicity. BiLT first constructs the finest-level classifier $f^{H}$ , which directly takes $\phi_{i}$ as the input, i.e.,

$$
\mathbf {z} _ {i} ^ {H} := f ^ {H} (\phi_ {i}),
$$

where $z_{i}^{H} \in R^{C_{H}}$ is the output logits at level H, and $z_{ij}^{H}, j \in [C_{H}]$ indicates the probability of image $x_{i}$ belonging to the class j. Subsequently, for the class-level $h \in \{1, \ldots, H-1\}$ , its corresponding classifier $f^{h}$ is developed following the previous h-1 outputs:

$$
\mathbf {z} _ {i} ^ {h} := f ^ {h} (g ^ {h} (\mathbf {z} _ {i} ^ {h + 1})),
$$

where $z_{i}^{h} \in R^{C^{h}}$ and $g^{h}$ is a trainable transformation layer to explore nonlinear relationships between different levels. In this paper, $g^{h}$ is implemented by a sequential module, including batch normalization, linear transformation, batch normalization, and ELU activation (Liang and Davis 2023).

We still adopt a similar manner as Eq.1 to learn BiLT. Here the loss for each sample $x_{i}$ at h-th level is defined as:

$$
\mathcal {L} _ {\mathrm{BiLT}} ^ {h} (\mathcal {Y} ^ {h}, f ^ {h}) = \mathbf {y} _ {i} ^ {h} \log \left(\operatorname{softmax} \left(\mathbf {z} _ {i} ^ {h}\right)\right),
$$

where $\mathbf{y}_i^h$ is a one-hot encoding with the $y_{i}^{h}$ -th indices as 1.

Intuitively, the learning process of our proposed BiLT boosts a bidirectional semantic information flow across all

![](images/c601d6a844ba70ff185b920070ca71526addb6cc5b65b9157a8c0c225d90177c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Backbone"] --> B["Feature Extraction"]
    B --> C["Bidirectional Logits Tree"]
    C --> D["Label Space"]
    D --> E["Adaptive Intra-Granularity Difference Learning"]
    subgraph Feature Extraction
        F["D̃³"]
        G["D̃²"]
        H["D̃¹"]
        I["z¹"]
        J["z²"]
        K["z³"]
        L["f³"]
        M["f²"]
        N["f¹"]
        O["f³"]
        P["f²"]
        Q["f³"]
        R["φᵢ"]
    end

    subgraph Label Space
        S["y¹"]
        T["y²"]
        U["y³"]
    end

    F --> I
    G --> J
    H --> K
    I --> L
    J --> M
    K --> N
    L --> O
    M --> P
    N --> Q
    O --> R
    S --> T
    T --> U
    U --> V
    V --> W
    W --> X
    X --> Y
    Y --> Z
    Z --> R
```
</details>

Figure 1: The overall framework of our method. In the forward phase of the Bidirectional Logits Tree (BiLT), coarse-grained logits are derived from fine-grained counterparts, while the gradients from coarse-grained classifiers influence fine-grained classifiers and feature learning in the backward phase. Simultaneously, Adaptive Intra-Granularity Difference Learning (AIGDL) adjusts the output of BiLT and supervision by learning differences between categories within the same granularity.

![](images/2998fa78d29ca9993d06912a2fca3744a51b5f571bf25059c7acae7b6cd8254f.jpg)

<details>
<summary>line</summary>

| Epochs | Coarsest Level | Finest Level | Only CE |
| ------ | -------------- | ------------ | ------- |
| 0      | 0              | 0            | 0       |
| 5      | 60             | 50           | 65      |
| 10     | 70             | 60           | 75      |
| 15     | 75             | 65           | 78      |
| 20     | 78             | 70           | 80      |
| 25     | 80             | 72           | 82      |
| 30     | 82             | 75           | 83      |
| 35     | 83             | 76           | 84      |
| 40     | 84             | 77           | 85      |
</details>

(a) Sharing Same Features

![](images/8610cc4f463962914b938b75b6053b92f86f31638e8398164e5727e3120ac131.jpg)

<details>
<summary>line</summary>

| Epochs | Coarsest Level | Finest Level | Only CE |
| ------ | -------------- | ------------ | ------- |
| 0      | 0              | 0            | 0       |
| 10     | 75             | 70           | 65      |
| 20     | 80             | 75           | 70      |
| 30     | 85             | 78           | 72      |
| 40     | 90             | 80           | 75      |
</details>

(b) BiLT

Figure 2: Comparison of convergence speed of Sharing Same Features and BiLT   
![](images/b8b5e81c0c9a78cf8bfc9d42a57b91a118f06e7ab149f1bf0f4eb3aaca85e3ae.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Feature Space φ"] --> B["Classifiers"]
    B --> C["L¹(Y¹, f¹(Φ(x)))"]
    B --> D["L²(Y², f²(Φ(x)))"]
    B --> E["L³(Y³, f³(Φ(x)))"]
    B --> F["Classifiers"]
```
</details>

Figure 3: An illustrative example demonstrates the granularity competition problem, where the model prioritizes coarse-grained learning at Level 1 and Level 2, thereby rendering the fine-grained features at Level 3 difficult to distinguish.

granularities properly, as shown in Fig. 1. First of all, during the forward propagation, the finer outputs serve as the input features for the coarse-level model recursively, enabling cross-level information explorations. This to some extent promotes the learning priority of finer features in the base encoder $\Phi$ such that more discriminative features

![](images/ff7db02de251af5c4c175373aca902af96e64e6ff921615eae9a8c22d08e21cb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Vehicle"] --> B["Equal"]
    A --> C["Similar"]
    D["Vehicle"] --> E["Equal"]
    D --> F["Similar"]
    G["Vehicle"] --> H["Equal"]
    G --> I["Similar"]
    J["Vehicle"] --> K["Equal"]
    J --> L["Similar"]
    M["Vehicle"] --> N["Equal"]
    M --> O["Similar"]
    P["Vehicle"] --> Q["Equal"]
    P --> R["Similar"]
    S["Vehicle"] --> T["Equal"]
    S --> U["Similar"]
    V["Vehicle"] --> W["Equal"]
    V --> X["Similar"]
    Y["Vehicle"] --> Z["Equal"]
    Y --> AA["Similar"]
    AB["Vehicle"] --> AC["Equal"]
    AB --> AD["Similar"]
    AE["Vehicle"] --> AF["Equal"]
    AE --> AG["Similar"]
    AH["Vehicle"] --> AI["Equal"]
    AH --> AJ["Similar"]
    AK["Vehicle"] --> AL["Equal"]
    AK --> AM["Similar"]
    AN["Vehicle"] --> AO["Equal"]
    AN --> AP["Similar"]
    AQ["Vehicle"] --> AR["Equal"]
    AQ --> AS["Similar"]
    AT["Vehicle"] --> AU["Equal"]
    AT --> AV["Similar"]
    AW["Vehicle"] --> AX["Equal"]
    AW --> AY["Similar"]
    AZ["Vehicle"] --> BA["Equal"]
    AZ --> BB["Similar"]
    BC["Vehicle"] --> BD["Equal"]
    BC --> BE["Similar"]
    BF["Vehicle"] --> BG["Equal"]
    BF --> BH["Similar"]
    BI["Vehicle"] --> BJ["Equal"]
    BI --> BK["Similar"]
    BL["Vehicle"] --> BM["Equal"]
    BL --> BN["Similar"]
    BO["Vehicle"] --> BP["Equal"]
    BO --> BQ["Similar"]
    BR["Vehicle"] --> BS["Equal"]
    BR --> BT["Similar"]
    BU["Vehicle"] --> BV["Equal"]
    BU --> BW["Similar"]
```
</details>

Figure 4: Predefined label trees struggle to articulate differences amongst classes at the same hierarchical level, and our method aims to learn disparities among classes and apply relevant corrections accordingly.

will be abstracted for finer classifications. Meanwhile, during the backward propagation, each classifier is not only optimized by the classification risk at the current level but also supervised by those coarser levels. In this way, classifiers at different granularities will interact with each other, leverage hierarchical information effectively, and thus reconcile the granularity competition problem.

# Adaptive Intra-Granularity Difference Learning

So far, we have explored how to excavate fine-grained class relationships effectively at different levels. However, mining the semantic information between labels involved at the same granularity is also essential for promising performance. Current studies (Garg, Sani, and Anand 2022; Liang and Davis 2023) generally introduce a cost matrix defined on LCA (Least Common Ancestor) to calibrate the final predictions. Based on the fact that there are often obscure semantic differences between sub-classes at the same granularity, such a strategy considering intra-granularity classes

equally might struggle to articulate them. Fig. 4 provides an example to illustrate it. We know that both “bicycles”, “motorcycles”, “trucks” and “car” belong to “Vehicle”. Yet, in the feature space, it is obvious that the distance between “bicycles” and “motorcycles” should be closer than “bicycles” and “trucks” since their appearance is highly similar. Apparently, this cannot be achieved through simple LCA-based methods.

We propose a novel Adaptive Intra-Granularity Difference Learning (AIGDL) method to remedy this. Concretely, let $\mathbf{p}_{i}^{h} = \text{softmax}(\mathbf{z}_{i}^{h})$ be the prediction probability and $D^{h} \in R^{C^{h} \times C^{h}}$ be the intra-granularity class distance matrix at level h, where $D_{ij}^{h}$ represents the LCA distance between class $y_{i}^{h}$ and class $y_{j}^{h}$ . We first follow the widely adopted Condition Risk Minimization (CRM) paradigm (Garg, Sani, and Anand 2022) to calibrate:

$$
\underset {k} {\operatorname{argmin}} R (\mathbf {y} _ {i} ^ {h} = k | \mathbf {x} _ {i}) = \underset {k} {\operatorname{argmin}} \sum_ {j = 1} ^ {C ^ {h}} \mathbf {D} _ {k, j} ^ {h} \cdot p ^ {h} (\mathbf {y} _ {i} ^ {h} = k | \mathbf {x} _ {i}).
$$

Compared with the typical likelihood maximization, this CRM adjustment guarantees us to select the Bayes optimal prediction $R(\mathbf{y}_{i}^{h}=k|\mathbf{x}_{i})$ resulting in the lowest possible overall cost (Duda and Hart 1974).

On top of $D^{h}$ , a learnable intra-granularity distance matrix $\Delta^{h} \in R^{C^{h} \times C^{h}}$ is further introduced to capture the nuanced semantic relationships between classes. Thereafter, the decision rule is augmented as follows:

$$
\begin{array}{l} \underset {k} {\operatorname{argmin}} R (\mathbf {y} _ {i} ^ {h} | \mathbf {x} _ {i}) \\ = \underset {k} {\operatorname{argmin}} \sum_ {j = 1} ^ {C ^ {h}} (D _ {k, j} ^ {h} - \beta \cdot \Delta_ {k, j} ^ {h}) \cdot p ^ {h} (\mathbf {y} _ {i} ^ {h} = k | \mathbf {x} _ {i}) \\ = \underset {k} {\operatorname{argmax}} \sum_ {j = 1} ^ {C ^ {h}} (\beta \cdot \Delta_ {k, j} ^ {h} - D _ {k, j} ^ {h}) \cdot p ^ {h} (\mathbf {y} _ {i} ^ {h} = k | \mathbf {x} _ {i}) \\ = \underset {k} {\operatorname{argmax}} \sum_ {j = 1} ^ {C ^ {h}} \widetilde {D} _ {k, j} ^ {h} \cdot p ^ {h} (\mathbf {y} _ {i} ^ {h} = k | \mathbf {x} _ {i}), \\ \end{array}
$$

where $\widetilde{D}^{h} = \beta \cdot \Delta^{h} - D^{h}$ and $\beta$ signifies the weight of the learnable matrix $\Delta^{h}$ . Meanwhile, to avoid self-cost, the diagonal entries of $\Delta^{h}$ (i.e., $\Delta_{ii}^{h}$ ) are masked as zeros and then each row of $\Delta^{h}$ undergoes an L2 normalization.

To effectively learn intra-granularity differences, we thus introduce a label smoothing strategy (Müller, Kornblith, and Hinton 2019) for training. Different from previous studies (Bertinetto et al. 2020), as the goal is to pursue a model with better mistake, we propose to employ the above class-wise matrix to label smoothing explicitly. Specifically, the label smoothing technique for AIGDL could be expressed as:

$$
\widetilde {\mathbf {y}} _ {i} ^ {h} = (1 - \epsilon) \mathbf {y} _ {i} ^ {h} + \epsilon \cdot \frac {\exp \left(\gamma (\boldsymbol {\beta} \cdot \boldsymbol {\Delta} _ {y _ {i} ^ {h}} ^ {h} - \mathbf {D} _ {y _ {i} ^ {h}} ^ {h})\right)}{\sum_ {j = 1} ^ {C ^ {h}} \exp \left(\gamma (\boldsymbol {\beta} \cdot \boldsymbol {\Delta} _ {y _ {i} ^ {h} , j} ^ {h} - \mathbf {D} _ {y _ {i} ^ {h} , j} ^ {h})\right)},
$$

where $\epsilon$ is the smoothing factor, $\gamma$ is the temperature parameter, and $D_{k}^{h}$ and $\Delta_{k}^{h}$ are the k-th row of $D^{h}$ and $\Delta^{h}$ respectively. In this way, the semantic similarities among these subclasses can be implicitly reflected at the label level, ensuring consistency in predictions. Moreover, it is to be noted that the above smoothing method is a general version of the Soft-Label method (Bertinetto et al. 2020) by setting $\epsilon = 1$ and $\beta = 0$ .

Similarly, the loss function of AIGDL for each level is defined as:

$$
\mathcal {L} _ {\mathrm{AIGDL}} ^ {h} = \frac {1}{N} \sum_ {i = 1} ^ {N} \widetilde {\mathbf {y}} _ {i} ^ {h} \log \left(\operatorname{softmax} \left(\widetilde {\mathbf {D}} ^ {h} \mathbf {p} _ {i} ^ {h}\right)\right).
$$

# Overall Optimization Objective

Combining all the above components, the overall optimization objective is defined as:

$$
\begin{array}{l} \mathcal {L} = \sum_ {h} \lambda_ {h} \cdot \left(\mathcal {L} _ {\mathrm{BiLT}} ^ {h} + \mathcal {L} _ {\mathrm{AIGDL}} ^ {h}\right) \\ = \sum_ {h} \frac {\lambda_ {h}}{N} \sum_ {i = 1} ^ {N} (\mathbf {y} _ {i} ^ {h} \log {(\mathbf {p} _ {i} ^ {h})} + \widetilde {\mathbf {y}} _ {i} ^ {h} \log {(\mathrm{softmax} (\widetilde {\mathbf {D}} ^ {h} \mathbf {p} _ {i} ^ {h}))}), \\ \end{array}
$$

where $\lambda_{h} = \exp(\alpha \cdot (h - H))$ is the weight for each level. As the hierarchical level becomes coarser, its contribution to fine-grained feature learning diminishes. So the corresponding loss weight $\lambda_{h}$ decreases as the hierarchical level becomes coarser.

# Experiments

# Experiments Setup

Datasets. We evaluate methods on four datasets: FGVC-Aircraft (Maji et al. 2013), CIFAR-100 (Krizhevsky, Hinton et al. 2009), iNaturalist2019 (Van Horn et al. 2018) and tieredImageNet-H(Ren et al. 2018), all of which have been used in previous studies(Liang and Davis 2023; Garg, Sani, and Anand 2022; Bertinetto et al. 2020). Dataset statistics and split settings are provided in the Appendix.

Competitors. We compare our method against the following competitors: HXE(Bertinetto et al. 2020), Soft-Labels(Bertinetto et al. 2020), Flamingo(Chang et al. 2021), CRM(Karthik et al. 2021), HAF(Garg, Sani, and Anand 2022), HAFrame(Liang and Davis 2023), and HiE(Jain, Karthik, and Gandhi 2024). We also include the vanilla cross-entropy loss as a baseline, denoted as Cross-Entropy. Note that, all competitors are implemented following the guidelines suggested by the corresponding paper.

# Performance Comparisons

Overall Performance. Tab. 1 and Tab. 2 present the detailed performance of the FGVC-Aircraft and CIFAR-100 datasets, respectively. Detailed performance results for the iNaturalist2019 dataset are provided in the Appendix. Notably, all metrics are correlated to log-scaled LCA distance except for Top-1 Accuracy (Karthik et al. 2021). According to the reported results, we can draw the following remarks: In most cases, our proposed BiLT could outperform all competitors at all metrics, except for the Top-1 Accuracy and Hier Dist@1 on the CIFAR-100 and iNaturalist2019 datasets. Even in failure cases, the performance of BiLT is still comparable. This speaks to the efficacy of our approach.

<table><tr><td>Method</td><td>Mistake Severity(↓)</td><td>Hier Dist@1(↓)</td><td>Hier Dist@5(↓)</td><td>Hier Dist@20(↓)</td><td>Top-1 Accuracy(↑)</td></tr><tr><td>Cross-Entropy</td><td>2.12 +/- 0.0288</td><td>0.44 +/- 0.0188</td><td>2.10 +/- 0.0053</td><td>2.67 +/- 0.0022</td><td>79.35 +/- 0.7021</td></tr><tr><td>HXE</td><td>2.04 +/- 0.0074</td><td>0.43 +/- 0.0283</td><td>1.96 +/- 0.0119</td><td>2.60 +/- 0.0085</td><td>78.75 +/- 0.9481</td></tr><tr><td>Soft-Labels</td><td>2.10 +/- 0.0124</td><td>0.49 +/- 0.0223</td><td>2.07 +/- 0.0149</td><td>2.65 +/- 0.0058</td><td>77.59 +/- 0.9698</td></tr><tr><td>Flamingo</td><td>2.10 +/- 0.0352</td><td>0.40 +/- 0.0116</td><td>2.06 +/- 0.0099</td><td>2.65 +/- 0.0043</td><td>80.72 +/- 0.5849</td></tr><tr><td>CRM</td><td>2.08 +/- 0.0366</td><td>0.42 +/- 0.0163</td><td>1.74 +/- 0.0053</td><td>2.44 +/- 0.0018</td><td>79.57 +/- 0.5880</td></tr><tr><td>HAF</td><td>2.53 +/- 0.0610</td><td>0.67 +/- 0.0295</td><td>2.10 +/- 0.0063</td><td>2.61 +/- 0.0028</td><td>73.68 +/- 1.2166</td></tr><tr><td>HAFrame</td><td>2.01 +/- 0.0103</td><td>0.39 +/- 0.0162</td><td>1.75 +/- 0.0060</td><td>2.45 +/- 0.0015</td><td>80.52 +/- 0.8375</td></tr><tr><td>HiE</td><td>2.06 +/- 0.0279</td><td>0.44 +/- 0.0146</td><td>1.82 +/- 0.0028</td><td>2.46 +/- 0.0013</td><td>78.66 +/- 0.9660</td></tr><tr><td>Ours</td><td>2.00 +/- 0.0220</td><td>0.38 +/- 0.0097</td><td>1.72 +/- 0.0032</td><td>2.44 +/- 0.0016</td><td>81.23 +/- 0.5820</td></tr></table>

Table 1: Performance comparisons on the FGVC-Aircraft dataset with different metrics. The first and second best results are highlighted with bold text and underline, respectively. 

<table><tr><td>Method</td><td>Mistake Severity(↓)</td><td>Hier Dist@1(↓)</td><td>Hier Dist@5(↓)</td><td>Hier Dist@20(↓)</td><td>Top-1 Accuracy(↑)</td></tr><tr><td>Cross-Entropy</td><td>2.35 +/- 0.0225</td><td>0.52 +/- 0.0076</td><td>2.25 +/- 0.0146</td><td>3.18 +/- 0.0079</td><td>77.85 +/- 0.2699</td></tr><tr><td>HXE</td><td>2.40 +/- 0.0137</td><td>0.62 +/- 0.0205</td><td>2.05 +/- 0.0082</td><td>3.02 +/- 0.0171</td><td>72.76 +/- 0.6816</td></tr><tr><td>Soft-Labels</td><td>2.33 +/- 0.0270</td><td>0.62 +/- 0.0144</td><td>1.29 +/- 0.0056</td><td>2.73 +/- 0.0122</td><td>74.34 +/- 0.4588</td></tr><tr><td>Flamingo</td><td>2.32 +/- 0.0186</td><td>0.51 +/- 0.0094</td><td>2.07 +/- 0.0214</td><td>3.08 +/- 0.0077</td><td>77.83 +/- 0.2942</td></tr><tr><td>CRM</td><td>2.31 +/- 0.0229</td><td>0.51 +/- 0.0019</td><td>1.15 +/- 0.0037</td><td>2.20 +/- 0.0034</td><td>77.80 +/- 0.2462</td></tr><tr><td>HAF</td><td>2.24 +/- 0.0177</td><td>0.50 +/- 0.0075</td><td>1.42 +/- 0.0057</td><td>2.64 +/- 0.0037</td><td>77.51 +/- 0.3091</td></tr><tr><td>HAFrame</td><td>2.21 +/- 0.0108</td><td>0.49 +/- 0.0066</td><td>1.11 +/- 0.0018</td><td>2.18 +/- 0.0013</td><td>77.71 +/- 0.2319</td></tr><tr><td>HiE</td><td>2.20 +/- 0.0232</td><td>0.47 +/- 0.0056</td><td>1.19 +/- 0.0031</td><td>2.86 +/- 0.0097</td><td>78.63 +/- 0.3101</td></tr><tr><td>Ours</td><td>2.17 +/- 0.0197</td><td>0.48 +/- 0.0106</td><td>1.08 +/- 0.0057</td><td>2.17 +/- 0.0030</td><td>77.77 +/- 0.4140</td></tr></table>

Table 2: Performance comparisons on the CIFAR-100 dataset with different metrics. The first and second best results are highlighted with bold text and underline, respectively.

![](images/d407255d8360cecf2b816a84a389a0c806c702efcfa5e8c3affdc5460fe7b771.jpg)

<details>
<summary>bar_stacked</summary>

| Method | Cross-Entropy (%) | HXE (%) | Soft-Labels (%) | Flamingo (%) | CRM (%) | HAF (%) | HAFrame (%) | HiE (%) | Ours (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Category 1 | 33.68 | 20.27 | 46.05 | 36.34 | 24.23 | 39.43 | 38.03 | 28.19 | 39.59 |
| Category 2 | 21.73 | 44.93 | 33.33 | 36.65 | 21.74 | 41.61 | 24.57 | 16.51 | 36.72 |
| Category 3 | 41.61 | 43.56 | 35.93 | 35.93 | 20.51 | 61.90 | 37.40 | 55.30 | 23.69 |
The percentage values for each category are explicitly labeled on the bars.
</details>

(a) FGVC-Aircraft

![](images/f586d109cfd366b01738ac798a34274cb81bfb912f2f0a130c780cbeceb00e15.jpg)

<details>
<summary>bar</summary>

| Method | Percentage (%) |
| :--- | :--- |
| Cross-Entropy | 39.82 |
| HXE | 35.43 |
| Soft-Labels | 38.67 |
| Flamingo | 39.46 |
| CRM | 40.04 |
| HAF | 41.61 |
| HAFrame | 40.58 |
| HiE | 43.81 |
| Ours | 44.08 |
</details>

(b) CIFAR-100

![](images/1df9521c67c117fd243de5685329ecf0931d1f2ddfd85752d546b3940af45d5a.jpg)

<details>
<summary>bar</summary>

| Method | Percentage (%) |
| :--- | :--- |
| Cross-Entropy | 70.53 |
| HXE | 70.04 |
| Soft-Labels | 72.37 |
| Flamingo | 72.89 |
| CRM | 71.56 |
| HAF | 73.84 |
| HAFrame | 75.57 |
| HiE | 76.20 |
| Ours | 76.64 |
</details>

(c) iNaturalist2019   
Figure 5: The probability of mistakes made by our method and competitors across the three datasets. In each subfigure, different colors represent varying LCA distances, increasing from left to right. The length of each bar indicates the severity of the mistake, while the values on the bars show the probability of the model making a mistake at a specific LCA distance.

Mistake Severity Distribution. Fig. 5 shows the mistake severity distribution for our method and the competitors. Specifically, here we report the performance comparisons at the finest granularity by examining the LCA distances between wrong and ground truth classes. It is obvious that, when misclassifying an example, our proposed BiLT could make a more reasonable prediction. As shown in Fig. 5a, when the model makes a mistake, our method has a 39.59% probability that the LCA distance between the prediction and the ground truth is 1. Additionally, there is a 76.31% probability (39.59% + 36.72%) that the LCA distance is less than or equal to 2, resulting in a significantly lower proportion of severe mistakes compared to those of competitors.

Intra-Granularity Difference Visualization. Fig. 6 shows the heat map of the learnable intra-granularity difference matrix $\Delta^{H}$ , illustrating relationships between the finest-level classes on CIFAR-100. Heat map of $\Delta^{H}$ for FGVC-Aircraft are provided in the Appendix. In a specific coarse class (zoomed portion A), the colors of the 3rd and the 4th classes are greener, indicating a weaker correlation compared to other fine-grained classes. At the coarse-grained level, square (B) is more yellow than square (A), indicating finer-grained classes in (B) share closer relations with each other. This illustrates that the label tree fails to accurately describe the difference between classes, necessitating the use of $\Delta^{H}$ to represent their relationships.

![](images/42dd61c00f3f47c9c87508bfb1230f9798a804a3e9e5278cd4b15846d7bb9977.jpg)

<details>
<summary>text_image</summary>

A
B
</details>

Figure 6: Heat maps of $\Delta^{H}$ for CIFAR-100, with colors transitioning from purple to yellow as values increase. This heat map is composed of $100 \times 100$ small squares, which exactly indicates 100 classes (5th level) within CIFAR-100. 20 green or yellow squares, each composed of 5 smaller squares, represent coarser labels for 20 classes (4th level), and larger squares represent coarser classes.

# Ablation Study

Sensitivity analysis of $\alpha$ . Fig.7 presents the sensitivity analysis of $\alpha$ on the FGVC-Aircraft dataset, while the analysis for the CIFAR-100 dataset is provided in the Appendix. The weight assigned to level h in the overall optimization objective is given by $\lambda_{h} = \exp(\alpha \cdot (h - H))$ . A smaller $\alpha$ increases the weights of the coarser levels. When $\alpha$ is set to 0, all levels receive equal weight, resulting in a uniform optimization objective. Within the $\alpha$ range of 0.5 to 1.0, the model reduces Mistake Severity while maintaining Accuracy. However, a large $\alpha$ significantly decreases the weight of the coarse level, leading to a notable increase in Mistake Severity. This phenomenon illustrates the importance of coarse-grained classification for fine-grained learning.

![](images/ae3de0aa5ab88b9cb2bf05e7afc5d9010942238a15e727fcfed3c2f3f7b74308.jpg)

<details>
<summary>scatter</summary>

| α    | Accuracy |
| ---- | -------- |
| 0.0  | 80.4     |
| 0.1  | 80.4     |
| 0.5  | 80.8     |
| 1.0  | 81.3     |
| 3.0  | 80.4     |
| 5.0  | 80.5     |
| 10.0 | 80.1     |
</details>

![](images/ad0f2627c1544193ed22462083a5f2f2ce8412d9266bd63e19758eb418d8b243.jpg)

<details>
<summary>scatter</summary>

| α    | Mistake Severity |
| ---- | ---------------- |
| 0.0  | 2.04             |
| 0.1  | 2.02             |
| 0.5  | 2.00             |
| 1.0  | 2.03             |
| 3.0  | 2.04             |
| 5.0  | 2.05             |
| 10.0 | 2.08             |
</details>

![](images/9417399d2249c9fd23f218a3892067bdf6e0b2482d73dbaaf14ffc9e45e06c5f.jpg)

<details>
<summary>scatter</summary>

| β    | Accuracy |
| ---- | -------- |
| 0.0  | 81.6     |
| 0.25 | 81.4     |
| 0.5  | 81.9     |
| 0.75 | 81.5     |
| 1.0  | 81.3     |
| 2.0  | 81.0     |
| 3.0  | 81.0     |
</details>

![](images/160b1f25852aecc14f1b117ead832c8cdcfcdeab779a6f51de972632028d8904.jpg)

<details>
<summary>scatter</summary>

| β    | Mistake Severity |
| ---- | ---------------- |
| 0.0  | 2.04             |
| 0.25 | 1.98             |
| 0.5  | 1.96             |
| 0.75 | 1.98             |
| 1.0  | 1.98             |
| 2.0  | 2.00             |
| 3.0  | 2.00             |
</details>

Figure 7: Sensitivity analysis about $\alpha$ and $\beta$ on FGVC-Aircraft.

Sensitivity analysis of $\beta$ . Fig.7 shows the sensitivity analysis of $\beta$ on the FGVC-Aircraft dataset. For the sensitivity analysis of $\beta$ on the CIFAR-100 dataset please see Appendix. $\beta$ is the weight for the learnable adjustment matrix $\Delta^{h}$ . Too large or too small $\beta$ leads to decreased performance. When $\beta = 0$ , the learnable adjustment matrix $\Delta^{h}$ has no effect. Conversely, too large $\beta$ may cause hierarchical level spanning, where some levels deviate significantly from the originally defined distance matrix after adjustment. The model achieved the best performance at $\beta = 0.5$ .

Sensitivity analysis of $\epsilon$ and $\gamma$ . Fig.8 shows the sensitivity analysis of $\epsilon$ and $\gamma$ on the FGVC-Aircraft and CIFAR-100 datasets. $\gamma$ is the temperature coefficient in the soft label generation process, and $\epsilon$ is the proportion of the soft label. When $\epsilon = 0$ , it means only hard label is used, and when $\epsilon = 1$ , it means only soft label is used. $\epsilon$ and $\gamma$ regulate the label encoding together. In the FGVC-Aircraft dataset, Mistake Severity is low when $\epsilon$ is 0.3 and $\gamma$ is around 0.7. In the CIFAR-100 dataset, Mistake Severity is low when $\epsilon$ and $\gamma$ are both around 0.3. It can be observed that using only hard label or only soft label is always not the optimal solution.

![](images/0a65e64c127a0c1f14d453d551634d023a6b0e004758500bb47b7e39825af5eb.jpg)

<details>
<summary>bar</summary>

| ε    | γ    |
| ---- | ---- |
| 0.0  | 0.1  |
| 0.1  | 0.3  |
| 0.2  | 0.5  |
| 0.3  | 0.7  |
| 0.4  | 0.9  |
| 0.5  | 0.7  |
| 0.6  | 0.5  |
| 0.7  | 0.3  |
| 0.8  | 0.1  |
| 0.9  | 0.1  |
</details>

(a) FGVC-Aircraft

![](images/a5c3d22aaf6c043e2afe0b8577d197ecd28f8280dca156b2d2a2879dd62f223f.jpg)

<details>
<summary>bar</summary>

| ε    | γ=0.0 | ε=0.1 | ε=0.3 | ε=0.5 | ε=0.7 | ε=1.0 | γ=0.1 | γ=0.3 | γ=0.5 | γ=0.7 | γ=0.9 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 2.0  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 2.1  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 2.2  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 2.3  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.0  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.1  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.3  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.5  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.7  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
| 0.9  | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   | 2.1   |
The chart displays a single data series with three distinct color groups (yellow, green, purple) representing different categories or conditions within the 'ε' and 'γ' dimensions for each category.
</details>

(b) CIFAR-100   
Figure 8: Sensitivity analysis about $\epsilon$ and $\gamma$ on FGVC-Aircraft and CIFAR-100.

# Conclusion

In this paper, we focus on the granularity competition problem between different granularities in fine-grained visual classification (FGVC) tasks. However, coarse-grained features are naturally easy to learn, leading feature-shared-based methods focus only on coarse features instead of fine-grained ones. To address this issue, we propose a novel method, the Bidirectional Logits Tree (BiLT), which organizes classifiers from the finest to the coarsest levels, rather than a parallel framework where all classifiers share the same feature vector. Additionally, we observe that predefined label trees cannot accurately capture semantic differences between labels at the same granularity, and therefore propose Adaptive Intra-Granularity Difference Learning (AIGDL). This method empowers BiLT to learn the fine-grained semantic differences of classes within the same granularity. Finally, extensive experiments and visualizations justify the effectiveness of our method.

# Acknowledgments

This work was supported in part by the National Key R&D Program of China under Grant 2018AAA0102000, in part by National Natural Science Foundation of China: 62236008, U21B2038, U23B2051, 61931008, 62122075, 62206264, and 92370102, in part by Youth Innovation Promotion Association CAS, in part by the Strategic Priority Research Program of the Chinese Academy of Sciences, Grant No. XDB0680000, in part by the Innovation Funding of ICT, CAS under Grant No.E000000, and in part by the Postdoctoral Fellowship Program of CPSF under Grant GZB20240729.

# References

Bertinetto, L.; Mueller, R.; Tertikas, K.; Samangooei, S.; and Lord, N. A. 2020. Making better mistakes: Leveraging class hierarchies with deep networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12506–12515.   
Bukchin, G.; Schwartz, E.; Saenko, K.; Shahar, O.; Feris, R.; Giryes, R.; and Karlinsky, L. 2021. Fine-grained angular contrastive learning with coarse labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8730–8740.   
Chang, D.; Pang, K.; Zheng, Y.; Ma, Z.; Song, Y.-Z.; and Guo, J. 2021. Your "flamingo" is my "bird": Fine-grained, or not. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11476–11485.   
Chen, J.; Wang, P.; Liu, J.; and Qian, Y. 2022. Label relation graphs enhanced hierarchical residual network for hierarchical multi-granularity classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4858–4867.   
Collins, K. M.; Bhatt, U.; and Weller, A. 2022. Eliciting and learning with soft labels from every annotator. In Proceedings of the AAAI Conference on Human Computation and Crowdsourcing, 40–52.   
Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, 248–255.   
Du, R.; Xie, J.; Ma, Z.; Chang, D.; Song, Y.-Z.; and Guo, J. 2021. Progressive learning of category-consistent multi-granularity features for fine-grained visual classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12): 9521–9535.   
Duda, R. O.; and Hart, P. E. 1974. Pattern classification and scene analysis. In A Wiley-Interscience publication.   
Frome, A.; Corrado, G. S.; Shlens, J.; Bengio, S.; Dean, J.; Ranzato, M.; and Mikolov, T. 2013. Devise: A deep visual-semantic embedding model. In Advances in Neural Information Processing Systems.   
Garg, A.; Sani, D.; and Anand, S. 2022. Learning hierarchy aware features for reducing mistake severity. In European Conference on Computer Vision, 252–267.

Gong, X.; Bisht, N.; and Xu, G. 2024. Does Label Smoothing Help Deep Partial Label Learning? In International Conference on Machine Learning.   
Grcic, M.; Gadetsky, A.; and Brbic, M. 2024. Fine-grained Classes and How to Find Them. In International Conference on Machine Learning, 16275–16294.   
Han, B.; Xu, Q.; Yang, Z.; Bao, S.; Wen, P.; Jiang, Y.; and Huang, Q. 2024. AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation. In Advances in Neural Information Processing Systems.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.   
Jain, K.; Karthik, S.; and Gandhi, V. 2024. Test-time amendment with a coarse classifier for fine-grained classification. In Advances in Neural Information Processing Systems.   
Jiang, X.; Tang, H.; Gao, J.; Du, X.; He, S.; and Li, Z. 2024. Delving into Multimodal Prompting for Fine-Grained Visual Classification. In Proceedings of the AAAI Conference on Artificial Intelligence, 2570–2578.   
Karthik, S.; Prabhu, A.; Dokania, P. K.; and Gandhi, V. 2021. No Cost Likelihood Manipulation at Test Time for Making Better Mistakes in Deep Networks. In International Conference on Learning Representations.   
Kingma, D. P.; and Ba, J. 2015. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations.   
Krizhevsky, A.; Hinton, G.; et al. 2009. Learning multiple layers of features from tiny images.   
Landrieu, L.; and Garnot, V. S. F. 2021. Leveraging class hierarchies with metric-guided prototype learning. In British Machine Vision Conference.   
Liang, T.; and Davis, J. 2023. Inducing neural collapse to a fixed hierarchy-aware frame for reducing mistake severity. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1443–1452.   
Liu, M.; Roy, S.; Li, W.; Zhong, Z.; Sebe, N.; and Ricci, E. 2024. Democratizing Fine-grained Visual Recognition with Large Language Models. In International Conference on Learning Representations.   
Liu, S.; Chen, J.; Pan, L.; Ngo, C.-W.; Chua, T.-S.; and Jiang, Y.-G. 2020. Hyperbolic visual embedding learning for zero-shot recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9273–9281.   
Maji, S.; Rahtu, E.; Kannala, J.; Blaschko, M.; and Vedaldi, A. 2013. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151.   
Müller, R.; Kornblith, S.; and Hinton, G. E. 2019. When does label smoothing help? In Advances in Neural Information Processing Systems.   
Ni, J.; Cheng, W.; Chen, Z.; Asakura, T.; Soma, T.; Kato, S.; and Chen, H. 2021. Superclass-conditional gaussian mixture model for learning fine-grained embeddings. In International Conference on Learning Representations.

Papyan, V.; Han, X.; and Donoho, D. L. 2020. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40): 24652–24663.   
Park, H.; Noh, J.; Oh, Y.; Baek, D.; and Ham, B. 2023. Acls: Adaptive and conditional label smoothing for network calibration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3936–3945.   
Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; et al. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems.   
Pu, Y.; Han, Y.; Wang, Y.; Feng, J.; Deng, C.; and Huang, G. 2024. Fine-grained recognition with learnable semantic data augmentation. IEEE Transactions on Image Processing, 33:3130–3144.   
Qin, Y.; Wang, X.; Beutel, A.; and Chi, E. 2021. Improving calibration through the relationship with adversarial robustness. In Advances in Neural Information Processing Systems, 14358–14369.   
Ren, M.; Triantafillou, E.; Ravi, S.; Snell, J.; Swersky, K.; Tenenbaum, J. B.; Larochelle, H.; and Zemel, R. S. 2018. Meta-learning for semi-supervised few-shot classification. In International Conference on Learning Representations.   
Silla, C. N.; and Freitas, A. A. 2011. A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22: 31–72.   
Touvron, H.; Sablayrolles, A.; Douze, M.; Cord, M.; and Jégou, H. 2021. Grafit: Learning fine-grained image representations with coarse labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 874–884.   
Van Horn, G.; Mac Aodha, O.; Song, Y.; Cui, Y.; Sun, C.; Shepard, A.; Adam, H.; Perona, P.; and Belongie, S. 2018. The inaturalist species classification and detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 8769–8778.   
Wang, D.; Shen, Z.; Shao, J.; Zhang, W.; Xue, X.; and Zhang, Z. 2015. Multiple granularity descriptors for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision, 2399–2406.   
Wang, R.; Zou, C.; Zhang, W.; Zhu, Z.; and Jing, L. 2023. Consistency-aware Feature Learning for Hierarchical Fine-grained Visual Classification. In Proceedings of the 31st ACM International Conference on Multimedia, 2326–2334.   
Wang, W.; Sun, Y.; Li, W.; and Yang, Y. 2024. Transhp: Image classification with hierarchical prompting. In Advances in Neural Information Processing Systems.   
Wang, Y.; Wang, Z.; Hu, Q.; Zhou, Y.; and Su, H. 2021. Hierarchical semantic risk minimization for large-scale classification. IEEE Transactions on Cybernetics, 52(9): 9546–9558.   
Wei, J.; Liu, H.; Liu, T.; Niu, G.; Sugiyama, M.; and Liu, Y. 2022. To Smooth or Not? When Label Smoothing Meets Noisy Labels. In International Conference on Machine Learning, 23589–23614.

Wu, H.; Merler, M.; Uceda-Sosa, R.; and Smith, J. R. 2016. Learning to make better mistakes: Semantics-aware visual food recognition. In Proceedings of the 24th ACM International Conference on Multimedia, 172–176.

Xian, Y.; Akata, Z.; Sharma, G.; Nguyen, Q.; Hein, M.; and Schiele, B. 2016. Latent embeddings for zero-shot classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 69–77.

Xu, S.; Sun, Y.; Zhang, F.; Xu, A.; Wei, X.; and Yang, Y. 2023. Hyperbolic Space with Hierarchical Margin Boosts Fine-Grained Learning from Coarse Labels. In Advances in Neural Information Processing Systems.

Yuan, L.; Tay, F. E.; Li, G.; Wang, T.; and Feng, J. 2020. Revisiting knowledge distillation via label smoothing regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3903–3911.

Zagoruyko, S.; and Komodakis, N. 2016. Wide Residual Networks. In British Machine Vision Conference.

Zhang, C.-B.; Jiang, P.-T.; Hou, Q.; Wei, Y.; Han, Q.; Li, Z.; and Cheng, M.-M. 2021. Delving deep into label smoothing. IEEE Transactions on Image Processing, 30: 5984–5996.

Zhang, J.; Chen, Z.; Zhang, H.; Xiao, C.; and Li, B. 2023. {DiffSmooth}: Certifiably robust learning via diffusion models and local smoothing. In 32nd USENIX Security Symposium (USENIX Security 23), 4787–4804.

Zhang, M.-L.; and Zhou, Z.-H. 2013. A review on multi-label learning algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8): 1819–1837.

Zhang, S.; Zheng, S.; Shui, Z.; and Yang, L. 2024. HLS-FGVC: Hierarchical Label Semantics Enhanced Fine-Grained Visual Classification. In IEEE International Conference on Acoustics, Speech and Signal Processing, 7370–7374.

Zhao, B.; Li, F.; and Xing, E. 2011. Large-scale category structure aware image categorization. In Advances in Neural Information Processing Systems.

# Appendix

# Related Work

Fine-grained Category Discovery Methods These methods primarily focus on discovering fine-grained categories using coarse-grained information. (Bukchin et al. 2021) was the first to address the problem of identifying fine-grained categories from coarse-grained information. They proposed a method combining coarse-grained pretraining and self-supervised contrastive learning, marking the first attempt to integrate these approaches for this task. Grafit (Touvron et al. 2021) later combined coarse-grained labels with fine-grained latent spaces to enhance fine-grained retrieval and classification accuracy. SCGM (Ni et al. 2021) used the Gaussian Mixture Models to link coarse-grained and fine-grained categories. More recently, FALCON (Grcic, Gadetsky, and Brbic 2024) addressed a more challenging scenario, discovering latent fine-grained categories without prior observations. They tackled this problem using an alternating optimization approach and gave a rich theoretical analysis. Since our method relies on multi-granularity category, these approaches provide a foundation for our method.

# Experiments

Datasets For FGVC-Aircraft, we use the original hierarchical labels and the split setting provided by the dataset. For CIFAR-100, we use the hierarchical labels from (Landrieu and Garnot 2021) and the split setting from (Garg, Sani, and Anand 2022). For iNaturalist2019, we use the hierarchical labels and split setting provided by (Bertinetto et al. 2020). For tieredImageNet-H, we use the hierarchical labels and split setting provided by (Ren et al. 2018). For all datasets, the distance between any two nodes in the label tree is defined using the Lowest Common Ancestor (LCA). Dataset statistics are shown in Tab. 3.

<table><tr><td>Datasets</td><td>Levels</td><td>Classes</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>FGVC-Aircraft</td><td>3</td><td>100</td><td>3,334</td><td>3,333</td><td>3,333</td></tr><tr><td>CIFAR-100</td><td>5</td><td>100</td><td>45,000</td><td>5,000</td><td>10,000</td></tr><tr><td>iNaturalist2019</td><td>7</td><td>1010</td><td>187,385</td><td>40,121</td><td>40,737</td></tr><tr><td>tieredImageNet-H</td><td>12</td><td>608</td><td>425,600</td><td>15,200</td><td>15,200</td></tr></table>

Table 3: Statistics of the datasets.

Evaluation Metrics. We evaluate the performance of the methods by the following metrics:

\- Mistake Severity is proposed in (Bertinetto et al. 2020).

$$
\text { Mistake   Severity } = \frac {\mathrm{LCA} (\widehat {\mathcal {Y}} ^ {H} , \mathcal {Y} ^ {H})}{| \mathcal {Y} ^ {H} | - | \widehat {\mathcal {Y}} ^ {H} \cap \mathcal {Y} ^ {H} |},
$$

where $\widehat{y}^{H}$ is the predicted label set, $y^{H}$ is the ground truth label set, and $|\cdot|$ means the size of set. It measures the average LCA distance between the incorrectly predicted label and the ground truth label in the label tree, which reflects the severity of the mistakes made by the model.

![](images/1e58731e7c70747de90b4f6835b284ee5ff1ad588f5d6216d466ea564f4c2d66.jpg)

<details>
<summary>text_image</summary>

A
B
C
</details>

Figure 9: Heat maps of $\Delta^{H}$ for FGVC-Aircraft, with colors transitioning from purple to yellow as values increase.

\- Hier Dist@k, also proposed in (Bertinetto et al. 2020), measures the average LCA distance between the top-k predicted labels and the ground truth label in the label tree, respectively. Given the top-k predicted label set $\widehat{\mathcal{Y}}_k^H$ and the top-k ground truth label set $\mathcal{Y}_k^H$ , the definition of Hier Dist@k is

$$
\mathbf {H i e r} \operatorname{Dist} @ k = \frac {\operatorname{LCA} \left(\widehat {\mathcal {Y}} _ {k} ^ {H} , \mathcal {Y} _ {k} ^ {H}\right)}{| \mathcal {Y} ^ {H} |},
$$

It reflects the overall quality of the top- $k$ predictions, which is important for certain downstream tasks.

\- Top-1 Accuracy, a commonly used metric in fine-grained visual classification tasks, defined as

$$
\text { Top - 1   Accuracy } = \frac {| \widehat {\mathcal {Y}} ^ {H} \cap \mathcal {Y} ^ {H} |}{| \mathcal {Y} ^ {H} |}.
$$

Implementation Details. We implemented our model with PyTorch $^{1}$ (Paszke et al. 2019) and all experiments were conducted on NVIDIA Tesla A100 80G GPUs. For CIFAR-100, we use WideResNet-28 (Zagoruyko and Komodakis 2016) as the backbone network for all methods. For FGVC-Aircraft and iNaturalist2019, we use ResNet-50 (He et al. 2016) as the backbone network for all methods and initialize it with the pre-trained model from ImageNet (Deng et al. 2009). Models are trained for 200 epochs on CIFAR-100 and 100 epochs on FGVC-Aircraft and iNaturalist2019. All methods, except for HXE and Soft-Label, use the SGD optimizer with a momentum of 0.9 and a weight decay of 0.0005. For HXE and Soft-Label, following (Garg, Sani, and Anand 2022), the Adam (Kingma and Ba 2015) optimizer is used with a learning rate of 0.001. For SGD training, the learning rate is initialized to 0.01 for the backbone network and 0.1 for the transformation layer and classifier. All methods are trained with a cosine learning rate scheduler as in (Chang et al. 2021). We use a batch size of 64

<table><tr><td>Method</td><td>Mistake Severity(↓)</td><td>Hier Dist@1(↓)</td><td>Hier Dist@5(↓)</td><td>Hier Dist@20(↓)</td><td>Top-1 Accuracy(↑)</td></tr><tr><td>Cross-Entropy</td><td>2.29 +/- 0.0185</td><td>0.67 +/- 0.0080</td><td>1.98 +/- 0.0029</td><td>3.41 +/- 0.0070</td><td>70.66 +/- 0.2274</td></tr><tr><td>HXE</td><td>2.29 +/- 0.0206</td><td>0.75 +/- 0.0121</td><td>1.84 +/- 0.0082</td><td>2.41 +/- 0.0039</td><td>67.16 +/- 0.3120</td></tr><tr><td>Soft-Labels</td><td>2.19 +/- 0.0133</td><td>0.71 +/- 0.0099</td><td>1.28 +/- 0.0071</td><td>2.04 +/- 0.0085</td><td>68.47 +/- 0.2941</td></tr><tr><td>Flamingo</td><td>2.13 +/- 0.0063</td><td>0.64 +/- 0.0014</td><td>1.79 +/- 0.0126</td><td>3.28 +/- 0.0110</td><td>70.67 +/- 0.2095</td></tr><tr><td>CRM</td><td>2.24 +/- 0.0155</td><td>0.66 +/- 0.0062</td><td>1.19 +/- 0.0046</td><td>1.76 +/- 0.0050</td><td>70.66 +/- 0.2274</td></tr><tr><td>HAF</td><td>2.13 +/- 0.0192</td><td>0.63 +/- 0.0045</td><td>1.55 +/- 0.2188</td><td>2.68 +/- 0.4208</td><td>70.57 +/- 0.1645</td></tr><tr><td>HAFrame</td><td>2.06 +/- 0.0087</td><td>0.60 +/- 0.0030</td><td>1.14 +/- 0.0025</td><td>1.74 +/- 0.0017</td><td>70.89 +/- 0.1759</td></tr><tr><td>HiE</td><td>2.04 +/- 0.0162</td><td>0.58 +/- 0.0041</td><td>1.18 +/- 0.1247</td><td>2.09 +/- 0.2103</td><td>71.43 +/- 0.2584</td></tr><tr><td>Ours</td><td>2.04 +/- 0.0070</td><td>0.59 +/- 0.0019</td><td>1.13 +/- 0.0017</td><td>1.72 +/- 0.0018</td><td>70.98 +/- 0.0450</td></tr></table>

Table 4: Performance comparisons on the iNaturalist2019 dataset with different metrics. The first and second best results are highlighted with bold text and underline, respectively.

<table><tr><td>Method</td><td>Mistake Severity(↓)</td><td>Hier Dist@1(↓)</td><td>Hier Dist@5(↓)</td><td>Hier Dist@20(↓)</td><td>Top-1 Accuracy(↑)</td></tr><tr><td>Cross-Entropy</td><td>6.95 +/- 0.0208</td><td>1.83 +/- 0.0117</td><td>5.69 +/- 0.0192</td><td>7.34 +/- 0.0291</td><td>73.63 +/- 0.1165</td></tr><tr><td>HXE</td><td>6.93 +/- 0.0297</td><td>1.81 +/- 0.0109</td><td>5.71 +/- 0.0101</td><td>6.99 +/- 0.0091</td><td>71.29 +/- 0.2378</td></tr><tr><td>Soft-Labels</td><td>6.94 +/- 0.0263</td><td>1.82 +/- 0.0117</td><td>5.67 +/- 0.0099</td><td>6.92 +/- 0.0109</td><td>70.18 +/- 0.2063</td></tr><tr><td>Flamingo</td><td>6.93 +/- 0.0391</td><td>1.92 +/- 0.0135</td><td>5.75 +/- 0.0130</td><td>7.41 +/- 0.0098</td><td>72.34 +/- 0.1488</td></tr><tr><td>CRM</td><td>6.89 +/- 0.0272</td><td>1.82 +/- 0.0155</td><td>4.82 +/- 0.0062</td><td>6.03 +/- 0.0041</td><td>73.54 +/- 0.1495</td></tr><tr><td>HAF</td><td>6.89 +/- 0.0281</td><td>1.82 +/- 0.0125</td><td>5.52 +/- 0.0176</td><td>6.95 +/- 0.0120</td><td>73.52 +/- 0.1613</td></tr><tr><td>HAFrame</td><td>6.89 +/- 0.0251</td><td>1.79 +/- 0.0216</td><td>4.94 +/- 0.0118</td><td>6.15 +/- 0.0065</td><td>74.00 +/- 0.3549</td></tr><tr><td>HiE</td><td>6.85 +/- 0.0306</td><td>1.84 +/- 0.0189</td><td>5.25 +/- 0.0143</td><td>6.74 +/- 0.0099</td><td>72.78 +/- 0.2512</td></tr><tr><td>Ours</td><td>6.77 +/- 0.0371</td><td>1.72 +/- 0.0241</td><td>5.41 +/- 0.0232</td><td>6.86 +/- 0.0158</td><td>74.41 +/- 0.3347</td></tr></table>

Table 5: Performance comparisons on the tieredImageNet-H dataset with different metrics. The first and second best results are highlighted with bold text and underline, respectively.

for CIFAR-100 and FGVC-Aircraft, and 256 for iNaturalist2019. The same data augmentation strategy as in (Garg, Sani, and Anand 2022) is used for all datasets. We select the best model based on the validation set and report the results on the test set. We run each method 5 times with different random seeds (0-4), and results are presented with a $95\%$ confidence interval following (Liang and Davis 2023).

# Performance Comparisons

Intra-Granularity Difference Visualization. Fig. 9 shows heat maps of the learnable intra-granularity difference matrix $\Delta^{H}$ for the finest-level classes on FGVC-Aircraft. Same as CIFAR-100, in a specific coarse class (the zoomed portion in Figure), the colors of the 2nd and the 3rd classes are bluer, indicating a weaker correlation compared to other fine-grained classes. At the coarse-grained level, square (B) is greener than square (A and C), indicating finer-grained classes in (A and C) share closer relations with each other.

# Ablation Study

Sensitivity analysis of $\alpha$ . Fig.10 presents the sensitivity analysis of $\alpha$ on the CIFAR-100 dataset. Similar to the FGVC-Aircraft dataset, setting $\alpha$ within 0.5 to 3.0 can enhance fine-grained learning by leveraging coarse-grained classification.

Sensitivity analysis of $\beta$ . Fig.10 presents the sensitivity analysis of $\beta$ on the CIFAR-100 dataset. Like the FGVC-Aircraft dataset, $\beta$ within 0.5 to 0.75 can enhance fine-grained learning.

![](images/9274231fbe44380c5d508820c41c1f23a2f08e3dd3da63146853228e09d4c487.jpg)

<details>
<summary>scatter</summary>

| α    | Accuracy |
| ---- | -------- |
| 0.0  | 77.8     |
| 0.1  | 77.7     |
| 0.5  | 78.8     |
| 1.0  | 78.9     |
| 3.0  | 79.0     |
| 5.0  | 78.4     |
| 10.0 | 78.4     |
</details>

![](images/f0f1c1c444cb3a07dae6f8a0dee4b32993117ee87c8faff44532acfaf24f523d.jpg)

<details>
<summary>scatter</summary>

| α    | Mistake Severity |
| ---- | ---------------- |
| 0.0  | 2.16             |
| 0.1  | 2.15             |
| 0.5  | 2.21             |
| 1.0  | 2.26             |
| 3.0  | 2.25             |
| 5.0  | 2.26             |
| 10.0 | 2.30             |
</details>

![](images/3a38abd6a73af2b6ffba596edaac9fb2d89e358beb33c213599a71c472ee4bb1.jpg)

<details>
<summary>scatter</summary>

| β    | Accuracy |
| ---- | -------- |
| 0.0  | 77.6     |
| 0.25 | 77.6     |
| 0.5  | 77.7     |
| 0.75 | 77.8     |
| 1.0  | 77.8     |
| 2.0  | 77.4     |
| 3.0  | 77.4     |
</details>

![](images/8c24882010d446d54897f093fc45910533703e0cde1f514fd2bbdcb80bc9effd.jpg)

<details>
<summary>scatter</summary>

| β    | Mistake Severity |
| ---- | ---------------- |
| 0.0  | 2.19             |
| 0.25 | 2.18             |
| 0.5  | 2.17             |
| 0.75 | 2.15             |
| 1.0  | 2.20             |
| 2.0  | 2.16             |
| 3.0  | 2.19             |
</details>

Figure 10: Sensitivity analysis about $\alpha$ and $\beta$ on CIFAR-100 dataset.

Different components. The ablation study, which involves BiLT, AIGDL, and label smoothing, is presented in Table 6 on the FGVC-Aircraft dataset. The results show that using all three components together achieve the best performance. When used individually, each component generally improves performance, except for Label Smoothing, which

<table><tr><td>BiLT</td><td>AIGDL</td><td>LabelSmoothing</td><td>Mistake Severity(↓)</td><td>Hier Dist@1(↓)</td><td>Hier Dist@5(↓)</td><td>Top-1 Accuracy(↑)</td></tr><tr><td></td><td></td><td></td><td>2.12</td><td>0.44</td><td>2.1</td><td>79.35</td></tr><tr><td>✓</td><td></td><td></td><td>2.04</td><td>0.39</td><td>2.03</td><td>81.07</td></tr><tr><td></td><td>✓</td><td></td><td>2.05</td><td>0.41</td><td>1.73</td><td>80.20</td></tr><tr><td></td><td></td><td>✓</td><td>2.10</td><td>0.51</td><td>1.81</td><td>75.84</td></tr><tr><td>✓</td><td>✓</td><td></td><td>1.99</td><td>0.39</td><td>1.73</td><td>80.47</td></tr><tr><td>✓</td><td></td><td>✓</td><td>1.96</td><td>0.37</td><td>1.72</td><td>81.01</td></tr><tr><td></td><td>✓</td><td>✓</td><td>2.05</td><td>0.40</td><td>1.73</td><td>80.47</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>1.95</td><td>0.36</td><td>1.72</td><td>81.34</td></tr></table>

Table 6: Ablation study over different components of our method. The first and second best results are highlighted with bold text and underline, respectively.

alone reduces Top-1 Accuracy. This decline occurs because Label Smoothing, while introducing class distance information, weakens ground-truth supervision, making the model's predictions overly conservative and reducing Top-1 Accuracy. Among the three components, BiLT provides the most significant performance improvement, highlighting BiLT's central role in our method.