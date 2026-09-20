# Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation

Jintao Tong $^{1}$ Yixiong Zou $^{✉1}$ Guangyao Chen $^{2}$ Yuhua Li $^{1}$ Ruixuan Li $^{1}$

# Abstract

Cross-Domain Few-Shot Segmentation (CD-FSS) aims to transfer knowledge from a source-domain dataset to unseen target-domain datasets with limited annotations. Current methods typically compare the distance between training and testing samples for mask prediction. However, we find an entanglement problem exists in this widely adopted method, which tends to bind source-domain patterns together and make each of them hard to transfer. In this paper, we aim to address this problem for the CD-FSS task. We first find a natural decomposition of the ViT structure, based on which we delve into the entanglement problem for an interpretation. We find the decomposed ViT components are crossly compared between images in distance calculation, where the rational comparisons are entangled with those meaningless ones by their equal importance, leading to the entanglement problem. Based on this interpretation, we further propose to address the entanglement problem by learning to weigh for all comparisons of ViT components, which learn disentangled features and re-compose them for the CD-FSS task, benefiting both the generalization and finetuning. Experiments show that our model outperforms the state-of-the-art CD-FSS method by 1.92% and 1.88% in average accuracy under 1-shot and 5-shot settings, respectively.

# 1. Introduction

Recent progress in deep neural networks (Long et al., 2015; Zhao et al., 2017; Dosovitskiy et al., 2020) has been driven by large-scale annotated datasets. However, the reliance

![](images/c5cb092d7d6741d6ceeb5bd02b98cc46701de242696c65a6a5ab9f0553e349cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Is"] --> B["comparison"]
    B --> C["visualization"]
    C --> D["prediction"]
    E["Is"] --> F["Disentangle & Composite"]
    F --> G["prediction"]
    H["Fs"] --> I["Self-Disentanglement"]
    J["Fq"] --> K["Disentangled Features"]
    I --> L["Composite"]
    K --> L
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#cff,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#ffc,stroke:#333
```
</details>

Figure 1: (a) A problem of feature entanglement exists in current works, which entangles multiple patterns and reduces the transferability. (b)(c) To handle this problem, we find a natural decomposition in ViT's feature, then analyze the entanglement problem based on this decomposition, and finally propose to self-disentangle and re-compose the ViT feature to address this problem for efficient cross-domain transferring and target-domain adaptation.

on abundant labeled data poses a major challenge, especially for dense prediction tasks like semantic segmentation. Cross-Domain Few-shot Semantic Segmentation (CD-FSS) (Shaban et al., 2017; Dong & Xing, 2018; Zhang et al., 2020b; Lei et al., 2022) has been introduced to address this issue, enabling predictions for target-domain unseen classes by limited annotated samples, with knowledge transferred from a data-sufficient source domain.

Existing CD-FSS works (Herzog, 2024; Su et al., 2024; He et al., 2024; Tong et al., 2024) usual perform segmentation by measuring the similarity between the support and query set based on features output by the encoder (Fig. 1a). However, we find this well-adopted method always leads to the entanglement of multiple patterns $^{1}$ and harms the transferability. For example, in Fig. 1a, the model tends to entangle the patterns of wings and bodies, i.e., detecting wings and bodies only when these two patterns appear simultaneously. However, if an image contains only the wings but the body is different from the training data (e.g., another kind of

![](images/e7adb2f5ff9fc42247871dc767d54abbf6d15a257c5a9227a19f1d9ff825d412.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph_Layer0["Layer0"]
        A1["MSA⁰"] --> B1["⊕"]
        B1 --> C1["MLP⁰"] --> D1["⊕"]
    end
    subgraph_Layern["Layern"]
        E1["MSAⁿ"] --> F1["⊕"]
        F1 --> G1["MLPⁿ"] --> H1["⊕"]
    end
    I["..."] --> J["..."]
    K["output"] --> L["x⁰ + MSA⁰ + MLP⁰ + MSA¹ + MLP¹ +...... + MSAⁿ + MLPⁿ"]
    L --> M["Block⁰"]
    N["Block¹"] --> O["Blockⁿ"]
    style L fill:#f9f,stroke:#333
    style M fill:#ccf,stroke:#333
    style O fill:#ccf,stroke:#333
```
</details>

Figure 2: The residual connection and consistent spatial size make the output of ViT components located in the same feature space, which inspires us to view the final output of a ViT as the cumulative composition of all ViT components.

bat), the model may fail to capture the wings, leading to segmentation errors. For the CD-FSS task, domain gaps and semantic gaps heavily exist between the source and target datasets. Therefore, transferring entangled patterns is much more difficult than transferring disentangled ones. Inspired by this issue, in this paper, we aim to address the entanglement problem for the CD-FSS task (Fig. 1bc).

Recent work on ViT interpretability (Gandelsman et al.) shows that the residual connections and the consistent spatial size make the output of each ViT component (e.g., MSA, MLP) located in the same feature space. Therefore, we find the final output of a ViT can naturally be seen as a cumulative composition of all ViT components (shown in Fig.2 and detailed in Section 2.2). Such a structural decomposition of ViT's output inspires us to ask: can the entangled semantic patterns also be decomposed in this way?

Based on this inspiration, we first delve into the entanglement problem for an interpretation. We find each ViT component captures distinct semantic patterns, e.g., bodies and wings, while the cumulative composition of these components implicitly binds all these patterns in ViT's output. By comparing distances between two images, as a mainstream of CD-FSS methods, the model essentially combines all possible comparisons between different components equally. Therefore, rational comparisons between patterns (wings vs. wings) are entangled with those meaningless comparisons (bodies vs. wings) by their equal importance, which we interpret to cause the feature entanglement problem.

Inspired by this interpretation, we further propose to handle the entanglement problem by learning to weigh for all comparisons between ViT components. Specifically, we first self-disentangle ViT's output by extracting features of different ViT components. Then, we introduce an Orthogonal Space Decoupling (OSD) module to further reduce the correlation of the disentangled features. Given these disentangled component features, we propose a Cross-Pattern Comparison (CPC) module, where the disentangled patterns are compared crossly for the re-composition, based on weights generated by OSD to emphasize the comparison between components with the same position. During the target-domain finetuning, we further introduce the Adaptive Fusion Weight (AFW) to dynamically learn the comparison weights for efficient adaptation (Fig. 1b).

To sum up, our primary contributions are as follows:

- To our knowledge, we are the first to analyze the feature entanglement problem from the aspect of the natural decomposition of ViT structures for the CD-FSS task.   
- We interpret the entanglement problem as a result of entangling rational comparisons between ViT components with those meaningless ones by their equal importance.   
- Based on this interpretation, we further propose to self-disentangle and re-compose ViT components for the CD-FSS task with the proposed orthogonal space decoupling module, cross-comparison module, and adaptive fusion weight module, which addresses the entanglement problem by learning to weigh for each comparison, benefiting both the generalization and finetuning for CD-FSS.   
- Extensive experiments show the effectiveness of our work on four different CD-FSS scenarios. Our model significantly outperforms state-of-the-art methods.

# 2. Delve into Feature Entanglement

In this section, we delve into the causes of feature entanglement by decomposing the structure of the ViT output.

# 2.1. Problem Definition

Cross-domain few-shot semantic segmentation (CD-FSS) aims to transfer knowledge learned from the source domain to unseen target domains with only a few annotated support images. Consider a source domain $D_{s} = (\mathcal{X}_{s}, \mathcal{Y}_{s})$ and a target domain $D_{t} = (\mathcal{X}_{t}, \mathcal{Y}_{t})$ , where X denotes the input distribution and Y denotes the label space. The input data distributions of $D_{s}$ and $D_{t}$ are distinct, and their label spaces do not overlap, i.e., $X_{s} \neq X_{t}, Y_{s} \cap Y_{t} = \emptyset$ . The model is trained solely on $D_{s}$ and without access to the target data, and then applied to segment novel classes in $D_{t}$ .

In this work, we adopt the meta-learning episodic manner following (Lei et al., 2022) to train and test our model. Specifically, both the training set from $D_{s}$ and the testing set from $D_{t}$ consist of several episodes. Each episode includes $K$ support samples $S = \{I_s^i, M_s^i\}_{i=1}^K$ ( $K$ image-mask pairs) and a query $Q = \{I_q, M_q\}$ , where $I$ represents the image and $M$ denotes the label. Within each episode, the model is expected to use the support sample $\{I_s, M_s\}_{i=1}^K$ and the query image $I_q$ to predict the query label.

# 2.2. Structural Decomposition of the ViT Output

ViT architecture. ViT (Dosovitskiy et al., 2020) is a residual network built from L layers, each of which contains a multi-head self-attention (MSA) followed by an MLP block. The input I is first split into N non-overlapping image patches. The patches are projected linearly into N d-dimensional vectors, and positional embeddings are added

![](images/bf3e2354f5525ed0edb436c3a15509f0fc876270e66234c1ada8818f12a99dc7.jpg)

<details>
<summary>text_image</summary>

support
query
</details>

Figure 3: Visualization of the cross-match between layers, where the same column means the same layer ID and bold lines indicate rational matches.

to them to create the image tokens $\{z_{i}^{0}\}_{i\in\{1,\ldots,N\}}$ . Due to the segmentation task, the CLS token is excluded (not included in the following formulas). Formally, the matrix $Z^{0}\in R^{d\times N}$ , with the tokens $z_{1}^{0},z_{2}^{0},\ldots,z_{N}^{0}$ as columns, constitutes the initial state of the residual stream. It is updated for L iterations via these two residual steps:

$$
\hat {Z} _ {l} = \mathrm{MSA} ^ {l} (Z ^ {l - 1}) + Z ^ {l - 1},   Z _ {l} = \mathrm{MLP} ^ {l} (\hat {Z} ^ {l}) + \hat {Z} ^ {l} \tag {1}
$$

Decomposition of the ViT. The residual structure of ViT allows us to express its output as a sum of the direct contributions of individual layers of the model. By unrolling Eq. 1 across layers, the image representation $\mathrm{ViT}(I)$ can be written as (Both here and in Eq. 1, we ignore a layer-normalization term to simplify derivations):

$$
\operatorname{ViT} (I) = Z ^ {0} + \sum_ {l = 1} ^ {L} \mathbf {M S A} ^ {l} \left(Z ^ {l - 1}\right) + \sum_ {l = 1} ^ {L} \mathbf {M L P} ^ {l} \left(\hat {Z} ^ {l}\right) \tag {2}
$$

We ignore here the indirect effects of the output of one layer on another downstream layer, and further simplify the MLPs and MSAs into Layers $^{2}$ :

$$
\operatorname{ViT} (I) = Z ^ {0} + \sum_ {l = 1} ^ {L} \text { Layer } ^ {l} \tag {3}
$$

# 2.3. Analyzing Entanglement by ViT Decomposition

Since most CD-FSS methods are based on distances between support and query set images, we begin our analysis by revisiting the distance comparison. For a support-query pair $\{I_{s}, I_{q}\}$ , their features extracted by ViT are:

$$
Z _ {s} = \operatorname{ViT} (I _ {s}), \quad Z _ {q} = \operatorname{ViT} (I _ {q}) \tag {4}
$$

The similarity score $S$ is computed using cosine similarity:

$$
S = Z _ {s} \cdot Z _ {q} / \| Z _ {s} \| \| Z _ {q} \| \tag {5}
$$

Substituting Eq 3 and Eq 4 into the similarity formula:

$$
S = \left(Z _ {s} ^ {0} + \sum_ {l = 1} ^ {L} \text { Layer } _ {s} ^ {l}\right) \cdot \left(Z _ {q} ^ {0} + \sum_ {l = 1} ^ {L} \text { Layer } _ {q} ^ {l}\right) / \| Z _ {s} \| \| Z _ {q} \|. \tag {6}
$$

From Eq. 6, we can see a cross-match of different layers in the distance calculation:

$$
\tilde {S} = (\sum_ {i = 1} ^ {L} \text { Layer } _ {s} ^ {i}) \cdot \sum_ {j = 1} ^ {L} (\text { Layer } _ {q} ^ {j}) = \sum_ {i = 1} ^ {L} \sum_ {j = 1} ^ {L} (\text { Layer } _ {s} ^ {i} \cdot \text { Layer } _ {q} ^ {j}) \tag {7}
$$

![](images/1dff22b596e7f992dd5c9d125bb712581331e42bd83bef9496bc789da6a76501.jpg)

<details>
<summary>heatmap</summary>

| Dataset | Target Domain Feature | Source Domain Feature |
| :--- | :--- | :--- |
| FSS-1000 | 0.2 | 0 |
| FSS-1000 | 0.3 | 1 |
| FSS-1000 | 0.4 | 2 |
| FSS-1000 | 0.5 | 3 |
| FSS-1000 | 0.6 | 4 |
| FSS-1000 | 0.7 | 5 |
| FSS-1000 | 0.8 | 6 |
| FSS-1000 | 1.1 | 7 |
| FSS-1000 | 1.2 | 8 |
| FSS-1000 | 1.3 | 9 |
| FSS-1000 | 1.4 | 10 |
| FSS-1000 | 1.5 | 11 |
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature: Source Domain Feature
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature:
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 
Source Domain Feature: 

Deepglobe
ISIC2018
Chest X-ray
</details>

Figure 4: Domain similarities between source- and target-domain features extracted from different layers. A brighter color means a higher domain similarity, indicating less overfitting to the source domain and less feature entanglement.

<table><tr><td>Target Dataset</td><td>FSS-1000</td><td>Deepglobe</td><td>ISIC</td><td>ChestX</td></tr><tr><td>Final Output</td><td>0.4288</td><td>0.3135</td><td>0.2527</td><td>0.2856</td></tr><tr><td>Layer-wise Avg.</td><td>0.6107</td><td>0.4988</td><td>0.5074</td><td>0.6612</td></tr><tr><td>Top-12 Avg.</td><td>0.8126</td><td>0.6441</td><td>0.6164</td><td>0.7823</td></tr><tr><td>Bottom-12 Avg.</td><td>0.1407</td><td>0.0130</td><td>0.0473</td><td>0.0163</td></tr></table>

Table 1: Simply shifting cross-matched layers heavily affects domain similarities, inspiring us to handle the entanglement problem by learning the cross-match of layers.

This implies the output of every layer is compared with all other layers. Since all layers are in the same feature space (Fig. 2), it is feasible to compare even the output of the first layer and the last layer, although the comparison results may be meaningless (Fig.3). However, in Eq. 6, we observe the matching process treats all layers equally, which means even the meaningless comparison between two distinct layers will have a non-trivial impact on the final distance.

As the feature entanglement can be viewed as a kind of overfitting to the source domain, which can be represented as the recognition based on meaningless patterns, such meaningless comparisons would lead the model to rely on patterns specific to such comparisons, leading to overfitting. However, Eq. 6 entangles these patterns and comparisons together with equally. Therefore, we hypothesize it is the entanglement in the cross-match of different layers that leads to the entanglement in the semantic features.

Validation of hypothesis. To validate this hypothesis, we use domain similarities between source and target domains to measure the feature entanglement, i.e., feature entanglement leads to overfitting to the source domain, and more overfitting leads to less transferable features across domains, reducing the domain similarity. We follow (Zou et al., 2024a) to take the CKA similarity $^{3}$ to measure the domain similarity. Specifically, we use different ViT layers to extract features from each domains, and then compare features from the source domain and target domains to measure the CKA similarity. Since ViT contains 12 layers, this would lead to $12 \times 12$ CKA values for each source-target domain pair. As shown in Fig. 4, the cross-match deviating from the diagonal shows much lower domain similarities, e.g., the

top-right corner which means matching Layer 0 of the target domain and Layer 11 of the source domain. In Table 1, we also calculate the CKA of the comparison between final outputs (i.e., viewed as the average of Fig. 4) and the layer-wise comparison (i.e., comparing outputs with the same layer ID, the diagonal of Fig. 4). We can see the layer-wise domain similarity is much higher than that of the final output. This verifies that the correct match between layers can lead to higher domain similarity, and therefore less feature entanglement. In other words, the decomposed components (Layers) are well-suited for generalization themselves. It is the cross-match of components that leads to feature entanglement.

Moreover, in Fig. 4, we can also observe a small fraction of cross-matches show higher CKA values than the diagonal (layer-wise) ones. This indicates the patterns captured by each layer are not strictly different from others (Fig. 3), possibly due to the dynamic calculation of the self-attention mechanism (Park & Kim, 2022). To verify it, in Table 1, we simply shift the match between layers, and the domain similarities are even higher than the layer-wise ones, indicating a learnable cross-match may be better than the naive layer-wise match for solving the entanglement problem.

# 2.4. Discussion and conclusion

To handle the feature entanglement problem for CD-FSS, we revisit ViT's inherent structure, which provides a natural decomposition of its features. Since all internal features of ViT layers (components) are in the same feature space due to the residual connection, ViT's final output implicitly combines all component features with the same importance. This also leads to the cross-match with equal importance between all ViT components. By taking the domain similarity as a measure of the source-domain overfitting caused by the entanglement problem, we find it is the meaningless cross-match between ViT components that majorly causes the entanglement problem, where the rational matches are entangled with those meaningless ones by their equal weights. Inspired by these, we aim to handle this problem by learning the cross-match weights of components.

# 3. Method

Building on our analysis of feature entanglement, we propose the concept of self-disentanglement and re-composition. Our framework is illustrated in Fig. 5. We first extract support and query features from different ViT layers, concatenate them along the channel dimension, and feed them into the Orthogonal Space Decoupling (OSD) module for weight allocation and semantic disentanglement. Subsequently, the outputs of OSD are input into the Cross-Pattern Comparison (CPC) module, where the disentangled patterns are compared crossly for the re-composition of patterns. For the re-composition, during source-domain training, score maps are composed with weights from OSD for efficient pattern learning. During target-domain finetuning, the Adaptive Fusion Weight (AFW) is introduced to dynamically learn the comparison weights for efficient adaptation.

# 3.1. Orthogonal Space Decoupling

Since misaligned and correct matches are assigned equal weights during comparison, we need to rectify misaligned matches. This involves two steps: (1) adjusting the semantics of each feature and assigning different weights before comparison, and (2) allocating different weights to each similarity after comparison. Therefore, we propose the OSD module as an explicit global decoupling and weight allocation mechanism. It helps semantic disentanglement by aggregating feature channels, enforcing orthogonal constraints, and assigning appropriate weights to different patterns.

Specifically, given a support and query set, a sequence of L pairs of support and query feature maps $\{(F_{l}^{s}, F_{l}^{q})\}_{l=1}^{L}$ is extracted from various ViT layers. Each representation $F_{l} \in R^{d \times N}$ is reshaped to $F_{l} \in R^{d \times n \times n}$ , where n is the patch number and the d is channel dimension. These support and query patterns are first concatenated along the channel dimension to form a complete representation:

$$
F _ {c o n} ^ {*} = \operatorname{concat} \left(\left\{F _ {l} ^ {*} \right\} _ {l = 1} ^ {L}\right) \tag {8}
$$

where $F_{con}^{*} \in R^{Ld \times n \times n}$ and $*$ denotes that both the support and query patterns undergo the same operation.

Then, these concatenated features are fed into the OSD, where explicit constraints on each pattern channel enable semantic decoupling, while handling weight allocation. The OSD consists of a fully connected layer $W_{in} \in R^{Ld \times r}$ , a convolutional layer $W_{orth} \in R^{r \times r \times 1 \times 1}$ , and a fully connected layer $W_{out} \in R^{r \times Ld}$ . Here, r is low a rank (default set to 8) to save computational resources. The concatenated features are reduced to a low-dimensional orthogonal space, applying orthogonal constraints and allocating weights:

$$
F _ {d o w n} ^ {*} = W _ {i n} (F _ {c o n} ^ {*}); \quad F _ {o r t h} ^ {*} = W _ {o r t h} (F _ {d o w n} ^ {*}) \tag {9}
$$

where $F_{orth}^{*} \in R^{r \times n \times n}$ . Next, we compute the orthogonal regularization (Xie et al., 2017) by reshaping $F_{orth}^{*}$ to $R^{r \times n^{2}}$ and using it as a loss term to constrain the extracted pattern, promoting their disentanglement:

$$
L _ {o r t h} = \left\| F _ {o r t h} F _ {o r t h} ^ {T} - I \right\| _ {F} ^ {2} \tag {10}
$$

Finally, we map $F_{orth}^{*}$ back to the original space and split the concatenated support and query features:

$$
F _ {u p} ^ {*} = W _ {o u t} (F _ {o r t h} ^ {*}); \quad \{F _ {l} ^ {*} \} _ {l = 1} ^ {L} = s p l i t (F _ {u p} ^ {*}) \tag {11}
$$

During source-domain training, OSD is trained jointly with the encoder. During target-domain fine-tuning, $W_{in}$ and $W_{out}$ are frozen, and we fine-tune the compact $W_{orth}$ .

# 3.2. Cross-Pattern Comparison for Re-Composition

Cross Comparison. Based on feature entanglement caused by misaligned matches and the dynamic nature of

![](images/2ecb035f8f3287689a95d781abc7a692f0614f03aca056f675c475311263a9e0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Adaptive Fusion"]
    B --> C["Prior"]
    B --> D["Prior"]
    B --> E["Prior"]
    C --> F["Cross-Pattern Comparison"]
    D --> F
    E --> F
    F --> G["OSD"]
    G --> H["concatenate"]
    H --> I["Block"]
    H --> J["Block"]
    H --> K["..."]
    H --> L["Block"]
    I --> M["Vision Encoder"]
    J --> M
    K --> M
    L --> M
    M --> N["query"]
    N --> O["support"]
    O --> P["MAP"]
    P --> Q["Disentangled Feature"]
```
</details>

✗: learnable in source training and target finetuning

![](images/40133198024cbb923035876119f4ebd00a5a7174249773595f5f5a66df78b1df.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Projector"] --> B["1x1 Conv"]
    B --> C["Layered Structure"]
    C --> D["Projector"]
    D --> E["F^s_con"]
    D --> F["F^q_con"]
    style A fill:#e6f3ff,stroke:#333
    style B fill:#e6f3ff,stroke:#333
    style C fill:#e6f3ff,stroke:#333
    style D fill:#e6f3ff,stroke:#333
    style E fill:#e6f3ff,stroke:#333
    note right of B L_orth
    note left of A Ld
```
</details>

: only learnable in source training

![](images/7d6db3854db3b8023b6cb94625371e7baee330d15485015e4e70ca16e9d05c06.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Cross-Pattern Comparison (CPC)"] --> B["distance"]
    B --> C["Score Map"]
    B --> D["Score Map"]
    B --> E["Score Map"]
    B --> F["..."]
    G["Adaptive Fusion Weight (AFW)"] --> H["source domain × Average Weight"]
    G --> I["target domain × Adaptive Fusion Weight"]
```
</details>

K: only learnable in target finetuning   
Figure 5: Overview of our method. We extract L support and query features from various blocks. These features are concatenated along the channel dimension and fed into OSD to impose orthogonal constraints for weight allocation and semantic decoupling. Outputs of OSD are then fed into the CPC module, where support and query features are compared crossly, yielding $L \times L$ score maps. In source-domain training, these maps are composed with average weights for efficient pattern learning. In target-domain fine-tuning, the AFW dynamically learns the composition weights for efficient adaption.

ViT, we propose the Cross-Pattern Comparison (CPC) module. After semantic decoupling and weight allocation by the OSD, we use mask average pooling (MAP) (Zhang et al., 2020b) to obtain L sets of foreground prototypes $P_{fg} \in R^{L \times d \times 1 \times 1}$ and background prototypes $P_{bg} \in R^{L \times d \times 1 \times 1}$ from support features. These disentangled support prototypes and query features are then input into the CPC module, where they are cross-compared for re-composition.

Specifically, for the query feature sets $F^{q} \in R^{L \times d \times n \times n}$ and the support prototypes $[P_{bg}, P_{fg}]$ , we compute the distance through cross-pairing to obtain the cross-pattern comparison maps (score maps), denote as C:

$$
C _ {b g / f g} = \text { distance } (F ^ {q}, P _ {b g / f g}); \quad C = \text { concat } (C _ {b g}, C _ {f g}) \tag {12}
$$

where C is reshaped to $R^{L^{2}\times2\times n\times n}$ (2 means background and foreground), and the distance can be calculated in various ways; we default to using cosine similarity, while also exploring other distance metrics (see Experiment 4.3):

$$
\text { distance } _ {\cos} = F ^ {q} \cdot P _ {b g / f g} / \| F ^ {q} \| \| P _ {b g / f g} \| \tag {13}
$$

Adaptive Fusion Weight. For re-composite the obtained $L^{2}$ sets of cross-pattern comparison maps, during source-domain training, the comparison maps C are composed with average weights for efficient pattern learning; during target-domain fine-tuning, Adaptive Fusion Weight (AFW), which includes background and foreground weight, is introduced for efficient adaption. The reason for not using AFW during training is that it is a small parameter matrix of size $L^{2} \times 2$ (just 288 for ViT-B), where the “2” corresponds to the composition weight for $C_{bg}$ and $C_{fg}$ . If trained jointly with the encoder in the source domain, it is prone to overfitting the source data. As a lightweight module, directly introducing it in the target domain allows for flexible adjustment and adaptation based on the target domain, resulting in better performance. The formula is as follows:

$$
\text { source }: \quad C _ {\text { fusion }} = \frac {\sum_ {l = 0} ^ {L ^ {2}} C (l)}{L ^ {2}} \tag {14}
$$

$$
\text { target }: \quad C _ {\text { fusion }} = \frac {W _ {A F W} \otimes C}{L ^ {2}} \tag {15}
$$

where $C(l) \in \mathbb{R}^{2 \times n \times n}, \otimes$ indicates the element-wise multiplication. The final prediction pred as describe:

$$
\operatorname{pred} = \operatorname{argmax} \left(\zeta_ {l} \left(C _ {\text {fusion}}\right)\right)) \tag {16}
$$

where $\zeta_l(*)$ is a function that bilinearly interpolates $C_{fusion}$ to the spatial size of the input image by expanding along the spatial dimension, i.e., $\zeta_l: \mathbb{R}^{2 \times n \times n} \to \mathbb{R}^{2 \times h \times w}$ . Here, $h$ and $w$ are the image's height and width.

Loss Strategy. During both source-domain training and target-domain fine-tuning, we employ the standard Binary Cross-Entropy (BCE) loss $L_{BCE}$ , with the orthogonal loss $L_{orth}$ from OSD added as a regularization term to promote

<table><tr><td rowspan="2">Method</td><td rowspan="2">Mark</td><td rowspan="2">Backbone</td><td colspan="2">FSS-1000</td><td colspan="2">Deepglobe</td><td colspan="2">ISIC</td><td colspan="2">Chest X-ray</td><td colspan="2">Average</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>PANet (Wang et al., 2019)</td><td>ECCV-20</td><td>Res-50</td><td>69.15</td><td>71.68</td><td>36.55</td><td>45.43</td><td>25.29</td><td>33.99</td><td>57.75</td><td>69.31</td><td>47.19</td><td>55.10</td></tr><tr><td>RPMMs (Yang et al., 2020a)</td><td>ECCV-20</td><td>Res-50</td><td>65.12</td><td>67.06</td><td>12.99</td><td>13.47</td><td>18.02</td><td>20.04</td><td>30.11</td><td>30.82</td><td>31.56</td><td>32.85</td></tr><tr><td>PFENet (Tian et al., 2020)</td><td>TPAMI-20</td><td>Res-50</td><td>70.87</td><td>70.52</td><td>16.88</td><td>18.01</td><td>23.50</td><td>23.83</td><td>27.22</td><td>27.57</td><td>34.62</td><td>34.98</td></tr><tr><td>RePRI (Boudiaf et al., 2021)</td><td>CVPR-21</td><td>Res-50</td><td>70.96</td><td>74.23</td><td>25.03</td><td>27.41</td><td>23.27</td><td>26.23</td><td>65.08</td><td>65.48</td><td>46.09</td><td>48.34</td></tr><tr><td>HSNet (Min et al., 2021)</td><td>ICCV-21</td><td>Res-50</td><td>77.53</td><td>80.99</td><td>29.65</td><td>35.08</td><td>31.20</td><td>35.10</td><td>51.88</td><td>54.36</td><td>47.57</td><td>51.38</td></tr><tr><td>PATNet (Lei et al., 2022)</td><td>ECCV-22</td><td>Res-50</td><td>78.59</td><td>81.23</td><td>37.89</td><td>42.97</td><td>41.16</td><td>53.58</td><td>66.61</td><td>70.20</td><td>56.06</td><td>61.99</td></tr><tr><td>PATNet (Lei et al., 2022)</td><td>ECCV-22</td><td>ViT-base</td><td>72.03</td><td>-</td><td>22.37</td><td>-</td><td>44.25</td><td>-</td><td>76.43</td><td>-</td><td>53.77</td><td>-</td></tr><tr><td>PerSAM (Zhang et al., 2024)</td><td>ICLR-24</td><td>ViT-base</td><td>60.92</td><td>66.53</td><td>36.08</td><td>40.65</td><td>23.27</td><td>25.33</td><td>29.95</td><td>30.05</td><td>37.56</td><td>40.64</td></tr><tr><td>APM (Tong et al., 2024)</td><td>NeurIPS-24</td><td>Res-50</td><td>79.29</td><td>81.83</td><td>40.86</td><td>44.92</td><td>41.71</td><td>51.16</td><td>78.25</td><td>82.81</td><td>60.03</td><td>65.18</td></tr><tr><td>ABCDFSS (Herzog, 2024)</td><td>CVPR-24</td><td>Res-50</td><td>74.60</td><td>76.20</td><td>42.60</td><td>45.70</td><td>45.70</td><td>53.30</td><td>79.80</td><td>81.40</td><td>60.67</td><td>64.97</td></tr><tr><td>DRA (Su et al., 2024)</td><td>CVPR-24</td><td>Res-50</td><td>79.05</td><td>80.40</td><td>41.29</td><td>50.12</td><td>40.77</td><td>48.87</td><td>82.35</td><td>82.31</td><td>60.86</td><td>65.42</td></tr><tr><td>APSeg (He et al., 2024)</td><td>CVPR-24</td><td>ViT-base</td><td>79.71</td><td>81.90</td><td>35.94</td><td>39.98</td><td>45.43</td><td>53.98</td><td>84.10</td><td>84.50</td><td>61.30</td><td>65.09</td></tr><tr><td>SDRC (Ours)</td><td>Ours</td><td>ViT-base</td><td>80.31</td><td>82.55</td><td>43.15</td><td>46.83</td><td>46.57</td><td>55.02</td><td>82.86</td><td>84.79</td><td>63.22</td><td>67.30</td></tr></table>

Table 2: Mean-IoU of 1-shot and 5-shot results on the CD-FSS benchmark. The best and second-best results are highlighted in bold and underlined, respectively. The comparison with domain transfer methods is provided in Appendix C.

semantic decoupling. Notably, in target-domain finetuning, we do not access the query data. Instead, we treat the support as the query for the calculation of $L_{BCE}$ and $L_{orth}$ . We optimize the model using the final loss L:

$$
L _ {B C E} = B C E (p r e d, y) \tag {17}
$$

$$
L = L _ {B C E} + \lambda L _ {\text { orth }} \tag {18}
$$

where y is the query ground truth in source-domain training and denotes support mask in target-domain fine-tuning, and $\lambda$ is a hyperparameter, with a default value of 0.1, that adjusts the weight of the orthogonal loss $L_{orth}$ .

# 4. Experiments

# 4.1. Dataset and Implementation Details

The benchmark proposed by PATNet (Lei et al., 2022) is adopted, following the same data preprocessing procedures as the dataset it employs. PASCAL VOC 2012 (Everingham et al., 2010) with SBD (Hariharan et al., 2011) augmentation serves as the training dataset. FSS-1000 (Li et al., 2020), DeepGlobe (Demir et al., 2018), ISIC2018 (Codella et al., 2019; Tschandl et al., 2018), and Chest X-ray (Candemir et al., 2013; Jaeger et al., 2013) are considered as target domains for evaluation. See the Appendix B for details.

Following previous prototype-based work (Dong & Xing, 2018; Zhang et al., 2020b; Wang et al., 2019), we built a lightweight encoder-only baseline that employs ViT-B (Dosovitskiy et al., 2020) pre-trained on ImageNet (Russakovsky et al., 2015) as the backbone network. The hyperparameter $\lambda$ , which adjusts the weight of the orthogonal loss, is set to 0.1, and the rank $r$ of the OSD module is set to 8. For other details, please refer to the appendix.

# 4.2. Comparison with State-of-the-Art Works

In Table2, we compare our method with existing works, where we achieve a significant improvement under both 1-shot and 5-shot settings. Specifically, we surpass the performance of the state-of-the-art by 1.92% and 1.88% under 1-shot and 5-shot settings, respectively. Notably, APSeg also utilizes ViT as its backbone; however, its parameter count is significantly larger than ours due to its SAM-based encoder-decoder architecture, while we employ an encoder-only structure. Consequently, our method not only surpasses APSeg in performance but also entails substantially lower computational costs. Furthermore, we showcase the qualitative results of our method in 1-way 1-shot segmentation, as depicted in Figure 6. These results demonstrate the substantial enhancement in generalization ability across significant domain gaps while maintaining a comparable accuracy in the face of similar domain shifts using our method.

![](images/e03f84b265e7cd36c91e451802b4aa91eda33560109802f62211812a49907849.jpg)

<details>
<summary>text_image</summary>

FSS-1000
Deepglobe
ISIC
Chest X-ray
support
query
baseline
ours
</details>

Figure 6: Qualitative results of our model for 1-shot setting. The support labels are highlighted in blue, while the predictions and ground truth of query images are presented in red.

# 4.3. Ablation Study

Impact of each design. As shown in Table 3, introducing the CPC, which serves as the foundation for the other designs, improved the average mIoU by 9.62% and 9.04%

<table><tr><td rowspan="2">CPC</td><td rowspan="2">AFW</td><td rowspan="2">OSD</td><td colspan="2">FSS-1000</td><td colspan="2">Deepglobe</td><td colspan="2">ISIC</td><td colspan="2">Chest X-ray</td><td colspan="2">Average</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td></td><td></td><td></td><td>77.80</td><td>80.69</td><td>33.18</td><td>37.39</td><td>36.99</td><td>41.20</td><td>51.54</td><td>55.26</td><td>49.88</td><td>53.64</td></tr><tr><td>√</td><td></td><td></td><td>79.20</td><td>81.46</td><td>41.71</td><td>43.25</td><td>42.99</td><td>47.97</td><td>74.81</td><td>78.05</td><td>59.50</td><td>62.68</td></tr><tr><td>√</td><td>√</td><td></td><td>79.22</td><td>82.02</td><td>42.59</td><td>45.22</td><td>43.11</td><td>50.73</td><td>80.36</td><td>80.36</td><td>61.32</td><td>65.22</td></tr><tr><td>√</td><td></td><td>√</td><td>80.05</td><td>82.18</td><td>41.87</td><td>44.68</td><td>45.63</td><td>52.61</td><td>75.45</td><td>78.32</td><td>60.75</td><td>64.45</td></tr><tr><td>√</td><td>√</td><td>√</td><td>80.31</td><td>82.55</td><td>43.15</td><td>46.83</td><td>46.57</td><td>55.02</td><td>82.86</td><td>84.79</td><td>63.22</td><td>67.30</td></tr></table>

Table 3: Detailed ablation study results of our various designs on four target datasets under 1-shot setting and 5-shot setting.

<table><tr><td rowspan="2">Metric</td><td colspan="2">baseline</td><td colspan="2">ours</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>Euclidean</td><td>48.92</td><td>53.07</td><td>62.49</td><td>66.53</td></tr><tr><td>Dot</td><td>49.18</td><td>53.03</td><td>62.75</td><td>66.58</td></tr><tr><td>EMD</td><td>50.02</td><td>53.23</td><td>63.37</td><td>67.01</td></tr><tr><td>Cosine</td><td>49.88</td><td>53.64</td><td>63.22</td><td>67.30</td></tr></table>

Table 4: Impact of the different distance metric for CPC.

for the 1-shot and 5-shot settings, respectively. The OSD is an explicit global decoupling module, guiding layers to focus on the patterns emphasized by the current layer and promoting disentanglement. Meanwhile, the AFW adjusts the weights of comparison maps for each pattern based on different target domains, resulting in more accurate segmentation predictions. These results demonstrate that each design in our approach significantly enhances performance.

Impact of different distance metric. Our method is highly versatile, as we evaluated it using various distance metrics, including Euclidean distance (Snell et al., 2017), cosine similarity (Vinyals et al., 2016), dot product (Chen et al., 2019), and EMD (Zhang et al., 2020a), as shown in Table 4. Regardless of the distance metric, our method consistently outperforms the baseline with significant gains. Notably, under 1-shot setting, EMD is the best distance metric, while cosine similarity performs best under 5-shot setting.

Effectiveness of cross-comparison. In Table 5, we report the performance of both the position-wise pattern comparison and cross-pattern comparison, confirming the effectiveness of the cross-layer strategy. Due to the dynamic nature of ViT and intra-class variations, features extracted from different layers may still form correct matches. Therefore, the CPC effectively compares these patterns for re-composition.

<table><tr><td>w/o AFW&amp;OSD</td><td>1-shot</td><td>5-shot</td></tr><tr><td>baseline</td><td>49.88</td><td>53.64</td></tr><tr><td>position-wise comparison</td><td>55.14</td><td>59.39</td></tr><tr><td>cross-pattern comparison</td><td>59.50</td><td>62.68</td></tr></table>

Table 5: Validate the effectiveness of cross-comparison.

Impact of parameters on performance. As shown in Table 6, the OSD and the AFW are lightweight modules with minimal parameters. The OSD is trained jointly with the encoder on the source domain, but on the target dataset, only the $W_{orth}$ of OSD is fine-tuned, reducing computational

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2" colspan="2">Encoder (ViT-B)</td><td rowspan="2">AFW</td><td colspan="3">OSD (rank=8)</td></tr><tr><td> $W_{in}$ </td><td> $W_{orth}$ </td><td> $W_{out}$ </td></tr><tr><td colspan="2">Params(K)</td><td colspan="2"> $8.6 \times 10^4$ </td><td>0.288</td><td>6.144</td><td>0.064</td><td>6.144</td></tr><tr><td>rank</td><td>64</td><td>32</td><td>16</td><td>8</td><td>4</td><td>2</td><td></td></tr><tr><td>mIoU</td><td>62.61</td><td>63.43</td><td>63.25</td><td>63.22</td><td>61.73</td><td>60.39</td><td></td></tr></table>

Table 6: Analysis the impact of parameters on performance.

costs while ensuring effective global decoupling for different domains. The AFW is not involved in source-domain training and is directly adopted during the target stage. We also explored the impact of the OSD's rank value on performance. With rank=8, the performance is slightly lower than 32, but the parameter count is only $\frac{1}{4}$ , so we set rank to 8.

Re-composition strategy for comparison maps. During source domain training, we composite the comparison maps with average weights. During target domain fine-tuning, the AFW is introduced to adaptively learn the combination weights for efficient adaption. As shown in Table 7, training AFW jointly with the encoder during source-domain training does not achieve better results compared to adapting it directly on the target domain. Training AFW in the source domain can lead to weights being biased toward the source data, so we opt for fine-tuning directly on the target domain.

<table><tr><td>AFW</td><td>1-shot</td><td>5-shot</td></tr><tr><td>w/ source-domain training</td><td>61.01</td><td>64.93</td></tr><tr><td>w/o source-domain training</td><td>63.22</td><td>67.30</td></tr></table>

Table 7: Validation of the training strategy for AFW.

# 4.4. Self-Disentanglement and Re-Composition

Disentangle Feature by decomposing ViT structure: To validate our approach, we visualize the features of each layer of ViT-B, as shown in Fig. 7. Since ViT-B has 12 layers, the features are grouped in pairs (with 6 layers per row). The results reveal that the extracted features from different layers capture distinct semantic information, focusing on elements such as the fish tail, body, fins, and outline. The

<table><tr><td rowspan="2">OSD</td><td colspan="2">FSS-1000</td><td colspan="2">Deepglobe</td><td colspan="2">ISIC</td><td colspan="2">ChestX</td></tr><tr><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>support MI</td><td>0.6444</td><td>0.6059</td><td>0.8599</td><td>0.7958</td><td>0.8738</td><td>0.7890</td><td>0.9095</td><td>0.6506</td></tr><tr><td>query MI</td><td>0.6437</td><td>0.6051</td><td>0.8593</td><td>0.7962</td><td>0.8712</td><td>0.7829</td><td>0.9110</td><td>0.6517</td></tr></table>

Table 8: The MI between features; lower values correspond to lower correlations (better disentanglement).

![](images/2524662d709432893cb71da0c4eb0bec17d01ad53f7e4c0f7c71a9983141e8a2.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 grayscale thermal or fluorescence images showing fish and fish silhouettes, no text or symbols present
</details>

Figure 7: Visualization of features extracted from different layers of ViT demonstrates the feasibility of disentangling the entangled patterns by decomposing the ViT structure.

result demonstrates that ViT inherently has the potential for semantic disentanglement and supports the feasibility of our insight: decompose the entangled semantic patterns through a structural decomposition of the ViT output.

Further semantic disentanglement by OSD: Mutual information reflects the correlation between features. Lower mutual information indicates weaker correlations, meaning the features are more semantically independent. Therefore, we measure the average mutual information (MI) separately between the support features and between the query features to verify that OSD promotes further semantic disentanglement. As shown in Table 8, after applying OSD, the mutual information between both support features and between query features decreases, verifying OSD's decoupling.

Cross-Pattern Comparison for re-composition: In Fig. 8, we visualize the results of cross-comparison (some samples) and re-composition. Through cross-comparison, the model focuses on different regions of the segmented object. These comparison maps are then re-composed, allowing the model to accurately identify the complete segmentation region.

Re-composition by Adaptive Fusion Weight: We visualize the AFW as heatmaps, where brighter colors indicate higher re-composition weights. The result in Fig.9 highlights that: 1) AFW learns different re-composition weights for each domain; 2) The highest weight learned by AFW is not necessarily at the diagonal of the matrix, suggesting that cross-matching and re-composition are more effective than position-wise matching. Additionally, we observe an interesting phenomenon: without any constraints imposed on AFW, the learned re-composition weights for foreground and background tend to be mutually exclusive, a trend particularly noticeable in the Deepglobe and ISIC datasets.

# 5. Analysis of Performance and Efficiency

Orthogonal Loss Weight As shown in Table 9, we validated the impact of the weight of the orthogonal loss on performance. The results indicate that the optimal choice of the weight falls within a wide interval, which means the tuning of this hyper-parameter is not difficult. Additionally, we used the same orthogonal loss weight in the Swin Transformer architecture as we did in the ViT architecture (Table 17). The performance indicates that our method design is not sensitive to this weight.

![](images/8d299ceeba6ae1d969a268135280e57a4e555ea1c2370921aa42c772cf7a3cd8.jpg)

<details>
<summary>text_image</summary>

query image & mask
some examples of the comparison maps for re-composition
composition result
</details>

Figure 8: The heatmaps of some examples of cross-comparison maps and the results after re-composition.

![](images/518d028fdc176ebcd2e34dedd0783de970c34dc992c7f278e5e04a63839dd4de.jpg)

<details>
<summary>heatmap</summary>

| Dataset | Background Weights - query means | Background Weights - support prototype | Background Weights - query means | Foreground Weights - query means | Foreground Weights - support prototype | Foreground Weights - query means | FSS-1000 - query means | Deepglobe - query means | ISIC2018 - query means | Chest X-ray - query means |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FSS-1000 | 13.10 | 0.00 | 0.00 | 13.10 | 0.00 | 0.00 | 13.10 | 0.00 | 13.10 | 0.00 |
| Deepglobe | 13.10 | 0.00 | 0.01 | 13.10 | 0.00 | 0.04 | 13.10 | 0.06 | 13.10 | 0.06 |
| ISIC2018 | 13.10 | 0.00 | 0.02 | 13.10 | 0.02 | 0.04 | 13.10 | 0.04 | 13.10 | 0.04 |
| Chest X-ray | 13.10 | 0.00 | 0.02 | 13.10 | 0.02 | 0.04 | 13.10 | 0.06 | 13.10 | 0.06 |
| FSS-1000 - support prototype | -13.15 | 2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -2.75 |
| Deepglobe - support prototype | -13.15 | 2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -2.75 |
| ISIC2018 - support prototype | -13.15 | 2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -2.75 |
| Chest X-ray - support prototype | -13.15 | 2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -13.15 | -2.75 | -2.75 | -2.75 |
| FSS-1000 - support prototype | 9.95 | 2.75 | -2.75 | -9.95 | -2.75 | -2.75 | -9.95 | -2.75 | -9.95 | -2.75 |
| Deepglobe - support prototype | 9.95 | 2.75 | -2.75 | -9.95 | -2.75 | -2.75 | -9.95 | -2.75 | -9.95 | -2.75 |
| ISIC2018 - support prototype | 9.95 | 2.75 | -2.75 | -9.95 | -2.75 | -2.75 | -9.95 | -2.75 | -9.95 | -2.75 |
| Chest X-ray - support prototype | 9.95 | 2.75 | -2.75 | -9.95 | -2.75 | -2.75 | -9.95 | -2.75 | -9.95 | -2.75 |
| FSS-1000 - support prototype & weather method: FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1 |
| Deepglobe: FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1000_FSS-1<nl>
<fcel>ISIC2018: FSS-100<fcel>-13.1<fcel>-23<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<nl>
<fcel>Chest X-ray: FSS-1<fcel>-13.1<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<fcel>-6<nl>
</details>

Figure 9: Visualization of AFW on four target datasets, which includes background and foreground weight. A brighter color indicates a higher re-composition weight.

<table><tr><td>Orth. Loss Weight</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.5</td></tr><tr><td>1-shot Avg mIoU</td><td>62.59</td><td>63.01</td><td>63.22</td><td>63.18</td><td>62.87</td></tr></table>

Table 9: Impact of orthogonal loss weight on performance.

Impact of Background Prototypes's Number Most current prototype-based methods (e.g., PANet, SSP) utilize a single background prototype to model background patterns, and have demonstrated good performance in both recent works and our experiments. Indeed, it is preferable to consider different background classes for different images. Therefore, as shown in Table 10, we further introduce clustering to obtain multiple background prototypes (Yang et al., 2020b; Li et al., 2021a). By comparing the single-background prototype to multi-background prototypes, we observe a slight performance improvement. However, the gains are not substantial enough to justify the additional computational overhead of clustering.

<table><tr><td></td><td>FSS1000</td><td>Deepglobe</td><td>ISIC</td><td>ChestX</td><td>Mean</td></tr><tr><td>Single BG prototype</td><td>80.31</td><td>43.15</td><td>46.57</td><td>82.86</td><td>63.22</td></tr><tr><td>Multi BG prototype</td><td>80.96</td><td>43.53</td><td>46.78</td><td>83.09</td><td>63.59</td></tr></table>

Table 10: Comparison between single and multiple background prototypes under 1-shot setting.

Computational Efficiency As shown in Table 11, we compared our method with PATNet (Lei et al., 2022), HSNet (Min et al., 2021), and SSP (Fan et al., 2022). Our approach exhibits greater computational efficiency than the other methods, as it does not require additional networks and instead leverages the inherent structure of the ViT for feature separation.

<table><tr><td></td><td>PATNet</td><td>HSNet</td><td>SSP</td><td>Ours</td></tr><tr><td>FLOPs (G)</td><td>22.63</td><td>20.11</td><td>18.97</td><td>18.86</td></tr></table>

Table 11: Analysis of computational efficiency.

# 6. Theoretical Analysis of the Effectiveness of ViT Disentanglement

In cross-domain few-shot segmentation tasks, models need to transfer knowledge from the source domain S with abundant annotations to the target domain T with limited data. Let H represent the hypothesis space of the segmentation model. The upper bound of the generalization error for target domain risk $\epsilon_{\mathcal{T}}(h)$ is defined as:

$$
\epsilon_ {\mathcal {T}} (h) \leq \epsilon_ {\mathcal {S}} (h) + d _ {\mathcal {H}} (\mathcal {S}, \mathcal {T}) + \lambda , \tag {19}
$$

where h denotes features extracted by the encoder, $\epsilon_{\mathcal{S}}(h)$ is the source domain risk, $d_{\mathcal{H}}(\mathcal{S},\mathcal{T})$ represents the H-divergence (domain gap) between the source and target domains, and $\lambda$ is the irreducible ideal joint risk.

Our approach reduces $\epsilon_{\mathcal{T}}(h)$ through two mechanisms:

1) Adaptive Fusion Weights (AFW): adaptively assigns higher weights to semantically appropriate matches, leading to better alignment (as confirmed by the experiments on the source domain in Answer 3), thereby optimizing source domain output and reducing $\epsilon_{S}(h)$ .   
2) Domain-Invariant Component Isolation: minimizes $d_{\mathcal{H}}(\mathcal{S}, \mathcal{T})$ by isolating domain-invariant patterns (e.g., object parts) via:

$$
d _ {\mathcal {H}} (\mathcal {S}, \mathcal {T}) \approx \sum_ {i = 1} ^ {L} \sum_ {j = 1} ^ {L} w _ {i j} d _ {\mathcal {H}} ^ {(i j)} (\mathcal {S}, \mathcal {T}) \leq \sum_ {i = 1} ^ {L} \sum_ {j = 1} ^ {L} d _ {\mathcal {H}} ^ {(i j)} (\mathcal {S}, \mathcal {T}), \tag {20}
$$

where $d_{\mathcal{H}}^{(ij)}$ denotes the inter-layer domain discrepancy. By leveraging the self-disentangling property and the orthogonal constraints from the OSD module, inappropriate matches are learned to have small $w_{ij}$ , reducing the inter-layer mutual information $I[h_{i}h_{j}^{T}]$ , thereby tightening the boundary.

# 7. Related Work

Cross-Domain Few-Shot Segmentation (CD-FSS) CD-FSS has received increasing attention recently. PAT-Net (Lei et al., 2022) establishes a CD-FSS benchmark and proposes feature transformation layers to map domain-specific features into domain-agnostic ones for fast adaptation. APSeg (He et al., 2024), based on SAM (Kirillov et al., 2023), introduces a novel auto-prompt network for guiding features in cross-domain segmentation. DRA (Su et al., 2024) adopts a compact adapter to align diverse target domain features with the source domain, while ABCDFSS (Herzog, 2024) introduces tiny adaptors that learn to refine features at test-time only. APM (Tong et al., 2024) proposes a lightweight frequency masker to achieve feature enhancement. These methods focus on optimizing encoders' final features to obtain a better representation. In contrast, our approach focuses on decomposing ViT's final features based on the natural decomposition of ViT's structure, and re-composing them for better comparison.

Feature Disentanglement Learning (FDL) FDL aims to learn an interpretable representation for image variants, which has long been a popular solution for addressing domain shifts. InfoGAN (Chen et al., 2016) maximizes mutual information to learn disentangled representations in an unsupervised manner. Disentangled-VAE (Li et al., 2021b) excavate category-distilling information from visual and semantic features for generalized zero-shot learning. DFR (Cheng et al., 2023) separates discriminative features from class-irrelevant components. However, these methods require additional complex VAE-Discriminator networks, resulting in significant computational overhead. In contrast, our approach does not introduce any extra branch networks. Instead, it leverages feature space consistency across different layers of the ViT to extract distinct patterns, decomposing the entangled semantic patterns through a structural decomposition of the ViT output from a novel perspective.

# 8. Conclusion

In this paper, we analyze and interpret the feature entanglement problem from a novel aspect of the natural decomposition of ViT. Based on it, we propose self-disentanglement and re-composition for CD-FSS. Experiments show our effectiveness and achieve a new state-of-the-art in CD-FSS.

# Acknowledgements

This work is supported by the National Key Research and Development Program of China under grant 2024YFC3307900; the National Natural Science Foundation of China under grants 62206102, 62436003, 62376103 and 62302184; the National Natural Science Foundation of China under grants 62402015; the Postdocotoral Fellowship Program of CPSF under grants GZB20230024;

the China Postdoctoral Science Foundation under grant 2024M750100; Major Science and Technology Project of Hubei Province under grant 2024BAA008; Hubei Science and Technology Talent Service Project under grant 2024DJC078; and Ant Group through CCF-Ant Research Fund. The computation is completed in the HPC Platform of Huazhong University of Science and Technology.

# Impact Statement

Given current interpretations of ViT structures, we find a natural decomposition exists in features extracted by ViT, which inspires us to propose the concept of self-disentanglement and re-composition of ViT features. Our research can also be applied in other fields such as transfer learning and domain adaptation. What's more, our method also provides an interpretable perspective and theoretical foundation for the feature disentanglement from the perspective of ViT's self-decomposition. Future research will aim to broaden our evaluations to encompass a wider range of target domains, enhancing our understanding of their performance in various real-world scenarios.

# References

Boudiaf, M., Kervadec, H., Masud, Z. I., Piantanida, P., Ben Ayed, I., and Dolz, J. Few-shot segmentation without meta-learning: A good transductive inference is all you need? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13979–13988, 2021.   
Candemir, S., Jaeger, S., Palaniappan, K., Musco, J. P., Singh, R. K., Xue, Z., Karargyris, A., Antani, S., Thoma, G., and McDonald, C. J. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE transactions on medical imaging, 33(2):577–590, 2013.   
Cao, Y., Liu, Y., Chen, Z., Shi, G., Wang, W., Zhao, D., and Lu, T. Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829, 2024.   
Chen, W.-Y., Liu, Y.-C., Kira, Z., Wang, Y.-C. F., and Huang, J.-B. A closer look at few-shot classification. In International Conference on Learning Representations, 2019.   
Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., and Abbeel, P. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. Advances in neural information processing systems, 29, 2016.   
Cheng, H., Wang, Y., Li, H., Kot, A. C., and Wen, B. Disentangled feature representation for few-shot image clas-

sification. IEEE transactions on neural networks and learning systems, 2023.

Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368, 2019.

Demir, I., Koperski, K., Lindenbaum, D., Pang, G., Huang, J., Basu, S., Hughes, F., Tuia, D., and Raskar, R. Deepglobe 2018: A challenge to parse the earth through satellite images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 172–181, 2018.

Dong, N. and Xing, E. P. Few-shot semantic segmentation with prototype learning. In BMVC, volume 3, 2018.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010.

Fan, Q., Pei, W., Tai, Y.-W., and Tang, C.-K. Self-support few-shot semantic segmentation. In European Conference on Computer Vision, pp. 701–719. Springer, 2022.

Gandelsman, Y., Efros, A. A., and Steinhardt, J. Interpreting clip's image representation via text-based decomposition. In The Twelfth International Conference on Learning Representations.

Hariharan, B., Arbeláez, P., Bourdev, L., Maji, S., and Malik, J. Semantic contours from inverse detectors. In 2011 international conference on computer vision, pp. 991–998. IEEE, 2011.

He, W., Zhang, Y., Zhuo, W., Shen, L., Yang, J., Deng, S., and Sun, L. Apseg: Auto-prompt network for cross-domain few-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23762–23772, 2024.

Herzog, J. Adapt before comparison: A new perspective on cross-domain few-shot segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23605–23615, 2024.

Jaeger, S., Karargyris, A., Candemir, S., Folio, L., Siegelman, J., Callaghan, F., Xue, Z., Palaniappan, K., Singh,

R. K., Antani, S., et al. Automatic tuberculosis screening using chest radiographs. IEEE transactions on medical imaging, 33(2):233–245, 2013.   
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023.   
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Similarity of neural network representations revisited. In International Conference on Machine Learning, pp. 3519–3529. PMLR, 2019.   
Lei, S., Zhang, X., He, J., Chen, F., Du, B., and Lu, C.-T. Cross-domain few-shot semantic segmentation. In European Conference on Computer Vision, pp. 73–90. Springer, 2022.   
Li, G., Jampani, V., Sevilla-Lara, L., Sun, D., Kim, J., and Kim, J. Adaptive prototype learning and allocation for few-shot segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8334–8343, 2021a.   
Li, X., Wei, T., Chen, Y. P., Tai, Y.-W., and Tang, C.-K. Fss-1000: A 1000-class dataset for few-shot segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2869–2878, 2020.   
Li, X., Xu, Z., Wei, K., and Deng, C. Generalized zero-shot learning via disentangled representation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 1966–1974, 2021b.   
Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2117–2125, 2017.   
Liu, Y., Zou, Y., Li, Y., and Li, R. The devil is in low-level features for cross-domain few-shot segmentation. arXiv preprint arXiv:2503.21150, 2025.   
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., and Kipf, T. Object-centric learning with slot attention. Advances in neural information processing systems, 33:11525–11538, 2020.

Long, J., Shelhamer, E., and Darrell, T. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431–3440, 2015.   
Min, J., Kang, D., and Cho, M. Hypercorrelation squeeze for few-shot segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6941–6952, 2021.   
Nie, J., Xing, Y., Zhang, G., Yan, P., Xiao, A., Tan, Y.-P., Kot, A. C., and Lu, S. Cross-domain few-shot segmentation via iterative support-query correspondence mining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3380–3390, 2024.   
Park, N. and Kim, S. How do vision transformers work? In 10th International Conference on Learning Representations, ICLR 2022, 2022.   
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015.   
Seitzer, M., Horn, M., Zadaianchuk, A., Zietlow, D., Xiao, T., Simon-Gabriel, C.-J., He, T., Zhang, Z., Schölkopf, B., Brox, T., et al. Bridging the gap to real-world object-centric learning. arXiv preprint arXiv:2209.14860, 2022.   
Shaban, A., Bansal, S., Liu, Z., Essa, I., and Boots, B. One-shot learning for semantic segmentation. arXiv preprint arXiv:1709.03410, 2017.   
Snell, J., Swersky, K., and Zemel, R. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017.   
Su, J., Fan, Q., Pei, W., Lu, G., and Chen, F. Domain-rectifying adapter for cross-domain few-shot segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24036–24045, 2024.   
Tian, Z., Zhao, H., Shu, M., Yang, Z., Li, R., and Jia, J. Prior guided feature enrichment network for few-shot segmentation. IEEE transactions on pattern analysis and machine intelligence, 44(2):1050–1065, 2020.   
Tong, J., Zou, Y., Li, Y., and Li, R. Lightweight frequency masker for cross-domain few-shot semantic segmentation. Advances in Neural Information Processing Systems, 37:96728–96749, 2024.   
Tschandl, P., Rosendahl, C., and Kittler, H. The ham10000 dataset, a large collection of multi-source dermatoscopic

images of common pigmented skin lesions. Scientific data, 5(1):1–9, 2018.   
Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.   
Wang, K., Liew, J. H., Zou, Y., Zhou, D., and Feng, J. Panet: Few-shot image semantic segmentation with prototype alignment. In proceedings of the IEEE/CVF international conference on computer vision, pp. 9197–9206, 2019.   
Xie, D., Xiong, J., and Pu, S. All you need is beyond a good init: Exploring better solution for training extremely deep convolutional neural networks with orthonormality and modulation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6176–6185, 2017.   
Yang, B., Liu, C., Li, B., Jiao, J., and Ye, Q. Prototype mixture models for few-shot semantic segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pp. 763–778. Springer, 2020a.   
Yang, B., Liu, C., Li, B., Jiao, J., and Ye, Q. Prototype mixture models for few-shot semantic segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pp. 763–778. Springer, 2020b.   
Zhang, C., Cai, Y., Lin, G., and Shen, C. Deepemd: Few-shot image classification with differentiable earth mover's distance and structured classifiers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12203–12213, 2020a.   
Zhang, R., Jiang, Z., Guo, Z., Yan, S., Pan, J., Dong, H., Qiao, Y., Gao, P., and Li, H. Personalize segment anything model with one shot. In The Twelfth International Conference on Learning Representations, 2024.   
Zhang, X., Wei, Y., Yang, Y., and Huang, T. S. Sg-one: Similarity guidance network for one-shot semantic segmentation. IEEE transactions on cybernetics, 50(9):3855–3865, 2020b.   
Zhao, H., Shi, J., Qi, X., Wang, X., and Jia, J. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2881–2890, 2017.   
Zhu, J., Wang, H., and Shi, M. Multi-modal large language model enhanced pseudo 3d perception framework for visual commonsense reasoning. IEEE Transactions on Circuits and Systems for Video Technology, 2024.

Zou, Y., Zhang, S., Li, Y., and Li, R. Margin-based few-shot class-incremental learning with class-level overfitting mitigation. Advances in neural information processing systems, 35:27267–27279, 2022.   
Zou, Y., Liu, Y., Hu, Y., Li, Y., and Li, R. Flatten long-range loss landscapes for cross-domain few-shot learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23575–23584, 2024a.   
Zou, Y., Zhang, S., Zhou, H., Li, Y., and Li, R. Compositional few-shot class-incremental learning. In International Conference on Machine Learning, pp. 62964–62977. PMLR, 2024b.

# Appendix for Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation

# A. Centered Kernel Alignment (CKA)

Centered Kernel Alignment (CKA) (Kornblith et al., 2019) is a widely used metric for measuring the similarity between two data representations (Zou et al., 2022; 2024b; Liu et al., 2025). It normalizes the Hilbert-Schmidt Independence Criterion (HSIC) to mitigate scale differences and ensure a more stable similarity measure.

HSIC quantifies dependence between two sets of features in a reproducing kernel Hilbert space (RKHS). For centered feature matrices X and Y, Equation (1) gives:

$$
\frac {1}{(n - 1) ^ {2}} \mathrm{tr} (\mathbf {X X} ^ {\top} \mathbf {Y Y} ^ {\top}) = | \mathrm{cov} (\mathbf {X} ^ {\top}, \mathbf {Y} ^ {\top}) | F ^ {2}. \tag {21}
$$

HSIC extends this to kernel-based methods, where $K_{ij} = k(x_i, x_j)$ and $\mathbf{L}_{ij} = l(y_i, y_j)$ define kernel matrices. The empirical HSIC estimator is:

$$
\mathrm{HSIC} (\mathbf {K}, \mathbf {L}) = \frac {1}{(n - 1) ^ {2}} \mathrm{tr} (\mathbf {K H L H}), \tag {22}
$$

where H is the centering matrix:

$$
\mathbf {H} = \mathbf {I} _ {n} - \frac {1}{n} \mathbf {1 1} ^ {\top}. \tag {23}
$$

Here, $I_{n}$ is the identity matrix, and 1 is a vector of ones. HSIC effectively quantifies statistical dependence and converges to its population estimate at a rate of $1/\sqrt{n}$ .

To address HSIC's sensitivity to scale, CKA introduces normalization. The CKA metric between two kernel matrices $\mathbf{K}$ and $\mathbf{L}$ is:

$$
\operatorname{CKA} (\mathbf {K}, \mathbf {L}) = \frac {\operatorname{HSIC} (\mathbf {K} , \mathbf {L})}{\sqrt {\operatorname{HSIC} (\mathbf {K} , \mathbf {K}) \cdot \operatorname{HSIC} (\mathbf {L} , \mathbf {L})}}. \tag {24}
$$

The numerator measures the similarity between the two kernels, while the denominator normalizes it using self-similarities within each representation. This normalization ensures CKA's invariance to isotropic scaling, making it a robust similarity measure for feature representations.

# B. More Details about Datasets

We adopt the benchmark established by PATNet (Lei et al., 2022). Fig. 10 illustrates segmentation examples for four target datasets. Further details are as follows:

![](images/6071422b750b49e515d804f2613b589a1d35561ff013bcecc9e3cf7116834d52.jpg)  
FSS-1000

![](images/cb2db69cf95ee54767c4ca99639fa3d43cbafd2805b2d4135c48cdc9f7386cba.jpg)

![](images/e88cdbdddbad1c38cfcedb1b6102319716d3d8248f7f3dfa5b8f804fbf0d59da.jpg)

![](images/e08a80096f59efc70ddb096487336eaf9117a9448fcb2019b3e0fe16c6acdacc.jpg)  
Deepglobe

![](images/349fbdcccd51571ce0e2d486ffccc757c510c0f59bebcccef382ac89561eb21b.jpg)

ISIC2018   
![](images/3f50f38d4fc422b858d1e3d0f45aa62a7e98672355f0f40285ca72a4d8eee8c9.jpg)

![](images/374d5a573ff8081d297045fc535c318aa5daa7b96a4107507ce2dfb0c245fae0.jpg)

![](images/d81e9b0b8ec01174549731192b6214eb930ea038ff1cb623bb793b5f49d3c56b.jpg)  
Chest X-ray   
Figure 10: Examples of images and their corresponding ground truth masks from four target domain datasets.

PASCAL-5 $^{i}$ (Shaban et al., 2017) is an extended version of PASCAL VOC 2012 (Everingham et al., 2010), incorporating additional annotation details from the SDS dataset (Hariharan et al., 2011). We use PASCAL as the source-domain dataset for model training and then evaluate the performance on four target-domain datasets.

FSS-1000 (Li et al., 2020) is a natural image dataset, encompassing 1,000 distinct categories, each represented by 10 samples. In this study, we adhere to the official dataset split for semantic segmentation and report our results on the specified test set, which comprises 240 classes and a total of 2,400 images. FSS-1000 is employed as the target domain for performance evaluation.

Deepglobe (Demir et al., 2018) is a dataset comprising satellite imagery with dense pixel-level annotations across seven categories: urban, agriculture, rangeland, forest, water, barren, and unknown. Since ground-truth labels are only available for the training set, we utilize the official training dataset, which includes 803 images, for evaluation. We adopt Deepglobe as the target domain for evaluation and follow the same processing methodology as PATNet.

ISIC2018 (Codella et al., 2019; Tschandl et al., 2018) is a skin cancer screening dataset, consisting of lesion images where each image contains a single primary lesion. The dataset is processed and utilized following the standards established by PATNet. We consider ISIC2018 as the target domain for evaluation.

Chest X-ray (Candemir et al., 2013; Jaeger et al., 2013)

is a dataset for Tuberculosis screening, consisting of 566 high-resolution images (4020 × 4892 pixels). These images are drawn from 58 Tuberculosis cases and 80 normal cases. To handle the large image dimensions, we resize them to 1024 × 1024 pixels.

# C. Comparison with Domain Transfer Methods

We compare our method with traditional disentanglement representation methods and multi-layer fusion approaches to validate its effectiveness. For a fair comparison, all methods are implemented on the same baseline and evaluated under the 1-shot setting on the CD-FSS benchmark.

Disentanglement representation methods Disentanglement representation methods has long been a popular solution for addressing domain shifts. InfoGAN (Chen et al., 2016) maximizes mutual information to learn disentangled representations in an unsupervised manner. Disentangled-VAE (Li et al., 2021b) excavate category-distilling information from visual and semantic features for generalized zero-shot learning. DFR (Cheng et al., 2023) separates discriminative features from class-irrelevant components. However, these methods require additional complex VAE-Discriminator networks, resulting in significant computational overhead. Moreover, under the few-shot setting, these methods struggle to leverage limited data to learn a suitable latent space for disentanglement. In contrast, our approach does not introduce any extra branch networks. Effectively addresses few-shot scenarios while maintaining low computational overhead. As shown in Table 12 our method outperforms existing disentanglement-based approaches on the CD-FSS task.

<table><tr><td></td><td>FSS</td><td>Deepglobe</td><td>ISIC</td><td>Chest</td><td>Average</td></tr><tr><td>baseline</td><td>77.80</td><td>33.18</td><td>36.99</td><td>51.54</td><td>49.88</td></tr><tr><td>InfoGAN</td><td>78.73</td><td>35.62</td><td>38.19</td><td>65.45</td><td>54.50</td></tr><tr><td>Disentangled-VAE</td><td>78.67</td><td>36.02</td><td>37.79</td><td>66.38</td><td>54.72</td></tr><tr><td>DFR</td><td>79.18</td><td>39.21</td><td>40.62</td><td>72.85</td><td>57.97</td></tr><tr><td>SDRC (Ours)</td><td>80.31</td><td>43.15</td><td>46.57</td><td>82.86</td><td>63.22</td></tr></table>

Table 12: Compare our method to previous disentanglement-based methods under 1-shot setting.

Feature fusion methods FPN (Lin et al., 2017) proposes an in-network feature pyramid architecture that enhances semantic richness across all scales by combining high- and low-resolution features. MEP3P (Zhu et al., 2024) enhanced the original visual features input into MLLMs with image depth features and pseudo-3D positions. MMFuser (Cao et al., 2024) integrated features from multiple layers, enriching the visual inputs for MLLMs by capturing multi-level representations from the vision encoder. These methods enhance the encoder's output representation by utilizing multi-layer information. They aims to increase the feature's robustness and discrimination by aggregating information from multiple layers. In contrast, our approach decouples the encoder's output into independent semantic representations, which improves transferability in cross-domain settings. As shown in Table 13 our method outperforms existing fusion-based approaches on the CD-FSS task (for multimodal methods, we apply this approach solely to improve the vision encoder).

<table><tr><td></td><td>FSS</td><td>Deepglobe</td><td>ISIC</td><td>Chest</td><td>Average</td></tr><tr><td>baseline</td><td>77.80</td><td>33.18</td><td>36.99</td><td>51.54</td><td>49.88</td></tr><tr><td>FPN</td><td>78.73</td><td>37.51</td><td>37.64</td><td>69.59</td><td>55.87</td></tr><tr><td>MEP3P</td><td>79.96</td><td>42.92</td><td>40.43</td><td>73.87</td><td>59.30</td></tr><tr><td>MMFuser</td><td>80.29</td><td>38.65</td><td>42.01</td><td>75.33</td><td>59.07</td></tr><tr><td>SDRC (Ours)</td><td>80.31</td><td>43.15</td><td>46.57</td><td>82.86</td><td>63.22</td></tr></table>

Table 13: Compare our method to previous fusion-based methods under 1-shot setting.

Slot attention-based methods Our approach differs from slot attention-based methods (Locatello et al., 2020; Seitzer et al., 2022) in two fundamental aspects: 1) Slot attention primarily disentangles distinct objects through object-centric representation optimization, exhibiting coarser granularity, whereas our method focuses on disentangling different patterns within the same object at a finer granularity level. 2) While slot attention employs additional iterative attention modules (external networks) for disentanglement, we leverage the inherent property of ViT layers that naturally attend to distinct spatial regions, augmented with orthogonality constraints to reinforce semantic separation, without requiring additional networks. To validate the effectiveness of our method, we conduct comparative evaluations against two representative slot attention-based disentanglement approaches, Slot-Attention (Locatello et al., 2020) and DINOSAUR (Seitzer et al., 2022), as shown in Table 14.

<table><tr><td></td><td>FSS1000</td><td>DeepGlobe</td><td>ISIC</td><td>ChestX</td><td>Mean</td></tr><tr><td>Baseline</td><td>77.80</td><td>33.18</td><td>36.99</td><td>51.54</td><td>49.88</td></tr><tr><td>Slot-Attention</td><td>79.05</td><td>37.83</td><td>41.22</td><td>69.59</td><td>56.92</td></tr><tr><td>DINOSAUR</td><td>79.62</td><td>38.57</td><td>40.89</td><td>72.33</td><td>57.85</td></tr><tr><td>SDRC (Ours)</td><td>80.31</td><td>43.15</td><td>46.57</td><td>82.86</td><td>63.22</td></tr></table>

Table 14: Comparison with slot attention-based methods under 1-shot setting.

# D. Comparison with Methods under Special Setting

Existing CD-FSS methods adopt a batch size of 1 during testing to prevent unfair advantages from incorporating information from other samples. However, IFA (Nie et al., 2024) uses a batch size of 96 during testing, which causes it to calculate foreground and background prototypes by aggregating all samples in a batch. For a fair comparison, we conduct comparisons with IFA using a batch size of 96.

<table><tr><td rowspan="2">Method</td><td colspan="2">FSS-1000</td><td colspan="2">Deepglobe</td><td colspan="2">ISIC</td><td colspan="2">Chest X-ray</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>IFA</td><td>80.1</td><td>82.4</td><td>50.6</td><td>58.8</td><td>66.3</td><td>69.8</td><td>74.0</td><td>74.6</td></tr><tr><td>Ours</td><td>83.1</td><td>85.7</td><td>51.1</td><td>59.4</td><td>69.7</td><td>72.5</td><td>84.1</td><td>87.2</td></tr></table>

Table 15: Comparison with IFA under its specific testing setting, which uses a batch size of 96.

E. Effectiveness in Swin Transformers 

<table><tr><td></td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Stage 4</td></tr><tr><td>Shape</td><td> $H/4 \times W/4 \times C$ </td><td> $H/8 \times W/8 \times 2C$ </td><td> $H/16 \times W/16 \times 4C$ </td><td> $H/32 \times W/32 \times 8C$ </td></tr><tr><td>Layer Num</td><td>2</td><td>2</td><td>18</td><td>2</td></tr></table>

Table 16: Configuration of Swin-B transformer architecture.

As shown in Table 17, we further validated the effectiveness of our method on Swin Transformer (Liu et al., 2021). Since the layers in Swin Transformer do not reside in the same feature space, two additional steps are required: (1) For features from stage 2 to stage 4, we upsample them spatially to $H / 4 \times W / 4$ ; (2) For features from stage 1 to stage 3, we add three mapping linear layers $(C, 8C)$ , $(2C, 8C)$ , and $(4C, 8C)$ to map them to the same feature space as that of stage 4. These mapping layers are trained alongside the model during the source domain training phase. The performance results demonstrate that our method is well-suited to Swin Transformer, and that Swin Transformer shows a significant improvement in performance compared to ViT.

<table><tr><td></td><td>FSS1000</td><td>Deepglobe</td><td>ISIC</td><td>ChestX</td><td>Mean</td></tr><tr><td>ViT-B</td><td>77.80</td><td>33.18</td><td>36.99</td><td>51.54</td><td>49.88</td></tr><tr><td>Swin-B</td><td>79.85</td><td>37.24</td><td>39.90</td><td>66.73</td><td>55.93</td></tr><tr><td>Swin-B + Ours</td><td>81.02</td><td>46.63</td><td>49.19</td><td>83.85</td><td>65.17</td></tr></table>

Table 17: Performance under 1-shot setting with Swin Transformer architecture.

# F. Applications in Other Settings

# 1) Benefit for few-shot segmentation task

We measure the FSS performance of our methods on Pascal, which consists of 20 classes and is set to a 4-fold configuration in the FSS setup. This means training is conducted on 5 classes, while testing is performed on 15 classes that were not seen during the training phase. The experimental results show that our method can also effectively improve the performance of general FSS tasks.

# 2) Benefit for domain generalization task

Under the domain generalization setting, our method trained on Pascal and tested on FSS1000 (with removed support sets and finetuning stage) demonstrates that feature disentanglement enhances model generalizability, yielding concomitant benefits for domain generalization.

<table><tr><td>1 shot</td><td>Fold0</td><td>Fold1</td><td>Fold2</td><td>Fold3</td><td>Mean</td></tr><tr><td>Baseline</td><td>61.5</td><td>68.2</td><td>66.7</td><td>52.5</td><td>62.2</td></tr><tr><td>Ours</td><td>63.1</td><td>70.3</td><td>67.8</td><td>55.4</td><td>64.2</td></tr></table>

Table 18: Performance on Pascal under FSS setting.

<table><tr><td></td><td>Baseline</td><td>Ours</td></tr><tr><td>FSS</td><td>72.6</td><td>75.3</td></tr></table>

Table 19: Performance under domain generalization setting.

# G. More Visualization Results about Self-Disentanglement and Re-Composition

Based on the feature space consistency, we find a natural decomposition in ViT's output, which inspires us to propose the concept of self-disentanglement and re-composition of ViT features for the CD-FSS task, which disentangles features without the need for additional branch networks. In the main text, we presented some visualization examples of self-disentanglement and re-composition. Here, we provide additional examples to further validate our insights.

Disentangle Feature by decomposing ViT structure: To validate our approach, we visualize the features of each layer of ViT-B, as shown in Figure 11. The outputs of all 12 layers of ViT-B show that features extracted at different layers exhibit distinct semantic tendencies, focusing on different regions of the segmented object (e.g. the outline, head, wings, body, and tail of a bird). The result demonstrates that ViT inherently has the potential for semantic disentanglement and supports the feasibility of our insight: decompose the entangled semantic patterns through a structural decomposition of the ViT output.

Cross-Pattern Comparison for re-composition: Due to the dynamic nature of ViT, the patterns captured by the different layer may be semantically similar (as demonstrated through experiments in the main text). Therefore, we adopt the Cross-Pattern Comparison (CPC) module, where the disentangled patterns are cross-compared to facilitate effective re-composition. Here, we present additional cross-comparison (some samples) visualizations and re-composition result visualizations, as shown in Figure 12. Through cross-comparison, the model focuses on different regions of the segmented object. These comparison maps are then re-composed, allowing the model to accurately identify the complete segmentation region.

![](images/b6a487b56082abdb36d98eadad5a1b21012e7814866aa324135449ac50462e1b.jpg)

<details>
<summary>text_image</summary>

Image Mask
1 ... Layer ... 12
</details>

Figure 11: Visualization of features extracted from different layers of ViT demonstrates the feasibility of disentangling the entangled patterns by decomposing the ViT structure.

![](images/b15b7584130264732122f58a30e5eb14e8e129b92e1c984a3991a0cb82fc39fe.jpg)

<details>
<summary>text_image</summary>

query image & mask
some examples of the comparison maps for re-composition
re-composition
</details>

Figure 12: Visualization of the heatmaps showing some examples of cross comparison maps and the results after re-composition.