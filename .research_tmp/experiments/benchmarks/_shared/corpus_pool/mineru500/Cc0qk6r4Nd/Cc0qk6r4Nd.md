# INTERNAL CROSS-LAYER GRADIENTS FOR EXTENDING HOMOGENEITY TO HETEROGENEITY IN FEDERATED LEARNING

Yun-Hin Chan, Rui Zhou, Running Zhao, Zhihan Jiang & Edith C.H. Ngai\*  
Department of Electrical and Electronic Engineering, The University of Hong Kong {chanyunhin, zackery, rnzhao, zhjiang}@connect.hku.hk, chngai@eee.hku.hk

# ABSTRACT

Federated learning (FL) inevitably confronts the challenge of system heterogeneity in practical scenarios. To enhance the capabilities of most model-homogeneous FL methods in handling system heterogeneity, we propose a training scheme that can extend their capabilities to cope with this challenge. In this paper, we commence our study with a detailed exploration of homogeneous and heterogeneous FL settings and discover three key observations: (1) a positive correlation between client performance and layer similarities, (2) higher similarities in the shallow layers in contrast to the deep layers, and (3) the smoother gradient distributions indicate the higher layer similarities. Building upon these observations, we propose InCo Aggregation that leverages internal cross-layer gradients, a mixture of gradients from shallow and deep layers within a server model, to augment the similarity in the deep layers without requiring additional communication between clients. Furthermore, our methods can be tailored to accommodate model-homogeneous FL methods such as FedAvg, FedProx, FedNova, Scaffold, and MOON, to expand their capabilities to handle the system heterogeneity. Copious experimental results validate the effectiveness of InCo Aggregation, spotlighting internal cross-layer gradients as a promising avenue to enhance the performance in heterogeneous FL.

# 1 INTRODUCTION

Federated learning (FL) is proposed to enable a federation of clients to effectively cooperate towards a global objective without exchanging raw data (McMahan et al., 2017). While FL makes it possible to fuse knowledge in a federation with privacy guarantees (Huang et al., 2021; McMahan et al., 2017; Jeong & Hwang, 2022), its inherent attribute of system heterogeneity (Li et al., 2020a), i.e., varying resource constraints of local clients, may hinder the training process and even lower the quality of the jointly-learned models (Kairouz et al., 2021; Li et al., 2020a; Mohri et al., 2019; Gao et al., 2022).

System heterogeneity refers to a set of varying physical resources $\{R_{i}\}_{i=1}^{n}$ , where $R_{i}$ denotes the resource of client i, a high-level idea of resource that holistically governs the aspects of computation, communication, and storage. Existing works cater to system heterogeneity through a methodology called model heterogeneity, which aligns the local models of varying architectures to make full use of local resources (Diao et al., 2021; Baek et al., 2022; Alam et al., 2022; Huang et al., 2022; Fang & Ye, 2022; Lin et al., 2020). Specifically, model heterogeneity refers to a set of different local models $\{M_{i}\}_{i=1}^{n}$ with $M_{i}$ being the model of client i. Let $R(M)$ denote the resource requirement for the model M. Model heterogeneity is a methodology that manages to meet the constraints $\{R(M_{i}) \leq R_{i}\}_{i=1}^{n}$ . In the case of model heterogeneity, heterogeneous devices are allocated to a common model prototype tailored to their varying sizes, such as ResNets with different depths or widths of layers (Liu et al., 2022; Diao et al., 2021; Horvath et al., 2021; Baek et al., 2022; Caldas et al., 2018; Ilhan et al., 2023), strides of layers (Tan et al., 2022), or numbers of kernels (Alam et al., 2022), to account for their inherent resource constraints. While several methods have been proposed to incorporate heterogeneous models into federated learning (FL), their performances often fall short compared to FL training using homogeneous models of the same size (He et al., 2020; Diao et al., 2021). Therefore, gaining a comprehensive understanding of the factors that limit the performance of heterogeneous models in FL is imperative. The primary objective of this paper is to investigate

![](images/4db19afa87a64b16ee0d73d00aff228a1a0dc08ddc93696ff341f023a23a8f1d.jpg)

<details>
<summary>line</summary>

| Stage | IID with homo | Non-IID with homo | Non-IID with hetero |
|-------|---------------|-------------------|---------------------|
| 0     | 1.0           | 1.0               | 1.0                 |
| 1     | 0.95          | 0.9               | 0.85                |
| 2     | 0.9           | 0.8               | 0.65                |
| 3     | 0.8           | 0.5               | 0.3                 |
</details>

(a) The CKA similarity of ResNets.

![](images/64b2d1e25598a8c679d79c2c15c3c916ab8992f8aacbca420d1bd6aea76521a8.jpg)

<details>
<summary>line</summary>

| Block | IID with homo | Non-IID with homo | Non-IID with hetero |
|-------|---------------|-------------------|---------------------|
| 0     | 0.95          | 0.95              | 0.95                |
| 1     | 0.95          | 0.95              | 0.95                |
| 2     | 0.95          | 0.95              | 0.95                |
| 3     | 0.95          | 0.95              | 0.95                |
| 4     | 0.95          | 0.95              | 0.95                |
| 5     | 0.95          | 0.95              | 0.95                |
| 6     | 0.78          | 0.78              | 0.78                |
| 7     | 0.78          | 0.70              | 0.65                |
</details>

(b) The CKA similarity of ViTs.

![](images/fa0ea031187e6d71f1ab500e082d5684b210eb5090b4a206b4a72e3a5a5ab7f9.jpg)

<details>
<summary>line</summary>

| Rounds | CKA    | Accuracy |
| ------ | ------ | -------- |
| 0      | 0.960  | 45       |
| 10     | 0.975  | 50       |
| 20     | 0.980  | 55       |
| 30     | 0.985  | 60       |
| 40     | 0.985  | 65       |
</details>

(c) The relations between CKA and accuracy.   
Figure 1: CKA similarity in different environments and the relation between accuracy and CKA similarity. (a) and (b): The CKA similarity of different federated settings. (c): The positive relation between CKA and accuracy during the training process.

the underlying reasons behind this limitation and propose a potential solution that acts as a bridge between model homogeneity and heterogeneity to tackle this challenge.

In light of this, we first conduct a case study to reveal the obstacles affecting the performance of heterogeneous models in FL. The observations from this case study are enlightening: (1) With increasing heterogeneity in data distributions and model architectures, we observe a decline in model accuracy and layer-wise similarity (layer similarity) as measured by Centered Kernel Alignment (CKA) $^{1}$ (Kornblith et al., 2019), a quantitative metric of bias (Luo et al., 2021; Raghu et al., 2021); (2) The deeper layers share lower layer similarity across the clients, while the shallower layers exhibit greater alignment. These insights further shed light on the notion that shallow layers possess the ability to capture shared features across diverse clients, even within the heterogeneous FL setting. Moreover, these observations indicate that the inferior performances in heterogeneous FL are related to the lower similarity in the deeper layers. Motivated by these findings, we come up with an idea: Can we enhance the similarity of deeper layers, thereby attaining improved performance?

To answer this question, we narrow our focus to the gradients, as the dissimilarity of deep layers across clients is a direct result of gradient updates (Ruder, 2016; Chen et al., 2021). Interestingly, we observe that (3) the gradient distributions originating from shallow layers are smoother and possess higher similarity than those from deep layers, establishing a connection between the gradients and the layer similarity. Therefore, inspired by these insights, we propose a method called InCo Aggregation, deploying different model splitting methods and utilizing the Internal Cross-layer gradients (InCo) in a server model to improve the similarity of its deeper layers without additional communications with the clients. More specifically, cross-layer gradients are mixtures of the gradients from the shallow and the deep layers. We utilize cross-layer gradients as internal knowledge, effectively transferring knowledge from the shallow layers to the deep layers. Nevertheless, mixing these gradients directly poses a significant challenge called gradient divergence (Wang et al., 2020; Zhao et al., 2018). To tackle this issue, we normalize the cross-layer gradients and formulate a convex optimization problem that rectifies their directions. In this way, InCo Aggregation automatically assigns optimal weights to the cross-layer gradients, thus avoiding labor-intensive parameter tuning. Furthermore, InCo Aggregation can extend to model-homogeneous FL methods that previously do not support model heterogeneity, such as FedAvg(McMahan et al., 2017), FedProx (Li et al., 2020b), FedNova (Wang et al., 2020), Scaffold (Karimireddy et al., 2020), and MOON (Li et al., 2021a), to develop their abilities in managing the model heterogeneity problem.

Our main contributions are summarized as follows:

- We first conduct a case study on homogeneous and heterogeneous FL settings and find that (1) client performance is positively correlated to layer similarities across different client models, (2) similarities in the shallow layers are higher than the deep layers, and (3) smoother gradient distributions hint for higher layer similarities.   
- We propose InCo Aggregation, applying model splitting and the internal cross-layer gradients inside a server model. Moreover, our methods can be seamlessly applied to various model-homogeneous FL methods, equipping them with the ability to handle model heterogeneity.   
- We establish the non-convex convergence of utilizing cross-layer gradients in FL and derive the convergence rate.   
- Extensive experiments validate the effectiveness of InCo Aggregation, showcasing its efficacy in strengthening model-homogeneous FL methods for heterogeneous FL scenarios.

![](images/703ce028e22b982edfa9202d8b1a8a3172297148f23a8ab7653723da61bf0249.jpg)

<details>
<summary>line</summary>

| Rounds | Stage2.conv0 | Stage2.conv1 | Stage2.conv2 |
| ------ | ------------ | ------------ | ------------ |
| 0      | 0.35         | 0.35         | 0.35         |
| 10     | 0.30         | 0.18         | 0.14         |
| 20     | 0.33         | 0.16         | 0.15         |
| 30     | 0.34         | 0.20         | 0.16         |
| 40     | 0.35         | 0.22         | 0.17         |
| 50     | 0.36         | 0.19         | 0.18         |
</details>

(a) Similarity of gradients in Stage 2.

![](images/a37ec8c43be303abd5e883e5167f4e65a7298a3712248241a943f8f2ddc07577.jpg)

<details>
<summary>line</summary>

| Rounds | Stage3.conv0 | Stage3.conv1 | Stage3.conv2 |
| ------ | ----------- | ----------- | ----------- |
| 0      | 0.12        | 0.12        | 0.04        |
| 10     | 0.05        | 0.02        | 0.03        |
| 20     | 0.06        | 0.02        | 0.03        |
| 30     | 0.06        | 0.03        | 0.03        |
| 40     | 0.07        | 0.04        | 0.04        |
| 50     | 0.06        | 0.04        | 0.04        |
</details>

(b) Similarity of gradients in Stage 3.

![](images/bd9650b5b3b0aee934a445689badaff2057df5188cf490e1303acc8e43a6f5a9.jpg)

<details>
<summary>line</summary>

| Gradient value | Stage3.conv0 | Stage3.conv1 | Stage3.conv2 |
| -------------- | ----------- | ----------- | ----------- |
| -0.0004        | 1250        | 1250        | 1250        |
| 0.0000         | 1750        | 1750        | 1750        |
| 0.0004         | 1250        | 1250        | 1250        |
| 0.0008         | 1250        | 1250        | 1250        |
</details>

(c) Gradient distributions in Non-IID with hetero.

![](images/d409c3c227fe309a08149439d36ab863f1abe7159b83e957c8dfa412c260ba7e.jpg)

<details>
<summary>line</summary>

| Gradient value | Density (Stage3.conv0) | Density (Stage3.conv1) | Density (Stage3.conv2) |
| -------------- | ---------------------- | --------------------- | --------------------- |
| -0.0004        | 0                      | 0                     | 0                     |
| -0.0002        | 500                    | 500                   | 500                   |
| 0.0000         | 1750                   | 1600                  | 1500                  |
| 0.0002         | 1750                   | 1600                  | 1500                  |
| 0.0004         | 1750                   | 1600                  | 1500                  |
| 0.0006         | 1750                   | 1600                  | 1500                  |
| 0.0008         | 1750                   | 1600                  | 1500                  |
</details>

(d) Gradient distributions in IID with homo.   
Figure 2: Cross-environment similarity and gradients distributions. (a) and (b): Similarity from Stage 2 and Stage 3. (c) and (d): The gradient distributions of Non-IID with hetero and IID with homo.

# 2 PRELIMINARY

To investigate the performance of clients in diverse federated learning settings, we present a case study encompassing both homogeneous and heterogeneous model architectures with CIFAR-10 and split data based on IID and Non-IID with ResNets (He et al., 2016) and ViTs (Dosovitskiy et al., 2020). We use CKA (Kornblith et al., 2019) similarities among models to measure the level of bias exhibited by each model. More detailed results of the case study are provided in Appendix G.

# 2.1 A CASE STUDY IN DIFFERENT FEDERATED LEARNING ENVIRONMENTS

Case Analysis. Generally, we find three intriguing observations from Table 1 and Figure 1:

(i) The deeper layers or stages have lower CKA similarities than the shallow layers. (ii) The settings with higher accuracy also obtain higher CKA similarities in the deeper layers or stages. (iii) The CKA similarity is positively related to the accuracy of clients, as shown in Figure 1c. These observations indicate that increasing the similarity of deeper layers can serve as a viable approach to improving client performance. Considering that shallower layers exhibit higher similarity, a potential direction emerges: to improve the CKA similarity in deeper layers according to the knowledge from the shallower layers.

Table 1: Accuracy of the case study. 

<table><tr><td></td><td>Settings</td><td>Test Accuracy</td></tr><tr><td rowspan="3">ResNet</td><td>IID with homo</td><td>81.0</td></tr><tr><td>Non-IID with homo</td><td>62.3(↓18.7)</td></tr><tr><td>Non-IID with hetero</td><td>52.3(↓28.7)</td></tr><tr><td rowspan="3">ViT</td><td>IID with homo</td><td>81.0</td></tr><tr><td>Non-IID with homo</td><td>54.8(↓26.2)</td></tr><tr><td>Non-IID with hetero</td><td>50.1(↓30.9)</td></tr></table>

# 2.2 DEEP INSIGHTS OF GRADIENTS IN THE SHALLOWER LAYERS

Gradients as Knowledge. In FL, there are two primary types of knowledge that can be utilized: features, which are outputs from middle layers, and gradients from respective layers. We choose to use gradients as our primary knowledge for two essential reasons. Firstly, our FL environment lacks a shared dataset, impeding the establishment of a connection between different clients using features derived from the same data. Secondly, utilizing features in FL would significantly increase communication overheads. Hence, taking these practical considerations into account, we select gradients as the knowledge.

![](images/a0b69e08bd918fe834edea5dd2a7493b7b7447c757594254cf9c03271f097036.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| -0.0002        | 100     |
| 0.0000         | 550     |
| 0.0002         | 300     |
| 0.0004         | 100     |
| 0.0006         | 50      |
| 0.0008         | 20      |
</details>

(a) Stage3 conv0 in Non-IID with hetero.

![](images/225cbda00e58a353b36bdd661261472f9df77001c1ab69de4d5c68acee8fc4fc.jpg)

<details>
<summary>line</summary>

| Gradient value | Round 40 | Round 41 | Round 42 | Round 43 | Round 44 | Round 45 | Round 46 | Round 47 | Round 48 | Round 49 |
| -------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| -0.0004        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| -0.0002        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| 0.0000         | 800      | 600      | 500      | 400      | 300      | 200      | 100      | 50       | 25       | 10       |
| 0.0002         | 800      | 600      | 500      | 400      | 300      | 200      | 100      | 50       | 25       | 10       |
| 0.0004         | 800      | 600      | 500      | 400      | 300      | 200      | 100      | 50       | 25       | 10       |
| 0.0006         | 800      | 600      | 500      | 400      | 300      | 200      | 100      | 50       | 25       | 10       |
| 0.0008         | 800      | 600      | 500      | 400      | 300      | 200      | 100      | 50       | 25       | 10       |
| Peak           |          |          |          |          |          |          |          |          |          |          |
| Value          |          |          |          |          |          |          |          |          |          |          |
| Peak           |          |          |          |          |          |          |          |          |          |          |
| Peak           |          |          |          |          |          |          |          |          |          |          |
| Peak           |          |          |          |          |          |          |          |          |          |          |
| Peak           |          |          |          |          |          |          |          |          |          |          |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This    |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
|
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This    <nl>| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
| Peak           |          | This     | This     | This     | This     | This     | This     | This     | This     | This     |
</details>

(b) Stage3 conv1 in Non-IID with hetero.

![](images/91cade0174e9d35ef330418653c11d69bbe4e9dc1a0e5fa7b9b1e79818e881c7.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| -0.0002        | 300     |
| 0.0000         | 600     |
| 0.0002         | 300     |
| 0.0004         | 500     |
| 0.0006         | 600     |
| 0.0008         | 300     |
</details>

(c) Stage3 conv0 in IID with homo.

![](images/bfac7a3c1d3305da4108f1277a92d622887732faf74b6799fb8aaca89fbd52a1.jpg)  
(d) Stage3 conv1 in IID with homo.   
Figure 3: The gradient distributions from round 40 to 50 in different environments.

Cross-environment Similarity. In this subsection, we deeply investigate the cross-environment similarity of gradients between two environments, i.e., IID with homo and Non-IID with hetero, to shed light on the disparities between shallow and deep layers in the same stage $^{2}$ and identify the gaps between the homogeneous and heterogeneous FL. As depicted in Figure 2a and 2b, gradients from shallow layers (Stage2.conv0 and Stage3.conv0) exhibit higher cross-environment CKA similarity than those from deep layers such as Stage2.conv1, and Stage3.conv2. Notably, even the lowest similarities (red lines) in Stage2.conv0 and Stage3.conv0 surpass the highest similarities in deep layers. These findings underscore the superior quality of gradients obtained from shallow layers.

![](images/582f7dafe0e7f69ae0862ad22b5603910b8eed0430fb69d566cfb6a6d1e3524e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Clients"] --> B["Resource Constraints: {(R_i)}_{i=1}^3"]
    B --> C["Deeper Layers"]
    C --> D["Global Model"]
    D --> E["Client Resources"]
    E --> F["Split Config"]
    F --> G["R1 Split"]
    G --> H["Client Resources"]
    H --> I["R2 Split"]
    I --> J["Client Resources"]
    J --> K["R3 Split"]
    K --> L["Client Resources"]
    L --> M["R1 Split"]
    M --> N["Client Resources"]
    N --> O["R2 Split"]
    O --> P["Client Resources"]
    P --> Q["R3 Split"]
    Q --> R["Client Resources"]
    R --> S["R1 Split"]
    S --> T["Client Resources"]
    T --> U["R2 Split"]
    U --> V["Client Resources"]
    V --> W["R3 Split"]
    W --> X["Client Resources"]
    X --> Y["R1 Split"]
    Y --> Z["Client Resources"]
    Z --> AA["R2 Split"]
    AA --> AB["Client Resources"]
    AB --> AC["R3 Split"]
    AC --> AD["Client Resources"]
    AD --> AE["R1 Split"]
    AE --> AF["Client Resources"]
    AF --> AG["R2 Split"]
    AG --> AH["Client Resources"]
    AH --> AI["R3 Split"]
    AI --> AJ["Client Resources"]
    AJ --> AK["R1 Split"]
    AK --> AL["Client Resources"]
    AL --> AM["R2 Split"]
    AM --> AN["Client Resources"]
    AN --> AO["R3 Split"]
    AO --> AP["Client Resources"]
    AP --> AQ["R1 Split"]
    AQ --> AR["Client Resources"]
    AR --> AS["R2 Split"]
    AS --> AT["Client Resources"]
    AT --> AU["R3 Split"]
    AU --> AV["Client Resources"]
    AV --> AW["R1 Split"]
    AW --> AX["Client Resources"]
    AX --> AY["R2 Split"]
    AY --> AZ["Client Resources"]
    AZ --> BA["R3 Split"]
    BA --> BB["Client Resources"]
    BB --> BC["R1 Split"]
    BC --> BD["Client Resources"]
    BD --> BE["R2 Split"]
    BE --> BF["Client Resources"]
    BF --> BG["R3 Split"]
    BG --> BH["Client Resources"]
    BH --> BI["R1 Split"]
    BI --> BJ["Client Resources"]
    BJ --> BK["R2 Split"]
    BK --> BL["Client Resources"]
    BL --> BM["R3 Split"]
    BM --> BN["Client Resources"]
    BN --> BO["R1 Split"]
    BO --> BP["Client Resources"]
    BP --> BQ["R2 Split"]
    BQ --> BR["Client Resources"]
    BR --> BS["R3 Split"]
    BS --> BT["Client Resources"]
    BT --> BU["R1 Split"]
    BU --> BV["Client Resources"]
    BV --> BW["R2 Split"]
    BW --> BX["Client Resources"]
    BX --> BY["R3 Split"]
    BY --> BZ["Client Resources"]
```
</details>

(a) Layer splitting.

![](images/9a122d95567e7631781b3247d2e1c2730c91b2419fc867d8bba5b0cee15cc224.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Stage Splitting"] --> B["Central Server"]
    B --> C["Client Config R1 - R2 - R3"]
    B --> D["Global Model - Deeper"]
    C --> E["Splitting"]
    D --> E
    E --> F["Stage 1 R1 Split"]
    E --> G["Stage 2 R2 Split"]
    E --> H["Stage 3 R3 Split"]
    F --> I["Aggregate"]
    G --> I
    H --> I
    I --> J["Clients R1 Split"]
    I --> K["Resource Constraints: {(Ri)l=1"] R2 R3]
```
</details>

(b) Stage splitting.

![](images/9fb21de1c9717a118fd46f4982a3c7fd26eac4b8772c87f30e3d6ff147259b08.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Clients"] --> B["Resource Constraints: {(R_i)}^3_{i=1}"]
    B --> C["Deeper Layers"]
    C --> D["Split Config"]
    D --> E["Client Resources R1, R2, R3"]
    E --> F["Global Model"]
    F --> G["Deeper"]
    G --> H["Central Server"]
    H --> I["Split Config"]
    I --> J["Client Resources R1, R2, R3"]
    J --> K["Global Model"]
    K --> L["Deeper"]
    L --> M["Aggregate"]
    M --> N["Clients"]
    N --> O["Resource Constraints: {(R_i)}^3_{i=1}"]
    O --> P["Deeper Layers"]
    P --> Q["Split Config"]
    Q --> R["Client Resources R1, R2, R3"]
    R --> S["Global Model"]
    S --> T["Deeper"]
    T --> U["Central Server"]
```
</details>

(c) Hetero splitting.   
Figure 5: The system architecture of three different model splitting methods: (a) layer splitting, (b) stage splitting, and (c) heterogeneous (hetero) splitting. (a): Layer splitting divides the entire model layer by layer. (b): Stage splitting separates each stage layer by layer. (c): Hetero splitting partitions the whole model in different widths and depths depending on the available resources $R_{i}$ of client i.

relative to those obtained from deep layers, and also indicate that the layers within the same stage exhibit similar patterns to the layers throughout the entire model.

Gradient Distributions. To dig out the latent relations between gradients and layer similarity, we delve deeper into the analysis of gradient distributions across different FL environments. More specifically, the comparison of Figure 2c and Figure 2d reveals that gradients from shallow layers (Stage3.conv0) exhibit greater similarity in distribution between Non-IID with hetero and IID with homo environments, in contrast to deep layers (Stage3.conv1 and Stage3.conv2). Additionally, as depicted in Figure 3c and Figure 3d, the distributions of gradients from a deep layer (Figure 3d) progressively approach the distribution of gradients from a shallow layer (Figure 3c) with each round, in contrast to Figure 3a and Figure 3b, where the distributions from deep layers (Figure 3b) are less smooth than those from shallow layers (Figure 3a) in Non-IID with hetero during rounds 40 to 50. Consequently, drawing from the aforementioned gradient analysis, we can enhance the quality of gradients from deep layers in Non-IID with hetero environments by leveraging gradients from shallow layers, i.e., cross-layer gradients as introduced in the subsequent section.

# 3 INCo AGGREGATION

We provide a concise overview of the three key components in InCo Aggregation at first.

The first component is model splitting, including three types of model splitting methods, as shown in Figure 5. The second component involves the combination of gradients from a shallow layer and a deep layer, referred to as internal cross-layer gradients. To address gradient divergence, the third component employs gradient normalization and introduces a convex optimization formulation. We elaborate on these three critical components of InCo Aggregation as follows.

# 3.1 MODEL SPLITTING

To facilitate model heterogeneity, we propose three model splitting methods: layer splitting, stage splitting, and hetero splitting, as illustrated in Figure 5. These methods distribute models with varying sizes to clients based on

their available resources, denoted as $R_{i}$ . In layer splitting, the central server initiates a global model and splits it layer by layer, considering the client resources $R_{i}$ , as depicted in Figure 5a. In contrast, stage splitting separates each stage layer by layer in Figure 5b. For instance, Figure 5b illustrates how the smallest client with $R_{1}$ resources obtains the first layer from each stage in stage splitting, whereas it acquires the first three layers from the entire model in layer splitting. Furthermore, hetero splitting, depicted in Figure 5c, involves the server splitting the global model into distinct widths and depths for

![](images/22e8791121c9dfeab6f5a280e25a6c8b003625cc7cc51c9b157bf1d84fa13b7b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["...."] --> B["Stage i - 1"]
    B --> C["Layer 0, G₀ᵗ"]
    C --> D["Layer 1, G₁ᵗ + G₀ᵗ"]
    D --> E["..."]
    E --> F["Layer N, Gₙᵗ + G₀ᵗ"]
    F --> G["...."]
    G --> H["Stage i + 1"]
    C --> I["G₀ᵗ"]
    D --> J["..."]
    F --> K["..."]
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
```
</details>

Figure 4: Cross-layer gradients for the server model in InCo.

![](images/53440ca7509b7ab1cdbc1926be820cfc9bea1636e7c7f68b555a5163e36bb910.jpg)

<details>
<summary>text_image</summary>

Client optimum
★
g_k^t = g_k^t + g_0^t
w_k^{t+3}
w_k^{t+2}
w_k^{t+1}
w_k^t
g_0^t
g_0^t
Global optimum
</details>

(a) Gradient divergence.

![](images/25e3d06048d9b14942fef593c6c521652c6b03d21be1a336445aacaf95cb3fdc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Client optimum ★"] --> B["w_k^{t+3}"]
    A --> C["w_k^{t+2}"]
    A --> D["w_k^{t+1}"]
    A --> E["g_k^{t}"]
    B --> F["w_k^{t+1}"]
    C --> G["w_k^{t+2}"]
    D --> H["g_k^{t+3}"]
    E --> I["g_k^{t+4}"]
    F --> J["w_k^{t+4}"]
    G --> K["g_k^{t+5}"]
    H --> L["g_k^{t+6}"]
    I --> M["g_k^{t+7}"]
    J --> N["Global optimum ★"]
    K --> N
    L --> N
    M --> N
    N --> O["Red arrow: g_norm^t = g_k^t + g_0^t"]
```
</details>

(b) Normalized gradients.

![](images/21e287c34d9386fe4caae084efd227998a90b0db3572fc3901510c0267d81da5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Client optimum ★"] --> B["Global optimum ★"]
    B --> C["Global optimum"]
    C --> D["g^i_opt = g^t_r - θ^i g^t_r"]
    D --> E["w^t_r + 1 w^t+2 w^t+3"]
    E --> F["w^t_r + 2 w^t+3"]
    F --> G["-θ^i g^t_r"]
    style A fill:#00FF00,stroke:#000
    style B fill:#00FF00,stroke:#000
    style C fill:#FFA500,stroke:#000
    style D fill:#FFA500,stroke:#000
    style E fill:#FFA500,stroke:#000
    style F fill:#FFA500,stroke:#000
    style G fill:#FFA500,stroke:#000
```
</details>

(c) Normalized and optimized.   
Figure 6: A depiction of gradient divergence, as shown in Figure 6a, along with its solutions. Despite the normalization portrayed in Figure 6b, the impact of gradient divergence persists. To mitigate this issue, we propose a convex optimization problem that is restricting gradient directions, as demonstrated in Figure 6c and supported by Theorem 3.1.

different clients, similar to the approaches in HeteroFL (Diao et al., 2021) and FedRolex (Alam et al., 2022). Layer splitting and stage splitting offer flexibility for extending model-homogeneous methods to system heterogeneity, while hetero splitting enables the handling of client models with varied widths and depths. Finally, the server aggregates client weights based on their original positions in the server models.

# 3.2 INTERNAL CROSS-LAYER GRADIENTS

Deploying model splitting methods directly in FL leads to a significant decrease in client accuracy, as demonstrated in Table 1. However, based on the findings of the case study, we observe that gradients from shallow layers contribute to increasing the similarity among layers from different clients, and CKA similarity exhibits a positive correlation with client accuracy. Therefore, we enhance the quality of gradients from deep layers by incorporating the utilization of cross-layer gradients. More specifically, when a server model updates the deep layers, we combine and refine the gradients from these layers with the gradients from the shallower layers to obtain appropriately updated gradients. Figure 4 provides a visual representation of how cross-layer gradients are employed. We assume that this stage has N layers. The first layer with the same shape in a stage (block) is referred to as Layer 0, and its corresponding gradients at time step t are $G_{0}^{t}$ . For Layer k, where $k \in \{1, 2, ..., N\}$ within the same stage, the cross-layer gradients are given by $G_{k}^{t} + G_{0}^{t}$ . Despite a large number of works on short-cut paths in neural networks, our method differs fundamentally in terms of the goals and the operations. We provide a thorough discussion in Appendix B.

# 3.3 GRADIENTS DIVERGENCE ALLEVIATION

However, the direct utilization of cross-layer gradients leads to an acute issue known as gradient and weight divergence (Wang et al., 2020; Zhao et al., 2018), as depicted in Figure 6a. To counter this effect, we introduce gradient normalization (Figure 6b) and the proposed convex optimization problem to restrict gradient directions, as illustrated in Figure 6c.

Cross-layer Gradients Normalization. Figure 6b depicts the benefits of utilizing normalized gradients. The normalized cross-layer gradient $g_0^{t'} + g_k^{t'}$ directs the model closer to the global optimum than the original cross-layer gradient $g_0^t + g_k^t$ . In particular, our normalization approach emphasizes the norm of gradients, i.e., $g_0^{t'} = g_0^t / ||g_0^t||$ and $g_k^{t'} = g_k^t / ||g_k^t||$ . The normalized cross-layer gradient is computed as $(g_0^{t'} + g_k^{t'}) \times (||g_0^t|| + ||g_k^t||) / 2$ in practice.

Convex Optimization. In addition to utilizing normalized gradients, incorporating novel projective gradients that leverage knowledge from both $g_{0}^{t}$ and $g_{k}^{t}$ serves to alleviate the detrimental impact of gradient divergence arising from the utilization of cross-layer gradients. Moreover, Our objective is to find the optimal projective gradients, denoted as $g_{opt}$ , which strike a balance between being as close as possible to $g_{k}$ while maintaining alignment with $g_{0}$ . This alignment ensures that $g_{k}$ is not hindered by the influence of $g_{0}$ while allowing $g_{opt}$ to acquire the beneficial knowledge for $g_{k}$ from $g_{0}$ . In other words, we aim for $g_{opt}$ to capture the advantageous information contained within $g_{0}$ without impeding the progress of $g_{k}$ . Pursuing this line of thought, we introduce a constraint aimed at ensuring the optimization directions of gradients, outlined as $\langle g_{0}^{t}, g_{k}^{t} \rangle \geq 0$ , where $\langle \cdot, \cdot \rangle$ is the dot product. To establish a convex optimization problem incorporating this constraint, we denote the projected gradient as $g_{opt}$ and formulate the following primal convex optimization problem,

$$
\min _ {g _ {o p t} ^ {t}} \quad | | g _ {k} ^ {t} - g _ {o p t} ^ {t} | | _ {2} ^ {2}, \text {   s.t.   } \langle g _ {o p t} ^ {t}, g _ {0} ^ {t} \rangle \geq 0, \tag {1}
$$

where we preserve the optimization direction of $g_{0}^{t}$ in $g_{opt}^{t}$ while minimizing the distance between $g_{opt}^{t}$ and $g_{k}^{t}$ . We prioritize the proximity of $g_{opt}^{t}$ to $g_{k}^{t}$ over $g_{0}^{t}$ since $g_{k}^{t}$ represents the true gradients of

layer k. By solving this problem through Lagrange dual problem (Bot et al., 2009), we derive the following outcomes,

Theorem 3.1. (Divergence alleviation). If gradients are vectors, for the layers that require cross-layer gradients, their updated gradients can be expressed as,

$$
g _ {o p t} ^ {t} = \left\{ \begin{array}{l l} g _ {k} ^ {t}, & \text { if   } \beta \geq 0 \\ g _ {k} ^ {t} - \theta^ {t} g _ {0} ^ {t}, & \text { if   } \beta <   0, \end{array} \right. \tag {2}
$$

where $\theta^t = \frac{\beta}{\alpha}$ , $\alpha = (g_0^t)^T g_0^t$ and $\beta = (g_0^t)^T g_k^t$ .

Remark 3.2. This theorem can be extended to the matrix form.

We provide proof for Theorem 3.1 and demonstrate how matrix gradients are incorporated into the problem in Appendix C. Our analytic solution in Equation 2 automatically determines the optimal settings for parameter $\theta^{t}$ , eliminating the need for cumbersome manual adjustments. In our practical implementation, we consistently update the server model using the expression $g_{k}^{t}-\theta^{t}g_{0}^{t}$ , irrespective of whether $\beta\geq0$ or $\beta<0$ . This procedure is illustrated in Algorithm 1 in Appendix H.

Communication Overheads. According to the entire process, the primary process (internal cross-layer gradients) is conducted on the server. Therefore, our method does not impose any additional communication overhead between clients and the server.

# 4 CONVERGENCE ANALYSIS

In this section, we demonstrate the convergence of cross-layer gradients and propose the convergence rate in non-convex scenarios. To simplify the notations, we adopt $L_{i}$ to be the local objective. At first, we show the following assumptions frequently used in the convergence analysis for FL (Tan et al., 2022; Li et al., 2020b; Karimireddy et al., 2020).

Assumption 4.1. (Lipschitz Smooth). Each objective function $L_{i}$ is $L$ -Lipschitz smooth and satisfies that $||\nabla L_{i}(x) - \nabla L_{i}(y)|| \leq L||x - y||, \forall (x,y) \in D_{i}, i \in 1,\dots,K$ .

Assumption 4.2. (Unbiased Gradient and Bounded Variance). At each client, the stochastic gradient is an unbiased estimation of the local gradient, with $\mathbb{E}[g_{i}(x)] = \nabla L_{i}(x)$ , and its variance is bounded by $\sigma^{2}$ , meaning that $\mathbb{E}[||g_{i}(x) - \nabla L_{i}(x)||^{2}] \leq \sigma^{2}, \forall i \in 1, \ldots, K$ , where $\sigma^{2} \geq 0$ .

Assumption 4.3. (Bounded Expectation of Stochastic Gradients). The expectation of the norm of the stochastic gradient at each client is bounded by $\rho$ , meaning that $\mathbb{E}[||g_i(x)||] \leq \rho, \forall i \in 1, \dots, K$ .

Assumption 4.4. (Bounded Covariance of Stochastic Gradients). The covariance of the stochastic gradients is bounded by $\Gamma$ , meaning that $Cov(g_{i,l_k}, g_{i,l_j}) \leq \Gamma, \forall i \in 1, \dots, K$ , where $l_k, l_j$ are the layers belonging to a model at client $i$ .

Following these assumptions, we present proof of non-convex convergence concerning the utilization of cross-layer gradients in Federated Learning (FL). We outline our principal theorems as follows.

Theorem 4.5. (Per round drift). Supposed Assumption 4.1 to Assumption 4.4 are satisfied, the loss function of an arbitrary client at round $t + 1$ is bounded by,

$$
\mathbb {E} \left[ L _ {t + 1, 0} \right] \leq \mathbb {E} \left[ L _ {t, 0} \right] - \left(\eta - \frac {L \eta^ {2}}{2}\right) \sum_ {e = 0} ^ {E - 1} \left| | \nabla L _ {t, e} | \right| ^ {2} + \frac {L E \eta^ {2}}{2} \sigma^ {2} + 2 \eta (\Gamma + \rho^ {2}) + L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma). \tag {3}
$$

The Theorem 4.5 demonstrates the bound of the local objective function after every communication round. Non-convex convergence can be guaranteed by the appropriate $\eta$ .

Theorem 4.6. (Non-convex convergence). The loss function L is monotonously decreased with the increasing communication round when,

$$
\eta <   \frac {2 \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t , e} | | ^ {2} - 4 (\Gamma + \rho^ {2})}{L \left(\sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t , e} | | ^ {2} + E \rho^ {2} + 2 (2 \rho^ {2} + \sigma^ {2} + \Gamma)\right)}. \tag {4}
$$

Moreover, after we prove the non-convex convergence for the cross-layer gradients, the non-convex convergence rate is described as follows.

Theorem 4.7. (Non-convex convergence rate). Supposed Assumption 4.1 to Assumption 4.4 are satisfied and $\kappa = L_0 - L^*$ , for an arbitrary client, given any $\epsilon > 0$ , after

$$
T = \frac {2 \kappa}{E \eta ((2 - L \eta) \epsilon - 3 L \eta \sigma^ {2} - 2 (2 + L \eta) \Gamma - 4 (1 + L \eta) \rho^ {2})} \tag {5}
$$

Table 2: Test accuracy of model-homogeneous methods with 100 clients and sample ratio 0.1. We shade in gray the methods that are combined with our proposed method, InCo Aggregation. We bold the best results and denote the improvements compared to the original methods in red. See Appendix H.5 for the error bars of InCo methods. 

<table><tr><td rowspan="2">Base</td><td rowspan="2">Methods</td><td colspan="2">Fashion-MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td><td colspan="2">CINIC10</td></tr><tr><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td></tr><tr><td rowspan="10">ResNet (Stage splitting)</td><td>HeteroAvg</td><td>87.8±1.1</td><td>86.0±1.0</td><td>85.1±2.0</td><td>86.9±2.3</td><td>64.8±2.9</td><td>66.7±3.3</td><td>48.6±2.6</td><td>56.5±1.6</td></tr><tr><td>HeteroProx</td><td>86.8±1.5</td><td>83.9±1.8</td><td>87.8±2.1</td><td>89.9±1.7</td><td>72.5±2.1</td><td>73.1±1.9</td><td>56.4±2.0</td><td>60.9±1.8</td></tr><tr><td>HeteroScaffold</td><td>85.2±0.8</td><td>86.4±0.7</td><td>80.6±2.3</td><td>86.3±2.7</td><td>65.5±3.0</td><td>69.7±2.8</td><td>50.8±2.9</td><td>57.8±3.4</td></tr><tr><td>HeteroNova</td><td>84.9±1.3</td><td>86.7±1.1</td><td>84.4±1.4</td><td>88.0±1.7</td><td>60.1±3.7</td><td>68.0±3.5</td><td>46.1±2.3</td><td>52.1±2.2</td></tr><tr><td>HeteroMOON</td><td>87.9±0.4</td><td>88.3±0.3</td><td>83.0±2.3</td><td>86.5±1.6</td><td>65.1±2.9</td><td>68.4±2.6</td><td>50.1±2.3</td><td>54.7±1.8</td></tr><tr><td>InCoAvg</td><td>90.2(↑2.4)</td><td>88.4(↑2.4)</td><td>87.6(↑2.5)</td><td>89.0(↑2.1)</td><td>67.8(↑3.0)</td><td>70.7(↑4.0)</td><td>53.0(↑4.4)</td><td>57.5(↑1.0)</td></tr><tr><td>InCoProx</td><td>88.8(↑2.0)</td><td>86.4(↑2.5)</td><td>89.0(↑1.2)</td><td>90.8(↑0.9)</td><td>74.5(↑2.0)</td><td>76.8(↑3.7)</td><td>59.1(↑2.7)</td><td>62.5(↑1.6)</td></tr><tr><td>InCoScaffold</td><td>88.3(↑3.1)</td><td>90.1(↑3.7)</td><td>85.4(↑4.8)</td><td>87.8(↑1.5)</td><td>67.3(↑1.8)</td><td>73.8(↑4.1)</td><td>53.5(↑2.7)</td><td>61.7(↑3.9)</td></tr><tr><td>InCoNova</td><td>86.6(↑1.7)</td><td>87.4(↑0.7)</td><td>86.4(↑2.0)</td><td>88.4(↑0.4)</td><td>62.8(↑2.7)</td><td>69.7(↑2.7)</td><td>48.0(↑1.9)</td><td>54.1(↑2.0)</td></tr><tr><td>InCoMOON</td><td>89.1(↑1.2)</td><td>89.5(↑1.2)</td><td>85.6(↑2.6)</td><td>89.3(↑2.8)</td><td>68.2(↑3.1)</td><td>71.8(↑3.4)</td><td>54.3(↑4.2)</td><td>57.6(↑2.9)</td></tr><tr><td rowspan="10">VT (Layer splitting)</td><td>HeteroAvg</td><td>92.2±0.6</td><td>92.0±0.6</td><td>92.9±1.0</td><td>93.8±0.9</td><td>93.6±1.0</td><td>94.1±0.9</td><td>84.2±1.6</td><td>85.3±1.3</td></tr><tr><td>HeteroProx</td><td>90.9±0.8</td><td>91.7±0.6</td><td>91.2±1.3</td><td>92.4±1.8</td><td>92.0±1.5</td><td>92.6±1.3</td><td>84.0±1.8</td><td>84.8±2.0</td></tr><tr><td>HeteroScaffold</td><td>91.9±0.6</td><td>92.1±0.4</td><td>92.5±0.9</td><td>93.7±0.6</td><td>93.8±0.8</td><td>94.3±0.4</td><td>83.8±1.9</td><td>85.3±1.6</td></tr><tr><td>HeteroNova</td><td>92.1±0.9</td><td>92.4±0.4</td><td>92.3±1.0</td><td>94.1±1.2</td><td>93.6±0.5</td><td>94.5±0.6</td><td>85.3±1.7</td><td>86.7±1.5</td></tr><tr><td>HeteroMOON</td><td>92.0±0.4</td><td>92.3±0.3</td><td>92.7±1.1</td><td>94.0±0.9</td><td>93.5±0.8</td><td>94.6±0.5</td><td>84.7±1.4</td><td>85.6±1.4</td></tr><tr><td>InCoAvg</td><td>93.0(↑0.8)</td><td>93.1(↑1.1)</td><td>94.2(↑1.3)</td><td>95.0(↑1.2)</td><td>94.6(↑1.0)</td><td>95.0(↑0.9)</td><td>85.9(↑1.7)</td><td>86.8(↑1.5)</td></tr><tr><td>InCoProx</td><td>92.6(↑1.7)</td><td>92.5(↑0.8)</td><td>93.9(↑2.7)</td><td>94.4(↑2.0)</td><td>94.0(↑2.0)</td><td>94.8(↑2.2)</td><td>85.1(↑1.1)</td><td>86.0(↑1.2)</td></tr><tr><td>InCoScaffold</td><td>92.9(↑1.0)</td><td>93.0(↑0.9)</td><td>94.0(↑1.5)</td><td>94.8(↑1.1)</td><td>94.6(↑0.8)</td><td>95.0(↑0.7)</td><td>85.7(↑1.9)</td><td>86.5(↑1.2)</td></tr><tr><td>InCoNova</td><td>93.1(↑1.0)</td><td>93.6(↑1.2)</td><td>94.7(↑2.4)</td><td>95.6(↑1.5)</td><td>94.8(↑1.2)</td><td>95.7(↑1.2)</td><td>86.2(↑0.9)</td><td>88.2(↑1.2)</td></tr><tr><td>InCoMOON</td><td>92.8(↑0.8)</td><td>93.0(↑0.7)</td><td>94.7(↑2.0)</td><td>95.1(↑1.1)</td><td>94.2(↑0.7)</td><td>95.1(↑0.5)</td><td>86.0(↑1.3)</td><td>86.8(↑1.2)</td></tr></table>

communication rounds, we have

$$
\frac {1}{T E} \sum_ {t = 0} ^ {T - 1} \sum_ {e = 0} ^ {E - 1} \mathbb {E} [ | | \nabla L _ {t, e} | | ^ {2} ] \leq \epsilon , i f \eta <   \frac {2 \epsilon - 4 (\Gamma + \rho^ {2})}{L (\epsilon + E \rho^ {2} + 2 (2 \rho^ {2} + \sigma^ {2} + \Gamma))}. \tag {6}
$$

Following these theorems, the convergence of internal cross-layer gradients is guaranteed. The proof is presented in Appendix D.

# 5 EXPERIMENTS

In this section, we conduct comprehensive experiments aimed at demonstrating three fundamental aspects: (1) the efficacy of InCo Aggregation and its extensions for various FL methods (Section 5.2), (2) the robustness analysis and ablation study of InCo Aggregation (Section 5.3), (3) in-depth analyses of the underlying principles behind InCo Aggregation (Section 5.4). Our codes are released on GitHub $^{3}$ . More experimental details and results can be found in Appendix H.

# 5.1 EXPERIMENT SETUP

Dataset and Data Distribution. We conduct experiments on Fashion-MNIST (Xiao et al., 2017), SVHN (Netzer et al., 2011), CIFAR-10 (Krizhevsky et al., 2009) and CINIC-10 (Darlow et al., 2018) under non-iid settings. We evaluate the algorithms under two Dirichlet distributions with $\alpha = 0.5$ and $\alpha = 1.0$ for all datasets.

Baselines. To demonstrate the effectiveness of InCo Aggregation, we use five baselines in model-homogeneous FL: FedAvg (McMahan et al., 2017), FedProx (Li et al., 2020b), FedNova (Wang et al., 2020), Scaffold (Karimireddy et al., 2020), and MOON (Li et al., 2021a) for ResNets and ViTs. In the context of model heterogeneity, we extend the training procedures of these baselines by incorporating model splitting methods, denoting the modified versions with the prefix "Hetero". Furthermore, by incorporating these methods with InCo Aggregation, we prefix the names with "InCo". Moreover, we also extend our methods to four state-of-the-art methods in model-heterogeneous FL: HeteroFL(Diao et al., 2021), InclusiveFL(Liu et al., 2022), FedRolex(Alam et al., 2022) and ScaleFL(Ilhan et al., 2023) for ResNets. We take the average accuracy of three different random seeds.

Federated Settings. In heterogeneous FL, we consider two architectures, ResNets and ViTs. The largest models are ResNet26 and ViT-S/12 (ViT-S with 12 layers). We deploy stage splitting for ResNets and obtain five sub-models, which can be recognized as ResNet10, ResNet14, ResNet18, ResNet22, and ResNet26. For the pre-trained ViT models, we employ layer splitting and result in five sub-models, which are ViT-S/8, ViT-S/9, ViT-S/10, ViT-S/11, and ViT-S/12. Moreover, we consider five different model capacities $\beta=\{1,\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{16}\}$ in hetero splitting, where for instance, $\frac{1}{2}$

Table 3: Test accuracy of model-heterogeneity methods with 100 clients and sample ratio 0.1. We shade in gray the methods that are combined with our proposed method, InCo Aggregation. We denote the improvements compared to the original methods in red. See Appendix H.5 for the error bars of InCo methods. 

<table><tr><td rowspan="2">Base</td><td rowspan="2">Splitting</td><td rowspan="2">Methods</td><td colspan="2">Fashion-MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td><td rowspan="2">Comm.overheads</td><td rowspan="2">FLOPs</td></tr><tr><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td></tr><tr><td rowspan="10">ResNet</td><td rowspan="2">Hetero</td><td>HeteroFL</td><td>88.9±1.0</td><td>89.7±0.7</td><td>90.5±1.6</td><td>92.2±1.3</td><td>65.2±3.2</td><td>68.4±3.6</td><td>4.6M</td><td>33.4M</td></tr><tr><td>+InCo</td><td>90.0(↑1.1)</td><td>90.4(↑0.7)</td><td>92.1(↑1.6)</td><td>93.5(↑1.3)</td><td>68.2(↑3.0)</td><td>71.2(↑2.8)</td><td>4.6M</td><td>33.8M</td></tr><tr><td rowspan="2">Stage</td><td>InclusiveFL</td><td>89.1±1.1</td><td>89.8±1.0</td><td>88.6±2.0</td><td>90.0±2.2</td><td>65.7±3.5</td><td>68.4±3.3</td><td>12.3M</td><td>75.2M</td></tr><tr><td>+InCo</td><td>90.1(↑1.0)</td><td>90.5(↑0.7)</td><td>90.6(↑2.0)</td><td>90.9(↑0.9)</td><td>69.1(↑3.4)</td><td>72.3(↑3.9)</td><td>12.3M</td><td>75.6M</td></tr><tr><td rowspan="2">Hetero</td><td>FedRolex</td><td>88.2±1.0</td><td>90.2±0.8</td><td>90.9±1.3</td><td>91.6±1.7</td><td>64.7±4.1</td><td>72.3±3.0</td><td>4.6M</td><td>33.4M</td></tr><tr><td>+InCo</td><td>90.4(↑2.2)</td><td>91.3(↑1.1)</td><td>92.8(↑1.9)</td><td>93.4(↑1.8)</td><td>67.9(↑3.2)</td><td>75.6(↑3.3)</td><td>4.6M</td><td>33.8M</td></tr><tr><td rowspan="2">Hetero</td><td>ScaleFL</td><td>90.9±0.5</td><td>91.0±0.4</td><td>92.6±1.0</td><td>92.9±0.9</td><td>71.1±2.9</td><td>74.7±3.1</td><td>9.5M</td><td>51.9M</td></tr><tr><td>+InCo</td><td>91.5(↑0.6)</td><td>91.7(↑0.7)</td><td>93.4(↑0.8)</td><td>93.6(↑0.7)</td><td>73.8(↑2.7)</td><td>76.1(↑2.4)</td><td>9.5M</td><td>52.3M</td></tr><tr><td>N/A</td><td>AllSmall</td><td>83.5±1.7</td><td>84.0±1.7</td><td>72.1±3.5</td><td>81.0±2.9</td><td>39.2±2.0</td><td>44.9±2.3</td><td>0.07M</td><td>3.7M</td></tr><tr><td>N/A</td><td>AllLarge</td><td>91.8±0.5</td><td>92.5±0.8</td><td>93.4±0.8</td><td>93.8±0.5</td><td>79.6±2.9</td><td>82.5±1.0</td><td>17.5M</td><td>112.4M</td></tr></table>

![](images/afd5d0eba1f735e245883cf608d6e93bd4d0379eb584134f3f63b036a084ffd5.jpg)

<details>
<summary>line</summary>

| BatchSize | InCoAvg | HeteroAvg |
| --------- | ------- | --------- |
| 32        | 70.0    | 65.0      |
| 64        | 68.0    | 65.0      |
| 128       | 67.0    | 62.0      |
| 256       | 65.0    | 57.0      |
</details>

(a) Different batch sizes in CIFAR-10.

![](images/f5baa1f06ffa2e45dd5ff74c8583189e1fcee1ac961f32657192978049b4dabd.jpg)

<details>
<summary>line</summary>

| BatchSize | InCoAvg | HeteroAvg |
| --------- | ------- | --------- |
| 32        | 56      | 51        |
| 64        | 54      | 49        |
| 128       | 50      | 47        |
| 256       | 49      | 44        |
</details>

(b) Different batch sizes in CINIC-10.

![](images/d62bf4a73da23bff1bdfc14a92d206b972955ea878c42d1a5213ebd9bc954f9d.jpg)

<details>
<summary>line</summary>

| Noise Std | InCoAvg | HeteroAvg |
| --------- | ------- | --------- |
| 1*std     | 67      | 63        |
| 2*std     | 66      | 61        |
| 3*std     | 64      | 60        |
| 4*std     | 62      | 59        |
| 5*std     | 59      | 55        |
</details>

(c) Different noise perturbations in CIFAR-10.

![](images/50956efea86519bdc13de7da2d7872185139f4f5a7526f63d8abec3b303ee599.jpg)

<details>
<summary>line</summary>

| Noise Std | InCoAvg | HeteroAvg |
| --------- | ------- | --------- |
| 1*std     | 52.0    | 47.5      |
| 2*std     | 52.5    | 47.0      |
| 3*std     | 47.0    | 42.5      |
| 4*std     | 46.0    | 39.5      |
| 5*std     | 44.0    | 38.0      |
</details>

(d) Different noise perturbations in CINIC-10.

Figure 7: Robustness analysis for InCo Aggregation.   
![](images/7589574df8cc3715879ed7652693617b65613117f21c9bca7f5ab3a6fbb8ba7f.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoAvg    | α=1.0        | 88.0     |
| InCoAvg    | α=0.5        | 90.0     |
| w/o N      | α=1.0        | 86.0     |
| w/o N      | α=0.5        | 87.0     |
| w/o O      | α=1.0        | 86.5     |
| w/o O      | α=0.5        | 88.0     |
| w/o N&O    | α=1.0        | 83.0     |
| w/o N&O    | α=0.5        | 88.5     |
| HeteroAvg  | α=1.0        | 86.0     |
| HeteroAvg  | α=0.5        | 87.5     |
</details>

(a) Fashion-MNIST.

![](images/27c3d841f10dc1d2369cb5ee81d6a07e3e9d19dd65be9d6b242c0cf3dcd98d40.jpg)

<details>
<summary>line</summary>

| Method     | Distribution α=1.0 | Distribution α=0.5 |
| ---------- | ------------------ | ------------------ |
| InCoAvg    | 89.0               | 87.0               |
| w/o N      | 87.0               | 83.0               |
| w/o O      | 86.5               | 85.5               |
| w/o N&O    | 86.0               | 85.0               |
| HeteroAvg  | 86.5               | 84.5               |
</details>

(b) SVHN.

![](images/a84001061dac0e53bdc5f3c552f1a3c1dea6f09c2cb99c85385a109f635613df.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoAvg    | 1.0          | 71       |
| w/o N      | 1.0          | 68       |
| w/o O      | 1.0          | 69       |
| w/o N&O    | 1.0          | 68       |
| HeteroAvg  | 1.0          | 67       |
| InCoAvg    | 0.5          | 68       |
| w/o N      | 0.5          | 60       |
| w/o O      | 0.5          | 65       |
| w/o N&O    | 0.5          | 64       |
| HeteroAvg  | 0.5          | 64       |
</details>

(c) CIFAR-10.

![](images/16771b4276fa01b36cb42191934266364e48235c51963e128b3d76bf23af6576.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoAvg    | α=1.0        | 57.0     |
| InCoAvg    | α=0.5        | 53.0     |
| w/o N      | α=1.0        | 56.0     |
| w/o N      | α=0.5        | 51.0     |
| w/o O      | α=1.0        | 56.0     |
| w/o O      | α=0.5        | 50.0     |
| w/o N&O    | α=1.0        | 56.0     |
| w/o N&O    | α=0.5        | 49.0     |
| HeteroAvg  | α=1.0        | 56.0     |
| HeteroAvg  | α=0.5        | 48.0     |
</details>

(d) CINIC-10.   
Figure 8: Ablation studies for InCo Aggregation. The federated settings are the same as Table 2.

indicates the widths and depths are half of the largest model ResNet26. Our experimental setup involves 100 clients, categorized into five distinct groups, with a sample ratio of 0.1. The detailed model sizes are shown in Appendix H.4.

# 5.2 INCo AGGREGATION IMPROVES ALL BASELINES.

Table 2 and Table 3 present the test accuracy of 100 clients with a sample ratio of 0.1. Table 2 provides compelling evidence for the efficacy of InCo Aggregation in enhancing the performance of all model-homogeneous baselines. Table 3 demonstrates the improvements of deploying InCo Aggregation in the model-heterogeneous methods. Moreover, Table 3 highlights that InCo Aggregation introduces no additional communication overhead and only incurs 0.4M FLOPs, which are conducted on the server side, indicating that InCo Aggregation does not impose any burden on client communication and computation resources.

# 5.3 ROBUSTNESS ANALYSIS AND ABLATION STUDY.

We delve into the robustness analysis of InCo Aggregation, examining two aspects: the impact of varying batch sizes and noise perturbations on gradients during transmission. Additionally, we perform an ablation study for InCo Aggregation. We provide more experiments in Appendix H.

Effect of Batch Size and Noise Perturbation. Notably, when compared to FedAvg as depicted in Figure 7a and Figure 7b, our method exhibits significant improvements while maintaining comparable performance across all settings. Furthermore, as illustrated in Figure 7c and Figure 7d, we explore the impact of noise perturbations by simulating noise with standard deviations following the gradients.

Ablation Study. Our ablation study includes the following methods: (i) InCoAvg w/o Normalization (HeteroAvg with cross-layer gradients and optimization), (ii) InCoAvg w/o Optimization (HeteroAvg with normalized cross-layer gradients), (iii) InCoAvg w/o Normalization and Optimization (HeteroAvg with cross-layer gradients), and (iv) HeteroAvg (FedAvg with stage splitting). The ablation study of InCo Aggregation is depicted in Figure 8, demonstrating the efficiency of InCo Aggregation.

![](images/b769ca7ed3b0b52482beaecb4e35296c3b5917bdb0930226420cfd9ff56ecc50.jpg)  
(a) $\theta$ in all layers.

![](images/7c7a47ca64ac85281f1a1eb1a78948331eabaca9dcc977f18ac29e5a45b4cbd0.jpg)  
(b) $\beta$ in Layer 11.

![](images/53eb9c4dda73b2d9cabfcc12fcab0776d43198438e6b3e09ea4af4c9b0c9c2dd.jpg)  
(c) $\beta$ in Layer 13.

![](images/5868806cd2793bd6d96708df281b4fc56eb4ddb5abcb683d925f53fbb0b33d86.jpg)  
(d) FedAvg.

![](images/c664bc4bdc4ec9724f17601c30d1f1a2a9fb2d5b6bb3cda49b06c11409294a05.jpg)  
(e) HeteroAvg.

![](images/51205b13b2e9cd5908fbb13cf17575c5533ec4ef2ec9f2fdcdf19adb871bff2d.jpg)  
(f) InCoAvg.

Figure 9: Important coefficients of Theorem 3.1 and t-SNE visualization of features. (a): $\theta$ in all layers. (b): $\beta$ in Layer 11. (c): $\beta$ in Layer 13. (d) to (f): t-SNE visualization of features learned by different methods on CIFAR-10. We select data from one class and three clients (client 0: ResNet10, client 1: ResNet14, client 2: ResNet26) to simplify the notations in t-SNE figures.   
![](images/e10b7ba7ba5bb060048660445b8b15da1275b70433e5ddd899e276e8279cd347.jpg)

<details>
<summary>line</summary>

| Stage | InCoAvg | HeteroAvg | FedAvg |
|-------|---------|-----------|--------|
| 0     | 1.0     | 1.0       | 0.9    |
| 1     | 0.95    | 0.9       | 0.8    |
| 2     | 0.9     | 0.7       | 0.2    |
| 3     | 0.6     | 0.4       | 0.2    |
</details>

(a) CIFAR10 with $\alpha = 0.5$ .

![](images/7bf4c744cf790846d08b773b446898b5a7594652ed610542b424bc612a49b774.jpg)  
(b) InCoAvg.

![](images/c89dfd1a3a46c36d8c5f3ffa9dff47e99e3cbe5e561c7211068694db3e4d68a4.jpg)  
(c) HeteroAvg.

![](images/bc0173334f732bdc733a02e302ed0900b5aef9308f22aaa470e0e575f6144992.jpg)  
(d) FedAvg.

<table><tr><td>CID</td><td>InCoAvg</td><td>HeteroAvg</td></tr><tr><td>0</td><td>54.4</td><td>54.8</td></tr><tr><td>20</td><td>70.7</td><td>67.8</td></tr><tr><td>40</td><td>75.3</td><td>72.6</td></tr><tr><td>60</td><td>73.3</td><td>69.9</td></tr><tr><td>80</td><td>72.0</td><td>68.5</td></tr></table>

(e) Client Accuracy.   
Figure 10: CKA layer similarity, Heatmaps, and accuracy of different clients. (a): The layer similarity of different methods. (b) to (d): Heatmaps for different methods in stage 2 and stage 3. (e): Accuracy of each client group. (0: ResNet10, 20: ResNet14, 40: ResNet18, 60: ResNet22, 80: ResNet26)

# 5.4 THE REASONS FOR THE IMPROVEMENTS

We undertake a comprehensive analysis to gain deeper insights into the mechanisms underlying the efficacy of InCo Aggregation. Our analysis focuses on the following three key aspects: (1) The investigation of important coefficients $\theta$ and $\beta$ in Theorem 3.1. (2) An examination of the feature spaces generated by different methods. (3) The evaluation of CKA similarity across various layers. Moreover, we discuss the differences between adding noises and InCo gradients in Appendix H.6.

Analysis for $\theta$ and $\beta$ . In our experiments, we set $\theta = 1$ for InCoAvg w/o Optimization, the blue dash line in Figure 9a. However, under Theorem 3.1, we observe that the value of $\theta$ varies for different layers, indicating the effectiveness of the theorem in automatically determining the appropriate $\theta$ values. $\beta > 0$ denotes the same direction between shallow layer gradients and the current layer gradients. Furthermore,

Table 4: The Percentage of $\beta > 0$ 

<table><tr><td rowspan="2">Methods</td><td colspan="2">Percentage of  $\beta > 0$ </td></tr><tr><td>Layer 11</td><td>Layer 13</td></tr><tr><td>InCoAvg</td><td>83.8</td><td>74.4</td></tr><tr><td>InCoAvg w/o O</td><td>53.5</td><td>50.2</td></tr></table>

Table 4 provides empirical evidence supporting the efficacy of Theorem 3.1 in heterogeneous FL.

t-SNE Visualizations. Figure 9d and Figure 9e provide visual evidence of bias stemming from model heterogeneity in the FedAvg and HeteroAvg. In contrast, Figure 9f demonstrates that InCoAvg effectively addresses bias. These findings highlight the superior generalization capability of InCoAvg compared to HeteroAvg and FedAvg, indicating that InCoAvg mitigates bias issues in client models.

Analysis for CKA Layer Similarity. Figure 10a reveals that InCoAvg exhibits a significantly higher CKA layer similarity compared to FedAvg. Consistent with the t-SNE visualization, FedAvg's heatmaps exhibit block-wise patterns in Figure 10d due to its inability to extract features from diverse model architectures. Notably, the smallest models in InCoAvg (top left corner) exhibit lower similarity (more black) with other clients compared to HeteroAvg in stage 3. This discrepancy arises because the accuracy of the smallest models in InCoAvg is similar to that of HeteroAvg, but the performance of larger models in InCoAvg surpasses that of HeteroAvg, as indicated in Figure 10e. Consequently, a larger similarity gap emerges between the smallest models and the other models. Addressing the performance of the smallest models in InCo Aggregation represents our future research direction.

# 6 CONCLUSIONS

We propose a novel FL training scheme called InCo Aggregation, which aims to enhance the capabilities of model-homogeneous FL methods in heterogeneous FL settings. Our approach leverages normalized cross-layer gradients to promote similarity among deep layers across different clients. Additionally, we introduce a convex optimization formulation to address the challenge of gradient divergence. Through extensive experimental evaluations, we demonstrate the effectiveness of InCo Aggregation in improving heterogeneous FL performance.

# REFERENCES

Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang. FedRolex: Model-heterogeneous federated learning with rolling sub-model extraction. Advances in Neural Information Processing Systems, 35:29677–29690, 2022.   
Sergio A. Alvarez. Gaussian rbf centered kernel alignment (cka) in the large-bandwidth limit. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):6587–6593, 2023. doi:10.1109/TPAMI.2022.3216518.   
Hankyul Baek, Won Joon Yun, Yunseok Kwak, Soyi Jung, Mingyue Ji, Mehdi Bennis, Jihong Park, and Joongheon Kim. Joint superposition coding and training for federated learning over multi-width neural networks. In IEEE INFOCOM 2022-IEEE Conference on Computer Communications, pp. 1729–1738. IEEE, 2022.   
Radu Ioan Bot, Sorin-Mihai Grad, and Gert Wanka. Duality in vector optimization. Springer Science & Business Media, 2009.   
Stephen Boyd, Stephen P Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.   
Sebastian Caldas, Jakub Konečny, H Brendan McMahan, and Ameet Talwalkar. Expanding the reach of federated learning by reducing client resource requirements. arXiv preprint arXiv:1812.07210, 2018.   
Yun Hin Chan and Edith Ngai. Fedhe: Heterogeneous models and communication-efficient federated learning. IEEE International Conference on Mobility, Sensing and Networking (MSN 2021), 2021.   
Yun-Hin Chan and Edith C-H Ngai. Exploiting features and logits in heterogeneous federated learning. arXiv preprint arXiv:2210.15527, 2022.   
Chen Chen, Hong Xu, Wei Wang, Baochun Li, Bo Li, Li Chen, and Gong Zhang. Communication-efficient federated learning with adaptive parameter freezing. In 2021 IEEE 41st International Conference on Distributed Computing Systems (ICDCS), pp. 1–11. IEEE, 2021.   
Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, and Qi Tian. Data-free learning of student networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3514–3522, 2019.   
Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13(1):795–828, 2012.   
Luke N Darlow, Elliot J Crowley, Antreas Antoniou, and Amos J Storkey. Cinic-10 is not imagenet or cifar-10. arXiv preprint arXiv:1810.03505, 2018.   
Enmao Diao, Jie Ding, and Vahid Tarokh. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients. In International Conference on Learning Representations, 2021.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
Xiuwen Fang and Mang Ye. Robust federated learning with noisy and heterogeneous clients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10072–10081, 2022.   
Dashan Gao, Xin Yao, and Qiang Yang. A survey on heterogeneous federated learning, 2022. URL https://arxiv.org/abs/2210.04505.   
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6):1789–1819, March 2021. ISSN 1573-1405. doi: 10.1007/s11263-021-01453-z. URL http://dx.doi.org/10.1007/s11263-021-01453-z.   
Chaoyang He, Murali Annavaram, and Salman Avestimehr. Group knowledge transfer: Federated learning of large cnns at the edge. Advances in Neural Information Processing Systems, 33:14068–14080, 2020.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 630–645. Springer, 2016.   
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. NIPS Deep Learning and Representation Learning Workshop, 2015.   
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.   
Samuel Horvath, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, and Nicholas Lane. Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. Advances in Neural Information Processing Systems, 34:12876–12889, 2021.   
Wenke Huang, Mang Ye, and Bo Du. Learn from others and be yourself in heterogeneous federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10143–10153, 2022.   
Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, and Sanjeev Arora. Evaluating gradient inversion attacks and defenses in federated learning. Advances in Neural Information Processing Systems, 34:7232–7241, 2021.   
Fatih Ilhan, Gong Su, and Ling Liu. Scalefl: Resource-adaptive federated learning with heterogeneous clients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24532–24541, 2023.   
Wonyong Jeong and Sung Ju Hwang. Factorized-fl: Personalized federated learning with parameter factorization & similarity matching. In Advances in Neural Information Processing Systems, 2022.   
Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. 2021.   
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning, pp. 5132–5143. PMLR, 2020.   
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.   
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pp. 3519–3529. PMLR, 2019.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
Daliang Li and Junpu Wang. FedMD: Heterogenous federated learning via model distillation. NeurIPS Workshop on Federated Learning for Data Privacy and Confidentiality, 2019.   
Qinbin Li, Bingsheng He, and Dawn Song. Model-contrastive federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10713–10722, 2021a.   
Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th International Conference on Data Engineering (ICDE), pp. 965–978. IEEE, 2022.

Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. IEEE signal processing magazine, 37(3):50–60, 2020a.   
Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. Proceedings of the 3rd MLSys Conference, 2020b.   
Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning: Do different neural networks learn the same representations? arXiv preprint arXiv:1511.07543, 2015.   
Yiying Li, Wei Zhou, Huaimin Wang, Haibo Mi, and Timothy M Hospedales. FedH2L: Federated learning with model and statistical heterogeneity. arXiv preprint arXiv:2101.11296, 2021b.   
Tao Lin, Lingjing Kong, Sebastian U Stich, and Martin Jaggi. Ensemble distillation for robust model fusion in federated learning. Advances in Neural Information Processing Systems, 33:2351–2363, 2020.   
Ruixuan Liu, Fangzhao Wu, Chuhan Wu, Yanlin Wang, Lingjuan Lyu, Hong Chen, and Xing Xie. No one left behind: Inclusive federated learning over heterogeneous devices. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3398–3406, 2022.   
Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, and Jiashi Feng. No fear of heterogeneity: Classifier calibration for federated learning with non-iid data. Advances in Neural Information Processing Systems, 34:5972–5984, 2021.   
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. PMLR, 2017.   
Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. In International Conference on Machine Learning, pp. 4615–4625. PMLR, 2019.   
Ari Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural networks with canonical correlation. Advances in neural information processing systems, 31, 2018.   
Gaurav Kumar Nayak, Konda Reddy Mopuri, Vaisakh Shaj, Venkatesh Babu Radhakrishnan, and Anirban Chakraborty. Zero-shot knowledge distillation in deep networks. In International Conference on Machine Learning, pp. 4743–4751. PMLR, 2019.   
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.   
Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. Advances in neural information processing systems, 30, 2017.   
Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in Neural Information Processing Systems, 34:12116–12128, 2021.   
Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.   
Felix Sattler, Tim Korjakow, Roman Rischke, and Wojciech Samek. Fedaux: Leveraging unlabeled auxiliary data in federated learning. IEEE Transactions on Neural Networks and Learning Systems, 2021.   
Tao Shen, Jie Zhang, Xinkang Jia, Fengda Zhang, Gang Huang, Pan Zhou, Kun Kuang, Fei Wu, and Chao Wu. Federated mutual learning. arXiv preprint arXiv:2006.16765, 2020.   
Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, and Chengqi Zhang. Fedproto: Federated prototype learning across heterogeneous clients. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8432–8440, 2022.

Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. Advances in neural information processing systems, 33:7611–7623, 2020.   
Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John Hopcroft. Towards understanding learning representations: To what extent do different neural networks learn the same representation. Advances in neural information processing systems, 31, 2018.   
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms, 2017.   
Cong Xie, Sanmi Koyejo, and Indranil Gupta. Asynchronous federated optimization. 12th Annual Workshop on Optimization for Machine Learning, 2020.   
Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pp. 10524–10533. PMLR, 2020.   
Jiahui Yu and Thomas S Huang. Universally slimmable networks and improved training techniques. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1803–1811, 2019.   
Running Zhao, Jiangtao Yu, Hang Zhao, and Edith C.H. Ngai. Radio2text: Streaming speech recognition using mmwave radio signals. 7(3), sep 2023. doi: 10.1145/3610873. URL https://doi.org/10.1145/3610873.   
Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018.   
Zhuangdi Zhu, Junyuan Hong, and Jiayu Zhou. Data-free knowledge distillation for heterogeneous federated learning. In International Conference on Machine Learning, pp. 12878–12889. PMLR, 2021.

# A CENTERED KERNEL ALIGNMENT

Centered Kernel Alignment (CKA) originally serves as a similarity measure for different kernel functions (Cortes et al., 2012). Later, its purpose has been extended to discovering meaningful similarities between internal representations of neural networks (Kornblith et al., 2019). Compared with alternative methods to monitor representation learning, such as Canonical Correlation Analysis-based methods (Raghu et al., 2017; Morcos et al., 2018) and neuron alignment methods (Li et al., 2015; Wang et al., 2018), CKA achieves the state-of-the-art performance in measuring the difference between representations of neural network. This is based on the fact that CKA reliably identifies correspondences between representations from architecturally corresponding layers in two networks trained with different initializations. $^{4}$

Denote $X \in R^{n \times p}$ and $Y \in R^{n \times q}$ as two representations of n data points with possibly different dimensions (i.e., $p \neq q$ ). These two representations fall into the following three categories: (1) internal outputs at two different layers of an individual network, (2) internal layer outputs of two architecturally identical networks trained from different initialization or by different datasets, or (3) internal layer outputs of two networks with different architectures possibly trained by different datasets. The application of CKA in our paper belongs to the third category, where we examine the CKA similarities of a corresponding layer output between every pair of local client models in the context of federated learning.

Let $k_{x}(\cdot, \cdot)$ and $k_{y}(\cdot, \cdot)$ be the kernel functions for $X$ and $Y$ respectively. Then the resulted kernel matrices of $k_{x}$ and $k_{y}$ with respect to $\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}$ and $\mathbf{y}_{1}, \ldots, \mathbf{y}_{n}$ are $K_{x}$ and $K_{y}$ , whose $(i, j)$ -entries are $K_{x}(i, j) = k_{x}(\mathbf{x}_{i}, \mathbf{x}_{j})$ and $K_{y}(i, j) = k_{y}(\mathbf{y}_{i}, \mathbf{y}_{j})$ . Then CKA is defined as

$$
\operatorname{CKA} (K _ {x}, K _ {y}) := \frac {\operatorname{tr} (K _ {x} H K _ {y} H)}{\sqrt {\operatorname{tr} (K _ {x} H K _ {x} H) \operatorname{tr} (K _ {y} H K _ {x} H)}}, \tag {7}
$$

where $H = I_{n} - \frac{1}{n}\mathbf{11}^{\mathrm{T}}$ is the centering matrix.

As for the kernels in CKA, we select linear kernel (i.e., $K_{x} = XX^{T}$ , $K_{y} = YY^{T}$ ) $^{5}$ over Radial Basis Function (RBF) kernel $^{6}$ from common kernels for the following reasons. First, experiments in Kornblith et al. (2019) manifest that linear and RBF kernels work equally well in similarity measurement of feature representations. Furthermore, it is recently validated that CKA based on an RBF kernel converges to linear CKA in the large-bandwidth limit (Alvarez, 2023). Hence, in our investigation we stick with linear CKA for computational efficiency, where the resulting linear CKA is

$$
\begin{array}{l} \mathrm{CKA} _ {\text { linear }} (X, Y) = \mathrm{CKA} (X X ^ {\mathrm{T}}, Y Y ^ {\mathrm{T}}) \\ = \frac {\operatorname{tr} \left(X X ^ {\mathrm{T}} H Y Y ^ {\mathrm{T}} H\right)}{\sqrt {\operatorname{tr} \left(X X ^ {\mathrm{T}} H X X ^ {\mathrm{T}} H\right) \operatorname{tr} \left(Y Y ^ {\mathrm{T}} H Y Y ^ {\mathrm{T}} H\right)}} \\ = \frac {\operatorname{tr} \left(Y ^ {\mathrm{T}} H X X ^ {\mathrm{T}} H Y\right)}{\sqrt {\operatorname{tr} \left(X ^ {\mathrm{T}} H X X ^ {\mathrm{T}} H X\right) \operatorname{tr} \left(Y ^ {\mathrm{T}} H Y Y ^ {\mathrm{T}} H Y\right)}} \\ = \frac {\left| \left| Y ^ {\mathrm{T}} H X \right| \right| _ {\mathrm{F}} ^ {2}}{\left| \left| X ^ {\mathrm{T}} H X \right| \right| _ {\mathrm{F}} \left| \left| Y ^ {\mathrm{T}} H Y \right| \right| _ {\mathrm{F}}}. \tag {8} \\ \end{array}
$$

In our design, we measure the averaged CKA similarities according to the outputs from the same batch of test data. The range of CKA is between 0 and 1, and a higher CKA score means more similar paired features.

# B COMPARISONS WITH RESIDUAL CONNECTIONS

Remarks on self-mixture approaches in neural networks. The goal of residual connections is to avoid exploding and vanishing gradients to facilitate the training of a single model (He et al., 2016), while cross-layer gradients aim to increase the layer similarities across a group of models that are jointly optimized in federated learning. Specifically, residual connections modify forward passes by

adding the shallow-layer outputs to those of the deep layers. In contrast, cross-layer gradients operate on the gradients calculated by back-propagation. We present the distinct gradient outcomes of the two methods in the following.

![](images/d1b6ef4d55e9e2c8d13abbb59eb8ad101f6773b9b07ccb73b7615554c856f09c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x_{i-2}"] --> B["W_{i-1}"]
    B --> C["x_{i-1}"]
    C --> D["W_i"]
    D --> E["x_i"]
    E --> F["W_{i+1}"]
    F --> G["x_{i+1}"]
    H["g_{w_{i-1}}"] --> B
```
</details>

(a) Cross-layer gradients

![](images/778969c50fa90f49a657a12847594df6cb643b39966dbd4e0740b6fb24496620.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x_{i-2}"] --> B["W_{i-1}"]
    B --> C["x_{i-1}"]
    C --> D["W_i"]
    D --> E["x_i"]
    E --> F["W_{i+1}"]
    F --> G["x_{i+1}"]
    G --> H["x'_{i+1}"]
    H --> I["x_i"]
```
</details>

(b) Residual connections   
Figure 11: Comparison of cross-layer gradients and residual connections

Consider three consecutive layers of a feedforward neural network indexed by $i - 1, i, i + 1$ . With a slight abuse of symbols, we use $f(\cdot; W_k)$ to denote the calculation in the $k$ -th layer. Given the input $\mathbf{x}_{\mathbf{i} - 2}$ to layer $i - 1$ , the output from the previous layer becomes the input to the next layer, thus generating $\mathbf{x}_{\mathbf{i} - 1}, \mathbf{x}_{\mathbf{i}}, \mathbf{x}_{\mathbf{i} + 1}$ sequentially.

$$
\mathbf {x} _ {\mathbf {i} - \mathbf {1}} = f (\mathbf {x} _ {\mathbf {i} - \mathbf {2}}; W _ {i - 1})
$$

$$
\mathbf {x} _ {\mathbf {i}} = f \left(\mathbf {x} _ {\mathbf {i} - 1}; W _ {i}\right) \tag {9}
$$

$$
\mathbf {x} _ {\mathbf {i} + \mathbf {1}} = f (\mathbf {x} _ {\mathbf {i}}; W _ {i + 1})
$$

In the case of residual connections, there is an additional operation that directs $x_{i}$ to $x_{i+1}$ , formulated as $x_{i+1}' = x_{i+1} + x_{i}$ . The gradient of $W_{i}$ is

$$
\begin{array}{l} g _ {W _ {i}} = \frac {\partial l o s s}{\partial W _ {i}} \\ = \frac {\partial \text {loss}}{\partial \mathbf {x} _ {\mathrm{i+1}} ^ {\prime}} \cdot \frac {\partial \mathbf {x} _ {\mathrm{i+1}} ^ {\prime}}{\partial \mathbf {x} _ {\mathrm{i}}} \cdot \frac {\partial \mathbf {x} _ {\mathrm{i}}}{\partial W _ {i}} (10) \\ = \frac {\partial \text { loss }}{\partial \left(\mathbf {x} _ {\mathbf {i} + 1} + \mathbf {x} _ {\mathbf {i}}\right)} \cdot \left(\frac {\partial \mathbf {x} _ {\mathbf {i} + 1}}{\partial \mathbf {x} _ {\mathbf {i}}} + \mathbb {I}\right) \cdot \frac {\partial \mathbf {x} _ {\mathbf {i}}}{\partial W _ {i}} (10) \\ = \frac {\partial l o s s}{\partial (\mathbf {x _ {i + 1}} + \mathbf {x _ {i}})} \cdot \left(\frac {\partial \mathbf {x _ {i + 1}}}{\partial W _ {i}} + \frac {\partial \mathbf {x _ {i}}}{\partial W _ {i}}\right) \\ \end{array}
$$

In the case of cross-layer gradients, the gradient of $W_{i}$ is

$$
\begin{array}{l} g _ {W _ {i}} = \frac {\partial \text {loss}}{\partial W _ {i}} + \frac {\partial \text {loss}}{\partial W _ {i - 1}} \tag {11} \\ = \frac {\partial l o s s}{\partial \mathbf {x} _ {i + 1}} \cdot \left(\frac {\partial \mathbf {x _ {i + 1}}}{\partial W _ {i}} + \frac {\partial \mathbf {x _ {i + 1}}}{\partial W _ {i - 1}}\right) \\ \end{array}
$$

We note that both residual connections and cross-layer gradients are subject to certain constraints. Residual connections require identical shapes for the layer outputs, while cross-layer gradients operate on the layer weights with the same shape.

# C PROOF OF THEOREM 3.1

This section demonstrates the details of the proof of Theorem 3.1. We will present the proof of Theorem 3.1 in the vector form and the matrix form.

# C.1 VECTOR FORM

We state the convex optimization problem Theorem 1 in the vector form in the following,

$$
\min _ {g _ {o p t}} \quad | | g _ {k} - g _ {o p t} | | _ {2} ^ {2}, \tag {12}
$$

$$
s. t. \quad \langle g _ {o p t}, g _ {0} \rangle \geq 0.
$$

Because the superscript $t$ would not influence the proof of the theorem, we simplify the notation $g^t$ to $g$ . We use Equation 12 instead of Equation 1 to complete this proof. The Lagrangian of Equation 12 is shown as,

$$
\begin{array}{l} L (g _ {o p t}, \lambda) = (g _ {k} - g _ {o p t}) ^ {T} (g _ {k} - g _ {o p t}) - \lambda g _ {o p t} ^ {T} g _ {0} \\ = g _ {k} ^ {T} g _ {k} - g _ {\text { opt }} ^ {T} g _ {k} - g _ {k} ^ {T} g _ {\text { opt }} + g _ {\text { opt }} ^ {T} g _ {\text { opt }} - \lambda g _ {\text { opt }} ^ {T} g _ {0} \tag {13} \\ = g _ {k} ^ {T} g _ {k} - 2 g _ {o p t} ^ {T} g _ {k} + g _ {o p t} ^ {T} g _ {o p t} - \lambda g _ {o p t} ^ {T} g _ {0}. \\ \end{array}
$$

Let $\frac{\partial L(g_{opt},\lambda)}{\partial g_{opt}} = 0$ , we have

$$
g _ {o p t} = g _ {k} + \lambda g _ {0} / 2, \tag {14}
$$

which is the optimum point for the primal problem Equation 12. To get the Lagrange dual function $L(\lambda) = \inf_{g_{opt}} L(g_{opt}, \lambda)$ , we substitute $g_{opt}$ by $g_k + \lambda g_0 / 2$ in $L(g_{opt}, \lambda)$ . We have

$$
\begin{array}{l} L (\lambda) = g _ {k} ^ {T} g _ {k} - 2 \left(g _ {k} + \frac {\lambda g _ {0}}{2}\right) ^ {T} g _ {k} + \left(g _ {k} + \frac {\lambda g _ {0}}{2}\right) ^ {T} \left(g _ {k} + \frac {\lambda g _ {0}}{2}\right) - \lambda \left(g _ {k} + \frac {\lambda g _ {0}}{2}\right) ^ {T} g _ {0} \\ = g _ {k} ^ {T} g _ {k} - 2 g _ {k} ^ {T} g _ {k} - \lambda g _ {0} ^ {T} g _ {k} + g _ {k} ^ {T} g _ {k} + \frac {\lambda g _ {0} ^ {T} g _ {k}}{2} + \frac {\lambda g _ {k} ^ {T} g _ {0}}{2} + \frac {\lambda^ {2} g _ {0} ^ {T} g _ {0}}{4} - \lambda g _ {k} ^ {T} g _ {0} - \frac {\lambda^ {2} g _ {0} ^ {T} g _ {0}}{2} \\ = g _ {k} ^ {T} g _ {k} - 2 g _ {k} ^ {T} g _ {k} + g _ {k} ^ {T} g _ {k} - \lambda g _ {0} ^ {T} g _ {k} + \lambda g _ {0} ^ {T} g _ {k} + \frac {\lambda^ {2} g _ {0} ^ {T} g _ {0}}{4} - \frac {\lambda^ {2} g _ {0} ^ {T} g _ {0}}{2} - \lambda g _ {k} ^ {T} g _ {0} \\ = - \frac {g _ {0} ^ {T} g _ {0}}{4} \lambda^ {2} - g _ {k} ^ {T} g _ {0} \lambda . \tag {15} \\ \end{array}
$$

Thus, the Lagrange dual problem is described as follows,

$$
\max _ {\lambda} L (\lambda) = - \frac {g _ {0} ^ {T} g _ {0}}{4} \lambda^ {2} - g _ {k} ^ {T} g _ {0} \lambda , \tag {16}
$$

$$
s. t. \lambda \geq 0.
$$

$L(\lambda)$ is a quadratic function. Because $g_0^T g_0 \geq 0$ , the maximum of $L(\lambda)$ is at the point $\lambda = -\frac{2b}{a}$ where $a = g_0^T g_0$ and $b = g_k^T g_0$ if we do not consider the constraint. It is clear that this convex optimization problem holds strong duality because it satisfies Slater's constraint qualification(Boyd et al., 2004), which indicates that the optimum point of the dual problem Equation 16 is also the optimum point for the primal problem Equation 12. We substitute $\lambda$ by $-\frac{2b}{a}$ in Equation 14, and we have

$$
g _ {o p t} = \left\{ \begin{array}{l l} g _ {k}, & \text { if } b \geq 0, \\ g _ {k} - \theta^ {t} g _ {0}, & \text { if } b <   0, \end{array} \right. \tag {17}
$$

where $\theta^t = \frac{b}{a}$ , $a = (g_0)^T g_0$ and $b = g_k^T g_0$ . We add the superscript $t$ to all gradients, and we finish the proof of Theorem 3.1.

# C.2 MATRIX FORM

The proof of the matrix form is similar to Appendix C.1. We update Equation 12 to the matrix form as follows,

$$
\min _ {G _ {\text { opt }}} \left| \left| G _ {k} - G _ {\text { opt }} \right| \right| _ {F} ^ {2}, \tag {18}
$$

$$
s. t. \quad \langle G _ {o p t}, G _ {0} \rangle \geq 0.
$$

Similar to Equation 13, the Lagragian of Equation 18 is,

$$
L (G _ {o p t}, \lambda) = t r (G _ {k} ^ {T} G _ {k}) - t r (G _ {o p t} ^ {T} G _ {k}) - t r (G _ {k} ^ {T} G _ {o p t}) + t r (G _ {o p t} ^ {T} G _ {o p t}) - \lambda t r (G _ {o p t} ^ {T} G _ {0}), \tag {19}
$$

where $tr(A)$ means the trace of the matrix $A$ . We can obtain the optimum point for Equation 18 according to $\frac{\partial L(G_{opt},\lambda)}{\partial G_{opt}} = 0$ . We have

$$
G _ {o p t} = G _ {k} + \lambda G _ {0} / 2. \tag {20}
$$

Similar to the analysis in Appendix C.1 and Equation 15, we get the Lagrange dual problem as follows,

$$
\max _ {\lambda} L (\lambda) = - \frac {t r (G _ {0} ^ {T} G _ {0})}{4} \lambda^ {2} - t r (G _ {k} ^ {T} G _ {0}) \lambda , \tag {21}
$$

$$
s. t. \lambda \geq 0,
$$

where $tr(G_0^T G_0) \geq 0$ . Following the same analysis in Appendix C.1, we have

$$
G _ {o p t} = \left\{ \begin{array}{l l} G _ {k}, & \text { if } b \geq 0, \\ G _ {k} - \theta^ {t} G _ {0}, & \text { if } b <   0, \end{array} \right. \tag {22}
$$

where $\theta^t = \frac{b}{a}$ , $a = tr(G_0^T G_0)$ and $b = tr(G_k^T G_0)$ . At last, we have finished the proof of the matrix form of Theorem 3.1.

# D PROOF OF CONVERGENCE ANALYSIS

We show the details of convergence analysis for cross-layer gradients. $W_{t,e}^{l_{i}}$ are the weights from the layers which need cross-layer gradients at round t of the local step e. To simplify the notations, we use $W_{t,e}^{l_{i}}$ instead of $W_{t,e}^{l_{i}}$ .

Lemma D.1. (Per Round Progress.) Suppose our functions satisfy Assumption 4.1 and Assumption 4.2. The expectation of a loss function of any arbitrary clients at communication round t after E local steps are bounded as,

$$
\mathbb {E} [ L _ {t, E - 1} ] \leq \mathbb {E} [ L _ {t, 0} ] - (\eta - \frac {L \eta^ {2}}{2}) \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} + \frac {L E \eta^ {2}}{2} \sigma^ {2}. \tag {23}
$$

Proof.

Considering an arbitrary client, we omit the client index $i$ in this lemma. Let $W_{t,e + 1} = W_{t,e} - \eta g_{t,e}$ , we have

$$
L _ {t, e + 1} \leq L _ {t, e} + \left\langle \nabla L _ {t, e}, W _ {t, e + 1} - W _ {t, e} \right\rangle + \frac {L}{2} \left| \left| W _ {t, e + 1} - W _ {t, e} \right| \right| ^ {2} \tag {24}
$$

$$
\leq L _ {t, e} - \eta \langle \nabla L _ {t, e}, g _ {t, e} \rangle + \frac {L}{2} | | \eta g _ {t, e} | | ^ {2},
$$

where Equation 24 follows Assumption 4.1. We take expectation on both sides of Equation 24, then

$$
\begin{array}{l} \mathbb {E} [ L _ {t, e + 1} ] \leq \mathbb {E} [ L _ {t, e} ] - \eta \mathbb {E} [ \langle \nabla L _ {t, e}, g _ {t, e} \rangle ] + \frac {L}{2} \mathbb {E} [ | | \eta g _ {t, e} | | ^ {2} ] \\ = \mathbb {E} [ L _ {t, e} ] - \eta | | \nabla L _ {t, e} | | ^ {2} + \frac {L \eta^ {2}}{2} \mathbb {E} [ | | g _ {t, e} | | ^ {2} ] \\ \stackrel {(a)} {=} \mathbb {E} [ L _ {t, e} ] - \eta | | \nabla L _ {t, e} | | ^ {2} + \frac {L \eta^ {2}}{2} (\mathbb {E} [ | | g _ {t, e} | | ] ^ {2} + V a r (| | g _ {t, e} | |)) \tag {25} \\ = \mathbb {E} [ L _ {t, e} ] - \eta | | \nabla L _ {t, e} | | ^ {2} + \frac {L \eta^ {2}}{2} (| | \nabla L _ {t, e} | | ^ {2} + V a r (| | g _ {t, e} | |)) \\ \stackrel {(b)} {\leq} \mathbb {E} [ L _ {t, e} ] - (\eta - \frac {L \eta^ {2}}{2}) | | \nabla L _ {t, e} | | ^ {2} + \frac {L \eta^ {2}}{2} \sigma^ {2}, \\ \end{array}
$$

where (a) follows $Var(X) = \mathbb{E}[X^2] - \mathbb{E}^2[X]$ , and (b) is Assumption 4.2. Telescoping local step 0 to $E - 1$ , we have

$$
\mathbb {E} [ L _ {t, E - 1} ] \leq \mathbb {E} [ L _ {t, 0} ] - (\eta - \frac {L \eta^ {2}}{2}) \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} + \frac {L E \eta^ {2}}{2} \sigma^ {2}, \tag {26}
$$

then we finish the proof of Lemma D.1.

Lemma D.2. (Bound Client Dirft.) Suppose our functions satisfy Assumption 4.2, Assumption 4.3 and Assumption 4.4. After each aggregation, the updates, $\Delta W$ , for the layers need cross-layer gradients have bounded drift:

$$
\mathbb {E} [ | | \Delta W | | ^ {2} ] \leq 2 \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma). \tag {27}
$$

Proof.

We have $W_{t + 1,0} - W_{t,E - 1} = \Delta W = \eta (g_{l_0} + g_{l_i}),\forall l_i$ need cross-layer gradients. Because all gradients are in the same aggregation round, we omit the time subscript in this proof process. Since $\eta$ is a constant, we also simplify it. $g_{l_0}$ and $g_{l_i}$ the gradients from the same client, indicating that they are dependent, then

$$
\left| \left| \Delta W \right| \right| ^ {2} = \left| \left| g _ {l _ {0}} + g _ {l _ {i}} \right| \right| ^ {2} \tag {28}
$$

$$
\stackrel {(c)} {\leq} | | g _ {l _ {0}} | | ^ {2} + 2 | | \langle g _ {l _ {0}}, g _ {l _ {i}} \rangle | | + | | g _ {l _ {i}} | | ^ {2},
$$

where (c) is Cauchy–Schwarz inequality. We take the expectation on both sides, then

$$
\begin{array}{l} \mathbb {E} [ | | \Delta W | | ^ {2} ] \leq \mathbb {E} [ | | g _ {l _ {0}} | | ^ {2} ] + 2 \mathbb {E} [ | | \langle g _ {l _ {0}}, g _ {l _ {i}} \rangle | | ] + \mathbb {E} [ | | g _ {l _ {i}} | | ^ {2} ] \\ \stackrel {(a)} {=} \mathbb {E} [ | | g _ {l _ {0}} | | ] ^ {2} + V a r (| | g _ {l _ {0}} | |) + \mathbb {E} [ | | g _ {l _ {i}} | | ] ^ {2} + V a r (| | g _ {l _ {i}} | |) + 2 \mathbb {E} [ | | \langle g _ {l _ {0}}, g _ {l _ {i}} \rangle | | ] \\ \stackrel {(d)} {\leq} 2 \left(\rho^ {2} + \sigma^ {2}\right) + 2 \mathbb {E} \left[ \left| \left| g _ {l _ {0}}, g _ {l _ {i}} \right| \right| \right] \tag {29} \\ \stackrel {(e)} {=} 2 (\rho^ {2} + \sigma^ {2}) + 2 (C o v (g _ {l _ {0}}, g _ {l _ {i}}) + \mathbb {E} [ | | g _ {l _ {0}} | | ] \mathbb {E} [ | | g _ {l _ {i}} | | ]) \\ \stackrel {(f)} {\leq} 2 (\rho^ {2} + \sigma^ {2}) + 2 (\Gamma + \rho^ {2}) \\ = 4 \rho^ {2} + 2 \sigma^ {2} + 2 \Gamma , \\ \end{array}
$$

where (d) follows assumption Assumption 4.2 and Assumption 4.3, (e) follows the covariance formula, and (f) follows assumption Assumption 4.4. We put back $\eta^{2}$ to the final step of Equation 29. At last, we complete the proof of Lemma D.2.

# D.1 PROOF OF THEOREM 4.5 AND THEOREM 4.6

We state Theorem 4.5 again in the following,

(Per round drift) Supposed Assumption 4.1 to Assumption 4.4 are satisfied, the loss function of an arbitrary client at round $t + 1$ is bounded by,

$$
\mathbb {E} \left[ L _ {t + 1, 0} \right] \leq \mathbb {E} \left[ L _ {t, 0} \right] - \left(\eta - \frac {L \eta^ {2}}{2}\right) \sum_ {e = 0} ^ {E - 1} \left| | \nabla L _ {t, e} | \right| ^ {2} + \tag {30}
$$

$$
\frac {L E \eta^ {2}}{2} \sigma^ {2} + 2 \eta (\Gamma + \rho^ {2}) + L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma).
$$

Proof.

Following the Assumption 4.1, we have

$$
L _ {t + 1, 0} \leq L _ {t, E - 1} + \left\langle \nabla L _ {t, E - 1}, W _ {t + 1, 0} - W _ {t, E - 1} \right\rangle + \frac {L}{2} \left| \left| W _ {t + 1, 0} - W _ {t, E - 1} \right| \right| ^ {2} \tag {31}
$$

$$
= L _ {t, E - 1} + \eta \langle \nabla L _ {t, E - 1}, g _ {l _ {0}} + g _ {l _ {1}} \rangle + \frac {L}{2} \eta^ {2} | | \Delta W | | ^ {2}.
$$

Taking the expectation on both sides, we obtain

$$
\mathbb {E} \left[ L _ {t + 1, 0} \right] = \mathbb {E} \left[ L _ {t, E - 1} \right] + \eta \mathbb {E} \left[ \left\langle \nabla L _ {t, E - 1}, g _ {l _ {0}} + g _ {l _ {1}} \right\rangle \right] + \frac {L}{2} \eta^ {2} \mathbb {E} [ | | \Delta W | | ^ {2} ]. \tag {32}
$$

The first item is Lemma D.1, and the third item is Lemma D.2. We consider the second item $\mathbb{E}[\langle \nabla L_{t,E - 1},g_{l_0} + g_{l_1}\rangle ]$ in the following, then,

$$
\begin{array}{l} \mathbb {E} \left[ \left\langle \nabla L _ {t, E - 1}, g _ {l _ {0}} + g _ {l _ {1}} \right\rangle \right] = \mathbb {E} \left[ \nabla L _ {t, E - 1} g _ {l _ {0}} \right] + \mathbb {E} \left[ \nabla L _ {t, E - 1} g _ {l _ {1}} \right] \\ \stackrel {(e)} {=} C o v \left(\nabla L _ {t, E - 1}, g _ {l _ {0}}\right) + \mathbb {E} \left[ \left| \left| \nabla L _ {t, E - 1} \right| \right| \right] \mathbb {E} \left[ \left| \left| g _ {l _ {0}} \right| \right| \right] + C o v \left(\nabla L _ {t, E - 1}, g _ {l _ {1}}\right) + \mathbb {E} \left[ \left| \left| \nabla L _ {t, E - 1} \right| \right| \right] \mathbb {E} \left[ \left| \left| g _ {l _ {1}} \right| \right| \right] \tag {33} \\ \stackrel {(f)} {\leq} 2 (\Gamma + \rho^ {2}). \\ \end{array}
$$

Combining two lemmas and Equation 33, we have

$$
\mathbb {E} \left[ L _ {t + 1, 0} \right] \leq \mathbb {E} \left[ L _ {t, 0} \right] - \left(\eta - \frac {L \eta^ {2}}{2}\right) \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} + \frac {L E \eta^ {2}}{2} \sigma^ {2} + 2 \eta (\Gamma + \rho^ {2}) + L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma), \tag {34}
$$

then we finish the proof of Theorem 4.5.

For Theorem 4.6, we consider the sum of the second term to the last term in Equation 34 to be smaller than 0, i.e.,

$$
- (\eta - \frac {L \eta^ {2}}{2}) \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} + \frac {L E \eta^ {2}}{2} \sigma^ {2} + 2 \eta (\Gamma + \rho^ {2}) + L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma) <   0, \tag {35}
$$

then, we have

$$
\eta <   \frac {2 \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t , e} | | ^ {2} - 4 (\Gamma + \rho^ {2})}{L \left(\sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t , e} | | ^ {2} + E \rho^ {2} + 2 \left(2 \rho^ {2} + \sigma^ {2} + \Gamma\right) \right.}. \tag {36}
$$

We finish the proof of Theorem 4.6.

# D.2 PROOF OF THEOREM 4.7

Telescoping the communication rounds from $t = 0$ to $t = T - 1$ with the local step from $e = 0$ to $e = E - 1$ on the expectation on both sides of Equation 34, we have

$$
\frac {1}{T E} \sum_ {t = 0} ^ {T - 1} \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} \leq \frac {\frac {2}{T E} \sum_ {t = 0} ^ {T - 1} (\mathbb {E} [ L _ {t , 0} ] - \mathbb {E} [ L _ {t + 1 , 0} ]) + L \eta^ {2} \sigma^ {2} + 4 \eta (\Gamma + \rho^ {2}) + 2 L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma)}{2 \eta - L \eta^ {2}}. \tag {37}
$$

Given any $\epsilon > 0$ , let

$$
\frac {\frac {2}{T E} \sum_ {t = 0} ^ {T - 1} (\mathbb {E} [ L _ {t , 0} ] - \mathbb {E} [ L _ {t + 1 , 0} ]) + L \eta^ {2} \sigma^ {2} + 4 \eta (\Gamma + \rho^ {2}) + 2 L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma)}{2 \eta - L \eta^ {2}} \leq \epsilon , \tag {38}
$$

and we denote $\kappa = L_0 - L^*$ , then Equation 38 becomes

$$
\frac {\frac {2 \kappa}{T E} + L \eta^ {2} \sigma^ {2} + 4 \eta (\Gamma + \rho^ {2}) + 2 L \eta^ {2} (2 \rho^ {2} + \sigma^ {2} + \Gamma)}{2 \eta - L \eta^ {2}} \leq \epsilon , \tag {39}
$$

because $\sum_{t=0}^{T-1} (\mathbb{E}[L_{t,0}] - \mathbb{E}[L_{t+1,0}]) \leq \kappa$ . We consider $T$ in Equation 39, i.e.,

$$
T \geq \frac {2 \kappa}{E \eta ((2 - L \eta) \epsilon - 3 L \eta \sigma^ {2} - 2 (2 + L \eta) \Gamma - 4 (1 + L \eta) \rho^ {2})}, \tag {40}
$$

then, we have

$$
\frac {1}{T E} \sum_ {t = 0} ^ {T - 1} \sum_ {e = 0} ^ {E - 1} | | \nabla L _ {t, e} | | ^ {2} \leq \epsilon , \tag {41}
$$

when

$$
\eta <   \frac {2 \epsilon - 4 (\Gamma + \rho^ {2})}{L (\epsilon + E \rho^ {2} + 2 (2 \rho^ {2} + \sigma^ {2} + \Gamma)}. \tag {42}
$$

We complete the proof of Theorem 4.7.

# E MORE RELATED WORKS

# E.1 FEDERATED LEARNING.

In 2017, Google proposed a novel machine learning technique, i.e., Federated Learning (FL), to organize collaborative computing among edge devices or servers (McMahan et al., 2017). It enables multiple clients to collaboratively train models while keeping training data locally, facilitating privacy protection. Various synchronous or asynchronous FL schemes have been proposed and achieved good performance in different scenarios. For example, FedAvg (McMahan et al., 2017) takes a weighted average of the models trained by local clients and updates the local models iteratively. FedProx (Li et al., 2020b) generalized and re-parametrized FedAvg, guaranteeing the convergence when learning over non-iid data. FedAsyn (Xie et al., 2020) employed coordinators and schedulers to achieve an asynchronous training process.

# E.2 HETEROGENEOUS MODELS.

The clients in homogeneous federated learning frameworks have identical neural network architectures, while the edge devices or servers in real-world settings show great diversity. They usually have different memory and computation capabilities, making it difficult to deploy the same machine-learning model in all the clients. Therefore, researchers have proposed various methods supporting heterogeneous models in the FL environment.

Knowledge Distillation. Knowledge distillation (KD) (Hinton et al., 2015) was proposed by Hinton et al., aiming to train a student model with the knowledge distilled from a teacher model, which becomes an important research area in Machine Learning (Gou et al., 2021; Zhao et al., 2023). Inspired by the knowledge distillation, several studies(Li & Wang, 2019; Li et al., 2021b; He et al., 2020) are proposed to address the system heterogeneity problem. In FedMD(Li & Wang, 2019), the clients distill and transmit logits from a large public dataset, which helps them learn from both logits and private local datasets. In RHFL (Fang & Ye, 2022), the knowledge is distilled from the unlabeled dataset and the weights of clients are computed by the symmetric cross-entropy loss function. Unlike the aforementioned methods, data-free KD is a new approach to completing the knowledge distillation process without the training data. The basic idea is to optimize noise inputs to minimize the distance to prior knowledge(Nayak et al., 2019). Chen et al.(Chen et al., 2019) train Generative Adversarial Networks (GANs)(Goodfellow et al., 2014) to generate training data for the entire KD process utilizing the knowledge distilled from the teacher model. In FedHe(Chan & Ngai, 2021), a server directly averages the logits transmitted from clients. FedGen(Zhu et al., 2021) adopts a generator to simulate the prior knowledge from all the clients, which is used along with the private data from clients in local training. In FedGKT(He et al., 2020), a neural network is separated into two segments, one held by clients, the other preserved in a server, in which the features and logits from clients are sent to the server to train the large model. In Felo (Chan & Ngai, 2022), the representations from the intermediate layers are the knowledge instead of directly using the logits.

Public or Generated Data. In FedML (Shen et al., 2020), latent information from homogeneous models is applied to train heterogeneous models. FedAUX (Sattler et al., 2021) initialized heterogeneous models by unsupervised pre-training and unlabeled auxiliary data. FCCL (Huang et al., 2022) calculate a cross-correlation matrix according to the global unlabeled dataset to exchange knowledge. However, these methods require a public dataset. The server might not be able to collect sufficient data due to data availability and privacy concerns.

Model Compression. Although HeteroFL (Diao et al., 2021) derives local models with different sizes from one large model, the architectures of local and global models still have to share the same model architecture, and it is inflexible that all models have to be retrained when the best participant joins or leaves the FL training process. Federated Dropout (Caldas et al., 2018) randomly selects sub-models from the global models following the dropout way. SlimFL (Baek et al., 2022) incorporated width-adjustable slimmable neural network (SNN) architectures(Yu & Huang, 2019) into FL which can tune the widths of local neural networks. FjORD (Horvath et al., 2021) tailored model widths to clients' capabilities by leveraging Ordered Dropout and a self-distillation methodology. FedRoLex (Alam et al., 2022) proposes a rolling sub-model extraction scheme to adapt to the heterogeneous model environment. However, similar to HeteroFL, they only vary the number of parameters for each layer.

# F PROBLEM FORMULATION

In this section, we introduce federated learning with model heterogeneity. Federated learning aims to foster collaboration with clients to jointly train a shared global model while preserving the privacy of their local data. However, in the context of model heterogeneity, it becomes challenging to maintain the same architectures across all clients. Specifically, we consider a set of physical resources denoted as $\{R_{i}\}_{i=1}^{n}$ , where $R_{i}$ represents the available resources of client i. For each local client model $\{M_{i}\}_{i=1}^{n}$ , the resource requirement $R(M_{i})$ must be smaller than or equal to the available resources of client i, i.e. $\{R(M_{i}) \leq R_{i}\}_{i=1}^{n}$ . To satisfy this constraint, the client models have varying sizes and architectures. Let $w_{i}$ denote the weights of the client model $M_{i}$ , and $f(x, w_{i})$ represent the forward function of model $M_{i}$ with input x. Moreover, each client has a local dataset $D_{i} = \{(x_{k,i}, y_{k,i}) | k \in \{1, 2, ..., |D_{i}|\}\}$ , where $|D_{i}|$ signifies the size of a dataset $D_{i}$ . The loss

function $l_{i}$ of client i is shown as follows,

$$
\min _ {w} \quad l _ {i} (w _ {i}) = \frac {1}{| D _ {i} |} \sum_ {k = 1} ^ {| D _ {i} |} l _ {C S E} (f (x _ {k}; w _ {i}), y _ {k}), \tag {43}
$$

where $l_{CSE}$ is a cross-entropy function. Moreover, if we denote $K = \sum_{i=1}^{n} |D_i|$ as the total size of all local datasets, the global optimization problem is,

$$
\min _ {w _ {1}, w _ {2}, \dots , w _ {n}} L (w _ {1}, \dots , w _ {n}) = \sum_ {i = 1} ^ {n} \frac {\left| D _ {i} \right|}{K} l _ {i} (w _ {i}), \tag {44}
$$

where the optimized model weights $\{w_{1}, w_{2}, ..., w_{n}\}$ are the parameters from $M_{i=1}^{n}$ . In our case, $\{w_{1}, w_{2}, ..., w_{n}\}$ are split from the server model weight $w_{s}$ from the server model $M_{s}$ . The goal of our paper is to propose a method that can effectively optimize Equation 44.

# G CONFIGURATIONS AND MORE RESULTS OF THE CASE STUDY

# G.1 CONFIGURATIONS

In the case study, we have five ResNet models which are stage splitting from ResNet26, resulting in ResNet10, ResNet14, ResNet18, ResNet22, and ResNet26. Five ViTs models are ViT-S/8, ViT-S/9, ViT-S/10, ViT-S/11, ViT-S/12, the results from the layer splitting of ViT-S/12. The model prototypes are the same as the experiment settings. To quantify a model's degree of bias towards its local dataset, we use CKA similarities among the clients based on the outputs from the same stages in ResNet (ResNets of different sizes always contain four stages) and the outputs from the same layers in Vision Transformers (ViTs) (Dosovitskiy et al., 2020). Specifically, we measure the averaged CKA similarities according to the outputs from the same batch of test data. The range of CKA is between 0 and 1, and a higher CKA score means more similar paired features. We train FedAvg under three settings: IID with the homogeneous setting, Non-IID with the homogeneous setting, and Non-IID with the heterogeneous setting. FedAvg only aggregates gradients from the models sharing the same architectures under the heterogeneous model setting (Lin et al., 2020). For ResNets, we conduct training 100 communication rounds, while only 20 rounds for ViTs. The local training epochs for clients are five for all settings. We use Adam(Kingma & Ba, 2015) optimizer with default parameter settings for all client models, and the batch size is 64. We use two small federated scales. One is ten clients deployed the same model architecture (ResNet18 for ResNets and ViT-S/12 for ViTs), which is called a homogeneous setting. The other is ten clients with five different model architectures, which is a heterogeneous setting. This setting means that we have five groups whose architectures are heterogeneous, but the clients belonging to the same group have the same architectures.

![](images/9f61289e6935e48da79075da2fa59edc21ab492e32927e4cce11e73487956b23.jpg)

<details>
<summary>line</summary>

| Rounds | blocks.7.norm1 | blocks.7.norm2 |
| ------ | -------------- | -------------- |
| 0      | 0.007          | 0.004          |
| 2      | 0.005          | 0.016          |
| 6      | 0.024          | 0.003          |
| 10     | 0.007          | 0.005          |
| 14     | 0.020          | 0.004          |
| 18     | 0.015          | 0.006          |
</details>

(a) Similarity of gradients in Block 7.

![](images/86939dd0845e9e47cd579de4de6cd9967545694ce7aff78c8c3d6a81ad85ab7e.jpg)

<details>
<summary>line</summary>

| Rounds | blocks.11.norm1 | blocks.11.norm2 |
| ------ | ---------------- | ---------------- |
| 0      | 0.01             | 0.00             |
| 2      | 0.045            | 0.01             |
| 4      | 0.02             | 0.00             |
| 6      | 0.015            | 0.03             |
| 8      | 0.06             | 0.04             |
| 10     | 0.07             | 0.03             |
| 12     | 0.03             | 0.01             |
| 14     | 0.025            | 0.01             |
| 16     | 0.03             | 0.015            |
| 18     | 0.03             | 0.01             |
</details>

(b) Similarity of gradients in Block 11.

![](images/cd7af0c0d3bff87fc4322574c184eea3dd10d889a7cf2309e5e6400b56448e66.jpg)

<details>
<summary>line</summary>

| Rounds | CKA    | Accuracy |
| ------ | ------ | -------- |
| 0      | 0.940  | 45       |
| 10     | 0.950  | 50       |
| 20     | 0.960  | 55       |
| 30     | 0.965  | 60       |
| 40     | 0.970  | 65       |
</details>

(c) The relations between CKA and accuracy in stage1.

![](images/32a6cddcca1907469e0b9e08aa107801da613e4e1381396f24f24f31e6325b50.jpg)

<details>
<summary>line</summary>

| Rounds | CKA    | Accuracy |
| ------ | ------ | -------- |
| 0      | 0.915  | 45       |
| 10     | 0.930  | 50       |
| 20     | 0.938  | 55       |
| 30     | 0.945  | 60       |
| 40     | 0.948  | 65       |
| 45     | 0.950  | 65       |
</details>

(d) The relations between CKA and accuracy in stage2.   
Figure 12: Cross-environment similarity and more results between accuracy and CKA similarity. (a) and (b): Cross-environment similarity from Block 7 and Block 11 from ViTs. (c) and (d): The positive relation between stage1 and stage2.

# G.2 RELATIONS BETWEEN CKA AND ACCURACY.

In this subsection, we continue to describe more results about the relations between CKA similarity and accuracy. Similar to Figure 1c from stage0, Figure 12c and Figure 12d show the positive relations between CKA and accuracy from stage1 and stage2.

# G.3 GRADIENT ANALYSIS FOR VITs

Similar to the gradient analyses conducted for ResNets, we have performed the analysis of gradient distributions for ViTs. In our investigation, we have analyzed the outputs from the norm1 and norm2 layers within the ViT blocks and have also applied InCo Aggregation to these layers. The selection

![](images/ca6b3c50feec127f5617b18514f57b288f6dba23af755de2dd359872700c73ca.jpg)

<details>
<summary>line</summary>

| Gradient value | Density (Round 10) | Density (Round 11) | Density (Round 12) | Density (Round 13) | Density (Round 14) | Density (Round 15) | Density (Round 16) | Density (Round 17) | Density (Round 18) | Density (Round 19) |
| -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| -0.0004        | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   |
| -0.0002        | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 |
| 0.0000         | 400                 | 400                 | 400                 | 400                 | 400                 | 400                 | 400                 | 400                 | 400                 | 400                 |
| 0.0002         | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 | 350                 |
| 0.0004         | 300                 | 300                 | 300                 | 300                 | 300                 | 300                 | 300                 | 300                 | 300                 | 300                 |
| 0.0006         | 250                 | 250                 | 250                 | 250                 | 250                 | 250                 | 250                 | 250                 | 250                 | 250                 |
| 0.0010         | 200                 | 200                 | 200                 | 200                 | 200                 | 200                 | 200                 | 200                 | 200                 | 200                 |
</details>

(a) Block7 norm1 in IID with homo.

![](images/c60f85581efb06fa1a666623065c17620f1bbb2241cbd45343f79ec7ef6dd184.jpg)

<details>
<summary>line</summary>

| Gradient value | Round 10 | Round 11 | Round 12 | Round 13 | Round 14 | Round 15 | Round 16 | Round 17 | Round 18 | Round 19 |
| -------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| -0.0004        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| -0.0002        | 350      | 350      | 350      | 350      | 350      | 350      | 350      | 350      | 350      | 350      |
| 0.0000         | 300      | 300      | 300      | 300      | 300      | 300      | 300      | 300      | 300      | 300      |
| 0.0002         | 250      | 250      | 250      | 250      | 250      | 250      | 250      | 250      | 250      | 250      |
| 0.0004         | 200      | 200      | 200      | 200      | 200      | 200      | 200      | 200      | 200      | 200      |
| 0.0006         | 150      | 150      | 150      | 150      | 150      | 150      | 150      | 150      | 150      | 150      |
| 0.0008         | 100      | 100      | 100      | 100      | 100      | 100      | 100      | 100      | 100      | 100      |
| 0.0010         | 50       | 50       | 50       | 50       | 50       | 50       | 50       | 50       | 50       | 50       |
</details>

(b) Block7 norm2 in IID with homo.

![](images/4709c6dc678a082222f107ea563211afc2cba42932faae32fd2d0c0d5bbc58c2.jpg)

<details>
<summary>line</summary>

| Gradient value | Round 10 | Round 11 | Round 12 | Round 13 | Round 14 | Round 15 | Round 16 | Round 17 | Round 18 | Round 19 |
| -------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| -0.0004        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| -0.0002        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| 0.0002         | 600      | 550      | 500      | 450      | 400      | 350      | 300      | 250      | 200      | 150      |
| 0.0004         | 550      | 500      | 450      | 400      | 350      | 300      | 250      | 200      | 150      | 100      |
| 0.0006         | 500      | 450      | 400      | 350      | 300      | 250      | 200      | 150      | 100      | 50       |
| 0.0010         | 450      | 400      | 350      | 300      | 250      | 200      | 150      | 100      | 50       | 25       |
| -              | -        | -        | -        | -        | -        | -        | -        | -        | -        | -        |
</details>

(c) Block11 norm1 in IID with homo.

![](images/e015b2986c487322e52299b58aac8c17e7320c9dcc894e19e4f5b621629e6f3f.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| -0.0002        | 350     |
| 0.0000         | 400     |
| 0.0002         | 350     |
| 0.0004         | 350     |
| 0.0006         | 350     |
| 0.0008         | 350     |
| 0.0010         | 350     |
</details>

(d) Block11 norm2 in IID with homo.

Figure 13: The gradient distributions from round 10 to 20 of ViTs in IID with homo.   
![](images/cd4343aa0009a1d1ebd3666825cfa2e2e88dddc11f08da7c56afed54ca0757d4.jpg)

<details>
<summary>line</summary>

| Gradient value | Density (Round 10) | Density (Round 11) | Density (Round 12) | Density (Round 13) | Density (Round 14) | Density (Round 15) | Density (Round 16) | Density (Round 17) | Density (Round 18) | Density (Round 19) |
| -------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| -0.0004        | 0                  | 0                  | 0                  | 0                  | 0                  | 0                  | 0                  | 0                  | 0                  | 0                  |
| -0.0002        | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 |
| 0.0000         | 400                | 400                | 400                | 400                | 400                | 400                | 400                | 400                | 400                | 400                |
| 0.0002         | 350                | 350                | 350                | 350                | 350                | 350                | 350                | 350                | 350                | 350                |
| 0.0004         | 300                | 300                | 300                | 300                | 300                | 300                | 300                | 300                | 300                | 300                |
| 0.0006         | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 |
| 0.0010         | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 | 25                 |
</details>

(a) Block7 norm1 in Non-IID with hetero.

![](images/78365f318f4a842c33ad205cc9ae7c52c446e979355b793bef39744183d7fc02.jpg)

<details>
<summary>line</summary>

| Gradient value | Round 10 | Round 11 | Round 12 | Round 13 | Round 14 | Round 15 | Round 16 | Round 17 | Round 18 | Round 19 |
| -------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| -0.0004        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| -0.0002        | 250      | 225      | 200      | 175      | 150      | 125      | 100      | 75       | 50       | 25       |
| 0.0002         | 250      | 225      | 200      | 175      | 150      | 125      | 100      | 75       | 50       | 25       |
| 0.0006         | 250      | 225      | 200      | 175      | 150      | 125      | 100      | 75       | 50       | 25       |
| 0.0010         | 250      | 225      | 200      | 175      | 150      | 125      | 100      | 75       | 50       | 25       |
</details>

(b) Block7 norm2 in Non-IID with hetero.

![](images/53bc209d2696cb56f3c5c8286dbbf4dc4c255c7aa419a7998f34f7b82bc5bb05.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| -0.0002        | 50      |
| 0.0000         | 150     |
| 0.0002         | 250     |
| 0.0004         | 300     |
| 0.0006         | 250     |
| 0.0010         | 150     |
</details>

(c) Block11 norm1 in Non-IID with hetero.

![](images/abee272a27842c729770e94da028cfce35ed7451af4877f7dfeb4c5a8657431a.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| -0.0002        | 350     |
| 0.0000         | 250     |
| 0.0002         | 350     |
| 0.0004         | 250     |
| 0.0006         | 350     |
| 0.0010         | 250     |
</details>

(d) Block11 norm2 in Non-IID with hetero.   
Figure 14: The gradient distributions from round 10 to 20 of ViTs in Non-IID with hetero.

of norm1 and norm2 layers is motivated by the significance of Layer Norm in the architecture of transformers (Xiong et al., 2020). Additionally, we have chosen Block7 and Block11 for analysis as, in the context of heterogeneous models, Block7 is the final layer of the smallest ViTs, while Block11 represents the final layer of the largest ViTs.

From Figure 12a and Figure 12b, we observe that the cross-environment similarities derived from the shallow layer norm (norm1) are higher compared to those from the deep layer norm (norm2). Moreover, similar to the analysis conducted for ResNets, we discover that the distributions of norm1 in ViTs exhibit greater smoothness compared to norm2, as depicted in Figure 13 and Figure 14. These findings reinforce the notion that InCo Aggregation is indeed suitable for ViTs.

![](images/2765e61d1d6013e62a903f13a6f312fa920462650e5ce9d21cbd48244868e2ad.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0010        | 0       |
| -0.0005        | 140     |
| 0.0000         | 120     |
| 0.0005         | 100     |
| 0.0010         | 80      |
| 0.0015         | 60      |
| 0.0020         | 40      |
</details>

(a) Stage2 conv0 in Non-IID with hetero.

![](images/ef4d09efb202c61be4b5316c3299d8f5fa5c486d11f4da2fe651bd7cbb638b9c.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0005        | 0       |
| 0.0000         | 140     |
| 0.0005         | 120     |
| 0.0010         | 80      |
| 0.0015         | 40      |
| 0.0020         | 0       |
</details>

(b) Stage2 conv1 in Non-IID with hetero.

![](images/36f3835135afb919556204178ef254e373ab06d8f93d4a19525b2e379aab7a6a.jpg)

<details>
<summary>line</summary>

| Gradient value | Density |
| -------------- | ------- |
| -0.0004        | 0       |
| 0.0000         | 500     |
| 0.0004         | 450     |
| 0.0008         | 40      |
</details>

(c) Stage3 conv0 in Non-IID with hetero with different seed.

![](images/43212db57354916076e8fc00ea88b50ed1d849e0e7d182a7e602e819abfba59d.jpg)

<details>
<summary>line</summary>

| Gradient value | Density (Round 40) | Density (Round 41) | Density (Round 42) | Density (Round 43) | Density (Round 44) | Density (Round 45) | Density (Round 46) | Density (Round 47) | Density (Round 48) | Density (Round 49) |
| -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| -0.0004        | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   | 0                   |
| -0.0002        | 100                 | 150                 | 200                 | 250                 | 300                 | 350                 | 400                 | 450                 | 500                 | 550                 |
| 0.0000         | 600                 | 550                 | 500                 | 450                 | 400                 | 350                 | 300                 | 250                 | 200                 | 150                 |
| 0.0002         | 500                 | 450                 | 400                 | 350                 | 300                 | 250                 | 200                 | 150                 | 100                 | 50                  |
| 0.0004         | 400                 | 350                 | 300                 | 250                 | 200                 | 150                 | 100                 | 50                  | 25                  | 10                  |
| 0.0006         | 300                 | 250                 | 200                 | 150                 | 100                 | 50                  | 25                  | 10                  | 5                   | 2                   |
| 0.0008         | 200                 | 150                 | 100                 | 50                  | 25                  | 10                  | 5                   | 2                   | 1                   | 1                   |
</details>

(d) Stage3 conv1 in Non-IID with hetero with different seed.   
Figure 15: The gradient distributions from round 40 to 50 of ResNets in Non-IID with hetero. (a) and (b) Stage2 conv0 and conv1. (c) and (d) Stage3 conv0 and conv1 with different seed.

# G.4 GRADIENT DISTRIBUTIONS FROM STAGE 2 AND DIFFERENT SEED.

Figure 15a and Figure 15b demonstrate the gradient distributions in Stage 2. In contrast to the gradient distributions of Stage 3, the differences in gradient distributions across different layers are less evident for Stage 2. This can be observed from Figure 1a, where the CKA similarity for Stage 2 is considerably higher than that of Stage 3. The higher similarity indicates that Stage 2 is relatively less biased and more generalized compared to Stage 3, resulting in less noticeable differences in gradient distributions. This observation further supports the relationship between similarity and smoothness, as higher similarity leads to smoother distributions. Moreover, Figure 15c and Figure 15d illustrate that the gradient distributions still keep the same properties in different random seed, indicating that the relations between similarity and smooth gradients are not affected by SGD noise.

# G.5 OTHER RELATIONS BETWEEN CKA AND THE STATISTICS OF GRADIENTS.

Figure 16 provides an overview of additional relationships between layer similarity and the cross-environment gradient statistics derived from IID with homo and Non-IID with hetero. We calculate the difference between gradients from the same layer across these two environments. To clarify the tendency of similarity for each stage, we normalize the results according to the smallest value within each stage. As shown in Figure 16, none of these gradient statistics exhibit stronger correlations with the similarity of gradients compared to the smoothness, discussed in Section 2.2 and Appendix G.3.

![](images/26b465426a6b5103f6fb6d74e9af317f872fb3cab58887599bfa32c2316850f5.jpg)

<details>
<summary>line</summary>

| Layer  | Norm CKA | Norm Mean |
| ------ | -------- | --------- |
| S2.c0  | 2.0      | 1.0       |
| S2.c1  | 1.0      | 0.5       |
| S3.c0  | 1.7      | -1.0      |
| S3.c1  | 1.0      | -1.0      |
| S3.c2  | 1.0      | -1.0      |
</details>

(a) CKA with mean.

![](images/e4961264c1fe077051e56b3f190ca0877ed6dd53f81b9a22c060bc8dd4809739.jpg)

<details>
<summary>line</summary>

| Layer   | Norm CKA | Norm Variance |
| ------- | -------- | ------------- |
| S2.c0   | 2.0      | 1.30          |
| S2.c1   | 1.8      | 1.25          |
| S2.c2   | 1.0      | 1.00          |
| S3.c0   | 1.7      | 1.25          |
| S3.c1   | 1.0      | 1.00          |
| S3.c2   | 1.1      | 1.05          |
</details>

(b) CKA with variance.

![](images/301493c7eda908c546fc919e0beebad6f0c56f967722b359384733655f57cede.jpg)

<details>
<summary>line</summary>

| Layer  | Norm CKA | Norm Covariance |
|--------|----------|-----------------|
| S2.c0  | 2.0      | 1.0             |
| S2.c1  | 1.6      | 3.5             |
| S2.c2  | 1.0      | 1.0             |
| S3.c0  | 1.7      | 1.0             |
| S3.c1  | 1.0      | 4.0             |
| S3.c2  | 1.0      | 1.0             |
</details>

(c) CKA with covariance.

![](images/eb0df718dca32f7a7ada6f68302a21f4e417294880e55f9d9c3e33031419c730.jpg)

<details>
<summary>line</summary>

| Layer  | CKA  | Difference |
| ------ | ---- | ---------- |
| S2.c0  | 2.0  | 4.0        |
| S2.c1  | 1.2  | 1.4        |
| S2.c2  | 1.0  | 4.0        |
| S3.c0  | 1.8  | 1.0        |
| S3.c1  | 1.0  | 1.4        |
| S3.c2  | 1.0  | 1.0        |
</details>

(d) CKA with difference.   
Figure 16: Other relationships between CKA and the cross-environment statistics of gradients from IID with homo and Non-IID with hetero. We use abbreviations for "stage" and "conv," represented as "s" and "c" respectively. For example, "s2c0" represents stage 2, conv0.

![](images/375f4c0a4d89f8e2469363d44f66fff1c6cccf2d9872a7076f9473eda8951199.jpg)

<details>
<summary>heatmap</summary>

| Client ID | Client1 | Client2 | Client3 | Client4 | Client5 | Client6 | Client7 | Client8 | Client9 |
|---|---|---|---|---|---|---|---|---|---|
| Client0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Client7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.34 | 0.34 | 0.34 |
| Client8 | 0.0 | 0.0 | 0.0 | 0.34 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |
| Client9 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |
The chart displays a single column of numerical values for each client's data point, with values ranging from approximately -1.3 to +1.3, and the color scale ranges from -1.3 (dark purple) to +1.3 (light orange). The title is 'Client ID' but it is not explicitly labeled in the image.
</details>

(a) The CKA similarity of IID with homo for stage 0.

![](images/7bed2367765c77d5f9d37fcbca3499ccc71e621234f337e55e0f39d4361426ce.jpg)  
(b) The CKA similarity of IID with homo for stage 1.

![](images/4452353bec8e052194df353baa9bfd172faf22d8902e275305b2f1ce9ee3b74c.jpg)  
(c) The CKA similarity of IID with homo for stage 2.

![](images/d9ee7512664cc7598db0bc8605031b3b3b69b8aa90d8f71dc679cba849dad75c.jpg)  
(d) The CKA similarity of IID with homo for stage 3.

![](images/2fbd309560ce2c17c59547888706d308df621d47b730cbbbb6941b5cd5f14bd1.jpg)

<details>
<summary>heatmap</summary>

| Client ID | Cluster1 | Cluster2 | Cluster3 | Cluster4 | Cluster5 | Cluster6 | Cluster7 | Cluster8 | Cluster9 |
|---|---|---|---|---|---|---|---|---|---|
| Cler0 | 0.9 | 0.85 | 0.8 | 0.75 | 0.7 | 0.65 | 0.6 | 0.55 | 0.5 |
| Cler1 | 0.95 | 0.88 | 0.82 | 0.78 | 0.72 | 0.68 | 0.62 | 0.58 | 0.52 |
| Cler2 | 0.98 | 0.9 | 0.84 | 0.81 | 0.74 | 0.71 | 0.66 | 0.62 | 0.54 |
| Cler3 | 0.99 | 0.92 | 0.86 | 0.83 | 0.76 | 0.73 | 0.69 | 0.64 | 0.56 |
| Cler4 | 1.0 | 0.94 | 0.88 | 0.85 | 0.78 | 0.75 | 0.71 | 0.66 | 0.58 |
| Cler5 | 1.01 | 0.96 | 0.9 | 0.87 | 0.8 | 0.77 | 0.73 | 0.68 | 0.6 |
| Cler6 | 1.02 | 0.97 | 0.92 | 0.89 | 0.82 | 0.79 | 0.75 | 0.71 | 0.62 |
| Cler7 | 1.03 | 0.98 | 0.94 | 0.91 | 0.84 | 0.81 | 0.77 | 0.73 | 0.64 |
| Cler8 | 1.04 | 0.99 | 0.96 | 0.93 | 0.86 | 0.83 | 0.79 | 0.75 | 0.66 |
| Cler9 | 1.05 | 1.0 | 0.98 | 0.95 | 0.88 | 0.85 | 0.81 | 0.77 | 0.68 |
The chart displays a heatmap with color intensity corresponding to the values on the vertical axis (ranging from ~0.2 to ~1.1). The x-axis labels are 'Cler1' through 'Cler9'. The y-axis is labeled 'Client ID'. The color scale ranges from ~0.2 (lightest) to ~1.1 (darkest), indicating the magnitude of the measured variable at each client's position relative to the client's position.
</details>

(e) The CKA similarity of Non-IID with homo for stage 0.

![](images/731f0721686466fabb2483bf93de27b168bc4674f06184ad0597f9ce6acce26f.jpg)  
(f) The CKA similarity of Non-IID with homo for stage 1.

![](images/4dbcfb77337325d54f4ab82eca28e6a8bb279959ba224455a316dec8e1b4b8c9.jpg)  
(g) The CKA similarity of Non-IID with homo for stage 2.

![](images/c778d358237105a77696d2752da5dad1107257d14dcea47754a7171ca6549415.jpg)  
(h) The CKA similarity of Non-IID with homo for stage 3.

![](images/5db204208b173a9ad121fbe1e35eea4e76f3b4f1e1f01491394392624d5ebd04.jpg)  
(i) The CKA similarity of Non-IID with hetero for stage 0.

![](images/79dd95fbcc135f0cdc52bd8a43b1d996650c6539efd9c9841b8d91f183ce3d5a.jpg)  
(j) The CKA similarity of Non-IID with hetero for stage 1.

![](images/713ef57a360c650296ef8c633a884c2a1ced0c1243db57f138b54aaf10e9c229.jpg)  
(k) The CKA similarity of Non-IID with hetero for stage 2.

![](images/1f18f23c39f8d00c8e58e972167f9f1f1a66c87593fdce1f5223b9ea2cd9c17e.jpg)  
(1) The CKA similarity of Non-IID with hetero for stage 3.   
Figure 17: The CKA similarity of IID with homo, Non-IID with homo and Non-IID with hetero for ResNets.

# G.6 HEATMAPS FOR THE CASE STUDY

In this part, we will show the heatmaps for all stages of ResNets and layer 4 to layer 7 of ViTs in Figure 17 and Figure 18. These heatmaps are the concrete images for Figure 1. We can see that the CKA similarity is lower with the deeper stages or layers no matter in ResNets and ViTs. However, it is notable that the different patterns for CKA similarity between ResNets and ViTs from the comparison between Figure 17 and Figure 18. To get a clear analysis, we focus on the last stage of ResNets and layer 7 of ViTs, which are the most biased part of the entire model. Like Figure 17l in ResNets, almost all clients are dissimilar, while only a part of clients has low similarity in ViTs from Figure 18l. Along with the experiment results from Table 2, the improvements in ViTs from FedInCo are modest. One possible reason is that we neglect more biased clients and regard all clients as having the same level of bias in ViTs, which is a possible improvement for FedInCo.

![](images/86792274c7353bedd22d862cc43ba473761fde68cdf2fbae5c2359071e476d9b.jpg)

<details>
<summary>heatmap</summary>

|        | Date0  | Date1  | Date2  | Date3  | Date4  | Date5  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Class0 | -0.9   | -0.8   | -0.7   | -0.6   | -0.5   | -0.4   |
| Class1 | -0.8   | -0.7   | -0.6   | -0.5   | -0.4   | -0.3   |
| Class2 | -0.7   | -0.6   | -0.5   | -0.4   | -0.3   | -0.2   |
| Class3 | -0.6   | -0.5   | -0.4   | -0.3   | -0.2   | -0.1   |
| Class4 | -0.5   | -0.4   | -0.3   | -0.2   | -0.1   | 0.0    |
| Class5 | -0.4   | -0.3   | -0.2   | -0.1   | 0.0    | 0.1    |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    |...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...  | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...  | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...  | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...  | ...    |
| ...    | ...    | ...    & ...| ...    | ...    | ...  | ...   |
| ...    | ...    | ...    &...|...    |...    |...  |...    |
| ...    | ...    | ...    &...|...    |...    |...  |...    |
| ...    | ...    | ...     &...|...   |...   |...  |...    |
| Note: The actual values in the 'Class' column are not provided in the code, so they are represented as placeholders (e.g., '1.8') in the heatmap.
</details>

(a) The CKA similarity of IID with homo for layer 4.

![](images/3c65eec4edb080b42e6c89d5c16ca073cd511a8a91e4f7cc50ba65a6396d5321.jpg)  
(b) The CKA similarity of IID with homo for layer 5.

![](images/b773e3c819085f6a3b31b4195a99baa0255b53d0bf1d49f2b209517789b7537c.jpg)

<details>
<summary>heatmap</summary>

| Client ID | Client1 | Client2 | Client3 | Client4 | Client5 | Client6 | Client7 | Client8 | Client9 |
| --------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Client0   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client1   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client2   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client3   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client4   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client5   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client6   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client7   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client8   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
| Client9   | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    | 1.80    |
</details>

(c) The CKA similarity of IID with homo for layer 6.

![](images/fd42bbe434661bc365e776887dc65db1ef52b886e7e03fc674d7f4d164569c98.jpg)  
(d) The CKA similarity of IID with homo for layer 7.

![](images/a772fbd1f63a764761e441d3a389137c05420c641613409d201a2182a9e082fc.jpg)  
(e) The CKA similarity of Non-IID with homo for layer 4.

![](images/29ba31a8d279a6b20fab951e40afd9b7fb91303ed47637789b850101e59eb651.jpg)  
(f) The CKA similarity of Non-IID with homo for layer 5.

![](images/0afdb28df7e32ee62ed9b05a7c80a36b1ea2bc66c4abd4d2313b32b6fb247fd2.jpg)

<details>
<summary>heatmap</summary>

| Client | Client1 | Client2 | Client3 | Client4 | Client5 | Client6 | Client7 | Client8 | Client9 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Client1 | 1.00    | 0.95    | 0.90    | 0.85    | 0.80    | 0.75    | 0.70    | 0.65    | 0.60    |
| Client2 | 0.95    | 0.90    | 0.85    | 0.80    | 0.75    | 0.70    | 0.65    | 0.60    | 0.55    |
| Client3 | 0.90    | 0.85    | 0.80    | 0.75    | 0.70    | 0.65    | 0.60    | 0.55    | 0.50    |
| Client4 | 0.85    | 0.80    | 0.75    | 0.70    | 0.65    | 0.60    | 0.55    | 0.50    | 0.45    |
| Client5 | 0.80    | 0.75    | 0.70    | 0.65    | 0.60    | 0.55    | 0.50    | 0.45    | 0.40    |
| Client6 | 0.75    | 0.70    | 0.65    | 0.60    | 0.55    | 0.50    | 0.45    | 0.40    | 0.35    |
| Client7 | 0.70    | 0.65    | 0.60    | 0.55    | 0.50    | 0.45    | 0.40    | 0.35    | 0.30    |
| Client8 | 0.65    | 0.60    | 0.55    | 0.50    | 0.45    | 0.40    | 0.35    | 0.30    | 0.25    |
| Client9 | 0.60    | 0.55    | 0.50    | 0.45    | 0.40    | 0.35    | 0.30    | 0.25    | 0.20    |
</details>

(g) The CKA similarity of Non-IID with homo for layer 6.

![](images/5c36aaa4d3c809a5053eead1c9450b4d2f0544920d5540a38054d083dc3bd8da.jpg)

<details>
<summary>heatmap</summary>

| | Client | Client1 | Client2 | Client3 | Client4 | Client5 | Client6 | Client7 | Client8 | Client9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Client0 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client1 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client2 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client3 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client4 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client5 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client6 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client7 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client8 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| Client9 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
The values in the table represent the correlation coefficients between each client and their respective clients, respectively for each client's pair.
</details>

(h) The CKA similarity of Non-IID with homo for layer 7.

![](images/24efe59f607b64cc8015a5438e1d3b8e6fc83ce082ed40559b69b6e076e5ee96.jpg)  
(i) The CKA similarity of Non-IID with hetero for layer 4.

![](images/a4470d5c6b58330cdcb3ae1b80c706bb7dbcc9ba8ddc685a9578413bf63da61d.jpg)  
(j) The CKA similarity of Non-IID with hetero for layer 5.

![](images/083be4c155494def00ca222117b497aecc1e913a6987cc6b8bee07d9d5bacbdc.jpg)  
(k) The CKA similarity of Non-IID with hetero for layer 6.

![](images/c8786ba1993de0fc033fe457e179a82aee06dcdcad78b030f8603ecf1b077a6e.jpg)  
(1) The CKA similarity of Non-IID with hetero for layer 7.   
Figure 18: The CKA similarity of IID with homo, Non-IID with homo and Non-IID with hetero for ViTs.

# H MORE DETAILS OF THE EXPERIMENTS

# H.1 PROCEDURE FOR INCO AGGREGATION

The pseudo-codes for InCo Aggregation are shown in Algorithm 1. InCo Aggregation is operated in a server model, indicating that the methods focused on the client can be aligned with InCo Aggregation, as shown in our experiments.

Algorithm 1 InCo Aggregation (InCoAvg as the example) 

<table><tr><td colspan="2">Require: Dataset  $D_{k}, k \in \{1, ..., K\}$ ,  $K$  clients, and their weights  $w_{1}, ..., w_{K}$ .</td><td>10: else $g_{l_{k}}^{t+1} = g_{l_{k}}^{t}$ </td></tr><tr><td colspan="2">Ensure: Weights for all clients  $w_{1}, ..., w_{K}$ .</td><td>12: end if $w_{l_{k}}^{t+1} = w_{l_{k}}^{t} + g_{l_{k}}^{t+1}$ </td></tr><tr><td colspan="2">1: Server process:</td><td>13: end for</td></tr><tr><td colspan="2">2: while not converge do</td><td>14: Sends the updated  $w_{i}^{t+1}$  to sampled clients.</td></tr><tr><td colspan="2">3: Receives  $g_{w_{i}}^{t}$  from the sampled client.</td><td>15: end while</td></tr><tr><td colspan="2">4: Parameter aggregation for  $g_{w_{i}}^{t}$ .</td><td>16: Client processes:while random clients  $i, i \in 1, ..., K$  do</td></tr><tr><td colspan="2">5: for each layer  $l_{k}$  in the server model do</td><td>17:</td></tr><tr><td colspan="2">6: if  $l_{k}$  needs cross-layer gradients then</td><td>18:</td></tr><tr><td colspan="2">7:  $g_{l_{k}}^{t'}$ ,  $g_{l_{0}}^{t'} \leftarrow$  Normalizes  $g_{l_{k}}^{t}$  and  $g_{l_{0}}^{t}$ .</td><td>19: Receives model weights  $w_{i}^{t-1}$ .</td></tr><tr><td colspan="2">8:  $\theta^{t}, \alpha, \beta$  from Theorem 3.1.</td><td>20: Updates client models  $w_{i}^{t-1}$  to  $w_{i}^{t}$ .</td></tr><tr><td colspan="2">9:  $g_{l_{k}}^{t+1} = \frac{(g_{l_{k}}^{t'} - \theta^{t} g_{l_{0}}^{t'}) \times (||g_{l_{k}}^{t} || + ||g_{l_{0}}^{t} ||)}{2}$ .</td><td>21: Sends  $g_{w_{i}}^{t} = w_{i}^{t} - w_{i}^{t-1}$  to the server.</td></tr><tr><td colspan="2"></td><td>22: end while</td></tr></table>

Table 5: Test accuracy of 100 clients and sample ratio 0.1. We shade in gray the methods that are combined with our proposed method, InCo Aggregation. We show the error bars for InCo Aggregation in this table. 

<table><tr><td rowspan="2">Base</td><td rowspan="2">Methods</td><td colspan="2">Fashion-MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td><td colspan="2">CINIC10</td></tr><tr><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td></tr><tr><td rowspan="10">ResNet</td><td>FedAvg</td><td>86.7±1.0</td><td>87.7±0.6</td><td>74.8±3.2</td><td>81.6±2.5</td><td>52.3±3.4</td><td>61.3±3.2</td><td>43.1±2.7</td><td>49.2±3.1</td></tr><tr><td>FedProx</td><td>75.1±1.8</td><td>76.6±1.5</td><td>32.0±2.8</td><td>43.7±2.9</td><td>19.2±2.2</td><td>23.4±2.4</td><td>17.4±1.7</td><td>19.8±1.4</td></tr><tr><td>Scaffold</td><td>87.9±0.5</td><td>88.0±0.3</td><td>76.3±3.4</td><td>82.4±3.1</td><td>54.3±3.6</td><td>61.8±3.0</td><td>43.5±2.4</td><td>49.4±3.1</td></tr><tr><td>FedNova</td><td>12.7±0.2</td><td>15.6±0.2</td><td>13.4±0.4</td><td>15.3±0.3</td><td>10.4±0.3</td><td>14.3±0.2</td><td>12.0±0.3</td><td>14.0±0.2</td></tr><tr><td>MOON</td><td>87.7±0.4</td><td>87.5±0.3</td><td>72.8±4.3</td><td>81.2±3.2</td><td>47.2±2.7</td><td>58.8±2.6</td><td>40.8±2.1</td><td>49.2±1.9</td></tr><tr><td>InCoAvg</td><td>90.2±1.2</td><td>88.4±1.8</td><td>87.6±2.8</td><td>89.0±2.6</td><td>67.8±3.2</td><td>70.7±3.4</td><td>53.0±3.2</td><td>57.5±3.3</td></tr><tr><td>InCoProx</td><td>88.8±2.3</td><td>86.4±3.2</td><td>89.0±1.3</td><td>90.8±1.2</td><td>74.5±2.3</td><td>76.8±1.8</td><td>59.1±3.2</td><td>62.5±2.4</td></tr><tr><td>InCoScaffold</td><td>88.3±1.4</td><td>90.1±1.2</td><td>85.4±2.4</td><td>87.8±3.5</td><td>67.3±3.6</td><td>73.8±2.9</td><td>53.5±3.3</td><td>61.7±3.0</td></tr><tr><td>InCoNova</td><td>86.6±1.4</td><td>87.4±1.3</td><td>86.4±2.5</td><td>88.4±1.8</td><td>62.8±3.9</td><td>69.7±4.2</td><td>48.0±2.7</td><td>54.1±1.7</td></tr><tr><td>InCoMOON</td><td>89.1±1.3</td><td>89.5±1.2</td><td>85.6±3.8</td><td>89.3±2.0</td><td>68.2±3.1</td><td>71.8±2.3</td><td>54.3±3.0</td><td>57.6±2.7</td></tr><tr><td rowspan="10">ViT</td><td>FedAvg</td><td>92.0±0.7</td><td>91.9±0.5</td><td>92.4±0.9</td><td>93.9±0.8</td><td>93.7±1.0</td><td>94.2±0.8</td><td>83.8±1.4</td><td>85.1±0.9</td></tr><tr><td>FedProx</td><td>89.8±0.5</td><td>89.7±0.5</td><td>71.4±3.8</td><td>81.1±2.9</td><td>82.6±3.3</td><td>84.7±2.3</td><td>67.8±2.8</td><td>71.3±3.0</td></tr><tr><td>Scaffold</td><td>92.0±0.4</td><td>92.0±0.5</td><td>92.2±0.8</td><td>93.8±0.6</td><td>93.5±0.7</td><td>94.5±0.5</td><td>83.3±1.6</td><td>85.5±1.2</td></tr><tr><td>FedNova</td><td>70.3±0.5</td><td>76.7±0.4</td><td>27.4±0.4</td><td>49.8±0.5</td><td>30.7±0.3</td><td>54.4±0.5</td><td>31.6±1.5</td><td>50.7±1.3</td></tr><tr><td>MOON</td><td>92.1±0.3</td><td>92.1±0.2</td><td>92.5±1.2</td><td>93.9±0.9</td><td>93.6±0.8</td><td>94.6±0.3</td><td>84.3±1.6</td><td>85.3±1.2</td></tr><tr><td>InCoAvg</td><td>93.0±0.6</td><td>93.1±0.5</td><td>94.2±0.6</td><td>95.0±0.4</td><td>94.6±0.7</td><td>95.0±0.6</td><td>85.9±1.9</td><td>86.8±1.3</td></tr><tr><td>InCoProx</td><td>92.6±0.3</td><td>92.5±0.3</td><td>93.9±0.7</td><td>94.4±0.6</td><td>94.0±1.0</td><td>94.8±0.7</td><td>85.1±1.4</td><td>86.0±0.8</td></tr><tr><td>InCoScaffold</td><td>92.9±0.3</td><td>93.0±0.2</td><td>94.0±1.1</td><td>94.8±0.6</td><td>94.6±0.5</td><td>95.0±0.2</td><td>85.7±1.3</td><td>86.5±1.1</td></tr><tr><td>InCoNova</td><td>93.1±0.3</td><td>93.6±0.3</td><td>94.7±0.9</td><td>95.6±0.5</td><td>94.8±0.4</td><td>95.7±0.3</td><td>86.2±1.8</td><td>88.2±1.0</td></tr><tr><td>InCoMOON</td><td>92.8±0.5</td><td>93.0±0.3</td><td>94.7±0.8</td><td>95.1±0.5</td><td>94.2±0.8</td><td>95.1±0.5</td><td>86.0±0.9</td><td>86.8±1.3</td></tr></table>

# H.2 DATASETS

We conduct experiments on Fashion-MNIST, SVHN, CIFAR-10, and CINIC-10. CINIC-10 is a dataset of the mix of CIFAR-10 and ImageNet within ten classes. We use $3 \times 224 \times 224$ with the ViT models and $3 \times 32 \times 32$ with the ResNet models for all datasets.

# H.3 HYPER-PARAMETERS

We deploy stage splitting for ResNets and obtain five sub-models, which can be recognized as ResNet10, ResNet14, ResNet18, ResNet22, and ResNet26. For the pre-trained ViT models, we employ layer splitting and obtain five sub-models, which are ViT-S/8, ViT-S/9, ViT-S/10, ViT-S/11, and ViT-S/12 from the PyTorch Image Models (timm) $^{7}$ . Our implementations of FedAvg, FedProx, FedNova, Scaffold and MOON are referred to (Li et al., 2022). We use Adam optimizer with a learning rate of 0.001, $\beta_{1} = 0.9$ and $\beta_{2} = 0.999$ , default parameter settings for all methods of ResNets. The local training epochs are fixed to 5. The batch size is 64 for all experiments. Furthermore, the global communication rounds are 500 for ResNets, and 200 for ViTs for all datasets. Global communication rounds for MOON and InCoMOON are 100 to prevent the extreme overfitting in Fashion-MNIST. The hyper-parameter $\frac{\mu}{2}$ for FedProx and InCoProx is 0.05 for ViTs and ResNets. We conduct our experiments with 4 NVIDIA GeForce RTX 3090s. All baselines and their InCo extensions are conducted in the same hyper-parameters. The settings of hetero splitting for ScaleFL followed the source codes. $^{8}$

# H.4 MODEL SIZES

We demonstrate the model sizes for each client model in Table 7, Table 8, and Table 9.

# H.5 ERROR BARS OF INCO AGGREGATION

We illustrate the error bars of InCo Aggregation and the results from model-homogeneous baselines (not use stage or layer splitting in the model heterogeneous environment) in Table 5. For the model-heterogeneity methods, we demonstrate the error bars of InCo Aggregation in Table 6. These results show the stability of InCo Aggregation. In all cases of ResNets and many cases of ViTs, the worst

Table 6: Test accuracy of model-heterogeneity methods with 100 clients and sample ratio 0.1. We shade in gray the methods that are combined with our proposed method, InCo Aggregation. We show the error bars for InCo Aggregation in this table. 

<table><tr><td rowspan="2">Base</td><td rowspan="2">Splitting</td><td rowspan="2">Methods</td><td colspan="2">Fashion-MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td></tr><tr><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td><td> $\alpha = 0.5$ </td><td> $\alpha = 1.0$ </td></tr><tr><td rowspan="10">ResNet</td><td rowspan="2">Hetero</td><td>HeteroFL</td><td>88.9±1.0</td><td>89.7±0.7</td><td>90.5±1.6</td><td>92.2±1.3</td><td>65.2±3.2</td><td>68.4±3.6</td></tr><tr><td>+InCo</td><td>90.0±1.2</td><td>90.4±1.1</td><td>92.1±1.0</td><td>93.5±1.5</td><td>68.2±3.8</td><td>71.2±3.4</td></tr><tr><td rowspan="2">Stage</td><td>InclusiveFL</td><td>89.1±1.1</td><td>89.8±1.0</td><td>88.6±2.0</td><td>90.0±2.2</td><td>65.7±3.5</td><td>68.4±3.3</td></tr><tr><td>+InCo</td><td>90.1±1.5</td><td>90.5±1.3</td><td>90.6±1.7</td><td>90.9±1.9</td><td>69.1±2.8</td><td>72.3±3.1</td></tr><tr><td rowspan="2">Hetero</td><td>FedRolex</td><td>88.2±1.0</td><td>90.2±0.8</td><td>90.9±1.3</td><td>91.6±1.7</td><td>64.7±4.1</td><td>72.3±3.0</td></tr><tr><td>+InCo</td><td>90.4±1.4</td><td>91.3±1.1</td><td>92.8±1.5</td><td>93.4±1.6</td><td>67.9±2.9</td><td>75.6±2.6</td></tr><tr><td rowspan="2">Hetero</td><td>ScaleFL</td><td>90.9±0.5</td><td>91.0±0.4</td><td>92.6±1.0</td><td>92.9±0.9</td><td>71.1±2.9</td><td>74.7±3.1</td></tr><tr><td>+InCo</td><td>91.5±1.0</td><td>91.7±1.1</td><td>93.4±0.9</td><td>93.6±0.9</td><td>73.8±3.2</td><td>76.1±2.6</td></tr><tr><td>N/A</td><td>AllSmall</td><td>83.5±1.7</td><td>84.0±1.7</td><td>72.1±3.5</td><td>81.0±2.9</td><td>39.2±2.0</td><td>44.9±2.3</td></tr><tr><td>N/A</td><td>AllLarge</td><td>91.8±0.5</td><td>92.5±0.8</td><td>93.4±0.8</td><td>93.8±0.5</td><td>79.6±2.9</td><td>82.5±1.0</td></tr></table>

Table 7: Model parameters for different architectures of ResNets (Stage splitting). 

<table><tr><td rowspan="2">Sizes</td><td colspan="5">ResNets (Stage splitting)</td></tr><tr><td>ResNet10</td><td>ResNet14</td><td>ResNet18</td><td>ResNet22</td><td>ResNet26</td></tr><tr><td>Params</td><td>4.91M ( $\times$ 0.281)</td><td>10.81M ( $\times$ 0.619)</td><td>11.18M ( $\times$ 0.641)</td><td>17.08M ( $\times$ 0.979)</td><td>17.45M ( $\times$ 1)</td></tr></table>

![](images/f5442be22f3bba58cc6027d38e8ce3e3dc554bab672c64443f5f89334cc732c3.jpg)

<details>
<summary>line</summary>

| Layer | InCoAvg | HeteroAvg | FedAvg |
|-------|---------|-----------|--------|
| 4     | 0.85    | 0.85      | 0.85   |
| 5     | 0.75    | 0.75      | 0.75   |
| 6     | 0.52    | 0.48      | 0.48   |
| 7     | 0.50    | 0.45      | 0.45   |
</details>

(a) CIFAR10 with $\alpha = 0.5$ .

![](images/e2a1744345af20a7e32c4995b8beb3e5c7c9f8b5d2458f74834f1eb612c05525.jpg)  
(b) InCoAvg.

![](images/247f7b45c1cecb12764700a897e7d902f1523838bcb5992eaf6100f6ae612ce4.jpg)  
(c) HeteroAvg.

![](images/a96637c807fade1ccf258e82b0436e58b18db5d17028494d643777d60b786137.jpg)  
(d) FedAvg.   
Figure 19: CKA layer similarity and Heatmaps of ViTs. (a): The layer similarity of different methods. (b) to (d): Heatmaps for different methods in layer 6 and layer 7.

results of InCo Aggregation are better than the Averaging Aggregation, demonstrating the efficacy of InCo Aggregation.

# H.6 DIFFERENCES BETWEEN ADDING NOISES AND INCO GRADIENTS.

The convergence speed of InCo gradients surpasses that of the other two methods, as illustrated in Figure 20a. Table 20c demonstrates that InCo gradients outperform other methods across different datasets. The primary distinction between InCo gradients and adding noises lies in their ability to determine the precise gradient modification for each node in the model. Gaussian noise lacks the capability to specify the exact modification required for each node, leading to less controlled and targeted adjustments. This is evident in Figure 20b, where the Frobenius norm of noises is larger and exhibits greater instability compared to InCo gradients.

# H.7 LAYER SIMILARITY AND HEATMAPS OF VITs

Figure 19 illustrates the layer similarity of the last four layers, along with the corresponding heatmaps for Layer 6 and Layer 7. Furthermore, Figure 19a demonstrates that InCo Aggregation significantly enhances the layer similarity, validating the initial motivation behind our proposed method. Additionally, since the disparity in layer similarity between Layer 6 and Layer 7 is minimal, the heatmaps for these layers do not exhibit significant differences, as depicted in Figure 19b through Figure 19d.

Table 8: Model parameters for different architectures of ViTs (Layer splitting). 

<table><tr><td rowspan="2">Sizes</td><td colspan="5">ViTs (Layer splitting)</td></tr><tr><td>ViT-S/8</td><td>ViT-S/9</td><td>ViT-S/10</td><td>ViT-S/11</td><td>ViT-S/12</td></tr><tr><td>Params</td><td>14.57M ( $\times$ 0.672)</td><td>16.34M ( $\times$ 0.754)</td><td>18.12M ( $\times$ 0.836)</td><td>19.90M ( $\times$ 0.912)</td><td>21.67M ( $\times$ 1)</td></tr></table>

Table 9: Model parameters for different architectures of ResNets (Heterogeneous splitting). 

<table><tr><td rowspan="2">Sizes</td><td colspan="5">ResNets (Hetero splitting)</td></tr><tr><td> $\frac{1}{16}$ </td><td> $\frac{1}{8}$ </td><td> $\frac{1}{4}$ </td><td> $\frac{1}{2}$ </td><td>ResNet26</td></tr><tr><td>Params</td><td>0.07M ( $\times$ 0.004)</td><td>0.28M ( $\times$ 0.016)</td><td>1.10M ( $\times$ 0.06)</td><td>4.37 ( $\times$ 0.25)</td><td>17.45M ( $\times$ 1)</td></tr></table>

# H.8 MORE ABLATION STUDIES AND ROBUSTNESS ANALYSIS.

We conduct additional experiments on different baselines to demonstrate the effectiveness of InCo Aggregation. Figure 21 to Figure 23 present the results of the ablation study for FedProx, FedNova, and Scaffold, incorporating InCo Aggregation. These results highlight the efficacy of InCo Aggregation across different baselines. Additionally, Figure 24 and Figure 25 illustrate the robustness analysis for FedProx and Scaffold. In Figure 24 and Figure 25, InCoProx and InCoScaffold consistently obtains the best performances across all settings. These experiments provide further evidence of the efficiency of InCo Aggregation.

# I LIMITATIONS AND FUTURE DIRECTIONS

The objective of this study is to expand the capabilities of model-homogeneous methods to effectively handle model-heterogeneous FL environments. However, the analysis of layer similarity reveals that the smallest models do not derive substantial benefits from InCo Aggregation, implying the limited extensions for these smallest models. Exploring methods to enhance the performance of the smallest models warrants further investigation. Furthermore, our research mainly focuses on image classification tasks, specifically CNN models (ResNets) and Transformers (ViTs). However, it is imperative to validate our conclusions in the context of language tasks, and other model architectures such as LSTM Hochreiter & Schmidhuber (1997). Additionally, it is important to consider that the participating clients in the training process may have different model architectures. For example, some clients may employ CNN models, while others may use Vision Transformers (ViTs). We believe that it is worth extending this work to encompass a wider range of tasks and diverse model architectures that hold great value and potential for future research.

![](images/371fcb6e368fe47159d5eebf1867208218a36aa67ef09945616a13adf98b788a.jpg)

<details>
<summary>line</summary>

| Rounds | HeteroAvg | Add noise | InCo gradients |
| ------ | --------- | --------- | -------------- |
| 0      | 50.0      | 50.0      | 50.0           |
| 100    | 58.0      | 56.0      | 57.0           |
| 200    | 62.0      | 60.0      | 63.0           |
| 300    | 65.0      | 63.0      | 66.0           |
| 400    | 67.0      | 65.0      | 68.0           |
| 500    | 68.0      | 66.0      | 69.0           |
</details>

(a) Convergence speeds.

![](images/48d4fcdd765bc6ca0eb9bdc4bcb770c4b672586b4a0f7d9f332a88b68ad18f0d.jpg)

<details>
<summary>line</summary>

| Rounds | Add noise | InCo gradients |
| ------ | --------- | -------------- |
| 0      | 3.0       | 1.5            |
| 50     | 3.5       | 1.6            |
| 100    | 3.2       | 1.4            |
| 150    | 3.8       | 1.3            |
| 200    | 3.6       | 1.2            |
| 250    | 3.9       | 1.1            |
| 300    | 3.7       | 1.0            |
| 350    | 4.0       | 0.9            |
| 400    | 3.8       | 0.8            |
| 450    | 4.2       | 0.7            |
| 500    | 3.9       | 0.6            |
</details>

(b) Frobenius norm.

<table><tr><td rowspan="2">Methods</td><td colspan="2">FashionMNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td></tr><tr><td>0.5</td><td>1.0</td><td>0.5</td><td>1.0</td><td>0.5</td><td>1.0</td></tr><tr><td>HeteroAvg</td><td>87.8</td><td>86.0</td><td>85.1</td><td>86.9</td><td>64.8</td><td>66.7</td></tr><tr><td>Add noises</td><td>87.0</td><td>86.7</td><td>83.3</td><td>86.2</td><td>62.5</td><td>64.9</td></tr><tr><td>InCo gradients</td><td>90.2</td><td>88.4</td><td>87.6</td><td>89.0</td><td>67.8</td><td>70.7</td></tr></table>

(c) Accuracy for different datasets.   
Figure 20: Convergence speeds, Frobenius norm of adding noise and InCo gradients (InCoAvg), and accuracy results for different datasets. (a): Convergence speeds of HeteroAvg, adding noise and InCo gradients. (b): Frobenius norm of noise and InCo gradients. (c): Accuracy for different datasets.

![](images/578626e2680dc1455725eda72fe7c0753816f9a8831f9d6de86a83784f4ba99b.jpg)

<details>
<summary>line</summary>

| Method     | α=1.0 | α=0.5 |
| ---------- | ----- | ----- |
| InCoProx   | 86.5  | 89.0  |
| w/o N      | 82.0  | 85.0  |
| w/o O      | 86.0  | 85.5  |
| w/o NoO    | 81.0  | 85.0  |
| HeteroProx | 84.0  | 87.0  |
</details>

(a) Fashion-MNIST.

![](images/61edc9584bcc58e10523f24a2dde3e1b9072f7f1a72b1a6b4b4caca2a28fdb47.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoProx   | α=1.0        | 91       |
| InCoProx   | α=0.5        | 89       |
| w/o N      | α=1.0        | 90       |
| w/o N      | α=0.5        | 88       |
| w/o O      | α=1.0        | 89       |
| w/o O      | α=0.5        | 88       |
| w/o N&O    | α=1.0        | 89       |
| w/o N&O    | α=0.5        | 88       |
| HeteroProx | α=1.0        | 90       |
| HeteroProx | α=0.5        | 88       |
</details>

(b) SVHN.

![](images/fc4a6b8bc795e6fae5ee7dd9239109c739064da2a2f1d299d53b4ca7571d0bb8.jpg)

<details>
<summary>line</summary>

| Method       | Distribution | Accuracy |
| ------------ | ------------ | -------- |
| InCoProx     | α=1.0        | 77.0     |
| InCoProx     | α=0.5        | 74.5     |
| w/o N        | α=1.0        | 76.0     |
| w/o N        | α=0.5        | 73.5     |
| w/o O        | α=1.0        | 75.5     |
| w/o O        | α=0.5        | 73.5     |
| w/o N&O      | α=1.0        | 76.5     |
| w/o N&O      | α=0.5        | 73.0     |
| HeteroProx   | α=1.0        | 73.0     |
| HeteroProx   | α=0.5        | 72.5     |
</details>

(c) CIFAR-10.

![](images/099ba94f6f3ef196544c8f2d9ea037cbe456412b5f7b0287252700e487f85a1f.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoProx   | α=1.0        | 62       |
| InCoProx   | α=0.5        | 59       |
| w/o N      | α=1.0        | 61       |
| w/o N      | α=0.5        | 57       |
| w/o O      | α=1.0        | 62       |
| w/o O      | α=0.5        | 58       |
| w/o N&O    | α=1.0        | 60       |
| w/o N&O    | α=0.5        | 57       |
| HeteroProx | α=1.0        | 61       |
| HeteroProx | α=0.5        | 57       |
</details>

(d) CINIC-10.

Figure 21: Ablation studies for InCo Aggregation for FedProx. The federated settings are the same as Table 2.   
![](images/b44a7f232af1feab2f8b4fa41a64507d6d389364acca0866b1fdac467a340f01.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoNova   | α=1.0        | 87.5     |
| InCoNova   | α=0.5        | 86.5     |
| w/o N      | α=1.0        | 87.0     |
| w/o N      | α=0.5        | 84.0     |
| w/o O      | α=1.0        | 86.5     |
| w/o O      | α=0.5        | 83.5     |
| w/o N&O    | α=1.0        | 87.0     |
| w/o N&O    | α=0.5        | 82.0     |
| HeteroNova | α=1.0        | 86.5     |
| HeteroNova | α=0.5        | 85.0     |
</details>

(a) Fashion-MNIST.

![](images/c7aa060ddb4d56136599800225cafe853269908c42f2c4555167b7592d60fa6b.jpg)

<details>
<summary>line</summary>

| Method     | Distribution | Accuracy |
| ---------- | ------------ | -------- |
| InCoNova   | α=1.0        | 88.5     |
| InCoNova   | α=0.5        | 86.0     |
| w/o N      | α=1.0        | 87.0     |
| w/o N      | α=0.5        | 84.5     |
| w/o O      | α=1.0        | 87.0     |
| w/o O      | α=0.5        | 85.0     |
| w/o N&O    | α=1.0        | 86.0     |
| w/o N&O    | α=0.5        | 83.0     |
| HeteroNova | α=1.0        | 88.0     |
| HeteroNova | α=0.5        | 84.5     |
</details>

(b) SVHN.

![](images/a160e10126f70168a57c55f96d8f46947a1d682bfc85a6b4f29b3756721c43c0.jpg)

<details>
<summary>line</summary>

| Method     | α=1.0 | α=0.5 |
| ---------- | ----- | ----- |
| InCoNova   | 70.0  | 63.0  |
| w/o N      | 67.0  | 58.0  |
| w/o O      | 67.0  | 61.0  |
| w/o N&O    | 68.0  | 62.0  |
| HeteroNova | 69.0  | 61.0  |
</details>

(c) CIFAR-10.

![](images/2dfb5bd44b97162538d146bcd2706b6e0454156b3ddc06b1d9c32c40838f5d21.jpg)

<details>
<summary>line</summary>

| Method       | Distribution | Accuracy |
| ------------ | ------------ | -------- |
| InCoNova     | α=1.0        | 54       |
| w/o N        | α=1.0        | 52       |
| w/o O        | α=1.0        | 53       |
| w/o N&O      | α=1.0        | 52       |
| HeteroNova   | α=1.0        | 52       |
| InCoNova     | α=0.5        | 48       |
| w/o N        | α=0.5        | 47       |
| w/o O        | α=0.5        | 46       |
| w/o N&O      | α=0.5        | 46       |
| HeteroNova   | α=0.5        | 46       |
</details>

(d) CINIC-10.

Figure 22: Ablation studies for InCo Aggregation for FedNova. The federated settings are the same as Table 2.   
![](images/5f70fd3ef07bb8769c8dfd9dad7321d328e6c3c261999dc65850028dc9dbd0fe.jpg)

<details>
<summary>line</summary>

| Method          | Distribution | Accuracy |
| --------------- | ------------ | -------- |
| InCoScaffold    | α=1.0        | 90       |
| InCoScaffold    | α=0.5        | 88       |
| w/o N           | α=1.0        | 87       |
| w/o N           | α=0.5        | 86       |
| w/o O           | α=1.0        | 86       |
| w/o O           | α=0.5        | 87       |
| w/o N&O         | α=1.0        | 86       |
| w/o N&O         | α=0.5        | 85       |
| NeteroScaffold  | α=1.0        | 86       |
| NeteroScaffold  | α=0.5        | 85       |
</details>

(a) Fashion-MNIST.

![](images/b9ae4bd2539f0293cf37c1923d2fb25dcf3b8f69927947c8d8c077e0de91dd63.jpg)

<details>
<summary>line</summary>

| Method       | Distribution | Accuracy |
| ------------ | ------------ | -------- |
| InCoScaffold | α=1.0        | 88.0     |
| InCoScaffold | α=0.5        | 86.0     |
| U/o N        | α=1.0        | 85.0     |
| U/o N        | α=0.5        | 83.0     |
| w/o O        | α=1.0        | 86.0     |
| w/o O        | α=0.5        | 84.0     |
| w/o N&O      | α=1.0        | 86.0     |
| w/o N&O      | α=0.5        | 83.0     |
| HeteroScaffold | α=1.0    | 87.0     |
| HeteroScaffold | α=0.5    | 80.0     |
</details>

(b) SVHN.

![](images/b07de69f3f46f12bdf57efba1166807318ce16dfc5c11dd63327fe3c878b7389.jpg)

<details>
<summary>line</summary>

| Method          | Distribution | Accuracy |
| --------------- | ------------ | -------- |
| InCoScaffold    | α=1.0        | 74       |
| InCoScaffold    | α=0.5        | 68       |
| w/o N           | α=1.0        | 72       |
| w/o N           | α=0.5        | 66       |
| w/o O           | α=1.0        | 74       |
| w/o O           | α=0.5        | 65       |
| w/o N&O         | α=1.0        | 70       |
| w/o N&O         | α=0.5        | 66       |
| HeteroScaffold  | α=1.0        | 70       |
| HeteroScaffold  | α=0.5        | 65       |
</details>

(c) CIFAR-10.

![](images/9e037b88c7c2c88350171b88e98f3aca0f703e5ce3e782e945372aaa8929082f.jpg)

<details>
<summary>line</summary>

| Method          | Distribution | Accuracy |
| --------------- | ------------ | -------- |
| InCoScaffold    | α=1.0        | 62       |
| InCoScaffold    | α=0.5        | 54       |
| w/o N           | α=1.0        | 58       |
| w/o N           | α=0.5        | 52       |
| w/o O           | α=1.0        | 60       |
| w/o O           | α=0.5        | 52       |
| w/o N&O         | α=1.0        | 58       |
| w/o N&O         | α=0.5        | 52       |
| HeteroScaffold  | α=1.0        | 62       |
| HeteroScaffold  | α=0.5        | 54       |
</details>

(d) CINIC-10.   
Figure 23: Ablation studies for InCo Aggregation for Scaffold. The federated settings are the same as Table 2.

![](images/8dfcfa8b124938e86ba609abfa0aecd3ade31dfdd3b8031046810b86d19193aa.jpg)

<details>
<summary>line</summary>

| BatchSize | InCoProx | HeteroProx |
| --------- | -------- | ---------- |
| 32        | 75       | 72         |
| 64        | 74       | 72         |
| 128       | 72       | 69         |
| 256       | 69       | 65         |
</details>

(a) Different batch sizes.

![](images/0fc7bfab91172f76e58e145ab950f890b43d3581993e636e9b6bdd6f8f775304.jpg)

<details>
<summary>line</summary>

| Noise Std | InCoProx | HeteroProx |
| --------- | -------- | ---------- |
| 1*std     | 73.0     | 70.5       |
| 2*std     | 71.0     | 70.0       |
| 3*std     | 72.0     | 67.0       |
| 4*std     | 70.5     | 65.0       |
| 5*std     | 69.0     | 63.0       |
</details>

(b) Different noise perturbations.   
Figure 24: Robustness analysis for InCo Aggregation for FedProx in CIFAR-10.

![](images/332a69f91010085b37e3d75dd6c55a29089a9667620a311be55ca18deb43a9dc.jpg)

<details>
<summary>line</summary>

| BatchSize | InCoScaffold | HeteroScaffold |
| --------- | ------------ | -------------- |
| 32        | 69.5         | 65.0           |
| 64        | 67.5         | 65.5           |
| 128       | 65.5         | 62.5           |
| 256       | 64.0         | 62.0           |
</details>

(a) Different batch sizes.

![](images/c3c3d538fe631983e1caac26e95fb8f662b0c53183c41337a958e4cb017af269.jpg)

<details>
<summary>line</summary>

| Noise Std | InCoScaffold | HeteroScaffold |
| --------- | ------------ | -------------- |
| 1*std     | 68.0         | 63.0           |
| 2*std     | 64.0         | 63.5           |
| 3*std     | 64.5         | 60.0           |
| 4*std     | 63.0         | 58.5           |
| 5*std     | 62.5         | 55.0           |
</details>

(b) Different noise perturbations.   
Figure 25: Robustness analysis for InCo Aggregation for Scaffold in CIFAR-10.