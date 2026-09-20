# FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation

Yuntian Bo, Yazhou Zhu, Lunbo Li, Haofeng Zhang\*

School of Computer Science and Engineering, Nanjing University of Science and Technology, China

{yuntian.bo, zyz\_nj, lunboli, zhanghf}@njust.edu.cn

# Abstract

Existing few-shot medical image segmentation (FSMIS) models fail to address a practical issue in medical imaging: the domain shift caused by different imaging techniques, which limits the applicability to current FSMIS tasks. To overcome this limitation, we focus on the cross-domain few-shot medical image segmentation (CD-FSMIS) task, aiming to develop a generalized model capable of adapting to a broader range of medical image segmentation scenarios with limited labeled data from the novel target domain. Inspired by the characteristics of frequency domain similarity across different domains, we propose a Frequency-aware Matching Network (FAMNet), which includes two key components: a Frequency-aware Matching (FAM) module and a Multi-Spectral Fusion (MSF) module. The FAM module tackles two problems during the meta-learning phase: 1) intra-domain variance caused by the inherent support-query bias, due to the different appearances of organs and lesions, and 2) inter-domain variance caused by different medical imaging techniques. Additionally, we design an MSF module to integrate the different frequency features decoupled by the FAM module, and further mitigate the impact of inter-domain variance on the model's segmentation performance. Combining these two modules, our FAMNet surpasses existing FSMIS models and Cross-domain Few-shot Semantic Segmentation models on three cross-domain datasets, achieving state-of-the-art performance in the CD-FSMIS task. Code is available at https://github.com/primebo1/FAMNet.

# Introduction

To bridge the gap between limited labeled samples and the need for precise segmentation, few-shot medical image segmentation (FSMIS) (Guha Roy et al. 2019; Ouyang et al. 2022; Hansen et al. 2022; Zhu et al. 2023; Sun et al. 2022; Feng et al. 2021; Shen et al. 2023; Lin et al. 2023; Ding et al. 2023; Cheng et al. 2024) has emerged. By training on base categories, FSMIS models can leverage only a few annotated samples to segment new categories in medical images directly. Nevertheless, due to the limited generalization capability, they often exhibit diminished performance when tested on the data with domain shifts, which restricts their applicability to only a single domain.

![](images/269627480c5fe49758ee0b15db2be2ff5ac748eb05cbd84422457cc3627208db.jpg)  
Figure 1: Motivation of the proposed method. (a) CT and MRI scans in the spatial and frequency domains. Frequency spectra are processed using a Hamming window (Hamming 1977) and are center-shifted. (b) Quantitative metrics for the similarity of CT and MRI in the spatial and frequency domains using structural similarity index measure (SSIM) (Wang et al. 2004) and normalized mean square error (NMSE). Metrics are calculated using registered images.

Recently, some researchers have started investigating cross-domain few-shot semantic segmentation (CD-FSS) (Lei et al. 2022; Chen et al. 2024; Herzog 2024; Su et al. 2024; Nie et al. 2024; He et al. 2024), which has demonstrated impressive segmentation capabilities on datasets like Deepglobe (Demir et al. 2018) and FSS-1000 (Wei et al. 2019). Although this operation paves the way for cross-domain applications in few-shot scenarios, these models cannot be directly applied to the medical field due to the unique characteristics of medical images, e.g., grayscale, intensity variations, and foreground-background imbalance. Meanwhile, existing domain generalization methods in medical imaging (Ouyang et al. 2021; Zhou et al. 2022; Xu et al. 2022; Su et al. 2023) mainly focus on domain randomization, neglecting the model itself and the few-shot setting.

Two major challenges hinder the development of cross-domain few-shot medical image segmentation (CD-FSMIS), which we try to address in this paper: 1) Intra-domain variations: Medical images exhibit significant variability between individual organs, e.g., size, fat content, and pathology, making it difficult to find similar support-query pairs, leading to support-query bias and reduced prototype representation in

prototypical networks. 2) Inter-domain variations:

Even within the same organ or region, the spatial domain similarity demonstrates low correlation across different domains, as illustrated in Figure 1(b). However, subtle distinctions are evident in the frequency domain, where inter-domain variations are primarily in high and low-frequency bands, while mid-frequency bands are relatively similar.

We therefore propose a novel method termed Frequency-aware Matching Network (FAMNet) for CD-FSMIS in this paper. Specifically, the core of our FAMNet, the Frequency-aware Matching (FAM) module, performs support-query matching in specific frequency bands, eliminating support-query bias by fusing foreground features and highlighting synergistic parts. Simultaneously, FAM incorporates frequency domain information within the feature space, reducing reliance on frequency bands with significant domain differences. This allows the model to focus more on resilient, domain-agnostic frequency bands, effectively addressing both intra-domain and inter-domain variations. Building upon the FAM module, we subsequently developed a Multi-spectral Fusion (MSF) module. While fusing the frequency-decoupled features from FAM, the MSF module extracts the critical information that remains in the domain-specific frequency bands after decoupling in the spatial domain. With FAM and MSF, our method not only demonstrates strong generalization capabilities but also effectively leverages domain-invariant interactive information from the sample space, showcasing excellent segmentation performance. In summary, our contributions are as follows:

- We extend few-shot medical image segmentation to a new task, termed cross-domain few-shot medical image segmentation, aimed at training a generalizable model to segment a novel class in unseen target domains with only a few annotated examples.   
- We propose a novel FAM module that concurrently mitigates the adverse impacts of intra-domain and inter-domain variances on model performance. Moreover, an MSF module is introduced for multi-spectral feature fusion, further suppressing domain-variant information to enhance the model's generalizability.   
- On three cross-domain datasets, our proposed method archives the state-of-the-art performance. The effectiveness and superiority of our method are further verified through various ablation studies and visualization.

# Related Works

# Few-shot Medical Image Segmentation

The FSMIS task has been proposed to address data scarcity typically found in medical scenarios, which aims to train models capable of segmenting novel organs or lesions with only a few annotated samples. Current FSMIS models can be categorized into two approaches: interactive networks (Guha Roy et al. 2019; Sun et al. 2022; Feng et al. 2021; Ding et al. 2023) and prototypical networks (Ouyang et al. 2022; Hansen et al. 2022; Shen et al. 2023; Zhu et al. 2023; Lin et al. 2023; Cheng et al. 2024). In the former category, SENet (Guha Roy et al. 2019) pioneered the use of interactive networks in FSMIS tasks, followed by MR-rNet (Feng et al. 2021), GCN-DE (Sun et al. 2022), and CRAPNet (Ding et al. 2023). The core idea behind these models is to enhance support-query interaction through attention mechanisms. For the latter category, SSL-ALPNet (Ouyang et al. 2022) introduced a self-supervised framework that generates adaptive local prototypes and supervised by superpixel-based pseudo-labels during training. ADNet (Hansen et al. 2022) proposed a learnable threshold for segmentation and relied on a single foreground prototype to compute anomaly scores for all query pixels, rather than learning prototypes for each class. CATNet (Lin et al. 2023) utilized a cross-masked attention Transformer to enhance support-query interaction and improve feature representation. GMRD (Cheng et al. 2024) captured the complexity of prototype class distributions by generating multiple representative descriptors. Unfortunately, all existing FSMIS methods are limited to single-domain applications, neglecting the domain shifts encountered in medical imaging.

# Cross-domain Few-shot Semantic Segmentation

Expanding on few-shot semantic segmentation (FSS), recent studies (Herzog 2024; He et al. 2024; Su et al. 2024; Nie et al. 2024; Chen et al. 2024; Lei et al. 2022) focus on CD-FSS, considering a more practical setting where both label space and data distribution are disjoint between the training and testing datasets. PATNet (Lei et al. 2022) employs a Pyramid-Anchor-Transformation module (PATM) to map domain-specific features into domain-agnostic ones. PMNet (Chen et al. 2024) proposes a lightweight matching network to densely exploit pixel-to-pixel and pixel-to-patch correlations between support-query pairs. DRAdapter (Su et al. 2024) utilizes local-global style perturbation to train an adapter that rectifies diverse target domain styles to the source domain, maximizing the utilization of the well-optimized source domain segmentation model. Nevertheless, existing CD-FSS models often suffer from substantial performance degradation when applied to medical images due to significant differences from natural images, such as color, intensity, and foreground-background imbalance.

# Methodology

# Problem Setting

The CD-FSMIS task aims to construct a generalizable model $\Theta$ to segment novel organs or lesions in an unseen domain with few annotated medical images. To elaborate, the model $\Theta$ is optimized using a single source domain dataset $\mathcal{D}^s$ encompassing the base categories $\mathcal{C}_{base}$ . Subsequently, the model's performance is assessed on a target domain dataset $\mathcal{D}^t$ , which comprises novel target categories $\mathcal{C}_{target}$ with only a few labeled images. It is crucial to highlight that the sets of categories $\mathcal{C}_{base}$ and $\mathcal{C}_{target}$ are disjoint, i.e. $\mathcal{C}_{base} \cap \mathcal{C}_{target} = \emptyset$ , and a domain shift exists between the source domain $\mathcal{D}^s$ and the target domain $\mathcal{D}^t$ . During the training phase, the model has no access to the target domain.

Our approach adheres to the episode-based meta-learning paradigm. For each meta-learning task, we randomly divide

![](images/f3fe9cd613caaa8da613a574aeb318461bc4ec5d8624072f09203c167ba54f83.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Support Image"] --> B["Encoder"]
    B --> C["CpG"]
    C --> D["Fs^ini"]
    D --> E["AAP"]
    E --> F["Fs"]
    F --> G["ABM"]
    G --> H["Ff^l"]
    H --> I["Multi-Spectral Fusion (MSF)"]
    I --> J["ReLU"]
    J --> K["GAP"]
    K --> L["Pf^fg"]
    L --> M["Cosine"]
    M --> N["M̃q"]
    N --> O["Attention-based Matching (ABM)"]
    O --> P["JSM"]
    P --> Q["Aggregation"]
    Q --> R["MLP"]
    R --> S["Fb"]
    S --> T["GAP"]
    T --> U["Pf^fg"]
    U --> V["Cosine"]
    V --> W["M̃q"]
    
    subgraph Coarse Prediction Generation
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
    end
    
    subgraph Frequency Aware Matching (FAM)
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
    end
    
    subgraph Multi-Spectral Fusion (MSF)
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#cfc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#cfc,stroke:#333
    style Q fill:#fcc,stroke:#333
    style R fill:#ffc,stroke:#333
    style S fill:#cfc,stroke:#333
```
</details>

Figure 2: The overall architecture of our method, consists of three main technical components: the Coarse Prediction Generation (CPG) module, the Frequency-aware Matching (FAM) module, and the Multi-Spectral Fusion (MSF) module. Note that in ABM, JSM denotes joint space matching. In the case of DAFBs, the attention matrix is directly utilized for attention weighting. Conversely, for DSFBs, an element-wise subtraction is applied prior to the weighting process.

the data into multiple episodes. Each episode $(\mathcal{S}, \mathcal{Q})$ comprises: 1) a support set $S = \{x_{i}^{s}, M_{i}^{s}\}_{i=1}^{K}$ containing K support samples, and 2) a query set $Q = \{x_{i}^{q}, M_{i}^{q}\}_{i=1}^{N_{q}}$ containing $N_{q}$ query samples, where $x_{i}$ denotes the i-th image, and $M_{i}$ denotes the corresponding segmentation ground truth. During inference, the model's segmentation performance is evaluated by providing a support set and a query set from the novel target domain.

# Overall Architecture

The proposed network is depicted in Figure 2, which can be briefly divided into three main parts: 1) a Coarse Prediction Generation (CPG) module for generating a coarse prediction of the query mask, 2) a Frequency-aware Matching (FAM) module for performing frequency-aware matching between support and query foreground features, and 3) a Multi-Spectral Fusion (MSF) module for fusing features based on their respective frequency bands.

First, the support and query images are fed into a weight-sharing feature encoder to extract their corresponding feature maps. Next, a coarse prediction of the query foreground mask is obtained using the CPG module. Then, the support foreground feature computed in CPG, the generated coarse query mask, and the extracted query feature are input into the proposed FAM module for frequency-aware matching. In this module, features are divided into three frequency bands, each independently fuses the support and query features through distinct weighting mechanisms

guided by matching results, yielding fused features for each band. To reintegrate these multi-spectral features and further suppress the influence of domain-specific frequency bands (DSFBs), the divided features are fused through the MSF module. A global average pooling operation is then performed to obtain the foreground prototype required by the prototypical network. Finally, we compute the cosine similarity between the foreground prototype and the query feature to produce the final prediction of the query mask.

# Feature Extraction

We use a ResNet-50 (He et al. 2016) feature encoder $\mathcal{E}_{\theta}(\cdot)$ pre-trained on MS-COCO (Lin et al. 2014) as a weight-shared backbone to extract the support and query feature maps, where $\theta$ denotes the backbone parameters.

The support and query feature maps are denoted as $\mathbf{F}_{s}^{ini} = \mathcal{E}_{\theta}(x_{i}^{s})$ and $\mathbf{F}_{q}^{ini} = \mathcal{E}_{\theta}(x_{i}^{q})$ , and $F_{s}^{ini}, F_{q}^{ini} \in R^{C \times H \times W}$ , where C denotes the channel depth of the feature, and H and W denote the height and width of the feature respectively.

# Coarse Prediction Generation (CPG) Module

A prototypical network-based method is performed to obtain a coarse segmentation mask of the query image. Given a support image $x_{i}^{s}$ and its corresponding foreground binary mask $M^{s}$ , the support foreground prototype $p_{s}^{fg}$ can be generated by using Masked Average Pooling (MAP). Mathematically,

this process can be denoted as:

$$
\mathbf {p} _ {s} ^ {f g} = \frac {\sum_ {u , v} \mathbf {F} _ {s} ^ {i n i} (u , v) \mathbf {M} ^ {s} (u , v)}{\sum_ {u , v} \mathbf {M} ^ {s} (u , v)}, \tag {1}
$$

where $(u,v)$ is the index of pixels on the feature map.

Following this, we directly use the computed $p_{s}^{fg}$ and the extracted query feature map $F_{q}^{ini}$ to predict a coarse query foreground mask $\widetilde{M}_{q}^{Coarse}$ :

$$
\widetilde {\mathbf {M}} _ {q} ^ {C o a r s e} = 1. 0 - \sigma \left(d (\mathbf {F} _ {q} ^ {i n i}, \mathbf {p} _ {s} ^ {f g}) - \tau\right), \tag {2}
$$

where $d(a,b) = -\alpha \cos(a,b)$ is the negative cosine similarity with a fixed scaling factor $\alpha = 20$ (Wang et al. 2019), $\sigma(\cdot)$ denotes the Sigmoid activation and $\tau$ is a learnable threshold introduced by (Hansen et al. 2022).

# Frequency-Aware Matching (FAM) Module

Multi-Spectral Decoupling of Foreground Features. To mitigate the discrepancy between support and query foregrounds, we first extract the foreground features from $f_{s}$ and $f_{q}$ using the support mask and coarse query mask. Since the number of foreground pixels in the support and query images often differs, we apply Adaptive Average Pooling (AAP) (Liu et al. 2018) to standardize the number of foreground pixels to a fixed value N. This process can be formulated as:

$$
\left\{ \begin{array}{l} \mathbf {F} _ {s} = \mathrm{AAP} (\mathbf {F} _ {s} ^ {i n i} \odot \mathbf {M} ^ {s}, N) \\ \mathbf {F} _ {q} = \mathrm{AAP} (\mathbf {F} _ {q} ^ {i n i} \odot \mathcal {R} (\widetilde {\mathbf {M}} _ {q} ^ {C o a r s e}), N) \end{array} , \right. \tag {3}
$$

where $\odot$ denotes the Hadamard product, and $\mathbf{F}_s$ , $\mathbf{F}_q \in \mathbb{R}^{C \times N}$ denote the extracted foreground features, and $\mathrm{AAP}(a, n)$ is the AAP operation that adjusts the input feature map $a$ to a fixed output size $n$ along the last dimension, and $\mathcal{R}$ denotes the mathematical function that rounds decimals to 0 or 1.

For each foreground feature, we utilize the two-dimensional Fast Fourier Transform (FFT) to convert the signal from the spatial domain to the frequency domain while preserving spatial information. We employ a reshape function $\rho$ to transform the feature into a square, $\rho : R^{C \times N} \to R^{C \times \sqrt{N} \times \sqrt{N}}$ , with $\rho^{-1}$ serving as its inverse. For the support foreground feature, this process is formalized as:

$$
\phi_ {s} = \operatorname{SC} \left(\mathcal {F} \left(\rho \left(\mathbf {F} _ {s}\right)\right)\right), \tag {4}
$$

where $\mathcal{F}$ denotes FFT, $\phi_s \in \mathbb{C}^{C \times \sqrt{N} \times \sqrt{N}}$ denotes the frequency domain feature representation, SC denotes the function to adjusts the frequency signal to center the zero-frequency component.

Subsequently, we apply a band-pass filter to decompose the frequency-domain signal into three bands, namely high, medium, and low frequencies. Finally, we revert the frequency signals back to the spatial domain using the Inverse Fast Fourier Transform (IFFT):

$$
\mathbf {F} _ {s} (i) = \rho^ {- 1} (\mathcal {F} ^ {- 1} (\mathcal {B} _ {p} (\phi_ {s}, I (i)))) = \left\{ \begin{array}{l l} \mathbf {F} _ {s} ^ {l}, & i = 1 \\ \mathbf {F} _ {s} ^ {m}, & i = 2, \\ \mathbf {F} _ {s} ^ {h}, & i = 3 \end{array} \right. \tag {5}
$$

where $\mathbf{F}_{s}(i)$ denotes the support foreground feature in a specific frequency band, $F^{-1}$ denotes IFFT, $I(i)$ is a binary mask vector where values are set to 1 for preserved components and 0 for discarded components, and $\mathcal{B}_{p}(\phi,I)$ is the band-pass filter that can be defined by:

$$
\phi^ {\prime} (u, v) = \left\{ \begin{array}{l l} 0, & \text { if   } I (u, v) = 0 \\ \phi (u, v), & \text { otherwise } \end{array} \right. \tag {6}
$$

We perform the same operation on the query foreground feature to obtain $\mathbf{F}_q^l$ , $\mathbf{F}_q^{m}$ , $\mathbf{F}_q^h$ .

Multi-Spectrum Attention-based Matching. As shown in Figure 1, high-frequency and low-frequency signals vary significantly across different domains. During typical training, models often rely on these DSFBs to better adapt to the tasks in the current scenario. However, this reliance leads to over-fitting to certain prominent features. For example, a model may achieve precise segmentation on CT images by focusing on the contained information of DSFBs, such as the high-frequency edge information. Yet, this approach often suffers from degradation when transferred to an unseen domain where edge information is less distinct, such as MRI. Our proposed method aims to enhance the model's generalization capabilities by matching support and query features while simultaneously reducing the model's reliance on DSFBs.

Specifically, this module consists of three processes, as illustrated in the top-right part of Figure 2. First, given the support and query foreground feature pair $(\mathbf{f}_{s}, \mathbf{f}_{q})$ belonging to the same frequency band B, we apply linear transformations using two learnable matrices $W_{s}$ , $W_{q} \in R^{N \times N}$ to map the features into a joint space. This further reduces the intra-domain differences between the support and query, enhancing the stability of matching. This operation can be formalized as:

$$
\left\{ \begin{array}{l} \mathbf {F} _ {s} ^ {\prime} = \mathbf {F} _ {s} \mathbf {W} _ {s} \\ \mathbf {F} _ {q} ^ {\prime} = \mathbf {F} _ {q} \mathbf {W} _ {q} \end{array} , \right. \tag {7}
$$

where $\mathbf{F}_s^\prime, \mathbf{F}_q^\prime \in \mathbb{R}^{C \times N}$ denote the transformed support and query foreground features, respectively.

Secondly, based on the transformed features, we compute the attention-based similarity score matrix between the features using cosine similarity:

$$
\mathbf {A} (\mathbf {F} _ {s} ^ {\prime}, \mathbf {F} _ {q} ^ {\prime}) = \sigma (\frac {\mathbf {F} _ {s} ^ {\prime} \cdot \mathbf {F} _ {q} ^ {\prime}}{\| \mathbf {F} _ {s} ^ {\prime} \| \| \mathbf {F} _ {q} ^ {\prime} \|}), \tag {8}
$$

where $A \in R^{1 \times N}$ denotes the computed attention matrix, and $\sigma$ denotes the sigmoid activation function. During training, our module updates the attention scores between features via the learnable transformation matrices $W_{s}$ and $W_{q}$ , enabling the model to better learn domain-agnostic similarities between features.

Thirdly, based on the notion that DSFBs and domain-agnostic frequency bands (DAFBs) should be treated differently, we apply distinct attention weightings to feature pairs in three frequency bands. To match the size of $F_{s}^{\prime}$ and $F_{q}^{\prime}$ , we first obtain $A^{\prime} \in R^{C \times N}$ by repeating A along the channel dimension. For feature pairs in DAFBs, We directly multiply the attention matrix with the features element-wise to

highlight the similar components while suppressing the dissimilar ones:

$$
\left\{ \begin{array}{l} \mathbf {F} _ {s, 0} ^ {\prime \prime} = \mathbf {A} ^ {\prime} \odot \mathbf {F} _ {s} ^ {\prime} \\ \mathbf {F} _ {q, 0} ^ {\prime \prime} = \mathbf {A} ^ {\prime} \odot \mathbf {F} _ {q} ^ {\prime}, \end{array} \right. \tag {9}
$$

where $F_{s,0}^{\prime\prime}$ and $F_{q,0}^{\prime\prime} \in R^{C \times N}$ are the weighted features belonging to the DAFBs.

While for feature pairs in DSFBs, an inverse attention weighting is performed to suppress the similar components between features, thereby reducing the model's reliance on these components:

$$
\left\{ \begin{array}{l} \mathbf {F} _ {s, 1} ^ {\prime \prime} = \left(1 - \mathbf {A} ^ {\prime}\right) \odot \mathbf {F} _ {s} ^ {\prime} \\ \mathbf {F} _ {q, 1} ^ {\prime \prime} = \left(1 - \mathbf {A} ^ {\prime}\right) \odot \mathbf {F} _ {q} ^ {\prime}, \end{array} \right. \tag {10}
$$

where $F_{s,1}^{\prime\prime}$ and $F_{q,1}^{\prime\prime} \in R^{C \times N}$ are the weighted features belonging to the DSFBs.

Thus, the FAM module completes the enhancement or suppression of full-spectrum similarity between features through A, which is derived from full-spectrum attention weights. We pass the concatenated feature pair through an MLP afterward to fuse the features belonging to the support and query, obtaining a new fused feature that acts as the final feature representation of B:

$$
\mathbf {F} _ {b} = \operatorname{MLP} \left(\operatorname{Cat} \left(\mathbf {F} _ {s, f b} ^ {\prime \prime}, \mathbf {F} _ {q, f b} ^ {\prime \prime}\right), \varphi\right), \quad f b \in \{0, 1 \} \tag {11}
$$

where $F_{b} \in R^{C \times N}$ denotes the fused feature in a specific frequency band, $\mathrm{MLP}(\cdot, \varphi)$ is a function of a MLP with parameters $\varphi$ . Notably, the MLP consists of two fully connected layers with a ReLU activation function in between, which can better learn the fusion patterns and reduce the parameters of the MLP, $\operatorname{Cat}(a, b)$ denotes the function that concatenates a and b along the last dimension, and fb serves as an indicator specifying DAFBs or DSFBs.

# Multi-Spectral Fusion (MSF) Module

Recent research and our experiments in the Supplementary Materials have revealed the subtle relationship between frequency domain signals and image feature information: 1) Different frequency bands contain different information. Low and high frequencies contain color and style information, while the middle-frequency band contains more structural and shape information (Huang et al. 2021). 2) In cross-domain tasks, low and high frequencies exhibit significant differences across different domains. 3) Directly discarding high and low-frequency features is unreasonable, as frequency domain decomposition fails to completely decouple domain-variant information (DVI) and domain-invariant information (DII).

These insights lead us to a question: Is there a mechanism that can extract the DII remaining in the DSFBs guided by the DII in the DAFBs? We thus considered the cross-attention mechanism, which is widely used in multi-modal feature fusion: When feature $\Phi$ is used as the key (K) and value (V), and another feature $\Psi$ is used as the query (Q), the resulting V is the representation of feature $\Phi$ weighted by the similarity between feature $\Psi$ and feature $\Phi$ .

Propelled by this knowledge, we propose the MSF module, a cross-attention-based feature fusion module. This module retains high and low-frequency information while using mid-frequency information to extract DII from high and low-frequency features and suppress DVI.

To be specific, for each feature triplet $(\mathbf{F}_{f}^{l}, \mathbf{F}_{f}^{m}, \mathbf{F}_{f}^{h})$ , the three fused features do not overlap in the frequency domain, i.e., $\phi(\mathbf{F}_{f}^{l}) \cap \phi(\mathbf{F}_{f}^{m}) = \varnothing$ , $\phi(\mathbf{F}_{f}^{l}) \cap \phi(\mathbf{F}_{f}^{h}) = \varnothing$ , and $\phi(\mathbf{F}_{f}^{m}) \cap \phi(\mathbf{F}_{f}^{h}) = \varnothing$ , where $\phi(\mathbf{F})$ represents the spectrum of F. However, $\zeta(\mathbf{F}_{f}^{l}) \cap \zeta(\mathbf{F}_{f}^{m}) \neq \varnothing$ and $\zeta(\mathbf{F}_{f}^{m}) \cap \zeta(\mathbf{F}_{f}^{h}) \neq \varnothing$ , where $\zeta(\mathbf{F})$ denotes the contained information of F. We use $F_{f}^{l}$ or $F_{f}^{h}$ , along with $F_{f}^{m}$ to compute the attention between the given features. The output matrix of cross-attention (CA) can be represented as:

$$
f _ {\mathrm{CA}} (\mathbf {Q}, \mathbf {K}, \mathbf {V}) = \operatorname{softmax} \left(\frac {\mathbf {Q} \mathbf {K} ^ {T}}{\sqrt {d}}\right) \mathbf {V} = \mathbf {S V}, \tag {12}
$$

where $d$ denotes a scaling factor, $\mathbf{S} \in \mathbb{R}^{N \times N}$ denotes the attention weight matrix. Consequently, the process of obtaining the refined features for the low and high-frequency components can be formalized by:

$$
\left\{ \begin{array}{l} \mathbf {F} _ {f} ^ {l ^ {\prime}} = f _ {\mathrm{CA}} \left(\left(\mathbf {F} _ {f} ^ {m}\right) ^ {T} \mathbf {W} _ {Q}, \left(\mathbf {F} _ {f} ^ {l}\right) ^ {T} \mathbf {W} _ {K}, \left(\mathbf {F} _ {f} ^ {l}\right) ^ {T} \mathbf {W} _ {V}\right) ^ {T} \\ \mathbf {F} _ {f} ^ {h ^ {\prime}} = f _ {\mathrm{CA}} \left(\left(\mathbf {F} _ {f} ^ {m}\right) ^ {T} \mathbf {W} _ {Q}, \left(\mathbf {F} _ {f} ^ {h}\right) ^ {T} \mathbf {W} _ {K}, \left(\mathbf {F} _ {f} ^ {h}\right) ^ {T} \mathbf {W} _ {V}\right) ^ {T}, \end{array} \right. \tag {13}
$$

where $F_{f}^{l'}$ , $F_{f}^{h'}$ denote the refined fused foreground features, and $W_{Q}$ , $W_{K}$ , $W_{V} \in R^{C \times C}$ are the learnable linear transformation matrices.

Finally, a straightforward addition is performed to integrate the features from the three frequency bands, followed by a ReLU activation function, as the module output $F_{f}$ :

$$
\mathbf {F} _ {f} = \mathrm{ReLU} (\mathbf {F} _ {f} ^ {l ^ {\prime}} + \mathbf {F} _ {f} ^ {m} + \mathbf {F} _ {f} ^ {h ^ {\prime}}) \in \mathbb {R} ^ {C \times N}. \tag {14}
$$

We use the fused foreground features to compute the final query mask. A global average pooling (GAP) operation is performed to obtain the frequency-aware and query-informed foreground prototype:

$$
\mathbf {p} _ {f} ^ {f g} (c) = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {F} _ {f} (c, i), \tag {15}
$$

where $c$ denotes the channel index.

Hence, the final query foreground prediction of our proposed model can be calculated in a similar way in Eq. 2:

$$
\widetilde {\mathbf {M}} _ {q} ^ {f g} = 1. 0 - \sigma \left(d (\mathbf {F} _ {q} ^ {i n i}, \mathbf {p} _ {f} ^ {f g}) - \tau\right), \tag {16}
$$

while the background prediction can be obtained by $\widetilde{\mathbf{M}}_q^{bg} = 1 - \widetilde{\mathbf{M}}_q^{fg}$ accordingly.

# Objective Function

We adopt the binary cross-entropy loss $L_{ce}$ to evaluate the error between the predicted query mask and its corresponding ground truth. Mathematically, our final prediction loss $L_{final}$ can be expressed as:

$$
\begin{array}{l} \mathcal {L} _ {f i n a l} = \mathcal {L} _ {c e} (\mathbf {M} ^ {q}, \widetilde {\mathbf {M}} _ {q} ^ {f g}, \widetilde {\mathbf {M}} _ {q} ^ {b g}) \\ = - \frac {1}{H W} \sum_ {h, w} \mathbf {M} ^ {q} \log \left(\widetilde {\mathbf {M}} _ {q} ^ {f g}\right) + (1 - \mathbf {M} ^ {q}) \log \left(\widetilde {\mathbf {M}} _ {q} ^ {b g}\right). \tag {17} \\ \end{array}
$$

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ref.</td><td colspan="5">Abdominal CT → MRI</td><td colspan="5">Abdominal MRI → CT</td></tr><tr><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>PANet</td><td>ICCV&#x27;19</td><td>39.24</td><td>26.47</td><td>37.35</td><td>26.79</td><td>32.46</td><td>40.29</td><td>30.61</td><td>26.66</td><td>30.21</td><td>31.94</td></tr><tr><td>SSL-ALP</td><td>TMI&#x27;22</td><td>70.74</td><td>55.49</td><td>67.43</td><td>58.39</td><td>63.01</td><td>71.38</td><td>34.48</td><td>32.32</td><td>51.67</td><td>47.46</td></tr><tr><td>ADNet</td><td>MIA&#x27;22</td><td>50.33</td><td>39.36</td><td>37.88</td><td>39.37</td><td>41.73</td><td>64.25</td><td>37.39</td><td>25.62</td><td>42.94</td><td>42.55</td></tr><tr><td>QNet</td><td>IntelliSys&#x27;23</td><td>58.82</td><td>42.69</td><td>51.67</td><td>44.58</td><td>49.44</td><td>70.98</td><td>38.64</td><td>30.17</td><td>43.28</td><td>45.77</td></tr><tr><td>CATNet</td><td>MICCAI&#x27;23</td><td>44.58</td><td>43.67</td><td>50.27</td><td>46.34</td><td>46.21</td><td>54.52</td><td>41.73</td><td>40.24</td><td>45.84</td><td>45.60</td></tr><tr><td>RPT</td><td>MICCAI&#x27;23</td><td>49.22</td><td>42.45</td><td>47.14</td><td>48.84</td><td>46.91</td><td>65.87</td><td>40.07</td><td>35.97</td><td>51.22</td><td>48.28</td></tr><tr><td>PATNet</td><td>ECCV&#x27;22</td><td>57.01</td><td>50.23</td><td>53.01</td><td>51.63</td><td>52.97</td><td>75.94</td><td>46.62</td><td>42.68</td><td>63.94</td><td>57.29</td></tr><tr><td>IFA</td><td>CVPR&#x27;24</td><td>48.81</td><td>45.79</td><td>51.46</td><td>51.42</td><td>49.37</td><td>50.05</td><td>36.45</td><td>32.69</td><td>43.08</td><td>40.57</td></tr><tr><td>Ours</td><td>—</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td><td>73.57</td><td>57.79</td><td>61.89</td><td>65.78</td><td>64.75</td></tr></table>

Table 1: Quantitative comparison of different methods Dice score (%) on the Cross-Modality Dataset. The best value is shown in bold font, and the second best is underlined.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ref.</td><td colspan="4">Cardiac LGE → b-SSFP</td><td colspan="4">Cardiac b-SSFP → LGE</td></tr><tr><td>LV-BP</td><td>LV-MYO</td><td>RV</td><td>Mean</td><td>LV-BP</td><td>LV-MYO</td><td>RV</td><td>Mean</td></tr><tr><td>PANet</td><td>ICCV&#x27;19</td><td>51.43</td><td>25.75</td><td>25.75</td><td>36.66</td><td>36.24</td><td>26.37</td><td>23.47</td><td>28.69</td></tr><tr><td>SSL-ALP</td><td>TMI&#x27;22</td><td>83.47</td><td>22.73</td><td>66.21</td><td>57.47</td><td>65.81</td><td>25.64</td><td>51.24</td><td>47.56</td></tr><tr><td>ADNet</td><td>MIA&#x27;22</td><td>58.75</td><td>36.94</td><td>51.37</td><td>49.02</td><td>40.36</td><td>37.22</td><td>43.66</td><td>40.41</td></tr><tr><td>QNet</td><td>IntelliSys&#x27;23</td><td>50.64</td><td>37.88</td><td>45.24</td><td>44.58</td><td>31.08</td><td>34.03</td><td>39.45</td><td>34.85</td></tr><tr><td>CATNet</td><td>MICCAI&#x27;23</td><td>64.63</td><td>42.41</td><td>56.13</td><td>54.39</td><td>45.77</td><td>43.51</td><td>46.02</td><td>45.10</td></tr><tr><td>RPT</td><td>MICCAI&#x27;23</td><td>60.84</td><td>42.28</td><td>57.30</td><td>53.47</td><td>50.39</td><td>40.13</td><td>50.50</td><td>47.00</td></tr><tr><td>PATNet</td><td>ECCV&#x27;22</td><td>65.35</td><td>50.63</td><td>68.34</td><td>61.44</td><td>66.82</td><td>53.64</td><td>59.74</td><td>60.06</td></tr><tr><td>IFA</td><td>CVPR&#x27;24</td><td>64.04</td><td>43.22</td><td>74.58</td><td>62.28</td><td>68.07</td><td>36.07</td><td>60.42</td><td>54.85</td></tr><tr><td>Ours</td><td>—</td><td>86.64</td><td>51.84</td><td>76.26</td><td>71.58</td><td>77.37</td><td>52.05</td><td>54.75</td><td>61.39</td></tr></table>

Table 2: Quantitative comparison of different methods Dice score (%) on the Cross-Sequence Dataset. The best value is shown in bold font, and the second best is underlined.

To capture more precise and sufficient query foreground features, an accurate coarse prediction of the query foreground is needed. We continue to use the binary cross-entropy loss to quantify the dissimilarity between the coarse prediction and $M_{q}$ :

$$
\mathcal {L} _ {\text { coarse }} = \mathcal {L} _ {c e} (\mathbf {M} _ {q}, \widetilde {\mathbf {M}} _ {q} ^ {\text { coarse }}, 1 - \widetilde {\mathbf {M}} _ {q} ^ {\text { coarse }}). \tag {18}
$$

Overall, the computation of the total loss $L_{total}$ for our proposed model can be denoted as $L_{total} = L_{final} + L_{coarse}$ .

# Experiments

# Datasets

We detail our proposed task into three cross-domain settings and evaluate our method on the following three datasets:

The Cross-Modality dataset comprises two abdominal datasets. The first is Abdominal MRI obtained from (Kavur et al. 2021), which includes 20 3D T2-SPIR MRI scans. The second is Abdominal CT, which comprises 20 3D abdominal CT scans from (Landman et al. 2015). We select four common categories from the two datasets: the left kidney (LK), right kidney (RK), liver, and spleen, for assessment.

The Cross-Sequence dataset is a cardiac dataset from (Zhuang et al. 2022), which includes 45 3D LGE MRI scans and 45 b-SSFP MRI scans, both comprising 3 distinct labels: the blood pool (LV-BP), the left ventricle myocardium (LV-MYO), and the right ventricle myocardium (RV).

The Cross-Institution dataset consists of 321 3D prostate T2-weighted MRI scans collected by the University College London hospitals (UCLH) and 82 3D prostate MRI scans from the National Cancer Institute (NCI), Bethesda, Maryland, USA. The data from UCLH are collected from 4 studies: INDEX (Dickinson et al. 2013), the SmartTarget Biopsy Trial (Hamid et al. 2019), PICTURE (Simmons et al. 2014), Promise12 (Simmons et al. 2014), and organized by (Li et al. 2023). The data from NCI are provided in (Choyke et al. 2016). All the data are annotated by (Li et al. 2023). We select three common categories, bladder, central gland (CG) and rectum, for assessment.

# Implementation Details

Our method is implemented on an NVIDIA GeForce RTX 4080S GPU. Initially, we employ the 3D supervoxel clustering method (Hansen et al. 2022) to generate pseudomasks as the supervision in the episode-based meta-learning task, and we follow the same pre-processing techniques as (Hansen et al. 2022). The experiments are conducted under the 1-way 1-shot condition. During inference, we randomly

<table><tr><td rowspan="2" colspan="4">Baseline CPG FAM MSF</td><td colspan="5">CT → MRI</td></tr><tr><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>√</td><td></td><td></td><td></td><td>39.24</td><td>26.47</td><td>37.35</td><td>26.79</td><td>32.46</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>69.22</td><td>49.52</td><td>45.73</td><td>51.41</td><td>53.97</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>71.68</td><td>55.45</td><td>67.20</td><td>53.75</td><td>62.02</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td></tr></table>

Table 3: Ablation studies for the effect of each component in Dice score (%).

sample a scan from the source domain and select a middle slice containing the foreground as the support image, with the remaining slices as the query images. For all datasets, we train the model for 39K iterations, comprising 3000 iterations per epoch with the batch size set to 1. To comprehensively test the performance of our proposed model, we conduct bidirectional evaluations within each dataset. For instance, in the Cross-Modality dataset, we evaluate performance both on CT → MRI and MRI → CT directions.

Additionally, for the training of our model, the output size N for the adaptive average pooling in FAM is set to $30^{2}$ . In the multi-spectral decoupling of foreground features, we divide the frequency band into low, mid, and high frequencies with a ratio of 3:4:3. We chose the Stochastic Gradient Descent (SGD) optimizer with an initial learning rate of 0.001, a momentum of 0.9 and a a decay factor of 0.95 every 1K iterations.

# Evaluation metric

In order to evaluate the model under a uniform standard, we adopt the Sorensen-Dice coefficient (Ouyang et al. 2022) that is commonly used in FSMIS tasks, as the evaluation metric. The Dice score is used to evaluate the overlap between the segmentation results and the ground truth, which can be denoted as

$$
\operatorname{DSC} (X, Y) = \frac {2 | X \cap Y |}{| X | + | Y |}, \tag {19}
$$

where X and Y denote the two masks respectively, and the DSC denotes the Dice score ranges from 0 to 1, with 1 indicating complete overlap and 0 indicating no overlap.

# Quantitative and Qualitative Results

To demonstrate the effectiveness of our proposed method, we compare its performance with various FSMIS models, including PANet (Wang et al. 2019), SSL-ALPNet (Ouyang et al. 2022), ADNet (Hansen et al. 2022), QNet (Shen et al. 2023), CATNet (Lin et al. 2023), and RPT (Zhu et al. 2023). Additionally, two CD-FSS models, PATNet (Lei et al. 2022) and IFA (Nie et al. 2024) are also used for comparison. Note that we evaluate CD-FSS models without fine-tuning.

As shown in Table 1, our proposed method significantly outperforms all existing FSMIS and CD-FSS models under both CT → MRI and MRI → CT directions. Specifically, in the CT → MRI direction, the Dice score reached 65.79%, which is 2.78% higher than the second-best method. More significantly, the proposed model exhibited an overall 7.46% higher performance compared to the highest corresponding method in the MRI $\rightarrow$ CT direction. While PATNet performs 2.37% better than our model in the liver category in the MRI $\rightarrow$ CT direction, it underperforms in smaller categories like RK, LK, and spleen. This discrepancy arises from the imbalanced foreground and background in medical images, which the CD-FSS models do not adequately address. In contrast, our model is better adapted to medical scenarios, resulting in a higher overall Dice score.

<table><tr><td colspan="3">Frequency band</td><td colspan="5">CT → MRI</td></tr><tr><td>Low</td><td>Mid</td><td>High</td><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>-</td><td>+</td><td>-</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td></tr><tr><td>-</td><td>+</td><td>+</td><td>66.28</td><td>55.68</td><td>62.41</td><td>60.79</td><td>61.29</td></tr><tr><td>+</td><td>+</td><td>-</td><td>68.14</td><td>53.47</td><td>64.09</td><td>54.36</td><td>60.02</td></tr><tr><td>+</td><td>+</td><td>+</td><td>64.46</td><td>48.77</td><td>62.21</td><td>59.28</td><td>58.68</td></tr></table>

Table 4: Ablation study (in Dice score %) for the distinct attention weightings in DSFBs & DAFBs. Given attention matrix A, + indicates using A for attention weighting, and - indicates using 1-A for attention weighting.

As depicted in Table 2, Our method consistently performs exceptionally well on the Cross-Sequence dataset compared to other methods, achieving the highest Dice scores of 71.58% and 61.39% in both directions, surpassing the second-best method by 10.14% and 1.33%, respectively. In the LV-BP category under the LGE → b-SSFP scenario, our model even surpasses the second-best model by 21.29%, reaching 86.64%, which is comparable to FSMIS models' segmentation accuracy in non-cross-sequence conditions.

For quantitative and qualitative results on the Cross-Institution dataset and visual segmentation results on three cross-domain datasets, please refer to the Supplementary Materials. All experiments demonstrate that our FAMNet is a medical image segmentation model with excellent generalization capabilities and minimal data dependency.

# Ablation Studies

Effect of each component. In this section, we discuss the effect of each component. Table 3 shows the contribution of each module to the overall model performance. Combined with CPG, the proposed FAM module significantly enhances baseline (PANet) performance by 29.56%, primarily by mitigating overfitting to DSFBs and implementing inter-domain debiasing for the support and query features. Additionally, the MSF module aids in integrating the frequency-decoupled features from the FAM module, further suppressing DVI, and contributing an additional 3.77% improvement in model performance.

Distinct attention weightings in DAFBs & DSFBs. In the FAM module, we apply distinct attention weighting methods to assign weights to features belonging to DSFBs and DAFBs. This approach reduces the model's dependency on the support-query correlation within DSFBs while enhancing its focus on DAFBs. Table 4 illustrates the ablation study for distinct attention weightings. It is evident that using the uniform attention weighting method throughout leads to a

significant drop in the Dice score, with a reduction of $7.11\%$ , and applying positive attention weighting to any DSFB results in a decrease in model performance. This decline is attributed to substantial overfitting caused by the attention mechanism to the source domain's support-query correlation, which weakens the model's ability to generalize to novel target domains.

For further discussion, we present extensive ablation experiments in the Supplementary Materials, which include more method comparisons and hyperparameter analysis.

# Conclusion

In this paper, we have addressed a novel task: cross-domain few-shot medical image segmentation (CD-FSMIS). We proposed a Frequency-aware Matching Network (FAMNet), which comprises a Frequency-aware Matching (FAM) module to enhance the model's generalization capabilities and reduce support-query bias by performing attention-based matching of the foreground features for specific frequency bands, which simultaneously handle intra-domain and inter-domain variations. Furthermore, we introduced a Multi-Spectral Fusion (MSF) module to integrate features decoupled by the FAM module and further suppress the detrimental impact of domain-variant information on the model's robustness. Extensive experiments on three cross-domain datasets demonstrated the excellent generalization ability and data independence of the proposed method.

# Acknowledgment

This work was partly supported by the National Natural Science Foundation of China (NSFC) under Grant Nos. 62371235, 62076132 and 62072246, partly by the Key Research and Development Plan of Jiangsu Province (Industry Foresight and Key Core Technology Project) under Grant BE2023008-2.

# References

Chen, H.; Dong, Y.; Lu, Z.; Yu, Y.; and Han, J. 2024. Pixel Matching Network for Cross-Domain Few-Shot Segmentation. In WACV, 978–987.   
Cheng, Z.; Wang, S.; Xin, T.; Zhou, T.; Zhang, H.; and Shao, L. 2024. Few-Shot Medical Image Segmentation via Generating Multiple Representative Descriptors. IEEE Transactions on Medical Imaging, 43(6): 2202 – 2214.   
Choyke, P.; Turkbey, B.; Pinto, P.; Merino, M.; and Wood, B. 2016. Data from PROSTATE-MRI. The Cancer Imaging Archive.   
Demir, I.; Koperski, K.; Lindenbaum, D.; Pang, G.; Huang, J.; Basu, S.; Hughes, F.; Tuia, D.; and Raskar, R. 2018. DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images. In CVPR, 172–17209.   
Dickinson, L.; Ahmed, H.; Kirkham, A.; Allen, C.; Freeman, A.; Barber, J.; Hindley, R.; Leslie, T.; Ogden, C.; Persad, R.; Winkler, M.; and Emberton, M. 2013. A multi-centre prospective development study evaluating focal therapy using high intensity focused ultrasound for localised prostate

cancer: The INDEX study. Contemporary Clinical Trials, 36(1): 68–80.   
Ding, H.; Sun, C.; Tang, H.; Cai, D.; and Yan, Y. 2023. Few-shot Medical Image Segmentation with Cycle-resemblance Attention. In WACV, 2487–2496.   
Feng, R.; Zheng, X.; Gao, T.; Chen, J.; Wang, W.; Chen, D. Z.; and Wu, J. 2021. Interactive Few-Shot Learning: Limited Supervision, Better Medical Image Segmentation. IEEE Transactions on Medical Imaging, 40(10): 2575–2588.   
Guha Roy, A.; Siddiqui, S.; Pölsterl, S.; Navab, N.; and Wachinger, C. 2019. 'Squeeze & Excite' Guided Few-Shot Segmentation of Volumetric Images. Medical Image Analysis, 59: 101587.   
Hamid, S.; Donaldson, I. A.; Hu, Y.; Rodell, R.; Villarini, B.; Bonmati, E.; Tranter, P.; Punwani, S.; Sidhu, H. S.; Willis, S.; van der Meulen, J.; Hawkes, D.; McCartan, N.; Potyka, I.; Williams, N. R.; Brew-Graves, C.; Freeman, A.; Moore, C. M.; Barratt, D.; Emberton, M.; and Ahmed, H. U. 2019. The SmartTarget Biopsy Trial: A Prospective, Within-person Randomised, Blinded Trial Comparing the Accuracy of Visual-registration and Magnetic Resonance Imaging/Ultrasound Image-fusion Targeted Biopsies for Prostate Cancer Risk Stratification. European Urology, 75(5): 733–740.   
Hamming, R. W. 1977. Digital Filters. Signal Processing Series. Englewood Cliffs: Prentice-Hall.   
Hansen, S.; Gautam, S.; Jenssen, R.; and Kampffmeyer, M. 2022. Anomaly detection-inspired few-shot medical image segmentation through self-supervision with supervoxels. Medical Image Analysis, 78: 102385.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition. In CVPR, 770–778.   
He, W.; Zhang, Y.; Zhuo, W.; Shen, L.; Yang, J.; Deng, S.; and Sun, L. 2024. APSeg: Auto-Prompt Network for Cross-Domain Few-Shot Semantic Segmentation. In CVPR, 23762–23772.   
Herzog, J. 2024. Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation. In CVPR, 23605–23615.   
Huang, J.; Guan, D.; Xiao, A.; and Lu, S. 2021. FSDR: Frequency Space Domain Randomization for Domain Generalization. In CVPR, 6891–6902.   
Kavur, A. E.; Gezer, N. S.; Barış, M.; Aslan, S.; Conze, P.-H.; Groza, V.; Pham, D. D.; Chatterjee, S.; Ernst, P.; Özkan, S.; Baydar, B.; Lachinov, D.; Han, S.; Pauli, J.; Isensee, F.; Perkonigg, M.; Sathish, R.; Rajan, R.; Sheet, D.; Dovletov, G.; Speck, O.; Nürnberger, A.; Maier-Hein, K. H.; Bozdağı Akar, G.; Ünal, G.; Dicle, O.; and Selver, M. A. 2021. CHAOS Challenge - combined (CT-MR) healthy abdominal organ segmentation. Medical Image Analysis, 69: 101950.   
Landman, B.; Xu, Z.; Igelsias, J.; Styner, M.; Langerak, T.; and Klein, A. 2015. Miccai multi-atlas labeling beyond the cranial vault-workshop and challenge. In MICCAI Workshop, 12.   
Lei, S.; Zhang, X.; He, J.; Chen, F.; Du, B.; and Lu, C.-T. 2022. Cross-Domain Few-Shot Semantic Segmentation. In ECCV, 73–90.

Li, Y.; Fu, Y.; Gayo, I. J.; Yang, Q.; Min, Z.; Saeed, S. U.; Yan, W.; Wang, Y.; Noble, J. A.; Emberton, M.; et al. 2023. Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration. Medical Image Analysis, 90: 102935.   
Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft COCO: Common Objects in Context. In ECCV, 740–755.   
Lin, Y.; Chen, Y.; Cheng, K.-T.; and Chen, H. 2023. Few Shot Medical Image Segmentation with Cross Attention Transformer. In MICCAI, 233–243.   
Liu, S.; Qi, L.; Qin, H.; Shi, J.; and Jia, J. 2018. Path Aggregation Network for Instance Segmentation. In CVPR, 8759–8768.   
Nie, J.; Xing, Y.; Zhang, G.; Yan, P.; Xiao, A.; Tan, Y.-P.; Kot, A. C.; and Lu, S. 2024. Cross-Domain Few-Shot Segmentation via Iterative Support-Query Correspondence Mining. In CVPR, 3380–3390.   
Ouyang, C.; Biffi, C.; Chen, C.; Kart, T.; Qiu, H.; and Rueckert, D. 2022. Self-Supervised Learning for Few-Shot Medical Image Segmentation. IEEE Transactions on Medical Imaging, 41(7): 1837–1848.   
Ouyang, C.; Chen, C.; Li, S.; Li, Z.; Qin, C.; Bai, W.; and Rueckert, D. 2021. Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation. IEEE Transactions on Medical Imaging, 42: 1095–1106.   
Shen, Q.; Li, Y.; Jin, J.; and Liu, B. 2023. Q-Net: Query-Informed Few-Shot Medical Image Segmentation. In Arai, K., ed., Intelligent Systems and Applications, 610–628.   
Simmons, L. A.; Ahmed, H. U.; Moore, C. M.; Punwani, S.; Freeman, A.; Hu, Y.; Barratt, D.; Charman, S. C.; Van der Meulen, J.; and Emberton, M. 2014. The PICTURE study — Prostate Imaging (multi-parametric MRI and Prostate HistoScanning™) Compared to Transperineal Ultrasound guided biopsy for significant prostate cancer Risk Evaluation. Contemporary Clinical Trials, 37(1): 69–83.   
Su, J.; Fan, Q.; Pei, W.; Lu, G.; and Chen, F. 2024. Domain-Rectifying Adapter for Cross-Domain Few-Shot Segmentation. In CVPR, 24036–24045.   
Su, Z.; Yao, K.; Yang, X.; Huang, K.; Wang, Q.; and Sun, J. 2023. Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation. In AAAI, 2366–2374.   
Sun, L.; Li, C.; Ding, X.; Huang, Y.; Chen, Z.; Wang, G.; Yu, Y.; and Paisley, J. 2022. Few-shot medical image segmentation using a global correlation network with discriminative embedding. Computers in Biology and Medicine, 140:105067.   
Wang, K.; Liew, J. H.; Zou, Y.; Zhou, D.; and Feng, J. 2019. PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment. In ICCV, 9196–9205.   
Wang, Z.; Bovik, A.; Sheikh, H.; and Simoncelli, E. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612.

Wei, T.; Li, X.; Chen, Y. P.; Tai, Y.-W.; and Tang, C.-K. 2019. FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation. In CVPR, 2866–2875.

Xu, Y.; Xie, S.; Reynolds, M.; Ragoza, M.; Gong, M.; and Batmanghelich, K. 2022. Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation. In MICCAI, 671–681.

Zhou, Z.; Qi, L.; Yang, X.; Ni, D.; and Shi, Y. 2022. Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization. In CVPR, 20856–20865.

Zhu, Y.; Wang, S.; Xin, T.; and Zhang, H. 2023. Few-Shot Medical Image Segmentation via a Region-Enhanced Prototypical Transformer. In MICCAI, 271–280. Springer.

Zhuang, X.; Xu, J.; Luo, X.; Chen, C.; Ouyang, C.; Rueckert, D.; Campello, V. M.; Lekadir, K.; Vesal, S.; Ravi Kumar, N.; Liu, Y.; Luo, G.; Chen, J.; Li, H.; Ly, B.; Sermesant, M.; Roth, H.; Zhu, W.; Wang, J.; Ding, X.; Wang, X.; Yang, S.; and Li, L. 2022. Cardiac segmentation on late gadolinium enhancement MRI: A benchmark study from multi-sequence cardiac MR segmentation challenge. Medical Image Analysis, 81: 102528.

# Appendix for “FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation”

Yuntian Bo, Yazhou Zhu, Lunbo Li, Haofeng Zhang\*

School of Computer Science and Engineering, Nanjing University of Science and Technology, China

{yuntian.bo, zyz\_nj, lunboli, zhanghf}@njust.edu.cn

# Ablation Studies

Unless otherwise specified, all experiments are conducted using the same training setting and model configuration, consistent with the implementation details.

<table><tr><td rowspan="2">Attention Mechanisms</td><td rowspan="2">Affected Fg Pixels</td><td colspan="5">CT → MRI</td></tr><tr><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td rowspan="3">Hard Attention</td><td>20%</td><td>74.28</td><td>56.49</td><td>69.44</td><td>55.25</td><td>63.87</td></tr><tr><td>50%</td><td>74.29</td><td>58.47</td><td>66.45</td><td>57.00</td><td>64.05</td></tr><tr><td>80%</td><td>76.44</td><td>56.06</td><td>64.66</td><td>53.54</td><td>62.68</td></tr><tr><td>Soft Attention</td><td>100%</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td></tr></table>

Table 1: Comparison of hard attention at different dropout rates and soft attention in Dice score (%).

# Attention Mechanism in FAM

In this section, we discuss the implementation of attention in the FAM module. In our experiments, the FAM module employs soft attention to weight the support and query foreground features, thereby mitigating the model's reliance on DSFBs. An alternative approach involves using hard attention to discard similar points directly. We conducted analytical experiments to analyze the feasibility of this method and compared the results with our soft attention approach. The comparative results in Table 1 demonstrated the superiority of our method. Specifically, we selected the top $20\%$ , $50\%$ , $80\%$ of points based on the computed similarity matrix and discarded them accordingly.

# Feasibility of Directly Discarding Frequency Bands

In a previous section “Multi-Spectral Fusion (MSF) Module”, we concluded that directly discarding DSFBs is unreasonable. Table 2 presents experimental evidence supporting this conclusion. The experiments were conducted using the FAM module but without the inclusion of the MSF module. We observed that discarding any one or more frequency bands adversely affects the model’s performance compared to using all frequency bands, with a reduction of up to 3.90%.

<table><tr><td colspan="3">Frequency band</td><td colspan="5">CT → MRI</td></tr><tr><td>Low</td><td>Mid</td><td>High</td><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>√</td><td>√</td><td>√</td><td>71.68</td><td>55.45</td><td>67.20</td><td>53.75</td><td>62.02</td></tr><tr><td></td><td>√</td><td>√</td><td>71.62</td><td>54.89</td><td>64.21</td><td>55.54</td><td>61.57</td></tr><tr><td>√</td><td>√</td><td></td><td>70.77</td><td>55.61</td><td>66.64</td><td>53.12</td><td>61.54</td></tr><tr><td></td><td>√</td><td></td><td>71.71</td><td>50.05</td><td>55.06</td><td>55.66</td><td>58.12</td></tr></table>

Table 2: The impact of directly discarding a specific frequency band in Dice score (%).

Based on these results, we conjecture that this outcome is related to the content-irrelevance of frequency decoupling. For instance, MRI images contain a large amount of fine textures that are absent in CT images, causing a domain shift. When discarding the high-frequency band, the fine texture information should be discarded as well. However, the high-frequency band also includes edge information, which plays a significant positive role in segmentation (Cheng et al. 2024), and this information remains consistent between CT and MRI (edge information of organs is similar in both modalities). Directly discarding the high-frequency band leads to the loss of both DVI and DII, resulting in the model learning in an information-deficient environment and thus decreasing segmentation performance. Furthermore, directly discarding a specific frequency band prevents the attention weights in that band from being trained, leading to over-fitting in the remaining frequency bands.

# Performing Matching Exclusive to Specific Frequency Bands

Instead of matching features across all frequency bands, this section focuses on matching exclusive to specific frequency bands. At least one DAFB is used to train the attention weights for support-query correlation to prevent overfitting issues, as detailed in the section “Distinct Attention Weightings in DAFBs & DSFBs”. The results of the study are shown in Table 3. When the low or high-frequency bands are not matched, the Dice score decreases by 2.59% and 2.01%, respectively, compared to matching across all frequency bands. Excluding both low and high-frequency bands from matching results in a minimum Dice score of 61.63%. This decline occurs because the query sample may

<table><tr><td colspan="3">Frequency band</td><td colspan="5">CT → MRI</td></tr><tr><td>Low</td><td>Mid</td><td>High</td><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>√</td><td>√</td><td>√</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td></tr><tr><td></td><td>√</td><td>√</td><td>72.62</td><td>55.68</td><td>67.25</td><td>57.26</td><td>63.20</td></tr><tr><td>√</td><td>√</td><td></td><td>74.01</td><td>57.08</td><td>66.62</td><td>57.40</td><td>63.78</td></tr><tr><td></td><td>√</td><td></td><td>72.90</td><td>56.00</td><td>64.26</td><td>53.05</td><td>61.63</td></tr></table>

Table 3: Ablation study (in Dice score %) for the impact of matching is exclusive to specific frequency bands. √ signifies feature matching in the corresponding frequency band.

be out of distribution or the support prototype may not accurately represent the mean of a category due to potential intra-domain shifts between support and query samples, a phenomenon particularly evident in the 1-shot setting. These inter-domain variations have detrimental effects on the segmentation performance of the CD-FSMIS model. Figure 3 illustrates the debiasing effect achieved by our proposed FAM module.

# Impact of Different Frequency Band Division Ratios

In this section, we discuss the impact of different frequency band division ratios on model performance. The division ratio determines our segmentation of DSFBs and DAFBs. In the FAM module, we apply different attention weightings to the image features belonging to DSFBs and DAFBs based on this segmentation. In the MSF module, we extract DII from features in DAFBs to extract residual DII from DSFBs, while simultaneously suppressing DVI in the final module output. Table 4 shows the impact of different division ratios. From Table 4, we observe that our model's performance peaks with a Dice score of 66.29% when the low:mid:high ratio is 3.5:3:3.5. When the proportion of the mid-frequency band exceeds 30%, the model's performance shows a decreasing trend as the mid-frequency band increases. We attribute this to the blurred boundary between DSFBs and DAFBs. During frequency band division, DAFBs inevitably include some frequency domain information from DSFBs. As the bandwidth of the mid-frequency band increases, DAFBs contain more domain-specific frequency signals. This deviates from the design intent of the FAM and MSF modules. In the FAM module, certain support-query matching relationships from DSFBs are also reinforced, which could be crucial for optimization during source domain training, leading to overfitting of source domain information. In the MSF module, since the mid-frequency band contains DVI, its guiding significance in the cross-attention extraction process decreases, resulting in certain DVI in DSFBs being enhanced rather than suppressed. When the proportion of the mid-frequency band falls below 30%, the model's performance declines sharply due to insufficient information in the mid-frequency band, rendering both the FAM and MSF modules inadequately trained.

<table><tr><td colspan="3">Division Ratio (Low:Mid:High)</td><td colspan="5">CT → MRI</td></tr><tr><td>Low</td><td>Mid</td><td>High</td><td>Liver</td><td>LK</td><td>RK</td><td>Spleen</td><td>Mean</td></tr><tr><td>2.0</td><td>6.0</td><td>2.0</td><td>75.11</td><td>56.49</td><td>66.35</td><td>57.27</td><td>63.81</td></tr><tr><td>2.5</td><td>5.0</td><td>2.5</td><td>73.19</td><td>57.06</td><td>70.75</td><td>56.06</td><td>64.27</td></tr><tr><td>3.0</td><td>4.0</td><td>3.0</td><td>73.01</td><td>57.28</td><td>74.68</td><td>58.21</td><td>65.79</td></tr><tr><td>3.5</td><td>3.0</td><td>3.5</td><td>72.75</td><td>60.52</td><td>74.44</td><td>57.44</td><td>66.29</td></tr><tr><td>4.0</td><td>2.0</td><td>4.0</td><td>74.01</td><td>59.35</td><td>72.77</td><td>57.37</td><td>65.88</td></tr><tr><td>4.5</td><td>1.0</td><td>4.5</td><td>68.68</td><td>55.77</td><td>68.42</td><td>60.31</td><td>63.30</td></tr></table>

Table 4: Ablation study (in Dice score %) for the impact of different frequency band division ratios.

# Impact of Foreground Pixel Number $N$

In this section, we analyze the fixed number N of foreground pixels, which standardizes the number of foreground pixels through the adaptive average pooling. In our experiment, we selected N as $30^{2}$ . We further vary N and evaluate the model's performance in CT → MRI direction, as illustrated in Figure 1. It can be observed that as N increases, the Dice score initially rises steadily and then fluctuates around a certain value. When N is $60^{2}$ , the model achieves the highest Dice score, but the difference compared to the reported results is minimal. Meanwhile, since the value of N is directly related to the subsequent network design, a large N introduces a significant number of unnecessary parameters and computational load. Hence, it is crucial to choose a moderate value of N to maintain our model's efficiency.

![](images/820cda56b9e24164051331d58ab82c873b69c770d99c77104ce1eed353d56b78.jpg)

<details>
<summary>bar</summary>

| Values of N | Dice Score (%) |
| :--- | :--- |
| 20² | 60.28 |
| 23² | 62.91 |
| 25² | 64.21 |
| 27² | 65.11 |
| 30² | 65.79 |
| 35² | 65.84 |
| 40² | 65.50 |
| 60² | 66.01 |
</details>

Figure 1: Ablation study (in Dice score %) for the impact of different N values.

# Quantitative and Qualitative Result on the Cross-Institution Dataset

As shown in Table 5, our FAMNet also excels on the Cross-Institution dataset, achieving a 2.27% and 3.18% higher Dice score in the UCLH → NCI and NCI → UCLH directions, respectively, compared to the second-best method. The performance improvement is attributed to FAMNet's ability to better address the variations arising from the use of instruments of different models or different manufacturers across institutions. These discrepancies typically result in differences in image contrast, signal-to-noise ratio, and other factors, which are particularly pronounced in the fre-

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ref.</td><td colspan="4">Prostate UCLH → NCI</td><td colspan="4">Prostate NCI → UCLH</td></tr><tr><td>Bladder</td><td>CG</td><td>Rectum</td><td>Mean</td><td>Bladder</td><td>CG</td><td>Rectum</td><td>Mean</td></tr><tr><td>PANet</td><td>ICCV&#x27;19</td><td>51.85</td><td>38.89</td><td>36.28</td><td>42.34</td><td>49.90</td><td>36.27</td><td>43.58</td><td>43.25</td></tr><tr><td>SSL-ALP</td><td>TMI&#x27;22</td><td>54.86</td><td>42.59</td><td>41.65</td><td>46.37</td><td>66.6</td><td>43.12</td><td>56.94</td><td>55.55</td></tr><tr><td>ADNet</td><td>MIA&#x27;22</td><td>53.92</td><td>46.11</td><td>42.26</td><td>47.43</td><td>62.84</td><td>51.34</td><td>60.45</td><td>58.21</td></tr><tr><td>QNet</td><td>IntelliSys&#x27;23</td><td>39.66</td><td>35.71</td><td>34.25</td><td>36.54</td><td>41.22</td><td>43.21</td><td>39.19</td><td>41.21</td></tr><tr><td>CATNet</td><td>MICCAI&#x27;23</td><td>47.61</td><td>45.29</td><td>42.51</td><td>45.14</td><td>51.98</td><td>48.81</td><td>55.01</td><td>51.93</td></tr><tr><td>RPT</td><td>MICCAI&#x27;23</td><td>52.17</td><td>41.22</td><td>46.93</td><td>46.77</td><td>62.35</td><td>51.66</td><td>64.63</td><td>59.55</td></tr><tr><td>PATNet</td><td>ECCV&#x27;22</td><td>50.04</td><td>52.71</td><td>44.57</td><td>49.10</td><td>69.91</td><td>53.02</td><td>62.79</td><td>61.91</td></tr><tr><td>IFA</td><td>CVPR&#x27;24</td><td>41.02</td><td>39.15</td><td>36.45</td><td>38.87</td><td>52.54</td><td>39.85</td><td>34.27</td><td>42.22</td></tr><tr><td>Ours</td><td>—</td><td>53.06</td><td>48.48</td><td>52.57</td><td>51.37</td><td>77.64</td><td>48.73</td><td>68.91</td><td>65.09</td></tr></table>

Table 5: Quantitative comparison of different methods Dice score (%) on the Cross-Institution Dataset. The best value is shown in bold font, and the second best is underlined.

![](images/d498ec8961c51b0f7ebaa8f8d68cf5c9090da2ff49ef5eac54e39d195b1369d1.jpg)

<details>
<summary>text_image</summary>

Query Image
GT
Prediction
Uncertainty
1.0
0.8
0.6
0.4
0.2
0.0
</details>

Figure 2: Illustration of query images, ground truths, predictions, and uncertainty maps in the directions of CT → MRI, LGE → b-SSFP, NCI → UCLH by FAMNet.

quency domain, especially in the high and low-frequency bands (Tang, Peli, and Acton 2003; Bernhardt et al. 2005). FAMNet effectively mitigates the impact of these domain shifts on segmentation performance, thereby enhancing its robustness and accuracy across different institutional data.

# Visualization

# Uncertainty Maps

Figure 2 illustrates three segmentation examples by FAM-Net from the CD-FSMIS task in the directions of CT → MRI, LGE → b-SSFP, and NCI → UCLH. By observing the predictions and uncertainty maps, it is evident that the edges of the target regions are typically highlighted. This indicates a higher model uncertainty in segmenting these areas, due to the differences in edge depiction across domains, e.g., CT images generally have sharper edges compared to MRI images. Additionally, the model exhibits greater uncertainty when segmenting smaller targets compared to larger organs. This is attributed to the varying focus on detailed re-

![](images/fc7e0613df66ea98553868dd0c1042c18ed32efb00d5847ed61955a44d30bb69.jpg)

<details>
<summary>text_image</summary>

(a)
Support Image
Support Label
(a)
Query Image
Query GT
(b)
Support Image
Support Label
(b)
Query Image
Query GT
(c)
Support Image
Support Label
(c)
Query Image
Query GT
support_fg_pixel
query_fg_pixel
support_prototype
support_fg_pixel
query_fg_pixel
FAMNet_fg_pixel
support_prototype
FAMNet_prototype
support_fg_pixel
query_fg_pixel
FAMNet_fg_pixel
support_prototype
FAMNet_prototype
</details>

Figure 3: T-SNE visualization for intra-domain variations, i.e., support-query bias, and debiasing capability of FAM-Net. 'fg' represents the foreground. In (b), the foregrounds in the support and query images exhibit significant differences in brightness and contrast, with notable discrepancies in texture and structural details as well. In (c), there is a pronounced size imbalance in the foregrounds, accompanied by substantial variations in brightness.

gions by different imaging techniques. Furthermore, when multiple targets are in close proximity or intersect, there is an increased uncertainty, which may lead to false segmentation.

# FAMNet's Capability to Mitigate Intra-domain Variations

Figure 3 visualizes the capability of our FAMNet to significantly narrow the support-query bias using t-SNE (Van der Maaten and Hinton 2008). As shown in Figure 3(a), typical support and query samples are closely situated and highly intermixed in the feature space, and the support prototype effectively represents the mean of the query foreground's overall distribution, resulting in excellent segmentation performance. Figure 3(b)(c) illustrates that when there is a moderate or significant bias between support and query samples, the support prototype deviates from the cluster of query foreground pixels in the feature space. This deviation reduces the representational capability of the support prototype, thereby degrading the performance of the cosine similarity-based mask calculation method. FAMNet calculates a new prototype through support-query matching, correcting the bias of the support prototype, thereby reducing or eliminating the detrimental impact of intra-domain variation on segmentation results.

# References

Bernhardt, P.; Batz, L.; Ruhrnschopf, E.-P.; and Hoheisel, M. 2005. Spatial frequency-dependent signal-to-noise ratio as a generalized measure of image quality. In Medical Imaging 2005: Physics of Medical Imaging, volume 5745, 407–418. SPIE.   
Cheng, Z.; Wang, S.; Xin, T.; Zhou, T.; Zhang, H.; and Shao, L. 2024. Few-Shot Medical Image Segmentation via Generating Multiple Representative Descriptors. IEEE Transactions on Medical Imaging, 43(6): 2202 – 2214.   
Tang, J.; Peli, E.; and Acton, S. 2003. Image enhancement using a contrast measure in the compressed domain. IEEE Signal Processing Letters, 10(10): 289–292.   
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. Journal of machine learning research, 9(11).