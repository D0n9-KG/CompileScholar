# HSOD-BIT-V2: A New Challenging Benchmark for Hyperspectral Salient Object Detection

Yuhao Qiu, Shuyan Bai, Tingfa Xu $^{\dagger}$ , Peifu Liu, Haolin Qin, Jianan Li $^{\dagger}$

Beijing Institute of Technology

# Abstract

Salient Object Detection (SOD) is crucial in computer vision, yet RGB-based methods face limitations in challenging scenes, such as small objects and similar color features. Hyperspectral images provide a promising solution for more accurate Hyperspectral Salient Object Detection (HSOD) by abundant spectral information, while HSOD methods are hindered by the lack of extensive and available datasets. In this context, we introduce HSOD-BIT-V2, the largest and most challenging HSOD benchmark dataset to date. Five distinct challenges focusing on small objects and foreground-background similarity are designed to emphasize spectral advantages and real-world complexity. To tackle these challenges, we propose Hyper-HRNet, a high-resolution HSOD network. Hyper-HRNet effectively extracts, integrates, and preserves effective spectral information while reducing dimensionality by capturing the self-similar spectral features. Additionally, it conveys fine details and precisely locates object contours by incorporating comprehensive global information and detailed object saliency representations. Experimental analysis demonstrates that Hyper-HRNet outperforms existing models, especially in challenging scenarios.

# 1 Introduction

Salient object detection (SOD) is crucial in various applications (2021; 2021), typically using RGB images to identify prominent objects. However, RGB images struggle with accurate localization in challenging scenes, such as color similarity between foreground and background, due to reliance on shape and color features (2021). In contrast, spectral curves offer a more detailed characterization of objects' intrinsic properties (2024; 2024), as shown in Figure 1 (a). Hyperspectral salient object detection (HSOD) methods utilize the abundant spectral information available in hyperspectral images (HSIs) to capture detailed object features, delivering enhanced performance even in challenging conditions (2022). Thus, integrating HSIs into SOD shows great promise for improving accuracy in challenging scenarios.

Convolutional neural networks (CNNs) enhance feature representation, boosting performance in HSOD task (2019). However, deep learning methods require extensive high-quality data, which is scarce in HSOD. Previous datasets

![](images/86f33e4f37dfab94e6cd26bf11e968f86ddaf5822dae2bf52c210ac582b2a2c8.jpg)  
Figure 1: (a) Exemplary challenging scenarios from HSOD-BIT-V2, where objects are hard to identify in pseudo-color images but exhibit salience in spectral curves. (b) Challenge attributes of HSOD-BIT (2024) and HSOD-BIT-V2, highlighting their capability to represent real-world challenges.

primarily sourced from publicly available HSIs not curated for HSOD (2002), suffer from imprecise annotation and inadequate quantity and quality. Although the dedicated HS-SOD dataset (2018) represents progress, it remains small and of low quality. The HSOD-BIT dataset (2024) expands the dataset scale and introduces challenges like non-uniform lighting and overexposure, yet its limited diversity still hinders the full utilization of spectral advantages.

To bridge these gaps, we construct HSOD-BIT-V2, the largest and most challenging HSOD dataset to date, featuring 500 high-quality HSIs. This dataset includes eight natural scene backgrounds and, for the first time in HSOD, introduces snowfields and fallen leaves, greatly enhancing diversity. The dataset emphasizes spectral advantages through five challenging attributes, focusing on small objects and foreground-background similarities. As depicted in Figure 1 (b), our dataset contains 417 challenging samples and outperforms HSOD-BIT across all attributes. With its expanded scale and varied challenges, HSOD-BIT-V2 provides enhanced data support for the HSOD task and serves as a new benchmark for algorithmic evaluation.

HSOD methods currently encounter three main challenges: (i) Spectral redundancy raises computational costs, reduces effective information density, and diminishes detection accuracy due to the Hughes phenomenon (2003). Existing dimensionality reduction techniques, such as PCA (2023), often lead to information loss. (ii) Effectively capturing spec-

tral features is vital due to the high spectral self-similarity and spatial sparsity of HSIs. Despite related research emphasizes this need (2022), fully exploiting spectral features remains difficult. (iii) Accurately distinguishing object contours is essential for dense supervision tasks. Current methods often lose fine details through interpolation or pooling during downsampling (2023), making accurate contour detection challenging.

To tackle these challenges, we present Hyper-HRNet, a high-resolution HSOD network. Hyper-HRNet optimizes HSI utilization and minimizes spectral dimensionality while preserving crucial spectral information. It achieves this by synergizing CNN and Transformer for effective spectral feature extraction and reconstruction. Additionally, it supplements high-resolution flow decoding with intact global information and detailed object saliency representations to convey fine details and precisely locate object contours.

Firstly, Hyper-HRNet introduces Hyperspectral Attention Reconstruction to effectively capture spectral features and address spectral redundancy. This component combines CNN and Transformer for effective spectral dimensionality reduction and reconstruction. The CNN adaptively captures high- and low-frequency spectral details, preserving edge and saliency features. Concurrently, the Transformer processes spectral feature map as a token to capture contextual spectral information and address long-range dependencies often inadequately handled by CNNs. This process harnesses the self-similar spectral features to enable seamless interactions in spectral-wise, thereby reducing spectral dimensionality and preserving effective spectral features.

Finally, Hyper-HRNet employs Global Ternary Perception Decoder to convey fine details and precisely delineate object contours. It fuses high-resolution flow from the backbone and enhances decoding through two modules: (i) Global Attention Feature Aggregator, which utilizes features processed with PixelShuffle to produce a saliency map containing intact global information and offset fine details loss typically seen in multi-scale decoding; (ii) Ternary-Aware Weight, which converts saliency predictions into ternary weights to emphasize essential regions between background and object, thereby improving contour localization accuracy.

Extensive experiments have been conducted to evaluate the performance of Hyper-HRNet on HSOD-BIT-V2, HSOD-BIT and HS-SOD datasets. Our model surpasses mainstream models, especially in challenging backgrounds.

Our contributions can be summarized as follows:

- We construct HSOD-BIT-V2, the largest and most challenging HSOD dataset to date, featuring five distinct attributes designed to highlight spectral advantages.   
- We introduce Hyper-HRNet, a novel network that effectively leverages HSIs to address spectral dimensionality and accurately delineate object contours.   
- We propose Hyperspectral Attention Reconstruction to optimize HSI utilization and reduce spectral dimensionality while preserving essential spectral information.   
- We develop Global Ternary Perception Decoder to enhance decoding by integrating intact global information and detailed object saliency representations.

# 2 Related Work

Salient Object Detection. Traditional SOD methods relied on low-level features to measure saliency (1998), often emphasizing high-contrast edges rather than salient objects due to limited feature representation (2019). CNNs made great strides in SOD (2015; 2019a). Recent works adopt a two-stage framework to generate a trimap for ensuring clear edges (2021), while also enhancing global context modeling through Transformer-based patch-wise branches (2023). Regrettably, these methods are limited to RGB data and tend to perform poorly when directly applied to HSIs.

Hyperspectral Salient Object Detection. Despite advances in SOD, HSOD remains unexplored. Previous methods relied on shallow features like spectral gradients (2013; 2018), and utilized PCA for dimensionality reduction (2013), which often led to information loss or inadequate saliency capture. Deep learning models address these issues by incorporating spectral saliency and edge features to reduce information loss (2023), and using CNNs with knowledge distillation for dimensionality reduction (2024). However, challenges remain in spectral feature utilization, information loss, and edge delineation. Therefore, we propose an attention-based component to better capture spectral self-similarity and a novel decoder to enhance object contours.

Hyperspectral Salient Object Detection Datasets. Acquiring HSIs is intricate, resulting in a scarcity of suitable data. Previous datasets, which relied on publicly available data not specifically curated for HSOD (2004; 2011), feature low-precision annotations and inferior quality. The first tailored HSOD dataset HS-SOD is small and limited to common scenes (2018). The HSOD-BIT dataset (2024), while larger and including some challenges, still lacks sufficient challenging data to fully showcase spectral advantages. Hence, HSOD requires larger, more diverse, and higher-quality datasets spanning various environmental scenarios.

# 3 HSOD-BIT-V2 Dataset

# 3.1 Overview

HSOD-BIT-V2 overcomes limitations in scale, quality, and challenge of current datasets. Table 1 shows it surpassing existing HS-SOD and HSOD-BIT, with 500 HSIs, $1240 \times 1680$ spatial resolutions, and 200 spectral bands. Unlike HS-SOD, which focuses on common scenes, and HSOD-BIT, with limited challenging data, HSOD-BIT-V2 covers 8 natural backgrounds with diverse and challenging data, highlighting small objects and foreground-background similarity.

# 3.2 Dataset Construction

HSOD-BIT-V2 includes 8 natural backgrounds across various weather conditions, as shown in Figure 3 (a), ensuring diversity and representativeness. Each scene type features multiple scenarios, with consistent imaging parameters for uniformity. To expand the dataset, we integrated and processed HSOD-BIT (2024), maintaining data coherence. Original data underwent dark current noise reduction, calibration, and quality evaluation, excluding low-quality or insufficiently challenging images. From the 500 processed data cubes, 406 images were used for training, and 94 for

<table><tr><td>Property</td><td>HS-SOD</td><td>HSOD-BIT</td><td>HSOD-BIT-V2</td></tr><tr><td>Data Volume</td><td>60</td><td>319</td><td>500</td></tr><tr><td>Spatial Resolution</td><td> $768 \times 1024$ </td><td> $1240 \times 1680$ </td><td> $1240 \times 1680$ </td></tr><tr><td>Spectral Bands</td><td>81</td><td>200</td><td>200</td></tr><tr><td>Spectral Resolution</td><td>5nm</td><td>3nm</td><td>3nm</td></tr><tr><td>Spectral Range</td><td>380-700nm</td><td>400-1000 nm</td><td>400-1000 nm</td></tr><tr><td>Challenges</td><td>0</td><td>278</td><td>459</td></tr><tr><td>F-B similiarty</td><td>0</td><td>30</td><td>160</td></tr><tr><td>Small object</td><td>0</td><td>5</td><td>186</td></tr><tr><td>Scene Type</td><td>4</td><td>6</td><td>8</td></tr></table>

Table 1: Statistical Comparison of HSOD Datasets.

![](images/8e2a61aee4be783778b4ca08928c377fbeeb36d40b58590dec018ed45859fd3b.jpg)

<details>
<summary>text_image</summary>

MS
CS
HDR
CB
SO
</details>

Figure 2: Examples of pseudo-color images and corresponding ground truth from HSOD-BIT-V2.

testing. Pseudo-color images were generated for easier annotation, with ground truth labels assigned using Matlab's ImageLabeler toolbox. Examples are shown in Figure 2.

# 3.3 Statistics

We perform further rigorous statistical analysis on HSOD-BIT-V2 to validate its scientific integrity, providing a solid foundation for the splitting of training and testing sets.

Challenge Attributes Statistics. To evaluate HSOD method thoroughly, we categorize challenges into five attributes: Complex Background (CB), Color Similarity (CS), High Dynamic Range (HDR), Small Object (SO), and Material Similarity (MS). MS is particularly difficult for HSI-based methods. Our dataset contains a substantial proportion of challenging data and notably numerous tiny objects. Figure 3 (b) shows the distribution and sizes of these attributes, which are balanced to effectively address real-world challenges.

Foreground Scale Analysis. Our study of foreground scale shows a uniform distribution, with small objects (less than $1\%$ of the image) comprising $38.4\%$ of the dataset, as shown in Figure 3 (c). The diverse object scales improve the HSOD model's performance, making it more versatile and effective in detecting salient objects of varying sizes, which is crucial for real-world scenarios with objects at different distances.

Centroid Spatial Distribution. Figure 3 (d) shows the spatial distribution of object centroids, represented by centroid probabilities across the dataset. Red regions denote dense clusters of centroid positions, with a uniform outward distribution from the center and higher concentration near the center. This pattern conforms to the natural inclination of

![](images/60ab44fa224b8b92fc072d7c48e6fe05dc51abdabfa13f5b6c3c47d399f02559.jpg)

<details>
<summary>pie</summary>

Scene Category
| Scene Category | Percentage (%) |
| :--- | :--- |
| lawn | 18 |
| fallen leaves | 5 |
| path | 21 |
| wall peace | 19 |
| playground | 10 |
| sky background | 9 |
| snowfield | 2 |
| other | 16 |
</details>

(a) Distribution of Background Types

![](images/c0a0c528875b2eae54f6d075483c505229b2f8b3d216276c3580d03fc80fbe82.jpg)

<details>
<summary>bar</summary>

| Proportion of Object Pixels | Number of Samples |
| --------------------------- | ----------------- |
| 0.00                        | 200               |
| 0.05                        | 40                |
| 0.10                        | 30                |
| 0.15                        | 25                |
| 0.20                        | 20                |
| 0.25                        | 15                |
| 0.30                        | 10                |
| 0.35                        | 5                 |
| 0.40                        | 5                 |
| 0.45                        | 5                 |
| 0.50                        | 5                 |
</details>

(c) Distribution of Object Scales

![](images/6ebf83aadc3ecd6f9adb152d57cc53616f0081719234c7a5b2ed304662aaec03.jpg)

<details>
<summary>bar</summary>

| Category | train | test |
|---|---|---|
| SO | 155 | 35 |
| HDR | 110 | 20 |
| MS | 15 | 5 |
| CS | 125 | 30 |
| CB | 225 | 55 |
| Large Obj. | 105 | 20 |
| Medium Obj. | 210 | 40 |
| Small Obj. | 185 | 30 |
</details>

(b) Attribute Distribution and Size Partition

![](images/08ac8819c3d27a698fd4d2182113df40d6b27c26d056710989589edf75d83203.jpg)

<details>
<summary>scatter</summary>

| X of the Center of the Salient Obj. | Y of the Center of the Salient Obj. |
| ---------------------------------- | ---------------------------------- |
| (various values)                  | (various values)                 |
</details>

(d) Distribution of Object Centroids   
Figure 3: Diagram of HSOD-BIT-V2 statistics.

human vision to prioritize prominent objects in the central field of view, while allocating less attention to the periphery.

# 4 Method

Given a HSI $I \in R^{H \times W \times C}$ , HSOD aims to generate a saliency map $Y \in R^{H \times W \times 1}$ , a binary image highlighting salient object. Hyper-HRNet interpolates and reconstructs HSI to preserve crucial information by capturing spectral self-similarity. It also retains high-resolution flow and integrates intact global information and detailed object saliency to enhance decoding results, depicted in Figure 4.

# 4.1 Hyperspectral Attention Reconstruction

Hyper-HRNet utilizes the Hyperspectral Attention Reconstruction (HAR) to downsample the channels from C to $C'$ by interpolating every 4 channels into 1 from I, then reconstructing the dimension-reduced HSI, depicted in Figure 4. HAR first applies a $3 \times 3$ convolution for embedding, then uses cascaded Hybrid Perceptual Spectral Attention Reconstruction Blocks (HPSAB). These blocks incorporate Transformer-like architecture with Hybrid Perceptual Spectral Attention (HPSA) which combines Transformer-based Multi-head Spectral-wise Self-Attention (MSSA) and CNN-based Adaptive Spectral Attention Mechanism (ASAM) to capture spectral-wise self-similar relationships. HAR effectively reconstructs the interpolated data, addressing spectral redundancy while preserving essential information.

Multi-head Spectral-wise Self-Attention (MSSA). MSSA enhances the self-attention mechanism to capture spectral-wise contextual relationships. Given the input $F_{in} \in R^{H \times W \times C'}$ obtained through interpolation and embedding from I, it is reshaped into tokens $X \in R^{HW \times C'}$ and linearly projected into Q, K, $V \in R^{HW \times C'}$ . Then, Q, K, and V are split into N parts along the spectral channel dimension: $Q = [Q_1, \cdots, Q_N]$ , $K = [K_1, \cdots, K_N]$ , and $V = [V_1, \cdots, V_N]$ . Then, MSSA treats each spectral rep-

![](images/37f69b9785a7b06b3b6f2564dd9a00915bdf1a4ffb4e2de2881c31a8ed769ead.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Interpolate"] --> B["Embedding"]
    B --> C["HPSAB"]
    C --> D["Layer Norm"]
    D --> E["HPSA"]
    E --> F["+"]
    F --> G["Layer Norm"]
    G --> H["FFN"]
    H --> I["+"]
    I --> J["C×L"]
    J --> K["3×3Conv"]
    K --> L["3×3Conv"]
    L --> M["HRNet"]
    M --> N["F1 ∈ ℝ^C×H×W"]
    M --> O["F2 ∈ ℝ^4C×H/4×W/4"]
    M --> P["F3 ∈ ℝ^16C×H/4×W/4"]
    M --> Q["F4 ∈ ℝ^64C×H/8×W/8"]
    N --> R["CMFI"]
    O --> S["CMFI"]
    P --> T["CMFI"]
    Q --> U["CMFI"]
    R --> V["D1"]
    S --> W["D2"]
    T --> X["D3"]
    U --> Y["TAW"]
    V --> Z["TAW"]
    W --> AA["TAW"]
    X --> AB["TAW"]
    Y --> AC["TAW"]
    Z --> AD["TAW"]
    AA --> AE["TAW"]
    AB --> AF["TAW"]
    AC --> AG["TAW"]
    AD --> AH["TAW"]
    AE --> AI["TAW"]
    AF --> AJ["TAW"]
    AG --> AK["Gm"]
    AH --> AL["Gm"]
    AI --> AM["Global Ternary Perception Decoder"]
    AJ --> AM
    AK --> AM
    AL --> AM
    AM --> AN["GAFA"]
    AN --> AO["Gm"]
    AO --> AP["Global Ternary Perception Decoder"]

    subgraph HYPERSPECTRAL Attention Reconstruction
        B
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
        U
        V
        W
        X
        Y
        Z
        AA
        AB
        AC
        AD
        AE
        AF
        AG
        AH
        AI
        AJ
        AK
    end

    subgraph CMFI
            L
            M
            N
            O
            P
            Q
            R
            S
            T
            U
            V
            W
            X
            Y
            Z
            AA
            AB
            AC
            AD
            AE
            AF
            AG
            AH
            AI
            AJ
            AK
    end

    subgraph GAFA
            M1["F1 → PU×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M2["f1 → PU×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M3["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M4["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M5["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M6["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M7["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M8["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M9["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M10["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M11["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M12["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M13["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M14["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M15["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M16["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M17["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M18["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M19["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M20["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M21["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M22["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M23["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M24["f1 → PU ×4 → ffuse → ffuse → ffuse → TF → f_ref"] --> M25["f1 ← Element-wise Summation"] --> M6["Sigmoid Function"] --> N["Saliency Map"] --> O["Saliency Detection Loss"] --> P["Saliency Detection Loss"]
    style HYPERSPECTRAL Attention Reconstruction fill:#f9f,stroke:#333,stroke-width:2px,color:#fff
    style CMFI fill:#ccf,stroke:#333,stroke-width:2px,color:#fff
    style GAFA fill:#cfc,stroke:#333,stroke-width:2px,color:#fff
```
</details>

Figure 4: The overall architecture of the proposed Hyper-HRNet is shown in the top part of the figure. The bottom part illustrates the detailed elucidation of composition within HPSAB and the blocks within GTPD.

resentation as a token and computes self-attention $SSA_{j}$ :

$$
\boldsymbol {A} _ {j} = \operatorname{softmax} (\boldsymbol {\sigma} _ {j} \boldsymbol {K} _ {j} ^ {T} \boldsymbol {Q} _ {j}), \quad \boldsymbol {S S A} _ {j} = \boldsymbol {V} _ {j} \boldsymbol {A} _ {j}, \tag {1}
$$

where $K_{j}^{T}$ denotes the transpose of $K_{j}$ . Due to the significant variation in spectral density across wavelengths, we reweight the matrix multiplication $K_{j}^{T}Q_{j}$ within $A_{j}$ using a learnable parameter $\sigma_{j}\in R^{1}$ to adapt to the spectral density variation. The outputs from N heads are then concatenated, linearly projected, and position embedding to generate the output feature maps $M\in R^{H\times W\times C^{\prime}}$ :

$$
\boldsymbol {M} = (\sum_ {j = 1} ^ {\mathrm{N}} (\boldsymbol {S} \boldsymbol {S} \boldsymbol {A} _ {j})) \boldsymbol {W} + \boldsymbol {f} _ {p} (\boldsymbol {V}), \tag {2}
$$

where $\pmb{W} \in \mathbb{R}^{C' \times C'}$ is a learnable parameter, $\pmb{f}_p(\cdot)$ is position embedding function including two depth-wise $3 \times 3$ convolutions, GELU activation, and reshape operation.

Adaptive Spectral Attention Mechanism (ASAM). To adaptively extract spectral details, ASAM utilizes the high-frequency saliency feature from max-pooling and the low-frequency degree feature from average-pooling (2022). The input feature $F_{in}$ is processed through two branches along spectral dimension: max-pooling for discriminative object features $F_{max}$ , and average-pooling for holistic object features $F_{avg}$ . Because varying emphasis across different stages, learnable parameters $\alpha$ and $\beta$ are used to weigh $F_{avg}$ and $F_{max}$ . The weighted tensors are combined to produce the adaptive spectral feature $F_{add} \in R^{1 \times 1 \times C'}$ :

$$
\boldsymbol {F} _ {\text { add }} = \frac {1}{2} (\boldsymbol {F} _ {\text { avg }} \oplus \boldsymbol {F} _ {\text { max }}) \oplus \boldsymbol {\alpha} \otimes \boldsymbol {F} _ {\text { avg }} \oplus \boldsymbol {\beta} \otimes \boldsymbol {F} _ {\text { max }}, \tag {3}
$$

where $\otimes$ means element-wise multiplication, $\oplus$ means element-wise summation. After applying the Sigmoid activation and performing element-wise multiplication with $F_{in}$ , the output feature maps S are obtained as follows:

$$
\boldsymbol {S} = \boldsymbol {F} _ {i n} \times \boldsymbol {\delta} (\boldsymbol {f} _ {k} (\boldsymbol {F} _ {a d d})), \tag {4}
$$

where $\delta$ stands for Sigmoid activation function, $f_{k}$ stands for 1D with an adaptive kernel size of k (2022). The final Spectral-wise Attention feature $\pmb{H} \in \mathbb{R}^{\mathrm{H} \times \mathrm{W} \times \mathrm{C}'}$ is obtained by element-wise summing $S$ and $M$ .

Finally, the reconstructed image is restored to $C$ channel using a $3 \times 3$ convolution. These processes are supervised by $I$ to maximize the preservation of spectral information.

# 4.2 Global Ternary Perception Decoder

Hyper-HRNet enhances object contours and decoding results through Global Ternary Perception Decoder (GTPD). Fusing cross-scale high-resolution flow from HRNet backbone (2020), GTPD supplements intact global information and ternary contour-aware saliency for precise decoding and accurate saliency predictions, as shown in Figure 4.

Cross-level Multi-scale Feature Interaction (CMFI). To discern scale and positional changes of objects in multi-scale features and highlight salient regions, GTPD uses CMFI based on the Split-Transform-Merge strategy (2017). The multi-scale features $F = \{F_i \in R^{C_i \times H_i \times W_i} | i = 1, 2, 3, 4\}$ from HRNet are split along the channel dimension into two parts for CMFI: $\{F_i^1, F_i^2\}$ . $F_i^1$ captures local context at small scales and expands the receptive field through GCN (2017), while $F_i^2$ facilitates cross-scale interaction using its rich shallow-level details and semantic information in deep-level features $F_{i+1}^2$ . They are treated as follows:

$$
\boldsymbol {D} _ {i} ^ {1} = \boldsymbol {f} _ {G C N} (\boldsymbol {f} _ {\text { main }} (\boldsymbol {F} _ {i} ^ {1}) + \boldsymbol {F} _ {i} ^ {1}), \tag {5}
$$

$$
\boldsymbol {D} _ {i} ^ {2} = \operatorname{Cat} (\boldsymbol {f} _ {s u b} (\boldsymbol {F} _ {i} ^ {2}), \boldsymbol {F} _ {i + 1} ^ {2}), \tag {6}
$$

where $\boldsymbol{f}_{main}(\cdot)$ performs downsampling via average pooling and $3 \times 3$ convolution, and corresponding upsampling. $\boldsymbol{f}_{GCN}(\cdot)$ denotes GCN. $\boldsymbol{f}_{sub}(\cdot)$ includes one $1 \times 1$ and two $3 \times 3$ convolutions. $\operatorname{Cat}(\cdot)$ denotes concatenation. Finally, the decoded output features $D_{i}$ are obtained as:

$$
\boldsymbol {D} _ {i} = \operatorname{Cat} (\boldsymbol {D} _ {i} ^ {1}, \boldsymbol {D} _ {i} ^ {2}). \tag {7}
$$

Global Attention Feature Aggregator (GAFA). To offset the loss of fine details and and limitations in capturing long-range dependencies in CNN-based multi-scale decoding, GTPD uses GAFA to incorporate intact global information into the decoding process. GAFA utilizes intact contextual features to generate global saliency via Transformer under supervision. Specifically, $F_{i}$ employs Pixel Shuffle into spatial dimensions of $20 \times 20$ as $f_{i}$ which is then concatenated. After fusing $f_{i}$ , GAFA generates the global saliency map $G_{m}$ via Transformer and linear mapping as follows:

$$
\boldsymbol {G} _ {m} = \delta (\boldsymbol {f} _ {r e f} (\boldsymbol {T F} (\sum_ {i = 1} ^ {n} \boldsymbol {f} _ {f u s e} (\boldsymbol {f} _ {i}))), \tag {8}
$$

where $\boldsymbol{f}_{fuse}(\cdot)$ applies $3 \times 3$ convolution, Batch Normalization, ReLU activation, and reshape operation which transforms the shape from $R^{C_{i} \times H_{i} \times W_{i}}$ to $R^{H_{i}W_{i} \times C_{i}}$ . $f_{ref}$ is realized via MLP. TF involves the original self-attention as:

$$
\boldsymbol {T} \boldsymbol {F} (\boldsymbol {Q}, \boldsymbol {K}, \boldsymbol {V}) = \operatorname{softmax} \left(\frac {\boldsymbol {Q} \boldsymbol {K} ^ {T}}{\sqrt {d _ {h}}}\right) \boldsymbol {V}. \tag {9}
$$

To obtain the ground truth $GT_{m}$ for $G_{m}$ , ground truth map employs Pixel Unshuffle into spatial dimensions of $20 \times 20$ , preserving complete information. $GT_{m}$ is obtained as:

$$
\boldsymbol {G} \boldsymbol {T} _ {m} = \max \left(\operatorname{PS} \left(\boldsymbol {G} _ {m}\right)\right), \tag {10}
$$

where PS(·) denotes the Pixel Unshuffle operation, and maxc(·) signifies maximum along the channels.

Ternary-Aware Weight (TAW). To enhance object contour delineation, GTPD leverages TAW to generate ternary-aware saliency layer by layer, focusing on uncertain region. Saliency prediction is categorized into three regions: object (saliency around 1), background (saliency around 0), and uncertain region(contours between object and background). The uncertain region is crucial in challenging scenarios. TAW first utilizes the decoded features $D_{i}$ to produce the saliency prediction $P_{i}$ as:

$$
\boldsymbol {P} _ {i} = \delta (\boldsymbol {f} _ {p r e} (\boldsymbol {D} _ {i})), \tag {11}
$$

where $f_{pre}(\cdot)$ uses $3 \times 3$ convolution. The saliency prediction from the lower layer generates ternary contour-aware saliency, producing a Trimap $T_{i}$ as weights for subsequent decoding stages. $T_{i}$ labels each pixel: 1 for object, 0 for background, and 2 for uncertain regions. The TAW process can be summarized as follows:

$$
\boldsymbol {T} _ {i} = \operatorname{softmax} (\boldsymbol {f} _ {w} ((\boldsymbol {D} _ {i} \otimes \boldsymbol {P} _ {i}) + \boldsymbol {D} _ {i})), \tag {12}
$$

where $\pmb{f}_w(\cdot)$ function employs a $3 \times 3$ convolution.

Ultimately, hierarchical decoding is optimized using the supplementary features $G_{m}$ and $T_{i}$ , with dense supervision. The output from the topmost layer is the final saliency map.

# 4.3 Loss Function

Hyper-HRNet is trained with a hybrid loss function that comprises data reconstruction loss $L_{s}$ , saliency detection loss $L_{sod}$ , and global guidance loss $L_{g}$ , defined as follows:

$$
\boldsymbol {L} _ {m} = \boldsymbol {L} _ {s} + \boldsymbol {L} _ {s o d} + \boldsymbol {L} _ {g}. \tag {13}
$$

$L_{s}$ evaluates the discrepancy between the restored image and the original data. $L_{sod}$ measures the deviation between the predicted saliency map and the ground truth. $L_{g}$ supervises global saliency map by its corresponding ground truth.

![](images/e17462535b517c3d34975b2dcc42e66116b6320ba53d748ed5186d6049cc91d8.jpg)

<details>
<summary>text_image</summary>

HSOD-BIT
HSOD-BIT-V2
Pseudo-Color Ground Truth U2Net DMSSN SMN Hyper-HRNet (Ours)
</details>

Figure 5: Qualitative results on HSOD-BIT-V2 and HSOD-BIT datasets. Hyper-HRNet has best detection performance.

# 5 Experiment

We evaluate Hyper-HRNet on the HSOD-BIT-V2, HSOD-BIT, and HS-SOD datasets. To ensure fairness, all comparison methods are independently trained and tested on the same conditions across three datasets. Further experiments and details are available in the supplementary material.

# 5.1 Results On HSOD-BIT-V2 and HSOD-BIT

Quantitative Analysis. Table 2 provides a quantitative comparison of Hyper-HRNet with existing methods on HSOD-BIT-V2 and HSOD-BIT datasets. The results show that our method outperforms both RGB- and HSI-based methods across all metrics. Notably, it surpasses the soTA HSI-based method SMN and RGB-based method U2Net by 0.051 and 0.266 in REC, 0.098 and 0.207 in CC, and 0.073 and 0.040 in AUC on HSOD-BIT-V2. These gains highlight the effectiveness of our approach. While RGB-based methods perform well on simpler samples, they struggle in more challenging scenarios, emphasizing the limitations of converting HSI to pseudo-color images for RGB-based SOD methods.

Furthermore, the analysis reveals that performance on HSOD-BIT-V2 is significantly lower than on HSOD-BIT, highlighting the increased challenges presented by our dataset. Traditional HSI-based methods exhibit variable performance, likely due to the enhanced denoising in HSOD-BIT-V2, which improves spectral quality.

Qualitative Analysis. Figure 5 presents visual comparisons of saliency maps generated by Hyper-HRNet on HSOD-BIT-V2 and HSOD-BIT datasets, alongside several existing HSOD methods. Hyper-HRNet outperforms other methods by leveraging spectral information to minimize background noise and enhance object localization in challenging scenes. Moreover, by preserving essential spectral details and integrating global and key region information during decoding, Hyper-HRNet produces sharper contours.

Attribute-based Evaluations. We evaluate our approach on five challenging attributes of HSOD-BIT-V2, as detailed in Table 3. Our method outperforms both RGB- and HSI-based methods across most attributes. In MS attributes, where foreground and background consist of chemically similar mate-

<table><tr><td>Dataset</td><td colspan="6">HSOD-BIT-V2</td><td colspan="6">HSOD-BIT</td><td colspan="2"></td></tr><tr><td>Metrics</td><td colspan="14"> $MAE \downarrow PRE \uparrow REC \uparrow avgF_1 \uparrow AUC \uparrow CC \uparrow MAE \downarrow PRE \uparrow REC \uparrow avgF_1 \uparrow AUC \uparrow CC \uparrow$ </td></tr><tr><td colspan="15">RGB-based SOD Methods</td></tr><tr><td>Itti (1998)</td><td>0.230</td><td>0.280</td><td>0.419</td><td>0.240</td><td>0.803</td><td>0.277</td><td>0.252</td><td>0.335</td><td>0.399</td><td>0.341</td><td>0.793</td><td>0.351</td><td>-</td><td>-</td></tr><tr><td>BASNet (2019b)</td><td>0.049</td><td>0.638</td><td>0.634</td><td>0.553</td><td>0.876</td><td>0.618</td><td>0.071</td><td>0.741</td><td>0.742</td><td>0.695</td><td>0.901</td><td>0.703</td><td>87.06 M</td><td>127.56 G</td></tr><tr><td>U2Net (2020)</td><td>0.046</td><td>0.649</td><td>0.597</td><td>0.513</td><td>0.948</td><td>0.621</td><td>0.062</td><td>0.814</td><td>0.683</td><td>0.739</td><td>0.951</td><td>0.746</td><td>44.01 M</td><td>47.65 G</td></tr><tr><td>SelfReformer (2023)</td><td>0.048</td><td>0.581</td><td>0.498</td><td>0.528</td><td>0.827</td><td>0.530</td><td>0.068</td><td>0.766</td><td>0.628</td><td>0.704</td><td>0.884</td><td>0.676</td><td>90.70 M</td><td>128.26 G</td></tr><tr><td colspan="15">HSI-based HSOD Methods</td></tr><tr><td>SAD (2013)</td><td>0.177</td><td>0.335</td><td>0.398</td><td>0.253</td><td>0.863</td><td>0.331</td><td>0.209</td><td>0.395</td><td>0.350</td><td>0.364</td><td>0.822</td><td>0.395</td><td>-</td><td>-</td></tr><tr><td>SED (2013)</td><td>0.106</td><td>0.359</td><td>0.178</td><td>0.237</td><td>0.781</td><td>0.264</td><td>0.138</td><td>0.415</td><td>0.131</td><td>0.345</td><td>0.746</td><td>0.301</td><td>-</td><td>-</td></tr><tr><td>SG (2013)</td><td>0.168</td><td>0.342</td><td>0.350</td><td>0.243</td><td>0.823</td><td>0.298</td><td>0.188</td><td>0.401</td><td>0.278</td><td>0.351</td><td>0.782</td><td>0.363</td><td>-</td><td>-</td></tr><tr><td>SED-SAD (2013)</td><td>0.180</td><td>0.345</td><td>0.367</td><td>0.265</td><td>0.865</td><td>0.333</td><td>0.208</td><td>0.400</td><td>0.317</td><td>0.381</td><td>0.828</td><td>0.407</td><td>-</td><td>-</td></tr><tr><td>SED-SG (2013)</td><td>0.165</td><td>0.332</td><td>0.302</td><td>0.243</td><td>0.820</td><td>0.287</td><td>0.189</td><td>0.391</td><td>0.247</td><td>0.381</td><td>0.776</td><td>0.351</td><td>-</td><td>-</td></tr><tr><td>SUDF (2019)</td><td>0.166</td><td>0.375</td><td>0.614</td><td>0.362</td><td>0.873</td><td>0.412</td><td>0.203</td><td>0.545</td><td>0.619</td><td>0.528</td><td>0.910</td><td>0.582</td><td>0.10 M</td><td>82.90 G</td></tr><tr><td>SMN (2023)</td><td>0.039</td><td>0.607</td><td>0.713</td><td>0.575</td><td>0.915</td><td>0.639</td><td>0.034</td><td>0.837</td><td>0.868</td><td>0.751</td><td>0.963</td><td>0.846</td><td>10.23 M</td><td>14.76 G</td></tr><tr><td>DMSSN (2024)</td><td>0.072</td><td>0.635</td><td>0.602</td><td>0.548</td><td>0.830</td><td>0.553</td><td>0.086</td><td>0.663</td><td>0.637</td><td>0.637</td><td>0.852</td><td>0.625</td><td>1.76 M</td><td>10.89 G</td></tr><tr><td>Hyper-HRNet</td><td>0.028</td><td>0.653</td><td>0.764</td><td>0.589</td><td>0.988</td><td>0.737</td><td>0.020</td><td>0.854</td><td>0.891</td><td>0.795</td><td>0.996</td><td>0.916</td><td>29.57 M</td><td>18.96 G</td></tr><tr><td>Hyper-HRNet-Lite</td><td>0.046</td><td>0.590</td><td>0.689</td><td>0.550</td><td>0.940</td><td>0.641</td><td>0.026</td><td>0.845</td><td>0.878</td><td>0.770</td><td>0.987</td><td>0.907</td><td>7.24 M</td><td>7.85 G</td></tr></table>

Table 2: Quantitative Results on HSOD-BIT and HSOD-BIT-V2 Datasets.

<table><tr><td>Challenges</td><td colspan="2">CB</td><td colspan="2">SC</td><td colspan="2">HDR</td><td colspan="2">SO</td><td colspan="2">SM</td></tr><tr><td>Metrics</td><td colspan="2">MAE↓AUC↑</td><td colspan="2">MAE↓AUC↑</td><td colspan="2">MAE↓AUC↑</td><td colspan="2">MAE↓AUC↑</td><td colspan="2">MAE↓AUC↑</td></tr><tr><td>U2Net</td><td>0.054</td><td>0.949</td><td>0.058</td><td>0.967</td><td>0.047</td><td>0.919</td><td>0.012</td><td>0.964</td><td>0.020</td><td>0.987</td></tr><tr><td>SelfReformer</td><td>0.058</td><td>0.841</td><td>0.066</td><td>0.753</td><td>0.049</td><td>0.818</td><td>0.019</td><td>0.720</td><td>0.027</td><td>0.837</td></tr><tr><td>SMN</td><td>0.041</td><td>0.897</td><td>0.030</td><td>0.851</td><td>0.057</td><td>0.923</td><td>0.036</td><td>0.809</td><td>0.042</td><td>0.857</td></tr><tr><td>DMSSN</td><td>0.089</td><td>0.774</td><td>0.068</td><td>0.782</td><td>0.050</td><td>0.846</td><td>0.060</td><td>0.669</td><td>0.095</td><td>0.502</td></tr><tr><td>Hyper-HRNet</td><td>0.029</td><td>0.979</td><td>0.014</td><td>0.973</td><td>0.018</td><td>0.989</td><td>0.008</td><td>0.979</td><td>0.030</td><td>0.994</td></tr></table>

Table 3: Attribute Evaluations on HSOD-BIT-V2 Dataset.

<table><tr><td>Methods</td><td> $MAE \downarrow$ </td><td> $avgF_1 \uparrow$ </td><td> $AUC \uparrow$ </td><td> $CC \uparrow$ </td></tr><tr><td>Itti (1998)</td><td>0.246</td><td>0.237</td><td>0.783</td><td>0.268</td></tr><tr><td>SAD (2013)</td><td>0.236</td><td>0.235</td><td>0.834</td><td>0.295</td></tr><tr><td>SED (2013)</td><td>0.185</td><td>0.236</td><td>0.817</td><td>0.277</td></tr><tr><td>SG (2013)</td><td>0.218</td><td>0.233</td><td>0.827</td><td>0.296</td></tr><tr><td>SED-SAD (2013)</td><td>0.209</td><td>0.250</td><td>0.830</td><td>0.286</td></tr><tr><td>SED-SG (2013)</td><td>0.188</td><td>0.240</td><td>0.826</td><td>0.287</td></tr><tr><td>SUDF (2019)</td><td>0.242</td><td>0.256</td><td>0.723</td><td>0.250</td></tr><tr><td>SMN (2023)</td><td>0.069</td><td>0.658</td><td>0.916</td><td>0.718</td></tr><tr><td>DMSSN (2024)</td><td>0.068</td><td>0.564</td><td>0.937</td><td>0.703</td></tr><tr><td>Hyper-HRNet</td><td>0.056</td><td>0.770</td><td>0.953</td><td>0.810</td></tr></table>

Table 4: Quantitative Results on the HS-SOD Dataset.

rials and closely matching colors, extracting discriminative features from HSIs is more challenging than from RGB images. Nonetheless, Hyper-HRNet consistently exceeds other HSI-based methods and nearly matches the performance of the soTA RGB-based methods U2Net, improving AUC by 0.005 and falling short of MAE by only 0.010. Specifically, Hyper-HRNet outperforms the soTA HSI-based approach SMN, with a 0.137 improvement in AUC and a 0.012 reduction in MAE for MS attributes. Additionally, RGB-based methods perform poorly on other challenging attributes, underscoring the limitations of applying SOD methods to pseudo-color images derived from HSIs.

Efficiency Analysis. Table 2 presents computational complexity of our method, excluding traditional approaches. Unlike RGB-based methods, which often rely on complex net-

![](images/8a324110a92959c0c1406f905e4f0af8c0079830929bcd40282f13336fe2f840.jpg)

<details>
<summary>natural_image</summary>

Two-panel scientific image showing a green textured surface with arrows pointing to circular features, and a blue abstract shape on the right (no text or symbols)
</details>

![](images/20a2d64fbc9c0fb1b2ba68e1645562f8716a40554692073aa147bf8f9b494d0c.jpg)

<details>
<summary>natural_image</summary>

Three-panel scientific image showing a textured surface with highlighted regions and a color-coded horizontal bar (no text or symbols)
</details>

Figure 6: Visualization of Hybrid Perceptual Spectral Attention features by HARM block. Attention features effectively preserve the salient information of the salient objects.

work structures and neglect spectral dimensions, our method significantly reduces both parameters and FLOPs. However, the complexity of HRNet results in a larger model size. To improve efficiency, we introduce Hyper-HRNet-Lite with the lightweight Lite-HRNet backbone (2021). Among HSI-based methods, SUDF uses CNNs for only feature extraction followed by manifold learning and superpixel clustering, leading to low parameters but high FLOPs, while DMSSN reduces parameters via knowledge distillation. Direct parameter comparisons with them are less meaningful. Hyper-HRNet-Lite minimizes FLOPs and achieves comparable performance to the soTA method SMN with fewer parameters, balancing efficiency, speed, and efficacy.

Visualization of Spectral Attention Feature. Figure 6 shows the Spectral Attention features from our proposed HAR, along with the pseudo-color images and ground truth. These features emphasize the spectral characteristics of HSIs, enhancing the contrast between salient objects and backgrounds while preserving crucial spectral information.

# 5.2 Results on HS-SOD Dataset

Quantitative Analysis. compares Hyper-HRNet with existing HSI-based methods on the HS-SOD dataset, using con-

![](images/4c89047d54b7b93b08352c744325dcf63b8f3b0b282b2a004c01b259efc827c9.jpg)

Figure 7: Qualitative results on HS-SOD dataset. 

<table><tr><td>HAR</td><td>GTPD</td><td> $MAE \downarrow$ </td><td> $avgF_1 \uparrow$ </td><td> $AUC \uparrow$ </td><td> $CC \uparrow$ </td></tr><tr><td>√</td><td>✗</td><td>0.034</td><td>0.508</td><td>0.883</td><td>0.655</td></tr><tr><td>✗</td><td>√</td><td>0.030</td><td>0.534</td><td>0.890</td><td>0.692</td></tr><tr><td>√</td><td>√</td><td>0.028</td><td>0.589</td><td>0.988</td><td>0.737</td></tr></table>

Table 5: Ablation Study of Key Components.

sistent training configurations from previous works (2023; 2024), utilizing 48 data for training and the rest for testing. Hyper-HRNet outperforms DMSSN and SMN, improving AUC by 0.160 and 0.037, CC by 0.107 and 0.092, and reducing MAE by 0.012 and 0.013, respectively.

Qualitative Analysis. Figure 7 presents visual comparisons of Hyper-HRNet against other HSI-based methods on the HS-SOD dataset. While other methods struggle with blurry edges and recognition distortions, Hyper-HRNet leverages reconstructed hyperspectral information to produce saliency maps with clear and precise contours, demonstrating its superior performance in the HSOD task.

# 5.3 Ablation Study

We conducted ablation studies on our HSOD-BIT-V2.

Effect of key components. To validate the efficacy of each component within Hyper-HRNet, as detailed in Table 5, we conducted a comparative analysis using HRNet as the baseline. The results indicate substantial performance improvements with the separate integration of HAR and GTPD. Moreover, their combined integration achieves superior results, confirming the effectiveness of the two components.

Effect of HAR. To validate the effectiveness of minimizing spectral redundancy in HAR, as shown in Table 6, we conducted comparative experiments on dimensionality reduction methods using Hyper-HRNet without dimensionality reduction as the baseline. The results show significant performance gains with the integration of HAR. Specifically, HAR enhances AUC by 0.050 and CC by 0.183, while reducing MAE by 0.045 compared to the Convolution Layer.

To further validate the effectiveness of HPSA, as shown in Table 7, we individually removed its key components, labeled as w/o MSSA and w/o ASAM, and replaced HPSA with other self-attention mechanisms, including the conven-

<table><tr><td>Interpolate</td><td>PCA</td><td>Conv</td><td>HAR</td><td> $MAE \downarrow$ </td><td> $avgF_1 \uparrow$ </td><td> $AUC \uparrow$ </td><td> $CC \uparrow$ </td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>0.169</td><td>0.178</td><td>0.664</td><td>0.193</td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td><td>0.084</td><td>0.283</td><td>0.728</td><td>0.268</td></tr><tr><td>X</td><td>√</td><td>X</td><td>X</td><td>0.073</td><td>0.321</td><td>0.754</td><td>0.447</td></tr><tr><td>X</td><td>X</td><td>√</td><td>X</td><td>0.079</td><td>0.426</td><td>0.833</td><td>0.472</td></tr><tr><td>X</td><td>X</td><td>X</td><td>√</td><td>0.028</td><td>0.589</td><td>0.988</td><td>0.737</td></tr></table>

Table 6: Comparative Experiments between Different Dimensionality Reduction Methods and HAR. 

<table><tr><td>Method</td><td> $MAE \downarrow$ </td><td> $avgF_1 \uparrow$ </td><td> $AUC \uparrow$ </td><td> $CC \uparrow$ </td></tr><tr><td>w/o MSSA</td><td>0.088</td><td>0.496</td><td>0.832</td><td>0.545</td></tr><tr><td>w/o ASAM</td><td>0.095</td><td>0.478</td><td>0.822</td><td>0.539</td></tr><tr><td>ViT</td><td>0.121</td><td>0.385</td><td>0.805</td><td>0.516</td></tr><tr><td>MSST</td><td>0.114</td><td>0.467</td><td>0.825</td><td>0.536</td></tr><tr><td>Hyper-HRNet</td><td>0.028</td><td>0.589</td><td>0.988</td><td>0.737</td></tr></table>

Table 7: Ablation Study of HAR.

<table><tr><td>Method</td><td> $MAE \downarrow$ </td><td> $avgF_1 \uparrow$ </td><td> $AUC \uparrow$ </td><td> $CC \uparrow$ </td></tr><tr><td>w/o CMFI</td><td>0.039</td><td>0.523</td><td>0.880</td><td>0.616</td></tr><tr><td>w/o GAFA</td><td>0.044</td><td>0.522</td><td>0.899</td><td>0.617</td></tr><tr><td>w/o TAW</td><td>0.039</td><td>0.526</td><td>0.920</td><td>0.666</td></tr><tr><td>Hyper-HRNet</td><td>0.028</td><td>0.589</td><td>0.988</td><td>0.737</td></tr></table>

Table 8: Ablation Study of GTPD.

tional self-attention mechanism in ViT (2020) and spectral-spatial hybrid attention mechanism MSS (2024). Removing MSSA or ASAM consistently led to a performance decline, with both ViT's spatial attention and HSOD's MSSA underperforming relative to HPSA. These results confirm the effectiveness of HAR and HPSA.

Effect of GTPD. Table 8 validates the effectiveness of the key modules within GTPD: CMFI, GAFA, and TAW. We conducted experiments by removing these modules individually, labeled as w/o CMFI, w/o GAFA, and w/o TAW. Removing cross-level multi-scale feature interaction, global attention saliency map, or ternary contour-aware weights consistently led to decreased prediction performance. These findings highlight the importance of three critical modules.

# 6 Conclusion

In this work, we introduce HSOD-BIT-V2, the largest and most challenging HSOD dataset to date, and propose a novel high-resolution network, Hyper-HRNet. Our dataset includes eight natural backgrounds and five challenging attributes that highlight the spectral advantages of HSIs. Our method optimizes HSI utilization, reduces spectral dimensionality, and preserves key spectral information. Additionally, it also accurately locates object contours through capturing intact global information and ternary contour-aware saliency. While we believe this work will advance HSOD research and establishes a new benchmark for future research, opportunities for improvement remain. Future efforts will focus on expanding the dataset and advancing hyperspectral image dimensionality reduction and reconstruction.

# Acknowledgments

This work was financially supported by the National Key Scientific Instrument and Equipment Development Project of China (No. 61527802), the National Natural Science Foundation of China (No. 62101032), the Young Elite Scientist Sponsorship Program of China Association for Science and Technology (No. YESS20220448), and the Young Elite Scientist Sponsorship Program of Beijing Association for Science and Technology (No. BYESS2022167).

# References

Ahmadi, M.; Karimi, N.; and Samavi, S. 2021. Context-aware saliency detection for image retargeting using convolutional neural networks. Multimedia Tools and Applications, 80: 11917–11941.   
Borji, A.; Cheng, M.-M.; Hou, Q.; Jiang, H.; and Li, J. 2019. Salient object detection: A survey. Computational visual media, 5: 117–150.   
Chakrabarti, A.; and Zickler, T. 2011. Statistics of real-world hyperspectral images. In CVPR 2011, 193–200. IEEE.   
Chen, H.; Li, Y.; Deng, Y.; and Lin, G. 2021. CNN-based RGB-D salient object detection: Learn, select, and fuse. International Journal of Computer Vision, 129(7): 2076–2096.   
Chen, H.; Zhao, W.; Xu, T.; Shi, G.; Zhou, S.; Liu, P.; and Li, J. 2024. Spectral-Wise Implicit Neural Representation for Hyperspectral Image Reconstruction. IEEE Transactions on Circuits and Systems for Video Technology, 34(5): 3714–3727.   
Dosovitskiy, A. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.   
Foster, D. H.; Nascimento, S. M.; and Amano, K. 2004. Information limits on neural identification of colored surfaces in natural scenes. Visual neuroscience, 21(3): 331–336.   
Huang, H.; Cai, M.; Lin, L.; Zheng, J.; Mao, X.; Qian, X.; Peng, Z.; Zhou, J.; Iwamoto, Y.; Han, X.-H.; et al. 2021. Graph-based pyramid global context reasoning with a saliency-aware projection for covid-19 lung infections segmentation. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1050–1054. IEEE.   
Hughes, G. 2003. On the mean accuracy of statistical pattern recognizers. IEEE Transactions on Information Theory, 14(1): 55–63.   
İmamoğlu, N.; Ding, G.; Fang, Y.; Kanezaki, A.; Kouyama, T.; and Nakamura, R. 2019. Salient object detection on hyperspectral images using features learned from unsupervised segmentation task. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2192–2196. IEEE.   
Imamoglu, N.; Oishi, Y.; Zhang, X.; Ding, G.; Fang, Y.; Kouyama, T.; and Nakamura, R. 2018. Hyperspectral image dataset for benchmarking on salient object detection. In 2018 Tenth international conference on quality of multimedia experience (qoMEX), 1–3. IEEE.

Itti, L.; Koch, C.; and Niebur, E. 1998. A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on pattern analysis and machine intelligence, 20(11): 1254–1259.   
Le Moan, S.; Mansouri, A.; Hardeberg, J. Y.; and Voisin, Y. 2013. Saliency for spectral image analysis. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 6(6): 2472–2479.   
Li, G.; Fang, Q.; Zha, L.; Gao, X.; and Zheng, N. 2022. HAM: Hybrid attention module in deep convolutional neural networks for image classification. Pattern Recognition, 129: 108785.   
Liang, J.; Zhou, J.; Bai, X.; and Qian, Y. 2013. Salient object detection in hyperspectral imagery. In 2013 IEEE International conference on image processing, 2393–2397. IEEE.   
Liu, P.; Xu, T.; Chen, H.; Zhou, S.; Qin, H.; and Li, J. 2023. Spectrum-driven Mixed-frequency Network for Hyperspectral Salient Object Detection. IEEE Transactions on Multimedia.   
Nascimento, S. M.; Ferreira, F. P.; and Foster, D. H. 2002. Statistics of spatial cone-excitation ratios in natural scenes. JOSA A, 19(8): 1484–1490.   
Peng, C.; Zhang, X.; Yu, G.; Luo, G.; and Sun, J. 2017. Large kernel matters—improve semantic segmentation by global convolutional network. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4353–4361.   
Qin, H.; Xu, T.; Liu, P.; Xu, J.; and Li, J. 2024. DMSSN: Distilled Mixed Spectral-Spatial Network for Hyperspectral Salient Object Detection. IEEE Transactions on Geoscience and Remote Sensing.   
Qin, X.; Zhang, Z.; Huang, C.; Dehghan, M.; Zaiane, O. R.; and Jagersand, M. 2020. U2-Net: Going deeper with nested U-structure for salient object detection. Pattern recognition, 106: 107404.   
Qin, X.; Zhang, Z.; Huang, C.; Gao, C.; Dehghan, M.; and Jagersand, M. 2019a. Basnet: Boundary-aware salient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 7479–7489.   
Qin, X.; Zhang, Z.; Huang, C.; Gao, C.; Dehghan, M.; and Jagersand, M. 2019b. Basnet: Boundary-aware salient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 7479–7489.   
Tang, L.; Li, B.; Zhong, Y.; Ding, S.; and Song, M. 2021. Disentangled high quality salient object detection. In Proceedings of the IEEE/CVF international conference on computer vision, 3580–3590.   
Wang, J.; Sun, K.; Cheng, T.; Jiang, B.; Deng, C.; Zhao, Y.; Liu, D.; Mu, Y.; Tan, M.; Wang, X.; et al. 2020. Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10): 3349–3364.   
Wang, Z.; Chen, H.; Li, J.; Xu, T.; Zhao, Z.; Duan, Z.; Gao, S.; and Lin, X. 2024. Opto-intelligence spectrometer using diffractive neural networks. Nanophotonics, 13(20): 3883–3893.

Wu, Z.; Su, H.; Tao, X.; Han, L.; Paoletti, M. E.; Haut, J. M.; Plaza, J.; and Plaza, A. 2022. Hyperspectral anomaly detection with relaxed collaborative representation. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–17.   
Xie, S.; Girshick, R.; Dollár, P.; Tu, Z.; and He, K. 2017. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1492–1500.   
Yu, C.; Xiao, B.; Gao, C.; Yuan, L.; Zhang, L.; Sang, N.; and Wang, J. 2021. Lite-HRNet: A Lightweight High-Resolution Network. In CVPR.   
Yun, Y. K.; and Lin, W. 2023. Towards a complete and detail-preserved salient object detection. IEEE Transactions on Multimedia.   
Zhang, L.; Zhang, Y.; Yan, H.; Gao, Y.; and Wei, W. 2018. Salient object detection in hyperspectral imagery using multi-scale spectral-spatial gradient. Neurocomputing, 291: 215–225.   
Zhao, R.; Ouyang, W.; Li, H.; and Wang, X. 2015. Saliency detection by multi-context deep learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1265–1274.