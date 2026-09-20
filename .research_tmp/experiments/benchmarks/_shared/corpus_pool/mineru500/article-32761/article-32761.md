# Ultra-High Resolution Segmentation via Boundary-Enhanced Patch-Merging Transformer

Haopeng Sun $^{1,2}$ \*, Yingwei Zhang $^{1,2}$ \*, Lumin Xu $^{4}$ , Sheng Jin $^{5,6}$ , Yiqiang Chen $^{1,2,3}$ †

$^{1}$ Institute of Computing Technology, Chinese Academy of Sciences

$^{2}$ University of Chinese Academy of Sciences

$^{3}$ Peng Cheng Laboratory $^{4}$ The Chinese University of Hong Kong

$^{5}$ The University of Hong Kong $^{6}$ SenseTime Research and Tetras.AI

sunhaopeng22s@ict.ac.cn, zhangyingwei@ict.ac.cn, luminxu@link.cuhk.edu.hk, jinsheng@tetras.ai, yqchen@ict.ac.cn

# Abstract

Segmentation of ultra-high resolution (UHR) images is a critical task with numerous applications, yet it poses significant challenges due to high spatial resolution and rich fine details. Recent approaches adopt a dual-branch architecture, where a global branch learns long-range contextual information and a local branch captures fine details. However, they struggle to handle the conflict between global and local information while adding significant extra computational cost. Inspired by the human visual system's ability to rapidly orient attention to important areas with fine details and filter out irrelevant information, we propose a novel UHR segmentation method called Boundary-enhanced Patch-merging Transformer (BPT). BPT consists of two key components: (1) Patch-Merging Transformer (PMT) for dynamically allocating tokens to informative regions to acquire global and local representations, and (2) Boundary-Enhanced Module (BEM) that leverages boundary information to enrich fine details. Extensive experiments on multiple UHR image segmentation benchmarks demonstrate that our BPT outperforms previous state-of-the-art methods without introducing extra computational overhead. Codes will be released to facilitate research.

# Introduction

With the advancement of remote sensing technology, the acquisition of abundant ultra-high resolution (UHR) geospatial images has become possible (Liu et al. 2023; Guan et al. 2024a,b; Li et al. 2024e,d). Semantic segmentation of UHR geospatial images has opened new horizons in computer vision, playing an increasingly important role in earth sciences and urban applications such as disaster control, environmental monitoring, land resource management, conservation, and urban planning (Chen et al. 2019; Ji, Zhao, and Lu 2023; Ji et al. 2023; Zhu et al. 2024b). The main challenge of this task is how to balance the semantic dispersion of high-resolution targets in the small receptive field and the loss of high-precision details in the large receptive field (Zhao et al. 2018), as well as the dilemma of trading off computation cost and segmentation accuracy.

Convolutional neural networks (Ronneberger, Fischer, and Brox 2015; Zhao et al. 2017; Chen et al. 2018; Yu et al. 2018; Jiang et al. 2023, 2024; Zhang et al. 2024b; Sun et al. 2024) and vision transformers (Cheng and Sun 2024; Ji et al. 2023; Shen et al. 2023b,a; Lu et al. 2025, 2024a,b; Shen et al. 2023c; Long et al. 2024; Hu et al. 2024; Xie et al. 2024c,a,b; Zhou et al. 2023, 2024) have succeeded in representing images as uniform grids of pixels or patches of fixed size. However, as shown in Fig. 1 (a) and (b), the previous works are sub-optimal for high-resolution remote sensing imagery analysis due to either overlooking fine-grained information (Chen et al. 2019; Cheng et al. 2020; Zhang et al. 2024a, 2023; Tao et al. 2023) or introducing substantial computational overhead (Li et al. 2021; Ji, Zhao, and Lu 2023; Li et al. 2024c,a,b). On the one hand, large areas of ocean contains global contextual information but little textual details can be represented by a single token for efficient global semantic learning. On the other hand, buildings and roads with fine-grained structures require more tokens to preserve details. Inspired by the human visual system which rapidly orients attention to important areas with fine details and filter out large amount of irrelevant information in complex large scenes, we propose a novel method termed Boundary-Enhanced Patch-Merging Transformer (BPT) to tackle this problem. As shown in Fig. 1 (c), the specifically designed Patch-Merging Transformer (PMT) adopts a dynamic patch merging approach, where vision tokens represent different regions with dynamic shapes and sizes. It efficiently captures both global and local information by dynamically adopting a fine resolution for regions containing critical details while preserving global information.

To further enhance performance, the recent approaches (Guo et al. 2022; Ji, Zhao, and Lu 2023; Ji et al. 2023; Liu et al. 2023; He, Nie, and Ma 2024) for UHR images follow a dual-branch framework to preserve both global and local information. These methods involve two deep branches: one downsamples the entire image and extracts global information, while the other crops local patches feeding them to the network sequentially and merging their predictions to obtain local cues. However, the former branch loses local details leading to inaccurate edge segmentation, while the latter branch lacks global context resulting in semantic errors. Directly fusing features

![](images/9d1b75ba2800dc1479ba88058d5a0659132a0f2d3b4847f9b02bf5a9026575f8.jpg)  
Source Image

![](images/a9f0a04a2f5d16ef0c122aa7e5e03022211b65654d232ada71463e7b3330ad65.jpg)  
Ground Truth

![](images/fa1239d6c18944e94253228540d832ade4a88bde63279589de6e79283a6ae0b2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Global Prediction"] --> B["Standard Grids"]
    C["Global and Local Prediction"] --> D["Constant"]
    B --> E["Crop"]
    D --> E
    E --> F["Concat"]
```
</details>

(a)

![](images/9c0188645a3b5cf06025b19411038973ed8db229fda36c1ac954fd9735b7acda.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Training Only BEM"] --> B["PMT"]
    C["Boundary Learning"] --> B
    B --> D["Feature Fusion"]
    D --> E["PMT and BEM Prediction"]
    F["PMT Grids"] --> B
    F --> E
```
</details>

(c)   
Figure 1: (a) Existing methods representing images as standard grids of pixels are sub-optimal for UHR segmentation. (b) Dual-branch framework preserves both global and local information at the cost of increased computation. (c) Our proposed model captures both global and local information by dynamically allocating tokens to informative regions (PMT) and leveraging boundary information (BEM).

from different branches may lead to information confusion, and the introduction of two-branch structure increases the memory cost.

To tackle this problem, we propose the Boundary-Enhanced Module (BEM) to improve the edge of segmentation masks. By introducing intermediate supervision on the low-level features, the boundary details are learned in high resolutions. It is notable that the auxiliary boundary learning module can be discarded during inference, resulting in no additional computation overhead. In addition, to prevent the confusion caused by directly fusing global and local information, we propose a feature fusion module to adaptively weigh different information to obtain the more accurate features.

Extensive experiments on five public UHR image segmentation benchmark datasets demonstrate that BPT outperforms the previous state-of-the-art methods, validating the effectiveness of our proposed method.

Our main contributions can be summarized as follows:

- We propose a novel efficient UHR image segmentation method termed Boundary-Enhanced Patch-Merging Transformer (BPT) to address the issue of global and local information fusion and strike the computation-accuracy balance.   
- We propose the Patch-Merging Transformer (PMT) which dynamically represents remote sensing regions with tokens with various shape and size, capturing both the global contextual information and rich local details.   
- We propose the Boundary-Enhanced Module (BEM) to integrate detailed boundary spatial information without introducing an extra time-consuming branch.   
- Extensive experiments on five public benchmark datasets demonstrate that the proposed method outperforms the existing approaches.

# Related Work

# Generic Image Segmentation

Deep learning methods have significantly advanced the field of semantic segmentation, especially for natural images and daily photos. Early semantic segmentation models (Yuan et al. 2024b,a,c) were primarily based on Fully Convolutional Networks (FCNs). FCN-based methods typically utilized an encoder to downsample images, thereby reducing spatial resolution while extracting high-level semantic features, and a decoder to upsample images, restoring spatial resolution while classifying each pixel. For instance, Deeplabv3 (Chen et al. 2018) adopted an atrous spatial pyramid pooling module to capture long-range context, and PSPNet (Zhao et al. 2017) devised a pyramid pooling strategy to capture both local and global context information. UNet (Ronneberger, Fischer, and Brox 2015) proposed a symmetric encoder-decoder structure for efficient and precise segmentation, preserving detailed information and enhancing segmentation accuracy. With the superior ability of Transformers to capture long-distance information, Transformer-based networks (Xie et al. 2021; Cheng, Schwing, and Kirillov 2021; Nie et al. 2024; Qian et al. 2024; Yin et al. 2022, 2023; Zhu et al. 2024a; Chen et al. 2024; Wang et al. 2024) have become a new research focus in segmentation, with representative works including SegFormer (Xie et al. 2021) and MaskFormer (Cheng, Schwing, and Kirillov 2021). However, due to memory limitations, it is challenging to apply generic semantic segmentation methods to ultra-high resolution (UHR) images. Existing UHR image segmentation approaches generally downsample images to regular resolutions or crop images into small patches, processing them sequentially and merging their predictions. Cropping can lead to global semantic errors, while downsampling can result in inaccurate segmentation of details, thus producing suboptimal results. Additionally, existing networks commonly use uniform grids of pixels or fixed-size patches, which are suboptimal for high-resolution

![](images/776bd4b5c6b7ac7165ebe0078ab9c753f6a54ba466782213c9e995e2ea82d107.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Patch Feature Extraction"]
    B --> C["Patch Merging Block"]
    C --> D["..."]
    D --> E["Patch Merging Block"]
    E --> F["Patch Recovering Block"]
    F --> G["..."]
    G --> H["Patch Recovering Block"]
    H --> I["Feature Fusion Module"]
    I --> J["Seg Head"]
    J --> K["LFinal"]
    K --> L["Extract Boundaries"]
    L --> M["Ground Truth"]
    M --> N["Boundary Truth"]
    N --> O["Boundary Head"]
    O --> P["LBoundary"]
    P --> B
    H --> Q["Seg Head"]
    Q --> R["LSemantic"]
    R --> M
```
</details>

(a)

![](images/a7a6ac6ae8938ea1dad54c6a044853f094525a5418a8d07be55d7ee64a35afaa.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Patch Recovering Block"] --> B["Upsample"]
    B --> C["Norm"]
    C --> D["Linear"]
    D --> E["Transformer Block"]
    E --> F["Linear"]
    F --> G["Norm"]
    G --> H["Color Grid"]
```
</details>

(b)

![](images/e1a0a6f2c1a5c1e42a7d1a668c74fce3fda3a97791215762c3ef125767d7dea7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Transformer Block"] --> B["Norm"]
    B --> C["Spatial Reduction"]
    C --> D["Patch Feature Merging"]
    D --> E["Linear"]
    D --> F["Linear"]
    D --> G["Linear"]
    E --> H["Similarity"]
    F --> H
    G --> H
    H --> I["Updated Patch Feature Set Z"]
    I --> J["Norm"]
    I --> K["DWConv"]
    I --> L["MLP"]
    C --> M["Patch Feature Selection"]
    M --> N["Patch Feature Merging Set X"]
    N --> O["Updated Patch Feature Set Y"]
    O --> P["Updated Patch Feature Set Z"]
```
</details>

(e)

![](images/e8de662d623dec3ef9a58869e1b87dae245bbaf5cd833eb8bb03c90d826e52e1.jpg)  
(c)

![](images/b5803fe86f2e65f1affaac55e62303ebf55ece1cbbd2a9e1f68e6b11827523f2.jpg)  
(d)

![](images/7a9c11fc52c06b534e40f11702259e1aab3323adc315167dcf6a52a9d6e6df42.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["PMT Features"] --> B["Corresponding Patch"]
    B --> C["Conv+Norm+ReLU"]
    C --> D["PMT Patch Features"]
    D --> E["Relationship Matrix"]
    E --> F["Merging Features"]
    F --> G["Output"]
    H["Boundary Features"] --> I["Conv+Norm+ReLU"]
    I --> J["Boundary Patch Features"]
    J --> K["Conv+Norm+ReLU"]
    K --> L["Feature Fusion Module"]
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style K fill:#fcc,stroke:#333
```
</details>

(f)   
Figure 2: (a) Overview of Boundary-Enhanced Patch-Merging Transformer (BPT), which consists of PMT and BEM. Dotted lines represent that only needed during the training phase. (b) Patch Recovering Block, (c) Patch Feature Extraction, (d) Boundary & Seg Head, (e) Patch Merging Block, (f) Feature Fusion Module.

remote sensing imagery analysis. Instead, we propose the Patch-Merging Transformer (PMT) to dynamically allocate vision tokens with varying shapes and sizes to represent different image regions, enhancing the segmentation performance while maintaining computational efficiency.

# Ultra-High Resolution Image Segmentation

Advancements in photography and sensor technologies have increased the accessibility of Ultra-High Resolution (UHR) geospatial images, opening new horizons for the computer vision community. Most existing methods for UHR image segmentation utilize multi-branch networks to learn both global and local information. GLNet (Chen et al. 2019) incorporated global and local information deeply in a two-stream branch manner. CascadePSP (Cheng et al. 2020) proposed a multi-branch network to learn features at different scales, generating high-quality results. FCtL (Li et al. 2021) exploited three different cropping scales to fuse multiscale feature information. ISDNet (Guo et al. 2022) integrated shallow and deep networks to learn global and local information respectively. WSDNet (Ji et al. 2023) in

troduced a multi-level discrete wavelet transform into the global and local branches to reduce computational overhead. GPWFormer (Ji, Zhao, and Lu 2023) used a hybrid CNN-Transformer in a dual-branch style to efficiently harvest both low-level and high-level context. GeoAgent (Liu et al. 2023) employed a reinforcement learning network to dynamically adjust the size of patches for global context. These existing methods often design complex multi-encoder-decoder streams and stages to gradually fuse global and local information, resulting in high memory requirements. In addition, directly merging these features can lead to information confusion and segmentation errors. To address these issues, we propose a novel efficient UHR image segmentation method called Boundary-Enhanced Patch-Merging Transformer (BPT). BPT adopts a single-branch structure capturing both global and local information by accounting for varying receptive field requirements in different instances. In addition, the auxiliary Boundary-Enhanced Module (BEM) leverages boundary information for learning fine details.

# Method

# Overview

We introduce our proposed method for ultra-high resolution (UHR) image segmentation, termed Boundary-Enhanced Patch-Merging Transformer (BPT). As shown in Fig. 2, BPT comprises two main components: Patch-Merging Transformer (PMT) and Boundary-Enhanced Module (BEM). Each UHR image is first cropped into several patches. PMT dynamically merges patches based on instance sizes, extracting both global and local information for more accurate preliminary segmentation. Additionally, the BEM learns boundary information to guide the fusion of fine boundary details and local information, enhancing segmentation accuracy without extra time or memory consumption.

# Patch Merging Transformer (PMT)

Our proposed Patch-Merging Transformer (PMT) is composed of Patch Feature Extraction (PFE), Patch Merging Block (PMB) and Patch Recovering Block (PRB). The input image is first uniformly divided in to small patches and processed through PFE to extract patch features. Then, the patches are automatically merged through PMB to reduce their number and thereby decrease memory consumption. Next, the merged patches are restored to the original number of patches through PRB and get the feature maps.

Patch Feature Extraction (PFE). The input UHR image is evenly partitioned into image patches to reduce resolution. Unlike previous methods, we use smaller, more numerous patches for finer segmentation. We crop the source image into patches with a size of $32 \times 32$ pixels. A higher number of patches is beneficial for preserving detailed information, and since our method can dynamically merge patches, it does not consume a large amount of memory. In implementation, we adopt PVT block as the base transformer block, for its high computation efficiency.

Patch Merging Block (PMB). We employ the Patch Merging Block (PMB) to gradually fuse patches and extract deep semantic information.

Patch Selection. Inspired by (Zeng et al. 2022), we apply the clustering algorithm to merge similar patch features, reducing the number of patches. Specifically, we use a variant of k-nearest neighbor based density peaks clustering algorithm (DPC-KNN). Given a set of patches $X = [x_{1}, \ldots, x_{i}, \ldots, x_{j}, \ldots]$ , we compute the local density $\rho$ of each patch based on its k-nearest neighbors:

$$
\rho_ {i} = \exp \left(- \frac {1}{k} \sum_ {x _ {j} \in \mathrm{KNN} (x _ {i})} \| x _ {i} - x _ {j} \| _ {2} ^ {2}\right), \tag {1}
$$

where $\mathrm{KNN}(x_{i})$ denotes the k-nearest neighbors of patch i. $x_{i}$ and $x_{j}$ are their corresponding patch features. We then compute the distance indicator as the minimal distance between a patch and any other patch with higher local density. For the patch with the highest local density, its indicator is set as the maximal distance between it and any other patches:

$$
\delta_ {i} = \left\{ \begin{array}{l l} \min _ {j: \rho_ {j} > \rho_ {i}} \| x _ {i} - x _ {j} \| _ {2}, & \text { if } \exists j \text { s.t. } \rho_ {j} > \rho_ {i} \\ \max _ {j} \| x _ {i} - x _ {j} \| _ {2}, & \text { otherwise } \end{array} \right. \tag {2}
$$

where $\delta_{i}$ is the distance indicator and $\rho_{i}$ is the local density. We combine the local density and the distance indicator to score each patch as $\rho_{i} \times \delta_{i}$ . Higher scores indicate higher potential as patch centers. We determine patch centers by selecting tokens with the highest scores and then assign other tokens to the nearest patch center based on feature distances.

Patch Feature Merging. To focus on the important image features when merging patch features, we utilize an importance-based patch feature merging strategy. Specifically, we use the importance score $P_{j}$ to represent the importance of each patch, estimated from the patch local density and distance indicator. The merged patch feature set $Y = [y_{1}, y_{2}, \ldots]$ is calculated as:

$$
y = \sum_ {j \in C _ {i}} S o f t m a x (\rho_ {j} \cdot \delta_ {j}) x _ {j}, \tag {3}
$$

where $C_{i}$ is the set of the i-th cluster, $x_{j}$ and $p_{j}$ are the original patch features and the corresponding importance score, respectively, and y is the features of the merged patch.

Patch Feature Updating. We enhance the merged patch features by considering the relations between the patch features before and after merging. Formally, we compute the similarity matrix S between the merged features (patch feature set after merging) and original features (patch feature set before merging) as:

$$
\text { Similarity } (y _ {m}, x _ {i}) = S _ {m, i} = \frac {\exp (W _ {q} y _ {m} \cdot W _ {k} x _ {i})}{\sum_ {n = 1} ^ {N} \exp (W _ {q} y _ {n} \cdot W _ {k} x _ {i})}, \tag {4}
$$

where $x_{i}$ and $y_{m}$ are the i-th of X and m-th of Y, $W_{q}$ and $W_{k}$ are weights of learned linear projections for patch features, and N is the number of elements in Y. The merged patch features are updated by adding a residual term capturing the details of the original features:

$$
z _ {m} = y _ {m} + W _ {o} \frac {\sum_ {i = 1} ^ {N _ {x}} S _ {m , i} W _ {v} x _ {i}}{\sum_ {i = 1} ^ {N _ {x}} S _ {m , i}}, \tag {5}
$$

where $N_{x}$ is the number of X, and $W_{v}$ and $W_{o}$ are learned weights to project merged features. In practice, we need to first transform tokens to feature maps before the projection process and perform the inverse transform after. We transform feature maps into patch features before merging and perform the inverse transformation afterward. Specifically, patch features correspond to specific positions in feature maps. The patches that undergo merging will have identical features, reducing the number of patches while expanding their regions. Patch features are inverse-transformed back into feature maps based on the position correspondence. Finally, the obtained feature maps are passed sequentially through Norm, DWConv, and MLP.

Patch Recovering Block (PRB). Patch Recovering Blocks (PRB) aggregate patches of different scales (ordinary and dynamically-merged patch features) and reconstruct output features for upsampling. PRB uses two linear

layers and one PVT transformer block to reduce computation burdens. For each merged patch feature containing abstract semantics, PRB upsamples patches and recovers feature mapping based on its merging history. During token merging in PFM, we record positional correspondence between original and merged patch features. In PRB's upsampling process, these records copy merged patch features into corresponding upsampled patches, executed progressively until all patch features are aggregated.

# Boundary-Enhanced Module (BEM)

We propose a Boundary-Enhanced Module (BEM) to guide low-level layers in learning boundary information (Wu et al. 2022) and refine global information for more precise results. This process occurs only during training, avoiding additional memory or time consumption during inference.

Boundary Prediction Task (BPT). Downsampling and upsampling processes can lose the boundary information, degrading segmentation performance at boundaries. To preserve the fine boundary details, we introduce an auxiliary task guiding low-level layers to learn boundary prediction. The boundary prediction is formulated as a binary segmentation task. We use the Canny operator and Dilation operation to obtain the boundaries mask.

Feature Fusion Module (FFM). The PMT path is semantically accurate but loses spatial and geometric details, especially at boundaries. The BEM path preserves captures boundary details, but lacks global semantic information. As the features of the two paths differ in representation level, simply averaging these features may cause information conflict and segmentation errors. To solve this problem, we propose the feature fusion module to solve information conflict, and achieve better segmentation accuracy. As shown in Figure 2(f), we concatenate features $F_{PMT}$ and $F_{BEM}$ . A relationship matrix $R^{C \times PN \times PN}$ is obtained by inputting it into the convolutional network (C is the number of channels, PN the number of patches). We then perform channel-wise multiplication on the patches and channels corresponding to this relationship matrix, selecting features. Finally, we concatenate features to obtain the final feature representations.

# Loss Functions

Instead of supervising only the final segmentation map, we jointly supervise all three parts: $L_{Semantic}$ , $L_{Boundary}$ , and $L_{Final}$ , as each serves a specific purpose in our design. The total loss $L_{Total}$ is the weighted combination of these parts:

$$
L _ {\text { Total }} = \lambda_ {1} L _ {\text { Semantic }} + \lambda_ {2} L _ {\text { Boundary }} + \lambda_ {3} L _ {\text { Final }}, \tag {6}
$$

$\lambda_{1}, \lambda_{2}$ and $\lambda_{3}$ are balancing hyper-parameters.

Semantic Loss. To enable the PMT branch to learn better semantic information, we adopt the boundary relaxation loss (RL) (Zhu et al. 2019), sampling only part of the pixels within objects for supervision. Since pixel numbers on different surfaces vary greatly, we also use focal loss (FL) for optimization. The semantic loss is:

$$
L _ {\text { Semantic }} = \alpha_ {1} L _ {\mathrm{FL}} + \beta_ {1} L _ {\mathrm{RL}}. \tag {7}
$$

Boundary Loss. Boundary pixel prediction faces a class imbalance problem due to fewer boundary pixels compared to non-boundary pixels. Using the weighted cross-entropy loss alone often yields coarse segmentation results. To solve this problem, we employ both binary cross-entropy (BCE) and dice loss (DL) (Milletari, Navab, and Ahmadi 2016) to optimize boundary learning. The boundary loss is:

Table 1: Comparisons with the state-of-the-art methods on the DeepGlobe dataset. ↑ means higher is better, ↓ means lower is better. \* means cropping the UHR image to small patches and predicting the results separately. 

<table><tr><td></td><td>Method</td><td>mIoU(%)↑</td><td>F1(%)↑</td><td>Acc(%)↑</td><td>Mem(M)↓</td></tr><tr><td rowspan="11">Generic</td><td>U-Net*</td><td>37.3</td><td>-</td><td>-</td><td>949</td></tr><tr><td>DeepLabv3+*</td><td>63.1</td><td>-</td><td>-</td><td>1279</td></tr><tr><td>FCN-8s*</td><td>71.8</td><td>82.6</td><td>87.6</td><td>1963</td></tr><tr><td>U-Net</td><td>38.4</td><td>-</td><td>-</td><td>5507</td></tr><tr><td>ICNet</td><td>40.2</td><td>-</td><td>-</td><td>2557</td></tr><tr><td>PSPNet</td><td>56.6</td><td>-</td><td>-</td><td>6289</td></tr><tr><td>DeepLabv3+</td><td>63.5</td><td>-</td><td>-</td><td>3199</td></tr><tr><td>FCN-8s</td><td>68.8</td><td>79.8</td><td>86.2</td><td>5227</td></tr><tr><td>BiseNetV1</td><td>53.0</td><td>-</td><td>-</td><td>1801</td></tr><tr><td>DANet</td><td>53.8</td><td>-</td><td>-</td><td>6812</td></tr><tr><td>STDC</td><td>70.3</td><td>-</td><td>-</td><td>2580</td></tr><tr><td rowspan="12">UHR</td><td>CascadePSP</td><td>68.5</td><td>79.7</td><td>85.6</td><td>3236</td></tr><tr><td>PPN</td><td>71.9</td><td>-</td><td>-</td><td>1193</td></tr><tr><td>PointRend</td><td>71.8</td><td>-</td><td>-</td><td>1593</td></tr><tr><td>MagNet</td><td>72.9</td><td>-</td><td>-</td><td>1559</td></tr><tr><td>MagNet-Fast</td><td>71.8</td><td>-</td><td>-</td><td>1559</td></tr><tr><td>GLNet</td><td>71.6</td><td>83.2</td><td>88.0</td><td>1865</td></tr><tr><td>ISDNet</td><td>73.3</td><td>84.0</td><td>88.7</td><td>1948</td></tr><tr><td>FCtL</td><td>73.5</td><td>83.8</td><td>88.3</td><td>3167</td></tr><tr><td>WSDNet</td><td>74.1</td><td>85.2</td><td>89.1</td><td>1876</td></tr><tr><td>GeoAgent</td><td>75.4</td><td>85.3</td><td>89.6</td><td>3990</td></tr><tr><td>GPWFormer</td><td>75.8</td><td>85.4</td><td>89.9</td><td>2380</td></tr><tr><td>BPT (Ours)</td><td>76.6</td><td>85.7</td><td>90.1</td><td>2074</td></tr></table>

$$
L _ {\text { Boundary }} = \alpha_ {2} L _ {\mathrm{DL}} + \beta_ {2} L _ {\mathrm{BCE}}. \tag {8}
$$

Final Loss. In UHR segmentation tasks, using only pixel-level cross-entropy supervision can deteriorate detailed structural information. To better capture objects with significant size differences, we use cross-entropy loss (CE) and focal loss (FL) to supervise final inference results. The final loss is:

$$
L _ {\text { Final }} = \alpha_ {3} L _ {\mathrm{FL}} + \beta_ {3} L _ {\mathrm{CE}} \tag {9}
$$

# Experiments

# Experimental Setup

Datasets and Evaluation Metrics. To validate our method's effectiveness, we conducted experiments on five UHR image datasets: DeepGlobe (Demir et al. 2018), Inria Aerial (Maggiori et al. 2017), CityScapes (Cordts et al. 2016), ISIC (Tschandl, Rosendahl, and Kittler 2018), and CRAG (Graham et al. 2019).

The DeepGlobe dataset comprises 803 UHR images, split into 455/207/142 for training, validation, and testing, respectively. Each image is $2448 \times 2448$ pixels, with annotations for seven landscape classes. The Inria Aerial

Table 2: Comparisons with the state-of-the-art methods on the Inria Aerial dataset. 

<table><tr><td></td><td>Method</td><td>mIoU (%)↑</td><td>F1 (%)↑</td><td>Acc (%)↑</td><td>Mem (M)↓</td></tr><tr><td rowspan="3">Generic</td><td>DeepLabv3+</td><td>55.9</td><td>-</td><td>-</td><td>5122</td></tr><tr><td>FCN-8s</td><td>69.1</td><td>81.7</td><td>93.6</td><td>2447</td></tr><tr><td>STDC</td><td>72.4</td><td>-</td><td>-</td><td>7410</td></tr><tr><td rowspan="8">UHR</td><td>CascadePSP</td><td>69.4</td><td>81.8</td><td>93.2</td><td>3236</td></tr><tr><td>GLNet</td><td>71.2</td><td>-</td><td>-</td><td>2663</td></tr><tr><td>ISDNet</td><td>74.2</td><td>84.9</td><td>95.6</td><td>4680</td></tr><tr><td>FCtL</td><td>73.7</td><td>84.1</td><td>94.6</td><td>4332</td></tr><tr><td>WSDNet</td><td>75.2</td><td>86.0</td><td>96.0</td><td>4379</td></tr><tr><td>GeoAgent</td><td>76.0</td><td>86.1</td><td>96.4</td><td>5780</td></tr><tr><td>GPWFormer</td><td>76.5</td><td>86.2</td><td>96.7</td><td>4710</td></tr><tr><td>BPT (Ours)</td><td>77.1</td><td>86.3</td><td>96.8</td><td>4450</td></tr></table>

dataset includes 180 UHR images (5000 × 5000 pixels) with binary masks for building/non-building areas, divided into 126/27/27 for training, validation, and testing. The Cityscapes dataset contains 5000 images with 19 semantic classes, split into 2979/500/1525 for training, validation, and testing. The ISIC dataset consists of 2596 UHR images, divided into 2077/260/259 for training, validation, and testing. The CRAG dataset comprises 213 images with glandular morphology annotations, split into 173 for training and 40 for testing, with an average size of 1512 × 1516.

In all experiments, we follow common practices (Ji, Zhao, and Lu 2023) by adopting mIoU, F1 score, and pixel accuracy (Acc) to evaluate segmentation performance, with mIoU being the primary metric. We also assess efficiency through GPU memory cost, measured using the "gpustat" tool with a mini-batch size of 1.

Baselines. We compare BPT with several representative baselines. Some methods are designed for UHR images (denoted as "UHR") and others are not ("Generic"). Generic segmentation baselines include U-Net (Ronneberger, Fischer, and Brox 2015), PSPNet (Zhao et al. 2017), DeepLabv3+ (Chen et al. 2018), FCN-8s (Long, Shelhamer, and Darrell 2015), BiseNetV1 (Yu et al. 2018), BiseNetV2 (Yu et al. 2021), DANet (Fu et al. 2019), STDC (Fan et al. 2021), while UHR segmentation baselines include CascadePSP (Cheng et al. 2020), PPN (Wu et al. 2020), PointRend (Kirillov et al. 2020), MagNet (Huynh et al. 2021), MagNet-Fast (Huynh et al. 2021), GLNet (Chen et al. 2019), ISDNet (Guo et al. 2022), FCtL (Li et al. 2021), WSDNet (Ji et al. 2023), GPWFormer (Ji, Zhao, and Lu 2023), GeoAgent (Liu et al. 2023), DenseCRF (Krähenbühl and Koltun 2011), DGF (Wu et al. 2018), SegFix (Yuan et al. 2020). The results of baseline models are referenced from (Ji, Zhao, and Lu 2023).

Implementation. For PMT, patch size is set to $32 \times 32$ . The number of Patch-Merging Transformer Blocks and Patch Recovering Blocks is four. We pre-train the model on the ImageNet-1K dataset using AdamW with a momentum of 0.9 and a weight decay of $5 \times 10^{-2}$ . The initial learning rate is $1 \times 10^{-3}$ , and the learning rate follows the cosine schedule. Models are pre-trained for 300 epochs. For segmentation training, we train models on MMSegmentation codebase with GTX 3090 GPUs. We optimize models using AdamW with an initial learning rate of $1 \times 10^{-4}$ , decayed using a polynomial schedule with a power of 0.9. Hyperparameters are set as follows: $\alpha_{1} = 0.6$ , $\beta_{1} = 0.4$ , $\alpha_{2} = 0.3$ , $\beta_{2} = 0.7$ , $\alpha_{3} = 0.5$ , $\beta_{3} = 0.5$ , $\lambda_{1} = 0.3$ , $\lambda_{2} = 0.3$ , $\lambda_{3} = 0.4$ . Following common practices (Ji, Zhao, and Lu 2023), maximum training iterations are set to 40k, 80k, 160k, 80k and 80k for Inria Aerial, DeepGlobe, Cityscapes, ISIC and CRAG, respectively.

Table 3: Comparisons with the state-of-the-art methods on the Cityscapes datasets. 

<table><tr><td></td><td>Method</td><td>mIoU (%)↑</td><td>Mem (M)↓</td></tr><tr><td rowspan="4">Generic</td><td>BiseNetV1</td><td>74.4</td><td>2147</td></tr><tr><td>BiseNetV2</td><td>75.8</td><td>1602</td></tr><tr><td>PSPNet</td><td>74.9</td><td>1584</td></tr><tr><td>DeepLabv3</td><td>76.7</td><td>1468</td></tr><tr><td rowspan="9">UHR</td><td>DenseCRF</td><td>62.9</td><td>1575</td></tr><tr><td>DGF</td><td>63.3</td><td>1727</td></tr><tr><td>SegFix</td><td>65.8</td><td>2033</td></tr><tr><td>MagNet</td><td>67.6</td><td>2007</td></tr><tr><td>MagNet-Fast</td><td>66.9</td><td>2007</td></tr><tr><td>ISDNet</td><td>76.0</td><td>1510</td></tr><tr><td>GeoAgent</td><td>77.8</td><td>2953</td></tr><tr><td>GPWFormer</td><td>78.1</td><td>1897</td></tr><tr><td>BPT (Ours)</td><td>78.5</td><td>1686</td></tr></table>

Table 4: Comparisons with the state-of-the-art methods on the CRAG and ISIC datasets.

<table><tr><td>Method</td><td>ISICmIoU (%)↑</td><td>CRAGmIoU (%)↑</td></tr><tr><td>PSPNet</td><td>77.0</td><td>88.6</td></tr><tr><td>DeepLabV3+</td><td>70.5</td><td>88.9</td></tr><tr><td>DANet</td><td>51.4</td><td>82.3</td></tr><tr><td>GLNet</td><td>75.2</td><td>85.9</td></tr><tr><td>GeoAgent</td><td>80.2</td><td>89.4</td></tr><tr><td>GPWFormer</td><td>80.7</td><td>89.9</td></tr><tr><td>BPT(Ours)</td><td>81.6</td><td>90.9</td></tr></table>

# Experimental Results

DeepGlobe. As shown in Table 1, we compare our proposed BPT with the aforementioned baseline methods on the DeepGlobe test dataset. The results demonstrate that BPT surpasses all other methods in terms of mIoU, F1, and accuracy. Notably, BPT significantly outperforms GeoAgent and GPWFormer in mIoU, without extra gpu memory cost.

Inria Aerial. Table 2 presents the comparisons on the Inria Aerial test dataset. This dataset poses a greater challenge due to its high resolution, with each image containing 25 million pixels—approximately four times that of DeepGlobe and finer foreground regions. The results indicate that our BPT outperforms the baselines by substantial margins in mIoU, while maintaining comparable memory costs.

Cityscapes. To further validate the generality of our method, we present results on the Cityscapes dataset in Table 3. BPT consistently outperforms all other methods in

Table 5: Ablation studies on the DeepGlobe, Inria Aerial, and Cityscapes datasets. 

<table><tr><td rowspan="2">ExpID</td><td colspan="4">Methods</td><td colspan="2">DeepGlobe</td><td colspan="2">Inria Aerial</td><td colspan="2">Cityscapes</td></tr><tr><td>PMB</td><td>PRB</td><td>BPT</td><td>FFM</td><td>mIoU (%)↑</td><td>Mem (%)↓</td><td>mIoU (%)↑</td><td>Mem (%)↓</td><td>mIoU (%)↑</td><td>Mem (%)↓</td></tr><tr><td>#1</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>76.6</td><td>2074</td><td>77.1</td><td>4450</td><td>78.5</td><td>1686</td></tr><tr><td>#2</td><td>✗</td><td>✗</td><td>✓</td><td>✓</td><td>73.9(-2.7)</td><td>2153</td><td>74.8(-2.3)</td><td>4592</td><td>76.9(-1.6)</td><td>1755</td></tr><tr><td>#3</td><td>✗</td><td>✓</td><td>✓</td><td>✓</td><td>75.0(-1.6)</td><td>2100</td><td>75.8(-1.3)</td><td>4562</td><td>77.7(-0.8)</td><td>1743</td></tr><tr><td>#4</td><td>✓</td><td>✗</td><td>✓</td><td>✓</td><td>75.5(-1.1)</td><td>2132</td><td>73.8(-1.0)</td><td>4496</td><td>77.8(-0.8)</td><td>1700</td></tr><tr><td>#5</td><td>✓</td><td>✓</td><td>✗</td><td>✗</td><td>75.5(-1.1)</td><td>2018</td><td>76.1(-1.0)</td><td>4364</td><td>77.9(-0.6)</td><td>1622</td></tr><tr><td>#6</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td><td>76.1(-0.5)</td><td>2035</td><td>76.7(-0.4)</td><td>4396</td><td>78.3(-0.2)</td><td>1642</td></tr></table>

mIoU, while also demonstrating efficient memory usage.

ISIC and CRAG. The ISIC dataset's image resolution is comparable to that of Inria Aerial, whereas CRAG features lower resolution images than the other datasets. Table 4 illustrates the experimental results, where BPT consistently achieves excellent performance across both datasets.

# Ablation Study

We present ablation studies in Table 5 to evaluate the effectiveness of the proposed Patch-Merging Transformer (PMT) and Boundary-Enhanced Module (BEM) on the DeepGlobe, Inria Aerial, and Cityscapes datasets.

Effect of Patch-Merging Transformer (PMT). PMT comprises Patch Merging Blocks (PMB) and Patch Recovering Blocks (PRB). To assess the impact of PMB, we built a baseline transformer network by replacing PMB with convolutional layer to downsample, as shown in ExpID #3, resulting in significant performance drops (-1.6%, -1.3%, -0.8% mIoU compared to ExpID #1). Similarly, replacing PRB with a deconvolutional head led to performance declines of -1.1%, -1.0%, and -0.8% mIoU (ExpID #4). Comparing ExpID #1 and #2 demonstrates the effectiveness of PMT.

Effect of Boundary-Enhanced Module (BEM). BEM significantly enhances model performance with minimal memory increase (ExpID #1 and #5), highlighting the importance of boundary information. The Feature Fusion Module (FFM) is also crucial for guiding information fusion, resulting in improved accuracy (ExpID #1 vs. #6).

Qualitative Analysis. To demonstrate the effectiveness of our proposed method, we conduct a qualitative analysis on the DeepGlobe dataset in Fig. 3. We visualize (a) the source image, (b) patch tokens generated by PMT, (c) the ground-truth mask, (d) results of the previous SOTA method, i.e. GPWFormer, (e) results of our proposed BPT.

Analysis of Patch Merging. As shown in Fig. 3(b), our method clusters patches of the same category into dynamic tokens of various shapes and sizes. Large areas of water, with minimal textual details, are represented by fewer tokens, while small, fine-grained areas use more tokens to preserve local information. Dynamic token allocation is crucial for preserving both global and local information without increasing computational cost.

Comparisons with GPWFormer. We compare our BPT with the previous state-of-the-art method, GPWFormer. Our BPT produces more precise results with better semantics and finer segmentation boundaries, showcasing its superior performance (Fig. 3(d)(e)).

![](images/5757ec32729263dc84500a4f01bcc9ad870d8369e961d2b2d95c38c3d7dc9fb8.jpg)  
Figure 3: Qualitative analysis on the DeepGlobe dataset. (a) Source image. (b) Patch tokens generated by PMT. (c) Ground-truth mask. (d) Results of GPWFormer. (e) Results of BPT (ours).

# Conclusion

In this work, we propose a novel UHR image segmentation method, termed Boundary-Enhanced Patch-Merging Transformer (BPT). The Patch-Merging Transformer dynamically and adaptively merges patches to effectively capture both global semantic information and local fine-grained details. The Boundary-Enhanced Module (BEM) enhances segmentation accuracy by enriching fine details. Extensive experiments demonstrate that our BPT consistently outperforms existing methods on various UHR image segmentation benchmarks.

# Acknowledgments

This work is supported by the Strategic Priority Research Program of Chinese Academy of Sciences (No.XDA28040500), the Natural Science Foundation of China (No.62302487), and two projects from the Science and Technology Innovation Program of Hunan Province (No.2024JJ9031 and No.2022RC4006).

# References

Chen, L.-C.; Zhu, Y.; Papandreou, G.; Schroff, F.; and Adam, H. 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 801–818.   
Chen, Q.; Wang, T.; Yang, Z.; Li, H.; Lu, R.; Sun, Y.; Zheng, B.; and Yan, C. 2024. SDPL: Shifting-Dense Partition Learning for UAV-View Geo-Localization. IEEE Trans. Circuits Syst. Video Technol., 34(11): 11810–11824.   
Chen, W.; Jiang, Z.; Wang, Z.; Cui, K.; and Qian, X. 2019. Collaborative global-local networks for memory-efficient segmentation of ultra-high resolution images. In CVPR, 8924–8933.   
Cheng, B.; Schwing, A.; and Kirillov, A. 2021. Per-pixel classification is not all you need for semantic segmentation. NeurIPS, 34: 17864–17875.   
Cheng, H. K.; Chung, J.; Tai, Y.-W.; and Tang, C.-K. 2020. CascadePSP: Toward class-agnostic and very high-resolution segmentation via global and local refinement. In CVPR, 8890–8899.   
Cheng, S.; and Sun, H. 2024. SPT: Sequence Prompt Transformer for Interactive Image Segmentation. arXiv:2412.10224.   
Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016. The cityscapes dataset for semantic urban scene understanding. In ICCV, 3213–3223.   
Demir, I.; Koperski, K.; Lindenbaum, D.; Pang, G.; Huang, J.; Basu, S.; Hughes, F.; Tuia, D.; and Raskar, R. 2018. Deepglobe 2018: A challenge to parse the earth through satellite images. In CVPR Workshops, 172–181.   
Fan, M.; Lai, S.; Huang, J.; Wei, X.; Chai, Z.; Luo, J.; and Wei, X. 2021. Rethinking BiSeNet for real-time semantic segmentation. In CVPR, 9716–9725.   
Fu, J.; Liu, J.; Tian, H.; Li, Y.; Bao, Y.; Fang, Z.; and Lu, H. 2019. Dual attention network for scene segmentation. In CVPR, 3146–3154.   
Graham, S.; Chen, H.; Gamper, J.; Dou, Q.; Heng, P.-A.; Snead, D.; Tsang, Y. W.; and Rajpoot, N. 2019. MILD-Net: Minimal information loss dilated network for gland instance segmentation in colon histology images. Med. Image Anal., 52: 199–211.   
Guan, R.; Li, Z.; Tu, W.; Wang, J.; Liu, Y.; Li, X.; Tang, C.; and Feng, R. 2024a. Contrastive Multiview Subspace Clustering of Hyperspectral Images Based on Graph Convolutional Networks. IEEE TGRS., 62: 1–14.   
Guan, R.; Tu, W.; Li, Z.; Yu, H.; Hu, D.; Chen, Y.; Tang, C.; Yuan, Q.; and Liu, X. 2024b. Spatial-Spectral Graph Contrastive Clustering with Hard Sample Mining for Hyperspectral Images. IEEE TGRS., 1–16.   
Guo, S.; Liu, L.; Gan, Z.; Wang, Y.; Zhang, W.; Wang, C.; Jiang, G.; Zhang, W.; Yi, R.; Ma, L.; et al. 2022. ISDNet: Integrating shallow and deep networks for efficient ultra-high resolution segmentation. In CVPR, 4361–4370.

He, J.; Nie, T.; and Ma, W. 2024. Geolocation representation from large language models are generic enhancers for spatio-temporal learning. arXiv:2408.12116.   
Hu, Q.; Yi, Z.; Zhou, Y.; Li, T.; Huang, F.; Liu, M.; Li, Q.; and Wang, Z. 2024. MonoBox: Tightness-free Box-supervised Polyp Segmentation using Monotonicity Constraint. arXiv e-prints, arXiv–2404.   
Huynh, C.; Tran, A. T.; Luu, K.; and Hoai, M. 2021. Progressive semantic segmentation. In CVPR, 16755–16764.   
Ji, D.; Zhao, F.; and Lu, H. 2023. Guided patch-grouping wavelet transformer with spatial congruence for ultra-high resolution segmentation. arXiv:2307.00711.   
Ji, D.; Zhao, F.; Lu, H.; Tao, M.; and Ye, J. 2023. Ultra-high resolution segmentation with ultra-rich context: A novel benchmark. In CVPR, 23621–23630.   
Jiang, J.; Feng, Y.; Chen, J.; Guo, D.; and Zheng, J. 2023. Latent-space Unfolding for MRI Reconstruction. In Proc. 31st ACM Int. Conf. Multimedia, 1294–1302.   
Jiang, J.; He, Z.; Quan, Y.; Wu, J.; and Zheng, J. 2024. PGIUN: Physics-Guided Implicit Unrolling Network for Accelerated MRI. IEEE Trans. Comput. Imaging.   
Kirillov, A.; Wu, Y.; He, K.; and Girshick, R. 2020. PointRend: Image segmentation as rendering. In CVPR, 9799–9808.   
Krähenbühl, P.; and Koltun, V. 2011. Efficient inference in fully connected CRFs with Gaussian edge potentials. NeurIPS, 24.   
Li, L.; Xing, J.; Yu, X.; and Zhang, X.-P. 2024a. Deviation Wing Loss for High-Performance 2D Pose Estimation. In IEEE ICME, 1–6. IEEE.   
Li, L.; Yang, W.; Yu, X.; Xing, J.; and Zhang, X.-P. 2024b. Translating Motion to Notation: Hand Labanotation for Intuitive and Comprehensive Hand Movement Documentation. In Proc. 32nd ACM Int. Conf. Multimedia, 4092–4100.   
Li, Q.; Yang, W.; Liu, W.; Yu, Y.; and He, S. 2021. From contexts to locality: Ultra-high resolution image segmentation via locality-aware contextual correlation. In CVPR, 7252–7261.   
Li, S.; Ye, M.; Zhou, L.; Li, N.; Xiao, S.; Tang, S.; and Zhu, X. 2024c. Cloud Object Detector Adaptation by Integrating Different Source Knowledge. In Proc. 38th Annu. Conf. Neural Inf. Process. Syst.   
Li, Y.; Long, Q.; Zhou, Y.; Cao, N.; Liu, S.; Zheng, F.; Zhu, Z.; Ning, Z.; Xiao, M.; Wang, X.; et al. 2024d. COMAE: COMprehensive Attribute Exploration for Zero-shot Hashing. arXiv preprint arXiv:2402.16424.   
Li, Y.; Lu, Y.; Dong, Z.; Yang, C.; Chen, Y.; and Gou, J. 2024e. SGLP: A Similarity Guided Fast Layer Partition Pruning for Compressing Large Deep Models. arXiv preprint arXiv:2410.14720.   
Liu, Y.; Shi, S.; Wang, J.; and Zhong, Y. 2023. Seeing Beyond the Patch: Scale-Adaptive Semantic Segmentation of High-resolution Remote Sensing Imagery based on Reinforcement Learning. In CVPR, 16868–16878.

Long, J.; Shelhamer, E.; and Darrell, T. 2015. Fully convolutional networks for semantic segmentation. In CVPR, 3431–3440.   
Long, X.; Zeng, J.; Meng, F.; Ma, Z.; Zhang, K.; Zhou, B.; and Zhou, J. 2024. Generative multi-modal knowledge retrieval with large language models. In AAAI, 18733–18741.   
Lu, H.; Tang, J.; Xu, X.; Cao, X.; Zhang, Y.; Wang, G.; Du, D.; Chen, H.; and Chen, Y. 2024a. Scaling Multi-Camera 3D Object Detection through Weak-to-Strong Eliciting. arXiv:2404.06700.   
Lu, H.; Xu, T.; Zheng, W.; Zhang, Y.; Zhan, W.; Du, D.; Tomizuka, M.; Keutzer, K.; and Chen, Y. 2024b. DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving. arXiv preprint arXiv:2412.09043.   
Lu, H.; Zhang, Y.; Lian, Q.; Du, D.; and Chen, Y. 2025. Towards generalizable multi-camera 3D object detection via perspective debiasing. AAAI.   
Maggiori, E.; Tarabalka, Y.; Charpiat, G.; and Alliez, P. 2017. Can semantic labeling methods generalize to any city? the inria aerial image labeling benchmark. In IEEE IGARSS, 3226–3229.   
Milletari, F.; Navab, N.; and Ahmadi, S.-A. 2016. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 3DV, 565–571. IEEE.   
Nie, T.; Qin, G.; Ma, W.; Mei, Y.; and Sun, J. 2024. ImputeFormer: Low rankness-induced transformers for generalizable spatiotemporal imputation. In Proc. 30th ACM SIGKDD Conf. Knowl. Discov. Data Min., 2260–2271.   
Qian, H.; Chen, Y.; Lou, S.; Khan, F.; Jin, X.; and Fan, D.-P. 2024. MaskFactory: Towards High-quality Synthetic Data Generation for Dichotomous Image Segmentation. In NeurIPS.   
Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 234–241. Springer.   
Shen, F.; Du, X.; Zhang, L.; and Tang, J. 2023a. Triplet Contrastive Learning for Unsupervised Vehicle Re-identification. arXiv:2301.09498.   
Shen, F.; Shu, X.; Du, X.; and Tang, J. 2023b. Pedestrian-specific Bipartite-aware Similarity Learning for Text-based Person Retrieval. In Proc. 31st ACM Int. Conf. Multimedia.   
Shen, F.; Xie, Y.; Zhu, J.; Zhu, X.; and Zeng, H. 2023c. Git: Graph interactive transformer for vehicle re-identification. IEEE Trans. Image Process.   
Sun, H.; Xu, L.; Jin, S.; Luo, P.; Qian, C.; and Liu, W. 2024. PROGRAM: PROtotype GRAPh Model based Pseudo-Label Learning for Test-Time Adaptation. In ICLR.   
Tao, H.; Li, J.; Hua, Z.; and Zhang, F. 2023. DUDB: Deep Unfolding Based Dual-Branch Feature Fusion Network for Pan-sharpening remote sensing images. IEEE TGRS.   
Tschandl, P.; Rosendahl, C.; and Kittler, H. 2018. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data, 5(1): 1–9.

Wang, T.; Yang, Z.; Chen, Q.; Sun, Y.; and Yan, C. 2024. Rethinking Pooling for Multi-Granularity Features in Aerial-View Geo-Localization. IEEE Signal Process. Lett., 31: 3005–3009.   
Wu, H.; Zheng, S.; Zhang, J.; and Huang, K. 2018. Fast end-to-end trainable guided filter. In CVPR, 1838–1847.   
Wu, T.; Lei, Z.; Lin, B.; Li, C.; Qu, Y.; and Xie, Y. 2020. Patch proposal network for fast semantic segmentation of high-resolution images. In AAAI, 12402–12409.   
Wu, X.; Jiang, B.; Zhong, Y.; and Chen, H. 2022. Multi-target Markov boundary discovery: Theory, algorithm, and application. IEEE Trans. Pattern Anal. Mach. Intell., 45(4): 4964–4980.   
Xie, E.; Wang, W.; Yu, Z.; Anandkumar, A.; Alvarez, J. M.; and Luo, P. 2021. SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS, 34:12077–12090.   
Xie, J.; Cai, Y.; Chen, J.; Xu, R.; Wang, J.; and Li, Q. 2024a. Knowledge-Augmented Visual Question Answering With Natural Language Explanation. IEEE Trans. Image Process.   
Xie, J.; Chen, J.; Liu, Z.; Cai, Y.; Huang, Q.; and Li, Q. 2024b. Video Question Generation for Dynamic Changes. IEEE Trans. Circuits Syst. Video Technol.   
Xie, J.; Zhou, Z.; Wu, Z.; Zhang, X.; Wang, J.; Cai, Y.; and Li, Q. 2024c. Automated Defect Report Generation for Enhanced Industrial Quality Control. In Proc. AAAI Conf. Artif. Intell., 19306–19314.   
Yin, B.; Zhang, X.; Hou, Q.; Sun, B.-Y.; Fan, D.-P.; and Van Gool, L. 2022. Camoformer: Masked separable attention for camouflaged object detection. arXiv:2212.06570.   
Yin, B.; Zhang, X.; Li, Z.; Liu, L.; Cheng, M.-M.; and Hou, Q. 2023. DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation. arXiv:2309.09668.   
Yu, C.; Gao, C.; Wang, J.; Yu, G.; Shen, C.; and Sang, N. 2021. Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation. Int. J. Comput. Vis., 129: 3051–3068.   
Yu, C.; Wang, J.; Peng, C.; Gao, C.; Yu, G.; and Sang, N. 2018. BiSeNet: Bilateral segmentation network for real-time semantic segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), 325–341.   
Yuan, Y.; Xie, J.; Chen, X.; and Wang, J. 2020. SegFix: Model-agnostic boundary refinement for segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII, 489–506. Springer.   
Yuan, Z.; Cao, J.; Li, Z.; Jiang, H.; and Wang, Z. 2024a. SD-MVS: Segmentation-Driven Deformation Multi-View Stereo with Spherical Refinement and EM Optimization. In Proc. AAAI Conf. Artif. Intell., volume 38, 6871–6880.   
Yuan, Z.; Cao, J.; Wang, Z.; and Li, Z. 2024b. Tsar-Mvs: Textureless-aware Segmentation and Correlative Refinement Guided Multi-View Stereo. Pattern Recognit., 154: 110565.

Yuan, Z.; Liu, C.; Shen, F.; Li, Z.; Mao, T.; and Wang, Z. 2024c. MSP-MVS: Multi-granularity Segmentation Prior Guided Multi-View Stereo. arXiv:2407.19323.   
Zeng, W.; Jin, S.; Liu, W.; Qian, C.; Luo, P.; Ouyang, W.; and Wang, X. 2022. Not all tokens are equal: Human-centric visual analysis via token clustering transformer. In CVPR, 11101–11111.   
Zhang, F.; Chen, G.; Wang, H.; Li, J.; and Zhang, C. 2023. Multi-scale video super-resolution transformer with polynomial approximation. IEEE Trans. Circuits Syst. Video Technol., 33(9): 4496–4506.   
Zhang, F.; Chen, G.; Wang, H.; and Zhang, C. 2024a. CF-DAN: Facial-expression recognition based on cross-fusion dual-attention network. Comput. Visual Media, 1–16.   
Zhang, Z.; Chen, M.; Xiao, S.; Peng, L.; Li, H.; Lin, B.; Li, P.; Wang, W.; Wu, B.; and Cai, D. 2024b. Pseudo Label Refinery for Unsupervised Domain Adaptation on Cross-dataset 3D Object Detection. In CVPR, 15291–15300.   
Zhao, H.; Qi, X.; Shen, X.; Shi, J.; and Jia, J. 2018. ICNet for real-time semantic segmentation on high-resolution images. In Proceedings of the European Conference on Computer Vision (ECCV), 405–420.   
Zhao, H.; Shi, J.; Qi, X.; Wang, X.; and Jia, J. 2017. Pyramid scene parsing network. In ICCV, 2881–2890.   
Zhou, Y.; Liang, D.; Chen, S.; Huang, S.-J.; Yang, S.; and Li, C. 2023. Improving lens flare removal with general-purpose pipeline and multiple light sources recovery. In Proc. IEEE/CVF Int. Conf. Comput. Vis., 12969–12979.   
Zhou, Y.; Song, L.; Wang, B.; and Chen, W. 2024. MetaGPT: Merging Large Language Models Using Model Exclusive Task Arithmetic. arXiv preprint arXiv:2406.11385.   
Zhu, H.; Zhu, Y.; Xiao, J.; Ma, Y.; Zhang, Y.; Li, J.; and Dai, F. 2024a. MISA: Mining Saliency-Aware Semantic Prior for Box Supervised Instance Segmentation. In IJCAI.   
Zhu, H.; Zhu, Y.; Xiao, J.; Xiao, T.; Ma, Y.; Zhang, Y.; and Dai, F. 2024b. Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation. arXiv:2412.03968.   
Zhu, Y.; Sapra, K.; Reda, F. A.; Shih, K. J.; Newsam, S.; Tao, A.; and Catanzaro, B. 2019. Improving semantic segmentation via video propagation and label relaxation. In CVPR, 8856–8865.