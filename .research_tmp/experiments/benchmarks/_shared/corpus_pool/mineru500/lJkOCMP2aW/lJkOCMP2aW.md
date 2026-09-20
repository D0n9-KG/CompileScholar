# PATHFORMER: MULTI-SCALE TRANSFORMERS WITH ADAPTIVE PATHWAYS FOR TIME SERIES FORECASTING

Peng Chen $^{1*}$ , Yingying Zhang $^{2}$ , Yunyao Cheng $^{3}$ , Yang Shu $^{1\dagger}$ , Yihang Wang $^{1*}$ , Qingsong Wen $^{2}$ , Bin Yang $^{1}$ , Chenjuan Guo $^{1}$

$^{1}$ East China Normal University, $^{2}$ Alibaba Group, $^{3}$ Aalborg University
{pchen,yhwang}@stu.ecnu.edu.cn,congrong.zyy@alibaba-inc.com
{yshu,cjguo,byang}@dase.ecnu.edu.cn,yunyaoc@cs.aau.dk
qingsongedu@gmail.com

# ABSTRACT

Transformers for time series forecasting mainly model time series from limited or fixed scales, making it challenging to capture different characteristics spanning various scales. We propose Pathformer, a multi-scale Transformer with adaptive pathways. It integrates both temporal resolution and temporal distance for multi-scale modeling. Multi-scale division divides the time series into different temporal resolutions using patches of various sizes. Based on the division of each scale, dual attention is performed over these patches to capture global correlations and local details as temporal dependencies. We further enrich the multi-scale Transformer with adaptive pathways, which adaptively adjust the multi-scale modeling process based on the varying temporal dynamics of the input, improving the accuracy and generalization of Pathformer. Extensive experiments on eleven real-world datasets demonstrate that Pathformer not only achieves state-of-the-art performance by surpassing all current models but also exhibits stronger generalization abilities under various transfer scenarios. The code is made available at https://github.com/decisionintelligence/pathformer.

# 1 INTRODUCTION

Time series forecasting is an essential function for various industries, such as energy, finance, traffic, logistics, and cloud computing (Chen et al., 2012; Cirstea et al., 2022b; Ma et al., 2014; Zhu et al., 2023; Pan et al., 2023; Pedersen et al., 2020), and is also a foundational building block for other time series analytics, e.g., outlier detection Campos et al. (2022); Kieu et al. (2022b). Motivated by its widespread application in sequence modeling and impressive success in various fields such as CV and NLP (Dosovitskiy et al., 2021; Brown et al., 2020), Transformer (Vaswani et al., 2017) receives emerging attention in time series (Wen et al., 2023; Wu et al., 2021; Chen et al., 2022; Liu et al., 2022c). Despite the growing performance, recent works have started to challenge the existing designs of Transformers for time series forecasting by proposing simpler linear models with better performance (Zeng et al., 2023). While the capabilities of Transformers are still promising in time series forecasting (Nie et al., 2023), it calls for better designs and adaptations to fulfill its potential.

Real-world time series exhibit diverse variations and fluctuations at different temporal scales. For instance, the utilization of CPU, GPU, and memory resources in cloud computing reveals unique temporal patterns spanning daily, monthly, and seasonal scales Pan et al. (2023). This calls for multi-scale modeling (Mozer, 1991; Ferreira et al., 2006) for time series forecasting, which extracts temporal features and dependencies from various scales of temporal intervals. There are two aspects to consider for multiple scales in time series: temporal resolution and temporal distance. Temporal resolution corresponds to how we view the time series in the model and determines the length of each temporal patch or unit considered for modeling. In Figure 1, the same time series can be divided

![](images/3b7a3511eca59ca0024263e9aeb6e821ec4b3238957e5414a39ad2a61afac5e5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Patch1"] --> B["Time Step 1"]
    C["Patch2"] --> D["Time Step 2"]
    E["Patch3"] --> F["Time Step 3"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
```
</details>

Figure 1: Left: Time series are divided into patches of varying sizes as temporal resolution. The intervals in blue, orange, and red represent different patch sizes. Right: Local details (black arrows) and global correlations (color arrows) are modeled through different temporal distances.

into small patches (blue) or large ones (yellow), leading to fine-grained or coarse-grained temporal characteristics. Temporal distance corresponds to how we explicitly model temporal dependencies and determines the distances between the time steps considered for temporal modeling. In Figure 1, the black arrows model the relations between nearby time steps, forming local details, while the colored arrows model time steps across long ranges, forming global correlations.

To further explore the capability of extracting correlations in Transformers for time series forecasting, in this paper, we focus on the aspect of enhancing multi-scale modeling with the Transformer architecture. Two main challenges limit the effective multi-scale modeling in Transformers. The first challenge is the incompleteness of multi-scale modeling. Viewing the data from different temporal resolutions implicitly influences the scale of the subsequent modeling process (Shabani et al., 2023). However, simply changing temporal resolutions cannot emphasize temporal dependencies in various ranges explicitly and efficiently. On the contrary, considering different temporal distances enables modeling dependencies from different ranges, such as global and local correlations (Li et al., 2019). However, the exact temporal distances of global and local intervals are influenced by the division of data, which is incomplete from a single view of temporal resolution. The second challenge is the fixed multi-scale modeling process. Although multi-scale modeling reaches a more complete understanding of time series, different series prefer different scales depending on their specific temporal characteristics and dynamics. For example, comparing the two series in Figure 1, the series above shows rapid fluctuations, which may imply more attention to fine-grained and short-term characteristics. The series below, on the contrary, may need more focus on coarse-grained and long-term modeling. The fixed multi-scale modeling for all data hinders the grasp of critical patterns of each time series, and manually tuning the optimal scales for a dataset or each time series is time-consuming or intractable. Solving these two challenges calls for adaptive multi-scale modeling, which adaptively models the current data from certain multiple scales.

Inspired by the above understanding of multi-scale modeling, we propose Multi-scale Transformers with Adaptive Pathways (Pathformer) for time series forecasting. To enable the ability of more complete multi-scale modeling, we propose a multi-scale Transformer block unifying multi-scale temporal resolution and temporal distance. Multi-scale division is proposed to divide the time series into patches of different sizes, forming views of diverse temporal resolutions. Based on each size of divided patches, dual attention encompassing inter-patch and intra-patch attention is proposed to capture temporal dependencies, with inter-patch attention capturing global correlations across patches and intra-patch attention capturing local details within individual patches. We further propose adaptive pathways to activate the multi-scale modeling capability and endow it with adaptive modeling characteristics. At each layer of the model, a multi-scale router adaptively selects specific sizes of patch division and the subsequent dual attention in the Transformer based on the input data, which controls the extraction of multi-scale characteristics. We equip the router with trend and seasonality decomposition to enhance its ability to grasp the temporal dynamics. The router works with an aggregator to adaptively combine multi-scale characteristics through weighted aggregation. The layer-by-layer routing and aggregation form the adaptive pathways of multi-scale modeling throughout the Transformer. To the best of our knowledge, this is the first study that introduces adaptive multi-scale modeling for time series forecasting. Specifically, we make the following contributions:

\- We propose a multi-scale Transformer architecture. It integrates the two perspectives of temporal resolution and temporal distance and equips the model with the capacity of a more complete multi-scale time series modeling.

- We further propose adaptive pathways within multi-scale Transformers. The multi-scale router with temporal decomposition works together with the aggregator to adaptively extract and aggregate multi-scale characteristics based on the temporal dynamics of input data, realizing adaptive multi-scale modeling for time series.   
- We conduct extensive experiments on different real-world datasets and achieve state-of-the-art prediction accuracy. Moreover, we perform transfer learning experiments across datasets to validate the strong generalization of the model.

# 2 RELATED WORK

Time Series Forecasting. Time series forecasting predicts future observations based on historical observations. Statistical modeling methods based on exponential smoothing and its different flavors serve as a reliable workhorse for time series forecasting (Hyndman & Khandakar, 2008; Li et al., 2022a). Among deep learning methods, GNNs model spatial dependency for correlated time series forecasting (Jin et al., 2023a; Wu et al., 2020; Zhao et al., 2024; Cheng et al., 2024; Miao et al., 2024; Cirstea et al., 2021). RNNs model the temporal dependency (Chung et al., 2014; Kieu et al., 2022a; Wen et al., 2017; Cirstea et al., 2019). DeepAR (Rangapuram et al., 2018) uses RNNs and autoregressive methods to predict future short-term series. CNN models use the temporal convolution to extract the sub-series features (Sen et al., 2019; Liu et al., 2022a; Wang et al., 2023). TimesNet (Wu et al., 2023a) transforms the original one-dimensional time series into a two-dimensional space and captures multi-period features through convolution. LLM-based methods also show effective performance in this field (Jin et al., 2023b; Zhou et al., 2023). Additionally, some methods are incorporating neural architecture search to discover optimal architectures(Wu et al., 2022; 2023b).

Transformer models have recently received emerging attention in time series forecasting (Wen et al., 2023). Informer (Zhou et al., 2021) proposes prob-sparse self-attention to select important keys, Triformer (Cirstea et al., 2022a) employs a triangular architecture, which manages to reduce the complexity. Autoformer (Wu et al., 2021) proposes auto-correlation mechanisms to replace self-attention for modeling temporal dynamics. FEDformer (Zhou et al., 2022) utilizes fourier transformation from the perspective of frequency to model temporal dynamics. However, researchers have raised concerns about the effectiveness of Transformers for time series forecasting, as simple linear models prove to be effective or even outperform previous Transformers (Li et al., 2022a; Challu et al., 2023; Zeng et al., 2023). Meanwhile, PatchTST (Nie et al., 2023) employs patching and channel independence with Transformers to effectively enhance the performance, showing that the Transformer architecture still has its potential with proper adaptation in time series forecasting.

Multi-scale Modeling for Time Series. Modeling multi-scale characteristics proves to be effective for correlation learning and feature extraction in the fields such as computer vision (Wang et al., 2021; Li et al., 2022b; Wang et al., 2022b) and multi-modal learning (Hu et al., 2020; Wang et al., 2022a), which is relatively less explored in time series forecasting. N-HiTS (Challu et al., 2023) employs multi-rate data sampling and hierarchical interpolation to model features of different resolutions. Pyraformer (Liu et al., 2022b) introduces a pyramid attention to extract features at different temporal resolutions. Scaleformer (Shabani et al., 2023) proposes a multi-scale framework, and the need to allocate a predictive model at different temporal resolutions results in higher model complexity. Different from these methods, which use fixed scales and cannot adaptively change the multi-scale modeling for different time series, we propose a multi-scale Transformer with adaptive pathways that adaptively model multi-scale characteristics based on diverse temporal dynamics.

# 3 METHODOLOGY

To effectively capture multi-scale characteristics, we propose multi-scale Transformers with adaptive pathways (named Pathformer). As depicted in Figure 2, the whole forecasting network is composed of Instance Norm, stacking of Adaptive Multi-Scale Blocks (AMS Blocks), and Predictor. Instance Norm (Kim et al., 2022) is a normalization technique employed to address the distribution shift between training and testing data. Predictor is a fully connected neural network, proposed due to its applicability to forecasting for long sequences (Zeng et al., 2023; Das et al., 2023).

The core of our design is the AMS Block for adaptive modeling of multi-scale characteristics, which consists of the multi-scale Transformer block and adaptive pathways. Inspired by the idea of patch-

![](images/57218da4470d37838dec701bd115da6b599a3e34b784e458b89acbcb21cdf64b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Predictor"] --> B["Adaptive Multi-Scale Block"]
    B --> C["Adaptive Multi-Scale Block"]
    C --> D["Adaptive Multi-Scale Block"]
    D --> E["Instance Norm"]
    E --> F["Original Time Series"]
    F --> G["Multi-Scale Router"]
    G --> H["PatchSize S1"]
    G --> I["PatchSize S2"]
    G --> J["PatchSize S3"]
    G --> K["..."]
    G --> L["PatchSize SM"]
    H --> M["Patch Size Selection(TopK)"]
    I --> M
    J --> M
    K --> M
    L --> M
    M --> N["Patch Division"]
    N --> O["patch X¹"]
    N --> P["..."]
    N --> Q["patch XP"]
    O --> R["Dual Attention"]
    P --> R
    Q --> R
    R --> S["Multi-Scale Aggregator"]
    S --> T["weight"]
    T --> U["Multi-Scale Transformer Block"]
    U --> V["Adaptive Multi-Scale Block"]
    V --> W["Adaptive Multi-Scale Block"]
    W --> X["Adaptive Multi-Scale Block"]
    X --> Y["Adaptive Multi-Scale Block"]
    Y --> Z["Adaptive Multi-Scale Block"]
    Z --> AA["Adaptive Multi-Scale Block"]
    AA --> AB["Adaptive Multi-Scale Block"]
    AB --> AC["Adaptive Multi-Scale Block"]
    AC --> AD["Adaptive Multi-Scale Block"]
    AD --> AE["Adaptive Multi-Scale Block"]
    AE --> AF["Adaptive Multi-Scale Block"]
    AF --> AG["Adaptive Multi-Scale Block"]
    AG --> AH["Adaptive Multi-Scale Block"]
    AH --> AI["Adaptive Multi-Scale Block"]
```
</details>

Figure 2: The architecture of Pathformer. The Multi-scale Transformer Block (MST Block) comprises patch division with multiple patch sizes and dual attention. The adaptive pathways select the patch sizes with the top K weights generated by the router to capture multi-scale characteristics, and the selected patch sizes are represented in blue. Then, the aggregator applies weighted aggregation to the characteristics obtained from the MST Block.

ing in Transformers (Nie et al., 2023), the multi-scale Transformer block integrates multi-scale temporal resolutions and distances by introducing patch division with multiple patch sizes and dual attention on the divided patches, equipping the model with the capability to comprehensively model multi-scale characteristics. Based on various options of multi-scale modeling in the Transformer block, adaptive pathways utilize the multi-scale modeling capability and endow it with adaptive modeling characteristics. A multi-scale router selects specific sizes of patch division and the subsequent dual attention in the Transformer based on the input data, which controls the extraction of multi-scale features. The router works with an aggregator to combine these multi-scale characteristics through weighted aggregation. The layer-by-layer routing and aggregation form the adaptive pathways of multi-scale modeling throughout the Transformer blocks. In the following parts, we describe the multi-scale Transformer block and the adaptive pathways of the AMS Block in detail.

# 3.1 MULTI-SCALE TRANSFORMER BLOCK

Multi-scale Division. For the simplicity of notations, we use a univariate time series for description, and the method can be easily extended to multivariate cases by considering each variable independently. In the multi-scale Transformer block, We define a collection of M patch size values as $S = \{S_{1}, \ldots, S_{M}\}$ , with each patch size S corresponding to a patch division operation. For the input time series $X \in R^{H \times d}$ , where H denotes the length of the time series and d denotes the dimension of features, each patch division operation with the patch size S divides X into P (with $P = H/S$ ) patches as $(X^{1}, X^{2}, \ldots, X^{P})$ , where each patch $X^{i} \in R^{S \times d}$ contains S time steps. Different patch sizes in the collection lead to various scales of divided patches and give various views of temporal resolutions for the input series. This multi-scale division works with the dual attention mechanism described below for multi-scale modeling.

Dual Attention. Based on the patch division of each scale, we propose dual attention to model temporal dependencies over the divided patches. To grasp temporal dependencies from different temporal distances, we utilize patch division as guidance for different temporal distances, and the dual attention mechanism consists of intra-patch attention within each divided patch and inter-patch attention across different patches, as shown in Figure 3(a).

Consider a set of patches $(X^{1}, X^{2}, \ldots, X^{P})$ divided with the patch size S, intra-patch attention establishes relationships between time steps within each patch. For the i-th patch $X^{i} \in R^{S \times d}$ , we first embed the patch along the feature dimension d to get $X_{intra}^{i} \in R^{S \times d_{m}}$ , where $d_{m}$ represents the dimension of embedding. Then we perform trainable linear transformations on $X_{intra}^{i}$ to obtain the key and value in attention operations, denoted as $K_{intra}^{i}$ , $V_{intra}^{i} \in R^{S \times d_{m}}$ . We employ a trainable query matrix $Q_{intra}^{i} \in R^{1 \times d_{m}}$ to merge the context of the patch and subsequently compute the

![](images/c0931543545b451ed7d4d3f1bb1f80c1505f0864c091b69f8aaa8ae6ff82b737.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Fusion"] --> B["Intra-patch attention"]
    A --> C["Inter-patch attention"]
    B --> D["Intra-patch embedding"]
    B --> E["Inter-patch embedding"]
    D --> F["Patch Division"]
    E --> F
    F --> G["Patch Size Pool"]
    G --> H["PatchSize S₁"]
    G --> I["PatchSize S₂"]
    G --> J["PatchSize Sₘ"]
    K["Concat"] --> L["Cross Attention"]
    L --> M["Linear"]
    L --> N["Linear"]
    L --> O["Linear"]
    M --> P["Initialize"]
    N --> P
    O --> P
    P --> Q["patch X¹"]
    P --> R["patch X²"]
    P --> S["patch X³"]
    Q --> T["..."]
    R --> U["..."]
    S --> V["..."]
    T --> W["..."]
    U --> X["..."]
    V --> Y["..."]
    W --> Z["..."]
    X --> AA["..."]
    Y --> AB["..."]
    Z --> AC["..."]
    AA --> AD["..."]
    AB --> AE["..."]
```
</details>

(a) Multi-Scale Transformer Block

![](images/c2987baa58e0fc60678b745f38dfda301db7e70858230ff05161dc65458ce4d7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Inter-patch attention"] --> B["Flatten"]
    B --> C["Self-Attention"]
    C --> D["Linear"]
    C --> E["Linear"]
    C --> F["Linear"]
    D --> G["Q"]
    E --> H["K"]
    F --> I["V"]
    G --> J["patch X¹"]
    H --> K["patch X²"]
    I --> L["patch X³"]
```
</details>

![](images/5ba2747224230ec9d0cb048d2ce5a3c1f4a95ad610052f66242e5e6a282d138e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Routing Path Selection"] --> B["Routing Weights Generation"]
    B --> C["+"]
    C --> D["Weighted Sum"]
    D --> E["Multi-kernel Average pooling"]
    E --> F["IDFT"]
    F --> G["TopK Frequency"]
    G --> H["DFT"]
    H --> I["..."]
    style A fill:#f9f,stroke:#333
    style I fill:#bbf,stroke:#333
```
</details>

(b) Multi-Scale Router   
Figure 3: (a) The structure of the Multi-Scale Transformer Block, which mainly consists of Patch Division, Inter-patch attention, and Intra-patch attention. (b) The structure of the Multi-Scale Router.

cross-attention between $Q_{intra}^{i}$ , $K_{intra}^{i}$ , $V_{intra}^{i}$ to model local details within the i-th patch:

$$
\mathrm{Attn} _ {\text { intra }} ^ {i} = \operatorname{Softmax} (Q _ {\text { intra }} ^ {i} (K _ {\text { intra }} ^ {i}) ^ {T} / \sqrt {d _ {m}}) V _ {\text { intra }} ^ {i}. \tag {1}
$$

After intra-patch attention, each patch has transitioned from its original input length of S to the length of 1. The attention results from all the patches are concatenated to produce the output of intra-attention on the divided patches as $Attn_{intra} \in R^{P \times d_{m}}$ , which represents the local details from nearby time steps in the time series:

$$
\mathrm{Attn} _ {\text { intra }} = \operatorname{Concat} \left(\mathrm{Attn} _ {\text { intra }} ^ {1}, \dots , \mathrm{Attn} _ {\text { intra }} ^ {P}\right). \tag {2}
$$

Inter-patch attention establishes relationships between patches to capture global correlations. For the patch-divided time series $X \in R^{P \times S \times d}$ , we first perform feature embedding along the feature dimension from d to $d_{m}$ and then rearrange the data to combine the two dimensions of patch quantity S and feature embedding $d_{m}$ , resulting in $X_{inter} \in R^{P \times d_{m}^{\prime}}$ , where $d_{m}^{\prime} = S \cdot d_{m}$ . After such embedding and rearranging process, the time steps within the same patch are combined, and thus we perform self-attention over $X_{inter}$ to model correlations between patches. Following the standard self-attention protocol, we obtain the query, key, and value through linear mapping on $X_{inter}$ , denoted as $Q_{inter}, K_{inter}, V_{inter} \in R^{P \times d_{m}^{\prime}}$ . Then, we compute the attention Attn $_{inter}$ , which involves interaction between patches and represents the global correlations of the time series:

$$
\mathrm{Attn} _ {\text {inter}} = \operatorname{Softmax} (Q _ {\text {inter}} (K _ {\text {inter}}) ^ {T} / \sqrt {d _ {m} ^ {\prime}}) V _ {\text {inter}}. \tag {3}
$$

To fuse global correlations and local details captured by dual attention, we rearrange the outputs of intra-patch attention to $Attn_{intra} \in R^{P \times S \times d_m}$ , performing linear transformations on the patch size dimension from 1 to S, to combine time steps in each patch, and then add it with inter-patch attention $Attn_{inter} \in R^{P \times S \times d_m}$ to obtain the final output of dual attention $Attn \in R^{P \times S \times d_m}$ .

Overall, the multi-scale division provides different views of the time series with different patch sizes, and the changing patch sizes further influence the dual attention, which models temporal dependencies from different distances guided by the patch division. These two components work together to enable multiple scales of temporal modeling in the Transformer.

# 3.2 ADAPTIVE PATHWAYS

The design of the multi-scale Transformer block equips the model with the capability of multiscale modeling. However, different series may prefer diverse scales, depending on their specific temporal characteristics and dynamics. Simply applying more scales may bring in redundant or useless signals, and manually tuning the optimal scales for a dataset or each time series is time-consuming or intractable. An ideal model needs to figure out such critical scales based on the input data for more effective modeling and better generalization of unseen data.

Pathways and Mixture of Experts are used to achieve adaptive modeling (Dean, 2021; Shazeer et al., 2016). Based on these concepts, we propose adaptive pathways based on multi-scale Transformer to model adaptive multi-scale, depicted in Figure 2. It contains two main components: the multi-scale router and the multi-scale aggregator. The multi-scale router selects specific sizes of patch division based on the input data, which activates specific parts in the Transformer and controls the extraction of multi-scale characteristics. The router works with the multi-scale aggregator to combine these characteristics through weighted aggregation, obtaining the output of the Transformer block.

Multi-Scale Router. The multi-scale router enables data-adaptive routing in the multi-scale Transformer, which selects the optimal sizes for patch division and thus controls the process of multi-scale modeling. Since the optimal or critical scales for each time series can be impacted by its complex inherent characteristics and dynamic patterns, like the periodicity and trend, we introduce a temporal decomposition module in the router that encompasses both seasonality and trend decomposition to extract periodicity and trend patterns, as illustrated in Figure 3(b).

Seasonality decomposition involves transforming the time series from the temporal domain into the frequency domain to extract the periodic patterns. We utilize the Discern Fourier Transform (DFT) (Cooley & Tukey, 1965), denoted as DFT( $\cdot$ ), to decompose the input X into Fourier basis and select the $K_{f}$ basis with the largest amplitudes to keep the sparsity of frequency domain. Then, we obtain the periodic patterns $X_{sea}$ through an inverse DFT, denoted as IDFT( $\cdot$ ). The process is as follows:

$$
\mathrm{X} _ {\text { sea }} = \mathrm{IDFT} (\{f _ {1}, \dots , f _ {K _ {f}} \}, A, \Phi), \tag {4}
$$

where $\Phi$ and A represent the phase and amplitude of each frequency from DFT(X), $\{f_{1},\ldots,f_{K_{f}}\}$ represents the frequencies with the top $K_{f}$ amplitudes. Trend decomposition uses different kernels of average pooling for moving averages to extract trend patterns based on the remaining part after the seasonality decomposition $X_{rem} = X - X_{sea}$ . For the results obtained from different kernels, a weighted operation is applied to obtain the representation of the trend component:

$$
\mathrm{X} _ {\text {trend}} = \operatorname{Softmax} (L (\mathrm{X} _ {\text {rem}})) \cdot \left(\operatorname{Avgpool} (\mathrm{X} _ {\text {rem}}) _ {\text {kernel} _ {1}}, \dots , \operatorname{Avgpool} (\mathrm{X} _ {\text {rem}}) _ {\text {kernel} _ {N}}\right), \tag {5}
$$

where $\operatorname{Avgpool}(\cdot)_{\operatorname{kernel}_{i}}$ is the pooling function with the i-th kernel, N corresponds to the number of kernels, $\operatorname{Softmax}(L(\cdot))$ controls the weights for the results from different kenerls. We add the seasonality pattern and trend pattern with the original input X, and then perform a linear mapping $\operatorname{Linear}(\cdot)$ to transform and merge them along the temporal dimension to get $X_{trans} \in R^{d}$ .

Based on the results $X_{trans}$ from temporal decomposition, the router employs a routing function to generate the pathway weights, which determines the patch sizes to choose for the current data. To avoid consistently selecting a few patch sizes, causing the corresponding scales to be repeatedly updated while neglecting other potentially useful scales in the multi-scale Transformer, we introduce noise terms to add randomness in the weight generation process. The whole process of generating pathway weights is as follows:

$$
R \left(\mathrm{X} _ {\text { trans }}\right) = \operatorname{Softmax} \left(\mathrm{X} _ {\text { trans }} W _ {r} + \epsilon \cdot \text { Softplus } \left(\mathrm{X} _ {\text { trans }} W _ {\text { noise }}\right)\right), \epsilon \sim \mathcal {N} (0, 1), \tag {6}
$$

where $R(\cdot)$ represents the whole routing function, $W_{r}$ and $W_{noise} \in R^{d \times M}$ are learnable parameters for weight generation, with d denoting the feature dimension of $X_{trans}$ and M denoting the number of patch sizes. To introduce sparsity in the routing and encourage the selection of critical scales, we perform top K selection on the pathway weights, keeping the top K pathway weights and setting the rest weights as 0, and denote the final result as $\bar{R}(X_{\text{trans}})$ .

Multi-Scale Aggregator. Each dimension of the generated pathway weights $\bar{R}(X_{\mathrm{trans}}) \in \mathbb{R}^{M}$ correspond to a patch size in the multi-scale Transformer, with $\bar{R}(X_{\mathrm{trans}})_{i} > 0$ indicating performing this size $S_{i}$ of patch division and the dual attention and $\bar{R}(X_{\mathrm{trans}})_{i} = 0$ indicating ignoring this patch size for the current data. Let $X_{out}^{i}$ denote the output of the multi-scale Transformer with the patch size $S_{i}$ , due to the varying temporal dimensions produced by different patch sizes, the aggregator first performs a transformation function $T_{i}(\cdot)$ to align the temporal dimension from different scales. Then, the aggregator performs weighted aggregation for the multi-scale outputs based on the pathway weights to get the final output of this AMS block:

$$
\mathrm{X} _ {\text { out }} = \sum_ {i = 1} ^ {M} \mathcal {I} (\bar {R} (\mathrm{X} _ {\text { trans }}) _ {i} > 0) R (\mathrm{X} _ {\text { trans }}) _ {i} T _ {i} (\mathrm{X} _ {\text { out }} ^ {i}). \tag {7}
$$

$\mathcal{I}(\bar{R}(X_{\text{trans}})_i > 0)$ is the indicator function which outputs 1 when $\bar{R}(X_{\text{trans}})_i > 0$ , and otherwise outputs 0, indicating that only the top K patch sizes and the corresponding outputs from the Transformer are considered or needed during aggregation.

# 4 EXPERIMENTS

# 4.1 TIME SERIES FORECASTING

Datasets. We conduct experiments on nine real-world datasets to assess the performance of Path-former, encompassing a range of domains, including electricity transportation, weather forecasting, and cloud computing. These datasets include ETT (ETTh1, ETTh2, ETTm1, ETTm2), Weather, Electricity, Traffic, ILI, and Cloud Cluster (Cluster-A, Cluster-B, Cluster-C).

Baselines and Metrics. We choose some state-of-the-art models to serve as baselines, including PatchTST (Nie et al., 2023), NLinear (Zeng et al., 2023), Scaleformer (Shabani et al., 2023), TIDE (Das et al., 2023), FEDformer (Zhou et al., 2022), Pyraformer (Liu et al., 2022b), and Autoformer (Wu et al., 2021). To ensure fair comparisons, all models follow the same input length ( $H = 36$ for the ILI dataset and $H = 96$ for others) and prediction length ( $F \in \{24, 49, 96, 192\}$ for Cloud Cluster datasets, $F \in \{24, 36, 48, 60\}$ for ILI dataset and $F \in \{96, 192, 336, 720\}$ for others). We select two common metrics in time series forecasting: Mean Absolute Error (MAE) and Mean Squared Error (MSE).

Implementation Details. Pathformer utilizes the Adam optimizer (Kingma & Ba, 2015) with a learning rate set at $10^{-3}$ . The default loss function employed is L1 Loss, and we implement early stopping within 10 epochs during the training process. All experiments are conducted using PyTorch and executed on an NVIDIA A800 80GB GPU. Pathformer is composed of 3 Adaptive Multi-Scale Blocks (AMS Blocks). Each AMS Block contains 4 different patch sizes. These patch sizes are selected from a pool of commonly used options, namely $\{2, 3, 6, 12, 16, 24, 32\}$ .

Main Results. Table 1 shows the prediction results of multivariable time series forecasting, where Pathformer stands out with the best performance in 81 cases and the second-best in 5 cases out of the overall 88 cases. Compared with the second-best baseline, PatchTST, Pathformer demonstrates a significant improvement, with an impressive 8.1% reduction in MSE and a 6.4% reduction in MAE. Compared with the strong linear models NLinear, Pathformer also outperforms them comprehensively, especially on large datasets such as Electricity and Traffic. This demonstrates the potential of Transformer architecture for time series forecasting. Compared with the multi-scale models Pyraformer and Scaleformer, Pathformer exhibits good performance improvements, with a substantial 36.4% reduction in MSE and a 19.1% reduction in MAE. This illustrates that the proposed comprehensive modeling from both temporal resolution and temporal distance with adaptive pathways is more effective for multi-scale modeling.

# 4.2 TRANSFER LEARNING

Experimental Setting. To assess the transferability of Pathformer, we benchmark it against three baselines: PatchTST, FEDformer, and Autoformer, devising two distinct transfer experiments. In the context of evaluating transferability across different datasets, models initially undergo pre-training on the ETTh1 and ETTm1. Subsequently, we fine-tune them using the ETTh2 and ETTm2. For assessing transferability towards future data, models are pre-trained on the first 70% of the training data sourced from three clusters: Cluster-A, Cluster-B, and Cluster-C. This pre-training is followed by fine-tuning the remaining 30% of the training data specific to each cluster. In terms of methodology for baselines, we explore two approaches: direct prediction (zero-shot) and full-tuning. Deviating from these approaches, Pathformer integrates a part-tuning strategy. In this approach, specific parameters, like those of the router network, undergo fine-tuning, resulting in a significant reduction in computational resource demands.

Transfer Learning Results. Table 2 presents the outcomes of our transfer learning evaluation. Across both direct prediction and full-tuning methods, Pathformer surpasses the baseline models, highlighting its enhanced generalization and transferability. One of the key strengths of Pathformer lies in its adaptive capacity to select varying scales for different temporal dynamics. This adaptability allows it to effectively capture complex temporal patterns present in diverse datasets, consequently demonstrating superior generalization and transferability. Part-tuning is a lightweight fine-tuning method that demands fewer computational resources and reduces training time on average by $52\%$ , while still achieving prediction accuracy nearly comparable to Pathformer full-tuning. Moreover, it outperforms the full-tuning of other baseline models on the majority of datasets. This demonstrates that Pathformer can provide effective lightweight transfer learning for time series forecasting.

Table 1: Multivariate time series forecasting results. The input length H = 96 (H = 36 for ILI). The best results are highlighted in bold, and the second-best results are underlined. 

<table><tr><td colspan="2">Method</td><td colspan="2">Pathformer</td><td colspan="2">PatchTST</td><td colspan="2">NLinear</td><td colspan="2">Scaleformer</td><td colspan="2">TiDE</td><td colspan="2">FEDformer</td><td colspan="2">Pyraformer</td><td colspan="2">Autoformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.382</td><td>0.400</td><td>0.394</td><td>0.408</td><td>0.386</td><td>0.392</td><td>0.396</td><td>0.440</td><td>0.427</td><td>0.450</td><td>0.376</td><td>0.419</td><td>0.664</td><td>0.612</td><td>0.449</td><td>0.459</td></tr><tr><td>192</td><td>0.440</td><td>0.427</td><td>0.446</td><td>0.438</td><td>0.440</td><td>0.430</td><td>0.434</td><td>0.460</td><td>0.472</td><td>0.486</td><td>0.420</td><td>0.448</td><td>0.790</td><td>0.681</td><td>0.500</td><td>0.482</td></tr><tr><td>336</td><td>0.454</td><td>0.432</td><td>0.485</td><td>0.455</td><td>0.480</td><td>0.443</td><td>0.462</td><td>0.476</td><td>0.527</td><td>0.527</td><td>0.459</td><td>0.465</td><td>0.891</td><td>0.738</td><td>0.521</td><td>0.496</td></tr><tr><td>720</td><td>0.479</td><td>0.461</td><td>0.495</td><td>0.474</td><td>0.486</td><td>0.472</td><td>0.494</td><td>0.500</td><td>0.644</td><td>0.605</td><td>0.506</td><td>0.507</td><td>0.963</td><td>0.782</td><td>0.514</td><td>0.512</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.279</td><td>0.331</td><td>0.294</td><td>0.343</td><td>0.290</td><td>0.339</td><td>0.364</td><td>0.407</td><td>0.304</td><td>0.359</td><td>0.346</td><td>0.388</td><td>0.645</td><td>0.597</td><td>0.358</td><td>0.397</td></tr><tr><td>192</td><td>0.349</td><td>0.380</td><td>0.378</td><td>0.394</td><td>0.379</td><td>0.395</td><td>0.466</td><td>0.458</td><td>0.394</td><td>0.422</td><td>0.429</td><td>0.439</td><td>0.788</td><td>0.683</td><td>0.456</td><td>0.452</td></tr><tr><td>336</td><td>0.348</td><td>0.382</td><td>0.382</td><td>0.410</td><td>0.421</td><td>0.431</td><td>0.479</td><td>0.476</td><td>0.385</td><td>0.421</td><td>0.496</td><td>0.487</td><td>0.907</td><td>0.747</td><td>0.482</td><td>0.486</td></tr><tr><td>720</td><td>0.398</td><td>0.424</td><td>0.412</td><td>0.433</td><td>0.436</td><td>0.453</td><td>0.487</td><td>0.492</td><td>0.463</td><td>0.475</td><td>0.463</td><td>0.474</td><td>0.963</td><td>0.783</td><td>0.515</td><td>0.511</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.316</td><td>0.346</td><td>0.324</td><td>0.361</td><td>0.339</td><td>0.369</td><td>0.355</td><td>0.398</td><td>0.356</td><td>0.381</td><td>0.379</td><td>0.419</td><td>0.543</td><td>0.510</td><td>0.505</td><td>0.475</td></tr><tr><td>192</td><td>0.366</td><td>0.370</td><td>0.362</td><td>0.383</td><td>0.379</td><td>0.386</td><td>0.428</td><td>0.455</td><td>0.391</td><td>0.399</td><td>0.426</td><td>0.441</td><td>0.557</td><td>0.537</td><td>0.553</td><td>0.496</td></tr><tr><td>336</td><td>0.386</td><td>0.394</td><td>0.390</td><td>0.402</td><td>0.411</td><td>0.407</td><td>0.524</td><td>0.487</td><td>0.424</td><td>0.423</td><td>0.445</td><td>0.459</td><td>0.754</td><td>0.655</td><td>0.621</td><td>0.537</td></tr><tr><td>720</td><td>0.460</td><td>0.432</td><td>0.461</td><td>0.438</td><td>0.478</td><td>0.442</td><td>0.558</td><td>0.517</td><td>0.480</td><td>0.456</td><td>0.543</td><td>0.490</td><td>0.908</td><td>0.724</td><td>0.671</td><td>0.561</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.170</td><td>0.248</td><td>0.177</td><td>0.260</td><td>0.177</td><td>0.257</td><td>0.182</td><td>0.275</td><td>0.182</td><td>0.264</td><td>0.203</td><td>0.287</td><td>0.435</td><td>0.507</td><td>0.255</td><td>0.339</td></tr><tr><td>192</td><td>0.238</td><td>0.295</td><td>0.248</td><td>0.306</td><td>0.241</td><td>0.297</td><td>0.251</td><td>0.318</td><td>0.256</td><td>0.323</td><td>0.269</td><td>0.328</td><td>0.730</td><td>0.673</td><td>0.281</td><td>0.340</td></tr><tr><td>336</td><td>0.293</td><td>0.331</td><td>0.304</td><td>0.342</td><td>0.302</td><td>0.337</td><td>0.340</td><td>0.375</td><td>0.313</td><td>0.354</td><td>0.325</td><td>0.366</td><td>1.201</td><td>0.845</td><td>0.339</td><td>0.372</td></tr><tr><td>720</td><td>0.390</td><td>0.389</td><td>0.403</td><td>0.397</td><td>0.405</td><td>0.396</td><td>0.435</td><td>0.433</td><td>0.419</td><td>0.410</td><td>0.421</td><td>0.415</td><td>3.625</td><td>1.451</td><td>0.433</td><td>0.432</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.156</td><td>0.192</td><td>0.177</td><td>0.218</td><td>0.168</td><td>0.208</td><td>0.288</td><td>0.365</td><td>0.202</td><td>0.261</td><td>0.238</td><td>0.314</td><td>0.896</td><td>0.556</td><td>0.249</td><td>0.329</td></tr><tr><td>192</td><td>0.206</td><td>0.240</td><td>0.224</td><td>0.258</td><td>0.217</td><td>0.255</td><td>0.368</td><td>0.425</td><td>0.242</td><td>0.298</td><td>0.275</td><td>0.329</td><td>0.622</td><td>0.624</td><td>0.325</td><td>0.370</td></tr><tr><td>336</td><td>0.254</td><td>0.282</td><td>0.277</td><td>0.297</td><td>0.267</td><td>0.292</td><td>0.447</td><td>0.469</td><td>0.287</td><td>0.335</td><td>0.339</td><td>0.377</td><td>0.739</td><td>0.753</td><td>0.351</td><td>0.391</td></tr><tr><td>720</td><td>0.340</td><td>0.336</td><td>0.350</td><td>0.345</td><td>0.351</td><td>0.346</td><td>0.640</td><td>0.574</td><td>0.351</td><td>0.386</td><td>0.389</td><td>0.409</td><td>1.004</td><td>0.934</td><td>0.415</td><td>0.426</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.145</td><td>0.236</td><td>0.180</td><td>0.264</td><td>0.185</td><td>0.266</td><td>0.182</td><td>0.297</td><td>0.194</td><td>0.277</td><td>0.186</td><td>0.302</td><td>0.386</td><td>0.449</td><td>0.196</td><td>0.313</td></tr><tr><td>192</td><td>0.167</td><td>0.256</td><td>0.188</td><td>0.275</td><td>0.189</td><td>0.276</td><td>0.188</td><td>0.300</td><td>0.193</td><td>0.280</td><td>0.197</td><td>0.311</td><td>0.386</td><td>0.443</td><td>0.211</td><td>0.324</td></tr><tr><td>336</td><td>0.186</td><td>0.275</td><td>0.206</td><td>0.291</td><td>0.204</td><td>0.289</td><td>0.210</td><td>0.324</td><td>0.206</td><td>0.296</td><td>0.213</td><td>0.328</td><td>0.378</td><td>0.443</td><td>0.214</td><td>0.327</td></tr><tr><td>720</td><td>0.231</td><td>0.309</td><td>0.247</td><td>0.328</td><td>0.245</td><td>0.319</td><td>0.232</td><td>0.339</td><td>0.242</td><td>0.328</td><td>0.233</td><td>0.344</td><td>0.376</td><td>0.445</td><td>0.236</td><td>0.342</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.587</td><td>0.758</td><td>1.724</td><td>0.843</td><td>2.725</td><td>1.069</td><td>0.232</td><td>0.339</td><td>2.154</td><td>0.992</td><td>2.624</td><td>1.095</td><td>1.420</td><td>2.012</td><td>2.906</td><td>1.182</td></tr><tr><td>36</td><td>1.429</td><td>0.711</td><td>1.536</td><td>0.752</td><td>2.530</td><td>1.032</td><td>2.745</td><td>1.075</td><td>2.436</td><td>1.042</td><td>2.516</td><td>1.021</td><td>7.394</td><td>2.031</td><td>2.585</td><td>1.038</td></tr><tr><td>48</td><td>1.505</td><td>0.742</td><td>1.821</td><td>0.832</td><td>2.510</td><td>1.031</td><td>2.748</td><td>1.072</td><td>2.532</td><td>1.051</td><td>2.505</td><td>1.041</td><td>7.551</td><td>2.057</td><td>3.024</td><td>1.145</td></tr><tr><td>60</td><td>1.731</td><td>0.799</td><td>1.923</td><td>0.842</td><td>2.492</td><td>1.026</td><td>2.793</td><td>1.059</td><td>2.748</td><td>1.142</td><td>2.742</td><td>1.122</td><td>7.662</td><td>2.100</td><td>2.761</td><td>1.114</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.479</td><td>0.283</td><td>0.492</td><td>0.324</td><td>0.645</td><td>0.388</td><td>2.678</td><td>1.071</td><td>0.568</td><td>0.352</td><td>0.576</td><td>0.359</td><td>2.085</td><td>0.468</td><td>0.597</td><td>0.371</td></tr><tr><td>192</td><td>0.484</td><td>0.292</td><td>0.487</td><td>0.303</td><td>0.599</td><td>0.365</td><td>0.564</td><td>0.351</td><td>0.612</td><td>0.371</td><td>0.610</td><td>0.380</td><td>0.867</td><td>0.467</td><td>0.607</td><td>0.382</td></tr><tr><td>336</td><td>0.503</td><td>0.299</td><td>0.505</td><td>0.317</td><td>0.606</td><td>0.367</td><td>0.570</td><td>0.349</td><td>0.605</td><td>0.374</td><td>0.608</td><td>0.375</td><td>0.869</td><td>0.469</td><td>0.623</td><td>0.387</td></tr><tr><td>720</td><td>0.537</td><td>0.322</td><td>0.542</td><td>0.337</td><td>0.645</td><td>0.388</td><td>0.576</td><td>0.349</td><td>0.647</td><td>0.410</td><td>0.621</td><td>0.375</td><td>0.881</td><td>0.473</td><td>0.639</td><td>0.395</td></tr><tr><td rowspan="4">Cluster-A</td><td>24</td><td>0.100</td><td>0.205</td><td>0.126</td><td>0.234</td><td>0.134</td><td>0.235</td><td>0.128</td><td>0.247</td><td>0.128</td><td>0.244</td><td>0.131</td><td>0.260</td><td>0.131</td><td>0.268</td><td>0.372</td><td>0.461</td></tr><tr><td>48</td><td>0.160</td><td>0.264</td><td>0.208</td><td>0.302</td><td>0.214</td><td>0.310</td><td>0.182</td><td>0.319</td><td>0.192</td><td>0.299</td><td>0.175</td><td>0.307</td><td>0.170</td><td>0.311</td><td>0.390</td><td>0.471</td></tr><tr><td>96</td><td>0.227</td><td>0.321</td><td>0.313</td><td>0.372</td><td>0.335</td><td>0.410</td><td>0.274</td><td>0.328</td><td>0.247</td><td>0.338</td><td>0.293</td><td>0.349</td><td>0.243</td><td>0.375</td><td>0.466</td><td>0.514</td></tr><tr><td>192</td><td>0.349</td><td>0.400</td><td>0.452</td><td>0.453</td><td>0.442</td><td>0.452</td><td>0.372</td><td>0.451</td><td>0.356</td><td>0.422</td><td>0.350</td><td>0.439</td><td>0.378</td><td>0.437</td><td>0.585</td><td>0.584</td></tr><tr><td rowspan="4">Cluster-B</td><td>24</td><td>0.121</td><td>0.224</td><td>0.126</td><td>0.237</td><td>0.130</td><td>0.241</td><td>0.125</td><td>0.241</td><td>0.128</td><td>0.240</td><td>0.128</td><td>0.243</td><td>0.129</td><td>0.263</td><td>0.242</td><td>0.369</td></tr><tr><td>48</td><td>0.172</td><td>0.270</td><td>0.183</td><td>0.290</td><td>0.173</td><td>0.285</td><td>0.164</td><td>0.280</td><td>0.165</td><td>0.288</td><td>0.156</td><td>0.287</td><td>0.168</td><td>0.296</td><td>0.299</td><td>0.425</td></tr><tr><td>96</td><td>0.242</td><td>0.322</td><td>0.272</td><td>0.352</td><td>0.281</td><td>0.365</td><td>0.252</td><td>0.342</td><td>0.244</td><td>0.334</td><td>0.277</td><td>0.389</td><td>0.315</td><td>0.436</td><td>0.366</td><td>0.471</td></tr><tr><td>192</td><td>0.437</td><td>0.427</td><td>0.476</td><td>0.461</td><td>0.479</td><td>0.456</td><td>0.438</td><td>0.447</td><td>0.452</td><td>0.467</td><td>0.414</td><td>0.478</td><td>0.389</td><td>0.485</td><td>0.597</td><td>0.563</td></tr><tr><td rowspan="4">Cluster-C</td><td>24</td><td>0.064</td><td>0.169</td><td>0.075</td><td>0.188</td><td>0.100</td><td>0.205</td><td>0.074</td><td>0.204</td><td>0.082</td><td>0.199</td><td>0.076</td><td>0.212</td><td>0.107</td><td>0.247</td><td>0.189</td><td>0.341</td></tr><tr><td>48</td><td>0.102</td><td>0.218</td><td>0.118</td><td>0.241</td><td>0.163</td><td>0.286</td><td>0.110</td><td>0.242</td><td>0.121</td><td>0.266</td><td>0.108</td><td>0.246</td><td>0.142</td><td>0.284</td><td>0.210</td><td>0.363</td></tr><tr><td>96</td><td>0.162</td><td>0.276</td><td>0.188</td><td>0.305</td><td>0.245</td><td>0.318</td><td>0.177</td><td>0.321</td><td>0.201</td><td>0.305</td><td>0.171</td><td>0.323</td><td>0.181</td><td>0.328</td><td>0.289</td><td>0.421</td></tr><tr><td>192</td><td>0.304</td><td>0.369</td><td>0.354</td><td>0.413</td><td>0.375</td><td>0.457</td><td>0.326</td><td>0.428</td><td>0.341</td><td>0.424</td><td>0.338</td><td>0.453</td><td>0.332</td><td>0.396</td><td>0.419</td><td>0.511</td></tr></table>

Table 2: Transfer Learning results. The best results are in bold, and the second results are underlined. 

<table><tr><td rowspan="2" colspan="2">Mdoels</td><td rowspan="2" colspan="2">Predict</td><td rowspan="2" colspan="2">Pathformer Part-tuning</td><td rowspan="2" colspan="2">Full-tuning</td><td colspan="4">PatchTST</td><td colspan="4">FEDformer</td><td colspan="4">Autoformer</td></tr><tr><td colspan="2">Predict</td><td colspan="2">Full-tuning</td><td colspan="2">Predict</td><td colspan="2">Full-tuning</td><td colspan="2">Predict</td><td colspan="2">Full-tuning</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.340</td><td>0.369</td><td>0.287</td><td>0.333</td><td>0.276</td><td>0.328</td><td>0.346</td><td>0.369</td><td>0.287</td><td>0.337</td><td>0.420</td><td>0.449</td><td>0.326</td><td>0.337</td><td>0.397</td><td>0.439</td><td>0.342</td><td>0.386</td></tr><tr><td>192</td><td>0.411</td><td>0.406</td><td>0.358</td><td>0.382</td><td>0.350</td><td>0.376</td><td>0.422</td><td>0.420</td><td>0.366</td><td>0.385</td><td>0.475</td><td>0.475</td><td>0.409</td><td>0.430</td><td>0.543</td><td>0.511</td><td>0.415</td><td>0.428</td></tr><tr><td>336</td><td>0.384</td><td>0.401</td><td>0.342</td><td>0.384</td><td>0.337</td><td>0.374</td><td>0.408</td><td>0.419</td><td>0.377</td><td>0.405</td><td>0.416</td><td>0.446</td><td>0.378</td><td>0.416</td><td>0.521</td><td>0.515</td><td>0.415</td><td>0.442</td></tr><tr><td>720</td><td>0.450</td><td>0.448</td><td>0.416</td><td>0.437</td><td>0.401</td><td>0.426</td><td>0.479</td><td>0.467</td><td>0.410</td><td>0.432</td><td>0.529</td><td>0.517</td><td>0.46</td><td>0.487</td><td>0.694</td><td>0.602</td><td>0.452</td><td>0.469</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.220</td><td>0.294</td><td>0.181</td><td>0.260</td><td>0.172</td><td>0.251</td><td>0.189</td><td>0.284</td><td>0.177</td><td>0.261</td><td>0.256</td><td>0.378</td><td>0.201</td><td>0.285</td><td>0.331</td><td>0.406</td><td>0.212</td><td>0.293</td></tr><tr><td>192</td><td>0.258</td><td>0.306</td><td>0.240</td><td>0.299</td><td>0.237</td><td>0.294</td><td>0.263</td><td>0.322</td><td>0.243</td><td>0.304</td><td>0.427</td><td>0.441</td><td>0.266</td><td>0.324</td><td>0.435</td><td>0.461</td><td>0.275</td><td>0.331</td></tr><tr><td>336</td><td>0.325</td><td>0.350</td><td>0.305</td><td>0.339</td><td>0.302</td><td>0.334</td><td>0.332</td><td>0.365</td><td>0.305</td><td>0.339</td><td>0.429</td><td>0.448</td><td>0.335</td><td>0.369</td><td>0.506</td><td>0.501</td><td>0.333</td><td>0.370</td></tr><tr><td>720</td><td>0.422</td><td>0.408</td><td>0.406</td><td>0.398</td><td>0.391</td><td>0.392</td><td>0.429</td><td>0.419</td><td>0.405</td><td>0.395</td><td>0.530</td><td>0.503</td><td>0.423</td><td>0.417</td><td>0.680</td><td>0.573</td><td>0.444</td><td>0.433</td></tr><tr><td rowspan="4">Cluster-A</td><td>24</td><td>0.121</td><td>0.223</td><td>0.100</td><td>0.205</td><td>0.097</td><td>0.202</td><td>0.143</td><td>0.250</td><td>0.115</td><td>0.221</td><td>0.200</td><td>0.326</td><td>0.171</td><td>0.298</td><td>0.382</td><td>0.471</td><td>0.349</td><td>0.445</td></tr><tr><td>48</td><td>0.186</td><td>0.281</td><td>0.159</td><td>0.261</td><td>0.144</td><td>0.254</td><td>0.231</td><td>0.322</td><td>0.192</td><td>0.289</td><td>0.240</td><td>0.360</td><td>0.219</td><td>0.342</td><td>0.372</td><td>0.463</td><td>0.362</td><td>0.450</td></tr><tr><td>96</td><td>0.249</td><td>0.334</td><td>0.215</td><td>0.313</td><td>0.193</td><td>0.302</td><td>0.350</td><td>0.396</td><td>0.290</td><td>0.359</td><td>0.326</td><td>0.418</td><td>0.299</td><td>0.392</td><td>0.395</td><td>0.490</td><td>0.375</td><td>0.432</td></tr><tr><td>192</td><td>0.372</td><td>0.416</td><td>0.312</td><td>0.381</td><td>0.292</td><td>0.371</td><td>0.524</td><td>0.491</td><td>0.406</td><td>0.433</td><td>0.381</td><td>0.463</td><td>0.338</td><td>0.432</td><td>0.948</td><td>0.761</td><td>0.592</td><td>0.602</td></tr><tr><td rowspan="4">Cluster-B</td><td>24</td><td>0.140</td><td>0.243</td><td>0.120</td><td>0.226</td><td>0.117</td><td>0.221</td><td>0.145</td><td>0.248</td><td>0.124</td><td>0.231</td><td>0.167</td><td>0.283</td><td>0.147</td><td>0.271</td><td>0.226</td><td>0.342</td><td>0.192</td><td>0.318</td></tr><tr><td>48</td><td>0.202</td><td>0.298</td><td>0.174</td><td>0.275</td><td>0.170</td><td>0.270</td><td>0.207</td><td>0.306</td><td>0.178</td><td>0.282</td><td>0.225</td><td>0.310</td><td>0.162</td><td>0.283</td><td>0.247</td><td>0.361</td><td>0.234</td><td>0.354</td></tr><tr><td>96</td><td>0.296</td><td>0.357</td><td>0.253</td><td>0.327</td><td>0.244</td><td>0.321</td><td>0.298</td><td>0.365</td><td>0.264</td><td>0.242</td><td>0.347</td><td>0.427</td><td>0.318</td><td>0.408</td><td>0.307</td><td>0.430</td><td>0.280</td><td>0.399</td></tr><tr><td>192</td><td>0.464</td><td>0.468</td><td>0.441</td><td>0.425</td><td>0.425</td><td>0.420</td><td>0.529</td><td>0.495</td><td>0.471</td><td>0.463</td><td>0.528</td><td>0.497</td><td>0.434</td><td>0.478</td><td>0.618</td><td>0.614</td><td>0.584</td><td>0.578</td></tr><tr><td rowspan="4">Cluster-C</td><td>24</td><td>0.069</td><td>0.173</td><td>0.064</td><td>0.166</td><td>0.062</td><td>0.165</td><td>0.074</td><td>0.184</td><td>0.072</td><td>0.182</td><td>0.109</td><td>0.243</td><td>0.097</td><td>0.229</td><td>0.212</td><td>0.344</td><td>0.194</td><td>0.332</td></tr><tr><td>48</td><td>0.144</td><td>0.254</td><td>0.104</td><td>0.219</td><td>0.101</td><td>0.215</td><td>0.138</td><td>0.246</td><td>0.115</td><td>0.233</td><td>0.150</td><td>0.285</td><td>0.118</td><td>0.260</td><td>0.228</td><td>0.366</td><td>0.214</td><td>0.362</td></tr><tr><td>96</td><td>0.174</td><td>0.284</td><td>0.166</td><td>0.275</td><td>0.162</td><td>0.272</td><td>0.194</td><td>0.303</td><td>0.182</td><td>0.298</td><td>0.228</td><td>0.342</td><td>0.190</td><td>0.325</td><td>0.281</td><td>0.436</td><td>0.263</td><td>0.405</td></tr><tr><td>192</td><td>0.327</td><td>0.386</td><td>0.316</td><td>0.374</td><td>0.301</td><td>0.365</td><td>0.376</td><td>0.413</td><td>0.349</td><td>0.407</td><td>0.344</td><td>0.444</td><td>0.332</td><td>0.441</td><td>0.508</td><td>0.537</td><td>0.417</td><td>0.507</td></tr></table>

# 4.3 ABLATION STUDIES

To ascertain the impact of different modules within Pathformer, we perform ablation studies focusing on inter-patch attention, intra-patch attention, time series decomposition, and Pathways. The W/O Pathways configuration entails using all patch sizes from the patch size pool for every dataset, eliminating adaptive selection. Table 3 illustrates the unique impact of each module. The influence of Pathways is significant; omitting them results in a marked decrease in prediction accuracy. This emphasizes the criticality of optimizing the mix of patch sizes to extract multi-scale characteristics, thus markedly improving the model's prediction accuracy. Regarding efficiency, intra-patch attention is notably adept at discerning local patterns, contrasting with inter-patch attention which primarily captures wider global patterns. The time series decomposition module decomposes trend

Table 3: Ablation study. W/O Inter, W/O Intra, W/O Decompose represent removing the inter-patch attention, intra-patch attention, and time series decomposition, respectively. 

<table><tr><td colspan="2">Models</td><td colspan="2">W/O Inter</td><td colspan="2">W/O Intra</td><td colspan="2">W/O Decompose</td><td colspan="2">W/O Pathways</td><td colspan="2">Pathformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.162</td><td>0.196</td><td>0.170</td><td>0.203</td><td>0.162</td><td>0.198</td><td>0.168</td><td>0.204</td><td>0.156</td><td>0.192</td></tr><tr><td>192</td><td>0.219</td><td>0.248</td><td>0.220</td><td>0.249</td><td>0.212</td><td>0.244</td><td>0.219</td><td>0.250</td><td>0.206</td><td>0.240</td></tr><tr><td>336</td><td>0.262</td><td>0.290</td><td>0.272</td><td>0.292</td><td>0.256</td><td>0.285</td><td>0.269</td><td>0.290</td><td>0.254</td><td>0.282</td></tr><tr><td>720</td><td>0.350</td><td>0.349</td><td>0.358</td><td>0.357</td><td>0.344</td><td>0.340</td><td>0.349</td><td>0.348</td><td>0.340</td><td>0.336</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.166</td><td>0.259</td><td>0.182</td><td>0.264</td><td>0.152</td><td>0.244</td><td>0.168</td><td>0.256</td><td>0.145</td><td>0.236</td></tr><tr><td>192</td><td>0.185</td><td>0.270</td><td>0.193</td><td>0.275</td><td>0.176</td><td>0.264</td><td>0.185</td><td>0.272</td><td>0.167</td><td>0.256</td></tr><tr><td>336</td><td>0.216</td><td>0.301</td><td>0.214</td><td>0.297</td><td>0.195</td><td>0.281</td><td>0.210</td><td>0.296</td><td>0.186</td><td>0.275</td></tr><tr><td>720</td><td>0.239</td><td>0.322</td><td>0.253</td><td>0.327</td><td>0.235</td><td>0.316</td><td>0.254</td><td>0.332</td><td>0.231</td><td>0.309</td></tr></table>

Table 4: Parameter sensitivity study. The prediction accuracy varies with K. 

<table><tr><td colspan="4">K=1</td><td colspan="2">K=2</td><td colspan="2">K=3</td><td colspan="2">K=4</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.283</td><td>0.333</td><td>0.279</td><td>0.331</td><td>0.286</td><td>0.337</td><td>0.282</td><td>0.333</td></tr><tr><td>192</td><td>0.357</td><td>0.380</td><td>0.349</td><td>0.380</td><td>0.354</td><td>0.383</td><td>0.359</td><td>0.384</td></tr><tr><td>336</td><td>0.342</td><td>0.379</td><td>0.348</td><td>0.382</td><td>0.338</td><td>0.377</td><td>0.347</td><td>0.380</td></tr><tr><td>720</td><td>0.411</td><td>0.430</td><td>0.398</td><td>0.424</td><td>0.406</td><td>0.428</td><td>0.407</td><td>0.432</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.162</td><td>0.247</td><td>0.145</td><td>0.236</td><td>0.147</td><td>0.238</td><td>0.152</td><td>0.244</td></tr><tr><td>192</td><td>0.175</td><td>0.260</td><td>0.167</td><td>0.256</td><td>0.176</td><td>0.265</td><td>0.178</td><td>0.266</td></tr><tr><td>336</td><td>0.192</td><td>0.278</td><td>0.186</td><td>0.275</td><td>0.181</td><td>0.274</td><td>0.190</td><td>0.277</td></tr><tr><td>720</td><td>0.234</td><td>0.311</td><td>0.231</td><td>0.309</td><td>0.230</td><td>0.308</td><td>0.235</td><td>0.313</td></tr></table>

and periodic patterns to improve the ability to capture the temporal dynamics of its input, assisting in the identification of appropriate patch sizes for combination.

Varying the Number of Adaptively Selected Patch Sizes. Pathformer adaptively selects the top K patch sizes for combination, adjusting to different time series samples. We evaluate the influence of different K values on prediction accuracy in Table 4. Our findings show that K = 2 and K = 3 yield better results than K = 1 and K = 4, highlighting the advantage of adaptively modeling critical multi-scale characteristics for improved accuracy. Additionally, distinct time series samples benefit from feature extraction using varied patch sizes, but not all patch sizes are equally effective.

Visualization of Pathways Weights. We show three samples and depict their average Pathways weights for each patch size in Figure 4. Our observations reveal that the samples possess unique Pathways weight distributions. Both Samples 1 and 2, which demonstrate longer seasonality and similar trend patterns, show similar visualized Pathways weights. This manifests in the higher weights they attribute to the larger patch sizes. On the other hand, Sample 3, which is characterized by its shorter seasonality pattern, aligns with higher weights for the smaller patch sizes. These observations underscore Pathformer's adaptability, emphasizing its ability to discern and apply the optimal patch size combinations for the diverse seasonality and trend patterns across samples.

![](images/d33424b799e13472f5fc31b77cd7fb924b9bdd84dee444475103869d2a1e3e7b.jpg)

<details>
<summary>line</summary>

| Time | Sample1 | Sample2 | Sample3 |
|------|---------|---------|---------|
| 0    | 0.0     | 0.0     | 0.0     |
| 1    | 0.5     | 0.3     | 0.4     |
| 2    | 0.7     | 0.6     | 0.8     |
| 3    | 0.9     | 0.8     | 1.0     |
| 4    | 0.6     | 0.5     | 0.7     |
| 5    | 0.4     | 0.3     | 0.5     |
| 6    | 0.2     | 0.1     | 0.3     |
| 7    | 0.1     | 0.0     | 0.1     |
| 8    | 0.3     | 0.2     | 0.4     |
| 9    | 0.5     | 0.4     | 0.6     |
| 10   | 0.7     | 0.6     | 0.8     |
| 11   | 0.9     | 0.8     | 1.0     |
| 12   | 1.1     | 1.0     | 1.2     |
| 13   | 1.3     | 1.2     | 1.4     |
| 14   | 1.5     | 1.4     | 1.6     |
| 15   | 1.7     | 1.6     | 1.8     |
| 16   | 1.9     | 1.8     | 2.0     |
| 17   | 2.1     | 2.0     | 2.2     |
| 18   | 2.3     | 2.2     | 2.4     |
| 19   | 2.5     | 2.4     | 2.6     |
| 20   | 2.7     | 2.6     | 2.8     |
| 21   | 2.9     | 2.8     | 3.0     |
| 22   | 3.1     | 3.0     | 3.2     |
| 23   | 3.3     | 3.2     | 3.4     |
| 24   | 3.5     | 3.4     | 3.6     |
| 25   | 3.7     | 3.6     | 3.8     |
| 26   | 3.9     | 3.8     | 4.0     |
| 27   | 4.1     | 4.0     | 4.2     |
| 28   | 4.3     | 4.2     | 4.4     |
| 29   | 4.5     | 4.4     | 4.6     |
| 30   | 4.7     | 4.6     | 4.8     |
| 31   | 4.9     | 4.8     | 5.0     |
| 32   | 5.1     | 5.0     | 5.2     |
| 33   | 5.3     | 5.2     | 5.4     |
| 34   | 5.5     | 5.4     | 5.6     |
| 35   | 5.7     | 5.6     | 5.8     |
| 36   | 5.9     | 5.8     | 6.0     |
| 37   | 6.1     | 6.0     | 6.2     |
| 38   | 6.3     | 6.2     | 6.4     |
| 39   | 6.5     | 6.4     | 6.6     |
| 40   | 6.7     | 6.6     | 6.8     |
| 41   | 6.9     | 6.8     | 7.0     |
| 42   | 7.1     | 7.0     | 7.2     |
| 43   | 7.3     | 7.2     | 7.4     |
| 44   | 7.5     | 7.4     | 7.6     |
| 45   | 7.7     | 7.6     | 7.8     |
| 46   | 7.9     | 7.8     | 8.0     |
| 47   | 8.1     | 8.0     | 8.2     |
| 48   | 8.3     | 8.2     | 8.4     |
| 49   | 8.5     | 8.4     | 8.6     |
| 50   | 8.7     | 8.6     | 8.8     |
| ... (repeated) for all three samples at each time point are not explicitly labeled in the image; the actual values may vary due to the use of random sampling data or a random number generator used by the code.
</details>

![](images/b6499d88931d01eaca6286ed52c662048dcfd47e56faa66382dfea96fbff6cf1.jpg)

<details>
<summary>heatmap</summary>

Path weight of Sample1
| | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| B1- | 0.502 | 0.230 | 0.153 | 0.115 |
| B2- | 0.487 | 0.219 | 0.131 | 0.163 |
| B3- | 0.367 | 0.294 | 0.124 | 0.214 |
</details>

![](images/4a2bfa595c850d4ed8622393e5a97733e593ce7dbd2313ee6de9e51bc6656bfb.jpg)

<details>
<summary>heatmap</summary>

Path weight of Sample2
| | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| B1 | 0.326 | 0.325 | 0.171 | 0.178 |
| B2 | 0.407 | 0.254 | 0.124 | 0.214 |
| B3 | 0.382 | 0.374 | 0.067 | 0.176 |
</details>

![](images/65f4bc941af06cede5acc8f517c7fac21b6c78e3c3553083ba068cea9a8d0caa.jpg)

<details>
<summary>heatmap</summary>

Path weight of Sample3
| | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| B1 | 0.092 | 0.203 | 0.261 | 0.444 |
| B2 | 0.122 | 0.183 | 0.268 | 0.426 |
| B3 | 0.142 | 0.177 | 0.205 | 0.475 |
</details>

Figure 4: The average pathways weights of different patch sizes for the Weather. $B_{1}$ , $B_{2}$ , and $B_{3}$ denote distinct AMS (Adaptive Multi-Scale) blocks, while $S_{1}$ , $S_{2}$ , $S_{3}$ , and $S_{4}$ represent varying patch sizes within each AMS block, with patch size decreasing sequentially.

# 5 CONCLUSION

In this paper, we propose Pathformer, a Multi-Scale Transformer with Adaptive Pathways for time series forecasting. It integrates multi-scale temporal resolutions and temporal distances by introducing patch division with multiple patch sizes and dual attention on the divided patches, enabling the comprehensive modeling of multi-scale characteristics. Furthermore, adaptive pathways dynamically select and aggregate scale-specific characteristics based on the different temporal dynamics. These innovative mechanisms collectively empower Pathformer to achieve outstanding prediction performance and demonstrate strong generalization capability on several forecasting tasks.

# ACKNOWLEDGMENTS

This work was supported by National Natural Science Foundation of China (62372179) and Alibaba Innovative Research Program.

# REFERENCES

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), 2020.   
David Campos, Tung Kieu, Chenjuan Guo, Feiteng Huang, Kai Zheng, Bin Yang, and Christian S. Jensen. Unsupervised time series outlier detection with diversity-driven convolutional ensembles. Proceedings of the VLDB Endowment, 2022.   
Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza Ramírez, Max Mergenthaler Canseco, and Artur Dubrawski. NHITS: neural hierarchical interpolation for time series forecasting. In Association for the Advancement of Artificial Intelligence (AAAI), 2023.   
Cathy WS Chen, Richard Gerlach, Edward MH Lin, and WCW Lee. Bayesian forecasting for financial risk management, pre and post the global financial crisis. Journal of Forecasting, 2012.   
Weiqi Chen, Wenwei Wang, Bingqing Peng, Qingsong Wen, Tian Zhou, and Liang Sun. Learning to rotate: Quaternion transformer for complicated periodical time series forecasting. In International Conference on Knowledge Discovery & Data Mining (KDD), 2022.   
Yunyao Cheng, Peng Chen, Chenjuan Guo, Kai Zhao, Qingsong Wen, Bin Yang, and Christian S. Jensen. Weakly guided adaptation for robust time series forecasting. Proceedings of the VLDB Endowment, 2024.   
Junyoung Chung, Çaglar Gülçehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, 2014.   
Razvan-Gabriel Cirstea, Bin Yang, and Chenjuan Guo. Graph attention recurrent neural networks for correlated time series forecasting. In International Conference on Knowledge Discovery & Data Mining (KDD), 2019.   
Razvan-Gabriel Cirstea, Tung Kieu, Chenjuan Guo, Bin Yang, and Sinno Jialin Pan. EnhanceNet: Plugin neural networks for enhancing correlated time series forecasting. In IEEE International Conference on Data Engineering (ICDE), 2021.   
Razvan-Gabriel Cirstea, Chenjuan Guo, Bin Yang, Tung Kieu, Xuanyi Dong, and Shirui Pan. Triformer: Triangular, variable-specific attentions for long sequence multivariate time series forecasting. In International Joint Conference on Artificial Intelligence (IJCAI), 2022a.   
Razvan-Gabriel Cirstea, Bin Yang, Chenjuan Guo, Tung Kieu, and Shirui Pan. Towards spatiotemporal aware traffic time series forecasting. In IEEE International Conference on Data Engineering (ICDE), 2022b.   
James W Cooley and John W Tukey. An algorithm for the machine calculation of complex fourier series. Mathematics of computation, 1965.   
Abhimanyu Das, Weihao Kong, Andrew Leach, Rajat Sen, and Rose Yu. Long-term forecasting with tide: Time-series dense encoder. arXiv, 2023.   
Jeff Dean. Introducing pathways: A next-generation ai architecture, 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.   
Marco AR Ferreira, David M Higdon, Herbert KH Lee, and Mike West. Multi-scale and hidden resolution time series models. 2006.   
Ronghang Hu, Amanpreet Singh, Trevor Darrell, and Marcus Rohrbach. Iterative answer prediction with pointer-augmented multimodal transformers for textvqa. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020.   
Rob J Hyndman and Yeasmin Khandakar. Automatic time series forecasting: the forecast package for r. Journal of statistical software, 2008.   
Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I Webb, Irwin King, and Shirui Pan. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. arXiv, 2023a.   
Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-LLM: Time series forecasting by reprogramming large language models. arXiv, 2023b.   
Tung Kieu, Bin Yang, Chenjuan Guo, Razvan-Gabriel Cirstea, Yan Zhao, Yale Song, and Christian S. Jensen. Anomaly detection in time series with robust variational quasi-recurrent autoencoders. In IEEE International Conference on Data Engineering (ICDE), 2022a.   
Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng. Robust and explainable autoencoders for unsupervised time series outlier detection. In IEEE International Conference on Data Engineering (ICDE), 2022b.   
Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations (ICLR), 2022.   
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), International Conference on Learning Representations (ICLR), 2015.   
Hao Li, Jie Shao, Kewen Liao, and Mingjian Tang. Do simpler statistical methods perform better in multivariate long sequence time-series forecasting? In International Conference on Information & Knowledge Management (CIKM), 2022a.   
Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2019.   
Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022b.   
Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: Time series modeling and forecasting with sample convolution and interaction. In Advances in Neural Information Processing Systems (NeurIPS), 2022a.   
Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations (ICLR), 2022b.   
Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2022c.   
Yu Ma, Bin Yang, and Christian S. Jensen. Enabling time-dependent uncertain eco-weights for road networks. In Proceedings of the ACM on Management of Data, 2014.

Hao Miao, Yan Zhao, Chenjuan Guo, Bin Yang, Zheng Kai, Feiteng Huang, Jiandong Xie, and Christian S. Jensen. A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data. In IEEE International Conference on Data Engineering (ICDE), 2024.   
Michael Mozer. Induction of multiscale temporal structure. In Advances in Neural Information Processing Systems (NeurIPS), 1991.   
Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR), 2023.   
Zhicheng Pan, Yihang Wang, Yingying Zhang, Sean Bin Yang, Yunyao Cheng, Peng Chen, Chenjuan Guo, Qingsong Wen, Xiduo Tian, Yunliang Dou, et al. Magicscaler: Uncertainty-aware, predictive autoscaling. Proceedings of the VLDB Endowment, 2023.   
Simon Aagaard Pedersen, Bin Yang, and Christian S. Jensen. Anytime stochastic routing with hybrid learning. Proceedings of the VLDB Endowment, 2020.   
Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2018.   
Rajat Sen, Hsiang-Fu Yu, and Inderjit S. Dhillon. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2019.   
Mohammad Amin Shabani, Amir H. Abdi, Lili Meng, and Tristan Sylvain. Scaleformer: Iterative multi-scale refining transformers for time series forecasting. In International Conference on Learning Representations (ICLR), 2023.   
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations (ICLR), 2016.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems (NeurIPS), 2017.   
Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: multiscale local and global context modeling for long-term series forecasting. In International Conference on Learning Representations (ICLR), 2023.   
Junke Wang, Zuxuan Wu, Wenhao Ouyang, Xintong Han, Jingjing Chen, Yu-Gang Jiang, and SerNam Lim. M2TR: multi-modal multi-scale transformers for deepfake detection. In International Conference on Multimedia Retrieval (ICMR), 2022a.   
Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In International Conference on Computer Vision (ICCV), 2021.   
Wenxiao Wang, Lu Yao, Long Chen, Binbin Lin, Deng Cai, Xiaofei He, and Wei Liu. Crossformer: A versatile vision transformer hinging on cross-scale attention. In International Conference on Learning Representations (ICLR), 2022b.   
Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: A survey. In International Joint Conference on Artificial Intelligence (IJCAI), 2023.   
Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, and Dhruv Madeka. A multi-horizon quantile recurrent forecaster. arXiv, 2017.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations (ICLR), 2023a.   
Xinle Wu, Dalin Zhang, Chenjuan Guo, Chaoyang He, Bin Yang, and Christian S. Jensen. AutoCTS: Automated correlated time series forecasting. Proceedings of the VLDB Endowment, 2022.   
Xinle Wu, Dalin Zhang, Miao Zhang, Chenjuan Guo, Bin Yang, and Christian S. Jensen. AutoCTS+: Joint neural architecture and hyperparameter search for correlated time series forecasting. Proceedings of the ACM on Management of Data, 2023b.   
Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In International Conference on Knowledge Discovery & Data Mining (KDD), 2020.   
Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Association for the Advancement of Artificial Intelligence (AAAI), 2023.   
Kai Zhao, Chenjuan Guo, Peng Han, Miao Zhang, Yunyao Cheng, and Bin Yang. Multiple time series forecasting with dynamic graph modeling. Proceedings of the VLDB Endowment, 2024.   
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Association for the Advancement of Artificial Intelligence (AAAI), 2021.   
Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning (ICML), 2022.   
Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, and Rong Jin. One fits all: Power general time series analysis by pretrained lm. arXiv, 2023.   
Zhaoyang Zhu, Weiqi Chen, Rui Xia, Tian Zhou, Peisong Niu, Bingqing Peng, Wenwei Wang, Hengbo Liu, Ziqing Ma, Xinyue Gu, et al. Energy forecasting with robust, flexible, and explainable machine learning algorithms. AI Magazine, 2023.

# A APPENDIX

# A.1 EXPERIMENTAL DETAILS

# A.1.1 DATASETS

The Special details about experiment datasets are as follows: ETT $^{1}$ datasets consist of 7 variables, originating from two different electric transformers. It covers the period from January 2016 to January 2018. Each electric transformer has data recorded at 15-minute and 1-hour granularities, labeled as ETTh1, ETTh2, ETTm1, and ETTm2. Weather $^{2}$ dataset comprises 21 meteorological indicators in Germany, collected every 10 minutes. Electricity $^{3}$ dataset contains the power consumption of 321 users, recorded every hour, spanning from July 2016 to July 2019. ILI $^{4}$ collects weekly data on patients with influenza-like illness from the Centers for Disease Control and Prevention of the United States spanning the years 2002 to 2021. Traffic $^{5}$ comprises hourly data sourced from the California Department of Transportation. This dataset delineates road occupancy rates measured by various sensors on the freeways of the San Francisco Bay area. Cloud cluster datasets are private business data, documenting customer resource demands at 1-minute intervals for three clusters: cluster-A, cluster-B, cluster-C, where A,B,C represent different cities, covering the period from February 2023 to April 2023. For dataset preparation, we follow the established practice from previous studies (Zhou et al., 2021; Wu et al., 2021). Detailed statistics are shown in Table 5.

Table 5: The statistics of datasets 

<table><tr><td>Datasets</td><td>ETTh1&amp;ETTh2</td><td>ETTm1&amp;ETTm2</td><td>Weather</td><td>Electricity</td><td>ILI</td><td>Traffic</td><td>Cluster</td></tr><tr><td>Variables</td><td>7</td><td>7</td><td>21</td><td>321</td><td>7</td><td>862</td><td>6</td></tr><tr><td>Timestamps</td><td>17420</td><td>69680</td><td>52696</td><td>26304</td><td>966</td><td>17544</td><td>256322</td></tr><tr><td>Split Ratio</td><td>6:2:2</td><td>6:2:2</td><td>7:1:2</td><td>7:1:2</td><td>7:1:2</td><td>7:1:2</td><td>7:1:2</td></tr></table>

# A.1.2 BASELINES

In the realm of time series forecasting, numerous models have surfaced in recent years. We choose models with superior predictive performance from 2021 to 2023 as baselines, including the 2021 state-of-the-art (SOTA) Autoformer, the 2022 SOTA FEDformer, and the 2023 SOTA PatchTST and NLinear, among others. The specific code repositories for each of these models are as follows:

- PatchTST: https://github.com/yuqinie98/PatchTST   
• NLinear: https://github.com/cure-lab/LTSF-Linear   
• FEDformer: https://github.com/MAZiqing/FEDformer   
- Scaleformer: https://github.com/borealisai/scaleformer   
- TiDE: https://github.com/google-research/google-research/tree/master/tide   
- Pyraformer: https://github.com/ant-research/Pyraformer   
• Autoformer: https://github.com/thuml/Autoformer

# A.2 UNIVARIATE TIME SERIES FORECASTING

We conducted univariate time series forecasting experiments on the ETT and Cloud cluster datasets. As shown in Table 6, Pathformer stands out with the best performance in 50 cases and as the second-best in 5 out of 56 instances. Pathformer has outperformed the second-best baseline PatchTST, especially on the Cloud cluster datasets. Our model Pathformer demonstrates excellent predictive performance in both multivariate and univariate time series forecasting.

Table 6: Univariate time series forecasting results. The input length H = 96, and the prediction length $F \in \{96, 192, 336, 720\}$ (for cloud clusters datasets $F \in \{24, 48, 96, 192\}$ ). The best results are highlighted in bold. 

<table><tr><td colspan="2">Models</td><td colspan="2">Pathformer</td><td colspan="2">PatchTST</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.057</td><td>0.180</td><td>0.057</td><td>0.179</td><td>0.079</td><td>0.215</td><td>0.071</td><td>0.206</td></tr><tr><td>192</td><td>0.075</td><td>0.208</td><td>0.076</td><td>0.209</td><td>0.104</td><td>0.245</td><td>0.114</td><td>0.262</td></tr><tr><td>336</td><td>0.076</td><td>0.216</td><td>0.093</td><td>0.240</td><td>0.119</td><td>0.270</td><td>0.107</td><td>0.258</td></tr><tr><td>720</td><td>0.090</td><td>0.238</td><td>0.097</td><td>0.245</td><td>0.142</td><td>0.299</td><td>0.126</td><td>0.283</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.128</td><td>0.274</td><td>0.127</td><td>0.273</td><td>0.128</td><td>0.271</td><td>0.153</td><td>0.306</td></tr><tr><td>192</td><td>0.177</td><td>0.330</td><td>0.178</td><td>0.328</td><td>0.185</td><td>0.330</td><td>0.204</td><td>0.351</td></tr><tr><td>336</td><td>0.180</td><td>0.340</td><td>0.221</td><td>0.374</td><td>0.231</td><td>0.378</td><td>0.246</td><td>0.389</td></tr><tr><td>720</td><td>0.213</td><td>0.371</td><td>0.250</td><td>0.403</td><td>0.278</td><td>0.420</td><td>0.268</td><td>0.409</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.029</td><td>0.126</td><td>0.030</td><td>0.127</td><td>0.033</td><td>0.140</td><td>0.056</td><td>0.183</td></tr><tr><td>192</td><td>0.042</td><td>0.160</td><td>0.043</td><td>0.165</td><td>0.058</td><td>0.186</td><td>0.081</td><td>0.216</td></tr><tr><td>336</td><td>0.058</td><td>0.185</td><td>0.059</td><td>0.185</td><td>0.084</td><td>0.231</td><td>0.076</td><td>0.218</td></tr><tr><td>720</td><td>0.079</td><td>0.217</td><td>0.081</td><td>0.218</td><td>0.102</td><td>0.250</td><td>0.110</td><td>0.267</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.062</td><td>0.179</td><td>0.064</td><td>0.181</td><td>0.072</td><td>0.206</td><td>0.065</td><td>0.189</td></tr><tr><td>192</td><td>0.096</td><td>0.230</td><td>0.097</td><td>0.231</td><td>0.102</td><td>0.245</td><td>0.118</td><td>0.256</td></tr><tr><td>336</td><td>0.128</td><td>0.268</td><td>0.129</td><td>0.270</td><td>0.130</td><td>0.279</td><td>0.154</td><td>0.305</td></tr><tr><td>720</td><td>0.179</td><td>0.326</td><td>0.181</td><td>0.330</td><td>0.178</td><td>0.325</td><td>0.182</td><td>0.335</td></tr><tr><td rowspan="4">Cluster-A</td><td>24</td><td>0.137</td><td>0.218</td><td>0.174</td><td>0.256</td><td>0.203</td><td>0.303</td><td>0.455</td><td>0.483</td></tr><tr><td>48</td><td>0.218</td><td>0.280</td><td>0.299</td><td>0.343</td><td>0.308</td><td>0.364</td><td>0.508</td><td>0.504</td></tr><tr><td>96</td><td>0.298</td><td>0.337</td><td>0.434</td><td>0.409</td><td>0.361</td><td>0.403</td><td>0.563</td><td>0.524</td></tr><tr><td>192</td><td>0.390</td><td>0.401</td><td>0.589</td><td>0.480</td><td>0.409</td><td>0.447</td><td>0.669</td><td>0.583</td></tr><tr><td rowspan="4">Cluster-B</td><td>24</td><td>0.100</td><td>0.206</td><td>0.107</td><td>0.218</td><td>0.130</td><td>0.253</td><td>0.197</td><td>0.339</td></tr><tr><td>48</td><td>0.146</td><td>0.251</td><td>0.158</td><td>0.265</td><td>0.149</td><td>0.272</td><td>0.247</td><td>0.390</td></tr><tr><td>96</td><td>0.219</td><td>0.301</td><td>0.234</td><td>0.327</td><td>0.230</td><td>0.342</td><td>0.313</td><td>0.429</td></tr><tr><td>192</td><td>0.454</td><td>0.404</td><td>0.461</td><td>0.444</td><td>0.415</td><td>0.412</td><td>0.512</td><td>0.544</td></tr><tr><td rowspan="4">Cluster-C</td><td>24</td><td>0.080</td><td>0.191</td><td>0.092</td><td>0.210</td><td>0.120</td><td>0.258</td><td>0.206</td><td>0.354</td></tr><tr><td>48</td><td>0.117</td><td>0.232</td><td>0.138</td><td>0.261</td><td>0.151</td><td>0.302</td><td>0.229</td><td>0.365</td></tr><tr><td>96</td><td>0.176</td><td>0.286</td><td>0.222</td><td>0.330</td><td>0.198</td><td>0.342</td><td>0.293</td><td>0.420</td></tr><tr><td>192</td><td>0.345</td><td>0.390</td><td>0.404</td><td>0.443</td><td>0.361</td><td>0.444</td><td>0.441</td><td>0.524</td></tr></table>

# A.3 VARYING THE INPUT LENGTH WITH TRANSFORMER MODELS

In time series forecasting tasks, the size of the input length determines how much historical information the model receives. We select models with better predictive performance from the main experiments as baselines. We configure different input lengths to evaluate the effectiveness of Pathformer and visualize the prediction results for input lengths of 48,192. From Figure 5, Pathformer consistently outperforms the baselines on the ETTh1, ETTh2, Weather, and Electricity. As depicted in Table 7 and Table 8, for H = 48, 192, Pathformer stands out with the best performance in 46, 44 cases out of 48, respectively. Based on the results above, it is evident that Pathformer outperforms the baselines across different input lengths. As the input length increases, the prediction metrics of Pathformer continue to decrease, indicating that it is capable of modeling longer sequences.

![](images/52bd0b0607ca91aa9a55c0e3df2bcd267f34d4b2b20663fbc6d0d692d1d52ddb.jpg)  
Figure 5: Results with different input length for ETTh1, ETTh2, Weather and Electricity.

Table 7: Multivariate time series forecasting results. The input length H = 48, and the prediction length $F \in \{96, 192, 336, 720\}$ . The best results are highlighted in bold. 

<table><tr><td colspan="2">Models</td><td colspan="2">Pathformer</td><td colspan="2">PatchTST</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.390</td><td>0.403</td><td>0.410</td><td>0.417</td><td>0.382</td><td>0.419</td><td>0.406</td><td>0.432</td></tr><tr><td>192</td><td>0.454</td><td>0.434</td><td>0.469</td><td>0.448</td><td>0.451</td><td>0.456</td><td>0.451</td><td>0.452</td></tr><tr><td>336</td><td>0.483</td><td>0.445</td><td>0.516</td><td>0.469</td><td>0.499</td><td>0.487</td><td>0.461</td><td>0.464</td></tr><tr><td>720</td><td>0.507</td><td>0.475</td><td>0.509</td><td>0.487</td><td>0.510</td><td>0.504</td><td>0.498</td><td>0.500</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.295</td><td>0.335</td><td>0.307</td><td>0.348</td><td>0.330</td><td>0.373</td><td>0.344</td><td>0.383</td></tr><tr><td>192</td><td>0.366</td><td>0.381</td><td>0.397</td><td>0.399</td><td>0.440</td><td>0.436</td><td>0.425</td><td>0.426</td></tr><tr><td>336</td><td>0.368</td><td>0.390</td><td>0.412</td><td>0.420</td><td>0.543</td><td>0.504</td><td>0.445</td><td>0.452</td></tr><tr><td>720</td><td>0.428</td><td>0.435</td><td>0.434</td><td>0.441</td><td>0.471</td><td>0.483</td><td>0.483</td><td>0.481</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.420</td><td>0.392</td><td>0.424</td><td>0.403</td><td>0.428</td><td>0.432</td><td>0.745</td><td>0.556</td></tr><tr><td>192</td><td>0.446</td><td>0.410</td><td>0.468</td><td>0.429</td><td>0.476</td><td>0.460</td><td>0.715</td><td>0.556</td></tr><tr><td>336</td><td>0.469</td><td>0.431</td><td>0.501</td><td>0.453</td><td>0.526</td><td>0.494</td><td>0.816</td><td>0.590</td></tr><tr><td>720</td><td>0.512</td><td>0.465</td><td>0.553</td><td>0.484</td><td>0.630</td><td>0.528</td><td>0.746</td><td>0.572</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.181</td><td>0.256</td><td>0.189</td><td>0.272</td><td>0.185</td><td>0.274</td><td>0.211</td><td>0.299</td></tr><tr><td>192</td><td>0.251</td><td>0.301</td><td>0.260</td><td>0.371</td><td>0.256</td><td>0.318</td><td>0.277</td><td>0.388</td></tr><tr><td>336</td><td>0.323</td><td>0.349</td><td>0.328</td><td>0.359</td><td>0.329</td><td>0.365</td><td>0.347</td><td>0.380</td></tr><tr><td>720</td><td>0.420</td><td>0.406</td><td>0.429</td><td>0.415</td><td>0.447</td><td>0.432</td><td>0.441</td><td>0.432</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.188</td><td>0.223</td><td>0.212</td><td>0.243</td><td>0.241</td><td>0.309</td><td>0.291</td><td>0.357</td></tr><tr><td>192</td><td>0.227</td><td>0.257</td><td>0.254</td><td>0.277</td><td>0.308</td><td>0.356</td><td>0.349</td><td>0.391</td></tr><tr><td>336</td><td>0.276</td><td>0.297</td><td>0.310</td><td>0.316</td><td>0.385</td><td>0.406</td><td>0.409</td><td>0.424</td></tr><tr><td>720</td><td>0.345</td><td>0.349</td><td>0.385</td><td>0.365</td><td>0.438</td><td>0.432</td><td>0.437</td><td>0.431</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.201</td><td>0.280</td><td>0.225</td><td>0.293</td><td>0.240</td><td>0.349</td><td>0.211</td><td>0.322</td></tr><tr><td>192</td><td>0.210</td><td>0.285</td><td>0.229</td><td>0.299</td><td>0.248</td><td>0.357</td><td>0.224</td><td>0.331</td></tr><tr><td>336</td><td>0.236</td><td>0.305</td><td>0.239</td><td>0.316</td><td>0.265</td><td>0.370</td><td>0.259</td><td>0.362</td></tr><tr><td>720</td><td>0.272</td><td>0.338</td><td>0.282</td><td>0.349</td><td>0.326</td><td>0.405</td><td>0.313</td><td>0.407</td></tr></table>

Table 8: Multivariate time series forecasting results. The input length $H = 192$ , and the prediction length $F \in \{96, 192, 336, 720\}$ . The best results are highlighted in bold. 

<table><tr><td colspan="2">Models</td><td colspan="2">Pathformer</td><td colspan="2">PatchTST</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.377</td><td>0.394</td><td>0.384</td><td>0.403</td><td>0.388</td><td>0.423</td><td>0.430</td><td>0.441</td></tr><tr><td>192</td><td>0.428</td><td>0.421</td><td>0.428</td><td>0.425</td><td>0.433</td><td>0.456</td><td>0.487</td><td>0.467</td></tr><tr><td>336</td><td>0.424</td><td>0.419</td><td>0.452</td><td>0.436</td><td>0.445</td><td>0.462</td><td>0.478</td><td>0.474</td></tr><tr><td>720</td><td>0.474</td><td>0.459</td><td>0.453</td><td>0.459</td><td>0.476</td><td>0.490</td><td>0.518</td><td>0.519</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.283</td><td>0.334</td><td>0.285</td><td>0.340</td><td>0.397</td><td>0.424</td><td>0.362</td><td>0.401</td></tr><tr><td>192</td><td>0.343</td><td>0.374</td><td>0.356</td><td>0.387</td><td>0.439</td><td>0.458</td><td>0.430</td><td>0.447</td></tr><tr><td>336</td><td>0.332</td><td>0.374</td><td>0.351</td><td>0.396</td><td>0.471</td><td>0.481</td><td>0.408</td><td>0.447</td></tr><tr><td>720</td><td>0.393</td><td>0.421</td><td>0.395</td><td>0.427</td><td>0.479</td><td>0.490</td><td>0.440</td><td>0.469</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.295</td><td>0.335</td><td>0.295</td><td>0.345</td><td>0.381</td><td>0.424</td><td>0.510</td><td>0.428</td></tr><tr><td>192</td><td>0.336</td><td>0.361</td><td>0.330</td><td>0.365</td><td>0.412</td><td>0.441</td><td>0.619</td><td>0.545</td></tr><tr><td>336</td><td>0.359</td><td>0.384</td><td>0.364</td><td>0.388</td><td>0.435</td><td>0.455</td><td>0.561</td><td>0.500</td></tr><tr><td>720</td><td>0.432</td><td>0.420</td><td>0.423</td><td>0.424</td><td>0.473</td><td>0.474</td><td>0.580</td><td>0.512</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.169</td><td>0.250</td><td>0.169</td><td>0.254</td><td>0.223</td><td>0.305</td><td>0.244</td><td>0.321</td></tr><tr><td>192</td><td>0.230</td><td>0.290</td><td>0.230</td><td>0.294</td><td>0.281</td><td>0.339</td><td>0.302</td><td>0.362</td></tr><tr><td>336</td><td>0.286</td><td>0.328</td><td>0.281</td><td>0.329</td><td>0.321</td><td>0.364</td><td>0.346</td><td>0.390</td></tr><tr><td>720</td><td>0.375</td><td>0.384</td><td>0.373</td><td>0.384</td><td>0.417</td><td>0.420</td><td>0.423</td><td>0.428</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.152</td><td>0.189</td><td>0.160</td><td>0.205</td><td>0.239</td><td>0.316</td><td>0.298</td><td>0.363</td></tr><tr><td>192</td><td>0.198</td><td>0.237</td><td>0.204</td><td>0.245</td><td>0.274</td><td>0.326</td><td>0.322</td><td>0.379</td></tr><tr><td>336</td><td>0.246</td><td>0.276</td><td>0.258</td><td>0.285</td><td>0.334</td><td>0.369</td><td>0.378</td><td>0.409</td></tr><tr><td>720</td><td>0.329</td><td>0.331</td><td>0.329</td><td>0.337</td><td>0.401</td><td>0.412</td><td>0.435</td><td>0.431</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.136</td><td>0.232</td><td>0.146</td><td>0.240</td><td>0.231</td><td>0.343</td><td>0.198</td><td>0.313</td></tr><tr><td>192</td><td>0.143</td><td>0.248</td><td>0.152</td><td>0.252</td><td>0.258</td><td>0.361</td><td>0.218</td><td>0.335</td></tr><tr><td>336</td><td>0.172</td><td>0.274</td><td>0.178</td><td>0.271</td><td>0.273</td><td>0.372</td><td>0.252</td><td>0.352</td></tr><tr><td>720</td><td>0.218</td><td>0.299</td><td>0.223</td><td>0.308</td><td>0.308</td><td>0.402</td><td>0.275</td><td>0.371</td></tr></table>

# A.4 MORE COMPARISONS WITH SOME BASIC BASELINES

To validate the effectiveness of Pathformer, we conducted extensive experiments with some recent basic baselines that exhibited good performance: DLinear, NLinear, and N-HiTS, using long input sequence length (H = 336). As depicted in Table 9, our proposed model Pathformer outperforms

Table 9: Multivariate time series forecasting results. The input length H = 336 (for ILI dataset H = 106), and the prediction length $F \in \{96, 192, 336, 720\}$ (for ILI dataset $F \in \{24, 36, 48, 60\}$ ). The best results are highlighted in bold. 

<table><tr><td colspan="2">Method</td><td colspan="2">Pathformer</td><td colspan="2">DLinear</td><td colspan="2">NLinear</td><td colspan="2">N-HiTS</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.369</td><td>0.395</td><td>0.375</td><td>0.399</td><td>0.374</td><td>0.394</td><td>0.378</td><td>0.393</td></tr><tr><td>192</td><td>0.414</td><td>0.418</td><td>0.405</td><td>0.416</td><td>0.408</td><td>0.415</td><td>0.427</td><td>0.436</td></tr><tr><td>336</td><td>0.401</td><td>0.419</td><td>0.439</td><td>0.443</td><td>0.429</td><td>0.427</td><td>0.458</td><td>0.484</td></tr><tr><td>720</td><td>0.440</td><td>0.452</td><td>0.472</td><td>0.490</td><td>0.440</td><td>0.453</td><td>0.561</td><td>0.501</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.276</td><td>0.334</td><td>0.289</td><td>0.353</td><td>0.277</td><td>0.338</td><td>0.274</td><td>0.345</td></tr><tr><td>192</td><td>0.329</td><td>0.372</td><td>0.383</td><td>0.418</td><td>0.344</td><td>0.381</td><td>0.353</td><td>0.401</td></tr><tr><td>336</td><td>0.324</td><td>0.377</td><td>0.448</td><td>0.465</td><td>0.357</td><td>0.400</td><td>0.382</td><td>0.425</td></tr><tr><td>720</td><td>0.366</td><td>0.410</td><td>0.605</td><td>0.551</td><td>0.394</td><td>0.436</td><td>0.625</td><td>0.557</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.285</td><td>0.336</td><td>0.299</td><td>0.353</td><td>0.306</td><td>0.348</td><td>0.302</td><td>0.350</td></tr><tr><td>192</td><td>0.331</td><td>0.361</td><td>0.335</td><td>0.365</td><td>0.349</td><td>0.375</td><td>0.347</td><td>0.383</td></tr><tr><td>336</td><td>0.362</td><td>0.382</td><td>0.369</td><td>0.386</td><td>0.375</td><td>0.388</td><td>0.369</td><td>0.402</td></tr><tr><td>720</td><td>0.412</td><td>0.414</td><td>0.425</td><td>0.421</td><td>0.433</td><td>0.422</td><td>0.431</td><td>0.441</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.163</td><td>0.248</td><td>0.167</td><td>0.260</td><td>0.167</td><td>0.255</td><td>0.176</td><td>0.255</td></tr><tr><td>192</td><td>0.220</td><td>0.286</td><td>0.224</td><td>0.303</td><td>0.221</td><td>0.293</td><td>0.245</td><td>0.305</td></tr><tr><td>336</td><td>0.275</td><td>0.325</td><td>0.281</td><td>0.342</td><td>0.274</td><td>0.327</td><td>0.295</td><td>0.346</td></tr><tr><td>720</td><td>0.363</td><td>0.381</td><td>0.397</td><td>0.421</td><td>0.368</td><td>0.384</td><td>0.401</td><td>0.413</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.144</td><td>0.184</td><td>0.176</td><td>0.237</td><td>0.182</td><td>0.232</td><td>0.158</td><td>0.195</td></tr><tr><td>192</td><td>0.191</td><td>0.229</td><td>0.220</td><td>0.282</td><td>0.225</td><td>0.269</td><td>0.211</td><td>0.247</td></tr><tr><td>336</td><td>0.234</td><td>0.268</td><td>0.265</td><td>0.319</td><td>0.271</td><td>0.301</td><td>0.274</td><td>0.300</td></tr><tr><td>720</td><td>0.316</td><td>0.323</td><td>0.323</td><td>0.362</td><td>0.338</td><td>0.348</td><td>0.351</td><td>0.353</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.134</td><td>0.218</td><td>0.140</td><td>0.237</td><td>0.141</td><td>0.237</td><td>0.147</td><td>0.249</td></tr><tr><td>192</td><td>0.142</td><td>0.235</td><td>0.153</td><td>0.249</td><td>0.154</td><td>0.248</td><td>0.167</td><td>0.269</td></tr><tr><td>336</td><td>0.162</td><td>0.257</td><td>0.169</td><td>0.267</td><td>0.171</td><td>0.265</td><td>0.186</td><td>0.290</td></tr><tr><td>720</td><td>0.200</td><td>0.290</td><td>0.203</td><td>0.301</td><td>0.210</td><td>0.297</td><td>0.243</td><td>0.340</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.411</td><td>0.705</td><td>2.215</td><td>1.081</td><td>1.683</td><td>0.868</td><td>1.862</td><td>0.869</td></tr><tr><td>36</td><td>1.365</td><td>0.727</td><td>1.963</td><td>0.963</td><td>1.703</td><td>0.859</td><td>2.071</td><td>0.934</td></tr><tr><td>48</td><td>1.537</td><td>0.764</td><td>2.130</td><td>1.024</td><td>1.719</td><td>0.884</td><td>2.134</td><td>0.932</td></tr><tr><td>60</td><td>1.418</td><td>0.772</td><td>2.368</td><td>1.096</td><td>1.819</td><td>0.917</td><td>2.137</td><td>1.968</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.373</td><td>0.241</td><td>0.410</td><td>0.282</td><td>0.410</td><td>0.279</td><td>0.402</td><td>0.282</td></tr><tr><td>192</td><td>0.380</td><td>0.252</td><td>0.423</td><td>0.287</td><td>0.423</td><td>0.284</td><td>0.420</td><td>0.297</td></tr><tr><td>336</td><td>0.395</td><td>0.256</td><td>0.436</td><td>0.296</td><td>0.435</td><td>0.290</td><td>0.448</td><td>0.313</td></tr><tr><td>720</td><td>0.425</td><td>0.280</td><td>0.466</td><td>0.315</td><td>0.464</td><td>0.307</td><td>0.539</td><td>0.353</td></tr></table>

these baselines for the input length 336. Zeng et al. (2023) point out that the previous Transformer cannot extract temporal relations well from longer input sequences, but our proposed Pathformer performs better with a longer input length, indicating that considering adaptive multi-scale modeling can be an effective way to enhance such a relation extraction ability of Transformers.

# A.5 DISCUSSION

# A.5.1 COMPARE WITH PATCHTST

PatchTST divides time series into patches, with empirical evidence proving that patching is an effective method to enhance model performance in time series forecasting. Our proposed model Pathformer extends the patching approach to incorporate multi-scale modeling. The main differences with PatchTST are as follows: (1) Partitioning with Multiple Patch Sizes: PatchTST employs a single patch size to partition time series, obtaining features with a singular resolution. In contrast, Pathformer utilizes multiple different patch sizes at each layer for partitioning. This approach captures multi-scale features from the perspective of temporal resolutions. (2) Global correlations between patches and local details in each patch: PatchTST performs attention between divided patches, overlooking the internal details in each patch. In contrast, Pathformer not only considers the correlations between patches but also the detailed information within each patch. It introduces dual attention(inter-patch attention and intra-patch attention) to integrate global correlations and local details, capturing multi-scale features from the perspective of temporal distances. (3) Adaptive Multi-scale Modeling: PatchTST employs a fixed patch size for all data, hindering the grasp of critical patterns in different time series. We propose adaptive pathways that dynamically select varying patch sizes tailored to the features of individual samples, enabling adaptive multi-scale modeling.

# A.5.2 COMPARE WITH N-HITS

N-HiTS utilizes the modeling of multi-scale features for time series forecasting, but it differs from Pathformer in the following aspects: (1) N-HiTS models time series features of different resolutions through multi-rate data sampling and hierarchical interpolation. In contrast, Pathformer not only takes into account time series features of different resolutions but also approaches multi-scale modeling from the perspective of temporal distance. Simultaneously considering temporal resolutions and temporal distances enables a more comprehensive approach to multi-scale modeling. (2) N-HiTS employs fixed sampling rates for multi-rate data sampling, lacking the ability to adaptively perform multi-scale modeling based on differences in time series samples. In contrast, Pathformer has the capability for adaptive multi-scale modeling. (3) N-HiTS adopts a linear structure to build its model framework, whereas Pathformer enables multi-scale modeling in a Transformer architecture.

# A.5.3 COMPARE WITH SCALEFORMER

Scaleformer also utilizes the modeling of multi-scale features for time series forecasting. It differs from Pathformer in the following aspects: (1) Scaleformer obtains multi-scale features with different temporal resolutions through downsampling. In contrast, Pathformer not only considers time series features of different resolutions but also models from the perspective of temporal distance, taking into account global correlations and local details. This provides a more comprehensive approach to multi-scale modeling through both temporal resolutions and temporal distances. (2) Scaleformer requires the allocation of a predictive model at different temporal resolutions, resulting in higher model complexity than Pathformer. (3) Scaleformer employs fixed sampling rates, while Pathformer has the capability for adaptive multi-scale modeling based on the differences in time series samples.

# A.6 EXPERIMENTS ON LARGE DATASETS

The current time series forecasting benchmarks are relatively small, and there is a concern that the predictive performance of the model might be influenced by overfitting. To address this issue, we explore larger datasets to validate the effectiveness of the proposed model. The detailed process is as follows: We seek larger datasets from two perspectives: data volume and the number of variables. We add two datasets, the Wind Power dataset, and the PEMS07 dataset, to evaluate the performance of Pathformer on larger datasets. The Wind Power dataset comprises 7397147 timestamps, reaching a sample size in the millions, and the PEMS07 dataset includes 883 variables. As depicted in Table 10, Pathformer demonstrates superior predictive performance on these larger datasets compared with some state-of-the-art methods such as PatchTST, DLinear, and Scaleformer.

Table 10: Results on large datasets: PEMS07 and Wind Power. 

<table><tr><td colspan="2">Methods</td><td colspan="2">Pathformer</td><td colspan="2">PatchTST</td><td colspan="2">DLinear</td><td colspan="2">Scaleformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">PEMS07</td><td>96</td><td>0.135</td><td>0.243</td><td>0.146</td><td>0.259</td><td>0.564</td><td>0.536</td><td>0.152</td><td>0.268</td></tr><tr><td>192</td><td>0.177</td><td>0.271</td><td>0.185</td><td>0.286</td><td>0.596</td><td>0.555</td><td>0.195</td><td>0.302</td></tr><tr><td>336</td><td>0.188</td><td>0.278</td><td>0.205</td><td>0.289</td><td>0.475</td><td>0.482</td><td>0.276</td><td>0.394</td></tr><tr><td>720</td><td>0.208</td><td>0.296</td><td>0.235</td><td>0.325</td><td>0.543</td><td>0.523</td><td>0.305</td><td>0.410</td></tr><tr><td rowspan="4">Wind Power</td><td>96</td><td>0.062</td><td>0.146</td><td>0.070</td><td>0.158</td><td>0.078</td><td>0.184</td><td>0.089</td><td>0.167</td></tr><tr><td>192</td><td>0.123</td><td>0.214</td><td>0.131</td><td>0.237</td><td>0.133</td><td>0.252</td><td>0.163</td><td>0.246</td></tr><tr><td>336</td><td>0.200</td><td>0.283</td><td>0.215</td><td>0.307</td><td>0.205</td><td>0.325</td><td>0.225</td><td>0.352</td></tr><tr><td>720</td><td>0.388</td><td>0.414</td><td>0.404</td><td>0.429</td><td>0.407</td><td>0.457</td><td>0.414</td><td>0.426</td></tr></table>

# A.7 VISUALIZATION

We visualize the prediction results of Pathformer on the Electricity dataset. As illustrated in Figure 6, for prediction lengths F = 96, 192, 336, 720, the prediction curve closely aligns with the Ground Truth curve, indicating the outstanding predictive performance of Pathformer. Meanwhile, Pathformer demonstrates effectiveness in capturing multi-period and complex trends present in diverse samples. This serves as evidence of its adaptive modeling capability for multi-scale characteristics.

![](images/8844df2bce692588efcf53435c7e431885a7dde72e88926b4e1927a788f159ab.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -0.5        | -0.6     |
| 25   | 1.8         | 2.7      |
| 50   | -0.3        | 3.5      |
| 75   | 1.6         | 2.9      |
| 100  | -0.2        | 2.0      |
| 125  | -0.8        | -0.9     |
| 150  | -0.5        | 2.8      |
| 175  | 3.0         | 2.4      |
| 200  | -0.4        | -0.5     |
</details>

(a) Prediciton Length $F = 96$

![](images/a10d514be8e89302e844bd63e7af2fc6aafb5494ca93fee0be08816392ff9179.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -0.5        | -0.4     |
| 10   | 1.8         | 2.7      |
| 20   | 2.9         | 3.5      |
| 30   | 1.9         | 2.1      |
| 40   | -0.6        | -0.7     |
| 50   | 2.8         | 3.0      |
| 60   | -0.4        | -0.5     |
| 70   | 2.0         | 2.1      |
| 80   | -0.3        | -0.4     |
| 90   | 1.2         | 1.3      |
| 100  | -0.2        | -0.3     |
| 110  | 0.8         | 0.9      |
| 120  | -0.7        | -0.8     |
| 130  | -1.0        | -1.1     |
| 140  | -0.5        | -0.6     |
| 150  | 2.9         | 2.8      |
| 160  | -0.4        | -0.5     |
| 170  | 3.1         | 2.9      |
| 180  | -0.6        | -0.7     |
| 190  | 2.5         | 2.6      |
| 200  | -0.3        | -0.4     |
| 210  | 1.8         | 2.7      |
| 220  | -0.5        | -0.6     |
| 230  | 2.7         | 2.8      |
| 240  | -0.4        | -0.5     |
| 250  | 2.6         | 2.7      |
| 260  | -0.6        | -0.7     |
| 270  | 1.2         | 1.3      |
| 280  | -0.8        | -0.9     |
| 290  | -1.1        | -1.2     |
| 300  | -1.3        | -1.4     |
</details>

(b) Prediciton Length $F = 192$

![](images/41cdcc23cce353656bde215e18bab0d4d33e5f522a3443b858492981e4e7eb66.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -1.5        | 0.4      |
| 50   | -1.0        | 1.9      |
| 100  | -1.5        | -1.2     |
| 150  | -1.0        | 0.8      |
| 200  | -0.5        | 0.7      |
| 250  | -1.5        | -0.3     |
| 300  | 0.8         | 0.6      |
| 350  | -1.0        | 0.6      |
| 400  | -1.5        | -1.0     |
</details>

(c) Prediciton Length F = 336

![](images/b3cf0bb622deef4e51b3584bd40df599b1059ecbc634e8e3cd2b00b96ddfbd27.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -0.5        | 2.8      |
| 50   | 2.9         | 3.5      |
| 100  | -0.3        | 2.0      |
| 150  | 2.8         | 1.4      |
| 200  | -0.7        | 2.7      |
| 250  | -1.2        | 1.3      |
| 300  | -0.4        | -0.2     |
| 350  | 3.2         | 2.6      |
| 400  | -0.6        | 2.5      |
| 450  | -0.8        | 1.3      |
| 500  | -1.0        | -0.5     |
| 550  | 2.7         | 2.4      |
| 600  | -0.5        | -0.8     |
| 650  | 2.9         | 2.7      |
| 700  | -0.3        | 2.6      |
| 750  | -0.6        | -0.4     |
| 800  | -1.1        | -1.0     |
</details>

(d) Prediciton Length $F = 720$   
Figure 6: Visualization of Pathformer's prediction results on Electricity. The input length $H = 96$