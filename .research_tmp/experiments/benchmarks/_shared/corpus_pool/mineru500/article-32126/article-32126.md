# Spike2Former: Efficient Spiking Transformer for High-performance Image Segmentation

Zhenxin Lei $^{1,2*}$ , Man Yao $^{2\dagger}$ , Jiakui Hu $^{3,2*}$ , Xinhao Luo $^{2}$ , Yanye Lu $^{3,4}$ , Bo Xu $^{2}$ , Guoqi Li $^{2\dagger}$

$^{1}$ University of Chinese Academy of Sciences

$^{2}$ Institute of Automation, Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology, Chinese Academy of Sciences

$^{3}$ Institute of Medical Technology, Peking University Health Science Center, Peking University

$^{4}$ National Biomedical Imaging Center, Peking University

# Abstract

Spiking Neural Networks (SNNs) have a low-power advantage but perform poorly in image segmentation tasks. The reason is that directly converting neural networks with complex architectural designs for segmentation tasks into spiking versions leads to performance degradation and non-convergence. To address this challenge, we first identify the modules in the architecture design that lead to the severe reduction in spike firing, make targeted improvements, and propose Spike2Former architecture. Second, we propose normalized integer spiking neurons to solve the training stability problem of SNNs with complex architectures. We set a new state-of-the-art for SNNs in various semantic segmentation datasets, with a significant improvement of +12.7% mIoU and 5.0× efficiency on ADE20K, +14.3% mIoU and 5.2× efficiency on VOC2012, and +9.1% mIoU and 6.6× efficiency on CityScapes. Our code is available at https://github.com/BICLab/Spike2Former

# Introduction

Spiking Neural Networks (SNNs) emulate the spatiotemporal dynamics and spike-based communication of biological neurons. The former ensures the network's representation (Maass 1997), while the spike-driven paradigm introduced by the latter allows SNNs to perform sparse computing when deployed on neuromorphic chips (Merolla et al. 2014; Davies et al. 2018), thereby benefiting from low power consumption (Roy, Jaiswal, and Panda 2019; Schuman et al. 2022). A notable example is the sensing-computing neuromorphic chip Speck, which operates at a power level as low as $0.7\mathrm{mW}$ in typical visual scenes (Yao et al. 2024d).

The complex neuronal dynamics and binary activations make it challenging to train large-scale SNNs. It took a long time for the SNN field to effectively address this issue through surrogate gradient training (Wu et al. 2018; Neftci, Mostafa, and Zenke 2019a) and residual learning design (Fang et al. 2021; Hu et al. 2024b). Currently, SNNs have achieved commendable performance on simple image classification tasks (Yao et al. 2024b,a). Unfortunately, when it comes to complex visual tasks that require the use of sophisticated neural network architectures, SNNs fall short.

For instance, in image segmentation, an additional segmentation head is required alongside the backbone used for image classification, resulting in a network structure that is significantly more complex than that for classification. Simply converting complex Artificial Neural Networks (ANNs) into spiking versions or directly applying residual designs from classification tasks often leads to a notable drop in performance. Consequently, the few existing SNN models (Kim, Chough, and Panda 2022; Zhang, Fan, and Zhang 2023; Yao et al. 2024a; Su et al. 2024; Patel et al. 2021) that tackle image segmentation tasks tend to perform poorly.

This work explores the application of SNNs with more complex architectures to image segmentation tasks. Specifically, Mask2Former (Cheng et al. 2022) is a classic Transformer-based per-mask classification architecture consisting of three parts: a backbone network; a Feature Pyramid Network (FPN) pixel decoder with a multi-scale deformable transformer encoder block; and a transformer decoder block. Directly converting the Mask2Former architecture into a spiking version results in obvious performance degradation and non-convergence. To address this, we investigated which modules in the spiking Mask2Former contribute to the significant loss of information, where the spiking neurons in these modules nearly cease to fire.

The first module with severe spike degradation is the deformable attention transformer encoder block in the FPN decoder. In the vanilla Mask2Former, the query operation in the deformable attention block is inherently sparse; if the queried information consists of sparse spikes, this could result in significant information loss. To address this, we incorporate convolution blocks in encoder blocks and redesign the deformable attention blocks to preserve more effective information and energy efficiency. The second key module that leads to information loss is the final mask embedding layer. This layer is crucial as it outputs the final segmentation results; however, being at the deepest part of the network, the semantic information that reaches this point is already greatly diminished. To end this, we build an auxiliary information branch to enhance the representation of mask embedding.

Another challenge is that, beyond architectural design, spiking neurons inherently introduce information loss by converting continuous values into binary spikes. This issue has long plagued the SNN field, leading to the development

of various methods aimed at refining the membrane potential distribution, such as attention mechanism (Yao et al. 2023) and information maximization loss (Guo et al. 2022). Recently, Luo et al. proposed an integer training and spike inference method in spiking CNNs to address this challenge. However, this approach does not extend to more complex architectures like Mask2Former, which require numerical stability and precise representation in the interaction of cross-modal features (Cheng, Schwing, and Kirillov 2021; Carion et al. 2020; Li et al. 2023). To address this, we propose a novel normalization method that normalizes integers during training to facilitate training, while ensuring that the spike-driven nature during inference remains unaffected.

The proposed methods are validated on the popular datasets ADE20k, CityScapes, Pascal VOC2012 for semantic segmentation. Spike2Former far exceeds the best models in SNNs in both performance and energy consumption and is comparable to the ANNs. The main contributions are:

1) Spike2Former. We analysis the challenge of applying SNNs to complex architecture and propose a transformer-based image segmentation method, Spike2Former, that incorporates the Spike-driven Deformable Transformer Encoder (SDTE) and Spike-Driven Mask Embedding (SDME) module to resolve the information deficiency and improve model performance.

2) NI-LIF Spiking Neuron. We design a novel spiking neuron, NI-LIF, to increase the training stability and reduce the information deficiency in complex segmentation method. The NI-LIF normalizes the integer activation during training and can be equivalent to spike-driven in inference, capitalizing on the low power nature of SNNs.

3) Performance: The proposed Spike2Former obtains a remarkable performance improvement in popular segmentation tasks with low power consumption, demonstrating the potential of SNNs in complex scenarios. Spike2Former achieves a new state-of-the-art ADE20k (+12.7% mIoU and 5.0×), Pascal VOC2012 (+14.3% mIoU and 5.2×), CityScapes (+9.1% mIoU and 6.6×).

# Related Works

# SNN Training

Training methods in Spiking Neural Networks (SNNs) mainly fall into two categories: ANN-to-SNN conversion and direct training. While ANN-to-SNN conversion inherits from traditional neural networks, it suffers from longer time steps and limited real-time processing capabilities (Li et al. 2022; Bu et al. 2022; Wu et al. 2021). Direct training with surrogate gradients (Neftci, Mostafa, and Zenke 2019b) offers better flexibility but typically yields lower performance. Inspired by recent Integer-based Leaky Integrate-and-Fire (I-LIF) neurons, which bridge the gap between training and inference through virtual timesteps, we adopt the direct training approach for its architectural flexibility (Hu et al. 2024b; Fang et al. 2021; Wu et al. 2018). Recently, Luo et al. introduced a novel spiking neuron that trains the model with Integer activation as a virtual timesteps and converts the Integer into binary spikes by expanding virtual timesteps during inference. We take this inspiration and further improve it for more complex application scenes.

# SNN Backbone Design

SNN backbone designs can be broadly categorized into CNN-based and Transformer-based approaches. CNN-based SNNs primarily focus on improving the Spike-ResNet architecture (Zheng et al. 2021) through variations such as SEW-ResNet (Fang et al. 2021) and MS-ResNet (Hu et al. 2024b). These variations utilize diverse residual connections to alleviate performance degradation and support deeper architectures. Recently, Transformer-based methods have gained prominence in SNN backbone design. Approaches such as (Wang et al. 2023; Zhou et al. 2023; Leroux, Finkbeiner, and Neftci 2023; Yao et al. 2024b,a; Hu et al. 2024a; Yao et al. 2024c) incorporate spiking neurons into self-attention mechanisms to enhance model performance. These methods are relatively straightforward and lack complex interactions, making them easier to train within the SNN framework. Additionally, Luo et al. recently introduced a spike-driven SpikeYOLO model, adapted from YOLOv8 for object detection, which achieves competitive performance. However, to mitigate information loss during feature interactions, this method simplifies the model design, failing to fully address the challenges of applying SNNs to complex architectures.

# Image Segmentation

In ANNs, image segmentation can be categorized into per-pixel and per-mask classification. Per-pixel classification, often using CNNs, employs Feature Pyramid Networks (FPN) to generate segmentation masks. MaskFormer (Cheng, Schwing, and Kirillov 2021) redefined this as per-mask classification, generating binary masks and assigning them semantic classes. Mask2Former (Cheng et al. 2022) and MaskDINO (Li et al. 2023) further improve the MaskFormer with more precise refinement of learnable query. In SNNs, image segmentation remains challenging (Li et al. 2022; Kirkland et al. 2020; Patel et al. 2021). In 2022, (Kim, Chough, and Panda 2022) introduced spike-driven decoders (spike-FCN and spike-Deeplab) by converting ANN architectures to SNNs, while Spiking-CGNet (Zhang, Fan, and Zhang 2023) in 2023 developed a segmentation-specific backbone. More recently, Meta-SpikeFormer (Yao et al. 2024a) directly trained on FPN achieved competitive results on ADE20k (Zhou et al. 2017), highlighting SNN potential. However, the performance and energy efficiency of these method are poor and limit their application to various scenarios.

# Method

Spike2Former adopts the architecture of Mask2Former, which includes a pyramid backbone, a pixel decoder with a deformable transformer encoder, a transformer decoder, and a mask embedding module. In this section, we present the improvement in the deformable transformer encoder and mask embedding module, followed by a discussion of the newly proposed spiking neuron NI-LIF.

# Information Deficiency in Query

Query features are essential in transformer-based methods (Ding et al. 2023; Wang et al. 2024; Jin et al. 2023). They are

![](images/4bb357ee5e5d31c7da9036f981347ef6392e7e0601805cffadc84c962fc33e47.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Backbone Output"] --> B["Spike-Driven Deformable Transformer Encoder Blocks × N"]
    B --> C["ESC"]
    B --> D["SDDA"]
    B --> E["Channel MLP"]
    C --> F["FPN Output F1,F2,F3"]
    D --> F
    E --> F
    F --> G["FPN Output F1,F2,F3"]
    G --> H["Spike-Driven Transformer Decoder Block × L"]
    H --> I["Learnable Object Query"]
    I --> J["Semantic/Panoptic Segmentation"]
    J --> K["NI-LIF Spiking Neuron"]
    K --> L["ME-Shortcut"]
    L --> M["Spike Degradation Phenomenon"]
    M --> N["Matrix Multiply"]
    N --> O["Element-wise Addition"]
    O --> P["NI-LIF"]
    P --> Q["Conv"]
    Q --> R["MLP"]
    R --> S["NI-LIF"]
    S --> T["Mask Embedding"]
    T --> U["ξ_mask"]
    U --> V["BN"]
    V --> W["ξ_pixel"]
    W --> X["×"]
    X --> Y["×"]
    Y --> Z["×"]
    Z --> AA["×"]
    AA --> AB["×"]
    AB --> AC["SNPs"]
    subgraph Legend
        AD["NI-LIF Spiking Neuron"] --> AE["ME-Shortcut"]
        AE --> AF["Spike Degradation Phenomenon"]
        AF --> AG["Matrix Multiply"]
        AG --> AH["Element-wise Addition"]
        AI["SEM"] --> AJ["SEMICON"]
        AK["SEMICON"] --> AL["SEMICONsymbol"]
        AM["SEMICONsymbol"] --> AN["SEMICONsymbolsymbol"]
        AO["SEMICONsymbolsymbol"] --> AP["SEMICONsymbolsymbolsymbol"]
        AQ["SEMICONsymbolsymbolsymbol"] --> AR["SEMICONsymbolsymbolsymbolsymbol"]
        AS["SEMICONsymbolsymbolsymbolsymbol"] --> AT["SEMICONsymbolsymbolsymbolsymbolsymbol"]
        AU["SEMICONsymbolsymbolsymbolsymbol"] --> AV["SEMICONsymbolsymbolsymbolsymbolsymbolsymbol"]
        AW["SEMICONsymbolsymbolsymbolsymbol"] --> AX["SEMICONsymbolsymbolsymbolsymbolsymbolsymbolsymbol"]
        AY["SEMICONsymbolsymbolsymbol"] --> AZ["SEMICONsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbol"]
    end
```
</details>

(B) Micro Design in SDTE   
![](images/1e5802c692779325b0a1b413ff171ce14482b30060dc204d936b7eaef6ef0c96.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Energy-Efficient Separable Convolution
        A["PWConv"] --> B["BN"]
        C["DWConv"] --> D["BN"]
        E["PWConv"] --> F["BN"]
    end

    subgraph Spike-Driven Deformable Attention
        G["Input"] --> H["ESC"]
        H --> I["Query"]
        I --> J["Offsets"]
        J --> K["Offset Sampling"]
        K --> L["Δp"]
        L --> M["Conv BN"]
        M --> N["Conv BN"]
        N --> O["Conv BN"]
        O --> P["Grid"]
        P --> Q["A"]
    end

    style Energy-Efficient Separable Convolution fill:#f9f,stroke:#333
    style Spike-Driven Deformable Attention fill:#ccf,stroke:#333
```
</details>

(C) NI-LIF Spiking Neuron   
![](images/68a6f0edfc768604785e3cecd327485b6a0ee0aea5a5ed81c1a3938edfc0d073.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph I_LIF Quantified Error
        A["Train: 1 1 0 0 2 0"] --> B["Infer: 10 10 0 0 0 1 1 0"]
    end
    subgraph NI_LIF Normalized Integer with D
        C["Train: 0.5 0.5 0 0 1.0 0"] --> D["Infer: 10 10 0 0 0 1 1 0"]
    end
    A --> E["Training: w = 0 × a + 1 × b + 2 × c + 0 × d"]
    C --> F["Inference: w = 0 × a + 1 × b + 1 × c + 0 × d"]
    E --> G["Multiply-ACcumulation(MAC)"]
    F --> H["Multiply-ACcumulation(MAC)"]
    I_LIF Inference: Spikes & Spike & Norm Integer & Conv Insert & Multiply-ACcumulation
```
</details>

Figure 1: (A) The architecture of Spike2Former, we propose Spike-Driven Mask Embedding(SDME) and Spike-driven Deformable Transformer Encoder to introduce the transformer-based method to SNNs. (B) Micro Design in Spike-Driven Deformable Transformer Encoder(SDTE) including Spike-driven Deformable attention(SDDA) and Spike Separable Convolution(ESC). (C) Comparison between NI-LIF and I-LIF. Integer activation results in information loss, especially in cross-model interaction. NI-LIF normalizes the integer during training to preserve information and converts them to spikes during inference with only sparse addition.

primarily used to interact with image features in the Multi-Scale Deformable Attention (MSDeformAttn) transformer encoder and transformer decoder within Mask2Former. However, SNNs face challenges in refining queries due to the significant loss of rich semantic representations when converting query into binary spikes. This raise the question: How can queries preserve information effectively in SNNs? To address this, we propose modifications in the architecture design, including deformable attention blocks and the mask embedding module, which are particularly prone to bias in SNNs. The details of these modifications will be discussed in the following of this section.

Spike-driven Deformable Transformer Encoder Spike-driven Deformable Transformer Encoder(SDTE) consists of a stack of successive blocks, including an Energy-efficient

Separable Convolution(ESC) module to enhance local connections, a Spike-Driven Deformable Attention(SDDA) module that conducts deformable attention within queries, and a Channel-MLP layer to learn non-linear representations.

Energy-efficient Separable Convolution Blocks Yao et al. utilize a separable convolution to enhance the inductive bias. However, this design significantly increases energy consumption due to the direct connection of depthwise and pointwise convolutions. Thus, we propose adding spiking neuron before the second pointwise convolution for energy efficiency, denoted as $ESC(\cdot)$ , and formulated as follows:

$$
\mathbf {U} _ {p w 1} = \operatorname{Conv} _ {p w 1} (S N (\mathbf {U})) \tag {1}
$$

$$
\mathbf {U} _ {d w} = \operatorname{Conv} _ {d w} (S N (\mathbf {U} _ {p w 1})) \tag {2}
$$

$$
\mathbf {U} _ {p w 2} = \operatorname{Conv} _ {p w 2} (S N (\mathbf {U} _ {d w})) \tag {3}
$$

where $U \in R^{\frac{H}{16} \times \frac{W}{16} \times dim}$ means the input membrane potential (dim denotes the embedding dimension), $\mathrm{Conv}_{dw}(\cdot)$ denotes depthwise convolutions, and $\mathrm{Conv}_{pw}(\cdot)$ denote pointwise convolutions.

Spike-Driven Deformable Attention The MSDeformAttn transformer encoder in Mask2Former refines features by attending to the global context, dynamically computing weights from the inputs, and utilizing deformability to adapt the receptive field size (Cavagnero et al. 2024). Although effective in ANNs, the sparse sampling strategy in deformable attention leads to information loss and unstable gradients (Fig. 2). Thus, we propose converting the attention weights into spikes instead of spiking the feature queries to better preserve the semantic information contained within the queries. Furthermore, using multi-scale features as queries significantly increases energy consumption, especially with high-resolution inputs. We recommend using single-scale deep image features with convolution blocks to enhance local connectivity, thereby improving energy efficiency while maintaining performance.

Specifically, for the calculation of attention weight and sampling offsets, we propose adding depthwise convolution (DWConv) to enhance the understanding of scene context. As shown in Fig. 1(B), given an input feature map $x_{g}$ , the Spike-Driven Deformable Attention can be formulated as:

$$
\text { DeformableSDSA } (\mathbf {p _ {q}}, \mathbf {x _ {g}}) = \sum_ {g = 1} ^ {G} \sum_ {k = 1} ^ {K} \mathbf {W} _ {g} \mathbf {A} _ {g k} \mathbf {W} _ {g} ^ {\prime} \tag {4}
$$

$$
\cdot \mathbf {x} _ {g} (p _ {0} + p _ {k} + \Delta p _ {g k})
$$

where G represents the total number of aggregation groups. For the g-th group, Wg and $W^{\prime}g$ represent the location-irrelevant projection weights. Agk denotes the attention weight corresponding to the k-th sampling point within the g-th group. $\Delta pgk$ represents the offset for the k-th sampling location $p_{k}$ within the g-th group. Subsequently, Wg, Agk, and $\Delta p_{gk}$ are calculated as follows:

$$
\mathbf {W} g = \operatorname{ESC} (x _ {g}), \tag {5}
$$

$$
x _ {g} ^ {\prime} = \mathrm{BN} (\mathrm{DWConv} (S N (x _ {g}))), \tag {6}
$$

$$
\mathbf {A} g k = S N (\mathrm{BN} (\mathrm{Conv} (S N (x _ {g} ^ {'})))) \tag {7}
$$

$$
\Delta p _ {g k} = \mathrm{BN} (\mathrm{Conv} (S N (x _ {g} ^ {'}))). \tag {8}
$$

Where the SN represents the spiking neuron. We add spiking neuron for Agk to convert the attention weight into spike and maintain the effective information in feature query. Finally, we apply an ESC block to the input that has been sampled to the embedding dimension. The output from the SDTE is then fed into the SpikeFPN (Yao et al. 2024a) to generate per-pixel embedding.

Spike-Driven Transformer Decoder The Spike-Driven Transformer Decoder (SDTD) contains a Spike-Driven Cross-Attention (SDCA) layer, a Spike-Driven Self-Attention (SDSA) layer, and a Channel-MLP layer. The formulation of SDTD can be written as:

![](images/ddd9bb0801192cbd0990b8b04999a8066bf2a4754c3eb527329d31d1ea14ff3e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Spike Query"] --> B["Sampled Query"]
    B --> C["Attention Weight"]
    D["Offsets"] --> B
    E["Information Deficiency"] --> B
    F["Offsets"] --> B
    G["Offsets"] --> B
    H["Offsets"] --> B
    I["Offsets"] --> B
    J["Offsets"] --> B
    K["Offsets"] --> B
    L["Offsets"] --> B
    M["Offsets"] --> B
    N["Offsets"] --> B
    O["Offsets"] --> B
    P["Offsets"] --> B
    Q["Offsets"] --> B
    R["Offsets"] --> B
    S["Offsets"] --> B
    T["Offsets"] --> B
    U["Offsets"] --> B
    V["Offsets"] --> B
    W["Offsets"] --> B
    X["Offsets"] --> B
    Y["Offsets"] --> B
    Z["Offsets"] --> B
    AA["Offsets"] --> B
    AB["Offsets"] --> B
    AC["Offsets"] --> B
    AD["Offsets"] --> B
    AE["Offsets"] --> B
    AF["Offsets"] --> B
    AG["Offsets"] --> B
    AH["Offsets"] --> B
    AI["Offsets"] --> B
    AJ["Offsets"] --> B
    AK["Offsets"] --> B
    AL["Offsets"] --> B
    AM["Offsets"] --> B
    AN["Offsets"] --> B
    AO["Offsets"] --> B
    AP["Offsets"] --> B
    AQ["Offsets"] --> B
    AR["Offsets"] --> B
    AS["Offsets"] --> B
    AT["Offsets"] --> B
    AU["Offsets"] --> B
    AV["Offsets"] --> B
    AW["Offsets"] --> B
    AX["Offsets"] --> B
    AY["Offsets"] --> AY
    AZ["Offsets"] --> AZ
    BA["Offsets"] --> BA
    BB["Offsets"] --> BB
    BC["Offsets"] --> BC
    BD["Offsets"] --> BD
    BE["NOT"]
    BF["NOT"]
```
</details>

![](images/2cb006fd1f1f786ff3f42087b482a299cf1367e161283a690a5c4c6b57d72a9b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Query"] --> B["Sampled Query"]
    B --> C["Spike Attention Weight"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
```
</details>

Figure 2: Illustration of offset-sampling operation in Spike-Driven Deformable Attention (SDDA). While directly spiking the query (Sampled Spike Query) leads to information loss during attention, spiking the attention weights (Spike Attention Weight) effectively preserves this crucial query information.

$$
\mathbf {Q} ^ {\prime} = \mathbf {Q} + \mathrm{SDCA} (\mathbf {Q}, \mathbf {F} _ {\mathbf {i}}), i = 1, 2, 3 \tag {9}
$$

$$
\mathbf {Q} ^ {\prime \prime} = \mathbf {Q} ^ {\prime} + \mathrm{SDSA} (\mathbf {Q} ^ {\prime}), \tag {10}
$$

$$
\mathbf {Q} ^ {\prime \prime \prime} = \mathbf {Q} ^ {\prime \prime} + \text { ChannelMLP } (\mathbf {Q} ^ {\prime \prime}). \tag {11}
$$

where $Q \in R^{N \times C}$ are the N learnable query with with learnable positional embedding and $F_{i} \in R^{H_{i} \times W_{i} \times B_{i}}$ indicate the multi-scale feature maps obtained from the pixel decoder, with $i \in \{1, 2, 3\}$ .

Additionally, we replace the re-parameterization convolution (Yao et al. 2024a) with the combination of Linear and BatchNorm for energy efficiency. The formulation of SDSA can be written as:

$$
\mathbf {Q} _ {\mathbf {s}}, \mathbf {K} _ {\mathbf {s}}, \mathbf {V} _ {\mathbf {s}} = S N (\mathrm{BN} (\operatorname{Conv} (S N (\mathbf {Q} _ {\mathbf {l} - \mathbf {1}})))), \tag {12}
$$

$$
\mathbf {Q} _ {\mathbf {l}} = \mathrm{BN} (\operatorname{Conv} (S N (\mathbf {Q} _ {\mathbf {s}} \mathbf {K} _ {\mathbf {s}} ^ {T} \mathbf {V} _ {\mathbf {s}} * S c a l e)) \tag {13}
$$

where the scale of SDSA can be re-parameterized into the spiking neuron's threshold. Note that the SDCA obtains the $K_{s}$ and $V_{s}$ from multi-scale feature maps $F_{i}$ .

Spike-Driven Mask Embedding The mask embedding module in Mask2Former (Cheng et al. 2022) uses a Multi-Layer Perceptron (MLP) to convert the per-segment embedding Q to N mask embedding $\zeta_{mask} \in R^{N \times C}$ , where C denotes the object class and N denotes the number of query and obtain the binary mask prediction $\hat{M} \in [0,1]$ through dot product between per-segment embedding M and per-pixel embedding $\zeta_{pixel}$ . However, the rich diversity semantic

<table><tr><td>Method</td><td>Model</td><td>mIoU (%)</td><td>Params (M)</td><td> $T \times D$ </td><td>Power (mJ)</td></tr><tr><td colspan="6">Pascal VOC2012</td></tr><tr><td rowspan="3">ANN</td><td>FCN(Long, Shelhamer, and Darrell 2015)</td><td>62.2</td><td>49</td><td>-</td><td>909.6</td></tr><tr><td>DeepLabv3(Chen et al. 2017)</td><td>66.7</td><td>68</td><td>-</td><td>1240.6</td></tr><tr><td>DeepLabv3+(Chen et al. 2018)</td><td>77.2</td><td>41</td><td>-</td><td>326.6</td></tr><tr><td>ANN2SNN</td><td>Spike Calibration(Li et al. 2022)</td><td>59.6</td><td>-</td><td>256×1</td><td>-</td></tr><tr><td rowspan="8">SNN</td><td>SpikeFCN(Kim, Chough, and Panda 2022)</td><td>9.9</td><td>50</td><td>20×1</td><td>383.5</td></tr><tr><td>SpikeDeepLab(Kim, Chough, and Panda 2022)</td><td>22.3</td><td>68</td><td>20×1</td><td>523.2</td></tr><tr><td rowspan="2">SpikeFPN(Yao et al. 2024a)</td><td>58.1</td><td>17</td><td>1×1</td><td>81.4</td></tr><tr><td>61.1</td><td>59</td><td>4×1</td><td>179.4</td></tr><tr><td rowspan="4">Spike2Former (Ours)</td><td>61.8</td><td>34</td><td>1×2</td><td>50.6</td></tr><tr><td>62.1</td><td>34</td><td>2×2</td><td>98.3</td></tr><tr><td>75.1</td><td>34</td><td>1×4</td><td>63.0</td></tr><tr><td>75.4</td><td>34</td><td>4×4</td><td>221.9</td></tr><tr><td colspan="6">ADE20k</td></tr><tr><td rowspan="5">ANN</td><td>DeepLabv3+(Chen et al. 2018)</td><td>42.7</td><td>41</td><td>-</td><td>818.8</td></tr><tr><td rowspan="2">MaskFormer(Cheng, Schwing, and Kirillov 2021)</td><td>44.5</td><td>41</td><td>-</td><td>243.8</td></tr><tr><td>46.7</td><td>42</td><td>-</td><td>253</td></tr><tr><td rowspan="2">Mask2Former(Cheng et al. 2022)</td><td>47.2</td><td>44</td><td>-</td><td>326.6</td></tr><tr><td>47.7</td><td>47</td><td>-</td><td>340.6</td></tr><tr><td rowspan="2">SNN</td><td>SpikeFPN(Yao et al. 2024a)</td><td>33.6</td><td>17</td><td>4×1</td><td>88.1</td></tr><tr><td>Spike2Former (Ours)</td><td>46.3</td><td>34</td><td>1×4</td><td>68.0</td></tr><tr><td colspan="6">CityScapes</td></tr><tr><td rowspan="5">ANN</td><td>DeepLabv3+(Chen et al. 2018)</td><td>79.1</td><td>65</td><td>-</td><td>1241.3</td></tr><tr><td rowspan="2">MaskFormer(Cheng, Schwing, and Kirillov 2021)</td><td>78.5</td><td>41</td><td>-</td><td>463.5</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="2">Mask2Former(Cheng et al. 2022)</td><td>79.6</td><td>44</td><td>-</td><td>601.6</td></tr><tr><td>80.6</td><td>47</td><td>-</td><td>623.4</td></tr><tr><td rowspan="2">SNN</td><td>SCGNet-L(Zhang, Fan, and Zhang 2023)</td><td>66.5</td><td>1.9</td><td>4×1</td><td>24.6</td></tr><tr><td>Spike2Former (Ours)</td><td>75.2</td><td>34</td><td>1×4</td><td>93.8</td></tr></table>

Table 1: The semantic segmentation results on ADE20k, CityScapes, and Pascal VOC2012 are presented. We obtain +14.3%, +12.7%, and +9.1%mIoU on Pascal VOC, ADE20k, and CityScapes separately compared with the state-of-the-art SNN method. The results of MaskFormer and Mask2Former are presented with R50 (Above) and Swin-t (Below) as backbone, respectively. Power is the estimation of energy consumption as formulated in (Yao et al. 2024b).

information corresponding to the segment object within N mask embedding $\zeta_{mask}$ will suffer a significant reduction when converting to binary spikes and further affect the binary mask predictions. Furthermore, as shown in Fig. 3 we also found the spike degradation phenomenon in mask embedding that the spiking firing rate before the embedding operation suffers a significant loss and infect the performance. Thus, we proposed a Membrane Embedding Shortcut (MEShortcut) to connect the membrane potential from the output of the transformer decoder to mask embedding, which significantly ensures the training stability and enhances the effective representation of N mask embedding $\zeta_{mask}$ .

As shown in Fig. 1(A), we build a ME-Shortcut with a channel convolution (Howard et al. 2019) parallel with the MLP layer to enhance the precise representation of $\zeta_{mask}$ and reduce the spike degradation phenomenon. In the following, we convert the $\zeta_{mask}$ to spikes before multiplying to pixel embedding $\zeta_{pixel}$ for spike-driven. Finally, we obtain the

binary mask predictions by:

$$
\zeta_ {m a s k} = S N (\mathrm{MLP} (S N (\mathbf {Q})) + w _ {s} * \mathrm{BN} (\operatorname{Conv} (S N (\mathbf {Q})))), \tag {14}
$$

$$
\hat {\mathbf {M}} = \zeta_ {\text { mask }} \cdot \zeta_ {\text { pixel }}. \tag {15}
$$

where the $SN(\cdot)$ is the spiking neuron layer, $w_{s}$ is a learnable weight initialized with 1 to select key feature.

# NI-LIF Spiking Neuron

The spiking neuron layer integrates spatio-temporal information into the membrane potential and then converts it into binary spikes for spike-driven computing in the following layer. Different from the image classification task, dense prediction necessitates a higher demand for numerical stability. Recent work (Luo et al. 2025) shows a performance increase in Object Detection with Integer Leaky Integrate-and-Fire (I-LIF). However, when we attempt to extend the I-LIF to a more complex architecture like Mask2Former, we found

that gradient instability occurs, especially in the spike-driven transformer decoder layer. We argue that the potential reason is that the interaction between object queries and multi-scale feature maps requires numerical stability. However, the large integer value in multi-scale feature maps obtained from I-LIF failed to quantify the precise value and brought quantization error and information loss. Therefore, we intend to tackle this issue by normalizing the $S[t]$ with the virtual timesteps D to maintain the gradient stable and preserve the information.

We improved I-LIF through normalizing integer-value with virtual timesteps to enhance the numerical and gradient stability, as depicted in Fig. 1(C), whose dynamics are:

$$
U [ t ] = H [ t - 1 ] + X [ t ], \tag {16}
$$

$$
S [ t ] = \text { Clip } (\text { round } (U [ t ]), 0, D) / D, \tag {17}
$$

$$
H [ t ] = \beta (U [ t ] - S [ t ] \times D). \tag {18}
$$

where $X[t]$ is the spatial input current at timestep $t$ , $U[t]$ denotes the membrane potential that integrates the temporal information $H[t - 1]$ and spatial $X[t]$ . round $(\cdot)$ is a round operation, Clip $(x, min, max)$ implies clipping the input $x$ to [min, max], and $D$ is a hyper-parameter to emit the maximum integer value.

Inference As shown in Fig. 1(C), I-LIF (Luo et al. 2025) converts the integer spikes into binary spikes during inference stages, and the input to the spiking neuron at $l + 1$ layer can be described as:

$$
X ^ {l + 1} [ t ] = W ^ {l} S ^ {l} [ t ] \tag {19}
$$

We also extend the T time step to $T \times D$ , and convert the normalized value $S^{l}[t]$ to a spike sequence $\{S^{l}[t,d]\}_{d=1}^{D}$ , which can be written as:

$$
\sum_ {d = 1} ^ {D} S ^ {l} [ t, d ] = S ^ {l} [ t ] \times D \tag {20}
$$

Therefore, the input of the neuron at $l + 1$ layer can be calculated by:

$$
X ^ {l} [ t ] = \sum_ {d = 1} ^ {D} (W ^ {l} (S ^ {l} [ t, d ] \times D)), \tag {21}
$$

$$
X ^ {l} [ t ] = \sum_ {d = 1} ^ {D} (W _ {D} ^ {l} (S ^ {l} [ t, d ])).
$$

where the $W_{D}^{l} = W^{l} \times D$ . The spike sequence $S^{l}[t, d]$ only contains 0/1 after being multiplied by the quantization step, and the MAC (Multiply-Accumulation operations) can be converted to sparse AC (Accumulation operations), which enables sparse addition during inference.

# Experiment

Dataset. We conduct semantic segmentation on ADE20k (Zhou et al. 2017), CityScapes (Cordts et al. 2016), and Pascal VOC2012 (Everingham et al. 2010) datasets. The details of the training strategy are shown in Tab. 2.

Training setting. We present all our main semantic segmentation results in mean Intersection over Union (mIoU) under single scale inference setting. We use Meta-Spikeformer (Params:15M) (Yao et al. 2024a) as our backbone, which is pre-trained on ImageNet-1k for 200 epochs. More training details can be found in the Appendix.

<table><tr><td>Setting</td><td>ADE20k</td><td>CityScapes</td><td>PasvalVOC2012</td></tr><tr><td>Input Size</td><td>512×512</td><td>512×1024</td><td>512×512</td></tr><tr><td>Learning Rate</td><td>2e-4</td><td>2e-3</td><td>2e-3</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Training Steps</td><td>160k</td><td>90k</td><td>80k</td></tr></table>

Table 2: Hyper-parameters setting in Spike2Former.

# Experiment Results

In Tab. 1, we comprehensively compare Spike2Former with other ANN and SNN methods in mIoU, parameters, and power. The proposed Spike2Former significantly improves the performance upper bound of SNNs on three public datasets. We obtain 46.3% mIoU, 75.2% mIoU, and 75.4% mIoU in ADE20k, CityScapes, and Pascal VOC2012 which is +14.3%, +9.1%, and +14.3% higher than the previous state-of-the-art SNN method (Yao et al. 2024a; Zhang, Fan, and Zhang 2023), respectively. Spike2Former also demonstrates significant advantages over the existing SNNs in terms of energy consumption: Spike2Former vs. SpikeFPN (Yao et al. 2024a): mIoU 44.6% vs. 33.6%; Power: 68mJ vs. 88.1mJ in ADE20k dataset and Spike2Former vs. SpikeFPN: mIoU 75.4% vs. 61.1%; Power 63.0mJ vs. 179.4mJ in Pascal VOC2012 dataset. Moreover, the performance gap between SNNs and ANNs is significantly narrowed. Spike2Former got +1.2% mIoU compared with MaskFormer(R50) and is competitive with the current classical ANN architecture Mask2Former(R50) in ADE20k dataset, which is 46.3% vs. 44.5% vs. 47.2% mIoU in mIoU while the energy consumption is much lower for 3.58× and 4.80× energy efficiency.

# Ablation study

We conduct ablation studies on the various components of Spike2Former to evaluate the contribution of each parts.

Spike-Driven Deformable Transformer Encoder Tab. 3 highlights the performance of our spike-driven Deformable Transformer Encoder (SDTE). As shown in Fig. 2, directly spiking query features reduces mIoU by 3.2%, likely due to excess retention of attention weights, diminishing effective information of query. Our SDTE improves mIoU by 2.5% over SpikeFPN (Yao et al. 2024a) without Transformer encoder and by 1.1% over a vanilla Transformer. While using multi-scale features as queries, as in Mask2Former, yields 46.7% mIoU, it doubles energy consumption (68.0mJ vs. 136.5mJ), prompting us to prioritize single-scale queries for energy efficiency.

Information Deficiency in Query Preserving query information is crucial for effectively applying SNNs to Mask2Former. As demonstrated in Tab. 3, removing the ME-Shortcut connection leads to a significant performance drop (approximately -4.5% mIoU). Fig. 3 visualizes the average binary mask prediction for each query over the validation set of ADE20k, revealing that incorporating shortcut connections (ME-Shortcut) results in more sophisticated and dis

<table><tr><td>Ablation</td><td>Method</td><td>Power (mJ)</td><td>mIoU (%)</td></tr><tr><td></td><td>Spike2Former(Baseline)</td><td>68.0</td><td>46.3</td></tr><tr><td>ANN Conversion</td><td>Mask2Former(Spiking Version)</td><td>-</td><td>5.2</td></tr><tr><td rowspan="3">Spike-Driven Mask Embedding</td><td>Vanilla Spike Mask Embedding</td><td>67.9</td><td>41.8(-4.5)</td></tr><tr><td>w/o. Conv in ME-Shortcut</td><td>67.9</td><td>45.7(-0.6)</td></tr><tr><td>ME-Shortcut → Membrane Shortcut</td><td>67.8</td><td>45.0(-1.3)</td></tr><tr><td rowspan="6">Spike-driven Deformable Transformer Encoder</td><td>w/o. SD Deformable Transformer Encoder</td><td>65.2</td><td>43.8(-2.5)</td></tr><tr><td>w/o. ESC in SD Deformable attention</td><td>64.5</td><td>45.6(-0.7)</td></tr><tr><td>w/o. DWConv in SD Deformable attention</td><td>67.8</td><td>45.7(-0.6)</td></tr><tr><td>Spiking in Query</td><td>67.8</td><td>43.1(-3.2)</td></tr><tr><td>Multi-Scale feature as query</td><td>136.5</td><td>46.7(+0.4)</td></tr><tr><td>Vanilla Transformer Encoder</td><td>66.9</td><td>44.9(-1.4)</td></tr><tr><td rowspan="4">Normalize Method in NI-LIF</td><td>I-LIF</td><td>68.1</td><td>37.9(-8.4)</td></tr><tr><td>Norm Only in Cross-Attn</td><td>68.1</td><td>43.2(-3.1)</td></tr><tr><td>Norm with Const 8</td><td>67.4</td><td>44.7(-1.3)</td></tr><tr><td>Norm with Sigmoid&amp;L2</td><td>-</td><td>*</td></tr></table>

Table 3: We conduct ablation on the proposed Spike-Driven Mask Embedding (SDME), Spike-driven Deformable Transformer Encoder (SDTE) and Normalized Integer-LIF (NI-LIF) in ADE20k dataset. We set $T \times D = 1 \times 4$ and modify just one modification of the baseline to test how the power and performance vary. \* Does not converge. The ablation on ANN indicates the direct conversion of Mask2Former with SNNs.

![](images/6bc58e3445de1ad2f1463f2e02a25c42da0caf1f9e7d4bf4a737aa6519b45d07.jpg)  
Figure 3: Segment Patterns Represents by Queries. We Average its corresponding segment predictions over the whole validation set of ADE20k. All the predictions are resized to the resolution of $200 \times 200$ for illustration purposes.

tinct patterns within the mask embedding, indicating higher-quality binary mask predictions. Moreover, ME-Shortcut increases the firing rate of the final spiking neuron threefold (from 0.08626 to 0.26617). This suggests that the performance drop without ME-Shortcut is attributable to spike degradation, which our proposed shortcut effectively mitigates. Despite its conceptual simplicity, ME-Shortcut is a key design for the successful application of SNNs to complex architecture.

NI-LIF Spiking Neuron Tab. 3 shows that directly applying I-LIF (Luo et al. 2025) to Spike2Former results in sub-optimal performance (37.9% mIoU compared to 46.3%), primarily due to information deficiency in the cross-attention layer of transformer decoder. Normalizing activations within the cross-attention layer alone yields a substantial +5.3% mIoU improvement compared with I-LIF, highlighting the

need for numerical stability and precise representation of Transformer-based model. Furthermore, when applying the NI-LIF to the whole network, the performance increases 8.4%. This improvement demonstrates the adaptability of NI-LIF in complex architecture and the effectiveness of NI-LIF in reducing the quantization error and enhancing the precise representation of features.

Further analysis in Tab. 1 (Pascal VOC2012) reveals the impact of varying timesteps (T) and quantization steps (D). Increasing timesteps from T=1 to T=2 (with D=2) raises energy consumption from 50.6 mJ to 98.3 mJ with only 0.3%mIoU. Additionally, increasing quantization steps from D=2 to D=4 improves performance from 61.8% to 75.1% mIoU, with a moderate power increase from 50.6 mJ to 63.0 mJ. This suggests that increasing quantization steps can enhance the performance while maintaining reasonable energy consumption.

# Conclusion

This work reduces the performance gap between Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs) in image segmentation. The proposed Spike2Former introduces architectural innovations and spiking neuron optimizations, including two key modifications to address information deficiency. The Normalized Integer LIF (NI-LIF) mitigates information loss and enhances training stability by converting normalized integers into binary spikes. Spike2Former achieves state-of-the-art performance on three benchmark datasets, highlighting the potential of SNNs for complex segmentation tasks. Our analysis of spike degradation and information deficiency emphasizes the need to reduce information loss in SNNs for advanced architectures. This work lays a foundation for extending SNNs to dense prediction tasks with sophisticated designs.

# Acknowledgments

This work was partially supported by CAS Project for Young Scientists in Basic Research (YSBR116), National Distinguished Young Scholars (62325603), National Natural Science Foundation of China (62236009, U22A20103, 62441606, 62406322), Beijing Science and Technology Plan (Z241100004224011), Beijing Natural Science Foundation for Distinguished Young Scholars(JQ21015), China Postdoctoral Science Foundation (GZB20240824, 2024M753497), and Natural Science Foundation of China under Grant (62394314, 62394311).

# References

Bu, T.; Fang, W.; Ding, J.; DAI, P.; Yu, Z.; and Huang, T. 2022. Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks. In International Conference on Learning Representations.   
Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-end object detection with transformers. In European conference on computer vision, 213–229. Springer.   
Cavagnero, N.; Rosi, G.; Cuttano, C.; Pistilli, F.; Ciccone, M.; Averta, G.; and Cermelli, F. 2024. Pem: Prototype-based efficient maskformer for image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15804–15813.   
Chen, L.-C.; Papandreou, G.; Kokkinos, I.; Murphy, K.; and Yuille, A. L. 2017. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4): 834–848.   
Chen, L.-C.; Zhu, Y.; Papandreou, G.; Schroff, F.; and Adam, H. 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), 801–818.   
Cheng, B.; Misra, I.; Schwing, A. G.; Kirillov, A.; and Girdhar, R. 2022. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1290–1299.   
Cheng, B.; Schwing, A.; and Kirillov, A. 2021. Per-pixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34: 17864–17875.   
Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3213–3223.   
Davies, M.; Srinivasa, N.; Lin, T.-H.; Chinya, G.; Cao, Y.; Choday, S. H.; Dimou, G.; Joshi, P.; Imam, N.; Jain, S.; et al. 2018. Loihi: A neuromorphic manycore processor with on-chip learning. IEEE Micro, 38(1): 82–99.

Ding, H.; Wang, B.; Kang, G.; Li, W.; He, C.; Zhao, Y.; and Wei, Y. 2023. DropQueries: A Simple Way to Discover Comprehensive Segment Representations. IEEE Transactions on Multimedia.   
Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.; and Zisserman, A. 2010. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88: 303–338.   
Fang, W.; Yu, Z.; Chen, Y.; Huang, T.; Masquelier, T.; and Tian, Y. 2021. Deep residual learning in spiking neural networks. Advances in Neural Information Processing Systems, 34: 21056–21069.   
Guo, Y.; Chen, Y.; Zhang, L.; Liu, X.; Wang, Y.; Huang, X.; and Ma, Z. 2022. IM-loss: information maximization loss for spiking neural networks. Advances in Neural Information Processing Systems, 35: 156–166.   
Howard, A.; Sandler, M.; Chu, G.; Chen, L.-C.; Chen, B.; Tan, M.; Wang, W.; Zhu, Y.; Pang, R.; Vasudevan, V.; et al. 2019. Searching for mobilenetv3. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1314–1324.   
Hu, J.; Yao, M.; Qiu, X.; Chou, Y.; Cai, Y.; Qiao, N.; Tian, Y.; XU, B.; and Li, G. 2024a. High-Performance Temporal Reversible Spiking Neural Networks with $\text{O}$ (L) $\text{O}$ Training Memory and $\text{O}$ (1) $\text{O}$ Inference Cost. In Forty-first International Conference on Machine Learning.   
Hu, Y.; Deng, L.; Wu, Y.; Yao, M.; and Li, G. 2024b. Advancing Spiking Neural Networks Toward Deep Residual Learning. IEEE Transactions on Neural Networks and Learning Systems, 1–15.   
Jin, S.; Li, S.; Li, T.; Liu, W.; Qian, C.; and Luo, P. 2023. You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception. arXiv preprint arXiv:2312.05525.   
Kim, Y.; Chough, J.; and Panda, P. 2022. Beyond classification: Directly training spiking neural networks for semantic segmentation. Neuromorphic Computing and Engineering, 2(4): 044015.   
Kirkland, P.; Di Caterina, G.; Soraghan, J.; and Matich, G. 2020. SpikeSEG: Spiking segmentation via STDP saliency mapping. In 2020 International Joint Conference on Neural Networks (IJCNN), 1–8. IEEE.   
Leroux, N.; Finkbeiner, J.; and Neftci, E. 2023. Online transformers with spiking neurons for fast prosthetic hand control. In 2023 IEEE Biomedical Circuits and Systems Conference (BioCAS), 1–6. IEEE.   
Li, F.; Zhang, H.; Xu, H.; Liu, S.; Zhang, L.; Ni, L. M.; and Shum, H.-Y. 2023. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3041–3050.   
Li, Y.; He, X.; Dong, Y.; Kong, Q.; and Zeng, Y. 2022. Spike calibration: Fast and accurate conversion of spiking neural network for object detection and segmentation. arXiv preprint arXiv:2207.02702.

Long, J.; Shelhamer, E.; and Darrell, T. 2015. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3431–3440.   
Luo, X.; Yao, M.; Chou, Y.; Xu, B.; and Li, G. 2025. Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection. In European Conference on Computer Vision, 253–272. Springer.   
Maass, W. 1997. Networks of spiking neurons: the third generation of neural network models. Neural Networks, 10(9):1659–1671.   
Merolla, P. A.; Arthur, J. V.; Alvarez-Icaza, R.; Cassidy, A. S.; Sawada, J.; Akopyan, F.; Jackson, B. L.; Imam, N.; Guo, C.; Nakamura, Y.; et al. 2014. A million spiking-neuron integrated circuit with a scalable communication network and interface. Science, 345(6197): 668–673.   
Neftci, E. O.; Mostafa, H.; and Zenke, F. 2019a. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6): 51–63.   
Neftci, E. O.; Mostafa, H.; and Zenke, F. 2019b. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6): 51–63.   
Patel, K.; Hunsberger, E.; Batir, S.; and Eliasmith, C. 2021. A spiking neural network for image segmentation. arXiv preprint arXiv:2106.08921.   
Roy, K.; Jaiswal, A.; and Panda, P. 2019. Towards spike-based machine intelligence with neuromorphic computing. Nature, 575(7784): 607–617.   
Schuman, C. D.; Kulkarni, S. R.; Parsa, M.; Mitchell, J. P.; Kay, B.; et al. 2022. Opportunities for neuromorphic computing algorithms and applications. Nature Computational Science, 2(1): 10–19.   
Su, Q.; He, W.; Wei, X.; Xu, B.; and Li, G. 2024. Multiscale full spike pattern for semantic segmentation. Neural Networks, 176: 106330.   
Wang, P.; Cai, Z.; Yang, H.; Swaminathan, A.; Manmatha, R.; and Soatto, S. 2024. Mixed-Query Transformer: A Unified Image Segmentation Architecture. arXiv preprint arXiv:2404.04469.   
Wang, Q.; Zhang, T.; Han, M.; Wang, Y.; Zhang, D.; and Xu, B. 2023. Complex dynamic neurons improved spiking transformer network for efficient automatic speech recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 102–109.   
Wu, J.; Xu, C.; Han, X.; Zhou, D.; Zhang, M.; Li, H.; and Tan, K. C. 2021. Progressive tandem learning for pattern recognition with deep spiking neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11): 7824–7840.   
Wu, Y.; Deng, L.; Li, G.; Zhu, J.; and Shi, L. 2018. Spatiotemporal backpropagation for training high-performance spiking neural networks. Frontiers in Neuroscience, 12: 331.

Yao, M.; Hu, J.; Hu, T.; Xu, Y.; Zhou, Z.; Tian, Y.; XU, B.; and Li, G. 2024a. Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips. In The Twelfth International Conference on Learning Representations.   
Yao, M.; Hu, J.; Zhou, Z.; Yuan, L.; Tian, Y.; Xu, B.; and Li, G. 2024b. Spike-driven transformer. Advances in Neural Information Processing Systems, 36.   
Yao, M.; Qiu, X.; Hu, T.; Hu, J.; Chou, Y.; Tian, K.; Liao, J.; Leng, L.; Xu, B.; and Li, G. 2024c. Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training. arXiv preprint arXiv:2411.16061.   
Yao, M.; Richter, O.; Zhao, G.; Qiao, N.; Xing, Y.; Wang, D.; Hu, T.; Fang, W.; Demirci, T.; De Marchi, M.; Deng, L.; Yan, T.; Nielsen, C.; Sheik, S.; Wu, C.; Tian, Y.; Xu, B.; and Li, G. 2024d. Spike-based dynamic computing with asynchronous sensing-computing neuromorphic chip. Nature Communications, 15(1): 4464.   
Yao, M.; Zhao, G.; Zhang, H.; Hu, Y.; Deng, L.; Tian, Y.; Xu, B.; and Li, G. 2023. Attention spiking neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8): 9393–9410.   
Zhang, H.; Fan, X.; and Zhang, Y. 2023. Energy-efficient spiking segmenter for frame and event-based images. Biomimetics, 8(4): 356.   
Zheng, H.; Wu, Y.; Deng, L.; Hu, Y.; and Li, G. 2021. Going deeper with directly-trained larger spiking neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 11062–11070.   
Zhou, B.; Zhao, H.; Puig, X.; Fidler, S.; Barriuso, A.; and Torralba, A. 2017. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 633–641.   
Zhou, Z.; Zhu, Y.; He, C.; Wang, Y.; YAN, S.; Tian, Y.; and Yuan, L. 2023. Spikformer: When Spiking Neural Network Meets Transformer. In The Eleventh International Conference on Learning Representations.

# Spike2Former vs. Mask2Former

# Architecture design

Spike2Former adopts the same meta-architecture as Mask2Former (Cheng et al. 2022), which comprises a backbone for image feature extraction, a Feature Pyramid Network (FPN) pixel decoder with a Multi-scale Deformable self-attention transformer encoder for generating per-pixel embeddings, a stack of transformer decoders to process object queries in relation to the image features, and an embedding module to decode the per-pixel embeddings into binary mask predictions and class embeddings. Spike2Former introduces key modifications to the Multi-Scale Deformable Self-Attention transformer encoder and the mask embedding module, while making minor adjustments to the remaining components. The following section delves into the specific changes implemented within the transformer decoder and pixel decoder.

# Spike-Driven Transformer Decoder

Transformer Decoder Design in Mask2Former In Mask2Former, learnable object queries interact with multiscale image features in a round robin fashion through a cross-attention mechanism. This strategy enables the queries to capture information about objects at various scales. After each Transformer decoder layer, Mask2Former employs an mask embedding module to generate binary mask predictions. These predictions are then converted into attention masks with a Sigmoid activation function and fed into the subsequent cross-attention layer, guiding the object queries to focus on regions containing segmented objects. However, this design proves less effective in the SNNs. The sparse nature of spike-based computation limits the ability of the attention mask, derived from the previous layer's binary predictions, to enhance the query's representation in the subsequent cross-attention layer. Furthermore, incorporating the mask attention can hinder the query's ability to effectively perceive and process the raw image features. Therefore, we omit this design, opting not to use mask attention within the cross-attention layers of our Spike2Former.

Spike-Driven Self-Attention The spike-driven self-attention has various variants. In this work, We take the inspiration of Meta-Spike2Former (Yao et al. 2024a) and proposed a spike-driven self-attention capable of dualing with sequence feature (Object Query). In Mata-Spikeformer, the calculation of self-attention can be written as:

$$
\mathbf {Q} _ {\mathbf {s}} = S N (\text { RepConv } _ {1} (\mathbf {U})), \tag {1}
$$

$$
\mathbf {K} _ {\mathbf {s}} = S N (\operatorname{RepConv} _ {2} (\mathbf {U})), \tag {2}
$$

$$
\mathbf {V} _ {\mathbf {s}} = S N \left(\operatorname{RepConv} _ {3} (\mathbf {U})\right), \tag {3}
$$

$$
\operatorname{SDSA} \left(\mathbf {Q} _ {\mathbf {s}}, \mathbf {K} _ {\mathbf {s}}, \mathbf {V} _ {\mathbf {s}}\right) = S N \left(\mathbf {Q} _ {\mathbf {s}} \left(\mathbf {K} _ {\mathbf {s}} ^ {\mathrm{T}} \mathbf {V} _ {\mathbf {s}}\right)\right), \tag {4}
$$

$$
\mathbf {U} ^ {\prime} = \mathbf {U} + \operatorname{RepConv} _ {4} (S D S A (\mathbf {Q} _ {\mathbf {s}}, \mathbf {K} _ {\mathbf {s}}, \mathbf {V} _ {\mathbf {s}})). \tag {5}
$$

where U represents the input membrane potential, and $\text{RepConv}(\cdot)$ denotes the re-parameterization convolution operation with a kernel size of $3 \times 3$ .

While this type of convolution has demonstrated significant energy efficiency gains with high-resolution images, we

![](images/4b9e0913f8dc0bb7dbbcb80c18edc3c3f57e974e5c95dfc1668dc6523ebb398d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Spike-Driven Transformer Decoder Blocks"] --> B["SD Cross-Attention"]
    A --> C["SD Self-Attention"]
    A --> D["Channel MLP"]
    B --> E["Conv BN"]
    B --> F["Conv BN"]
    B --> G["Conv BN"]
    C --> H["Conv BN"]
    C --> I["Conv BN"]
    C --> J["Conv BN"]
    D --> K["Linear BN"]
    D --> L["Linear BN"]
    M["Feature Map"] --> N["Query"]
    M --> O["Conv"]
    M --> P["Conv"]
    M --> Q["Conv"]
    R["Spike-Driven Cross-Attention"] --> S["Conv BN"]
    T["Spike-Driven Self-Attention"] --> U["Conv BN"]
    V["Query"] --> W["Conv BN"]
    X["Query"] --> Y["Conv BN"]
    Z["Linear"] --> AA["Linear BN"]
    AB["Linear"] --> AC["Linear BN"]
```
</details>

Figure 1: Illustration of the proposed Spike-Driven Transformer Decoder Blocks.

observed that it was not optimal for the specific demands of image segmentation. Therefore, to better suit the characteristics of this task, we replace the $3 \times 3$ re-parameterization convolution with a combination of a $1 \times 1$ standard convolution followed by Batch Normalization. This adjustment provides a more effective balance between computational efficiency and feature representation for accurate segmentation. The calculation of our Spike-Driven Self-Attention(SDSA) can be written as:

$$
\mathbf {Q} _ {\mathbf {s}}, \mathbf {K} _ {\mathbf {s}}, \mathbf {V} _ {\mathbf {s}} = S N (\mathrm{BN} (\operatorname{Conv} (S N (\mathbf {U} _ {\mathbf {l} - \mathbf {1}})))), \tag {6}
$$

$$
\mathbf {U} ^ {\prime} = \mathrm{BN} (\operatorname{Conv} (S N (\mathbf {Q _ {s}} \mathbf {K _ {s}} ^ {T} \mathbf {V _ {s}} * S c a l e)) \tag {7}
$$

where the scale of SDSA can be re-parameterized into the spiking neuron's threshold.

Spike-Driven Cross-Attention The calculation of spike-driven cross-attention are similar to the self-attention operation. Moreover, the $K_{s}$ and $V_{s}$ in cross-attention are obtained from the multi-scale image features generated from the pixel decoder. We calculate the spike-driven cross-attention as follow:

$$
\mathbf {Q} _ {\mathbf {s}} = S N (\mathrm{BN} (\operatorname{Conv} (S N (\mathbf {U} _ {\mathbf {l} - \mathbf {1}})))), \tag {8}
$$

$$
\mathbf {K} _ {\mathbf {s}}, \mathbf {V} _ {\mathbf {s}} = S N (\mathrm{BN} (\operatorname{Conv} (S N (\mathbf {F} _ {\mathbf {i}})))), i = 1, 2, 3, \tag {9}
$$

$$
\mathbf {U} ^ {\prime} = \mathrm{BN} (\operatorname{Conv} (S N (\mathbf {Q} _ {\mathbf {s}} \mathbf {K} _ {\mathbf {s}} ^ {\mathbf {T}} \mathbf {V} _ {\mathbf {s}} * S c a l e)) \tag {10}
$$

Where $\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3$ indicates the multi-scale feature maps in $\{\frac{1}{16}, \frac{1}{8}, \frac{1}{4}\}$ , The scale of SDCA can be re-parameterized into the spiking neuron's threshold.

# Feature Pyramid Network Pixel Decoder

In Spike2Former, we introduce a subtle yet efficient modification to the FPN pixel decoder. Specifically, we substitute the $3 \times 3$ channel convolution in the original Spike-FPN with a more computationally efficient $3 \times 3$ depth-wise convolution. This adjustment aims to reduce the energy consumption associated with generating per-pixel embeddings.

As evidenced by the results presented in Table 1, this alteration yields a notable improvement in energy efficiency. We observe a significant reduction in energy consumption

<table><tr><td>Method</td><td>Power(mJ)</td><td>mIoU(%)</td></tr><tr><td>Vanilla SpikeFPN</td><td>95.8</td><td>45.9</td></tr><tr><td>DWConv1x1 SpikeFPN</td><td>68.0</td><td>46.3</td></tr></table>

Table 1: Ablation study on the depth-wise convolution in pixel decoder in ADE20k dataset. We equipped Spike2Former with different pixel decoder, reporting its energy consumption and performance.

(-27.8 mJ) while simultaneously achieving a slight performance gain of approximately 0.4% mIoU. This outcome highlight the effectiveness of this modification in balancing computational cost and segmentation accuracy.

# Dataset and Training Settings

In this section, we present the detail of dataset, the training setting on each dataset, and the theoretical energy consumption calculation method.

# Semantic segmentation

We conduct semantic segmentation on three popular datasets: ADE20k(Zhou et al. 2017), CityScapes(Cordts et al. 2016), and Pascal VOC2012(Everingham et al. 2010). The evaluation matrix are mean Interact of Union(mIoU).

ADE20k ADE20k contains 20,000 images for training and 2,000 images for validation. The dataset is a subset of ADE20k-Full, featuring 150 semantic categories. In this work, we validate the Spike2Former with an input resolution of $512 \times 512$ and a learning rate of 0.0002. We employ the data augmentation strategy as described in (Cheng et al. 2022). During training, the images are resized so that the shortest side does not exceed 512 pixels. We train the model on ADE20k for 160k iterations with a batch size of 36.

CityScapes CityScapes is an urban, egocentric street view dataset with high-resolution images (1024 × 2048 pixels) across 19 semantic categories. We use 2,975 images for training, 500 images for validation, and 1,525 images for testing. During training, we crop the images to 512 × 1024 and apply data augmentation using the same strategy as in (Cheng et al. 2022). We train the model for 90k iterations with a batch size of 32. During inference, we resize the longer side of the images to 2048 pixels and use multi-scale testing as (Cheng, Schwing, and Kirillov 2021).

Pascal VOC 2012 The Pascal VOC 2012 dataset comprises approximately 11,530 images, of which 5,717 are used for training and 5,823 for testing. The image resolutions vary, typically ranging from 300x500 to 500x400 pixels. The dataset includes 20 semantic categories such as people, animals, vehicles, and furniture. The scenes mainly depict everyday natural environments, including streets, homes, and outdoor settings. We use an input resolution of $512 \times 512$ with a batch size of 32 for training and resize the longer side of the images to 512 pixels during inference.

# Panoptic segmentation

Panoptic segmentation is a more challenging segmentation task that provides comprehensive scene understanding by simultaneously performing semantic and instance segmentation. The core requirement is to assign every pixel a semantic label, such as "road" or "tree," while also distinguishing between different instances of objects within the same category, like multiple "cars." This task is particularly challenging because it must effectively combine instance-level and pixel-level segmentation into a single, coherent output, a contrast to semantic segmentation, which does not differentiate between individual instances of the same class. Panoptic segmentation is essential for advanced applications like autonomous driving, where precise scene interpretation is critical. To the best of our knowledge, no panoptic segmentation results have been reported in SNNs. In this work, we present the first panoptic segmentation results in SNNs on the widely-used COCO2017 panoptic dataset. We evaluate the performance with Panoptic Quality(PQ) in 'All' object, 'Thing', and 'Stuff', denoting as $PQ_{All}$ , $PQ_{Thing}$ , and $PQ_{Stuff}$ , separately.

COCO Panoptic COCO panoptic is one of the most widely used datasets for panoptic segmentation, containing 133 categories (80 "thing" categories with instance-level annotations and 53 "stuff" categories). It includes 118,000 images for training and 5,000 images for validation. All images are sourced from the COCO2017 dataset. We employ large-scale jittering augmentation as described in (Du et al. 2021) and crop the images to $1024 \times 1024$ during training.

# Theoretical Energy Consumption

The power consumption of ANNs and SNNs can be estimated using the following equations:

$$
E _ {A N N} = O ^ {2} \times C _ {i n} \times C _ {o u t} \times k ^ {w} \times E _ {M A C}, \tag {11}
$$

$$
E _ {S N N} = (T \times D) \times R _ {L I F} \times O ^ {2} \tag {12}
$$

$$
\times C _ {i n} \times C _ {o u t} \times k ^ {2} \times E _ {A C}. \tag {13}
$$

Here, O represents the output size, $C_{in}$ and $C_{out}$ denote the number of input and output channels respectively, k is the kernel size, $R_{LIF}$ represents the firing rate of the LIF spiking neuron, T is the timestep, and D is the upper limit of integer-activation during training.

We adopt a widely used energy consumption evaluation method for SNNs(Yao et al. 2024b,a; Wang et al. 2023), assuming a 32-bit floating-point implementation on a 45nm technology. In this context, $E_{MAC} = 4.6pJ$ and $E_{AC} = 0.9pJ$ represent the energy consumed per MAC (Multiply-AC cumulate operations) and AC (AC cumulate operations) operation respectively (Yao et al. 2024b).

As shown in Eq. 13, the reduced power consumption of SNNs stems from their reliance on sparse addition operations, reflected in the use of $E_{AC}$ .

# Training Setting

We conducted our experiments using the popular codebases MMSegmentation (Contributors 2020) and MMDetection (Chen et al. 2019), which are widely used for semantic

<table><tr><td>Method</td><td>Model</td><td>All(%)</td><td>PQThing(%)</td><td>Stuff(%)</td><td>Power(mJ)</td><td>Params(M)</td></tr><tr><td rowspan="4">ANN</td><td rowspan="2">MaskFormer</td><td>46.5</td><td>51.0</td><td>39.8</td><td>832.6</td><td>45</td></tr><tr><td>47.7</td><td>51.7</td><td>41.7</td><td>823.4</td><td>42</td></tr><tr><td rowspan="2">Mask2Former</td><td>51.9</td><td>57.7</td><td>43.0</td><td>1039.6</td><td>44</td></tr><tr><td>53.2</td><td>59.3</td><td>44.0</td><td>1067.2</td><td>47</td></tr><tr><td rowspan="2">SNN</td><td>SpikeFPN*</td><td>14.5</td><td>9.8</td><td>21.9</td><td>448.0</td><td>36</td></tr><tr><td>Spike2Former(Ours)</td><td>30.9</td><td>31.7</td><td>30.0</td><td>222.6</td><td>34</td></tr></table>

Table 2: The performance of panptic segmentation on COCO2017 with 133 categories. We use $T \times D = 1 \times 4$ in COCO2017 panoptic. The result of MaskFormer and Mask2Former are equipped with R50(above) and Swin-t(Below) as backbone. \* indicates the self-implement result (Yao et al. 2024a). 

<table><tr><td>Model</td><td>mIoU(%)</td><td>T × D</td><td>Power(mJ)</td></tr><tr><td rowspan="7">Spike2Former</td><td>5.2*</td><td>1×1</td><td>60.2</td></tr><tr><td>61.8</td><td>1×2</td><td>50.6</td></tr><tr><td>62.1</td><td>2×2</td><td>98.3</td></tr><tr><td>75.1</td><td>1×4</td><td>63.0</td></tr><tr><td>75.3</td><td>2×4</td><td>117.4</td></tr><tr><td>75.4</td><td>4×4</td><td>221.9</td></tr><tr><td>76.0</td><td>1×8</td><td>66.9</td></tr></table>

Table 3: Ablation study on the various $T \times D$ in Pascal VOC2012 with input resolution of $512 \times 512$ . \* indicates the setting is not work well because of the huge information deficiency.

and panoptic segmentation in the context of ANNs, respectively. To enable the use of these versatile frameworks within the SNN domain, we introduced modifications to support SNN training. These adaptations pave the way for efficient development of future SNN-based segmentation models. All the experiment are train and evaluate on 4 NVIDIA A100 GPUs.

# Experiment Result on Panoptic Segmentation

Tab. 2 presents a comprehensive comparison of Spike2Former against state-of-the-art ANN and SNN methods on the COCO2017 panoptic segmentation benchmark, considering metrics such as mIoU, parameter count, and power consumption. For this evaluation, we re-implemented Spike-FPN (Yao et al. 2024a) within our Spike-mmdetection codebase.

Our proposed Spike2Former significantly advances the performance limits of SNNs for this task. We achieve 30.9% PQ, 31.7% PQ, and 75.4% PQ for the "All," "Things," and "Stuff" categories, respectively. These results represent substantial improvements of +16.4%, +21.9%, and +8.1% over the previous SNN state-of-the-art, Spike-FPN (Yao et al. 2024a). Furthermore, Spike2Former exhibits considerable advantages in energy efficiency compared to existing SNNs. Specifically, Spike2Former attains a PQ $_{All}$ of 30.9% with a power consumption of 222.6mJ, while SpikeFPN (Yao et al. 2024a) achieves 33.6% PQ $_{All}$ at the cost of 448.0mJ.

While there remains a performance gap between Spike2Former and ANN methods, our approach demonstrates a more energy-efficient way to perform panoptic segmentation. For instance, Spike2Former exhibits a $4.8\times$ improvement in energy efficiency compared to Mask2Former with Swin-t as backbone.

# Ablation Study

# Ablation Study on T and D

We conducted an ablation study to investigate the effects of varying timesteps (T) and quantization steps (D). As illustrated in Table 3, increasing T from 1 to 2 (while keeping D constant at 2) leads to a considerable rise in energy consumption, from 50.6 mJ to 98.3 mJ. Conversely, increasing D from 2 to 4 (with T fixed at 1) yields a substantial performance boost, raising the mIoU from 61.8% to 75.1%, while incurring only a moderate power increase from 50.6 mJ to 63.0 mJ. This suggests that increasing the quantization steps is an effective strategy for enhancing performance while maintaining reasonable energy efficiency.

Notably, we observed a significant performance drop when the virtual timestep was set to 1. We hypothesize that this is because a pure 0/1 spike sequence is not well-suited for the Mask2Former architecture, adversely affecting numerical stability and hindering effective interactions between features. Therefore, we recommend using a virtual timestep of $2 \sim 4$ to normalize the output into integer spikes for complex models.

# Visualization

As shown in Fig. 2, Fig. 3, and Fig. 4, we visualize the semantic segmentation results of Spike2Former on the ADE20k, CityScapes, and Pascal VOC2012 datasets. Our proposed Spike2Former demonstrates remarkable capabilities in identifying small objects, accurately classifying object categories, and producing precise segmentation along object boundaries.

![](images/4e8cc452f314c6b4f589f04ca94ec031941fd7c76ceba223dd184f300ce4f717.jpg)  
Figure 2: Visualization of semantic segmentation predictions on the ADE20K dataset: Spike2Former with Meta-Spikeformer backbone which achieves 46.3 mIoU on the validation set. First and third columns: ground truth. Second and fourth columns: prediction. Last row shows failure cases

![](images/cd05d2f07e7a09a7194399034b1ab9cea72d801e97bac93d4e1f2f56a1aab313.jpg)  
Figure 3: Visualization of semantic segmentation predictions on the CityScapes dataset: Spike2Former with Meta-Spikeformer backbone which achieves 75.2 mIoU on the validation set. First columns: ground truth. Second columns: prediction. Last row shows failure cases

![](images/b452abce4c5d43fa066aa4e1290e9b7f725a6591f89ad74bc830d40423b567ef.jpg)

<details>
<summary>natural_image</summary>

Two side-by-side images showing a person standing beside a large green agricultural machine in a forest setting (no text or symbols visible)
</details>

![](images/63762ac0e5707c72385f1db9cc32cd7df345c26643da2dc7919b33de041df6cf.jpg)

<details>
<summary>natural_image</summary>

Side-by-side comparison of a light blue motorcycle against a dark background (no text or symbols visible)
</details>

![](images/cd5a5c41cdc5afbec1b8b0574a7bdaea269deab3b1628a2a106835e27d2c4a7f.jpg)

<details>
<summary>natural_image</summary>

Two identical yellow birds in flight against a dark cloudy sky (no text or symbols)
</details>

![](images/5aeae9556e87e6d01cbb6eb469af483a2096bf308064520453a7ab76e5513143.jpg)

<details>
<summary>natural_image</summary>

Two human hands reaching toward each other against a dark background with faint floral patterns (no text or symbols)
</details>

![](images/416e4aae0d0a48de0be101f0988fec8e8769d324e4bbf93ac68ab1a7d8d86ef7.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing a pink silhouette of a horse in different poses, set against a dark outdoor background with trees and a building (no text or symbols visible)
</details>

![](images/092d4dfbc41ebf198f44eb72f7cf8b80ed3ad88e56ff019f97002379e5f9a1b7.jpg)

<details>
<summary>natural_image</summary>

Two side-by-side images showing red 3D-rendered airplanes in flight against a dark background, with no visible text or symbols.
</details>

![](images/2c35cf219cec5233c702a5af2dfd520a6fd5913841a4e93ee41bf40fc970fabf.jpg)

<details>
<summary>natural_image</summary>

Side-by-side profile of a person wearing a cap and T-shirt, holding two purple objects (no text or symbols visible)
</details>

![](images/0b0fb87172102bbcfc9a257ecdbea74ea053ddcc07d1f64a8502e324762f4cac.jpg)

<details>
<summary>natural_image</summary>

Interior scene with two dark room views and a green-lit sofa, no visible text or symbols
</details>

![](images/6af2d5a46e94a73fa60a6327b82fb3a4816220022009de019e63419fb2bc3e63.jpg)

<details>
<summary>natural_image</summary>

Illustration of a cyclist riding a bicycle with green lane markings (no text or symbols)
</details>

![](images/021f0879dcb4727f3d742e748cae7b6addc1fe23874e04d18d9483a43b22be87.jpg)

<details>
<summary>natural_image</summary>

Composite photo showing a smiling person in a red shirt and a man in glasses, both looking at the camera (no visible text or symbols)
</details>

![](images/44bd31f2acfddb89bd415128b6986b9f3094969f7514841023db72b58ba068c9.jpg)

<details>
<summary>natural_image</summary>

Two abstract color regions with purple and black patches, no text or symbols present
</details>

![](images/8cecc0944ceb6b9ca2f1790216bed0f040751a86af2bd7c875da32f45632a05d.jpg)

<details>
<summary>natural_image</summary>

Night street scene with a bright blue shipping container and a truck on the road (no visible text or symbols)
</details>

Figure 4: Visualization of semantic segmentation predictions on the Pascal VOC2012 dataset: Spike2Former with Meta-Spikeformer backbone which achieves 75.4 mIoU on the validation set. First and third columns: ground truth. Second and fourth columns: prediction. Last row shows failure cases

# References

Chen, K.; Wang, J.; Pang, J.; Cao, Y.; Xiong, Y.; Li, X.; Sun, S.; Feng, W.; Liu, Z.; Xu, J.; Zhang, Z.; Cheng, D.; Zhu, C.; Cheng, T.; Zhao, Q.; Li, B.; Lu, X.; Zhu, R.; Wu, Y.; Dai, J.; Wang, J.; Shi, J.; Ouyang, W.; Loy, C. C.; and Lin, D. 2019. MMDetection: Open MMLab Detection Toolbox and Benchmark. arXiv preprint arXiv:1906.07155.   
Cheng, B.; Misra, I.; Schwing, A. G.; Kirillov, A.; and Girdhar, R. 2022. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1290–1299.   
Cheng, B.; Schwing, A.; and Kirillov, A. 2021. Per-pixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34: 17864–17875.   
Contributors, M. 2020. MMSegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark. https://github.com/open-mmlab/mmsegmentation.   
Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3213–3223.   
Du, X.; Zoph, B.; Hung, W.-C.; and Lin, T.-Y. 2021. Simple training strategies and model scaling for object detection. arXiv preprint arXiv:2107.00057.   
Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.; and Zisserman, A. 2010. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88: 303–338.   
Wang, Q.; Zhang, T.; Han, M.; Wang, Y.; Zhang, D.; and Xu, B. 2023. Complex dynamic neurons improved spiking transformer network for efficient automatic speech recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 102–109.   
Yao, M.; Hu, J.; Hu, T.; Xu, Y.; Zhou, Z.; Tian, Y.; XU, B.; and Li, G. 2024a. Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips. In The Twelfth International Conference on Learning Representations.   
Yao, M.; Hu, J.; Zhou, Z.; Yuan, L.; Tian, Y.; Xu, B.; and Li, G. 2024b. Spike-driven transformer. Advances in Neural Information Processing Systems, 36.   
Zhou, B.; Zhao, H.; Puig, X.; Fidler, S.; Barriuso, A.; and Torralba, A. 2017. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 633–641.