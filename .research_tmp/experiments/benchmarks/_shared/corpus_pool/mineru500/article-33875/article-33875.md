# Content-aware Balanced Spectrum Encoding in Masked Modeling for Time Series Classification

Yudong Han $^{1,2*}$ , Haocong Wang $^{1*}$ , Yupeng Hu $^{1\dagger}$ , Yongshun Gong $^{1}$ , Xuemeng Song $^{3}$ , Weili Guan $^{4}$

$^{1}$ School of Software, Shandong University, $^{2}$ Beijing Institute of Technology,

$^{3}$ School of Computer Science and Technology, Shandong University, $^{4}$ Harbin Institute of Technology (Shenzhen) {hanyudong.sdu, sduwhc, sxmustc, honeyguan}@gmail.com, {huyupeng, ysgong}@sdu.edu.cn

# Abstract

Due to the superior ability of global dependency, transformer and its variants have become the primary choice in Masked Time-series Modeling (MTM) towards time-series classification task. In this paper, we experimentally analyze that existing transformer-based MTM methods encounter with two under-explored issues when dealing with time series data: (1) they encode features by performing long-dependency ensemble averaging, which easily results in rank collapse and feature homogenization as the layer goes deeper; (2) they exhibit distinct priorities in fitting different frequency components contained in the time-series, inevitably leading to spectrum energy imbalance of encoded feature. To tackle these issues, we propose an auxiliary content-aware balanced decoder (CBD) to optimize the encoding quality in the spectrum space within masked modeling scheme. Specifically, the CBD iterates on a series of fundamental blocks, and thanks to two tailored units, each block could progressively refine the masked representation via adjusting the interaction pattern based on local content variations of time-series and learning to recalibrate the energy distribution across different frequency components. Moreover, a dual-constraint loss is devised to enhance the mutual optimization of vanilla decoder and our CBD. Extensive experimental results on ten time-series classification datasets show that our method nearly surpasses a bunch of baselines. Meanwhile, a series of explanatory results are showcased to sufficiently demystify the behaviors of our method.

# Introduction

Time-series representation learning has emerged as a fundamental and preposed task for time-series analysis. Different from supervised learning (Han et al. 2023b, 2021; Hu et al. 2021a,b, 2024), which heavily relies on human-labeled ground truth, self-supervised learning (Nie et al. 2022; Tonekaboni, Eytan, and Goldenberg 2021; Ozyurt, Feuerriegel, and Zhang 2022) has shown its flexibility using the intrinsic characteristic of data itself towards scalable representation learning. The widely embraced scheme that initially introduced from computer vision domain (He et al. 2022), Masked Time-series Modeling (MTM), randomly masks a portion of input timestamps, and then reconstructs the invisible timestamps based on the visible ones. Recently, transformer and its variants have become the predominant choice in MTM, and achieve new state-of-the-art performance on various classification benchmarks (Dong et al. 2023; Cheng et al. 2023). This greatly attributes to its powerful modeling ability of long-range dependency between different timestamps, which facilitates learning the context-aware feature.

![](images/236853969da44f216ce1981810372acc0916f4b0058197c9d6c4e1322248d72a.jpg)

<details>
<summary>line</summary>

(a) Interaction Matrix Rank
| Layers of Encoder | Ours Rank | MTM Rank |
|---|---|---|
| 0 | 17 | 10 |
| 2 | 16 | 6 |
| 4 | 15 | 11 |
| 6 | 18 | 10 |
| 8 | 15 | 8 |
</details>

![](images/ebb1f867958014e71d292a3a677a9ab0ac95057dff7cf005b1ea943063aad130.jpg)

<details>
<summary>line</summary>

| Frequency Spectrum | Origin | Ours | MTM |
| ------------------ | ------ | ---- | --- |
| 0                  | 2      | 2    | 20  |
| 5                  | 22     | 20   | 18  |
| 10                 | 5      | 5    | 8   |
| 15                 | 3      | 3    | 4   |
| 20                 | 2      | 2    | 2   |
| 25                 | 1      | 1    | 1   |
| 30                 | 1      | 1    | 1   |
| 35                 | 1      | 1    | 1   |
| 40                 | 1      | 1    | 1   |
| 45                 | 1      | 1    | 1   |
| 50                 | 1      | 1    | 1   |
| 55                 | 1      | 1    | 1   |
| 60                 | 1      | 1    | 1   |
</details>

Figure 1: (a) Comparison of rank of the interaction matrix across different layers of the encoder in vanilla MTM and our method; (b) Energy distribution comparison of raw data, reconstructed results of vanilla MTM, and that of our method.

Despite their promising performance towards representation learning, these transformer-based methods overlook two potential problems: (1) feature homogenization. Several studies (Dong, Cordonnier, and Loukas 2021; Han et al. 2023a, 2024) point out that the feature encoded by transformer-based backbone easily incurs rank collapse due to the long-dependency ensemble averaging (Park and Kim 2022), and we experimentally conclude that this phenomenon also exists when encoding the time-series data. As illustrated in Figure 1 (a), we showcase the learned interaction matrix and calculate their ranks from the encoder of vanilla MTM (Vaswani et al. 2017) and our improved method. We observe that vanilla MTM tends to produce low-rank features, which manifests to encode homogenized information and lacks sufficient semantic richness. (2) energy imbalance. Based on frequency principle (Xu 2020; Rahaman et al. 2019) and architecture-induced frequency preference (Park and Kim 2022), we observe that vanilla

transformer-based feature learning is inclined to capture low-frequency energy of time-series. In other words, they will easily memorize the sketchy trend of time-series but need more steps to comprehend variation details conveyed by mid/high-frequency energy. As depicted in Figure 1 (b), we transform the original time-series and reconstructed ones into the Fourier domain respectively, and show their difference of spectrum energy distribution. It can be seen that the reconstruction result from vanilla MTM primarily concentrates on low-frequency energy and lacks adequate attention to other frequency band information, thereby leading to the obvious inconsistency with original time-series.

However, how to tackle above issues within MTM remains an under-explored problem. In this paper, we theoretically analyze that (1) the rank of encoding feature intrinsically hinges on property of its corresponding spectrum space, and (2) the distribution of feature space could also be directly revealed by the energy distribution in the spectrum space. Based on above considerations, we reveal the possibility of resorting to the unified spectrum encoding to address them. Specifically, we propose an auxiliary content-aware balanced spectrum decoding branch to achieve the spectrum reconstruction in MTM, where several cascaded blocks with two tailored units are employed to motivate the encoder to retard homogenized and imbalanced feature learning, respectively. Content-aware Interaction Modulation (CIM) Unit is first employed, which maps the intermediate representations from encoder into spectrum space and modulates them with a content-aware complex network (Trabelsi et al. 2017). According to frequency-domain convolutional theory (Huang et al. 2023), this process essentially equals to dynamic kernel convolutional operation, which not only restricts unnecessary interaction scope compared with long-dependency tactic but also has the flexibility to adjust the receptive field based on variation of content. Following CIM, Spectrum Energy Rebalance (SER) Unit, inspired by Bernstein approximation, is leveraged to adjust the energy distribution of encoded feature across different frequency components, thus compensating for overlooked details in mid/high frequency bands. Moreover, dual-constraint loss is further devised to reduce the gap between temporal space and spectrum space, thereby avoiding the information loss in two decoding branches.

We conduct experiments on ten classification datasets from diverse domains with both univariate and multivariate settings, to demonstrate the superiority of our method in representation learning. More importantly, the related visualization and mechanism exploration results sufficiently indicate the behavior of our model.

In summary, our main contributions are as follows:

\- We propose a two-pronged masked time-series reconstruction framework, which seamlessly integrates the content-aware balanced decoder into vanilla temporal decoder, along with devised dual-constraint loss, to establish the connection between temporal domain and spectrum domain.

\- Towards content-aware balanced decoder, we endow it with two iterative units, to effectively retard feature ho-

mogenization and achieve spectrum energy rebalance.

\- We conduct extensive experiments on numerous real-world datasets to validate the effectiveness of our proposed CBD, the superiority of our overall model compared with state-of-art methods, and convincingly reveal the intriguing behavior of our model.

# Related Work

Time-Series Pretraining Recent self-supervised Time-Series pre-training techniques can be roughly categorized into two groups: Time-Series Contrastive Learning and Masked Times-Series Modeling.

Time-Series Contrastive Learning constructs multiple views to increase variance and align features for robust global representation learning. Existing work (Zhang et al. 2022) focus on designing different views for contrast to capture different preference on learned representation. For example, TS2Vec (Yue et al. 2022) utilizes temporal contrast to improve discrimination of dynamic semantic variations. TS-TCC (Eldele et al. 2021) formulates a cross-view prediction task that incorporates both temporal and contextual contrast. TimesURL (Liu and Chen 2024) enhances learning process through frequency-temporal augmentation and double universum construction. More flexibly, TS-GAC (Wang et al. 2024) emphasizes the role of spatial consistency in a graph contrastive framework. CSL (Liang et al. 2023) leverages shapelet-based learning to better fit series data.

Concurrent with these strands of research, Masked Time-Series Modeling learns representations by reconstructing masked timestamps from partial observations. TST (Zerveas et al. 2021) marks the first attempt to employ a transformer-based encoder for this task. PatchTST(Nie et al. 2023) predicts masked subseries-level patches to capture local semantics. SimMTM (Dong et al. 2023) argues that random masking would disrupt temporal variations, thereby proposing to aggregate the point-wise representations derived from multiple masked variations for robust reconstruction. To mitigate the gap between masked representation in pre-trained stage and unmasked ones in fine-tuning stage, TimeMAE (Cheng et al. 2023) leverages a decoupled scheme to respectively encode the semantic-enhanced representation with high information density. In this paper, we primarily focus on optimizing the masked time-series modeling. Different from above studies, our method reveals two crucial issues in existing transformer-based masked modeling framework: feature homogenization and energy imbalance from the spectrum perspective.

Learning from Spectrum Perspective More and more investigations delve into learning the robust representation by transforming the temporal domain to frequency domain due to its exclusive properties, which have been actively applied to many fields, such as graph classification, domain generalization, and time-series. For example, FACT (Xu et al. 2021) reckons that model that highlights phase information could better deal with cross-domain setting, thus developing a Fourier-based data augmentation strategy. SFA (Zhang et al. 2023b) introduces a spectrum feature alignment method to address the issue of imbalanced feature when performing graph contrast. In the field of

![](images/34bd70c0383033e1c7ffb56a93b0716d576af5be76fd31230dfb74dc65732e6c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Time Series Input"] --> B["Reconstructed Data"]
    B --> C{Reconstruction}
    C --> D["FFT F"]
    C --> E["IFFT F⁻¹"]
    D --> F["Dual Constraint"]
    E --> F
    G["Masked Token with Random Initialization"] --> B
    H["Visible tokens"] --> B
    I["Masked tokens"] --> B
    J["Temporal data"] --> B
    K["Frequency data"] --> B
    L["Reconstructed data"] --> B
    M["CBD (FD)"] --> N["Reconstructed Data"]
    O["IFT"] --> P["Dual Constraint"]
```
</details>

(a) Architecture of Our Two-pronged Reconstruction Framework

![](images/74250c13a841b3bfebce61a2e13fb828ee0aeba83600584a3690cf0427d5dd09.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["FFT"] --> B["Layer Norm"]
    C["FFN"] --> D["Layer Norm"]
    E["CBD Block"] --> F["IFFT"]
    G["SER Unit"] --> H["CIM Unit"]
    I["FFT"] --> J["Layer Norm"]
    K["x U"] --> L["+"]
    M["Ã(λ₁)"] --> N["..."]
    O["Ã(λ₂)"] --> N
    P["Ã(λ₅₋₁)"] --> Q["..."]
    R["Ã(λₛ)"] --> Q
    S["f(k/K)"] --> T["P = Σ_{k=0}^K f(k/K)(K_k)(1 - Ã(λₛ))^{K-k}Ã(λₛ)^k"]
    T --> U["A"]
    U --> V["P = Σ_{k=0}^K f(k/K)(K_k)(1 - Ã(λₛ))^{K-k}Ã(λₛ)^k"]
    V --> W["Bre"]
    W --> X["Wre"]
    X --> Y["ZF→im"]
    Y --> Z["+"]
    AA["ZF→re"] --> AB["-"]
    AC["ZF"] --> AD["●"]
    AE["ZF"] --> AF["○"]
    AG["ZF"] --> AH["×"]
    AI["ZF"] --> AJ["×"]
    AK["ZF"] --> AL["×"]
    AM["ZF"] --> AN["×"]
    AO["ZF"] --> AP["×"]
    AQ["ZF"] --> AR["×"]
    AS["ZF"] --> AT["×"]
    AU["ZF"] --> AV["×"]
    AW["ZF"] --> AX["×"]
    AY["..."] --> AZ["Ã(λs₋₁)"]
    BA["Ã(λₛ)"] --> BB["Ã(λₛ)"]
```
</details>

(b) Content-aware Balanced Decoder and its sub units   
Figure 2: (a) Schematic illustration of our two-pronged framework. TED denotes temporal encoder, and TD and CBD represents temporal decoder and frequency decoder (content-aware balanced decoder). (b) The details of content-aware balanced decoder.

time-series analysis, FEDformer (Zhou et al. 2022) launches a frequency-enhanced transformer to emphasize the global view of learned feature for time series forecasting. However, how to effectively retard homogenized and imbalanced feature learning from the spectrum perspective in masked time-series framework still remains an under-explored challenge.

# Methodology

In this section, we begin by reviewing the masked modeling framework towards time-series data. Following this, we delve into an in-depth analysis to the key components of our framework, Content-aware Balanced Decoder (CBD), and elaborate on the implementation of its two tailored units. Finally, we provide the design details of loss function that optimizes the overall network.

# Review of Masked Time-Series Modeling

Masked Time-Series Modeling (MTM) adopts a transformer-based architecture to reconstruct large portions of masked timestamps via the visible timestamps. Formally, given $\{x_{i}\}_{i=1}^{N}$ as a batch of N time-series samples, where $x_{i} \in R^{L \times C}$ contains L timestamps and C variables, we randomly mask a certain proportion of timestamps along the temporal dimension. Following the orthodox pipeline of masked modeling framework, MTM encoder takes these visible timestamps as input to generate an intermediate representation $Z = \{z_{i}\}_{i=1}^{N}$ , while the following decoder achieves temporal reconstruction of masked timestamps. By aligning the reconstruction results with unmasked timestamps, the encoder could learn the representation that capture context-aware dependencies within time-series data.

# Content-aware Balanced Decoder

As suggested by MAE (He et al. 2022), the decoder design is crucial to the MTM model, as it not only models the relationship between representations of masked tokens and visible tokens, but also determines the semantic preference of learned representation. To address the issues of feature homogenization and spectrum energy imbalance in traditional transformer-based MTM methods, we armed the temporal encoder (TD) with an auxiliary content-aware balanced decoder (CBD), where two tailored units are iteratively deployed. The overall architecture and unfold details are illustrated in Figure 2. In what follows, we elaborate these two units sequentially.

Content-aware Interaction Modulation Unit (CIM) Inspired by the fact that the rank of feature matrix can be determined by its spectrum distribution (i.e., lower bound), $\text{rank}(\mathbf{Z}) \geq \text{rank}(\boldsymbol{\Sigma})$ , where $\Sigma$ are the eigenvalue of Z, we devise a content-aware modulation strategy that operates directly in the spectral space. Specifically, we first transform the intermediate feature Z to discrete Fourier domain,

$$
\mathbf {Z} _ {F} = \mathcal {F} (\mathbf {Z}), \tag {1}
$$

where $\mathcal{F}$ denotes the Fourier transformation, and each of components $\mathbf{Z}_F$ can be written as,

$$
\mathbf {Z} _ {F} (\lambda_ {s}) = \sum_ {t = 1} ^ {T} \mathbf {Z} (t) \cdot e ^ {- j \frac {2 \pi \lambda_ {s} t}{T}}, \tag {2}
$$

and the inverse Fourier transformation $F^{-1}$ corresponds to

$$
\mathbf {Z} (t) = \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbf {Z} _ {F} (\lambda_ {s}) \cdot e ^ {- j \frac {2 \pi \lambda_ {s} t}{T}}, \tag {3}
$$

where T denotes the total length of intermediate feature of time-series, and $\lambda_{s}$ denotes the s-th frequency component in frequency domain. The real and imaginary part of $Z_{F}$ can be denoted as $Z_{F\rightarrow re}$ and $Z_{F\rightarrow im}$ , which exhibits equal contribution for frequency modeling (Trabelsi et al. 2017). In order to achieve more exhaustive spectrum modulation, we encode the real and imaginary parts separately to generate the modulation signals, which is summarized as,

$$
\mathbf {M} (\lambda_ {s}) = \sigma (\mathbf {W} _ {F} \mathbf {Z} _ {F} (\lambda_ {s}) + \mathbf {b} _ {F}), \tag {4}
$$

where $\sigma$ is the activation function, $W_{F} = W_{F \to re} + j \cdot W_{F \to im}$ and $b_{F} = b_{F \to re} + j \cdot b_{F \to im}$ . More specifically, the generated M can be further unfolded into real and imaginary parts as follows,

$$
\left\{\begin{array}{l}\mathbf {M} _ {r e} (\lambda_ {s}) = \sigma \left(\mathbf {W} _ {F \rightarrow r e} \mathbf {Z} _ {F \rightarrow r e} (\lambda_ {s}) \right.\\\quad - \left. \mathbf {W} _ {F \rightarrow i m} \mathbf {Z} _ {F \rightarrow i m} (\lambda_ {s}) + \mathbf {B} _ {F \rightarrow r e}\right),\\\mathbf {M} _ {i m} (\lambda_ {s}) = \sigma \left(\mathbf {W} _ {F \rightarrow i m} \mathbf {Z} _ {F \rightarrow i m} (\lambda_ {s}) \right.\\\quad + \left. \mathbf {W} _ {F \rightarrow r} \mathbf {Z} _ {F \rightarrow r e} (\lambda_ {s}) + \mathbf {B} _ {F \rightarrow i m}\right).\end{array}\right. \tag {5}
$$

Afterwards, the generated complex signal is used to modulate counterparts of original frequency-domain feature, which can be written as,

$$
\tilde {\mathbf {Z}} _ {F} (\lambda_ {s}) = \mathbf {M} (\lambda_ {s}) \odot \mathcal {F} (\mathbf {Z}). \tag {6}
$$

Based on theorem 1, Eq.(6) essentially equals to a dynamic convolutional operation. In a deeper analysis from the perspective of temporal domain, it serves as two purposes: (1) Unlike the long-range dependency mechanism in self-attention block, convolutional operation leads to less freedom of interaction, which naturally restore the rank of features (Han et al. 2023a). (2) its dynamicity indicates that the learned kernel is determined by the content variation of each timestamp, which effectively diversifying the interaction pattern. In other words, for smoothing sub-series, a large interaction scope is possibly unnecessary, while for sub-series that has more obvious fluctuation, enlarging the interaction scope to obtain more context is crucial for understanding its comprehensive semantic. By means of Eq.(6), feature homogenization and rank collapse can be effectively mitigated via direct spectrum modulation. The proof of theorem 1 is illustrated in supplementary materials.

Theorem 1. (Frequency-domain convolution theorem) The multiplication of two signals in the Fourier domain equals to Fourier transformation of a convolution of these two signals in temporal domain, which can be summarized as,

$$
\mathcal {F} \left[ \mathbf {K} (t) \otimes \mathbf {Z} (t) \right] = \mathcal {F} (\mathbf {K} (t)) \odot \mathcal {F} (\mathbf {Z} (t)), \tag {7}
$$

where $\otimes$ and $\odot$ denote the convolutional operation and element-multiplication operation, respectively, $\mathbf{K}(t)$ and $\mathbf{Z}(t)$ represent two signals with respect with time variable t, and $\mathcal{F}(\cdot)$ denotes the Fourier transformation.

Spectrum Energy Rebalance Unit (SER) For time-series data, timestamps with obvious fluctuations generally contain more high-frequency components (Qin et al. 2021) in spectrum space. However, several previous literatures indicate that networks tend to prioritize learning low-frequency components over high-frequency ones(Xu and Zhou 2021; Xu, Zhang, and Xiao 2019), resulting in challenges in capturing high-frequency details. Furthermore, according to theorem 2, overlooking the certain bands of frequency components would greatly impact the temporal information encoding, thereby hindering balanced representation learning. The proof of theorem 2 is shown in supplementary materials.

Theorem 2. (Parseval's theorem) Assume that $\mathbf{Z}$ denotes the original temporal domain feature, and $\mathbf{Z}_F$ is the corresponding spectrum representation, then the energy of feature in temporal domain equals to the energy in frequency domain,

$$
\sum_ {t = - \infty} ^ {\infty} | \mathbf {Z} (t) | ^ {2} = \sum_ {s = - \infty} ^ {\infty} | \mathbf {Z} _ {F} (\lambda_ {s}) | ^ {2}, \tag {8}
$$

where $\mathbf{Z}_{F} = \mathcal{F}(\mathbf{Z})$ , $\mathcal{F}(\cdot)$ denotes the Fourier transformation, $\lambda_{s}$ represents the s-th spectrum component, and t denotes the time variable

To tackle it, we design Spectrum Energy Rebalance unit (SER) to flexibly adjust the energy distribution in spectrum space to approximate that of original time-series data. We assume that the ideal response function from spectrum components to energy can be denoted as $\mathcal{G}(\cdot)$ , and the initial response is $A = \sqrt{\tilde{Z}_{F\to re}^{2} + \tilde{Z}_{F\to im}^{2}} = \|\tilde{Z}_{F}\|_{2}$ . SER serves as the energy rebalance mapping function $\mathcal{P}(\cdot)$ , which makes the initial response A head to real response $\mathcal{G}(\tilde{\mathbf{Z}}_{F})$ , i.e., $\mathcal{G}(\tilde{\mathbf{Z}}_{F}) = \mathcal{P}(\cdot) \circ \|\cdot\|_{2}(\tilde{\mathbf{Z}}_{F})$ , where $\circ$ denotes the compound operation. To better simulate the effect of $\mathcal{P}(\cdot)$ , we adopt Bernstein polynomials. Before unfolding our design details of SER, we first give the definition of Bernstein polynomial approximation in the following.

Definition 1. (Bernstein polynomial approximation) Given an arbitrary continuous mapping function $f(w)$ from $w \in [0,1]$ , the $K$ -order Bernstein polynomial approximation of $f(w)$ is defined as,

$$
p _ {K} (w) = \sum_ {k = 0} ^ {K} \Theta_ {k} \cdot \mathcal {B} _ {k} ^ {K} (w) = \sum_ {k = 0} ^ {K} f (\frac {k}{K}) \cdot \binom {K} {k} (1 - w) ^ {K - k} w ^ {k}, \tag {9}
$$

and we have $p_{K}(w) \to f(w)$ as $K \to \infty$ , where $\Theta_{k}$ works as the coefficient of $\mathcal{B}_{k}^{K}(w)$ , and $\mathcal{B}_{k}^{K}(w)$ serves as the base of polynomial.

Given that the domain of definition of $p_{K}(w)$ is restricted to [0, 1], we therefore adopt the softmax function to normalize the input energy distribution $\mathbf{A}(\lambda_{s})$ , and we substitute the normalized energy distribution $\tilde{\mathbf{A}}(\lambda_{s})$ into the approximated polynomial $p_{K}(w)$ ,

$$
p _ {K} = \sum_ {k = 0} ^ {K} f (\frac {k}{K}) \cdot \binom {K} {k} (1 - \tilde {\mathbf {A}} (\lambda_ {s})) ^ {K - k} \tilde {\mathbf {A}} (\lambda_ {s}) ^ {k}. \tag {10}
$$

Beyond that, in order to adjust $\mathcal{P}$ (i.e., $p_{K}$ ) in a data-driven manner, we leverage an energy-aware gating network to control the filter coefficients,

$$
f (\frac {k}{K}) = (\mathbf {W} _ {c} \mathbf {A} ^ {\dagger} + \mathbf {b} _ {c}) _ {k}, \tag {11}
$$

where $\mathbf{A}^{\dagger} = \left[\tilde{\mathbf{A}} (\lambda_1),\tilde{\mathbf{A}} (\lambda_2),\dots,\tilde{\mathbf{A}} (\lambda_S)\right]\in \mathbb{R}^S$ . $\mathbf{W}_c\in$ $\mathbb{R}^{K\times S}$ and $\mathbf{b}_c\in \mathbb{R}^K$ are the learnable weights and bias of gating network, respectively.

Such mechanism potentially undertakes the role that it allows the model to have elasticity to compensate for the missing spectrum components in the training process. Interestingly, from a graph perspective, different ways of frequency response can be interpreted as different specific filters (Xu, Zhang, and Xiao 2019; Bianchi et al. 2022). When K is sufficiently large, function P could simulate arbitrary spectrum rebalance functions (i.e., filters). Finally, the learned

spectrum rebalance function is utilized to modulate the frequency component to obtain the balanced spectrum feature,

$$
\tilde {\mathbf {Z}} _ {\mathbf {F} \rightarrow \mathbf {b a l}} (\lambda_ {s}) = p _ {K} (\lambda_ {s}) \odot \tilde {\mathbf {Z}} _ {F} (\lambda_ {s}). \tag {12}
$$

Iterative Architecture As illustrated in Figure 2(b), it is built upon the vanilla transformer architecture, which modifies the self-attention block by integrating CIM and SER unit. For each layer $u \in [1, U]$ , where $U$ is the total number of layer, a CBD block processes and updates the masked representation as follows,

$$
\tilde {\mathbf {Z}} _ {F} ^ {(u)} = \text { CBDBlock } (\tilde {\mathbf {Z}} _ {F} ^ {(u - 1)}; \boldsymbol {\Omega} ^ {(u)}), \tag {13}
$$

where $\mathbf{\Omega}^{(u)}=\{\mathbf{W}_{F}^{(u)},\mathbf{b}_{F}^{(u)},\mathbf{W}_{c}^{(u)},\mathbf{b}_{c}^{(u)}\}$ denotes the set of learned parameters in each block. At each step, CBD block retards rank collapse of interaction matrix and mitigates feature smoothing via CIM unit, and further leverages SER unit to balance spectrum energy. Moreover, each block has independent parameters, allowing blocks at different depths to collaboratively refine the masked representation.

# Optimization Strategy

Dual-Constraint Pre-trained Loss During the pre-training stage, as shown in Figure 2(b), temporal encoder (TED) encodes the masked time series and decodes it into two outputs: temporal reconstruction $\tilde{\mathbf{Z}}_{T}^{(U)}$ and frequency reconstruction $\tilde{\mathbf{Z}}_{F}^{(U)}$ . We denote the ground truth of temporal domain and its frequency domain as $\mathbf{Z}_{T}^{(gt)}$ and $\mathbf{Z}_{F}^{(gt)}$ , the pre-training loss is formulated as:

$$
\begin{array}{l} \mathcal {L} = \mathcal {L} _ {T} ^ {(r e)} (\tilde {\mathbf {Z}} _ {T}, \mathbf {Z} _ {T} ^ {(g t)}) + \mathcal {L} _ {F} ^ {(d u a l)} (\mathcal {F} (\tilde {\mathbf {Z}} _ {T}), \mathbf {Z} _ {F} ^ {(g t)}) \\ + \gamma \left[ \mathcal {L} _ {F} ^ {(r e)} (\tilde {\mathbf {Z}} _ {F} ^ {(U)}, \mathbf {Z} _ {F} ^ {(g t)}) + \mathcal {L} _ {T} ^ {(d u a l)} (\mathcal {F} ^ {- 1} (\tilde {\mathbf {Z}} _ {F} ^ {(U)}), \mathbf {Z} _ {T} ^ {(g t)}) \right], \tag {14} \\ \end{array}
$$

where $\mathcal{L}_{\sim}^{(re)}$ represents the reconstruction loss in traditional masked modeling, while $\mathcal{L}_{\sim}^{(dual)}$ serves as the dual-constraint loss aforementioned. For temporal domain loss $L_{T}^{\sim}$ , we compute Mean Square Error (MSE) between the reconstructed and raw time-series. As to frequency domain loss $L_{F}^{\sim}$ , we compute squared Euclidean distance between their real and imaginary parts. $\gamma$ is the loss weight to control the contribution of each branch. Specially, the dual-constraint loss ensures that the two decoding branches can enhance the information consistency between temporal domain and frequency domain in the training process.

Fine-tuned Loss In the fine-tuning stage, the output Z from the common encoder is fed into the final classification head $f_{c}$ . Then the fine-tuned loss is termed as,

$$
\mathcal {L} = \mathcal {L} _ {C E} (f _ {c} (\mathbf {Z}), y), \tag {15}
$$

where the Cross-Entropy Loss is utilized to minimize the distance between label y and the prediction $f_{c}(\mathbf{Z})$ .

# Experiments

In this section, we first introduce the experiment setup. Then, we present and analyze the quantitative results of our model and a bunch of baselines on time series classification. Finally, we demonstrate the effectiveness of our model through analytical experiments.

# Experiment Setup

Datasets We conduct experiments on ten publicly available datasets to validate the performance of our model, including Human Activity Recognition (HAR)(Anguita et al. 2012) and nine large datasets from the UEA(Bagnall et al. 2018) and UCR archive(Dau et al. 2019), which is referred as PS, SRSCP1, MI, FM, AWR, SAD, ECG5000, FB, Uware. These datasets not only encompass both univariate and multivariate datasets, but also cover diverse domains. More details are provided in supplementary materials.

Baselines We compare our model with recent advanced self-supervised learning methods. To highlight our performance, we select five contrastive learning methods and four masked modeling methods. More details about these baselines are provided in supplementary part:

- Contrastive Learning Methods: TS-TCC(Eldele et al. 2021), TS2Vec(Yue et al. 2022), TS-GAC(Wang et al. 2024), TimesURL(Liu and Chen 2024), CSL(Liang et al. 2023).   
- Reconstructed Methods: PatchTST(Nie et al. 2023), CRT(Zhang et al. 2023a), SimMTM(Dong et al. 2023), TimeMAE(Cheng et al. 2023).

Evaluation In the classification task, we evaluate the performance in two mainstream manner: linear evaluation and fine-tuning evaluation. The former manner freezes the pretrained parameters of encoder and trains the classifier layer with labeled data. The latter manner updates both the encoder and classifier layer based on the pre-trained parameters. We utilize accuracy as the metrics for classification tasks and mark the best and second best values. All experiments are conducted with five different seeds and the average results are taken for comparisons.

Implement Details For the encoder, we use the same backbone as TimeMAE with the default 8 transformer layers, while the decoder is configured with 2 layers for both branches. We also adopt the same masking strategy as TimeMAE. We set the batch size as 128 and choose AdamW optimizer with a learning rate of 1e-4. All methods are conducted with NVIDIA A10 and implemented by PyTorch. More implementation and baseline experiments details are provided in supplementary materials.

# Quantitative Experimental Results

Table 1 presents the classification results. Firstly, we observe that our method achieves the highest accuracy across most of datasets, regardless of linear evaluation or fine-tuning task. Particularly, HAR and MI show significant improvements, surpassing the second-best by $2.38\%$ and $1.90\%$ in linear probing accuracy. The superior performance of linear evaluation suggests that our method effectively narrows the gap between pretrained representation and task-specific representation, demonstrating strong feature generalization capabilities. Meanwhile, the performance gain on fine-tuning highlight model's adaptability to specific tasks. Notably, we exclude TimesURL from fine-tune evaluation due to its implementation constraints to ensure the rigor of our experiments. Secondly, our method supports both univariate and

<table><tr><td rowspan="2">Methods</td><td colspan="10">Linear Evaluation</td></tr><tr><td>HAR</td><td>PS</td><td>SRSCP1</td><td>MI</td><td>FM</td><td>AWR</td><td>SAD</td><td>ECG5000</td><td>FB</td><td>UWare</td></tr><tr><td>TS-TCC(IJCAI&#x27;21)</td><td> $89.22 \pm 0.70$ </td><td> $14.27 \pm 0.39$ </td><td> $83.64 \pm 0.99$ </td><td> $55.47 \pm 0.68$ </td><td> $48.00 \pm 1.63$ </td><td> $93.23 \pm 0.68$ </td><td> $95.20 \pm 0.15$ </td><td> $92.67 \pm 0.78$ </td><td> $50.29 \pm 0.31$ </td><td> $83.17 \pm 0.29$ </td></tr><tr><td>TS2Vec(AAAI&#x27;22)</td><td> $90.36 \pm 0.32$ </td><td> $10.82 \pm 0.38$ </td><td> $83.61 \pm 0.99$ </td><td> $51.00 \pm 0.75$ </td><td> $47.10 \pm 4.22$ </td><td> $98.30 \pm 0.09$ </td><td> $97.31 \pm 0.19$ </td><td> $93.50 \pm 0.65$ </td><td> $78.90 \pm 0.54$ </td><td> $93.40 \pm 0.53$ </td></tr><tr><td>TS-GAC(AAAI&#x27;24)</td><td> $91.40 \pm 0.16$ </td><td> $13.53 \pm 0.48$ </td><td> $86.04 \pm 0.31$ </td><td> $56.00 \pm 0.46$ </td><td> $53.39 \pm 0.54$ </td><td> $98.50 \pm 0.06$ </td><td> $97.99 \pm 0.05$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>TimesURL(AAAI&#x27;24)</td><td> $87.18 \pm 0.46$ </td><td> $16.88 \pm 0.42$ </td><td> $85.32 \pm 0.95$ </td><td> $61.20 \pm 0.35$ </td><td> $56.00 \pm 1.84$ </td><td> $98.00 \pm 0.13$ </td><td> $95.86 \pm 0.10$ </td><td> $93.73 \pm 0.58$ </td><td> $73.95 \pm 0.34$ </td><td> $94.22 \pm 0.38$ </td></tr><tr><td>CSL(VLDB&#x27;24)</td><td> $83.40 \pm 0.33$ </td><td> $17.63 \pm 0.36$ </td><td> $84.60 \pm 0.74$ </td><td> $61.00 \pm 0.45$ </td><td> $56.25 \pm 2.63$ </td><td> $97.67 \pm 1.35$ </td><td> $94.95 \pm 0.26$ </td><td> $93.16 \pm 0.84$ </td><td> $79.50 \pm 0.42$ </td><td> $95.31 \pm 0.20$ </td></tr><tr><td>PatchTST(ICML&#x27;22)</td><td> $77.89 \pm 1.98$ </td><td> $14.42 \pm 0.26$ </td><td> $68.36 \pm 2.15$ </td><td> $61.00 \pm 1.36$ </td><td> $55.00 \pm 2.10$ </td><td> $90.63 \pm 0.86$ </td><td> $91.54 \pm 0.41$ </td><td> $90.25 \pm 0.45$ </td><td> $59.11 \pm 0.34$ </td><td> $90.01 \pm 0.30$ </td></tr><tr><td>CRT(TNNLS&#x27;23)</td><td> $89.21 \pm 0.56$ </td><td> $7.60 \pm 1.17$ </td><td> $58.03 \pm 1.11$ </td><td> $51.67 \pm 2.86$ </td><td> $54.00 \pm 1.71$ </td><td> $82.50 \pm 1.17$ </td><td> $85.81 \pm 0.29$ </td><td> $91.00 \pm 0.34$ </td><td> $75.73 \pm 0.08$ </td><td> $80.87 \pm 0.24$ </td></tr><tr><td>SimMTM(NeurIPS&#x27;23)</td><td> $87.90 \pm 0.35$ </td><td> $15.45 \pm 0.12$ </td><td> $90.78 \pm 0.30$ </td><td> $62.00 \pm 0.36$ </td><td> $55.33 \pm 3.21$ </td><td> $95.42 \pm 0.16$ </td><td> $95.05 \pm 0.13$ </td><td> $92.57 \pm 0.18$ </td><td> $51.98 \pm 0.22$ </td><td> $91.58 \pm 0.30$ </td></tr><tr><td>TimeMAE(Arxiv&#x27;23)</td><td> $91.31 \pm 0.84$ </td><td> $14.13 \pm 0.34$ </td><td> $85.53 \pm 1.84$ </td><td> $62.60 \pm 1.67$ </td><td> $53.80 \pm 1.64$ </td><td> $95.07 \pm 0.72$ </td><td> $95.76 \pm 0.51$ </td><td> $93.92 \pm 0.17$ </td><td> $55.31 \pm 0.33$ </td><td> $86.78 \pm 1.64$ </td></tr><tr><td>Ours</td><td> $93.78 \pm 0.70$ </td><td> $18.22 \pm 0.12$ </td><td> $86.72 \pm 0.98$ </td><td> $64.50 \pm 0.20$ </td><td> $56.40 \pm 2.94$ </td><td> $98.52 \pm 0.29$ </td><td> $97.98 \pm 0.21$ </td><td> $94.51 \pm 0.08$ </td><td> $79.57 \pm 0.38$ </td><td> $95.99 \pm 0.12$ </td></tr><tr><td rowspan="2"></td><td colspan="10">Fine-tune Evaluation</td></tr><tr><td>HAR</td><td>PS</td><td>SRSCP1</td><td>MI</td><td>FM</td><td>AWR</td><td>SAD</td><td>ECG5000</td><td>FB</td><td>UWare</td></tr><tr><td>TS-TCC(IJCAI&#x27;21)</td><td> $91.66 \pm 0.42$ </td><td> $20.72 \pm 0.68$ </td><td> $83.76 \pm 1.07$ </td><td> $57.81 \pm 0.30$ </td><td> $46.00 \pm 2.12$ </td><td> $96.61 \pm 0.33$ </td><td> $98.71 \pm 0.09$ </td><td> $93.42 \pm 0.61$ </td><td> $63.68 \pm 0.52$ </td><td> $89.35 \pm 0.05$ </td></tr><tr><td>TS2Vec(AAAI&#x27;22)</td><td> $95.10 \pm 0.36$ </td><td> $16.07 \pm 0.66$ </td><td> $84.43 \pm 0.61$ </td><td> $53.00 \pm 0.49$ </td><td> $49.21 \pm 2.33$ </td><td> $98.01 \pm 0.06$ </td><td> $98.36 \pm 0.34$ </td><td> $93.63 \pm 0.52$ </td><td> $79.82 \pm 0.14$ </td><td> $93.45 \pm 0.06$ </td></tr><tr><td>TS-GAC(AAAI&#x27;24)</td><td> $94.83 \pm 0.10$ </td><td> $16.76 \pm 0.35$ </td><td> $86.44 \pm 2.50$ </td><td> $56.96 \pm 2.54$ </td><td> $49.11 \pm 4.15$ </td><td> $98.06 \pm 0.07$ </td><td> $99.01 \pm 0.20$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>CSL(VLDB&#x27;24)</td><td> $89.31 \pm 0.40$ </td><td> $18.13 \pm 0.48$ </td><td> $83.95 \pm 0.68$ </td><td> $64.50 \pm 0.64$ </td><td> $56.00 \pm 3.27$ </td><td> $98.66 \pm 0.16$ </td><td> $99.44 \pm 0.23$ </td><td> $94.42 \pm 0.37$ </td><td> $79.60 \pm 0.30$ </td><td> $95.53 \pm 0.27$ </td></tr><tr><td>PatchTST(ICML&#x27;22)</td><td> $88.45 \pm 0.70$ </td><td> $18.42 \pm 0.64$ </td><td> $81.25 \pm 1.78$ </td><td> $62.40 \pm 2.35$ </td><td> $57.81 \pm 2.25$ </td><td> $98.05 \pm 0.06$ </td><td> $97.75 \pm 0.16$ </td><td> $94.40 \pm 0.21$ </td><td> $75.65 \pm 0.23$ </td><td> $91.36 \pm 0.17$ </td></tr><tr><td>CRT(TNNLS&#x27;23)</td><td> $90.09 \pm 0.75$ </td><td> $8.38 \pm 0.19$ </td><td> $67.65 \pm 1.36$ </td><td> $50.67 \pm 2.65$ </td><td> $53.00 \pm 2.56$ </td><td> $87.62 \pm 0.84$ </td><td> $98.23 \pm 0.13$ </td><td> $92.53 \pm 0.32$ </td><td> $79.24 \pm 0.13$ </td><td> $85.64 \pm 0.09$ </td></tr><tr><td>SimMTM(NeurIPS&#x27;23)</td><td> $93.50 \pm 0.32$ </td><td> $21.95 \pm 0.20$ </td><td> $92.72 \pm 0.93$ </td><td> $62.00 \pm 0.20$ </td><td> $60.00 \pm 2.15$ </td><td> $98.57 \pm 0.10$ </td><td> $99.30 \pm 0.10$ </td><td> $92.76 \pm 0.13$ </td><td> $82.53 \pm 0.18$ </td><td> $94.64 \pm 0.13$ </td></tr><tr><td>TimeMAE(Arxiv&#x27;23)</td><td> $95.11 \pm 0.18$ </td><td> $19.49 \pm 0.59$ </td><td> $87.71 \pm 1.27$ </td><td> $61.80 \pm 2.28$ </td><td> $61.00 \pm 1.87$ </td><td> $97.73 \pm 0.43$ </td><td> $99.20 \pm 0.03$ </td><td> $94.25 \pm 0.14$ </td><td> $67.21 \pm 0.44$ </td><td> $92.81 \pm 0.47$ </td></tr><tr><td>Ours</td><td> $95.92 \pm 0.78$ </td><td> $23.27 \pm 0.37$ </td><td> $86.25 \pm 1.14$ </td><td> $65.20 \pm 0.31$ </td><td> $60.60 \pm 3.38$ </td><td> $98.67 \pm 0.19$ </td><td> $99.44 \pm 0.12$ </td><td> $94.84 \pm 0.10$ </td><td> $80.83 \pm 0.45$ </td><td> $95.85 \pm 0.23$ </td></tr></table>

Table 1: Comparisons with State-of-the-Art methods with different evaluations (%)

![](images/9b627b2dc0c32d0ddd97d0934bbd249fbc6d6969b52b97495e1eb7d9926909d0.jpg)  
Figure 3: Learned Interaction Matrix on the HAR dataset.

multivariate setting. It is worth noting that TS-GAC considers variables as nodes in graph network, which struggles to deal with single-variable cases. This demonstrates both flexibility and generalization of our method. Finally, compared with contrastive-based baselines, our improved masked modeling scheme could strike a balance between dynamic local variation and holistic semantic representation. Compared to masked modeling baselines, our content-aware balanced decoder brings diverse and balanced decoding potential from the spectrum perspective.

# Analytical experiment

Analysis of Content-aware Interaction Modulation We pretrain our network and vanilla transformer-based reconstruction model on the HAR dataset for 20, 50 and 80 epochs. Then we visualize the interaction matrix obtained from the last encoder layer in Figure 3. Both methods exhibit similar encoding patterns in the early stages, empha-

![](images/2763e2a11612dc5ea230646e52fba1a36dc5f7c835c36fe780c43f26706bf2f7.jpg)  
Figure 4: Energy rebalance of SER. The top row depicts the modulation during the energy rebalancing, while the bottom represents the corresponding learned Bernstein polynomials.

sizing long-dependency interactions. However, our method employs a more prudent and distinct tactics for interaction scope control in the initial phase. This ability stems from our Content-aware Interaction Modulation Unit (CIM), enabling more accurate and flexible interaction scope compared to the pure self-attention decoding. As training progresses, the encoding receptive fields of both methods gradually shrink, but our approach maintains a more semantically compact receptive field. More visualizations on other datasets are provided in supplementary materials.

Analysis of Spectrum Energy Rebalance We investigate an in-depth investigation into the modulation mechanism of the Spectrum Energy Rebalance Unit (SER). Figure 4 visualizes the original and modulated spectrum energy distributions for HAR, PhonemeSpectra, and FingerMovements. SER adaptively adjusts the energy distribution by learning

data-specific Bernstein coefficients. For HAR (top left), the red line shows the original network concentrating energy in the low-frequency range, indicating a bias towards low-frequency information. As frequency increases, energy decreases, leading to insufficient fitting of mid-to-high frequencies. With SER, the green line shows a marked increase in mid-to-high-frequency energy. The blue line also illustrates SER's modulation role, which serves as the normalized scaling factor from original energy distribution to balanced one. This flexibility allows SER to compensate for overlooked information and prevent representation degradation. Notably, the model does not excessively prioritize mid-to-high frequencies, as seen in PhonemeSpectra. To further clarify SER's effect, we visualize the learned Bernstein polynomials at the bottom row, summarizing the mapping from normalized energy to the final energy response and highlighting the rebalancing preferences for each dataset.

<table><tr><td>Method</td><td>LP.Acc.(%)</td><td>FT.Acc.(%)</td></tr><tr><td> $\mathcal{L}_{T}^{(re)}$ </td><td>89.47</td><td>93.34</td></tr><tr><td> $\mathcal{L}_{F}^{(re)}$ </td><td>91.61</td><td>94.02</td></tr><tr><td> $\mathcal{L}_{T}^{(re)} + \mathcal{L}_{F}^{(dual)}$ </td><td>91.68</td><td>94.16</td></tr><tr><td> $\mathcal{L}_{F}^{(re)} + \mathcal{L}_{T}^{(dual)}$ </td><td>91.78</td><td>94.12</td></tr><tr><td> $\mathcal{L}_{T}^{(re)} + \mathcal{L}_{F}^{(re)}$ </td><td>93.37</td><td>95.61</td></tr><tr><td>w/o SER</td><td>91.92</td><td>94.06</td></tr><tr><td>w/o CIM</td><td>93.45</td><td>95.78</td></tr><tr><td>Ours</td><td>93.78</td><td>95.92</td></tr></table>

Table 2: Ablation study of designed losses and components in CBD unit.

Ablation Study To take a closer look on each part in our framework, we devise several variants and measure their performance using both linear probing (LP) and fine-tuning (FT) on HAR. Results are reported in Table 2 and Table 3. (1) Effect of Designed Losses. Key observations from designed losses include: (1) Both the vanilla temporal decoder and our spectrum-aware decoder benefit from the dual-constraint loss. Notably, the frequency-constraint loss $\mathcal{L}_F^{(dual)}$ , without the frequency decoding branch, improves linear evaluation accuracy by $2.21\%$ . (2) Our frequency decoder outperforms the traditional temporal decoder in feature encoding, achieving improvements of $2.14\%$ in LP and $0.68\%$ in FT by addressing feature homogenization and spectrum imbalance. (3) The two-pronged decoders could collaboratively improve encoding capability more than either alone, yielding improvements of $3\%$ and $1.76\%$ in LP accuracy compared to the standalone temporal and frequency decoders, respectively. Combining them with dual-

<table><tr><td>K Value</td><td>1</td><td>2</td><td>4</td><td>8</td><td>12</td><td>16</td></tr><tr><td>LP Acc. (%)</td><td>92.39</td><td>92.76</td><td>93.61</td><td>93.04</td><td>93.78</td><td>93.48</td></tr><tr><td>LP F1 (%)</td><td>91.86</td><td>92.36</td><td>93.13</td><td>92.61</td><td>93.32</td><td>93.05</td></tr></table>

Table 3: Ablation study of K value in SER.

<table><tr><td>Model</td><td>Avg. LP Acc.</td><td>Avg. FT Acc.</td></tr><tr><td>PatchTST</td><td>69.82</td><td>76.55</td></tr><tr><td>+ CBD</td><td>71.32 (+1.50)</td><td>77.56 (+1.01)</td></tr><tr><td>CRT</td><td>67.64</td><td>71.31</td></tr><tr><td>+ CBD</td><td>72.95 (+5.31)</td><td>75.58 (+4.27)</td></tr><tr><td>SimMTM</td><td>73.81</td><td>79.80</td></tr><tr><td>+ CBD</td><td>74.74 (+0.93)</td><td>80.39 (+0.59)</td></tr></table>

Table 4: Performance by integrating CBD (frequency decoder) to three advanced mask modeling models.

constraint loss $\mathcal{L}_F^{(dual)}$ and $\mathcal{L}_T^{(dual)}$ results in the best representation learning performance. (2) Effectiveness of Each Component in CBD. Table 2 also examines the roles of the CIM and SER units in the frequency decoder. The SER can rebalance the spectrum energy without the CIM. This mechanism can encourage the model to emphasize high-frequency overlooked details. Meanwhile, the CIM dynamically controls the interaction scope based on the local signal variation via flexible large kernel convolutions. With the CIM, the model reduces unnecessary information aggregation and facilitate feature encoding. (3) Performance Variation with Different Order $K$ . To explore the theoretical boundary of Bernstein approximation, we examined performance with varying order $K$ of the Bernstein polynomial in Table 3. Theoretically, as K approaches infinity, the polynomial can fit any function. However, we found that performance improves with increasing $K$ up to 12, after which it declines. This suggests that $K = 12$ is sufficient for SER to learn the scaling factor towards spectrum rebalancing, while higher value will introduce unnecessary noise. Thus, we select $K = 12$ as the optimal value.

CBD Generality From Table 4, we can find that the CBD, as an additional decoder, improves the classification performance of diverse advanced masked modeling models. Specifically, for CRT, our module increases the LP accuracy by an average of 5.31% and FT accuracy by 4.27% across ten datasets. It is worth mentioning that since TimeMAE does not use the origin data for reconstruction, the CBD may receive inaccurate supervisory signals, leading to performance decline. Detailed experimental results are available in supplementary materials.

# Conclusion

In this paper, we propose a two-pronged reconstruction framework to improve the quality of time-series representation. Specifically, we harness the content-aware balanced decoder with two built-in units to form our powerful frequency decoder, which works collaboratively with the vanilla temporal decoder to effectively address the issues of feature homogenization and spectral energy imbalance. The extensive experiments convincingly demonstrate the superiority of our method based on the observation on quantitative results, behavior analysis, and representation visualization.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China, No.:62276155, No.:62376137, No.:U24A20328, No.:62476071, No.:62206156, and No.:62206157; in part by the National Natural Science Foundation of Shandong Province, No.:ZR2021MF040, No.:ZR2024QF104 and No.:ZR2022QF047.

# References

Anguita, D.; Ghio, A.; Oneto, L.; Parra, X.; and Reyes-Ortiz, J. L. 2012. Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine. In Ambient Assisted Living and Home Care: 4th International Workshop, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings 4, 216–223. Springer.   
Bagnall, A.; Dau, H. A.; Lines, J.; Flynn, M.; Large, J.; Bostrom, A.; Southam, P.; and Keogh, E. 2018. The UEA multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075.   
Bianchi, F. M.; Grattarola, D.; Livi, L.; and Alippi, C. 2022. Graph Neural Networks With Convolutional ARMA Filters. IEEE Trans. Pattern Anal. Mach. Intell., 44(7): 3496–3507.   
Cheng, M.; Liu, Q.; Liu, Z.; Zhang, H.; Zhang, R.; and Chen, E. 2023. TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders. arXiv preprint arXiv:2303.00320.   
Cortes, C.; and Vapnik, V. 1995. Support-vector networks. Machine learning, 20: 273–297.   
Dau, H. A.; Bagnall, A.; Kamgar, K.; Yeh, C.-C. M.; Zhu, Y.; Gharghabi, S.; Ratanamahatana, C. A.; and Keogh, E. 2019. The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6): 1293–1305.   
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.   
Dong, J.; Wu, H.; Zhang, H.; Zhang, L.; Wang, J.; and Long, M. 2023. SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling. arXiv preprint arXiv:2302.00861.   
Dong, Y.; Cordonnier, J.; and Loukas, A. 2021. Attention is not all you need: pure attention loses rank doubly exponentially with depth. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, 2793–2803. PMLR.   
Eldele, E.; Ragab, M.; Chen, Z.; Wu, M.; Kwoh, C. K.; Li, X.; and Guan, C. 2021. Time-series representation learning via temporal and contextual contrasting. arXiv preprint arXiv:2106.14112.   
Han, D.; Pan, X.; Han, Y.; Song, S.; and Huang, G. 2023a. FLatten Transformer: Vision Transformer using Focused Linear Attention. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, 5938–5948. IEEE.   
Han, Y.; Guo, Y.; Yin, J.; Liu, M.; Hu, Y.; and Nie, L. 2021. Focal and Composed Vision-semantic Modeling for Visual

Question Answering. In Shen, H. T.; Zhuang, Y.; Smith, J. R.; Yang, Y.; César, P.; Metze, F.; and Prabhakaran, B., eds., MM '21: ACM Multimedia Conference, Virtual Event, China, October 20 - 24, 2021, 4528–4536. ACM.   
Han, Y.; Hu, Y.; Song, X.; Tang, H.; Xu, M.; and Nie, L. 2024. Exploiting the Social-Like Prior in Transformer for Visual Reasoning. In AAAI, 2058–2066. AAAI Press.   
Han, Y.; Yin, J.; Wu, J.; Wei, Y.; and Nie, L. 2023b. Semantic-Aware Modular Capsule Routing for Visual Question Answering. TIP, 32: 5537–5549.   
He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. 2022. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 16000–16009.   
Hu, Y.; Liu, M.; Su, X.; Gao, Z.; and Nie, L. 2021a. Video Moment Localization via Deep Cross-Modal Hashing. IEEE Trans. Image Process., 30: 4667–4677.   
Hu, Y.; Nie, L.; Liu, M.; Wang, K.; Wang, Y.; and Hua, X. 2021b. Coarse-to-Fine Semantic Alignment for Cross-Modal Moment Localization. IEEE Trans. Image Process., 30: 5933–5943.   
Hu, Y.; Wang, K.; Liu, M.; Tang, H.; and Nie, L. 2024. Semantic Collaborative Learning for Cross-Modal Moment Localization. ACM Trans. Inf. Syst., 42(2): 50:1–50:26.   
Huang, Z.; Zhang, Z.; Lan, C.; Zha, Z.; Lu, Y.; and Guo, B. 2023. Adaptive Frequency Filters As Efficient Global Token Mixers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, 6026–6036. IEEE.   
Liang, Z.; Zhang, J.; Liang, C.; Wang, H.; Liang, Z.; and Pan, L. 2023. Contrastive Shapelet Learning for Unsupervised Multivariate Time Series Representation Learning. CoRR, abs/2305.18888.   
Liu, J.; and Chen, S. 2024. TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning. In Wooldridge, M. J.; Dy, J. G.; and Natarajan, S., eds., Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, 13918–13926. AAAI Press.   
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.   
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.   
Ozyurt, Y.; Feuerriegel, S.; and Zhang, C. 2022. Contrastive learning for unsupervised domain adaptation of time series. arXiv preprint arXiv:2206.06243.   
Park, N.; and Kim, S. 2022. How Do Vision Transformers Work? In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Köpf, A.; Yang, E. Z.; DeVito, Z.; Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Wallach, H. M.; Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E. B.; and Garnett, R., eds., Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, 8024–8035.   
Qin, Z.; Zhang, P.; Wu, F.; and Li, X. 2021. FcaNet: Frequency Channel Attention Networks. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, 763–772. IEEE.   
Rahaman, N.; Baratin, A.; Arpit, D.; Draxler, F.; Lin, M.; Hamprecht, F. A.; Bengio, Y.; and Courville, A. C. 2019. On the Spectral Bias of Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, 5301–5310. PMLR.   
Tonekaboni, S.; Eytan, D.; and Goldenberg, A. 2021. Unsupervised representation learning for time series with temporal neighborhood coding. arXiv preprint arXiv:2106.00750.   
Trabelsi, C.; Bilaniuk, O.; Serdyuk, D.; Subramanian, S.; Santos, J. F.; Mehri, S.; Rostamzadeh, N.; Bengio, Y.; and Pal, C. J. 2017. Deep Complex Networks. CoRR, abs/1705.09792.   
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. Journal of machine learning research, 9(11).   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.   
Wang, Y.; Xu, Y.; Yang, J.; Wu, M.; Li, X.; Xie, L.; and Chen, Z. 2024. Graph-Aware Contrasting for Multivariate Time-Series Classification. arXiv:2309.05202.   
Xu, Q.; Zhang, R.; Zhang, Y.; Wang, Y.; and Tian, Q. 2021. A fourier-based framework for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14383–14392.   
Xu, Z. J.; Zhang, Y.; and Xiao, Y. 2019. Training Behavior of Deep Neural Network in Frequency Domain. In Gedeon, T.; Wong, K. W.; and Lee, M., eds., Neural Information Processing - 26th International Conference, ICONIP 2019, Sydney, NSW, Australia, December 12-15, 2019, Proceedings, Part I, volume 11953 of Lecture Notes in Computer Science, 264–274. Springer.   
Xu, Z. J.; and Zhou, H. 2021. Deep Frequency Principle Towards Understanding Why Deeper Learning Is Faster. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence,

EAAI 2021, Virtual Event, February 2-9, 2021, 10541–10550. AAAI Press.

Xu, Z.-Q. J. 2020. Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks. Communications in Computational Physics, 28(5): 1746–1767.

Yue, Z.; Wang, Y.; Duan, J.; Yang, T.; Huang, C.; Tong, Y.; and Xu, B. 2022. Ts2vec: Towards universal representation of time series. In Proceedings of the AAAI Conference on Artificial Intelligence, 8980–8987.

Zerveas, G.; Jayaraman, S.; Patel, D.; Bhamidipaty, A.; and Eickhoff, C. 2021. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, 2114–2124.

Zhang, W.; Yang, L.; Geng, S.; and Hong, S. 2023a. Self-Supervised Time Series Representation Learning via Cross Reconstruction Transformer. IEEE Transactions on Neural Networks and Learning Systems.

Zhang, X.; Zhao, Z.; Tsiligkaridis, T.; and Zitnik, M. 2022. Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in Neural Information Processing Systems, 35: 3988–4003.

Zhang, Y.; Zhu, H.; Song, Z.; Koniusz, P.; and King, I. 2023b. Spectral feature augmentation for graph contrastive learning and beyond. In Proceedings of the AAAI Conference on Artificial Intelligence, 11289–11297.

Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R. 2022. FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, 27268–27286. PMLR.

# A Theorem Proof

# A.1 Proof of frequency-domain convolution theorem

Frequency-domain convolution theorem. The multiplication of two signals in the Fourier domain equals to the Fourier transformation of a convolution of these two signals in their original domain. This can be given by:

$$
\mathcal {F} (\mathbf {K} (v) \otimes \mathbf {Z} (v)) = \mathcal {F} (\mathbf {K} (v)) \odot \mathcal {F} (\mathbf {Z} (v)), \tag {16}
$$

where $\otimes$ and $\odot$ denote the convolutional operation and element-multiplication operation, and F refers to the Fourier transformation. $\mathbf{K}(v)$ and $\mathbf{Z}(v)$ represent two signals with respect with time variable v.

Proof. Suppose that the temporal dimension is denoted as $T$ , then

$$
\mathcal {F} (\mathbf {K} (v) \otimes \mathbf {Z} (v)) = \sum_ {i = 0} ^ {T - 1} (\mathbf {K} (v _ {i}) \otimes \mathbf {Z} (v _ {i})) e ^ {- j 2 \pi f v _ {i}}, \tag {17}
$$

where $j$ represents the imaginary unit. According to convolution theorem, which is written as $\mathbf{K}(v_i) \otimes \mathbf{Z}(v_i) = \sum_{j=0}^{T-1} \mathbf{K}(\tau_j) \mathbf{Z}(v_i - \tau_j)$ , then

$$
\mathcal {F} (\mathbf {K} (v) \otimes \mathbf {Z} (v)) = \sum_ {v = 0} ^ {T - 1} \sum_ {\tau = 0} ^ {T - 1} (\mathbf {K} (\tau) \mathbf {Z} (v - \tau)) e ^ {- j 2 \pi f v} \tag {18}
$$

$$
= \sum_ {v = 0} ^ {T - 1} \sum_ {\tau = 0} ^ {T - 1} \mathbf {Z} (v - \tau) e ^ {- j 2 \pi f v} \mathbf {K} (\tau).
$$

Let $x = v - \tau$ , then

$$
\begin{array}{l} \mathcal {F} (\mathbf {K} (v) \otimes \mathbf {Z} (v)) = \sum_ {v = 0} ^ {T - 1} \sum_ {\tau = 0} ^ {T - 1} \mathbf {Z} (x) e ^ {- j 2 \pi f (x + \tau)} x \mathbf {K} (\tau) \\ = \sum_ {v = 0} ^ {T - 1} \sum_ {\tau = 0} ^ {T - 1} \mathbf {Z} (x) e ^ {- j 2 \pi f x} e ^ {- j 2 \pi f \tau} \mathbf {K} (\tau) \\ = \sum_ {v = 0} ^ {T - 1} \mathbf {K} (\tau) e ^ {- j 2 \pi f \tau} \sum_ {\tau = 0} ^ {T - 1} \mathbf {Z} (x) e ^ {- j 2 \pi f x} \\ = \mathcal {F} (\mathbf {K} (v)) \odot \mathcal {F} (\mathbf {Z} (v)). \tag {19} \\ \end{array}
$$

Proved.

# A.2 Proof of Parseval's theorem

Parseval's theorem. The energy of feature in temporal domain can be completely characterized by the energy in the frequency domain, which can be formulated as follows:

$$
\sum_ {t = - \infty} ^ {\infty} | \mathbf {Z} (v) | ^ {2} = \sum_ {s = - \infty} ^ {\infty} | \mathbf {Z} _ {F} (\lambda_ {s}) | ^ {2}, \tag {20}
$$

where Z represents the original temporal domain feature, and $Z_{F}$ is the corresponding spectrum representation. They satisfy the equation $\mathbf{Z}_{F}(\lambda_{s}) = \sum_{i=0}^{T} \mathbf{Z}(v_{i}) e^{-j2\pi \lambda_{s} v_{i}}$ , where v is the temporal variable, $\lambda_{s}$ denotes the s-th frequency component.

Proof. Considering the representation of raw time series as $\mathbf{Z} \in \mathbb{R}^{T \times d}$ , where $T$ denotes the total length of time series, and we denote the temporal dimension as $v$ , then

<table><tr><td>Dataset</td><td>Example</td><td>Length</td><td>Channel</td><td>Class</td><td>Type</td></tr><tr><td>HAR</td><td>11,770</td><td>128</td><td>9</td><td>6</td><td>HAR</td></tr><tr><td>PS</td><td>6,668</td><td>217</td><td>11</td><td>39</td><td>SOUND</td></tr><tr><td>SRSCP1</td><td>561</td><td>896</td><td>6</td><td>2</td><td>EEG</td></tr><tr><td>MI</td><td>378</td><td>3,000</td><td>64</td><td>2</td><td>EEG</td></tr><tr><td>FM</td><td>416</td><td>50</td><td>28</td><td>2</td><td>EEG</td></tr><tr><td>AWR</td><td>575</td><td>144</td><td>9</td><td>25</td><td>MOTION</td></tr><tr><td>SAD</td><td>8,798</td><td>93</td><td>13</td><td>10</td><td>SPEECH</td></tr><tr><td>ECG5000</td><td>5,000</td><td>140</td><td>1</td><td>5</td><td>ECG</td></tr><tr><td>FordB</td><td>4,446</td><td>500</td><td>1</td><td>2</td><td>SENSOR</td></tr><tr><td>UWare</td><td>4,478</td><td>945</td><td>1</td><td>8</td><td>HAR</td></tr></table>

Table 5: Details of ten widely-used datasets in experiments.

$$
\sum_ {i = 0} ^ {T} \left| \mathbf {Z} (v _ {i}) \right| ^ {2} = \sum_ {i = 0} ^ {T} \mathbf {Z} (v _ {i}) \mathbf {Z} ^ {*} (v _ {i}), \tag {21}
$$

where $\mathbf{Z}^{*}(v)$ is the conjugate of $\mathbf{Z}(v)$ . According to inverse Fourier transformation, $\mathbf{Z}^{*}(v_{i}) = \sum_{s=0}^{S} \mathbf{Z}_{F}^{*}(\lambda_{s}) e^{j2\pi \lambda_{s} v_{i}}$ , we can obtain,

$$
\begin{array}{l} \sum_ {i = 0} ^ {T} \left| \mathbf {Z} (v _ {i}) \right| ^ {2} = \sum_ {i = 0} ^ {T} \mathbf {Z} (v _ {i}) \left[ \sum_ {s = 0} ^ {S} \mathbf {Z} _ {F} ^ {*} (\lambda_ {s}) e ^ {j 2 \pi \lambda_ {s} v _ {i}} \right] \\ = \sum_ {s = 0} ^ {S} \mathbf {Z} _ {F} ^ {*} (\lambda_ {s}) \left[ \sum_ {i = 0} ^ {T} \mathbf {Z} (v _ {i}) e ^ {j 2 \pi \lambda_ {s} v _ {i}} \right] \tag {22} \\ = \sum_ {s = 0} ^ {S} \mathbf {Z} _ {F} ^ {*} (\lambda_ {s}) \mathbf {Z} _ {F} (\lambda_ {s}) \\ = \sum_ {s = 0} ^ {S} | \mathbf {Z} _ {F} (\lambda_ {s}) | ^ {2}. \\ \end{array}
$$

Proved.

# B Experiment Setup

# B.1 Dataset Details

We conduct experiments to assess the superiority of our method under linear probing and fine-tuning settings on ten datasets, including Human Activity Recognition (HAR)(Anguita et al. 2012) and nine large-scale datasets from the UEA(Bagnall et al. 2018) and UCI(Dau et al. 2019) archive: PhonemeSpectra (PS), SelfRegulationSCP1 (SRSCP1), MotorImagery (MI), FingerMovements (FM), ArticularyWordRecognition (AWR), SpokenArabicDigits (SAD), ECG5000, FordB and UWaveGestureLibraryAll (UWare). These datasets cover diverse types of signals (human activity recognition, electroencephalography, electrocardiography, speech, sound, motion and sensor), different length (from 50 to 3000) and multivariate channel dimensions (from 1 to 64). When it comes to data processing, for HAR, we directly download the preprocessed files provided by TS-TCC (Eldele et al. 2021) while for the rest we adopt their pre-defined train-test splits. Table 5 summarizes the statistics of each dataset.

# B.2 Implementation Details

In the pre-training stage, we set the layer of transformer-based encoder to 8 following TimeMAE (Cheng et al. 2023). To preserve more information into the encoder and stimulate its encoding capacity, we set the layer of the temporal decoder and frequency decoder to 2. The hidden embedding size in transformer is set to 128 for all datasets. We use AdamW optimizer with a weight decay of $3e-4$ , $\beta_{1} = 0.9$ , and $\beta_{2} = 0.99$ . The learning rate of 1e-4 is adopted. The batch size and mask ratio is set to 128 and 75% by default, respectively. Due to the distribution difference of frequency information in different datasets, the value of $\gamma$ corresponding to the best performance of each data is slightly different. All methods are run with an NVIDIA A10 and implemented by PyTorch (Paszke et al. 2019).

For the reproduction details on baselines, we adopt the default hyperparameter setting claimed in their paper. For methods that can not provide insufficient hyperparameter details, such as the patch length in CRT (Zhang et al. 2023a), in their released code repository, we select the optimal one as their final experimental results. Notably, CLS (Liang et al. 2023) utilizes SVM (Cortes and Vapnik 1995) as the downstream classifier in its code, which differs from the learnable linear classification heads used in most of methods. For fair comparison, we replace it with a linear classification head consistent with TimesURL (Liu and Chen 2024) and re-measure the results.

# B.3 Baselines

To demonstrate the effectiveness of our plug-and-play frequency decoder in improving various MTM methods, and the superiority of our overall method, we select four advanced MTM frameworks as baselines. Additionally, five contrastive-based approaches are considered to further validate our performance.

1. Contrastive Learning Approaches:

(a) TS-TCC (Eldele et al. 2021) designs a tough cross-view prediction task to perform temporal and contextual contrastive learning.   
(b) TS2Vec (Yue et al. 2022) performs hierarchical contrastive learning to learn multi-scale contextual information at timestamp level and instance level, respectively.   
(c) TS-GAC (Wang et al. 2024) incorporates both node-level and graph-level contrastive strategies into graph learning framework to learn sensor- and global-level features.   
(d) TimesURL (Liu and Chen 2024) strengthens the universal time-series representations via frequency-temporal augmentation, where double Universums is elaborately constructed as a hard negative.   
(e) CSL (Liang et al. 2023) utilizes shapelet-based embeddings within a contrastive learning framework to effectively learn generalizable representations for multivariate time series.

2. Masked Time-series Modeling Approaches:

![](images/aa6f201d026ed0a42670c886ff46fa9e97f4e9a29f716fc914caec0f13868171.jpg)

![](images/f1fabe416c251805461a2e24c3582f124e392339bb173aa018712bc37770db23.jpg)

![](images/9cfc96e2e287e0c436b860514629a7cb6181dd65245277251417f9f38179125e.jpg)

![](images/c3bbc3817e286e9f426fb720ab9bd09a661dee280a2b73ca0d69205829591e32.jpg)

![](images/3abe6308ba11c5ebb4467df32a3929f4f32e2e178f37d25a4a2b47869363b716.jpg)

![](images/e264354680406a04924bce97629ac9e0e09cf0f6cb92158eb7360251d7657ccd.jpg)  
Figure 5: Learned Interaction Matrix on the PhonemeSpectra dataset.

![](images/812ecd19591e03e9bcbdab887bd211d63340f4bc82639abd55d2ba34035ee7c0.jpg)

![](images/795526e4c4c9b321ecc39dffc56c6c989a984a935c082be063a5aa25e7826d19.jpg)

![](images/a60c4cb5d7a9e55a56a0182c7987ac78403d86780ffd96e3e7b7d30fc1cd3199.jpg)

![](images/e2c7c33b957cf657ddd4d9baba9d2232aa165139df236e8503451347caf8f9fd.jpg)

![](images/cee492d1c6c121e37e6a7000f12fd49b022497f0772f31bf1b9ecde5be7b5e93.jpg)

![](images/337829ddcf7fc038596231d21a9bca78c70ed3298b9e5e7c99d7d67e3ea31c01.jpg)  
Figure 6: Learned Interaction Matrix on the FingerMovements dataset.

(a) PatchTST (Nie et al. 2023) improves long-term forecasting accuracy by segmenting multivariate time series into subseries-level patches, and employing a channel-independent strategy for representation learning.   
(b) CRT (Zhang et al. 2023a) proposes a cross-domain dropping-reconstruction task by randomly dropping certain patches in both time and frequency domains to emphasize temporal-spectral correlations.   
(c) SimMTM (Dong et al. 2023) reconstructs the original time series by aggregating point-wise representations from multiple masked variations.   
(d) TimeMAE (Cheng et al. 2023) aligns the reconstructed features with masked features encoded by a momentum encoder and further incorporates discretized encoding to enhance the reconstruction quality.

# C More Analytical Experiments

# C.1 Supplementary Material to Analysis of Content-aware Interaction Modulation

To improve the comprehensiveness of CIM analytic experiment (i.e., only the HAR dataset is provided in the main manuscript), we further include the visualization of interaction matrix regrading PhonemeSpectra and FingerMove

<table><tr><td>Method</td><td>Masked Ratio</td><td>Masking Rule</td><td>Masking Number</td></tr><tr><td>PatchTST</td><td>40%</td><td>Random</td><td>1</td></tr><tr><td>CRT</td><td>75%</td><td>Random</td><td>1</td></tr><tr><td>SimMTM</td><td>50%</td><td>Random</td><td>3</td></tr><tr><td>TimeMAE</td><td>60%</td><td>Random</td><td>1</td></tr><tr><td>Ours</td><td>75%</td><td>Random</td><td>1</td></tr></table>

Table 6: Masking Strategies of different methods.

ments datasets in Figure 5 and Figure 6. We analyze them from two aspects: (1) we observe that the trend of interactive pattern of these two datasets are different from HAR dataset. On HAR dataset, as training progresses (20 epoch → 50 epoch → 80 epoch), the effect of frequency decoder helps the model restrict the scope of receptive interactions. By contrast, the opposite trend regarding PS and FM datasets is exhibited, where the scope of receptive interaction gradually enlarges, supporting delicate expansion of interactive regions. (2) Compared with MTM, the learned interactive matrix of Ours could better mitigate the mindless interactive scope, which effectively optimize the rank of interactive matrix. These visualizations consistently demonstrate the adaptability and flexibility of our Content-aware Interaction Modulation unit in controlling the scope of interaction and mitigating the feature homogenization.

# C.2 Ablation Study on the Masking Strategy

In this section, we will provide a detailed description of the masking strategy employed in our approach, along with the ablation study on the masked ratio.

Masking Strategy For the masking number, we only randomly the timestamps along temporal dimension once, which is different from SimMTM that masks the timestamps with multiple times (SimMTM does so for better noise reduction). Moreover, we adopt the same masking rules as TimeMAE, where the raw time series is first encoded by a convolutional layer, and the masked to increase the information density of feature. For the masked ratio, we use 75% masked ratio for all datasets. This value is not the empirical value for these datasets, but it keep pace with the setting of vanilla MTM and CRT. Considering that MTM-based baselines may choose the optimal masked ratio according to their model design, we respect the original works and do not make any additional adjustment to their ratio setting. Instead, we conduct experiments according to the default configurations claimed in their papers, and their masking strategies are elaborated in Table 6.

Results with Different Masked Ratio We report some comparison results regarding mask ratio in Figure7, from which we give a three-point analysis: (1) The masked ratio serves as a barometer of the difficulty level of reconstruction task and has a significant impact on the model's representation learning. (2) The relationship between model performance and masked ratio is indeed not linear. Yet, there is a general trend of an initial increase followed by a decrease in model performance as the masked ratio increases, as shown

![](images/9829083076feab9590e429ebb6eeabd8ee63c87ef64d89a172b49731d7b37174.jpg)  
SimMTM TimeMAE Ours

Figure 7: Accuracy scores of HAR and PhonemeSpectra of LP and FT with respect to different masked ratios. in MAE(He et al. 2022) in CV domain, BERT(Devlin et al. 2018) in NLP domain, TimeMAE, SimMTM, and Ours (not limited to these). Specifically, when the masked ratio is low, it fails to stimulate the model to excavate sufficient contextual information thereby learning the satisfactory semantic representation via reconstruction process. As the mask ratio increases, the model learns more contextual information and enhances its generalization ability. However, excessively high mask ratio unavoidably result in the loss of crucial information. (3) In time-series domain, different datasets exhibit distinct optimal mask ratios. Arguably, for datasets with obvious periodicity, a smaller mask ratio is sufficient for the model to capture the contextual information via reconstruction, because the model could easily memorize the repetitive pattern. By contrast, for datasets with inapparent periodicity, a larger mask ratio allows the model to focus on more comprehensive contextual variation and accurately learn their representation, whereas a smaller mask ratio may be unable to find the complete pattern of time series.

# C.3 Full Results of CBD Generality

Table 7 presents the full results of our Content-aware Balanced Decoder (CBD) with other Masked Time-series Modeling Methods. Specifically, we maintain the original reconstruction branch (i.e., temporal decoder) and add a frequency reconstruction branch (i.e., CBD). We set the layer number of CBD to 2, and set the balanced weight $\gamma$ to 0.5. We can observe that across most datasets, our CBD significantly enhances the classification performance of the original MTM architecture. This indicates that CBD provides more diverse and balanced feature benefited from the spectrum space.

# C.4 Visualization Analysis

To demonstrate the superiority of our method on representation learning, we employ T-SNE (Van der Maaten and Hin-

<table><tr><td rowspan="2">Methods</td><td colspan="10">Linear Evaluation</td></tr><tr><td>HAR</td><td>PS</td><td>SRSCP1</td><td>MI</td><td>FM</td><td>AWR</td><td>SAD</td><td>ECG5000</td><td>FB</td><td>UWare</td></tr><tr><td>PatchTST</td><td>77.89</td><td>14.42</td><td>68.36</td><td>61.00</td><td>55.00</td><td>90.63</td><td>91.54</td><td>90.25</td><td>59.11</td><td>90.01</td></tr><tr><td>PatchTST + CBD</td><td>78.84</td><td>15.96</td><td>75.78</td><td>61.20</td><td>59.37</td><td>92.97</td><td>89.20</td><td>90.20</td><td>59.51</td><td>90.13</td></tr><tr><td>CRT</td><td>89.21</td><td>7.60</td><td>58.03</td><td>51.67</td><td>54.00</td><td>82.50</td><td>85.81</td><td>91.00</td><td>75.73</td><td>80.87</td></tr><tr><td>CRT + CBD</td><td>92.32</td><td>9.16</td><td>71.00</td><td>54.12</td><td>56.91</td><td>88.75</td><td>88.94</td><td>93.01</td><td>92.89</td><td>82.43</td></tr><tr><td>SimMTM</td><td>87.90</td><td>15.45</td><td>90.78</td><td>62.00</td><td>55.33</td><td>95.42</td><td>95.05</td><td>92.57</td><td>51.98</td><td>91.58</td></tr><tr><td>SimMTM + CBD</td><td>88.12</td><td>17.54</td><td>90.79</td><td>63.00</td><td>55.30</td><td>96.83</td><td>97.68</td><td>93.35</td><td>51.92</td><td>92.88</td></tr><tr><td rowspan="2"></td><td colspan="10">Fine-tune Evaluation</td></tr><tr><td>HAR</td><td>PS</td><td>SRSCP1</td><td>MI</td><td>FM</td><td>AWR</td><td>SAD</td><td>ECG5000</td><td>FB</td><td>UWare</td></tr><tr><td>PatchTST</td><td>88.45</td><td>18.42</td><td>81.25</td><td>62.40</td><td>57.81</td><td>98.05</td><td>97.75</td><td>94.40</td><td>75.65</td><td>91.36</td></tr><tr><td>PatchTST + CBD</td><td>87.81</td><td>19.35</td><td>84.77</td><td>63.00</td><td>59.38</td><td>98.44</td><td>97.79</td><td>94.44</td><td>79.17</td><td>91.48</td></tr><tr><td>CRT</td><td>90.09</td><td>8.38</td><td>67.65</td><td>50.67</td><td>53.00</td><td>87.62</td><td>98.23</td><td>92.53</td><td>79.24</td><td>85.64</td></tr><tr><td>CRT + CBD</td><td>92.39</td><td>9.61</td><td>73.78</td><td>55.11</td><td>55.57</td><td>93.73</td><td>98.23</td><td>94.91</td><td>93.19</td><td>89.32</td></tr><tr><td>SimMTM</td><td>93.50</td><td>21.95</td><td>92.72</td><td>62.00</td><td>60.00</td><td>98.57</td><td>99.30</td><td>92.76</td><td>82.53</td><td>94.64</td></tr><tr><td>SimMTM + CBD</td><td>94.27</td><td>24.37</td><td>92.83</td><td>65.00</td><td>62.00</td><td>98.53</td><td>99.54</td><td>94.04</td><td>80.37</td><td>95.90</td></tr></table>

Table 7: Full evaluation results of the CBD scalability and generality based on other MTM Methods (%)

![](images/6a157de69ce5d5e5a461f4338eb175094657f116621d89356c72d06044f1765d.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -50  | 20   | red     |
| -30  | 10   | blue    |
| -10  | 0    | green   |
| 10   | -10  | purple  |
| 30   | -20  | yellow  |
| 50   | -30  | orange  |
</details>

![](images/7b2556019e2c5633ef6ba3869eafed0cab8fb67ebf05a9116a54277fa3363f7f.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -50  | 10   | yellow  |
| -40  | 30   | orange  |
| -30  | -20  | purple  |
| -20  | -40  | purple  |
| -10  | -50  | purple  |
| 0    | -60  | purple  |
| 10   | -40  | red     |
| 20   | -20  | green   |
| 30   | 10   | blue    |
| 40   | 30   | red     |
| 50   | 50   | blue    |
| 60   | 70   | red     |
</details>

![](images/2d453d9d37496d6fdb5fb955b21b85bc18d963ee3f397f6b719da2d8b9215207.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -35  | 10   | red     |
| -25  | 20   | green   |
| -15  | 30   | blue    |
| -5   | 40   | purple  |
| 5    | 20   | orange  |
| 15   | 0    | yellow  |
| 35   | -20  | purple  |
| 55   | -40  | orange  |
| 75   | -60  | yellow  |
</details>

![](images/1f07d43d6afc0f5bf3da6327c52a888bf1f362c7fb4567aa1ea6adcbb5b80558.jpg)  
Figure 8: T-SNE visualization of feature vectors on the HAR dataset.

ton 2008) algorithm to display the features learned from HAR and ArticularyWordRecognition dataset on several methods in Figure 8 and Figure 9. Specifically, (a) and (b) show the representations obtained by TimeMAE and TS-GAC, respectively. (c) shows the vanilla masked time-series model, and (d) illustrates the visualized results from our overall model. We observe that our method exhibits better class boundary separability compared to the other three methods, particularly on the classes marked by color red, green and blue in HAR. This observation underlines the

![](images/62cf182698b1bca89305cd52a00ee1475e222f76578887e6809519365ba200e7.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -10  | 20   |
| -5   | 15   |
| 0    | 10   |
| 5    | 5    |
| 10   | 0    |
| 15   | -5   |
| -10  | -10  |
| -5   | -5   |
| 0    | 0    |
| 5    | 5    |
| 10   | 10   |
| 15   | 15   |
| -10  | -15  |
| -5   | -10  |
| 0    | -5   |
| 5    | 0    |
| 10   | 5    |
| 15   | 10   |
| -10  | -20  |
| -5   | -15  |
| 0    | -10  |
| 5    | -5   |
| 10   | 0    |
| 15   | 5    |
| -10  | -20  |
| -5   | -15  |
| 0    | -10  |
| 5    | -5   |
| 10   | 0    |
| 15   | 5    |
| -10  | -20  |
| -5   | -15  |
| 0    | -10  |
| 5    | -5                   |
| 10   | 0                    |
| 15   | 5                    |
| -10  | -20  |
| -5   | -15  |
| 0    | -10  |
| 5    | -5   |
| 10   | 0                    |
| 15   | 5                    |
| -10  | -20  |
| -5   | -15  |
| 0    | -10  |
| 5    | -5   |
| 10   | 0                    |
| 15   | 5                    |
| -10  | -20  |
| -5   | -10  |
| 0    | -5   |
| 5    | 0    |
| 10   | 5    |
| 15   | 10   |
| -10  | -20  |
| -5   | -10  |
| 0    | -5   |
| 5    | 0    |
| 10   | 5    |
| 15   | 10   |
| -10  | -20  |
| -5   | -10  |
| 0    | -5   |
| 5    | 0    |
</details>

![](images/6310791056d1c6bd8720b31536a96cc443727ecd87a3418019a517ed5f407345.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -20  | 0    |
| -15  | 5    |
| -10  | 10   |
| -5   | 15   |
| 0    | 10   |
| 5    | 5    |
| 10   | 0    |
| 15   | -5   |
| 20   | -10  |
</details>

![](images/2f14125f2637e87009de3eb0e762a002f4b3fd54d9a9be61f8f4f2e1b2427285.jpg)

<details>
<summary>scatter</summary>

| x       | y       |
| ------- | ------- |
| -14.0   | 6.0     |
| -12.0   | 8.0     |
| -10.0   | 9.0     |
| -8.0    | 7.0     |
| -6.0    | 5.0     |
| -4.0    | 3.0     |
| -2.0    | 1.0     |
| 0.0     | -1.0    |
| 2.0     | -3.0    |
| 4.0     | -5.0    |
| 6.0     | -7.0    |
| 8.0     | -9.0    |
| 10.0    | -11.0   |
| 12.0    | -13.0   |
| 14.0    | -15.0   |
</details>

![](images/668d4f948b3577cda33cb7565a211f98e2f964e1c11e3cf2c7ac2b6e3f21100a.jpg)

<details>
<summary>scatter</summary>

| x       | y       |
| ------- | ------- |
| -14.0   | 8.0     |
| -12.0   | 6.0     |
| -10.0   | 4.0     |
| -8.0    | 2.0     |
| -6.0    | 0.0     |
| -4.0    | -2.0    |
| -2.0    | -4.0    |
| 0.0     | -6.0    |
| 2.0     | -8.0    |
| 4.0     | -10.0   |
| 6.0     | -12.0   |
| 8.0     | -14.0   |
| 10.0    | -16.0   |
| 12.0    | -14.0   |
| 14.0    | -12.0   |
| 16.0    | -10.0   |
| 18.0    | -8.0    |
| 20.0    | -6.0    |
| 22.0    | -4.0    |
| 24.0    | -2.0    |
| 26.0    | 0.0     |
| 28.0    | 2.0     |
| 30.0    | 4.0     |
| 32.0    | 6.0     |
| 34.0    | 8.0     |
| 36.0    | 10.0    |
| 38.0    | 12.0    |
| 40.0    | 14.0    |
| 42.0    | 16.0    |
| 44.0    | 18.0    |
| 46.0    | 20.0    |
| 48.0    | 22.0    |
| 50.0    | 24.0    |
| 52.0    | 26.0    |
| 54.0    | 28.0    |
| 56.0    | 30.0    |
| 58.0    | 32.0    |
| 60.0    | 34.0    |
| 62.0    | 36.0    |
| 64.0    | 38.0    |
| 66.0    | 40.0    |
| 68.0    | 42.0    |
| 70.0    | 44.0    |
| 72.0    | 46.0    |
| 74.0    | 48.0    |
| 76.0    | 50.0    |
| 78.0    | 52.0    |
| 80.0    | 54.0    |
| 82.0    | 56.0    |
| 84.0    | 58.0    |
| 86.0    | 60.0    |
| 88.0    | 62.0    |
| 90.0    | 64.0    |
| 92.0    | 66.0    |
| 94.0    | 68.0    |
| 96.0    | 70.0    |
| 98.0    | 72.0    |
| 100.0   | 74.0    |
</details>

Figure 9: T-SNE visualization of feature vectors on the ArticularyWordRecognition dataset.

promising role of our overall framework in enhancing representation discriminative learning.