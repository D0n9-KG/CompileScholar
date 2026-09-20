# PromptRestorer: A Prompting Image Restoration Method with Degradation Perception

Cong Wang $^{1,*}$ , Jinshan Pan $^{2,*}$ , Wei Wang $^{3,*}$ , Jiangxin Dong $^{2}$ , Mengzhu Wang $^{4}$ , Yakun Ju $^{1}$ , Junyang Chen $^{5}$

$^{1}$ The Hong Kong Polytechnic University, $^{2}$ Nanjing University of Science and Technology, $^{3}$ Dalian University of Technology, $^{4}$ Hebei University of Technology, $^{5}$ Shenzhen University

# Abstract

We show that raw degradation features can effectively guide deep restoration models, providing accurate degradation priors to facilitate better restoration. While networks that do not consider them for restoration forget gradually degradation during the learning process, model capacity is severely hindered. To address this, we propose a Prompting image Restorer, termed as PromptRestorer. Specifically, PromptRestorer contains two branches: a restoration branch and a prompting branch. The former is used to restore images, while the latter perceives degradation priors to prompt the restoration branch with reliable perceived content to guide the restoration process for better recovery. To better perceive the degradation which is extracted by a pre-trained model from given degradation observations, we propose a prompting degradation perception modulator, which adequately considers the characters of the self-attention mechanism and pixel-wise modulation, to better perceive the degradation priors from global and local perspectives. To control the propagation of the perceived content for the restoration branch, we propose gated degradation perception propagation, enabling the restoration branch to adaptively learn more useful features for better recovery. Extensive experimental results show that our PromptRestorer achieves state-of-the-art results on 4 image restoration tasks, including image deraining, deblurring, dehazing, and desnowing.

# 1 Introduction

Image restoration aims to recover clear high-quality images from given degraded ones. It is highly ill-posed since only degraded images can be exploited, statistical observations are thus required to well-pose the problems $[37, 67, 66, 41, 42]$ . Although conventional approaches can recover images to some extent, they typically involve solving optimization algorithms that are difficult due to the non-convexity and non-smooth problems. Additionally, the observations may not always hold, which can cause algorithms to fail.

With the emergence of convolutional neural networks (CNNs) [48] and Transformers [22, 45], which perform well at implicitly learning the priors from large-scale data, learning-based methods have dominated recent image restoration tasks and achieved impressive performance [81, 51, 103, 72, 61, 93, 33, 13]. However, these methods are usually built without explicitly considering the specific degradation information, which accordingly limits model capacity (Case 1 in Fig. 1). An alternative approach is to design a conditional branch to learn additional information to provide the restoration network with useful content for modulation [30, 17, 34, 35] (Case 2 in Fig. 1). However, we note that while conditional branches in these models are learnable, they may not effectively provide degradation information, as the optimizable parameters result in gradually clearer features during the learning process, leading to the degradation vanishing which accordingly limits model performance.

![](images/a1476b8d3bd1b29fe07e6b8fea8d0ddd734de8ad11810822c8d3a265b78725c8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Enc."]
    B --> C["Dec."]
    C --> D["Output"]
    E["Condition"] --> F["Learnable Net"]
    F --> C
    G["Input"] --> H["Enc."]
    H --> I["Dec."]
    I --> J["Output"]
    K["Pre-trained Model"] --> L["Enc."]
    L --> M["Dec."]
    M --> N["Output"]
```
</details>

![](images/cb9a75c760f29df342a3fcd26912b318e03d3af7134cd276a028526b45483a56.jpg)

<details>
<summary>line</summary>

| Iterations (x10^4) | Case 1 PSNR (dB) | Case 2 PSNR (dB) | Case 3 PSNR (dB) |
| ------------------ | ---------------- | ---------------- | ---------------- |
| 0                  | 26.5             | 26.5             | 26.5             |
| 2                  | 27.0             | 27.0             | 27.0             |
| 4                  | 27.5             | 27.5             | 27.5             |
| 6                  | 28.0             | 28.0             | 28.0             |
| 8                  | 28.5             | 28.5             | 28.5             |
| 10                 | 29.0             | 29.0             | 29.0             |
| 12                 | 29.5             | 29.5             | 29.5             |
| 14                 | 30.0             | 30.0             | 30.0             |
| 16                 | 30.5             | 30.5             | 30.5             |
| 18                 | 31.0             | 31.0             | 31.0             |
| 20                 | 31.5             | 31.5             | 31.5             |
| 22                 | 32.0             | 32.0             | 32.0             |
| 24                 | 32.5             | 32.5             | 32.5             |
| 26                 | 33.0             | 33.0             | 33.0             |
| 28                 | 33.5             | 33.5             | 33.5             |
| 30                 | 34.0             | 34.0             | 34.0             |
| 32                 | 34.5             | 34.5             | 34.5             |
| 34                 | 35.0             | 35.0             | 35.0             |
| 36                 | 35.5             | 35.5             | 35.5             |
| 38                 | 36.0             | 36.0             | 36.0             |
| 40                 | 36.5             | 36.5             | 36.5             |
| 42                 | 37.0             | 37.0             | 37.0             |
| 44                 | 37.5             | 37.5             | 37.5             |
| 46                 | 38.0             | 38.0             | 38.0             |
| 48                 | 38.5             | 38.5             | 38.5             |
| 50                 | 39.0             | 39.0             | 39.0             |
</details>

(b) Learning Curves

![](images/9022917101390c13c289474abe8e69f606d0f13ab81a33eaf0980732f7a90053.jpg)

<details>
<summary>text_image</summary>

Degraded Image
Input
PSNR
PSNR
Case 1
25.8
99.2
4823
Case 2
26.1
29.7
4823
Case 3
27.0
30.0
4823
GT
∞
20
</details>

(c) Visual Performance   
Figure 1: (a) compares different restoration frameworks. Unlike existing approaches that are built within the architectures such as Cases 1-2, which are unable to memorize the degradation well during the learning process, we propose a prompting method (Case 3) that directly exploits raw degradation features extracted by a pre-trained model from the given degradation observations to guide restoration. In (b), we observe that both Cases 1-2 outperform our method in early iterations, as they effectively memorize degraded information. However, both Cases 1-2 experience degradation vanishing with further iterations (better demonstrated in Sec. 4.3), while our prompting method persists in guiding the restoration network with accurate degradation priors, accordingly producing better restoration quality. In (c), visual performance demonstrates that our prompting method recovers sharper images. Quantitative results are reported in Tab. 5.

Recently, prompt learning has been shown an effective tool to improve model performance by designing various prompts $[115, 98, 104, 23, 114, 46, 53, 43]$ . The prompt usually serves as the guidance tool to correct the networks toward better results $[28]$ . However, prompt learning still keeps a margin for image restoration, and existing prompts may not be suitable for image restoration since they cannot effectively model degradation priors well. Hence, we ask: Is there a reasonable prompting manner to correct degraded image restoration networks to facilitate better recovery?

The answer is in the riddle. This paper proposes the PromptRestorer, a Prompting image Restorer, to overcome degradation vanishing in image restoration via promoting by exploring degradation input itself for better restoration (Case 3 in Fig. 1). Our idea is simple: we directly exploit the raw degraded features extracted by a pre-trained model from the degraded inputs to generate more reliable prompting content to guide image restoration. Raw degraded features preserve accurately degraded information, which can consistently prompt the restoration network with accurate degraded priors, enabling the restoration network to perceive the degradation for better restoration. Hence, we design the PromptRestorer, which consists of two branches: (a) the restoration branch and (b) the prompting branch. The former is used to restore images and the latter is used to generate reliable prompting features to guide the restoration network for better restoration. To better perceive the degradation, we propose a Prompting Degradation Perception Modulator (PromptDPM), which consists of Global Prompting Perceptor (G2P) and Local Prompting Perceptor (L2P). The G2P adequately exploits the self-attention mechanism to form global prompting attention, while the L2P considers the pixel-level perception to build local prompting content. To control the propagation of perceived features in the restoration branch, we propose Gated Degradation Perception Propagation (GDP), enabling the restoration network to adaptively learn more useful features to facilitate better restoration.

The main contributions of this work are summarized below:

- We propose PromptRestorer, which is the first approach to our knowledge that takes advantage of the prompting learning for general image restoration by considering raw degradation features in restoration, enabling the restoration model to overcome degradation vanishing while consistently retaining the degradation priors to facilitate better restoration.   
- We propose a prompting degradation perception modulator that is used to perceive degradation from global and local perspectives, which is able to provide the restoration network

with more reliable perceived content learned from the degradation priors, enabling it to better guide the restoration process.

\- We propose gated degradation perception propagation that exploits a gating mechanism to control the propagation of the perceived features, enabling the model to adaptively learn more useful features for better image restoration.

Fig. 1 summarises framework comparisons, and their learning curves and visual performance. Deeper analysis and discussion about them are provided in Sec. 4.2.

# 2 Related Work

In this section, we review image restoration, conditional modulation, and prompt learning.

Image Restoration. Recently, CNN-based architectures $[110, 112, 103, 4, 24, 102, 89, 91, 87, 19, 88, 117]$ and Transformer-based models $[96, 56, 49, 12, 93, 93]$ have been shown to outperform conventional restoration approaches $[37, 83, 64, 47, 7, 79]$ . These learning-based methods usually adopt U-Net architectures $[50, 18, 103, 100, 1, 93, 108, 101]$ , which have been demonstrated the effectiveness because of hierarchical multi-scale representation and effective learning between shallow and deeper layers by skip connection $[111, 59, 102, 31]$ . We refer the readers to recent excellent literature reviews on image restoration $[5, 54, 82]$ , which summarise the main designs in deep image restoration models.

Although these models have achieved promising performance, they do not explicitly take degradation into consideration for model design which is vital for restoration, limiting the model capacity.

Conditional Modulation. Conditional modulation usually involves implicitly modulating the additional content to guide the restoration $[30, 10, 16, 17, 36, 35, 34, 60, 57, 92]$ . These approaches usually contain two branches: a basic network and a conditional network. The conditional network provides additional information to guide the basic network for restoration via spatial feature transform (SFT) $[92]$ . Among these methods, blur kernel $[30]$ , semantics $[92]$ , and degraded input $[57, 16]$ which serve as the additional conditions are broadly known.

The learnable nature of the conditional network in these models does not effectively provide degradation information for the basic network. As parameters are optimized in the learning process, features become gradually clear.

Prompt Learning. Prompt learning methods have been studied broadly in natural language processing (NLP) $[75, 78, 8]$ . Due to high effectiveness, prompt learning is recently used in vision-related tasks $[115, 98, 104, 23, 114, 46, 53, 43, 71, 29, 40, 86, 28]$ . In vision prompt learning, many works seek useful prompts to correct task networks toward better performance $[28]$ .

Although prompt learning has shown promise in various vision tasks, it still keeps a margin in general image restoration. This paper proposes an effective prompting method, enabling the restoration model to overcome the degradation vanishing in the learning process for better restoration.

# 3 PromptRestorer

Our goal aims to overcome degradation vanishing and better perceive degradation in deep restoration models to improve image recovery quality. To achieve this, we introduce a prompting strategy that helps the model consistently memorize degradation information, enabling it to prompt restoration with better degradation for better restoration. To better perceive degradation, we propose the Prompting Degradation Perception Modulator (PromptDPM), which can provide more reliable perceived content to guide the restoration network. To control the propagation of the perceived content, we propose the Gated Degradation Perception Propagation (GDP), enabling the restoration network to adaptively learn more useful features for better restoration.

# 3.1 Overall Pipeline

Fig. 2 shows the framework of our PromptRestorer, which contains two branches: (a) the restoration branch and (b) the prompting branch. The restoration branch is used to restore images, where each block is prompted by the prompting branch. The prompting branch first generates the accurate

![](images/baf4acec73e4f559a1c3330c7dde348dc1aae730efc4455f7d7a4e670e36adb5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["(a) Restoration Branch Input (I)"] --> B["Extraction"]
    B --> C["Continuous Gated Transformer"]
    C --> D["PGD"]
    D --> E["×L1"]
    E --> F["Down ×2↓"]
    F --> G["×L2"]
    G --> H["Down ×2↓"]
    H --> I["×L3"]
    I --> J["Up ×2↑"]
    J --> K["×L2"]
    K --> L["Up ×2↑"]
    L --> M["×L1"]
    M --> N["Reconstruction"]
    N --> O["Output (Ĥ)"]
    
    P["Pre-trained Model"] --> Q["Raw Degradation Features"]
    Q --> R["Y1"]
    R --> S["Layer Norm"]
    S --> T["Multi-Head Attention"]
    T --> U["+"]
    U --> V["Layer Norm"]
    V --> W["1×1 Conv"]
    W --> X["GELU"]
    X --> Y["1×1 Conv"]
    Y --> Z["DConv"]
    Z --> AA["+"]
    AA --> AB["Output (Ĥ)"]
    
    AC["(b) Prompting Branch"] --> AD["Transformer Block (TFB)"]
    AD --> AE["Improved ConvNeXt"]
    AE --> AF["Layer Norm"]
    AF --> AG["Layer Norm"]
    AG --> AH["1×1 Conv"]
    AH --> AI["GELU"]
    AI --> AJ["1×1 Conv"]
    AJ --> AK["DConv"]
    AK --> AL["+"]
    AL --> AM["Output (Ĥ)"]
    
    AN["(d) Prompting Degradation Perception Modulator (PromptDPM)"] --> AO["Y1"]
    AO --> AP["G2P"]
    AP --> AQ["L2P"]
    AQ --> AR["Concatl"]
    AR --> AS["1×1 Conv"]
    AS --> AT["Output (Ĥ)"]
    
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#ffc,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#ffc,stroke:#333
    style N fill:#ffc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#cfc,stroke:#333
    style Q fill:#cfc,stroke:#333
    style R fill:#cfc,stroke:#333
    style S fill:#cfc,stroke:#333
    style T fill:#cfc,stroke:#333
    style U fill:#cfc,stroke:#333
    style V fill:#cfc,stroke:#333
    style W fill:#cfc,stroke:#333
    style X fill:#cfc,stroke:#333
    style Y fill:#cfc,stroke:#333
```
</details>

Figure 2: Overall pipeline of our PromptRestorer. PromptRestorer contains two branches: (a) the restoration branch and (b) the prompting branch. The restoration branch is used to restore images, where each block (c) in CGT is prompted by the prompting branch. The prompting branch first generates precise degradation features extracted by a pre-trained model from degradation observations, then these features prompt the restoration branch to facilitate better restoration via PromptDPM (d).

degradation feature extracted by a pre-trained model, and then the feature is to prompt the restoration branch, enabling the restoration branch to better perceive the degradation prior for better recovery.

Restoration Branch. Given a degraded input image $I \in R^{H \times W \times 3}$ , we first applies a $3 \times 3$ convolution as the feature extraction to obtain low-level embeddings $X_{0} \in R^{H \times W \times C}$ ; where $H \times W$ denotes the spatial dimension and C is the number of channels. Next, the shallow features $X_{0}$ gradually are hierarchically encoded into deep features $X_{l} \in R^{\frac{H}{l} \times \frac{W}{l} \times lC}$ . After encoding the degraded input into low-resolution latent features $X_{3} \in R^{\frac{H}{3} \times \frac{W}{3} \times 3C}$ , the decoder progressively recovers the high-resolution representations. Finally, a reconstruction layer which contains 4 Transformer blocks as the refinement followed by a $3 \times 3$ convolution is applied to decoded features to generate residual image $S \in R^{H \times W \times 3}$ to which degraded image is added to obtain the restored output image: $\hat{H} = I + S$ . Both encoder and decoder at l-level consist of multiple Continuous Gated Transformers (CGT) with expanding channel capacity. To help better recovery, the encoder features are concatenated with the decoder features via skip connections [74] by $1 \times 1$ convolutions.

Prompting Branch. The prompting branch, as shown in Fig. 2(b), aims to generate and perceive degradation features and then provide useful guidance content for the restoration branch. We note VQGAN [25] has been demonstrated that it can generate high-quality images while representing the features of input images. However, it tends to damage image structure after vector quantization [116, 11, 32]. To avoid this problem, we only exploit the encoder of pre-trained VQGAN to represent the deep features of the degraded inputs. We first use the pretrained encoder to extract degraded features $\mathbf{Y}_l\in \mathbb{R}^{\frac{H}{L}\times \frac{W}{L}\times lC}$ ; where $l$ denotes the $l$ -level layer in the pre-trained encoder. Then, the degraded features are exploited to generate reliable prompting content to prompt the restoration branch by PromptDPM (see Sec. 3.2). The generated prompting content is transmitted to each CGT to guide the restoration branch.

Continuous Gated Transformers. CGT exploits the perceived features from PromptDPM to provide the Transformer block with more reliable content to overcome degradation vanishing to facilitate better restoration. Each CGT consists of three Transformer blocks (Fig. 2(c)) with residual connections [38] and the input in each block is gated by GDP (see Sec. 3.3) to control the propagation of perceived features. Let $\mathcal{P},\mathcal{G}$ , and $\mathcal{T}$ respectively denote the operations of PromptDPM (expressed in (2)), GDP (expressed in (7)), and Transformer, the features flow in $k^{th}$ block in one CGT at $l$ -level encoder/decoder, which can be expressed as:

$$
\mathbf {X} _ {k} = \mathcal {T} (\mathbf {G} _ {k - 1}); \mathbf {G} _ {k - 1} = \mathcal {G} \big (\mathbf {X} _ {k - 1}, \mathbf {P} _ {l} \big); \mathbf {P} _ {l} = \mathcal {P} (\mathbf {X} _ {k - 1}, \mathbf {Y} _ {l}), \tag {1}
$$

![](images/52165cfe356300213e59a9fdb4667f57a7f1439c8f3520e5d2bb480c182cf9df.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) G2P
        A["Layer Norm"] --> B["Global Perception Attention"]
        B --> C["Layer Norm"]
        C --> D["Improved ConvNeXt"]
    end

    subgraph (b) L2P
        E["Local Perception Modulator"] --> F["DConv"]
        F --> G["GELU"]
        G --> H["1×1 Conv"]
        H --> I["Concat"]
        I --> J["Softmax"]
        J --> K["Max"]
        K --> L["Concat"]
        L --> M["Softmax"]
        M --> N["KV-Induced Attention"]
        O["Q-Induced Attention"] --> P["Q"]
        Q["Q"] --> R["Concat"]
        S["Q"] --> T["Concat"]
        U["Q"] --> V["Concat"]
        W["Q"] --> X["Concat"]
        Y["Y"] --> Z["Layer Norm"]
        AA["1×1 Conv"] --> AB["DConv"]
        AC["V"] --> AD["K"]
        AE["V"] --> AF["Concat"]
        AG["V"] --> AH["Concat"]
        AI["V"] --> AJ["Concat"]
        AK["V"] --> AL["Concat"]
    end

    subgraph (b) L2P
        M["Local Perception Modulator"] --> N
        O["Q-Induced Attention"] --> P
        P --> Q
        Q --> R
        R --> S
        S --> T
        T --> U
        U --> V
        V --> W
        W --> X
        X --> Y
        Y --> Z
        Z --> AA
        AA --> AB
        AB --> AC
        AC --> AD
        AD --> AE
        AE --> AI
        AI --> AJ
        AJ --> AK
        AK --> AL
        AL --> M
        M --> N
        N --> O
        O --> P
        P --> Q
        Q --> R
        R --> S
        S --> T
        T --> U
        U --> V
        V --> W
        W --> X
        X --> Y
        Y --> Z
    end

    A -->|X| B
    C -->|Y| D
    D -->|X| E
    E -->|X| F
    F -->|X| G
    G -->|X| H
    H -->|X| I
    I -->|X| J
    J -->|X| K
    K -->|X| L
    L -->|X| M
    M -->|X| N
    N -->|X| O
    O -->|X| P
    P -->|X| Q
    Q -->|X| R
    R -->|X| S
    S -->|X| T
    T -->|X| U
    U -->|X| V
    V -->|X| W
    W -->|X| X
    X -->|X| Y
    Y -->|X| Z
    
    subgraph (b) L2P
        Z["Matrix Multiplication"]
        AA["Reshape"]
        AB["Restoration-Induced Band"]
    end

    subgraph (b) L2P
        AC["Degradation-Induced Band"]
        AD["Restoration-Induced Band"]
    end

    subgraph (b) L2P
        AE["1×1 Conv"]
        AF["DConv"]
        AG["Q"]
        AH["K"]
        AI["Q"]
        AJ["K"]
        AK["K"]
        AL["Q"]
        AM["K"]
        AN["K"]
        AO["Q"]
        AP["K"]
        AQ["K"]
    end

    style (b) L2P fill:#f9f,stroke:#333,stroke-width:2px
```
</details>

Figure 3: (a) Global Prompting Perceptor (G2P); (b) Local Prompting Perceptor (L2P).

where $X_{k}$ means the output of $k^{th}$ Transformer block in one CGT, especially $X_{0}$ is the downsampled/upsampled features at $(l-1)$ -level encoder/decoder; $P_{l}$ refers to the generated features of PromptDPM at l-level layer; $G_{k-1}$ means the gated features between $X_{k-1}$ and $P_{l}$ , which serves as the input of $k^{th}$ Transformer block. Each Transformer block consists of multi-head attention [101] followed by an improved ConvNeXt [62] as the feed-forward network (see Fig. 2(c)).

# 3.2 Prompting Degradation Perception Modulator

To better perceive the degradation to prompt the restoration network with more reliable perceived content from the degradation priors, we propose the PromptDPM (see Fig. 2(d)). The PromptDPM consists of 1) Global Prompting Perceptor (G2P, introduced in Sec. 3.2.1) and 2) Local Prompting Perceptor (L2P, introduced in Sec. 3.2.2) to respectively perceive the degradation from global and local perspectives, enabling to generate more useful content to guide the restoration branch. From a restoration tensor $\mathbf{X} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ and a degradation tensor $\mathbf{Y} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ , we prompt $\mathbf{X}$ with $\mathbf{Y}$ :

$$
\mathcal {P} (\mathbf {X}, \mathbf {Y}) = W _ {p} \left(\mathcal {C} \left[ \Psi^ {\text { global }} (\mathbf {X}, \mathbf {Y}), \Psi^ {\text { local }} (\mathbf {X}, \mathbf {Y}) \right]\right) + \mathbf {X}, \tag {2}
$$

where $\Psi^{\mathrm{global}}(\cdot ,\cdot)$ and $\Psi^{\mathrm{local}}(\cdot ,\cdot)$ respectively denote the operations of G2P and L2P; $\mathcal{C}[\cdot ,\cdot ]$ means the concatenation at channel dimension; $W_{p}(\cdot)$ refers to the $1\times 1$ point-wise convolution.

# 3.2.1 Global Prompting Perceptor

The G2P, shown in Fig. 3(a), fully exploits the self-attention mechanism to form the global prompting attention induced by the degraded features. The G2P contains the global perception attention followed by an improved ConvNeXt [62]. Our global perception attention consists of 1) Query-Induced Attention (Q-InAtt) and 2) Key-Value-Induced Attention (KV-InAtt). The Q-InAtt considers re-forming the query vector induced by degradation features to build a representative query to perform attention, while the KV-InAtt re-considers key and value vectors induced by other degradation counterparts to search for more similar content with the restoration query. From a layer normalized restoration tensor $\mathbf{X} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ , our G2P first generates restoration query (Q), key (K), and value (V) projections from the restoration features. It is achieved by applying $1 \times 1$ convolutions to aggregate pixel-wise cross-channel context followed by $3 \times 3$ depth-wise convolutions $W_{d}(\cdot)$ to encode channel-wise spatial context, yielding $\mathbf{Q} = W_{d}W_{p}\mathbf{X}$ , $\mathbf{K} = W_{d}W_{p}\mathbf{X}$ , and $\mathbf{V} = W_{d}W_{p}\mathbf{X}$ . Meanwhile, we similarly convert the degradation tensor $\mathbf{Y} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ into degradation query ( $\widetilde{\mathbf{Q}}$ ), key ( $\widetilde{\mathbf{K}}$ ), and value ( $\widetilde{\mathbf{V}}$ ) projections: $\widetilde{\mathbf{Q}} = W_{d}W_{p}\mathbf{Y}$ , $\widetilde{\mathbf{K}} = W_{d}W_{p}\mathbf{Y}$ , and $\widetilde{\mathbf{V}} = W_{d}W_{p}\mathbf{Y}$ . Then, we respectively conduct Q-InAtt and KV-InAtt:

$$
\mathbf {A} _ {\mathrm{Q} - \text { InAtt }} = \mathcal {A} _ {\mathrm{Q} - \text { InAtt }} \left(W _ {p} (\mathcal {C} [ \mathbf {Q}, \widetilde {\mathbf {Q}} ]), \mathbf {K}, \mathbf {V}\right); \mathbf {A} _ {\mathrm{KV} - \text { InAtt }} = \mathcal {A} _ {\mathrm{KV} - \text { InAtt }} \left(\mathbf {Q}, W _ {p} (\mathcal {C} [ \mathbf {K}, \widetilde {\mathbf {K}} ]), W _ {p} (\mathcal {C} [ \mathbf {V}, \widetilde {\mathbf {V}} ])\right), \tag {3}
$$

where $\mathcal{A}_{(\cdot)}\left(\hat{\mathbf{Q}},\hat{\mathbf{K}},\hat{\mathbf{V}}\right) = \hat{\mathbf{V}}\cdot \mathrm{Softmax}\left(\hat{\mathbf{K}}\cdot \hat{\mathbf{Q}} /\alpha\right)$ ; Here, $\alpha$ is a learnable scaling parameter to control

the magnitude of the dot product of $\hat{K}$ and $\hat{Q}$ before applying the softmax function. Similar to the conventional multi-head SA [22], we divide the number of channels into ‘heads’ and learn separate attention maps. Then two induced attentions are fused and followed by an improved ConvNeXt:

$$
\mathbf {A} ^ {\prime} = W _ {p} \left(\mathcal {C} \left[ \mathbf {A} _ {\mathrm{Q-InAtt}}, \mathbf {A} _ {\mathrm{KV-InAtt}} \right]\right) + \mathbf {X}; \mathbf {A} = W _ {p} W _ {d} \phi W _ {p} W _ {d} \left(L N \left(\mathbf {A} ^ {\prime}\right)\right) + \mathbf {A} ^ {\prime}, \tag {4}
$$

where the $W_{p}W_{d}\phi W_{p}W_{d}(\cdot)$ means the improved ConvNeXt shown in the latter of Fig. 2(c); $LN(\cdot)$ means the operation of layer normalization [6].

# 3.2.2 Local Prompting Perceptor

The L2P, as shown in Fig. 3(b), adequately considers the pixel-level degradation perception, enabling to better perceive degradation from spatially neighboring pixel positions. The L2P consists of a local perception modulator followed by a separable depth-level convolution. The local perception modulator contains two core components: 1) Degradation-Induced Band (Deg-InBan) and 2) Restoration-Induced Band (Res-InBan). The former is achieved by exploiting the degradation features to induce spatially useful content from restoration content to guide restoration gating fusion, while the latter utilizes the deep restoration features to induce more useful features from another degradation counterpart to form the degradation gating. Given the degradation tensor $\mathbf{Y} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ , we first exploit the point-wise convolution and $3 \times 3$ depth-wise convolution to encode two degradation projections, yielding $\widetilde{\mathbf{Q}} = W_d^Q W_p^Q \mathbf{Y}$ and $\widetilde{\mathbf{K}} = W_d^K W_p^K \mathbf{Y}$ . Meanwhile, the restoration tensor $\mathbf{X} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \hat{C}}$ are also encoded into two restoration projections: $\mathbf{Q} = W_d^Q W_p^Q \mathbf{X}$ and $\mathbf{K} = W_d^K W_p^K \mathbf{X}$ . Then, we respectively conduct Deg-InBan and Res-InBan:

$$
\mathbf {Z} _ {\text {Deg - InBan}} = \sigma \left(W _ {p} \phi W _ {d} (\mathcal {C} [ \widetilde {\mathbf {Q}}, \mathbf {Q} ])\right) \odot \mathbf {K}; \mathbf {Z} _ {\text {Res - InBan}} = \widetilde {\mathbf {Q}} \odot \sigma \left(W _ {\phi} W _ {d} (\mathcal {C} [ \widetilde {\mathbf {K}}, \mathbf {K} ])\right), \tag {5}
$$

where $\sigma(\cdot)$ denotes the sigmoid function that controls the gating level. Then, the perceived features in the two bands are fused via concatenation and $1 \times 1$ convolution and followed by a depth-level separable convolution $W_{p}\phi W_{d}(\cdot)$ :

$$
\mathbf {Z} ^ {\prime} = W _ {p} \left(\mathcal {C} \left[ \mathbf {Z} _ {\text {Deg - InBan}}, \mathbf {Z} _ {\text {Res - InBan}} \right]\right) + \mathbf {X}; \mathbf {Z} = W _ {p} \phi W _ {d} \left(\mathbf {Z} ^ {\prime}\right) + \mathbf {Z} ^ {\prime}. \tag {6}
$$

# 3.3 Gated Degradation Perception Propagation

The GDP aims to control the propagation of the perceived degradation, enabling to adaptively learn more useful features in Transformer blocks to facilitate better restoration. Given the output restoration tensor $X_{k-1} \in R^{\hat{H} \times \hat{W} \times \hat{C}}$ of $(k - 1)^{th}$ Transformer block in one CGT and the perceived tensor $P_{l} \in R^{\hat{H} \times \hat{W} \times \hat{C}}$ which is the output feature of one PromptDPM at l-level, the input of $k^{th}$ Transformer block can be obtained by gating the $X_{k-1}$ with $P_{l}$ by $1 \times 1$ convolution and gated control function sigmoid $\sigma(\cdot)$ with residual learning [39]:

$$
\mathcal {G} \left(\mathbf {X} _ {k - 1}, \mathbf {P} _ {l}\right) = \sigma \left(W _ {p} \mathbf {P} _ {l}\right) \odot \mathbf {X} _ {k - 1} + \mathbf {X} _ {k - 1}. \tag {7}
$$

# 3.4 Learning Strategy

To train the network, two objective loss functions are adopted, including image reconstruction loss $(\mathcal{L}_i)$ for pixel recovery and frequency loss $(\mathcal{L}_f)$ for detail enhancement [18]:

$$
\mathcal {L} = \mathcal {L} _ {i} + \lambda \mathcal {L} _ {f}, \text { where } \mathcal {L} _ {i} = \| \hat {\mathbf {H}} - \mathbf {H} \| _ {1}; \mathcal {L} _ {f} = \| \mathcal {F} (\hat {\mathbf {H}}) - \mathcal {F} (\mathbf {H}) \| _ {1}, \tag {8}
$$

where H denotes the ground truth image; F denotes the Fast Fourier transform; $\lambda$ is a weight that is empirically set to be 0.1.

# 4 Experiment

We evaluate PromptRestorer on benchmarks for 4 image restoration tasks: (a) deraining, (b) deblurring, (c) desnowing, and (d) dehazing. We train separate models for different image restoration tasks. Our PromptRestorer employs a 3-level encoder-decoder. From level-1 to level-3, the number of CGT is [2, 3, 6], attention heads are [2, 4, 8], and number of channels is [48, 96, 192]. The expanding channel capacity factor $\beta$ is 4. For downsampling and upsampling, we adopt pixel-unshuffle and pixel-shuffle [77], respectively. We train models with AdamW optimizer with the initial learning rate $3e^{-4}$ gradually reduced to $1e^{-6}$ with the cosine annealing [63]. The patch size is set as $256 \times 256$ .

Table 1: Image deraining results. Our PromptRestorer advances recent 14 state-of-the-arts on average. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Test100 [107]</td><td colspan="2">Rain100H [97]</td><td colspan="2">Rain100L [97]</td><td colspan="2">Test2800 [27]</td><td colspan="2">Test1200 [106]</td><td colspan="2">Average</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>DerainNet [26]</td><td>22.77</td><td>0.810</td><td>14.92</td><td>0.592</td><td>27.03</td><td>0.884</td><td>24.31</td><td>0.861</td><td>23.38</td><td>0.835</td><td>22.48</td><td>0.796</td></tr><tr><td>SEMI [95]</td><td>22.35</td><td>0.788</td><td>16.56</td><td>0.486</td><td>25.03</td><td>0.842</td><td>24.43</td><td>0.782</td><td>26.05</td><td>0.822</td><td>22.88</td><td>0.744</td></tr><tr><td>DIDMDN [106]</td><td>22.56</td><td>0.818</td><td>17.35</td><td>0.524</td><td>25.23</td><td>0.741</td><td>28.13</td><td>0.867</td><td>29.65</td><td>0.901</td><td>24.58</td><td>0.770</td></tr><tr><td>UMRL [99]</td><td>24.41</td><td>0.829</td><td>26.01</td><td>0.832</td><td>29.18</td><td>0.923</td><td>29.97</td><td>0.905</td><td>30.55</td><td>0.910</td><td>28.02</td><td>0.880</td></tr><tr><td>RESCAN [55]</td><td>25.00</td><td>0.835</td><td>26.36</td><td>0.786</td><td>29.80</td><td>0.881</td><td>31.29</td><td>0.904</td><td>30.51</td><td>0.882</td><td>28.59</td><td>0.857</td></tr><tr><td>PreNet [72]</td><td>24.81</td><td>0.851</td><td>26.77</td><td>0.858</td><td>32.44</td><td>0.950</td><td>31.75</td><td>0.916</td><td>31.36</td><td>0.911</td><td>29.42</td><td>0.897</td></tr><tr><td>MSPFN [44]</td><td>27.50</td><td>0.876</td><td>28.66</td><td>0.860</td><td>32.40</td><td>0.933</td><td>32.82</td><td>0.930</td><td>32.39</td><td>0.916</td><td>30.75</td><td>0.903</td></tr><tr><td>DCSFN [90]</td><td>27.46</td><td>0.887</td><td>28.98</td><td>0.887</td><td>34.70</td><td>0.961</td><td>30.96</td><td>0.903</td><td>32.92</td><td>0.937</td><td>31.00</td><td>0.915</td></tr><tr><td>MPRNet [103]</td><td>30.27</td><td>0.897</td><td>30.41</td><td>0.890</td><td>36.40</td><td>0.965</td><td>33.64</td><td>0.938</td><td>32.91</td><td>0.916</td><td>32.73</td><td>0.921</td></tr><tr><td>SPAIR [69]</td><td>30.35</td><td>0.909</td><td>30.95</td><td>0.892</td><td>36.93</td><td>0.969</td><td>33.34</td><td>0.936</td><td>33.04</td><td>0.922</td><td>32.91</td><td>0.926</td></tr><tr><td>Uformer [93]</td><td>29.17</td><td>0.880</td><td>30.06</td><td>0.884</td><td>36.34</td><td>0.966</td><td>33.36</td><td>0.935</td><td>31.98</td><td>0.909</td><td>32.18</td><td>0.915</td></tr><tr><td>MAXIM-2S [84]</td><td>31.17</td><td>0.922</td><td>30.81</td><td>0.903</td><td>38.06</td><td>0.977</td><td>33.80</td><td>0.943</td><td>32.37</td><td>0.922</td><td>33.24</td><td>0.933</td></tr><tr><td>Restormer [101]</td><td>32.00</td><td>0.923</td><td>31.46</td><td>0.904</td><td>38.99</td><td>0.978</td><td>34.18</td><td>0.944</td><td>33.19</td><td>0.926</td><td>33.96</td><td>0.935</td></tr><tr><td>SFNet [20]</td><td>31.47</td><td>0.919</td><td>31.90</td><td>0.908</td><td>38.21</td><td>0.974</td><td>33.69</td><td>0.937</td><td>32.55</td><td>0.911</td><td>33.56</td><td>0.929</td></tr><tr><td>PromptRestorer</td><td>31.84</td><td>0.920</td><td>31.72</td><td>0.908</td><td>39.04</td><td>0.977</td><td>34.40</td><td>0.947</td><td>33.27</td><td>0.928</td><td>34.05</td><td>0.936</td></tr></table>

![](images/337b1e9fb149cd3b4fa9267530e3adfa9f75ce5b46f2ae25ec4aa0d67e48f433.jpg)  
PSNR   
(a) Input

![](images/2f60f6cfdb81bfdc74f3c6a9488f8faa544ba7d31d73eea28c766354fae9868c.jpg)  
20.83 dB   
(b) RESCAN

![](images/d0077bbaae7e995fee069e3626ebd32447f81aef70add6c6c7bf9c4f72d83f0f.jpg)  
21.89 dB   
(c) DCSFN

![](images/72944d2e139bfcc87bf23346f421cd53a1f26850f2a3ca6a3c73febd9b4bb343.jpg)  
21.51 dB   
(d) MPRNet

![](images/35d58d328e2af56cb864f5875db1c0b32114d98dd4b9325cc3c9288bdcf4338a.jpg)  
22.41 dB   
(e) Restormer

![](images/00baba8706fee18a6e61c8fdce7c7bae2c31ac705b4c79f412737776a87bb167.jpg)  
22.46 dB   
(f) PromptRestorer

![](images/b101e60523b64f391ccfae082614686f72380aa42abed65a0a5ab8fa464318ba.jpg)  
∞   
(g) GT   
Figure 4: Image deraining example on Rain100H [97].

Table 2: Image deblurring results. Our PromptRestorer is trained only on the GoPro dataset [65] and directly applied to the HIDE [76] and RealBlur [73] benchmark datasets. 

<table><tr><td>Benchmark</td><td>Metrics</td><td>Nah et al. [65]</td><td>SRN [81]</td><td>DBGAN [109]</td><td>MT-RNN [68]</td><td>DMPHN [105]</td><td>Suin et al. [80]</td><td>SPAIR [69]</td><td>MIMO-UNet+ [18]</td><td>MPRNet [103]</td><td>Restormer [101]</td><td>PromptRestorer</td></tr><tr><td rowspan="2">GoPro [65]</td><td>PSNR ↑</td><td>21.00</td><td>30.26</td><td>31.10</td><td>31.15</td><td>31.20</td><td>31.85</td><td>32.06</td><td>32.45</td><td>32.66</td><td>32.92</td><td>33.06</td></tr><tr><td>SSIM ↑</td><td>0.914</td><td>0.934</td><td>0.942</td><td>0.945</td><td>0.940</td><td>0.948</td><td>0.953</td><td>0.957</td><td>0.959</td><td>0.961</td><td>0.962</td></tr><tr><td rowspan="2">HIDE [76]</td><td>PSNR ↑</td><td>25.73</td><td>28.36</td><td>28.94</td><td>29.15</td><td>29.09</td><td>29.98</td><td>30.29</td><td>29.99</td><td>30.96</td><td>31.22</td><td>31.36</td></tr><tr><td>SSIM ↑</td><td>0.874</td><td>0.915</td><td>0.915</td><td>0.918</td><td>0.924</td><td>0.930</td><td>0.931</td><td>0.930</td><td>0.939</td><td>0.942</td><td>0.944</td></tr><tr><td rowspan="2">RealBlur-R [73]</td><td>PSNR ↑</td><td>32.51</td><td>35.66</td><td>33.78</td><td>35.79</td><td>35.70</td><td>-</td><td>-</td><td>35.54</td><td>35.99</td><td>36.19</td><td>36.06</td></tr><tr><td>SSIM ↑</td><td>0.841</td><td>0.947</td><td>0.909</td><td>0.951</td><td>0.948</td><td>-</td><td>-</td><td>0.947</td><td>0.952</td><td>0.957</td><td>0.954</td></tr><tr><td rowspan="2">RealBlur-J [73]</td><td>PSNR ↑</td><td>27.87</td><td>28.56</td><td>24.93</td><td>28.44</td><td>28.42</td><td>-</td><td>28.81</td><td>27.63</td><td>28.70</td><td>28.96</td><td>28.82</td></tr><tr><td>SSIM ↑</td><td>0.827</td><td>0.867</td><td>0.745</td><td>0.862</td><td>0.860</td><td>-</td><td>0.875</td><td>0.837</td><td>0.873</td><td>0.879</td><td>0.873</td></tr></table>

![](images/ac00a160c9e4de5025e05f211f9172cad607fba16fb01c9cf75ba1e92dfcee9b.jpg)  
PSNR   
(a) Degraded Image (b) Degraded Patch

![](images/2fa4ba8d5f2fe62efa6287b57d0864fb7b23bceef3c79a3db46f160707459b12.jpg)  
24.83 dB   
(b) Degraded Patch

![](images/fd15238f46a2af9a4cd1c6c2e40a655d3a8d7390c0eecc6294437308f2c739b3.jpg)  
30.40 dB   
(c) Nah et al.

![](images/58f0cb66856722966c038dde0e593efc672cedff45f4a05afffbedc23a1f92ee.jpg)  
31.04 dB   
(d) SRN

![](images/4e9490102338f2c8f138840dfb2a9aacfd11a61a7eed0aa2e0cdbde7e2f0249c.jpg)  
31.07 dB   
(e) MPRNet

![](images/ef88ac584540ab52a7dc1322872fe58ebdee873d7cfc3d0ccf3f3189d9fa2d4a.jpg)  
31.68 dB   
(f) PromptRestorer

![](images/74414c4986b3c75ba94bbf89d7a3df8d3b51aaa304f9af6381e92dcc01ae9e4c.jpg)  
∞   
(g) GT   
Figure 5: Image deblurring example on GoPro [65].

# 4.1 Main Results

Image Deraining Results. Similar to existing methods $[44, 103, 69]$ , we report PSNR/SSIM scores using Y channel in YCbCr color. Tab. 1 shows that our PromptRestorer outperforms current state-of-the-art approaches when averaged across all five datasets. Compared to the recent best method Restormer $[101]$ , PromptRestorer achieves 0.09 dB improvement on average. On individual datasets, the gain can be as large as 0.22 dB, e.g., Test2800 $[27]$ . In Fig. 4, we present a challenging visual deraining example, where our PromptRestorer is able to generate a clearer result with finer details.

Image Deblurring Results. We evaluate deblurring results on both synthetic datasets (GoPro [65], HIDE [76]) and real-world datasets (RealBlur-R [73], RealBlur-J [73]). Tab. 2 summarises the results, where our PromptRestorer advances current state-of-the-art approaches on GoPro [65] and HIDE [76]. Compared with MPRNet [103], our PromptRestorer obtains a performance 0.12 dB gains. Fig. 5 provides a visual deblurring example. Our PromptRestorer produces a sharper result with fewer artifacts.

Table 3: Image dehazing results on SOTS-Indoor [52] and real-world benchmarks Dense-Haze [2] and NH-Haze [3]. Our PromptRestorer significantly advances state-of-the-arts on SOTS-Indoor [52]. 

<table><tr><td>Benchmark</td><td>Metrics</td><td>DCP [37]</td><td>DehazeNet [9]</td><td>AODNet [51]</td><td>GridNet [58]</td><td>FFANet [70]</td><td>MSBDN [21]</td><td>UHD [113]</td><td>MAXIM [84]</td><td>DeHamer [33]</td><td>PromptRestorer</td></tr><tr><td rowspan="2">SOTS-Indoor [52]</td><td>PSNR ↑</td><td>16.61</td><td>19.82</td><td>20.51</td><td>32.16</td><td>36.39</td><td>32.77</td><td>21.75</td><td>38.11</td><td>36.63</td><td>42.54</td></tr><tr><td>SSIM ↑</td><td>0.8546</td><td>0.8209</td><td>0.8162</td><td>0.9836</td><td>0.9886</td><td>0.9812</td><td>0.8786</td><td>0.9910</td><td>0.9881</td><td>0.9945</td></tr><tr><td rowspan="2">Dense-Haze [2]</td><td>PSNR ↑</td><td>11.01</td><td>9.48</td><td>12.82</td><td>14.96</td><td>12.22</td><td>15.13</td><td>12.16</td><td>-</td><td>16.62</td><td>15.86</td></tr><tr><td>SSIM ↑</td><td>0.4165</td><td>0.4383</td><td>0.4683</td><td>0.5326</td><td>0.4440</td><td>0.5551</td><td>0.4594</td><td>-</td><td>0.5602</td><td>0.5680</td></tr><tr><td rowspan="2">NH-Haze [3]</td><td>PSNR ↑</td><td>12.72</td><td>11.76</td><td>15.69</td><td>18.33</td><td>18.13</td><td>17.97</td><td>16.05</td><td>-</td><td>20.66</td><td>20.36</td></tr><tr><td>SSIM ↑</td><td>0.4419</td><td>0.3988</td><td>0.5728</td><td>0.6667</td><td>0.6473</td><td>0.6591</td><td>0.4612</td><td>-</td><td>0.6844</td><td>0.7203</td></tr></table>

![](images/7cd0989c01a7254fa34ad37127ed167811ec3c71184931c4eaeb110ee265c824.jpg)  
PSNR   
(a) Input

![](images/d9287227e894a570e4645193cf5a7b51ff0906d2f4dab11fd5038a34bda408d3.jpg)  
26.18 dB   
(b) GridNet

![](images/fefec5ef9dc5dd39c866fce92221283c2d5efa0088309004b7d4bf3b80fe9af1.jpg)  
29.14 dB   
(c) MSBDN

![](images/0db6294c35e344ad315cbfccbb95070472a5cbbded07f0de1615be06388d1c2e.jpg)  
15.52 dB   
(d) UHD

![](images/3ec9e4c888b6a058298dc2a13636ddd807b66a6ed7ceeb204c0b1fdf29c07c27.jpg)  
32.77 dB   
(e) DeHamer

![](images/510ae09bd199c942afcf501afe6a93abf4662448237e19b4dd994161fa4e95c7.jpg)  
36.86 dB   
(f) PromptRestorer

![](images/0c3f2d9669e2c55b39d4e669c559ce1e037310b7e1e6445eb16cd663f8e17f05.jpg)  
∞   
(g) GT   
Figure 6: Image dehazing example on SOTS-Indoor [52].

Table 4: Image desnowing results on CSD (2000) [65], SRRS (2000) [76], and Snow100K (2000) [73]. Our PromptRestorer achieves the best metrics on all datasets on the image desnowing problem. 

<table><tr><td>Benchmark</td><td>Metrics</td><td>DesnowNet [61]</td><td>JSTASR [14]</td><td>HDCW-Net [15]</td><td>TransWeather [85]</td><td>MSP-Former [13]</td><td>Uformer [94]</td><td>Restormer [101]</td><td>PromptRestorer</td></tr><tr><td rowspan="2">CSD (2000) [15]</td><td>PSNR ↑</td><td>20.13</td><td>27.96</td><td>29.06</td><td>31.76</td><td>33.75</td><td>33.80</td><td>35.43</td><td>37.48</td></tr><tr><td>SSIM ↑</td><td>0.81</td><td>0.88</td><td>0.91</td><td>0.93</td><td>0.96</td><td>0.96</td><td>0.97</td><td>0.99</td></tr><tr><td rowspan="2">SRRS (2000) [14]</td><td>PSNR ↑</td><td>20.38</td><td>25.82</td><td>27.78</td><td>28.29</td><td>30.76</td><td>30.12</td><td>32.24</td><td>33.99</td></tr><tr><td>SSIM ↑</td><td>0.84</td><td>0.89</td><td>0.92</td><td>0.92</td><td>0.95</td><td>0.96</td><td>0.96</td><td>0.99</td></tr><tr><td rowspan="2">Snow100K (2000) [61]</td><td>PSNR ↑</td><td>30.50</td><td>23.12</td><td>31.54</td><td>31.82</td><td>33.43</td><td>33.81</td><td>34.67</td><td>36.02</td></tr><tr><td>SSIM ↑</td><td>0.94</td><td>0.86</td><td>0.95</td><td>0.95</td><td>0.96</td><td>0.94</td><td>0.95</td><td>0.97</td></tr></table>

![](images/d895528e5627e693645b74538e4a1fdd2be4ff0406b8547a7fe443784cb9e346.jpg)  
PSNR   
(a) Input

![](images/2770f94962c25bea2882f56e5a16c9815788f7821a1c09dfdfd5fda896e84c26.jpg)  
24.18 dB   
(b) JSTASR

![](images/42d7abfb7933cecd3057a924e6cab19e7c381557b6a2c3e469c1852ae7017f26.jpg)  
26.81 dB   
(c) HDCWNet

![](images/4827cccdb33197bfb75e3efde258be29e3dc358834d4caf962aa41a4dd3f25f3.jpg)  
29.07 dB   
(d) Uformer

![](images/5ad9859c44c40c29677e008fb3c34aec1e277d7c1bc3bd7c74b8615083ec50d6.jpg)  
29.72 dB   
(e) Restormer

![](images/ec5f2326865d768b870cfaacd805c633809fad3dbd578e4bb5b98420b6ccd2c8.jpg)  
34.70 dB   
(f) PromptRestorer

![](images/a1dfe6964e0f00907a42d617ff5f95ea8c498bc398b0ad38c4b3817c0d5d64b8.jpg)  
∞   
(g) GT   
Figure 7: Image desnowing example on CSD (2000) [15].

Image Dehazing Results. We perform the image dehazing experiments on both synthetic benchmark RESIDE SOTS-Indoor $[52]$ , and real-world hazy benchmarks Dense-Haze $[2]$ and NH-Haze $[3]$ . Tab. 3 summarise the quantitative results. Compared to the recent works DeHamer $[33]$ and MAXIM $[84]$ , our method receives 4.33 dB and 5.91 dB PSNR gains on the SOTS-Indoor, respectively. On the real-world benchmark NH-Haze $[3]$ , our PromptRestorer can achieve 0.7203 of the SSIM result, which is a new record and significantly outperforms current state-of-the-art approaches DeHamer $[33]$ . The results on both synthetic and real-world benchmarks have demonstrated the effectiveness of our PromptRestorer on the image dehazing task. Fig. 6 shows the visual results, where our PromptRestorer is more effective in removing haze than other methods.

Image Desnowing Results. For the image desnowing task, we compare our PromptRestorer on the CSD $[15]$ , SRRS $[14]$ , and Snow100K $[61]$ datasets with existing state-of-the-art methods $[61, 14, 15, 13, 85]$ . We also compare recent Transformer-based general image restoration approaches Restormer $[101]$ and Uformer $[93]$ . As shown in Tab. 4, our PromptRestorer yields a 2.05 dB PSNR improvement over the state-of-the-art approach $[101]$ on the CSD benchmark $[15]$ . The visual results in Fig. 7 show that our PromptRestorer is able to remove spatially varying snowflakes than competitors.

Table 6: Ablation experiments on PromptDPM. Each component in L2P and G2P is effective.   
(a) Effect on L2P. Both Res-InBan and Deg-InBan play positive roles for image restoration. 

<table><tr><td>Experiment</td><td>PSNR</td><td>FLOPs (G)</td><td>Params (M)</td></tr><tr><td>w/o L2P</td><td>30.819</td><td>148.34</td><td>12.60</td></tr><tr><td>w/o Res-InBan</td><td>30.952</td><td>153.39</td><td>12.97</td></tr><tr><td>w/o Deg-InBan</td><td>30.964</td><td>153.39</td><td>12.97</td></tr><tr><td>Full (Ours)</td><td>31.015</td><td>157.04</td><td>13.24</td></tr></table>

(b) Effect on G2P. Both Q-InAtt and KV-InAtt play a positive effect on high-quality image restoration. 

<table><tr><td>Experiment</td><td>PSNR</td><td>FLOPs (G)</td><td>Params (M)</td></tr><tr><td>w/o G2P</td><td>30.697</td><td>123.44</td><td>10.69</td></tr><tr><td>w/o Q-InAtt</td><td>30.914</td><td>148.93</td><td>12.65</td></tr><tr><td>w/o KV-InAtt</td><td>30.906</td><td>147.26</td><td>12.52</td></tr><tr><td>Full (Ours)</td><td>31.015</td><td>157.04</td><td>13.24</td></tr></table>

# 4.2 Analysis and Discussion

For ablation experiments, following $[84, 20]$ , we train the image deblurring model on GoPro dataset $[65]$ for 1000 epochs only and set the number of Transformer in each CGT is 1. Params mean the number of learnable parameters. Testing is performed on the GoPro testing dataset $[65]$ . FLOPs are computed on image size $256 \times 256$ . Next, we describe the influence of each component individually.

Effect on Prompting. The core design of our PromptRestorer is the ‘prompting’, which exploits a pre-trained model to extract raw degradation features from the degraded observations and then generate perceived content to guide the restoration branch (i.e., Case 3 in Fig. 1).

Compared to existing frameworks such as Cases 1-2 in Fig. 1, our proposed prompting strategy shows superior performance, as demonstrated in Tab. 5. Our method achieved 0.877 dB gains compared to Case 1, and 0.369 dB higher than Case 2. Interestingly, the learnable condition branch in Case 2 $^{2}$ , despite consuming more FLOPs and Params, results in worse performance than ours. Our approach directly exploits raw degradation features to prompt restoration with persistent degradation priors to facilitate better recovery. Fig. 1(c) shows two examples, where our model that exploits raw degradation features as prompting generates sharper and clearer images.

Table 5: Effect on prompting. Our method that directly exploits the raw degradation to prompt restoration performs better. 

<table><tr><td>Case in Fig. 1</td><td>PSNR</td><td>FLOPs (G)</td><td>Params (M)</td></tr><tr><td>1</td><td>30.138</td><td>105.58</td><td>10.16</td></tr><tr><td>2</td><td>30.646</td><td>157.04</td><td>16.77</td></tr><tr><td>3 (Ours)</td><td>31.015</td><td>157.04</td><td>13.24</td></tr></table>

Effect on PromptDPM. We analyze the impact of PromptDPM on restoration quality in Tab. 6 by disabling one component at a time. Each model in L2P and G2P consumes similar Params and FLOPs, while our full model achieves the best performance. Disabling L2P or G2P results in a decrease in performance by 0.196 dB and 0.318 dB, respectively. These experiments conclusively demonstrate the effectiveness of each component in L2P and G2P for restoration.

Effect on GDP. To understand the impact of GDP, we disable it to compare with full model in Tab. 7.

Note that the computational cost of the GDP is negligible compared to disabling it as it only involves a $1 \times 1$ convolution and sigmoid function for the gating mechanism, while it leads to a gain of 0.091 dB. This finding highlights the significance of controlling the propagation of the perceived degradation features.

Table 7: Effect on GDP. Our GDP which controls the degradation propagation is effective. 

<table><tr><td>Experiment</td><td>PSNR</td><td>FLOPs (G)</td><td>Params (M)</td></tr><tr><td>w/o GDP</td><td>30.924</td><td>154.60</td><td>12.95</td></tr><tr><td>w/ GDP (Ours)</td><td>31.015</td><td>157.04</td><td>13.24</td></tr></table>

# 4.3 Visualization Understanding for Degradation Vanishing

To emphasize the understanding of degradation vanishing, we visualize the features learned in the condition/prompting branches to better comprehend the learned status of these branches in Fig. 8. Notably, both Cases 1-2 exhibit sharper results in later iterations compared to earlier ones, which fail to provide the restoration branch with sufficient degraded information and cause the restoration

models to not perceive the degradation well, thereby hindering the model capacity. In contrast, as the restoration branch needs to adapt perceived features from the PromptDPM which is to perceive the raw degradation features from inputs, our model (Case 3) initially exhibits inferior performance (around 20K iterations) as shown in Fig. 1(b). However, with better adaptation to the degradation information after more iterations, the prompting branch can better prompt the restoration branch consistently with more reliable perceived content learned from the raw degradation, enabling our restoration branch to overcome degradation vanishing and improve restoration quality, as shown in Fig. 1(c).

![](images/6b33577fc5d2b51556693fb619be3f8eab604d26381589bf575214759c40c1c3.jpg)

<details>
<summary>text_image</summary>

GT Features
Case 1
10K Iterations
500K Iterations
Case 2
10K Iterations
500K Iterations
Case 3
Consistent Degraded Features
</details>

Figure 8: Visualization. We show the average features over the channel dimension in the condition/prompting branches for the second example in Fig. 1(c). We obtain GT/degraded features by inputting GT/degraded images into the pre-trained VQGAN. As single-branch models (Case 1 in Fig. 1) do not have condition branches, we visualize the last layer in the 1-level encoder for reference.

# 5 Concluding Remarks

In this paper, we investigate the degradation vanishing in the learning process for image restoration. To solve this problem, we have proposed the PromptRestorer which explores the raw degradation features extracted by a pre-trained model from the given degraded observations to guide the restoration process to facilitate better recovery. Extensive experiments have demonstrated that our PromptRestorer favors against state-of-the-art approaches on 4 restoration tasks, including image deraining, deblurring, dehazing, and desnowing.

# References

[1] A. Abuolaim and M. S. Brown. Defocus deblurring using dual-pixel data. In ECCV, 2020.   
[2] C. O. Ancuti, C. Ancuti, M. Sbert, and R. Timofte. Dense-haze: A benchmark for image dehazing with dense-haze and haze-free images. In ICIP, pages 1014-1018, 2019.   
[3] C. O. Ancuti, C. Ancuti, and R. Timofte. Nh-haze: An image dehazing benchmark with non-homogeneous hazy and haze-free images. In CVPR workshops, pages 444–445, 2020.   
[4] S. Anwar and N. Barnes. Densely residual laplacian super-resolution. TPAMI, 2020.   
[5] S. Anwar, S. Khan, and N. Barnes. A deep journey into super-resolution: A survey. ACM Computing Surveys, 2019.   
[6] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv:1607.06450, 2016.   
[7] D. Berman, T. Treibitz, and S. Avidan. Non-local image dehazing. In CVPR, pages 1674-1682, 2016.   
[8] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. arXiv:2005.14165, 2020.   
[9] B. Cai, X. Xu, K. Jia, C. Qing, and D. Tao. Dehazenet: An end-to-end system for single image haze removal. IEEE TIP, 25(11):5187–5198, 2016.   
[10] H. Cai, J. He, Y. Qiao, and C. Dong. Toward interactive modulation for photo-realistic image restoration. In CVPR Workshops, pages 294-303, 2021.   
[11] C. Chen, X. Shi, Y. Qin, X. Li, X. Han, T. Yang, and S. Guo. Real-world blind super-resolution via feature matching with implicit high-resolution priors. In ACM MM, pages 1329-1338, 2022.   
[12] H. Chen, Y. Wang, T. Guo, C. Xu, Y. Deng, Z. Liu, S. Ma, C. Xu, C. Xu, and W. Gao. Pre-trained image processing transformer. In CVPR, 2021.

[13] S. Chen, T. Ye, Y. Liu, T. Liao, Y. Ye, and E. Chen. Msp-former: Multi-scale projection transformer for single image desnowing. arXiv preprint arXiv:2207.05621, 2022.   
[14] W.-T. Chen, H.-Y. Fang, J.-J. Ding, C.-C. Tsai, and S.-Y. Kuo. Jstasr: Joint size and transparency-aware snow removal algorithm based on modified partial convolution and veiling effect removal. In ECCV, pages 754–770, 2020.   
[15] W.-T. Chen, H.-Y. Fang, C.-L. Hsieh, C.-C. Tsai, I. Chen, J.-J. Ding, S.-Y. Kuo, et al. All snow removed: Single image desnowing algorithm using hierarchical dual-tree complex wavelet representation and contradict channel loss. In ICCV, pages 4196-4205, 2021.   
[16] X. Chen, Y. Liu, Z. Zhang, Y. Qiao, and C. Dong. Hdrunet: Single image hdr reconstruction with denoising and dequantization. In CVPR Workshops, pages 354-363, 2021.   
[17] X. Chen, Z. Zhang, J. S. Ren, L. Tian, Y. Qiao, and C. Dong. A new journey from sdrtv to hdrtv. In ICCV, pages 4500-4509, 2021.   
[18] S.-J. Cho, S.-W. Ji, J.-P. Hong, S.-W. Jung, and S.-J. Ko. Rethinking coarse-to-fine approach in single image deblurring. In ICCV, 2021.   
[19] X. Cui, C. Wang, D. Ren, Y. Chen, and P. Zhu. Semi-supervised image deraining using knowledge distillation. IEEE TCSVT, 32(12):8327–8341, 2022.   
[20] Y. Cui, Y. Tao, Z. Bing, W. Ren, X. Gao, X. Cao, K. Huang, and A. Knoll. Selective frequency network for image restoration. In ICLR, 2023.   
[21] H. Dong, J. Pan, L. Xiang, Z. Hu, X. Zhang, F. Wang, and M. Yang. Multi-scale boosted dehazing network with dense feature fusion. In CVPR, pages 2154-2164, 2020.   
[22] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.   
[23] Y. Du, F. Wei, Z. Zhang, M. Shi, Y. Gao, and G. Li. Learning to prompt for open-vocabulary object detection with vision-language model. In CVPR, pages 14064-14073, 2022.   
[24] A. Dudhane, S. W. Zamir, S. Khan, F. Khan, and M.-H. Yang. Burst image restoration and enhancement. In CVPR, 2022.   
[25] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873-12883, 2021.   
[26] X. Fu, J. Huang, X. Ding, Y. Liao, and J. Paisley. Clearing the skies: A deep network architecture for single-image rain removal. TIP, 2017.   
[27] X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding, and J. Paisley. Removing rain from single images via a deep detail network. In CVPR, 2017.   
[28] Y. Gan, X. Ma, Y. Lou, Y. Bai, R. Zhang, N. Shi, and L. Luo. Decorate the newcomers: Visual domain prompt for continual test time adaptation. In AAAI, 2023.   
[29] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, and Y. Qiao. Clip-adapter: Better vision-language models with feature adapters. arXiv preprint arXiv:2110.04544, 2021.   
[30] J. Gu, H. Lu, W. Zuo, and C. Dong. Blind super-resolution with iterative kernel correction. In CVPR, 2019.   
[31] S. Gu, Y. Li, L. V. Gool, and R. Timofte. Self-guided network for fast image denoising. In ICCV, 2019.   
[32] Y. Gu, X. Wang, L. Xie, C. Dong, G. Li, Y. Shan, and M. Cheng. VQFR: blind face restoration with vector-quantized dictionary and parallel decoder. In ECCV, pages 126–143, 2022.   
[33] C.-L. Guo, Q. Yan, S. Anwar, R. Cong, W. Ren, and C. Li. Image dehazing transformer with transmission-aware 3d position embedding. In CVPR, pages 5812-5820, 2022.   
[34] J. He, C. Dong, Y. Liu, and Y. Qiao. Interactive multi-dimension modulation for image restoration. TPAMI, 44(12):9363–9379, 2022.

[35] J. He, C. Dong, and Y. Qiao. Interactive multi-dimension modulation with dynamic controllable residual learning for image restoration. In A. Vedaldi, H. Bischof, T. Brox, and J. Frahm, editors, ECCV, volume 12365, pages 53–68, 2020.   
[36] J. He, Y. Liu, Y. Qiao, and C. Dong. Conditional sequential modulation for efficient global image retouching. In ECCV, volume 12358, pages 679–695. Springer, 2020.   
[37] K. He, J. Sun, and X. Tang. Single image haze removal using dark channel prior. TPAMI, 2010.   
[38] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.   
[39] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.   
[40] R. Herzig, O. Abramovich, E. Ben-Avraham, A. Arbelle, L. Karlinsky, A. Shamir, T. Darrell, and A. Globerson. Promptonomyvit: Multi-task prompt learning improves video transformers using synthetic scene data. CoRR, abs/2212.04821, 2022.   
[41] Z. Hu, S. Cho, J. Wang, and M.-H. Yang. Deblurring low-light images with light streaks. In CVPR, 2014.   
[42] J.-B. Huang, A. Singh, and N. Ahuja. Single image super-resolution from transformed self-exemplars. In CVPR, 2015.   
[43] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim. Visual prompt tuning. In ECCV, pages 709–727, 2022.   
[44] K. Jiang, Z. Wang, P. Yi, B. Huang, Y. Luo, J. Ma, and J. Jiang. Multi-scale progressive fusion network for single image deraining. In CVPR, 2020.   
[45] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah. Transformers in vision: A survey. arXiv:2101.01169, 2021.   
[46] M. U. Khattak, H. A. Rasheed, M. Maaz, S. Khan, and F. S. Khan. Maple: Multi-modal prompt learning. CoRR, abs/2210.03117, 2022.   
[47] J. Kopf, B. Neubert, B. Chen, M. Cohen, D. Cohen-Or, O. Deussen, M. Uyttendaele, and D. Lischinski. Deep photo: Model-based photograph enhancement and viewing. ACM TOG, 2008.   
[48] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.   
[49] M. Kumar, D. Weissenborn, and N. Kalchbrenner. Colorization transformer. In ICLR, 2021.   
[50] O. Kupyn, T. Martyniuk, J. Wu, and Z. Wang. DeblurGAN-v2: Deblurring (orders-of-magnitude) faster and better. In ICCV, 2019.   
[51] B. Li, X. Peng, Z. Wang, J. Xu, and D. Feng. Aod-net: All-in-one dehazing network. In ICCV, pages 4780-4788, 2017.   
[52] B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang. Benchmarking single-image dehazing and beyond. TIP, 28(1):492–505, 2019.   
[53] M. Li, L. Chen, Y. Duan, Z. Hu, J. Feng, J. Zhou, and J. Lu. Bridge-prompt: Towards ordinal action understanding in instructional videos. In CVPR, pages 19880–19889, 2022.   
[54] S. Li, I. B. Araujo, W. Ren, Z. Wang, E. K. Tokuda, R. H. Junior, R. Cesar-Junior, J. Zhang, X. Guo, and X. Cao. Single image deraining: A comprehensive benchmark analysis. In CVPR, 2019.   
[55] X. Li, J. Wu, Z. Lin, H. Liu, and H. Zha. Recurrent squeeze-and-excitation context aggregation net for single image deraining. In ECCV, 2018.   
[56] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte. SwinIR: Image restoration using swin transformer. In ICCV Workshops, 2021.   
[57] X. Liu, J. Hu, X. Chen, and C. Dong. Udc-unet: Under-display camera image restoration via u-shape dynamic network. In L. Karlinsky, T. Michaeli, and K. Nishino, editors, ECCV Workshops, volume 13805, pages 113–129, 2022.   
[58] X. Liu, Y. Ma, Z. Shi, and J. Chen. Griddehazenet: Attention-based multi-scale network for image dehazing. In ICCV, pages 7313-7322, 2019.

[59] X. Liu, M. Suganuma, Z. Sun, and T. Okatani. Dual residual networks leveraging the potential of paired operations for image restoration. In CVPR, 2019.   
[60] Y. Liu, J. He, X. Chen, Z. Zhang, H. Zhao, C. Dong, and Y. Qiao. Very lightweight photo retouching network with conditional sequential modulation. TMM, 2022.   
[61] Y.-F. Liu, D.-W. Jaw, S.-C. Huang, and J.-N. Hwang. Desnownet: Context-aware deep network for snow removal. TIP, 27(6):3064–3073, 2018.   
[62] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In CVPR, pages 11976–11986, 2022.   
[63] I. Loshchilov and F. Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017.   
[64] T. Michaeli and M. Irani. Nonparametric blind super-resolution. In ICCV, 2013.   
[65] S. Nah, T. Hyun Kim, and K. Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In CVPR, 2017.   
[66] J. Pan, Z. Hu, Z. Su, and M.-H. Yang. $l_{0}$ -regularized intensity and gradient prior for deblurring text images and beyond. TPAMI, 39(2):342–355, 2017.   
[67] J. Pan, D. Sun, H. Pfister, and M.-H. Yang. Blind image deblurring using dark channel prior. In CVPR, 2016.   
[68] D. Park, D. U. Kang, J. Kim, and S. Y. Chun. Multi-temporal recurrent neural networks for progressive non-uniform single image deblurring with incremental temporal training. In ECCV, 2020.   
[69] K. Purohit, M. Suin, A. Rajagopalan, and V. N. Boddeti. Spatially-adaptive image restoration using distortion-guided networks. In ICCV, 2021.   
[70] X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia. Ffa-net: Feature fusion attention network for single image dehazing. In AAAI, volume 34, pages 11908–11915, 2020.   
[71] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.   
[72] D. Ren, W. Zuo, Q. Hu, P. Zhu, and D. Meng. Progressive image deraining networks: A better and simpler baseline. In CVPR, 2019.   
[73] J. Rim, H. Lee, J. Won, and S. Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In ECCV, 2020.   
[74] O. Ronneberger, P. Fischer, and T. Brox. U-Net: convolutional networks for biomedical image segmentation. In MICCAI, 2015.   
[75] T. Schick and H. Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In EACL, pages 255–269, 2021.   
[76] Z. Shen, W. Wang, X. Lu, J. Shen, H. Ling, T. Xu, and L. Shao. Human-aware motion deblurring. In ICCV, 2019.   
[77] W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In CVPR, 2016.   
[78] T. Shin, Y. Razeghi, R. L. L. IV, E. Wallace, and S. Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In EMNLP, pages 4222-4235, 2020.   
[79] H. Singh, A. Kumar, L. K. Balyan, and G. K. Singh. A novel optimally gamma corrected intensity span maximization approach for dark image enhancement. In DSP, pages 1-5, 2017.   
[80] M. Suin, K. Purohit, and A. N. Rajagopalan. Spatially-attentive patch-hierarchical network for adaptive motion deblurring. In CVPR, 2020.   
[81] X. Tao, H. Gao, X. Shen, J. Wang, and J. Jia. Scale-recurrent network for deep image deblurring. In CVPR, 2018.

[82] C. Tian, L. Fei, W. Zheng, Y. Xu, W. Zuo, and C.-W. Lin. Deep learning on image denoising: An overview. Neural Networks, 2020.   
[83] R. Timofte, V. De Smet, and L. Van Gool. Anchored neighborhood regression for fast example-based super-resolution. In ICCV, 2013.   
[84] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li. Maxim: Multi-axis mlp for image processing. In CVPR, pages 5769-5780, 2022.   
[85] J. M. J. Valanarasu, R. Yasarla, and V. M. Patel. Transweather: Transformer-based restoration of images degraded by adverse weather conditions. In CVPR, pages 2353-2363, 2022.   
[86] C. Wang, J. Pan, W. Lin, J. Dong, and X.-M. Wu. Selfpromer: Self-prompt dehazing transformers with depth-consistency. arXiv preprint arXiv:2303.07033, 2023.   
[87] C. Wang, J. Pan, and X. Wu. Online-updated high-order collaborative networks for single image deraining. In AAAI, pages 2406–2413, 2022.   
[88] C. Wang, Y. Wu, Z. Su, and J. Chen. Joint self-attention and scale-aggregation for self-calibrated deraining network. In ACM MM, pages 2517-2525, 2020.   
[89] C. Wang, X. Xing, Y. Wu, Z. Su, and J. Chen. DCSFN: deep cross-scale fusion network for single image rain removal. In ACM MM, pages 1643–1651. ACM, 2020.   
[90] C. Wang, X. Xing, Y. Wu, Z. Su, and J. Chen. DCSFN: deep cross-scale fusion network for single image rain removal. In ACM MM, pages 1643-1651, 2020.   
[91] C. Wang, H. Zhu, W. Fan, X. Wu, and J. Chen. Single image rain removal using recurrent scale-guide networks. Neurocomputing, 467:242–255, 2022.   
[92] X. Wang, K. Yu, C. Dong, and C. C. Loy. Recovering realistic texture in image super-resolution by deep spatial feature transform. In CVPR, 2018.   
[93] Z. Wang, X. Cun, J. Bao, and J. Liu. Uformer: A general u-shaped transformer for image restoration. arXiv:2106.03106, 2021.   
[94] Z. Wang, X. Cun, J. Bao, W. Zhou, J. Liu, and H. Li. Uformer: A general u-shaped transformer for image restoration. In CVPR, pages 17683–17693, 2022.   
[95] W. Wei, D. Meng, Q. Zhao, Z. Xu, and Y. Wu. Semi-supervised transfer learning for image rain removal. In CVPR, 2019.   
[96] F. Yang, H. Yang, J. Fu, H. Lu, and B. Guo. Learning texture transformer network for image super-resolution. In CVPR, 2020.   
[97] W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan. Deep joint rain detection and removal from a single image. In CVPR, 2017.   
[98] Y. Yao, A. Zhang, Z. Zhang, Z. Liu, T. Chua, and M. Sun. CPT: colorful prompt tuning for pre-trained vision-language models. CoRR, abs/2109.11797, 2021.   
[99] R. Yasarla and V. M. Patel. Uncertainty guided multi-scale residual learning-using a cycle spinning cnn for single image de-raining. In CVPR, 2019.   
[100] Z. Yue, Q. Zhao, L. Zhang, and D. Meng. Dual adversarial network: Toward real-world noise removal and noise generation. In ECCV, 2020.   
[101] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, pages 5718-5729, 2022.   
[102] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao. Learning enriched features for real image restoration and enhancement. In ECCV, 2020.   
[103] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao. Multi-stage progressive image restoration. In CVPR, 2021.   
[104] Y. Zang, W. Li, K. Zhou, C. Huang, and C. C. Loy. Unified vision and language prompt learning. CoRR, abs/2210.07225, 2022.

[105] H. Zhang, Y. Dai, H. Li, and P. Koniusz. Deep stacked hierarchical multi-patch network for image deblurring. In CVPR, 2019.   
[106] H. Zhang and V. M. Patel. Density-aware single image de-raining using a multi-stream dense network. In CVPR, 2018.   
[107] H. Zhang, V. Sindagi, and V. M. Patel. Image de-raining using a conditional generative adversarial network. TCSVT, 2019.   
[108] K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte. Plug-and-play image restoration with deep denoiser prior. TPAMI, 2021.   
[109] K. Zhang, W. Luo, Y. Zhong, L. Ma, B. Stenger, W. Liu, and H. Li. Deblurring by realistic blurring. In CVPR, 2020.   
[110] Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu. Image super-resolution using very deep residual channel attention networks. In ECCV, 2018.   
[111] Y. Zhang, K. Li, K. Li, B. Zhong, and Y. Fu. Residual non-local attention networks for image restoration. In ICLR, 2019.   
[112] Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu. Residual dense network for image restoration. TPAMI, 2020.   
[113] Z. Zheng, W. Ren, X. Cao, X. Hu, T. Wang, F. Song, and X. Jia. Ultra-high-definition image dehazing via multi-guided bilateral learning. In CVPR, pages 16185–16194, 2021.   
[114] Z. Zheng, X. Yue, K. Wang, and Y. You. Prompt vision transformer for domain generalization. CoRR, abs/2208.08914, 2022.   
[115] K. Zhou, J. Yang, C. C. Loy, and Z. Liu. Conditional prompt learning for vision-language models. In CVPR, pages 16795-16804, 2022.   
[116] S. Zhou, K. C. K. Chan, C. Li, and C. C. Loy. Towards robust blind face restoration with codebook lookup transformer. In NeurIPS, 2022.   
[117] H. Zhu, C. Wang, Y. Zhang, Z. Su, and G. Zhao. Physical model guided deep image deraining. In ICME, 2020.