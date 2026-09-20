# Hierarchical Alignment-enhanced Adaptive Grounding Network for Generalized Referring Expression Comprehension

Yaxian Wang $^{1,2}$ , Henghui Ding $^{3*}$ , Shuting He $^{4}$ , Xudong Jiang $^{5}$ , Bifan Wei $^{6,7*}$ , Jun Liu $^{1,2}$

$^{1}$ School of Computer Science and Technology, Xi'an Jiaotong University, China

$^{2}$ Ministry of Education Key Laboratory of Intelligent Networks and Network Security, Xi'an Jiaotong University, China

$^{3}$ Institute of Big Data, Fudan University, China

$^{4}$ Shanghai University of Finance and Economics, China

$^{5}$ Nanyang Technological University, Singapore

$^{6}$ School of Continuing Education, Xi'an Jiaotong University, China

$^{7}$ Shaanxi Province Key Laboratory of Big Data Knowledge Engineering, Xi'an Jiaotong University, China

wyx1566@stu.xjtu.edu.cn, henghui.ding@gmail.com, shutting.he@sufe.edu.cn, {weibifan,liukeen}@xjtu.edu.cn

# Abstract

In this work, we address the challenging task of Generalized Referring Expression Comprehension (GREC). Compared to the classic Referring Expression Comprehension (REC) that focuses on single-target expressions, GREC extends the scope to a more practical setting by further encompassing no-target and multi-target expressions. Existing REC methods face challenges in handling the complex cases encountered in GREC, primarily due to their fixed output and limitations in multi-modal representations. To address these issues, we propose a Hierarchical Alignment-enhanced Adaptive Grounding Network (HieA2G) for GREC, which can flexibly deal with various types of referring expressions. First, a Hierarchical Multi-modal Semantic Alignment (HMSA) module is proposed to incorporate three levels of alignments, including word-object, phrase-object, and text-image alignment. It enables hierarchical cross-modal interactions across multiple levels to achieve comprehensive and robust multi-modal understanding, greatly enhancing grounding ability for complex cases. Then, to address the varying number of target objects in GREC, we introduce an Adaptive Grounding Counter (AGC) to dynamically determine the number of output targets. Additionally, an auxiliary contrastive loss is employed in AGC to enhance object-counting ability by pulling in multi-modal features with the same counting and pushing away those with different counting. Extensive experimental results show that HieA2G achieves new state-of-the-art performance on the challenging GREC task and also the other 4 tasks, including REC, Phrase Grounding, Referring Expression Segmentation (RES), and Generalized Referring Expression Segmentation (GRES), demonstrating the remarkable superiority and generalizability of the proposed HieA2G.

# Introduction

Generalized Referring Expression Comprehension (GREC) (He et al. 2023; Liu, Ding, and Jiang 2023; Wu et al. 2024a) aims to detect an arbitrary number of target objects based on a given free-form text expression. In contrast to the classic Referring Expression Comprehension (REC) (Mao et al.

Image   
![](images/60ac9ecd0821d6519513141289903666df84df2879b3031ef0bce9dbdc5d455c.jpg)

![](images/c936d5ea33564aab785a8efbf138f66794941c500c4ff67ef156786405135944.jpg)  
the player in blue   
(a) REC

![](images/41ac7b206fbe4c987fb23e15aca1951101e683b4ce0c5816cb47c4f364b25ba3.jpg)  
A player in black pants runs towards the ball.   
(b) Phrase Grounding

No-Target   
![](images/a7bb4107974a955db9975e8712d3f8ec5388800a180640961c18b315764dacbd.jpg)  
the player in blue top and black pants

Single-Target   
![](images/9872d19d0febd8036d271f127a536fd007841c1788f33aed835dc0a48a418617.jpg)  
the player in blue

Multi-Target   
![](images/0fa760a21937ee8ab1ff8494c20df7410d35f8c7d097ea7fbc9362cd684ec6ce.jpg)  
all people   
(c) GREC   
Figure 1: Different visual grounding tasks. (a) Classic REC: text expressions can only specify a single object; (b) Phrase grounding detects all objects mentioned in expressions; (c) GREC (He et al. 2023; Liu, Ding, and Jiang 2023) supports the text expressions indicating an arbitrary number of target objects from 0 to multiple, which is a more challenging task.

2016; Yu et al. 2016) that only supports the single-target text expressions, GREC narrows the gap with real-world scenarios by further encompassing no-target expressions that do not match any object in the image, and multi-target expressions that refer to multiple target objects. This task has great potential value for various practical applications such as visual-language navigation, embodied AI, and human-robot interaction.

Although recent methods (Zhu et al. 2022; Deng et al. 2023) have achieved remarkable performance in REC, they are constrained to predict only one target that is most related to the text expression, leading to the inability to deal with no-target and multi-target expressions. As shown in Figure 1, the text expression “the player in blue top and black pants” does not match any object within the image. In this case, the traditional REC models (Kamath et al. 2021; Yan et al. 2023; Li and Sigal 2021; Luo et al. 2020) still produce a

false-negative bounding box. When given the multi-target text expression “all people”, existing REC models also fail to locate all matched targets in the image, arising from the fact that they are enforced to locate only a single target most related to the text expression. Despite phrase grounding (Plummer et al. 2015; Yu et al. 2020) can locate multiple objects, it tends to locate all objects based on the key noun phrases in the text expression without comprehending the entire text semantics. GREC is a more challenging task that requires a comprehensive understanding of the intricate semantics of text expressions and visual contents to handle any quantity of target objects ranging from zero to multiple. Therefore, it is necessary to advance a robust GREC model to adapt to this kind of complex generalized scenario.

The first challenge of GREC lies in how to effectively align the diverse text expressions with the corresponding images for comprehensive multi-modal understanding. Existing methods (Kamath et al. 2021; Radford et al. 2021; Liu, Jiang, and Ding 2024; Xu et al. 2022; Liu et al. 2022) have made significant efforts to alleviate the cross-modal semantic gap. Nevertheless, they tend to rely solely on single-level alignment, either word-object or text-image alignment, leading to insufficient vision-language interaction and further inhibiting the effective learning of exhaustive multi-modal information. For one image, diverse and flexible text expressions can specify different numbers of target objects from various perspectives as shown in Figure 1, highlighting the significance of multi-level cross-modal interactions. For example, for a no-target case, the model is often required to capture the fine-grained attribute details of local objects and have a comprehensive understanding of the global contextual information, further rejecting providing any object response. Therefore, it is far from satisfactory to handle the complex cases in GREC leveraging the single-level alignment between the flexible text expressions and images.

The second challenge of GREC is how to output different numbers of target objects dynamically for each specific image-text pair. Given a complex text expression specifying multiple targets, a potential approach is to split the text expression into multiple text expressions and query the model multiple rounds to obtain the target objects one by one. However, text expressions with implicit multi-target information are difficult to decompose, and such an approach can not solve the inherent requirement in GREC, which desires an efficient model to give all targets in a single forward process. More importantly, multi-target expressions such as “three players” and “all people” necessitate a model to possess an explicit or implicit object-counting ability. Although a threshold-based strategy (He et al. 2023) has demonstrated its advantages in selecting the target objects from multiple candidate object proposals, it is often challenging to decide an appropriate threshold for all samples. Moreover, using a unified threshold struggles to adapt to the characteristics of different samples, resulting in inaccurate prediction results for some samples. Therefore, it is crucial to design a more advanced strategy for selecting target objects.

To address the above challenges, we propose HieA2G, a Hierarchical Alignment-enhanced Adaptive Grounding Network, for GREC to deal with various types of referring expressions flexibly. Specifically, we design a Hierarchical Multi-modal Semantic Alignment (HMSA) module to achieve comprehensive and robust multi-modal understanding by coupling three levels of alignments including word-object, phrase-object, and text-image alignment. Due to the absence of fine-grained region-level annotations corresponding to the entity words, we propose a text mask recovery auxiliary task to reconstruct the masked text semantics with the visual object features to promote word-object alignment. In this way, the visual features are facilitated to fuse the semantic information of the masked entity fully. Compared with the entity word, the attribute-related information is essential to distinguish the object of the same category. By matching the descriptive phrase with the visual object, the encoder can derive distinctive visual features and understand a larger range of semantic units. After that, the high-level text-image alignment matches the overall semantics between the text and image, enabling a more comprehensive perception of global information. On the one hand, the HMSA module can help provide holistic and robust multi-modal understanding to facilitate more accurate object localization. On the other hand, it endows the model with the ability to exploit information at various levels of detail, which allows it to accomplish other various tasks like REC and phrase grounding. Furthermore, to address the varying number of target objects in GREC, we design an Adaptive Grounding Counter (AGC) to dynamically determine the number of output target objects for each specific image-text pair. Additionally, an auxiliary contrastive loss is employed in AGC to enhance the object-counting ability by pulling together the multi-modal features with the same counting and pushing away those with different counting.

Our contributions are summarized as follows: (1) We propose a Hierarchical Alignment-enhanced Adaptive Grounding Network (HieA2G) for GREC to support text expressions indicating an arbitrary number of target objects. (2) We design a Hierarchical Multi-modal Semantic Alignment module to enable hierarchical vision-language interactions across multiple levels for comprehensive and robust multimodal semantic understanding. (3) We propose an Adaptive Grounding Counter to dynamically determine the number of output targets for each specific image-text pair, which can help deal with the multi/single/no-target text expressions flexibly. (4) Extensive experimental results show that HieA2G achieves new SOTA results on the challenging GREC task. It also exhibits superior performance across the other four visual grounding tasks including REC, Phrase Grounding, Referring Expression Segmentation (RES), and Generalized Referring Expression Segmentation (GRES).

# Related Work

Referring Expression Comprehension. REC aims to detect one specific object from an image based on a referring expression. Existing methods can be classified into two groups: two-stage (Hu et al. 2017; Zhuang et al. 2018; Yang, Li, and Yu 2019; Liu et al. 2019a; Li, Bu, and Cai 2021) and one-stage (Liao et al. 2020; Zhou et al. 2021; Yang et al. 2022a; Deng et al. 2023; Ye et al. 2021) methods. The recent advancements in large language models (Wu et al.

![](images/da4bc6deb66c7e35e0c27b45ef2086b2600a7278fc37a2e5be7294b007f42fbf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Visual Encoder"] --> B["Vi"]
    B --> C["Fusion"]
    C --> D["Transformer Encoder"]
    D --> E["Transformer Decoder"]
    E --> F["Box Head"]
    F --> G["Mask Head"]
    G --> H["AGC"]
    H --> I["Tw"]
    I --> J["Lw2o"]
    K["Text Tokens"] --> L["Text Encoder"]
    M["Noun Phrases"] --> N["Text Encoder"]
    O["Masking"] --> P["Text Image Alignment"]
    Q["Word-Object Alignment"] --> R["Phrase-Object Alignment"]
    S["Text Tokens"] --> T["Text Image Alignment"]
    U["Text Images"] --> V["Text Images"]
    W["P"] --> C
    X["P"] --> D
    Y["M"] --> E
    Z["P"] --> K
    AA["P"] --> M
    AB["P"] --> N
    AC["P"] --> O
    AD["P"] --> Q
    AE["P"] --> S
```
</details>

Figure 2: The framework of our proposed HieA2G. First, the visual encoder and the text encoder extract the visual feature $V_{I}$ and text feature $T_{w}$ . Then, a Transformer encoder is employed to perform multi-modal feature interaction further. The learnable object queries and the output of the Transformer encoder are fed to the Transformer decoder, whose output is object embeddings $O_{e}$ corresponding to the object queries. Next, based on $O_{e}$ , the Hierarchical Multi-modal Semantic Alignment (HMSA) module is employed to facilitate multi-level cross-modal interaction via word-object, phrase-object, and text-image alignment. Moreover, an Adaptive Grounding Counter (AGC) is utilized to decide the output number of target objects dynamically.

2024d; Chen et al. 2023; You et al. 2024) have also brought new opportunities to vision-language tasks requiring localization like REC. They have achieved promising results by collecting large-scale datasets, pretraining, and fine-tuning the large language models. It's worth noting that our work focuses on GREC, which detects an arbitrary number of target objects. Therefore, the top-1 selection method for REC cannot be applied directly in this setting. Although GREC (He et al. 2023) has modified some REC methods (Luo et al. 2020; Kamath et al. 2021; Ding et al. 2023c; Yan et al. 2023) to output different numbers of bounding boxes, they still struggle to deal with such complex and flexible referring expressions, leading to unsatisfactory performance.

Referring Expression Segmentation. RES aims to segment one object based on text expression (Liu, Jiang, and Ding 2022). Driven by the success of Transformer, recent works (Ding et al. 2021, 2023a,b,c; He and Ding 2024; He et al. 2024; Meng et al. 2022; Li et al. 2024; Wu et al. 2024b,c; Liu et al. 2023; Liu, Li, and Ding 2024; Li and Sigal 2021; Kim et al. 2022; Yang et al. 2022b; Wang et al. 2022) have extensively use it to extract visual and language features. ReLA (Liu, Ding, and Jiang 2023) introduces the Generalized Referring Expression Segmentation (GRES) benchmark, which further include multi-target and no-target samples. They only study region-language and region-image relationships. In contrast, we propose a hierarchical multimodal alignment to enhance the comprehensive understanding of visual-linguistic context. Moreover, rather than a binary classification to judge only the existence of objects, we design an adaptive object-counting strategy to facilitate robust object perception.

# Methodology

# Architecture Overview

Figure 2 shows the overall architecture of our proposed HieA2G. First, the text expression T is fed into the text encoder to obtain the word features $T_{w} = \{w_{k}|k \in \{1,2,\ldots,K\}\}$ , where K is the number of words. For the phrase feature, we first extract noun phrases from the text expression, which are then represented as $T_{p} = \{p_{i}|i \in \{1,2,\ldots,M\}\}$ by average pooling the word features in each phrase, where M is the number of phrases. For the image input I, we adopt a visual encoder to obtain the visual feature $V_{I}$ and flatten it into a 2D feature combined with positional embeddings. The image feature and text feature are projected into the same space, and then are fused by concatenation to feed to the Transformer encoder for multi-modal deep fusion. Next, the output of the Transformer encoder and the learnable object queries are fed into the Transformer decoder. Subsequently, we obtain the text-aware object embeddings $O_{e} = \{o_{j}|j \in \{1,2,\ldots,N\}\}$ corresponding to the N object queries. Based on the object embeddings $O_{e}$ , the Hierarchical Multi-modal Semantic Alignment (HMSA) module is performed to hierarchically incorporate the multimodal information across multiple levels for a comprehensive multi-modal understanding. Furthermore, we propose an Adaptive Grounding Counter (AGC) to determine the output number of target objects dynamically and then select the desired outputs from candidate object proposals.

# Hierarchical Multi-modal Semantic Alignment

To fully model the relationships between the various types of text expression and the images, we propose a Hierarchical Multi-modal Semantic Alignment (HMSA) to facilitate multi-level cross-modal interactions through word-object, phrase-object, and text-image alignment. HMSA can help exploit information at various levels of detail and promote comprehensive and robust text-aware object embeddings for better box regression and mask segmentation.

Word-Object Alignment. To enable a more directly fine-grained word-object alignment, we introduce a masked text recovery task by enforcing the model to recover the missed

key information in the text based on the matched object features. The object embeddings are then facilitated to fuse the semantic information of the masked entity fully. Specifically, we first extract the entity noun in the text and randomly mask it with a [MASK] token. The masked text is encoded by the text encoder as $T_{w}^{*}$ . Then, combining the object embeddings $O_{e}$ and the masked text feature $T_{w}^{*}$ , a Transformer layer is utilized to reconstruct the text semantic feature $\hat{T}_{w}$ . We design a masked text recovery loss $L_{w2o}$ by measuring the semantic similarity between the original complete text feature $T_{w}$ and the reconstructed text feature $\hat{T}_{w}$ as:

$$
\mathcal {L} _ {w 2 o} = \alpha (1 - \cos (T _ {w}, \hat {T} _ {w})), \tag {1}
$$

where $\alpha$ is set to 0 when given a no-target sample, otherwise set to 1. Due to the weak relevance and even total irrelevance between the textual and the visual features for no-target samples, it is meaningless and impossible to reconstruct the missing information for this kind of image-text pair, even interfering with model optimization.

Phrase-Object Alignment. With the explicit phrase-object annotations in the Flickr30K Entities (Plummer et al. 2015) dataset, it is promising to encourage the model to match each phrase with the corresponding object query. By matching the descriptive phrase with the visual object, we can derive distinctive object features and understand a larger range of semantic units. Specifically, we first project the phrase features $T_{p} \in \mathbb{R}^{M \times C_{p}}$ and the object embeddings $\mathcal{O}_{e} \in \mathbb{R}^{N \times C_{v}}$ into the same sub-space by linear layers:

$$
\widehat {T} _ {p} = W _ {1} T _ {p}, \quad \widehat {\mathcal {O}} _ {e} = W _ {2} \mathcal {O} _ {e}, \tag {2}
$$

where $W_{1}$ and $W_{2}$ are learnable parameters, $\widehat{T}_{p} \in R^{M \times C}$ and $\widehat{O}_{e} \in R^{N \times C}$ are the projected phrase features and projected object embeddings, M and N is the number of phrases and object queries, respectively. The matching relation map $S \in R^{M \times N}$ between all noun phrases and the object queries is calculated as follows:

$$
S = \text { Sigmoid } (\widehat {T} _ {p} \cdot \widehat {\mathcal {O}} _ {e} ^ {\top}). \tag {3}
$$

For all object queries, we adopt a bipartite matching (Cheng et al. 2022) to find the matched ground-truth bounding box. Then, we can obtain a ground-truth binary map $Y \in R^{M \times N}$ between phrases and object queries, indicating their matching relationships. With the predicted matching relation map $S \in R^{M \times N}$ , we design a phrase-object contrastive loss $L_{p2o}$ , implemented with the binary cross-entropy loss:

$$
\mathcal {L} _ {p 2 o} = - \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {N} Y _ {i, j} \log S _ {i, j} + (1 - Y _ {i, j}) \log (1 - S _ {i, j}). \tag {4}
$$

In this way, the model is encouraged to generate higher scores for positive phrase-object alignments and lower scores for negative ones. Therefore, the object embeddings can be endowed with stronger discriminative ability by capturing fine-grained attributes within the phrase semantics. Text-Image Alignment. The image feature should have high feature similarity with the matched text expression and low feature similarity with the unmatched text expression. To fully model the global relationships between the image I and text expression T, we define a global match score $S^{T}(I,T)$ for each image-text pair to calculate their similarity via word-object pairs as follows:

$$
S ^ {T} (I, T) = \frac {1}{N} \sum_ {j = 1} ^ {N} \sum_ {k = 1} ^ {K} a _ {j, k} \langle \hat {o} _ {j}, \hat {w} _ {k} \rangle , \tag {5}
$$

$$
a _ {j, k} = \frac {\exp \langle \hat {o} _ {j} , \hat {w} _ {k} \rangle}{\sum_ {l = 1} ^ {K} \exp \langle \hat {o} _ {j} , \hat {w} _ {l} \rangle}, \tag {6}
$$

where $\langle .,.\rangle$ represents the dot product operation of two embeddings, $\hat{o}_j$ and $\hat{w}_k$ denotes the projected embedding of $j$ -th object query and $k$ -th word, and $S^T(I,T)$ is computed by normalizing along the text dimension. Similarly, $S^I(I,T)$ can be obtained by normalizing along the image dimension.

The global match score $S^{T}(I,T)$ measures the degree of semantic alignment between an image and its corresponding text expression. In this case, maximizing the match score of matched image-text pairs helps ensure their strong correspondence. For an image-text pair in a batch, the objective function is defined as follows:

$$
\mathcal {L} _ {t 2 i} ^ {T T} (I) = - \log \frac {\exp (S ^ {T} (I , T))}{\sum_ {T ^ {\prime} \in \mathcal {B} _ {T}} \exp (S ^ {T} (I , T ^ {\prime}))}, \tag {7}
$$

$$
\mathcal {L} _ {t 2 i} ^ {T I} (T) = - \log \frac {\exp (S ^ {T} (I , T))}{\sum_ {I ^ {\prime} \in \mathcal {B} _ {I}} \exp (S ^ {T} (I ^ {\prime} , T))}, \tag {8}
$$

where $B_{T}$ , $B_{I}$ represents a collection of the text expressions and images in a batch, $\mathcal{L}_{t2i}^{TT}(I)$ and $\mathcal{L}_{t2i}^{TI}(T)$ are normalized along text and image dimension, respectively. Similarly, $\mathcal{L}_{t2i}^{IT}(I)$ and $\mathcal{L}_{t2i}^{II}(T)$ can be obtained using $S^{I}(I,T)$ .

The final text-image alignment loss for each image-text pair is computed as follows:

$$
\mathcal {L} _ {t 2 i} = \mathcal {L} _ {t 2 i} ^ {T T} (I) + \mathcal {L} _ {t 2 i} ^ {I T} (I)) + \mathcal {L} _ {t 2 i} ^ {T I} (T) + \mathcal {L} _ {t 2 i} ^ {I I} (T), \tag {9}
$$

In this way, the thorough text-image alignment boosts a more comprehensive understanding of global multi-modal information.

The final loss of HMSA is computed as $L_{align} = L_{w2o} + L_{p2o} + L_{t2i}$ . By incorporating these three-level alignments, the object embeddings corresponding to the object queries can be gradually refined for better box regression and mask segmentation, further obtaining N potential object proposals via box head and mask head.

# Adaptive Grounding Counter

To adapt to the generalized setting such as no-target and multi-target samples, we design an Adaptive Grounding Counter (AGC) to decide the output number of target objects dynamically for each specific image-text pair. With AGC, the desired target objects can be selected effectively from the N candidate object proposals corresponding to the object queries. Specifically, we formulate it as a classification task to predict an output number. For different images, the same text expression can specify different target objects. For

![](images/ab4bbc76a501b1c12397258fb886fd0217590a0b081f58d12c131d800834dc25.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Layer"] --> B["A"]
    B --> C["C"]
    C --> D["MLP"]
    D --> E["Count: 2"]
    F["Input Layer"] --> G["A"]
    G --> H["Block"]
    H --> I["push pull"]
    I --> J["Memory Bank"]
    J --> K["M"]
    style A fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style J fill:#ccf,stroke:#333
    subgraph Input Layer
        L1["O_e"] --> M1["A"]
        M1 --> N1["Bar"]
        N1 --> O1["C"]
        O1 --> P1["Green Block"]
        P1 --> Q1["Blue Box"]
        Q1 --> R1["Blue Box"]
        R1 --> S1["Blue Box"]
        S1 --> T1["Blue Box"]
        T1 --> U1["Blue Box"]
        U1 --> V1["Blue Box"]
        V1 --> W1["Blue Box"]
        W1 --> X1["Blue Box"]
        X1 --> Y1["Blue Box"]
        Y1 --> Z1["Blue Box"]
        Z1 --> AA1["Blue Box"]
        AA1 --> AB1["Blue Box"]
        AB1 --> AC1["Blue Box"]
        AC1 --> AD1["Blue Box"]
        AD1 --> AE1["Blue Box"]
        AE1 --> AF1["Blue Box"]
        AF1 --> AG1["Blue Box"]
        AG1 --> AH1["Blue Box"]
        AH1 --> AI1["Blue Box"]
        AI1 --> AJ1["Blue Box"]
        AJ1 --> AK1["Blue Box"]
        AK1 --> AL1["Blue Box"]
        AL1 --> AM1["Blue Box"]
        AM1 --> AN1["Blue Box"]
        AN1 --> AO1["Blue Box"]
        AO1 --> AP1["Blue Box"]
        AP1 --> AQ1["Blue Box"]
        AQ1 --> AR1["Blue Box"]
        AR1 --> AS1["Blue Box"]
        AS1 --> AT1["Blue Box"]
        AT1 --> AU["Blue Box"]
        AU --> AV1["Blue Box"]
        AV1 --> AW1["Blue Box"]
        AW1 --> AX1["Blue Box"]
        AX1 --> AY1["Blue Box"]
        AY1 --> AZ1["Blue Box"]
        AZ1 --> BA1["Blue Box"]
        BA1 --> BB1["Blue Box"]
        BB1 --> BC1["Blue Box"]
        BC1 --> BD1["Blue Box"]
        BD1 --> BE1["Blue Box"]
        BE1 --> BF1["Blue Box"]
        BF1 --> BG1["Blue Box"]
        BG1 --> BH1["Blue Box"]
        BH1 --> BI1["Blue Box"]
        BI1 --> BJ1["Blue Box"]
        BJ1 --> BK1["Blue Box"]
        BK1 --> BL1["Blue Box"]
        BL1 --> BM1["Blue Box"]
        BM1 --> BN1["Blue Box"]
        BN1 --> BO1["Blue Box"]
        BO1 --> BP1["Blue Box"]
        BP1 --> BQ1["Blue Box"]
        BQ1 --> BR1["Blue Box"]
        BR1 --> BS1["Blue Box"]
        BS1 --> BT1["Blue Box"]
        BT1 --> BU1["Blue Box"]
        BU1 --> BV1["Blue Box"]
        BV1 --> BW1["Blue Box"]
        BW1 --> BX1["Blue Box"]
        BX1 --> BY1["Blue Box"]
        BY1 --> BZ1["Blue Box"]
        BZ1 --> CA1["Blue Box"]
        CA1 --> CB1["Blue Box"]
        CB1 --> CC1["Blue Box"]
        CC1 --> CD1["Blue Box"]
        CD1 --> CE1["Blue Box"]
        CE1 --> CF1["Blue Box"]
        CF1 --> CG1["Blue Box"]
        CG1 --> CH1["Blue Box"]
        CH1 --> CI1["Blue Box"]
        CI1 --> CJ1["Blue Box"]
        CJ1 --> CK1["Blue Box"]
        CK1 --> CL1["Blue Box"]
        CL1 --> CM1["Blue Box"]
        CM1 --> CN1["Blue Box"]
        CN1 --> CO1["Blue Box"]
        CO1 --> CP1["Blue Box"]
        CP1 --> CQ1["Blue Box"]
        CQ1 --> CY1["Blue Box"]
        CY1 --> CZ1["Blue Box"]
        CZ1 --> DA1["Blue Box"]
        DA1 --> DB1["Blue Box"]
        DB1 --> DC2["Blue Box"]
        DC2 --> DD2["Blue Box"]
        DD2 --> DJ2["Blue Box"]
        DJ2 --> DK2["Blue Box"]
        DK2 --> DL2["Blue Box"]
        DL2 --> DV2["Blue Box"]
        DV2 --> DW2["Blue Box"]
        DW2 --> DX2["Blue Box"]
        DX2 --> DX3["DB Bank"]
    end
    style Input Layer fill:#f9f,stroke:#333
    style Input Layer fill:#ccf,stroke:#333
    style Output Layer fill:#cfc,stroke:#333
```
</details>

Figure 3: The detail of the Adaptive Grounding Counter.

example, “all kids” can denote an arbitrary number of objects in different images. Therefore, it is necessary to incorporate both the sentence feature of the text expression and object embeddings to predict the label. To better train AGC, we conducted a statistical analysis of all samples in the gRefCOCO dataset (Liu, Ding, and Jiang 2023; He et al. 2023). The distribution of objects follows a long-tailed pattern, with most samples falling within the range of 0 to 3, and only a small proportion exceeding the number of 3.

Therefore, it is defined as a classification task into five classes. With word features $T_{w}$ and object embeddings $O_{e}$ at hand, we adopt the average pooling to obtain the global text feature and visual feature. Then, the two features are concatenated to obtain a global multi-modal feature $M_{g}$ , which is used to predict the object counting label $y_{c}$ as:

$$
M _ {g} = [ \mathrm{AP} (T _ {w}); \mathrm{AP} (\mathcal {O} _ {e}) ], \tag {10}
$$

$$
y _ {c} = \mathrm{MLP} (M _ {g}), \tag {11}
$$

where [;] denotes concatenation operation, AP denotes average pooling, MLP denotes a two-layer perceptron, and $y_{c} \in \{0, 1, 2, 3, 3+\}$ . Note that, only when the counting is larger than 3, the threshold-based strategy is adopted, that is object proposals with class scores above the threshold are selected. Otherwise, the sorted target objects with high scores are selected according to the counting.

To further promote the object counting ability, we incorporate contrastive learning in AGC by pulling together the multi-modal features $M_{g}$ with the same counting and pushing away those with the different counting in Figure 3. The number of negative samples is related to batch size. However, the size of the batch size is limited by GPU memory. Therefore, to facilitate contrastive learning, we introduce a memory bank M (He et al. 2020) to maintain a larger number of negative samples. Inspired by (Khosla et al. 2020), a supervised contrastive loss $L_{con}$ is introduced as follows:

$$
\mathcal {L} _ {\text { con }} = - \frac {1}{| P (i) |} \sum_ {p \in P (i)} \log \frac {\exp (M _ {g} ^ {i} \cdot M _ {g} ^ {p} / \tau)}{\sum_ {a \in A (i)} \exp (M _ {g} ^ {i} \cdot M _ {g} ^ {a} / \tau)}, \tag {12}
$$

where $i$ denotes anchor index, $P(i) = \{p \in A(i), y_c^p = y_c^i\}$ is a collection of indices for the positive samples in $\mathcal{M}$ , $|P(i)|$ denotes the cardinality of the collection, $A(i)$ denotes a collection of indices for all positive and negative samples in $\mathcal{M}$ , $M_g$ denotes the global multi-modal feature, and $\tau$ is a temperature hyperparameter.

The final loss of AGC is computed as $\mathcal{L}_{agc} = \mathcal{L}_{cls} + \mathcal{L}_{con}$ , where $\mathcal{L}_{cls}$ denotes the object counting classification loss, implemented by a cross-entropy loss.

<table><tr><td rowspan="2">Methods</td><td colspan="2">val</td><td colspan="2">testA</td><td colspan="2">testB</td></tr><tr><td>Pr</td><td>N-acc.</td><td>Pr</td><td>N-acc.</td><td>Pr</td><td>N-acc.</td></tr><tr><td> $MCN^†$ </td><td>28.0</td><td>30.6</td><td>32.3</td><td>32.0</td><td>26.8</td><td>30.3</td></tr><tr><td> $VLT^†$ </td><td>36.6</td><td>35.2</td><td>40.2</td><td>34.1</td><td>30.2</td><td>32.5</td></tr><tr><td> $MDETR^†$ </td><td>42.7</td><td>36.3</td><td>50.0</td><td>34.5</td><td>36.5</td><td>31.0</td></tr><tr><td> $UNITEXT^†$ </td><td>58.2</td><td>50.6</td><td>46.4</td><td>49.3</td><td>42.9</td><td>48.2</td></tr><tr><td> $Ferret^*$ </td><td>54.8</td><td>48.9</td><td>49.5</td><td>45.2</td><td>43.5</td><td>43.8</td></tr><tr><td> $HieA2G_{R101}$ </td><td>67.8</td><td>60.3</td><td>66.0</td><td>60.1</td><td>56.5</td><td>56.0</td></tr></table>

Table 1: Results on gRefCOCO dataset (Liu, Ding, and Jiang 2023) in terms of $\Pr@(F_{1}=1,\mathrm{IoU}\geq0.5)$ and N-acc. for GREC task. $\dagger$ denotes these methods have been modified to generate multiple boxes following (He et al. 2023). \* denotes the model adapted for the GREC task.

# Training Objective

To further supervise task-specific training, a series of losses for the box head and mask head are introduced as follows:

$$
\mathcal {L} _ {\text { det }} = \lambda_ {b b o x} \mathcal {L} _ {b b o x} + \lambda_ {g i o u} \mathcal {L} _ {g i o u} + \lambda_ {\text { class }} \mathcal {L} _ {\text { class }}, \tag {13}
$$

$$
\mathcal {L} _ {\text {seg}} = \lambda_ {\text {mask}} \mathcal {L} _ {\text {mask}} + \lambda_ {\text {dice}} \mathcal {L} _ {\text {dice}}, \tag {14}
$$

where $\lambda_{*}$ are the hyperparameters, $L_{class}$ is cross-entropy loss for box classification, $L_{bbox}$ and $L_{giou}$ are L1 loss (Ren et al. 2015) and GIoU loss (Rezatofighi et al. 2019) for box regression. The focal loss $L_{mask}$ (Lin et al. 2017) and dice loss $L_{dice}$ (Milletari, Navab, and Ahmadi 2016) are introduced to supervise mask segmentation (Ding et al. 2018).

We first pretrain HieA2G on a combined dataset formed by the training data of RefCOCO/+/g, Flickr30K Entities, and gRefCOCO datasets using the joint loss $L_{pretrain} = L_{align} + L_{det}$ . The goal of pretraining is to incorporate comprehensive multi-modal information into object queries. Then based on the pretrained weights, HieA2G is finetuned on various downstream tasks with the task-specific loss such as $L_{det}$ , $L_{mask}$ and an additional loss $L_{agc}$ introduced specifically for GREC and GRES.

# Experiments

# Experimental Setup

Datasets. The proposed HieA2G is evaluated mainly on the gRefCOCO dataset (He et al. 2023; Liu, Ding, and Jiang 2023) for GREC and GRES. We also conducted experiments on a phrase grounding dataset called Flickr30K Entities (Plummer et al. 2015), and three widely-used REC and RES benchmarks including RefCOCO (Yu et al. 2016), RefCOCO+ (Yu et al. 2016), and RefCOCOg (Mao et al. 2016). Implementation Details. We adopt ResNet101 (He et al. 2016) and Swin-B (Liu et al. 2021) as our visual encoder, and RoBERTa-base (Liu et al. 2019b) as our text encoder.

# Performance Comparison

Results on GREC. As shown in Table 1, our HieA2G with the ResNet101 backbone achieves superior performance on both metrics across three splits of the gRefCOCO dataset. It shows an average performance gain of $14.2\%$ in $\mathrm{Pr}@\mathrm{(F_1 = 1}$ , IoU $\geq 0.5$ ) over Ferret (You et al. 2024) using a Multimodal

![](images/fe784b7d99cd5800d092dc9e267ba0480b4f920bf72ed44b96c1f3ff156d8bb7.jpg)  
(a) top right two airplanes

![](images/71efea2bde83f0a2060638580d11799a21b69978729241a899f7b30fde06caf2.jpg)  
(b) closest bus on right and red bus 2nd from front

![](images/bb589cfd15e7448d49c4229f07fdb98d21bde2601ceb026f406ebe6b88532c9f.jpg)  
(c) blurred person in crowd directly right of woman in pink shirt on bike and driver

![](images/99589afd2f135941ac2b71475764cdf0bbf37d506819cecf033efaefc16702c4.jpg)  
(d) All zebras

![](images/1f1bde29c3259fa6690f846c14b58400ea948c3162e77e136a1156a8470bdcb4.jpg)  
(e) the board in man hand

![](images/766e87998634684553b5f873d6d672c2353245fe0e4301d749368069be3237c0.jpg)  
(f) second to left middle row green round thing   
(A) Successful cases on gRefCOCO dataset.

![](images/46ea8b45018f6bd86301843166e5aa949880445d0b836da6333ac92b34c5bfd2.jpg)  
(a) the person repairing the car and their umbrellas

![](images/3c222dedfd86a3cac0764c1d0ea0abc125b3f69d76993c780bb94e0fcaf65e3c.jpg)  
(b) the racket on the ground and the player jumping in the air

![](images/3d081ea82a45516af4408bd3be4f38a7090103b2c731cd223c3d3b1697f0df4f.jpg)  
(c) the white laptop on the table in front of the lady

![](images/58280f6852ddf9a91839f3801a655732b4d6f918aacd9b15b0b37d1b6e3bfb67.jpg)  
(d) all people and their skiing boards

![](images/793a56eb2c774c77e25713ea0dc684b13d2922454995a38fd1e191cac0b77070.jpg)  
(e) the front of the truck right grill and white car on the left

![](images/ca1d00ff3b9f0ab00e05c86882725aed4d9c72184740e2c05ab556c73f6de490.jpg)  
(f) the young lady with glasses is walking towards us   
(B) Failure cases on gRefCOCO dataset.

Figure 4: Visualization for the success cases and failure cases of HieA2G on gRefCOCO dataset. The ground truth is denoted by red bounding boxes, whereas green bounding boxes denote the predictions. The $F_{1}$ score of all success cases in (A) is 1.0. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val-u</td><td>test-u</td></tr><tr><td>MAttNet</td><td>76.7</td><td>81.1</td><td>70.0</td><td>65.3</td><td>71.6</td><td>56.0</td><td>66.6</td><td>67.3</td></tr><tr><td>RefTR</td><td>85.7</td><td>88.7</td><td>81.2</td><td>77.6</td><td>82.3</td><td>69.0</td><td>79.3</td><td>80.0</td></tr><tr><td>MDETR</td><td>86.8</td><td>89.9</td><td>81.4</td><td>79.5</td><td>84.1</td><td>70.6</td><td>81.6</td><td>80.9</td></tr><tr><td>SeqTR</td><td>83.7</td><td>86.5</td><td>81.2</td><td>71.5</td><td>76.3</td><td>64.9</td><td>74.9</td><td>74.2</td></tr><tr><td>TransVG++</td><td>86.3</td><td>88.4</td><td>81.0</td><td>75.4</td><td>80.5</td><td>66.3</td><td>76.2</td><td>76.3</td></tr><tr><td>LISA-7B</td><td>85.4</td><td>88.8</td><td>82.6</td><td>74.2</td><td>79.5</td><td>68.4</td><td>79.3</td><td>80.4</td></tr><tr><td>GSVA-7B</td><td>86.3</td><td>89.2</td><td>83.8</td><td>72.8</td><td>78.8</td><td>68.0</td><td>81.6</td><td>81.8</td></tr><tr><td>HieA2GR101</td><td>87.8</td><td>90.3</td><td>84.0</td><td>80.7</td><td>85.6</td><td>72.9</td><td>83.7</td><td>83.8</td></tr></table>

Table 2: Results comparison on RefCOCO/+/g for REC task.

<table><tr><td rowspan="2">Methods</td><td colspan="3">val</td><td colspan="3">test</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>VisualBert</td><td>68.1</td><td>84.0</td><td>86.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VisualBert</td><td>70.4</td><td>84.5</td><td>86.3</td><td>71.3</td><td>85.0</td><td>86.5</td></tr><tr><td>MDETR</td><td>82.5</td><td>92.9</td><td>94.9</td><td>83.4</td><td>93.5</td><td>95.3</td></tr><tr><td>Shrika-7B</td><td>75.8</td><td>-</td><td>-</td><td>76.5</td><td>-</td><td>-</td></tr><tr><td>Ferret-7B</td><td>80.4</td><td>-</td><td>-</td><td>82.2</td><td>-</td><td>-</td></tr><tr><td> $HieA2G_{R101}$ </td><td>82.9</td><td>93.2</td><td>95.1</td><td>83.7</td><td>93.8</td><td>95.6</td></tr></table>

Table 3: Results comparison on Flickr30K Entities dataset in Recall@k (ANY-BOX protocol) for Phrase Grounding task.

Large Language Model (MLLM), and an average performance gain of 9.4% in N-acc. over UNITEXT (Yan et al. 2023). These results indicate that HieA2G has a significant advantage in handling various types of text expressions to flexibly detect target objects ranging from zero to multiple.

Results on REC. As illustrated in Table 2, HieA2G achieves consistent performance gains across all splits of the three datasets compared to existing classic REC methods. HieA2G with the ResNet101 backbone even outperforms GSVA-7B (Xia et al. 2024) based on MLLM. The promising results can be attributed to the hierarchical multi-modal semantic alignment design, which promotes a comprehensive understanding of information at different granularities.

Results on Phrase Grounding. The main results on the

<table><tr><td rowspan="2">Methods</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val-u</td><td>test-u</td></tr><tr><td>MAttNet</td><td>56.5</td><td>62.4</td><td>51.7</td><td>46.7</td><td>52.4</td><td>40.1</td><td>47.6</td><td>48.6</td></tr><tr><td>MCN</td><td>62.4</td><td>64.2</td><td>59.7</td><td>50.6</td><td>55.0</td><td>44.7</td><td>49.2</td><td>49.4</td></tr><tr><td>VLT</td><td>65.7</td><td>68.3</td><td>62.7</td><td>55.5</td><td>59.2</td><td>49.4</td><td>52.9</td><td>56.7</td></tr><tr><td>HieA2GR101</td><td>73.3</td><td>75.9</td><td>69.0</td><td>64.8</td><td>69.7</td><td>56.1</td><td>62.9</td><td>63.5</td></tr><tr><td>LAVT</td><td>72.7</td><td>75.8</td><td>68.8</td><td>62.1</td><td>68.4</td><td>55.1</td><td>61.2</td><td>62.1</td></tr><tr><td>ReLA</td><td>73.8</td><td>76.5</td><td>70.2</td><td>66.0</td><td>71.0</td><td>57.7</td><td>65.0</td><td>66.0</td></tr><tr><td>LISA-7B</td><td>74.9</td><td>79.1</td><td>72.3</td><td>65.1</td><td>70.8</td><td>58.1</td><td>67.9</td><td>70.6</td></tr><tr><td>GSVA-7B</td><td>77.2</td><td>78.9</td><td>73.5</td><td>65.9</td><td>69.6</td><td>59.8</td><td>72.7</td><td>73.3</td></tr><tr><td>HieA2GSwinB</td><td>75.1</td><td>77.6</td><td>71.1</td><td>66.5</td><td>71.4</td><td>58.9</td><td>65.3</td><td>66.6</td></tr></table>

Table 4: Results comparison on RefCOCO/+/g for RES task.

Flickr30K Entities are shown in Table 3. HieA2G with ResNet101 improves performance over the previous SOTA MDETR on both val and test splits, suggesting that our method effectively enhances the multi-modal interactions.

Results on RES. As shown in Table 4, HieA2G outperforms the previous SOTA method ReLA (Liu, Ding, and Jiang 2023) with the same Swin-B backbone. It also shows competitive performance on RefCOCO and RefCOCO+ datasets to MLLM-based LISA-7B (Lai et al. 2024) and GSVA-7B. The results demonstrate that the comprehensive multi-modal representation ability of our HieA2G can contribute a lot to accurate segmentation for referring objects.

Results on GRES. GRES aims to generate masks for an arbitrary number of target objects. Unlike the previous SOTA GRES method ReLA using a simple binary classification branch for object-existence judgment, HieA2G has an explicit object-counting ability to facilitate accurate object perception in the generalized scenario. In Table 5, HieA2G with the Swin-B backbone achieves clear performance improvements over ReLA across all three evaluation sets on different metrics. It is even slightly better than the strong MLLM-based GSVA-7B in CIoU and GIoU. Besides generating high-quality masks, HieA2G with either backbone demonstrates outstanding performance in N-acc. and T-acc.,

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Backbone</td><td colspan="4">val</td><td colspan="4">testA</td><td colspan="4">testB</td></tr><tr><td>cIoU</td><td>gIoU</td><td>N-acc.</td><td>T-acc.</td><td>cIoU</td><td>gIoU</td><td>N-acc.</td><td>T-acc.</td><td>cIoU</td><td>gIoU</td><td>N-acc.</td><td>T-acc.</td></tr><tr><td>MAttNet</td><td>ResNet101</td><td>47.5</td><td>48.2</td><td>41.2</td><td>96.1</td><td>58.7</td><td>59.3</td><td>44.0</td><td>97.6</td><td>45.3</td><td>46.1</td><td>41.3</td><td>95.3</td></tr><tr><td>VLT</td><td>DarkNet53</td><td>52.5</td><td>52.0</td><td>47.2</td><td>95.7</td><td>62.2</td><td>63.2</td><td>48.7</td><td>95.9</td><td>50.5</td><td>50.9</td><td>47.8</td><td>94.7</td></tr><tr><td>VLT+ReLA</td><td>DarkNet53</td><td>58.7</td><td>59.4</td><td>-</td><td>-</td><td>66.6</td><td>65.4</td><td>-</td><td>-</td><td>56.2</td><td>57.4</td><td>-</td><td>-</td></tr><tr><td>CRIS</td><td>ResNet101</td><td>55.3</td><td>56.3</td><td>-</td><td>-</td><td>63.8</td><td>63.4</td><td>-</td><td>-</td><td>51.0</td><td>51.8</td><td>-</td><td>-</td></tr><tr><td>HieA2G</td><td>ResNet101</td><td>62.5</td><td>67.1</td><td>60.9</td><td>97.4</td><td>67.6</td><td>70.5</td><td>60.2</td><td>97.7</td><td>58.8</td><td>61.5</td><td>56.5</td><td>96.4</td></tr><tr><td>LAVT</td><td>Swin-B</td><td>57.6</td><td>58.4</td><td>49.3</td><td>96.2</td><td>65.3</td><td>65.9</td><td>49.3</td><td>95.1</td><td>55.0</td><td>55.8</td><td>48.5</td><td>95.3</td></tr><tr><td>ReLA</td><td>Swin-B</td><td>62.4</td><td>63.6</td><td>56.4</td><td>96.3</td><td>69.3</td><td>70.0</td><td>59.0</td><td>97.8</td><td>59.9</td><td>61.0</td><td>58.4</td><td>95.4</td></tr><tr><td>LISA-7B</td><td>ViT-H</td><td>61.8</td><td>61.6</td><td>54.7</td><td>-</td><td>68.5</td><td>66.3</td><td>50.0</td><td>-</td><td>60.6</td><td>58.8</td><td>51.9</td><td>-</td></tr><tr><td>GSVA-7B</td><td>ViT-H</td><td>63.3</td><td>66.5</td><td>62.4</td><td>-</td><td>69.9</td><td>71.1</td><td>65.3</td><td>-</td><td>60.5</td><td>62.2</td><td>60.6</td><td>-</td></tr><tr><td>HieA2G</td><td>Swin-B</td><td>64.2</td><td>68.4</td><td>62.8</td><td>98.3</td><td>70.4</td><td>72.0</td><td>63.4</td><td>98.5</td><td>61.0</td><td>62.8</td><td>60.8</td><td>97.5</td></tr></table>

Table 5: Results comparison on gRefCOCO dataset in terms of cIoU, gIoU, N-acc. and T-acc. for GRES task.

<table><tr><td rowspan="2">#</td><td colspan="3">HMSA</td><td colspan="2">AGC</td><td colspan="2">GREC</td></tr><tr><td>W2O</td><td>P2O</td><td>T2I</td><td>Classifier</td><td> $\mathcal{L}_{con}$ </td><td>Pr</td><td>N-acc.</td></tr><tr><td>#1</td><td></td><td></td><td></td><td>√</td><td>√</td><td>65.2</td><td>54.9</td></tr><tr><td>#2</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>67.5</td><td>58.1</td></tr><tr><td>#3</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>67.1</td><td>57.3</td></tr><tr><td>#4</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>67.0</td><td>56.4</td></tr><tr><td>#5</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>53.9</td><td>48.0</td></tr><tr><td>#6</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>66.5</td><td>57.3</td></tr><tr><td>#7</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>67.8</td><td>60.3</td></tr></table>

Table 6: Ablation study of different components of HieA2G for GREC. Notably, W2O, P2O and T2I indicate the word-object, phrase-object, and text-image alignment.

indicating its robust object perception ability.

# Ablation Study

Effect of Hierarchical Multi-modal Semantic Alignment. From the first to fourth rows of Table 6, we perform an in-depth study of the HMSA module to validate its effectiveness. In the first row, removing all three-level alignments of the HMSA module leads to a significant decrease of 2.6% in $\Pr@(F_{1}=1,\mathrm{IoU}\geq0.5)$ and 5.4% in N-acc. Furthermore, we can find that removing any one of the alignments, including W2O, P2O, and T2I, all leads to performance degradation compared with the overall model in the seventh row. This demonstrates that different levels of alignment can refine the object embeddings of the object queries, further facilitating accurate grounding by combining them.

Effect of Adaptive Grounding Counter. In the last three rows of Table 6, we test the effectiveness of AGC. The overall AGC is removed in the fifth row and replaced by the default threshold-based strategy (He et al. 2023) to filter the output objects. We can observe a $13.9\%$ and $12.3\%$ performance drop in terms of $\mathrm{Pr}@\mathrm{(F_1 = 1,IoU\geq 0.5)}$ and N-acc., which reflects that our advanced adaptive selection strategy contributes a lot to the output of target objects. Then, we add the classifier in the sixth row, which achieves $12.6\%$ and $9.3\%$ performance gain in both metrics respectively. When combined with $\mathcal{L}_{con}$ in the last row, further improvement can be brought for all metrics. This suggests that $\mathcal{L}_{con}$ is helpful to enhance the model's object counting ability.

# Qualitative Analysis

We visualize some qualitative examples of our method on the validation split of gRefCOCO dataset to discuss the strengths and weaknesses of HieA2G as shown in Figure 4. Analysis of Success Cases. As shown in (A) of Figure 4, our model can deal with various complex multi-target expressions in (a)-(d) and no-target expressions in (e)-(f). For example, HieA2G can count accurately “two airplanes” in (a) with shared attributes, and can differentiate an ordinal number like “2nd” to detect the correct bus in (b). It can also explicitly recognize all specified target objects for complex text expressions in (b), (c), and (d). Moreover, HieA2G can grasp the fine-grained attribute details to reject the no-target expression “the board in man hand” in (e). It has a comprehensive understanding of the global contextual information of all objects in (f), thereby rejecting to give a detection result due to no object in the image satisfying the description. Analysis of Failure Cases. We show some failure cases of HieA2G in (B) of Figure 4. There are two main types of failure cases. ① For the three cases (a)-(c) in the first row, due to the ambiguous visual clues in the image, HieA2G struggles to detect all target objects for the first two cases and fails to reject giving a target for the last case. ② For the three cases (d)-(f) in the second row, due to the occlusion of the key visual clues, HieA2G fails to detect a desired object for the first two cases and gives a false negative target for the last no-target case. The analysis of failure cases reveals the limitations of HieA2G, while also shedding light on potential directions for our future research.

# Conclusion

We propose a Hierarchical Alignment-enhanced Adaptive Grounding Network (HieA2G) for the challenging GREC task. The proposed Hierarchical Multi-modal Semantic Alignment (HMSA) module enables multi-level cross-modal interactions to achieve comprehensive and robust multi-modal understanding for better grounding. Adaptive Grounding Counter (AGC) determines the number of output targets dynamically to help select the outputs, effectively tackling the varying number of target objects in flexible referring expressions. The experimental results demonstrate the remarkable superiority and generalizability of the proposed HieA2G on multiple visual grounding tasks including REC, GREC, phrase grounding, RES, and GRES.

# References

Chen, K.; Zhang, Z.; Zeng, W.; Zhang, R.; Zhu, F.; and Zhao, R. 2023. Shikra: Unleashing multimodal llm's referential dialogue magic. arXiv preprint arXiv:2306.15195.

Cheng, B.; Misra, I.; Schwing, A. G.; Kirillov, A.; and Girdhar, R. 2022. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Deng, J.; Yang, Z.; Liu, D.; Chen, T.; Zhou, W.; Zhang, Y.; Li, H.; and Ouyang, W. 2023. TransVG++: End-to-end visual grounding with language conditioned vision transformer. IEEE TPAMI.

Ding, H.; Jiang, X.; Shuai, B.; Liu, A. Q.; and Wang, G. 2018. Context contrasted feature and gated multi-scale aggregation for scene segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2393–2402.

Ding, H.; Liu, C.; He, S.; Jiang, X.; and Loy, C. C. 2023a. MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2694–2703.

Ding, H.; Liu, C.; He, S.; Jiang, X.; Torr, P. H.; and Bai, S. 2023b. MOSE: A new dataset for video object segmentation in complex scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20224–20234.

Ding, H.; Liu, C.; Wang, S.; and Jiang, X. 2021. Vision-language transformer and query generation for referring segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 16321–16330.

Ding, H.; Liu, C.; Wang, S.; and Jiang, X. 2023c. VLT: Vision-language transformer and query generation for referring segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6): 7900–7916.

He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9729–9738.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778.

He, S.; and Ding, H. 2024. RefMask3D: Language-Guided Transformer for 3D Referring Segmentation. In ACM International Conference on Multimedia, 8316–8325.

He, S.; Ding, H.; Jiang, X.; and Wen, B. 2024. SegPoint: Segment Any Point Cloud via Large Language Model. In European Conference on Computer Vision, 349–367.

He, S.; Ding, H.; Liu, C.; and Jiang, X. 2023. GREC: Generalized referring expression comprehension. arXiv preprint arXiv:2308.16182.

Hu, R.; Rohrbach, M.; Andreas, J.; Darrell, T.; and Saenko, K. 2017. Modeling relationships in referential expressions with compositional modular networks. In CVPR.

Kamath, A.; Singh, M.; LeCun, Y.; Synnaeve, G.; Misra, I.; and Carion, N. 2021. MDETR-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Khosla, P.; Teterwak, P.; Wang, C.; Sarna, A.; Tian, Y.; Isola, P.; Maschinot, A.; Liu, C.; and Krishnan, D. 2020. Supervised contrastive learning. Advances in Neural Information Processing Systems, 33: 18661–18673.

Kim, N.; Kim, D.; Lan, C.; Zeng, W.; and Kwak, S. 2022. ReSTR: Convolution-free referring image segmentation using transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Lai, X.; Tian, Z.; Chen, Y.; Li, Y.; Yuan, Y.; Liu, S.; and Jia, J. 2024. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9579–9589.

Li, L.; Bu, Y.; and Cai, Y. 2021. Bottom-up and bidirectional alignment for referring expression comprehension. In ACM MM, 5167–5175.

Li, M.; and Sigal, L. 2021. Referring transformer: A one-step approach to multi-task visual grounding. Advances in Neural Information Processing Systems, 34: 19652–19664.

Li, X.; Ding, H.; Yuan, H.; Zhang, W.; Pang, J.; Cheng, G.; Chen, K.; Liu, Z.; and Loy, C. C. 2024. Transformer-Based Visual Segmentation: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 10138–10163.

Liao, Y.; Liu, S.; Li, G.; Wang, F.; Chen, Y.; Qian, C.; and Li, B. 2020. A real-time cross-modality correlation filtering method for referring expression comprehension. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10880–10889.

Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollár, P. 2017. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision.

Liu, C.; Ding, H.; and Jiang, X. 2023. GRES: Generalized Referring Expression Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23592–23601.

Liu, C.; Ding, H.; Zhang, Y.; and Jiang, X. 2023. Multimodal mutual attention and iterative interaction for referring image segmentation. IEEE Transactions on Image Processing, 32: 3054–3065.

Liu, C.; Jiang, X.; and Ding, H. 2022. Instance-specific feature propagation for referring segmentation. IEEE Transactions on Multimedia, 25: 3657–3667.

Liu, C.; Jiang, X.; and Ding, H. 2024. Primitivenet: decomposing the global constraints for referring segmentation. Visual Intelligence, 2(1): 16.

Liu, C.; Li, X.; and Ding, H. 2024. Referring Image Editing: Object-level Image Editing via Referring Expressions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13128–13138.

Liu, Q.; Wen, Y.; Han, J.; Xu, C.; Xu, H.; and Liang, X. 2022. Open-world semantic segmentation via contrasting and clustering vision-language embedding. In European Conference on Computer Vision, 275–292.

Liu, X.; Wang, Z.; Shao, J.; Wang, X.; and Li, H. 2019a. Improving referring expression grounding with cross-modal attention-guided erasing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.; Levy, O.; Lewis, M.; Zettlemoyer, L.; and Stoyanov, V. 2019b. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.   
Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision.   
Luo, G.; Zhou, Y.; Sun, X.; Cao, L.; Wu, C.; Deng, C.; and Ji, R. 2020. Multi-task collaborative network for joint referring expression comprehension and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10034–10043.   
Mao, J.; Huang, J.; Toshev, A.; Camburu, O.; Yuille, A. L.; and Murphy, K. 2016. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.   
Meng, L.; Li, H.; Chen, B.-C.; Lan, S.; Wu, Z.; Jiang, Y.-G.; and Lim, S.-N. 2022. AdaViT: Adaptive Vision Transformers for Efficient Image Recognition. In CVPR.   
Milletari, F.; Navab, N.; and Ahmadi, S.-A. 2016. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In 3DV, 565–571.   
Plummer, B. A.; Wang, L.; Cervantes, C. M.; Caicedo, J. C.; Hockenmaier, J.; and Lazebnik, S. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE International Conference on Computer Vision, 2641–2649.   
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 8748–8763.   
Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards real-time object detection with region proposal networks. In NeurIPS.   
Rezatofighi, H.; Tsoi, N.; Gwak, J.; Sadeghian, A.; Reid, I.; and Savarese, S. 2019. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 658–666.   
Wang, Z.; Lu, Y.; Li, Q.; Tao, X.; Guo, Y.; Gong, M.; and Liu, T. 2022. CRIS: Clip-driven referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11686–11695.   
Wu, C.; Liu, Y.; Ji, J.; Ma, Y.; Wang, H.; Luo, G.; Ding, H.; Sun, X.; and Ji, R. 2024a. 3D-GRES: Generalized 3d referring expression segmentation. In Proceedings of the 32nd ACM International Conference on Multimedia, 7852–7861.   
Wu, J.; Li, X.; Li, X.; Ding, H.; Tong, Y.; and Tao, D. 2024b. Towards robust referring image segmentation. IEEE Transactions on Image Processing.   
Wu, J.; Li, X.; Xu, S.; Yuan, H.; Ding, H.; Yang, Y.; Li, X.; Zhang, J.; Tong, Y.; Jiang, X.; Ghanem, B.; and Tao, D. 2024c. Towards Open Vocabulary Learning: A Survey. IEEE

Transactions on Pattern Analysis and Machine Intelligence, 46(7): 5092–5113.

Wu, Z.; Weng, Z.; Peng, W.; Yang, X.; Li, A.; Davis, L. S.; and Jiang, Y. 2024d. Building an Open-Vocabulary Video CLIP Model With Better Architectures, Optimization and Data. IEEE TPAMI.

Xia, Z.; Han, D.; Han, Y.; Pan, X.; Song, S.; and Huang, G. 2024. Gsva: Generalized segmentation via multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Xu, J.; De Mello, S.; Liu, S.; Byeon, W.; Breuel, T.; Kautz, J.; and Wang, X. 2022. GroupViT: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18134–18144.

Yan, B.; Jiang, Y.; Wu, J.; Wang, D.; Luo, P.; Yuan, Z.; and Lu, H. 2023. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yang, L.; Xu, Y.; Yuan, C.; Liu, W.; Li, B.; and Hu, W. 2022a. Improving visual grounding with visual-linguistic verification and iterative reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9499–9508.

Yang, S.; Li, G.; and Yu, Y. 2019. Dynamic graph attention for referring expression comprehension. In ICCV.

Yang, Z.; Wang, J.; Tang, Y.; Chen, K.; Zhao, H.; and Torr, P. H. 2022b. LAVT: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18155–18165.

Ye, J.; Lin, X.; He, L.; Li, D.; and Chen, Q. 2021. One-stage visual grounding via semantic-aware feature filter. In ACM MM, 1702–1711.

You, H.; Zhang, H.; Gan, Z.; Du, X.; Zhang, B.; Wang, Z.; Cao, L.; Chang, S.-F.; and Yang, Y. 2024. Ferret: Refer and Ground Anything Anywhere at Any Granularity. In International Conference on Learning Representations.

Yu, L.; Poirson, P.; Yang, S.; Berg, A. C.; and Berg, T. L. 2016. Modeling context in referring expressions. In ECCV. Yu, T.; Hui, T.; Yu, Z.; Liao, Y.; Yu, S.; Zhang, F.; and Liu, S. 2020. Cross-modal omni interaction modeling for phrase grounding. In ACM MM.

Zhou, Y.; Ji, R.; Luo, G.; Sun, X.; Su, J.; Ding, X.; Lin, C.-W.; and Tian, Q. 2021. A real-time global inference network for one-stage referring expression comprehension. IEEE Transactions on Neural Networks and Learning Systems.

Zhu, C.; Zhou, Y.; Shen, Y.; Luo, G.; Pan, X.; Lin, M.; Chen, C.; Cao, L.; Sun, X.; and Ji, R. 2022. SeqTR: A simple yet universal network for visual grounding. In European Conference on Computer Vision, 598–615.

Zhuang, B.; Wu, Q.; Shen, C.; Reid, I.; and Van Den Hengel, A. 2018. Parallel attention: A unified framework for visual object discovery through dialogs and queries. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 4252–4261.