# DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection

Jia Syuen Lim $^{*}$ Zhuoxiao Chen $^{*}$ Mahsa Baktashmotlagh
Zhi Chen Xin Yu Zi Huang Yadan Luo $^{\dagger}$

The University of Queensland

{jiasyuen.lim, zhuoxiao.chen, m.baktashmotlagh}@uq.edu.au

{zhi.chen, xin.yu, helen.huang, y.luo}@uq.edu.au

# Abstract

Class-agnostic object detection (OD) can be a cornerstone or a bottleneck for many downstream vision tasks. Despite considerable advancements in bottom-up and multi-object discovery methods that leverage basic visual cues to identify salient objects, consistently achieving a high recall rate remains difficult due to the diversity of object types and their contextual complexity. In this work, we investigate using vision-language models (VLMs) to enhance object detection via a self-supervised prompt learning strategy. Our initial findings indicate that manually crafted text queries often result in undetected objects, primarily because detection confidence diminishes when the query words exhibit semantic overlap. To address this, we propose a Dispersing Prompt Expansion (DiPEx) approach. DiPEx progressively learns to expand a set of distinct, non-overlapping hyperspherical prompts to enhance recall rates, thereby improving performance in downstream tasks such as out-of-distribution OD. Specifically, DiPEx initiates the process by self-training generic parent prompts and selecting the one with the highest semantic uncertainty for further expansion. The resulting child prompts are expected to inherit semantics from their parent prompts while capturing more fine-grained semantics. We apply dispersion losses to ensure high inter-class discrepancy among child prompts while preserving semantic consistency between parent-child prompt pairs. To prevent excessive growth of the prompt sets, we utilize the maximum angular coverage (MAC) of the semantic space as a criterion for early termination. We demonstrate the effectiveness of DiPEx through extensive class-agnostic OD and OOD-OD experiments on MS-COCO and LVIS, surpassing other prompting methods by up to 20.1% in AR and achieving a 21.3% AP improvement over SAM. The code is available at https://github.com/jason-lim26/DiPEx.

# 1 Introduction

In real-world applications, the class of interest may constantly change, prompting the need for new tasks like out-of-distribution (OOD) detection $[53, 11]$ , open-world detection $[60, 52, 22, 62, 55]$ and open-vocabulary $[48, 54, 31, 28]$ object detection (OD) to ensure reliable operation of detectors. A significant bottleneck in these OD tasks is the ability to locate all objects in a scene - typically referred to as class-agnostic OD $[36]$ . Ensuring a high recall rate is essential in this task as it lays the foundation for correctly classifying objects, thereby improving the average precision for classes of interest. Conversely, a low recall implies that some objects will be missed entirely, negatively impacting downstream recognition tasks.

![](images/a3b0d88d1418329a4e9cc654413205a04ba71b285397d4ac24e31efac3f676bf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Class-Agnostic Prompt"] --> B["Text Encoder"]
    C["V1 V2 V3 ... VM"] --> B
    D["G-DINO"] --> B
    E["Vision Encoder"] --> F["G-DINO"]
    G["Text Encoder"] --> F
    H["Class-Agnostic Prompt"] --> I["Text Encoder"]
    J["V1 V2 V3 ... VM"] --> I
    K["Known Classes &quot;keyboard&quot; + &quot;laptop&quot;"] --> I
    L["Class-Agnostic Detection"] --> M["G-DINO"]
    N["Laptop OOD keyboard OOD OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboardOOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard OOD keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood keyboard Oood controller"]
```
</details>

(a) Class-agnostic OD and Downstream OOD-OD

![](images/e242792923b6cb287c598e5dd555bdd0b45460bbcea37217a1f4182002e91c43.jpg)

<details>
<summary>bar</summary>

| Category | @Small | @Medium | @Large |
|---|---|---|---|
| foreground | 0.00 | 0.15 | 0.65 |
| entity | 0.15 | 0.45 | 0.75 |
| universal | 0.10 | 0.40 | 0.80 |
| class | 0.05 | 0.30 | 0.75 |
| object | 0.15 | 0.40 | 0.70 |
| substance | 0.20 | 0.45 | 0.70 |
| stuff | 0.25 | 0.50 | 0.75 |
| items | 0.30 | 0.55 | 0.80 |
| elements | 0.10 | 0.40 | 0.75 |
| objects | 0.25 | 0.50 | 0.75 |
| small | 0.05 | 0.35 | 0.70 |
| generic | 0.15 | 0.45 | 0.75 |
| liry | 0.05 | 0.30 | 0.70 |
| activity | 0.00 | 0.25 | 0.65 |
| animal | 0.00 | 0.25 | 0.65 |
| attribute | 0.15 | 0.25 | 0.65 |
| body | 0.15 | 0.35 | 0.75 |
| knowledge | 0.15 | 0.35 | 0.75 |
| communication | 0.15 | 0.35 | 0.75 |
| event | 0.15 | 0.25 | 0.65 |
| feeling | 0.15 | 0.25 | 0.65 |
| food group | 0.15 | 0.35 | 0.75 |
| location motivation | 0.15 | 0.35 | 0.75 |
| natural object | 0.25 | 0.35 | 0.75 |
| natural phenomenon | 0.35 | 0.35 | 0.75 |
| person possession | 0.15 | 0.35 | 0.75 |
| plant process | 0.25 | 0.35 | 0.75 |
| quantity relation | 0.15 | 0.35 | 0.75 |
| shape state substance | 0.15 | 0.35 | 0.75 |
| time | 0.15 | 0.35 | 0.75 |
AR = AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR*AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR *AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR; AR*AR *AR
</details>

(b) UNIVERSAL and CLASS-WIDE Queries   
Figure 1: (a) An exemplar of the studied class-agnostic OD and downstream OOD-OD tasks. (B) Zero-shot class-agnostic OD performance of Grounding DINO [33] on MS-COCO [32], with the hand-crafted UNIVERSAL query from ChatGPT and CLASS-WIDE query from WordNet [14].

Conventional solutions to the under-explored class-agnostic OD task often rely on bottom-up strategies $[47, 61, 40, 41]$ such as selective search $[47]$ or EdgeBox $[19]$ , which generate a large ranked set of class-agnostic proposals based on low-level visual cues. To address the low precision and scalability issues of these approaches, another line of research has explored multi-object discovery by leveraging (self)-supervised features from vision transformers (ViT) (e.g., DINO $[39]$ , MoCo-v2 $[5]$ , SwAV $[3]$ ), or external motion information to support region proposal regression. However, these methods still fall short, achieving only about 30% average recall (AR) on benchmark datasets like MS-COCO due to the lack of intrinsic knowledge about a wide range of objects. The newly released vision-language models (VLMs) such as Grounding DINO $[33]$ , GLIP $[29]$ , T-Rex2 $[21]$ , which are pretrained on large-scale grounding datasets, have opened up new opportunities for acquiring common knowledge for generic object localization. VLMs have demonstrated impressive zero-shot recognition capacities given the provided textual prompt. However, to effectively locate all objects, one would need to input all class names accurately, which is impractical in real-world applications.

To better understand the limitation of modern VLMs in generic object localization, we investigated the design of hand-crafted text queries (Section 2) to enhance detection recall through two approaches: (1) We employed a UNIVERSAL query, using ChatGPT to generate 13 types of broad nouns and adjectives (e.g., “objects”, “generic”) as queries for the Grounding DINO model, aiming to detect a wide array of objects without focusing on specific categories; (2) We implemented a CLASS-WIDE query, selecting 25 high-level semantic words (e.g., “plant”, “animal”) from the top layer of the WordNet hierarchy (also used for the ImageNet vocabulary) to cover extensive object categories. Our findings, depicted in Figure 1b and Table 1, reveal that while VLMs can generalize across universal object categories, combining all queries into one string significantly reduces detection performance (by up to 52% in AR) due to the “semantic overlap” among words. This suggests that optimal detection requires conducting multiple separate inferences, presenting substantial computational demands for large datasets.

To overcome the aforementioned limitations, we propose a novel self-supervised Dispersing Prompt Expansion (DiPEx) strategy. This approach progressively expands a set of non-overlapping hyperspherical prompts for capturing all objects in a given dataset, thereby benefiting downstream tasks such as out-of-distribution object detection. Specifically, we start with a generic parent prompt that is self-supervised using the UNIVERSAL and CLASS-WIDE text queries. To capture more fine-grained semantics, we split the parent prompts with high semantic uncertainty into a set of distinct child prompts. We initialize child prompts by diversifying the parent token embedding, randomly rotating it to different angles on the hypersphere to yield a range of unique prompts. Dispersion losses are employed to minimize semantic overlap among child prompts while maintaining semantic consistency across parent-child prompt pairs. To prevent excessive growth of the prompt sets, we estimate the maximum angular coverage (MAC) of the semantic space as a criterion to terminate the prompt expansion process, balancing semantic richness and computational overhead. Extensive experiments on the MS-COCO and LVIS datasets verify the effectiveness and versatility of the proposed DiPEx strategy. With a single pass of inference, DiPEx can achieve by up to 20.1% improvements in average recall (particularly 35.2% for small objects) and outperforms segment anything model (SAM) [26] by 21.3% in average precision.

Table 1: Zero-shot class-agnostic object detection performance of Grounding DINO [33] on MS-COCO [32], with hand-crafted prompts from various sources. We report average recall (AR) and precision (AP) limited to a maximum of 100 detections per image. $\Delta$ AR quantifies the percentage decrease in AR comparing "query-merging" to "prediction-merging" for forming multi-word queries. 

<table><tr><td>Word Source</td><td>Merging Strategy</td><td>AR</td><td> $\Delta AR$ </td><td>AR@S</td><td>AR@M</td><td>AR@L</td><td>AP</td></tr><tr><td rowspan="2">ChatGPT [38]</td><td>query-merging</td><td>0.345</td><td rowspan="2">-52.46%</td><td>0.122</td><td>0.360</td><td>0.718</td><td>0.067</td></tr><tr><td>prediction-merging</td><td>0.526</td><td>0.317</td><td>0.606</td><td>0.781</td><td>0.274</td></tr><tr><td rowspan="2">WordNet [14]</td><td>query-merging</td><td>0.461</td><td rowspan="2">-23.64%</td><td>0.234</td><td>0.522</td><td>0.774</td><td>0.229</td></tr><tr><td>prediction-merging</td><td>0.570</td><td>0.382</td><td>0.646</td><td>0.796</td><td>0.344</td></tr><tr><td rowspan="2">ChatGPT [38]+WordNet [14]</td><td>query-merging</td><td>0.408</td><td rowspan="2">-44.36%</td><td>0.162</td><td>0.471</td><td>0.751</td><td>0.121</td></tr><tr><td>prediction-merging</td><td>0.589</td><td>0.410</td><td>0.665</td><td>0.798</td><td>0.353</td></tr></table>

Related Study. The full discussions can be found in Section A.1. Traditional bottom-up approaches for region proposal generation, such as those by $[47]$ and $[27]$ , often face precision constraints despite high recall rates, limiting their scalability. Recent advancements in Vision Transformers (ViTs) by $[4]$ and $[10]$ have enabled self-supervised learning on massive datasets, extracting semantically meaningful features. Methods like LOST $[45]$ and TokenCut $[51]$ use graph-based techniques but are limited to detecting a single object per image. MOST $[43]$ addresses this with entropy-based box analysis but struggles with generalization. MAVL $[36]$ uses a late fusion strategy with text queries, requiring full supervision and multiple inferences. Our approach eliminates the need for labels and achieves state-of-the-art performance with one-pass inference using non-overlapping prompts. Vision-Language Models (VLMs), like those by $[42]$ and $[20]$ , have shown potential in learning generic concepts. HierKD $[35]$ and OV-DETR $[58]$ align image representations with captions and extend DETR to open-vocabulary settings. GLIP $[29]$ , Grounding DINO $[33]$ , and T-Rex2 $[21]$ integrate object detection and visual grounding. However, VLMs' effectiveness depends on textual cues, and prompt tuning, as introduced by CoOp $[24]$ and improved by CoCoOp and MaPLe $[25]$ , offers a solution by optimizing soft prompts while keeping the model's parameters frozen. ProDA $[34]$ learns diverse prompts using a Gaussian model. DFKD-VLFM $[56]$ and PromptStyler $[7]$ attempted to diversify a fixed number of prompts through contrastive approach. Despite these advancements, full supervision is typically required. UPL $[17]$ and POUF $[46]$ introduced unsupervised prompt learning, but adaptation for object detection remains limited. DiPEx is the first to apply prompt learning to class-agnostic object detection through a progressive self-training approach.

# 2 Pilot Study

In this section, we detail our preliminary exploration of the zero-shot detection capabilities using state-of-the-art VLM, Grounding DINO [33], to detect all objects irrespective of the associated classes on the MS-COCO dataset [32] as illustrated in Figure 1b. We conduct experiments using two types of text queries: UNIVERSAL queries generated by ChatGPT for general object detection, and CLASS-WIDE queries derived from WordNet, representing broad object categories. Our experiments reveal that semantic overlap between text queries impacts detection performance. To support this hypothesis, we conduct a case study showing that similar concatenated prompts reduce the model's detection confidence.

# 2.1 Hand-crafted Queries for Class-agnostic Object Detection

UNIVERSAL Query. We employ ChatGPT to generate 13 synonyms of universal concepts, including nouns and adjectives, which are displayed as x-axis labels. The zero-shot object detection results, measured by average recall (AR) and precision (AP) across the top 100 confident boxes for each query text, are presented. The plot reveals that more general terms such as “generic” and “items” yield the highest AR. Surprisingly, more specific descriptors like “foreground”, “small”, or “tiny” tend to reduce AR and do not effectively aid in identifying foreground or small objects.

CLASS-WIDE Query. We utilize 25 semantically independent beginner words (listed as x-axis labels in the bottom figure) from the highest level of the WordNet hierarchy [14] as class-wide text queries. A variation in AR (0.26\~0.43) is observed with different textual queries from WordNet, with a mean AR of 0.35. Compared to the mean AR of 0.37 across class-agnostic queries generated by ChatGPT, the zero-shot detection ability remains similar, regardless of the types of queries used.

Discussion on Multi-Word Queries. The zero-shot results presented in Figure 1b were obtained using single-word prompts for the Grounding DINO. To explore whether combining multiple words

![](images/db51cb5e03f42ae5683c405e796e984e049a1da57400a070b38f3f6300cebc07.jpg)

<details>
<summary>text_image</summary>

Text Query: "plates."
Text Query: "dishes."
Text Query: "plates. dishes."
phonics 0.44
phonics 0.45
phonics 0.46
phonics 0.47
phonics 0.48
phonics 0.49
phonics 0.50
phonics 0.51
phonics 0.52
phonics 0.53
phonics 0.54
phonics 0.55
phonics 0.56
phonics 0.57
phonics 0.58
phonics 0.59
phonics 0.60
phonics 0.61
phonics 0.62
phonics 0.63
phonics 0.64
phonics 0.65
phonics 0.66
phonics 0.67
phonics 0.68
phonics 0.69
phonics 0.70
phonics 0.71
phonics 0.72
phonics 0.73
phonics 0.74
phonics 0.75
phonics 0.76
phonics 0.77
phonics 0.78
phonics 0.79
phonics 0.80
phonics 0.81
phonics 0.82
phonics 0.83
phonics 0.84
phonics 0.85
phonics 0.86
phonics 0.87
phonics 0.88
phonics 0.89
phonics 0.90
phonics 0.91
phonics 0.92
phonics 0.93
phonics 0.94
phonics 0.95
phonics 0.96
phonics 0.97
phonics 0.98
phonics 0.99
phonics 1.00
phonics 1.01
phonics 1.02
phonics 1.03
phonics 1.04
phonics 1.05
phonics 1.06
phonics 1.07
phonics 1.08
phonics 1.09
phonics 1.10
phonics 1.11
phonics 1.12
phonics 1.13
phonics 1.14
phonics 1.15
phonics 1.16
phonics 1.17
phonics 1.18
phonics 1.19
phonics 1.20
phonics 1.21
phonics 1.22
phonics 1.23
phonics 1.24
phonics 1.25
phonics 1.26
phonics 1.27
phonics 1.28
phonics 1.29
phonics 1.30
phonics 1.31
phonics 1.32
phonics 1.33
phonics 1.34
phonics 1.35
phonics 1.36
phonics 1.37
phonics 1.38
phonics 1.39
phonics 1.40
phonics 1.41
phonics 1.42
phonics 1.43
phonics 1.44
phonics 1.45
phonics 1.46
phonics 1.47
phonics 1.48
phonics 1.49
phonics 1.50
phonics 1.51
phonics 1.52
phonics 1.53
phonics 1.54
phonics 1.55
phonics 1.56
phonics 1.57
phonics 1.58
phonics 1.59
phonics 1.60
phonics 1.61
phonics 1.62
phonics 1.63
phonics 1.64
phonics 1.65
phonics 1.66
phonics 1.67
phonics 1.68
phonics 1.69
phonics 1.70
phonics 1.71
phonics 1.72
phonics 1.73
phonics 1.74
phonics 1.75
phonics 1.76
phonics 1.77
phonics 1.78
phonics 1.79
phonics 1.80
phonics 1.81
phonics 1.82
phonics 1.83
phonics 1.84
phonics 1.85
phonics 1.86
phonics 1.87
phonics 1.88
phonics 1.89
phonics 1.90
</details>

Figure 2: A case study investigating the impact of semantic overlap between text queries on the detection confidence of the pre-trained Grounding DINO [33]. Semantic overlaps are quantified by the angular distance, denoted as $\Theta$ , between tokenized embeddings of word pairs using BERT [9].

as prompts from a given source (e.g., WordNet) could improve zero-shot detection performance, we developed strategies for merging at both the input stage (query-merging) and the output stage (prediction-merging) as shown in Table 1. The query-merging strategy concatenates all input text queries (e.g., “foreground . elements . … tiny . objects .”) and performs a single-pass inference to obtain detections. The prediction-merging strategy, on the other hand, uses each text query individually for separate inference and then combines all box predictions. Table 1 shows that applying query-merging to UNIVERSAL words results in a 52.46% reduction in AR compared to prediction-merging, whereas CLASS-WIDE queries (e.g., from WordNet) achieve a smaller decrease in AR of only 23.64%. These findings suggest that large semantic overlaps in concatenated queries (e.g., “stuff”, “objects” and “item” from ChatGPT) may greatly contribute to diminished object detection performance. To further investigate this phenomenon, we conducted a case study analyzing the impact of semantic overlap on detection performance, which is presented in the following section.

# 2.2 Confidence Diminishing when Text Query Semantically Overlap

To verify our hypothesis, we conduct a case study to demonstrate how semantic overlap in multi-word query leads to diminished detection confidence. We quantify semantic overlap by calculating the angular distance between pairs of textual token embeddings generated by BERT $[9]$ . As shown in Figure 2, a small angular distance $\theta$ of $53.73^{\circ}$ between the text tokens “plates” and “dishes” diminishes the model’s confidence. Consequently, some boxes that could be precisely localized with high confidence using the single token “plates” are omitted. In contrast, concatenating two text tokens with a larger angular distance (e.g., $60.99^{\circ}$ between “plates” and “cup”) maintained high detection confidence. This combination resulted in bounding box predictions that encompassed all boxes predicted with each individual token (“plates” or “cup”). This case study supports our hypothesis that semantic overlap between concatenated text queries can interfere with the detection confidence of the model. Therefore, we propose that developing a method to learn a set of semantically non-overlapping prompts for the target dataset could enable efficient object localization with one-pass inference using VLMs.

# 3 Proposed Approach

In this section, we first mathematically formulate the task of class-agnostic detection using a general VLM and, without loss of generality, illustrate the process using Grounding DINO [33] as an exemplar model. We detail the steps of the proposed dispersing prompt expansion in Section 3.2, followed by the early termination strategy of the prompt set growth.

# 3.1 Problem Formulation

Class-agnostic OD. Let I denote the input image and T the associated text query. For the zero-shot object detection in a class-agnostic setting, we consider the text query T to be of the form of “a photo of a {class}”, where the class token {class} is sampled from our predefined UNIVERSAL (e.g., “objects”) or CLASS-WIDE (e.g., “plant”) sets as described in Section 2. The text query is then tokenized and projected into word embeddings as $P = \{v_{1}, v_{2}, \ldots, v_{M}, c\}$ , where

![](images/583eb8ae75ebe7155c9874dcaa2ccaf878a913e0cee28a2ed991563a0749960f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    V0["V0"] --> V11["V1,1"]
    V0 --> V12["V1*"]
    V0 --> V13["V1,3"]
    V11 --> V21["V2,1"]
    V12 --> V22["V2,2"]
    V13 --> V23["V2,3"]
    V21 --> ...[...]
```
</details>

L-Layer Prompt Expansion   
![](images/3a97710fcd317f7894769b1e01f2385498431e9639d6187c0c7be359904c8ba9.jpg)

<details>
<summary>text_image</summary>

dim3
v_{l+1,2}
θ₂
θ₁
v_l*
v_{l+1,1}
dim1
dim2
</details>

Child Prompt Initialization

![](images/b4ead75450ec01e966cb4ec3424221eb76f5f5046ef66956609d3d305100c17d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    v0 --> v1_1
    v1_1 --> v2_1
    v2_1 --> v2_2
    v2_2 --> v1_3
    v1_3 --> v2_3
    v2_3 --> v1_1
    style v0 fill:#000,stroke:#000
    style v1_1 fill:#000,stroke:#000
    style v2_1 fill:#000,stroke:#000
    style v2_2 fill:#000,stroke:#000
    style v1_3 fill:#000,stroke:#000
    style v2_3 fill:#000,stroke:#000
    subgraph α_max
        direction LR
        v1_1 --> v2_1
        v2_1 --> v2_2
        v1_3 --> v2_3
    end
```
</details>

Maximum Angular Coverage   
Figure 3: An illustration of the ① proposed prompt expansion strategy that selectively grows a set of child prompts for the highlighted parent prompt across L iterations; ② diversifying initialized embeddings of the child prompt on a hypersphere and ③ quantifying maximum angular coverage $\alpha_{max}$ for early termination of the prompt growth.

$v = \{v_i\}_{i=1}^M \in R^{M \times d}$ indicates a set of $M$ contextual embeddings and $c$ is the query text embedding. Here, $d$ indicates the dimensions of learnable tokens. The visual embeddings $E_v$ extracted from the visual encoder and prompt embeddings $P$ are fused jointly to prompt the VLM and generate the final bounding box predictions $O = f(E_v, P) \in R^{N_B \times 4}$ , with $f$ being the VLM, and $N_B$ being the number of predicted boxes. Formally, the objective of class-agnostic OD is to ensure that the generated bounding boxes can capture any objects as comprehensively as possible.

Adapt Prompt Tuning for Class-agnostic OD. Instead of relying on hand-crafted templates, prompt tuning approaches like CoOp [24] and CoCoOp [23], originally developed for classification tasks, aim to learn the context embeddings v with a frozen VLM using a supervised contrastive learning loss. To adapt these prompt learning approaches to the Grounding DINO [33] detection framework, we first construct a pseudo label set $D_{PSL}$ from the zero-shot detection results with UNIVERSAL and CLASS-WIDE text queries (see Section A.3 for details). The prompt learning is then supervised by the standard box regression loss $L_{box}$ , $L_{giou}$ and focal classification loss $L_{cls}$ as implemented in [33].

# 3.2 Dispersing Prompt Expansion (DiPEx)

Unlike previous prompt tuning approaches, the proposed DiPEx strategy aims to iteratively grow a set of learnable prompts $P = \{P_{1}, P_{2}, \ldots, P_{L}\}$ in a tree hierarchy of depth L. To maximize the utility of prompts and ensure minimal semantic overlap among them, we assume v resides on the surface of a unit-hypersphere, i.e., $\|v_{i}\|_{2} = 1$ . This assumption transforms the overlap minimization problem into maximizing the angular distances among the learned prompts. In the initial round, we set a single learnable parent prompt $P_{1} = \{v\}$ , which is self-trained using $D_{PSL}$ with the same procedure outlined above. In each subsequent round l for $l \in [1, L]$ , we identify the parent prompts of highest uncertainty and grow K child prompts $P_{l+1} \in R^{K \times d}$ from it. The learned $v_{l}^{*}$ is then frozen and stored in a parent queue $P_{parent}$ . Prompt growth is terminated when the maximum angular coverage $\alpha_{max}$ exceeds a certain threshold $T_{\alpha}$ .

Child Prompt Initialization. Continuing from the previous discussion, we now describe the process of child prompt initialization, which aims to inherit the semantics from parent prompts while capturing more fine-grained semantics. After the l-th round of training, we expand the parent prompt with the highest uncertainty, denoted as $v_{l}^{*} \subset P_{l}$ , into a set of learnable child prompts (Figure 3). We empirically adopt the logit activation frequency of the prompts as a measure of uncertainty, visualized in Figure 6. The rationale is that if a prompt is activated for most samples, it covers overly broad semantics (e.g., animals) and may need to be decomposed into narrower categories (e.g., cats and dogs). To disentangle the complex semantic of $P_{l}^{*}$ , we set up K child prompts $P_{l+1} = \{v_{l+1,k}\}_{k=1}^{K}$ for the selected parent prompt $v_{l}^{*}$ . To diversify the initialized embedding for each child prompt, we introduce K random angular offsets $\Theta = \{\theta_{k}\}_{k=1}^{K}$ to rotate $v_{l}^{*}$ on the hypersphere by different angles $\theta_{k} \sim [-\theta, \theta]$ . Given that $v_{l}^{*}$ is a d-dim vector, we randomly sample two axes i and j where $i, j \sim [1, d]$ for rotation. The k-th child prompt embedding $v_{l+1,k}$ is then obtained by applying the

corresponding rotation matrix $\mathfrak{R}_k\in \mathbb{R}^{d\times d}$ , which are defined as follows:

$$
\mathbf {v} _ {l + 1, k} = \mathbf {v} _ {l} ^ {*} \Re_ {k}, \quad \Re_ {k} = \left[ \begin{array}{c c c c c c c} 1 & \dots & 0 & \dots & 0 & \dots & 0 \\ \vdots & \ddots & \vdots & & \vdots & & \vdots \\ 0 & \dots & \cos \theta_ {k} & \dots & - \sin \theta_ {k} & \dots & 0 \\ \vdots & & \vdots & 1 & \vdots & & \vdots \\ 0 & \dots & \sin \theta_ {k} & \dots & \cos \theta_ {k} & \dots & 0 \\ \vdots & & \vdots & & \vdots & \ddots & \vdots \\ 0 & \dots & 0 & \dots & 0 & \dots & 1 \end{array} \right]. \tag {1}
$$

Here, the non-identity elements are placed at the intersections of the i-th and j-th rows and columns, corresponding to the plane of rotation, as illustrated by the grey ellipses in Figure 3. As the initialized embeddings of the child prompts are diversified while maintaining consistency with the central parent embedding (red dot), this leads to varying detection results. This enriched prediction diversity allows us to facilitate online self-training, where we adopt the predictions with the highest confidence as pseudo labels for each child prompt, which in turn supervise the next round of prompt learning with respect to $L_{bbox}$ , $L_{giou}$ and $L_{cls}$ for the next iteration.

Optimization. We expect the learned child prompts to follow an accurate semantic hierarchy, having minimal overlap with other child tokens while maintaining semantic consistency with their original parent prompts. We leverage the following dispersion losses to enlarge the angular distances among the child-child and decrease the distances between child-parent prompt pairs:

$$
\mathcal {L} _ {\text { parent - child }} = - \frac {1}{K} \sum_ {i = 1} ^ {K} \left(\frac {\mathbf {v} _ {i} ^ {\top} \mathbf {v} _ {l} ^ {*}}{\| \mathbf {v} _ {i} \| \| \mathbf {v} _ {l} ^ {*} \|} / \tau_ {p}\right), \tag {2}
$$

$$
\mathcal {L} _ {\mathrm{child-child}} = \frac {1}{K} \sum_ {i = 1} ^ {K} \log \frac {1}{K - 1} \sum_ {j \neq i} \exp (\frac {\mathbf {v} _ {i} ^ {\top} \mathbf {v} _ {j}}{\| \mathbf {v} _ {i} \| \| \mathbf {v} _ {j} \|} / \tau_ {c}),
$$

where $v_{l}^{*}$ is retrieved from the parent prompt queue $P_{parent}$ as a fixed prototype. The temperature coefficients $\tau_{p}$ and $\tau_{c}$ adjust the angular separation. The overall optimization can be formulated as:

$$
\mathcal {L} = \mathcal {L} _ {\text { parent - child }} + \gamma \mathcal {L} _ {\text { child - child }} + \gamma_ {\text { bbox }} \mathcal {L} _ {\text { bbox }} + \gamma_ {\text { giou }} \mathcal {L} _ {\text { giou }} + \gamma_ {\text { cls }} \mathcal {L} _ {\text { cls }}, \tag {3}
$$

where $\gamma$ is the loss coefficient that controls the $L_{child-child}$ . The rest coefficients i.e., $\gamma_{bbox}$ , $\gamma_{giou}$ , $\gamma_{cls}$ follows [33]. Until the optimization convergence, the prompt expansion will repeat if needed.

Expansion Termination with Maximum Angular Coverage (MAC). While prompt expansion is effective in capturing fine-grained semantics, it inevitably introduces computational overhead, impacting inference efficiency for downstream tasks. To balance the semantic richness and inference costs, we gather all learned prompts P and evaluate the maximum angular coverage (MAC) among all pairs. MAC is defined as:

$$
\alpha_ {\max} = \max _ {\mathbf {v} _ {i}, \mathbf {v} _ {j} \in \mathbf {P}} \arccos (\frac {\mathbf {v} _ {i} ^ {\top} \mathbf {v} _ {j}}{\| \mathbf {v} _ {i} \| \| \mathbf {v} _ {j} \|}). \tag {4}
$$

The $\alpha_{max}$ reveals the breadth of vocabularies covered by the current prompts. Notably, our empirical study shows that as the number of expansion rounds increases, the MAC increases monotonically and eventually converges. This convergence serves as an effective signal to terminate prompt expansion. The overall algorithm is summarized in Algorithm 1.

# 4 Experiments

# 4.1 Experimental Setup

Datasets. We conduct our experiments using two detection datasets: 1). MS-COCO [32], a large-scale object detection and instance segmentation dataset, comprising approximately 115K training images and 5K validation images across 80 classes. 2). LVIS [15] includes 2.2 million high-quality instance segmentation masks covering 1,000 class labels, resulting in a long-tailed data distribution. It consists of around 100K training images and 19.8K validation images. For class-agnostic object detection (CA-OD) setting, we merge all categories from both datasets into a single class to perform class-agnostic detection. To further validate the efficacy of DiPEx in downstream out-of-distribution object detection (OOD-OD) tasks, we evaluate our method using a rectified version of the OOD-OD benchmark. Unlike previous benchmarks [12], where samples that do not contain ID instances are manually selected and ID and OOD performance are evaluated separately, we tested our approach

Algorithm 1 The Proposed DiPEx for Class-Agnostic Object Detection   
Input: f: vision-language model (VLM)
Output: P: set of fine-tuned prompts for f to detect class-agnostic objects
    Initialize a single learnable parent prompt $P_{1} = \{v_{1}\}$ Optimize $P_{1}$ using zero-shot detection results from f
    Initialize a growing set of learnable prompts $P = \{P_{1}\}$ and an empty parent queue $P_{parent} = \{\}$ for each round $l \in \{1, 2, \cdots, L\}$ do
    Identify the parent prompt with the highest uncertainty $v_{l}^{*} \in P_{l}$ Freeze $v_{l}^{*}$ and add it to the parent queue $P_{parent}$ Expand $v_{l}^{*}$ into K learnable child prompts $P_{l+1} = \{v_{l+1,k}\}_{k=1}^{K}$ via Equation (1)
    Grow the set of learnable prompts: $P = P_{l} \cup P_{l+1}$ Optimize the prompts in P using Equation (3) with $P_{parent}$ Compute maximum angular coverage (MAC) via Equation (4)
    if MAC converges then
    Break; terminate the prompt growth
    end if
end for

Table 2: Class-agnostic object detection on the MS-COCO dataset. [ ] indicate the prompt word for Grounding DINO. The prompting methods indicated with ‘\*’ are adapted to the OD task. 

<table><tr><td>Method</td><td>Description</td><td>AR1</td><td>AR10</td><td>AR100</td><td>AR@S</td><td>AR@M</td><td>AR@L</td><td>AP</td></tr><tr><td>Selective Search [47]</td><td>non-parametric</td><td>0.1</td><td>1.1</td><td>7.8</td><td>0.9</td><td>7.2</td><td>20.7</td><td>0.1</td></tr><tr><td>UP-DETR [8]</td><td>self-training</td><td>0.2</td><td>1.4</td><td>1.4</td><td>0.0</td><td>0.2</td><td>5.8</td><td>0.1</td></tr><tr><td>DETReg [1]</td><td>self-training</td><td>0.6</td><td>3.7</td><td>12.9</td><td>0.2</td><td>12.8</td><td>35.3</td><td>1.4</td></tr><tr><td>FreeSOLO [49]</td><td>self-training</td><td>3.7</td><td>9.7</td><td>12.6</td><td>0.5</td><td>12.3</td><td>34.1</td><td>4.2</td></tr><tr><td>Exemplar-FreeSOLO [18]</td><td>self-training</td><td>8.2</td><td>13.0</td><td>17.9</td><td>-</td><td>-</td><td>-</td><td>12.6</td></tr><tr><td>MOST [43]</td><td>self-training</td><td>3.1</td><td>6.4</td><td>6.4</td><td>0.1</td><td>1.6</td><td>24.5</td><td>3.3</td></tr><tr><td>CutLER [50]</td><td>self-training</td><td>6.8</td><td>19.6</td><td>32.8</td><td>13.7</td><td>37.5</td><td>60.0</td><td>29.6</td></tr><tr><td>Grounding DINO [&quot;generic&quot;] [9]</td><td>zero-shot</td><td>10.3</td><td>37.8</td><td>44.1</td><td>17.7</td><td>51.6</td><td>80.0</td><td>28.3</td></tr><tr><td>Grounding DINO+CoOp* [24]</td><td>self-training</td><td>10.4</td><td>39.1</td><td>61.3</td><td>36.4</td><td>72.7</td><td>88.8</td><td>34.6</td></tr><tr><td>Grounding DINO+CoCoOp* [23]</td><td>self-training</td><td>7.6</td><td>34.1</td><td>58.1</td><td>33.9</td><td>68.3</td><td>86.1</td><td>24.6</td></tr><tr><td>DiPEx</td><td>self-training</td><td>10.5</td><td>40.8</td><td>63.2</td><td>39.2</td><td>74.3</td><td>89.8</td><td>35.9</td></tr></table>

on the MS-COCO, which includes a mixture of both ID and OOD objects. While we followed the settings outlined in OOD-OD [12], with 20 base classes in PASCAL-VOC [13] designated as ID classes and the remaining classes treated as OOD. Our choice of dataset enhances the rigor of our evaluation by combining both ID and OOD instances, providing a more realistic assessment of our method's real-world conditions.

Evaluation Metrics. We report results for class-agnostic object detection on both the MS-COCO and LVIS validation splits. For evaluation, we adopt official metrics from the COCO 2017 challenge. Specifically, we report average precision (AP) at IoU thresholds from 0.5 to 0.95, along with average recall (AR) across the same threshold range. We also report AR by object scale: AR@S for small, AR@M for medium, and AR@L for large objects. Details on our implementation, including those of prior works used as baselines, are provided in Appendix A.2.

# 4.2 Main Results on Class-agnostic OD and OOD-OD

Class-agnostic OD on MS-COCO. To validate our proposed method for class-agnostic object detection, we compared it against ten different baseline methods on the MS-COCO dataset, using various metrics as reported in Table 2. We observed that non-parametric methods generally underperform compared to self-training methods due to their inability to learn and extract semantic and geometric information about objects from the dataset. In contrast, Grounding DINO, leveraging pre-trained knowledge, demonstrates strong zero-shot capabilities and achieves AR $_{100}$ of 44.1% with a single text prompt, “generic”. Furthermore, CoOp, which fine-tunes prompts for Grounding DINO, enhances class-agnostic detection performance by 39.0% in AR $_{100}$ compared to direct zero-shot inference. Our method, which expands the learnable prompts to a wider angular distance, surpasses all baselines by achieving the highest performance across all metrics and outperforming the leading baseline, CoOp, by 3.1% in AR $_{100}$ . Notably, for small objects which are challenging to localize, our method improves

Table 3: Class-agnostic object detection on the LVIS dataset. $^{\dagger}$ indicate the model is fine-tuned on the LVIS training set by self-training without box annotations. 

<table><tr><td>Method</td><td>AR1</td><td>AR10</td><td>AR200</td><td>AR@S</td><td>AR@M</td><td>AR@L</td><td>AP</td><td>AP@S</td><td>AP@M</td><td>AP@L</td></tr><tr><td>Selective Search [47]</td><td>0.1</td><td>1.1</td><td>13.0</td><td>6.1</td><td>19.9</td><td>37.6</td><td>0.2</td><td>0.4</td><td>0.2</td><td>0.2</td></tr><tr><td>G-DINO [“object”] [9]</td><td>4.1</td><td>17.9</td><td>27.2</td><td>13.0</td><td>44.1</td><td>71.1</td><td>5.4</td><td>5.6</td><td>10.0</td><td>9.4</td></tr><tr><td>G-DINO [“generic”] [9]</td><td>3.8</td><td>16.5</td><td>20.2</td><td>6.5</td><td>34.5</td><td>67.7</td><td>9.0</td><td>4.1</td><td>17.4</td><td>30.7</td></tr><tr><td>G-DINO [“items”] [9]</td><td>4.0</td><td>17.8</td><td>28.0</td><td>13.9</td><td>45.3</td><td>70.7</td><td>11.6</td><td>6.3</td><td>19.6</td><td>32.0</td></tr><tr><td>SAM [26]</td><td>-</td><td>-</td><td>42.7</td><td>27.7</td><td>66.3</td><td>75.5</td><td>6.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $^\dagger$  CutLER [50]</td><td>2.4</td><td>9.3</td><td>21.8</td><td>10.8</td><td>35.1</td><td>55.5</td><td>4.5</td><td>2.7</td><td>9.1</td><td>15.1</td></tr><tr><td> $^\dagger$  HASSOD [2]</td><td>0.2</td><td>10.6</td><td>26.9</td><td>15.6</td><td>42.2</td><td>56.9</td><td>4.9</td><td>2.8</td><td>7.9</td><td>12.2</td></tr><tr><td> $^\dagger$  G-DINO + CoOp* [24]</td><td>4.2</td><td>19.1</td><td>40.3</td><td>23.6</td><td>63.5</td><td>83.5</td><td>14.0</td><td>8.3</td><td>23.7</td><td>32.3</td></tr><tr><td> $^\dagger$  G-DINO + CoCoOp* [23]</td><td>4.2</td><td>19.2</td><td>40.7</td><td>24.1</td><td>63.8</td><td>84.1</td><td>13.6</td><td>8.1</td><td>22.4</td><td>30.1</td></tr><tr><td> $^\dagger$  DiPEx</td><td>4.3</td><td>20.1</td><td>48.4</td><td>31.9</td><td>72.6</td><td>88.2</td><td>15.2</td><td>9.3</td><td>25.3</td><td>32.8</td></tr></table>

Table 4: The downstream out-of-distribution object detection (OOD-OD) on the MS-COCO dataset, where the ground truth boxes contain both known and unknown classes. 

<table><tr><td rowspan="2">Method</td><td colspan="2">KNOWN</td><td colspan="8">UNKNOWN</td></tr><tr><td>AP</td><td>AP50</td><td> $AR_{100}$ </td><td>AR@S</td><td>AR@M</td><td>AR@L</td><td>AP</td><td>AP@S</td><td>AP@M</td><td>AP@L</td></tr><tr><td>Selective Search [47]</td><td>-</td><td>-</td><td>8.3</td><td>1.0</td><td>8.5</td><td>23.2</td><td>0.1</td><td>0.0</td><td>0.0</td><td>0.5</td></tr><tr><td>MOST [43]</td><td>-</td><td>-</td><td>5.3</td><td>0.1</td><td>1.3</td><td>22.5</td><td>0.4</td><td>0.1</td><td>0.4</td><td>1.2</td></tr><tr><td>CutLER [50]</td><td>-</td><td>-</td><td>34.5</td><td>15.8</td><td>41.5</td><td>62.7</td><td>5.7</td><td>2.3</td><td>6.9</td><td>13.7</td></tr><tr><td>VOS [12]</td><td>36.6</td><td>56.7</td><td>10.0</td><td>2.2</td><td>6.1</td><td>27.1</td><td>2.8</td><td>0.8</td><td>2.2</td><td>7.2</td></tr><tr><td>PROB [62]</td><td>28.2</td><td>43.8</td><td>13.2</td><td>1.9</td><td>11.2</td><td>40.3</td><td>0.9</td><td>0.6</td><td>0.9</td><td>2.1</td></tr><tr><td>UnSniffer [30]</td><td>35.8</td><td>55.8</td><td>20.6</td><td>11.8</td><td>19.9</td><td>34.8</td><td>2.9</td><td>1.5</td><td>3.1</td><td>5.3</td></tr><tr><td>G-DINO [“generic”]</td><td>46.3</td><td>59.7</td><td>43.3</td><td>18.0</td><td>52.1</td><td>82.6</td><td>12.5</td><td>6.9</td><td>17.8</td><td>25.7</td></tr><tr><td>DiPEx</td><td>46.3</td><td>59.7</td><td>59.9</td><td>35.8</td><td>72.9</td><td>89.7</td><td>15.7</td><td>9.7</td><td>21.8</td><td>25.2</td></tr></table>

AR@S by 7.7% compared to CoOp, indicating that expanded prompts better capture a range of object sizes. Additionally, the proposed DiPEx achieved the highest AP of 35.9%, demonstrating the superior quality of class-agnostic detection.

Class-agnostic OD on LVIS. To further validate the efficacy of DiPEx, we conducted extensive experiments on the challenging LVIS dataset, which includes thousands of classes with a long-tail distribution. As shown in Table 3, prompt tuning methods such as CoOp [24] and CoCoOp [23] outperform zero-shot Grounding DINO when using hand-crafted prompts (e.g., “items”, “generic”, “objects”). Additionally, CoCoOp surpasses multi-object discovery baselines like CutLER [50] and HASSOD [2], by 86.7% and 51.3% in AR $_{200}$ , respectively. Notably, SAM [26], which was pretrained on a vast of dataset containing millions of images and billions of masks, demonstrates strong zero-shot capabilities, surpassing all other baselines. In contrast, our proposed DiPEx outperforms SAM by 13.3% in AR $_{200}$ and 21.3% in AP after only four epochs of self-training, Furthermore, DiPEx exceeds CoOp by 20.1% in AR $_{200}$ .

Downstream OOD-OD on MS-COCO. To evaluate the generalization of our proposed DiPEx in out-of-distribution object detection (OOD-OD), we compared its performance on both known and unknown classes against various baselines. As shown in Table 4, the zero-shot Grounding DINO uses known class names as prompts, supplemented with a simple “generic” prompt for unknowns, outperforms all other non-VLM methods (e.g., 25.5% higher AR $_{100}$ compared to CutLER [50]). This improvement stems from VLMs leveraging rich semantic knowledge from language models to better comprehend object information in images. DiPEx enhances this further by expanding text prompts in embedding space, enabling it to capture and differentiate objects of varying sizes and diverse semantics from learned classes. This approach delivers a significant performance gain, achieving a 38.3% increase in AR $_{100}$ and a 25.6% in AP increase over zero-shot predictions. Furthermore, the expanded prompts can be directly applied alongside various known class vocabularies to detect unknown objects, eliminating the need for retraining.

# 4.3 Ablation Study and Model Analysis

We investigate the impact of various factors on prompting performance including the learnable prompt lengths, the number of expansion rounds L, and angular coverage achieved across rounds. To facilitate model analysis, we present the distribution of prompt logit activation and visualization of detection results. Further ablation studies refers to Section A.3.

![](images/5438536235b7c586875402f57d8c4c5af1d04013223e432b7f632c05cdc52834.jpg)

<details>
<summary>bar</summary>

| Number of Prompts (N) | DiPEx  | CoOp   | CoCoOp |
| --------------------- | ------ | ------ | ------ |
| 3                     | 0.630  | 0.612  | 0.606  |
| 5                     | 0.630  | 0.612  | 0.594  |
| 7                     | 0.630  | 0.610  | 0.585  |
| 9                     | 0.634  | 0.612  | 0.582  |
</details>

![](images/693aa97ec55d43a87e869fee5140c3707e0702201c01b14444ddc4de2da270bc.jpg)

<details>
<summary>bar</summary>

| Number of Prompts (N) | DiPEx  | CoOp   | CoCoOp |
| --------------------- | ------ | ------ | ------ |
| 3                     | 0.348  | 0.346  | 0.272  |
| 5                     | 0.342  | 0.344  | 0.318  |
| 7                     | 0.348  | 0.344  | 0.268  |
| 9                     | 0.358  | 0.346  | 0.246  |
</details>

Figure 4: Impact of the prompt length on the MS-COCO dataset. The average recall (AR) and precision (AP) are reported to compare the derived DiPEx against CoOp [24] and CoCoOp [23].

![](images/9c12bdf1cd0ccf1fee42f7a44d3727931e7a9e2e9e06032008f12492580a8f02.jpg)

<details>
<summary>heatmap</summary>

| Prompts \ Prompts | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 65 | 48 | 58 | 51 | 60 | 61 | 64 | 56 |
| 1 | 65 | 0 | 58 | 68 | 58 | 67 | 60 | 67 | 56 |
| 2 | 48 | 58 | 0 | 45 | 40 | 53 | 46 | 49 | 50 |
| 3 | 58 | 68 | 45 | 0 | 40 | 58 | 39 | 43 | 58 |
| 4 | 51 | 58 | 40 | 40 | 0 | 56 | 40 | 43 | 51 |
| 5 | 60 | 67 | 53 | 58 | 56 | 0 | 54 | 56 | 58 |
| 6 | 61 | 60 | 46 | 39 | 40 | 54 | 0 | 40 | 55 |
| 7 | 64 | 67 | 49 | 43 | 43 | 56 | 40 | 0 | 56 |
| 8 | 56 | 56 | 50 | 58 | 51 | 58 | 55 | 56 | 0 |
</details>

![](images/cff9b0c5051338680010d41ae8368f32f8d3303908a64ba0ba3efd553cd8a725.jpg)

![](images/3e94daeaeccf11a665ed8665e1b95aed6eff12d46512cce625cb96fe8c506554.jpg)  
Figure 5: The heatmap visualization presents the angular coverage across all learned prompts through the 2nd, the 3rd, and the 4th round of training. The maximum angular coverage (MAC) monotonically increases from $67.7^{\circ}$ in the 2nd round to $75.95^{\circ}$ in the final round. The gradual reduction in rate of change in angular coverage towards the final round suggests that the model nearing convergence.

Impact on Number of Prompts. In Figure 4, we compare the impact of prompt length N for DiPEx against CoOp [24] & CoCoOp [23]. Overall, DiPEx shows consistent improvement in performance with a greater number of prompts – not merely due to quantity, but rather because a larger set fosters greater diversification, enabling the model to capture more comprehensive semantics. In contrast, CoOp's [24] performance remains constant, while CoCoOp's [23] performance declines, suggesting that more prompts do not necessarily guarantee enhanced performance.

Impact on Expansion Rounds and Angular Coverage. To substantiate our hypothesis that a higher maximum angular coverage (MAC) correlates with a broader spectrum of vocabularies, we computed the MAC using Equation (4). The coverage results are visualized as heatmaps in Figure 5. At the initial stage of expansion (leftmost heatmap), we observe that the prompts are quite uniformly distributed, with a mean coverage of $47.56^{\circ}$ , This suggests that the prompts are actively exploring the embedding space to capture diverse semantics. As the expansion progresses to the third round (middle heatmap), the MAC increases from $67.78^{\circ}$ to $75.70^{\circ}$ . Specifically, row/col 7 (selected parent prompt) demonstrates the closest angular distances among the child prompts. This observation is crucial as it suggests that child prompts should not diverge excessively from the root semantics to maintain coherence. By the fourth round of expansion (rightmost heatmap), the pattern remains consistent with the third round. There is a reduced rate of change of MAC, achieving a maximum coverage of $75.95^{\circ}$ and a mean coverage of $11.51^{\circ}$ among the child prompts. This plateau in MAC indicates that maximum semantic expansion has been reached, suggesting that the model is approaching convergence and further expansion may not be necessary.

The Distribution of Prompt Logit Activation. We previously established prompt logit activation frequency as an uncertainty measure to guide parent prompt selection for splitting. To investigate the dynamics of expanding highly uncertain parent prompts, we visualize the activation statistics (i.e., the frequency of logit activations) of tokens within the 2nd and 3rd expansion rounds. As illustrated in Figure 6, the distribution of these logits exhibits a long-tailed pattern, suggesting substantial uncertainty and numerous semantic overlaps among the mined semantics. The figure on the right demonstrates that, following the expansion of highly activated prompts, the distribution of

![](images/ef4ee856dd502c7dc861442964da3c69de7fb5437b2320e2ff86dcbe15024c05.jpg)

<details>
<summary>bar</summary>

| Expanded Child Prompts | Activation Frequency |
| ---------------------- | -------------------- |
| 1                      | 0.140                |
| 2                      | 0.120                |
| 3                      | 0.125                |
| 4                      | 0.115                |
| 5                      | 0.110                |
| 6                      | 0.105                |
| 7                      | 0.100                |
| 8                      | 0.095                |
| 9                      | 0.090                |
| 10                     | 0.085                |
</details>

![](images/3e6dfad1a2ac8e1d96a1a09399caf65828a95711ded0916d36c1bebf187f4443.jpg)

<details>
<summary>bar</summary>

| Expanded Child Prompts | Activation Frequency |
| ---------------------- | -------------------- |
| 1                      | 0.075                |
| 2                      | 0.065                |
| 3                      | 0.060                |
| 4                      | 0.058                |
| 5                      | 0.057                |
| 6                      | 0.056                |
| 7                      | 0.055                |
| 8                      | 0.054                |
| 9                      | 0.053                |
| 10                     | 0.052                |
| 11                     | 0.051                |
| 12                     | 0.050                |
| 13                     | 0.049                |
| 14                     | 0.048                |
| 15                     | 0.047                |
| 16                     | 0.046                |
| 17                     | 0.045                |
| 18                     | 0.044                |
| 19                     | 0.043                |
| 20                     | 0.042                |
| 21                     | 0.041                |
| 22                     | 0.040                |
| 23                     | 0.039                |
| 24                     | 0.038                |
| 25                     | 0.037                |
| 26                     | 0.036                |
| 27                     | 0.035                |
| 28                     | 0.034                |
| 29                     | 0.033                |
| 30                     | 0.032                |
| 31                     | 0.031                |
| 32                     | 0.030                |
| 33                     | 0.029                |
| 34                     | 0.028                |
| 35                     | 0.027                |
| 36                     | 0.026                |
| 37                     | 0.025                |
| 38                     | 0.024                |
| 39                     | 0.023                |
| 40                     | 0.022                |
| 41                     | 0.021                |
| 42                     | 0.020                |
| 43                     | 0.019                |
| 44                     | 0.018                |
| 45                     | 0.017                |
| 46                     | 0.016                |
| 47                     | 0.015                |
| 48                     | 0.014                |
| 49                     | 0.013                |
| 50                     | 0.012                |
| 51                     | 0.011                |
| 52                     | 0.010                |
| 53                     | 0.009                |
| 54                     | 0.008                |
| 55                     | 0.007                |
| 56                     | 0.006                |
| 57                     | 0.005                |
| 58                     | 0.004                |
| 59                     | 0.003                |
| 60                     | 0.002                |
| 61                     | 0.001                |
| 62                     | 0.001                |
| 63                     | 0.001                |
| 64                     | 0.001                |
| 65                     | 0.001                |
| 66                     | 0.001                |
| 67                     | 0.001                |
| 68                     | 0.001                |
| 69                     | 0.001                |
| 70                     | 0.001                |
| 71                     | 0.001                |
| 72                     | 0.001                |
| 73                     | 0.001                |
| 74                     | 0.001                |
| 75                     | 0.001                |
| 76                     | 0.001                |
| 77                     | 0.001                |
| 78                     | 0.001                |
| 79                     | 0.001                |
| 80                     | 0.001                |
| 81                     | 0.001                |
| 82                     | 0.001                |
| 83                     | 0.001                |
| 84                     | 0.001                |
| 85                     | 0.001                |
| 86                     | 0.001                |
| 87                     | 0.001                |
| 88                     | 0.001                |
| 89                     | 0.001                |
| 90                     | 0.001                |
| 91                     | 0.001                |
| 92                     | 0.001                |
| 93                     | 0.001                |
| 94                     | 0.001                |
| 95                     | 0.001                |
| 96                     | 0.001                |
| 97                     | 0.001                |
| 98                     | 0.001                |
| 99                     | 0.001                |
| Note: The actual values may vary slightly due to the random nature of the data generation process (e.g., random number generation) and the specific number of prompts (e.g., random number generation). The provided values are placeholders or not explicitly stated in the code.
</details>

Figure 6: The distribution of logit activation of the learned prompts in the 2nd round (left) and the 3rd round (right). The prompt of the highest activation frequency is identified for further expansion.

![](images/527dcf0d45ca0d85f5d0f18d17447f8168d75a4c9945acd7947965565e32eea2.jpg)

<details>
<summary>text_image</summary>

MOST
CutLER
Zero-Shot G-DINO
DiPEx
Ground-Truth
</details>

Figure 7: Visualization of the class-agnostic detection performance by baselines and the proposed DiPEx on MS-COCO [32]. More visualizations are provided in Appendix (Figures 9 and 10).

child prompts becomes more uniform, suggesting the discovery of fine-grained semantics. These observations support our choice of uncertainty measure and verify the validity of DiPEx, indicating that expanding based on highly uncertain parent prompts effectively alleviates semantic ambiguity.

Qualitative Study. In this section, we present visualized class-agnostic box predictions on images sampled from the MS-COCO dataset $[32]$ , as shown in Figure 7. The proposed DiPEx method demonstrates a superior ability to detect more bounding boxes than all baseline methods, particularly for small objects. For example, people in the distance (rows 1 and 3) and some bonsai (row 2) are missed by all baselines but successfully detected by DiPEx, showcasing its strong capability in localizing challenging small objects. For large objects, such as a motorcycle (row 3) and two people shaking hands in the near distance (row 1), DiPEx localizes them with significantly higher confidence compared to the zero-shot predictions of Grounding DINO using the prompt “generic”. Additionally, DiPEx successfully identifies objects that are not annotated in the MS-COCO ground truth, such as plates (row 1), a pillowcase (row 2), and a frame on the wall (row 2). This highlights DiPEx’s ability to identify a comprehensive set of class-agnostic objects, even those missed in human annotations.

# 5 Conclusion and Limitations

This work introduces DiPEx, a novel self-supervised dispersing prompt expansion approach for class-agnostic object detection. We demonstrate through comprehensive experiments and analysis that DiPEx effectively detects a wide range of unseen objects of varying sizes and achieves broad vocabulary coverage. The progressively expanded prompt sets maintain good angular distances, promoting the formation of a semantic hierarchy and facilitating downstream detection tasks with a single inference pass. While the proposed DiPEx does not rely on box annotations, it requires self-training on the entire dataset for each round of prompt expansion, resulting in increased computational overhead. Additionally, some hyperparameters like temperature coefficients $\tau_{p}$ , $\tau_{c}$ and learnable prompt length K, may require manual tuning for optimal performance. Future research directions include exploring methods to learn hierarchical prompts at once rather than through expansion. Extensive benchmarking on additional downstream tasks, such as open-vocabulary and open-world detection, is necessary to comprehensively validate the proposed approach.

# Acknowledgments and Disclosure of Funding

This research is partially supported by the Australian Research Council (DE240100105, DP240101814, DP230101196)

# References

[1] Amir Bar, Xin Wang, Vadim Kantorov, Colorado J. Reed, Roei Herzig, Gal Chechik, Anna Rohrbach, Trevor Darrell, and Amir Globerson. Detreg: Unsupervised pretraining with region priors for object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14585–14595, 2022.   
[2] Shengcao Cao, Dhiraj Joshi, Liangyan Gui, and Yu-Xiong Wang. HASSOD: hierarchical adaptive self-supervised object detection. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2023.   
[3] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2020.   
[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 9630–9640, 2021.   
[5] Xinlei Chen, Haoqi Fan, Ross B. Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. CoRR, abs/2003.04297, 2020.   
[6] Ming-Ming Cheng, Ziming Zhang, Wen-Yan Lin, and Philip H. S. Torr. BING: binarized normed gradients for objectness estimation at 300fps. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3286–3293, 2014.   
[7] Junhyeong Cho, Gilhyun Nam, Sungyeon Kim, Hunmin Yang, and Suha Kwak. Promptstyler: Prompt-driven style generation for source-free domain generalization. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 15656–15666. IEEE, 2023.   
[8] Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen. UP-DETR: unsupervised pre-training for object detection with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1601–1610, 2021.   
[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 4171–4186, 2019.   
[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.   
[11] Xuefeng Du, Gabriel Gozum, Yifei Ming, and Yixuan Li. SIREN: shaping representations for detecting out-of-distribution objects. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2022.   
[12] Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li. VOS: learning what you don't know by virtual outlier synthesis. In International Conference on Learning Representations (ICLR), 2022.   
[13] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John M. Winn, and Andrew Zisserman. The pascal visual object classes (VOC) challenge. International Journal of Computer Vision (IJCV), 88(2):303–338, 2010.

[14] Christiane Fellbaum. WordNet: An electronic lexical database. MIT press, 1998.   
[15] Agrim Gupta, Piotr Dollár, and Ross B. Girshick. LVIS: A dataset for large vocabulary instance segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5356–5364, 2019.   
[16] Weizhen He, Weijie Chen, Binbin Chen, Shicai Yang, Di Xie, Luojun Lin, Donglian Qi, and Yueting Zhuang. Unsupervised prompt tuning for text-driven object detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 2651–2661. IEEE, 2023.   
[17] Tony Huang, Jack Chu, and Fangyun Wei. Unsupervised prompt learning for vision-language models. CoRR, abs/2204.03649, 2022.   
[18] Taoseef Ishtiak, Qing En, and Yuhong Guo. Exemplar-freesolo: Enhancing unsupervised instance segmentation with exemplars. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15424–15433, 2023.   
[19] Ayush Jaiswal, Yue Wu, Pradeep Natarajan, and Premkumar Natarajan. Class-agnostic object detection. In IEEE Winter Conference on Applications of Computer Vision, pages 918–927, 2021.   
[20] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning (ICML), volume 139, pages 4904–4916, 2021.   
[21] Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-rex2: Towards generic object detection via text-visual prompt synergy. CoRR, abs/2403.14610, 2024.   
[22] K. J. Joseph, Salman H. Khan, Fahad Shahbaz Khan, and Vineeth N. Balasubramanian. Towards open world object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5830–5840, 2021.   
[23] Zhou Kaiyang, Yang Jingkang, Loy Chen Change, and Liu Ziwei. Conditional prompt learning for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[24] Zhou Kaiyang, Yang Jingkang, Loy Chen Change, and Liu Ziwei. Learning to prompt for vision-language models. International Journal of Computer Vision (IJCV), 2022.   
[25] Muhammad Uzair Khattak, Hanoona Abdul Rasheed, Muhammad Maaz, Salman H. Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19113–19122. IEEE, 2023.   
[26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 3992–4003, 2023.   
[27] Philipp Krähenbühl and Vladlen Koltun. Geodesic object proposals. In European Conference on Computer Vision (ECCV), volume 8693, pages 725–739, 2014.   
[28] Liangqi Li, Jiaxu Miao, Dahu Shi, Wenming Tan, Ye Ren, Yi Yang, and Shiliang Pu. Distilling DETR with visual-linguistic knowledge for open-vocabulary object detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 6478–6487, 2023.   
[29] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10955–10965, 2022.   
[30] Wenteng Liang, Feng Xue, Yihao Liu, Guofeng Zhong, and Anlong Ming. Unknown sniffer for object detection: Don't turn a blind eye to unknown objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3230–3239. IEEE, 2023.

[31] Chuang Lin, Peize Sun, Yi Jiang, Ping Luo, Lizhen Qu, Gholamreza Haffari, Zehuan Yuan, and Jianfei Cai. Learning object-language alignments for open-vocabulary object detection. In International Conference on Learning Representations (ICLR), 2023.   
[32] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In European Conference on Computer Vision (ECCV), volume 8693, pages 740–755, 2014.   
[33] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2023.   
[34] Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, and Xinmei Tian. Prompt distribution learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5196–5205. IEEE, 2022.   
[35] Zongyang Ma, Guan Luo, Jin Gao, Liang Li, Yuxin Chen, Shaoru Wang, Congxuan Zhang, and Weiming Hu. Open-vocabulary one-stage detection with hierarchical visual-language knowledge distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14054–14063. IEEE, 2022.   
[36] Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Fahad Shahbaz Khan, Rao Muhammad Anwer, and Ming-Hsuan Yang. Class-agnostic object detection with multi-modal transformer. In European Conference on Computer Vision (ECCV), volume 13670, pages 512–531, 2022.   
[37] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. Deep spectral methods: A surprisingly strong baseline for unsupervised semantic segmentation and localization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8354–8365. IEEE, 2022.   
[38] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.   
[39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. CoRR, abs/2304.07193, 2023.   
[40] Pedro H. O. Pinheiro, Ronan Collobert, and Piotr Dollár. Learning to segment object candidates. In Annual Conference on Neural Information Processing Systems (NeurIPS), pages 1990–1998, 2015.   
[41] Jordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Ferran Marqués, and Jitendra Malik. Multiscale combinatorial grouping for image segmentation and object proposal generation. IEEE Trans. Pattern Anal. Mach. Intell., 39(1):128–140, 2017.   
[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), volume 139, pages 8748–8763, 2021.   
[43] Sai Saketh Rambhatla, Ishan Misra, Rama Chellappa, and Abhinav Shrivastava. MOST: multiple object localization with self-supervised transformers for object discovery. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 15777–15788, 2023.   
[44] Jianbo Shi and Jitendra Malik. Normalized cuts and image segmentation. IEEE Trans. Pattern Anal. Mach. Intell., 22(8):888–905, 2000.   
[45] Oriane Siméoni, Gilles Puy, Huy V. Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet, and Jean Ponce. Localizing objects with self-supervised transformers and no labels. In British Machine Vision Conference (BMVC), page 310, 2021.

[46] Korawat Tanwisuth, Shujian Zhang, Huangjie Zheng, Pengcheng He, and Mingyuan Zhou. POUF: prompt-oriented unsupervised fine-tuning for large pre-trained models. In International Conference on Machine Learning (ICML), volume 202, pages 33816–33832, 2023.   
[47] Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers, and Arnold W. M. Smeulders. Selective search for object recognition. International Journal of Computer Vision (IJCV), 104(2):154–171, 2013.   
[48] Luting Wang, Yi Liu, Penghui Du, Zihan Ding, Yue Liao, Qiaosong Qi, Biaolong Chen, and Si Liu. Object-aware distillation pyramid for open-vocabulary object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11186–11196. IEEE, 2023.   
[49] Xinlong Wang, Zhiding Yu, Shalini De Mello, Jan Kautz, Anima Anandkumar, Chunhua Shen, and José M. Álvarez. Freesolo: Learning to segment objects without annotations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14156–14166, 2022.   
[50] Xudong Wang, Rohit Girdhar, Stella X. Yu, and Ishan Misra. Cut and learn for unsupervised object detection and instance segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3124-3134, 2023.   
[51] Yangtao Wang, Xi Shen, Shell Xu Hu, Yuan Yuan, James L. Crowley, and Dominique Vaufrey-daz. Self-supervised transformers for unsupervised object discovery using normalized cut. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14523-14533, 2022.   
[52] Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, and Shengjin Wang. Detecting everything in the open world: Towards universal object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11433-11443. IEEE, 2023.   
[53] Samuel Wilson, Tobias Fischer, Feras Dayoub, Dimity Miller, and Niko Sünderhauf. SAFE: sensitivity-aware features for out-of-distribution object detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 23508–23519. IEEE, 2023.   
[54] Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, and Chen Change Loy. Aligning bag of regions for open-vocabulary object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15254–15264. IEEE, 2023.   
[55] Zhiheng Wu, Yue Lu, Xingyu Chen, Zhengxing Wu, Liwen Kang, and Junzhi Yu. UC-OWOD: unknown-classified open world object detection. In European Conference on Computer Vision (ECCV), volume 13670, pages 193–210, 2022.   
[56] Yunyi Xuan, Weijie Chen, Shicai Yang, Di Xie, Luojun Lin, and Yueting Zhuang. Distilling vision-language foundation models: A data-free approach via prompt diversification. In Abdulmotaleb El-Saddik, Tao Mei, Rita Cucchiara, Marco Bertini, Diana Patricia Tobon Vallejo, Pradeep K. Atrey, and M. Shamim Hossain, editors, Proceedings of the 31st ACM International Conference on Multimedia, MM 2023, Ottawa, ON, Canada, 29 October 2023-3 November 2023, pages 4928–4938. ACM, 2023.   
[57] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. FILIP: fine-grained interactive language-image pre-training. In International Conference on Learning Representations (ICLR), 2022.   
[58] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection using captions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14393–14402, 2021.   
[59] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. CoRR, abs/2304.00685, 2023.   
[60] Xiaowei Zhao, Yuqing Ma, Duorui Wang, Yifan Shen, Yixuan Qiao, and Xianglong Liu. Revisiting open world object detection. IEEE Trans. Circuits Syst. Video Technol., 34(5):3496-3509, 2024.

[61] C. Lawrence Zitnick and Piotr Dollár. Edge boxes: Locating object proposals from edges. In European Conference on Computer Vision (ECCV), volume 8693, pages 391–405, 2014.   
[62] Orr Zohar, Kuan-Chieh Wang, and Serena Yeung. PROB: probabilistic objectness for open world object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11444–11453. IEEE, 2023.   
[63] Wei Li Zuwei Long. Open grounding dino: the third party implementation of the paper grounding dino. https://github.com/longzw1997/Open-GroundingDino, 2023.

# A Appendix / Supplemental Material

This supplementary material includes a comprehensive overview of related work on class-agnostic object detection, vision-language models (VLMs), and prompt tuning. Additionally, we provide detailed descriptions of baselines, implementation details for both the baselines and the proposed method, and an extensive ablation study are provided. The ablation study analyzes the impact of pseudo-labeled supervision and the effect of the hyperparameter $\gamma$ . Lastly, we present more comprehensive visualizations of class-agnostic box predictions.

• Section A.1: Related Work   
• Section A.2: Baselines and Implementation Details   
• Section A.3: More Ablation Studies   
• Figures 9 and 10: Additional Visualizations of Class-Agnostic Box Predictions

# A.1 Related Work

Class-Agnostic Object Detection. Traditional bottom-up approaches $[47, 27, 61, 40, 41, 6]$ for region proposal generation, often grapple with the constraints precision, despite high recall rates, reducing their scalability for general use in diverse environments. Recent breakthroughs in ViTs $[4, 10, 39]$ have enabled scaling up to massive datasets for self-supervised learning, extracting both local and global semantically meaningful features. This has led to numerous methods in unsupervised object discovery and localization. LOST $[45]$ is an early application, using a patch similarity graph and an inverse degree map to identify seed patches and extract bounding boxes. TokenCut $[51]$ constructs an undirected graph with image tokens as nodes, applying the normalized cut algorithm $[44]$ for foreground-background segregation. MOVE $[37]$ builds on LOST by employing deep spectral bipartitioning, offering a more principled and effective approach. However, both LOST $[45]$ and TokenCut $[51]$ are limited to detecting a single object per image. MOST $[43]$ addresses this limitation by using entropy-based box analysis (EBA) to segregate foreground tokens. Nevertheless, their performance remains sub-optimal, constrained by their limited capacity to generalize across diverse object categories. Closest to our work is MAVL $[36]$ , where they develop an MViT with late fusion strategy and use generic text queries like “all objects” to locate objects. However, their framework requires full supervision and multiple inferences with different textual prompts, yet still falls short of achieving optimal performance. In contrast, our approach eliminates the need for labels and achieves SOTA performance with one-pass inference with the non-overlapping prompts.

VLMs and Prompt Tuning. Recent advances in VLMs $[42, 20, 57]$ which are pretrained on expansive image-text pairs have demonstrated significant potential in learning generic concepts. HierKD $[35]$ introduces global language-to-visual knowledge distillation modules, which align global-level image representations with caption embeddings through contrastive loss. OV-DETR $[58]$ pioneered the extension of the DETR framework to an open-vocabulary setting by integrating a conditional binary matching mechanism. GLIP $[29]$ converted object detection into a grounding task, utilizing additional data to align phrase and region semantics. Recently, Grounding DINO $[33]$ introduced a dual-encoder-single-encoder framework to integrate object detection and visual grounding within a unified architecture. Similarly, T-Rex2 $[21]$ synergizes text and visual prompts through contrastive learning, leading to state-of-the-art performance in out-of-distribution object detection. Nonetheless, the effectiveness of VLMs is heavily influenced by the textual cues they are conditioned on, and efficiently adapting them to specific downstream applications remains a substantial challenge as manually engineering optimal prompts can often entail considerable effort and resources $[59]$ . Prompt tuning is a simple yet effective solution to adapt models to specific tasks by optimizing a small number of soft prompts in an end-to-end manner while keeping the original model's parameters frozen. The pioneering work of CoOp $[24]$ introduced context optimization by fine-tuning CLIP using learnable tokens. However, CoOp's generalizability was constrained, a limitation later addressed by CoCoOp $[23]$ , which conditioning input tokens on image embeddings. MaPLe $[25]$ advanced this by introducing a multi-modal prompting technique to overcome the limitations of uni-modal prompting methods. ProDA $[34]$ further innovated by learning a distribution of diverse prompts and employing a Gaussian model to capture visual variations. Despite these advancements, an inherent limitation persists across these methods: they all require full supervision. UPL $[17]$ first proposed unsupervised prompt learning for image recognition task, POUF $[46]$ later introduced a similar self-prompting mechanism to minimize entropy using optimal transport. However, these

Table 5: The impact on pseudo-labeled supervision on MS-COCO [32] dataset, when applying different pseudo-labels queried on Grounding DINO using different textual cues. In the main paper, we report the performance of DiPEx using merged pseudo labels for the first round of training. 

<table><tr><td>Method</td><td>AR100</td><td>AR@S</td><td>AR@M</td><td>AR@L</td><td>AP</td><td>AP@S</td><td>AP@M</td><td>AP@L</td></tr><tr><td>Grounding DINO @ [“generic”]</td><td>44.1</td><td>17.7</td><td>51.6</td><td>80.0</td><td>28.3</td><td>11.4</td><td>33.0</td><td>56.5</td></tr><tr><td>Grounding DINO @ [25 nouns]</td><td>40.5</td><td>16.0</td><td>46.6</td><td>75.0</td><td>12.1</td><td>4.4</td><td>12.0</td><td>26.7</td></tr><tr><td>Grounding DINO @ merged</td><td>51.9</td><td>24.8</td><td>61.6</td><td>85.8</td><td>19.1</td><td>7.6</td><td>19.5</td><td>42.0</td></tr><tr><td>DiPEx @ [“generic”]</td><td>65.5</td><td>42.7</td><td>76.2</td><td>90.3</td><td>37.0</td><td>20.5</td><td>43.3</td><td>62.4</td></tr><tr><td>DiPEx @ [25 nouns]</td><td>46.6</td><td>18.5</td><td>55.7</td><td>83.3</td><td>13.2</td><td>4.0</td><td>13.3</td><td>30.7</td></tr><tr><td>DiPEx @ merged</td><td>63.2</td><td>39.2</td><td>74.3</td><td>89.8</td><td>35.9</td><td>16.4</td><td>39.7</td><td>63.8</td></tr></table>

methods have yet to be adapted for the object detection domain. To our knowledge, UPT $[16]$ is the only existing work that optimizes prompts using dual complementary teaching specifically for object detection tasks. Our work, DiPEx, represents the first endeavor to apply prompt learning to class-agnostic object detection through a self-training approach.

# A.2 Baselines and Implementation Details

Baselines. We compare the proposed approach with fourteen baselines: 1) bottom-up selective search [47] that slides windows of different sizes to locate objects, 2) UP-DETR [8], an unsupervised pre-training method for OD that can be fine-tuned to detect class-agnostic objects. 3) DETReg [1], which learns to localize objects and encode an object's properties during unsupervised pre-training, 4) MOST [43], a multiple objects localizer based on patch correlations without any training, 5) FreeSOLO [49], which unifies pixel grouping, object localization and feature pre-training in a fully self-supervised manner, 6) Exemplar-FreeSOLO [18], an improved approach based on FreeSOLO through exemplar knowledge extraction, 7) CutLER [50], an unsupervised object detection method by encouraging the detector to explore objects missed in extracted coarse masks, 8) HASSOD [2], a clustering strategy that groups regions into object masks based on self-supervised features, 9) CoOp [24] and 10) CoCoOp [23], prompting techniques that utilize learnable vectors to model a prompt's context words, enabling zero-shot transfer to class-agnostic detection, 11) segment anything model (SAM) [26], a foundational model trained on 1 billion masks and 11 million images such that can perform zero-shot transfer to the class-agnostic OD task. For OOD-OD task, we further compare three baseline methods: 12) VOS [12] that regularizes the model's decision boundary between known and unknown classes by training with generated virtual outliers. 13) PROB [62] which utilizes a multivariate Gaussian distribution to learn objectness probability to separate known and unknown objects, 14) UnSniffer [30], which similarly introduces an object confidence, derived from learning known objects with varying degrees of overlap.

Implementation Details. Our code is developed on the Open Grounding-DINO framework $[63]$ , and operates on a single NVIDIA RTX A6000 GPU with 48 GB of memory. For our experiments, we choose a batch size of 8 for training, and set hyperparameter $\gamma = 0.1$ , $\tau_{p} = 0.1$ , $\tau_{c} = 0.1$ , $\theta = \pm15^{\circ}$ , K = 9, L = 3, and while adopting all remaining hyperparameters from the Open Grounding-DINO codebase. We empirically set the $T_{\alpha} = 75^{\circ}$ as our threshold for expansion termination. The original implementation of CoOP was developed for image classification tasks based on CLIP and supervised contrastive learning. We extend CoOP to class-agnostic object detection using pseudo labeling-based self-training, which remains consistent with our approach. All the implementation code and configurations files are provided in supplementary materials and will be publicly released upon acceptance of this work.

# A.3 More Ablation Studies

Pseudo-labels Construction For pseudo-labeling, we utilize off-the-shelf Grounding DINO with a “generic” text prompt, which demonstrates considerable zero-shot performance, as illustrated in our pilot study. Additionally, we generate pseudo-boxes by concatenating all 25 beginner nouns from WordNet [14]. We then merge the predictions from these two queries and apply Soft-NMS to eliminate overlaps. In the following Section A.3, we also investigate the performance of DiPEx alongside other prompt-tuning methods on the quality of pseudo-labels.

Impact on Pseudo-labeled Supervision. In this section, we investigate how the quality of pseudo-labels used for self-training impacts DiPEx's performance, given our reliance on these training samples. Specifically, we generate pseudo-labels by querying the off-the-shelf Grounding Dino model with three different approaches: 1) using a “generic” text prompt, 2). the 25 beginner nouns from WordNet [14], and a combination of both. As shown in Table 5, the “generic” text prompt alone demonstrated considerable performance. However, we observed an improvement in Average Recall (AR) when merging the predictions generated by “generic” with the 25 beginner nouns, leading us to this study. Consequently, we use these pseudo-labels to self-train our model.

Effect of Loss Coefficient $\gamma$ . To effectively separate child prompts while maintaining semantic coherence between parent and child prompts, selecting an appropriate $\gamma$ is essential. As illustrated in the bar plot below, a moderate $\gamma$ value typically yields optimal results. In contrast, a larger value (e.g., $\gamma = 5$ ) causes child prompts to diverge more significantly, distorting semantic integrity and potentially leading to over-regularization of the model.

![](images/c4285b01fda3b663f2e6d338c5650ed1af3e110740990c5483e7a60b738783de.jpg)

<details>
<summary>bar</summary>

| y | AR | AP |
|---|---|---|
| 0.1 | 0.63 | 0.36 |
| 0.5 | 0.62 | 0.28 |
| 1 | 0.61 | 0.26 |
| 3 | 0.61 | 0.27 |
| 5 | 0.60 | 0.24 |
</details>

Figure 8: Study of Loss Coefficient γ

![](images/4c66a860e13d35300a59e2de19e4574d49a6e565d59d87e41bf8c6909963819f.jpg)  
Figure 9: Additional visualizations of class-agnostic box predictions. Columns 1 – 4 correspond to the following methods: MOST [43], CutLER [50], zero-shot Grounding DINO ["generic"] [9], and our proposed DiPEx, respectively. The final column presents human-annotated ground truth bounding boxes from the MS-COCO dataset [32].

![](images/103cdae4652da36dcef8c8c2e4ba8c44cf56a920e0ca791eed03cb537bcd34eb.jpg)

<details>
<summary>text_image</summary>

Grid of 20-panel sequence showing a person in action, with bounding boxes highlighting key events and activity descriptions.
</details>

Figure 10: Additional visualizations of class-agnostic box predictions. Columns 1 – 4 correspond to the following methods: MOST [43], CutLER [50], zero-shot Grounding DINO ["generic"] [9], and our proposed DiPEx, respectively. The final column presents human-annotated ground truth bounding boxes from the MS-COCO dataset [32].