# MotionCraft: Crafting Whole-Body Motion with Plug-and-Play Multimodal Controls

Yuxuan Bian $^{1}$ , Ailing Zeng $^{2*}$ , Xuan Ju $^{1}$ , Xian Liu $^{1}$ , Zhaoyang Zhang $^{1}$ , Wei Liu $^{2}$ , Qiang Xu $^{1*}$

$^{1}$ The Chinese University of Hong Kong $^{2}$ Tencent https://cure-lab.github.io/MotionCraft

![](images/d73fa9e98819f16da8fc0add97c0bfba2ecb93478655a03ce508da648bd614a1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Video Animation"] --> B["Unified SMPL-X Representation"]
    B --> C["Text-to-Motion Backbone"]
    C --> D["Control Branch"]
    D --> E["Plug & Play"]
    
    subgraph Video Animation
        F1["Person in motion"] --> F2["Person in motion"]
        F3["Person in motion"] --> F4["Person in motion"]
        F5["Person in motion"] --> F6["Person in motion"]
        F7["Person in motion"] --> F8["Person in motion"]
        F9["Person in motion"] --> F10["Person in motion"]
    end
    
    subgraph Unified SMPL-X Representation
        G1["Text"] --> H1["Sit and Stand up"]
        H1 --> I1["Giving a speech ..."]
        I1 --> J1["Do 2 jumping jacks"]
        J1 --> K1["Dance in the HipHop ..."]
    end
    
    style Video Animation fill:#f9f,stroke:#333
    style Unified SMPL-X Representation fill:#ccf,stroke:#333
```
</details>

Figure 1: We propose MotionCraft, a diffusion transformer that crafts whole-body motion with plug-and-play multimodal controls, encompassing robust motion generation abilities including Text-to-Motion, Speech-to-Gesture, and Music-to-Dance.

# Abstract

Whole-body multimodal motion generation, controlled by text, speech, or music, has numerous applications including video generation and character animation. However, employing a unified model to achieve various generation tasks with different condition modalities presents two main challenges: motion distribution drifts across different tasks (e.g., co-speech gestures and text-driven daily actions) and the complex optimization of mixed conditions with varying granularities (e.g., text and audio). Additionally, inconsistent motion formats across different tasks and datasets hinder effective training toward multimodal motion generation. In this paper, we propose MotionCraft, a unified diffusion transformer that crafts whole-body motion with plug-and-play multimodal control. Our framework employs a coarse-to-fine training strategy, starting with the first stage of text-to-motion semantic pre-training, followed by the second stage of multimodal low-level control adaptation to handle conditions of varying granularities. To effectively learn and transfer motion knowledge across different distributions, we design MC-Attn for parallel modeling of static and dynamic human topology graphs. To overcome the motion format inconsistency of existing benchmarks, we introduce MC-Bench, the first available multimodal whole-body motion generation benchmark based on the unified SMPL-X format. Extensive experiments show that MotionCraft achieves state-of-the-art performance on various standard motion generation tasks.

# Introduction

Whole-body human motion generation with multimodal controls (Zhang et al. 2024b; Liu et al. 2024a; Li et al. 2023), which produces natural and coherent human movements based on multimodal conditions, has numerous applications, including human video generation (Hu 2024) and character animation (Zhang et al. 2023a).

Recent advancements in single-conditioned human motion generation have made it possible to generate realistic human movements from a variety of control signals with varying granularities, including text descriptions (Guo et al. 2022; Zhang et al. 2023c), music clips (Siyao et al. 2022; Li et al. 2023), and speech segments (Liu et al. 2024a; Chen et al. 2024). However, extending these capabilities to whole-body motion generation with multimodal control within a unified model introduces several significant challenges:

Motion distribution drifts: Under different conditions, the motion distribution often varies significantly (Zhang et al. 2024b; Ling et al. 2023). In text-to-motion (T2M), semantic text guidance mainly controls daily torso movements (Guo et al. 2022; Lin et al. 2023a), while speech-to-gesture (S2G) focuses on gestures and facial expressions under first-perspective audio (Liu et al. 2024a; Yi et al. 2023). Music-to-dance (M2D) includes a more dynamic and variable correlation between the third-perspective music with limb movements (Li et al. 2023). Previous research usually focused on a single task to avoid the weak generative transferability posed by distribution drifts.   
Optimization challenges under mixed conditions: Current multimodal motion generation work compress diverse control signals—such as semantic text guidance, first-person speech, and third-person music—into a common latent space for mixed modeling. This includes transformer

token embedding (Zhou, Wan, and Wang 2023) and the feature space used in ImageBind (Girdhar et al. 2023). However, this approach often leads to alignment issues across different modalities and introduces optimization challenges when learning conditions at different levels of granularity simultaneously (Team et al. 2023).

Non-uniform whole-body motion format and evaluation: Finally, there are no high-quality multimodal whole-body human motion generation benchmarks with unified motion representation and evaluation pipelines.

In this work, we propose a unified motion diffusion transformer, MotionCraft, that crafts whole-body motion with plug-and-play multimodal control, generating fine-grained text- and speech (music)-aligned motions. It also supports generating motion with multiple conditions simultaneously, such as text combined with speech or music.

For effective learning of conditions with varying granularities, MotionCraft employs a two-stage, coarse-to-fine multimodal generation framework. In the first stage, it captures high-level semantic motion generation abilities guided by coarse-grained text. In the second stage, control branches are added to the frozen backbone from the first stage, allowing the model to retain semantic generation capabilities while achieving fine-grained plug-and-play controls for specific low-level conditions (speech or music) without the optimization confusion associated with mixed training.

To address motion distribution drifts across various generation scenarios, we analyze human motion kinematics and distribution using t-distributed stochastic neighbor embedding. We find that motion distributions corresponding to different control signals can be decomposed into static human topology structures and dynamic topology relationships, which are generalizable across different scenarios. Different from the existing large language and vision model, the amount of motion data is still very small and unscalable. To model these human-centric spatiotemporal properties, we design MC-Attn, where the spatial branch learns and transfers motion topology knowledge across different distributions by parallel modeling of static and dynamic human topology graphs, while the temporal branch captures the temporal relationships within the motion sequences.

To overcome the inconsistent motion format limitation in existing benchmarks, such as Rot6D (Guo et al. 2022), SMPL (Loper et al. 2015), and SMPL-X (Pavlakos et al. 2019), we also introduce MC-Bench, the first available multimodal motion generation benchmark based on the unified whole-body SMPL-X format, including data construction and evaluation pipelines. Extensive experiments demonstrate that MotionCraft achieves competitive performance across various standard motion generation tasks, including text-to-motion, speech-to-gesture, and music-to-dance. Additionally, we provide comprehensive ablation studies, offering insights into model design and scaling effects for future multimodal whole-body motion generation models.

In summary, our contributions are as follows:

\- We propose MotionCraft, a two-stage, coarse-to-fine multimodal motion generation framework that supports control signals at different granularities, enabling efficient

plug-and-play multimodal motion generation.

- We design MC-Attn, the first attempt to achieve modeling of static and dynamic human topology against motion distribution drifts in multimodal motion generation.   
- We create MC-Bench, the first publicly available multimodal whole-body motion generation benchmark with a unified whole-body motion representation SMPL-X.

# Related Work

# Human Motion Generation Models

Conditioned human motion generation models have made significant progress, including text-to-motion (T2M) (Tevet et al. 2023; Zhang et al. 2023b; Liu et al. 2023; Zhang et al. 2024a, 2023c; Liang et al. 2024), speech-to-gesture (S2G) (Yi et al. 2023; Chen et al. 2024; Liu et al. 2022b), and music-to-dance (M2D) (Li et al. 2023; Tseng, Castellon, and Liu 2023; Siyao et al. 2022). Recently, increasing attention has been paid to multimodal motion generation (Ling et al. 2023; Zhang et al. 2024b; Luo et al. 2024). $M^3$ -GPT (Luo et al. 2024) injects quantized condition tokens into the vocabulary of large language models to achieve motion understanding and generation, but it overlooks the modeling of human topology priors. Motion-Verse (Zhang et al. 2024b) incorporates dynamic attention to assess relationships among body parts but fails to capture the overall static human topology, leading to limited generalization power and increased optimization complexity. Furthermore, it employs mixed training across all conditions based on ImageBind (Girdhar et al. 2023), which creates optimization challenges when learning conditions of varying granularities simultaneously and needs retraining for new control signals. MCM (Ling et al. 2023) attempts to address the optimization confusion of mixed training based on the ControlNet (Zhang, Rao, and Agrawala 2023) architecture, but it neglects any modeling of human topology structure, resulting in poor generalization across generation scenarios. Compared to previous methods in Tab. 1, MotionCraft generates whole-body motion under varying control signals with plug-and-play capability by using MC-Attn to capture static human topology and domain-specific dynamic skeleton relationships, incorporating control branches, and employing a coarse-to-fine training strategy.

# Human Motion Generation Benchmarks

Various conditioned human motion generation benchmarks have been constructed in recent years. For T2M, researchers have curated datasets encompassing action categories (Chung et al. 2021; Trivedi, Thatipelli, and Sarvadevabhatla 2021), sequential action labels (Zhang et al. 2022; Guo et al. 2020), and arbitrary natural language descriptions (Lin et al. 2023a; Guo et al. 2022; Tang et al. 2023). For M2D, AIST++ (Li et al. 2021) reconstructs 5 hours of dance based on SMPL (Loper et al. 2015) format from videos. Finedance (Li et al. 2023) collects dances of 14.6 hours across 22 genres and supplements the dataset with detailed gestures using the SMPL-H (Pavlakos et al. 2019) format. For S2G datasets (Liu et al. 2024a, 2022a;

Table 1: Comparison of MotionCraft with previous motion generation methods. MotionCraft jointly models the static human skeleton structure and dynamic human topology relationships to achieve flexible motion knowledge transfer across various whole-body generation scenarios, supporting plug-and-play with any new control signal modality. 

<table><tr><td>Model</td><td>Text2Motion</td><td>Music2Dance</td><td>Speech2Gesture</td><td>Static Body Prior</td><td>Dynamic Body Adaption</td><td>Whole Body</td><td>Unified Representation</td><td>Plug-and-Play</td></tr><tr><td>FineMoGen (Zhang et al. 2023c)</td><td>√</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>HumanTomato (Lu et al. 2023)</td><td>√</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>FineDance (Li et al. 2023)</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>Bailando (Siyao et al. 2022)</td><td>✗</td><td>√</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>EMAGE (Liu et al. 2024a)</td><td>✗</td><td>✗</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>TalkShow (Yi et al. 2023)</td><td>✗</td><td>✗</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>MCM (Ling et al. 2023)</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>Motion-Verse (Zhang et al. 2024b)</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>MotionCraft</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Yi et al. 2023), BEAT2 (Liu et al. 2024a) and BEAT (Liu et al. 2022a) have emerged as the most popular benchmarks, celebrated for their diverse range of motion and extensive data volume. BEAT2, built upon BEAT, utilizes SMPL-X and FLAME (Kim, Kim, and Choi 2023) to achieve higher-quality unified mesh-level data. Despite these developments, no publicly available benchmark supports unified representation for multimodal whole-body motion generation.

# Motivation

The key challenge in achieving whole-body human motion generation with multimodal controls is addressing motion distribution drifts across different generation scenarios (Zhang et al. 2024b) and the efficient learning of control signals at varying granularities (Ling et al. 2023).

Motion distribution drifts solution. Current motion generation models mainly focus on scenarios with a single condition since they struggle to handle the noticeable motion distribution drifts across different scenarios (Zhou, Wan, and Wang 2023). For instance, as shown in Fig. 2, T2M primarily involves everyday torso movements, S2G includes complex hand gestures, rich facial expressions, and almost stationary lower limbs, while M2D emphasizes varied and extensive limb movements with limited hand movements. However, many human-centric studies (Zeng et al. 2021; Ma, Bai, and Zhou 2022) have confirmed that representing the human skeletal topology as a directed weighted graph, with different body parts as vertices, can introduce kinematic priors in complex motion modeling, thereby improving generalizability under distribution shifts. Additionally, based on human kinematic (Loper et al. 2015; Pavlakos et al. 2019), it is natural to decompose the human skeleton into a combination of static and dynamic topologies. For instance, in any scenario, the root vertex (hip) always significantly influences its child vertices (lower limbs or upper arms), with symmetrical interactions between pairs of arms. However, in S2G, the correlation between the limbs and other body parts weakens, while the linking weight between hands and facial expressions strengthens. Therefore, modeling both dynamic and static topology graphs can efficiently generalize motion knowledge across different generation tasks, even with limited data and significant distribution drifts.

Efficient learning of conditions at varying granularities. Different motion generation scenarios correspond to conditions at varying granularities. For instance, text guidance typically provides sequence-level coarse-grained semantic control, while speech and music focus more on per-frame low-level control (Liu et al. 2024a; Li et al. 2023). Mixed learning of all conditions within a single space leads

![](images/a26f84c2e495fbd9d2244393a2f7eb2fadb32605cd554f77d69e5f486e3f4cfe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Person with motion icons"] --> B["<Text> &quot;a person steps back and to the left and appears to sit...&quot;"]
    A --> C["<Speech> &quot;I always like to chill in the...&quot;"]
    A --> D["<Music> &quot;Dancing in the hip-hop style...&quot;"]
    E["Hand gesture icons"] --> F["Text-to-Motion"]
    G["Hand gesture icons"] --> H["Speech-to-Gesture"]
    I["Hand gesture icons"] --> J["Music-to-Dance"]
```
</details>

Figure 2: The t-SNE latent space of motion in different generation tasks. It illustrates the motion distribution drifts across different generation scenarios.

to inevitable modality alignment loss and fails to decouple the learning process for each granularity (Zhang, Rao, and Agrawala 2023; Ling et al. 2023), causing optimization confusion. Motivated by other vision generation paradigms in the image/video domain, including StableDiffusion (Rombach et al. 2022) and Sora (Liu et al. 2024b), decoupling the generation under different conditions and using T2M as a basic pre-training task can build robust generative abilities for following multi-condition generation, resulting in more efficient and fine-grained multimodal control generation.

# Proposed Method

# MotionCraft Framework

The overview of MotionCraft is described in Fig. 3. Aimed at decoupling the conditioned generation learning at varying granularities, we adopt a two-branch architecture consisting of a main text-to-motion branch and a plug-and-play low-level control branch, along with a two-stage coarse-to-fine training strategy to efficiently grasp the motion topology knowledge across different scenarios with various control signal modalities. Both branches use a motion diffusion transformer specifically designed with MC-Attn to capture both static and dynamic motion topology properties.

Stage ① Text-to-Motion Semantic Pre-training. The main branch $f_{m}(\cdot)$ is optimized in Stage I, text-to-motion semantic pre-training, using text-to-motion paired data collected from diverse scenarios in MC-Bench. We choose text as the shared condition among various unimodal datasets, allowing MotionCraft to acquire sequence-level generation and coarse-grained text-guidance following abilities between text $H_{text} \in R^{B \times F_{t} \times D_{t}}$ and motion $H_{motion} \in R^{B \times F_{m} \times D_{m}}$ . Overall, text guidance pre-training in diverse generation scenarios helps follow fine-grained controls of other low-level conditions in Stage II.

Stage ② Multimodal Low-level Control Adaptation. During the low-level control adaptation fine-tuning stage,

![](images/9484c88fbbb49043d4645a6f9bd4088add425307566e4783d927a7ae4f9ec3ab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text2Motion"] --> B["1. Root"]
    C["Speech2Gesture"] --> B
    D["Music2Dance"] --> B
    B --> E["3. Head"]
    E --> F["..."]
    E --> G["10. Left Hand"]
    E --> H["11. Right Leg"]
    E --> I["12. Left Leg"]
    E --> J["12 Body Parts"]
    K["SMPL-X"] --> L["12 Body Parts"]
    M["<Text>&quot;A person raises his arms...&quot;"]
    N["<Speech>&quot;I always like to chill...&quot;"]
    O["<Music>&quot;Dancing in the hip-hop...&quot;"]
```
</details>

(I) SMPL-X based Representation Unification

![](images/9efcc8da1d032f48fa4408b81b7bcc5e81934e66338a7957337cfa3613ae60d3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Noisy SMPL-X"] --> B["Body-wise Encoding"]
    C["Text"] --> D["DiT-Layer"]
    B --> D
    D --> E["..."]
    E --> F["DiT-Layer"]
    F --> G["Body-wise Decoding"]
    G --> H["Human figures"]
```
</details>

(II) Stage 1: Text-to-Motion Semantic Pre-Training

![](images/f8c5c64c72ab2b827a7d055b5dfadb9018ee2497d00b5114acf6db175a19efbd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Noisy SMPL-X Text"] --> B["Body-wise Encoding"]
    B --> C["DiT-Layer"]
    C --> D["+"]
    D --> E["DiT-Layer"]
    E --> F["..."]
    F --> G["Body-wise Decoding"]
    G --> H["..."]
    H --> I["DiT-Layer"]
    I --> J["+"]
    J --> K["Zero-Linear"]
    K --> L["DiT-Layer"]
    L --> M["Zero-Linear"]
    M --> N["DiT-Layer"]
    N --> O["Zero-Linear"]
    O --> P["..."]
    P --> Q["DiT-Layer"]
    Q --> R["Zero-Linear"]
    R --> S["..."]
    S --> T["DiT-Layer"]
    T --> U["Zero-Linear"]
    U --> V["..."]
    V --> W["DiT-Layer"]
    W --> X["Zero-Linear"]
    X --> Y["..."]
    Y --> Z["DiT-Layer"]
    Z --> AA["Zero-Linear"]
    AA --> AB["..."]
    AB --> AC["DiT-Layer"]
    AC --> AD["Zero-Linear"]
    AD --> AE["..."]
    AE --> AF["DiT-Layer"]
    AF --> AG["Zero-Linear"]
    AG --> AH["..."]
    AH --> AI["DiT-Layer"]
    AI --> AJ["Zero-Linear"]
    AJ --> AK["..."]
    AK --> AL["DiT-Layer"]
    AL --> AM["Zero-Linear"]
    AM --> AN["..."]
    AN --> AO["DiT-Layer"]
    AO --> AP["Zero-Linear"]
    AP --> AQ["..."]
    AQ --> AR["DiT-Layer"]
    AR --> AS["Zero-Linear"]
    AS --> AT["..."]
    AT --> AU["DiT-Layer"]
    AU --> AV["Zero-Linear"]
    AV --> AW["..."]
    AW --> AX["DiT-Layer"]
    AX --> AY["Zero-Linear"]
    AY --> AZ["..."]
    AZ --> BA["DiT-Layer"]
    BA --> BB["Zero-Linear"]
    BB --> BC["..."]
    BC --> BD["DiT-Layer"]
    BD --> BE["Zero-Linear"]
    BE --> BF["..."]
    BF --> BG["DiT-Layer"]
    BG --> BH["Zero-Linear"]
    BH --> BI["..."]
    BI --> BJ["DiT-Layer"]
    BJ --> BK["Zero-Linear"]
    BK --> BL["..."]
    BL --> BM["DiT-Layer"]
    BM --> BN["Zero-Linear"]
    BN --> BO["..."]
    BO --> BP["DiT-Layer"]
    BP --> BQ["Zero-Linear"]
    BQ --> BR["..."]
    BR --> BS["DiT-Layer"]
    BS --> BT["Zero-Linear"]
    BT --> BU["..."]
    BU --> BV["DiT-Layer"]
    BV --> BW["Zero-Linear"]
    BW --> BX["..."]
    BX --> BY["DiT-Layer"]
    BY --> BZ["Zero-Linear"]
    BZ --> CA["..."]
    CA --> CB["DiT-Layer"]
    CB --> CC["Zero-Linear"]
    CC --> CD["..."]
    CD --> CE["DiT-Layer"]
    CE --> CF["Zero-Linear"]
    CF --> CG["..."]
    CG --> CH["DiT-Layer"]
    CH --> CI["Zero-Linear"]
    CI --> CJ["..."]
    CJ --> CK["DiT-Layer"]
    CK --> CL["Zero-Linear"]
    CL --> CM["..."]
    CM --> CN["DiT-Layer"]
    CN --> CO["Zero-Linear"]
    CO --> CP["..."]
    CP --> CQ["DiT-Layer"]
    CQ --> CR["Zero-Linear"]
    CR --> CS["..."]
    CS --> CT["DiT-Layer"]
    CT --> CU["Zero-Linear"]
    CU --> CV["..."]
    CV --> CW["DiT-Layer"]
    CW --> CX["Zero-Linear"]
    CX --> CY["..."]
    CY --> CZ["DiT-Layer"]
    CZ --> DA["Zero-Linear"]
    DA --> DB["..."]
    DB --> DC["DiT-Layer"]
    DC --> DD["Zero-Linear"]
    DD --> DE["..."]
    DE --> DF["DiT-Layer"]
    DF --> DG["Zero-Linear"]
    DG --> DH["..."]
    DH --> DI["DiT-Layer"]
```
</details>

(III) Stage 2: Multimodal Plug-and-play Low-level Control Adaptation

![](images/ad5a26c118d416db8a3a137b2ad424d0c598279c025c15b3608c34c59f6211f3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text"] --> B["MC-Attn"]
    B --> C["Stylization"]
    C --> D["FFN"]
    D --> E["Stylization"]
    E --> F["H motion"]
    E --> G["H text"]
    F --> H["MOE"]
    G --> I["MOE"]
    H --> J["Static Tropology Structure Modeling"]
    H --> K["Dynamic Topology Relation Modeling"]
    I --> L["Temporal Modeling"]
    J --> M["+"]
    K --> N["+"]
    L --> O["+"]
    M --> P["+"]
    N --> Q["+"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#fff,stroke:#333
    style P fill:#fff,stroke:#333
    style Q fill:#fff,stroke:#333
```
</details>

(IV)Motion Spatial & Temporal Topology Modeling in MC-Attn   
Figure 3: Architecture of MotionCraft. MotionCraft is a transformer-based diffusion model. In the first stage, MotionCraft uses text as a semantic control guide to learn coarse-grained cross-scenario motion knowledge across multiple datasets; in the second stage, MotionCraft freezes the backbone while adding a plug-and-play control branch to learn the different low-level control signals. The core of MotionCraft is MC-Attn, which optimizes the representation of motion token sequences by capturing the spatial properties of static and dynamic human topology graphs and learning temporal relationships in parallel.

we aim to model the correlation between various condition signals $H_{c} \in R^{B \times T_{c} \times D_{c}}$ and motion sequences $H_{motion} \in R^{B \times F_{m} \times D_{m}}$ . All main branch parameters $f_{m}(\cdot)$ are frozen to maintain their coarse-grained motion generation and semantic text guidance following abilities. A copy of the main branch parameters $\hat{f}_{m}(\cdot)$ is then used to initialize the control branch, connecting them with a zero-initialized linear layer $W_{p} \in R^{D_{m} \times D_{m}}$ to prevent early training noise from causing collapse. Then the condition signals (speech, music, or other low-level control signals) are fed into the control branch, where a position mask $M_{c} \in \{0,1\}^{F_{m}}$ aligns the condition signals to the motion sequence length $F_{m}$ , setting zeros for the $F_{m} - T_{c}$ missing frames in the original control signal sequence. The output of each control branch layer is directly added to the corresponding main branch layer input through the zero bridge linear, allowing new control signals to guide frame-level human motion generation.

# MC-Attn Design

The core of MotionCraft is MC-Attn, which parallel captures the static and dynamic human topology graphs, thereby enhancing the transferability of motion topology knowledge across diverse generation scenarios against non-neglectable distribution drifts. MC-Attn has three key components: a static-skeleton graph learner and a dynamic-topology relationship graph learner for parallel modeling motion spatial properties, and temporal attention for modeling the frame-level dynamics of each body part over time. The three modules share the same motion representation input $H_{m} \in R^{B \times F_{m} \times D_{m}}$ , the output of the last MC-Attn layer refined further by a MOE(Shazeer et al. 2017).

For the static-skeleton graph learner, the process begins by constructing the $N_{b}$ graph vertex representation $H_{s} \in R^{B \times F_{m} \times N_{b} \times D_{b}}$ , followed by initializing a diagonal unit matrix $A_{s} \in R^{N_{b} \times N_{b}}$ as the adjacency matrix for the initial

static topology graph $G_{s}$ , where each body part is connected only to itself to avoid training collapse from random connections. Through optimization, $\hat{A}_{s}$ captures the static, input-independent human topology, enabling the model to quickly grasp fundamental human structure for new scenarios, even with limited data. The module outputs $E_{s} = \hat{A}_{s} \cdot H_{s}$ .

While static human topology graphs capture the basic structure and facilitate quick convergence to new distributions, they do not adapt dynamically to new contexts, potentially causing underfitting in evolving scenarios (Zhang et al. 2024b). To address this, we introduce a dynamic-topology relationship graph learner that models dynamic distribution features and adjusts to distribution drifts based on control signals, complementing the static topology structure. Specifically, the dynamic-topology relationship graph learner represents each body part as a dynamic graph vertex $\mathbf{H}_d \in \mathbb{R}^{B \times F_m \times N_b \times D_b}$ , using attention scores $\mathbf{A}_d \in \mathbb{R}^{B \times F_m \times N_b \times N_b}$ as edge weights in the dynamic human topology graph $\mathcal{G}_d$ , thereby enhancing the model's ability to adapt its spatial structure beyond the static-skeleton learner. The final output is $\mathbf{E}_d = \mathbf{A}_d \cdot \mathbf{H}_d$ .

Various studies have shown that basic attention is sufficient for modeling temporal relationships (Nie et al. 2022; Bian et al. 2024). Therefore, we chose to use each body part as a unit $\mathbf{H}_t \in \mathbb{R}^{B \cdot N_b \times F_m \times D_b}$ and measure the temporal relationships between frames based on attention (Vaswani et al. 2017). Considering that external textual control signals are mostly sequential instructions in the temporal dimension, text information is also modeled here to produce the output, $\hat{\mathbf{E}}_t = \text{Softmax}(\mathbf{Q}_{H_t} \cdot [\mathbf{K}_{H_t}^T, \mathbf{K}_{H_{text}}^T] / \sqrt{D_b}) \cdot [\mathbf{V}_{H_t}^T, \mathbf{V}_{H_{text}}^T]$ , where $\mathbf{Q}_{H_t} = \mathbf{W}^{Q_{H_t}}\mathbf{H}_t$ , $\mathbf{K}_{H_t} = \mathbf{W}^{K_{H_t}}\mathbf{H}_t$ , $\mathbf{K}_{H_{text}} = \mathbf{W}^{K_{H_{text}}} \mathbf{H}_{text}$ , $\mathbf{V}_{H_t} = \mathbf{W}^{V_{H_t}}\mathbf{H}_t$ , $\mathbf{V}_{H_{text}} = \mathbf{W}^{V_{H_{text}}} \mathbf{H}_{text}$ , and [,] denotes concat operation. Other sequential control modalities, such as speech and music, are modeled in the control branch.

The final output $E = E_{s} + E_{d} + E_{t}$ of MC-Attn combines the spatiotemporal representations of the human skeleton and the temporal dynamics of each body part.

# MC-Bench Construction

To prevent the information loss when aligning different motion formats, we select HumanML3D (Guo et al. 2022) in SMPL format for T2M, FineDance (Li et al. 2023) in SMPL-H Rot-6D format for M2D, and BEAT2 (Liu et al. 2024a) in SMPL-X format for S2G from public datasets, as they are the most representative unimodal datasets in their respective areas. To enable whole-body multimodal control of human motion generation, we converted all data to the SMPL-X format. Key operations include filling in missing facial information in HumanML3D and FineDance with average expressions and converting FineDance from SMPL-H Rot-6D format to axis-angle representation for efficient alignment with SMPL-X parameters and minimal alignment errors compared to the official body-retargeting method. We then pre-train a motion encoder and a text encoder by aligning text and motion contrastively with a retrieval optimization goal (Lu et al. 2023) for a unified evaluation of the SMPL-X motion representation. For FineDance and BEAT2, which lack corresponding textual information, we generate pseudocaptions such as "A dancer is performing a street dance in the Jazz style to the rhythm of the wildfire" and "A person is giving a speech, and the content is ...".

# Experiments

# Implementation Details

We designed two model variants for the first stage of Text-to-Motion backbone training, MotionCraft-Basic and MotionCraft-Mix, which were trained on the HumanML3D subset in MC-Bench and the entire MC-Bench, respectively. In the second stage, we used BEAT2 (Liu et al. 2024a), a large dataset for speech gesture synthesis, and FineDance (Li et al. 2023), a high-quality choreography dataset, to train control branches for Speech-to-Gesture and Music-to-Dance. MotionCraft-Basic and MotionCraft-Mix share the same 4-layer transformer backbone configuration, dividing the body topology into 12 parts, each with a body-part hidden encoding dimension of 64. MC-Bench used a unified whole-body motion format SMPL-X (Pavlakos et al. 2019) in the form of axis-angle, instead of the joint positions or 6D rotation. Thus we retrained the motion and text encoder based on SMPL-X using OpenTMR (Lu et al. 2023) for evaluation.

# Evaluation Metrics

Text-to-Motion. We use Fréchet Inception Distance (FID) to measure the distribution distance between generated motion and the ground truth, and diversity (Div) to measure the average pairwise Euclidean distance among random pairs of generated motion. Furthermore, we use R-Precision to measure how often the top-k closest motions to their corresponding captions are achieved within a 32-sample batch. Finally, we employ Multi-Modal Distance (MM Dist) to quantify the average Euclidean distance between motion representations and their corresponding text features.

Speech-to-Gesture. We use $FID_{H}$ , $FID_{B}$ , and Div for quality and diversity measurement. $FID_{H}$ represents the difference between the hand motion distribution and the ground truth gesture distribution, while $FID_{B}$ focuses on the distance between the distributions of whole-body motion. Moreover, we use the Beat Alignment Score (Li et al. 2021) to measure the alignment between the motion and speech beats and employ L2 Loss to measure the difference between generated and real expressions.

Music-to-Dance. Similar to Speech-to-Gesture, we use $FID_{H}$ , $FID_{B}$ , and Div to measure the quality of music-to-motion generation for hand and whole-body movements, as well as the diversity of the generated motions.

# Quantitative and Qualitative Results

We evaluate MotionCraft on three representative tasks: ① Text-to-Motion, ② Speech-to-Gesture, and ③ Music-to-Dance, analysing both quantitative and qualitative results. $^{1}$ More visualization comparisons are in our supplementary.

Comparison on Text-to-Motion Generation. In the text-to-motion task, we compare MotionCraft with current SOTA baselines (Zhang et al. 2023c; Ling et al. 2023; Zhang et al. 2024b,a; Tevet et al. 2023; Zhang et al. 2023b) in two benchmarks: the HumanML3D subset with whole-body format SMPL-X of MC-Bench in Tab. 2 and the original HumanML3D (Guo et al. 2022) with the tensor-only format (The results are in supplementary due to page limit). In both benchmarks, MotionCraft achieved better text-guided generation capability, diversity, and motion generation quality. Notably, in the HumanML3D subset of MC-Bench, the inadequate evaluation abilities in the original HumanML3D benchmark with torso-only representation were significantly improved, providing a more comprehensive and objective comparison. This is because the whole-body SMPL-X representation requires the model to generate the torso movements, gestures, and expressions rather than the only torso. Additionally, we found that MotionCraft-Mix trained on the MC-Bench has a significant advantage over MotionCraft-Basic. This is because MotionCraft-Mix can efficiently transfer human topology knowledge against distribution drifts in various generation scenarios. Visualization is in Fig. 4, and MotionCraft can follow diverse textual descriptions with fine-grained control.

Comparison on Speech-to-Gesture Generation. In Tab. 3, we compared MotionCraft with MCM (Ling et al. 2023), Talkshow (Yi et al. 2023), and EMAGE (Liu et al. 2024a). Our model achieved good quality and diversity in both hand and whole-body motion generation and excelled in aligning with the rhythm of first-perspective speech. This is credited to our coarse-to-fine training strategy and the robust topology knowledge learned from the static and dynamic human topology graphs. However, in expressions, MotionCraft-Mix performs slightly worse than EMAGE and Talkshow. This arises from origin dataset limitations in HumanML3D and FineDance, where the face was filled with random or

![](images/f216db05d88e1c1337d1aa562e12ee7270dc797213bbf4f16e99ea345f40fefd.jpg)

<details>
<summary>text_image</summary>

Ours
FineMoGen
MCM
The man walks forward and looks like he trips on something.
Incorrect execution order!
Didn't trip!
Ours
EMAGE
MCM
Ours
FineMoGen
MCM
A person holds their arms out in front of them and performs a squatting motion.
Ours
EMAGE
MCM
A person is doing a speech: "Well, in my opinion, I think the best ..." 
Ours
FineDance
MCM
Ours
FineDance
MCM
A person is doing a speech: "the best job for me is to become a..."
More diverse and natural!
A dancer is performing a Street dance in the Jazz style.
A dancer is performing a Mix dance in the Korean style.
</details>

Figure 4: The qualitative results of MotionCraft and other state-of-the-art baselines on three representative tasks, text-to-motion, speech-to-gesture, and music-to-dance. More detailed visualization comparisons are in our supplementary. 

<table><tr><td rowspan="2">Method</td><td colspan="3">R Precision</td><td rowspan="2">FID ↓</td><td rowspan="2">Div ↑</td><td rowspan="2">MM Dist↓</td></tr><tr><td>Top-1 ↑</td><td>Top-2 ↑</td><td>Top-3 ↑</td></tr><tr><td>GT</td><td> $0.663^{\pm 0.006}$ </td><td> $0.807^{\pm 0.002}$ </td><td> $0.864^{\pm 0.002}$ </td><td> $0.000^{\pm 0.000}$ </td><td> $36.423^{\pm 0.183}$ </td><td> $15.567^{\pm 0.036}$ </td></tr><tr><td>T2M-GPT(Zhang et al. 2023b)</td><td> $0.529^{\pm 0.004}$ </td><td> $0.652^{\pm 0.003}$ </td><td> $0.732^{\pm 0.003}$ </td><td> $10.457^{\pm 0.108}$ </td><td> $36.114^{\pm 0.098}$ </td><td> $17.029^{\pm 0.039}$ </td></tr><tr><td>MDM(Tevet et al. 2023)</td><td> $0.383^{\pm 0.010}$ </td><td> $0.527^{\pm 0.012}$ </td><td> $0.604^{\pm 0.009}$ </td><td> $18.671^{\pm 0.370}$ </td><td> $36.156^{\pm 0.103}$ </td><td> $18.785^{\pm 0.054}$ </td></tr><tr><td>MotionDiffuse(Zhang et al. 2024a)</td><td> $0.525^{\pm 0.004}$ </td><td> $0.675^{\pm 0.009}$ </td><td> $0.743^{\pm 0.009}$ </td><td> $9.982^{\pm 0.379}$ </td><td> $36.187^{\pm 0.160}$ </td><td> $17.314^{\pm 0.066}$ </td></tr><tr><td>FineMoGen(Zhang et al. 2023c)</td><td> $0.565^{\pm 0.001}$ </td><td> $0.710^{\pm 0.004}$ </td><td> $0.775^{\pm 0.004}$ </td><td> $7.323^{\pm 0.143}$ </td><td> $36.324^{\pm 0.069}$ </td><td> $16.679^{\pm 0.029}$ </td></tr><tr><td>MCM(Ling et al. 2023)</td><td> $0.407^{\pm 0.002}$ </td><td> $0.559^{\pm 0.003}$ </td><td> $0.636^{\pm 0.001}$ </td><td> $15.540^{\pm 0.443}$ </td><td> $35.813^{\pm 0.137}$ </td><td> $18.673^{\pm 0.029}$ </td></tr><tr><td>MotionCraft-Basic</td><td> $0.590^{\pm 0.003}$ </td><td> $0.743^{\pm 0.002}$ </td><td> $0.804^{\pm 0.004}$ </td><td> $8.477^{\pm 0.102}$ </td><td> $36.210^{\pm 0.089}$ </td><td> $16.252^{\pm 0.035}$ </td></tr><tr><td>MotionCraft-Mix</td><td> $0.600^{\pm 0.003}$ </td><td> $0.747^{\pm 0.004}$ </td><td> $0.812^{\pm 0.006}$ </td><td> $6.707^{\pm 0.081}$ </td><td> $36.419^{\pm 0.047}$ </td><td> $16.334^{\pm 0.059}$ </td></tr></table>

Table 2: Results of Text-to-Motion in HumanML3D of MC-Bench. We compare the results of text-to-motion between ours and the SOTA methods. Red background indicates best results, yellow background indicates second best results.

average expressions, confusing the first training stage that affects the following S2G generation. Still, we find Motion-Craft-Mix possesses a notable performance boost against MotionCraft-Basic, further confirming that MC-Attn learned robust topology knowledge that can be generalized across different generation scenarios. Qualitative results in Fig. 4 clearly show that MotionCraft can effectively follow the beats and generate reasonable gestures and lip movements.

Comparison on Music-to-Dance Generation. MotionCraft achieves performance comparable to the SOTA baselines, as shown in Tab. 3. Both variants of our model perform well in diversity, attributed to the first stage of coarse text-to-motion generation training. This equips the model with extensive motion topology knowledge across various scenarios. However, MotionCraft-Mix has an increase in FID compared to MotionCraft-Basic. This is likely due to the FineDance dataset's lack of necessary text descriptions, leading to identical pseudo-captions for different segments of the same song during the first stage of training. This one-

to-many generation mode confuses when the model incorporates corresponding music information for each segment in the second stage, attempting to learn many-to-many relationships. Qualitative results in Fig. 4 show that MotionCraft can generate natural dances according to the music beats.

# Ablation Study

We conducted ablation explorations about the necessity of MC-Attn design and scaling up influences in Tab .4.

Different motion topology modeling designs. We have three key observations about decoupling the static and dynamic human topology graph learning. ① Only modeling static topology decreases performance in T2M but significantly improves performance in S2G and M2D. We attribute this to the static topology ensuring the model grasps basic spatial relations between body parts, enhancing generalization across various generation scenarios. However, the additional learnable spatial structure module, unrelated to input, increases learning difficulty in the T2M task. ② Only

<table><tr><td>S2G-Method</td><td> $FID_{H} \downarrow$ </td><td> $FID_{B} \downarrow$ </td><td>Face L2 Loss ↓</td><td>Beat Align Score ↑</td><td>Div ↑</td><td>M2D-Method</td><td> $FID_{H} \downarrow$ </td><td> $FID_{B} \downarrow$ </td><td>Div ↑</td></tr><tr><td>Talkshow</td><td>26.713</td><td>74.824</td><td>7.791</td><td>6.947</td><td>13.472</td><td>Edge</td><td>93.430</td><td>108.507</td><td>13.471</td></tr><tr><td>EMAGE</td><td>39.094</td><td>90.762</td><td>7.680</td><td>7.727</td><td>13.065</td><td>Finedance</td><td>10.747</td><td>72.229</td><td>13.813</td></tr><tr><td>MCM</td><td>23.946</td><td>71.241</td><td>16.983</td><td>7.993</td><td>13.167</td><td>MCM</td><td>4.717</td><td>78.577</td><td>14.890</td></tr><tr><td>MotionCraft-Basic</td><td>18.486</td><td>27.023</td><td>10.097</td><td>8.098</td><td>10.334</td><td>MotionCraft-Basic</td><td>3.858</td><td>76.248</td><td>16.667</td></tr><tr><td>MotionCraft-Mix</td><td>12.882</td><td>25.187</td><td>8.906</td><td>8.226</td><td>12.595</td><td>MotionCraft-Mix</td><td>2.849</td><td>67.159</td><td>18.483</td></tr></table>

Table 3: Results of Speech-to-Gesture in BEAT2 and Music-to-Dance in FineDance of MC-Bench. We respectively evaluate the $FID_{H}$ and $FID_{B}$ , Face L2 Loss×10 $^{-8}$ , Beat Align Score×10 $^{-1}$ , and diversity for S2G and the $FID_{H}$ , $FID_{B}$ , and the diversity for M2D. Red background indicates best results, yellow background indicates second best results.

<table><tr><td colspan="2">Method</td><td colspan="5">HumanML3D (Text-to-Motion)</td><td colspan="5">BEAT2 (Speech-to-Gesture)</td><td colspan="3">Finedance (Music-to-Dance)</td></tr><tr><td>Dynamic-Spatial</td><td>Static-Spatial</td><td>Top-1 ↑</td><td>Top-2 ↑</td><td>Top-3 ↑</td><td>FID ↓</td><td>Div ↑</td><td> $FID_H \downarrow$ </td><td> $FID_B \downarrow$ </td><td>Face L2 ↓</td><td>Beat Align Score ↑</td><td>Div ↑</td><td> $FID_H \downarrow$ </td><td> $FID_B \downarrow$ </td><td>Div ↑</td></tr><tr><td>X</td><td>X</td><td>0.583</td><td>0.729</td><td>0.794</td><td>8.911</td><td>35.954</td><td>15.587</td><td>31.839</td><td>12.448</td><td>7.908</td><td>11.752</td><td>7.088</td><td>150.733</td><td>17.984</td></tr><tr><td>X</td><td>√</td><td>0.557</td><td>0.706</td><td>0.772</td><td>9.041</td><td>36.101</td><td>12.929</td><td>27.928</td><td>12.287</td><td>8.077</td><td>12.230</td><td>5.104</td><td>112.186</td><td>18.503</td></tr><tr><td>√</td><td>X</td><td>0.582</td><td>0.732</td><td>0.798</td><td>8.455</td><td>36.241</td><td>15.517</td><td>28.631</td><td>12.544</td><td>7.708</td><td>11.313</td><td>4.972</td><td>102.103</td><td>16.385</td></tr><tr><td>√</td><td>√</td><td>0.600</td><td>0.747</td><td>0.812</td><td>6.707</td><td>36.419</td><td>12.882</td><td>25.187</td><td>8.906</td><td>8.226</td><td>12.595</td><td>2.849</td><td>67.159</td><td>18.483</td></tr><tr><td colspan="2">MotionCraft-Tiny-(4, 64, 77M)</td><td>0.600</td><td>0.747</td><td>0.812</td><td>6.707</td><td>36.419</td><td>12.882</td><td>25.187</td><td>8.906</td><td>8.226</td><td>12.595</td><td>2.849</td><td>67.159</td><td>18.483</td></tr><tr><td colspan="2">MotionCraft-Small-(4, 128, 130M)</td><td>0.653</td><td>0.794</td><td>0.847</td><td>5.593</td><td>36.264</td><td>15.346</td><td>27.140</td><td>8.322</td><td>8.023</td><td>11.906</td><td>2.370</td><td>59.471</td><td>17.036</td></tr><tr><td colspan="2">MotionCraft-Small-(8, 64, 145M)</td><td>0.635</td><td>0.779</td><td>0.802</td><td>6.193</td><td>36.311</td><td>15.702</td><td>28.094</td><td>8.589</td><td>8.031</td><td>11.824</td><td>3.749</td><td>66.958</td><td>16.478</td></tr><tr><td colspan="2">MotionCraft-Medium-(8, 128, 250M)</td><td>0.647</td><td>0.785</td><td>0.854</td><td>5.670</td><td>36.384</td><td>14.937</td><td>23.498</td><td>8.125</td><td>8.089</td><td>10.962</td><td>3.904</td><td>75.412</td><td>16.507</td></tr><tr><td colspan="2">MotionCraft-Large-(16, 128, 478M)</td><td>0.604</td><td>0.744</td><td>0.809</td><td>7.872</td><td>36.169</td><td>15.964</td><td>27.476</td><td>9.036</td><td>7.969</td><td>10.625</td><td>4.837</td><td>77.341</td><td>16.426</td></tr></table>

Table 4: Ablation Study. (a) Ablation on model design (Upper half). The results suggest that jointly modeling dynamic and static human skeleton topologies significantly improves performance since this provides robust topology knowledge against distribution drifts. (b) Ablation on scaling up impacts (Lower half). We design four scaling model variants, where \*\*-(a, b, c) denotes model \*\* with a transformer layer, b body-part encoding dimension, and total c parameter counts. We observe a rise-then-fall performance trend across three types of tasks as the model size increases. Red background indicates best results.

![](images/cb19430b08d4e74bd4aa6074dba928b2f87aac4619f9b38e39895b92ca347cf1.jpg)  
Figure 5: Multimodal video generation application with our generated motions conditioned on music (upper row) or speech (lower row). We project them to 2D images to serve as motion conditions for MimicMotion (Zhang et al. 2024c).

modeling dynamic topology nearly brings no benefit. This is because the initial optimization of the input-adaptive dynamic topology adjacency matrix is complex, especially for transferring topology knowledge against distribution drifts, making it hard to converge to the correct dynamic topology graph (Zhang et al. 2024b). Joint modeling of static and dynamic topologies effectively captures motion knowledge against distribution drifts, as in human-centric research (Zeng et al. 2021). The static topology learns basic human structure, providing foundational spatial knowledge across tasks, while the dynamic topology adjusts according to specific motion distributions and control signals.

Scaling up impacts. Based on the acknowledgment of the scalability of transformer models, we explored the impact of model size on task performance. We increased the size of MotionCraft-Mix from 77M to 478M, observing a rise-then-fall performance trend across three types of tasks as the model size increased with limited data. This verifies that increasing the model's parameter size can enhance generative capabilities, but without a corresponding increase in high-quality data, model performance may decline.

# Application: Multimodal Video Generation

To demonstrate the downstream application, in Fig. 5, we present two animation videos driven by MotionCraft in M2D and S2G. Our generated motion sequences can be combined with any off-the-shelf human video generation framework, such as MimicMotion (Zhang et al. 2024c), AnimateAnyone (Hu et al. 2023), and VividPose (Wang et al. 2024), enabling users to customize videos of any character based on specific control signals, such as speech or music. Notably, unlike the traditional 2D keypoints estimated from videos, our generated 3D motion approach allows for flexible adjustment of camera parameters to project different visible body regions (e.g., full body or upper body, as in Fig. 5). More detailed visualizations are in our supplementary.

# Conclusion

In this paper, we proposed MotionCraft, a unified framework for whole-body human motion generation with plug-and-play multimodal controls that generalizes across different generative distributions and efficiently handles control signals of varying granularity. MotionCraft employs a coarse-to-fine training strategy that achieves fine-grained, plug-and-play control for different conditions (including text, speech and music) without the optimization burden of mixed training. Our core design is MC-Attn, which effectively learns and transfers motion knowledge across different distributions by parallel modeling the static and dynamic human topology graphs. We introduced MC-Bench, the first available multimodal whole-body motion generation benchmark based on the unified whole-body SMPL-X representation. Extensive experiments show that MotionCraft achieves a competitive performance on standard motion generation tasks against current state-of-the-art baselines.

# References

Bian, Y.; Ju, X.; Li, J.; Xu, Z.; Cheng, D.; and Xu, Q. 2024. Multi-Patch Prediction: Adapting Language Models for Time Series Representation Learning. In Forty-first International Conference on Machine Learning.

Chen, J.; Liu, Y.; Wang, J.; Zeng, A.; Li, Y.; and Chen, Q. 2024. DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation. In CVPR.

Chen, X.; Jiang, B.; Liu, W.; Huang, Z.; Fu, B.; Chen, T.; and Yu, G. 2023. Executing your Commands via Motion Diffusion in Latent Space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18000–18010.

Chung, J.; Wuu, C.-h.; Yang, H.-r.; Tai, Y.-W.; and Tang, C.-K. 2021. Haa500: Human-centric atomic action dataset with curated videos. In Proceedings of the IEEE/CVF international conference on computer vision, 13465–13474.

Gärtner, E.; Andriluka, M.; Coumans, E.; and Sminchisescu, C. 2022. Differentiable dynamics for articulated 3d human motion reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 13190–13200.

Girdhar, R.; El-Nouby, A.; Liu, Z.; Singh, M.; Alwala, K. V.; Joulin, A.; and Misra, I. 2023. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15180–15190.

Guo, C.; Zou, S.; Zuo, X.; Wang, S.; Ji, W.; Li, X.; and Cheng, L. 2022. Generating Diverse and Natural 3D Human Motions From Text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5152–5161.

Guo, C.; Zuo, X.; Wang, S.; Zou, S.; Sun, Q.; Deng, A.; Gong, M.; and Cheng, L. 2020. Action2motion: Conditioned generation of 3d human motions. In Proceedings of the 28th ACM International Conference on Multimedia, 2021–2029.

Hu, L. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8153–8163.

Hu, L.; Gao, X.; Zhang, P.; Sun, K.; Zhang, B.; and Bo, L. 2023. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. arXiv preprint arXiv:2311.17117.

Kim, J.; Kim, J.; and Choi, S. 2023. Flame: Free-form language-based motion synthesis & editing. In Proceedings of the AAAI Conference on Artificial Intelligence.

Li, R.; Yang, S.; Ross, D. A.; and Kanazawa, A. 2021. Learn to Dance with AIST++: Music Conditioned 3D Dance Generation. arXiv:2101.08779.

Li, R.; Zhao, J.; Zhang, Y.; Su, M.; Ren, Z.; Zhang, H.; Tang, Y.; and Li, X. 2023. FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10234–10243.

Liang, H.; Bao, J.; Zhang, R.; Ren, S.; Xu, Y.; Yang, S.; Chen, X.; Yu, J.; and Xu, L. 2024. Omg: Towards open-vocabulary motion generation via mixture of controllers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 482–493.

Lin, J.; Zeng, A.; Lu, S.; Cai, Y.; Zhang, R.; Wang, H.; and Zhang, L. 2023a. Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset. Advances in Neural Information Processing Systems.

Lin, J.; Zeng, A.; Wang, H.; Zhang, L.; and Li, Y. 2023b. One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer. In CVPR, 21159–21168.

Ling, Z.; Han, B.; Wong, Y.; Kangkanhalli, M.; and Geng, W. 2023. Mcm: Multi-condition motion synthesis framework for multi-scenario. arXiv preprint arXiv:2309.03031.

Liu, H.; Zhu, Z.; Becherini, G.; Peng, Y.; Su, M.; Zhou, Y.; Zhe, X.; Iwamoto, N.; Zheng, B.; and Black, M. J. 2024a. EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1144–1154.

Liu, H.; Zhu, Z.; Iwamoto, N.; Peng, Y.; Li, Z.; Zhou, Y.; Bozkurt, E.; and Zheng, B. 2022a. Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. In European conference on computer vision, 612–630. Springer.

Liu, J.; Dai, W.; Wang, C.; Cheng, Y.; Tang, Y.; and Tong, X. 2023. Plan, posture and go: Towards open-world text-to-motion generation. arXiv preprint arXiv:2312.14828.

Liu, X.; Wu, Q.; Zhou, H.; Xu, Y.; Qian, R.; Lin, X.; Zhou, X.; Wu, W.; Dai, B.; and Zhou, B. 2022b. Learning hierarchical cross-modal association for co-speech gesture generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10462–10472.

Liu, Y.; Zhang, K.; Li, Y.; Yan, Z.; Gao, C.; Chen, R.; Yuan, Z.; Huang, Y.; Sun, H.; Gao, J.; He, L.; and Sun, L. 2024b. Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arXiv:2402.17177.

Loper, M.; Mahmood, N.; Romero, J.; Pons-Moll, G.; and Black, M. J. 2015. SMPL: A Skinned Multi-Person Linear Model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6): 248:1–248:16.

Lu, S.; Chen, L.-H.; Zeng, A.; Lin, J.; Zhang, R.; Zhang, L.; and Shum, H.-Y. 2023. HumanTOMATO: Text-aligned Whole-body Motion Generation. arxiv:2310.12978.

Luo, M.; Hou, R.; Chang, H.; Liu, Z.; Wang, Y.; and Shan, S. 2024. M3-GPT: An Advanced Multimodal, Multi-task Framework for Motion Comprehension and Generation. arXiv preprint arXiv:2405.16273.

Ma, J.; Bai, S.; and Zhou, C. 2022. Pretrained diffusion models for unified human motion synthesis. arXiv 2022. arXiv preprint arXiv:2212.02837.

Mahmood, N.; Ghorbani, N.; Troje, N. F.; Pons-Moll, G.; and Black, M. J. 2019. AMASS: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, 5442–5451.

McFee, B.; Raffel, C.; Liang, D.; Ellis, D. P.; McVicar, M.; Battenberg, E.; and Nieto, O. 2015. librosa: Audio and music signal analysis in python. In SciPy, 18–24.   
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.   
Oord, A. v. d.; Li, Y.; and Vinyals, O. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.   
Pavlakos, G.; Choutas, V.; Ghorbani, N.; Bolkart, T.; Osman, A. A. A.; Tzionas, D.; and Black, M. J. 2019. Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 10975–10985.   
Petrovich, M.; Black, M. J.; and Varol, G. 2022. TEMOS: Generating diverse human motions from textual descriptions. In European Conference on Computer Vision, 480–497. Springer.   
Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684–10695.   
Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.   
Siyao, L.; Yu, W.; Gu, T.; Lin, C.; Wang, Q.; Qian, C.; Loy, C. C.; and Liu, Z. 2022. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11050–11059.   
Tang, Y.; Liu, J.; Liu, A.; Yang, B.; Dai, W.; Rao, Y.; Lu, J.; Zhou, J.; and Li, X. 2023. FLAG3D: A 3D Fitness Activity Dataset with Language Instruction. In CVPR.   
Team, G.; Anil, R.; Borgeaud, S.; Wu, Y.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.   
Tevet, G.; Raab, S.; Gordon, B.; Shafir, Y.; Cohen-or, D.; and Bermano, A. H. 2023. Human Motion Diffusion Model. In The Eleventh International Conference on Learning Representations.   
Trivedi, N.; Thatipelli, A.; and Sarvadevabhatla, R. K. 2021. NTU-X: an enhanced large-scale dataset for improving pose-based recognition of subtle human actions. In Proceedings of the Twelfth Indian Conference on Computer Vision, Graphics and Image Processing, 1–9.   
Tseng, J.; Castellon, R.; and Liu, K. 2023. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 448–458.   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Wang, Q.; Jiang, Z.; Xu, C.; Zhang, J.; Wang, Y.; Zhang, X.; Cao, Y.; Cao, W.; Wang, C.; and Fu, Y. 2024. VividPose: Advancing Stable Video Diffusion for Realistic Human Image Animation. arXiv preprint arXiv:2405.18156v1.   
Xia, X.; Liu, T.; Han, B.; Wang, N.; Gong, M.; Liu, H.; Niu, G.; Tao, D.; and Sugiyama, M. 2020. Part-dependent label noise: Towards instance-dependent label noise. NeurIPS, 33: 7597–7610.   
Yi, H.; Liang, H.; Liu, Y.; Cao, Q.; Wen, Y.; Bolkart, T.; Tao, D.; and Black, M. J. 2023. Generating Holistic 3D Human Motion from Speech. In CVPR.   
Zeng, A.; Sun, X.; Yang, L.; Zhao, N.; Liu, M.; and Xu, Q. 2021. Learning skeletal graph neural networks for hard 3d pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision, 11436–11445.   
Zhang, J.; Yan, H.; Xu, Z.; Feng, J.; and Liew, J. H. 2023a. MagicAvatar: Multi-modal Avatar Generation and Animation. In arXiv:2308.14748.   
Zhang, J.; Zhang, Y.; Cun, X.; Huang, S.; Zhang, Y.; Zhao, H.; Lu, H.; and Shen, X. 2023b. T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).   
Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3836–3847.   
Zhang, M.; Cai, Z.; Pan, L.; Hong, F.; Guo, X.; Yang, L.; and Liu, Z. 2024a. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence.   
Zhang, M.; Jin, D.; Gu, C.; Hong, F.; Cai, Z.; Huang, J.; Zhang, C.; Guo, X.; Yang, L.; He, Y.; et al. 2024b. Large motion model for unified multi-modal motion generation. arXiv preprint arXiv:2404.01284.   
Zhang, M.; Li, H.; Cai, Z.; Ren, J.; Yang, L.; and Liu, Z. 2023c. FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing. NeurIPS.   
Zhang, S.; Ma, Q.; Zhang, Y.; Qian, Z.; Kwon, T.; Pollefeys, M.; Bogo, F.; and Tang, S. 2022. Egobody: Human body shape and motion of interacting people from head-mounted devices. In European conference on computer vision, 180–200. Springer.   
Zhang, Y.; Gu, J.; Wang, L.-W.; Wang, H.; Cheng, J.; Zhu, Y.; and Zou, F. 2024c. MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance. arXiv preprint arXiv:2406.19680.   
Zhou, Y.; Barnes, C.; Lu, J.; Yang, J.; and Li, H. 2019. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5745–5753.   
Zhou, Z.; Wan, Y.; and Wang, B. 2023. A unified framework for multimodal, multi-part human motion synthesis. arXiv preprint arXiv:2311.16471.

# Appendix

# Visualization Results

Due to the limitations of the PDF's static format and the page limit, additional visualizations and comparisons are available in the supplementary and project page, which include generation visualizations for individual tasks such as Text-to-Motion (T2M), Music-to-Dance (M2D), and Speech-to-Gesture (S2G), as well as visualizations of plug-and-play control generation within a single long sequence. The video also demonstrates the application of our generated motion sequences in video production and character animation.

# Related Works

Human Motion Generation Models Conditioned human motion generation models have made significant progress, including text-to-motion (T2M) (Tevet et al. 2023; Zhang et al. 2023b; Liu et al. 2023; Zhang et al. 2024a, 2023c; Liang et al. 2024), speech-to-gesture (S2G) (Yi et al. 2023; Chen et al. 2024; Liu et al. 2022b), and music-to-dance (M2D) (Li et al. 2023; Tseng, Castellon, and Liu 2023; Siyao et al. 2022). In text-to-motion, models (Tevet et al. 2023; Chen et al. 2023; Zhang et al. 2023b; Liu et al. 2023; Zhang et al. 2024a, 2023c; Liang et al. 2024) achieve text-controlled motion generation with semantic consistency by applying advanced generative models and aligning motion and text feature domains. For speech-to-gesture (Yi et al. 2023; Chen et al. 2024; Liu et al. 2022b), many efforts focus on mapping speech to human gestures through rhythm alignment and character style learning. Additionally, numerous studies (Li et al. 2023; Tseng, Castellon, and Liu 2023; Siyao et al. 2022) design spatial and temporal coherence constraints to ensure that models learn the corresponding style and rhythm from the input music. Recently, increasing attention has been given to multimodal motion generation (Ling et al. 2023; Zhang et al. 2024b; Luo et al. 2024). $M^3$ -GPT (Luo et al. 2024) injects quantized condition tokens into the vocabulary of large language models to achieve motion understanding and generation, but it overlooks the modeling of human topology priors. Motion-Verse (Zhang et al. 2024b) incorporates dynamic attention to assess relationships among body parts but fails to capture the overall static human topology, leading to limited generalization and increased optimization complexity. Furthermore, it employs mixed training across all conditions based on ImageBind (Girdhar et al. 2023), which creates optimization challenges when learning conditions of varying granularity simultaneously and needs retraining for new control signals. MCM (Ling et al. 2023) attempts to address the optimization confusion of mixed training based on the ControlNet (Zhang, Rao, and Agrawala 2023) architecture, but it neglects any modeling of human topology structure, resulting in poor generalization across generation scenarios. Moreover, MCM only focuses on tensor movements, lacking the ability to generate whole-body motion. Compared to previous methods in Tab. 1, MotionCraft generates whole-body motion under varying control signals with plug-and-play capability by using MC-Attn to capture static human topology and domain-specific dynamic skeleton relationships, incorporating control branches, and employing a coarse-to-fine training strategy.

Human Motion Generation Benchmarks Various conditioned human motion generation benchmarks have been constructed in recent years. For T2M, researchers have curated datasets encompassing action categories (Chung et al. 2021; Trivedi, Thatipelli, and Sarvadevabhatla 2021), sequential action labels (Zhang et al. 2022; Guo et al. 2020), and arbitrary natural language descriptions (Lin et al. 2023a; Guo et al. 2022; Tang et al. 2023) at various abstraction levels. Specifically, AMASS (Mahmood et al. 2019) consolidates 15 optical marker-based motion capture datasets into a comprehensive collection based on SMPL (Loper et al. 2015) representation. HumanML3D (Guo et al. 2022) extracted a high-quality subset within AMASS (Mahmood et al. 2019) based on H3D format for torso-only generation, including three arbitrary natural language descriptions per motion clip from diverse annotators. For M2D, AIST++ (Li et al. 2021) reconstructs 5 hours of dance based on SMPL (Loper et al. 2015) format from videos, despite its significant reconstruction error, and lack of capture of hand movements. Finedance (Li et al. 2023) collects dances of 14.6 hours across 22 genres and supplements the dataset with detailed gestures using the SMPL-H (Pavlakos et al. 2019) format. For S2G, datasets (Liu et al. 2024a, 2022a; Yi et al. 2023) are gathered from pseudo-labeled (PGT) and motion-captured sources. Mocap datasets are generally preferred due to significant errors in monocular 3D pose estimation in PGT (Gärtner et al. 2022). Recently, BEAT2 (Liu et al. 2024a) and BEAT (Liu et al. 2022a) have emerged as the most popular benchmarks, celebrated for their diverse range of motion and extensive data volume. BEAT2, building upon BEAT, utilizes SMPL-X and FLAME (Kim, Kim, and Choi 2023) to achieve higher-quality unified mesh-level data. Despite these developments, no publicly available benchmark supports unified representation for multimodal whole-body motion generation.

# MC-Bench Construction

Motion Representation From body-only motion generation (Guo et al. 2022; Li et al. 2021; Ling et al. 2023) to whole-body motion generation (Lu et al. 2023; Li et al. 2023; Liu et al. 2024a; Zhang et al. 2024b), previous research has explored various motion representations in generation tasks, including the default axis-angle input based on SMPL mesh parameters (Loper et al. 2015), 6D rotation (Li et al. 2023), quaternion (Pavlakos et al. 2019), and the extended H3D-format from SMPL (Guo et al. 2022), which adds redundant information like joint positions and velocities. In recent years, SMPL-X (Pavlakos et al. 2019), an extension of SMPL, has incorporated hand modeling to enable finer-grained finger joint modeling. Therefore, considering the practicality and efficiency of motion representation, we use the default axis-angle input of SMPL-X to model the main body and hands. Specifically, the $i$ -th pose is defined by a tuple of root axis-angle $(\dot{r}^r \in \mathbb{R}^3)$ around the X(Y and Z)-axis, root trajectory $(\dot{r}^t \in \mathbb{R}^3)$ along the X(Y and Z)-axis,

local joints axis-angle rotations $(\theta^{r} \in \mathbb{R}^{3N})$ , where N denotes the number of whole body joints, including both body joints and hand joints. For face motion representations, we follow the MotionX (Lin et al. 2023a) to adopt the $f^{s} \in R^{100}$ to represent the face shape, $f^{e} \in R^{50}$ in the Flame Format (Kim, Kim, and Choi 2023) to represent the face expression, and jaw axis-angle rotation $(\theta^{j} \in \mathbb{R}^{3})$ for jaw modeling. Additionally, we employ the standard SMPL-X model 10-dimensional parameters $\theta^{b} \in {}^{10}$ to represent the body shape. Thus, we represent the whole-body motion as $m_{i} = \{\dot{r}^{r}, \dot{r}^{t}, \theta^{r}, f^{s}, f^{e}, \theta^{j}, \theta^{b}\}$ .

Text-to-Motion Subset Construction In the first phase of semantic text-to-motion pre-training, our data primarily consists of three parts, as follows:

1. HumanML3D (Guo et al. 2022) is a representative 3D motion-text dataset containing 14,616 high-quality human motions paired with 44,970 text captions. Instead of using the body-only H3D format from the original HumanML3D, we re-extracted the corresponding instances from its original AMASS (Mahmood et al. 2019) data in the SMPL-X format and processed each motion frame based on our SMPL-X axis-angle format, setting the corresponding SMPL-X (Pavlakos et al. 2019) representation to zero for any missing body parts. The text was filtered and processed according to the original HumanML3D text caption processing workflow.   
2. BEAT2(Liu et al. 2024a) is a speech-to-gesture dataset that includes various speaking styles and speaker IDs. It provides SMPL-X (Pavlakos et al. 2019) axis-angle rotation motion representation, from which we directly extract the corresponding rotation information based on our whole-body motion format $\mathbf{m}_i = \{\dot{r}^r,\dot{r}^t,\theta^r,\mathbf{f}^s,\mathbf{f}^e,\theta^j,\theta^b\}$ . For the text part, we generate corresponding pseudo-semantic text captions using simple rules, such as "A person is giving a speech, and the content is ...".   
3. FineDance(Li et al. 2023) is currently one of the leading music-to-dance datasets in terms of choreography diversity and data volume, originally providing body-hand data representation based on SMPL-H (Pavlakos et al. 2019) rot6D. We first convert the simplified rot6D rotation matrix into the axis-angle format around the XYZ axis, leveraging the equivalence between different rotation representations (Zhou et al. 2019). Instead of using the official SMPL-H (Pavlakos et al. 2019) to SMPL-X (Pavlakos et al. 2019) retargeting optimization method, we directly map SMPL-H parameters to SMPL-X (Pavlakos et al. 2019). Our qualitative and quantitative experiments demonstrate that this simple approach is effective, with negligible retargeting errors. For the textual part, we apply basic rules to generate pseudo-semantic captions matching the corresponding music segments, such as "A dancer is performing a street dance in the Jazz style to the rhythm of the wildfire."

Speech-to-Gesture Subset Construction For all speeches in BEAT2 (Liu et al. 2024a), we use Librosa (McFee et al. 2015) to extract 2-dimensional temporal speech features related to speech prosody. The audio is sampled at 76,800 Hz with a hop size of 512. We segment the motion sequences and corresponding speech into 64-frame segments with a stride of 64 frames. In subsequent generations, we employ the outpainting-based sampling strategy from DiffSHEG (Chen et al. 2024) to achieve long-term gesture generation. The pre-process details of motion representation and semantic text captioning have been thoroughly explained above, so they will not be repeated here.

Music-to-Dance Subset Construction For all music in FineDance (Li et al. 2023), we use Librosa (McFee et al. 2015) to extract 35-dimensional temporal music features. The audio is sampled at 76,800 Hz with a hop size of 512. We segment the motion sequences and corresponding music into 120-frame segments with a stride size of 30 frames. The pre-process details of motion representation and semantic text captioning have been thoroughly explained above, so they will not be repeated here.

# Experiments

Implementation Details of MotionCraft We employ a 4-layer motion diffusion transformer as the backbone of MotionCraft, featuring a latent dimension of $12 \times 64$ and a feedforward embedding size of 256, where 12 corresponds to the number of body parts and 64 denotes the dimensionality of each body-specific hidden state. For the control branch of MotionCraft, we set the number of copied MotionCraft blocks to 2, representing half of the total. The encoder design for various low-level control signals (speech or music) aligns with the baselines (Liu et al. 2024a; Li et al. 2023). For the text encoder, we utilize a frozen CLIP ViT-B/32 encoder, enhanced with two additional transformer encoder layers. In the diffusion model, the variances $\beta_{t}$ are predefined to linearly range from 0.0001 to 0.02, with 1000 noising steps. Following MDM (Tevet et al. 2023), we set $x_{start}$ as the diffusion prediction goal instead of the noise. The model is trained using the Adam optimizer, starting with a learning rate of $2 \times 10^{-4}$ , which decays to $2 \times 10^{-5}$ via a cosine schedule. Training occurs on $8 \times$ NVIDIA Tesla V100-32GB GPUs, with a batch size of 64 per GPU, and takes approximately 48 hours.

Implementation Details of Text-Motion Retrieval Pre-Training in Evaluation Since we used SMPL-X-based axis-angle as the motion representation, the motion encoder and text encoder from previous studies could not be directly applied for evaluation. Therefore, following Human-Tomato (Lu et al. 2023), we retrained a text-whole-body-motion retrieval model specifically for our SMPL-X axis-angle motion representation to assess performance in a contrastive learning approach. This retrieval model employs a VAE-based architecture (Petrovich, Black, and Varol 2022) consisting of a motion encoder, a text encoder, and a motion decoder. The training objective is the weighted sum of:

$$
\min \mathcal {L} _ {r e c} + \lambda_ {K L} \mathcal {L} _ {K L} + \lambda_ {E} \mathcal {L} _ {E} + \lambda_ {N C E} \mathcal {L} _ {N C E},
$$

where the four loss terms are reconstruction loss, Kullback-Leibler (KL) divergence loss, cross-modal embedding simi-

<table><tr><td rowspan="2">Method</td><td colspan="3">R Precision</td><td rowspan="2">FID ↓</td><td rowspan="2">Div ↑</td><td rowspan="2">MM Dist↓</td></tr><tr><td>Top-1 ↑</td><td>Top-2 ↑</td><td>Top-3 ↑</td></tr><tr><td>GT</td><td> $0.511^{\pm 0.003}$ </td><td> $0.703^{\pm 0.003}$ </td><td> $0.797^{\pm 0.002}$ </td><td> $0.002^{\pm 0.000}$ </td><td> $9.503^{\pm 0.065}$ </td><td> $2.974^{\pm 0.008}$ </td></tr><tr><td>T2M-GPT(Zhang et al. 2023b)</td><td> $0.491^{\pm 0.003}$ </td><td> $0.680^{\pm 0.003}$ </td><td> $0.775^{\pm 0.002}$ </td><td> $0.116^{\pm 0.004}$ </td><td> $9.761^{\pm 0.081}$ </td><td> $3.118^{\pm 0.011}$ </td></tr><tr><td>MDM(Tevet et al. 2023)</td><td> $0.418^{\pm 0.005}$ </td><td> $0.604^{\pm 0.005}$ </td><td> $0.707^{\pm 0.004}$ </td><td> $0.489^{\pm 0.025}$ </td><td> $9.450^{\pm 0.066}$ </td><td> $3.630^{\pm 0.023}$ </td></tr><tr><td>MotionDiffuse(Zhang et al. 2024a)</td><td> $0.491^{\pm 0.001}$ </td><td> $0.681^{\pm 0.001}$ </td><td> $0.782^{\pm 0.001}$ </td><td> $0.630^{\pm 0.001}$ </td><td> $9.410^{\pm 0.049}$ </td><td> $3.113^{\pm 0.001}$ </td></tr><tr><td>FineMoGen(Zhang et al. 2023c)</td><td> $0.504^{\pm 0.002}$ </td><td> $0.690^{\pm 0.002}$ </td><td> $0.784^{\pm 0.002}$ </td><td> $0.151^{\pm 0.008}$ </td><td> $9.263^{\pm 0.094}$ </td><td> $2.998^{\pm 0.008}$ </td></tr><tr><td>Motion-Verse(Zhang et al. 2024b)</td><td> $0.496^{\pm 0.002}$ </td><td> $0.685^{\pm 0.002}$ </td><td> $0.785^{\pm 0.002}$ </td><td> $0.415^{\pm 0.002}$ </td><td> $9.176^{\pm 0.074}$ </td><td> $3.087^{\pm 0.012}$ </td></tr><tr><td>MCM(Ling et al. 2023)</td><td> $0.494^{\pm 0.003}$ </td><td> $0.682^{\pm 0.005}$ </td><td> $0.777^{\pm 0.003}$ </td><td> $0.075^{\pm 0.003}$ </td><td> $9.484^{\pm 0.074}$ </td><td> $3.086^{\pm 0.011}$ </td></tr><tr><td>MotionCraft-Basic</td><td> $0.501^{\pm 0.003}$ </td><td> $0.697^{\pm 0.003}$ </td><td> $0.796^{\pm 0.002}$ </td><td> $0.173^{\pm 0.002}$ </td><td> $9.543^{\pm 0.098}$ </td><td> $3.025^{\pm 0.008}$ </td></tr></table>

Table 5: Results of text-to-motion in origin HumanML3D benchmark. We compare the results of text-to-motion generation between ours and the SOTA methods. Our method achieves better semantic relevance, fidelity, and diversity performances. Red background indicates best results, yellow background indicates second best results.

<table><tr><td colspan="2">Method</td><td colspan="5">HumanML3D (Text-to-Motion)</td><td colspan="5">BEAT2 (Speech-to-Gesture)</td><td colspan="3">Finedance (Music-to-Dance)</td></tr><tr><td>Local-Unfreeze</td><td>Temporal-Patching</td><td>Top-1 ↑</td><td>Top-2 ↑</td><td>Top-3 ↑</td><td>FID ↓</td><td>Div ↑</td><td> $FID_H \downarrow$ </td><td> $FID_B \downarrow$ </td><td>Face L2 ↓</td><td>Beat Align Score ↑</td><td>Div ↑</td><td> $FID_H \downarrow$ </td><td> $FID_B \downarrow$ </td><td>Div ↑</td></tr><tr><td>X</td><td>X</td><td>0.653</td><td>0.794</td><td>0.847</td><td>5.593</td><td>36.264</td><td>15.346</td><td>27.140</td><td>8.322</td><td>8.023</td><td>11.024</td><td>2.370</td><td>59.471</td><td>17.036</td></tr><tr><td>X</td><td>√</td><td>0.628</td><td>0.776</td><td>0.834</td><td>5.944</td><td>36.189</td><td>17.583</td><td>27.605</td><td>8.792</td><td>8.007</td><td>10.920</td><td>3.496</td><td>64.784</td><td>16.371</td></tr><tr><td>√</td><td>X</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>17.962</td><td>26.556</td><td>8.561</td><td>8.035</td><td>11.248</td><td>2.493</td><td>56.847</td><td>16.894</td></tr><tr><td>√</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>18.554</td><td>28.434</td><td>8.630</td><td>7.980</td><td>11.157</td><td>3.229</td><td>61.518</td><td>16.502</td></tr></table>

Table 6: Additional Ablation Study. we explored the second stage model training strategy about the body-wise encoder (decoder) and motion sequence temporal relationship modeling paradigm. The “Local-Unfreeze” column indicates that during the second phase, only specific body parts corresponding to certain control signals are unfrozen in the body-wise encoder and decoder. For instance, in the Speech-to-Gesture task, only the encoders and decoders for hands and face are unfrozen, while in Music-to-Dance, only the encoder and decoder for the hands are unfrozen. The "Temporal-Patching" column means performing patching operations on adjacent frames, compressing a specified number of neighboring frames into a single basic unit on the time dimension for modeling, instead of treating each frame’s motion as a basic unit on the time dimension.

Red background indicates best results

larity loss, and InfoNCE (Oord, Li, and Vinyals 2018) loss, respectively. The hyperparameters are set to $\lambda_{KL} = 1 \times 10^{-5}$ , $\lambda_E = 1 \times 10^{-5}$ , $\lambda_{NCE} = 1 \times 10^{-1}$ .

More Results on Text-to-Motion To provide a more comprehensive comparison, in addition to evaluating on the HumanML3D subset with whole-body format SMPL-X in MC-Bench (Tab. 2), we also compare MotionCraft with current SOTA baselines (Zhang et al. 2023c; Ling et al. 2023; Zhang et al. 2024b,a; Tevet et al. 2023; Zhang et al. 2023b) on the original HumanML3D (Guo et al. 2022) using the body-only H3D format, which contains redundant information. Quantitative comparison results are shown in Tab. 5. It is evident that on the original HumanML3D text-to-motion benchmark, MotionCraft also achieves better text-guided generation capability, diversity, and motion generation quality.

Notably, in the HumanML3D subset of MC-Bench, the limited evaluation capabilities observed in the original HumanML3D benchmark with torso-only representation—where performance differences between models were minimal—were significantly enhanced. This improvement arises because the whole-body SMPL-X representation necessitates the model to generate torso movements, gestures, and expressions, rather than focusing solely on the torso.

More Results on Ablation Study In addition to the ablation experiments on dynamic-static motion topology modeling and model parameter scaling in Tab. 4, we further explored the second stage model training strategy and motion sequence temporal relationship modeling paradigm. The results are in Tab. 6.

Ablation on the second stage body-wise encoder and decoder training strategy. The “Local-Unfreeze” column in Tab. 6 indicates that during the second phase, only specific body parts corresponding to certain control signals are unfrozen in the body-wise encoder and decoder. For instance, in the Speech-to-Gesture task, only the encoders and decoders for hands and face are unfrozen, while in Music-to-Dance, only the encoder and decoder for the hands are unfrozen. Rows one and three of Tab. 6 clearly show that fully unfreezing the body-wise encoder and decoder during the second phase enhances encoding and decoding optimization for specific body parts in the generation scenario, thus improving the generation capabilities for targeted scenarios (e.g., hand and face modeling in Speech-to-Gesture, hand modeling in Music-to-Dance). Conversely, partial unfreezing helps retain the human body topology knowledge learned during the first phase of text semantic pre-training, thereby stabilizing the overall generation capability for full-body actions on downstream generation tasks.

Ablation on motion sequence temporal relationship modeling. Currently, there are two classic approaches for modeling temporal relationships in motion sequences: treating each frame's motion as a basic unit on the time dimension for sequence modeling, and performing patching operations on adjacent frames, compressing a specified number of neighboring frames into a single basic unit on the time di-

mension for modeling. In general time series analysis, the latter method has been widely shown to significantly improve performance in transformer-based models (Nie et al. 2022), as it mitigates the impact of extreme values and eliminates redundant information, allowing for higher information density in sub-sequence relationship modeling. However, as shown in Table 6, the conclusions for temporal dynamic modeling of motion sequences appear to be the opposite of those for general time series modeling. This discrepancy can be attributed to differences in data representation between motion and general time series:

- High-quality motion data (Guo et al. 2022; Li et al. 2023; Liu et al. 2024a) from motion capture systems generally do not suffer from extreme values, whereas general time series data (Nie et al. 2022) can be affected by various factors such as collection environment and economic or cultural influences, leading to inconsistent quality.   
- In SMPL-X axis-angle motion format, the rotation angles of child joints are influenced by their parent joints, meaning small changes in parent joint values can lead to significant motion changes due to the whole-body topological structure (Loper et al. 2015; Pavlakos et al. 2019). In other words, the axis-angle motion representation in SMPL-X is far more sensitive than general time series data, and using compressed sub-sequences as modeling units can introduce significant cumulative errors.

# Broader Impact and Limitation

In this section, we will discuss the possible social impact and limitations of MotionCraft.

Broader Impact. First, we explore the task of whole-body motion generation under multimodal controls and establish the first multimodal motion generation benchmark with a unified whole-body motion representation based on three high-quality single-control signal motion generation datasets. These could serve as a foundation for the multimodal control motion generation research community. Additionally, with large-scale motion data training under multimodal controls, our trained MotionCraft can function as a motion prior for other research, such as HumanTomato (Lu et al. 2023) and VPoser (Pavlakos et al. 2019). It can also help address noisy annotations in the current Motion Capture process (Xia et al. 2020; Lin et al. 2023b). Finally, expressive, multimodal-controllable, and high-quality motion generation can be applied to various practical downstream scenarios, including but not limited to human video generation, motion animations, and robotics.

Limitation. While this work achieves significant progress in whole-body motion generation under multimodal controls, some limitations remain. First, the utilization of more high-quality semantic text descriptions for whole-body motion generation requires further investigation. This work follows previous approaches by using sequential semantic descriptions without incorporating frame-level or fine-grained whole-body descriptions throughout the two-stage training. This issue is particularly important in the Speech-to-Gesture (Liu et al. 2024a) and Music-to-Dance (Li et al. 2023) tasks, where sequence-level semantic captions are naturally lacking, and pseudo-text descriptions are used to supplement them. Second, the current motion representation employs the axis-angle SMPL-X format. Although this allows direct rendering of the corresponding mesh via the SMPL-X model (Pavlakos et al. 2019), the 6D parameters for root rotation and root trajectory can significantly affect the overall motion generation quality, causing additional fluctuations during model training. In the future, we will explore adding projected 3D joint positions, similar to the redundant H3D format in HumanML3D (Guo et al. 2022), to provide extra constraints on root rotation and root trajectory. Additionally, we plan to unify more multimodal datasets to advance the development of a superior multimodal whole-body motion generation model.

2023) tasks, where sequence-level semantic captions are naturally lacking, and pseudo-text descriptions are used to supplement them. Second, the current motion representation employs the axis-angle SMPL-X format. Although this allows direct rendering of the corresponding mesh via the SMPL-X model (Pavlakos et al. 2019), the 6D parameters for root rotation and root trajectory can significantly affect the overall motion generation quality, causing additional fluctuations during model training. In the future, we will explore adding projected 3D joint positions, similar to the redundant H3D format in HumanML3D (Guo et al. 2022), to provide extra constraints on root rotation and root trajectory. Additionally, we plan to unify more multimodal datasets to advance the development of a superior multimodal whole-body motion generation model.