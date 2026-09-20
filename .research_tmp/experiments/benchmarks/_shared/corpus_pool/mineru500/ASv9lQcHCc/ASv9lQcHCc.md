# GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching

Haibin He $^{1*}$ , Maoyuan Ye $^{1*}$ , Jing Zhang $^{1\dagger}$ , Juhua Liu $^{1\dagger}$ , Bo Du $^{1}$ , Dacheng Tao $^{2}$

$^{1}$ School of Computer Science, National Engineering Research Center for Multimedia Software, and Institute of Artificial Intelligence, Wuhan University, China

$^{2}$ College of Computing & Data Science at Nanyang Technological University {haibinhe, yemaoyuan, liujuhua, dubo}@whu.edu.com jingzhang.cv@gmail.com, dacheng.tao@gmail.com https://github.com/Hxyz-123/GoMatching

# Abstract

Beyond the text detection and recognition tasks in image text spotting, video text spotting presents an augmented challenge with the inclusion of tracking. While advanced end-to-end trainable methods have shown commendable performance, the pursuit of multi-task optimization may pose the risk of producing sub-optimal outcomes for individual tasks. In this paper, we identify a main bottleneck in the state-of-the-art video text spotter: the limited recognition capability. In response to this issue, we propose to efficiently turn an off-the-shelf query-based image text spotter into a specialist on video and present a simple baseline termed GoMatching, which focuses the training efforts on tracking while maintaining strong recognition performance. To adapt the image text spotter to video datasets, we add a rescoring head to rescore each detected instance's confidence via efficient tuning, leading to a better tracking candidate pool. Additionally, we design a long-short term matching module, termed LST-Matcher, to enhance the spotter's tracking capability by integrating both long- and short-term matching results via Transformer. Based on the above simple designs, GoMatching delivers new records on ICDAR15-video, DSText, BOVText, and our proposed novel test with arbitrary-shaped text termed ArTVideo, which demonstrates GoMatching's capability to accommodate general, dense, small, arbitrary-shaped, Chinese and English text scenarios while saving considerable training budgets.

# 1 Introduction

Text spotting has received increasing attention due to its various applications, such as video retrieval $[1]$ and autonomous driving $[2]$ . Recently, numerous image text spotting (ITS) methods $[3–6]$ that simultaneously tackle text detection and recognition, have attained extraordinary accomplishment. In the video realm, video text spotting (VTS) involves a tracking task additionally. Although VTS methods $[7–12]$ make significant progress, a substantial discrepancy persists when compared to ITS. We observe that the text recognition proficiency of VTS models is far inferior to ITS models. To investigate this issue, we compare the state-of-the-art (SOTA) VTS model TransDETR $[12]$ and ITS model Deepsolo $[5]$ for image-level text spotting performance on ICDAR15-video $[13]$ and our established ArTVideo (i.e., Arbitrary-shaped Text in Video) test set (Sec.4.1) which comprises about 30% curved text.

As illustrated in Fig. 1(a), even when evaluating the image-level spotting performance on the VTS model's training set, the F1-score of TransDETR is only comparable to the zero-shot performance of

![](images/a3a615737460f4df6b515faed07d1b70ae4ab5336760127837bbdb5e0c381b54.jpg)

<details>
<summary>bar</summary>

| Image-Level Performance | Category | VTS model (on training set) | ITS model (cross-dataset, zero-shot) | ITS model (cross-dataset, zero-shot) |
|---|---|---|---|---|
| ICDAR15-Video Spotting F1-score (↑) | 51.5 | 51.4 | 38.6 | 23.1 |
| ICDAR15-Video Gap between Spot. & Det. (↓) | 51.5 | 51.4 | 38.6 | 23.1 |
| ArTVideo Spotting F1-score (↑) | 35.9 | 79.1 | 35.9 | 79.1 |
| ArTVideo Gap between Spot. & Det. (↓) | 42.8 | 8.0 | 42.8 | 8.0 |
</details>

(a)

![](images/b13a4398789b3b86c9f15df75ac82cc7aa07d085b3b56d9233478d12d5b641a1.jpg)

<details>
<summary>scatter</summary>

| Model       | GPU Hours | MOTA   |
|-------------|-----------|--------|
| GoMatching  | 0         | 72.04  |
| TransDETR   | 200       | 19.1GB |
</details>

(b)   
Figure 1: (a) 'Gap between Spot. & Det.': the gap between spotting and detection F1-score. As the spotting task involves recognizing the results of the detection process, the detection score is indeed the upper bound of spotting performance. The larger the gap, the poorer the recognition ability. Compared to the ITS model (Deepsolo [5]), the VTS model (TransDETR [12]) presents unsatisfactory image-level text spotting F1-scores, which lag far behind its detection performance, especially on ArTVideo with curved text. It indicates recognition capability is a main bottleneck in the VTS model. (b) GoMatching outperforms TransDETR by over 12 MOTA on ICDAR15-video while saving 197 training GPU hours and 10.8GB memory. Notice that since the pre-training strategies and settings vary between TransDETR and GoMatching, the comparison is focused on the fine-tuning stage.

Deepsolo. The performance of the VTS model on ArTVideo is much worse. Moreover, there is a huge gap between the spotting and detection-only performance of the VTS model, which indicates that the recognition capability is the main bottleneck. We attribute this discrepancy to two key aspects: 1) the model architecture and 2) the training data. First, in terms of model architecture, ITS studies [5, 6] have presented the advantages of employing advanced query formulation for text spotting in DETR frameworks [14, 15]. In contrast, existing Transformer-based VTS models still rely on Region of Interest (RoI) components or simply cropping detected text regions for recognition. On the other hand, some studies [16, 17] have indicated that there exists optimization conflict in detection and association during the end-to-end training of MOTR [18]. We hold that TransDETR [12], which further incorporates text recognition into MOTR-based architecture, may also suffer from optimization conflict. Second, regarding the training data, most text instances in current video datasets [13, 10, 19] are straight or oriented, and the bounding box labels are only quadrilateral, which constrains the data diversity and recognition performance as well. Overall, the limitations in model architecture and data probably lead to the unsatisfactory text spotting performance of the SOTA VTS model. Hence, leveraging model and data knowledge from ITS presents considerable value for VTS.

To achieve this, a straightforward approach is to take an off-the-shelf SOTA image text spotter and focus the training efforts on tracking across frames, akin to tracking-by-detection methods. An important question is how to efficiently incorporate a RoI-free image text spotter for VTS. In this paper, we propose a simple baseline via lonG and short term Matching, termed GoMatching, which leverages an off-the-shelf RoI-free image text spotter to identify text from each single frame and associates text instances across frames with a strong tracker.

Specifically, we select the state-of-the-art DeepSolo $[5]$ as the image text spotter and design a Long-Short Term Matching-based tracker termed LST-Matcher. Initially, to adapt the DeepSolo to video datasets while preserving its inherent knowledge, we freeze Deepsolo and introduce a rescoring mechanism. This mechanism entails training an additional lightweight text classifier called rescoring head via efficient tuning, and recalibrating confidence scores for detected instances to mitigate performance degradation caused by the image-video domain gap. The final score for each instance is determined by a fusion operation between the original score provided by the image text spotter and the calibrated score acquired from the rescoring head. The identified text instances are then sent to LST-Matcher for association. LST-Matcher can effectively harnesses both long- and short-term information, making it a highly capable tracker. As a result, our baseline significantly surpasses existing SOTA methods by a large margin with much lower training costs, as shown in Fig. 1(b).

In summary, the contribution of this paper is threefold. 1) We identify the limitations in current VTS methods and propose a novel and simple baseline, which leverages an off-the-shelf image text spotter with a strong customized tracker. 2) We introduce the rescoring mechanism and long-short term matching module to adapt image text spotter to video datasets and enhance the tracker's capabilities. 3) We establish the ArTVideo test set for addressing the absence of curved texts in current video datasets and evaluating the text spotters on videos with arbitrary-shape text. Extensive

experiments on public challenging datasets and the established ArTVideo test set demonstrate the effectiveness of our baseline and its outstanding performance with less training budgets. For example, GoMatching achieves the highest ranking on ICDAR15-video and DSText. Especially on bilingual dataset BOVText, GoMatching obtains a 45% improvement on MOTA compared to the recorded best performance [11]. On curved text dataset ArTVideo, GoMatching also surpasses previous SOTA method [12] by a substantial margin.

# 2 Related Works

# 2.1 Multi-Object Tracking

Multi-object tracking methods follow the tracking-by-detection (TBD) or tracking-by-query-propagation (TBQP) pipeline. TBD methods $[20–22]$ employ detectors for localization and then use association algorithms to get object trajectories. Different from extending tracks frame-by-frame, GTR $[23]$ proposes to generate entire trajectories at once in Transformer. TBQP paradigm extends query-based object detectors $[14, 15]$ to tracking. MOTR $[18]$ detects object locations and serially updates its tracking queries for detecting the same items in the following frames, achieving an end-to-end solution. However, MOTR suffers from optimization conflict between detection and association $[16, 17]$ , resulting in inferior detection performance. For the VTS task which additionally involves text recognition, a naive way of training all modules end-to-end may also lead to optimization conflict. In contrast, we explore inheriting prior knowledge of text spotting from ITS models while focusing on the tracking task.

# 2.2 Image Text Spotting

Early approaches $[24–26]$ crafted RoI-based modules to bridge text detection and recognition. However, these methods ignored one vital issue, i.e., the synergy problem between the two tasks. To overcome this dilemma, recent Transformer-based methods $[27, 3, 28, 6]$ get rid of the fetters of RoI modules, and chase a better representation for the two tasks. For example, DETR-based TESTR $[4]$ uses two decoders for each task in parallel. In contrast, DeepSolo $[5]$ proposes a unified and explicit query form for the two tasks, without harnessing dual decoders. However, the above methods cannot perform tracking in the video.

# 2.3 Video Text Spotting

Compared to ITS, existing SOTA VTS methods still rely on RoI for recognition. CoText $[11]$ adopts a lightweight text spotter with Masked-RoI, then uses several encoders to fuse features derived from the spotter, and finally feeds them to a tracking head with cosine similarity matching. TransDETR $[12]$ performs detection and tracking under the MOTR paradigm and then uses Rotated-RoI to extract features for the subsequent recognizer. They pursue training all modules in an end-to-end manner. In comparison, we explore how to efficiently turn a RoI-free ITS model into a VTS one. We reveal the probability of freezing off-the-shelf ITS part and focusing on tracking, thereby saving training budgets while reaching SOTA performance.

# 3 Methodology

# 3.1 Overview

The architecture of GoMatching is presented in Fig. 2. It consists of a frozen image text spotter, a rescoring head, and a Long-Short Term Matching module (LST-Matcher). We adopt an outstanding off-the-shelf image text spotter (i.e., DeepSolo) and freeze its parameters, with the aim of introducing strong text spotting capability into VTS while significantly reducing training cost. In DeepSolo, there are p sequences of queries used for final predictions, with each storing comprehensive semantics for a text instance. To alleviate spotting performance degradation caused by the image-video domain gap, we devise a rescoring mechanism, which determines the confidence scores for text instances by considering both the scores from the image text spotter and a new trainable rescoring head. Finally, we design LST-Matcher to generate instance trajectories by leveraging long-short term information.

![](images/33f88b3a8707458074d9a2e63c8fe649a18e82d8b52c17d2ea42228516a379f2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reception ATELIER"] --> B["Images"]
    C["Input Image"] --> D["Images"]
    B --> E["Image Text Spotter"]
    D --> E
    E --> F["Get queries from decoder"]
    F --> G["threshold Filtering"]
    G --> H["ST-Matcher"]
    G --> I["LT-Matcher"]
    H --> J["Trajectories for instances"]
    I --> J
    J --> K["Object IDs"]
    G --> L["Instance"]
    L --> M["Training only"]
    L --> N["Inference only"]
    L --> O["Both"]
    L --> P["Loss function"]
    L --> Q["OutPut"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#ccf,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#cff,stroke:#333
    style N fill:#cff,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#ffc,stroke:#333
    style Q fill:#ffc,stroke:#333
```
</details>

Figure 2: The overall architecture of GoMatching. The frozen image text spotter provides text spotting results for frames. The rescoring mechanism considers both instance scores from the image text spotter and a trainable rescoring head to reduce performance degradation due to the domain gap. Long-short term matching module (LST-Matcher) assigns IDs to text instances based on the queries in long-short term frames. The yellow star sign ‘★’ indicates the final output of GoMatching.

# 3.2 Rescoring Mechanism

Owing to the domain gap between image and video datasets, employing a frozen image text spotter for direct prediction may result in relative low recall due to low text confidence, further leading to a reduction in end-to-end spotting performance. To ease this issue, we devise a rescoring mechanism via a lightweight rescoring head and a simple score fusion operation. Specifically, the rescoring head is designed to recompute the score for each query from the decoder in the image text spotter. It consists of a simple linear layer and is initialized with the parameters of the image text spotter's classification head. The score fusion operation then decides the final scores by considering both the scores from the image text spotter and the rescoring head. Let $C_{o}^{t} = \{c_{o_{1}}^{t}, ..., c_{o_{p}}^{t}\}$ be a set of original scores produced by image text spotter in frame t. $C_{r}^{t} = \{c_{r_{1}}^{t}, ..., c_{r_{p}}^{t}\}$ is a set of recomputed scores obtained from the rescoring head. We obtain the maximum value for each query as the final score, denoted as $C_{f}^{t} = \{c_{f_{1}}^{t} = \max(c_{o_{1}}^{t}, c_{r_{1}}^{t}), ..., c_{f_{p}}^{t} = \max(c_{o_{p}}^{t}, c_{r_{p}}^{t})\}$ . With final scores, the queries in frames are filtered by a threshold before being sent to LST-Matcher for association.

# 3.3 Long-Short Term Matching Module

Long-short term matching module (LST-Matcher) consists of two sub-modules: the Short Term Matching module (ST-Matcher) and the Long Term Matching module (LT-Matcher), which own the same structure. ST-Matcher is steered to match simple instances between adjacent frames into trajectories, while LT-Matcher is responsible for using long term information to address the unmatched instances due to severe occlusions or strong appearance changes. Each of them contains a one-layer Transformer encoder and a one-layer Transformer decoder [23]. We use a simple multi-layer perceptron (MLP) to map the filtered text instance queries into embeddings as the input, getting rid of using RoI features as in most existing MOT methods. In the encoder, historical embeddings are enhanced by self-attention. The decoder takes embeddings in the current frame as query and enhanced historical embeddings as key for cross-attention, and computes the association score matrix. The current instances are then linked to the existing trajectories composed of historical embeddings or generate new trajectories according to the association score matrix.

![](images/e29a30ba519aeb2a17c22ce6169ecefc8efafb7c35a8a66b812f9f2f986d9a7d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["τ^H_all"] --> B["τ^{t-1}"]
    B --> C["ST-Matcher"]
    D["E^t"] --> E["Get unmatched embeddings"]
    F["E^t_s.u"] --> G["LT-Matcher"]
    H["τ^H_all"] --> I["τ^H_oth"]
    J["τ^s"] --> K["τ^l"]
    L["τ^N"] --> M["Frame"]
    C --> N["τ_s"]
    G --> O["τ_l"]
    K --> P["τ^N"]
```
</details>

Figure 3: The inference pipeline of LST-Matcher, which is a two-stage association process: (1) ST-Matcher associates the instances with trajectories in previous frames as denoted by blue lines. (2) LT-Matcher associates the remaining unmatched instances by utilizing other trajectories in history frames as denoted by red lines.

To be specific, supposing a given clip including $T$ frames and $N_{t}$ text instances in frame $t$ after threshold filtering. $Q^{t} = \{q_{1}^{t},\dots,q_{N_{t}}^{t}\}$ is the set of text instance queries in frame $t$ . Initially, we use a two-layer MLP to map these frozen queries into embeddings $E^{t} = \{e_{1}^{t},\dots,e_{N_{t}}^{t}\}$ . The set of embeddings in all frames is denoted as $E^{L} = E^{1}\cup \ldots \cup E^{T}$ . Let the universal set of embeddings in adjacent frames of the input batch be denoted as $E^{S} = E^{S_{2}}\cup E^{S_{3}}\cup \ldots \cup E^{S_{T}}$ and $E^{S_{t}} = E^{t - 1}\cup E^{t}$ . Based on the predictions of image text spotter, we obtain their corresponding bounding boxes $B^{t} = \{b_{1}^{t},\dots,b_{N_{t}}^{t}\}$ . Let $\tau = \{\tau_1,\dots,\tau_K\}$ be the set of ground-truth (GT) trajectories of all instances in the clip, where $\tau_{k} = \{\tau_{k}^{1},\dots,\tau_{k}^{T}\}$ describes a tube of instance locations $\tau_{k}^{t}\in \mathbb{R}^{4}\cup \{\emptyset \}$ through time. $\tau_{k}^{t} = \emptyset$ means the absence of instance $k$ in frame $t$ . Let $\hat{\alpha}_k^t$ be the matched instance index for $\tau_k^t$ according to the following equation:

$$
\hat {\alpha} _ {k} ^ {t} = \left\{ \begin{array}{l l} \emptyset , & \text { if } \tau_ {k} ^ {t} = \emptyset \text { or } \max _ {i} (I o U (b _ {i} ^ {t}, \tau_ {k} ^ {t})) <   0. 5 \\ & \operatorname{argmax} _ {i} (I o U (b _ {i} ^ {t}, \tau_ {k} ^ {t})), \quad \text { otherwise } \end{array} \right.. \tag {1}
$$

ST-Matcher calculates a short-term association score $v_{i}^{t}(e_{\hat{\alpha}_{k}^{t}}^{t}, E^{S_{t}}) \in \mathbb{R}^{N_{S_{t}}}$ for i-th instances in frame t, where $e_{\hat{\alpha}_{k}^{t}}^{t} \in R^{D}$ is a trajectory query and $N_{S_{t}} = N_{t} + N_{t-1}$ . LT-Matcher calculates a long-term trajectory-specific association score $u_{i}^{t}(e_{k}, E^{L}) \in \mathbb{R}^{N}$ for i-th instances in frame t, where $e_{k} \in \{e_{\hat{\alpha}_{k}^{1}}^{1}, e_{\hat{\alpha}_{k}^{2}}^{2}, ..., e_{\hat{\alpha}_{k}^{T}}^{T}\}$ , $N = \sum_{t=1}^{T} N_{t}$ . Specifically, when $v_{i}^{t}(e_{\hat{\alpha}_{k}^{t}}^{t}, E^{S_{t}}) = 0$ and $u_{i}^{t}(e_{k}, E^{L}) = 0$ , it means no association at time t. Then, ST-Matcher and LT-Matcher can predict distributions of short-term and long-term associations for all instance i in frame t which can be written as:

$$
P _ {s _ {a}} (e _ {\hat {\alpha} _ {k} ^ {t}} ^ {t}, E ^ {S _ {t}}) = \frac {\exp (v _ {i} ^ {t} (e _ {\hat {\alpha} _ {k} ^ {t}} ^ {t} , E ^ {S _ {t}}))}{\sum_ {j \in \{\emptyset , 1 , \ldots , N _ {t} \}} \exp (v _ {j} ^ {t} (e _ {\hat {\alpha} _ {k} ^ {t}} ^ {t} , E ^ {S _ {t}}))}, \tag {2}
$$

$$
P _ {l _ {a}} (e _ {k}, E ^ {L}) = \frac {\exp (u _ {i} ^ {t} (e _ {k} , E ^ {L}))}{\sum_ {j \in \{\emptyset , 1 , \dots , N _ {t} \}} \exp (u _ {j} ^ {t} (e _ {k} , E ^ {L}))}. \tag {3}
$$

To ensure sufficient training of ST-Matcher and LT-Matcher, embeddings set $E^{S}$ and $E^{L}$ are fed into ST-Matcher and LT-Matcher during training, respectively.

During inference, we engage a memory bank to store the instance trajectories from H history frames for long term association. All filtered instances in each frame are further processed by non-maximum-suppression (NMS) before being fed into LST-Matcher for association. Unlike the training phase, where ST-Matcher and LT-Matcheder are independent of each other, LST-Matcher comprises a two-stage associating procedure as described in Fig. 3. Concretely, ST-Matcher first matches the embedding $E^{t}$ in the current frame t with the trajectories $\tau^{t-1}$ in the previous frame t-1. Then, LT-Matcher employs other trajectories $\tau_{oth}^{H}$ in the memory bank to associate the unmatched ones $E_{s\_u}^{t}$ with low association score in ST-Matcher caused by the heavy occlusion or strong appearance changes. If the association score with any trajectory calculated in ST-Matcher or LT-Matcher is higher than a threshold $\theta$ , the instance is linked to the trajectory with the highest score. Otherwise, this instance is used to initiate a new trajectory. Finally, we combine the trajectories $\tau_{s}$ and $\tau_{l}$ predicted by ST-Matcher and LT-Matcher to obtain new trajectories $\tau^{N}$ for tracking in the next frame.

# 3.4 Optimization

Rescoring Loss. To train the rescoring head, we following DETR [14] and use Hungarian algorithm [29] to find a bipartite matching $\hat{\sigma}$ between the prediction set $\hat{Y}$ and the ground truth set $Y$ with minimum matching cost $\mathcal{C}$ :

$$
\hat {\sigma} = \arg \min _ {\sigma} \sum_ {i} ^ {N} \mathcal {C} (Y _ {i}, \hat {Y} _ {\sigma (i)}), \tag {4}
$$

where N is the number of ground truth instances per frame. The cost C can be defined as:

$$
\mathcal {C} (Y _ {i}, \hat {Y} _ {\sigma (i)}) = \lambda_ {c} \mathcal {L} _ {c l s} (\hat {p} _ {\sigma (i)} (c _ {i})) + \lambda_ {b} \sum_ {1} ^ {N} \| b _ {i} - \hat {b} _ {\sigma (i)} \|, \tag {5}
$$

where $\lambda_{c}$ and $\lambda_{b}$ serve as the hyper-parameters to balance different tasks. $\hat{p}_{\sigma(i)}(c_i)$ and $\hat{b}_{\sigma(i)}$ are the probability for ground truth class $c_{i}$ and the prediction of bounding box respectively, and $b_{i}$

represents the ground truth bounding box. $\mathcal{L}_{cls}$ is the focal loss [30]. Specifically, the focal loss for training the rescoring head can be formulated as:

$$
\mathcal {L} _ {r e s} = \sum_ {1} ^ {N} [ - \mathbb {1} _ {\{c _ {i} \neq \varnothing \}} \alpha (1 - \hat {p} _ {\hat {\sigma} (i)} (c _ {i})) ^ {\gamma} \log (\hat {p} _ {\hat {\sigma} (i)} (c _ {i})) - \mathbb {1} _ {\{c _ {i} = \varnothing \}} (1 - \alpha) (\hat {p} _ {\hat {\sigma} (i)} (c _ {i})) ^ {\gamma} \log (1 - \hat {p} _ {\hat {\sigma} (i)} (c _ {i})) ], \tag {6}
$$

where $\alpha$ and $\gamma$ are the hyper-parameters of focal loss.

Long-Short Association Loss. In ST-Matcher, we only consider each trajectory in the universal set of adjacent frames, while in LT-Matcher we consider each trajectory in all long term frames. For each trajectory, we optimize the log-likelihood of its assignments $\hat{\alpha}_{k}$ following GTR [23]:

$$
\mathcal {L} _ {s \_ a s s} (E ^ {S}, \hat {\tau} _ {k}) = - \sum_ {t = 2} ^ {T} \log P _ {s _ {a}} (\hat {\alpha} _ {k} ^ {t} | e _ {\hat {\alpha} _ {k} ^ {t}} ^ {t}, E ^ {S _ {t}}), \tag {7}
$$

$$
\mathcal {L} _ {l _ {-} a s s} (E ^ {L}, \hat {\tau} _ {k}) = - \sum_ {w} \sum_ {t = 1} ^ {T} \log P _ {l _ {a}} (\hat {\alpha} _ {k} ^ {t} | E _ {\hat {\alpha} _ {k} ^ {w}} ^ {w}, E ^ {L}), \tag {8}
$$

where $w\in \{1,\dots,T|\hat{\alpha}_k^w\neq \emptyset \}$

In ST-Matcher and LT-Matcher, empty trajectories would be generated for these unassociated queries, and their optimization goals can be defined as:

$$
\mathcal {L} _ {s \_ b g} (E ^ {S}) = - \sum_ {j: \nexists \hat {\alpha} _ {k} ^ {t} = j} \sum_ {t = 2} ^ {T} \log P _ {s _ {a}} (\alpha^ {t} = \emptyset | e _ {j} ^ {t}, E ^ {S _ {t}}), \tag {9}
$$

$$
\mathcal {L} _ {l \_ b g} (E ^ {L}) = - \sum_ {w = 1} ^ {T} \sum_ {j: \not \exists \hat {\alpha} _ {k} ^ {w} = j} \sum_ {t = 1} ^ {T} \log P _ {l _ {a}} (\alpha^ {t} = \emptyset | E _ {j} ^ {w}, E ^ {L}). \tag {10}
$$

Finally, we can get the long-short association loss as follows:

$$
\mathcal {L} _ {a s s o} = \mathcal {L} _ {s \_ b g} + \mathcal {L} _ {l \_ b g} + \sum_ {\hat {\tau} _ {k}} (\mathcal {L} _ {s \_ a s s} + \mathcal {L} _ {l \_ a s s}). \tag {11}
$$

Overall Loss. Combined with the rescore loss $L_{res}$ in Eq. (6) and the long-short association loss $L_{asso}$ in Eq. (11), the final training loss can be defined as:

$$
\mathcal {L} = \lambda_ {r e s} \mathcal {L} _ {r e s} + \lambda_ {a s s o} \mathcal {L} _ {a s s o}, \tag {12}
$$

where the hyper-parameters $\lambda_{res}$ and $\lambda_{asso}$ are the weights of $\mathcal{L}_{res}$ and $\mathcal{L}_{asso}$ , respectively.

# 4 Experiments

# 4.1 Datasets and Evaluation Metrics

ICDAR15-video [13] is a word-level video text reading benchmark annotated with quadrilateral bounding boxes, comprising a training set of 25 videos and a test set of 24 videos. It focuses on wild scenarios, such as driving on the road, exploring shopping streets, walking in a supermarket, etc.

BOVText $[10]$ is a large-scale, bilingual, and open-world benchmark for video text spotting, encompassing English and Chinese. The dataset is meticulously collected from YouTube and KuaiShou with different scenarios. The text box annotations are represented as quadrilaterals at the textline level.

DSText [19] is a newly proposed dataset, and focuses on dense and small text reading challenges in the video. This dataset provides 50 training videos and 50 test videos. Compared with the previous datasets, DSText mainly includes the following three new challenges: dense video texts, high-proportioned small texts, and various new scenarios, e.g., ‘Game’, ‘Sports’, etc. Similar to ICDAR15-video, DSText adopts word-level annotations, which are labeled with quadrilaterals.

ArTVideo is a novel word-level test set established in this work to evaluate the performance of arbitrary-shaped video text, which is absent in the VTS community. It contains 20 videos with about 30% curved text instances. Straight text is annotated with quadrilaterals, while curved text is annotated with polygons. More details are provided in the Appendix A.

Evaluation Metrics. To evaluate performance, we adopt three evaluation metrics commonly used in ICDAR15-video competition and DSText competition, including MOTA [31], MOTP, and IDF1 [32].

# 4.2 Implementation Details

In all experiments, we only use a single NVIDIA GeForce RTX 3090 (24G) GPU to train and evaluate GoMatching. As for the image text spotter in GoMatching, we apply the officially released DeepSolo [5]. During fine-tuning GoMatching on downstream video datasets, we only update the rescoring head and LST-Matcher, while keeping DeepSolo frozen. More inference settings can be seen in the Appendix B.

Training Setting. The text spotting part of GoMatching is initialized with off-the-shelf DeepSolo weights and kept frozen in all experiments. We optimize other modules on video datasets. We follow EfficientDet $[33]$ to adopt the scale-and-crop augmentation strategy with a resolution of 1280. The batch size T is 6. All frames in a batch are from the same video. Text instances with fusion scores higher than 0.3 are fed into the LST-Matcher during training. AdamW $[34]$ is used as the optimizer. We adopt the warmup cosine annealing learning rate strategy with the initial learning rate being set to 0.00005. The loss weights $\lambda_{res}$ and $\lambda_{asso}$ are set to 1.0 and 0.5, respectively. For focal loss, $\alpha$ is 0.25 and $\gamma$ is 2.0 as in $[14, 5]$ . The model is trained for 30k iterations on all downstream video datasets.

# 4.3 Comparison with State-of-the-art Methods

Results on ICDAR15-video. To evaluate the effectiveness of GoMatching on oriented video text, we conduct a comparison with the state-of-the-art methods on ICDAR15-video presented in Table 1a. As can be seen, GoMatching ranks first in all metrics on the ICDAR15-video leaderboard. By effectively combining a robust image text spotter with a strong tracker, GoMatching improves the best performance by $5.08\%$ MOTA, $0.75\%$ MOTP, and $3.16\%$ IDF1, respectively. Furthermore, owing to the substantial enhancement in recognition and tracking capabilities (details can be found in Sec. 4.4 and Appendix G), GoMatching outperforms the current SOTA single-model method TransDETR by $11.08\%$ MOTA, $3.92\%$ MOTP, and $7.31\%$ IDF1, respectively.

Results on BOVText. Except for the English word recognition scenario, GoMatching can readily adapt to other video text recognition scenarios, such as Chinese text line recognition. For BOVText, which focuses on English and Chinese textline recognition, we employ the DeepSolo trained on bilingual textline datasets and then fine-tune GoMatching on BOVText. The results are presented in Table 1b. It is evident that GoMatching achieves a new record on the BOVText dataset and surpasses previous methods significantly. GoMatching exhibits superior performance over the previous SOTA method CoText [11], with improvements of 41.5% on MOTA, 6.9% on MOTP, and 14.3% on IDF1. Such exceptional performance of GoMatching on BOVText suggests its proficiency in spotting both Chinese and English text in videos. Moreover, it can be easily extended to other languages by adapting the image spotter.

Results on DSText. We further conduct experiments on DSText with dense and small video text scenarios. Results are presented in Table 1c. It is worth noting that most of the previous methods on the DSText leaderboard used an ensemble of multiple models and large public datasets to enhance their performance [19]. For example, TencentOCR integrates the detection results of DBNet [35] and Cascade MaskRCNN [36] built with multiple backbone architectures, combines the Parseq [37] text recognizer, and further improves the end-to-end tracking with ByteTrack [22]. DA adopts Mask R-CNN [38] and DBNet to detect text, then uses BotSORT [21] to replace the tracker in VideoTextSCM [39] and employs the Parseq model for recognition. As a single model with a frozen image text spotter, GoMatching also shows competitive performance compared to other ensembling methods on the leaderboard. GoMatching ranks first (22.83%) on MOTA, second (80.43%) on MOTP, and third (46.09%) on IDF1. Moreover, compared to the SOTA single-model method, GoMatching achieves substantial improvements of 45.46% and 19.66% on MOTA and IDF1, respectively.

Results on ArTVideo. We test TransDETR and GoMatching on ArTVideo to compare the zero-shot text spotting capabilities for arbitrary-shaped text. For a fair comparison, both TransDETR and GoMatching are trained on ICDAR15-video. Unlike ICDAR15-video and DSText which only have straight text, ArTVideo has a substantial number of curved text, so we report results under four settings: tracking results on both straight and curved text, spotting results on both straight and curved text, tracking results on curved text only, and spotting results on curved text only. As shown in Table 1d, GoMatching outperforms TransDETR under all settings. Especially when involving an additional recognition task (end-to-end spotting) or only considering curved text, the performance advantages of GoMatching are more significant. This further confirms that the previous SOTA

Table 1: Comparison results with SOTA methods on four distinct datasets. ‘†’ denotes that the results are collected from the official competition website. ‘\*’: we use the officially released model for evaluation. ‘M-ME’ indicates whether multi-model ensembling is used. ‘Y’ and ‘N’ stand for yes and no. The best and second-best results are marked in bold and underlined, respectively.   
(a) Results on ICDAR15-video. 

<table><tr><td>Method</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>HIK_OCR [9]</td><td>52.98</td><td>74.88</td><td>61.85</td></tr><tr><td>CoText [11]</td><td>58.94</td><td>74.53</td><td>71.66</td></tr><tr><td>TransDETR [12]</td><td>60.96</td><td>74.61</td><td>72.80</td></tr><tr><td>h&amp;h_lab†</td><td>63.76</td><td>77.78</td><td>71.08</td></tr><tr><td>GOCR Offline†</td><td>63.05</td><td>74.31</td><td>76.95</td></tr><tr><td>CoText(Kuaishou_MMU)†</td><td>66.96</td><td>76.55</td><td>74.24</td></tr><tr><td>GoMatching (size:800) (ours)</td><td>68.51</td><td>77.52</td><td>76.59</td></tr><tr><td>GoMatching (size:1000) (ours)</td><td>72.04</td><td>78.53</td><td>80.11</td></tr><tr><td>GoMatching (size:1440) (ours)</td><td>70.52</td><td>78.25</td><td>78.70</td></tr></table>

(c) Results on DSText. 

<table><tr><td>Method</td><td>M-ME</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>TransDETR+HRNet $\dagger$ </td><td>Y</td><td>-28.58</td><td>80.36</td><td>26.20</td></tr><tr><td>SCUT-MMOCR-KS $\dagger$ </td><td>Y</td><td>-27.47</td><td>76.59</td><td>43.61</td></tr><tr><td>TextTrack $\dagger$ </td><td>Y</td><td>-25.09</td><td>74.95</td><td>26.38</td></tr><tr><td>abcmot $\dagger$ </td><td>Y</td><td>5.54</td><td>74.61</td><td>24.25</td></tr><tr><td>DA $\dagger$ </td><td>Y</td><td>10.51</td><td>78.97</td><td>53.45</td></tr><tr><td>TencentOCR $\dagger$ </td><td>Y</td><td>22.44</td><td>80.82</td><td>56.45</td></tr><tr><td>TransDETR [12]*</td><td>N</td><td>-22.63</td><td>79.73</td><td>26.43</td></tr><tr><td>GoMatching (ours)</td><td>N</td><td>22.83</td><td>80.43</td><td>46.09</td></tr></table>

(b) Results on BOVText. 

<table><tr><td>Method</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>EAST + CRNN [10]</td><td>-79.3</td><td>76.3</td><td>6.8</td></tr><tr><td>PSENet + CRNN [10]</td><td>-17.0</td><td>79.2</td><td>31.3</td></tr><tr><td>DB + CRNN [10]</td><td>-13.2</td><td>81.3</td><td>38.8</td></tr><tr><td>TransVTSpotter [10]</td><td>-1.4</td><td>82.0</td><td>43.6</td></tr><tr><td>CoText [11]</td><td>11.4</td><td>80.3</td><td>48.3</td></tr><tr><td>GoMatching (ours)</td><td>52.9</td><td>87.2</td><td>62.6</td></tr></table>

(d) Results on ArTVideo. 

<table><tr><td>Method</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td colspan="4">ArTVideo Tracking</td></tr><tr><td>TransDETR [12]</td><td>54.2</td><td>67.9</td><td>70.4</td></tr><tr><td>GoMatching (ours)</td><td>67.2</td><td>81.3</td><td>75.8</td></tr><tr><td colspan="4">ArTVideo End-to-End Spotting</td></tr><tr><td>TransDETR [12]</td><td>2.8</td><td>69.7</td><td>49.3</td></tr><tr><td>GoMatching (ours)</td><td>68.8</td><td>82.9</td><td>78.5</td></tr><tr><td colspan="4">ArTVideo-Curved Tracking</td></tr><tr><td>TransDETR [12]</td><td>4.4</td><td>60.5</td><td>50.2</td></tr><tr><td>GoMatching (ours)</td><td>59.5</td><td>76.3</td><td>73.5</td></tr><tr><td colspan="4">ArTVideo-Curved End-to-End Spotting</td></tr><tr><td>TransDETR [12]</td><td>-66.7</td><td>-61.9</td><td>26.9</td></tr><tr><td>GoMatching (ours)</td><td>56.8</td><td>78.0</td><td>73.9</td></tr></table>

methods have unsatisfactory recognition capabilities and limited adaptability to complex scenarios. Furthermore, as shown in Fig. 1(b), GoMatching achieves excellent performance while significantly reducing the training budget.

Some visual results are provided in Fig. 4. It shows that GoMatching performs well on straight and curved text, and even more complex scene text. More visual results (including some failure cases) and analysis are provided in the Appendix G.

# 4.4 Ablation Studies

We first conduct comprehensive ablation studies on ICDAR15-video to verify the effectiveness of each component. The experimental results are shown in Table 2. The impact of frame length on long-term association during inference is then studied, and the results are shown in Appendix C.

Effectiveness of Utilizing Queries. Comparing the first two rows in Table 2, we can find that using queries from the decoder of image text spotter is more beneficial for tracking than RoI features. By leveraging the unified queries from frozen DeepSolo, 0.98% and 1.05% improvements on MOTA and IDF1 are achieved. This is because queries integrate more text instance information, i.e., unifying multi-scale features, text semantics, and position information, which has been proven effective in DeepSolo. Although position information is essential for tracking, it is ignored in RoI features.

Effectiveness of Rescoring Mechanism. To verify the effectiveness of the rescoring mechanism, we test three different scoring mechanisms: the original score from DeepSolo, the score recomputed by the rescoring head, and the fusion score from the rescoring mechanism. As shown in row 2 and row 3 of Table 2, the rescoring head can alleviate the performance degradation caused by the domain gap between ICDAR15-image and ICDAR15-video, and achieve gains of 1.25% and 0.97% on MOTA and IDF1, respectively. Moreover, as shown in row 4, we can observe that combining the knowledge of rescoring head learned from the new dataset with the prior knowledge of DeepSolo can further improve MOTA and IDF1 by 0.33% and 0.32%, respectively. Appendix F contains more results.

Effectiveness of LST-Matcher. In this part, we conduct three experiments to prove the effectiveness of the LST-Matcher. As shown in row 4 of Table 2, we only use LT-Matcher to associate high-score text instances in the current frame with trajectories in the tracking memory bank. In row 5, we only use ST-Matcher to associate high-score text instances in the current frame with trajectories of the previous frame. In addition, as shown in row 6, we employ both LT-Matcher and ST-Matcher to test

![](images/1debcde4623e69f60da9e278d7b9bd04dc36315daa86e9726a9240f7e497c9e4.jpg)

Figure 4: Visual results of video text spotting. Images from top to bottom are the results on ICDAR15-video, BOVText, DSText, and ArTVideo, respectively. Text instances belonging to the same trajectory are assigned the same color.   
Table 2: Impact of difference components in the proposed GoMatching. ‘Query’ indicates that LST-Matcher employs the queries of high-score text instances for association, otherwise RoI features. Column ‘Scoring’ indicates the employed scoring mechanism, in which ‘O’ means using the original scores from DeepSolo, ‘R’ means using the scores recomputed by the rescoring head, and ‘F’ means using the fusion scores obtained from the rescoring mechanism. 

<table><tr><td>Index</td><td>Query</td><td>Scoring</td><td>LT-Matcher</td><td>ST-Matcher</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>1</td><td></td><td>O</td><td>√</td><td></td><td>66.20</td><td>78.52</td><td>75.07</td></tr><tr><td>2</td><td>√</td><td>O</td><td>√</td><td></td><td>67.22</td><td>78.54</td><td>76.12</td></tr><tr><td>3</td><td>√</td><td>R</td><td>√</td><td></td><td>68.47</td><td>78.29</td><td>77.09</td></tr><tr><td>4</td><td>√</td><td>F</td><td>√</td><td></td><td>68.80</td><td>78.24</td><td>77.41</td></tr><tr><td>5</td><td>√</td><td>F</td><td></td><td>√</td><td>69.40</td><td>78.34</td><td>73.60</td></tr><tr><td>6</td><td>√</td><td>F</td><td>√</td><td>√</td><td>70.52</td><td>78.25</td><td>78.70</td></tr></table>

LST-Matcher. We can easily observe that compared to LT-Matcher, LST-Matcher improves MOTA and IDF1 by 1.72% and 1.29% respectively, while compared to ST-Matcher, the improvement on MOTA and IDF1 are 1.12% and 5.1%, respectively. In Fig. 5, we also demonstrate that using LST-Matcher can effectively mitigate the issue of ID switches caused by the strong appearance changes due to motion blur. These results validate that combining short-term and long-term information leads to more robust tracking outcomes, thereby enhancing the performance of video text spotting.

Different training strategies. To investigate the impacts of different training strategies on GoMatching, we establish three distinct settings on the ICDAR15-video dataset, as shown in Table 3. 1) We only fine-tune the tracker while keeping the image text spotter frozen (the first row of Table 3). 2) We first fine-tune the image text spotter on images extracted from ICDAR15-video and then further fine-tune the tracker with the image text spotter fixed (the second row of Table 3). 3) We jointly train the spotter's decoder and tracker of GoMatching while trying different learning rates for the decoder (the last three rows of Table 3). As shown in the first two rows of Table 3, fine-tuning the image spotter on the downstream video dataset results in a performance decline compared to the default setting. This is due to two data-related factors: 1) minor variations in text content between frames in the same video lead to insufficient data diversity, causing the image text spotter to overfit more easily; and 2) image blurring from camera motion reduces the quality of data available for training the image text spotter. When the image text spotter and tracker are trained simultaneously (the last three rows), the model's performance significantly decreases. Even with the decoder's learning rate close to zero (i.e., 0.001), there is still a $1.13\%$ drop in IDF1. As the decoder's learning rate increases, the performance decline becomes more pronounced. This indicates that naive joint optimization of text spotting and tracking is challenging, likely due to conflicts between the two tasks. In future work,

![](images/2b51f12d01d0cd170b471ce87bb34565060ec031f9b06a51869b23cc2f986ccb.jpg)

<details>
<summary>text_image</summary>

ID Switches
1.00
2.00
3.00
4.00
5.00
6.00
7.00
8.00
9.00
10.00
11.00
12.00
13.00
14.00
15.00
16.00
17.00
18.00
19.00
20.00
21.00
22.00
23.00
24.00
25.00
26.00
27.00
28.00
29.00
30.00
31.00
32.00
33.00
34.00
35.00
36.00
37.00
38.00
39.00
40.00
41.00
42.00
43.00
44.00
45.00
</details>

Figure 5: Visualization results of ST-Matcher and LST-Matcher. The first row shows the failure case that suffers from the ID switches issue when using only ST-Matcher, caused by missed detection and erroneous matching. The second row shows that LST-Matcher effectively mitigates this issue via both long and short term matching. Text instances in the same color represent the same IDs.

Table 3: Results of GoMatching under various training settings on the ICDAR15-video dataset. ‘Only Image Spotter’ and ‘Only Tracker’ refer to fine-tuning either the image spotter or tracker with another module fixed. ‘End-to-End’ denotes that training the image spotter and tracker in an end-to-end manner. ‘0.001’, ‘0.01’ and ‘0.1’ correspond to the ratios of the learning rate employed by the decoder of the image text spotter relative to the base learning rate. Due to constraints in training resources, we only optimize the parameters of the decoder component for the image text spotter. 

<table><tr><td>Index</td><td>Training Setting</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>1</td><td>‘Only Tracker’</td><td>72.04</td><td>78.53</td><td>80.11</td></tr><tr><td>2</td><td>First ‘Only Image Spotter’, Then ‘Only Tracker’</td><td>70.82</td><td>78.09</td><td>79.64</td></tr><tr><td>3</td><td>‘End-to-End’, Image Spotter’s Decoder (‘0.001’)</td><td>71.48</td><td>79.14</td><td>78.98</td></tr><tr><td>4</td><td>‘End-to-End’, Image Spotter’s Decoder (‘0.01’)</td><td>70.15</td><td>78.17</td><td>77.67</td></tr><tr><td>5</td><td>‘End-to-End’, Image Spotter’s Decoder (‘0.1’)</td><td>68.03</td><td>75.46</td><td>77.16</td></tr></table>

it is worth trying to establish larger, more diverse video text spotting datasets and to explore more effective multi-task optimization strategies.

# 5 Conclusion

In this paper, we propose a simple yet strong baseline, termed GoMatching, for video text spotting. GoMatching harnesses the talent of an off-the-shelf query-based image text spotter and only needs to tune a lightweight tracker, effectively addressing the limitations of previous SOTA methods in recognition. Specifically, we design the rescoring mechanism and LST-Matcher to adapt the image text spotter to unseen video datasets while empowering GoMatching with excellent tracking capability. Moreover, we establish a novel test set ArTVideo for the evaluation of video text spotting models on arbitrary-shaped text, filling the gap in this area. Experiments on public benchmarks and ArTVideo demonstrate the superiority of our GoMatching in terms of both spotting accuracy and training cost.

# Acknowledgments and Disclosure of Funding

This work was supported in part by the National Key Research and Development Program of China under Grant 2023YFC2705700, in part by the National Natural Science Foundation of China under Grant U23B2048, 62076186 and 62225113, and in part by the Innovative Research Group Project of Hubei Province under Grant 2024AFA017. Dr Tao's research is partially supported by NTU RSR and Start Up Grants. The numerical calculations in this paper have been done on the supercomputing system in the Supercomputing Center of Wuhan University.

# References

[1] Jianfeng Dong, Xirong Li, Chaoxi Xu, Xun Yang, Gang Yang, Xun Wang, and Meng Wang. Dual encoding for video retrieval by text. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4065–4080, 2021.   
[2] Chongsheng Zhang, Yuefeng Tao, Kai Du, Weiping Ding, Bin Wang, Ji Liu, and Wei Wang. Character-level street view text spotting based on deep multisegmentation network for smarter autonomous driving. IEEE Transactions on Artificial Intelligence, 3(2):297–308, 2021.   
[3] Yuliang Liu, Jiaxin Zhang, Dezhi Peng, Mingxin Huang, Xinyu Wang, Jingqun Tang, Can Huang, Dahua Lin, Chunhua Shen, Xiang Bai, et al. Spts v2: single-point scene text spotting. arXiv preprint arXiv:2301.01635, 2023.   
[4] Xiang Zhang, Yongwen Su, Subarna Tripathi, and Zhuowen Tu. Text spotting transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9519–9528, 2022.   
[5] Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao. Deepsolo: Let transformer decoder with explicit points solo for text spotting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19348–19357, 2023.   
[6] Mingxin Huang, Jiaxin Zhang, Dezhi Peng, Hao Lu, Can Huang, Yuliang Liu, Xiang Bai, and Lianwen Jin. Estextspotter: Towards better scene text spotting with explicit synergy in transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19495–19505, 2023.   
[7] Xiaobing Wang, Yingying Jiang, Shuli Yang, Xiangyu Zhu, Wei Li, Pei Fu, Hua Wang, and Zhenbo Luo. End-to-end scene text recognition in videos based on multi frame tracking. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 1255–1260. IEEE, 2017.   
[8] Zhanzhan Cheng, Jing Lu, Yi Niu, Shiliang Pu, Fei Wu, and Shuigeng Zhou. You only recognize once: Towards fast video text spotting. In Proceedings of the 27th ACM International Conference on Multimedia, pages 855–863, 2019.   
[9] Zhanzhan Cheng, Jing Lu, Baorui Zou, Liang Qiao, Yunlu Xu, Shiliang Pu, Yi Niu, Fei Wu, and Shuigeng Zhou. Free: A fast and robust end-to-end video text spotter. IEEE Transactions on Image Processing, 30:822–837, 2020.   
[10] Weijia Wu, Yuanqiang Cai, Debing Zhang, Sibo Wang, Zhuang Li, Jiahong Li, Yejun Tang, and Hong Zhou. A bilingual, openworld video text dataset and end-to-end video text spotter with transformer. arXiv preprint arXiv:2112.04888, 2021.   
[11] Wejia Wu, Zhuang Li, Jiahong Li, Chunhua Shen, Hong Zhou, Size Li, Zhongyuan Wang, and Ping Luo. Real-time end-to-end video text spotter with contrastive representation learning. arXiv preprint arXiv:2207.08417, 2022.   
[12] Weijia Wu, Yuanqiang Cai, Chunhua Shen, Debing Zhang, Ying Fu, Hong Zhou, and Ping Luo. End-to-end video text spotting with transformer. International Journal of Computer Vision, 132(9):4019–4035, 2024.   
[13] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, et al. Icdar 2015 competition on robust reading. In 2015 13th international conference on document analysis and recognition (ICDAR), pages 1156–1160. IEEE, 2015.   
[14] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.   
[15] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. In ICLR, 2021.

[16] Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. Motrv2: Bootstrapping end-to-end multi-object tracking by pretrained object detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22056–22065, 2023.   
[17] En Yu, Tiancai Wang, Zhuoling Li, Yuang Zhang, Xiangyu Zhang, and Wenbing Tao. Motrv3: Release-fetch supervision for end-to-end multi-object tracking. arXiv preprint arXiv:2305.14298, 2023.   
[18] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end multiple-object tracking with transformer. In European Conference on Computer Vision, pages 659–675. Springer, 2022.   
[19] Weijia Wu, Yuzhong Zhao, Zhuang Li, Jiahong Li, Mike Zheng Shou, Umapada Pal, Dimosthenis Karatzas, and Xiang Bai. Icdar 2023 video text reading competition for dense and small text. arXiv preprint arXiv:2304.04376, 2023.   
[20] Zhongdao Wang, Liang Zheng, Yixuan Liu, Yali Li, and Shengjin Wang. Towards real-time multi-object tracking. In European Conference on Computer Vision, pages 107–122. Springer, 2020.   
[21] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.   
[22] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European Conference on Computer Vision, pages 1–21. Springer, 2022.   
[23] Xingyi Zhou, Tianwei Yin, Vladlen Koltun, and Philipp Krähenbühl. Global tracking transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8771–8780, 2022.   
[24] Minghui Liao, Guan Pang, Jing Huang, Tal Hassner, and Xiang Bai. Mask textspotter v3: Segmentation proposal network for robust scene text spotting. In Computer Vision–ECCV 2020:16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 706–722. Springer, 2020.   
[25] Wenhai Wang, Enze Xie, Xiang Li, Xuebo Liu, Ding Liang, Zhibo Yang, Tong Lu, and Chunhua Shen. Pan++: Towards efficient and accurate end-to-end spotting of arbitrarily-shaped text. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5349–5367, 2021.   
[26] Yuliang Liu, Chunhua Shen, Lianwen Jin, Tong He, Peng Chen, Chongyu Liu, and Hao Chen. Abcnet v2: Adaptive bezier-curve network for real-time end-to-end text spotting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8048–8064, 2021.   
[27] Yair Kittenplon, Inbal Lavi, Sharon Fogel, Yarin Bar, R Manmatha, and Pietro Perona. Towards weakly-supervised text spotting using a multi-task transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4604–4613, 2022.   
[28] Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao. Deepsolo++: Let transformer decoder with explicit points solo for multilingual text spotting. arXiv preprint arXiv:2305.19957, 2023.   
[29] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.   
[30] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.   
[31] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: The clear mot metrics. EURASIP Journal on Image and Video Processing, 2008:1–10, 2008.   
[32] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pages 17–35. Springer, 2016.

[33] Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781–10790, 2020.   
[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[35] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, and Xiang Bai. Real-time scene text detection with differentiable binarization. In Proceedings of the AAAI conference on artificial intelligence, pages 11474–11481, 2020.   
[36] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: High quality object detection and instance segmentation. IEEE transactions on pattern analysis and machine intelligence, pages 1483-1498, 2019.   
[37] Darwin Bautista and Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. In European Conference on Computer Vision, pages 178–196. Springer, 2022.   
[38] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.   
[39] Yuzhe Gao, Xing Li, Jiajian Zhang, Yu Zhou, Dian Jin, Jing Wang, Shenggao Zhu, and Xiang Bai. Video text tracking with a spatio-temporal complementary model. IEEE Transactions on Image Processing, pages 9321–9331, 2021.   
[40] Yuliang Liu, Lianwen Jin, Shuaitao Zhang, Canjie Luo, and Sheng Zhang. Curved scene text detection via transverse and longitudinal sequence connection. Pattern Recognition, 90:337–345, 2019.

# A More Details of ArTVideo

Due to the scarcity of curved text instances within existing video text spotting datasets, it is infeasible to evaluate the performance of video text spotting models on curved text. To fill this gap, we collected a test set named ArTVideo containing 20 video clips with a total of 884 frames, in which 18 videos were collected from YouTube and 2 videos from the BOVText test set. ArTVideo contains 6,526 text instances, including 4,632 straight text instances and 1,894 curved text instances, i.e., curved text accounts for about $30\%$ . As shown in Fig. 6, we provide high-quality word-level annotations for both straight and curved text in two different annotated ways. The straight text is labeled with quadrilaterals, while for curved text, we follow the CTW1500 [40] and adopt a polygon with 14 points to annotate the text contour. More statistics about ArTVideo are shown in Fig. 7.

![](images/4791b62ba3e5e3df87559ebb4cf28348e2ad40e18a7ffa1f4499c1a30e2e9719.jpg)

<details>
<summary>text_image</summary>

UNITED
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
BRAMALL
1855
</details>

Figure 6: Visual examples from our ArTVideo. The straight and curved text are labeled with quadrilaterals and polygons, respectively. The same background color in different frames (columns) denotes the same instance.

![](images/793a9e1500066f8cbcff543eeaf8ae76376f0b80efb8af0f9bd050501817b351.jpg)

<details>
<summary>bar</summary>

| Instance Counts per Frame | Percentage (%) |
| :--- | :--- |
| 1 | 7.2 |
| 2 | 10.1 |
| 3 | 12.1 |
| 4 | 7.9 |
| 5 | 7.4 |
| 6 | 10.0 |
| 7 | 10.3 |
| 8 | 10.4 |
| 9 | 10.4 |
| 10 | 4.3 |
| 11 | 1.6 |
| 12 | 5.6 |
| 13 | 7.7 |
| 14 | 1.2 |
| 15 | 0.8 |
| 16 | 2.1 |
| 17 | 1.7 |
The chart displays the percentage of instances per frame for ArTVideo. The x-axis represents the instance counts per frame (ranging from 1 to 20), and the y-axis represents the percentage (%) of occurrences. There is no label for the data series.
</details>

(a)

![](images/7c12cea2e0426ae44f322798320e7a9804bd2894e0aaf3d3acebcd78cf98dab6.jpg)

<details>
<summary>bar</summary>

| Text Length per Instance | Percentage (%) |
| :--- | :--- |
| 1 | 3 |
| 2 | 10 |
| 3 | 9 |
| 4 | 20 |
| 5 | 17 |
| 6 | 13 |
| 7 | 12 |
| 8 | 5 |
| 9 | 6 |
| 10 | 2 |
| 11 | 2.5 |
| 12 | 1 |
| 13 | 1.5 |
| 14 | 0 |
| 15 | 0 |
</details>

(b)

![](images/0db51d5a864fc77a658222b494eff1f7b0b74fe51e9b35b7128638304461706b.jpg)

<details>
<summary>text_image</summary>

CONTRACTORS
MILK
THE TRAVELER0104
2 Kopi
530kcal
NEXT 2.5PLATED
Floor
1ST
FIRST
Store
LIMITED
Electrolit
NYS C
ALL
SHEFFIELD
BEACH in
TACTO
HARD HIPVan
TO
KINEMASTER
ECOFFEE TO
FROCCOL
LANCE
MASTERTON
CREAM
553kj
STAGES
FROM HALLAM
INSTANT ON AWARDS AT
BEN KD
Made ON NATURE
BURLING
CONSOLID
CRICKET COMINGUO
RETAIL
132kcal MULTI-BRAND
HYDRATION
pcchiiBckNN MELCOME
PEDESTRIAN
CONCREATOR
ANIMIS
NLC BICYCLE 1862 STREET OPENS JERRYS HOME AS BELL ON WHITE REDHART redhart TIME Hardrock
BRANDY
ITSSINGLE GAME A GAME BETWEEN A
</details>

(c)   
Figure 7: Statistics of ArTVideo. (a) and (b) show the distribution of text instance numbers in each frame and the distribution of the text length of each instance, respectively. (c) shows the word cloud of text annotations in ArTVideo.

Table 4: Ablation studies on the number of frames $(T)$ for long-term association in LT-Matcher, and the max number of history frames in tracking memory bank is $H = T - 1$ . Experiments are conducted on ICDAR15-video and the best results are marked in bold. 

<table><tr><td>Number T</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>T = 32</td><td>70.13</td><td>78.07</td><td>78.24</td></tr><tr><td>T = 16</td><td>70.33</td><td>78.25</td><td>78.60</td></tr><tr><td>T = 8</td><td>70.44</td><td>78.25</td><td>78.70</td></tr><tr><td>T = 6</td><td>70.52</td><td>78.25</td><td>78.70</td></tr><tr><td>T = 4</td><td>70.51</td><td>78.27</td><td>78.16</td></tr></table>

Table 6: Results of using different image sizes on ICDAR15-video. 'Size' means the size of the shorter side of the input image during inference. The best results are highlighted in bold. 

<table><tr><td>Method</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td><td>FPS (↑)</td></tr><tr><td>TransDETR(Size: 800)</td><td>60.96</td><td>74.61</td><td>72.80</td><td>12.69</td></tr><tr><td>GoMatching(Size: 800)</td><td>68.51</td><td>77.52</td><td>76.59</td><td>14.41</td></tr><tr><td>GoMatching(Size: 1000)</td><td>72.04</td><td>78.53</td><td>80.11</td><td>10.60</td></tr></table>

Table 5: Results of different score fusion strategies on ICDAR5-video. ‘Mean’, ‘Geomean’, and ‘Maximum’ denote the arithmetic mean, geometric mean, and the maximum score fusion strategies, respectively. The best results are highlighted in bold. 

<table><tr><td>Strategy</td><td>MOTA (↑)</td><td>MOTP (↑)</td><td>IDF1 (↑)</td></tr><tr><td>Mean</td><td>70.46</td><td>78.38</td><td>78.29</td></tr><tr><td>Geo-mean</td><td>70.29</td><td>78.39</td><td>78.26</td></tr><tr><td>Maximum</td><td>70.52</td><td>78.25</td><td>78.70</td></tr></table>

Table 7: Comparison between TransDETR and GoMatching. ‘T-Para.’ and ‘A-Para.’ denote the number of all parameters and the trainable parameters in each model, respectively. 

<table><tr><td>Method</td><td>#T-Para. (M)</td><td>#A-Para. (M)</td></tr><tr><td>TransDETR</td><td>39.35</td><td>39.58</td></tr><tr><td>GoMatching</td><td>32.79</td><td>75.38</td></tr></table>

# B More Details of Inference Settings

Since there is no training set in ArTVideo, we directly use the model trained on ICDAR15-video to evaluate the generalization ability of GoMatching to arbitrary-shaped text. The association score threshold is set to 0.2. For ICDAR15-video, we set the shorter size of the input image to 800, 1000, and 1440, with 800 aligned with the setting in TransDETR and 1440 aligned with the setting in DeepSolo. As for BOVText, DSText, and ArTVideo, the shorter sizes are set to 1000, 1280, and 1440, respectively. All the ablation studies are conducted on the setting of 1440.

# C Impact of the Frame Number in LT-Matcher

In Table 4, we further study and analyze the impact of the number of frames for long-term association in LT-Matcher during inference. For the text spotting task, since a single frame may have a large number of text instances, sometimes reaching hundreds, excessive historical frame information would weaken the discrimination of text instance features, resulting in erroneous matching results. Therefore, we conduct a hyper-parameter search and find that the optimal frame number is 6.

# D Comparison of Different Score Fusion Strategies

To investigate the impact of the score fusion strategy in the rescoring mechanism, we further evaluate two other strategies: (1) the arithmetic mean score fusion strategy and (2) the geometric mean score fusion strategy, denoted as mean and geo-mean, respectively. The arithmetic mean score fusion strategy takes the average of the scores from the image text spotter and the rescoring head as the final score for each query, while the geo-mean score fusion strategy uses the geometric mean. These two strategies can be formulated as:

$$
c _ {m e a n} = (c _ {o} + c _ {r}) / 2, \tag {13}
$$

$$
c _ {g e o - m e a n} = \sqrt {c _ {o} * c _ {r}}, \tag {14}
$$

where $c_{mean}$ and $c_{geo-mean}$ denote the final score of the two strategies, respectively. $c_{o}$ and $c_{r}$ are the scores from the image text spotter and the rescoring head, respectively.

From Table 5, we can see that employing the maximum score fusion strategy achieves the best performance on MOTA and IDF1 among all the strategies. As for the other two strategies, extremely low confidence scores from the image text spotter may result in low final scores, probably leading to missed detections. Therefore, we adopt the maximum score fusion strategy in the rescoring mechanism of GoMatching by default.

With Rescoring Mechanism   
![](images/24e66c6fe73a82c677129696165f1165e59de3b6404d58dda37e18f8e029b764.jpg)

<details>
<summary>text_image</summary>

(0.49)ABANES
(0.72)ESTROPAIO
(0.69)ABONICS
(0.52)ESTROPAIO
10:33X (0.34)ITG
10:52 ESTROPAIO
10:58 SIMULY
(0.60)035
(0.30)24
(0.45)31
(0.61)WAJAS
(0.67)ERFICIES
(0.70)VERO
(0.66)100
(0.68)100
(0.43)AUN
(0.50)EDAD
(0.72)SAVA UNA
(0.62)VEHT
(0.51)SUDEBAD IENIT
(0.35)NO
(0.76)DELICAD
(0.30)O
(0.56)100
(0.52)100
(0.55)070
(0.50)065
</details>

Without Rescoring Mechanism   
![](images/79924bea09ed0ba61edf71a2f1231acbc72142c2806f64c7c0c5b60f68dab350.jpg)

<details>
<summary>text_image</summary>

ESTROPAO
(0.73) ESTROPAO
(0.40) BAN
(0.69) ESTROPAO
(0.64) MULTI
(0.51) 205
(0.59) WAUAS
(0.41) EDAD
(0.68) SALVAUNFA
(0.60) LICADAS TRUCUPART
(0.74) DELICAD
(0.37) 100
(0.39) 100
(0.45) 1070
(0.41) 100
</details>

![](images/2c57e126a8aa5158fc11baa48d0128ce5bf23ead49f054844378fe139f08dc63.jpg)

<details>
<summary>text_image</summary>

FRANKEURT
REFRESCOS
(0.80)REFRESCOS
CER
(0.79)CERVEZAS
(0.52)TUCACLIA
(0.52)170
(0.64)CASA
(0.50)FANTA
(0.46)150
(0.38)FANTA
(0.57)170
(0.64)ARDA
(0.46)ESTERIA
</details>

![](images/1ec44d1aa72e49ac578eed5e418a5d022d244cb2cdd50c754da55f97b4b45e78.jpg)

<details>
<summary>text_image</summary>

FRANKFURT
0.78 FRANKFURT
REFRESCOS
0.80 REFRESCOS
CERYAZAS
(0.47) TUCACLIA
(0.41) 170
(0.45) FANTA (0.43) NALANIA (0.39) 150
(0.35) FANTA
(0.48) 170
(0.63) ARDA
(0.46) ESTERIA
</details>

![](images/e65c7aaf8c2d4cccc943cb4cb189d68a62af5e278c2dbd8f3f9350ac9485d82a.jpg)

<details>
<summary>text_image</summary>

servi
(0.75)SERVILLETAS
servi
(0.59160)
servi
(0.30)COSQUE
(0.32)VELE
servi
(0.73)SERVILLETAS
servi
(0.65)CAPAS
(0.50)THX72
10.401 €
</details>

![](images/06fccf6e34916214751ad56b37bb658afd40419310832c55e8cd9206d586fa65.jpg)

<details>
<summary>text_image</summary>

(0.40) STAIL
servi-letas
(0.73) SERVILLETAS
servi-1 SERVILLETAS
(0.43) SERVILLETAS
servi-2 SERVILLETAS
(0.73) CAPAS
(0.51) THX72
1.90
</details>

Figure 8: Visual comparison of GoMatching with and without the rescoring mechanism. The values in parentheses indicate the confidence scores of the detected text. ‘→’ points to the filtered texts due to low confidence without using rescoring. ‘→’ points to the text whose confidence score has been improved by the rescoring mechanism. The rescoring mechanism increases the confidence scores of small texts and blurry texts, preventing them from being filtered out by the threshold and thereby cultivating a better tracking candidate pool.

Table 8: Video text detection performance on ICDAR2013-video [? ]. The best results are highlighted in bold. 

<table><tr><td>Method</td><td>Precision (↑)</td><td>Recall (↑)</td><td>F-measure (↑)</td></tr><tr><td>Free</td><td>79.7</td><td>68.4</td><td>73.6</td></tr><tr><td>TransDETR</td><td>80.6</td><td>70.2</td><td>75.0</td></tr><tr><td>GoMatching w/o rescoring</td><td>92.4</td><td>65.7</td><td>76.8</td></tr><tr><td>GoMatching</td><td>89.5</td><td>74.8</td><td>81.5</td></tr></table>

Table 9: Comparison results of detection AP on the ICDAR13-video between with and without the rescoring mechanism. 

<table><tr><td>Method</td><td>AP</td><td> $AP_S$ </td><td> $AP_M$ </td><td> $AP_L$ </td></tr><tr><td>w/o rescoring</td><td>26.2</td><td>11.6</td><td>40.1</td><td>49.8</td></tr><tr><td>w/ rescoring</td><td>29.3 (+3.1)</td><td>15.5 (+3.9)</td><td>42.7 (+2.6)</td><td>51.9 (+2.1)</td></tr></table>

# E More Comparisons between TransDETR and GoMatching

In Table 6, we provide the results of using three different image sizes in GoMatching during inference on ICDAR15-video. In the first row of Table 6, the shorter side of the input image is set to 800, which is the same as the default setting of TransDETR. It is evident from Table 6 that GoMatching significantly outperforms TransDETR in all settings. Meanwhile, it also shows that GoMatching outperforms TransDETR under its default setting in terms of both inference speed and spotting

![](images/365832266a843710c96eb8e611da809e671949693c5e516a063f736f093b4e6a.jpg)

<details>
<summary>text_image</summary>

TransDETR
GoMatching
BISCONS FRAUCES
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BISCONS
BIS CONS
</details>

Figure 9: More visualization results of TransDETR and GoMatching. Failure cases of text detection and recognition are highlighted with ellipses and rectangles, respectively.

accuracy. With the increase in image size (e.g., 1000), GoMatching offers better spotting performance at the cost of decreased inference speed.

Moreover, we compare the number of all parameters and the trainable parameters of GoMatching and TransDETR, as shown in Table 7. It is noteworthy that GoMatching requires a much smaller training budget than TransDETR owing to its simpler architecture design, making it a simple but strong baseline for video text spotting.

# F More Results for the Rescoring Mechanism

In Table 8, we provide the video text detection results of GoMatching on ICDAR13-video [?] and compare them with Free and TransDETR. As shown in the table, without the rescoring mechanism, GoMatching relies on the original results from DeepSolo, resulting in a $4.5\%$ decrease on Recall and only a $1.8\%$ improvement on F-measure compare to TransDETR. This is due to the domain gap between image data and video data, directly using an image spotter leads to a low confidence and consequently low Recall on video data. When encompasses the rescoring mechanism, GoMatching achieves a 9.1 improvement on Recall compared to DeepSolo and a $6.5\%$ F-measure enhancement compared to TransDETR. These improvements highlight the effectiveness of rescoring mechanism in alleviating the domain gap and leading to a better tracking candidate pool. The impressive results of GoMatching are not merely attributed to the introduction of a robust image text spotter.

To further explore how the rescoring mechanism eases the domain gap, we calculate the AP of detection results on ICDAR13-video, as shown in the Table 9. Incorporating the rescoring mechanism effectively improves the detection performance of DeepSolo on video datasets, particularly for small text, resulting in a $3.9\%$ improvement on $\mathrm{AP_S}$ . We also present more visual results to embody the potency of rescoring mechanism in Fig. 8.

# G More Visualization Results

We present more visualization results of TransDETR and GoMatching in Fig. 9, including some failure cases. It can be observed that GoMatching exhibits a significant improvement in text recognition performance compared to TransDETR. It should be noted that GoMatching may also experience failures due to the image-video domain gap and extreme cases, such as very small text instances and significant motion blur. These issues can be mitigated by employing a stronger text spotter with a more representative backbone and training on larger, more diverse datasets.