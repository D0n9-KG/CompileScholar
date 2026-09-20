# Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

Chengcheng Wang Wei He Ying Nie Jianyuan Guo Chuanjian Liu Kai Han\* Yunhe Wang\*

Huawei Noah's Ark Lab

{wangchengcheng11,hewei142,ying.nie,jianyuan.guo,liuchuanjian,kai.han,yunhe.wang}@huawei.com

# Abstract

In the past years, YOLO-series models have emerged as the leading approaches in the area of real-time object detection. Many studies pushed up the baseline to a higher level by modifying the architecture, augmenting data and designing new losses. However, we find previous models still suffer from information fusion problem, although Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) have alleviated this. Therefore, this study provides an advanced Gather-and-Distribute mechanism (GD) mechanism, which is realized with convolution and self-attention operations. This new designed model named as Gold-YOLO, which boosts the multi-scale feature fusion capabilities and achieves an ideal balance between latency and accuracy across all model scales. Additionally, we implement MAE-style pretraining in the YOLO-series for the first time, allowing YOLO-series models could be to benefit from unsupervised pretraining. Gold-YOLO-N attains an outstanding 39.9% AP on the COCO val2017 datasets and 1030 FPS on a T4 GPU, which outperforms the previous SOTA model YOLOv6-3.0-N with similar FPS by +2.4%. The PyTorch code is available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO, and the MindSpore code is available at https://gitee.com/mindspore/models/tree/master/research/cv/Gold\_YOLO.

# 1 Introduction

Object detection as a fundamental vision task that aims to recognize the categories and locate the positions of objects. It can be widely used in a wide range of applications, such as intelligent security, autonomous driving, robot navigation, and medical diagnosis. High-performance and low-latency object detector is receiving increasing attention for deployment on the edge devices.

Over the past few years, researchers have extensive research on CNN-based detection networks, gradually evolving the object detection framework from two-stage (e.g., Faster RCNN [42] and Mask RCNN [25]) to one-stage (e.g., YOLO [39]), and from anchor-based (e.g., YOLOv3 [41] and YOLOv4 [2]) to anchor-free (e.g., CenterNet [10], FCOS [46] and YOLOX [11]). [12, 7, 17] studied the optimal network structure through NAS for object detection task, and [16, 23, 19] explore another way to improve the performance of the model by distillation. Single-stage detection models, especially YOLO series models, have been widely welcomed in the industry due to their simple structure and balance between speed and accuracy.

Improvement of backbone is also an important research direction in the field of vision. As described in the survey $[20]$ , $[26, 27, 59, 21]$ has achieved a balance between precision and speed, while $[9, 35, 22, 18]$ has shown strong performance in precision. These backbones have improved the performance of the original model in different visual tasks, ranging from high-level tasks like object

![](images/0ab5aa34a6f4a0bb94fb366f66dd228d80bf17c0fdb78a883a50d66dac251b68.jpg)

<details>
<summary>line</summary>

| Throughput(FPS) on T4 | Gold-YOLO-L | Gold-YOLO-M | Gold-YOLO-S | Gold-YOLO-N | YOLOv5 | YOLOv6-3.0 | YOLOv7 | YOLOv8 | PPYOLOE | YOLOX |
| --------------------- | ----------- | ----------- | ----------- | ----------- | ------ | ---------- | ------ | ------ | ------- | ----- |
| 100                   | 52.0        | 51.5        | 51.0        | 50.5        | 50.0   | 50.5       | 51.0   | 50.5   | 50.0    | 50.0  |
| 200                   | 50.0        | 49.5        | 49.0        | 48.5        | 48.0   | 48.5       | 49.0   | 48.5   | 48.0    | 48.0  |
| 400                   | 47.0        | 46.5        | 46.0        | 45.5        | 45.0   | 45.5       | 46.0   | 45.5   | 45.0    | 45.0  |
| 600                   | 44.0        | 43.5        | 43.0        | 42.5        | 42.0   | 42.5       | 43.0   | 42.5   | 42.0    | 42.0  |
| 800                   | 41.0        | 40.5        | 40.0        | 39.5        | 39.0   | 39.5       | 40.0   | 39.5   | 39.0    | 39.0  |
| 1000                  | 38.0        | 37.5        | 37.0        | 36.5        | 36.0   | 36.5       | 37.0   | 36.5   | 36.0    | 36.0  |
| 1200                  | 35.0        | 34.5        | 34.0        | 33.5        | 33.0   | 33.5       | 34.0   | 33.5   | 33.0    | 33.0  |
</details>

(a) TensorRT 7, FP16 Throughput (FPS), BS=32

![](images/b3c67935cf703b27901cd17cbdcfac0901d34febf4741f1222d90fc34c0be9d8.jpg)

<details>
<summary>line</summary>

| Throughput(FPS on T4 (bs=32, TensorRT8)) | Gold-YOLO-L | Gold-YOLO-M | Gold-YOLO-S | Gold-YOLO-N | YOLOv5 | YOLOv6-3.0 | YOLOv7 | YOLOv8 | PPYOLOE | YOLOX |
| ---------------------------------------- | ----------- | ----------- | ----------- | ----------- | ------ | ---------- | ------ | ------ | ------- | ----- |
| 200                                      | 52.0        | 51.5        | 50.5        | 49.5        | 49.0   | 50.0       | 51.0   | 50.5   | 50.0    | 50.0  |
| 400                                      | 48.0        | 47.0        | 46.0        | 45.0        | 45.0   | 46.0       | 47.0   | 46.5   | 46.0    | 46.0  |
| 600                                      | 45.0        | 44.0        | 43.0        | 42.0        | 42.0   | 43.0       | 44.0   | 43.5   | 43.0    | 43.0  |
| 800                                      | 42.0        | 41.0        | 40.0        | 39.0        | 39.0   | 40.0       | 41.0   | 40.5   | 40.0    | 40.0  |
| 1000                                     | 40.0        | 39.0        | 38.0        | 37.0        | 37.0   | 38.0       | 39.0   | 38.5   | 38.0    | 38.0  |
| 1200                                     | 38.0        | 37.0        | 36.0        | 35.0        | 35.0   | 36.0       | 37.0   | 36.5   | 36.0    | 36.0  |
| 1400                                     | 36.0        | 35.0        | 34.0        | 33.0        | 33.0   | 34.0       | 35.0   | 34.5   | 34.0    | 34.0  |
</details>

(b) TensorRT 8, FP16 Throughput (FPS), BS=32   
Figure 1: Comparison of state-of-the-art efficient object detectors in Tesla T4 GPU. Both latency and throughput (batch size of 32) are given for a handy reference. (a) and (b) test with TensorRT 7 and 8, respectively.

detection to low-level tasks like image restoration. By using the encoder-decoder structure with the transformer, researchers have constructed a series of DETR-like object detection models, such as DETR [3] and DINO [56]. These models can capture long-range dependency between objects, enabling transformer-based detectors to achieve comparable or superior performance with most refined classical detectors. Despite the notable performance of transformer-based detectors, they fall short when compared to the speed of CNN-based models. Small-scale object detection models based on CNN still dominate the speed-accuracy trade-off, such as YOLOX [11] and YOLOv6-v8 [22, 48, 14]. We focus on the real-time object detection models, especially YOLO series for mobile deployment. Mainstream real-time object detectors consist of three parts: backbone, neck, and head. The backbone architecture has been widely investigated [41, 43, 9, 35] and the head architecture is typically straight forward, consisting of several convolutional or fully-connected layers. The necks in YOLO series usually use Feature Pyramid Network (FPN) and its variants to fuse multi-level features. These neck modules basically follow the architecture shown in Fig. 3. However, the current approach to information fusion has a notable flaw: when there is a need to integrate information across layers (e.g., level-1 and level-3 are fused), the conventional FPN-like structure fails to transmit information without loss, which hinders YOLOs from better information fusion.

Built upon the concept of global information fusion, TopFormer [58] has achieved remarkable results in semantic segmentation tasks. In this paper, we expanding on the foundation of TopFormer's theory, propose a novel Gather-and-Distribute mechanism (GD) for efficient information exchanging in YOLOs by globally fusing multi-level features and injecting the global information into higher levels. This significantly enhances the information fusion capability of the neck without significantly increasing the latency, improving the model's performance across varying object sizes. Specifically, GD mechanism comprises two branches: a shallow gather-and-distribute branch and a deep gather-and-distribute branch, which extract and fuse feature information via a convolution-based block and an attention-based block, respectively. To further facilitate information flow, we introduce a lightweight adjacent-layer fusion module which combines features from neighboring levels on a local scale. Our Gold-YOLO architectures surpasses the existing YOLO series, effectively demonstrating the effectiveness of our proposed approach.

To further improve the accuracy of the model, we also introduce a pre-training method, where we pre-train the backbone on ImageNet 1K using the MAE method, which significantly improves the convergence speed and accuracy of the model. For example, our Gold-YOLO-S with pre-training achieves 46.4% AP, which outperforms the previous SOTA YOLOv6-3.0-S with 45.0% AP at similar speed.

# 2 Related works

# 2.1 Real-time object detectors

After years of development, the YOLO-series model has become popular in the real-time object detection area. YOLOv1-v3 [39, 40, 41] constructs the initial YOLOs, identifies a single-stage

![](images/4cdc0d7e8f9ab3d32f0c9b84e7c60a0f499e84a99356e4b97e4fe3c421dbb88a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Backbone"] --> B["B2"]
    B --> C["Low-FAM"]
    C --> D["Low-IFM"]
    D --> E["Inject"]
    E --> F["P3"]
    F --> G["Inject"]
    G --> H["N3"]
    H --> I["High-IFM"]
    I --> J["High-GD"]
    J --> K["Neck"]
    K --> L["Head"]
    L --> M["Backbone"]
    style A fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
```
</details>

Figure 2: The architecture of the proposed Gold-YOLO.

detection structure consisting of three parts, backbone-neck-head, predicts objects of different sizes through multi-scale branches, become a representative single-stage object detection model. YOLOv4 [2] optimizes the previously used darknet backbone structure and propose a series of improvements, like the Mish activation function, PANet and data augmentation methods. YOLOv5 [13] inheriting the YOLOv4 [2] scheme with improved data augmentation strategy and a greater variety of model variants. YOLOX [11] incorporates Multi positives, Anchor-free, and Decoupled Head into the model structure, setting a new paradigm for YOLO-model design. YOLOv6 [32, 31] brings the reparameterization method to YOLO-series models for the first time, proposing EfficientRep Backbone and Rep-PAN Neck. YOLOv7 [48] focuses on analyzing the effect of gradient paths on the model performance and proposes the E-ELAN structure to enhance the model capability without destroying the original gradient paths. The YOLOv8 [14] takes the strengths of previous YOLO models and integrates them to achieve the SOTA of the current YOLO family.

# 2.2 Transformer-base object detection

Vision Transformer (ViT) emerged as a competitive alternative to convolutional neural networks (CNNs) that are widely used for different image recognition tasks. DETR $[3]$ applies the transformer structure to the object detection task, reconstructing the detection pipeline and eliminating many hand-designed parts and NMS components to simplify the model design and overall process. Combining the sparse sampling capability of deformable convolution with the global relationship modeling capability of transformer, Deformable DETR $[61]$ improve convergence speed while improve model speed and accuracy. DINO $[56]$ first time introduced Contrastive denoising, Mix query selection and a look forward twice scheme. The recent RT-DETR $[36]$ improved the encoder-decoder structure to solve the slow DETR-like model problem, outperforming YOLO-L/X in both accuracy and speed. However, the limitations of the DETR-like structure prevent it from showing sufficient dominance in the small model region, where YOLOs remain the SOTA of accuracy and velocity balance.

# 2.3 Multi-scale features for object detection

Traditionally, features at different levels carry positional information about objects of various sizes. Larger features encompass low-dimensional texture details and positions of smaller objects. In contrast, smaller features contain high-dimensional information and positions of larger objects. The original idea behind Feature Pyramid Networks (FPN) proposed by $[34]$ is that these diverse pieces of information can enhance network performance through mutual assistance. FPN provides an efficient architectural design for fusing multi-scale features through cross-scale connections and information exchange, thereby boosting the detection accuracy of objects of varied sizes.

Based on FPN, the Path Aggregation Network (PANet) [49] incorporates a bottom-up path to make information fusion between different levels more adequate. Similarly, EfficientDet [44] presents a new repeatable module (BiFPN) to increase the efficiency of information fusion between different levels. M2Det [60] introduced an efficient MLFPN architecture with U-shape and Feature Fusion Modules. Ping-Yang Chen [5] improved interaction between deep and shallow layers using bidirectional fusion modules. Unlike these inter-layer works, [37] explored individual feature information using the Centralized Feature Pyramid (CFP) method. Additionally, [53] extended FPN with the Asymptotic Feature Pyramid Network (AFPN) to interact across non-adjacent layers. In response to FPN's limitations in detecting large objects, [30] proposed a refined FPN structure. YOLO-F [6] achieved

![](images/6e2aadde9cc1bdd3051973834ffd1fe31df5f971db8ce1fb5a9d4ece773709a5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["level-1"] --> B["fuse"]
    C["level-2"] --> D["fuse"]
    E["level-3"] --> F["fuse"]
    B --> D
    D --> F
    F --> G["fuse"]
    G --> H["level-1"]
    G --> I["level-2"]
    G --> J["level-3"]
```
</details>

(a) traditional neck structure

![](images/78d57a6bf472af38b1fbb254769c164abefe99b1f20bd947720cc12d9d130117.jpg)

<details>
<summary>natural_image</summary>

Group of people gathered in a crowded indoor setting, some highlighted with a red box (no visible text or symbols)
</details>

![](images/b63f9fca1ce308a7d2b0a40b02bf228f0b709a0416be6fa263926a5615703903.jpg)

<details>
<summary>natural_image</summary>

Two side-by-side thermal or spectral images showing abstract color patterns (no text or symbols)
</details>

(b) traditional neck

![](images/6b39a765f7216776bc413c7df642e0b3132ae8ab40ec0c28fd9535d910d49813.jpg)

<details>
<summary>natural_image</summary>

Group of people in a dimly lit indoor setting, some wearing face masks and others with orange highlights (no visible text or symbols)
</details>

![](images/cef9471be15b14c2f7ae73ca6c2ef75d8ebc02c6df9e9f5e4660e01aa82abefd.jpg)

<details>
<summary>natural_image</summary>

Side-by-side comparison of a person's profile with a colorful abstract background (no text or symbols)
</details>

(c) our proposed neck   
Figure 3: (a) is example diagram of traditional neck information fusion structure. (b) and (c) is AblationCAM [38] visualization

state-of-the-art performance with single-level features. SFNet [33] aligns different level features with semantic flow to improves FPN performance in model. SAFNet [29] introduced Adaptive Feature Fusion and Self-Enhanced Modules. [4] presented a parallel FPN structure for object detection with bi-directional fusion. However, due to the excessive number of paths and indirect interaction methods in the network, the previous FPN-based fusion structures still have drawbacks in low speed, cross-level information exchange and information loss.

However, due to the excessive number of paths and indirect interaction methods in the network, the previous FPN-based fusion structures still have drawbacks in low speed, cross-level information exchange and information loss.

# 3 Method

# 3.1 Preliminaries

The YOLO series neck structure, as depicted in Fig.3, employs a traditional FPN structure, which comprises multiple branches for multi-scale feature fusion. However, it only fully fuse features from neighboring levels, for other layers information it can only be obtained indirectly ‘recursively’. In Fig.3, it shows the information fusion structure of the conventional FPN: where existing level-1, 2, and 3 are arranged from top to bottom. FPN is used for fusion between different levels. There are two distinct scenarios when level-1 get information from the other two levels:

1) If level-1 seeks to utilize information from level-2, it can directly access and fuse this information.   
2) If level-1 wants to use level-3 information, level-1 should recursively calling the information fusion module of the adjacent layer. Specifically, the level-2 and level-3 information must be fused first, then level-1 can indirectly obtain level-3 information by combining level-2 information.

This transfer mode can result in a significant loss of information during calculation. Information interactions between layers can only exchange information that is selected by intermediate layers, and not selected information is discarded during transmission. This leads to a situation where information at a certain level can only adequately assist neighboring layers and weaken the assistance provided to other global layers. As a result, the overall effectiveness of the information fusion may be limited.

To avoid information loss in the transmission process of traditional FPN structures, we abandon the original recursive approach and construct a novel gather-and-distribute mechanism (GD). By using a unified module to gather and fuse information from all levels and subsequently distribute it to different levels, we not only avoid the loss of information inherent in the traditional FPN structure but also enhance the neck's partial information fusion capabilities without significantly increasing latency. Our approach thus allows for more effective leveraging of the features extracted by the backbone, and can be easily integrated into any existing backbone-neck-head structure.

In our implementation, the process gather and distribute correspond to three modules: Feature Alignment Module (FAM), Information Fusion Module (IFM), and Information Injection Module (Inject).

- The gather process involves two steps. Firstly, the FAM collects and aligns features from various levels. Secondly, IFM fuses the aligned features to generate global information.   
- Upon obtaining the fused global information from the gather process, the inject module distribute this information across each level and injects it using simple attention operations, subsequently enhancing the branch's detection capability.

To enhance the model's ability to detect objects of varying sizes, we developed two branches: low-stage gather-and-distribute branch (Low-GD) and high-stage gather-and-distribute branch (High-GD). These branches extract and fuse large and small size feature maps, respectively. Further details are provided in Sections 4.1 and 4.2. As shown in Fig. 2, the neck's input comprises the feature maps $B2$ , $B3$ , $B4$ , $B5$ extracted by the backbone, where $B_i \in \mathbb{R}^{N \times C_{Bi} \times R_{Bi}}$ . The batch size is denoted by $N$ , the channels by $C$ , and the dimensions by $R = H \times W$ . Moreover, the dimensions of $R_{B2}$ , $R_{B3}$ , $R_{B4}$ , and $R_{B5}$ are $R$ , $\frac{1}{2} R$ , $\frac{1}{4} R$ , and $\frac{1}{8} R$ , respectively.

# 3.2 Low-stage gather-and-distribute branch

In this branch, the output B2, B3, B4, B5 features from the backbone are selected for fusion to obtain high resolution features that retain small target information. The structure show in Fig.4(a)

Low-stage feature alignment module. In low-stage feature alignment module (Low-FAM), we employ the average pooling (AvgPool) operation to down-sample input features and achieve a unified size. By resizing the features to the smallest feature size of the group $(R_{B4} = \frac{1}{4}R)$ , we obtain $F_{align}$ . The Low-FAM technique ensures efficient aggregation of information while minimizing the computational complexity for subsequent processing through the transformer module.

The target alignment size is chosen based on two conflicting considerations: (1) To retain more low-level information, larger feature sizes are preferable; however, (2) as the feature size increases, the computational latency of subsequent blocks also increases. To control the latency in the neck part, it is necessary to maintain a smaller feature size.

Therefore, we choose the $R_{B4}$ as the target size of feature alignment to achieve a balance between speed and accuracy.

Low-stage information fusion module. The low-stage information fusion module (Low-IFM) design comprises multi-layer reparameterized convolutional blocks (RepBlock) and a split operation. Specifically, RepBlock takes $F_{align}$ ( $channel = sum(C_{B2}, C_{B3}, C_{B4}, C_{B5})$ ) as input and produces $F_{fuse}$ ( $channel = C_{B4} + C_{B5}$ ). The middle channel is an adjustable value (e.g., 256) to accommodate varying model sizes. The features generated by the RepBlock are subsequently split in the channel dimension into $F_{inj\_P3}$ and $F_{inj\_P4}$ , which are then fused with the different level's feature.

The formula is as follows:

$$
F _ {\text { align }} = \text { Low\_FAM } ([ B 2, B 3, B 4, B 5 ]), \tag {1}
$$

$$
F _ {\text { fuse }} = \operatorname{RepBlock} \left(F _ {\text { align }}\right), \tag {2}
$$

$$
F _ {\text { inj\_P3 }}, F _ {\text { inj\_P4}} = \text { Split } (F _ {\text { fuse }}). \tag {3}
$$

Information injection module. In order to inject global information more efficiently into the different levels, we draw inspiration from the segmentation experience [47] and employ attention operations to fuse the information, as illustrated in Fig. 5. Specifically, we input both local information (which refers to the feature of the current level) and global inject information (generated by IFM), denoted as $F_{local}$ and $F_{inj}$ , respectively. We use two different Convs with $F_{inj}$ for calculation, resulting in $F_{global\_embed}$ and $F_{act}$ . While $F_{local\_embed}$ is calculated with $F_{local}$ using Conv. The fused feature $F_{out}$ is then computed through attention. Due to the size differences between $F_{local}$ and $F_{global}$ , we employ average pooling or bilinear interpolation to scale $F_{global\_embed}$ and $F_{act}$ according to the size of $F_{inj}$ , ensuring proper alignment. At the end of each attention fusion, we add the RepBlock to further extract and fuse the information.

![](images/524654389a0450c873055915af0ba178924a66ccf9d048d9dff7a2701d515226.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    B5 --> Bilinear
    B4 --> Bilinear
    B3 --> Bilinear
    B2 --> Bilinear
    Bilinear --> C
    B3 --> C
    B2 --> C
    C --> AvgPool
    AvgPool --> AvgPool
    AvgPool --> Conv
    Conv --> RegConv-blocks
    RegConv-blocks --> Conv
    Conv --> Split
    Split --> Inject_P4
    Split --> Inject_P3
    style C fill:#99ccff,stroke:#333
    style AvgPool fill:#99ccff,stroke:#333
    style Conv fill:#ffcccc,stroke:#333
    style Split fill:#ffcccc,stroke:#333
```
</details>

(a) low-stage gather-and-distribute branch

![](images/cfc5763a30996545de2000b863c1b16e77469f4e1cdfcd7e09daf4af79e1fb03.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    P3 --> avgpool["avgpool"]
    P4 --> avgpool
    avgpool --> HighFAM["High-FAM"]
    HighFAM --> MultiHeadAttention["Multi-Head Attention"]
    MultiHeadAttention --> FeedForwardNetwork["Feed-Forward Network"]
    FeedForwardNetwork --> SplitSplit["Split"]
    SplitSplit --> Project_N5["Project_N5"]
    SplitSplit --> Project_N4["Project_N4"]
    HighFAM --> HighIFM["High-IFM x L"]
    style HighFAM fill:#f9f,stroke:#333
    style Project_N5 fill:#ccf,stroke:#333
    style Project_N4 fill:#cfc,stroke:#333
```
</details>

(b) high-stage gather-and-distribute branch   
Figure 4: Gather-and-Distribute structure. In (a), the Low-FAM and Low-IFM is low-stage feature alignment module and low-stage information fusion module in low-stage branch, respectively. In (b), the High-FAM and High-IFM is high-stage feature alignment module and high-stage information fusion module, respectively.

In low stage, $F_{local}$ is equal to $Bi$ , so the formula is as follows:

$$
F _ {\text { global\_act\_Pi }} = \text { resize } (\text { Sigmoid } (\text { Conv } _ {\text { act }} (F _ {\text { inj\_Pi }}))), \tag {4}
$$

$$
F _ {\text { global\_embed\_Pi }} = \operatorname{resize} \left(\operatorname{Conv} _ {\text { global\_embed\_Pi }} \left(F _ {\text { inj\_Pi }}\right)\right), \tag {5}
$$

$$
F _ {\text {att\_fuse\_Pi}} = \operatorname{Conv} _ {\text {local\_embed\_Pi}} (B i) * F _ {\text {ing\_act\_Pi}} + F _ {\text {global\_embed\_Pi}}, \tag {6}
$$

$$
P i = \text { RepBlock } (F _ {\text { att\_fuse\_Pi }}). \tag {7}
$$

# 3.3 High-stage gather-and-distribute branch

The High-GD fuses the features $\{P3, P4, P5\}$ that are generated by the Low-GD, as shown in Fig.4(b)

High-stage feature alignment module. The high-stage feature alignment module (High-FAM) consists of avgpool, which is utilized to reduce the dimension of input features to a uniform size. Specifically, when the size of the input feature is $\{R_{P3}, R_{P4}, R_{P5}\}$ , avgpool reduces the feature size to the smallest size within the group of features ( $R_{P5} = \frac{1}{8}R$ ). Since the transformer module extracts high-level information, the pooling operation facilitates information aggregation while decreasing the computational requirements for the subsequent step in the Transformer module.

High-stage information fusion module. The high-stage information fusion module (High-IFM) comprises the transformer block (explained in greater detail below) and a splitting operation, which involves a three-step process: (1) the $F_{align}$ , derived from the High-FAM, are combined using the transformer block to obtain the $F_{fuse}$ . (2) The $F_{fuse}$ channel is reduced to $sum(C_{P4}, C_{P5})$ via a $Conv1 \times 1$ operation. (3) The $F_{fuse}$ is partitioned into $F_{inj\_N4}$ and $F_{inj\_N5}$ along the channel dimension through a splitting operation, which is subsequently employed for fusion with the current level feature.

The formula is as follows:

$$
F _ {\text { align }} = \text { High\_FAM } ([ P 3, P 4, P 5 ]), \tag {8}
$$

$$
F _ {\text { fuse }} = \text { Transformer } (F _ {\text { align }}), \tag {9}
$$

$$
F _ {\text { inj\_N4 }}, F _ {\text { inj\_N5}} = \text { Split } (C o n v 1 \times 1 (F _ {\text { fuse }})). \tag {10}
$$

The transformer fusion module in Eq. 8 comprises several stacked transformers, with the number of transformer blocks denoted by $L$ . Each transformer block includes a multi-head attention block, a Feed-Forward Network (FFN), and residual connections. To configure the multi-head attention block, we adopt the same settings as LeViT [15], assigning head dimensions of keys $K$ and queries $Q$ to $D$ (e.g., 16) channels, and $V = 2D$ (e.g., 32) channels. In order to accelerate inference, we substitute the velocity-unfriendly operator, Layer Normalization, with Batch Normalization for each convolution, and replace all GELU activations with ReLU. This minimizes the impact of the transformer module on the model's speed. To establish our Feed-Forward Network, we follow the methodologies presented in [28, 55] for constructing the FFN block. To enhance the local connections of the transformer block, we introduce a depth-wise convolution layer between the two 1x1 convolution layers. We also set the expansion factor of the FFN to 2, aiming to balance speed and computational cost.

![](images/6aa66271f282d646e1d7636e07470672002faef1cabdd8cab6a2e2015a7e6754.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x_local"] --> B["Conv 1x1"]
    C["x_global"] --> D["Conv 1x1"]
    B --> E["×"]
    D --> E
    E --> F["avgpool / bilinear"]
    F --> G["Sigmoid"]
    G --> H["×"]
    H --> I["RepConv-blocks"]
    I --> J["Inject"]
    J --> K["×"]
    K --> L["avgpool / bilinear"]
    L --> M["×"]
    M --> N["conv 1x1"]
    N --> O["Conv 1x1"]
```
</details>

(a) Information injection module   
![](images/84ba1784c9d95698d0e893acca747a6006187e82c870946b25ce2a0d56d89eea.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Inject-LAF"] --> B["RepConv-blocks"]
    B --> C["×"]
    C --> D["avgpool / bilinear"]
    D --> E["Sigmoid"]
    E --> F["Conv 1x1"]
    F --> G["LAF"]
    G --> H["x_local"]
    G --> I["x_global"]
    C --> J["avgpool / bilinear"]
    J --> K["Conv 1x1"]
    K --> L["x_global"]
    L --> M["Conv 1x1"]
    M --> N["avgpool / bilinear"]
    N --> O["×"]
    O --> P["avgpool / bilinear"]
    P --> Q["×"]
    Q --> R["avgpool / bilinear"]
    R --> S["×"]
    S --> T["avgpool / bilinear"]
    T --> U["×"]
    U --> V["avgpool / bilinear"]
    V --> W["×"]
    W --> X["avgpool / bilinear"]
    X --> Y["×"]
    Y --> Z["avgpool / bilinear"]
    Z --> AA["×"]
    AA --> AB["avgpool / bilinear"]
    AB --> AC["×"]
    AC --> AD["avgpool / bilinear"]
    AD --> AE["×"]
    AE --> AF["avgpool / bilinear"]
    AF --> AG["×"]
    AG --> AH["avgpool / bilinear"]
    AH --> AI["×"]
    AI --> AJ["avgpool / bilinear"]
    AJ --> AK["×"]
    AK --> AL["avgpool / bilinear"]
    AL --> AM["×"]
    AM --> AN["avgpool / bilinear"]
    AN --> AO["×"]
    AO --> AP["avgpool / bilinear"]
    AP --> AQ["×"]
    AQ --> AR["avgpool / bilinear"]
    AR --> AS["×"]
    AS --> AT["avgpool / bilinear"]
    AT --> AU["×"]
    AU --> AV["avgpool / bilinear"]
    AV --> AW["×"]
    AW --> AX["avgpool / bilinear"]
    AX --> AY["×"]
    AY --> AZ["avgpool / bilinear"]
    AZ --> BA["×"]
    BA --> BB["avgpool / bilinear"]
    BB --> BC["×"]
    BC --> BD["avgpool / bilinear"]
    BD --> BE["×"]
    BE --> BF["avgpool / bilinear"]
    BF --> BG["×"]
    BG --> BH["avgpool / bilinear"]
    BH --> BI["×"]
    BI --> BJ["avgpool / bilinear"]
    BJ --> BK["×"]
    BK --> BL["avgpool / bilinear"]
    BL --> BM["×"]
    BM --> BN["avgpool / bilinear"]
    BN --> BO["×"]
    BO --> BP["avgpool / bilinear"]
    BP --> BQ["×"]
    BQ --> BR["avgpool / bilinear"]
    BR --> BS["×"]
    BS --> BT["avgpool / bilinear"]
    BT --> BU["×"]
    BU --> BV["avgpool / bilinear"]
    BV --> BW["×"]
    BW --> BX["avgpool / bilinear"]
    BX --> BY["×"]
    BY --> BZ["avgpool / bilinear"]
    BZ --> CA["×"]
    CA --> CB["avgpool / bilinear"]
    CB --> CC["×"]
    CC --> CD["avgpool / bilinear"]
    CD --> CE["×"]
    CE --> CF["avgpool / bilinear"]
    CF --> CG["×"]
    CG --> CH["avgpool / bilinear"]
    CH --> CI["×"]
    CI --> CJ["avgpool / bilinear"]
    CJ --> CK["×"]
    CK --> CL["avgpool / bilinear"]
    CL --> CM["×"]
    CM --> CN["avgpool / bilinear"]
    CN --> CO["×"]
    CO --> CP["avgpool / bilinear"]
    CP --> CQ["×"]
    CQ --> CR["avgpool / bilinear"]
    CR --> CS["×"]
    CS --> CT["avgpool / bilinear"]
    CT --> CU["×"]
    CU --> CV["avgpool / bilinear"]
    CV --> CW["×"]
    CW --> CX["avgpool / bilinear"]
    CX --> CY["×"]
    CY --> CZ["avgpool / bilinear"]
    CZ --> DA["×"]
    DA --> DB["avgpool / bilinear"]
    DB --> DC["×"]
    DC --> DD["avgpool / bilinear"]
    DD --> DE["×"]
    DE --> DF["avgpool / bilinear"]
    DF --> DG["×"]
    DG --> DH["avgpool / bilinear"]
    DH --> DI["×"]
    DI --> DJ["avgpool / bilinear"]
    DJ --> DK["×"]
```
</details>

![](images/7e22e9dd288b3306134e923aa3543983227abed38ab5814e88f3a10775e108dc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Conv 1x1"] --> C["C"]
    B["avgpool"] --> C
    D["Conv 1x1"] --> C
    E["bilinear"] --> C
    F["B_n-1"] --> C
    G["B_n"] --> C
    H["B_n+1"] --> C
    I["LAF Low-stage"] --> A
    I --> B
    I --> D
    I --> E
```
</details>

![](images/fbc875169028d55e898f29511e3d48bf34c28b6272ac7316b45a682288199d64.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["avgpool"] --> C["C"]
    B["P_n-1"] --> C
    C --> D["Conv 1x1"]
    D --> E["P_n"]
    style A fill:#d4edda,stroke:#333
    style B fill:#d4edda,stroke:#333
    style C fill:#fff,stroke:#333
    style D fill:#fff,stroke:#333
    style E fill:#fff,stroke:#333
```
</details>

(b) lightweight adjacent layer fusion module   
Figure 5: The information injection module and lightweight adjacent layer fusion(LAF) module

Information injection module. The information injection module in High-GD is exactly the same as in Low-GD. In high stage, $F_{local}$ is equal to Pi, so the formula is as follows:

$$
F _ {\text { global\_act\_Ni }} = \text { resize } (\text { Sigmoid } (\text { Conv } _ {\text { act }} (F _ {\text { inj\_Ni}}))), \tag {11}
$$

$$
F _ {\text { global\_embed\_Ni }} = \operatorname{resize} \left(\operatorname{Conv} _ {\text { global\_embed\_Ni }} \left(F _ {\text { inj\_Ni }}\right)\right), \tag {12}
$$

$$
F _ {\text {att\_fuse\_Ni}} = \operatorname{Conv} _ {\text {local\_embed\_Ni}} (P i) * F _ {\text {ing\_act\_Ni}} + F _ {\text {global\_embed\_Ni}}, \tag {13}
$$

$$
N i = \text { RepBlock } (F _ {\text { att\_fuse\_Ni }}). \tag {14}
$$

# 3.4 Enhanced cross-layer information flow

We have achieved better performance than existing methods using only a global information fusion structure. To further enhance the performance, we drew inspiration from the PAFPN module in YOLOv6 [31] and introduced an Inject-LAF module. This module is an enhancement of the injection module and includes a lightweight adjacent layer fusion (LAF) module that is added to the input position of the injection module.

To achieve a balance between speed and accuracy, we designed two LAF models: LAF low-level model and LAF high-level model, which are respectively used for low-level injection (merging features from adjacent two layers) and high-level injection (merging features from adjacent one layer). There structure is shown in Fig. 5 (b).

To ensure that feature maps from different levels are aligned with the target size, the two LAF models in our implementation utilize only three operators: bilinear interpolation to up-sample features that are too small, average pooling to down-sample features that are too large, and 1x1 convolution to adjust features that differ from the target channel.

The combination of the LAF module with the information injection module in our model effectively balances the between accuracy and speed. By using simplified operations, we are able to increase the number of information flow paths between different levels, resulting in improved performance without significantly increased the latency.

# 3.5 Masked image modeling pre-training

Recent methods such as BEiT [1], MAE [24] and SimMIM [51], have demonstrated the effectiveness of masked image modeling (MIM) for vision tasks. However, these methods are not specifically tailored for convolutional networks (convnets). SparK [45] and ConvNeXt-V2 [50] are pioneers in exploring the potential of masked image modeling for convnets.

In this study, we adopt MIM Pre-training following the SparK's [45] methodology, which successfully identifies and overcomes two key obstacles in extending the success of MAE-style pretraining to

Table 1: Comparisons with other YOLO-series detectors on COCO 2017 val. FPS and latency are measured in FP16-precision on a Tesla T4 in the same environment with TensorRT 7. All our models are trained for 300 epochs. Both the accuracy and the speed performance of our models are evaluated with the input resolution of 640x640. ‘†’ represents that the self-distillation method is utilized, and ‘\*’ represents that the MIM pre-training method is utilized. 

<table><tr><td>Method</td><td>Input Size</td><td> $AP^{val}$ </td><td> $AP^{val}_{50}$ </td><td>FPS (bs=1)</td><td>FPS (bs=32)</td><td>Latency (bs=1)</td><td>Params</td><td>FLOPs</td></tr><tr><td>YOLOv5-N [13]</td><td>640</td><td>28.0%</td><td>45.7%</td><td>602</td><td>735</td><td>1.7 ms</td><td>1.9 M</td><td>4.5 G</td></tr><tr><td>YOLOv5-S [13]</td><td>640</td><td>37.4%</td><td>56.8%</td><td>376</td><td>444</td><td>2.7 ms</td><td>7.2 M</td><td>16.5 G</td></tr><tr><td>YOLOv5-M [13]</td><td>640</td><td>45.4%</td><td>64.1%</td><td>182</td><td>209</td><td>5.5 ms</td><td>21.2 M</td><td>49.0 G</td></tr><tr><td>YOLOv5-L [13]</td><td>640</td><td>49.0%</td><td>67.3%</td><td>113</td><td>126</td><td>8.8 ms</td><td>46.5 M</td><td>109.1 G</td></tr><tr><td>YOLOX-Tiny [11]</td><td>416</td><td>32.8%</td><td>50.3%</td><td>717</td><td>1143</td><td>1.4 ms</td><td>5.1 M</td><td>6.5 G</td></tr><tr><td>YOLOX-S [11]</td><td>640</td><td>40.5%</td><td>59.3%</td><td>333</td><td>396</td><td>3.0 ms</td><td>9.0 M</td><td>26.8 G</td></tr><tr><td>YOLOX-M [11]</td><td>640</td><td>46.9%</td><td>65.6%</td><td>155</td><td>179</td><td>6.4 ms</td><td>25.3 M</td><td>73.8 G</td></tr><tr><td>YOLOX-L [11]</td><td>640</td><td>49.7%</td><td>68.0%</td><td>94</td><td>103</td><td>10.6 ms</td><td>54.2 M</td><td>155.6 G</td></tr><tr><td>PPYOLOE-S [52]</td><td>640</td><td>43.1%</td><td>59.6%</td><td>327</td><td>419</td><td>3.1 ms</td><td>7.9 M</td><td>17.4 G</td></tr><tr><td>PPYOLOE-M [52]</td><td>640</td><td>49.0%</td><td>65.9%</td><td>152</td><td>189</td><td>6.6 ms</td><td>23.4 M</td><td>49.9 G</td></tr><tr><td>PPYOLOE-L [52]</td><td>640</td><td>51.4%</td><td>68.6%</td><td>101</td><td>127</td><td>10.1 ms</td><td>52.2 M</td><td>110.1 G</td></tr><tr><td>YOLOv7-Tiny [48]</td><td>416</td><td>33.3%</td><td>49.9%</td><td>787</td><td>1196</td><td>1.3 ms</td><td>6.2 M</td><td>5.8 G</td></tr><tr><td>YOLOv7-Tiny [48]</td><td>640</td><td>37.4%</td><td>55.2%</td><td>424</td><td>519</td><td>2.4 ms</td><td>6.2 M</td><td>13.7 G</td></tr><tr><td>YOLOv7 [48]</td><td>640</td><td>51.2%</td><td>69.7%</td><td>110</td><td>122</td><td>9.0 ms</td><td>36.9 M</td><td>104.7 G</td></tr><tr><td>YOLOv7-E6E [48]</td><td>1280</td><td>56.8%</td><td>74.4%</td><td>16</td><td>17</td><td>59.6 ms</td><td>151.7 M</td><td>843.2 G</td></tr><tr><td>YOLOv8-N [14]</td><td>640</td><td>37.3%</td><td>52.6%</td><td>561</td><td>734</td><td>1.8 ms</td><td>3.2 M</td><td>8.7 G</td></tr><tr><td>YOLOv8-S [14]</td><td>640</td><td>44.9%</td><td>61.8%</td><td>311</td><td>387</td><td>3.2 ms</td><td>11.2 M</td><td>28.6 G</td></tr><tr><td>YOLOv8-M [14]</td><td>640</td><td>50.2%</td><td>67.2%</td><td>143</td><td>176</td><td>7.0 ms</td><td>25.9 M</td><td>78.9 G</td></tr><tr><td>YOLOv8-L [14]</td><td>640</td><td>52.9%</td><td>69.8%</td><td>91</td><td>105</td><td>11.0 ms</td><td>43.7 M</td><td>165.2 G</td></tr><tr><td>YOLOv6-3.0-N [31]</td><td>640</td><td>37.0%/37.5%†</td><td>52.7%/53.1%†</td><td>779</td><td>1187</td><td>1.3 ms</td><td>4.7 M</td><td>11.4 G</td></tr><tr><td>YOLOv6-3.0-S [31]</td><td>640</td><td>44.3%/45.0%†</td><td>61.2%/61.8%†</td><td>339</td><td>484</td><td>2.9 ms</td><td>18.5 M</td><td>45.3 G</td></tr><tr><td>YOLOv6-3.0-M [31]</td><td>640</td><td>49.1%/50.0%†</td><td>66.1%/66.9%†</td><td>175</td><td>226</td><td>5.7 ms</td><td>34.9 M</td><td>85.8 G</td></tr><tr><td>YOLOv6-3.0-L [31]</td><td>640</td><td>51.8%/52.8%†</td><td>69.2%/70.3%†</td><td>98</td><td>116</td><td>10.3 ms</td><td>59.6 M</td><td>150.7 G</td></tr><tr><td>Gold-YOLO-N</td><td>640</td><td>39.6%/39.9%†</td><td>55.7%/55.9%†</td><td>563</td><td>1030</td><td>1.7 ms</td><td>5.6 M</td><td>12.1 G</td></tr><tr><td>Gold-YOLO-S</td><td>640</td><td>45.4%/46.1%†</td><td>62.5%/63.3%†</td><td>286</td><td>446</td><td>3.3 ms</td><td>21.5 M</td><td>46.0 G</td></tr><tr><td>Gold-YOLO-M</td><td>640</td><td>49.8%/50.9%†</td><td>67.0%/68.2%†</td><td>152</td><td>220</td><td>6.4 ms</td><td>41.3 M</td><td>87.5 G</td></tr><tr><td>Gold-YOLO-L</td><td>640</td><td>51.8%/53.2%†</td><td>68.9%/70.5%†</td><td>88</td><td>116</td><td>11.1 ms</td><td>75.1 M</td><td>151.7 G</td></tr><tr><td>Gold-YOLO-S*</td><td>640</td><td>45.5%/46.4%†</td><td>62.2%/63.4%†</td><td>286</td><td>446</td><td>3.3 ms</td><td>21.5 M</td><td>46.0 G</td></tr><tr><td>Gold-YOLO-M*</td><td>640</td><td>50.2%/51.1%†</td><td>67.5%/68.5%†</td><td>152</td><td>220</td><td>6.4 ms</td><td>41.3 M</td><td>87.5 G</td></tr><tr><td>Gold-YOLO-L*</td><td>640</td><td>52.3%/53.3%†</td><td>69.6%/70.9%†</td><td>88</td><td>116</td><td>11.1 ms</td><td>75.1 M</td><td>151.7 G</td></tr></table>

convolutional networks (convnets). These challenges include the convolutional operations' inability to handle irregular and randomly masked input images, as well as the inconsistency between the single-scale nature of BERT pretraining and the hierarchical structure of convnets.

To address the first issue, unmasked pixels are treated as sparse voxels of 3D point clouds and employ sparse convolution for encoding. For the latter issue, a hierarchical decoder is developed to reconstruct images from multi-scale encoded features. The framework adopts a UNet-style architecture to decode multi-scale sparse feature maps, where all spatial positions are filled with embedded masks. We pretrain our model's backbone on ImageNet 1K for multiple Gold-YOLO models, and results in notable improvements

# 4 Experiment

# 4.1 Setups

Datasets. We perform extensive experiments on the Microsoft COCO datasets to validate the proposed detector. For the ablation study, we train on COCO train2017 and validate on COCO val2017 datasets. We use the standard COCO AP metric with a single scale image as input, and report the standard mean average precision (AP) result under different IoU thresholds and object scales.

Implementation details. We followed the setup of YOLOv6-3.0 [31] use the same structure (except for neck) and training configurations. The backbone of the network was implemented with

the EfficientRep Backbone, while the head utilized the Efficient Decoupled Head. The optimizer learning schedule and other setting also same as YOLOv6, i.e. stochastic gradient descent (SGD) with momentum and cosine decay on learning rate. Warm-up, grouped weight decay strategy and the exponential moving average (EMA) are utilized. Self-distillation and anchor-aided training (AAT) also be used in training. The strong data augmentations we adopt Mosaic [2, 13] and Mixup [57].

We conducted MIM unsupervised pretraining on the backbone using the 1.28 million ImageNet-1K datasets [8]. Following the experiment settings in Spark [45], we employed a LAMB optimizer [54] and cosine-annealing learning rate strategy, with a masking ratio of $60\%$ and a mask patch size of 32. For the Gold-YOLO-L models, we employed a batch size of 1024, while for the Gold-YOLO-M models, a batch size of 1152 was used. MIM pretraining was not employed for Gold-YOLO-N due to the limited capacity of its small backbone.

All our models are trained on 8 NVIDIA A100 GPUs, and the speed performance is measured on an NVIDIA Tesla T4 GPU with TensorRT.

# 4.2 Comparisons

Our focus is primarily on evaluating the speed performance of our models after deployment. Specifically, we measure throughput (frames per second at a batch size of 1 or 32) and GPU latency, rather than FLOPs or the number of parameters. To compare our Gold-YOLO with other state-of-the-art detectors in the YOLO series, such as YOLOv5 [13], YOLOX [11], PPYOLOE [52], YOLOv7 [48], YOLOv8 [14] and YOLOv6-3.0 [31], we test the speed performance of all official models with FP16-precision on the same Tesla T4 GPU with TensorRT.

Gold-YOLO-N demonstrates notable advancements, achieving an improvement of 2.6%/2.4%/6.6% compared to YOLOv8-N, YOLOv6-3.0-N, and YOLOv7-Tiny (input size=416), respectively, while offering comparable or superior performance in terms of throughput and latency. When compared to YOLOX-S and PPYOLOE-S, Gold-YOLO-S demonstrates a notable increase in AP by 5.9%/3.1%, while operating at a faster speed of 50/27 FPS (with a batch size of 32).

Gold-YOLO-M outperforms YOLOv6-3.0-M, YOLOX-M and PPYOLOE-M by achieving 1.1%, 4.2% and 2.1% higher AP with a comparable speed. Additionally, it achieves 5.7% and 0.9% higher AP than YOLOv5-M and YOLOv8-M, respectively, while achieving a higher speed. Gold-YOLO-M outperforms YOLOv7 with a significant improvement of 98FPS (batch size = 32), while maintaining the same AP. Gold-YOLO-L also achieves a higher accuracy compared to YOLOv8-L and YOLOv6-3.0-L, with a noticeable accuracy advantage of 0.4% and 0.5% respectively, while maintaining similar FPS at a batch size of 32.

# 4.3 Ablation study

# 4.3.1 Ablation study on GD structure

To verify the validity of our analysis concerning the FPN and to assess the efficacy of the proposed gather-and-distribute mechanism, we examined each module in GD independently, focusing on AP, number of parameters, and latency on the T4 GPU. The Low-GD predominantly targets small and medium-sized objects, whereas the High-GD primarily detect large-sized objects, and the LAF module bolsters both branches. The experimental results are displayed in Table 2.

Table 2: Ablation study on GD structure. The test model is Gold-YOLO-S on T4 GPU evaluate. 

<table><tr><td>Low-GD</td><td>High-GD</td><td>LAF</td><td>AP</td><td>AP-small</td><td>AP-medium</td><td>AP-large</td><td>FPS (bs=32)</td><td>Params</td><td>FLOPs</td></tr><tr><td>√</td><td></td><td></td><td>44.65%</td><td>25.13%</td><td>49.70%</td><td>60.36%</td><td>454.4</td><td>18.7 M</td><td>45.1 G</td></tr><tr><td></td><td>√</td><td></td><td>42.27%</td><td>20.79%</td><td>47.74%</td><td>61.09%</td><td>493.7</td><td>20.9 M</td><td>43.6 G</td></tr><tr><td></td><td></td><td>√</td><td>44.36%</td><td>25.04%</td><td>49.53%</td><td>60.64%</td><td>526.2</td><td>18.1 M</td><td>43.3 G</td></tr><tr><td>√</td><td>√</td><td></td><td>45.57%</td><td>24.90%</td><td>50.38%</td><td>63.50%</td><td>461.8</td><td>21.5 M</td><td>45.8 G</td></tr><tr><td>√</td><td>√</td><td>√</td><td>46.11%</td><td>25.22%</td><td>51.23%</td><td>63.42%</td><td>446.2</td><td>21.5 M</td><td>46.0 G</td></tr></table>

# 4.3.2 Ablation study on LAF

In this ablation study, we conducted experiments to compare the effects of different module designs within the LAF framework and evaluate the influence of varying model sizes on accuracy. The

results of our study provide evidence to support the assertion that the existing LAF structure is indeed optimal. The difference between model-1 and model-2 is whether LAF uses add or concat, and model-3 increase the model size basis of model-2. The model-4 is based on model-3 but discards LAF. The experimental results are displayed in Table 3.

Table 3: Ablation study on LAF. Use TensorRT 7 on T4 GPU evaluate. 

<table><tr><td>model</td><td>concat</td><td>add</td><td>increase model size</td><td>AP</td><td>AP-small</td><td>AP-medium</td><td>AP-large</td><td>FPS (bs=32)</td><td>Params</td><td>FLOPs</td></tr><tr><td>1</td><td>√</td><td></td><td></td><td>46.11%</td><td>25.22%</td><td>51.23%</td><td>63.42%</td><td>446.2</td><td>21.5 M</td><td>46.0 G</td></tr><tr><td>2</td><td></td><td>√</td><td></td><td>45.43%</td><td>24.98%</td><td>50.66%</td><td>62.10%</td><td>413.2</td><td>20.9 M</td><td>47.5 G</td></tr><tr><td>3</td><td></td><td>√</td><td>√</td><td>46.49%</td><td>26.32%</td><td>51.29%</td><td>63.43%</td><td>356.0</td><td>30.8 M</td><td>54.1 G</td></tr><tr><td>4</td><td></td><td></td><td>√</td><td>46.47%</td><td>25.27%</td><td>50.80%</td><td>64.25%</td><td>373.8</td><td>26.4 M</td><td>52.9 G</td></tr></table>

# 4.3.3 Ablation study on other model and task

The GD mechanism is a general concept and can be applied beyond YOLOs. We have extend GD mechanism to other models and obtain significant improvement.

On Instance Segmentation task, we replace different necks in Mask R-CNN and train/test on the COCO instance datasets. The result as shown in the Table 4.

Table 4: Ablation study on Instance Segmentation Task. 

<table><tr><td>Model</td><td>Neck</td><td>FPS</td><td>Bbox mAP</td><td>Bbox mAP:50</td><td>Segm mAP</td><td>Segm mAP:50</td></tr><tr><td>MaskRCNN-ResNet50</td><td>FPN</td><td>21.6</td><td>38.2%</td><td>58.8%</td><td>34.7%</td><td>55.7%</td></tr><tr><td>MaskRCNN-ResNet50</td><td>AFPN</td><td>19.1</td><td>36.0%</td><td>53.6%</td><td>31.8%</td><td>50.7%</td></tr><tr><td>MaskRCNN-ResNet50</td><td>PAFPN</td><td>20.2</td><td>37.9%</td><td>58.6%</td><td>34.5%</td><td>55.3%</td></tr><tr><td>MaskRCNN-ResNet50</td><td>GD</td><td>18.7</td><td>40.7%</td><td>59.5%</td><td>36.0%</td><td>56.4%</td></tr></table>

On Semantic Segmentation task, we replace different necks in PointRend and train/test on the Cityscapes datasets. The result as shown in the Table 5.

Table 5: Ablation study on Semantic Segmentation Task. 

<table><tr><td>Model</td><td>Neck</td><td>FPS</td><td>mIoU</td><td>mAcc</td><td>aAcc</td></tr><tr><td>PointRend-ResNet50</td><td>FPN</td><td>11.21</td><td>76.47</td><td>84.05</td><td>95.96</td></tr><tr><td>PointRend-ResNet50</td><td>GD</td><td>11.07</td><td>78.54</td><td>85.60</td><td>96.12</td></tr><tr><td>PointRend-ResNet101</td><td>FPN</td><td>8.76</td><td>78.30</td><td>85.71</td><td>96.23</td></tr><tr><td>PointRend-ResNet101</td><td>GD</td><td>7.54</td><td>80.01</td><td>86.15</td><td>96.34</td></tr></table>

Table 6: Performance of GD mechanism on other object detection models. 

<table><tr><td>model</td><td>Neck</td><td>FPS</td><td>AP</td></tr><tr><td>EfficientDet</td><td>BiFPN</td><td>6.0</td><td>34.4</td></tr><tr><td>EfficientDet</td><td>GD</td><td>5.7</td><td>38.8</td></tr></table>

On object detection task, we replace different necks in EfficientDet and train/test on the COCO datasets. The result as shown in the Table 6.

# 5 Conclusion

In this paper, we revisit the traditional Feature Pyramid Network (FPN) architecture and critically analyze its constraints in terms of information transmission. Following this, we subsequently developed the Gold-YOLO series models for object detection tasks, achieving state-of-the-art results. In Gold-YOLO we introduce an innovative gather-and-distribute mechanism, strategically designed to enhance the efficacy and efficiency of information fusion and transmission, avoid unnecessary losses, thereby significantly improving the model's detection capabilities. We truly hope that our work will prove valuable in addressing real-world problems and may also ignite fresh ideas for researchers in this field.

# Acknowledgement

We gratefully acknowledge the support of MindSpore, CANN (Compute Architecture for Neural Networks) and Ascend AI Processor used for this research.

# References

[1] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers, 2022.   
[2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020.   
[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 213–229. Springer, 2020.   
[4] Ping-Yang Chen, Ming-Ching Chang, Jun-Wei Hsieh, and Yong-Sheng Chen. Parallel residual bi-fusion feature pyramid network for accurate single-shot object detection. IEEE Transactions on Image Processing, 30:9099–9111, 2021.   
[5] Ping-Yang Chen, Jun-Wei Hsieh, Chien-Yao Wang, and Hong-Yuan Mark Liao. Recursive hybrid fusion pyramid network for real-time small object detection on embedded devices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 402–403, 2020.   
[6] Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, and Jian Sun. You only look one-level feature. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13039–13048, 2021.   
[7] Yukang Chen, Tong Yang, Xiangyu Zhang, Gaofeng Meng, Xinyu Xiao, and Jian Sun. Detnas: Backbone search for object detection. Advances in Neural Information Processing Systems, 32, 2019.   
[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.   
[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
[10] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6569–6578, 2019.   
[11] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.   
[12] Golnaz Ghiasi, Tsung-Yi Lin, and Quoc V Le. Nas-fpn: Learning scalable feature pyramid architecture for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7036-7045, 2019.   
[13] Jocher Glenn. Yolov5 release v6.1. https://github.com/ultralytics/yolov5/releases/tag/v6.1, 2022.   
[14] Jocher Glenn. Ultralytics yolov8. https://github.com/ultralytics/ultralytics, 2023.   
[15] Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze. Levit: a vision transformer in convnet's clothing for faster inference. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12259–12269, 2021.   
[16] Jianyuan Guo, Kai Han, Yunhe Wang, Han Wu, Xinghao Chen, Chunjing Xu, and Chang Xu. Distilling object detectors via decoupled features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2154–2164, 2021.

[17] Jianyuan Guo, Kai Han, Yunhe Wang, Chao Zhang, Zhaohui Yang, Han Wu, Xinghao Chen, and Chang Xu. Hit-detector: Hierarchical trinity architecture search for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11405–11414, 2020.   
[18] Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. Cmt: Convolutional neural networks meet vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12175–12185, 2022.   
[19] Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Xinghao Chen, Chunjing Xu, Chang Xu, and Yunhe Wang. Positive-unlabeled data purification in the wild for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2653–2662, 2021.   
[20] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. IEEE transactions on pattern analysis and machine intelligence, 45(1):87–110, 2022.   
[21] Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. Ghostnet: More features from cheap operations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1580–1589, 2020.   
[22] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. Advances in Neural Information Processing Systems, 34:15908–15919, 2021.   
[23] Zhiwei Hao, Jianyuan Guo, Ding Jia, Kai Han, Yehui Tang, Chao Zhang, Han Hu, and Yunhe Wang. Learning efficient vision transformers via fine-grained manifold distillation. In Advances in Neural Information Processing Systems, 2022.   
[24] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners, 2021.   
[25] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.   
[26] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
[27] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.   
[28] Zilong Huang, Youcheng Ben, Guozhong Luo, Pei Cheng, Gang Yu, and Bin Fu. Shuffle transformer: Rethinking spatial shuffle for vision transformer. arXiv preprint arXiv:2106.03650, 2021.   
[29] Zhenchao Jin, Bin Liu, Qi Chu, and Nenghai Yu. Safnet: A semi-anchor-free network with enhanced feature pyramid for object detection. IEEE Transactions on Image Processing, 29:9445–9457, 2020.   
[30] Zhenchao Jin, Dongdong Yu, Luchuan Song, Zehuan Yuan, and Lequan Yu. You should look at all objects. In European Conference on Computer Vision, pages 332–349. Springer, 2022.   
[31] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangxiang Chu. Yolov6 v3. 0: A full-scale reloading. arXiv preprint arXiv:2301.05586, 2023.   
[32] Chuyi Li, Lulu Li, Hongliang Jiang, Kaiheng Weng, Yifei Geng, Liang Li, Zaidan Ke, Qingyuan Li, Meng Cheng, Weiqiang Nie, et al. Yolov6: A single-stage object detection framework for industrial applications. arXiv preprint arXiv:2209.02976, 2022.   
[33] Xiangtai Li, Ansheng You, Zhen Zhu, Houlong Zhao, Maoke Yang, Kuiyuan Yang, Shaohua Tan, and Yunhai Tong. Semantic flow for fast and accurate scene parsing. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 775–793. Springer, 2020.   
[34] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.

[35] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.   
[36] Wenyu Lv, Shangliang Xu, Yian Zhao, Guanzhong Wang, Jinman Wei, Cheng Cui, Yuning Du, Qingqing Dang, and Yi Liu. Detrs beat yolos on real-time object detection. arXiv preprint arXiv:2304.08069, 2023.   
[37] Y Quan, D Zhang, L Zhang, and J Tang. Centralized feature pyramid for object detection. arxiv 2022. arXiv preprint arXiv:2210.02093, 41.   
[38] Harish Guruprasad Ramaswamy et al. Ablation-cam: Visual explanations for deep convolutional network via gradient-free localization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 983–991, 2020.   
[39] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016.   
[40] Joseph Redmon and Ali Farhadi. Yolo9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7263-7271, 2017.   
[41] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.   
[42] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.   
[43] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.   
[44] Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781-10790, 2020.   
[45] Keyu Tian, Yi Jiang, Qishuai Diao, Chen Lin, Liwei Wang, and Zehuan Yuan. Designing bert for convolutional networks: Sparse and hierarchical masked modeling. arXiv preprint arXiv:2301.03580, 2023.   
[46] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9627-9636, 2019.   
[47] Qiang Wan, Zilong Huang, Jiachen Lu, Gang Yu, and Li Zhang. Seaformer: Squeeze-enhanced axial transformer for mobile semantic segmentation. arXiv preprint arXiv:2301.13156, 2023.   
[48] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022.   
[49] Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, and Jiashi Feng. Panet: Few-shot image semantic segmentation with prototype alignment. In proceedings of the IEEE/CVF international conference on computer vision, pages 9197–9206, 2019.   
[50] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders, 2023.   
[51] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling, 2022.   
[52] Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang, Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An evolved version of yolo. arXiv preprint arXiv:2203.16250, 2022.   
[53] Guoyu Yang, Jie Lei, Zhikuan Zhu, Siyu Cheng, Zunlei Feng, and Ronghua Liang. Afpn: Asymptotic feature pyramid network for object detection. arXiv preprint arXiv:2306.15988, 2023.

[54] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes, 2020.   
[55] Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, and Wei Wu. Incorporating convolution designs into visual transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 579–588, 2021.   
[56] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.   
[57] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.   
[58] Wenqiang Zhang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu, and Chunhua Shen. Topformer: Token pyramid transformer for mobile semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12083–12093, 2022.   
[59] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6848–6856, 2018.   
[60] Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang, Ying Chen, Ling Cai, and Haibin Ling. M2det: A single-shot object detector based on multi-level feature pyramid network. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 9259–9266, 2019.   
[61] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

# A Additional experiment

# A.1 More detailed accuracy and speed data for Gold-YOLO

In this section, we report the test performance of our Gold-YOLO with or without LAF module and pre-training. FPS and latency are measured in FP16-precision on a Tesla T4 in the same environment with TensorRT 7. Both the accuracy and the speed performance of our models are evaluated with the input resolution of 640x640. The result shown in Table 7.

Table 7: Test results of Gold-YOLO series model on COCO 2017 val. ‘†’ represents that the self-distillation method is utilized, ‘◇’ represents that the model don’t have LAF module, and ‘★’ represents that the MIM pre-training method is utilized. 

<table><tr><td>Method</td><td> $AP^{val}$ </td><td> $AP^{val}_{50}$ </td><td> $AP^{val}_{small}$ </td><td> $AP^{val}_{medium}$ </td><td> $AP^{val}_{large}$ </td><td>FPS $(bs=32)$ </td><td>Params</td><td>FLOPs</td></tr><tr><td>Gold-YOLO- $N^{\diamond}$ </td><td>38.37% / 38.82%†</td><td>54.43% / 54.96%†</td><td>18.37% / 18.40%†</td><td>42.62% / 43.45%†</td><td>55.13% / 56.00%†</td><td>1087</td><td>5.6 M</td><td>12.0 G</td></tr><tr><td>Gold-YOLO- $S^{\diamond}$ </td><td>44.47% / 45.57%†</td><td>61.55% / 62.92%†</td><td>24.04% / 24.90%†</td><td>49.34% / 50.38%†</td><td>61.95% / 63.50%†</td><td>462</td><td>21.5 M</td><td>45.8 G</td></tr><tr><td>Gold-YOLO- $M^{\diamond}$ </td><td>49.41% / 50.26%†</td><td>66.64% / 67.58%†</td><td>31.19% / 31.59%†</td><td>54.30% / 55.26%†</td><td>65.91% / 67.62%†</td><td>229</td><td>41.3 M</td><td>86.8 G</td></tr><tr><td>Gold-YOLO- $L^{\diamond}$ </td><td>51.68% / 52.65%†</td><td>69.07% / 70.25%†</td><td>34.86% / 34.20%†</td><td>56.92% / 57.70%†</td><td>69.00% / 69.71%†</td><td>119</td><td>75.0 M</td><td>150.6 G</td></tr><tr><td>Gold-YOLO-N</td><td>39.57% / 39.92%†</td><td>55.70% / 55.94%†</td><td>19.67% / 19.15%†</td><td>44.08% / 44.32%†</td><td>56.98% / 57.75%†</td><td>1030</td><td>5.6 M</td><td>12.1 G</td></tr><tr><td>Gold-YOLO-S</td><td>45.36% / 46.11%†</td><td>62.48% / 63.33%†</td><td>25.32% / 25.22%†</td><td>50.21% / 51.23%†</td><td>62.63% / 63.42%†</td><td>446</td><td>21.5 M</td><td>46.0 G</td></tr><tr><td>Gold-YOLO-M</td><td>49.77% / 50.86%†</td><td>67.01% / 68.23%†</td><td>32.32% / 31.01%†</td><td>55.29% / 56.24%†</td><td>66.27% / 67.83%†</td><td>220</td><td>41.3 M</td><td>87.5 G</td></tr><tr><td>Gold-YOLO-L</td><td>51.84% / 53.16%†</td><td>68.94% / 70.49%†</td><td>34.12% / 34.53%†</td><td>57.36% / 58.60%†</td><td>68.17% / 70.07%†</td><td>116</td><td>75.1 M</td><td>151.7 G</td></tr><tr><td>Gold-YOLO- $S^*$ </td><td>45.52% / 46.36%†</td><td>62.20% / 63.36%†</td><td>24.66% / 25.26%†</td><td>50.76% / 51.30%†</td><td>63.24% / 63.64%†</td><td>446</td><td>21.5 M</td><td>46.0 G</td></tr><tr><td>Gold-YOLO- $M^*$ </td><td>50.16% / 51.14%†</td><td>67.52% / 68.53%†</td><td>30.52% / 32.33%†</td><td>55.54% / 56.10%†</td><td>67.64% / 68.55%†</td><td>220</td><td>41.3 M</td><td>87.5 G</td></tr><tr><td>Gold-YOLO- $L^*$ </td><td>52.25% / 53.28%†</td><td>69.61% / 70.93%†</td><td>33.09% / 33.83%†</td><td>57.77% / 58.92%†</td><td>69.01% / 69.92%†</td><td>116</td><td>75.1 M</td><td>151.7 G</td></tr></table>

# A.2 MIM pre-training ablation experiment

We also compared the Gold-YOLO-S on COCO 2017 validation results for different MIM pre-training epochs without self-distillation. The result shown in Table 8.

Table 8: Test results on COCO 2017 val for different pre-training epoch setting. 

<table><tr><td>Epoch</td><td> $\mathbf{AP}^{val}$ </td><td> $\mathbf{AP}^{val}_{50}$ </td><td> $\mathbf{AP}^{val}_{small}$ </td><td> $\mathbf{AP}^{val}_{medium}$ </td><td> $\mathbf{AP}^{val}_{large}$ </td></tr><tr><td>400</td><td>45.39%</td><td>62.17%</td><td>25.01%</td><td>50.28%</td><td>62.74%</td></tr><tr><td>600</td><td>45.48%</td><td>62.18%</td><td>25.56%</td><td>50.63%</td><td>62.85%</td></tr><tr><td>800</td><td>45.52%</td><td>62.20%</td><td>24.66%</td><td>50.76%</td><td>63.24%</td></tr></table>

# B Comprehensive Latency and Throughput Benchmark

# B.1 Model Latency and Throughput on T4 GPU with TensorRT 8

Comparisons with other YOLO-series detectors on COCO 2017 val. FPS and latency are measured in FP16-precision on Tesla T4 in the same environment with TensorRT 8.2. The result shown in Table 9.

Table 9: Comparison of Latency and Throughput in YOLO series model on a T4 GPU using TensorRT 8.2. 

<table><tr><td>Method</td><td>Input Size</td><td>FPS (bs=1)</td><td>FPS (bs=32)</td><td>Latency (bs=1)</td></tr><tr><td>YOLOv5-N</td><td>640</td><td>702</td><td>843</td><td>1.4 ms</td></tr><tr><td>YOLOv5-S</td><td>640</td><td>433</td><td>515</td><td>2.3 ms</td></tr><tr><td>YOLOv5-M</td><td>640</td><td>202</td><td>235</td><td>4.9 ms</td></tr><tr><td>YOLOv5-L</td><td>640</td><td>126</td><td>137</td><td>7.9 ms</td></tr><tr><td>YOLOX-Tiny</td><td>416</td><td>766</td><td>1393</td><td>1.3 ms</td></tr><tr><td>YOLOX-S</td><td>640</td><td>313</td><td>489</td><td>2.6 ms</td></tr><tr><td>YOLOX-M</td><td>640</td><td>159</td><td>204</td><td>5.3 ms</td></tr><tr><td>YOLOX-L</td><td>640</td><td>104</td><td>117</td><td>9.0 ms</td></tr><tr><td>PPYOLOE-S</td><td>640</td><td>357</td><td>493</td><td>2.8 ms</td></tr><tr><td>PPYOLOE-M</td><td>640</td><td>163</td><td>210</td><td>6.1 ms</td></tr><tr><td>PPYOLOE-L</td><td>640</td><td>110</td><td>145</td><td>9.1 ms</td></tr><tr><td>YOLOv7-Tiny</td><td>640</td><td>464</td><td>568</td><td>2.1 ms</td></tr><tr><td>YOLOv7</td><td>640</td><td>128</td><td>135</td><td>7.6 ms</td></tr><tr><td>YOLOv6-3.0-N</td><td>640</td><td>785</td><td>1215</td><td>1.3 m s</td></tr><tr><td>YOLOv6-3.0-S</td><td>640</td><td>345</td><td>498</td><td>2.9 ms</td></tr><tr><td>YOLOv6-3.0-M</td><td>640</td><td>178</td><td>238</td><td>5.6 ms</td></tr><tr><td>YOLOv6-3.0-L</td><td>640</td><td>105</td><td>125</td><td>9.5 ms</td></tr><tr><td>Gold-YOLO-N</td><td>640</td><td>657</td><td>1191</td><td>1.4 ms</td></tr><tr><td>Gold-YOLO-S</td><td>640</td><td>308</td><td>492</td><td>3.1 ms</td></tr><tr><td>Gold-YOLO-M</td><td>640</td><td>157</td><td>241</td><td>6.1 ms</td></tr><tr><td>Gold-YOLO-L</td><td>640</td><td>94</td><td>137</td><td>10.3 ms</td></tr></table>

# B.2 Model Latency and Throughput on V100 GPU with TensorRT 7

Comparisons with other YOLO-series detectors on COCO 2017 val. FPS and latency are measured in FP16-precision on Tesla V100 in the same environment with TensorRT 7.2. The result shown in Table 10.

Table 10: Comparison of Latency and Throughput in YOLO series model on a V100 GPU using TensorRT 7.2. 

<table><tr><td>Method</td><td>Input Size</td><td>FPS (bs=1)</td><td>FPS (bs=32)</td><td>Latency (bs=1)</td></tr><tr><td>YOLOv5-N</td><td>640</td><td>577</td><td>1727</td><td>1.4 ms</td></tr><tr><td>YOLOv5-S</td><td>640</td><td>449</td><td>1249</td><td>1.7 ms</td></tr><tr><td>YOLOv5-M</td><td>640</td><td>271</td><td>698</td><td>3.0 ms</td></tr><tr><td>YOLOv5-L</td><td>640</td><td>178</td><td>440</td><td>4.7 ms</td></tr><tr><td>YOLOX-Tiny</td><td>416</td><td>569</td><td>2883</td><td>1.4 ms</td></tr><tr><td>YOLOX-S</td><td>640</td><td>386</td><td>1206</td><td>2.0 ms</td></tr><tr><td>YOLOX-M</td><td>640</td><td>245</td><td>600</td><td>3.4 ms</td></tr><tr><td>YOLOX-L</td><td>640</td><td>149</td><td>361</td><td>5.6 ms</td></tr><tr><td>PPYOLOE-S</td><td>640</td><td>322</td><td>1050</td><td>2.4 ms</td></tr><tr><td>PPYOLOE-M</td><td>640</td><td>222</td><td>566</td><td>4.0 ms</td></tr><tr><td>PPYOLOE-L</td><td>640</td><td>153</td><td>406</td><td>5.5 ms</td></tr><tr><td>YOLOv7-Tiny</td><td>640</td><td>453</td><td>1565</td><td>1.7 ms</td></tr><tr><td>YOLOv7</td><td>640</td><td>182</td><td>412</td><td>4.6 ms</td></tr><tr><td>YOLOv6-3.0-N</td><td>640</td><td>646</td><td>2660</td><td>1.2 m s</td></tr><tr><td>YOLOv6-3.0-S</td><td>640</td><td>399</td><td>1330</td><td>2.0 ms</td></tr><tr><td>YOLOv6-3.0-M</td><td>640</td><td>203</td><td>676</td><td>4.4 ms</td></tr><tr><td>YOLOv6-3.0-L</td><td>640</td><td>125</td><td>385</td><td>6.8 ms</td></tr><tr><td>Gold-YOLO-N</td><td>640</td><td>574</td><td>2457</td><td>1.7 ms</td></tr><tr><td>Gold-YOLO-S</td><td>640</td><td>391</td><td>1205</td><td>2.5 ms</td></tr><tr><td>Gold-YOLO-M</td><td>640</td><td>238</td><td>633</td><td>4.0 ms</td></tr><tr><td>Gold-YOLO-L</td><td>640</td><td>146</td><td>365</td><td>6.6 ms</td></tr></table>

# C Broader impacts and limitations

Broader impacts. The YOLO model can be widely applied in fields such as healthcare and intelligent transportation. In the healthcare domain, the YOLO series models can improve the early diagnosis rates of certain diseases and reduce the cost of initial diagnosis, thereby saving more lives. In the field of intelligent transportation, the YOLO model can assist in autonomous driving of vehicles, enhancing traffic safety and efficiency. However, there are also risks associated with the military application of the YOLO model, such as target recognition for drones and assisting military reconnaissance. We will make every effort to prevent the use of our model for military purposes.

Limitations. Generally, making finer adjustments on the structure will help further improve the model's performance, but this requires a significant amount of computational resources. Additionally, due to our algorithm's heavy usage of attention operations, it may not be as friendly to some earlier hardware support.

# D CAM visualization

Below are the CAM visualization results of the neck for YOLOv5, YOLOv6, YOLOv7, YOLOv8, and our Gold-YOLO, shown in Fig. 6. It can be observed that our model assigns higher weights to the detected regions of the targets.

And we compare the neck CAM visualization between Gold-YOLO and YOLOv6, shown in Fig. 7.

![](images/1f329a098d52623971e20897b9c4b8a850ef1d237ec21fb1ae0df585954925f3.jpg)

<details>
<summary>text_image</summary>

COCO
YOLOv5-N
YOLOv6-N
YOLOv7-T
YOLOv8-N
Gold-YOLO-N
</details>

Figure 6: The CAM visualization results of the neck for YOLOv5, YOLOv6, YOLOv7, YOLOv8, and our Gold-YOLO.

![](images/d68eb4b293a4db26f4e3cf08a90d35755c26441bb719e1ab67aed9812a84b8e4.jpg)

<details>
<summary>text_image</summary>

GD-Neck
FPN-Neck
P4
P3
N3
N4
</details>

Figure 7: Neck CAM visualization. We can observe that features at different levels exhibit distinct preferences for objects of different sizes. In the traditional grid-structured FPN, with increasing network depth and information interaction between different levels, the sensitivity of the feature map to object positions gradually diminishes, accompanied by information loss. Our proposed GD mechanism performs global fusion separately for high-level and low-level information, resulting in globally fused features that contain abundant position information for objects of various sizes.

# E Discussion

# E.1 The differences between feature alignment module of Gold-YOLO and other similar works.

Both M2Det and RHF-Net have incorporated additional information fusion modules within their alignment modules. In M2Det, the SFAM module includes an SE block, while in RHF-Net, the Spatial Pyramid Pooling block is augmented with a Bottleneck layer. In contrast to M2Det and RHF-Net, Gold-YOLO leans towards functional separation among modules, segregating feature alignment and feature fusion into distinct modules. Specifically, the FAM module in GoldD-YOLO focuses solely on feature alignment. This ensures computational efficiency within the FAM module. And the LAF module efficiently merges features from various levels with minimal computational cost, leaving a greater portion of fusion and injection functionalities to other modules.

Based on the GD mechanism, achieving SOTA performance for the YOLO model can be accomplished using simple and easily accessible operators. This strongly demonstrates the effectiveness of the approach we propose. Additionally, during the network construction process, we intentionally opted for simple and thoroughly validated structures. This choice serves to prevent potential development and performance issues arising from certain operators being unsupported by deployment devices. As a result, it guarantees the usability and portability of the entire mechanism.

# E.2 Simple calculation operation

In the process of network construction, we have drawn from and built upon the experiences of previous researchers. Rather than focusing solely on performance improvements achieved by enhancing specific operators or local structures, our emphasis lies on the conceptual shift brought about by the GD mechanism in comparison to the traditional FPN structure. Through the GD mechanism, achieving SOTA performance for the YOLO model is attainable using simple and easily applicable operators. This serves as strong evidence for the effectiveness of the proposed approach.

Additionally, during the network construction process, we intentionally opted for simple and thoroughly validated structures. This choice serves to prevent potential development and performance issues arising from certain operators being unsupported by deployment devices. As a result, it guarantees the usability and portability of the entire mechanism. Moreover, this decision also creates opportunities for future performance enhancements.

The GD mechanism is a general concept and can be applied beyond YOLOs. We have extend GD mechanism to other models and obtain significant improvement. The experiments demonstrate

that our proposed GD mechanism exhibits robust adaptability and generalization. This mechanism consistently brings about performance improvements across different tasks and models.