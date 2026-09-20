# Beyond Cropped Regions: New Benchmark and Corresponding Baseline for Chinese Scene Text Retrieval in Diverse Layouts

Gengluo Li $^{12}$ Huawen Shen $^{12}$ Yu Zhou $^{3}$

# Abstract

Chinese scene text retrieval is a practical task that aims to search for images containing visual instances of a Chinese query text. This task is extremely challenging because Chinese text often features complex and diverse layouts in real-world scenes. Current efforts tend to inherit the solution for English scene text retrieval, failing to achieve satisfactory performance. In this paper, we establish a Diversified Layout benchmark for Chinese Street View Text Retrieval (DL-CSVTR), which is specifically designed to evaluate retrieval performance across various text layouts, including vertical, cross-line, and partial alignments. To address the limitations in existing methods, we propose Chinese Scene Text Retrieval CLIP (CSTR-CLIP), a novel model that integrates global visual information with multi-granularity alignment training. CSTR-CLIP applies a two-stage training process to overcome previous limitations, such as the exclusion of visual features outside the text region and reliance on single-granularity alignment, thereby enabling the model to effectively handle diverse text layouts. Experiments on existing benchmark show that CSTR-CLIP outperforms the previous state-of-the-art model by 18.82% accuracy and also provides faster inference speed. Further analysis on DL-CSVTR confirms the superior performance of CSTR-CLIP in handling various text layouts. The dataset and code will be publicly available to facilitate research in Chinese scene text retrieval.

$^{1}$ Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China $^{2}$ School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China $^{3}$ VCIP & TMCC & DISSec, College of Computer Science, Nankai University, Tianjin, China. Correspondence to: Yu Zhou <yzhou@nankai.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](images/d7fce8efcf0b483ecc6d69a0e56bc2cf966175d44335250c398bb18775ca24cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text Detector"] --> B["BANK BANK"]
    B --> C["Visual Embedding"]
    C --> D["BANK"]
    E["Text Detector"] --> F["封顶大吉 大吉"]
    F --> G["Visual Embedding"]
    G --> H["封顶大吉"]
    I["Text Detector"] --> J["Vertical"]
    J --> K["Cross-Line"]
    K --> L["Partial"]
    M["Text Embedding"] --> N["Visual Embedding"]
    N --> O["Ours"]
    P["&quot;Text Embedding"] --> Q["封顶大吉"]
    R["&quot;Visual Embedding"] --> S["Ours"]
```
</details>

Figure 1. Pipeline comparison between (a) English scene text retrieval, (b) Chinese scene text retrieval adopted in previous work, and (c) Our proposed Chinese scene text retrieval framework that is based on full image information and multi-granularity alignment. The patterns in the circles represent the text layout forms.

# 1. Introduction

Text is an important object in scene images, and scene text related research topics including detection (Wang et al., 2022; Cao et al., 2025), recognition (Yang et al., 2025; Zhang et al., 2025), spotting (Lyu et al., 2025a; Wei et al., 2022), understanding (Shen et al., 2025; Zeng et al., 2023) and processing (Zeng et al., 2024b; Shu et al., 2025) have drawn increasing attention in recent years. Among them, scene text retrieval is an important topic of information retrieval, which involves searching for images that contain visual instances of a given query text within a collection of natural images (Mishra et al., 2013; Mafla et al., 2020). As texts in images generally convey valuable information, this task has been widely used in many applications, such as multimedia content retrieval, product recommendation, and automatic navigation (Karaoglu et al., 2016; Bai et al., 2018; Song et al., 2019).

In recent years, scene text retrieval has witnessed tremendous progress, while most of the existing methods focus on the language patterns of English. Whether these methods can be transferred to other language scenarios (especially

non-latin scripts) has not been fully explored. More specifically, as one of the most widely used non-latin languages, Chinese scene text retrieval differs distinctly from English scene text retrieval. As shown in Figure 1(a), in English setting, the query term is typically a single word, and text in images exhibits clear separations. As such, English scene text retrieval is essentially a simple local matching problem, which can be solved by measuring the similarity between the detected region and the query word. In contrast, in Chinese setting, there are no separations among words of the same sentences, the query terms can be any combination of consecutive characters, and Chinese has more highly variable layouts in real scenes. As shown in Figure 1(b), detection results often do not completely match the query words. Therefore, it is difficult to adapt the scene text retrieval pipeline from English to Chinese straightforwardly.

Targeting at the Chinese scene text retrieval task, Wang et al. (Wang et al., 2021) establish the CSVTR benchmark, and several studies have demonstrated promising performance on this benchmark. However, it can not well reflect the retrieval capabilities of models in real world. The query terms in this benchmark predominantly appear independently and are primarily horizontally oriented, neglecting the characteristics of diverse text layouts in Chinese scene text retrieval. In addition, most existing Chinese scene text retrieval models adopt paradigms developed for English scenes, utilizing cropped text regions from text detection results to represent the visual information of images, which brings several limitations. On the one hand, this method inevitably results in the loss of context, that is, the global semantic information can not contribute to the retrieval process, and parts of the query term may be lost. On the other hand, since redundant elements may be included in the detected boxes, the model is easily disturbed by irrelevant content. Moreover, previous methods strictly align the features of a text region with all the text it contains during training. This single-granularity alignment manner will undoubtedly cause some ambiguity, impairing models' ability to perceive and distinguish text. In Chinese scene text, the diverse combinations and layout forms of characters present significant challenges to previous solutions based on cropped regions. Therefore, enhancing the ability of scene text retrieval models to handle varied layouts is crucial for advancing scene text retrieval.

Since previous studies have not addressed the specific challenges of Chinese text retrieval, we propose a new benchmark and a scene text retrieval model that extends beyond cropped regions. As a first contribution, we introduce DLCSVTR, a new Diversified Layout benchmark for Chinese Street View Text Retrieval. We collect scene images containing visual instances of query terms in three of the most common text layouts, in addition to the horizontal layout, including (i) vertical layouts where the query terms are arranged vertically, (ii) cross-line layouts where query terms span rows or columns, and (iii) partial layouts where query terms are connected to other text. DL-CSVTR combines diverse layouts to evaluate the ability of models to handle various text instances, simulating the requirements in real Chinese scene text retrieval applications.

Following it, we are committed to enhancing model performance on diverse text layouts by freeing the model from the limitations imposed by cropped regions on its visual perception range, which we believe is key to addressing the challenges of these layouts. To this end, we take advantages of the Contrastive Language-Image Pre-training (CLIP) model to retain full-image information, and convert the problem from text regions matching to global semantics enhanced layout patterns learning, as shown in Figure 1(c). Specifically, our proposed Chinese Scene Text Retrieval CLIP model adopts a two-stage training paradigm. The goal of the first stage is to teach CLIP to focus on specific areas and enhance its OCR capabilities. During it, triplet inputs including the original image, the text segmentation map and the corresponding text are constructed, which are employed to train CLIP with a single-granularity alignment method. In the second stage, the model goes further into multi-granularity alignment, so as to learn the ability of perceiving diverse layouts. We apply the random granularity alignment algorithm to process the segmentation map and text in the triplets, producing many layout patterns for the model to learn. Meanwhile, global image features derived from the original CLIP model are injected in multi-scale layers of CSTR-CLIP, enabling the model to perform retrieval under the guidance of context.

Experiments on the previous Chinese scene text retrieval benchmark CSVTR show that CSTR-CLIP improves retrieval accuracy by 18.82% over the best previous method, while also delivering faster inference speed. Furthermore, both quantitative and qualitative experiments on DLCSVTR confirm the effectiveness and generalization ability of CSTR-CLIP in handling diverse text layouts.

In summary, the contributions of this paper include:

- We conduct a thorough analysis of previous benchmark datasets and methods for Chinese scene text retrieval, identifying a significant gap in their ability to handle diverse text layouts. To address it, we propose a new benchmark, DL-CSVTR, which includes Chinese text image data for various layouts, to evaluate models' retrieval capabilities in real scenarios.   
- We introduce CSTR-CLIP, a novel paradigm for Chinese scene text retrieval. CSTR-CLIP moves beyond the previous approach of text regions cropping, expands the model's visual receptive field by retaining full-image information, and enhances perception flexibility through multi-granularity alignment.

![](images/3b38165b2eab08dcee2be601e0642f0d324581010274755d7b6f3c90df365c55.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
|---|---|
| Horizontal | 92.62 |
| Vertical | 3.53 |
| Cross-Line | 1.07 |
| Partial | 2.75 |
| Text bbox containing the query | Not labeled (implied by legend) |
</details>

Figure 2. Layout distribution of the visual instance of the query word in the image in CSVTR.

\- Our experiments demonstrate that the proposed method not only achieves state-of-the-art performance on the existing CSVTR dataset but also surpasses previous models in retrieval capabilities on the newly introduced DL-CSVTR benchmark.

# 2. Related Work

Scene Text Retrieval Benchmark. Mishra et al. (Mishra et al., 2013) is the first work to introduce the scene text retrieval task with the IIIT Scene Text Retrieval benchmark, which comprises numerous scene images and predefined query terms linked to specific images. The model accuracy is evaluated using Mean Average Precision (mAP), and speed is measured with Frames Per Second (FPS). Later researchers enhance this by using the Street View Text and TotalText datasets, creating richer benchmarks with text annotations as query terms (Wang & Belongie, 2010; Ch'ng & Chan, 2017). To examine the retrieval problem in other language setting, Wang et al. (Wang et al., 2021) introduce the CSVTR dataset for Chinese scene text retrieval, where query terms typically appear independently and are arranged horizontally, as shown in Figure 8. It ignores the challenges that the diverse text layout of Chinese brings to the scene text retrieval task.

Scene Text Retrieval. For scene text retrieval, a straightforward solution is to use text detection (Shu et al., 2023) and recognition(Qiao et al., 2020b; 2021) or end-to-end spotting (Lyu et al., 2025b) models to extract characters or words from images, simplifying the problem to traditional text retrieval (Mishra et al., 2013; Jaderberg et al., 2016; Liu et al., 2020; Qiao et al., 2020a; Liao et al., 2020). However, this approach often leads to error accumulation and limited performance (Huang et al., 2024). Some studies design manual text embeddings to convert characters into vector representations for improved retrieval robustness (Almazán et al., 2014; Ghosh et al., 2015; Ghosh & Valveny,

2015; Wilkinson & Brun, 2016; Gómez et al., 2018; Mafla et al., 2021), while they still suffer from the suboptimality of manually designed embeddings (Zhou et al., 2022b). Subsequently, cross-modal embeddings are proposed to unify visual and textual representations within a common feature space, achieving strong retrieval performance and speed (Gómez et al., 2017; Mhiri et al., 2019; Wang et al., 2021; Zeng et al., 2024a). Recently, visual embedding methods propose to transform query words into visual representations for retrieval via visual matching (Wen et al., 2023; Luo et al., 2024), achieving remarkable performance but at the cost of slower computational speeds. All these approaches rely on text detection to crop text regions for feature extraction. Although cropping enhances the visual features of the text region, the loss of semantic information outside this region has limited further improvements in scene text retrieval.

CLIP's Attention Guidance and OCR Capability. CLIP (Radford et al., 2021) is a vision-language model trained on large-scale data, possessing powerful representation capabilities (Materzyńska et al., 2022; Yu et al., 2023). Moreover, researchers find that many high-similarity image-text pairs in the LAION-2B dataset (Schuhmann et al., 2022) include visual instances of captions within the images (Lin et al., 2024) and synthetic text images can be utilized as visual prompts to enhance image classification performance (Li et al., 2022; Shi & Yang, 2023). These studies suggest that CLIP possesses potential OCR functionality and is well-suited for retrieval tasks (Luo et al., 2022; Baldrati et al., 2023; Saito et al., 2023). However, CLIP's perception often involves all image elements, resulting in a bias towards prominent objects (Xing et al., 2023). To enhance region awareness, methods like ReCLIP (Subramanian et al., 2022) crop images using bounding boxes, though this can lead to the loss of visual information. Red-Circle (Shtedritski et al., 2023) adds contours to guide CLIP's attention, while MaskCLIP (Zhou et al., 2022a) and AlphaClip (Sun et al., 2024) use masks to focus on specific regions, guiding CLIP to focus on target areas while preserving global context. Inspired by these works, our approach leverages CLIP to extract full-image information and facilitate scene text retrieval with CLIP's intrinsic text knowledge. Additionally, text location information is exploited to guide the model's perception area.

# 3. DL-CSVTR Benchmark

Based on our findings, we propose a Diversified Layout benchmark for Chinese Street View Text Retrieval (DLCSVTR), which considers the diverse text layouts of query terms' visual instances in Chinese scene text retrieval. Specifically, it includes data from three common scene text layout types, where the visual representation of query terms strictly adheres to their respective layouts: vertical, cross-

![](images/821cb582550c1ad126f1850625622f28fb7617737218c0b527ee00e05b9017f4.jpg)

<details>
<summary>text_image</summary>

(a) CSVTR
Horizontal
(b) DL-CSVTR-V
Vertical
(c) DL-CSVTR-CL
Cross-Line
(d) DL-CSVTR-P
Partial
Data Source & Corresponding Text Layout
Example
7天连锁酒店
便捷换乘
封顶
&
大青
彭城广场
Visual instances of query
“7天连锁酒店”
“便捷换乘”
“封顶大吉”
“广场”
Text query
1667 / 23
370 / 31
450 / 33
1250 / 25
Number of images / queries
</details>

Figure 3. Common Chinese text layouts, where the visual representation of query terms is based on the text detection model's cropped results. In category (d), the red-highlighted area indicates the corresponding query term.

line, and partial, as shown in Figure 3.

To ensure the quality of annotations and minimize the impact of individual annotator bias, we adopt the strategy of simultaneous data collection by three annotators. First, we define 89 query terms for the three layouts. These query terms are primarily common conceptual nouns, including but not limited to trademark names, building names, and idioms. The details of the specific query word settings can be found in the supplementary material. Subsequently, the three annotators use these predefined query terms to search for images containing the specified text layouts of the query terms' visual instances. The results from all three annotators are then combined. After combining the results, the annotators screen the image set to remove duplicates and images where the query terms' visual representation did not match the specified text layouts, ensuring the uniqueness of the text layout for each query term in the images. The specific descriptions of each text layout are as follows:

Vertical. Vertical text is common in real-world scenes, characterized by text arranged from top to bottom, as shown in Figure 3(b). We collect 370 street view images containing vertically arranged text with 31 unique query terms. All these queries are visually represented in a vertical layout within the images.

Cross-Line. Cross-line text layouts occur when a conceptual noun spans multiple rows or columns, as shown in Figure 3(c). We collect 450 street view images featuring cross-line text layouts with 33 unique query terms. These queries are visually represented in either cross-row or cross-column layouts within the scene images.

Partial. Unlike English words, Chinese characters typically do not have distinct separations. When query terms appear in a long sentence, text line detection often includes characters beyond the query terms, as shown in Figure 3(d). We collect 1,250 images with 25 query terms where the text detector's detected regions might include unrelated extra not unrelated characters.

We observe that different Chinese text layouts lead to varying granularities in the outputs of text detectors. For example, a cross-line layout may produce detection boxes that capture only portions of the query terms, while a partial layout might include additional characters. Besides, vertically arranged text is often associated with surrounding visual elements, such as nearby buildings or scenery. Previous scene text retrieval models typically crop text regions based on detector results, discarding all visual information outside the detection boxes and thereby losing valuable context. Additionally, these models employ single-granularity alignment, strictly aligning the visual features of a text region to the textual features of all text contained within, thereby limiting the model's broader perceptual scope. Consequently, our proposed DL-CSVTR benchmark introduces new challenges to assess the model's ability of handling diverse text layouts, with the data being used exclusively for testing purposes. Detailed settings of the query terms and image visualizations can be found in the supplementart material.

# 4. Method

This section introduces our proposed scene text retrieval model, Chinese Scene Text Retrieval CLIP (CSTR-CLIP), which leverages multi-granularity perception guided by text regions and full-image information understanding capabilities. Additionally, it covers the training data-based random granularity alignment algorithm and explains how the model is applied to downstream tasks in scene text retrieval.

To overcome the limitations of the text region cropping paradigm, we use the entire image for feature extraction, expanding the visual receptive field. To distinguish different text labels in the image, we synthesize segmentation maps by annotating the position information of the text labels, using them to guide CLIP's focus on specific regions. A two-stage training process is designed to gradually equip CLIP with the ability to handle scene text retrieval across diverse text layouts. The model structure is illustrated in Figure 4.

Stage 1: Training CLIP's OCR and regional perception capabilities. As shown in Figure 4(a), the model is initialized with pre-trained Chinese CLIP weights (Yang et al., 2023), and Text Position Convolution is introduced to enhance the regional perception capabilities. During training, we keep the text encoder $E_T$ frozen.

Given an input image $I \in R^{H \times W \times 3}$ , it is first embedded with the CLIP RGB Convolutional layer $Conv_{RGB}$ . Then, we generate a segmentation map $G \in R^{H \times W \times 1}$ for each text label in I, highlighting the pixels in the text area while

![](images/7045f9a653da48ca28236d4061dcfb7deaff558f4acb04cdb21369565dc9c0ca.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Image"] --> B["RGB Conv"]
    C["G"] --> D["TextPos Conv"]
    B --> E["F_seg"]
    D --> E
    E --> F["Image Encoder Ei"]
    F --> G["ψi"]
    G --> H["LNCE"]
    H --> I["ψT"]
    I --> J["Text Encoder Et"]
    J --> K["川渝捞"]
    K --> L["T"]
```
</details>

(a) Stage 1: Focuses on Training CLIP's OCR and Regional Attention Capabilities

![](images/cf5a27613856250dcc79e1186d20996558611287410c769e150675ceff15878f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Random Splicing"] --> B["Random Mask &quot;拐火锅&quot; α"]
    B --> C["Random Expand &quot;川渝捞火锅&quot; β"]
    C --> D["“火锅” T_ip"]
    D --> E["“拐火锅” θ"]
    E --> F["G_ik"]
    E --> G["G_ip"]
```
</details>

(c) Random Alignment Granularity Processing (RAGP)

![](images/667505b7df2f75f468394b1d61e925d426174475e9fe7e8ced9d709a12960450.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Segmentation Map Set"] --> B["RAGP"]
    B --> C["GRandom"]
    C --> D["TextPos Conv"]
    D --> E["RGB Conv"]
    E --> F["RGB Conv"]
    F --> G["Encoder Layer L1"]
    G --> H["Fusion Layer I"]
    H --> I["Fusion Layer i"]
    I --> J["Fusion Layer K"]
    J --> K["Attention Block"]
    K --> L["LNCB"]
    L --> M["Text Encoder"]
    M --> N["TRandom"]
    N --> O["RAGP"]
    O --> P["川渝捞"]
    P --> Q["Image Encoder Eorigin"]
    Q --> R["Fusion Layer Lk"]
    R --> S["Fusion Layer i"]
    S --> T["Fusion Layer i"]
    T --> U["Fusion Layer Lk"]
    U --> V["Fusion Layer K"]
    V --> W["Attention Block"]
    W --> X["ψI Random"]
    X --> Y["LNCB"]
    Y --> Z["ψT Random"]
```
</details>

(b) Stage 2: Training with Global Information Integration and Multi-Granularity Alignment.   
Figure 4. Two-stage training framework for the CSTR-CLIP model, including: (a) Training of model's region attention and OCR capabilities based on single-granularity alignment; (b) Adapting CSTR-CLIP to diverse text layouts using random multi-granularity alignment and integrating global features. (c) The Random Alignment Granularity Processing (RAGP) module in stage 2.

setting the background pixels to zero. We design a single-channel convolutional layer, $Conv_{Textpos}$ , that accepts G as input to obtain the guided embedding. After that, these two embeddings are fused:

$$
F _ {\text { seg }} = \operatorname{Conv} _ {R G B} (I) + \operatorname{Conv} _ {T e x t p o s} (G) \tag {1}
$$

Next, $F_{seg}$ is fed into the CLIP image encoder $E_{I}$ to obtain the image feature $\psi_{I} \in R^{d}$ , where d is the dimension of the CLIP embedding space. The corresponding text is fed into the text encoder $E_{T}$ to obtain the text feature $\psi_{T} \in R^{d}$ :

$$
\psi_ {I}, \psi_ {T} = E _ {I} (F _ {\text { seg }}), E _ {T} (T) \tag {2}
$$

During training, for each image $I_{i}$ , we generate triplets $\{I_{i}, G_{ik}, T_{ik}\}$ by iterating over its text box annotations, where i is the image index and k is the text annotation index. The number of text line annotations determines the number of triplets generated. We use all triplets generated from the images as the training set, apply a single-granularity alignment method to align text regions with their contents, and train the model using NCE Loss:

$$
L _ {\mathrm{NCE}} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {\exp \left(\psi_ {I} ^ {i} \cdot \psi_ {T} ^ {i} / \tau\right)}{\sum_ {j = 1} ^ {N} \exp \left(\psi_ {I} ^ {i} \cdot \psi_ {T} ^ {j} / \tau\right)} \tag {3}
$$

where $\tau$ is the temperature parameter.

This stage enhances CLIP's OCR capabilities and its perception guided by the specified regions. The model is initially trained with synthetic data and then fine-tuned using real-world scene data. It can direct perception to specified regions through $Conv_{Textpos}$ . However, the single-granularity alignment method used during training leads to losing some visual features in non-guided areas.

Stage 2: Training with Global Information Integration and Multi-Granularity Alignment. We utilize the frozen original Chinese CLIP visual encoder $E_{origin}$ to enhance our first stage model, as illustrated in Figure 4(b). The scene image I is also fed into $E_{origin}$ simultaneously. For each layer l, the intermediate features $F_{focus}^{l}$ from $E_{I}$ are augmented by the corresponding global features $F_{origin}^{l}$ from $E_{origin}$ :

$$
F _ {\text { fuse }} ^ {l} = F _ {\text { focus }} ^ {l} + \text { FusionLayer } (F _ {\text { origin }} ^ {l}) \tag {4}
$$

where the Fusion Layer is a 1x1 convolutional layer.

This process aims to recover the visual feature loss in non-perception regions from the first stage, thereby enhancing the model's understanding of global information. This stage exclusively uses real-world scene data for training.

Random Alignment Granularity Processing (RAGP). To augment the training data for multi-granularity alignment, we apply RAGP on segmentation maps and text inputs, as illustrated in Figure 4(c). According to our previous analysis, single-granularity alignment limits the model's perception to

![](images/e27a3060540dc4fc4c114b414ea9330eeb6980df0319e77150e3f18b4b4ea098.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text Detector"] --> B["Gallery Set"]
    B --> C["CSTR-CLIP Image Encoder"]
    C --> D["CSTR-CLIP Text Encoder"]
    D --> E["query"]
    style A fill:#99ccff,stroke:#333
    style B fill:#f9f9f9,stroke:#333
    style C fill:#e0e0e0,stroke:#333
    style D fill:#e0e0e0,stroke:#333
    style E fill:#fff2cc,stroke:#333
```
</details>

Figure 5. The retrieval pipeline of CSTR-CLIP.

the entire text within the guided region, thereby reducing its flexibility. In cross-line and partial layouts, the corresponding text region may miss some queries or include redundant text, making single-granularity alignment less than ideal. The key to solving this problem is to enable the model to flexibly perceive text elements inside and near the text region, disrupting the exact match between the segmentation map $G_{ik}$ and the corresponding text $T_{ik}$ during training.

We truncate $T_{ik}$ by masking m characters from both ends, ensuring a minimum length of 2, with the deletion probability controlled by a hyperparameter $\beta$ . We assume that semantically related text elements are spatially close in the image. For $T_{ik}$ , we identify the text $T_{ip}$ whose bounding box centroid is closest to $T_{ik}$ and extend it in the direction connecting their centroids, with the probability of adding text controlled by a hyperparameter $\alpha$ . For the segmentation map $G_{ik}$ , we locate the segmentation map $G_{ip}$ whose bounding box centroid is closest to $T_{ik}$ and concatenate them, with the probability of concatenation controlled by a hyperparameter $\theta$ . In general, the text is first randomly expanded and then randomly masked, and the processing of the segmentation map is independent of the text processing.

CSTR-CLIP for Inference. As illustrated in Figure 5, we use PaddleOCR to detect text, and RAGP is not applied in the inference stage. Detected text regions are converted into segmentation maps and paired with the original image, forming the gallery set. If no text regions are detected, a fully highlighted image is used as the segmentation map. All related pairs in the gallery set are processed by the visual encoder to extract visual features, and the query word is encoded into text features by the text encoder. For an image, multiple binary groups are formed with the various text regions identified by the detector, resulting in multiple visual features guided by different text regions. The highest similarity score between these visual and text features determines the final score for the image's visual representation.

# 5. Experiments

Training Datasets. We follow the training setting of previous studies (Wen et al., 2023; Luo et al., 2024), employing a combination of synthetic and real data for model training.

- SynthText-CH. Following SynthText-900K (Gupta et al., 2016), we create the SynthText-CH dataset with 300K samples, recording bounding box coordinates to generate segmentation maps for the first training stage.   
- ReCTS. The ReCTS (Zhang et al., 2019) dataset comprises annotated Chinese text images in natural scenes. Segmentation maps are generated from these annotations for both training stages. It serves as the data for the second training stage.

Test Datasets. The test data comprises the existing Chinese scene text retrieval benchmark CSVTR and our proposed DL-CSVTR.

- CSVTR. This dataset includes 1,667 images which are gathered using Google Image Search, corresponding to 23 predefined queries. The visual appearance of the query terms in the images is primarily in a horizontal layout and presented independently.   
- DL-CSVTR. The DL-CSVTR dataset proposed in this paper includes three types of text layout benchmarks, with a total of 2,070 images and 89 predefined query terms. This dataset will be used to evaluate the model's retrieval ability across different text configurations. The tests will be conducted on the subsets corresponding to the three types of text layouts.

Baselines. We use previously published scene text retrieval models tested on CSVTR as baselines. We reproduce and evaluate the most competitive methods on DL-CSVTR. Since earlier methods do not utilize the CLIP pre-trained model, we replace the backbones in previous works (Wang et al., 2021; Luo et al., 2024) with the pre-trained CLIP model to ensure a fair comparison. These methods represent the state-of-the-art in cross-modal and visual embedding paradigms. The source of training data is the same as CSTR-CLIP, with the text area cropped according to the annotations for the training set.

Implementation Details. Due to CLIP's limited input size and the small text in scene images, we expand the input resolution of CSTR-CLIP to 640×640. Yet, this will lead to inappropriate positional encoding in the self-attention layer that processes the features extracted by visual extraction module, so we replace them with learnable ones, initialized using nearest neighbor interpolation. The visual extraction module utilizes the pre-trained ResNet50 version.

We optimize CSTR-CLIP using the Adam optimizer with an initial learning rate of 1e-6 and a batch size of 48. The first stage is trained for 8 epochs (6 with synthetic data and

2 with real data), while the second stage is trained for 10 epochs. In RAGP, we set the hyperparameters controlling the randomness to $\beta = 0.2$ , $\alpha = 0.3$ , and $\theta = 0.2$ . Models are trained on an NVIDIA A6000 GPU and tested on a GTX 1080 GPU.

# 5.1. Performance on CSVTR benchmark

<table><tr><td>Method</td><td>mAP %</td><td>FPS</td></tr><tr><td>Mishra et al(Mishra et al., 2013)</td><td>4.79</td><td>0.10</td></tr><tr><td>TDSL (Wang et al., 2021)</td><td>60.23</td><td>12</td></tr><tr><td>VSTR (Wen et al., 2023)</td><td>63.17</td><td>11</td></tr><tr><td>Luo et al(Luo et al., 2024)</td><td>69.75</td><td>11.2</td></tr><tr><td>CLIP (640x640)(Radford et al., 2021)</td><td>57.32</td><td>47.6</td></tr><tr><td>TDSL (Wang et al., 2021) $^{\dagger}$ </td><td>78.36</td><td>12.3</td></tr><tr><td>Luo et al(Luo et al., 2024) $^{\dagger}$ </td><td>80.74</td><td>10.5</td></tr><tr><td>CSTR-CLIP</td><td>88.57</td><td>21.5</td></tr></table>

Table 1. Comparison with existing methods on CSVTR. $^{\dagger}$ represents the version that replaces the backbone with the CLIP encoder. We highlight the best and the second results.

We compare our model with previous models on CSVTR, and the experimental results are shown in Table 1. The results clearly demonstrate that our model outperforms the previous methods, improving retrieval accuracy by $18.82\%$ while maintaining competitive inference speed. Additionally, our method continues to excel even when the encoders of state-of-the-art models based on cross-modal and visual embeddings are replaced with the CLIP encoder as an additional baseline. We attribute this improvement to our method's ability to retain visual features outside the text regions from a global image perspective. In scenarios where the detector fails to detect the text regions or the text regions in the image have poor clarity, our full-image based method leverages rich visual context information to achieve superior retrieval capabilities.

# 5.2. Performance on DL-CSVTR benchmark

As shown in Table 2, CLIP demonstrates strong retrieval ability across all DL-CSVTR benchmarks, highlighting the importance of perceiving beyond cropped regions for understanding diverse text layouts. While methods based on the cropped regions paradigm surpass the CLIP's retrieval ability in vertical layouts, they perform poorly in cross-line and partial layouts. This indicates that the cropped regions paradigm struggles to address the challenges posed by these more complex configurations.

However, after single-granularity training in the first stage, our method surpasses previous methods based on the cropped regions paradigm across all DL-CSVTR benchmarks, and further surpasses all baselines to achieve the best retrieval performance after the second stage of training. For vertical layouts, our method outperforms cropped region-based methods by leveraging full-image information, demonstrating that understanding information beyond cropped regions is crucial for improving the model's understanding of vertical text layouts. In cross-line and partial layouts, the model, limited by single-granularity alignment, does not surpass the CLIP baseline after the first stage. However, in the second stage, the fusion of full-image information with multi-granularity perception training, guided by text regions, overcomes the limitations of single-granularity alignment and non-guided region information loss from the first stage, resulting in impressive performance.

# 5.3. Ablation Study

In this section, we conduct a series of ablation experiments based on our model training stages. The detailed results are presented in Table 3.

Effectiveness of Stage 1 settings. We conduct an ablation study on the Stage 1 settings to evaluate their effectiveness. (See #1, #2 and #3) When the visual encoder is frozen, $Conv_{Textpos}$ can be considered a data augmentation method, adapting the image to the visual encoder's attention bias through single-granularity alignment. This approach performs well on CSVTR, where query terms typically appear independently and in a horizontal layout, resulting in better performance due to accurate text detection. However, in the more diverse settings of DL-CSVTR, the performance declines due to suboptimal detector outputs, particularly in partial and cross-line layouts where only parts of the query terms or extraneous elements are captured.

Involving the visual encoder $E_I$ in model training enhances performance across all datasets (See #2 and #3). This improvement results from fine-tuning the OCR capabilities and enhancing $Conv_{Textpos}$ 's guidance function at the visual encoder level. The gains are especially notable on CSVTR and DL-CSVTR-V, where jointly training the image encoder $E_I$ and $Conv_{Textpos}$ results in optimal attention guidance. However, improvements on DL-CSVTR-CL and DL-CSVTR-P are less pronounced, as the single-granularity alignment method overly restricts the model's focus, resulting in poorer performance when only part of the query's visual representation or irrelevant elements are included in the guidance area.

Effectiveness of Stage 2 settings. For the Stage 2, the ablation results of each setting are also reported (See #3, #4, #5 and #6). The inclusion of full-image features on CSVTR leads to some improvements, but RAGP does not yield significant gains, as query terms in CSVTR typically appear independently and in a horizontal layout. In DL-CSVTR-V, the addition of full-image information brings significant improvement, validating the importance of full-image layout information in understanding vertically aligned text. For

<table><tr><td>Method</td><td>DL-CSVTR-V (mAP %)</td><td>DL-CSVTR-CL (mAP %)</td><td>DL-CSVTR-P (mAP %)</td></tr><tr><td>Luo et al(Luo et al., 2024)</td><td>39.87</td><td>21.98</td><td>13.95</td></tr><tr><td>CLIP (640x640) (Radford et al., 2021)</td><td>55.48</td><td>54.43</td><td>37.31</td></tr><tr><td>TDSL(Wang et al., 2021) $^{\dagger}$ </td><td>62.51</td><td>40.46</td><td>24.88</td></tr><tr><td>Luo et al(Luo et al., 2024) $^{\dagger}$ </td><td>52.73</td><td>34.23</td><td>29.91</td></tr><tr><td>CSTR-CLIP (Stage1)</td><td>74.50</td><td>45.98</td><td>33.25</td></tr><tr><td>CSTR-CLIP</td><td>84.44</td><td>65.56</td><td>61.85</td></tr></table>

Table 2. Comparison with existing methods on DL-CSVTR. $^{\dagger}$ represents the version that replaces the backbone with the CLIP encoder. We highlight the best and the second results. 

<table><tr><td rowspan="2">#</td><td rowspan="2">Stage1 $Conv_{Textpos}$ </td><td rowspan="2">IE</td><td rowspan="2">Stage2Global features</td><td rowspan="2">RAGP</td><td colspan="4">mAP(%)</td></tr><tr><td>CSVTR</td><td>DL-CSVTR-V</td><td>DL-CSVTR-CL</td><td>DL-CSVTR-P</td></tr><tr><td>1</td><td>X</td><td>X</td><td>X</td><td>X</td><td>57.32</td><td>55.48</td><td>54.43</td><td>37.30</td></tr><tr><td>2</td><td>√</td><td>X</td><td>X</td><td>X</td><td>59.50</td><td>53.17</td><td>42.15</td><td>30.70</td></tr><tr><td>3</td><td>√</td><td>√</td><td>X</td><td>X</td><td>86.25</td><td>74.50</td><td>45.98</td><td>33.25</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>X</td><td>88.41</td><td>83.91</td><td>57.82</td><td>43.19</td></tr><tr><td>5</td><td>√</td><td>√</td><td>X</td><td>√</td><td>86.31</td><td>74.01</td><td>60.08</td><td>52.51</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td>√</td><td>88.57</td><td>84.44</td><td>65.56</td><td>61.85</td></tr></table>

Table 3. Ablation experimental results. In Stage 1, we perform ablation studies on the use of $Conv_{Textpos}$ and the inclusion of the image encoder (IE) in training. In Stage 2, we ablate the inclusion of global features from the original CLIP model and the use of multi-granularity alignment processing. We highlight the best and the second results.

DL-CSVTR-CL and DL-CSVTR-P, integrating full-image information and random granularity alignment processing significantly enhances the model's ability to retrieve text across rows and in partial layouts. RAGP refines the model's attention to partial text within the guided area and external text content, while full-image information enhances the model's overall perception and understanding of the image content. As shown in Figure 6, we visualize the fused features of the intermediate layers using the merged channel visualization method. The visualization results clearly indicate that after the Stage 2 training, CSTR-CLIP not only enhances its perception of the entire image but also improves its understanding of the text information near the guided area and further refines the internal text elements.

![](images/d9a64d39d80230c7b5babd69c1b223844564a306d90e3dc3f4d4cc5783e701a9.jpg)

<details>
<summary>text_image</summary>

(a)
FFocus
FLayer1
FLayer2
FLayer3
FLayer4
(b)
(c)
</details>

Figure 6. Visualization results of model intermediate layer features, including (a) original CLIP, (b) CSTR-CLIP after Stage 1 training, (c) CSTR-CLIP after Stage 2 training.

<table><tr><td>Query</td><td>Segmentation Map</td><td>Top3 Retrival results</td></tr><tr><td rowspan="2">“蜜雪冰城”</td><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Figure 7. An example of region-specified scene text retrieval. The green boxes in the search results indicate visual instances of the query terms.

# 5.4. Interactive Region-Specified Scene Text Retrieval

Our approach paves the way for more diverse text retrieval methods, enabling region-specific retrieval. Specifically, $Conv_{Textpos}$ allows the model to focus on user-defined regions, while multi-granularity alignment enhances the flexibility of the model's perception, permitting users to custom-design segmentation maps. As illustrated in Figure 7, by highlighting specific parts of the segmentation map, we direct the model's attention to the corresponding image region, with the visual examples of the query terms in the top-ranked results mostly appearing in the guided region. This enables our model to perform more accurate scene text retrieval with region-specific guidance.

# 6. Conclusion

This paper identifies the limitations of the previous scene text retrieval paradigm in handling the diverse text layouts of Chinese. To tackle it, a new benchmark DL-CSVTR is proposed, which contains three types of common text layouts for evaluating models' retrieval capabilities in real scenarios. To overcome the limitations of the previous text region cropping paradigms in Chinese scene text retrieval, we propose the CSTR-CLIP model. Our model is designed with a multi-granularity-aware paradigm, leveraging the entire image while being guided by text regions. This approach not only achieves the best retrieval accuracy and speed on previous benchmarks but also demonstrates significant performance improvements under various text layout conditions on DL-CSVTR. The text region-guided design allows users to specify regions of interest for more detailed retrieval in practical applications.

# Acknowledge

Supported by the National Natural Science Foundation of China (Grant NO 62376266 and 62406318), Key Laboratory of Ethnic Language Intelligent Analysis and Security Governance of MOE, Minzu University of China, Beijing, China.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Almazán, J., Gordo, A., Fornès, A., and Valveny, E. Word spotting and recognition with embedded attributes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014.   
Bai, X., Yang, M., Lyu, P., Xu, Y., and Luo, J. Integrating scene text and visual appearance for fine-grained image classification. IEEE Access, 2018.   
Baldrati, A., Bertini, M., Uricchio, T., and Del Bimbo, A. Composed image retrieval using contrastive learning and task-oriented clip-based features. ACM Transactions on Multimedia Computing, Communications and Applications, 2023.   
Cao, T., Lyu, J., Zeng, W., Mu, W., and Zhou, Y. The devil is in fine-tuning and long-tailed problems: A new benchmark for scene text detection. In IJCAI, 2025.   
Ch'ng, C. K. and Chan, C. S. Total-text: A comprehensive

dataset for scene text detection and recognition. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2017.   
Ghosh, S. K. and Valveny, E. Query by string word spotting based on character bi-gram indexing. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2015.   
Ghosh, S. K., Gomez, L., Karatzas, D., and Valveny, E. Efficient indexing for query by string text retrieval. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2015.   
Gómez, L., Rusinol, M., and Karatzas, D. Lsde: Levenshtein space deep embedding for query-by-string word spotting. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2017.   
Gómez, L., Mafla, A., Rusinol, M., and Karatzas, D. Single shot scene text retrieval. In Proceedings of the European Conference on Computer Vision (ECCV), 2018.   
Gupta, A., Vedaldi, A., and Zisserman, A. Synthetic data for text localisation in natural images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.   
Huang, M., Li, H., Liu, Y., Bai, X., and Jin, L. Bridging the gap between end-to-end and two-step text spotting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
Jaderberg, M., Simonyan, K., Vedaldi, A., and Zisserman, A. Reading text in the wild with convolutional neural networks. International Journal of Computer Vision, 2016.   
Karaoglu, S., Tao, R., Gevers, T., and Smeulders, A. W. Words matter: Scene text for image classification and retrieval. IEEE Transactions on Multimedia, 2016.   
Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning (ICML), 2022.   
Liao, M., Pang, G., Huang, J., Hassner, T., and Bai, X. Mask textspotter v3: Segmentation proposal network for robust scene text spotting. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.   
Lin, Y., He, C., Wang, A. J., Wang, B., Li, W., and Shou, M. Z. Parrot captions teach clip to spot text, 2024. URL https://arxiv.org/abs/2312.14232.

Liu, Y., Chen, H., Shen, C., He, T., Jin, L., and Wang, L. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.   
Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., and Li, T. Clip4clip: An empirical study of clip for end-to-end video clip retrieval and captioning. Neurocomputing, 2022.   
Luo, H., Ibrayim, M., Hamdulla, A., and Deng, Q. Visual and semantic guided scene text retrieval. The Journal of Supercomputing, 2024.   
Lyu, J., Wang, W., Yang, D., Zhong, J., and Zhou, Y. Arbitrary reading order scene text spotter with local semantics guidance. In AAAI, volume 39, pp. 5919–5927, 2025a.   
Lyu, J., Wei, J., Zeng, G., Li, Z., Xie, E., Wang, W., Ma, C., and Zhou, Y. TextBlockV2: Towards precise-detection-free scene text spotting with pre-trained language model. TOMM, 2025b.   
Mafla, A., Dey, S., Biten, A. F., Gomez, L., and Karatzas, D. Fine-grained image classification and retrieval by combining visual and locally pooled textual features. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020.   
Mafla, A., Tito, R., Dey, S., Gómez, L., Rusinol, M., Valveny, E., and Karatzas, D. Real-time lexicon-free scene text retrieval. Pattern Recognition, 2021.   
Materzyńska, J., Torralba, A., and Bau, D. Disentangling visual and written concepts in clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
Mhiri, M., Desrosiers, C., and Cheriet, M. Word spotting and recognition via a joint deep embedding of image and text. Pattern Recognition, 2019.   
Mishra, A., Alahari, K., and Jawahar, C. Image retrieval using textual cues. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2013.   
Qiao, L., Tang, S., Cheng, Z., Xu, Y., Niu, Y., Pu, S., and Wu, F. Text perceptron: Towards end-to-end arbitrary-shaped text spotting. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2020a.   
Qiao, Z., Zhou, Y., Yang, D., Zhou, Y., and Wang, W. SEED: Semantics enhanced encoder-decoder framework for scene text recognition. In CVPR, pp. 13528–13537, 2020b.

Qiao, Z., Zhou, Y., Wei, J., Wang, W., Zhang, Y., Jiang, N., Wang, H., and Wang, W. PIMNet: A parallel, iterative and mimicking network for scene text recognition. In ACM MM, pp. 2046–2055, 2021.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), 2021.   
Saito, K., Sohn, K., Zhang, X., Li, C.-L., Lee, C.-Y., Saenko, K., and Pfister, T. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 2022.   
Shen, H., Li, G., Zhong, J., and Zhou, Y. LDP: Generalizing to multilingual visual information extraction by language decoupled pretraining. In AAAI, volume 39, pp. 6805–6813, 2025.   
Shi, C. and Yang, S. Logoprompt: Synthetic text images can be good visual prompts for vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.   
Shtedritski, A., Rupprecht, C., and Vedaldi, A. What does clip know about a red circle? visual prompt engineering for vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.   
Shu, Y., Wang, W., Zhou, Y., Liu, S., Zhang, A., Yang, D., and Wang, W. Perceiving ambiguity and semantics without recognition: An efficient and effective ambiguous scene text detector. In ACM MM, pp. 1851–1862, 2023.   
Shu, Y., Zeng, W., Zhao, F., Chen, Z., Li, Z., Yang, X., Zhou, Y., Rota, P., Bai, X., Jin, L., et al. Visual text processing: A comprehensive review and unified evaluation. arXiv preprint arXiv:2504.21682, 2025.   
Song, H., Wang, H., Huang, S., Xu, P., Huang, S., and Ju, Q. Text siamese network for video textual keyframe detection. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019.   
Subramanian, S., Merrill, W., Darrell, T., Gardner, M., Singh, S., and Rohrbach, A. Reclip: A strong zero-shot baseline for referring expression comprehension, 2022. URL https://arxiv.org/abs/2204.05991.

Sun, Z., Fang, Y., Wu, T., Zhang, P., Zang, Y., Kong, S., Xiong, Y., Lin, D., and Wang, J. Alpha-clip: A clip model focusing on wherever you want. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
Wang, H., Bai, X., Yang, M., Zhu, S., Wang, J., and Liu, W. Scene text retrieval via joint text detection and similarity learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4558–4567, 2021.   
Wang, K. and Belongie, S. Word spotting in the wild. In Proceedings of the European Conference on Computer Vision (ECCV), 2010.   
Wang, W., Zhou, Y., Lv, J., Wu, D., Zhao, G., Jiang, N., and Wang, W. TPSNet: Reverse thinking of thin plate splines for arbitrary shape scene text representation. In ACM MM, pp. 5014–5025, 2022.   
Wei, J., Zhang, Y., Zhou, Y., Zeng, G., Qiao, Z., Guo, Y., Wu, H., Wang, H., and Wang, W. TextBlock: Towards scene text spotting without fine-grained detection. In ACM MM, pp. 5892–5902, 2022.   
Wen, L., Wang, Y., Zhang, D., and Chen, G. Visual matching is enough for scene text retrieval. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM), 2023.   
Wilkinson, T. and Brun, A. Semantic and verbatim word spotting using deep neural networks. In Proceedings of the International Conference on Frontiers in Handwriting Recognition (ICFHR), 2016.   
Xing, Y., Kang, J., Xiao, A., Nie, J., Shao, L., and Lu, S. Bridging semantic gaps for language-supervised semantic segmentation. CoRR, 2023.   
Yang, A., Pan, J., Lin, J., Men, R., Zhang, Y., Zhou, J., and Zhou, C. Chinese clip: Contrastive vision-language pretraining in chinese, 2023. URL https://arxiv.org/abs/2211.01335.   
Yang, X., Qiao, Z., and Zhou, Y. IPAD: Iterative, parallel, and diffusion-based network for scene text recognition. IJCV, pp. 1–21, 2025.   
Yu, W., Liu, Y., Hua, W., Jiang, D., Ren, B., and Bai, X. Turning a clip model into a scene text detector. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
Zeng, G., Zhang, Y., Zhou, Y., Yang, X., Jiang, N., Zhao, G., Wang, W., and Yin, X.-C. Beyond OCR + VQA: Towards end-to-end reading and reasoning for robust and accurate textvqa. Pattern Recognition, 138:109337, 2023.

Zeng, G., Zhang, Y., Wei, J., Yang, D., Zhang, P., Gao, Y., Qin, X., and Zhou, Y. Focus, distinguish, and prompt: Unleashing CLIP for efficient and flexible scene text retrieval. In ACM MM, pp. 2525–2534, 2024a.   
Zeng, W., Shu, Y., Li, Z., Yang, D., and Zhou, Y. TextCtrl: Diffusion-based scene text editing with prior guidance control. NeurIPS, 37:138569–138594, 2024b.   
Zhang, R., Zhou, Y., Jiang, Q., Song, Q., Li, N., Zhou, K., Wang, L., Wang, D., Liao, M., Yang, M., et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019.   
Zhang, Y., Liu, C., Wei, J., Yang, X., Zhou, Y., Ma, C., and Ji, X. Linguistics-aware masked image modeling for self-supervised scene text recognition. In CVPR, 2025.   
Zhou, C., Loy, C. C., and Dai, B. Extract free dense labels from clip. In Proceedings of the European Conference on Computer Vision (ECCV), 2022a.   
Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022b.

# A. Analysis of DL-CSVTR

This section supplements the statistical analysis of CSVTR and the definition of query terms in our proposed DL-CSVTR.

# A.1. Analysis of text layout distribution in CSVTR

We have analyzed the four representations of query terms in CSVTR. The statistics are based on the image as the basic unit, with priority given to horizontal, then partial, then cross-line, and finally vertical layouts. For example, if the visual representation of a query term in an image exists both horizontally and cross-line, the image is classified as having a horizontal layout. The statistical results are shown in Figure 8.

![](images/723f723676172abec6bc9b78c91f9d340a9ff7d7eacf33bfccf734c31f9e3824.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Horizontal | 93 |
| other | 7 |
| Cross-Line | 1 |
| Vertical | 3 |
| Partial | 3 |
</details>

Figure 8. Distribution of text layouts for visual representations of query terms in CSVTR

The statistical results clearly show that the visual layout of most query terms is horizontal, which does not reflect the challenges that diverse text layouts pose to scene text retrieval tasks.

# A.2. Defining the query terms in DL-CSVTR

The query terms need to appear in a variety of scene images, so they must be common, high-frequency words. In the Chinese context, these are mostly Chinese trademark names, idioms, or noun concepts with specific semantic meanings. Based on our analysis and the definition of query terms in CSVTR, we designed query terms for the three types of text layout benchmarks in DL-CSVTR, as shown in Figure 2.

For each defined query term, we used common image search engines and manually screened images to find visual representations of the query terms that match the specific text layouts, ensuring that the three layouts remain distinct in our DL-CSVTR.

# A.3. Sample visualization in DL-CSVTR

We visualize data from the three benchmarks in DL-CSVTR to provide a clearer understanding of the DL-CSVTR dataset. The visualization results are presented in Figure 10.

In DL-CSVTR-V, the visual representation of the query terms is in a vertical layout, while in DL-CSVTR-CL and DL-CSVTR-P, the representation spans rows or columns and is connected to other characters.

<table><tr><td colspan="2">(a) DL-CSVTR-V</td><td colspan="2">(b) DL-CSVTR-CL</td><td colspan="2">(c) DL-CSVTR-P</td></tr><tr><td colspan="2">Vertical</td><td colspan="2">Cross-Line =||</td><td colspan="2">Partial</td></tr><tr><td><img src="images/410744aeb482418eb890f13e1bc1604744ed3d8efc428883c281808be1753541.jpg"/></td><td><img src="images/162fd523aeb9509efe8f8454562a072bd22627d56de34a5e0bf3636c2cbda431.jpg"/></td><td><img src="images/88bbdd110a3848f3dde5f1d4628c9c8b3dd5c9d38a4d1c21f18e036f8e3e4a94.jpg"/></td><td><img src="images/f12348aa946d8bfa995d437e74e5c2e0c049836cc1a519fcf09b52bd7179d0cb.jpg"/></td><td><img src="images/3818c7a943387fec6cc0b4b6895392eafd2667dd2518542787328001d1bf60ee.jpg"/></td><td><img src="images/3cae569069d46a8edb2725dfd62df8401380fefadcd5ad59e73e1ef470078c61.jpg"/></td></tr><tr><td><img src="images/d47d537a9647bf6330e302a365315d0e5b6aff00d6c4bb168ecdd0b493bb5799.jpg"/></td><td><img src="images/a5098ee5ed8cc4b380f302bf24302d3b94f3ae40c16feaf2f4e0656802bf6509.jpg"/></td><td><img src="images/ee2073c54478d4d5ce6fb9cfb03180b6ece1fef688ec0daae6ac8eacee12e3ec.jpg"/></td><td><img src="images/0d7eeb32d1fda11bebcae73b2dd5cd4358d1b07637cc0d9eb41282d6098590f2.jpg"/></td><td><img src="images/6b16c87add257570b5cb36925688abd5bb6a3dd08e73b7c49e028520892a55ba.jpg"/></td><td><img src="images/af96f7c8a75b09f7edeedc09d8ae629a32f8f5dad2cfb9a958a9654ebc7eed04.jpg"/></td></tr><tr><td colspan="2">{临榆炸鸡腿,乔迁之喜保持车距,名创优品,大吉大利,封顶大吉,年年有余,开业大吉,早生贵子,欢迎光临,清仓甩卖,瑞幸咖啡,米村拌饭,西贝莜面村,金榜题名,阿水大杯茶,高考加油,乔治莫兰迪,买一送一,夜泊秦淮,出入平安,奈雪的茶,小心地滑,库迪咖啡,新年快乐,欢度国庆,欢迎回家,琦王花生,生日快乐,薛记炒货,金桂乌龙,锅圈食汇,领鲜全城}</td><td colspan="2">{7天连锁酒店,七里山塘,中国建设银行,便携换乘,出入平安,南京大排档,夜泊秦淮,如家酒店,封顶大吉,汉庭酒店,海底捞火锅,自助银行服务,茉莉乌龙,金榜题名,银座佳驿酒店,麦当劳,一帆风顺,万达广场,亚朵酒店,全季酒店,半天妖,吾悦广场,太古里,家和万事兴,招商银行,洗手间,筷子街,茂昌眼镜,速8酒店,金足印象,锦江之星}</td><td colspan="2">{中心,会议,便利,动物,医院,协会,卫生,台球,咖啡,垃圾,大学,学院,安全,小学,广场,文明,服务,研究,社区,网咖,药房,超市,酒店,饭店,高考}</td></tr></table>

Figure 9. Distribution of text layouts for visual representations of query terms in CSVTR

# A.4. Supplementary examples of DL-CSVTR

We provide a portion of the DL-CSVTR dataset, ensuring that the images do not reveal information about the author or associated entities. The dataset includes the following:

1. DL-CSVTR-V: This folder contains a subset of vertically arranged data, where the visual instances of the query words appear in a vertical layout within the images.   
2. DL-CSVTR-CL: This folder contains a subset of cross-row layout data, where the visual instances of the query words appear in a cross-row or cross-column layout within the images.   
3. DL-CSVTR-P: This folder contains a subset of partial layout data, where the visual instances of the query words appear connected to other text within the images.

Algorithm 1 Training Triplet Generation   
Input: OCR dataset with images $\{I_{i}\}$ and their text annotations $\{T_{ik}\}$ , where i is the image index and k is the text line index.

Output: Set of triplets $\{(I_{i}, G_{ik}, T_{ik})\}$ for training

1: for each image $I_{i}$ in the dataset do

2: Initialize an empty set of triplets T

3: for each text line annotation $T_{ik}$ in $I_{i}$ do

4: Initialize a grayscale image $G_{ik}$ of the same size as $I_{i}$ with all pixel values set to 0

5: Get the polygonal coordinates $P_{ik}$ of the text line $T_{ik}$ 6: Set the pixel values within the polygon $P_{ik}$ in $G_{ik}$ to 255

7: Add the triplet $(I_{i}, G_{ik}, T_{ik})$ to T

8: end for

9: end for

10: Store the triplets T

11: return All stored triplets $\{(I_{i}, G_{ik}, T_{ik})\}$

Algorithm 2 Retrieval bigrams Generation for Multiple Images using PaddleOCR   
Input: Set of original images $\{I_{i}\}$ Output: Set of retrieval pairs $\{(I_{i}, G_{ik})\}$ 1: Initialize an empty set of pairs P
2: for each image $I_{i}$ in the image set $\{I_{i}\}$ do
3: Use PaddleOCR to perform text detection on image $I_{i}$ , obtaining a set of bounding boxes $\{B_{ik}\}$ 4: for each bounding box $B_{ik}$ in detection results $\{B_{ik}\}$ do
5: Initialize a grayscale image $G_{ik}$ of the same size as $I_{i}$ with all pixel values set to 0
6: Get the polygonal coordinates $P_{ik}$ of the bounding box $B_{ik}$ 7: Set the pixel values within the polygon $P_{ik}$ in $G_{ik}$ to 255
8: Add the pair $(I_{i}, G_{ik})$ to P
9: end for
10: end for
11: return All generated pairs P

# B. Details of the Model

This section provides additional explanations regarding the model details and specific implementation.

# B.1. Training triple generation

We generate training triplets based on the text line annotations of the images for both training and testing in the first and second stages. Specifically, the data generation algorithm creates the segmentation map mentioned in the text, based on the text line annotations. This segmentation map is a grayscale image that forms a triplet with the corresponding original image and text. Please refer to the pseudocode provided in Algorithm 1.

# B.2. Retrieval bigrams generation

In the retrieval stage, thanks to the standard two-tower model we use, we only need to create a segment map for the images in the gallery to get the bigrams. We use Paddle OCR, an OCR detection model with strong performance and fast inference speed, as the detector. The detector is used to give the location annotations of multiple text regions and generate a segmentmap to obtain candidate bigrams. Please refer to the pseudocode provided in Algorithm 2

# B.3. Supplementation of RAGP

To better understand the processing order of RAGP, we provide pseudocode to illustrate the process. Please refer to Algorithm 3.

Algorithm 3 Random Granularity Alignment Processing
Input: Text inputs $T_{ik}$ , Grayscale images $G_{ik}$ Parameter: $\beta, \alpha, \theta$ Output: Processed text inputs and grayscale images
1: for each text input $T_{ik}$ do
2: Random Expand:
3: Find $T_{ip}$ whose centroid is closest to $T_{ik}$ 4: With probability $\alpha$ , merge $T_{ik}$ and $T_{ip}$ 5: Random Mask:
6: if length( $T_{ik}$ ) $\geq$ 4 then
7: With probability $\beta$ , delete $m$ characters from $T_{ik}$ , ensuring length $\geq$ 2
8: end if
9: end for
10: for each grayscale image $G_{ik}$ do
11: Random Splicing:
12: Find $G_{ip}$ whose centroid is closest to $G_{ik}$ 13: With probability $\theta$ , merge $G_{ik}$ and $G_{ip}$ 14: end for

The pseudocode clearly illustrates the process of the RAGP we proposed. In the actual code implementation, we accomplish this random processing algorithm using the getitem function and other custom-defined functions when creating the dataset.

# C. Visualization of Retrieval Results.

# C.1. Retrieval visualization on DL-CSVTR

To intuitively analyze the retrieval capabilities of STR-CLIP and previous methods on Chinese text retrieval with diverse layouts, visualization results for three different text layout benchmarks from DL-CSVTR are presented in Figure 11.

Under the DL-CSVTR-V benchmark, as shown in Figure 11(a), previous methods based on cropping text regions discard visual information outside the text region, hindering the model's ability to understand vertically arranged text. For instance, in positive examples, we observed that vertical text in the scene is often associated with surrounding visual elements, such as vertical billboards and architectural styles, which are crucial for the model's comprehension of vertical layouts. Our method, STR-CLIP, achieves the best retrieval results by understanding text information and related visual elements from a full-image perspective. The AP comparison for each query term is shown in the figure 12.

Under the DL-CSVTR-CL benchmark, as shown in Figure 11(b), our proposed model benefits from breaking the limitations of the cropped text region paradigm. Even if the text region that guides the model to perceive only contains part of the query words, our full-image information perception and multi-granularity perception guided by the text region can solve this problem well. The AP comparison for each query term is shown in the figure 13.

Under the DL-CSVTR-P benchmark, as shown in Figure 11(c), our proposed model outperforms the previous single-granularity matching paradigm by enhancing the perception of specific elements within the text region. The AP comparison for each query term is shown in the figure 14.

# C.2. Retrieval visualization on CSVTR

The AP comparison for each query term is shown in the figure 15.

# D. Supplementary of Experiments

# D.1. Details of baseline reproduction

We modified the open-source code of TDSL (Wang et al., 2021) and Luo et al (Luo et al., 2024) which represent the most advanced scene text retrieval models for cross-modal embedding and visual embedding, respectively. Specifically, for TDSL,

we replaced the language and visual encoders with the pre-trained Chinese CLIP encoder and fine-tuned it using cropped text regions and annotations. For VSTR, we replaced the dual-end visual encoder with the visual encoder from Chinese CLIP and trained it using cropped text regions and annotated text instance images generated from annotations. After replacing the encoder, both models demonstrated stronger retrieval ability on the previous CSVTR benchmark.

# D.2. Supplement of model code

We provide the model code and data processing scripts used in our work, along with some data to illustrate our model training process. We have ensured that all content potentially leaking personal information has been removed. Specifically, the provided code and related data include the following components:

1. CSTR-CLIP-Stage1.py: This file contains the model code and dataset loading script for the first stage of training CSTR-CLIP.   
2. CSTR-CLIP-Stage2.py: This file contains the model code and dataset loading script for the second stage of training CSTR-CLIP. The RAGP algorithm we designed is incorporated into the dataset design.   
3. train-img: This folder contains a portion of the data used for training, including original images and segmentation maps.

<table><tr><td>#</td><td>Benchmark</td><td>Query</td><td colspan="5">Example</td></tr><tr><td rowspan="3">(a)</td><td rowspan="3">DL-CSVTR-V</td><td>“七里山塘”</td><td><img src="images/783abf8540567ac1dcc402f168047741891a469a0eb261d29bf311d4d45849bd.jpg"/></td><td><img src="images/e95057930948e63c528cdef39b16fbdb963b8a8e1af29d6ea16e5f4f84b2e605.jpg"/></td><td><img src="images/3f96c233743323c634524352a901fb5d5fed714669328291ba9f0a260114ec1c.jpg"/></td><td><img src="images/e2f3e07a97c814e5006875a976858c85a0d531218ff22c37a7ce61a0d2ede626.jpg"/></td><td><img src="images/607d1ae530acf31235629f822f4d650e0268452c33cd01581c0052586f228ac3.jpg"/></td></tr><tr><td>“如家酒店”</td><td><img src="images/f20b8f649926e93b69820c59a2e5045b2ca0a108846414081b24b57eafa57126.jpg"/></td><td><img src="images/dc71ed63845658ff8edd0d7e3d7dc6128c26704d079aa9c7101b506fb453e6f3.jpg"/></td><td><img src="images/fc833de1cd6c36033e61d05194e5b6cd4ada58a4d4ed3e26de4e2a0b1697df82.jpg"/></td><td><img src="images/83f69970ef87c16602f60d8e93a3c080477b0a42b3b23e7d73d64c5b92fad5db.jpg"/></td><td><img src="images/0ec821bb54108c75feac8e2a541b2834fe8eee47b68f27890f7f9d332249cb4c.jpg"/></td></tr><tr><td>“麦当劳”</td><td><img src="images/2671db2a044ce5d63170fe32944a2e1afd3c7a7fae62a854b8d158914de835a9.jpg"/></td><td><img src="images/c0cf299a98072e66db4e273f8c49e020d219458f0010474502db4998d301e368.jpg"/></td><td><img src="images/0523e06e0126c3b846dc3358431f6cda9eab820cff989e6517a88134c6acebb4.jpg"/></td><td><img src="images/6ca1c98ed535a92c2a4acc0fe7a1b38aab78b8b9a5368653629fc55df0acc2bf.jpg"/></td><td><img src="images/5db722db47027e6e0c28ae29c1a0d4946833ad4a6426c480143e834c66c46d8c.jpg"/></td></tr><tr><td rowspan="3">(b)</td><td rowspan="3">DL-CSVTR-CL</td><td>“封顶大吉”</td><td><img src="images/43dbc8c7e34506d229bc7653896264bdf557d33d67aa1378f2aa14efea375b9d.jpg"/></td><td><img src="images/25378dc2f0072c46f113ff65d7b93d7db9d44b4f16c22f8aef9e783cac8de7ab.jpg"/></td><td><img src="images/a56732f3773c3b77e7711f66f6ab5afa100377a5ad8e696266cbd09f1e34ecc4.jpg"/></td><td><img src="images/3490d8282555e06a1eb6f8ea103093597b7c5d997ac2e1795fe0007a7db20e22.jpg"/></td><td><img src="images/d73fb5e0e5f0fa9ddb487f5297bd3ee1388010a25fe10344a3405da9b01f5284.jpg"/></td></tr><tr><td>“库迪咖啡”</td><td><img src="images/dfab628c11a3608a03e263f03d6e3973635f50e08710b131516be6e07844341d.jpg"/></td><td><img src="images/c4baad49396449264f6b71abfadd2af82c4378399da60ba4bc1162f8ce3a702a.jpg"/></td><td><img src="images/ac67ec1fcecaa67b9fdb3fb147bbf3e8d64dc7d1f9d53eef5330146c12709670.jpg"/></td><td><img src="images/0e00c08923e9fb6e3b016b2a3f65dbee977651d3eee5a722979361d90023d969.jpg"/></td><td><img src="images/c1f1d3dc931a5d62beb8d8d39848af4bb053d5c84e867b1e7048a2bead4ac85d.jpg"/></td></tr><tr><td>“西贝莜面村”</td><td><img src="images/4fc806c8ba6e05e63065b9d083c491b4f8f2d57e9ea27136cb2c34471e7cfc2e.jpg"/></td><td><img src="images/e45c93e06c4ff54ca0f5b5ea5b857e39c9ca4dde23f047e20142573d162cdfb3.jpg"/></td><td><img src="images/c0f16e17db931a3393491e588cbbe87219cf88e9d60ff8de4c69d92cdcb08701.jpg"/></td><td><img src="images/bb81ba9735f097a911f25f7b78ba052e1892199b719005a166f10de6cc6e6278.jpg"/></td><td><img src="images/a8525ea68db7293852e604bcc4df0cd9ce286fc7c8785e9ad9c8e2010955b6a4.jpg"/></td></tr><tr><td rowspan="3">(c)</td><td rowspan="3">DL-CSVTR-P</td><td>“安全”</td><td><img src="images/a52be41c6783a7f83ddefdcfc3c7ee988c699080eb1edbff8e3b246c59767a56.jpg"/></td><td><img src="images/db3d2b9366fb1ee67b9b69f19152f75328c843ec4e44845dfea8d81aa2c610df.jpg"/></td><td><img src="images/e6732e1d2843e42557970b4ec6fcb0e27dabf1cf27a86b48beca2579ceaad59b.jpg"/></td><td><img src="images/e5d87df238f68e981d82e9f79ca9222dd02a6708c1239e04e0c0ca6703abdcdf.jpg"/></td><td><img src="images/9b8b923bb7b81e57fb134c8a102b3383d9acca8d45931528939407346cbd5c7d.jpg"/></td></tr><tr><td>“超市”</td><td><img src="images/35e447460eb054a78ae21ec839460d77bdfb0b8950d24a416fd903963fc2ad81.jpg"/></td><td><img src="images/af133fcba3be43f5f5e5773a2efc7603a2afa4246ffc18b00c5f7c1ac497b6b9.jpg"/></td><td><img src="images/d8894d5ded32176295f3ce6494bed63f5de916631b34bf5c405f7da49d92bc61.jpg"/></td><td><img src="images/a0cd4c197f57f08294f7e3f0520d74b38f7e8f61932bf3fd0a4d0cc3ab871a94.jpg"/></td><td><img src="images/a746e5eb41373e7368da8130e545de7e7c5c23a309a7b30c2509850d6976be03.jpg"/></td></tr><tr><td>“咖啡”</td><td><img src="images/211b6c7b74c2078a33acb22d6acd503365729f8d680777cac0041d8fc1e85fe7.jpg"/></td><td><img src="images/c800ab776e605ed917b4feaabf8c078159aebb9af17a38340e7d7237a5387de6.jpg"/></td><td><img src="images/ae1bc1d5b59fac38c1e20908a78106edee97c5c19f1dc5ba9bf9e608ed63085a.jpg"/></td><td><img src="images/5b6500c7d2ac543ee0080c23b42a52516ba46b12b41f6c4d2ab1bbad0dc59070.jpg"/></td><td><img src="images/efae1cfe2a858c729eebe5a43763f33483c2b3639125259c9d62998a39ee79f8.jpg"/></td></tr></table>

Figure 10. Sample presentation of three benchmarks in DL-CSVTR. Best viewed in zoom.

<table><tr><td>#</td><td>Benchmark</td><td>Query</td><td>Method</td><td colspan="5">Retrieval Results</td></tr><tr><td rowspan="3">(a)</td><td rowspan="3">DL-CSVTR-V</td><td rowspan="3">“锦江之星”</td><td>Visual Embeding</td><td><img src="images/6802366447a0236bd66e9fa49510da5bc128011f2794bd7ba8bf3d90902db4d4.jpg"/></td><td><img src="images/41388da48ec8bea404245a9fa564ce9cb8a362ab1ceae53e52e1b6167bcfd0b6.jpg"/></td><td><img src="images/af093054972e9af38adb8fec415799f745f8b6031d20f9d8b6664d91143cd9d8.jpg"/></td><td><img src="images/51687f7036107a6740c459f2a390623eb6042e0c93baee974c06dd7ded165025.jpg"/></td><td><img src="images/dadaa525b77af3a505d7c405a7958ba9627929d018e4e8584d3a9a0b1748fdfa.jpg"/></td></tr><tr><td>Cross Model Embeding</td><td><img src="images/db3b8220478299d8135bf02645c4a92761cd9ad9583af0fdaeba54a1b6efc01f.jpg"/></td><td><img src="images/0192743c8b0da7ce8db937bf7665ee26518dc6b7d4869154a9f3919413cbd400.jpg"/></td><td><img src="images/335f55b9606a8264359a060f21257a0fecf8623e62788cb01f1f325d60d24c96.jpg"/></td><td><img src="images/f75df0252014e540d6db2b99ae94943266b3206845bc0a2aed6490537b0e869b.jpg"/></td><td><img src="images/c87c5782cfa15fad7a334a40677a5a6a060235252e41e026c866e8fdee73c996.jpg"/></td></tr><tr><td>CSTR-CLIP</td><td><img src="images/86b36aa8f8cf7477271f58e7134aeca76d923fbc3b78433df0458d2756da2579.jpg"/></td><td><img src="images/988ced4db646f5218a601e58a536a74d49ad396e7eeefe52633be075dbd6d0a3.jpg"/></td><td><img src="images/d8d6b1f0084c8f88d2c92ecae80c3cd32d31fd5ae7644bcbda3236805ddeb833.jpg"/></td><td><img src="images/8dc13b79cc345fd132403ad392573f6ee6def20841dcd0a348e4831f83f930d0.jpg"/></td><td><img src="images/9df8d2ece68791ff596fb9578ce8a7801e41ff5e3acc5219b239ab3a9e93c2e9.jpg"/></td></tr><tr><td rowspan="3">(b)</td><td rowspan="3">DL-CSVTR-CL</td><td rowspan="3">“锅圈食汇”</td><td>Visual Embeding</td><td><img src="images/9caa37d983c46a9272de16b595e45eb8d3e210eb9a6c72efea02f30e7fd15d32.jpg"/></td><td><img src="images/7a7ba9ec0463ad2a856bf5922e9d28fceeaed33933ea683cac13eeda07ce3338.jpg"/></td><td><img src="images/38abfab19c3903395ff7279dee2701f62c95a71a2242613b3760cc6766baae26.jpg"/></td><td><img src="images/164b9a78b1f6f3aa2e3a5525cd78ec8c659ac61929fbee9397d1fa43c717e83a.jpg"/></td><td><img src="images/df5184db9040dec1c17f0c3a7d89b3f791cd1610d10d0d1f1ff66781bcbdeb9d.jpg"/></td></tr><tr><td>Cross Model Embeding</td><td><img src="images/e66ca6a1296057f19c7aea71d672ca3c13f18d5066f4214795a9d217e6d5e6d4.jpg"/></td><td><img src="images/bd4e56b7e017ce1937311408d570c0fddbb80626935ae12fed0e90d554d329d5.jpg"/></td><td><img src="images/9c2f5e826a6f72f016297f7c8210a4cfec7bba969f27e2bed6354295851f65fb.jpg"/></td><td><img src="images/6928e4438d12d5c1663aa2f7fb374c23b444655b6586e180b27ea8b422fb751f.jpg"/></td><td><img src="images/97c5eaed999821c047fbebc00e2547b3119784eece9b8c62b85df3e51df2a358.jpg"/></td></tr><tr><td>CSTR-CLIP</td><td><img src="images/5c296f0f141d4b870704f8864d131d4e777e3e6abc3322661b0ae4c4de2e44f5.jpg"/></td><td><img src="images/14179a7fd6271b5bd9cc46397234e555651c06c4899dea50a993643f1d1ab319.jpg"/></td><td><img src="images/b2c53f42e018bab69c7caee94305086093144b0baf6f50fa7098d112a31ce06b.jpg"/></td><td><img src="images/d241a4e9e0806dd599a8e9c6f8f96f9a7f6dd11ba0f85262b4397d3b51cd9a43.jpg"/></td><td><img src="images/2f2ce06a56557177d55b8ee2632ddf3f970a569e839f53523d5524bc9611d563.jpg"/></td></tr><tr><td rowspan="3">(c)</td><td rowspan="3">DL-CSVTR-P</td><td rowspan="3">“网咖”</td><td>Visual Embeding</td><td><img src="images/af9392fbf7edd31d8ad08f667f71a3d22f4714a3417f20c288fedee28679886f.jpg"/></td><td><img src="images/bd622ae5f9a6ec1f9817a81b96ebc94cccdb68506e8442cf04b8b421ea0a2137.jpg"/></td><td><img src="images/0df158d9488daebe425221ba6eca0ffb0216814bfce58cb197d364c72c735d38.jpg"/></td><td><img src="images/dddd2da2501caeb5d99c7b4d4fd983ff4ec67b065507cce0bed21e5c9af42855.jpg"/></td><td><img src="images/bd32e66a55646056c82e1d53395d11f52406004d7b197f21d9140567205f5a1c.jpg"/></td></tr><tr><td>Cross Model Embeding</td><td><img src="images/b8e3d17e5eb9b8d186fc1ef90535697552338e5511dbc2f05a784b8e689616d2.jpg"/></td><td><img src="images/bdc27bb8c8219b77ed87324b6bf6d518bacde3ad9a90f1871a8a298c6d048a39.jpg"/></td><td><img src="images/f36945a8e9648eacb754e941afe2ea379de8bc047258e5ad64735befc93db82e.jpg"/></td><td><img src="images/f82a0e80c1f70bdab7370664c5279e8d6f88433b66e579871b9fdaf2b9d63b9e.jpg"/></td><td><img src="images/5df738a3690956a0619492c8c0923b635bb7cae9553eb83cf5816f5c99dbfd7f.jpg"/></td></tr><tr><td>CSTR-CLIP</td><td><img src="images/27610f8b7139d016155db23dfe2671abe6ce7517154f4ed018933771b9e8d3ee.jpg"/></td><td><img src="images/b2c94a8781b9d6b0985eeed03ea180ae6c71e8520fce842137870ac595bfeefd.jpg"/></td><td><img src="images/e60c1df6f30c89da349474fbdcb68e470b5be0ec9e47a4680f1b7b0727b39fa3.jpg"/></td><td><img src="images/db7fe335b62900719b5f4abc0a2c9268f148eaabb4171270e6fc150db8b5636a.jpg"/></td><td><img src="images/91c0c6eb01e9d3dad7c0dd0443edd9d978d166c54b29054d57b2bda4cbb01e93.jpg"/></td></tr></table>

Figure 11. Visualization of retrieval results. (a) An example on DL-CSVTR-V benchmark, in which rank@1-5 retrieval results are provided. (b) An example on DL-CSVTR-CL benchmark, in which rank@1-5 retrieval results are provided. (c) An example on DL-CSVTR-P benchmark, in which rank@1-5 retrieval results are provided. The correct results are highlighted in green while the incorrect ones are highlighted in red. Best viewed in zoom.

![](images/de76780613f84f8e174f6d02fa9db3134c49291ca2c723f529fa3f3dcc77d75c.jpg)

<details>
<summary>bar</summary>

AP Comparison of Different Models – DL-CSVTR-V
| Queries | Cross-Model Embedding | Visual Embedding | CSTR-CUP |
|---|---|---|---|
| 7天连锁酒店 | 0.73 | 0.33 | 0.63 |
| 七里山塘 | 0.85 | 0.49 | 0.91 |
| 中国建设银行 | 0.93 | 0.86 | 0.91 |
| 便携换乘 | 0.16 | 0.30 | 0.92 |
| 出入平安 | 0.65 | 0.67 | 0.88 |
| 南京大排档 | 0.39 | 0.34 | 0.81 |
| 夜泊秦淮 | 0.62 | 0.48 | 1.00 |
| 如家酒店 | 0.18 | 0.03 | 0.58 |
| 封顶大吉 | 0.89 | 0.92 | 0.88 |
| 汉庭酒店 | 0.54 | 0.69 | 0.79 |
| 海底捞火锅 | 0.80 | 0.51 | 0.94 |
| 自助银行服务 | 0.87 | 0.11 | 0.91 |
| 茉莉乌龙 | 0.93 | 0.90 | 1.00 |
| 金榜题名 | 0.56 | 0.28 | 0.69 |
| 银座佳驿酒店 | 0.31 | 1.00 | 0.65 |
| 麦当劳 | 0.94 | 0.73 | 1.00 |
| 一帆风顺 | 0.87 | 0.28 | 1.00 |
| 万达广场 | 0.41 | 0.72 | - |
| 亚朵酒店 | 0.79 | 0.39 | 0.90 |
| 全季酒店 | 0.73 | 0.05 | 0.93 |
| 半天妖 | 0.91 | 1.00 | 0.97 |
| 吾悦广场 | 0.54 | 0.15 | 0.83 |
| 太古里 | 0.42 | 0.13 | 0.35 |
| 家和万事兴 | - | 0.65 | - |
| 招商银行 | - | - | - |
| 洗手间 | - | - | - |
| 饷子街 | - | - | - |
| 茂昌眼镜 | - | - | - |
| 遠8酒店 | - | - | - |
| 金足印象 | - | - | - |
| 锐江之星 | - | - | - |
The chart displays AP values for three different models (Cross-Model Embedding, Visual Embedding, CSTR-CUP) across various bank queries and locations.
</details>

Figure 12. Comparison of AP of each query word under the DL-CSVTR-V benchmark.

![](images/be1c095931619036c9415757567b881fe43bd01108441569a05ca1438552dd80.jpg)

<details>
<summary>bar</summary>

AP Comparison of Different Models – DL-CSVTR-CL
| Queries | Cross-Model Embedding | Visual Embedding | CSTR-CUP |
|---|---|---|---|
| 脂榆炸鸡腿 | 0.80 | 0.60 | 0.91 |
| 保持车距 | 0.06 | 0.25 | 0.25 |
| 大吉大利 | 0.22 | 0.12 | 0.09 |
| 年年有余 | 0.30 | 0.05 | 0.64 |
| 早生贯子 | 0.03 | 0.02 | 0.07 |
| 清仓甩费 | 0.51 | 0.07 | 0.22 |
| 米村拌饭 | 0.39 | 0.08 | 0.87 |
| 金榜题名 | 0.25 | 0.49 | 0.19 |
| 青考加油 | 0.17 | 0.51 | 0.39 |
| 乔治莫兰迪 | 0.33 | 0.47 | 0.75 |
| 出入平安 | 0.21 | 0.02 | 0.44 |
| 杀雪的茶 | 0.71 | 0.33 | 0.98 |
| 库迦咖啡 | 0.29 | 0.29 | 0.35 |
| 欢度国庆 | 0.76 | 0.22 | 1.00 |
| 瑞王花生 | 0.67 | 0.10 | 0.96 |
| 薛记炒货 | 0.75 | 0.25 | 0.95 |
| 锦圈食汇 | 0.61 | 0.07 | 1.00 |
| 养迁之意 | 0.36 | 0.61 | 0.68 |
| 名创优品 | 0.46 | 0.94 | 0.97 |
| 封顶大吉 | 0.31 | 0.79 | 0.94 |
| 开业大吉 | 0.31 | 0.73 | 0.42 |
| 欢迎光临 | 0.15 | 0.66 | 0.33 |
| 瑞华咖啡 | 0.80 | 0.74 | 0.75 |
| 西贝莜面村 | 0.41 | 0.33 | 0.97 |
| 阿水大杯茶 | 0.55 | 0.03 | 0.98 |
| 买一送一 | 0.28 | 0.02 | 0.48 |
| 夜边兼淮 | 0.03 | 0.05 | 1.00 |
| 小心地满 | 0.12 | 0.02 | 0.21 |
| 新年快乐 | 0.31 | 0.68 | 0.36 |
| 欢迎回家 | 0.63 | 0.02 | 1.00 |
| 生日快乐 | 0.55 | 0.82 | 0.56 |
| 金桂乌龙 | 0.71 | 0.18 | 1.00 |
| 领鲜全域 | 0.72 | 0.67 | 1.00 |
</details>

Figure 13. Comparison of AP of each query word under the DL-CSVTR-CL benchmark.

![](images/64815d9f01576a08689657bf61c2b0ac37d5a85ba31e834287a51e3521fb4436.jpg)

<details>
<summary>bar</summary>

AP Comparison of Different Models – DL-CSVTR-P
| Queries | Cross-Model Embedding | Visual Embedding | CSTR-CUP |
|---|---|---|---|
| 中心 | 0.07 | 0.60 | 0.40 |
| 会议 | 0.10 | 0.23 | 0.30 |
| 便利 | 0.13 | 0.16 | 0.54 |
| 动物 | 0.06 | 0.28 | 0.73 |
| 医院 | 0.24 | 0.27 | 0.63 |
| 协会 | 0.08 | 0.15 | 0.44 |
| 卫生 | 0.22 | 0.47 | 0.74 |
| 台球 | 0.19 | 0.40 | 0.98 |
| 咖啡 | 0.25 | 0.87 | 0.61 |
| 垃圾 | 0.06 | 0.16 | 0.83 |
| 大学 | 0.16 | 0.35 | 0.59 |
| 学院 | 0.08 | 0.32 | 0.49 |
| 安全 | 0.16 | 0.13 | 0.74 |
| 小学 | 0.16 | 0.64 | 0.72 |
| 广场 | 0.06 | 0.08 | 0.27 |
| 文明 | 0.21 | 0.07 | 0.51 |
| 服务 | 0.22 | 0.22 | 0.31 |
| 研究 | 0.06 | 0.45 | 0.31 |
| 社区 | 0.12 | 0.11 | 0.48 |
| 网咖 | 0.72 | 0.37 | 1.00 |
| 药房 | 0.51 | 0.15 | 0.91 |
| 超市 | 0.17 | 0.42 | 0.69 |
| 酒店 | 0.18 | 0.45 | 0.52 |
| 饭店 | 0.35 | 0.11 | 0.78 |
| 高考 | 0.76 | 0.12 | 0.81 |
</details>

Figure 14. Comparison of AP of each query word under the DL-CSVTR-P benchmark.

![](images/38faa2ba7b5e1be9d85c25457cf28b0ace06f399d748ac405b3b2863a79e95d9.jpg)

<details>
<summary>bar</summary>

AP Comparison of Different Models – CSVTR
| Queries | Cross-Model Embedding | Visual Embedding | CSTR-CUP |
|---|---|---|---|
| 7天连锁酒店 | 0.95 | 0.90 | 0.95 |
| 零芝林大药房 | 0.88 | 0.91 | 0.91 |
| 北京华联 | 0.57 | 0.71 | 0.80 |
| 微妹火锅 | 0.94 | 0.90 | 0.96 |
| 汉庭酒店 | 0.83 | 0.77 | 0.90 |
| 沪上网娱 | 0.91 | 0.90 | 0.89 |
| 华莱士 | 0.88 | 0.89 | 0.96 |
| 黄润鸡米饭 | 0.82 | 0.83 | 0.89 |
| 绝味鸭脖 | 0.91 | 0.83 | 0.96 |
| 兰州拉面 | 0.88 | 0.77 | 0.90 |
| 蜀雪冰城 | 0.87 | 0.90 | 0.91 |
| 她家酒店 | 0.89 | 0.78 | 0.94 |
| 世纪联华 | 0.72 | 0.83 | 0.92 |
| 洋河蓝色经典 | 0.82 | 0.83 | 0.80 |
| 永和豆浆 | 0.80 | 0.85 | 0.89 |
| 正新鸿捷 | 0.95 | 0.95 | 0.99 |
| 中国电信 | 0.86 | 0.87 | 0.90 |
| 中国工商银行 | 0.81 | 0.70 | 0.82 |
| 中国建设银行 | 0.77 | 0.67 | 0.85 |
| 中国农业银行 | 0.88 | 0.86 | 0.91 |
| 中国银行 | 0.51 | 0.29 | 0.59 |
| 中国邮政储蓄银行 | 0.77 | 0.71 | 0.88 |
| 注意安全 | 0.65 | 0.83 | 0.91 |
</details>

Figure 15. Comparison of AP of each query word under the CSVTR benchmark.