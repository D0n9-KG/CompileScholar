# Explore In-Context Learning for 3D Point Cloud Understanding

Zhongbin Fang $^{1}$ , Xiangtai Li $^{2}$ , Xia Li $^{3}$ , Joachim M. Buhmann $^{3}$ , Chen Change Loy $^{2}$ Mengyuan Liu $^{4}$

$^{1}$ Sun Yat-sen University $^{2}$ S-Lab, Nanyang Technological University $^{3}$ Department of Computer Science, ETH Zurich

$^{4}$ Key Laboratory of Machine Perception, Shenzhen Graduate School, Peking University fangzhb5@mail2.sysu.edu.cn, xiangtai.li@ntu.edu.sg, xia.li@inf.ethz.ch ccloy@ntu.edu.sg, liumengyuan@pku.edu.cn

https://github.com/fanglaosi/Point-In-Context

![](images/aab849485ac1915e29babff639fe4e5a7bae9324a3fc1c6d45b270032b2ccc3c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["(a) Language understanding"] --> B["GPT-3"]
    B --> C["Translation"]
    C --> D["I love ice cream"]
    E["Delicious food! Terrible dishes!"] --> F["Positive"]
    F --> G["Sentiment"]
    G --> H["Negative"]
    
    I["(b) 2D image inpainting"] --> J["MAE-VQGAN"]
    J --> K["Segmentation"]
    K --> L["Inpainting"]
    
    M["(c) 3D point cloud modeling"] --> N["Reconstruction"]
    N --> O["Point-In-Context"]
    O --> P["Registration"]
    P --> Q["Denoising"]
    Q --> R["Mask"]
```
</details>

Figure 1: (a) In-context learning in NLP [6], with different text prompts for corresponding tasks: translation and sentiment analysis. (b) In-context learning in 2D vision [4], with 2D visual prompts for different tasks: segmentation and inpainting. (c) Our proposed in-context learning for 3D point clouds, with 3D visual prompts for different tasks: reconstruction, denoising, registration, etc.

# Abstract

With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions (coordinates) that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks.

# 1 Introduction

In recent years, large-scale models $[1, 32, 15, 5]$ with enormous parameters pre-trained on broad data have emerged in computer vision and natural language processing. Capable of handling diverse tasks simultaneously, these models can be adapted for new tasks when prompted. Text-to-image generation models such as DALL-E $[34]$ and language models like GPT $[33]$ are examples of this development, representing significant progress toward general intelligence. However, training these models is resource-intensive, which makes full fine-tuning $[18, 21, 20]$ or even parameter-efficient tuning techniques $[50, 17, 25]$ such as prompt tuning impractical for many users.

In-context learning, originating from natural language processing (NLP) $[33, 35, 6]$ , holds potential as the mainstream approach for efficient model adaptation and generalization. Unlike other methods that necessitate model parameter updates for new tasks, in-context learning incorporates domain-specific input-output pairs, known as in-context examples or prompts, into a test example. In NLP, the prompt can be a machine translation pair or sentiment analysis, as shown in Fig. 1 (a). This allows the model to produce optimal results without requiring any parameter updates for previous tasks. Several works $[4, 38, 39, 48]$ explore in-context learning in computer vision. The visual prompt $[4]$ is the first to adopt a pre-trained neural network for filling missing patches in grid-like images, as shown in Fig. 1(b). Other studies investigate the effect of visual prompts or generalization to more vision tasks. The above methods adopt Mask Image Modeling (MIM) architecture $[13, 3]$ for in-context task transfer. Inspired by MIM, Masked Point Modeling (MPM) is proposed recently and is widely used in different point cloud tasks $[44]$ . To our knowledge, no work has explored in-context learning for 3D point cloud understanding using the MPM framework.

Our work is the first to explore in-context learning for 3D point cloud understanding, as shown in Fig. 1(c). Given the absence of previous works, we propose a benchmark based on the ShapeNet [7] and ShapeNetPart [42], encompassing four different tasks: point cloud reconstruction, denoising, registration, and part segmentation. Meanwhile, we benchmark several representative baselines [27, 40, 26, 12], including individual models for each task and the models equipped with shared backbone with multiple task heads. Then, to tackle the benchmark mentioned above, we present the Point-In-Context (PIC) to explore in-context learning for 3D point clouds.

A straightforward extension of the 2D MIM architecture to point clouds for in-context learning may encounter two primary obstacles. First, using the conventional position embedding approach could potentially lead to information leakage, attributed to the use of invisible center points $^{1}$ . Second, unlike 1D word embeddings or 2D images, 3D point cloud data, which are inherently unordered [27], present the risk of having their positional information become disarrayed when partitioned into a patch sequence. Consequently, it is indispensable to devise a new sampling and grouping strategy for 3D point cloud data. In response to this issue, we propose a simple yet effective solution, termed joint sampling. This technique involves recording the indices of sampled center points and using the K-nearest neighbor strategy to sample both the input and target concurrently. In addition, leveraging the mask point transformer architecture, we explore two distinct baseline methodologies for PIC, which encompass separating inputs and targets akin to the Painter strategy [38] and concatenating inputs and targets in a manner analogous to the MAE approach [13] for reconstruction. Contrary to the focus of few-shot learning on a single specific task, our objective is to explore the in-context ability of the generative model, facilitating the execution of tasks commensurate with the 3D prompt.

Our main contributions are as follows. 1) We introduce a simple yet effective general framework for 3D visual prompting. Given two pairs of point clouds, we demonstrate that multiple 3D point cloud tasks can be treated as task-aware prompting, as seen in 2D image and NLP tasks. 2) We create a new benchmark, including four different point cloud tasks for 3D in-context learning, and evaluate several representative baselines. 3) Our comprehensive study addresses various 3D prompt examples, sampling methods, and masking strategies. Moreover, we show that the choice of in-context examples greatly impacts performance for 3D in-context learning.

# 2 Related Work

3D Point Cloud Classification. Deep neural networks have been proposed for the 3D point cloud classification task. Both PointNet [27] and its improved versions [28, 30] are pioneers of point-based

methods in 3D point cloud analysis, which adopt multi-layer perceptron (MLP) to handle point clouds directly. Several graph-based methods $[40, 22, 16]$ exploit geometric properties and propose different dynamic kernels. In particular, DGCNN $[40]$ designs EdgeConv, which dynamically computes each layer output. Recently, several works $[49]$ adopt pure transformer-based architecture to model the global context. Point Transformer $[49, 12, 43]$ applies the vectorized self-attention mechanism to construct a point Transformer layer for 3D point cloud learning. Meanwhile, PointMLP $[24]$ directly applies a pure residual MLP network to the 3D point cloud analysis. Recently, several works have explored the vision language models $[46]$ and joint 2D and 3D training $[31, 47]$ via transformer architectures. However, these approaches are only designed for a single task, so they cannot be used directly for in-context learning.

Masked Image Modeling (MIM) For 2D and 3D Vision. The GPT and BERT series $[8]$ have greatly enhanced natural language processing performance through masked modeling and fine-tuning downstream tasks. BEiT $[3]$ is the first to propose matching image patches with discrete tokens via d-VAE $[34]$ and pre-train a standard vision transformer $[37, 10]$ using masked image modeling $[8]$ . MAE $[13]$ then directly reconstructs the raw pixel values of masked tokens and achieves high efficiency with a high mask ratio. For 3D point cloud pre-training using MIM, several works $[44, 26, 45, 47, 9, 29, 19]$ aim to improve feature representation. Point-BERT $[44]$ adopts a BERT-like architecture, while Point-MAE $[26]$ transfers the MAE-like framework for pre-training. These methods use standard transformer networks to process 3D point clouds and achieve competitive performance on various downstream tasks. Our approach follows a similar point MIM pipeline but explores the in-context ability of point transformers and MIM, which has not been investigated previously.

In-Context Learning. In-context learning $[33, 35, 6]$ is a new learning paradigm in large language models like GPT-3. This paradigm enables an autoregressive language model to perform inference on unseen tasks by conditioning the input on specific input-output pairs, known as "context." This powerful paradigm allows users to customize a model's output to fit their downstream datasets without modifying the often inaccessible internal model parameters. Recent research in natural language processing has demonstrated the efficacy of in-context learning across a range of language tasks, including machine translation, sentiment analysis, and question-answering. Recently, several works $[4, 38, 39, 48, 36, 2]$ explore in-context learning in computer vision. The visual prompt $[4]$ is the first pure vision model, which was pre-trained to fill missing patches in images that are made of academic figures and infographics. Then, Painter $[38]$ generalizes visual in-context learning as learning to paint the image via different task prompts. The following works explore the effect of task prompts. Recently, SegGPT $[39]$ extends Painter by learning a generalized one-shot segmentation. In contrast, our method explores the effect of 3D prompts on in-context learning in the point cloud and proposes new baselines for benchmarking 3D in-context learning.

# 3 Method

In this section, we first introduce the task settings of in-context learning in the 3D point cloud, which is motivated by 2D visual in-context learning. Then, we elaborate on the dataset construction and task definitions. Next, based on the MPM framework, we point out the information leakage issue and propose a joint sampling strategy. Finally, we build two distinct baselines using the MPM training methodology.

# 3.1 Modeling In-Context Learning in 3D Point Cloud

In-context Learning in 2D. During training, the 2D in-context learning models [4] take two pairs as inputs, including reference pair (or task prompt pair), $R_{i}^{k} = \{I_{i}, T_{i}^{k}\}$ and query inputs, $Q_{j}^{k} = \{I_{j}, T_{j}^{k}\}$ , where $R_{i}^{k}$ is the task prompt containing one image $I_{i}$ and one target $T_{i}^{k}$ . $Q_{j}^{k}$ represent current input image $I_{j}$ . Here, $k$ represents the task index while $i$ and $j$ indicate different example indexes. When $i = j$ , we term the prompt as an ideal prompt. During training, both Visual Prompt [4] and Painter [38] combine two pairs of images that perform the same task into a grid-like image and randomly mask portions, following MAE [13]. During the inference stage, only example pairs and a query image are provided. They are combined into a grid-like image with a quarter mask to mask the $T_{j}^{k}$ , and a pre-trained model is used to restore the missing parts.

![](images/73c57fc27a3de34b0ea6a1a35807cc54ead2d9f9d9359a58daca04bcbc7c78ef.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Centers"]
    B --> C["Nos Embedding"]
    C --> D["Transformer"]
    D --> E["..."]
    
    subgraph (a) Masked Point Modeling for pre-training in previous work
        F1["Input"] --> G1["Target"]
        G1 --> H["Joint Sampling Module"]
        H --> I["Consistent"]
        I --> J["Output"]
    end
    
    subgraph (b) Our Proposed Joint sampling module
        K1["Input"] --> L1["Target"]
        L1 --> M["Conistent"]
        M --> N["Embedding"]
        N --> O["Separate input and target"]
        O --> P["Output"]
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F1 fill:#cff,stroke:#333
    style G1 fill:#ffc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#ffc,stroke:#333
    style K1 fill:#cff,stroke:#333
    style L1 fill:#ffc,stroke:#333
    style M1 fill:#ffc,stroke:#333
    style N1 fill:#ffc,stroke:#333
    style O1 fill:#ffc,stroke:#333
    style P1 fill:#ffc,stroke:#333
```
</details>

Figure 2: (a) The pre-training pipeline used in previous works. When performing Masked Point Modeling (MPM), these works $[44, 26, 45]$ use the center position of the target patches for position embedding, which results in information leakage. (b) Joint Sampling module. When selecting the center points for the input and target point clouds, the same indexes are used, which are sampled from the input point cloud. (c)(d) Different combination forms of input and target point clouds, which respectively denote the input form of our two baselines, PIC-Sep and PIC-Cat.

In-context Learning in 3D Point Cloud. Motivated by 2D in-context learning, we design a similar procedure for 3D in-context learning. During training, each input sample contains two pairs of point clouds that perform the same task as in 2D-context learning. Each pair consists of an input point cloud and its corresponding output point cloud for the given task. Similar to PointMAE [26], we adopt the farthest point sampling and K-nearest neighbor (KNN) techniques to convert the point clouds into a sentence-like data format. These point patches are subsequently encoded into tokens. During the inference, the input point cloud is a combination of example input and query point cloud, while the target point cloud consists of an example target along with masked tokens, as shown in Fig. 1(c). Based on different $R_{i}^{k}$ , given input $P_{j}^{k}$ , the model outputs a corresponding target $T_{j}^{k}$ .

# 3.2 Dataset and Tasks Definition

ShapeNet In-Context Datasets. Since there is no previous benchmark for 3D in-context learning, in order to establish the first benchmark for 3D in-context learning, we carefully curate datasets and define task specifications. Firstly, we obtain samples from publicly available datasets, such as ShapeNet [7], ShapeNetPart [42] and transform them into the "input-target" format as stated in Sec.3.1. Additionally, to augment the sample size for the part segmentation task, we conduct several random operations, including point cloud perturbation, rotation, and scaling, on the ShapeNetPart. Consequently, we construct an extensive dataset encompassing all four types of tasks (mentioned below), comprising 217,454 samples. Each sample comprises an input point cloud and its corresponding target for a specific task. Then, we standardize the inputs and outputs to solely contain only the point cloud's XYZ coordinates. For reconstruction and denoising tasks, our aim is to create a clean and aligned point cloud; whereas for the registration task, our aim is to create a clean, registered point cloud. Additionally, the output of the part segmentation task is several point clusters, representing different parts of the object.

Reconstruction. The objective of this task is to reconstruct a complete dense point cloud using only a sparse set of points. To evaluate the model's reconstruction capability, we establish five levels for input point clouds, which contain 512, 256, 128, 64, and 32 points respectively.

Denoising. In this task, the input consists of a point cloud with Gaussian noise. The objective is to remove the noise surrounding the point cloud, resulting in a clear and distinct object shape. To evaluate the model's performance across different noise levels, we establish five noise levels ranging from 100 to 500 noisy points.

Registration. The objective of this task is to restore a rotated point cloud to its original orientation. It is assumed that during both training and inference, the query point cloud and prompt are synchronized in terms of the rotation angle. To avoid coupling with the outputs of the denoising and reconstruction

![](images/21b13a1604b67aa3d46e3c0592dc4b08c359a0b33594ca3b433cc2c204db755e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pairs-wise Point clouds"] --> B["Input"]
    B --> C["Target"]
    C --> D["Embed Concat"]
    D --> E["Random mask"]
    E --> F["Transformer"]
    F --> G["Output"]
    G --> H["GT"]
    
    subgraph Inputs
        I1["Input Image"] --> I2["Input Image"]
        I3["Input Image"] --> I4["Input Image"]
        I5["Input Image"] --> I6["Input Image"]
        I7["Input Image"] --> I8["Input Image"]
        I9["Input Image"] --> I10["Input Image"]
        I10 --> J1["Prompt Input"]
        I10 --> J2["Prompt Target"]
        I10 --> J3["Query Input"]
        I10 --> J4["Query Target"]
        I10 --> J5["Masked Token"]
    end
    
    subgraph Transformers
        K1["Prompt Mask"] --> K2["Transformer"]
        K2 --> K3["Reconstruction"]
        K2 --> K4["Denoising"]
        K2 --> K5["Registration"]
        K2 --> K6["Part Segmentation"]
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I1 fill:#ffc,stroke:#333
    style I2 fill:#ffc,stroke:#333
    style I3 fill:#ffc,stroke:#333
    style I4 fill:#ffc,stroke:#333
    style I5 fill:#ffc,stroke:#333
    style I6 fill:#ffc,stroke:#333
    style I7 fill:#ffc,stroke:#333
    style I8 fill:#ffc,stroke:#333
    style J1 fill:#ffc,stroke:#333
    style J2 fill:#ffc,stroke:#333
    style J3 fill:#ffc,stroke:#333
    style J4 fill:#ffc,stroke:#333
    style J5 fill:#ffc,stroke:#333
    style J6 fill:#ffc,stroke:#333
    style K1 fill:#ffc,stroke:#333
    style K2 fill:#ffc,stroke:#333
    style K3 fill:#ffc,stroke:#333
    style K4 fill:#ffc,stroke:#333
    style K5 fill:#ffc,stroke:#333
    style K6 fill:#ffc,stroke:#333
```
</details>

Figure 3: Overall scheme of our Point-In-Context. Top: Training pipeline of the Masked Point Modeling (MPM) framework. During training, each sample comprises two pairs of input and target point clouds that tackle the same task. These pairs are fed into the transformer model to perform the masked point reconstruction task, which follows a random masking process. Bottom: In-context inference on multitask. Our Point-In-Context could infer results on various downstream point cloud tasks, including reconstruction, denoising, registration, and part segmentation.

tasks, we set the registration task's output as both an upright and an upside-down point cloud. To provide a comprehensive evaluation, we define five levels for the registration task. As the level increases, the range of optional rotation angles also increases.

Part Segmentation. The goal of this task is to segment an object into several components, typically 2-6 components. Conventionally, a C-dimensional one-hot code is assigned to every point to classify them, where P is the total count of categories. However, for in-context learning, we need to keep the input and output in the same space, which only contains XYZ coordinates, making it a regression task. To achieve this, we convert the P component labels to P discrete points containing XYZ coordinates that are uniformly distributed within a cube. Thus, the output of this task is clusters of points.

# 3.3 Point-In-Context Models

MPM For 3D In-Context Learning. Following previous works $[4, 38]$ , we adopt the masked point modeling framework for point clouds and propose Point-In-Context (PIC), where we treat the points in MPM as image tokens in MIM, allowing us to leverage the transformer for processing both types of data. As shown in Fig 3, during training, we aim to reconstruct the masked points, aided by the input and visible target 3D points. During the inference stage, given query point clouds as input, depending on the task prompted by the example pair, it can generate the corresponding target, such as reconstruction, denoising, registration, or part segmentation outputs of the query point cloud.

Information Leakage. Though MPM $[26, 45]$ exists as a base framework for us, simply adapting it to point clouds is not feasible. As shown in Fig. 2(a), previous pre-training pipelines embed positional information with the center point coordinates of all patches, even for those that are masked out (invisible). Since the patches masked out in the target are invisible in our setting, such an operation will cause information leakage, which does not satisfy the requirements. Furthermore, we find that the sine-cosine encoding sequence will significantly reduce the model performance compared to learned embedding, and even lead to the collapse of the training. The reason is that valuable position information is missing, making it impossible for the model to locate the patches that need to be reconstructed during the processing. Unlike 2D images, 3D point cloud patch sequences have no fixed position (unordered property $[27]$ ), so we need to align the patch sequences generated from the input and target point clouds.

Joint Sampling (JS) Module. To handle the above issues, we collect N central points from each input point cloud and retrieve their indexes, which we then use to obtain the central points of every patch in both the input and target point clouds. The process is shown in Fig. 2(b). The key of our JS module is the consistency between center point indices of corresponding patches in both target and input point clouds. In other words, the order of the input token sequence and the target token sequence are well-aligned. Such a design compensates for the missing positional embedding of the

Table 1: Comparison of task-specific, multitask, and in-context learning models on four 3D point cloud tasks. For reconstruction, denoising, and registration, we report Chamfer Distance [11] loss (x1000). For part segmentation, we report mIOU. 

<table><tr><td rowspan="2">Models</td><td rowspan="2">Venues</td><td colspan="6">Reconstruction CD ↓</td><td colspan="6">Denoising CD ↓</td><td colspan="6">Registration CD ↓</td><td>Part Seg.</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>mIOU↑</td></tr><tr><td colspan="21">Task-specific models (trained separately)</td></tr><tr><td>PointNet [27]</td><td>CVPR&#x27;17</td><td>3.7</td><td>3.7</td><td>3.8</td><td>3.9</td><td>4.1</td><td>3.9</td><td>4.1</td><td>4.0</td><td>4.1</td><td>4.0</td><td>4.2</td><td>4.1</td><td>5.3</td><td>5.9</td><td>6.9</td><td>7.7</td><td>8.5</td><td>6.9</td><td>77.45</td></tr><tr><td>DGCNN [40]</td><td>TOG&#x27;19</td><td>3.9</td><td>3.9</td><td>4.0</td><td>4.1</td><td>4.3</td><td>4.0</td><td>4.7</td><td>4.5</td><td>4.6</td><td>4.5</td><td>4.7</td><td>4.6</td><td>6.2</td><td>6.7</td><td>7.3</td><td>7.4</td><td>7.7</td><td>7.1</td><td>76.12</td></tr><tr><td>PCT [12]</td><td>CVM&#x27;21</td><td>2.4</td><td>2.4</td><td>2.5</td><td>2.6</td><td>3.0</td><td>2.6</td><td>2.3</td><td>2.2</td><td>2.2</td><td>2.2</td><td>2.3</td><td>2.2</td><td>5.3</td><td>5.7</td><td>6.3</td><td>6.9</td><td>7.2</td><td>6.3</td><td>79.46</td></tr><tr><td>ACT [9]</td><td>ICLR&#x27;21</td><td>2.4</td><td>2.5</td><td>2.3</td><td>2.5</td><td>2.8</td><td>2.5</td><td>2.2</td><td>2.3</td><td>2.2</td><td>2.3</td><td>2.5</td><td>2.3</td><td>5.1</td><td>5.6</td><td>5.9</td><td>6.0</td><td>7.0</td><td>5.9</td><td>81.24</td></tr><tr><td colspan="21">Multitask models: share backbone + multi-task heads</td></tr><tr><td>PointNet [27]</td><td>CVPR&#x27;17</td><td>87.2</td><td>86.6</td><td>87.3</td><td>90.8</td><td>92.2</td><td>88.8</td><td>17.8</td><td>22.0</td><td>25.6</td><td>30.4</td><td>33.2</td><td>25.8</td><td>25.4</td><td>22.6</td><td>24.9</td><td>25.7</td><td>26.9</td><td>25.1</td><td>15.33</td></tr><tr><td>DGCNN [40]</td><td>TOG&#x27;19</td><td>38.8</td><td>36.6</td><td>37.5</td><td>37.9</td><td>42.9</td><td>37.7</td><td>6.5</td><td>6.3</td><td>6.5</td><td>6.4</td><td>7.1</td><td>6.5</td><td>12.5</td><td>14.9</td><td>17.9</td><td>19.7</td><td>20.7</td><td>17.1</td><td>16.95</td></tr><tr><td>PCT [12]</td><td>CVM&#x27;21</td><td>34.7</td><td>44.1</td><td>49.9</td><td>50.0</td><td>52.3</td><td>46.2</td><td>11.2</td><td>10.3</td><td>10.7</td><td>10.2</td><td>10.5</td><td>10.6</td><td>24.4</td><td>26.0</td><td>29.6</td><td>32.8</td><td>34.7</td><td>29.5</td><td>16.71</td></tr><tr><td>Point-MAE [26]</td><td>ECCV&#x27;22</td><td>5.5</td><td>5.5</td><td>6.1</td><td>6.4</td><td>6.4</td><td>6.0</td><td>5.6</td><td>5.4</td><td>5.6</td><td>5.5</td><td>5.8</td><td>5.6</td><td>11.4</td><td>12.8</td><td>14.8</td><td>16.0</td><td>16.9</td><td>14.5</td><td>5.42</td></tr><tr><td>ACT [9]</td><td>ICLR&#x27;23</td><td>7.4</td><td>6.6</td><td>6.5</td><td>6.6</td><td>7.0</td><td>6.8</td><td>7.3</td><td>6.8</td><td>7.0</td><td>6.8</td><td>7.2</td><td>7.0</td><td>12.2</td><td>14.4</td><td>19.4</td><td>25.5</td><td>29.0</td><td>20.1</td><td>12.08</td></tr><tr><td>I2P-MAE [47]</td><td>CVPR&#x27;23</td><td>17.0</td><td>16.0</td><td>16.7</td><td>17.2</td><td>18.5</td><td>17.2</td><td>20.6</td><td>20.4</td><td>20.1</td><td>18.3</td><td>18.8</td><td>19.6</td><td>32.5</td><td>31.3</td><td>31.1</td><td>31.6</td><td>31.2</td><td>31.5</td><td>22.60</td></tr><tr><td>ReCon [29]</td><td>ICML&#x27;23</td><td>12.4</td><td>12.1</td><td>12.4</td><td>12.5</td><td>13.1</td><td>12.5</td><td>20.4</td><td>24.5</td><td>27.2</td><td>29.2</td><td>32.5</td><td>26.9</td><td>14.7</td><td>16.3</td><td>19.2</td><td>21.5</td><td>22.5</td><td>18.8</td><td>7.71</td></tr><tr><td colspan="21">In-context learning models</td></tr><tr><td>Copy</td><td></td><td>155</td><td>153</td><td>152</td><td>156</td><td>155</td><td>154</td><td>149</td><td>155</td><td>157</td><td>155</td><td>155</td><td>154</td><td>155</td><td>157</td><td>156</td><td>148</td><td>154</td><td>154</td><td>24.18</td></tr><tr><td>Point-BERT [44]</td><td>CVPR&#x27;22</td><td>288</td><td>285</td><td>292</td><td>286</td><td>308</td><td>292</td><td>292</td><td>293</td><td>298</td><td>296</td><td>299</td><td>296</td><td>291</td><td>295</td><td>294</td><td>295</td><td>298</td><td>294</td><td>0.65</td></tr><tr><td>Our PIC-Cat</td><td></td><td>3.2</td><td>3.6</td><td>4.6</td><td>4.9</td><td>5.5</td><td>4.3</td><td>3.9</td><td>4.6</td><td>5.3</td><td>6.0</td><td>6.8</td><td>5.3</td><td>10.0</td><td>11.4</td><td>13.8</td><td>16.9</td><td>18.6</td><td>14.1</td><td>78.95</td></tr><tr><td>Our PIC-Sep</td><td></td><td>4.7</td><td>4.3</td><td>4.3</td><td>4.4</td><td>5.7</td><td>4.7</td><td>6.3</td><td>7.2</td><td>7.9</td><td>8.2</td><td>8.6</td><td>7.6</td><td>8.6</td><td>9.2</td><td>10.2</td><td>11.3</td><td>12.4</td><td>10.3</td><td>74.95</td></tr></table>

target while avoiding information leakage. Therefore, it facilitates the model to learn the inherent association between input and target and streamlines the learning process. Subsequently, all point clouds search for neighborhoods containing M points based on the center points corresponding to each patch.

Point-In-Context Model Architecture. We use a standard transformer with an encoder-decoder structure as the backbone of our Point-In-Context, and a simple $1 \times 1$ convolutional layer as the task head for point cloud reconstruction. Inspired by Painter [38] and MAE [13], we explore two different baselines for PIC and name them PIC-Sep and PIC-Cat. For PIC-Sep, we take the input and masked target point clouds parallel to the transformer and then merge their features after several blocks, using a simple average for the fusion operation. For PIC-Cat, we concatenate the input and target to form a new point cloud. Then we mask it globally and feed it to the transformer for prediction. We denote the prompt pair as $R_{i}^{k} = \{P_{i}, T_{i}^{k}\}$ and query inputs, $Q_{j}^{k} = \{P_{j}, T_{j}^{k}\}$ , then PIC-Sep and PIC-Cat can be formalized as:

$$
P ^ {\text { Sep }} = \text { Transformer } ([ P _ {i} \| P _ {j} ], ([ T _ {i} ^ {k} \| T _ {j} ^ {k} ], M)), \tag {1}
$$

$$
P ^ {\text { Cat }} = \text { Transformer } ([ I _ {i} \| T _ {i} ^ {k} \| I _ {j} \| T _ {j} ^ {k} ], M), \tag {2}
$$

where $\parallel$ is the concatenate operation, and $M$ is the masked token to replace the invisible token. These two input forms are shown in Fig. 2(c)(d).

Loss Function. The model is trained to reconstruct the masked point patches. To this end, we use the $\ell_{2}$ Chamfer Distance as the training loss. Specifically, we calculate the Chamfer Distance between each predicted patch P and its corresponding ground truth G.

$$
\mathcal {L} (P, G) = \sum_ {p \in P} \min _ {g \in G} \| p - g \| _ {2} ^ {2} + \sum_ {g \in G} \min _ {p \in P} \| p - g \| _ {2} ^ {2} \tag {3}
$$

# 4 Experiments

Implementation Details. We sample 1024 points of each point cloud and divide it into N = 64 point patches, each with M = 32 neighborhood points. We set the mask ratio as 0.7. For PIC-Sep, we merge the feature of input and target at the third block. We randomly select a prompt pair that performs the same task with the query point cloud from the training set. We use an AdamW optimizer $[23]$ and cosine learning rate decay, with the initial learning rate as 0.001 and a weight decay of 0.05. All models are trained for 300 epochs.

<table><tr><td></td><td>Input</td><td>Prediction</td><td>Ground truth</td><td>Input</td><td>Prediction</td><td>Ground truth</td><td>Input</td><td>Prediction</td><td>Ground truth</td></tr><tr><td>Reconstruction</td><td><img src="images/0c15cac6089b46dd5063f1a7f6b9b230a8625f0ad73373bb6ff35950cc54ed4c.jpg"/></td><td><img src="images/0d49760ca9786e27e21687dd12f6b2e86e3734a19c231b0f2e761951ffadcec4.jpg"/></td><td><img src="images/9b590c395d60e3ca8e71132d5206a5be5d99bd22b79f265ace1d6e1f6a4faca3.jpg"/></td><td><img src="images/727169ab0ecd06c5b05cdc80bed23a121e0b43d5dbc23358aae772d39e9a87d8.jpg"/></td><td><img src="images/d883d662fd9d1e884cfbf9c0011b6f6a0ab57f4192f3f4d98dce26383c9d004a.jpg"/></td><td><img src="images/265764fed04764c940272d2e9b99c03f492ab1f8f2d755b38b5665ab23bbd277.jpg"/></td><td><img src="images/df2768588babfb5781f0266c0163c875f09803cfb2d4aa9aa257bd9f7296d100.jpg"/></td><td><img src="images/f5ce7295443f91a1d065bcb7eb797c7ca7f5583450672df9756439576b9fa985.jpg"/></td><td><img src="images/2f49729ad5406b370acf93c7c223d495b48e7479b26a4a16d6419329ec200d20.jpg"/></td></tr><tr><td>Denosing</td><td><img src="images/cbd24d1cd554347f27062d1db4619096c80c0e2bd11e0f91ca472dda76fc9b23.jpg"/></td><td><img src="images/bfa91486744f92bd3d938a63e6e0b2e5c83247b14235f28b91e0b03d6555562e.jpg"/></td><td><img src="images/330c30ad7555a214305c1e7f24ca3c916527c7ef8a98adf5eaedfef84f21c196.jpg"/></td><td><img src="images/568a925212cefebcc1ecb745902854c91d9468d1e956164384b3e387397a39da.jpg"/></td><td><img src="images/d1635dc57ad794ae189cb882fdf6676af6834d68ab6e9210961dcb8007dffbd5.jpg"/></td><td><img src="images/34495444ba5dd5b8f3bb9faa7464a4c8a4ae41024030408e59c0233aee09b37f.jpg"/></td><td><img src="images/184fdc447656876da4731574257bfe89f16132abee300f6700598bcaa93c4850.jpg"/></td><td><img src="images/461d2e27dc85c6de0fb5d85f2134e841a26cb2666599d2dd7af6aba8de4a1260.jpg"/></td><td><img src="images/450c3380c2c1e71668e90dfa3c4dea1f6bf8fc8963021aea6f0666e1005a3e5d.jpg"/></td></tr><tr><td>Registration</td><td><img src="images/d140f975ae43b82ddbcdd431fe8662252af480fafe0b17f30b55f7c90e9da5a3.jpg"/></td><td><img src="images/1322be15dca1cfe91b7d62266ea87fc000936b3a305bce7ef13358d24384273e.jpg"/></td><td><img src="images/bf18b908d7305efc79547a910c7c86a487d5318dc69652622f538df478790e74.jpg"/></td><td><img src="images/567e4ecd852884c8580bf3e5d7697bb1f38fc6298402e16e274ad5b7a12b82cb.jpg"/></td><td><img src="images/7ca0b833929d7217a26d59895edecc48eef86f8aa7b41d29806cd8340df22440.jpg"/></td><td><img src="images/da187f1b39d36bd3007a7f74c5032ee9982f43023647e9f873c8d100f6e57561.jpg"/></td><td><img src="images/c462d69537a87b1d4df3d438f97e30b472f2c5a72d28f0ba33aa73f1ade0039c.jpg"/></td><td><img src="images/b04353d748549a9ea009e36eed8ba18d7dcbcc5bfb439565c286ceaf2921c444.jpg"/></td><td><img src="images/9d21f14d71ad5d8abc2203052d1e722c11a263ffdda4accc87dcc8c1addefd15.jpg"/></td></tr><tr><td rowspan="2">Part Segmentation</td><td><img src="images/62f9d4c0763201dd6e88b63f3a61b5708eb1a4689acc9a90df48b1c3b607c7c2.jpg"/></td><td><img src="images/08405d715143e282cb4a88225d13ae8c32257365589a590c03151b9af87d9a2a.jpg"/></td><td><img src="images/0abbf8760cae2bcecbf212a31f639eff5f8f05abc60bfc6a497f95287f285c01.jpg"/></td><td><img src="images/f7c2fb00d59783277db9da44e6c0f7fb7b95155a537a60de644bf5c38cfa8223.jpg"/></td><td><img src="images/9a17654d5ebe1a7a9ae7ce6f619bdba37dd93241bced27d54430ecbc96b3aad9.jpg"/></td><td><img src="images/62c345d3b92e57fb31bc51b64a01646a15e4f5ff280a90082c65177b6719472d.jpg"/></td><td><img src="images/9774a3ec0917d311d03a5de8123be4e4d4847d03c500b122c1b5cdccbfdf8227.jpg"/></td><td><img src="images/b15b0f662e364c688ed0006768957e0836cc0cf387a5710f4a05736c38cd42be.jpg"/></td><td><img src="images/2f76d4239395dee03b411c4af7eee0cc701931928bb0891d9786df113451fe8b.jpg"/></td></tr><tr><td><img src="images/8e0fa3526a0e719dd68c1303b88e48cd04cf8b96f3a7a1dfd5c8e400dc6d8a8a.jpg"/></td><td><img src="images/2c6695f6bdb3dcb7b2d54f01ccacad130212a51053a2646a165780eed1993d5b.jpg"/></td><td><img src="images/4c1cadf3e9e6591b2dfbbaa8c5bdeb03236cbc11b3a3e9af7c867128385748e4.jpg"/></td><td><img src="images/0c47c1203cfb4540c56e976e437d77d4880aaad3df72638ebe686df245322837.jpg"/></td><td><img src="images/b0c2fc71112c1bfd016bea21e3ba32c765978686fa837257b57850912c1ba708.jpg"/></td><td><img src="images/5e4eb2cd959255f7e2e237fe679488e64694de3a2aabb87c3b44993de021b2b4.jpg"/></td><td><img src="images/13ababf5ffb440d4be9fe3fdb27e4277fb7b96153d0ecf8861b4b8c5665b0184.jpg"/></td><td><img src="images/15a1db0920a5395ee5c351cf20675da4883a245bf1f275e012d6188af0dc619b.jpg"/></td><td><img src="images/0aa48256f4199f784ac7f4562389f5c43f83265428d24b8895e241138a1fa1b6.jpg"/></td></tr></table>

Figure 4: Visualization of predictions obtained by our Point-In-Context and their corresponding targets in different tasks, such as reconstruction, denoising, registration, and part segmentation. For part segmentation, we visualize the generated target together with the mapping back, both adding category-specific colors for better comparisons.

# 4.1 Baseline Methods and Training Details

To evaluate the performance of the proposed framework, we evaluate the two variants PIC-Sep and PIC-Cat on the dataset mentioned in Sec. 3.2, both equipped with the proposed Joint Sampling module. We compare them with the following related methods:

Point-BERT [44] is a masked auto-encoder. Like the settings in PIC-Cat, we concatenate 2 pairs of input-target point cloud tokens as a token sequence and input it to Point-BERT, which takes a token sequence as the input and converts them into discrete point tokens from a pre-trained dVAE [40] vocabulary of size 8192.

Task-specific Models. We selected three representative methods: PointNet [27], DGCNN [40], PCT [12], and ACT [9], and individually train them on four different tasks mentioned in Sec. 3.2. Additionally, we also designed task-specific heads for each task.

Multitask Models. For a fair comparison, we develop a multitask model based on PointNet $[27]$ , DGCNN $[40]$ , and PCT $[12]$ , respectively, which are capable of multitask learning. These models feature a shared backbone network and task-specific heads designed to address the needs of different tasks. This design allows simultaneous learning of all four tasks.

Point-MAE $[26]$ is a masked auto-encoder. Unlike Point-BERT, Point-MAE directly rebuilds points in each local area. ACT $[9]$ , I2P-MAE $[47]$ , and ReCon $[29]$ are recent SOTA methods that involve other modalities like image and text knowledge in the pre-training stage and enhance the performance on different tasks after fine-tuning the models. Similar to multitask models, we use a pre-trained encoder and combine it with different task heads for simultaneous training on the four tasks.

Copy Example is a baseline that utilizes the target point cloud of the prompt as its prediction.

# 4.2 Main Results

We report extensive experimental results of various models on the dataset we proposed in Tab. 1. From where, we found that our PIC-Cat and PIC-Sep exhibit impressive results and are capable of adapting to different tasks after only one training, achieving state-of-the-art results in all four tasks amount multitask models. Besides, we visualize the in-context 3D inference results of PIC-Sep in

![](images/1841b59bdf51d9b8c268dd08299dbe03568f1efe5b6d1d2cc338f0094ed9497a.jpg)

<details>
<summary>line</summary>

| Level   | PointNet | PCT  | DGCNN | PIC-Cat | PIC-Sep |
| ------- | -------- | ---- | ----- | ------- | ------- |
| Level1  | 42       | 51   | 51    | 23      | 18      |
| Level2  | 44       | 53   | 52    | 25      | 20      |
| Level3  | 46       | 55   | 54    | 27      | 18      |
| Level4  | 49       | 56   | 55    | 28      | 16      |
| Level5  | 60       | 57   | 56    | 29      | 15      |
</details>

(a) Registration results on ModelNet40

![](images/06f0901d6876a129fb81cc1f7e31cc6ca8c2e74f82746e51af558af305eaad07.jpg)

<details>
<summary>text_image</summary>

Registration
1.1.1.1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.35.36.37.38.39.40.41.42.43.44.45.46.47.48.49.50.
</details>

(b) The generalization capacity

![](images/4578720ef0e0d5602e8b96cf6fe475cb3d8345a3c7f2c8ea2a534a7020807b51.jpg)

<details>
<summary>text_image</summary>

Completion
Aircraft 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
</details>

zation capacity

![](images/7b81ca85228c09000ab31d369a76915f564b31ceb4f9acfb983514510ef482a9.jpg)

<details>
<summary>text_image</summary>

Reconstruction
target Pred
</details>

(c) Limitation   
Figure 5: (a) Generalized to out-of-distribution data. We evaluate the registration task on the ModelNet40 [41] dataset, which is not present in the training set. The Chamfer Distance (x1000) is used as the evaluation metric. (b) Generalized to new tasks. Our model is able to rotate at any angle according to the example pairs and show the completion ability to the broken point cloud, which is not present in the training set. (c) Limitation of our model. PIC cannot reconstruct the detailed parts of complex point clouds very well.

Fig. 4, where our model can generate the corresponding predictions given the provided prompts among all four tasks, including reconstruction, denoising, registration, and part segmentation.

Comparison to Task-specific Models. Task-specific models including PointNet $[27]$ , DGCNN $[40]$ , and PCT $[12]$ outperform PIC-Cat and PIC-Sep in most indicators on all tasks. Specifically, in the part segmentation task, PIC-Cat achieves better results than PointNet and DGCNN, though they are well-designed for segmentation. It needs to be clarified that the performance of our model is directly related to the choice of prompts. When the quality of the prompt is better, PIC-Sep and PIC-Cat can achieve much better results, and this will be demonstrated in the following section.

Generalization. Usually, in-context Learning has a certain degree of generalization ability, thus allowing the model to quickly adapt to different tasks. This also applies to our models. First, we test the proposed methods on out-of-distribution point clouds by conducting a registration evaluation on the ModelNet40 [41] dataset. As shown in Fig. 5 (a), our models, PIC-Sep and PIC-Cat, both exhibit superior performance on open-class tasks compared to supervised learning models specifically trained for the registration task. We compare it with models trained on the single task, such as PointNet [27], DGCNN [40], and PCT [12]. These models suffer obvious performance drops when transferred to the new dataset. The more difficulty (higher level of rotating), the more drops.

Furthermore, we validate their generalization abilities on unseen tasks, such as registration and local completion. As shown in Fig. 5 (b), our proposed models work well on both tasks, validating their abilities to transfer learned knowledge. As a comparison, task-specific models are unable to infer the rotated point cloud unless trained with clear supervision. Besides, recovering a local hole is also an impossible task for them, even if they are trained in the reconstruction task.

# 4.3 Analysis

Effectiveness of Joint Sampling. We investigate whether the JS module has a valid effect on four tasks. As shown in Tab. 2, both PIC-Sep and PIC-Cat face a sharp decline in performance on the four tasks when the JS module is absent, and they even fail to achieve the most basic goal of reconstructing masked tokens. These findings validate our intuition that maintaining the consistency of the input and target token sequence positions is an indispensable design. That is to say, the JS module is a simple yet effective way to compensate for the missing positional information.

Table 2: Effectiveness of JS Module. 

<table><tr><td>Model</td><td>JS</td><td>Den. CD↓</td><td>Part Seg. mIOU↑</td></tr><tr><td rowspan="2">PIC-Cat</td><td>✕</td><td>29.3</td><td>17.03</td></tr><tr><td>√</td><td>5.3</td><td>78.95</td></tr><tr><td rowspan="2">PIC-Sep</td><td>✕</td><td>36.3</td><td>23.72</td></tr><tr><td>√</td><td>7.6</td><td>74.95</td></tr></table>

Ablation on Point Sampling. We study how the sampling method used by PIC-Sep in JS modules affects the performance of the model. We use the two most common sampling methods: Farthest Point Sampling (FPS) [27], and Random Sampling (RS) [14]. As shown in Tab. 3 (a), for the tasks of reconstruction, denoising, and registration, FPS produced better results than RS. Especially for

Table 3: Ablation study on our Point-In-Context. Gray: default setting   
(a) Different sampling strategies. 

<table><tr><td>#</td><td>Sample method</td><td>Rec. CD↓</td><td>Den. CD↓</td><td>Reg. CD↓</td><td>Part Seg. mIOU↑</td></tr><tr><td>1</td><td>RS</td><td>42.6</td><td>11.6</td><td>12.1</td><td>77.24</td></tr><tr><td>2</td><td>FPS</td><td>4.7</td><td>7.6</td><td>10.3</td><td>74.95</td></tr></table>

(b) Different loss functions. 

<table><tr><td>#</td><td>Loss function</td><td>Rec. CD↓</td><td>Den. CD↓</td><td>Reg. CD↓</td><td>Part Seg. mIOU↑</td></tr><tr><td>1</td><td> $\ell_1$ </td><td>5.0</td><td>8.1</td><td>11.1</td><td>72.35</td></tr><tr><td>2</td><td> $\ell_2$ </td><td>4.7</td><td>7.6</td><td>10.3</td><td>74.95</td></tr><tr><td>3</td><td> $\ell_1 + \ell_2$ </td><td>5.3</td><td>7.9</td><td>13.3</td><td>70.46</td></tr></table>

(c) Prompt position. 

<table><tr><td rowspan="2">#</td><td rowspan="2">order</td><td>Rec.</td><td>Den.</td><td>Reg.</td><td>Part Seg.</td></tr><tr><td>CD↓</td><td>CD↓</td><td>CD↓</td><td>mIOU↑</td></tr><tr><td>1</td><td>behind</td><td>4.8</td><td>8.2</td><td>8.0</td><td>74.04</td></tr><tr><td>2</td><td>before</td><td>4.7</td><td>7.6</td><td>10.3</td><td>74.95</td></tr></table>

point cloud reconstruction, where FPS can collect more key points than RS, and these key points can describe the original outline of the entire point cloud. However, in the task of part segmentation, the results of RS exceed those of FPS.

Loss Function. We conduct an exploration to determine which loss function is most suitable for our Point-In-Context model. During training, we experiment with using $\ell_{1}$ , $\ell_{2}$ , and a combination of $\ell_{1}$ and $\ell_{2}$ as the loss functions for our PIC-Sep. As Tab. 3 (b) shows, $\ell_{2}$ achieves the best result on all four tasks mentioned above.

Prompt Engineering. We investigate the influence of the layout of the prompt and the query on the experimental results. For PIC-Sep, we set up two layout options: one with the prompt before the query, and the other with the prompt after the query. As shown in Tab. 3 (c), the performance between the two designs has negligible differences. We simply choose the “before” option to align with 2D in-context learning works.

Mask Ratio. We conduct ablation experiments on the mask ratio at a wide range (20%-70%). As shown in Tab 4, training our Point-In-Context with a lower mask ratio weakens its performance across various tasks, especially on the mask ratio 20%. Meanwhile, the best results in the four tasks are distributed at different mask ratios, but considering all downstream tasks as a whole, the model can achieve the highest performance when the mask ratio is 70%. Different from language data, we also find keeping sparsity in training is necessary for mask point modeling for in-context learning. We find similar results as in MAE [13], a higher mask ratio is required to make sure that the model can learn hidden features well.

Prompt Selection. We explore the impact of the prompt selection on the model's prediction results. For the random selection method, during testing, we randomly select a pair of input-target point clouds from the training set that performs the same task as the query point cloud, serving as a task prompt. For the class-aware selection method, based on the previous one, we further select point clouds belonging to the same category as the query, such as airplanes, tables, chairs, etc. Additionally, we further delve into selecting two alternative prompts. To select pairs of examples that are paired with the query point cloud, we consider two factors: the Chamfer Distance (CD) [11] between the prompt and the query point cloud and the feature similarity between them (features are extracted from pre-trained PointNet [27]), which are respectively denoted

Table 4: Ablation study on mask ratio. 

<table><tr><td>#</td><td>Mask Ratio</td><td>Rec. CD↓</td><td>Den. CD↓</td><td>Reg. CD↓</td><td>Part Seg. mIOU↑</td></tr><tr><td>1</td><td>0.2</td><td>27.8</td><td>33.5</td><td>68.2</td><td>43.84</td></tr><tr><td>2</td><td>0.3</td><td>5.2</td><td>7.3</td><td>14.8</td><td>56.72</td></tr><tr><td>3</td><td>0.4</td><td>5.0</td><td>7.4</td><td>12.3</td><td>60.25</td></tr><tr><td>4</td><td>0.5</td><td>4.0</td><td>7.5</td><td>11.5</td><td>64.68</td></tr><tr><td>5</td><td>0.6</td><td>4.9</td><td>7.8</td><td>9.4</td><td>70.17</td></tr><tr><td>6</td><td>0.7</td><td>4.7</td><td>7.6</td><td>10.3</td><td>74.95</td></tr></table>

Table 5: Prompt selection methods. 

<table><tr><td>Model</td><td>Selection method</td><td>Rec. CD↓</td><td>Den. CD↓</td><td>Reg. CD↓</td></tr><tr><td rowspan="4">PIC-Cat</td><td>Random</td><td>4.3</td><td>5.3</td><td>14.1</td></tr><tr><td>Class-aware</td><td>4.3</td><td>5.3</td><td>10.5</td></tr><tr><td>Fea-aware</td><td>4.3</td><td>5.3</td><td>10.7</td></tr><tr><td>CD-aware</td><td>4.3</td><td>5.3</td><td>9.6</td></tr><tr><td rowspan="4">PIC-Sep</td><td>Random</td><td>4.7</td><td>7.6</td><td>10.3</td></tr><tr><td>Class-aware</td><td>4.6</td><td>7.4</td><td>5.1</td></tr><tr><td>Fea-aware</td><td>4.9</td><td>7.6</td><td>5.8</td></tr><tr><td>CD-aware</td><td>4.4</td><td>7.1</td><td>4.1</td></tr></table>

as CD-aware and Fea-aware. As depicted in Fig. 5, the CD-aware method demonstrates the best performance and even outperforms the task-specific models on the registration task. Note that we report the random method in the main results (Fig. 1), which means our model has a higher ceiling. This provides us a great opportunity to improve downstream task results by selecting higher-quality prompts, which will be the direction of future work.

<table><tr><td></td><td colspan="2">In-context learning models</td><td colspan="4">Multitask models</td><td></td></tr><tr><td>Input</td><td>PIC-Sep</td><td>PIC-Cat</td><td>PointMAE</td><td>PointNet</td><td>DGCNN</td><td>PCT</td><td>Target</td></tr><tr><td><img src="images/386aa69c8eb3c7cc8da5d0fd590eb7d2c54a2a43fbfcabcbc9068e4c0b152291.jpg"/></td><td><img src="images/afba2dac7829c543dc4581577ae7de3493532493ff571e221559a125b553c2b8.jpg"/></td><td><img src="images/aafa05718bfe5bf8b7671189d32e7e1258b90b6635478c6b069d06764f907683.jpg"/></td><td><img src="images/a9ccb117fa20b79ed800fad362b5574f918abfcce4ac3e3c33777e16237d0a44.jpg"/></td><td><img src="images/b2ad5838e3b7e2c266a7344b5d011f53183e344adb2acd23e34c0a6530a57779.jpg"/></td><td><img src="images/71522e9f120bddda3a232e589d9dbdbbd2e9013922ef3103887d25e2e82281a1.jpg"/></td><td><img src="images/6c471d9db4d8cb11976721bd384708eff9d3431609c88b10caafc6470f919c77.jpg"/></td><td><img src="images/4c0001b72a9ef5039739af131a22cb006812b72cd97dab580c75bea80505d751.jpg"/></td></tr><tr><td><img src="images/cd95c73110b0751e0348a75c4369ce7de446c30bb3c8eec32d1168213b9f96fe.jpg"/></td><td><img src="images/742d79896477d8481a55c10bc1cc6bae30ee28dac64530cf218964fe297542c3.jpg"/></td><td><img src="images/b48e6ef0fdcaaa4e5ede363f7f165d20d67018b6f1246877eaebc83a69365260.jpg"/></td><td><img src="images/2058baddda7d9bf02715e1fed85f654f9289def94d7946ed3db10e354bff0e72.jpg"/></td><td><img src="images/f67cc1c3ef0c31b959c4e0e1c88557c991fd89e9fd65ae51af253e1f1e014d29.jpg"/></td><td><img src="images/dd2a7beb524e548ed00caf5564d377d027a24eae51f4bdad1d56d12bcff1baa1.jpg"/></td><td><img src="images/cf45bc15d5c9166f768db53d843c8f3a4ba56ce711a1aeeb19cbfd29399c2b57.jpg"/></td><td><img src="images/509fb98187ebe0a3c061079e594ca5bb390f26f17baffeb42aa8b79485814aa8.jpg"/></td></tr><tr><td><img src="images/01a56187a1e64a42658642cdcd315fe4c170c03e842822cccc7a3dec2b703a92.jpg"/></td><td><img src="images/1bc5d3c4d67462773382f5b3d5f1c389b0676e44b71bbc0aa3750d6e0fd77db8.jpg"/></td><td><img src="images/17184a012f36486e9c551ad7f76a7f3a4b8ab854395e97707d612055d4d365fb.jpg"/></td><td><img src="images/eb9178d23763a8fe6c22363d2064631ef668cc96dac00665ab9e79c65e8eae5f.jpg"/></td><td><img src="images/61628cd06fee8bdc13dc59b5925fe2da8d53e86b627784a14606a52c7af9d439.jpg"/></td><td><img src="images/91abad4c4ef556cd9e79bfbfa55c5a897a2174accd93a0fef3a371d1f2da2389.jpg"/></td><td><img src="images/e6f931b888afd82780753e17c27b8849bb599458323d46b67dd55d6812873589.jpg"/></td><td><img src="images/bbe0416b7bed47a1ab76174268b197a35cb082ddbee51938b9c517210b753542.jpg"/></td></tr><tr><td><img src="images/ea7f16f0a493fc8f672fc68c3468e6ad419ff4840d489b8bfc02e969528f27d8.jpg"/></td><td><img src="images/f2e0868a2c872547cec9bde44adbe210b0a2417b444456adfad5a4ee6b5b5d73.jpg"/></td><td><img src="images/c62f6728c294a6843a4bd2145f56a08b2b2b4233ac82af54ba8576d48dfbc106.jpg"/></td><td><img src="images/a6d8677d072bf5ac2c6cbec55a48d35bea7218f2c833243b10e82f135afdd410.jpg"/></td><td><img src="images/5e08befa508b3ec43f405bc22423aa985b05b9d6e98072aad5586190c11c4b90.jpg"/></td><td><img src="images/19ef65fec4996e61076327b61fc40455ddd44d1b874884c769439e6664d32fe4.jpg"/></td><td><img src="images/02908279e8dfb3439737ea9d45158aba3a1a02f27cd13b3cb110fc94457be3ce.jpg"/></td><td><img src="images/19425f441b84d17a9efa0b98307d88b4ecafb469aa2d24d9a2d2a84e87337e68.jpg"/></td></tr><tr><td><img src="images/a228e564d62a153d43151e58c9ed43cda6df57adb471faf5b2dbd785c70c3339.jpg"/></td><td><img src="images/6d99b606ad78d90b20ccff632f0318b41373ee790be3184e40bc7594f56c9752.jpg"/></td><td><img src="images/e294b3d0f132e4801ea4166de784ba614338d9451271cd161728c41247cff243.jpg"/></td><td><img src="images/c499d1b9ed6420c2fa47cd02f24a6a27388fe2ac8a8d51182e965883597a562a.jpg"/></td><td><img src="images/931dc70ac4f20132d8e93efa265dffd46b2070b6e60f1ec75d44dccc73abe907.jpg"/></td><td><img src="images/ca087c351b24c3ec4949951acdeb1d4366bde55ca877eff1e609265707a38814.jpg"/></td><td><img src="images/ab0d7474d418b4e22e6fe17c452a210a4344fb30bf809186206a120f148e6595.jpg"/></td><td><img src="images/128fbc46a187f402ac1093e270b757532c3cc0735551e7933772b7267ee75746.jpg"/></td></tr><tr><td><img src="images/a1f30bf6b0397f626ae1562b73860d9537717f6a420d5131beab0907c8ceec52.jpg"/></td><td><img src="images/b9ceb60aa5605d7fb70429b168ae2610b4705d45caf7b4f0dcf51af7ae255324.jpg"/></td><td><img src="images/f5d064c3382c79df2946294e8e75db3425e06af5e2b2b9eb8d6eb072856c7b6a.jpg"/></td><td><img src="images/d0eadc38e9a4e963ab1fae1e3579ce2c7d33b2472a59c75a1946806ee91826ce.jpg"/></td><td><img src="images/904dad95f2e0a1948dbfc3152156fcbd61f9cfe056cef3900c62eb84ca3b2455.jpg"/></td><td><img src="images/1f6d22512283ad0cb8937f48fdbde12f1e5bc4f60c57ceb0de8d0365f478f9b0.jpg"/></td><td><img src="images/98908003cd9be7b082d4beab9699a06932178dce100020de4882c680d0037198.jpg"/></td><td><img src="images/353ee41bb627b8f9a4b3969aae039dacae3166e414d14297e811155530ca1153.jpg"/></td></tr></table>

Figure 6: Visualization of comparison results between PIC and multitask models.

Comparison Results between PIC and Multitask Models. We conduct a comparison of visualization results between our Point-In-Context models and multitask models on three tasks, including reconstruction, denoising, and registration. It is important to note that the multitask models in this comparison do not utilize the pre-trained backbone. As shown in Fig. 6, compared with other multitask models, our PIC-Sep and PIC-Cat output results are more satisfactory.

Limitation. Our experimental results show that our model can adapt to multiple downstream tasks after a single training with the assistance of prompt support. It performs well on the four proposed tasks, indicating its outstanding generalization ability. Nonetheless, our study has inherent limitations. Our model performs conditional generation for all tasks and presents unchallenged performance for concise point clouds. But for point clouds with complex contours, our model struggles and cannot reconstruct the detailed parts of complex point clouds very well, As shown in Fig. 5 (c).

Board Impact. Our work is the first to explore in-context learning for 3D point cloud understanding and is a very relevant but underexplored problem, including task definition, benchmark, and baseline models. Besides, we hope the setup of in-context learning in 3D and the curation of the in-context learning dataset is helpful to the community.

# 5 Conclusion

We propose Point-In-Context (PIC), the first framework adopting the in-context learning paradigm for 3D point cloud understanding. Specifically, we set up an extensive dataset of point cloud pairs with four fundamental tasks to achieve in-context ability. We propose effective designs that facilitate the training and solve the inherited information leakage problem. PIC shows its excellent learning capacity, achieves comparable results with single-task models, and outperforms multitask models on all four tasks. Besides, it shows good generalization ability to out-of-distribution samples and unseen tasks and has great potential via selecting higher-quality prompts. We hope it paves the way for further exploration of in-context learning in the 3D modalities.

Acknowledgements: This work is supported by the National Natural Science Foundation of China (No. 62203476). This study is also supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20120-0001). It is also supported by the interdisciplinary doctoral grants (iDoc 2021-360) from the Personalized Health and Related Technologies (PHRT) of the ETH domain.

# References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 2   
[2] Ivana Balažević, David Steiner, Nikhil Parthasarathy, Relja Arandjelović, and Olivier J. Hénaff. Towards in-context scene understanding. arXiv:2306.01667, 2023. 3   
[3] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2022. 2, 3   
[4] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. In NeurIPS, 2022. 1, 2, 3, 5   
[5] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv:2108.07258, 2021. 2   
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 1, 2, 3   
[7] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv:1512.03012, 2015. 2, 4, 14   
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 3   
[9] Runpei Dong, Zekun Qi, Linfeng Zhang, Junbo Zhang, Jianjian Sun, Zheng Ge, Li Yi, and Kaisheng Ma. Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning? In ICLR, 2023. 3, 6, 7   
[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3   
[11] Haoqiang Fan, Hao Su, and Leonidas J. Guibas. A point set generation network for 3d object reconstruction from a single image. In CVPR, 2017. 6, 9   
[12] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. Pct: Point cloud transformer. In CVM, 2021. 2, 3, 6, 7, 8, 13, 14   
[13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2, 3, 6, 9   
[14] Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, and Andrew Markham. Randla-net: Efficient semantic segmentation of large-scale point clouds. In CVPR, 2020. 8   
[15] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 2   
[16] Li Jiang, Hengshuang Zhao, Shu Liu, Xiaoyong Shen, Chi-Wing Fu, and Jiaya Jia. Hierarchical point-edge interaction network for point cloud semantic segmentation. In ICCV, 2019. 3   
[17] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv:2101.00190, 2021. 2   
[18] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv:2110.07602, 2021. 2   
[19] Yang Liu, Chen Chen, Can Wang, Xulin King, and Mengyuan Liu. Regress before construct: Regress autoencoder for point cloud self-supervised learning. In ACM MM, 2023. 3   
[20] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In CVPR, 2022. 2   
[21] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 2   
[22] Zhe Liu, Shunbo Zhou, Chuanzhe Suo, Peng Yin, Wen Chen, Hesheng Wang, Haoang Li, and Yun-Hui Liu. Lpd-net: 3d point cloud learning for large-scale place recognition and environment analysis. In ICCV, 2019. 3   
[23] Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. In ICLR, 2018. 6   
[24] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In ICLR, 2022. 3   
[25] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. Parameter-efficient image-to-video transfer learning. arXiv:2206.13559, 2022. 2   
[26] Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In ECCV, 2022. 2, 3, 4, 5, 6, 7, 13

[27] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 2, 5, 6, 7, 8, 9, 13, 14   
[28] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017. 2   
[29] Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining. In ICML, 2023. 3, 6, 7   
[30] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. In NeurIPS, 2022. 2   
[31] Guocheng Qian, Xingdi Zhang, Abdullah Hamdi, and Bernard Ghanem. Improving standard transformer models for 3d point cloud understanding with image pretraining. arXiv:2208.12259, 2022. 3   
[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2   
[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3   
[34] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2, 3   
[35] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. arXiv:2112.08633, 2021. 2, 3   
[36] Yanpeng Sun, Qiang Chen, Jian Wang, Jingdong Wang, and Zechao Li. Exploring effective factors for improving visual in-context learning. arXiv:2304.04748, 2023. 3   
[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3   
[38] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In CVPR, 2023. 2, 3, 5, 6   
[39] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. In ICCV, 2023. 2, 3   
[40] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. In TOG, 2019. 2, 3, 6, 7, 8, 13, 14   
[41] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In CVPR, 2015. 8   
[42] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for region annotation in 3d shape collections. In TOG, 2016. 2, 4   
[43] Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, and Jie Zhou. Pointr: Diverse point cloud completion with geometry-aware transformers. In ICCV, 2021. 3   
[44] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In CVPR, 2022. 2, 3, 4, 6, 7, 13   
[45] Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, and Hongsheng Li. Point-m2ae: multi-scale masked autoencoders for hierarchical point cloud pre-training. In NeurIPS, 2022. 3, 4, 5   
[46] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In CVPR, 2022. 3   
[47] Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3d representations from 2d pre-trained models via image-to-point masked autoencoders. In CVPR, 2023. 3, 6, 7   
[48] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. What makes good examples for visual in-context learning? arXiv:2301.13670, 2023. 2, 3   
[49] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H.S. Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021. 3   
[50] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. In IJCV, 2022. 2

# Supplementary Material

![](images/a139f18d54c44652233f4a0eb9bd5c476b4c91b3f146da099f74ab33366d1fee.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pairs-wise Point clouds"] --> B["Input"]
    B --> C["Target"]
    C --> D["Embed Concat"]
    D --> E["Mask"]
    E --> F["Transformer"]
    F --> G["Output"]
    G --> H["CD"]
    H --> I["GT"]
    I --> J["Multi-task results"]
    
    subgraph Input
        B1["Prompt Input"]
        B2["Prompt Target"]
        B3["Query Input"]
        B4["Query Target"]
        B5["Masked Token"]
    end
    
    subgraph Target
        C1["Prompt Input"]
        C2["Prompt Target"]
        C3["Query Input"]
        C4["Masked Token"]
    end
    
    subgraph Output
        G1["Crosslinked to GT"]
        H1["Crosslinked to GT"]
        I1["Crosslinked to GT"]
        J1["Crosslinked to GT"]
    end
    
    subgraph Multitask Results
        J1["Crosslinked to GT"]
        K1["Crosslinked to GT"]
        L1["Crosslinked to GT"]
    end
    
    A -->|2 pairs JS Module| B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```
</details>

Figure 7: Overall scheme of our Point-In-Context-Cat. Top: During training, each sample comprises two pairs of input and target point clouds that tackle the same task. Unlike PIC-Sep, PIC-Cat concatenates the input and target to form a new point cloud. Bottom: In-context inference on multitask. Our Point-In-Context could infer results on various downstream point cloud tasks.

Overview. The supplementary material includes sections as follows:

• Section A: Pipeline of Point-In-Context-Cat, including training and inference stages.   
- Section B: More results about multitask models trained using a pre-trained backbone with multitask heads.   
• Section C: More visualization results.

# A More Details of PIC

Pipeline of PIC-Cat. During the training phase, our approach involves selecting a pair of query point clouds and a pair of prompt point clouds from the training dataset. These point clouds are then grouped using the Joint Sampling module. Following this, we perform encoding and tokenization on each point cloud and concatenate them to create a new point cloud. A masking operation is applied to the entire point cloud to conduct the MPM task. We set the mask ratio as 60% for our specific approach, PIC-Cat. During the in-context inference stage, we only mask the last quarter of the tokens, which corresponds to the desired output. This approach allows our PIC-Cat model to reconstruct the masked tokens, leveraging its training experience. It is important to note that the task on the query point cloud is determined by the prompt in the example pair.

Comparison of Model Parameters, GFLOPs, and Test Speed. We compare the parameters and GFLOPs of each model in the main results of the main text in Tab. 6. Our model achieves a favorable balance between structural complexity and task performance, making it a compelling choice. Note that the parameters and GFLOPs of task-specific models are computed, including four individual models for four different tasks. Besides, we report the speed of models by samples/second tested on one NVIDIA RTX 3080 Ti GPU. Our PIC-Cat presents a high inference speed (953 samples/second), which is second only to DGCNN [40].

Table 6: The comparison of parameters, GFLOPs, and test speed. 

<table><tr><td rowspan="2"></td><td colspan="3">Task-specific models</td><td colspan="4">multi-task models</td><td colspan="3">In-context learning models</td></tr><tr><td>PointNet [27]</td><td>DGCNN [40]</td><td>PCT [12]</td><td>PointNet [27]</td><td>DGCNN [40]</td><td>PCT [12]</td><td>Point-MAE [26]</td><td>Point-BERT [44]</td><td>PIC-Cat</td><td>PIC-Sep</td></tr><tr><td>Params(M)</td><td>8.9</td><td>7.9</td><td>13.0</td><td>6.0</td><td>7.6</td><td>10.8</td><td>27.0</td><td>52.6</td><td>29.0</td><td>28.9</td></tr><tr><td>FLOPs(G)</td><td>1.9</td><td>3.1</td><td>6.3</td><td>1.9</td><td>10.2</td><td>6.4</td><td>11.8</td><td>12.0</td><td>12.1</td><td>8.4</td></tr><tr><td>Test speed</td><td>694</td><td>1500</td><td>694</td><td>844</td><td>1185</td><td>717</td><td>742</td><td>190</td><td>953</td><td>291</td></tr></table>

Table 7: Results of multitask models composed of a multitask head and a pre-train backbone trained on ShapeNet [7] for classification. For reconstruction, denoising, and registration, we report Chamfer Distance $\ell_{2}$ loss (x1000). For part segmentation, we report mIOU. 

<table><tr><td rowspan="2">Models</td><td rowspan="2">Acc.(%)</td><td colspan="6">Reconstruction CD ↓</td><td colspan="6">Denoising CD ↓</td><td colspan="6">Registration CD ↓</td><td>Part Seg.</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Avg.</td><td>mIOU↑</td></tr><tr><td colspan="21">multitask models: share backbone + multi-task heads</td></tr><tr><td>PointNet [27]</td><td>88.7</td><td>47.0</td><td>45.8</td><td>45.4</td><td>45.4</td><td>45.8</td><td>45.9</td><td>22.9</td><td>23.2</td><td>26.3</td><td>28.3</td><td>30.0</td><td>26.1</td><td>35.5</td><td>34.8</td><td>37.1</td><td>37.2</td><td>38.6</td><td>36.6</td><td>10.13</td></tr><tr><td>DGCNN [40]</td><td>89.4</td><td>46.7</td><td>47.2</td><td>48.1</td><td>48.6</td><td>48.5</td><td>47.8</td><td>8.2</td><td>8.3</td><td>8.4</td><td>8.8</td><td>9.2</td><td>8.6</td><td>14.2</td><td>15.8</td><td>18.2</td><td>21.8</td><td>23.5</td><td>18.7</td><td>21.35</td></tr><tr><td>PCT [12]</td><td>89.5</td><td>64.7</td><td>60.8</td><td>59.2</td><td>60.1</td><td>59.7</td><td>61.0</td><td>14.5</td><td>12.2</td><td>12.4</td><td>12.0</td><td>11.8</td><td>12.6</td><td>22.6</td><td>25.2</td><td>28.3</td><td>31.1</td><td>33.2</td><td>28.1</td><td>15.43</td></tr></table>

![](images/532619bba1a8e5c435239715337525be4433f9fd7e8d2cc901da97e700e7626a.jpg)

<details>
<summary>text_image</summary>

Reconstruction
Denoising
Registration
Part Segmentation
</details>

Figure 8: Additional visualization results of PIC-Sep. The output of our model is marked in red. Note that the results of part segmentation have been processed by adding XYZ coordinates.

# B More Results of Multitask Models

Pre-trained Backbone + Multitask Heads. For multitask models, we utilize a pre-trained backbone feature extraction network that is trained on the ShapeNet $[7]$ dataset for classification tasks. This pre-trained backbone network is equipped with multiple task-specific heads to perform multitask learning on our benchmark, allowing for the simultaneous handling of various tasks. As shown in Tab. 7, while these supervised models perform well when trained on individual tasks, they exhibit poor performance on multitask benchmarks.

# C More Visualization

More Visualization of PIC-Sep. We visualize more examples in Fig. 8, including reconstruction, denoising, registration, and part segmentation.