# LLaVA-OneVision: Easy Visual Task Transfer

Bo Li $^{2, \heartsuit}$ Yuanhan Zhang $^{2, \heartsuit}$ Dong Guo $^{1}$ Renrui Zhang $^{3, \heartsuit}$ Feng Li $^{4, \heartsuit}$ Hao Zhang $^{4, \heartsuit}$ Kaichen Zhang $^{2}$ Peiyuan Zhang $^{2}$ Yanwei Li $^{3, \heartsuit}$ Ziwei Liu $^{2}$ Chunyuan Li $^{1}$

$^{1}$ ByteDance $^{2}$ S-Lab, NTU $^{3}$ CUHK $^{4}$ HKUST

https://llava-vl.github.io/blog/llava-onevision

# Abstract

We present LLaVA-OneVision, a family of open large multimodal models (LMMs) developed by consolidating our insights into data, models, and visual representations in the LLaVA-NeXT blog series. Our experimental results demonstrate that LLaVA-OneVision is the first single model that can simultaneously push the performance boundaries of open LMMs in three important computer vision scenarios: single-image, multi-image, and video scenarios. Importantly, the design of LLaVA-OneVision allows strong transfer learning across different modalities/scenarios, yielding new emerging capabilities. In particular, strong video understanding and cross-scenario capabilities are demonstrated through task transfer from images to videos.

# 1 Introduction

It is a core aspiration in AI to build general-purpose assistants with Large Multimodal Models (LMM) [67]. LLaVA-OneVision is an open model, continuing to advance the line of research in building large vision-and-language assistant (LLaVA) [83] that can follow diverse instructions to complete a variety of computer vision tasks in the wild. As a cost-efficient recipe, it is typically developed by connecting vision encoders with large language models (LLM) using a simple connection module.

The first LLaVA model $[83]$ demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors similar to GPT-4V on previously unseen images and instructions for the first time. LLaVA-1.5 $[81]$ significantly expands and improves the capabilities by incorporating more academic-related instruction data, achieving SoTA performance on a dozens of benchmarks with a data-efficient recipe. LLaVA-NeXT $[82]$ inherits this property, further pushing performance boundaries through three key techniques: AnyRes for handling high-resolution images, expanding high-quality instruction data, and utilizing the best open LLM available at the time.

LLaVA-NeXT provides an extendable and scalable prototype, which facilitates several parallel explorations, reported in the LLaVA-NeXT blog series $[82, 169, 65, 64, 68]$ :

https://llava-vl.github.io/blog/

- The Video blog [169] shows that the image-only-trained LLaVA-NeXT model is surprisingly strong on video tasks with zero-shot modality transfer, due to the design of AnyRes to digest any vision signals as a sequence of images.   
- The Stronger blog [65] demonstrates the LLM model scaling success of this cost-efficient strategy. By simply scaling up the LLM, it achieves performance comparable to GPT-4V on selected benchmarks.

\- The Ablation blog [64] summarizes our empirical exploration except the visual instruction data itself, including the choice of architectures (scaling of LLM & vision encoder), visual representations (resolution & #tokens), as well as training strategies (trainable modules & high-quality data) in the pursuit of data scaling success.

\- The Interleave blog [68] describes the strategies to extend and improve the capability in new scenarios including multi-image, multi-frame (video) and multi-view (3D), while maintaining the single-image performance.

These explorations, conducted within a fixed compute budget, aimed to offer useful insights along the way as we navigate the project, rather than push performance limits. During the process, we have also been accumulating and curating a large collection of the high-quality datasets from January to June. By consolidating these insights and execute the experiments with “yolo run” on newly accumulated larger datasets, we introduce LLaVA-OneVision. We implement the new model with the available compute, without extensively de-risking individual components. This leaves room for further improvements in capabilities through additional data and model scaling following our recipe, Please see the detailed development timeline in Section A. In particular, our paper makes the following contributions:

\- Large multimodal models. We develop LLaVA-OneVision, a family of open large multimodal models (LMMs) that improves the performance boundaries of open LMMs in three important vision settings, including single-image, multi-image, and video scenarios.

\- Emerging Capabilities with Task Transfer. Our design in modeling and data representations allow task transfer across different scenarios, suggesting a simple approach to yield new emerging capabilities. In particular, LLaVA-OneVision demonstrate strong video understanding through task transfer from images.

\- Open-source. To pave the way towards building a general-purpose visual assistant, we release the following assets to the public: the generated multimodal instruction data, the codebase, the model checkpoints, and a visual chat demo.

# 2 Related Work

The SoTA proprietary LMMs, such as GPT-4V $[109]$ , GPT-4o $[110]$ , Gemini $[131]$ and Claude-3.5 $[3]$ , exhibit excellent performance in versertile vision scenarios, including single-image, multi-image and video settings. In the open research community, existing works typically develop models tailored to each individual scenario separately. Specifically, most focus on pushing the performance limits in single-image scenarios $[26, 83, 173, 73, 164, 35]$ , only a few recent papers have begun to explore multi-image scenarios $[70, 47]$ . While video LMMs excel in video understanding, they often do so at the expense of image performance $[72, 76]$ . It is rare to have a single open model that reports excellent performance in all three scenarios. LLaVA-OneVision aims to fill this gap by demonstrating state-of-the-art performance across a broad range of tasks, and showcasing interesting emerging capabilities through cross-scenario task transfer and composition.

To the best of our knowledge, LLaVA-NeXT-Interleave $[68]$ is the first attempt to report good performance in all three scenarios, LLaVA-OneVision inherits its training recipe and data for improved performance. Other versatial open LMMs with potentials to excel include VILA $[77]$ , InternLM-XComposer-2.5 $[162]$ . Unfortunately, their results are not fully evaluated and reported; we compare with them in the experiments. In addition to building systems with versatial capabilities, LLaVA-OneVision is benefited from large-scale high-quality data training, including model-synthesized knowledge and the new collection of diverse instruction tuning data. For the former, we inherit all the knowledge learning data in $[64]$ . For the latter, our are motivated by FLAN $[136, 88, 145]$ . The data collection process is con-current with Idefics2 $[63]$ and Cambrian-1 $[133]$ , but we focus on a smaller but more carefully curated collection of datasets. A similar conclusion is observed: a large amount of visual instruction tuning data can significantly improve performance. For comprehensive investigations on design choices of LMMs, we refer to several recent studies $[51, 63, 64, 104, 133, 10]$ .

![](images/82591f3825c6353c66c0e53283888ed75ccd35271efa31374e7aab8008ef6581.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Qwen-2"] --> D["Language Model fφ"]
    B["2-Layer MLP"] --> D
    C["SigLIP"] --> D
    D --> E["Projection pθ"]
    E --> F["Vision Encoder gψ"]
    F --> G["Zv"]
    G --> H["Hv"]
    H --> I["Xv"]
    I --> J["Visual Signal"]
    J --> K["Hq"]
    K --> L["Xq"]
    L --> M["Language Instruction"]
    M --> N["Video"]
    style A fill:#d4edda,stroke:#333
    style B fill:#d4edda,stroke:#333
    style C fill:#d4edda,stroke:#333
    style D fill:#e6f7ff,stroke:#333
    style E fill:#e6f7ff,stroke:#333
    style F fill:#e6f7ff,stroke:#333
    style G fill:#e6f7ff,stroke:#333
    style H fill:#e6f7ff,stroke:#333
    style I fill:#e6f7ff,stroke:#333
    style J fill:#e6f7ff,stroke:#333
    style K fill:#e6f7ff,stroke:#333
    style L fill:#e6f7ff,stroke:#333
    style M fill:#e6f7ff,stroke:#333
    style N fill:#fff2cc,stroke:#333
```
</details>

Figure 1: LLaVA-OneVision network architecture. Left: The current model instantiation; Right: the general form of LLaVA architecture in [83], but is extended to support more visual signals.

# 3 Modeling

# 3.1 Network Architecture

The model architecture inherits the minimalism design of LLaVA series, whose primary goals are (i) effectively leverage the pre-trained capabilities of both the LLM and visual model, as well as (ii) facilitate strong scaling behavior in terms of both data and model. The network architecture is illustrated in Figure 1.

- LLM. We choose Qwen-2 [148] as our LLM $f_{\phi}(\cdot)$ parameterized by $\phi$ , as it offers various model size and exhibits strong language capabilities to date among publicly available checkpoints.   
- Vision Encoder. We consider the SigLIP [158] as the visual encoder $g_{\psi}(\cdot)$ parameterized by $\psi$ , encoding an input image $\mathbf{X}_{\mathrm{v}}$ into its visual feature $\mathbf{Z}_{\mathrm{v}} = g(\mathbf{X}_{\mathrm{v}})$ . The grid features before and after the last Transformer layer are considered in our experiments.   
- Projector. We consider a 2-layer MLP [81] $p_{\theta}(\cdot)$ parameterized by $\theta$ , to project image features into the word embedding space, yielding a sequence of visual tokens $\mathbf{H}_{\mathrm{v}} = p(\mathbf{Z}_{\mathrm{v}})$ .

The model choice is based on our empirical insights in $[65, 64]$ that stronger LLM typically supercharge stronger multimodal capabilities in the wild, while SigLIP yields higher LMM performance among open vision encoders.

For a sequence of length $L$ , we compute the probability of the target answers $\mathbf{X}_{\mathrm{a}}$ by:

$$
p (\mathbf {X} _ {\mathrm{a}} | \mathbf {X} _ {\mathrm{v}}, \mathbf {X} _ {\mathrm{q}}) = \prod_ {i = 1} ^ {L} p (x _ {i} | \mathbf {X} _ {\mathrm{v}}, \mathbf {X} _ {\mathrm{q}, <   i}, \mathbf {X} _ {\mathrm{a}, <   i}), \tag {1}
$$

where $X_{q,<i}$ and $X_{a,<i}$ are the instruction and answer tokens in all turns before the current prediction token $x_{i}$ , respectively. For the conditionals in (1), we explicitly add $X_{v}$ to emphasize the fact that the visual signal is grounded for all answers. As explained in Section 3.2, the form of visual signal $X_{v}$ is general. The visual input fed into the vision encoder depends on the corresponding scenarios: the individual image crop in the single-image sequence, the individual image in a multi-image sequence and the individual frame in the video sequence, respectively.

# 3.2 Visual Representations

The representation of visual signals is key to the success of the visual encoding. It relates to two factors, the resolution in the raw pixel space and the number of tokens in the feature space, leading to the visual input representation configuration (resolution, #token). The scaling of both factors leads to improved performance, especially on tasks that require visual details. To strike a balance of performance and cost, we observe that the scaling of resolution is more effective than that of token numbers, and recommend an AnyRes strategy with pooling. The comparison is illustrated in Figure 2.

![](images/62a44c1e5cc35ce517569a0a0e33772db7410e27a38d2bf486c0f1eec18375c3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Visual Instruction Timing"] --> B["split"]
    B --> C["encode"]
    C --> D["Bilinear Interpolation"]
    D --> E["flatten"]
    E --> F["LLM"]
    A --> G["resize"]
    G --> H["encode"]
    H --> I["flatten"]
    I --> F
```
</details>

(a) Higher AnyRes with Bilinear Interpolation   
![](images/341df1785e66859b359c0b627c9b02969b3ddf8c1f0a9642b6aba6ef59f9e058.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Visual Instructors Testing"] --> B["resize & split"]
    B --> C["encode"]
    C --> D["flatten"]
    D --> E["LLM"]
    F["Introduction"] --> G["resize"]
    G --> H["encode"]
    H --> I["flatten"]
```
</details>

(b) The original AnyRes

Figure 2: The visual representations. Top: The new Higher AnyRes scheme with Bilinear Interpolation to deal with images of higher resolution; Bottom: the original AnyRes in [82]. 

<table><tr><td>Single-Image</td><td colspan="5">729 + N * 729 Tokens</td><td>... N Crops</td><td>(1 + 9) * 729 = 7290 Tokens</td></tr><tr><td>Multi-Image</td><td colspan="6">N * 729 Tokens</td><td>12 * 729 = 8748 Tokens</td></tr><tr><td>Video</td><td colspan="5">N * 196 Tokens</td><td>... N Frames</td><td>32 * 196 = 6272 Tokens</td></tr><tr><td colspan="7">Example on Token Strategy</td><td>Max Tokens</td></tr></table>

Figure 3: The visual representation strategy to allocate tokens for each scenario in LLaVA-OneVision. The maximum number of visual tokens across different scenarios is designed to be similar, ensuring balanced visual representations to accommodate cross-scenario capability transfer. Note that 729 is the #tokens for SigLIP to encode a visual input of resolution $384 \times 384$ .

For AnyRes with a configuration of width $a$ , height $b$ , it divides the image into $a \times b$ crops, each with the shape $(a, b)$ . Each crop has the same resolution suitable for the vision encoder. Assuming there are $T$ tokens per crop, the total number of visual tokens is $L = (a \times b + 1) \times T$ , where the base image is resized before being fed into the vision encoder. We consider a threshold $\tau$ , and reduce the #token per crop, using bilinear interpolation if needed:

$$
T _ {\text { new }} = \left\{ \begin{array}{l l} \frac {\tau}{(a \times b + 1)} & \text { if   } L > \tau \\ T & \text { if   } L \leq \tau \end{array} \right. \tag {2}
$$

A set of spatial configurations $(a, b)$ is defined to specify various methods for cropping images, thereby accommodating images of different resolutions and aspect ratios. Among them, the configuration

that requires a minimum number of crops is selected. Please see our detailed ablations of visual representation in [64].

The proposed Higher AnyRes strategy can serve as a flexible visual representation framework, adaptable for multi-image and video representation. The optimal configuration for performance and cost can be adjusted accordingly. We illustrate the configuration in Figure 3, describe the detailed in Section C.1 and provide high-level encoding strategies as below:

- Single-image. We consider a large maximum spatial configuration $(a, b)$ for single-image representation to maintain the original image resolution without resizing. Additionally, we purposefully allocate a large number of visual tokens per image, resulting in a long sequence to effectively represent the visual signal. This is based on the observation that there is a larger number of high-quality training samples with diverse instructions for images compared to videos. By representing an image with a long sequence that mimics video representation, we facilitate a smoother capability transfer from image to video understanding [169, 64].   
- Multi-image. Only the base image resolution is considered and fed into the vision encoder to obtain feature maps, eliminating the need for multi-crop of high resolution image and thus saving computational resources [68].   
- Video. Each frame of the video is resized to the base image resolution and processed by the vision encoder to generate feature maps. Bilinear interpolation is employed to reduce the number of tokens, allowing the consideration of a larger number of frames by reducing tokens per frame. Empirical evidence suggests this provides a better trade-off between performance and computational cost [169].

These representation configurations are designed for capability transfer with a fixed compute budget in our experiments. With increased computational resources, the number of tokens per image or frame can be increased during both training and inference stages to boost performance.

# 4 Data

In the realm of multimodal training from LLM, the axiom “quality over quantity” is especially true. This principle is paramount due to the extensive knowledge stored within pre-trained LLMs and Vision Transformers (ViTs). While it is essential to accumulate balanced, diverse, and high-quality instruction data by the end of the LMM’s training lifecycle, an often-overlooked aspect is the continuous exposure of the model to new, high-quality data for further knowledge acquisition whenever it is available. In this section, we discuss the data sources and strategies for high-quality knowledge learning and visual instruction tuning.

# 4.1 High-Quality Knowledge

The web-scale public image-text data is often of low-quality, rendering the data scaling of multimodal pre-training less efficient. Instead, we recommend to focus on high-quality knowledge learning, given a limited compute budget. This approach acknowledges that the pre-trained LLMs and ViTs already possess a substantial knowledge base, and the goal is to refine and enhance this knowledge with carefully curated data. By prioritizing the quality of data, we can maximize compute efficiency.

We consider data from three major categories for high-quality knowledge learning:

- Re-Captioned Detailed Description Data. LLaVA-NeXT-34B [82] is known for its strong detailed caption ability among open-source LMMs. We used the model to generate new captions for the images from the following datasets: COCO118K, BLIP558K, and CC3M. We combined them to form the Re-Captioned Detailed Description Data, totaling 3.5M samples. This can be viewed as an simple attempt of self-improvement AI, where the training data is generated by an early version of the model itself.   
- Document / OCR Data. We utilized the Text Reading subset from the UReader dataset, totaling 100K, which is easily accessible through PDF rendering. We used this text reading data along with the SynDOG EN/CN, to form the Document / OCR Data, totaling 1.1M samples.   
- Chinese and Language Data. We used the original ShareGPT4V [20] images and utilized GPT-4V provided by the Azure API to generate 92K detailed Chinese caption data, aiming to improve the model's capability in Chinese. Since we used a large portion of detailed caption

data, we also aim to balance the model's language understanding ability. We collected 143K samples from the Evo-Instruct dataset [16].

It is interesting to note that almost all (accounting for 99.8%) of the high-quality knowledge data is synthetic. This is due to the high cost and copyright constraints associated with collecting large-scale, high-quality data in the wild. In contrast, synthetic data can be easily scaled. We believe that learning from large-scale synthetic data is becoming a trend as AI models continue to grow more powerful.

# 4.2 Visual Instruction Tuning Data

Visual instruction tuning $[83]$ refers to the capability of an LMM to understand and act upon visual instructions. These instructions can be in the form of language, combined with visual media such as images and videos, which the LMM processes and follows to perform a task or provide a response. This involves integrating visual understanding with natural language processing to interpret the instructions and execute the required responses.

Data Collection and Curation. As demonstrated in previous works $[81, 133, 63]$ , visual instruction tuning data is crucial for LMM capability. Therefore, maintaining a high-quality dataset collection is crucial and beneficial to the community. We started to collect a large pool of instruction tuning datasets from various original sources, with an unbalanced data ratio among categories. Additionally, we utilize a few new subsets from the Cauldron $[63]$ and Cambrian $[133]$ dataset collections.

We categorize the data based on a three-level hierarchy: vision, instruction, and response.

- Vision Input. Three vision scenarios are considered, depding which visual input is considered in the multimodal sequence, including single-image, multi-image, video.   
- Language Instruction. The instructions, which often appears as questions, define the tasks to perform to deal with the visual input. We classify the data into five major categories: General QA, General OCR, Doc/Chart/Screen, Math Reasoning, and Language. These instructions define the skill sets that a trained LMM could cover. We use task categorization to help maintain and balance the skill distribution.   
- Language Response. The answer not only responds the user request, but also specifies the model behavior. It can be broadly categorized into free-form and fixed-form.

Free-form data is typically annotated by advanced models like GPT-4V/o and Gemini, while fixed-form data is derived from academic datasets, e.g. VQAv2, GQA, Visual Genome. For free-form data, we keep the original answers. However, for fixed-form data, we manually review the content and make necessary corrections to the question and answer formats. We adhere to the LLaVA-1.5 prompting strategy for multiple-choice data, short answer data, and specific task data (e.g., OCR). This step is crucial for guiding the model's behavior to correctly balance QA performance, conversational ability, and reasoning skills in more complicated tasks, as well as preventing potential conflicts from different data sources. We list the full details about each dataset in our collection, and their categorization and formatting prompt in Appendix E.3.

We divide the instruction data into two separate groups: one for single-image scenario and the other for all vision scenarios. This division is based on insights from our earlier studies $[68, 169]$ , which highlight the relationship between image and video models: a stronger image model can better transfer to multi-image and video tasks. Additionally, the quantity and quality of training datasets available for single images are significantly higher than those for videos and multi-image tasks.

Single-Image Data. Since single-image data is crucial for multimodal capabilities, we explicitly compile a large single-image data collection for model learning. We select from collected data sources to form a balanced collection, resulting in a total of 3.2 million samples. The overall distribution of single-image data is shown in Figure 4, with detailed information and the roadmap of data collection presented in Appendix E.1.

OneVision Data. In addition to the single-image stage training, we further fine-tune the model using a mixture of video, image, and multi-image data. We introduce a total of 1.6 million mixed data samples, comprising 560K multi-image data from [68], 350K videos collected in this project, and 800K single-image samples. Notably, in this stage, we do not introduce new single-image data but instead sample high-quality and balanced portions from the previous single-image data, as described

![](images/415f5a4cbec6cb3a18c34fa8a1b7ab8d158ef7f2ec3762cf677de4259abdbce1.jpg)

<details>
<summary>pie</summary>

Single-image 3.2M
| Category | Description | Value |
|---|---|---|
| General | CLEVR (0.7 K) | 36.1% |
| General | Image Textualization (99.6 K) | 99.6 |
| General | OKVQA (9.0 K) | 9.0 |
| General | ShareGPT4V (91.0 K) | 91.0 |
| General | Visual7W (14.4 K) | 14.4 |
| General | VQAv2 (82.8 K) | 82.8 |
| General | Doc/Chart/Screen | 20.6% |
| General | Math/Reasoning | 20.1% |
| General | Language | 14.3% |
| General | Magpie Pro (L3 MT) | 150.0 |
| General | Magpie Pro (L3 ST) | 150.0 |
| General | Magpie Pro (Qwen2 ST) | 150.0 |
| General | Magpie Pro (L3 ST) | 150.0 |
| General | Magpie Pro (Qwen2 ST) | 150.0 |
| General | Magpie Pro (L3 ST) | 150.0 |
| General | Magpie Pro (Qwen2 ST) | 150.0 |
| General | Magpie Pro (L3 ST) | - |
| General | Magpie Pro (Qwen2 ST) | - |
| General | Magpie Pro (L3 ST) | - |
| General | Magpie Pro (Qwen2 ST) | - |
| General | Magpie Pro (L3 ST) | - |
| General | Magpie Pro (Qwen2 ST) | - |
| General | Magpie Pro (L3 ST) | - |
| General | Magpie Pro (Qwen2 ST) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | Magpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | MAGpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | MAGpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | MAGpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | MAGpie Pro (L3 ST) + Magpie Pro (L3 ST) | - |
| General | MAGpie Pro (L3 ST) - Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L3 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L2 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L2 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L2 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L2 MT) | - |
| General | Magpie Pro (L3 MT) - Magpie Pro (L2 MT) | - |
| General / Magpie Pro: L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT & L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT, L2 MT,
| Language / OCR: IAM: 5.7 K; SynthDog-EN: 40.1 K; GQA: 72.1 K; GeoQA+: MathV360K: 17.2 K; GeoQA+: MathV360K: 17.2 K; MathQA: 29.8 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; MathQA: 29.8 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV360K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GeOQA+: MathV356K: 17.2 K; GQA: 72.1 K; Image Recognition: 17.8 K; UReader KG: 37.6 K; ScreenWords: 15.7 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K; UReader KG: 37.6K
General / Magpie Pro: IAM: 5.7 K; SynthDog-EN: 40.1 K; GQA: 72.1 K; Image Recognition: 18.8 K; UReader KG: 37.6 K; ScreenWords: 15.7 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.4 K; Image Recognition: 18.8 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37.6 K
General / Magpie Pro: IAM: 5.7 K; SynthDog-EN: 40.1 K; GQA: 72.1 K; Image Recognition: 18.8 K; UReader KG: 37.6 K; ScreenWords: 15.7 K; UReader KG: 37.6 K; UReader KG: 37.6 K; UReader KG: 37 .8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K; Image Recognition: 18.8K
General / Magpie Pro: IAM: 5.7K; SynthDog-EN: 40.1K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IAM: 5.7K
General / Magpie Pro: IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN * IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN : IIMN<nl>
</details>

Figure 4: Single-Image 3.2M. A High-Quality Single-Image Dataset Collection. Left: Data Distribution within Each Category. The outer circle shows the distribution of all data categories and the inner circle shows the distribution of data subsets. Right: The detailed quantities of datasets.

![](images/3e07e2df11730296d83a7cf062ea50c62e33fc280fe1b51ce7443620dd3ac05d.jpg)

<details>
<summary>pie</summary>

OneVision 1.6M
| Category | Value |
|---|---|
| Single-Image | 250 |
| Multi-Image | 300 |
| Video | 400 |
</details>

![](images/b8c9dc7ced046383c7cd251aa609c397753cc6942654066137531389df6da12a.jpg)

<details>
<summary>text_image</summary>

Single-Image (31.2%)
■ Magpie Pro (90.0K) ■ Vision FLAN (filtered) (55.8K) ■ Image Textualization (49.8K)
■ Caudron (40.2K) ■ UReader (39.9K) ■ ShareGPT4V (21.0K) ■ ALLaVA Inst. (21.0K)
■ Cambrian (filtered GPT4o) (24.9K) ■ LLAVA-Wild (train) (10.9K) ■ LAION-GPT4V (8.0K) ■ LLAVA-158K (7.0K)
■ Geo170K-QA (6.8K) ■ Geo170K-Align (6.0K) ■ ShareGPT4o (5.7K) ■ TabMWP (4.5K)
■ LLAVAR GPT4 (4.0K) ■ MapQA (4.3K) ■ MathQA (3.0K) ■ TextOCR (GPT4V) (2.5K)
■ TextCaps (2.2K) ■ ScienceQA (1.9K) ■ FigureQA (1.8K) ■ GeoQA+ (1.7K)
■ A12D (InternVL) (1.2K) ■ UniGeo (1.2K) ■ IconQA (1.1K) ■ LRV-Normal (filtered) (1.1K)
■ TQA (1.0K) ■ Geometry3K (1.0K) ■ Super-CLEVR (0.9K) ■ A12D (GPT4V) (0.7K)
■ VizWiz (0.7K) ■ VQA-AS (0.6K) ■ CLEVR-Math (0.5K) ■ PlotQA (0.5K)
■ GEOS (0.5K) ■ InfoVQA (0.9K) ■ PMC-VQA (0.4K) ■ Geo3K (0.2K)
■ VQA-RAD (0.2K) ■ LRV-Chart (0.2K)

Multi-Image (43.0%)
■ NLVR (86.4K) ■ Co-Instruct (50.0K) ■ ScanNet (49.9K)
■ RAVEN (35.0K) ■ IconQA (34.6K) ■ VIST (26.0K) ■ ScanQA (25.6K)
■ ContrastiveCaption (25.2K) ■ ALFRED (22.6K) ■ FlintstonesSV (22.3K) ■ ImageCode (16.6K)
■ DreamSim (15.9K) ■ Birds-to-Words (14.3K) ■ PororoSV (12.3K) ■ Spot-the-Diff (10.8K)
■ nuScenes (9.8K) ■ VISION (9.9K) ■ WebQA (9.3K) ■ RecipeQA-VisualCloze (8.7K)
■ RecipeQA-ImageCoherence (8.7K) ■ TQA (MI) (8.2K) ■ AESOP (6.9K) ■ HQ-Edit-Diff (7.0K)
■ MagicBrush-Diff (6.7K) ■ COMICS-Dialogue (5.9K) ■ MultiVQA (5.0K) ■ VizWiz (MI) (4.9K)
■ CLEVR-Change (3.9K) ■ NextQA (3.9K) ■ IEdit (3.5K) ■ Star (3.0K)
■ DocVQA (MI) (1.9K) ■ MIT-PropertyCoherence (1.9K) ■ MIT-StateCoherence (1.9K) ■ OCR-VQA (MI) (1.9K)

Video (25.9%)
■ ActivityNet (6.5K) ■ Charades (23.6K) ■ Ego4D (0.8K)
■ NextQA (9.5K) ■ ShareGPT4Video (255.0K) ■ Youcook2 (41.9K)
</details>

Figure 5: OneVision 1.6M. A high-quality single-image, multi-image and video dataset collection. Left: Data Distribution within each category. The outer circle shows the distribution of all data categories and the inner circle shows the distribution of data subsets. Right: The detailed quantities of datasets. “MI” means it is the multi-image version dataset proposed by DEMON [69].

in [68]. The data distribution and details are presented in Figure 5, with additional information available in Appendix E.2.

# 5 Training Strategies

To enable LLM for multimodal capabilities, we identify three critical functionalities, and systematically divide them into three distinct learning stages for the purpose of ablation studies. As with most existing research, prior LLaVA models mainly explore the single-image instruction tuning. However, other parts are less frequently investigated and therefore constitute the primary focus of this section.

We train the model via a curriculum learning principle, where training objectives and examples of increasing difficulty are observed in a stage-wise manner. With a fixed compute budget, this strategy helps decompose the training process and produces immediate checkpoints that can be re-used in more experiment trails.

- Stage-1: Language-Image Alignment. The goal is to well align the visual features into the word embedding space of LLMs.   
- Stage-1.5: High-Quality Knowledge Learning. To strike a balance between compute-efficiency and injecting new knowledge into LMMs, we recommend to consider the high-quality knowledge for LMM learning. The training configuration mirrors the settings used in Stage-2, ensuring consistency and allowing the model to integrate new information seamlessly.

\- Stage-2: Visual Instruction Tuning. To teach LMM to solve a diverse set of visual task with preferred responses, we organize the instruction data into different groups, described in Section 4.2. The model is scheduled to train on these groups in order.

Specifically, the visual instruction tuning process consists of two phases: (i) Single-Image Training: The model is first trained on 3.2 million single-image instructions, resulting in a model with strong performance in following a diverse set of instructions to complete visual tasks using a single image. (ii) OneVision Training: The model is then trained on a mixture of video, single-image, and multi-image data. In this phase, the model expands its capabilities from single-image scenarios to diverse scenarios. It learns to follow instructions to complete tasks in each new scenario and transfer the learned knowledge across different scenarios, resulting in new emergent capabilities. Note that the proposed OneVision training in the post-training stage is probably the simplest and most cost-efficient way to empower the LMMs with the multi-image and video understanding capabilities.

The training strategy is summarized in Table 1. We progressively train the model to deal with long sequence training. The maximum image resolution and the number of visual tokens gradually increase as training progresses. In Stage-1, the base image representation is considered with 729 tokens. In Stages 1.5 and 2, AnyRes is considered with up to 5 times and 10 times more visual tokens, respectively. Regarding trainable modules, Stage-1 updates only the projector, while the subsequent stages update the full model. It is also noted that the learning rate for the vision encoder is 5 times smaller than that for the LLM.

![](images/974b55cacc52d35ca2499a13c47ac3eeadd761aa044b78cfd3115067506c1c0d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Language-Image Alignment"] --> B["High-Quality Knowledge Learning"]
    B --> C["Visual Instruction Tuning"]
```
</details>

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Stage-1</td><td rowspan="2">Stage-1.5</td><td colspan="2">Stage-2</td></tr><tr><td>Single-Image</td><td>OneVision</td></tr><tr><td rowspan="2">Vision</td><td>Resolution</td><td>384</td><td> $384 \times \{2 \times 2, 1 \times \{2,3\}, \{2,3\} \times 1\}$ </td><td> $384 \times \{\{1 \times 1\}, \cdots, \{6 \times 6\}\}$ </td><td> $384 \times \{\{1 \times 1\}, \cdots, \{6 \times 6\}\}$ </td></tr><tr><td>#Tokens</td><td>729</td><td>Max  $729 \times 5$ </td><td>Max  $729 \times 10$ </td><td>Max  $729 \times 10$  (See Fig. 3)</td></tr><tr><td rowspan="2">Data</td><td>Dataset</td><td>LCS</td><td>Image (Sec. 4.1)</td><td>Image (Sec. 4.2)</td><td>(Multi)-Image &amp; Video (Sec. 4.2)</td></tr><tr><td>#Samples</td><td>558K</td><td>4M</td><td>3.2M</td><td>1.6M</td></tr><tr><td rowspan="4">Model</td><td>Trainable</td><td>Projector</td><td>Full Model</td><td>Full Model</td><td>Full Model</td></tr><tr><td>0.5B LLM</td><td>1.8M</td><td>0.8B</td><td>0.8B</td><td>0.8B</td></tr><tr><td>7.6B LLM</td><td>20.0M</td><td>8.0B</td><td>8.0B</td><td>8.0B</td></tr><tr><td>72.7B LLM</td><td>72.0M</td><td>73.2B</td><td>73.2B</td><td>73.2B</td></tr><tr><td rowspan="4">Training</td><td>Batch Size</td><td>512</td><td>256/512</td><td>256/512</td><td>256/512</td></tr><tr><td>LR:  $\psi_{vision}$ </td><td> $1 \times 10^{-3}$ </td><td> $2 \times 10^{-6}$ </td><td> $2 \times 10^{-6}$ </td><td> $2 \times 10^{-6}$ </td></tr><tr><td>LR:  $\{ \theta_{proj}, \phi_{LLM} \}$ </td><td> $1 \times 10^{-3}$ </td><td> $1 \times 10^{-5}$ </td><td> $1 \times 10^{-5}$ </td><td> $1 \times 10^{-5}$ </td></tr><tr><td>Epoch</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>

Table 1: Detailed configuration for each training stage of the LLaVA-OneVision model. The table outlines the progression of vision parameters, dataset characteristics, model specifications, and training hyperparameters across different stages of the curriculum learning process. We use a global batch size of 512 for the 0.5B model, and 256 for the 7B and 72B models.

# 6 Experimental Results

We conduct standardized and reproducible evaluations for LLaVA-OneVision models on all benchmarks using LMMs-Eval $[161]$ . For fair comparison with other leading LMMs, we primarily report results from original papers. When results are unavailable, we onboard the models in LMMs-Eval and evaluate them using consistent settings. All our results are reported with greedy decoding and 0-shot settings unless otherwise specified.

To reveal the generality and effectiveness of the designed paradigm, we comprehensively evaluate our LLaVA-OneVision models across different modalities in Table 2, including single-image, multi-image, and video benchmarks. Detailed results for each modality are presented in Table 3, Table 4, and Table 5, respectively. We denote the model checkpoint trained after the single-image stage and one-vision stage as LLaVA-OV (SI) or LLaVA-OV, respectively

Three model sizes are provided (0.5B, 7B and 72B), to accommodate applications with different performance-throughput trade-off, ranging from edge device to cloud serving. The GPT-4V and GPT-4o results are presented as references. Our largest model LLaVA-OneVision-72B yields superior performance between GPT-4V and GPT-4o on most benchmarks. It suggests that the proposed recipe is effective, revealing a promising path for further scaling. However, a relatively larger gap remains in complex tasks such as visual chat scenarios, we leave it as future research in stronger LLMs, larger training data and better preference learning.

# 6.1 Single-Image Benchmarks

To validate the performance for single-image tasks in real-world scenarios, we consider a comprehensive set of image benchmarks in Table 3. It can be categorized into three classes:

(1) Chart, Diagram, and Document Understanding. As the main visual formats for structured OCR data, we evaluate the results on AI2D [54], ChartQA [101], DocVQA [103], and InfoVQA [102] benchmarks. Though current open-source models such as InternVL [22] and Cambrian [133] achieve performance comparable to commercial models, LLaVA-OneVision goes a step further, surpassing GPT-4V [109] and approaching the performance level of GPT-4o [110].   
(2) Perception and Multi-discipline Reasoning. Including visual perception scenarios, we reveal the potentials of our model for more complex and challenging reasoning tasks. Specifically, we adopt the perception benchmarks including MME [151], MMBench [86], and MMVet [154], and reasoning benchmarks such as MathVerse [165], MathVista [90], and MMMU [157]. The results of LLaVA-OneVision significantly outperforms GPT-4V on various benchmarks, and comparable to GPT-4o on MathVista. This further confirms the superiority of our framework in visual perception and reasoning tasks.   
(3) Real-world Understanding and Visual Chat. We consider the evaluation of LMMs as general-purpose assistant in the wild as the most important metrics, beyond the lab environments. To validate the capabilities in real-world scenarios, we utilize several widely-adopted benchmarks, including RealworldQA $[141]$ , Vibe-Eval $[111]$ , MM-LiveBench $[161]$ , and LLaVA-Bench-Wilder $[65]$ . While our model still has room for improvement compared to GPT-4V and GPT-4o, it achieves competitive performance with open-source models of similar parameter size. Notably, our model performs well on MM-LiveBench $[161]$ , a benchmark for real-world internet content with constantly updated content, demonstrating the model's broad world knowledge and strong generalization abilities.

# 6.2 Multi-Image Benchmarks

We further evaluate LLaVA-OneVision in multi-image interleaved settings, where users may ask questions between multiples images. In particular, we perform comprehensive assessment on the diverse subtasks of LLaVA-Interleave Bench $[68]$ , such as Spot the Difference $[45]$ , Image Edit Instruction (IEI) $[68]$ , Visual Storytelling (VST) $[40]$ , Text-rich VQA (TR-VQA) $[85]$ , Multi-image VQA (MI-VQA) $[117]$ , Raven Puzzle $[24]$ , Q-Bench (QB) $[139]$ , and NLVR2 $[125]$ . We also utilize several multi-view benchmarks for evaluation, which depict 3D environments with multiple viewpoints, including 3D Dialogue (3D-Chat) and Task Decomposition (3D-TD) from 3D-LLM $[38]$ , ScanQA $[5]$ , ALFRED $[122]$ , and nuScenes VQA $[9]$ . We refer to these datasets as in-domain evaluations, since our training data includes the training split of them.

Moreover, we conduct evaluations on different out-domain tasks, which reveals the generalization capability of our approach. They include the multi-image split of math QA benchmark MathVerse $[165]$ and science QA benchmark SciVerse $[34]$ , multi-image perception benchmark BLINK $[31]$ , MMMU(multi-image) $[157]$ that contains all multi-image QA in MMMU, and MuirBench $[135]$ spanning 12 diverse multi-image tasks.

As shown in Table 4, LLaVA-OneVision (SI) consistently outperforms existing multi-image LMMs in all benchmarks. After additional tuning on multi-image and video data, LLaVA-OneVision shows a marked improvement over GPT-4V in specific areas, with significant margins. This highlights its strong performance in complex tasks such as multi-image reasoning, identifying differences, and understanding 3D environments. In addition, we observe a consistent performance enhancement on after the one-vision training stage, which is more evident on multi-view benchmarks that are absent

<table><tr><td>Capability</td><td>Benchmark</td><td>LLaVAOneVision-0.5B</td><td>LLaVAOneVision-7B</td><td>LLaVAOneVision-72B</td><td>GPT-4V(V-Preview)</td><td>GPT-4o</td></tr><tr><td rowspan="18">Single-Image</td><td>†AI2D [53]Science Diagrams</td><td>57.1%</td><td>81.4%</td><td>85.6%</td><td>78.2%</td><td>94.2%</td></tr><tr><td>†ChartQA [101]Chart Understanding</td><td>61.4%</td><td>80.0%</td><td>83.7%</td><td>78.5%</td><td>85.7%</td></tr><tr><td>†DocVQA [103] (test)Document Understanding</td><td>70.0%</td><td>87.5%</td><td>91.3%</td><td>88.4%</td><td>92.8%</td></tr><tr><td>†InfoVQA [102] (test)Infographic Understanding</td><td>41.8%</td><td>68.8%</td><td>74.9%</td><td>-</td><td>-</td></tr><tr><td>MathVerse [165] (vision-mini)Professional Math Reasoning</td><td>17.9%</td><td>26.2%</td><td>39.1%</td><td>32.8%</td><td>50.2%</td></tr><tr><td>MathVista [90] (testmini)General Math Understanding</td><td>34.8%</td><td>63.2%</td><td>67.5%</td><td>49.9%</td><td>63.8%</td></tr><tr><td>MMBench [86] (en-dev)Multi-discip</td><td>52.1%</td><td>80.8%</td><td>85.9%</td><td>75.0%</td><td>-</td></tr><tr><td>MME [28] (cog./perp.)Multi-discip</td><td>240/1238</td><td>418/1580</td><td>579/1682</td><td>517/1409</td><td>-</td></tr><tr><td>MMStar [19]Multi-discip</td><td>37.5%</td><td>61.7%</td><td>66.1%</td><td>57.1%</td><td>-</td></tr><tr><td>MMMU [157] (val)College-level Multi-disp</td><td>31.4%</td><td>48.8%</td><td>56.8%</td><td>56.8%</td><td>69.1%</td></tr><tr><td>MMVet [153]Multi-discip</td><td>29.1%</td><td>57.5%</td><td>63.7%</td><td>49.9%</td><td>76.2%</td></tr><tr><td>SeedBench [66] (image)Multi-discip; Large-scale</td><td>65.5%</td><td>75.4%</td><td>78.0%</td><td>49.9%</td><td>76.2%</td></tr><tr><td>†ScienceQA [93]High-school Science</td><td>67.2%</td><td>96.0%</td><td>90.3%</td><td>75.7%</td><td>-</td></tr><tr><td>ImageDC [65]Image Detail Description</td><td>83.3%</td><td>88.2%</td><td>91.2%</td><td>91.5%</td><td>-</td></tr><tr><td>RealworldQA [141]Realwold QA</td><td>55.6%</td><td>66.3%</td><td>71.9%</td><td>61.4%</td><td>-</td></tr><tr><td>Vibe-Eval [112]Chanllenging Cases</td><td>33.8%</td><td>51.7%</td><td>50.7%</td><td>57.9%</td><td>63.1%</td></tr><tr><td>MM-LiveBench [161] (2406)Internet Content Understanding</td><td>49.9%</td><td>77.1%</td><td>81.5%</td><td>-</td><td>92.4%</td></tr><tr><td>LLaVA-Wilder [65] (small)Realworld Chat</td><td>55.0%</td><td>67.8%</td><td>72.0%</td><td>81.0%</td><td>85.9%</td></tr><tr><td rowspan="5">Multi-Image</td><td>LLaVA-Interleave [68]Out-domain</td><td>33.3%</td><td>64.2%</td><td>79.9%</td><td>60.3%</td><td>-</td></tr><tr><td>MuirBench [135]Comprehensive Multi-image</td><td>25.5%</td><td>41.8%</td><td>54.8%</td><td>62.3%</td><td>-</td></tr><tr><td>Mantis [47]Multi-image in the Wild</td><td>39.6%</td><td>64.2%</td><td>77.6%</td><td>62.7%</td><td>-</td></tr><tr><td>BLINK [31]Unusual Visual Scenarios</td><td>52.1%</td><td>48.2%</td><td>55.4%</td><td>51.1%</td><td>-</td></tr><tr><td>†Text-rich VQA [84]OCR, Webpage, Ducument</td><td>65.0%</td><td>80.1%</td><td>83.7%</td><td>54.5%</td><td>-</td></tr><tr><td rowspan="9">Video</td><td>ActivityNetQA [155]Spatio-Temporal Reasoning</td><td>50.5%</td><td>56.6%</td><td>62.3%</td><td>57.0%</td><td>-</td></tr><tr><td>EgoSchema [98]Egocentric Video Understanding</td><td>26.8%</td><td>60.1%</td><td>62.0%</td><td>-</td><td>-</td></tr><tr><td>PerceptionTest [115]Perception and Reasoning</td><td>49.2%</td><td>57.1%</td><td>66.9%</td><td>-</td><td>-</td></tr><tr><td>SeedBench [66] (video)Multi-discip; Video</td><td>44.2%</td><td>56.9%</td><td>62.1%</td><td>60.5%</td><td>-</td></tr><tr><td>LongVideoBench [138] (val)Long Video</td><td>45.8%</td><td>56.3%</td><td>63.2%</td><td>60.7%</td><td>66.7%</td></tr><tr><td>MLVU [170]Long Video Understanding</td><td>50.3%</td><td>64.7%</td><td>68.0%</td><td>49.2%</td><td>64.6%</td></tr><tr><td>MVBench [71]Multi-discip</td><td>45.5%</td><td>56.7%</td><td>59.4%</td><td>43.5%</td><td>-</td></tr><tr><td>VideoChatGPT [97]Video Conversation</td><td>3.12</td><td>3.49</td><td>3.62</td><td>4.06</td><td>-</td></tr><tr><td>VideoMME [29]Multi-discip</td><td>44.0%</td><td>58.2%</td><td>66.2%</td><td>59.9%</td><td>71.9%</td></tr></table>

Table 2: Performance comparison to state-of-the-art commercial models with our LLaVA-OneVision models (0.5B to 72B parameters) across diverse evaluation benchmarks spanning multiple modalities. $\dagger$ indicates that the training set has been observed in our data mixture.

<table><tr><td rowspan="2">Model</td><td>AI2D</td><td>ChartQA</td><td>DocVQA</td><td>InfoVQA</td><td>MathVerse</td><td>MathVista</td><td>MMBench</td><td>MME</td><td>MMMU</td></tr><tr><td>test</td><td>test</td><td>val/test</td><td>val/test</td><td>mini-vision</td><td>testmini</td><td>en-dev</td><td>test</td><td>val</td></tr><tr><td>Qwen-VL-Max [8]</td><td>79.3</td><td>79.8</td><td>-/93.1</td><td>-</td><td>23.0</td><td>51.0</td><td>77.6</td><td>2281</td><td>51.4</td></tr><tr><td>Gemini-1.5-Pro [130]</td><td>94.4</td><td>87.2</td><td>-/93.1</td><td>-/81.0</td><td>-</td><td>63.9</td><td>-</td><td>-</td><td>62.2</td></tr><tr><td>Claude 3.5 Sonnet [3]</td><td>94.7</td><td>90.8</td><td>-/95.2</td><td>49.7</td><td>-</td><td>67.7</td><td>-</td><td>-</td><td>68.3</td></tr><tr><td>GPT-4V [109]</td><td>78.2</td><td>78.5*</td><td>-/88.4</td><td>-</td><td>32.8</td><td>49.9</td><td>75.0</td><td>517/1409</td><td>56.8</td></tr><tr><td>GPT-4o [110]</td><td>94.2</td><td>85.7</td><td>-/92.8</td><td>-</td><td>50.2</td><td>63.8</td><td>-</td><td>-</td><td>69.1</td></tr><tr><td>Cambrian-34B [133]</td><td>79.7</td><td>73.8</td><td>-/75.5</td><td>-</td><td>-</td><td>53.2</td><td>81.4</td><td>-</td><td>49.7</td></tr><tr><td>VILA-34B [77]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>82.4</td><td>1762</td><td>51.9</td></tr><tr><td>IXC-2.5-7B [162]</td><td>81.5</td><td>82.2</td><td>-/90.9</td><td>-/70.0</td><td>20.0</td><td>59.6</td><td>82.2</td><td>2229</td><td>42.9</td></tr><tr><td>InternVL-2-8B [22]</td><td>83.8</td><td>83.3</td><td>-/91.6</td><td>-/74.8</td><td>27.5</td><td>58.3</td><td>81.7</td><td>2210</td><td>49.3</td></tr><tr><td>InternVL-2-26B [22]</td><td>84.5</td><td>84.9</td><td>-/92.9</td><td>-/75.9</td><td>31.3</td><td>59.4</td><td>83.4</td><td>2260</td><td>48.3</td></tr><tr><td>LLaVA-OV-0.5B (SI)</td><td>54.2</td><td>61.0</td><td>75.0/71.2</td><td>44.8/41.3</td><td>17.3</td><td>34.6</td><td>43.8</td><td>272/1217</td><td>31.2</td></tr><tr><td>LLaVA-OV-0.5B</td><td>57.1</td><td>61.4</td><td>73.7/70.0</td><td>46.3/41.8</td><td>17.9</td><td>34.8</td><td>52.1</td><td>240/1238</td><td>31.4</td></tr><tr><td>LLaVA-OV-7B (SI)</td><td>81.6</td><td>78.8</td><td>89.3/86.9</td><td>69.9/65.3</td><td>26.9</td><td>56.1</td><td>81.7</td><td>483/1626</td><td>47.3</td></tr><tr><td>LLaVA-OV-7B</td><td>81.4</td><td>80.0</td><td>90.2/87.5</td><td>70.7/68.8</td><td>26.2</td><td>63.2</td><td>80.8</td><td>418/1580</td><td>48.8</td></tr><tr><td>LLaVA-OV-72B (SI)</td><td>85.1</td><td>84.9</td><td>93.5/91.8</td><td>77.7/74.6</td><td>37.7</td><td>66.5</td><td>86.6</td><td>563/1706</td><td>57.4</td></tr><tr><td>LLaVA-OV-72B</td><td>85.6</td><td>83.7</td><td>93.1/91.3</td><td>79.2/74.9</td><td>39.1</td><td>67.5</td><td>85.9</td><td>579/1682</td><td>56.8</td></tr></table>

<table><tr><td rowspan="2">Model</td><td colspan="10">MMVet MMStar S-Bench S-QA ImageDC MMLBench RealWorldQA Vibe-Eval LLaVA-W L-Wilder</td></tr><tr><td>test</td><td>test</td><td>image</td><td>test</td><td>test</td><td>2024-06</td><td>test</td><td>test</td><td>test</td><td>small</td></tr><tr><td>Qwen-VL-Max [8]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gemini-1.5-Pro [130]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>85.9</td><td>70.4</td><td>60.4</td><td>-</td><td>-</td></tr><tr><td>Claude 3.5 Sonnet [3]</td><td>75.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>92.3</td><td>59.9</td><td>66.2</td><td>102.9</td><td>83.1</td></tr><tr><td>GPT-4V [109]</td><td>49.9</td><td>57.1</td><td>49.9</td><td>75.7</td><td>91.5</td><td>-</td><td>61.4</td><td>57.9</td><td>98.0</td><td>81.0</td></tr><tr><td>GPT-4o [110]</td><td>76.2</td><td>-</td><td>76.2</td><td>-</td><td>92.5</td><td>92.4</td><td>58.6</td><td>63.1</td><td>106.1</td><td>85.9</td></tr><tr><td>Cambrian-34B [133]</td><td>-</td><td>-</td><td>-</td><td>85.6</td><td>-</td><td>-</td><td>67.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VILA-34B [77]</td><td>53.0</td><td>-</td><td>75.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.3</td><td>-</td><td>-</td></tr><tr><td>IXC-2.5-7B [162]</td><td>51.7</td><td>59.9</td><td>75.4</td><td>-</td><td>87.5</td><td>-</td><td>67.8</td><td>45.2</td><td>78.1</td><td>61.4</td></tr><tr><td>InternVL-2-8B [22]</td><td>60.0</td><td>59.4</td><td>76.0</td><td>97.0</td><td>87.1</td><td>73.4</td><td>64.4</td><td>46.7</td><td>84.5</td><td>62.5</td></tr><tr><td>InternVL-2-26B [22]</td><td>65.4</td><td>60.4</td><td>76.8</td><td>97.5</td><td>91.0</td><td>77.2</td><td>66.8</td><td>51.5</td><td>99.6</td><td>70.2</td></tr><tr><td>LLaVA-OV-0.5B (SI)</td><td>26.9</td><td>36.3</td><td>63.4</td><td>67.8</td><td>83.0</td><td>43.2</td><td>53.7</td><td>34.9</td><td>71.2</td><td>51.5</td></tr><tr><td>LLaVA-OV-0.5B</td><td>29.1</td><td>37.5</td><td>65.5</td><td>67.2</td><td>83.3</td><td>49.9</td><td>55.6</td><td>33.8</td><td>74.2</td><td>55.0</td></tr><tr><td>LLaVA-OV-7B (SI)</td><td>58.8</td><td>60.9</td><td>74.8</td><td>96.6</td><td>85.7</td><td>75.8</td><td>65.5</td><td>47.2</td><td>86.9</td><td>69.1</td></tr><tr><td>LLaVA-OV-7B</td><td>57.5</td><td>61.7</td><td>75.4</td><td>96.0</td><td>88.9</td><td>77.1</td><td>66.3</td><td>51.7</td><td>90.7</td><td>67.8</td></tr><tr><td>LLaVA-OV-72B (SI)</td><td>60.0</td><td>65.2</td><td>77.6</td><td>91.3</td><td>91.5</td><td>84.4</td><td>73.8</td><td>46.7</td><td>93.7</td><td>72.9</td></tr><tr><td>LLaVA-OV-72B</td><td>63.7</td><td>66.1</td><td>78.0</td><td>90.3</td><td>91.2</td><td>81.5</td><td>71.9</td><td>50.7</td><td>93.5</td><td>72.0</td></tr></table>

Table 3: LLaVA-OneVision performance on single-image benchmarks. \*GPT-4V reports 4-shot results on ChartQA. All results are reported as 0-shot accuracy.

in single-image data. This demonstrates the significance of our one-vision paradigm for empowering LMMs with comprehensive visual capabilities.

# 6.3 Video Benchmarks

Video is also a common modality to build world model, capturing the dynamic nature of the real world over time. We conduct experiments on several open-ended and multi-choice video benchmarks. These include ActivityNet-QA $[155]$ that contains human-annotated action-related QA pairs derived from ActivityNet dataset, EgoSchema $[98]$ and MLVU $[170]$ focusing on long video understanding, PerceptionTest $[115]$ designed to evaluate the perception skills, VideoMME $[29]$ and NeXTQA $[142]$ containing diverse video domains and durations (from minutes to hours), VideoDetailCaption $[87]$ and Video-ChatGPT $[96]$ for video detailed description and visua chat, respectively.

As shown in Table 5, LLaVA-OneVision achieves comparable or better results than previous open source models with much larger LLMs. The superiority of LLaVA-OneVision is particularly evident in complex benchmarks such as EgoSchema and VideoMME. Even compared to the advanced commercial model GPT-4V, LLaVA-OneVision performs competitively on the ActivityNet-QA, MLVU, and VideoMME benchmarks.

<table><tr><td rowspan="2">Model</td><td>IEI</td><td>ML-VQA</td><td>NLVR2</td><td>Puzzle</td><td>Q-Bench</td><td>Spot-Diff</td><td>TR-VQA</td><td>VST</td><td>3D-Chat</td><td>3D-TD</td><td>ScanQA</td><td>ALFRED</td><td>nuScenes</td><td>BLINK</td><td>Mantis</td><td>MathVerse</td><td>MuirBench</td><td>SciVerse</td></tr><tr><td colspan="8">in-domain multi-image</td><td colspan="5">in-domain multi-view</td><td colspan="5">out-domain</td></tr><tr><td>GPT-4V [109]</td><td>11.0</td><td>52.0</td><td>88.8</td><td>17.1</td><td>76.5</td><td>12.5</td><td>54.5</td><td>10.9</td><td>31.2</td><td>35.4</td><td>32.6</td><td>10.3</td><td>63.7</td><td>51.1</td><td>62.7</td><td>60.3</td><td>62.3</td><td>66.9</td></tr><tr><td>LLaVA-N-Image-7B $^{\dagger }$  [82]</td><td>13.2</td><td>39.4</td><td>68.0</td><td>9.0</td><td>51.0</td><td>12.9</td><td>59.6</td><td>10.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>41.8</td><td>46.1</td><td>13.5</td><td>-</td><td>12.2</td></tr><tr><td>VPG-C-7B [70]</td><td>15.2</td><td>46.8</td><td>73.2</td><td>2.4</td><td>57.6</td><td>27.8</td><td>38.9</td><td>21.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>43.1</td><td>52.4</td><td>24.3</td><td>-</td><td>23.1</td></tr><tr><td>Mantis-7B [47]</td><td>11.2</td><td>52.5</td><td>87.4</td><td>25.7</td><td>69.9</td><td>17.6</td><td>45.2</td><td>12.5</td><td>2.60</td><td>14.7</td><td>16.1</td><td>14.0</td><td>46.2</td><td>46.4</td><td>59.5</td><td>27.2</td><td>36.1</td><td>29.3</td></tr><tr><td>LLaVA-N-Inter-7B [68]</td><td>24.3</td><td>87.5</td><td>88.8</td><td>48.7</td><td>74.2</td><td>37.1</td><td>76.1</td><td>33.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>52.6</td><td>62.7</td><td>32.8</td><td>38.9</td><td>31.6</td></tr><tr><td>LLaVA-N-Inter-14B [68]</td><td>24.5</td><td>95.0</td><td>91.1</td><td>59.9</td><td>76.7</td><td>40.5</td><td>78.6</td><td>33.3</td><td>70.6</td><td>52.2</td><td>34.5</td><td>62.0</td><td>76.7</td><td>52.1</td><td>66.4</td><td>33.4</td><td>40.7</td><td>32.7</td></tr><tr><td>LLaVA-OV-0.5B (SI)</td><td>15.6</td><td>44.8</td><td>56.1</td><td>30.0</td><td>45.8</td><td>8.5</td><td>36.7</td><td>7.6</td><td>22.1</td><td>22.1</td><td>16.9</td><td>25.5</td><td>8.2</td><td>37.9</td><td>38.2</td><td>20.9</td><td>22.7</td><td>26.7</td></tr><tr><td>LLaVA-OV-0.5B</td><td>17.1</td><td>48.7</td><td>63.4</td><td>35.4</td><td>48.8</td><td>36.4</td><td>65.0</td><td>29.8</td><td>60.0</td><td>48.0</td><td>29.4</td><td>62.2</td><td>70.5</td><td>52.1</td><td>39.6</td><td>60.0</td><td>25.5</td><td>29.1</td></tr><tr><td>LLaVA-OV-7B (SI)</td><td>20.5</td><td>60.3</td><td>75.9</td><td>24.6</td><td>56.0</td><td>7.9</td><td>52.8</td><td>8.4</td><td>24.5</td><td>29.9</td><td>22.1</td><td>32.0</td><td>70.8</td><td>45.6</td><td>54.2</td><td>26.3</td><td>32.7</td><td>30.0</td></tr><tr><td>LLaVA-OV-7B</td><td>22.2</td><td>90.2</td><td>89.4</td><td>53.3</td><td>74.5</td><td>39.2</td><td>80.1</td><td>31.7</td><td>62.8</td><td>52.6</td><td>30.1</td><td>61.0</td><td>79.8</td><td>48.2</td><td>64.2</td><td>67.6</td><td>41.8</td><td>79.1</td></tr><tr><td>LLaVA-OV-72B (SI)</td><td>22.1</td><td>61.2</td><td>78.9</td><td>44.2</td><td>61.5</td><td>15.6</td><td>67.9</td><td>12.1</td><td>30.8</td><td>25.4</td><td>21.9</td><td>43.5</td><td>75.5</td><td>46.0</td><td>56.8</td><td>58.6</td><td>33.2</td><td>65.8</td></tr><tr><td>LLaVA-OV-72B</td><td>22.5</td><td>95.3</td><td>93.8</td><td>63.4</td><td>83.2</td><td>43.3</td><td>83.7</td><td>34.5</td><td>63.2</td><td>53.3</td><td>35.8</td><td>66.3</td><td>78.8</td><td>55.4</td><td>77.6</td><td>91.6</td><td>54.8</td><td>94.9</td></tr></table>

Table 4: LLaVA-OneVision performance on multi-image benchmarks with all results reported in accuracy. $^{\dagger}$ denotes the LLaVA-NeXT-Vicuna-7B (2024-01). We use IEI for Image Edit Instruction, MI-VQA for Multi-image VQA, NLVR2 for Natural Language for Visual Reasoning, SDiff for Spot the Difference, VST for Visual Story Telling, TR-VQA for Text-rich VQA. For MathVerse and SciVerse, we report the accuracy on their multi-image splits.

<table><tr><td rowspan="2">Model</td><td>ActNet-QA</td><td>EgoSchema</td><td>MLVU</td><td>MVBench</td><td>NextQA</td><td>PercepTest</td><td>SeedBench</td><td>VideoChatGPT</td><td>VideoDC</td><td>VideoMME</td><td>L-VideoBench</td></tr><tr><td>test</td><td>test</td><td>m-avg</td><td>test</td><td>mc</td><td>val</td><td>video</td><td>test</td><td>test</td><td>wo/w-subs</td><td>val</td></tr><tr><td>GPT-4V [109]</td><td>57.0</td><td>-</td><td>49.2</td><td>43.5</td><td>-</td><td>-</td><td>60.5</td><td>4.06</td><td>4.00</td><td>59.9/63.3</td><td>61.3</td></tr><tr><td>GPT-4o [110]</td><td>-</td><td>-</td><td>64.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>71.9/77.2</td><td>66.7</td></tr><tr><td>Gemini-1.5-Flash [131]</td><td>55.3</td><td>65.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>70.3/75.0</td><td>61.6</td></tr><tr><td>Gemini-1.5-Pro [131]</td><td>57.5</td><td>72.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>75.0/81.3</td><td>64.0</td></tr><tr><td>VILA-40B [77]</td><td>58.0</td><td>58.0</td><td>-</td><td>-</td><td>67.9</td><td>54.0</td><td>-</td><td>3.36</td><td>3.37</td><td>60.1/61.1</td><td>-</td></tr><tr><td>PLLaVA-34B [143]</td><td>60.9</td><td>-</td><td>-</td><td>58.1</td><td>-</td><td>-</td><td>-</td><td>3.48</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaVA-N-Video-34B [169]</td><td>58.8</td><td>49.3</td><td>-</td><td>-</td><td>70.2</td><td>51.6</td><td>-</td><td>3.34</td><td>3.48</td><td>52.0/54.9</td><td>50.5</td></tr><tr><td>LongVA-7B [163]</td><td>50.0</td><td>-</td><td>56.3</td><td>-</td><td>68.3</td><td>-</td><td>-</td><td>3.20</td><td>3.14</td><td>52.6/54.3</td><td>-</td></tr><tr><td>IXC-2.5-7B [162]</td><td>52.8</td><td>-</td><td>37.3</td><td>69.1</td><td>71.0</td><td>34.4</td><td>-</td><td>3.46</td><td>3.73</td><td>55.8/58.8</td><td>-</td></tr><tr><td>LLaVA-N-Video-32B [169]</td><td>54.3</td><td>60.9</td><td>65.5</td><td>-</td><td>77.3</td><td>59.4</td><td>-</td><td>3.59</td><td>3.84</td><td>60.2/63.0</td><td>-</td></tr><tr><td>LLaVA-OV-0.5B (SI)</td><td>49.0</td><td>33.1</td><td>47.9</td><td>43.3</td><td>53.6</td><td>48.6</td><td>43.4</td><td>3.08</td><td>3.51</td><td>41.7/40.4</td><td>41.9</td></tr><tr><td>LLaVA-OV-0.5B</td><td>50.5</td><td>26.8</td><td>50.3</td><td>45.5</td><td>57.2</td><td>49.2</td><td>44.2</td><td>3.12</td><td>3.55</td><td>44.0/43.5</td><td>45.8</td></tr><tr><td>LLaVA-OV-7B (SI)</td><td>55.1</td><td>52.9</td><td>60.2</td><td>51.2</td><td>61.6</td><td>54.9</td><td>51.1</td><td>3.54</td><td>3.51</td><td>55.0/59.1</td><td>54.3</td></tr><tr><td>LLaVA-OV-7B</td><td>56.6</td><td>60.1</td><td>64.7</td><td>56.7</td><td>79.4</td><td>57.1</td><td>56.9</td><td>3.51</td><td>3.75</td><td>58.2/61.5</td><td>56.4</td></tr><tr><td>LLaVA-OV-72B (SI)</td><td>62.1</td><td>58.6</td><td>60.9</td><td>57.1</td><td>67.2</td><td>62.3</td><td>60.9</td><td>3.55</td><td>3.66</td><td>64.8/66.9</td><td>58.3</td></tr><tr><td>LLaVA-OV-72B</td><td>62.3</td><td>62.0</td><td>68.0</td><td>59.4</td><td>80.2</td><td>66.9</td><td>62.1</td><td>3.62</td><td>3.60</td><td>66.2/69.5</td><td>61.3</td></tr></table>

Table 5: LLaVA-OneVision performance on video benchmarks. We report the score out of 5 for VideoDC, VideoChatGPT while other results are reported in accuracy. All results are reported as 0-shot accuracy.

Within the LLaVA-OV split, the smallest performance difference occurs in PerceptionTest, with a minimal improvement of 0.5 points when scaling the LLM from 0.5B to 7B. This contrasts with at least a 5-point improvement in other datasets. The modest gain at PerceptionTest suggests that LLaVA-OV's perception capabilities may mainly depend on its vision module, supporting findings from recent studies such as those by Qiao et al. [116], which separate the roles of the image encoder and the LLM in perception and reasoning tasks. Notably, for datasets like EgoSchema that demand significant reasoning, a larger LLM substantially enhances performance.

Moreover, in comparing LLaVA-OV-7B (SI) with LLaVA-OV-7B, the smallest improvement is seen with ActivityNet-QA. This suggests that LLaVA-OV-7B (SI), which is trained only on images, can

already perform well on this dataset. Delving into ActivityNet-QA, it becomes apparent that many questions can be answered by observing just a single frame from the video. For instance, the question "What's the color of the ball?" can be answered throughout the video as the ball is visible from start to finish. This scenario does not require the model to understand the video sequence, allowing LLaVA-OV-7B (SI) to perform well.

# 7 Emerging Capabilities with Task Transfer

In addition to reporting the LLaVA-OneVision's capabilities across various benchmarks, we also observe the emerging behaviors of the proposed model with task transfer and composition, paving a promising way to generalize to tackle real-world computer vision tasks in the wild. We illustrate several emerging capabilities using examples as below.

S1: Joint understanding of diagram and chart (Transfer from single-image to multi-image) The capability to understand tables and charts are separately learned from single image diagram and single-image chart understanding data, and the joint understanding task of table and chart do not appear in multi-image data. As shown in Table 6, LLaVA-OneVision is capable of understanding and reasoning over the joint of diagram and chart.

S2: GUI for multi-modal agent (Transfer from single-image and multi-image). Understanding GUIs and applying multimodal models to agentic tasks is of great value. In Table 7, LLaVA-OneVision recognizes the graphical user interface (GUI) screenshots of an iPhone and provides operational instructions to search for and open the TikTok app. This task requires strong OCR capabilities learned from single-image scenarios and relational reasoning skills developed from multi-image scenarios. The example highlights LLaVA-OneVision's proficiency in GUI understanding and task execution.

S3: Set-of-mark Prompting (Transfer from single-image task composition). Different from existing open LLMs, LLaVA-OneVision demonstrates excellent set-of-marks (SoM) reasoning $[149]$ , an emerging capability shown in Table 8. To the best of our knowledge, this is the first time that open LMMs report good emerged SoM ability, as we observe that LLaVA-OneVision is able to produce SoM reasoning for many examples in $[149]$ . This task is not explicitly included in our training data, it is hypothesized that the ability is composed by visual referring and OCR.

S4: Image-to-Video Editing Instruction (Transfer from single-image and video). LLaVA-OneVision could generate detailed video creation prompts based on a static image in Table 9. Given an image and a target video, the model constructs a coherent and vivid narrative for the video, detailing elements such as characters, actions, background settings, and scene specifics. This task leverages both single-image analysis and video comprehension. It is hypothesized that this ability is generalized from the composition of single-image editing instruction task and video detailed description task.

S5: Video-to-Video Difference (Transfer from multi-image and video). Understanding differences in images is a common ability in recent large multimodal models (LMMs), but our models extend this capability to videos. Table 10 showcases LLaVA-OneVision's ability to analyze differences between two video sequences with the same beginning frame but different endings. The model provides a detailed comparison, describing characters, actions, and scene changes. In Table 11, LLaVA-OneVision's describe the differences one by one between videos with a similar background but different main object in the foreground. This task leverages spot the difference in the multi-image analysis to generalize to video scenarios.

S6: Multi-camera Video Understanding in Self-driving (Transfer from single-image and multi-image to video). Understanding videos in a normal aspect ratio is straightforward, what about the videos with multi-views? In Table 12, we observe that LLaVA-OneVision could analyze and interprets multi-camera video footage from self-driving cars. Given video showing four camera views, the model describes each view in detail and plans the ego car's next move. This task combines multi-panel comprehension, video detailed description, and spatial-temporal reasoning.

S7: Composed Sub-video Understanding (Transfer from multi-image to video). Besides multi-view video, we see our model generalize to vertical videos with two sub-scenes. Table 13 demonstrates LLaVA-OneVision's ability to understand and describe the content and layout of a composed sub-video. Given a vertical video with a series of frames featuring a consistent background and a person in the foreground, the model provides a detailed analysis of visual elements, their arrangement, and the narrative context. This task requires single-image analysis, multi-image sequence comprehension, and contextual reasoning.

S8: Visual prompting in video (Task transfer from single-image to video). In Table 14, LLaVA-OneVision is able to understand the highlighted area with a semi-transparent circle in the video, and clearly see the number “10” on the back of the player. The capability of understanding visual prompts and OCR is a capability of single-image LMMs. Our model displays the capability of understanding visual prompts in videos, without training on video data with visual prompts.

S9: Visual Referring in Image in Video Understanding. The ability to refer to image query when answering questions about a video as shown in Table 15. This capability is not seen in LLaVA-NeXT or LLaVA-Interleave, this is probably because strong base single-image training is required for such capability to appear.

# 8 Conclusions

LLaVA-OneVision is a new, open LMM that shines when transferred to a broad range of tasks in the scenarios of single-image, multi-image and videos. The model is developed by consolidating the insights in the LLaVA-NeXT blog series, and is trained by scaling the recipe with a larger dataset and stronger LLMs. Our design allows new capabilities to emerge, through training multiple scenarios together and task transfer, eg, strong visual understanding ability from image to video. Our results demonstrate that LMMs trained with this open recipe and resources achieve state-of-the-art performance across various benchmarks. We also hope that LLaVA-OneVision serves as a valuable starting point for the community to build specific applications, and develop stronger LMMs for diverse vision scenarios through further scaling.

# S1: Joint Understanding of Diagram and Chart from Multi-Image

![](images/accae96ae44a7b54d8186949566067fcc1c76a0f73be58ea528c5e1df75e5712.jpg)

<details>
<summary>text_image</summary>

11
40°
</details>

Insurance Company   
Price 

<table><tr><td>State Farm</td></tr><tr><td>Allstate</td></tr><tr><td>Liberty Mutual</td></tr><tr><td>USAA</td></tr></table>

68   
63   
59   
90   
User

Ross owns a house similar to the brown sector in the following image. They want to get the insurance from Allstate. The price per unit area is given in the following figure. What would be the cost of insuring the whole house? All the computations are rounded to two places of decimal.

LLaVA-OV

To find the cost of insuring the whole house, we need to calculate the area of the sector and then multiply it by the price per unit area. The area of the sector can be calculated using the formula $A = (\theta / 360) * \pi * r^2$ , where $\theta$ is the central angle and $r$ is the radius. In this case, $\theta = 40^\circ$ and $r = 11$ . Plugging these values into the formula, we get $A = (40 / 360) * \pi * 11^2 \sim 38.01$ . The price per unit area is given in the table as \$63. Multiplying the area by the price per unit area, we get the cost of insuring the whole house as $38.01 * \$63 \sim \$2, 386.03$ .

Table 6: LLaVA-OneVision transfers its ability to understand diagram and table to multi-image scenarios, interpreting multiple images in a coherent manner.

# S2: Multi-image GUI Understanding for Agents

00:31   
Search   
![](images/4cf4fa84b8fe3913b07dc6af25be58456d33360fc9b1bbc4a52e78679a4e20d7.jpg)

Charts   
![](images/70be19c515ed4bb523af0224d5e1e07a4eb7bc6e74407e0a1dda10d17cb80fc3.jpg)

100   
![](images/d49fafc83668f1028223f9a95770b45afb55bc58bcf1201587bb671200f62b4b.jpg)

00:31   
![](images/7e070c300870063515e73703fc69129174fc480fdfb21a3d15f289f810e854dd.jpg)

<details>
<summary>text_image</summary>

tiktok
Cancel
tiktok
tiktok shop
tiktok downloader
tiktok live studio
tiktok in Stories
tiktok ltd..in Developers
tiktok likes
tiktok studio
tiktok followers
tiktok followers for free
</details>

![](images/de39178b5b8a67d9c6241f2cafd94f8ed6271e443d7a6f56855e87130d4424c6.jpg)

<details>
<summary>text_image</summary>

00:31
TikTok
TikTok Ltd.
Open
TNI RATINGS
AWARDS
AGE
CHART
4.7
Eternals' Classic
12+
#3
Apples
Years Old
Entertainment
Events
HAPPENING NOW
SPECIAL EVENT
TikTok LIVE Community Fest
Global celebration for LIVE creators and fans!
TikTok
Videos, Music & Live Streams
Open
Today
Games
Apps
Arcade
Search
</details>

![](images/3e2dcad258dce13e92331ae409968f0519351235b8f75baab3a4157a12b9a977.jpg)

<details>
<summary>text_image</summary>

00:31
4 App Store
TikTok
</details>

Suggested   
![](images/8fc52cef31f8bb12f5ebafca064d0f239b4016a2e1d63bc801f3830fc74df400.jpg)

<details>
<summary>text_image</summary>

DoorDash - Food Delivery
Food & Groceries, delivered
Get
Bumble For Friends: Meet IRL
Chat to new people, make a BPF
Get
In-App Purchases
Al Chat - AI Chatbot Assistant
ChatGPT & GPT-46 API's Powered
Get
In-App Purchases
Hay Day
Farm with Friends and Family
Get
In-App Purchases
Candy Crush Saga
Match 3 & Blast Puzzle Game
Get
In-App Purchases
Wink - Meet New People App
Games
Apps
Available
Search
</details>

![](images/ef62bb72540bc925d63af80852efa271b52aacd6885c2b313f5a0658de0727b8.jpg)

<details>
<summary>text_image</summary>

q w e r t y u i o p
a s d f g h j k l
z x c v b n m
123 space search
</details>

![](images/64dd8d623293674dbb7e18056039d60f8a39da6908bfd996d58cf4414abb8ed5.jpg)

<details>
<summary>text_image</summary>

SPECIAL EVENT
TikTok LIVE Community Fest
Digital celebration for LIVE creators and fans!
Open
TikTok
Videos, Music & Live Streams
Today
Games
Apps
Appraise
Search
</details>

User

There are four sequential mobile UI screens, and are transformed by three steps of tap operations. Please respectively describe the three tap operations.

LLaVA-OV

The image shows a sequence of four mobile application interface screens. Here's a breakdown of the three tap operations based on the visible changes:

1. First Tap Operation:

- The user starts by tapping on the search bar at the top of the screen.   
- The keyboard is shown at the bottom of the screen, indicating that the user can now type a search query.   
- The user types "TikTok" in the search bar and taps the search icon on the keyboard to initiate a search.

2. Second Tap Operation:

- After tapping the search icon or pressing 'Enter', the screen displays a list of search results.   
- The user taps on the first result, which is the TikTok application.   
- This action opens the TikTok's app store page, which includes the app icon, developer information ...

3. Third Tap Operation:

- The user taps the "Open" button on the app store page.   
- This action opens the TikTok application, and the screen transitions to the app's main interface.

Table 7: LLaVA-OneVision plays the role of agent. It recognizes multiple screenshots on the iPhone and take action to interact with the iPhone, providing operation instructions for automating tasks.

# S3: Set-of-mark Prompting

![](images/7623668a31d68426ee40c6ccc8166a22fbdf02e13c55eb49dfccdd7d09906347.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern office with white furniture and wall art (no visible text or symbols)
</details>

User   
Here's an image with objects labeled in numbers. Please describe the objects that are marked as 4, 5 and 7.   
The image shows a room with various items marked with numbers. Here's the description for the items marked as 4, 5, and 7: The item marked as 4 is a framed picture hanging on the wall. It features a black and white illustration of a dandelion with seeds blown away, creating a sense of movement. The item marked as 5 is a white bookshelf with multiple shelves. It appears to be a storage unit that is part of the room's furniture. It is located between the two desks. The item marked as 7 is a black chair with a modern design. It has a metal frame and a woven seat, suggesting a contemporary style. This chair is positioned in front of one of the desks.

LLaVA-OV

Table 8: LLaVA-OneVision exhibits excellent set-of-mark prompting capabilities, ie, referring to marks when answering questions. This example demonstrates that describing specific objects based on numerical labels within an image highlights its comprehension skills in handling fine-grained visual content.

# S4: Image-to-Video Editing Instruction

![](images/bd03d789dac32acc91ff6f81c16179935787dfb3713828a0408a61664e0e3487.jpg)

<details>
<summary>natural_image</summary>

Plate of cooked mushrooms and buns on a white dish (no text or symbols visible)
</details>

User

![](images/513bd7219cc78dd440d14f25ba32af325edc820af222bfb32d967971bd2af6b1.jpg)

<details>
<summary>natural_image</summary>

Six-panel photo sequence showing cooked mushrooms and fish on plates, with a close-up of black birds in the final image (no text or symbols)
</details>

Please provide the video generation instruction that can change the source image to the target video.

LLaVA-OV

To generate a video from the provided image, you would need to create a sequence of images that depict the progression of the mushrooms being transformed into penguins. Here's a step-by-step guide on how to create this video:

1. Start with the initial image of the mushrooms on a plate.

2. In the next image, begin to change the appearance of the mushrooms to resemble penguins. Start with the smallest mushrooms and gradually move to the larger ones.

3. Continue to transform the mushrooms into penguins, adding details such as eyes, beaks, and flippers.

4. As the video progresses, more mushrooms should be transformed, and the penguins should appear to interact with each other and the mushrooms.

5. The final images should show a group of penguins interacting with the mushrooms, creating a playful and whimsical scene. Remember to maintain a consistent style and color palette throughout the video to ensure a cohesive look. The transformation should be smooth and realistic, making it seem as though the mushrooms are naturally turning into penguins.

Table 9: LLaVA-OneVision learns to generate detailed video creation prompts based on a static image. This capability is generalized to videos from the image-to-image language editing generation.

# S5: Video-to-Video Difference

![](images/c1ad4c139ad7005cd4d441d0945c772166d312c733cfd637c695ce3bcb0712b0.jpg)

<details>
<summary>natural_image</summary>

Collage of four filmstrip-style photos showing a couple in a casual urban setting, no visible text or symbols.
</details>

![](images/862f7075db0e0f2a060f74953d6f8f3ad2b852991bc9b0ba1e33c8b6f51fabaa.jpg)

<details>
<summary>natural_image</summary>

Four-panel filmstrip-style image showing a man walking outdoors, with no visible text or symbols.
</details>

User

Here are two videos, please provide the analysis on their differences.

LLaVA-OV

The first video shows a sequence of images where a man in a blue plaid shirt approaches a woman in a red dress, engages in a conversation, and eventually gives her a kiss on the cheek. The woman in the red dress is looking at the man with a slight smile, and another woman in a light blue top is observing the scene. The setting appears to be a street with buildings in the background. The second video shows the same man in the blue plaid shirt walking away from the woman in the red dress, who is now looking in the opposite direction. The woman in the light blue top is no longer visible in this sequence. The man is walking down the street, passing by the buildings and other pedestrians. The overall mood of the video changes from the intimate and personal interaction in the first video to a more solitary and neutral scene in the second video.

Table 10: LLaVA-OneVision learns to analyze differences between videos with the same starting frame but different endings.

S5: Video-to-Video Difference   
![](images/fd59d2da4840f6275159dc57d943592525ac5a40f9e6cf2130aacd70a5ea7528.jpg)

<details>
<summary>natural_image</summary>

Sequence of six frames showing a green caterpillar on a leaf, with no visible text or symbols.
</details>

![](images/95fd628c2d1bbcda9f07492638f066ed10111da5401361c5f1175b376f2f78ed.jpg)

<details>
<summary>natural_image</summary>

Five-panel sequence showing a butterfly with blue wings and white spots perched on green leaves, captured in a frame.
</details>

User

What are the differences between the two videos?

LLaVA-OV

The first video features a caterpillar with green and black stripes on its body, resting on a leaf. The caterpillar has white, bristly hair-like structures along its back. The background is blurred, but it appears to be a garden or a natural setting with green foliage and some sunlight filtering through. The second video shows a butterfly with blue and black wings, resting on a leaf. The butterfly has white spots on its wings and an orange patch near the base of the wings. The background is also blurred but suggests a similar garden or natural setting with greenery. The differences between the two videos are:

1. The first video shows a caterpillar, while the second video shows a butterfly.   
2. The caterpillar has green and black stripes, while the butterfly has blue and black wings with white spots.   
3. The caterpillar has white, bristly hair-like structures along its back, which are not present on the butterfly.   
4. The butterfly has a more delicate and transparent appearance compared to the fuzzy texture of the caterpillar.   
5. The wings of the butterfly are spread out, while the caterpillar is in a curled position.   
6. The butterfly has long antennae, which are not visible in the caterpillar's frame.

Table 11: LLaVA-OneVision learns to analyze differences between videos with similar backgrounds but different foreground objects.

S6: Multi-camera Video Understanding in Self-driving   
![](images/02524a46b0fcf79357e30c677e50cf70d4cb4a56ed560a36df9c6077dcc3f105.jpg)

<details>
<summary>text_image</summary>

Street surveillance frames showing traffic detection and pedestrian crossing with bounding boxes and a 'THE WORLD?' sign
</details>

User

Here is a video showing four camera views of a driving car. The top two show front views, and the bottom two show back views. First describe the video of each view in detail, and then plan the next move of the ego car.

# LLaVA-OV

The video consists of six different camera views of a driving car, arranged in two columns with three rows. Each view captures the car from a different angle, showing both front and back perspectives.

In the front views:

1. The first image shows the car approaching an intersection with a pedestrian crossing. There are cars ahead, and the pedestrian is visible on the sidewalk.   
2. The second image shows the car at the intersection, with the pedestrian crossing and cars ahead. The pedestrian is now closer to the car.   
3. The third image shows a pedestrian visible on the sidewalk.

In the back views:

1. The first image shows the car approaching the intersection with pedestrians crossing and a black car aside.   
2. The second image shows the car at the intersection, with several cars behind.   
3. The third image shows the car passing through the intersection, with the pedestrian visible on the sidewalk.

The next move of the ego car would be to continue driving forward, following the road and traffic rules, and ensuring the pedestrian's safety as they cross the street. The next move of the ego car would be to continue driving forward, following the road and traffic rules, and ensuring the pedestrian's safety as they cross the street.

Table 12: LLaVA-OneVision analyzes and interprets multi-camera video footage in self-driving contexts.

# S7: Composed Sub-video Understanding

![](images/474b3715706ff3ad6c2609a20aa7f77ff859311999b0bb0999e28afa13deb48b.jpg)

<details>
<summary>text_image</summary>

HOUSE OF THE DRAGON HOUSE OF THE DRAGON HOUSE OF THE DRAGON HOUSE OF THE DRAGON
HOUSE OF THE DRAGON HOUSE OF THE DRAGON HOUSE OF THE DRAGON HOUSE OF THE DRAGON
</details>

User   
Please describe the video's content and layout in detail.

LLaVA-OV

The video showcases a series of frames featuring a central, obscured foreground view of a room, presumably a game room or office setup. The room is adorned with several posters on the wall, a model of a humanoid figure wearing armor, and various other items that suggest a space dedicated to hobbies or gaming. In the foreground, there is a person, likely a woman, who is wearing headphones and has a neutral to slightly surprised facial expression, indicating that she may be reacting to something off-screen.

The backdrop throughout the video is a dynamic, digital illustration from the television show "Game of Thrones," specifically from the series "House of the Dragon." This series is a prequel to "Game of Thrones" and centers around the House Targaryen, a prominent family within the show's fictional world. The images in question feature characters from the series, including members of House Targaryen. Some characters are seen in battle attire, riding dragons, and engaged in combat scenarios. The scenes are detailed and vivid, depicting intense moments from the show, such as the White Walkers and dragon battles.

The video starts with the person in the foreground wearing headphones and a black shirt. Behind her, the consistent backdrop from "House of the Dragon" unfolds, focusing on characters from the series. It is evident that the series elements are meant to evoke a sense of drama and tension that is central to the narrative of "House of the Dragon."

As the video progresses, the scenes transition through various moments from the series. The characters are shown in different scenarios, such as riding dragons, engaging in combat, and experiencing intense moments. The visuals are rich with color and detail, highlighting the epic scale and dramatic tone of the series. The consistent presence of the game room setup in the foreground provides a sense of continuity throughout the video.

The video captures the essence of "House of the Dragon" by using dramatic and action-packed scenes from the series. The person in the foreground remains a constant, adding an element of personal reaction to the intense moments presented in the background.

In summary, the video juxtaposes a calm, possibly surprised individual in a game room setup with dynamic and intense scenes from "House of the Dragon," creating a contrast between the peaceful foreground and the dramatic background

Table 13: LLaVA-OneVision learns to understand and describe composed sub-videos in detail.

S8: Visual Prompting in Video   
![](images/da741a9e9cab04dfbd86d7cd6f94d150bcbd235b1d061f79c0b899388a96ff26.jpg)

<details>
<summary>text_image</summary>

Soccer match photo showing players on the field with visible sponsor banners and player names like 'ASHFORD'
</details>

![](images/07bdec79ad05eb94d68ca38de06982604e3c3d7d4c5c9bf0c723d599365b868b.jpg)

<details>
<summary>natural_image</summary>

Soccer match in progress on a green field, showing players in white and red uniforms on the court (no visible text or symbols)
</details>

![](images/b75c2813679007c13b91cdd9b93277e1bf977308a0c0918271d041bedd423530.jpg)

<details>
<summary>text_image</summary>

中国抢速新春快乐
中国抢速新春快乐
中国抢速新春快乐
H. RASHFORD
</details>

![](images/83cb6be19e124dbb9dc387d9b3ced2fd7f6323060c2e52f86a37b7a0c0ce769a.jpg)

<details>
<summary>text_image</summary>

缴纳足球俱乐部
中国抢送新春战
HARSHFORD
</details>

User   
Describe the player highlighted in the video.   
LLaVA-OV

The player highlighted in the video is wearing a white kit, and the number "10" is visible on their shirt. This player is positioned in the center of the pitch and is actively involved in the play, demonstrating control and dribbling skills as they navigate around the field. The player is surrounded by opponents in red kits, indicating that they are likely a key player for their team, possibly a playmaker or attacking midfielder. Throughout the video, the player maintains possession of the ball and creates opportunities for their team, showcasing their importance in the game's dynamics.

Table 14: LLaVA-OneVision learns to provide detailed descriptions of highlighted subjects in video content.

# S9: Referring Image in Video Understanding

![](images/af5f6423dc1c2be0988a59329b349078d5f7a89b11e94cfd2b0b89af19dc0d59.jpg)

![](images/6ec9761a1062c2a194af5ced59ecaed800b6e3605ae5e87a3331d5d95e5e5c6f.jpg)

User There are 2 images. Does the person in image 1 appear in image 2? If yes, what is he doing?

LLaVA-OV Yes, the person from image 1 appears in image 2. He is playing soccer with two other individuals in a residential outdoor area.

User Who is the man in image 1?

LLaVA-OV The man is Lionel Messi.

![](images/68fad8490143dacf22210817337399f0a675f621f0d1ec46e539e42a9e2d1108.jpg)

![](images/d5753b3f60b47da7135b94be74f7f5693fc33adeb45eb614dac1b514670ae790.jpg)

<details>
<summary>natural_image</summary>

Sequence of five photos showing people playing with green balls on a grassy field, palm trees in background (no text or symbols)
</details>

User Here is an image and a video. Does the man shown in the image also appear in the video? If yes, describe his behavior in detail.

LLaVA-OV Yes, the man shown in the image appears in the video. He is seen engaging in a soccer activity in a grassy outdoor area. He is dressed in a white shirt, dark pants, and white shoes. Throughout the video, he is seen standing near a yellow and black soccer ball, and at various points, he is either preparing to kick the ball or has just kicked it. His actions suggest he is actively participating in the game, and he appears to be in motion, indicating movement and activity.

![](images/c0a2b8474a5156c0fc500b7cbb2c4b210afde8f9991107ee6eccf6ea3c10e5cb.jpg)

![](images/f0a609f9ffe467ac3aec0bc63ada19046aa5ff857a6f0ec72686058c15eda8dd.jpg)

<details>
<summary>natural_image</summary>

Sequence of five photos showing people playing on a grassy field with palm trees, no text or symbols visible
</details>

User Here is a video and an image. Does the man shown in the image also appear in the video? Explain it.

LLaVA-OV The man shown in the image does not appear in the video. The image shows a person in a red sports jersey with the number 7, while the video features individuals playing with a soccer ball in a grassy outdoor area. The clothing, background, and activity are different between the image and the video, indicating that they are separate and not related.

User Who is the man in the image?

LLaVA-OV The man is Cristiano Ronaldo.

Table 15: LLaVA-OneVision's capability in referring image and video understanding. It accurately identifies the same individual in two images in the first instance. It identifies the same individual in both the image and the video in the second instance and correctly concludes the absence of the individual in the third instance, indicating its understanding capability to relate visual query in both image and video understanding.

# References

[1] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In AAAI, 2019. 39   
[2] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms, 2019. 39   
[3] Anthropic. Claude-3.5. https://www.anthropic.com/news/claude-3-5-sonnet, 2024. 2, 11   
[4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In ICCV, 2015. 39   
[5] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129–19139, 2022. 9   
[6] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 40   
[7] Haoping Bai, Shancong Mou, Tatiana Likhomanenko, Ramazan Gokberk Cinbis, Oncel Tuzel, Ping Huang, Jiulong Shan, Jianjun Shi, and Meng Cao. Vision datasets: A benchmark for vision-based industrial inspection, 2023. 40   
[8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. Technical Report, 2023. 11, 37   
[9] Ankan Bansal, Yuting Zhang, and Rama Chellappa. Visual question answering on image sets. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16, pages 51–67. Springer, 2020. 9   
[10] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 2   
[11] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In ICCV, 2019. 39   
[12] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving, 2020. 40   
[13] Jimmy Carter. Textocr-gpt4v. https://huggingface.co/datasets/jimmycarter/textocr-gpt4v, 2024. 39   
[14] Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps, 2022. 39   
[15] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. arXiv preprint arXiv:2109.00590, 2021. 40   
[16] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 6, 7, 39   
[17] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression, 2022. 39

[18] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning, 2022. 39   
[19] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 10   
[20] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 5   
[21] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 38, 40   
[22] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 9, 11, 37, 39   
[23] Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and Dongmei Zhang. Hitab: A hierarchical table dataset for question answering and natural language generation. In ACL, 2022. 39   
[24] Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024. 9   
[25] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 40   
[26] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2024. 2   
[27] Maxwell Forbes, Christine Kaeser-Chen, Piyush Sharma, and Serge Belongie. Neural naturalist: Generating fine-grained image comparisons, 2019. 40   
[28] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 10, 36, 38   
[29] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 10, 11   
[30] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data, 2023. 40   
[31] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 9, 10   
[32] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model, 2023. 39

[33] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abraham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4d: Around the world in 3,000 hours of egocentric video, 2022. 38, 40   
[34] Ziyu Guo, Renrui Zhang, Hao Chen, Jialin Gao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Sciverse. https://sciverse-cuhk.github.io, 2024. 9   
[35] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023. 2   
[36] Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos, 2018. 40   
[37] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, 2018. 39, 40   
[38] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023. 9   
[39] Mehrdad Hosseinzadeh and Yang Wang. Image change captioning by learning from an auxiliary task. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2724–2733, 2021. 40   
[40] Ting-Hao K. Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, Aishwarya Agrawal, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016), 2016. 9   
[41] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 39   
[42] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing, 2024. 40   
[43] Phillip Isola, Joseph J. Lim, and Edward H. Adelson. Discovering states and transformations in image collections. In CVPR, 2015. 40   
[44] Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan Boyd-Graber, Hal Daumé III au2, and Larry Davis. The amazing mysteries of the gutter: Drawing inferences between panels in comic book narratives, 2017. 40   
[45] Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images. arXiv preprint arXiv:1808.10584, 2018. 9   
[46] Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images, 2018. 40

[47] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 2, 10, 12, 40   
[48] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017. 39   
[49] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In CVPR, 2018. 37, 39   
[50] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning, 2018. 39   
[51] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. Technical Report, 2024. 2   
[52] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023. 39   
[53] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016. 10, 37, 39   
[54] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016. 9, 36, 38   
[55] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4999–5007, 2017. 39   
[56] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5376–5384, 2017. 40   
[57] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. In NeurIPS, 2020. 39   
[58] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022. 37, 39   
[59] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li. Visual genome: Connecting language and vision using crowdsourced dense image annotations, 2016. 39   
[60] Benno Krojer, Vaibhav Adlakha, Vibhav Vineet, Yash Goyal, Edoardo Ponti, and Siva Reddy. Image retrieval from contextual descriptions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, Online, May 2022. Association for Computational Linguistics. 40   
[61] Shanghai AI Laboratory. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2023. 38

[62] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018. 39   
[63] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? Technical Report, 2024. 2, 6, 37   
[64] Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. Llava-next: What else influences visual instruction tuning beyond data?, May 2024. 1, 2, 3, 5, 34, 35   
[65] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, May 2024. 1, 3, 9, 10, 34, 36, 38   
[66] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. 10   
[67] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 2024. 1   
[68] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next: Tackling multi-image, video, and 3d in large multimodal models, June 2024. 1, 2, 5, 6, 7, 9, 10, 12, 34, 35, 36, 38   
[69] Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Fine-tuning multimodal llms to follow zero-shot demonstrative instructions, 2024. 7, 40   
[70] Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Hanwang Zhang, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, and Yueting Zhuang. Empowering vision-language models to follow interleaved vision-language instructions. arXiv preprint arXiv:2308.04152, 2023. 2, 12   
[71] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark, 2023. 10   
[72] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, 2024. 2   
[73] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. Technical Report, 2024. 2   
[74] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization, 2019. 40   
[75] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning, 2023. 39   
[76] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 2   
[77] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 2, 11, 12   
[78] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015. 37, 39

[79] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 2023. 39   
[80] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 39   
[81] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 1, 3, 6, 37   
[82] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. 1, 4, 5, 12, 34, 37   
[83] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1, 2, 3, 6, 39   
[84] Xuejing Liu, Wei Tang, Xinzhe Ni, Jinghui Lu, Rui Zhao, Zechao Li, and Fei Tan. What large language models bring to text-rich vqa?, 2023. 10   
[85] Xuejing Liu, Wei Tang, Xinzhe Ni, Jinghui Lu, Rui Zhao, Zechao Li, and Fei Tan. What large language models bring to text-rich vqa? arXiv preprint arXiv:2311.07306, 2023. 9   
[86] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, and Ziwei Liu. Mmbench: Is your multi-modal model an all-around player? Technical Report, 2023. 9, 10, 36   
[87] LMMs-Lab. Video detail caption, 2024. 11   
[88] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pages 22631–22648. PMLR, 2023. 2   
[89] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, and Yaofeng Sun. Deepseek-vl: towards real-world vision-language understanding. Technical Report, 2024. 37   
[90] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. arXiv preprint arXiv:2310.02255, 2023. 9, 10, 36, 38   
[91] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning, 2021. 39   
[92] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In ACL, 2021. 39   
[93] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022. 10, 39   
[94] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In International Conference on Learning Representations (ICLR), 2023. 39   
[95] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. In NeurIPS, 2021. 39, 40

[96] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 11   
[97] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 10   
[98] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36, 2024. 10, 11   
[99] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019. 39   
[100] U-V Marti and Horst Bunke. The iam-database: an english sentence database for offline handwriting recognition. International journal on document analysis and recognition, 5:39–46, 2002. 39   
[101] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022. 9, 10, 36, 37, 39   
[102] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 9, 10, 36, 39   
[103] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021. 9, 10, 36, 37, 39, 40   
[104] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024. 2   
[105] A. Mishra, K. Alahari, and C. V. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012. 39   
[106] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019. 39   
[107] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 947–952, 2019. 40   
[108] Jason Obeid and Enamul Hoque. Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model, 2020. 39   
[109] OpenAI. Gpt-4v. https://openai.com/index/gpt-4v-system-card/, 2023. 2, 9, 11, 12   
[110] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. 2, 9, 11, 12   
[111] Piotr Padlewski, Max Bain, Matthew Henderson, Zhongkai Zhu, Nishant Relan, Hai Pham, Donovan Ong, Kaloyan Aleksiev, Aitor Ormazabal, Samuel Phua, et al. Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models. arXiv preprint arXiv:2405.02287, 2024. 9   
[112] Piotr Padlewski, Max Bain, Matthew Henderson, Zhongkai Zhu, Nishant Relan, Hai Pham, Donovan Ong, Kaloyan Aleksiev, Aitor Ormazabal, Samuel Phua, Ethan Yeo, Eugenie Lamprecht, Qi Liu, Yuqi Wang, Eric Chen, Deyu Fu, Lei Li, Che Zheng, Cyprien de Masson d'Autume, Dani Yogatama, Mikel Artetxe, and Yi Tay. Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models, 2024. 10, 36, 38

[113] Dong Huk Park, Trevor Darrell, and Anna Rohrbach. Robust change captioning, 2019. 40   
[114] Renjie Pi, Jianshu Zhang, Jipeng Zhang, Rui Pan, Zhekai Chen, and Tong Zhang. Image textualization: An automatic framework for creating accurate and detailed image descriptions, 2024. 39   
[115] Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. 10, 11   
[116] Yuxuan Qiao, Haodong Duan, Xinyu Fang, Junming Yang, Lin Chen, Songyang Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Prism: A framework for decoupling and assessing the capabilities of vlms, 2024. 12   
[117] Harsh Raj, Janhavi Dadhania, Akhilesh Bhardwaj, and Prabuchandran KJ. Multi-image visual question answering. arXiv preprint arXiv:2112.13706, 2021. 9   
[118] Hareesh Ravi, Kushal Kafle, Scott Cohen, Jonathan Brandt, and Mubbasir Kapadia. Aesop: Abstract encoding of stories, objects, and pictures. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2032–2043, 2021. 40   
[119] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, 2022. 39   
[120] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015. 39   
[121] ShareGPT. https://sharegpt.com/, 2023. 37, 39   
[122] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020. 9, 40   
[123] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension, 2020. 39   
[124] Gunnar A. Sigurdsson, Gül Varol, Xiaolong Wang, Ivan Laptev, Ali Farhadi, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. ArXiv e-prints, 2016. 38, 40   
[125] Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. A corpus of natural language for visual reasoning. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 217–223, 2017. 9   
[126] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs, 2019. 40   
[127] Hao Tan, Franck Dernoncourt, Zhe Lin, Trung Bui, and Mohit Bansal. Expressing visual relationships via language, 2019. 40   
[128] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In AAAI, 2021. 39   
[129] Benny J. Tang, Angie Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning, 2023. 39   
[130] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 11

[131] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 12   
[132] Ting-Hao, Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, Lucy Vanderwende, Michel Galley, and Margaret Mitchell. Visual storytelling, 2016. 40   
[133] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 2, 6, 9, 11, 39   
[134] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2words: Automatic mobile ui summarization with multimodal learning, 2021. 36, 39   
[135] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 9, 10   
[136] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021. 2   
[137] Chris Wendler. wendlerc/renderedtext, 2023. 39   
[138] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. 10   
[139] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023. 9   
[140] Haoning Wu, Hanwei Zhu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Annan Wang, Wenxiu Sun, Qiong Yan, Xiaohong Liu, Guangtao Zhai, Shiqi Wang, and Weisi Lin. Towards open-ended visual quality comparison, 2024. 40   
[141] x.ai. Grok-1.5 vision preview. 9, 10   
[142] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, June 2021. 11, 38, 40   
[143] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 12   
[144] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. ArXiv, abs/2406.08464, 2024. 36, 37, 39   
[145] Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning. arXiv preprint arXiv:2402.11690, 2024. 2   
[146] Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning, 2024. 37, 39   
[147] Semih Yagcioglu, Aykut Erdem, Erkut Erdem, and Nazli Ikizler-Cinbis. Recipeqa: A challenge dataset for multimodal comprehension of cooking recipes, 2018. 40

[148] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 3, 35   
[149] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 13   
[150] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Alex Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model, 2023. 37, 39   
[151] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023. 9   
[152] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions, 2016. 39   
[153] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2023. 10   
[154] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 9   
[155] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9127–9134, 2019. 10, 11, 38, 40   
[156] Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntax-aware network for handwritten mathematical expression recognition. arXiv preprint arXiv:2203.01601, 2022. 39   
[157] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, and Yuxuan Sun. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 9, 10, 36, 38   
[158] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 3, 35   
[159] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In CVPR, 2019. 39, 40   
[160] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing, 2024. 40   
[161] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024. 8, 9, 10, 36   
[162] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 2, 11, 12   
[163] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 12

[164] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023. 2   
[165] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024. 9, 10   
[166] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, Peng Gao, and Hongsheng Li. Mavis: Mathematical visual instruction tuning, 2024. 39   
[167] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 38   
[168] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 39   
[169] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024. 1, 5, 6, 12, 34, 35, 36   
[170] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 10, 11   
[171] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding, 2024. 36   
[172] Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos, 2017. 38, 40   
[173] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2   
[174] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In CVPR, 2016. 39

# A Development Roadmap from LLaVA-NeXT to LLaVA-OneVision

LLaVA-OneVision is built upon techniques developed in the LLaVA-NeXT blog series $[82, 169, 65, 64, 68]$ from January to June 2024. The initial LLaVA-NeXT provided an extendable and scalable prototype, which facilitated several parallel explorations. These explorations, conducted within a fixed compute budget, aimed to offer useful insights along the way, rather than push performance limits. LLaVA-OneVision consolidates these insights and execute with “yolo run” – implements the new model with the available compute, without extensively de-risking individual components.

![](images/c6e3fb85abf83a38603f1b61cd95502ddeb1f89cb0e9bc6009c3787dabce0887.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Jan\nLLaVA-NeXT [82"]] --> B["April\nLLaVA-NeXT (Video) [169"]]
    A --> C["May\nLLaVA-NeXT (Stronger) [65"]]
    A --> D["May\nLLaVA-NeXT (Ablations) [64"]]
    A --> E["June\nLLaVA-NeXT (Interleave) [68"]]
    F["July\nLLaVA-OneVision"] --> B
    F --> C
    F --> D
    F --> E
```
</details>

Figure 6: The development timeline from LLaVA-NeXT to LLaVA-OneVision.

# 1. LLaVA-NeXT:

Improved reasoning, OCR, and world knowledge [82]

- Blog: https://llava-vl.github.io/blog/2024-01-30-llava-next/   
- A cost-efficient training recipe for LMMs with strong performance

# 2. LLaVA-NeXT (Video):

A Strong Zero-shot Video Understanding Model [169]

- Blog: https://llava-vl.github.io/blog/2024-04-30-llava-next-video/   
- Thanks to the design of AnyRes to digest vision signal, the image-only-trained LLaVA-NeXT model is surprisingly strong on video tasks with zero-shot modality transfer. DPO training with AI feedback on videos can further yield significant improvement.

# 3. LLaVA-NeXT (Stronger):

Stronger LLMs Supercharge Multimodal Capabilities in the Wild [65]

- Blog: https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/   
- The same cost-efficient recipe, supporting LLaMA3 (8B) and Qwen (72B &110B). Simply scaling up LLM catches up with GPT-4V on selected benchmarks. Developed an evaluation benchmark for daily-life visual chat, LLaVA-Bench (Wilder).

# 4. LLaVA-NeXT (Ablation):

What Else Influences Visual Instruction Tuning Beyond Data? [64]

- Blog: https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/   
- Ablating the choice of Architectures (Scaling LLM & Vision Encoder), Visual Representations (Resolution & #Tokens), and Training Strategies (Trainable modules & High-quality data).

# 5. LLaVA-NeXT (Interleave):

Tackling Multi-image, Video, 3D in Large Multimodal Models [68]

- Blog: https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/   
- Extending the capability to new scenarios including multi-image, multi-frame (video) and multi-view (3D), with new training data (M4-Instruct) and benchmark (LLaVA-Interleave Bench).

# B Author Contributions

- Bo Li contributes to maintaining the LLaVA-OneVision codebase, conducting the large-scale training of the LLaVA-OneVision models of all stages (including the stage with single-image, multi-image, and video data), based on upon our previous LLaVA-NeXT series. He contributes significantly to the single-image development such as LLaVA-NeXT-Ablations [64], high-quality recpationing, as well as collection and curation of the single-image data mixture.   
- Yuanhan Zhang contributes to a series of works in LLaVA-NeXT-Video [169], including video training and inference codebase, an effective pipeline for high-quality video data generation, and all the video training data.   
- Dong Guo contributes to collection and curation of the single-image data mixture and consistently provides technical support throughout the project.   
- Feng Li, Renrui Zhang, and Hao Zhang contribute to LLaVA-NeXT-Interleave [68], including the multi-image instruction data mixture, the multi-image evaluation benchmarks, and the early prototype of LLaVA-OneVision, i.e., a joint training stage with single-image, multi-image, and videos. They also contribute to the collection and curation of the single-image data mixture.   
- Kaichen Zhang maintains the training codebase and contributes to the integration of LLaVA-OneVision model into LMMs-Eval's evaluation pipeline.   
- Yanwei Li contributes to revising the paper.   
- Ziwei Liu makes valuable suggestions throughout the projects.   
- Chunyuan Li initiates and leads the series of projects, designs the roadmap and milestones, drives the excursion, as well as leads the paper writing.

# C Implmenetation Details

# C.1 Token Strategy for Mixed-Modality Data

We provide a detailed explanation of our token strategy for handling mixed-modality data within LLaVA-OneVision's architecture, which is illustrated in Figure 3.

For single-image data, we employ the AnyResMax-9 strategy, as previously outlined in blog $[64]$ . Using SO400M $[158]$ as the Vision Encoder, each input image (or grid) is processed into 729 visual tokens. Consequently, the maximum number of visual tokens for a single image is $729 \times (1 + 9)$ , where $1 \times 729$ represents the base tokens and $9 \times 729$ accounts for the grid tokens.

For multi-image data, we utilize a simple padding strategy. Each image is first resized to fit within a 384x384 frame by zero-padding, as required by SO400M, while maintaining the aspect ratio. After processing through the vision encoder, the zero-padding is removed from the tokens. Our training data includes up to 12 images per instance, resulting in a maximum of $12 \times 729$ multi-image tokens.

For video data, we adopt a strategy similar to LLaVA-NeXT-Video $[169]$ . Each frame is processed through the vision encoder and then subjected to $2 \times 2$ bilinear interpolation, resulting in 196 tokens per frame. We sample up to 32 frames per video, leading to a maximum of $32 \times 196$ video tokens.

As shown in Figure 3, the maximum number of tokens across different modalities is approximately equal. This design strategy aims to balance the data from various modalities, ensuring more equitable representation that is transferable from the perspective of the language model. For instance, a high-resolution image can be interpreted as a composition of multiple images, and multiple images can be understood as a shorter video.

# C.2 Language Templates and Special Tokens

We utilize the Qwen-2 series [148] language models with the template as OpenAI's ChatML $^{1}$ . During training, we adopt <image> as the marker for image tokens, following previous LLaVA models. This image special token is represented as -200 in the input index after tokenization. For multi-image

scenarios, we use multiple <image> interleaved with text to denote the positions of the images. For video scenarios, we place a single <image> at the beginning to indicate the inclusion of a video.

One more aspect related to the handling of image tokens is ensuring that there are no extra <image> in the data. For instance, in some code writing tasks, there could be <image>...</image> related to HTML code. To avoid potential misunderstandings, we manually removed around 10 such samples from the Magpie [144] and Screen2Words [134] datasets.

# D Evaluation Steers Development

# D.1 Post-Evaluation as a Development Tool

With the help of our comprehensive evaluation toolkit, LMMs-Eval $[161]$ , we conduct post-evaluations on a selected set of benchmarks after each training experiment concludes.

Our preference for selecting benchmarks is based on whether the targeted scenarios are sufficiently important and specific. These evaluations should not be too resource-intensive, meaning the benchmarks should not contain too many items, take too long to evaluate, or consume a large number of GPT-4V tokens (when using it as the judge model).

In our development, we evaluate on AI2D [54], ChartQA [101], DocVQA [103], and InfoVQA [102] to examine the model's fine-grained understanding of tables, charts, and diagrams, as well as MME [28] for formatting control, since it requires only Yes or No answers. We also include MMBench-Dev [86] and MMMU-Val [157] for multi-discipline evaluation. Quickly obtaining evaluation results on these benchmarks will guide our next steps in model development and data curation.

# D.2 Improving Model Performance on Key Scenarios

During our development process, we gradually recognized the significance of using static evaluation benchmarks as performance indicators. Our primary goal at this stage is not to overfit the model to certain datasets to achieve exceptionally high performance. Instead, we benchmark our models against GPT-4V's performance to set our target thresholds (e.g., initially 80%, gradually increasing to 95%-100%). Once the model meets the score requirements in static evaluations, it indicates that the model has sufficient capabilities in the selected scenarios. Furthermore, we cannot blindly pursue results on benchmarks, as even the test data for AI2D may have certain issues $^{2}$ .

Ultimately, our focus is on optimizing the model's visual chat and reasoning capabilities. In this stage, we monitored the model's performance on benchmarks such as MathVista [90], LLaVA-Wilder [65], MM-LiveBench [171], and Vibe-Eval [112]. These benchmarks require the model to engage in visual dialogue with challenging questions, and demand a diverse skill set with extensive world knowledge. This helps us create a model with strong generalization capabilities in real-world scenarios.

# D.3 Evaluation Task Information

In this section, we provide information on all the tasks used during the evaluation. Specifically, we use the default post\_prompt and pre\_prompt from the LMMs-Eval framework. These prompts are consistent with the evaluation of our previous LLaVA-NeXT [65, 169, 68]. The table below details the specific tasks used in LMMs-Eval and their corresponding task names.

# Tasks Information

# - Single-image:

\- ai2d, chartqa, docvqa\_val, infovqa\_val, mme, realworldqa, mathvista\_testmini, llava\_in\_the\_wild, mmvet, mmbench\_en\_dev, ocrbench, mmmu, llava\_wilder\_small, vibe\_eval, wildvision\_0617, live\_bench\_2406, mathverse\_testmini\_vision, seedbench, scienceqa\_img, mmstar, dc100\_en

# - Videos:

\- activitynetqa, videochatgpt, nextqa\_mc\_test, egoschema, video\_dc499, videmme, videomme\_w\_subtype, perceptiontest\_val\_mc, mlvu, mvbench

# - Multi-image:

\- llava\_interleave\_bench, muirbench

By referring to the task names listed here, the audience can directly retrieve the generation arguments and specific prompt information. For instance, the details for tasks=ai2d are available at lmms-eval/ai2d. By following these settings, researchers can easily reproduce our results.

# E Data Curation Roadmap of LLaVA-NeXT Series

In this section, we provide the in-depth experience and roadmap of data curation in the LLaVA-NeXT series. To achieve strong multimodal performance, we need to collect and curate high-quality data from various sources, which is crucial for the model's generalization capabilities.

# E.1 Single-Image Data Curation

As the primary data source, our principle for single-image data has always been that quality outweighs quantity. Given limited resources, we strive to use high-quality data to maximize the performance.

The first version of the LLaVA-NeXT models (LLaVA-NeXT-Vicuna-7B/13B, Mistral-7B, Hermes-Yi-34B), comprising 760K data samples [82], includes 665K samples from LLaVA-1.5 [81], 3,247 samples from AI2D [53], 18,317 samples from ChartQA [101], 10,194 samples from DocVQA [103], 20,000 samples from DVQA [49], 40,093 samples from SynthDOG-EN [58], and 15,131 samples from user requests on LLaVA's demo, re-annotated with GPT-4V. In the subsequent iteration, we added 20,000 samples from COCO Caption [78], forming a new 790K version. This 790K dataset supported the second release of LLaVA-NeXT models (LLaVA-NeXT-LLaMA3-8B, LLaVA-NeXT-Qwen-72B, LLaVA-NeXT-Qwen-110B).

In subsequent collections, we accumulated open-sourced datasets from the Internet and referred to the dataset collection processes of other advanced LMMs, such as Qwen-VL $[8]$ , DeepSeek-VL $[89]$ , Intern-VL $[22]$ , Vision-Flan $[146]$ , UReader $[150]$ , Idefics-2 (Cauldron) $[63]$ , and Cambrian. During the data iteration process, we strictly adhered to the initial LLaVA-1.5 strategy. For each dataset, we manually inspected and ensured its quality and QA format. We also designed specific formatting prompts to make data from different sources compatible with each other, thus avoiding conflicts.

Some data sources, such as AI2D and ChartQA, appear in different dataset collections and may be duplicated. Since Cauldron includes special formatting prompts, its data is not straightforward to re-format. Therefore, we prioritize using data from other collections that are closer to the raw format. For the Cambrian dataset, we only selected a subset of the GPT-4o re-annotated data. We also collected math-related data from the MathV and MAVIS datasets.

For the pure language data, we replaced the ShareGPT $[121]$ text data that LLaVA has been using since version 1.5. Given that our largest Qwen2-72B model has achieved performance levels close to latest GPT-4 model in language tasks, we need to use higher quality language data to maintain or further enhance its language capabilities. To achieve this, we sourced the highest quality language SFT data available, the Magpie-Pro dataset $[144]$ .

After undergoing the aforementioned process, we have obtained approximately 4 million raw SFT data samples, ensuring their quality and accuracy. Additionally, we utilized Azure's OpenAI GPT-4V and GPT-4o services to re-annotate our data, focusing on scenarios that were not adequately covered by the original data but are crucial. These scenarios include:

(1) Detailed Descriptions on Charts and Diagrams: For this scenario, we used images from the AI2D and InfoVQA training sets and employed GPT-4V to provide detailed descriptions of the images, resulting in 4,874 detailed descriptions for AI2D and 1,992 samples for InfoVQA.   
(2) Chinese Language: We used images from the LLaVA-158K dataset and employed GPT-4o to provide detailed descriptions in Chinese, resulting in a total of 91,466 samples.   
(3) Multi-turn Dialogue: Also with the LLaVA-158K dataset, we employed GPT-4o to create long dialogues with an average of more than 3 turns per conversation, obtaining a total of 26,048 samples.

When resources permit, we recommend a data validation process we used in early stage data sourcing. We extract approximately 100K samples from each newly added data source or collection (if the selected data source can form a collection) and add them to the 790K version of the dataset. We validate newly added data under the SO400M-Qwen-1.5-0.5B experimental setting. If the addition of new data results in a performance decline compared to the baseline, we conduct further manual inspections of the data and adjust the formatting prompt accordingly. This step requires abundant resources and must be carried out by highly professional researchers, as it cannot be substituted with average human annotators.

During the collection process, we manually labeled the datasets with two tags: {General, Language, Math/Reasoning, General OCR, Doc/Chart/Screen} and {Fixed-form, Free-form}. Based on these tags, we formed the final distribution of 3.2 million single-image data samples.

Starting with the initial distribution, we gradually increased the amount of free-form (most of them are GPT-4V/o annotated) data and observed the model's performance on various benchmarks and try to balance among them. These benchmarks include academic datasets, such as AI2D [54], MME [28], MMMU [157], MathVista [90], and visual chat datasets, such as LLaVA-Wilder [65], and VibeEval [112]. Ultimately, we gradually established an optimal data distribution for single-image tasks under the 7B setting.

# E.2 OneVision Data Curation

In addition to single-image data, we incorporate multi-image and video datasets to support a wider scope of visual scenarios. We aim to balance the capability among different data modalities, and achieve an overall superior performance with one framework as LLaVA-OneVision.

For multi-image data, we adopt the diverse interleaved multimodal tasks within M4-Instruct dataset from LLaVA-NeXT-Interleave $[68]$ . This dataset mainly comprises general multi-image tasks, such as spotting the difference, visual story telling, image editing instruction generation, interleaved multi-image dialogue, multi-image puzzle, low-level multi-image assessment, etc. Besides, we also utilize the multi-view datasets in M4-Instruct to indicate spatial information in the 3D world, including embodied VQA (dialogue and planning) and 3D scene VQA (captioning and grounding).

For video data, we first integrate the multi-frame data from M4-Instruct, including NExT-QA $[142]$ and ShareGPT4Video $[21]$ . Then, to enable more detailed temporal cues, we select several datasets commonly used in recent academic research for re-annotation, including Charades $[124]$ , ActivityNet $[155]$ , YouCook2 $[172]$ , and Ego4D $[33]$ . Initially, we annotated captions. Following ShareGPT4o $[61]$ , we sampled video frames at 1 frame per second (FPS) and used the pre-defined instructions to prompt GPT-4o for generating video captions. Additionally, following LLaVA-Hound $[167]$ , we developed open-ended question-answering pairs and their corresponding multiple-choice versions using the captions created by GPT-4o. We also employed GPT-4o to generate question-answer pairs, obtaining high-quality video data for OneVision training.

# E.3 Detailed Dataset Statistics

We primarily use tables to present the statistical information of all datasets utilized in both the Single-Image and OneVision stages. The information includes the dataset category, dataset name, number of samples, and prompt type. The dataset statistics are summarized in Table 16.

<table><tr><td>Dataset</td><td># Samples</td><td>Prompt ID</td><td>Dataset</td><td># Samples</td><td>Prompt ID</td></tr><tr><td colspan="6">General (1.14M, 36.1%)</td></tr><tr><td>AOKVQA [119]</td><td>66160</td><td>1</td><td>Cambrian (filtered) [133]</td><td>83131</td><td>-</td></tr><tr><td>CLEVR [48]</td><td>700</td><td>1</td><td>COCO Caption [78]</td><td>20000</td><td>9</td></tr><tr><td>Hateful Memes [57]</td><td>8500</td><td>1</td><td>IconQA [95]</td><td>2494</td><td>5</td></tr><tr><td>Image Textualization [114]</td><td>99583</td><td>11</td><td>LLaVA-158K [83]</td><td>158000</td><td>-</td></tr><tr><td>LLaVA-Wild (train) [83]</td><td>54517</td><td>-</td><td>LLaVAR [168]</td><td>20000</td><td>-</td></tr><tr><td>OKVQA [99]</td><td>8998</td><td>1</td><td>RefCOCO [152]</td><td>50586</td><td>7,8</td></tr><tr><td>ScienceQA [93]</td><td>4976</td><td>5</td><td>ShareGPT4O [121]</td><td>57289</td><td>11</td></tr><tr><td>ShareGPT4V [121]</td><td>92025</td><td>11</td><td>ST-VQA [11]</td><td>17247</td><td>1</td></tr><tr><td>TallyQA [1]</td><td>9868</td><td>1</td><td>Vision FLAN [146]</td><td>186070</td><td>-</td></tr><tr><td>Visual7W [174]</td><td>14366</td><td>5</td><td>VisText [129]</td><td>9969</td><td>15</td></tr><tr><td>VizWiz [37]</td><td>6614</td><td>2</td><td>VQARAD [62]</td><td>313</td><td>1</td></tr><tr><td>VQAv2 [4]</td><td>82783</td><td>1</td><td>VSR [79]</td><td>2157</td><td>3</td></tr><tr><td>WebSight</td><td>10000</td><td>18</td><td>InterGPS [91]</td><td>1280</td><td>5</td></tr><tr><td>ALLaVA Instruct [16]</td><td>70000</td><td>-</td><td></td><td></td><td></td></tr><tr><td colspan="6">Doc/Chart/Screen (20.6%, 647K)</td></tr><tr><td>AI2D (GPT4V Detailed Caption)</td><td>4874</td><td>12</td><td>AI2D (InternVL [22])</td><td>12413</td><td>4</td></tr><tr><td>AI2D (Original) [53]</td><td>3247</td><td>5</td><td>Chart2Text [108]</td><td>26961</td><td>13</td></tr><tr><td>ChartQA [101]</td><td>18317</td><td>1</td><td>Diagram Image2Text</td><td>300</td><td>17</td></tr><tr><td>DocVQA [103]</td><td>10194</td><td>1</td><td>DVQA [49]</td><td>20000</td><td>1</td></tr><tr><td>FigureQA [50]</td><td>1000</td><td>3</td><td>HiTab [23]</td><td>2500</td><td>1</td></tr><tr><td>Infographic VQA [102]</td><td>4404</td><td>1</td><td>LRV Chart [80]</td><td>1787</td><td>-</td></tr><tr><td>RoBUT SQA</td><td>8514</td><td>-</td><td>RoBUT WikiSQL</td><td>74989</td><td>-</td></tr><tr><td>RoBUT WTQ</td><td>38246</td><td>1</td><td>Screen2Words [134]</td><td>15730</td><td>10</td></tr><tr><td>TQA [55]</td><td>1365</td><td>5</td><td>UReader Caption [150]</td><td>91439</td><td>9</td></tr><tr><td>UReader IE [150]</td><td>17327</td><td>1</td><td>UReader KG [150]</td><td>37550</td><td>14</td></tr><tr><td>UReader QA [150]</td><td>252954</td><td>1</td><td>VisualMRC[128]</td><td>3027</td><td>-</td></tr><tr><td colspan="6">Math/Reasoning (20.1%,632K)</td></tr><tr><td>MAVIS Manual Collection [166]</td><td>87358</td><td>19</td><td>MAVIS Data Engine [166]</td><td>100000</td><td>19</td></tr><tr><td>CLEVR-Math [48]</td><td>5290</td><td>2</td><td>Geo170K Align [32]</td><td>60252</td><td>-</td></tr><tr><td>Geo170K QA [32]</td><td>67833</td><td>19</td><td>Geometry3K [91]</td><td>2101</td><td>6</td></tr><tr><td>GEOS [120]</td><td>508</td><td>6</td><td>Geometry3K (MathV360K) [92]</td><td>9734</td><td>6</td></tr><tr><td>GeoMVerse (MathV360K) [52]</td><td>9303</td><td>20</td><td>GeoQA+ (MathV360K) [18]</td><td>17172</td><td>6</td></tr><tr><td>MapQA (MathV360K) [14]</td><td>5235</td><td>1</td><td>MathQA [2]</td><td>29837</td><td>19</td></tr><tr><td>Super-CLEVR [75]</td><td>8652</td><td>2</td><td>TabMWP [94]</td><td>45184</td><td>2</td></tr><tr><td>UniGeo [17]</td><td>11959</td><td>6</td><td>GQA [41]</td><td>72140</td><td>1</td></tr><tr><td>LRV Normal [80]</td><td>10500</td><td>-</td><td>RAVEN [159]</td><td>2100</td><td>3</td></tr><tr><td>Visual Genome [59]</td><td>86417</td><td>7,8</td><td></td><td></td><td></td></tr><tr><td colspan="6">General OCR (8.9%,281K)</td></tr><tr><td>ChromeWriting [137]</td><td>8835</td><td>21</td><td>HME100K [156]</td><td>74502</td><td>21</td></tr><tr><td>IIIT5K [105]</td><td>2000</td><td>22</td><td>IAM [100]</td><td>5663</td><td>22</td></tr><tr><td>K12 Printing</td><td>12832</td><td>22</td><td>OCR-VQA [106]</td><td>80000</td><td>1</td></tr><tr><td>Rendered Text [137]</td><td>10000</td><td>22</td><td>SynthDog-EN [58]</td><td>40093</td><td>16</td></tr><tr><td>TextCaps [123]</td><td>21952</td><td>9</td><td>TextOCR-GPT4V [13]</td><td>25114</td><td>11</td></tr><tr><td colspan="6">Pure Language (450K) (14.3%, 647K)</td></tr><tr><td>Magpie Pro [144] (L3 MT)</td><td>149999</td><td>-</td><td>Magpie Pro (L3 ST)</td><td>150000</td><td>-</td></tr><tr><td>Magpie Pro (Qwen2 ST)</td><td>149996</td><td>-</td><td></td><td></td><td></td></tr><tr><td colspan="6">Multi-image Scenarios</td></tr><tr><td>Spot-the-Diff [46]</td><td>10.8K</td><td>20</td><td>Birds-to-Words [27]</td><td>14.3K</td><td>21</td></tr><tr><td>CLEVR-Change [113, 39]</td><td>3.9K</td><td>22</td><td>HQ-Edit-Diff [42]</td><td>7.0K</td><td>3</td></tr><tr><td>MagicBrush-Diff [160]</td><td>6.7K</td><td>4</td><td>IEdit [127]</td><td>3.5K</td><td>19</td></tr><tr><td>AESOP [118]</td><td>6.9K</td><td>23</td><td>FlintstonesSV [36]</td><td>22.3K</td><td>24</td></tr><tr><td>PororoSV [74]</td><td>12.3K</td><td>25</td><td>VIST [132]</td><td>26K</td><td>4</td></tr><tr><td>WebQA [15]</td><td>9.3K</td><td>8</td><td>TQA (MI) [56]</td><td>8.2K</td><td>9</td></tr><tr><td>OCR-VQA (MI) [107]</td><td>1.9K</td><td>17</td><td>DocVQA (MI) [103]</td><td>1.9K</td><td>18</td></tr><tr><td>RAVEN [159]</td><td>35K</td><td>5</td><td>MIT-StateCoherence [43]</td><td>1.9K</td><td>11</td></tr><tr><td>MIT-PropertyCoherence [43]</td><td>1.9K</td><td>12</td><td>RecipeQA ImageCoherence [147]</td><td>8.7K</td><td>14</td></tr><tr><td>VISION [7]</td><td>9.9K</td><td>13</td><td>Multi-VQA [69]</td><td>5K</td><td>-</td></tr><tr><td>IconQA [95]</td><td>34.6K</td><td>-</td><td>Co-Instruct [140]</td><td>50.0K</td><td>-</td></tr><tr><td>DreamSim [30]</td><td>15.9K</td><td>-</td><td>ImageCoDe [60]</td><td>16.6K</td><td>-</td></tr><tr><td>nuScenes [12]</td><td>9.8K</td><td>10</td><td>ScanQA [6]</td><td>25.6K</td><td>7</td></tr><tr><td>ALFRED [122]</td><td>22.6K</td><td>16</td><td>ContrastCaption [47]</td><td>25.2K</td><td>-</td></tr><tr><td>VizWiz (MI) [37]</td><td>4.9K</td><td>6</td><td>ScanNet [25]</td><td>49.9K</td><td>7</td></tr><tr><td>COMICS Dialogue [44]</td><td>5.9K</td><td>15</td><td>NLVR2 [126]</td><td>86K</td><td>26</td></tr><tr><td colspan="6">Multi-frame (Video) Scenarios</td></tr><tr><td>NExT-QA [142]</td><td>9.5K</td><td>2</td><td>ActivityNet [155]</td><td>6.5k</td><td>1</td></tr><tr><td>Ego-4D [33]</td><td>0.8K</td><td>2</td><td>Charades [124]</td><td>23.6K</td><td>1</td></tr><tr><td>YouCook2 [172]</td><td>41.9K</td><td>2</td><td>ShareGPT4Video [21]</td><td>255K</td><td>-</td></tr></table>

Table 16: The detailed statistics of Single-Image datasets used in LLaVA-OneVision. Prompt ID denotes the ID of Formatting Prompt which is corresponding to the ID in Table 18. - denotes no fromatting prompt is used.

Table 17: The detailed statistics of Multi-Image and Video datasets used in LLaVA-OneVision. Prompt ID denotes the ID of Formatting Prompt corresponding to the ID in Table 19. - denotes no fromatting prompt is used. "MI" means it is the multi-image version dataset from DEMON [69].

<table><tr><td>ID</td><td>Type</td><td>Postion</td><td>Prompt</td></tr><tr><td>1</td><td>VQA</td><td>Tail</td><td>Answer the question with a single word (or phrase).</td></tr><tr><td>2</td><td>VQA</td><td>Head</td><td>Hint: Please answer the question and provide the final answer at the end.</td></tr><tr><td>3</td><td>VQA (Yes/No)</td><td>Tail</td><td>Answer the question with Yes or No./Yes or No?/...</td></tr><tr><td>4</td><td>Choice</td><td>Tail</td><td>Answer with the given letter directly</td></tr><tr><td>5</td><td>Choice (Option Letter)</td><td>Tail</td><td>Answer with the option letter from the given choices directly. / Please respond with only the letter of the correct answer.</td></tr><tr><td>6</td><td>Choice (Option Letter)</td><td>Head</td><td>Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.</td></tr><tr><td>7</td><td>Region Caption</td><td>All</td><td>Provide a short description for this region.</td></tr><tr><td>8</td><td>Grounding</td><td>All</td><td>Provide the bounding box coordinate of the region this sentence describes.</td></tr><tr><td>9</td><td>Breif Caption</td><td>All</td><td>Provide a one-sentence caption for the provided image./Create a compact narrative representing the image presented./...</td></tr><tr><td>10</td><td>Screen Summarization</td><td>All</td><td>Summarize the main components in this picture./Provide a detailed account of this screenshot./...</td></tr><tr><td>11</td><td>Detailed Caption</td><td>All</td><td>Describe this image in detail./Explain the visual content of the image in great detail./...</td></tr><tr><td>12</td><td>Science Books</td><td>All</td><td>Here is a diagram figure extracted from some Grade 1 - 6 science books.\nPlease first describe the content of this figure in detail, including how the knowledge visually displayed in the diagram.\nThen start with a section title \"related knowledge:\", briefly and concisely highlight the related domain knowledge and theories that underly this diagram. Note that you do not need to provide much detail. Simply cover the most important concepts.</td></tr><tr><td>13</td><td>Information Extraction</td><td>Head</td><td>Provide the requested information directly.</td></tr><tr><td>14</td><td>Graph Sumarization</td><td>All</td><td>Please clarify the meaning conveyed by this graph./Explain what this graph is communicating./...</td></tr><tr><td>15</td><td>Photo Sumarization</td><td>All</td><td>Highlight a few significant elements in this photo./Mention a couple of crucial points in this snapshot./...</td></tr><tr><td>16</td><td>Chart Sumarization</td><td>All</td><td>What insights can be drawn from this chart?/Explain the trends shown in this chart./...</td></tr><tr><td>17</td><td>OCR</td><td>Head</td><td>OCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. If a word is split due to a line break in the image, use a space instead</td></tr><tr><td>18</td><td>Diagram Linkage</td><td>All</td><td>Dissect the diagram, highlighting the interaction between elements./Interpret the system depicted in the diagram, detailing component functions./...</td></tr><tr><td>19</td><td>Code Generation</td><td>All</td><td>Compose the HTML code to achieve the same design as this screen-shot.</td></tr><tr><td>20</td><td>Choice (with Reasoning)</td><td>Head</td><td>First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.</td></tr><tr><td>21</td><td>Math Computing</td><td>Tail</td><td>Round computations to 2 decimal places.</td></tr><tr><td>22</td><td>LaTeX OCR</td><td>All</td><td>Please write out the expression of the formula in the image using LaTeX format.</td></tr><tr><td>23</td><td>Text Reading</td><td>All</td><td>What is written in the image? Answer this question using the text in the image directly./Read and list the text in this image.</td></tr><tr><td>24</td><td>Choice (Full Option)</td><td>Tail</td><td>Please provide your answer by stating the letter followed by the full option.</td></tr><tr><td colspan="4">Video</td></tr><tr><td>1</td><td>Choice (Option Letter)</td><td>Tail</td><td>Answer with the option letter from the given choices directly. / Please respond with only the letter of the correct answer.</td></tr><tr><td>2</td><td>Choice (Full Option)</td><td>Tail</td><td>Please provide your answer by stating the letter followed by the full option.</td></tr><tr><td colspan="4">Multi-Image</td></tr><tr><td>3</td><td>Open-Ended</td><td>Head</td><td>What's the difference between 2 images?</td></tr><tr><td>4</td><td>Open-Ended</td><td>Head</td><td>Given the stories paired with the first several images, can you finish the story based on the last image?/With the narratives paired with the initial images, how would you conclude the story using the last picture?/...</td></tr><tr><td>5</td><td>Multi-Choice</td><td>Head</td><td>Here is a Raven's Progressive Matrice in a three-by-three form. You are provided with the first eight elements in eight images, please select the last one from four choices following the structural and analogical relations.</td></tr><tr><td>6</td><td>Multi-Choice</td><td>All</td><td>There are ten possible explanations for the ten different answers to a VQA: ... I will give you two sets of pictures, questions, and answers to determine if they belong to the same 'Question-Answer Differences'. You must choose your answer from the Choice List.</td></tr><tr><td>7</td><td>Open-Ended</td><td>Head</td><td>This is a 3D scenario.</td></tr><tr><td>8</td><td>Open-Ended</td><td>Head</td><td>I will give you several images and a question, your job is to seek information in the slide and answer the question correctly./Based on the images, please answer the following question./...</td></tr><tr><td>9</td><td>Multi-Choice</td><td>Head</td><td>Provided with a series of diagrams from a textbook, your responsibility is to correctly answer the following question. You must choose your answer from the Choice List./Using a selection of textbook diagrams, your task is to provide an accurate response to the subsequent query. You must choose your answer from the Choice List./...</td></tr><tr><td>10</td><td>Open-Ended</td><td>Head</td><td>Given six images taken from different cameras on a street view car, your task is to answer questions about the depicted scene. You must choose your answer from the Choice List. /Upon receiving six photographs captured from various cameras on a street-view car, your responsibility is to provide accurate responses to questions about the scene. You must choose your answer from the Choice List. /...</td></tr><tr><td>11</td><td>Multi-Choice</td><td>Head</td><td>I will provide you with two sets of pictures, each of which shows an object in the opposite state. Can you tell me if the states of these two sets of pictures are the same? You must choose your answer from the Choice List. /I have two sets of pictures that show an object in opposite states. Can you tell me if the states of these two sets of pictures are the same? You must choose your answer from the Choice List. /...</td></tr><tr><td>12</td><td>Multi-Choice</td><td>Head</td><td>Are the following four images of the same class? You must choose your answer from the Choice List. /Do the following four images belong to the same category? You must choose your answer from the Choice List. /...</td></tr><tr><td>13</td><td>Multi-Choice</td><td>Head</td><td>Are these two workpieces the same type?/Are these two workpieces of the same kind?/...</td></tr><tr><td>14</td><td>Multi-Choice</td><td>Head</td><td>Presented with a textual recipe tutorial, your task is to scrutinize it carefully and select the image that is incoherent in the provided sequence of images. You must choose your answer from the Choice List. /Given a text-based recipe guide, your responsibility is to meticulously review it and identify the image that doesn't fit in the following sequence of images. You must choose your answer from the Choice List. /...</td></tr><tr><td>15</td><td>Multi-Choice</td><td>Head</td><td>I will give you a series of comic panels. The dialogue box of the last panel is masked. Can you choose the most relevant one from the candidates? You must choose your answer from the Choice List. /Given previous full panels and one masked panel, your job is to select the most appropriate dialogue among four candidates. You must choose your answer from the Choice List. /...</td></tr><tr><td>16</td><td>Open-Ended</td><td>Head</td><td>Give you a main goal, your job is to figure out what to do now by looking at current envirments. Your past views as well as decisions are also provided./Given a primary objective and your current surroundings, use your previous decisions and perspectives to determine your next move./...</td></tr><tr><td>17</td><td>Multi-Choice</td><td>Head</td><td>I will give you two pictures of the book cover. Please look at the pictures and answer a question You must choose your answer from the Choice List. /I will provide you with two images of the book cover. Please examine the images and answer a question. You must choose your answer from the Choice List. /...</td></tr><tr><td>18</td><td>Multi-Choice</td><td>Head</td><td>I will give you some pictures, and each group of pictures will correspond to a question. Please answer it briefly. You must choose your answer from the Choice List. /For each group of pictures, there is a question. Please give a short answer to it. You must choose your answer from the Choice List. /...</td></tr><tr><td>19</td><td>Open-Ended</td><td>Head</td><td>Please give a editing Request to describe the transformation from the source image to the target image./What is the correct image edit instruction that can transfrom the source image to target image?/...</td></tr><tr><td>20</td><td>Open-Ended</td><td>Head</td><td>What's the difference between 2 images? /Identify the alterations between these two images. /...</td></tr><tr><td>21</td><td>Open-Ended</td><td>Head</td><td>What's the difference between 2 birds? /Identify the alterations between these two birds. /...</td></tr><tr><td>22</td><td>Open-Ended</td><td>Head</td><td>What's the difference between 2 images? /Identify the alterations between these two images. /...</td></tr><tr><td>23</td><td>Open-Ended</td><td>Head</td><td>Given the stories paired with the first several images, can you finish the story based on the last image?/With the narratives paired with the initial images, how would you conclude the story using the last picture?/...</td></tr><tr><td>24</td><td>Open-Ended</td><td>Head</td><td>Given the stories paired with the first several images, can you finish the story based on the last image?/With the narratives paired with the initial images, how would you conclude the story using the last picture?/...</td></tr><tr><td>25</td><td>Open-Ended</td><td>Head</td><td>Given the stories paired with the first several images, can you finish the story based on the last image?/With the narratives paired with the initial images, how would you conclude the story using the last picture?/...</td></tr><tr><td>26</td><td>Multi-Choice</td><td>All</td><td>Answer the following multiple-choice question: Here is a statement describing 2 images: ... Is it true or false?</td></tr></table>

Table 18: The information of formatting prompts for Single-Image data. The "Position" means the position of the formatting prompt in the prompt where "All" means the formatting prompt is the prompt. Sometimes, there are multiple prompts of the same meaning. In this case, the prompt column is formatted as "Prompt1/Prompt2/...".

Table 19: The information of formatting prompts for One-Vision data. The “Position" means the position of the formatting prompt in the prompt where “All" means the formatting prompt is the prompt. Sometimes, there are multiple prompts of the same meaning. In this case, the prompt column is formatted as “Prompt1/Prompt2/...”.

# E.4 Policy Information and Reproducibility

We will open-source most of the public datasets we used. These images and data are already publicly available for academic research; we incorporated them and converted the format for our use. However, a small portion of our data sources related to user data and those obtained using the Azure OpenAI Service cannot be directly released due to company policy. We will provide the exact data YAML files used in the final reproduction scripts and will offer reproducible experimental scripts, training logs, and final version checkpoints using fully public data as our compute resources allow.