# Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning

Hao Shao $^{1,2}$ Zhuofan Zong $^{1}$

Shengju Qian $^{1}$ Letian Wang $^{3}$

Han Xiao $^{1}$ Yu Liu $^{2✉}$

Guanglu Song $^{2}$ Hongsheng Li $^{1,4,5✉}$

$^{1}$ The Chinese University of Hong Kong $^{2}$ SenseTime Research $^{3}$ University of Toronto $^{4}$ Shanghai Artificial Intelligence Laboratory $^{5}$ CPII under InnoHK

# Abstract

Multi-Modal Large Language Models (MLLMs) have demonstrated impressive performance in various VQA tasks. However, they often lack interpretability and struggle with complex visual inputs, especially when the resolution of the input image is high or when the interested region that could provide key information for answering the question is small. To address these challenges, we collect and introduce the large-scale Visual CoT dataset comprising 438k question-answer pairs, annotated with intermediate bounding boxes highlighting key regions essential for answering the questions. Additionally, about 98k pairs of them are annotated with detailed reasoning steps. Importantly, we propose a multi-turn processing pipeline that dynamically focuses on visual inputs and provides interpretable thoughts. We also introduce the related benchmark to evaluate the MLLMs in scenarios requiring specific local region identification. Extensive experiments demonstrate the effectiveness of our framework and shed light on better inference strategies. The Visual CoT dataset, benchmark, and pre-trained models are available on this webpage to support further research in this area.

# 1 Introduction

With the success of large language models (LLMs) like GPT-4 $[1]$ and Gemini $[63]$ , researchers are enhancing these models by incorporating visual understanding capabilities. This enthusiasm has led to the emergence of multi-modal large language models (MLLM), such as LLaVA $[39, 40]$ , SPHINX $[17, 37]$ , and Qwen-VL $[3]$ . Involving the extraction of visual tokens from input images, these MLLMs mostly follow a two-stage schedule: first the alignment of these tokens with linguistic modalities, and then the joint processing in LLMs. MLLMs have demonstrated viability in various scenarios, such as image captioning, visual question answering, and optical character recognition, owing to their ability to generate plausible outputs and leverage the extensive knowledge of LLMs.

However, many popular MLLMs $[47, 58, 23, 85, 7, 9, 76, 75, 83]$ and related benchmarks $[35, 8, 22, 73, 74]$ are primarily trained to respond to instructions based on visual inputs, employing a decoder-only autoregressive design as a single black box. While these models exhibit impressive generation capabilities, they suffer from inaccurate information $[36]$ and even hallucinations $[18]$ . Moreover, the black-box design hinders the interpretability of visual-language models. Additionally, the potential of multi-turn in-context capability and the advantages of chain-of-thought $[70, 89, 81]$ for LLMs have not been extensively explored in MLLMs. Some recent works, such as multimodal-CoT $[90]$

and [80, 79], have shown improvements by incorporating text-level chain-of-thought reasoning or in-context learning. However, it remains uncharted whether existing MLLMs can benefit from chain-of-thought reasoning in the visual understanding process, along with their interpretability remains largely unexplored.

Furthermore, humans comprehend intricate visual information differently, often by focusing on specific image regions or details within a given sample. For instance, when asked for a detailed regional description, humans tend to scan the entire image first, locate the references, and then focus on the targets. In contrast, most MLLMs process aligned image contexts in a fixed-grain manner with a large amount of computation (e.g., CLIP [57], EVA2-CLIP [62], InternVL [12]). To mimic human-like efficient reasoning behaviors, models need to identify image regions containing essential visual details and dynamically zoom in to capture adjusted context, which current MLLMs struggle with, leading them to seek information primarily from the text domain.

Therefore, there is a pressing need to develop methods that can handle multi-turn, dynamic focused visual inputs, while providing more interpretable stages of reasoning to enhance the efficacy and applicability of MLLMs. However, two significant challenges hinder the design of such pipelines: the lack of intermediate visual chain-of-thought supervision in existing visual question-answering (VQA) datasets, and the reliance of popular MLLM pipelines on static image context inputs.

To address these challenges, we develop and release a 438k visual chain-of-thought dataset by annotating each visual question-answer pair with a bounding box. The bounding box highlights the key image region essential for answering the question. We suppose that accurately locating and comprehending this key region will significantly improve MLLM's response accuracy and relevance. Notably, about 98k question-answer pairs include extra detailed reasoning steps. These annotations are designed to instruct the MLLM in a logical, step-by-step process to identify the final bbox and generate the answer. Building on the dataset, we propose a novel pipeline that unleashes the visual CoT reasoning capability of MLLMs, which is designed to identify and output key regions in an image that provides detailed information relevant to the given question. It integrates the understanding of both the original image and detailed local image to generate the final answer. Besides, we provide the corresponding visual CoT benchmark and pre-trained models for reproducibility, aiming to foster further research in the visual chain-of-thought for MLLMs.

To summarize, this paper makes the following contributions:

- We present a visual chain-of-thought dataset comprising 438k data items, each consisting of a question, an answer, and an intermediate bounding box as CoT contexts. Some items also contain detailed reasoning steps. The dataset spans across five distinct domains.   
- We propose a novel multi-turn processing pipeline for MLLMs that can dynamically focus on visual inputs and provide intermediate interpretable thoughts.   
- We introduce the visual chain-of-thought benchmark for evaluating MLLMs in scenarios where they need to focus on specific local regions or reasons to identify objects.

# 2 Related Works

Multi-modal LLMs. Since the advent of large language models (LLMs), their success in various language applications has paved the way for the development of multi-modal large language models (MLLMs), which integrate vision and language modalities. Initially, MLLMs were treated as dispatch schedulers to connect vision expert models, such as VisualChatGPT $[71]$ , HuggingGPT $[59]$ , and MM-REACT $[80]$ , in order to extend language models to other tasks and modalities. More recently, MLLMs have focused on aligning these modalities through extensive training on image-caption pairs or image-question conversations. Notable methods like LLaVA $[40]$ train a projector that maps image tokens to aligned representations of pre-trained LLMs. Other approaches, such as BLIP-2 $[32, 31]$ , adopt a query transformer (Q-Former) to learn image embeddings using learnable queries after obtaining image features. MoVA $[96]$ designs an adaptive router to fuse task-specific vision experts with a coarse-to-fine mechanism. In terms of training strategy, recent works $[40, 3, 68, 94, 10, 44]$ commonly employ a 2-stage framework. The first stage involves pre-training on image-caption pairs, while the second stage focuses on alignment by using question-answering triplets. MLLMs have also been extended to various applications, including fine-grained localization $[69, 29]$ such as object detection $[86]$ , video understanding $[84, 34, 11]$ , and image generation $[25, 56]$ .

InfographicsVQA   
![](images/1c58d31960406b55088acbaf457f54b794a7669eb446582e2387d22da3157aa5.jpg)

<details>
<summary>text_image</summary>

Legal Sector COVID-19
Headline Survey Results
June 2010
85%
Home working
There has been a clear split of first
have found it either easy or diffic
transition to home working.
63%
Very Difficult / 
Difficult
Very Easy
Easy
</details>

Question: How many have found home working very difficult?   
Answer: 22%   
CoT BBox: [83, 884, 140, 910]

DocVQA   
![](images/0606f2f31f41548544a8a3d82816a6e23bb2bd46bb0092582bcfef9ed515480c.jpg)

<details>
<summary>text_image</summary>

Text/ Doc
Anita Golden Pepper, Ph.D.
Diana Jane Mason, M.Sc.N.
September, 1977
</details>

Question: What is the name of the second person in the document?   
Answer: Diana Jane Mason   
CoT BBox: [1059, 1929, 1473, 1960]

Flickr30k   
![](images/eb719b881bdf1c0205d14916e436d899324e00b2505166d8570d1aa9e9106d24.jpg)  
General VQA   
Question: What activity is the puppy engaging in?   
Answer: The puppy is running through the grass with a yellow toy in its mouth, which looks to be an activity of fetching.   
CoT BBox: [195, 181, 271, 247]

TextCaps   
![](images/ed9f63135885eb579ffc15c439346587ea9d6d4f1263565ab658831fd11ebfa9.jpg)  
Text/ Doc   
Question: What number is associated with the bus line?
Answer: 12   
CoT BBox: [525, 101, 570, 145]

Birds-200-2011   
![](images/2ef4ff67e0bf6b3986b7819b58623d332aed43231d67370c2581d99aba12aab5.jpg)

<details>
<summary>natural_image</summary>

Close-up of a bird perched on pine branches with a red box highlighting the perched position (no text or symbols visible)
</details>

Fine-Grained   
Question: Does the bird in the picture have blue crown and black upperparts?   
Answer: No   
CoT BBox: [142, 118, 320, 252]

Open Images   
![](images/cefca0e336073760afd27f84ba60191800287f946ba04f4d7d74dabf3212a303.jpg)  
Relation Reasoning   
Question: What is the running man wearing on his hand in the picture?   
Answer: baseball glove   
CoT BBox: [378, 589, 492, 691]

Figure 1: Examples of five domains covered in the visual CoT dataset, with corresponding question-answer annotations and visual CoT bboxes: chart, text/doc, general VQA, fine-grained understanding, and relation reasoning. The red bounding boxes in the images highlight the critical image regions that provide necessary and related information for answering the questions.

Reasoning Capability of LLMs and MLLMs. LLMs have demonstrated impressive reasoning capabilities, enabled by in-context learning (ICL) [4], which allows feeding prompted samples and context. This capability has been further enhanced by chain-of-thought (CoT) [70] prompting, which enables LLMs to generate coherent intermediate reasoning steps toward the final answer. Previous studies have shown that LLMs benefit from manually written demonstrations [70] as well as zero-shot prompting outputs [26]. Trar [92] proposes a routing module to dynamically select informative regions based on the attention map. However, due to the domain gap between vision and text data, MLLMs fail to naturally inherit this reasoning capability. To address this limitation, researchers have focused on enhancing the reasoning capability of MLLMs in both the training and prompting paradigms. For instance, Flamingo [2] bridges the gap between these two modalities by pre-training on interleaved visual and textual data. Similarly, other works leverage visual grounded-reasoning [45, 93] data in training, such as Shikra [6] and KOSMOS-2 [53]. More recently, V\*[72] and CogCoM[55] modify the general mechanism in MLLMs and collect a series of visual reasoning steps as training data. On the other hand, studies have also explored prompting models [19, 87, 88, 51, 91] to understand complex visual scenes and tasks, focusing on the details of prompting techniques in MLLMs.

# 3 Visual CoT Dataset

There is a shortage of multimodal datasets for training multi-modal large language models (MLLMs) that require to identify specific regions in an image for additional attention to improve response performance. This type of dataset with grounding bbox annotations could possibly help the MLLM output intermediate interpretable attention area and enhance performance. To fill the gap, we curate a visual CoT dataset, as illustrated in Fig. 1 and Tab. 1. This dataset specifically focuses on identifying critical regions within images, a feature essential for models to concentrate on relevant visual elements

Table 1: One data example with detailed reasoning steps, of which we have collected about 98k of this type. The red bounding box shows the important image region for answering the question.

# An example of detailed reasoning steps in GQA dataset

Question: What appliance is to the right of the cabinet?

\###

Please think step by step and provide the bounding box coordinate

of the region that can help you answer the question better.

\###

Reasoning steps: 1. Identify the cabinet in the image.

2. Observe the area to the right of the identified cabinet.

3. Look for any appliance located to the right side of the cabinet.

4. Determine the name of the appliance found in this location

CoT BBox: [163, 44, 206, 67]

![](images/913566abf796ca48302ba4d3460c03ccc5c155314a971cc17d2fe943a40f70db.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with built-in appliances and dining areas (no visible text or symbols)
</details>

# Answer

The appliance is a microwave.

Table 2: The overview of the visual CoT dataset. The dataset spans five distinct domains and includes various source datasets, ensuring a broad representation of visual data styles. 

<table><tr><td>Domain</td><td>Source Dataset</td><td>Size</td><td>Used GPT-4?</td><td>Dataset Description</td></tr><tr><td rowspan="5">Text/Doc</td><td>TextVQA [61]</td><td>16k</td><td>No</td><td>Images with text</td></tr><tr><td>TextCaps [60]</td><td>32k</td><td>Yes</td><td>Images with text</td></tr><tr><td>DocVQA [50]</td><td>33k</td><td>No</td><td>Doc Images</td></tr><tr><td>DUDE [65]</td><td>15k</td><td>No</td><td>Doc Images</td></tr><tr><td>SROIE [20]</td><td>4k</td><td>No</td><td>Invoice Images</td></tr><tr><td>Fine-Grained Understanding</td><td>Birds-200-2011 [66]</td><td>10k</td><td>No</td><td>Images of birds</td></tr><tr><td rowspan="2">General VQA</td><td>Flickr30k [54]</td><td>136k</td><td>Yes</td><td>Images</td></tr><tr><td>Visual7W [95]</td><td>43k</td><td>No</td><td>Images</td></tr><tr><td>Charts</td><td>InfographicsVQA [49]</td><td>15k</td><td>No</td><td>Infographic</td></tr><tr><td rowspan="3">Relation Reasoning</td><td>VSR [38]</td><td>3k</td><td>No</td><td>Images</td></tr><tr><td>GQA [21]</td><td>88k</td><td>Yes</td><td>Images(with detailed reasoning steps)</td></tr><tr><td>Open images [28]</td><td>43k</td><td>No</td><td>Images</td></tr></table>

to improve response accuracy. Each data sample consists of a question, answer, and a corresponding visual bounding box across five domains, as shown in Tab. 2. Some data samples also include extra detailed reasoning steps.

To ensure a robust foundation for detailed visual and textual analysis, our dataset deliberately integrates a diverse selection of data including text/doc, fine-grained understanding, charts, general VQA, and relation reasoning. These data domains are deliberately chosen to cultivate a comprehensive skill set across varied analytical tasks: 1) Text/doc enhances MLLM's capabilities on OCR and contextual understanding, crucial for applications requiring text interpretation in complex environments. 2) Fine-grained understanding aids in identifying and distinguishing subtle differences in visual appearance and patterns. 3) Charts foster the ability to interpret graphical data, which are essential for business and scientific applications. 4) General VQA exposes models to a wide array of visual queries, improving their general usability. 5) Relation reasoning data develops spatial and contextual awareness of MLLMs, vital for interactive and navigational tasks. Together, these modalities ensure the dataset not only fills existing gaps but also enhances the versatility and contextual awareness of MLLMs across varied scenarios.

# 3.1 Data Generation

To collect and build a diverse and comprehensive Visual CoT dataset, we select twelve source datasets across five distinct domains, primarily consisting of Visual Question Answering (VQA) and Image Captioning datasets. We reuse their images and useful annotations, such as question-answer pairs, image captions, and object relations, to aid in building our dataset. The data construction process involves both linguistic and visual annotators to create question-answer pairs, and provide intermediate chain-of-thought bounding boxes indicating the crucial image region for answering the question. For the linguistic annotations, we employ GPT-4 [1], known for its robust language

![](images/b824a031a497c5f8ce146c68880fcd860cd518247a27a21f32ba5169061d23bd.jpg)  
Figure 2: Statistics of the proposed visual CoT dataset. We visualize the CoT bbox distribution, average bbox size, and average relative size of bbox area R for each source dataset.

understanding and generation capabilities. For the visual annotations, we choose PaddleOCR [15], an efficient and accurate tool for optical character recognition. In the following sections, we elaborate on the generation methods employed for each domain-specific dataset.

Text/Doc. We choose five text-related datasets to create data in this domain: TextVQA [61], DocVQA [50], DUDE [65], TextCaps [60], SROIE [20]. The five datasets focus on text recognition and comprehension in a variety of images and documents. TextVQA, DocVQA, DUDE and SROIE have already provided question-answer pairs, which we directly adopt. TextCaps, providing only captions and OCR tokens, required us to employ a linguistic annotator to create corresponding questions and answers (see further details in Appendix E.1). For the visual CoT bboxes, we then apply PaddleOCR[15] to detect OCR-identified regions in the image, and specify the CoT bounding boxes as the region that consists of words and sentences aligning with the answer. Furthermore, we also design a filtering pipeline to improve content quality. This process ensures that the areas highlighted by the bounding boxes are directly relevant to the questions.

Fine-Grained Understanding. For this domain, we use Birds-200-2011 [66], which is a widely-used dataset for fine-grained visual categorization. This dataset is not only rich in visual data but also includes detailed annotations about various bird parts and their attributes, along with bird bounding boxes in each picture. To leverage this dataset for our MLLM, we have formulated questions that challenge the model to identify specific characteristics or features present in the birds. These questions are designed to test the MLLM's ability to discern and recognize fine-grained details in the images.

General VQA. We use Flickr30k [54] and Visual7W [95] as the dataset for general VQA tasks. In Flickr30k, each image encompassed five captions and the bounding boxes of most objects mentioned in the captions. Employing a similar approach to TextCaps, we use GPT-4 to generate questions that require focusing on small objects in the images. The visual CoT bounding boxes in our proposed dataset correspond to the bboxes of objects identified and annotated in the official dataset. Visual7W has already provided the question-answer pairs with object-level grounding annotations.

Charts. We select the InfographicsVQA [49] dataset for its high-resolution infographics, which are advantageous for training MLLMs to pinpoint answer locations. Like in our Text/Doc data, we apply OCR techniques to identify regions containing the answers, using these identified areas as the CoT bounding boxes for more precise model training.

Relation Reasoning. We select the Visual Spatial Reasoning (VSR) [38], GQA [21], and Open Images [28] datasets to construct data focusing on relation-reasoning. These datasets are rich in spatial relational information among objects in images. For our chain-of-thought (CoT) bounding boxes, we use the bounding boxes surrounding the objects relevant to the question. For instance, if the question is “What is the material of the desk left to the woman?”, the bounding box of the desk to the woman’s left is designated as the visual CoT bounding box, providing more visual context for the MLLM’s reasoning process. In GQA [21] each image is associated with a scene graph of objects and relations. Each question comes with a structured representation of its semantics. With these annotations, we utilize GPT-4 to generate detailed reasoning steps, as illustrated in Tab. 1. The related prompt is available in Appendix E.3.

# 3.2 Dataset Analysis

We provide a visualization of the data statistics in Fig. 2. We partition the bboxes in each dataset into three groups (large, medium, small) based on the relative bounding box size R, which is the ratio of the CoT bbox size relative to the total image size. The visualization reveals that the majority of the annotated key regions, particularly in text-oriented datasets, occupy only a small portion of the entire image, highlighting the importance of identifying these crucial areas to enhance performance. Specifically, the average bounding box size is $247.8^{2}$ pixels, which well aligns with the common input resolution for a vision encoder ranges between 224 and 336 pixels, while the original image size is usually too large and needs down-sampling that loses information. These regions account for only about 13.2% of the image area. This highlights the necessity for MLLMs to accurately pinpoint these crucial areas to enhance processing efficiency and effectiveness. If the model fails to correctly identify and focus on these key regions, the majority of the image processed could be irrelevant, leading to inefficient computation, hallucination, and potential degradation in performance.

# 4 Enhancing MLLMs with Chain-of-Thought Capabilities

Along with the visual CoT dataset, we also propose a visual CoT MLLM framework named VisCoT, which employs standard models without specialized modifications, serving as a baseline to enhance MLLMs with visual CoT capabilities. In this section, we briefly introduce the framework, and illustrate the pipeline in Fig. 3. Readers are referred to Appendix B for more details.

VisCoT Pipeline. To train the MLLM baseline with visual CoT data, we add a CoT prompt (“Please provide the bounding box coordinate of the region that can help you answer the question better.”) to the question, asking the model to identify the most informative region of the image. VisCoT then determines this region and generates its bounding box. During the training phase, we utilize the ground truth bounding box to extract visual information rather than a predicted one in the following steps. With the original image $X_{0}$ and the bbox, a visual sampler extracts the localized image $X_{1}$ containing detailed information. The same vision encoder and projector are then used to extract visual tokens $H_{1}$ . The MLLM then integrates visual tokens from both the original and localized images $\{H_{0}, H_{1}\}$ to provide more precise and comprehensive answers. For data without visual CoT annotations, this procedure is omitted as indicated by the dashed box in Fig. 3. Here, the MLLM directly answers based on the input image alone. Our VisCoT baseline is thus adaptable to data in both annotated and non-annotated formats simultaneously.

Visual Sampler. Given the original image and the predicted bbox, the visual sampler's role is to accurately select the relevant region that considers the visual encoder requirement and bbox corner cases. We first calculate the center point $[x_0, y_0]$ , half-width $w_{half}$ , and half-height $h_{half}$ of the bounding box predicted by VisCoT. To capture more context and meet the square receptive field requirement of the CLIP model, $\max\{\max\{w_{half}, h_{half}\}, res_{half}\}$ is chosen as the sample size $s$ . $\mathrm{res}_{half}$ is the half input size of the vision encoder. Consequently, the visual sampler crops the region $[x_0 - s, y_0 - s, x_0 + s, y_0 + s]$ for further processing. During inference, if the calculated cropped box extends beyond the image boundaries, the center point is adjusted towards the center of the image to ensure the box remains within the image frame. This adjustment is important for improving the overall performance, as it can mitigate the impact of any detection inaccuracies.

Inference. VisCoT offers two options to generate answers: with or without the visual CoT process. If the CoT feature is not needed, users can simply provide the MLLM with the image and question. To engage the CoT feature, users can append the additional visual CoT prompt after the question.

![](images/245d6991a0b359026db3fc1ea27c099ecb8ed576fc7ba9ca4ee0cd963519d133.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Visual Chain-of-Thought Procedure"] --> B["Large Language Model"]
    B --> C["Vision Encoder"]
    B --> D["Visual Sampler"]
    C --> E["In which country is this event taking place? Please provide the bounding box coordinate of the region that can help you answer the question better."]
    D --> F["Vision Encoder"]
    D --> G["Vorsicht localized Image"]
    H["[0.011, 0.125, 0.349, 0.146"]]] --> I["↑"]
    J["It's Germany, as the 'Vorsicht' sign suggests, which is German for 'Caution'"].] --> K["↑"]
```
</details>

Figure 3: VisCoT first extracts visual tokens from an image and pinpoints the key region relevant to the question. Then, it processes the localized visual information. Finally, the MLLM integrates the information from the overall and localized images to construct a comprehensive and accurate answer.

Table 3: Performance on the Visual CoT benchmark. Datasets highlighted in grey indicate their training splits were not used in our model's training phase. Res indicates input image resolution. 

<table><tr><td colspan="2"></td><td colspan="5">Doc/Text</td><td>Chart</td></tr><tr><td>MLLM</td><td>Res.</td><td>DocVQA</td><td>TextCaps</td><td>TextVQA</td><td>DUDE</td><td>SROIE</td><td>InfographicsVQA</td></tr><tr><td>LLaVA-1.5-7B [39]</td><td> $336^2$ </td><td>0.244</td><td>0.597</td><td>0.588</td><td>0.290</td><td>0.136</td><td>0.400</td></tr><tr><td>LLaVA-1.5-13B [39]</td><td> $336^2$ </td><td>0.268</td><td>0.615</td><td>0.617</td><td>0.287</td><td>0.164</td><td>0.426</td></tr><tr><td>SPHINX-13B [37]</td><td> $224^2$ </td><td>0.198</td><td>0.551</td><td>0.532</td><td>0.000</td><td>0.071</td><td>0.352</td></tr><tr><td>VisCoT-7B</td><td> $224^2$ </td><td>0.355</td><td>0.610</td><td>0.719</td><td>0.279</td><td>0.341</td><td>0.356</td></tr><tr><td>VisCoT-7B</td><td> $336^2$ </td><td>0.476</td><td>0.675</td><td>0.775</td><td>0.386</td><td>0.470</td><td>0.324</td></tr><tr><td colspan="2"></td><td colspan="2">General VQA</td><td colspan="3">Relation Reasoning</td><td>Fine-grainedAverage</td></tr><tr><td>MLLM</td><td>Res.</td><td>Flickr30k</td><td>Visual7W</td><td>GQA</td><td>Open images</td><td>VSR</td><td>Birds-200-2011</td></tr><tr><td>LLaVA-1.5-7B [39]</td><td> $336^2$ </td><td>0.581</td><td>0.575</td><td>0.534</td><td>0.412</td><td>0.572</td><td>0.530</td></tr><tr><td>LLaVA-1.5-13B [39]</td><td> $336^2$ </td><td>0.620</td><td>0.580</td><td>0.571</td><td>0.413</td><td>0.590</td><td>0.573</td></tr><tr><td>SPHINX-13B [37]</td><td> $224^2$ </td><td>0.607</td><td>0.558</td><td>0.584</td><td>0.467</td><td>0.613</td><td>0.505</td></tr><tr><td>VisCoT-7B</td><td> $224^2$ </td><td>0.671</td><td>0.580</td><td>0.616</td><td>0.833</td><td>0.682</td><td>0.556</td></tr><tr><td>VisCoT-7B</td><td> $336^2$ </td><td>0.668</td><td>0.558</td><td>0.631</td><td>0.822</td><td>0.614</td><td>0.559</td></tr></table>

Model Training VisCoT baseline is trained in two stages. In the first stage, consistent with LLaVA-1.5, we freeze the weights of the vision encoder and LLM, and utilize image-text caption data for training. In the second stage, all weights are trainable. For more details, see Appendix B.

# 5 Experiments

Firstly, we provide an overview of the construction and evaluation of the CoT benchmark. Subsequently, in the evaluation phase, we begin by accessing VisCoT on the proposed benchmark (refer to Sec. 5.2). Additionally, we conduct further experiments to analyze the impact of essential components within VisCoT through an ablation study in Sec. 5.3. Finally, we showcase the capabilities of VisCoT in engaging complex multimodal conversations in Sec. 5.4. The training details and detection performance of the visual CoT bboxes can be found in Appendix B & C.

# 5.1 Visual CoT Benchmark

In this section, we provide an overview of our visual CoT benchmark, which primarily focuses on scenarios where the MLLM needs to concentrate on specific regions within a complete image. We utilize 12 source datasets, as shown in Fig. 1, and when an official training/evaluation split exists, we adopt it. In cases where such a split does not exist, we randomly divide the dataset. Additionally, we incorporate the test split of SROIE, DUDE, and Visual7W to evaluate the model's zero-shot visual CoT capabilities. Following the methodology of previous MLLM studies [33, 46], we employ

Table 4: Ablation study on the different BBox selection strategies. ‘w/o CoT’ indicates a standard, non-CoT-based inference process. ‘GT BBox’ uses annotated ground truth bboxes. ‘Random’ and ‘Center’ refer to using random and center bboxes instead of model predictions. 

<table><tr><td rowspan="2">BBox Strategy</td><td colspan="5">Doc/Text</td><td>Chart</td></tr><tr><td>DocVQA</td><td>TextCaps</td><td>TextVQA</td><td>DUDE</td><td>SROIE</td><td>InfographicsVQA</td></tr><tr><td>Baseline</td><td>0.355</td><td>0.610</td><td>0.719</td><td>0.279</td><td>0.341</td><td>0.356</td></tr><tr><td>w/o CoT</td><td>0.170</td><td>0.502</td><td>0.463</td><td>0.175</td><td>0.044</td><td>0.332</td></tr><tr><td>GT BBox</td><td>0.774</td><td>0.827</td><td>0.840</td><td>0.718</td><td>0.633</td><td>0.778</td></tr><tr><td>Random</td><td>0.208</td><td>0.463</td><td>0.495</td><td>0.157</td><td>0.146</td><td>0.378</td></tr><tr><td>Center</td><td>0.220</td><td>0.533</td><td>0.558</td><td>0.204</td><td>0.205</td><td>0.366</td></tr><tr><td rowspan="2">BBox Strategy</td><td colspan="2">General VQA</td><td colspan="3">Relation Reasoning</td><td>Fine-grainedAverage</td></tr><tr><td>Flickr30k</td><td>Visual7W</td><td>GQA</td><td>Open images</td><td>VSR</td><td>Birds-200-2011</td></tr><tr><td>Baseline</td><td>0.671</td><td>0.580</td><td>0.616</td><td>0.833</td><td>0.682</td><td>0.556</td></tr><tr><td>w/o CoT</td><td>0.610</td><td>0.554</td><td>0.600</td><td>0.656</td><td>0.634</td><td>0.534</td></tr><tr><td>GT BBox</td><td>0.692</td><td>0.699</td><td>0.796</td><td>0.896</td><td>0.792</td><td>0.577</td></tr><tr><td>Random</td><td>0.627</td><td>0.458</td><td>0.477</td><td>0.763</td><td>0.585</td><td>0.683</td></tr><tr><td>Center</td><td>0.653</td><td>0.529</td><td>0.547</td><td>0.803</td><td>0.657</td><td>0.609</td></tr></table>

Table 5: Ablation study on the visual sampler design. 

<table><tr><td>Expanded Cropping</td><td>Centered Cropping</td><td>Doc/ Text</td><td>Chart</td><td>General VQA</td><td>Relation Reasoning</td><td>Fine-grained</td><td>Average</td></tr><tr><td></td><td></td><td>0.399</td><td>0.321</td><td>0.621</td><td>0.668</td><td>0.509</td><td>0.496</td></tr><tr><td>√</td><td></td><td>0.410</td><td>0.328</td><td>0.625</td><td>0.678</td><td>0.531</td><td>0.506</td></tr><tr><td></td><td>√</td><td>0.434</td><td>0.331</td><td>0.641</td><td>0.677</td><td>0.521</td><td>0.518</td></tr><tr><td>√</td><td>√</td><td>0.461</td><td>0.356</td><td>0.626</td><td>0.710</td><td>0.556</td><td>0.550</td></tr></table>

ChatGPT [52] and ask it to assign a numerical score between 0 and 1, where a higher score indicates better prediction accuracy. For detailed information on the prompt used for ChatGPT-based evaluation, please refer to Appendix E.4.

# 5.2 Performance Evaluation

In this section, we comprehensively evaluate VisCoT across various multi-modal tasks to thoroughly assess our model's visual understanding ability. Tab. 3 highlights the enhancements through the visual CoT benchmark. We also showcase the baseline performance of our model on other benchmarks in Appendix D, where it directly answers questions without employing the visual CoT process.

In Tab. 3, we test our model and LLaVA-1.5 on the proposed visual CoT benchmark as detailed in Sec. 5.1. To demonstrate the impact of the chain-of-thought process, we also include the ablation study that removes this reasoning process and directly generates the response in a standard, direct manner. Notably, our pipeline shows significant improvement in the doc/text-related tasks and high-resolution image processing, even when the training splits from corresponding datasets are not utilized for the model training. For instance, SROIE [20] is a dataset that involves extracting key information from scanned receipts, such as the company name and the total price. Our model achieves $8 \times$ performance compared to the standard pipeline without a chain-of-thought process. Furthermore, the visual CoT pipeline also shows superior results in other benchmark tasks, showing its efficacy in enhancing the model's comprehensive visual and textual interpretation abilities.

# 5.3 Ablation Study

In the ablation studies below, in default, we ablate VisCoT-7B with a resolution of 224 and mainly evaluate in the proposed visual CoT benchmark.

Visual CoT BBox Selection Strategies. Tab. 4 showcases the performance of our model on the visual CoT benchmark using different strategies for bbox selection. As anticipated, employing ground truth annotated bounding boxes instead of model predictions yields the highest performance, surpassing the baseline by a significant margin. This can be considered the upper bound of our model's potential.

Token Efficiency. The visual CoT pipeline utilizes double the visual tokens for answer generation, leading us to assess its performance at various resolutions: 224, 336, and 448. As depicted in Fig. 4, the visual CoT pipeline exhibits improved token efficiency in our model. For instance, when equipped with the visual CoT, our model's accuracy at 224 resolution surpasses that of the standard pipeline at 448 resolution, while only using half the visual tokens.

![](images/b97afe63b0810fc25cc24d395752b737977e5224e49e49f66a0d69985b80f9ab.jpg)

<details>
<summary>line</summary>

| Visual Token Number | Without Visual CoT | With Visual CoT |
| ------------------- | ------------------ | --------------- |
| 128                 | 0.43               | -               |
| 256                 | -                  | -               |
| 512                 | 0.49               | 0.55            |
| 1024                | 0.54               | 0.58            |
| 2048                | -                  | 0.63            |
</details>

Figure 4: Trade-offs between visual token numbers and average accuracy on the visual CoT benchmark.

![](images/4928d0f8438553cb84881dafd545d305ea63271700ba4ad3e8ceead279ca168f.jpg)

<details>
<summary>text_image</summary>

DocVQA
Question: What is the heading of the page?
GT Answer: Returns from 1956 Sugar Quota Questionnaire as of November 23, 1956
VisCoT Answer [score: 0.50]: RETURNS FROM 1968 SUGAR QUOTA QUESTIONNAIRE
VisCoT (GT BBox) Answer [score: 0.50]: Returns from 1966 sugar quota questionnaire
VisCoT (w/o CoT) Answer [score: 0.20]: "Monthly Financial Statement"
Flickr30k
Question: What is at the left side of the dog?
GT Answer: teddy bear
VisCoT Answer [score: 1.00]: teddy bear
VisCoT (GT BBox) Answer [score: 1.00]: teddy bear
VisCoT (w/o CoT) Answer [score: 0.50]: A stuffed animal
Birds-200-2011
Question: Does the bird in the picture have blue crown and black throat?
GT Answer: No
VisCoT Answer [score: 1.0]: No, the bird in the picture does not have a blue crown and black throat.
VisCoT (GT BBox) Answer [score: 1.00]: No
VisCoT (w/o CoT) Answer [score: 0.00]: The bird has a blue crown and a yellow throat.
GQA
Question: What is the name of the device to the right of the baby on the left side?
GT Answer: laptop
VisCoT Thoughts:
1. Locate the baby referenced in the question.
2. Ascertain the baby's position within the given context, specifically confirming that the baby is on the left side.
3. Identify the device situated to the right of the located baby.
4. Request or deduce the name of the identified device.
VisCoT Answer [score: 1.00]: laptop
VisCoT (GT BBox) Answer [score: 1.00]: The device is a laptop
VisCoT (w/o CoT) Answer [score: 0.00]: screen
</details>

Figure 5: Visualization results of visual CoT to illustrate the difference between various inference modes. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue. The scores are evaluated by the ChatGPT. Best viewed in color and zoomed in.

Interestingly, random box selection demonstrates similar performance to the ‘w/o CoT’ approach, suggesting limited impact when the box selection is arbitrary or the prediction is incorrect. However, selecting the ‘Center’ box exhibits an improvement over the “Random” strategy, indicating that the central region of an image often contains more relevant information. This ablation study provides two key insights: firstly, our model excels at accurately predicting visual bounding boxes, and secondly, the precision of these box predictions significantly influences overall performance.

Visual Sampler. We ablate the visual sampler design in Tab. 5. Expanded Cropping refers to enlarging the cropped region if the region is smaller than the vision encoder's input size. Centered Cropping denotes moving the cropped region toward the center if the region extends beyond the image. The results reveal that more image context can bring better performance, and we suppose that it mitigates the problem of detection inaccuracies.

# 5.4 Visualization

This section displays VisCoT's qualitative performance through Fig. 5, highlighting its visual CoT ability to identify critical regions in images that aid in answering questions and synthesizing the combined contexts of both original and zoomed-in images. We also provide comparative results with different configurations: VisCoT (GT BBox), and VisCoT (w/o CoT). The accuracy of detection and depth of understanding directly contribute to the quality of the generated answers.

# 6 Conclusion

In this paper, we introduced VisCoT, a pioneering approach that enhances multi-modal large language models with visual chain-of-thought reasoning. This methodology addresses critical gaps in MLLMs, particularly in interpretability and processing dynamic visual inputs. Our visual CoT dataset offers 438k annotated question-answer pairs for detailed visual analysis. Our novel multi-turn processing pipeline allows MLLMs to dynamically focus and interpret visual data, mirroring human cognition. VisCoT provides more interpretable reasoning stages, and the visual CoT benchmark advances the evaluation of MLLMs' focus on specific image areas. Extensive experiments validate the framework's effectiveness, offering a promising starting point for further exploration in visual CoT.

Acknowledgement. This project is funded in part by National Key R&D Program of China Project 2022ZD0161100, by the Centre for Perceptual and Interactive Intelligence (CPII) Ltd under the Innovation and Technology Commission (ITC)'s InnoHK, by General Research Fund of Hong Kong RGC Project 14204021. Hongsheng Li is a PI of CPII under the InnoHK.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.   
[3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.   
[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
[5] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.   
[6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm's referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.   
[7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.   
[8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

[9] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024.   
[10] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.   
[11] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023.   
[12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.   
[13] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna.lmsys.org (accessed 14 April 2023), 2023.   
[14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.   
[15] Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, et al. Pp-ocr: A practical ultra lightweight ocr system. arXiv preprint arXiv:2009.09941, 2020.   
[16] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.   
[17] Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024.   
[18] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. arXiv preprint arXiv:2308.06394, 2023.   
[19] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023.   
[20] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.   
[21] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.   
[22] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Guanglu Song, Peng Gao, et al. Mmsearch: Benchmarking the potential of large models as multi-modal search engines. arXiv preprint arXiv:2409.12959, 2024.   
[23] Yang Jiao, Shaoxiang Chen, Zequn Jie, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Lumen: Unleashing versatile vision-centric capabilities of large multimodal models. arXiv preprint arXiv:2403.07304, 2024.

[24] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.   
[25] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36, 2024.   
[26] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.   
[27] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.   
[28] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020.   
[29] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023.   
[30] Hugo Laurençon, Daniel van Strien, Stas Bekman, Leo Tronchon, Lucile Saulnier, Thomas Wang, Siddharth Karamcheti, Amanpreet Singh, Giada Pistilli, Yacine Jernite, et al. Introducing idefics: An open reproduction of state-of-the-art visual language model, 2023. URL https://huggingface.co/blog/idefics.Accessed, pages 09–18, 2023.   
[31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.   
[32] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.   
[33] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.   
[34] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023.   
[35] Yian Li, Wentao Tian, Yang Jiao, Jingjing Chen, and Yu-Gang Jiang. Eyes can deceive: Benchmarking counterfactual reasoning abilities of multi-modal large language models. arXiv preprint arXiv:2404.12966, 2024.   
[36] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.   
[37] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.   
[38] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 2023.   
[39] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023.   
[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

[41] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.   
[42] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.   
[43] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.   
[44] Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji. Cheap and quick: Efficient vision-language instruction tuning for large language models. Advances in Neural Information Processing Systems, 36, 2024.   
[45] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multi-task collaborative network for joint referring expression comprehension and segmentation. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 10034–10043, 2020.   
[46] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023.   
[47] Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, and Yu Liu. Exploring the role of large language models in prompt encoding for diffusion models. arXiv preprint arXiv:2406.11831, 2024.   
[48] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016.   
[49] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.   
[50] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.   
[51] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. arXiv preprint arXiv:2311.17076, 2023.   
[52] OpenAI. Chatgpt. https://openai.com/blog/chatgpt/, 2023.   
[53] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.   
[54] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.   
[55] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom: Train large vision-language models diving into details through chain of manipulations. arXiv preprint arXiv:2402.04236, 2024.   
[56] Shengju Qian, Huiwen Chang, Yuanzhen Li, Zizhao Zhang, Jiaya Jia, and Han Zhang. Strait: Non-autoregressive generation with stratified image transformer. arXiv preprint arXiv:2303.00750, 2023.

[57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[58] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15120–15130, 2024.   
[59] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Huggingpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36, 2024.   
[60] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020.   
[61] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.   
[62] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.   
[63] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
[64] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
[65] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19528–19540, 2023.   
[66] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.   
[67] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022.   
[68] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.   
[69] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. Advances in Neural Information Processing Systems, 36, 2024.   
[70] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.   
[71] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.

[72] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 17, 2023.   
[73] Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, et al. Cares: A comprehensive benchmark of trustworthiness in medical vision language models. arXiv preprint arXiv:2406.06007, 2024.   
[74] Peng Xia, Siwei Han, Shi Qiu, Yiyang Zhou, Zhaoyang Wang, Wenhao Zheng, Zhaorun Chen, Chenhang Cui, Mingyu Ding, Linjie Li, et al. Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models. arXiv preprint arXiv:2410.10139, 2024.   
[75] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, and Huaxiu Yao. Mmed-rag: Versatile multimodal rag system for medical vision language models. arXiv preprint arXiv:2410.13085, 2024.   
[76] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao. Rule: Reliable multimodal rag for factuality in medical vision language models. arXiv preprint arXiv:2407.05131, 2024.   
[77] Jinjin Xu, Liwu Xu, Yuzhe Yang, Xiang Li, Yanchun Xie, Yi-Jie Huang, and Yaqian Li. u-llava: Unifying multi-modal tasks via large language model. arXiv preprint arXiv:2311.05348, 2023.   
[78] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15325–15336, 2023.   
[79] Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, and Xin Geng. Exploring diverse in-context configurations for image captioning. Advances in Neural Information Processing Systems, 36, 2024.   
[80] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.   
[81] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.   
[82] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023.   
[83] Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al. Map-neo: Highly capable and transparent bilingual large language model series. arXiv preprint arXiv:2405.19327, 2024.   
[84] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.   
[85] Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Jingjing Chen, and Yu-Gang Jiang. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597, 2024.   
[86] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.   
[87] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. What makes good examples for visual in-context learning? Advances in Neural Information Processing Systems, 2023.   
[88] Yuechen Zhang, Shengju Qian, Bohao Peng, Shu Liu, and Jiaya Jia. Prompt highlighter: Interactive control for multi-modal llms. arXiv preprint arXiv:2312.04302, 2023.   
[89] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.

[90] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.   
[91] Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, and Sibei Yang. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models. Advances in Neural Information Processing Systems, 36:5168–5191, 2023.   
[92] Yiyi Zhou, Tianhe Ren, Chaoyang Zhu, Xiaoshuai Sun, Jianzhuang Liu, Xinghao Ding, Mingliang Xu, and Rongrong Ji. Trar: Routing the attention spans in transformer for visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2074–2084, 2021.   
[93] Chaoyang Zhu, Yiyi Zhou, Yunhang Shen, Gen Luo, Xingjia Pan, Mingbao Lin, Chao Chen, Liujuan Cao, Xiaoshuai Sun, and Rongrong Ji. Seqtr: A simple yet universal network for visual grounding. In European Conference on Computer Vision, pages 598–615. Springer, 2022.   
[94] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.   
[95] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4995–5004, 2016.   
[96] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046, 2024.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]   
(b) Did you describe the limitations of your work? [Yes] See Appendix F.   
(c) Did you discuss any potential negative societal impacts of your work? [Yes] See Appendix G.   
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]   
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments (e.g. for benchmarks)...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] We have provided the related details in Appendix. The code, training data, benchmark, and checkpoints can be found in this GitHub repo: https://github.com/deepcs233/Visual-CoT   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] See ‘Training Details’ in Section Experiments. We also provide reproducible scripts that contain all hyperparameters in this GitHub repo: https://github.com/deepcs233/Visual-CoT   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [No]   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See Appendix B.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]   
(b) Did you mention the license of the assets? [Yes]   
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] We provide our code, training data, and checkpoints in this GitHub repo: https://github.com/deepcs233/Visual-CoT   
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [Yes] See Appendix H.   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] See Appendix H.

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# A Overview

Our supplementary includes the following sections:

- Section B: Framework details. Details for model design, implementation and training data.   
- Section C: Detection performance of the visual CoT bboxes. Details for detection performance for the intermediate visual CoT bounding boxes.   
- Section D: More experiment results. Additional performance evaluation and performance analysis.   
- Section E: Prompt design. Prompt for generating the visual CoT dataset and evaluating the performance.   
• Section F: Limitations. Discussion of limitations of our work.   
- Section G: Potential negative societal impacts. Discussion of potential negative societal impacts of our work.   
• Section F: More visualization. More Visualization of our dataset and demos.   
• Section I: Disclaimer. Disclaimer for the visual CoT dataset and the related model.

Following NeurIPS Dataset and Benchmark track guidelines, we have shared the following artifacts:

<table><tr><td>Artifcat</td><td>Link</td><td>License</td></tr><tr><td>Code Repository</td><td>https://github.com/deepcs233/Visual-CoT</td><td>Apache-2.0 license</td></tr><tr><td>Data</td><td>https://huggingface.co/datasets/deepcs233/Visual-CoT</td><td>CC BY 4.0</td></tr><tr><td>Model Weights</td><td>https://huggingface.co/collections/deepcs233/viscot-65fe883e2a0cdd3c59fc5d63</td><td>Apache-2.0 license</td></tr></table>

The authors are committed to ensuring its regular upkeep and updates.

# B Framework details

# B.1 Model details

We choose the pre-trained ViT-L/14 of CLIP $[57]$ as the vision encoder and Vicuna-7/13B $[13]$ as our LLM, which has better instruction following capabilities in language tasks compared to LLaMA $[64]$ . Consider an input original image, we take the vision encoder to obtain the visual feature. Similar to LLaVA $[40, 39]$ , we use a simple linear layer to project the image features into the word embedding space to obtain the visual tokens $H_{0}$ which share the same dimensionality of the LLM.

# B.2 Implementation details

Following the setup described by Vicuna $[13]$ , our model undergoes a two-stage training process. In the first stage, we pre-train the model for 1 epoch using a learning rate of 2e-3 and a batch size of 128. For the second stage, we fine-tune the model for 1 epoch on our visual CoT dataset, employing a learning rate of 2e-5 and a batch size of 128. The Adam optimizer with zero weight decay and a cosine learning rate scheduler are utilized. To conserve GPU memory during fine-tuning, we employ FSDP (Full Shard Data Parallel) with ZeRO3-style. All models are trained using $32 \times A100s$ . In the case of training the setting with a 7B LLM and a resolution of 224, the first/second pre-training stage completes within 1/16 hours.

# B.3 Training data details

We train the model on a reorganized Vision-Language dataset. The training data is a composite of three sources: the second stage data from LLaVA, data from Shikra's $[6]$ second stage, and our visual CoT data. The inclusion of data from Shikra, which features various datasets with positional annotations, such as RefCOCO $[24]$ for REC, visual gemone $[27]$ for grounding caption. These datasets can enhance VisCoT's ability to accurately identify and understand locations within images. This enhancement is crucial for tasks requiring precise spatial awareness. We listed all training data in Table 6. We removed the images from the training set that are the same as those in the testing or validation set to prevent potential data leakage. Our training data includes three parts, and they are from LLaVA-1.5, a subset of Shikra, and our proposed visual CoT dataset separately.

Table 6: The overview of our training dataset. 

<table><tr><td>Dataset</td><td>Size</td><td>Source Datasets</td></tr><tr><td>LLaVA-1.5</td><td>665K</td><td>LLaVA, ShareGPT, VQAv2, GQA, OKVQA OCRVQA, A-OKVQA, TextCaps, RefCOCO, VG</td></tr><tr><td>Shikra</td><td>1.4M</td><td>RefCOCO(+/g), VG, PointQA-Local/Twice Visual-7W, Flickr30K</td></tr><tr><td>Visual CoT dataset</td><td>376K</td><td>TextVQA, TextCaps, DocVQA, Birds-200-2011 Flickr30K, InfographicsVQA, VSR, GQA, Open images</td></tr></table>

# C Detection performance of the visual CoT bboxes

In Table 7, we present the detection performance based on the predicted CoT bounding boxes. A higher performance indicates that our VisCoT identifies the key regions with greater accuracy.

# D More experiment results

# D.1 Performance evaluation

In Tab. 8 and Tab. 9, we showcase the baseline performance of our model, where it directly answers questions without employing the visual CoT process.

Multi-modal Large Language Models Benchmarks. In Tab. 8, we evaluate our model on recently proposed MLLM benchmarks such as MME [16], POPE [36], MMbench [42], ScienceQA [43],

Table 7: Detection performance (Top-1 Accuracy@0.5) on the visual CoT benchmark. The ground truth bounding boxes used for computing the metric are the intermediate CoT bounding boxes annotated in our CoT benchmark. 

<table><tr><td colspan="2"></td><td colspan="5">Doc/Text</td><td colspan="2">Chart</td></tr><tr><td>MLLM</td><td>Res.</td><td>DocVQA</td><td>TextCaps</td><td>TextVQA</td><td>DUDE</td><td>SROIE</td><td colspan="2">InfographicsVQA</td></tr><tr><td>VisCoT-7B</td><td> $224^2$ </td><td>13.6</td><td>41.3</td><td>46.8</td><td>5.0</td><td>15.7</td><td colspan="2">7.2</td></tr><tr><td>VisCoT-7B</td><td> $336^2$ </td><td>20.4</td><td>46.3</td><td>57.6</td><td>9.6</td><td>18.5</td><td colspan="2">10.0</td></tr><tr><td colspan="2"></td><td colspan="2">General VQA</td><td colspan="3">Relation Reasoning</td><td>Fine-grained</td><td rowspan="2">Average</td></tr><tr><td>MLLM</td><td>Res.</td><td>Flickr30k</td><td>Visual7W</td><td>GQA</td><td>Open images</td><td>VSR</td><td>Birds-200-2011</td></tr><tr><td>VisCoT-7B</td><td> $224^2$ </td><td>49.6</td><td>31.1</td><td>42.0</td><td>57.6</td><td>69.6</td><td>67.0</td><td>37.2</td></tr><tr><td>VisCoT-7B</td><td> $336^2$ </td><td>51.3</td><td>29.4</td><td>49.5</td><td>59.3</td><td>54.0</td><td>47.1</td><td>37.6</td></tr></table>

Table 8: Comparison with SoTA methods on 8 benchmarks. VisCoT achieves the best performance on the most of benchmarks, and ranks second on the other. For a fair comparison, VisCoT generates responses directly, without the visual CoT process. SQA [43]; VQA $^{T}$ : TextVQA [61]; MME $^{P}$ : MME-Preception [16]; MME $^{C}$ : MME-Cognition [16]; POPE [36]; MMB: MMBench [42]; MMB $^{CN}$ : MMBench-Chinese [42]. $\dagger$ uses 50M in-house instruction-finetuning data. \* uses multiple vision encoders. 

<table><tr><td>Method</td><td>LLM</td><td>Res.</td><td>SQA</td><td>GQA</td><td> $VQA^T$ </td><td>POPE</td><td> $MME^P$ </td><td> $MME^C$ </td><td>MMB</td><td> $MMB^{CN}$ </td></tr><tr><td>BLIP-2 [31]</td><td>Vicuna-13B</td><td> $224^2$ </td><td>-</td><td>41.0</td><td>42.5</td><td>85.3</td><td>1293.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InstructBLIP [14]</td><td>Vicuna-7B</td><td> $224^2$ </td><td>-</td><td>49.2</td><td>50.1</td><td>-</td><td>-</td><td>-</td><td>36.0</td><td>23.7</td></tr><tr><td>InstructBLIP [14]</td><td>Vicuna-13B</td><td> $224^2$ </td><td>-</td><td>49.5</td><td>50.7</td><td>78.9</td><td>1212.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Shikra [6]</td><td>Vicuna-13B</td><td> $224^2$ </td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>58.8</td><td>-</td></tr><tr><td>IDEFICS-9B [30]</td><td>LLaMA-7B</td><td> $224^2$ </td><td>44.2</td><td>38.4</td><td>25.9</td><td>-</td><td>-</td><td>-</td><td>48.2</td><td>25.2</td></tr><tr><td>IDEFICS-80B [30]</td><td>LLaMA-65B</td><td> $224^2$ </td><td>68.9</td><td>45.2</td><td>30.9</td><td>-</td><td>-</td><td>-</td><td>54.5</td><td>38.1</td></tr><tr><td>Qwen-VL $^†$ [3]</td><td>Qwen-7B</td><td> $448^2$ </td><td>67.1</td><td>59.3</td><td>63.8</td><td>-</td><td>-</td><td>-</td><td>38.2</td><td>7.4</td></tr><tr><td>Qwen-VL-Chat $^†$ [3]</td><td>Qwen-7B</td><td> $448^2$ </td><td>68.2</td><td>57.5</td><td>61.5</td><td>-</td><td>1487.5</td><td>360.7</td><td>60.6</td><td>56.7</td></tr><tr><td>LLaVA1.5 [40]</td><td>Vicuna-7B</td><td> $336^2$ </td><td>66.8</td><td>62.0</td><td>58.2</td><td>85.9</td><td>1510.7</td><td>-</td><td>64.3</td><td>58.3</td></tr><tr><td>LLaVA1.5 [40]</td><td>Vicuna-13B</td><td> $336^2$ </td><td>71.6</td><td>63.3</td><td>61.3</td><td>85.9</td><td>1531.3</td><td>295.4</td><td>67.7</td><td>63.6</td></tr><tr><td>SPHINX* [3]</td><td>LLaMA-13B</td><td> $224^2$ </td><td>69.3</td><td>62.6</td><td>51.6</td><td>80.7</td><td>1476.1</td><td>310.0</td><td>66.9</td><td>56.2</td></tr><tr><td>VisCoT</td><td>Vicuna-7B</td><td> $224^2$ </td><td>68.2</td><td>63.1</td><td>55.4</td><td>86.0</td><td>1453.6</td><td>308.3</td><td>67.9</td><td>59.7</td></tr><tr><td>VisCoT</td><td>Vicuna-13B</td><td> $224^2$ </td><td>71.6</td><td>64.2</td><td>57.8</td><td>85.6</td><td>1480.0</td><td>255.4</td><td>66.9</td><td>60.5</td></tr><tr><td>VisCoT</td><td>Vicuna-7B</td><td> $336^2$ </td><td>68.3</td><td>62.0</td><td>61.0</td><td>86.5</td><td>1514.4</td><td>275.0</td><td>67.3</td><td>60.1</td></tr><tr><td>VisCoT</td><td>Vicuna-13B</td><td> $336^2$ </td><td>73.6</td><td>63.3</td><td>62.3</td><td>83.3</td><td>1535.7</td><td>331.8</td><td>67.4</td><td>61.6</td></tr></table>

TextVQA [61], GQA [21]. Our model still achieves comparative results across all benchmarks. This performance indicates that the visual CoT data we proposed not only enhances visual comprehension in CoT-specific scenarios but also boosts the model's overall visual understanding in standard inference setups. As demonstrated in Tab. 10, the implementation of visual CoT enables our model to achieve superior performance even with a lower resolution and a reduced number of visual tokens. This finding highlights the efficiency and effectiveness of the visual CoT approach in enhancing model accuracy.

Visual grounding. Furthermore, we evaluate VisCoT on REC benchmarks with RefCOCO [24], RefCOCO+ [48], and RefCOCOg [48] datasets. Our model outperforms the previous state-of-the-art models, including the specialist models such as G-DINO-L [41] and UNINEXT [78]. Notably, even with a minimal setup (7B LLM & 224 resolution), our approach outperforms methods that utilize higher resolutions or larger LLM models. This demonstrates that our dataset, enhanced with intermediate bounding boxes, significantly improves the model's precision in locating and understanding referred objects or regions. "Top-1 Accuracy@0.5" refers to the accuracy of a model in predicting the correct bounding box as the top prediction when the Intersection over Union (IoU) between the predicted and ground truth bounding boxes meets or exceeds 50%.

# D.2 Performance analysis

Tab. 4 shows that our baseline with visual CoT performs better than the model without CoT. We further investigate whether different bounding box sizes affect performance improvement. In Fig. 6,

Table 9: Performance (Top-1 Accuracy@0.5) on Referring Expression Comprehension (REC) tasks. For a fair comparison, VisCoT generates responses directly, without the visual CoT process. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Res.</td><td colspan="3">RefCOCO+</td><td colspan="3">RefCOCO</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>test-A</td><td>test-B</td><td>val</td><td>test-A</td><td>test-B</td><td>val-u</td><td>test-u</td></tr><tr><td colspan="10">Specialist models</td></tr><tr><td>UNINEXT [78]</td><td> $640^2$ </td><td>85.24</td><td>89.63</td><td>79.79</td><td>92.64</td><td>94.33</td><td>91.46</td><td>88.73</td><td>89.37</td></tr><tr><td>G-DINO-L [41]</td><td> $384^2$ </td><td>82.75</td><td>88.95</td><td>75.92</td><td>90.56</td><td>93.19</td><td>88.24</td><td>86.13</td><td>87.02</td></tr><tr><td colspan="10">Generalist models</td></tr><tr><td>VisionLLM-H [69]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>86.70</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OFA-L [67]</td><td> $480^2$ </td><td>68.29</td><td>76.00</td><td>61.75</td><td>79.96</td><td>83.67</td><td>76.39</td><td>67.57</td><td>67.58</td></tr><tr><td>Shikra 7B [6]</td><td> $224^2$ </td><td>81.60</td><td>87.36</td><td>72.12</td><td>87.01</td><td>90.61</td><td>80.24</td><td>82.27</td><td>82.19</td></tr><tr><td>Shikra 13B [6]</td><td> $224^2$ </td><td>82.89</td><td>87.79</td><td>74.41</td><td>87.83</td><td>91.11</td><td>81.81</td><td>82.64</td><td>83.16</td></tr><tr><td>MiniGPT-v2-7B [5]</td><td> $448^2$ </td><td>79.97</td><td>85.12</td><td>74.45</td><td>88.69</td><td>91.65</td><td>85.33</td><td>84.44</td><td>84.66</td></tr><tr><td>MiniGPT-v2-7B-Chat [5]</td><td> $448^2$ </td><td>79.58</td><td>85.52</td><td>73.32</td><td>88.06</td><td>91.29</td><td>84.30</td><td>84.19</td><td>84.31</td></tr><tr><td>Qwen-VL-7B [3]</td><td> $448^2$ </td><td>83.12</td><td>88.25</td><td>77.21</td><td>89.36</td><td>92.26</td><td>85.34</td><td>85.58</td><td>85.48</td></tr><tr><td>Qwen-VL-7B-Chat [3]</td><td> $448^2$ </td><td>82.82</td><td>88.59</td><td>76.79</td><td>88.55</td><td>92.27</td><td>84.51</td><td>85.96</td><td>86.32</td></tr><tr><td>Ferret-7B [82]</td><td> $336^2$ </td><td>80.78</td><td>87.38</td><td>73.14</td><td>87.49</td><td>91.35</td><td>82.45</td><td>83.93</td><td>84.76</td></tr><tr><td>u-LLaVA-7B [77]</td><td> $224^2$ </td><td>72.21</td><td>76.61</td><td>66.79</td><td>80.41</td><td>82.73</td><td>77.82</td><td>74.77</td><td>75.63</td></tr><tr><td>SPHINX-13B [37]</td><td> $224^2$ </td><td>82.77</td><td>87.29</td><td>76.85</td><td>89.15</td><td>91.37</td><td>85.13</td><td>84.87</td><td>83.65</td></tr><tr><td>VisCoT-7B</td><td> $224^2$ </td><td>85.68</td><td>91.34</td><td>80.20</td><td>90.60</td><td>93.49</td><td>86.65</td><td>85.29</td><td>86.04</td></tr><tr><td>VisCoT-7B</td><td> $336^2$ </td><td>87.46</td><td>92.05</td><td>81.18</td><td>91.77</td><td>94.25</td><td>87.46</td><td>88.38</td><td>88.34</td></tr><tr><td>VisCoT-13B</td><td> $224^2$ </td><td>86.26</td><td>91.20</td><td>80.57</td><td>91.40</td><td>93.53</td><td>87.26</td><td>86.62</td><td>86.79</td></tr></table>

Table 10: Performance on VQA benchmarks. 

<table><tr><td>Model Res.</td><td>LLaVA-1.5-7B  $336^2$ </td><td>VisCoT-7B (w/o COT) $224^2$ </td><td>VisCoT-7B  $224^2$ </td><td>VisCoT-7B (w/o COT) $336^2$ </td><td>VisCoT-7B  $336^2$ </td></tr><tr><td>DocVQA</td><td>21.6</td><td>14.4</td><td> $\underline{39.0}$ </td><td>29.4</td><td>49.3</td></tr><tr><td>TextVQA</td><td>58.2</td><td>55.5</td><td> $\underline{62.9}$ </td><td>60.2</td><td>66.9</td></tr><tr><td>ChartQA</td><td>17.7</td><td>14.2</td><td> $\underline{19.2}$ </td><td>17.5</td><td>22.8</td></tr></table>

we divide each evaluation dataset into five equal parts based on their relative bounding box sizes. We observe that the visual CoT usually achieve greater improvement when the corresponding bounding box is relative smaller.

![](images/0f4174a3932ddb7d3629add94fda651979f27060b5435feb0016586b0a0a11f8.jpg)  
Figure 6: Visualization of performance improvement across different bounding box relative sizes for different source datasets. We find that visual CoT shows a larger improvement in cases where the queried object is relatively small. Red bars represent evaluation data samples where the model with CoT outperforms the model without CoT. Green bars indicate the opposite. The y-axis represents different ranges of relative sizes of bboxes R. For example, the 20-40% range indicates that the bboxes in this range occupy the relatively small 20-40% quantile within the entire dataset. For clarity, samples where both models achieve the same scores are omitted.

# E Prompt design

# E.1 Generating the dataset for TextCaps

You are an AI visual assistant, and you are seeing a single image. What you see is provided with several sentences and Ocr\_tokens, describing the same image you are looking at. Ocr\_tokens indicates the text in the image. Answer all questions as you are seeing the image. Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask THREE diverse questions and give corresponding answers. Again, do not ask about uncertain details. Do not just makeup questions and answers based on Ocr tokens. Your response should include questions asking about the textual information of the image, the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Please only ask questions that have definite answers:

- One can see the content in the image that the question asks about and can answer confidently;   
- One can determine confidently from the image that it is not in the image. Do not ask any questions that cannot be answered confidently.   
- One can not see the Ocr\_tokens, so the question must not mention ‘Ocr’

Craft Questions Around Ocr\_tokens: Create questions that directly pertain to these identified words or phrases. Ensure that the question is structured in a way that the answer MUST be a word or phrase directly from the Ocr\_tokens. Your answer cannot contain words outside of Ocr\_tokens. The answers must be within three words.

Please follow the provided format:

Question: [question]

Answer: [answer]

Here is the context you need to process:

Image description: { }

Ocr\_tokens: { }

# E.2 Generating the dataset for Flickr30k

You are an AI visual assistant, and you are seeing a single image. What you see are provided with five sentences, describing the same image you are looking at. Each sentence includes specific objects mentioned and their corresponding locations within the image (e.g., [a peach] is located at [area: 95162]) Answer all questions as you are seeing the image. Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers. The generated questions need closer examination of specific regions in the image to gather detailed information for answering. The generated answers must be based on the corresponding area. When creating your questions, keep the following considerations in mind:

- Direct Alignment: Ensure the "Focus Area" specified in each question directly corresponds to the content of the question. For instance, if the question refers to "two women", the focus area should align with the portion described as "[Two women]" in the image description.   
- Image-Only Basis: Respondents will only have access to the image itself and will NOT see the provided descriptions or area details. Ensure your questions can be answered by viewing the image alone.   
- Avoid Repetition: Each question should be distinctive without overlapping content.   
- Clarity and Precision: The answers to your questions should be both lucid and exact. Evade vagueness.   
- Restricted Question Formats: Refrain from phrasing questions like "What's in region xx?" or "What happens in description 1?". The terms "description" and "region" should not appear in your questions & answers.   
- MUST: The "Focus Area" you provide can answer the question you provide.

Please follow the provided format, area\_id is a number:

Question: [question]

Focus Area: [area: area\_id]

Answer: [answer]

Here is the data you need to process:

Describe 1: With a barn in the background a child puts her head through a hole in a cow cutout and smiles for the camera.

[a barn] is located at [area: 62407]

[a child] is located at [area: 62402]

[a hole] is located at [area: 62405]

...

# E.3 Generating the dataset with detailed reasoning steps for GQA

You are an AI visual assistant, and you are seeing a single image. I will provide a question-answer pair along with the corresponding reasoning steps. The question and answer are based on an image. You need to generate the pure reasoning text in a step-by-step format, with each step clearly numbered (1. 2. 3. ... etc). The reasoning text should help solve the question and reach the final answer without including or hinting at the answer itself. The reasoning text must not include any ID numbers.

Here is the data you need to process:

Question: What appliance is to the right of the cabinet?

Answer: The appliance is a microwave.

Reasoning steps: [{"operation": "select", "dependencies": [], "argument": "cabinet (3588933)"}, {"operation": "relate", "dependencies": [0], "argument": "appliance,to the right of,s (1564001)"}, {"operation": "query", "dependencies": [1], "argument": "name"}]

...

![](images/8c95ef18c7df8aa952893f65cd9bd1b81517b8267e7e65d4061a3a9b1cd81f92.jpg)

<details>
<summary>text_image</summary>

What does the top post it have written on it? Please provide the bounding box coordinate of the region that can help you answer the question better.
[160.9, 330.6, 199.0, 390.5]
my name is mary
Ground truth: Lost
my name is mary
</details>

Figure 7: Visualization results of the VisCoT. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue. In this case, our model incorrectly predicts the CoT region, leading to a wrong answer.

# E.4 Evaluation for the visual CoT benchmark using the ChatGPT

You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.

Question: { }

Standard answer: { }

Model's answer: { }

# F Limitations

In scenarios where the input image contains extensive information or the question is particularly complex, VisCoT may struggle to identify the most relevant region for answering the question. As shown in Figure 7, this challenge can sometimes result in the model being misled and producing incorrect responses.

Our data pipeline inherits the limitations of utilizing GPT-4 API. (1) Accuracy and Misinformation: Generated content may not always be accurate, which could lead to the spread of misinformation. To mitigate this, we have designed a comprehensive filtering script as a post-process to improve content quality. (2) Bias and Fairness: Since we do not have access to the training data of GPT-4, the generated instructional data might reflect inherent biases, potentially reinforcing social or cultural inequalities present in the base model training. In terms of data usage, we explicitly state that OpenAI's terms must be adhered to, and the data can only be used for research purposes.

# G Potential negative societal impacts

The potential negative societal impacts of our work are similar to other MLLMs and LLMs. The development of Visual CoT and MLLMs, while advancing AI, poses societal risks like increased privacy invasion, the perpetuation of biases, the potential for misinformation, job displacement, and ethical concerns regarding accountability and consent.

Sroie   
![](images/f55d57332c4fd2f295be71ef69d4ca9478dc85b68b8270c219b678aeb6b28ab4.jpg)

<details>
<summary>text_image</summary>

Text/ Doc
LIGHTROOM GALLERY SDN BHD
NO. 28, JALAN ASTANA IC,
BANDAR BUKIT RAJA, 41050
KLANG SELANGOR D.E., MALAYSIA
ROC No. 1 (107905) 1-1
</details>

Question: What is the company in the invoice shown in the picture?   
Answer: LIGHTROOM GALLERY SDN BHD   
CoT BBox: [173, 86, 680, 144]

DUDE   
![](images/9b983c88e36dd238b7e0a4b911f91ae6064938e952616afb36727483cbb4bff5.jpg)

<details>
<summary>text_image</summary>

CALIFORNIA DEPARTMENT OF SOCIAL SERVICES
COMMUNITY CARE LICENSING DIVISION
COLD REGIONAL Office, 801 TRAGER AVE., SUITE 100
PRAVING, CA 5006
FACILITY NUMBER: 410200506
FACILITY TYPE: 850
TELEPHONE: (650) 365-8094
ZIP CODE: 94061
DATE: 01/31/2013
TIME BEGAN: 09:00 AM
TIME COMPLETED: 10:30 AM
</details>

Question: What is the facility number?   
Answer: 410500506   
CoT BBox: [1310, 363, 1478, 400]

InfographicsVQA   
![](images/78002aa5658b1f82565eb31d47fc22ed3b0155922957666c2ab1c95b80e429b0.jpg)

<details>
<summary>text_image</summary>

SEC Bowl / Alabama Bowl Games '14'
Geek ALABAMA
Three Alabama Bowl Games
Museum NHL-Crivala Bowl
South Alabama vs. Bowling Green
BOWL
December 20th - 8:15 pm CT - ESPN
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Florida
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Blue
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Red
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Orange
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
Turkey CS - Green
</details>

Question: When is the South Alabama vs Bowling   
Green match?   
Answer: December 20th - 8:15 pm CT   
CoT BBox: [631, 678, 1403, 725]

TextCaps   
![](images/f995713d4d6f8caa5ee094b9460ba8dafb399965ded71b403d5fa4913d7777fa.jpg)

<details>
<summary>natural_image</summary>

Nighttime stadium scene with a soccer field and a colorful football advertisement featuring Liverpool logo (no readable text or symbols)
</details>

Question: Which sports team is being supported by the   
fans in the stadium?   
Answer: LIVERPOOL   
CoT BBox: [445, 356, 610, 395]

Flickr30k   
![](images/036539868546d4698e618f3d43e7d39910310c67ffebd4e2c7c3b3b6e4f0b5fc.jpg)

<details>
<summary>natural_image</summary>

Two-panel image showing a person roller skating on a track and a close-up of the same skis with red annotation boxes highlighting the motion (no text or symbols)
</details>

Question: Is there a participant who might be   
distinguishable by heavier build and what are they wearing on their legs?   
Answer: Yes, there is a heavier participant who is wearing   
striped socks while she skates around the rink.   
CoT BBox: [132, 287, 180, 350]

Flickr30k   
![](images/c5019de1eae34721223f0bcfc0e0107d9e23472c90d8b4b37f8764ebf16bd896.jpg)

<details>
<summary>natural_image</summary>

Two children in colorful raincoats on a beach, one with a red dashed arrow pointing to the other (no text or symbols visible)
</details>

Question: What kind of headwear can be seen on the   
children standing on the sandy beach?   
Answer: The children are wearing winter hats.   
CoT BBox: [136, 67, 215, 137]

Figure 8: Examples in the visual CoT dataset, with corresponding question-answer annotations and visual CoT bboxes. The red bounding boxes in the images highlight the critical image regions that provide necessary and related information for answering the questions.

# H More visualization

We provide more visualization results of our proposed visual CoT dataset in Fig. 8, Fig. 9.

We provide more visualization results of our VisCoT baseline in Fig. 10, Fig. 11, Fig. 12, Fig. 13.

# I Disclaimer

This dataset was collected and released solely for research purposes, with the goal of making the MLLMs dynamically focus on visual inputs and provide intermediate interpretable thoughts. The authors are strongly against any potential harmful use of the data or technology to any party.

Intended Use. The data, code, and model checkpoints are intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported

![](images/3a2845f778342057ab0a03cb3fd03ee105db40f6d53b1d869ddf7a379e0e8946.jpg)

<details>
<summary>text_image</summary>

General VQA
Visual7W
Question: Why was the picture taken?
Answer: To show the cake.
CoT BBox: [133, 203, 210, 246]
</details>

![](images/d44727bad902ac24e7535ad5922fd96273553bf2a4f76de7dd8c79f80baae02f.jpg)

<details>
<summary>text_image</summary>

VSR
Relation Reasoning
Question: Is the truck part of the cake?
Answer: No
CoT BBox: [1, 6, 156, 120]
</details>

![](images/b21bb7b6eed4b10bcb212418daf155a2a080da7cd9dd970e2f5e721c198c5eff.jpg)

<details>
<summary>text_image</summary>

Relation Reasoning
GQA
Question: Who is dressed in blue?
Answer: Boy
CoT BBox: [223, 173, 324, 198]
</details>

![](images/8ff4d59f49554e4e607668c09942b74bed2855703be9f5ac314a708fd2dae44c.jpg)

<details>
<summary>text_image</summary>

Fine-Grained
Birds-200-2011
Question: Does the bird in the picture have iridescent
underparts and white crown?
Answer: No
CoT BBox: [134, 69 430 263]
</details>

Figure 9: Examples in the visual CoT dataset, with corresponding question-answer annotations and visual CoT bboxes. The red bounding boxes in the images highlight the critical image regions that provide necessary and related information for answering the questions.

in the reference paper. The data, code, and model checkpoints are not intended to be used in clinical care or for any clinical decision making purposes.

Primary Intended Use. The primary intended use is to support AI researchers reproducing and building on top of this work. VisCoT and its associated models should be helpful for exploring various vision question answering (VQA) research questions.

Out-of-Scope Use. Any deployed use case of the model — commercial or otherwise — is out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are intended for research use only and not intended for deployed use cases.

![](images/f29f5faee351028ac07dcd0021d7d532f40fc21d68f587f6074819a1d54eeb18.jpg)

<details>
<summary>text_image</summary>

What is the team name on the jersey?
Please provide the bounding box
coordinate of the region that can help
you answer the question better.
[331.5, 385.9, 562.5, 522.3]
orioles
orioles
</details>

![](images/a5d82b24a92414cb52d8e2493c9ad6138b44ea897c032f648586517f8345feea.jpg)

<details>
<summary>text_image</summary>

First word in green? Please provide the bounding box coordinate of the region that can help you answer the question better.
[326.9, 434.1, 490.3, 492.7]
Fagiolina
Trasimento
fagiolina
</details>

![](images/aaedb2b2d90ba13d6e52acb8c78df0b6224eeee82afcdcc1f3c844f1731b8c11.jpg)

<details>
<summary>text_image</summary>

What is the team name on the jersey?
Please provide the bounding box
coordinate of the region that can help
you answer the question better.
[321.9, 547.8, 424.6, 571.3]
AZKARAN PRISON
AYR 3 0 0
azkaban
HAVE YOU
SEEN THIS
WIZARD?
AZKARAN PRISON
AYR 3 0 0
APPROACH WITH OVEREAT CALIFION
OF IN NOT TO BE SPECT TOTAL
MAGIC ASSETS FOR YOUR
ANS INFORMATION IS USED TO THE TARGET OF THIS
WORK IN THE RIGHT WOULD BE TOWLED BY MARS
NOTI APPROACH YOUR NAME OF YOUR HOME
</details>

Figure 10: Visualization results of the VisCoT. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue.

![](images/240384693e10d2228131b437ce8845d7e7abf5818443a09a4b25cb686717a7be.jpg)

<details>
<summary>text_image</summary>

What brand radio is this? Please provide the bounding box coordinate of the region that can help you answer the question better.
[628.5, 108.1, 732.0, 132.9]
TECSUN
tecsun
</details>

![](images/f3f7178e2a7db0a82c789c24ef0b0f1a72d0495e27bc1a871a83d4144598e88c.jpg)

<details>
<summary>text_image</summary>

What is the licence plate number?
Please provide the bounding box
coordinate of the region that can help
you answer the question better.
[546.5, 533.2, 626.9, 562.0]
65-02
</details>

![](images/74b37229d5c8ca00bd365e732d0a4f9185220b1f23e1a3ca6a07bef2c0cb1a7c.jpg)

<details>
<summary>text_image</summary>

Who is the first reference? Please
provide the bounding box coordinate
of the region that can help you
answer the question better.

[403.0, 598.0, 699.0, 630.0]

William R. Beisel, M.D.
Deputy for Science

William R. Beisel, M.D.
</details>

Figure 11: Visualization results of the VisCoT. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue.

![](images/1e1db414171db1b76366a1f080fbb909fb09e7826cb5654410a934a477063e02.jpg)

<details>
<summary>text_image</summary>

What activity is the man engaged in
while sitting down? Please provide
the bounding box coordinate of the
region that can help you answer the
question better.
[163.0, 226.0, 199.0, 257.0]
He is reading cards.
</details>

![](images/fc6138fe5b066c9b9bc00d25b57b3859c8a6ae51ffeced9f4f27631ae7910e98.jpg)

<details>
<summary>text_image</summary>

Which department is shown on page 10 top left corner? Please provide the bounding box coordinate of the region that can help you answer the question better.
[188.0, 131.0, 513.0, 162.0]
State of Illinois
Department of Human Rights
Department of human rights
Procedures for Housing Cases
Rule of EDR: The Bill District of Hunan Rights (DHR) is the state agency responsible for entering the Illinois State Rights Act. The rule of EDR is to be defined as: (1) State Council District of Illinois, and (2) Bill District of Houston, which is the organization of the building of the construction in Chicago. More formal rights are available to the State Council District of Houston, which, including forms and for housing claims.
Information for Housing Countries: DHR applies the court's interest in establishing the construction house rights. However, we cannot have legal or administrative other party, government or not required, but you would be represented, you must obtain own or her own rights to identify any other party or private entity may be arising from other parties or local authorities. If you are not willing to pay more money, you may also pay an insurance for any other party or private entity. If you are not willing to pay more money, you may also pay an insurance for any other party or private entity. If you are not willing to pay more money, you may also pay an insurance for any other party or private entity. If you are not willing to pay more money, you may also pay an insurance for any other party or private entity. If you are not willing to pay more money, you may also pay an insurance for any other party or private entity. If you have a right to pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay some money. If you are not willing to pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also pay more money, you may also earn more money than your own company.
File No. 10-0004 - Complaints may initiate a change writing, by phone in person. IDH does not suggest reasons to file, but discuss the risks of distribution and regulate the filing procedures. After a charge bill, the case is given that IDH's charge number must be reached on the exercise of the purchase price. The person or organization of the charge is fixed upon. Constitutional under Federal Housing Law, DHR will allow us to receive a copy of this document with the U.S. Department of Housing and other (Development of Real Estate Development) (EADS) per copy or equivalent agreement. Cases that are validated and/or HDH will refer to: "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don’t know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "I don't know," "U.S." or similar to the Office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the Office of Chicago; and I don't know whether it is able to write out the address of the Office of Chicago; and I don't know whether it is able to write out the address of the Office of Chicago; and I don't know whether it is able to write out the address of the Office of Chicago; and I don't know whether it is able to write out the address of the Office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address of the office of Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out the address in Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write outThe Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is able to Write out The Office of Chicago; and I don't know whether it is allowed until that time will be paid out. If you are not willing to pay more money, you may also pay more money than your own company.
Incorporates by: Complaints should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Complaints should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Complaints should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Complaints should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than your own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issuedby a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member whohas a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Compaintors should be issued by a member who has a right to pay more money than its own company.
Incorporatesby: Com paintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Com paintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Com paintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Com paintors should be issued by a member who has a right to pay more money than its own company.
Incorporates by: Com paintors should be issued by a member who has a right topay less than 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year after 1 year later
and then
Sectional Rights
The bill District of Illinois shall have no further written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did nothave written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not have written consent or permission from which he did not hawtharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewhharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewharlewhharlewharlewharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlewhharlef wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherling hwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarlurees wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinggwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarlureed wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlinghwarluree wherlingkhanrulicwah#ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch###ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch######ch#####<nl>
Sectional Rights
Table 90-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-99-
Sectional Rights
Table 80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-80-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-60-65
Sectional Rights
Table 75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75-75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -75 -23
Sectional Rights
Table 62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/62/33
Sectional Rights
Table 43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/43/8
Sectional Rights
Table 34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34/34 /8
Sectional Rights
Table 25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /25 /1
Sectional Rights
Table 18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 /18 / 1
Sectional Rights
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 | Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights |
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1 + Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
Table 1+ Sectional Rights:
</details>

![](images/553db4ff3eb19a9553e8eec7d3f6bc7fd6b8605b6a6bb09988b2375c10d8226d.jpg)

<details>
<summary>text_image</summary>

What is the address in the invoice shown in the picture? Please provide the bounding box coordinate of the region that can help you answer the question better.
[251.0, 395.0, 646.0, 427.0]
47100 PUCHONG, BELANGO
EVERGREEN LIGHT SDN BHD
(827655-U)
NO.3-1, JALAN PUTERI 7/11,
BANDAR PUTTERI
TEL/FAX: 03-03608719
(NO. ID: GET: 003389363472 )
TAX INVOICE
NO.: 142507
EDDIE #01-01 18/08/2017 17:21
E.CODE QTY UPRICE AMT
11598 2 12.26 24.525
PROGRAM LDW LED STICK
SUB-TOTAL 24.52
THM (STTE) 1.47
SUB-TOTAL 25.58
ROUNDING ADJUST 0.01
TOTAL ITEMS: 1
TOTAL QUANTITY: 2 RM
Total Incl GST 26.00
Pay Cash 26.00
Change 0.00
OUT Taxable Ant: GET Ant.
s-BRIE: 5% 24.52 1.47
THANK YOU
GOOD BOLD ARE NOT REFUNDABLE FOR CASH.
</details>

Figure 12: Visualization results of the VisCoT. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue.

![](images/3dfd300cb0e9d29c26798236a440aee013ea9028d8dffdce5c2c2bf1ae92363a.jpg)

<details>
<summary>text_image</summary>

What kind of furniture is left of the ladder? Please provide the bounding box coordinate of the region that can help you answer the question better.
[35.0, 251.0, 65.0, 296.0]
chairs
</details>

![](images/43e751ccedb072f78fdf5b95ea0069bfb3e3ffd581777d4faba59689acbbf296.jpg)

<details>
<summary>text_image</summary>

What color is the toy that the little girl
is carrying while walking? Please
provide the bounding box coordinate
of the region that can help you
answer the question better.
[237.0, 211.0, 297.0, 316.0]
The toy is pink and blue.
</details>

![](images/4cf9d3de1d8daed8cd13d5198d494c9fe3c537ed8aa5c90b636c25f63373a388.jpg)

<details>
<summary>text_image</summary>

What is the girl carrying? Please provide the bounding box coordinate of the region that can help you answer the question better.
[213.0, 185.0, 265.0, 374.0]
skateboard
</details>

Figure 13: Visualization results of the VisCoT. Model-generated bounding boxes are shown in red, while ground truth (GT) bounding boxes are in blue.