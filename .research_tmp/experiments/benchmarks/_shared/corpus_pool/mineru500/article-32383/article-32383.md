# EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding

Muye Huang $^{1,2}$ , Han Lai $^{1,2}$ , Xinyu Zhang $^{1,3}$ , Wenjun Wu $^{1,3}$ , Jie Ma $^{2*}$ , Lingling Zhang $^{1,2}$ , Jun Liu $^{1,2}$

$^{1}$ School of Computer Science and Technology, Xi'an Jiaotong University $^{2}$ MOE KLINNS Lab, Xi'an Jiaotong University

$^{3}$ Shaanxi Province Key Laboratory of Big Data Knowledge Engineering {huangmuye, hanlai, zhang1393869716, nickjun98}@stu.xjtu.edu.cn, {jiema, zhanglling, liukeen}@xjtu.edu.cn

# Abstract

Chart understanding enables automated data analysis for humans, which requires models to achieve highly accurate visual comprehension. While existing Visual Language Models (VLMs) have shown progress in chart understanding, the lack of high-quality training data and comprehensive evaluation benchmarks hinders VLM chart comprehension. In this paper, we introduce EvoChart, a novel self-training method for generating synthetic chart data to enhance VLMs' capabilities in real-world chart comprehension. We also propose EvoChart-QA, a noval benchmark for measuring models' chart comprehension abilities in real-world scenarios. Specifically, EvoChart is a unique self-training data synthesis approach that simultaneously produces high-quality training corpus and a high-performance chart understanding model. EvoChart-QA consists of 650 distinct real-world charts collected from 140 different websites and 1,250 expert-curated questions that focus on chart understanding. Experimental results on various open-source and proprietary VLMs tested on EvoChart-QA demonstrate that even the best proprietary model, GPT-4o, achieves only 49.8% accuracy. Moreover, the EvoChart method significantly boosts the performance of open-source VLMs on real-world chart understanding tasks, achieving 54.2% accuracy on EvoChart-QA.

Homepage — https://github.com/MuyeHuang/EvoChart

# 1 Introduction

Chart Question Answering (CQA) aims to answer specific questions based on the context provided by chart images, enabling automated data analysis, such as the business data reports. This process requires complex chart understanding and visual reasoning skills to interpret various elements, including visual components, text and values. Consequently, CQA tasks have attracted the interest of researchers (Kafle et al. 2018; Methani et al. 2020; Masry et al. 2022).

Recently, VLMs (Liu et al. 2023b; Dai et al. 2023; Lin et al. 2023; Zhu et al. 2024) have shown significant advancements in general visual capabilities, especially in chart understanding, achieving high scores on the ChartQA dataset (Zhang et al. 2024; Chen et al. 2023a; Meng et al. 2024). However, their real-world performance is notably weaker than their ChartQA test set performance. We conducted a test to illustrate this, as shown in Figure 1, we discarded complex reasoning problems in the ChartQA (Masry et al. 2022) training set and posed 103 basic understanding questions. We then evaluated various VLMs on these questions. The results, presented in the Appendix, show that performance dropped by over 40% compared to ChartQA scores, even for questions on training set charts. This highlights two points: first, current VLMs are capable of answering some chart-reasoning questions, but they lack a comprehensive understanding of charts. Second, the ChartQA dataset allows models to correctly answer questions without a comprehensive understanding of the charts, leading to an overestimation of the capabilities of current models (Wang et al. 2024).

![](images/0892b7cf8c78f7f38672f2cdd6d3d4cd097b61247b7ac58d84b9c8fa74b93fd7.jpg)

<details>
<summary>bar_line</summary>

| Year | Bad | Good |
|------|-----|------|
| 2010 | 66  | 34   |
| 2012 | 67  | 32   |
| 2014 | 61  | 39   |
| 2016 | 71  | 29   |
| 2018 | 54  | 45   |
</details>

Figure 1: Case of Modified ChartQA. “Original” refers to the question from the ChartQA dataset, while “Modified” refers to our modified version.

Firstly, the lack of high-quality chart training data is a major reason why current models lack robust chart understanding capabilities. Existing methods (Masry et al. 2022, 2023; Meng et al. 2024) for collecting chart training data fall into two categories: manual annotation and automatic synthesis. Manually annotated data have real-world chart appearances but suffer from coarse granularity and high human costs. Automatically synthesized data offer fine-grained annotations but lack real-world diversity, leading to poor model robustness. Thus, constructing chart datasets is a challenging bal-

ance between cost and quality, resulting in a scarcity of high-quality chart training data.

Secondly, the single source of charts and the excessive focus on high-level chart reasoning are primary reasons why the ChartQA dataset provides an overly optimistic estimation of VLM chart understanding capabilities. The ChartQA dataset has only four chart sources, focus on politics and economics. Each source of charts with similar styles, making it prone to overfitting. Additionally, datasets like ChartQA focuses heavily on numerical and logical reasoning, this allows the model to potentially answer questions correctly without a clear understanding of the chart. For example, “What is the difference between Bad and Good in 2015?”, the model may not explicitly know the values of “Good” and “Bad” in 2015, but still has the possibility of answering the question accurately.

To address these challenges, we propose a novel method, EvoChart, for synthesizing high-quality chart datasets with real-world characteristics. We also introduce EvoChart-QA, a carefully crafted benchmark for evaluating chart comprehension in real-world scenarios. EvoChart is a multi-stage self-training approach for chart data generation. In each stage, the chart generator produces a batch of synthetic chart data, and the model self-selects and refines the chart data, ensuring that the synthesized data is of high quality for current stage. Subsequently, the model trains on the self-selected data to progress to the next stage. This approach produces both a progressively challenging dataset and a robust chart understanding model. EvoChart-QA is a benchmark designed for basic chart understanding, featuring 650 charts from 140 real-world websites and 1250 expert-curated questions. The diverse chart styles accurately simulate real-world scenarios, with questions focused on chart understanding. Experiments on EvoChart-QA demonstrate that our EvoChart method achieves outstanding performance with 54.2% accuracy, also exhibits leading performance of 81.5% on the ChartQA dataset.

Our main contributions are summarized into three folds:

- We propose EvoChart, a method that combines chart dataset construction with model self-training, using a multi-stage approach to simultaneously output high-quality chart data and a chart understanding model.   
- We propose a novel real-world chart basic understanding benchmark, EvoChart-QA, which comprehensively evaluates a model's chart understanding capability through multi-source real-world charts and multi-type manually curated questions.   
- We conducted extensive experiments on the EvoChart method and EvoChart-QA. Results demonstrate that the EvoChart method significantly outperforms other data synthesis methods, and we also deeply analyze the performance of various VLMs on EvoChart-QA.

# 2 Related Work

# 2.1 Chart Question Answering Datasets

Since FigureQA (Kahou et al. 2018) pioneered the CQA task, numerous datasets for chart question answering have emerged. Synthetic datasets, such as DVQA (Kafle et al. 2018), PlotQA (Methani et al. 2020), RealCQA (Ahmed et al. 2023), ChartX (Xia et al. 2024) and UniChart (Masry et al. 2023). This datasets utilize synthetically generated charts or templat-base questions. Generate datasets such as ChartSFT (Meng et al. 2024), utilize a mixture of GPT-4 (OpenAI et al. 2024)-generated charts and questions. Mixed datasets as ChartQA (Masry et al. 2022) and Charxiv (Wang et al. 2024), the former is a dataset compiled semi-manually with the assistance of templates, while the latter is template-based and requires evaluation by GPT-4o. In contrast, EvoChart-QA focus on real-world scenarios and employ an automated evaluation method that does not necessitate the utilization of GPT-4.

# 2.2 Visual Language Models on CQA

VLMs are language models with visual understanding capabilities, and they have numerous applications in CQA tasks. Small VLMs like ChartReader (Cheng, Dai, and Hauptmann 2023), MatCha (Liu et al. 2023a), ScreenAI (Baechler et al. 2024) and UniChart (Masry et al. 2023) have shown superior performance on tasks like PlotQA and DVQA, highlighting the potential of VLMs in CQA. ChartLlama (Han et al. 2023) was a milestone, being the first to apply LLaVa1.5 (Liu et al. 2024) to CQA tasks and achieving impressive performance. Subsequently, works such as ChartPaLI (Carbune et al. 2024), ChartInstruct (Masry et al. 2024a), ChartAst-D (Meng et al. 2024), and TinyChart (Zhang et al. 2024) delved into the multimodal alignment and CQA reasoning aspects of VLMs in CQA, achieving remarkable performance. Recently, open-source general VLMs such as Phi3-Vision (Abdin et al. 2024) and Intern-VL2.0 (Chen et al. 2023b), through large-scale training, have achieved state-of-the-art performance on the ChartQA dataset.

# 2.3 Self-Training Approach

With the increasing capabilities of language models, numerous researchers have begun to explore the potential of leveraging language models for self-training. GPT3Mix (Yoo et al. 2021) proved that large language model augmentation of textual corpora is very effective. Further, ReST (Gülçehre et al. 2023) achieves cost-effective and efficient human preference alignment through a dual-loop self-training approach. Dennis et al. (Ulmer et al. 2024) obtained new data from the self-talk of multi-role-playing LLM Agents by adding a filtering check mechanism, realizing efficient self-training. Recently, Xu et al. (Xu et al. 2024) proposed ENVISIONS, which uses a neural-symbolic self-training approach to significantly improve mathematical and logical reasoning abilities without relying on external stronger models or evaluation tools. Inspired by their work, our proposed EvoChart focuses on a scalable self-training process.

# 3 EvoChart Method

We introduce the EvoChart, a unique self-training data synthesis approach that simultaneously produces high-quality training corpus and a high-performance chart understanding

![](images/951daa147ab70fa0d23b6a932b46ef01291f93a0044a56eabe83efb639060b35.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Compositional Chart Generation"] --> B["Derived from Chart Code Seed"]
    B --> C["Stage (k-1)"]
    C --> D["QA-pair Generator"]
    D --> E["Corpus Output"]
    E --> F["Train"]
    F --> G["Model (k-1)"]
    G --> H["Model Output"]
    H --> I["Employ as Evaluator"]
    I --> J["Chart Evaluation and Refinement"]
    J --> K["Action: “Add More Legends”"]
    K --> L["Chart Evaluator"]
    L --> M["Action Space"]
    
    subgraph Stage (k-1)
        C
        D
    end
    
    subgraph Stage (k)
        J
        K
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#fcc,stroke:#333
```
</details>

Figure 2: The overview of the proposed EvoChart method. The figure depicts a counterclockwise cyclical self-training process, where the Chart Evaluator of each stage k is trained based on the results of the previous stage k - 1.

model. EvoChart comprises three iterative phases: Compositional Chart Generation for generating charts with diverse appearance, Chart Evaluation and Refinement to select and refine charts suitable for the current stage, and QA-pair generation and training to produce training data and provide a stronger model for the subsequent stage. These phases operate cyclically throughout the construction of EvoChart, as illustrated in Figure 2. We will explore each of these steps in the following subsections.

# 3.1 Compositional Chart Generation

Compositional chart generation aims to produce high-quality and diverse charts at minimal cost, serving as the core component of EvoChart's construction. Previous approaches (Han et al. 2023; Meng et al. 2024; Zhang et al. 2024) either rely solely on GPT-4 for chart generation result in limited diversity and high costs, or using plotting libraries for random generation often leads to unrealistic themes and styles. To achieve continuous, low-cost, and diverse chart creation, we propose a two-step generation strategy within the compositional chart generation process:

1) Chart Code Seed Generation: This step serves as the initial phase of chart generation, aiming to produce fundamental chart code containing elements that are difficult to achieve via random processes. These elements include chart themes and appropriate units for the x- and y-axes. Notably, this step is executed only once during the entire chart dataset construction process. As (Xu et al. 2023) demonstrated the feasibility of large language models for code generation, we employ sophisticated prompt engineering techniques to guide GPT-4 in generating over 25k real-world chart code seeds. Our GPT-4 prompts are based on the following key aspects:

Chart Types: Different chart types are suited for different themes. We focus on the four most prevalent real-world chart types: line charts, bar charts, pie charts, and scatter charts. For each of these chart types, we generate themes that are specifically tailored to their characteristics and use cases.

Chart Themes: Chart themes are challenging to generate through any random process, and manual curation is both costly and prone to domain bias. By utilizing prompting GPT-4, we have generated 25,000 themes across over 200 domains, including politics, economics, technology and everyday life. These themes encompass various titles, units, and other relevant elements.

Chart Color Schemes: Chart color schemes significantly influence the visual appearance of a chart. We employ an automated approach to generate over 200 color palettes, ensuring aesthetically pleasing chart appearances. These color schemes include diverse colors for lines, bars, segments, and backgrounds, among other visual elements.

2) Composable Chart Generator: The Composable Chart Generator is responsible for producing diverse charts. It is invoked multiple times during the construction process. This step automates the creation of a wide variety of charts by randomly assigning configurations to the chart code. We have defined over a hundred configuration options, each with dozens of potential values. The generator automatically selects these options based on the Code Seed, ensuring diverse chart outputs. Due to the numerous configuration options, we will highlight a few key aspects below:

Chart Data: Numerical data constitutes the core message conveyed by a chart. While randomly generated data may result in excessively volatile values and unrealistic visualizations, we ensure that the generated chart data is adhered to the specified ranges provided in the Code Seed. This ensures that data values remain within the reasonable bounds defined by the chosen theme.

Axis Tick Interval: Real-world chart creators often omit some labels on axes. For any continuous axis label (e.g.,

year, month, quarter), we set a 25% probability of no omission, a 50% probability of omitting one out of three labels, and a 25% probability of omitting two out of four labels.

Other Configurations: A multitude of detailed configurations influence chart appearance, including line width, numeric label (position), line style (solid or dashed), bar stacking, axis visibility, font size, font type, and more. We introduce randomness into these configurations through a range of selectable options, and we employ ECharts (Li et al. 2018) for rendering charts.

# 3.2 Chart Evaluation and Refinement

Chart Evaluation and Refinement enhances the chart images generated by the Compositional Chart Generation process. While chart code seeds can produce diverse charts, refinement remains crucial due to the following reasons: 1) Random generation can lead to visually similar charts, causing overfitting and reducing the model's generalization ability. 2) Seed-based chart construction may result in poor chart aesthetics, negatively impacting data quality. To address these issues, we propose two steps: the Chart Evaluator and the Action Space. In the k-th stage of data synthesis (Stage-k), the Chart Evaluator assigns a multi-dimensional evaluation score $e_{k}$ to the charts. Based on the difference $\Delta E$ between $e_{k}$ and the previous score $e_{k-1}$ , the Action Space selects an action to modify the charts. The detailed process is described below.

The Chart Evaluator uses the current stage model to assess chart quality, producing a multi-dimensional evaluation score. To avoid hallucinations from questions like “Does this chart have flaws?”, we assess quality using a directness-based question-answering approach. The Action Space then selects actions to refine the charts based on the evaluation scores. A detailed list of action types is in the Appendix. The evaluation questions and actions are as follows:

Is-Chart & Is-Title-Clear: These questions check if the chart is correctly rendered. While existing VLMs struggle to comprehend charts in detail, they can still distinguish the names of different chart types. Therefore, we propose the following questions. For example, “Is the image a horizontal bar chart?” If the model answers incorrectly, the action is “Drop.” If correct, the action is “None.”

Label-Value & Value-Label: This question type evaluates chart quality by examining text-value alignment. For example, “What is the value of Medication in May?” We generate 10 questions per chart and calculate the average accuracy $e_{k}$ . If $e_{k}-e_{k-1}$ is significantly positive, the chart may be too simple, prompting a “value enhancement method.” If significantly negative, it may indicate errors or overlaps, prompting the “Drop” action.

Label-Visual & Visual-Label: This evaluates visual-text alignment, for example, “What is the bar color of Medication?” We generate 10 questions per chart and calculate accuracy $e_{k}$ . If $e_{k} - e_{k-1}$ is significantly positive, the chart’s visual information may be too simple, prompting a “visual enhancement method.” If significantly negative, it may indicate visual errors, prompting the “Drop” action.

Through Chart Evaluation and Refinement, we ensure that EvoChart generates accurate and challenging data relative to the current stage model in each stage. This ensures data diversity and simultaneously prevents the EvoChart model from overfitting to the EvoChart Corpus.

# 3.3 QA-pairs Generation and Training

QA-pairs Generation and Training aims to generate chart-based question-answer pairs, incorporating these data into the EvoChart corpus and training the EvoChart model for the next stage. We generate question-answer pairs using various question templates. Notably, we focus on basic chart understanding, the templates specifically focus on the alignment of visual-text-value information in charts (e.g., extracting values through visual information, extracting visual information through text). Additionally, we generate rich question-CoT pairs using composable vCoTs (Rose et al. 2024) and distinguish Direct from vCoT using Instruct. Since vCoTs solely serve to enhance model comprehension, we only mix in 20% of vCoT data in the training data. We have established 198 question templates with corresponding answers including Direct and over 500 vCoT templates. We generated 1.6M QA-pairs during the training in 3 stages. A detailed information can be found in the Appendix.

# 4 EvoChart-QA Benchmark

EvoChart-QA is a comprehensive and challenging benchmark for real-world chart understanding. We carefully selected 625 charts with diverse appearances, all sourced from real-world websites. Then we curated 1250 chart-based understanding questions through human experts. This process ensures that EvoChart-QA accurately reflects real-world scenarios. The comparison between EvoChart-QA and other benchmarks is shown in Table 1. In the following sections, we will elaborate on the chart selection process, question construction methods, and evaluation metrics used.

<table><tr><td>Name</td><td>Real Data</td><td>Real Chart</td><td>Open Vocab</td><td>Human Query</td><td>Multi Source</td><td>Flex Eval</td></tr><tr><td>FigureQA</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>DVQA</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>PlotQA</td><td>√</td><td>✕</td><td>√</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>ChartQA</td><td>√</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>✕</td></tr><tr><td>CharXiv</td><td>√</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>✕</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison with different benchmarks

# 4.1 Chart Selection

To enable EvoChart-QA to emulate real-world chart understanding scenarios, all charts in our dataset are carefully selected by human experts. Specifically, we crawled 1,000 charts from 140 different websites. Human experts then filtered out images with ambiguous meanings or damages, resulting in a final dataset of 625 valid images. These images include line charts, bar charts, pie charts, and scatter plots. Examples and sources of all images are detailed in the Appendix.

# 4.2 Question Construction

We focus on chart basic understanding questions. Following the definitions of prior researchers (Kafle et al. 2020; Methani et al. 2020), we concentrate on data and structural retrieval questions. Specifically, we categorize the problems into two types during manual construction: Direct Retrieval and Complex Retrieval. Direct Retrieval questions focus on understanding the image and directly extracting its content, while Complex Retrieval questions emphasize performing multiple visual reasoning steps on the chart.

1) Direct Retrieval. Direct Retrieval aims to directly extract elements from a chart based on the question. To comprehensively assess the model's ability to extract various elements from charts, we categorize chart elements into three types: label, value, and visual. Label elements refer to textual content in the chart, such as chart title, axis labels, etc. Value elements refer to data values conveyed by the chart, which are displayed or implicitly provided based on the chart author's intention. Visual elements refer to all visual descriptions in the chart, such as line color, largest segment, etc. For example: "What is the value of the green dashed line in 2015?" Although Direct Retrieval involves only extracting elements from charts, it remains a challenging task. On the one hand, real-world charts often exhibit non-standard variations. For example, they may include rich text such as numerous logos inserted in the image, or they may combine multiple chart forms within a single chart to facilitate expressions. On the other hand, questions posed by real-world users may contain ambiguous expressions. For example, "Which country's total GDP is represented by the bar in the lightest shade of blue?" This visual description is ambiguous, but it is a clearly question for human observers.

2) Complex Retrieval. Complex Retrieval involves querying information within a chart using complex, multi-step descriptions. Compared to Direct Retrieval, Complex Retrieval focuses on understanding the relative positions of elements within the chart. For example, “In the chart, the third bar to the left of the longest red bar represents the GDP of which country?”. Complex retrieval poses novel challenges for chart comprehension. This is because the descriptive information in complex retrieval relies entirely on the chart itself, requiring the model to have a comprehensive and clear understanding of the chart. For example, comprehending the previously mentioned “the longest red bar” is entirely based on the extraction of information from the chart’s content. Furthermore, this also necessitates the model to possess sophisticated visual reasoning capabilities, such as understanding “the third bar from the left” which demands visually-grounded inference.

# 4.3 Evaluation Metrics

We designed a automatic evaluation method for EvoChart-QA, combining flex and strict approaches, to fairly evaluate answer correctness. In EvoChart-QA construction, we label questions as “Strict” or “Flex” and use the corresponding Strict or Flex approach to evaluate correctness. For “Strict” type questions, answers have a definite value, such as numerical or textual values explicitly labeled in the chart. We employ a zero-tolerance approach for judging these questions. For “Flex” type questions, answers have estimated values, such as unlabeled numerical values. We employ a 5% tolerance approach to judge these questions. Finally, we employ average accuracy to evaluate the model’s performance. In contrast to our metrics, previous methods like ChartQA allowed a 5% tolerance for any numerical answer, leading to an optimistic estimation of model outputs. For example, years are numerical answers, and 1995 and 2008 would fall within the 5% tolerance in previous evaluation metrics, and our method does not exhibit this error.

# 5 Experiments

# 5.1 Setup

Datasets. To comprehensively evaluate the effectiveness of EvoChart, we chose to test it on both ChartQA and EvoChart-QA. ChartQA (Masry et al. 2022) is a dataset with two subsets: “Augment”, which is machine-generated, and “Human”, which is manually curated. “Augment” focuses on element extraction tasks within machine-synthesized images, while “Human” emphasizes complex numerical and logical reasoning tasks in real-world charts. EvoChart-QA is a novel real-world benchmark that we proposed.

Models. We conducted extensive evaluations on both open-source and proprietary models. For open-source models, we tested Phi3-Vision-4B (Abdin et al. 2024), QwenVL-Chat-7B (Bai et al. 2023), LlaVa1.6-Vicuna-7B (Liu et al. 2024), Intern-VL-2.0-8B (Chen et al. 2023b), Llama3-Llava-Next-8B (Li et al. 2024), CogVLM2-19B (Wang et al. 2023), LlaVa1.6-YI-34B, Intern-VL-2.0-40B, ChartLlama-13B (Han et al. 2023), ChartAst-S-13B (Meng et al. 2024), ChartIns-Llama2-7B (Masry et al. 2024a), ChartIns-FlanT5-3B, ChartGemma-2B (Masry et al. 2024b), and TinyChart-3B (Zhang et al. 2024). For proprietary models, we tested Gemini-1.5-Flash (Team et al. 2024), Gemini-1.5-Pro, Qwen-VL-Plus, Qwen-VL-Max, GPT-4-turbo (OpenAI et al. 2024), and GPT-4o. For all models, we employed a zero-shot approach. The specific configurations of all models are provided in the Appendix.

Settings. In EvoChart method, we utilize Phi3-Vision (Abdin et al. 2024) as the initialization model. We conducted a 3-Stage data synthesis and training process, with each Stage undergoing 1 Epoch of SFT with a learning rate of 2e-5 and using cosine learning rate scheduler. All experiments were completed on 4 NVIDIA A800 80G GPUs.

# 5.2 Experimental Results

Tables 2 and 3 present the performance of EvoChart and other open-source or proprietary VLMs on EvoChart-QA and ChartQA. We elaborate on the experimental results in two aspects: EvoChart-QA and EvoChart.

EvoChart-QA Results. All models exhibit relatively poor performance on EvoChart-QA, with accuracies not exceeding 55%. Among proprietary models, GPT-4o achieves the highest accuracy at 49.8%. InternVL-2.0-40B demonstrates the strongest performance among open-source general-purpose models, reaching 49.0%. Within the domain of chart-expert models, our proposed EvoChart method yields the best-performing model, achieving the highest accuracy

<table><tr><td rowspan="2">Model</td><td colspan="3">Line</td><td colspan="3">Bar</td><td colspan="3">Pie</td><td colspan="3">Scatter</td><td colspan="3">Overall</td></tr><tr><td>Dir.</td><td>Comp.</td><td>All</td><td>Dir.</td><td>Comp.</td><td>All</td><td>Dir.</td><td>Comp.</td><td>All</td><td>Dir.</td><td>Comp.</td><td>All</td><td>Dir.</td><td>Comp.</td><td>All</td></tr><tr><td colspan="16">Proprietary Models</td></tr><tr><td>Gemini-1.5-Flash</td><td>26.7</td><td>17.1</td><td>25.0</td><td>28.4</td><td>20.7</td><td>26.8</td><td>41.9</td><td>22.9</td><td>33.8</td><td>33.5</td><td>19.4</td><td>29.3</td><td>30.5</td><td>20.3</td><td>27.9</td></tr><tr><td>Gemini-1.5-Pro</td><td>42.1</td><td>21.4</td><td>38.5</td><td>28.4</td><td>20.7</td><td>26.8</td><td>41.9</td><td>22.9</td><td>33.8</td><td>33.5</td><td>19.4</td><td>29.3</td><td>36.0</td><td>21.2</td><td>32.2</td></tr><tr><td>Qwen-VL-Plus</td><td>22.7</td><td>11.4</td><td>20.8</td><td>28.8</td><td>13.8</td><td>25.5</td><td>33.3</td><td>9.4</td><td>23.1</td><td>27.2</td><td>13.4</td><td>23.1</td><td>27.0</td><td>11.9</td><td>23.1</td></tr><tr><td>Qwen-VL-Max</td><td>35.5</td><td>17.1</td><td>32.2</td><td>44.4</td><td>23.0</td><td>39.8</td><td>48.1</td><td>25.0</td><td>38.2</td><td>33.5</td><td>19.4</td><td>29.3</td><td>39.9</td><td>21.6</td><td>35.2</td></tr><tr><td>GPT-4-turbo</td><td>40.0</td><td>25.7</td><td>37.5</td><td>44.7</td><td>29.9</td><td>41.5</td><td>55.0</td><td>34.4</td><td>46.2</td><td>46.8</td><td>14.9</td><td>37.3</td><td>44.8</td><td>27.2</td><td>40.3</td></tr><tr><td>GPT-4o</td><td>52.7</td><td>32.9</td><td>49.2</td><td>52.7</td><td>44.8</td><td>51.0</td><td>53.5</td><td>49.0</td><td>51.6</td><td>56.3</td><td>23.9</td><td>46.7</td><td>53.4</td><td>39.1</td><td>49.8</td></tr><tr><td colspan="16">Open-source Models</td></tr><tr><td>Phi3-Vision-4B</td><td>43.3</td><td>27.1</td><td>40.5</td><td>47.9</td><td>27.6</td><td>43.5</td><td>33.3</td><td>27.1</td><td>30.7</td><td>50.0</td><td>14.9</td><td>39.6</td><td>44.6</td><td>24.7</td><td>39.5</td></tr><tr><td>QwenVL-Chat-7B</td><td>20.6</td><td>17.1</td><td>20.0</td><td>18.2</td><td>9.2</td><td>16.2</td><td>31.0</td><td>14.6</td><td>24.0</td><td>24.7</td><td>11.9</td><td>20.9</td><td>21.9</td><td>13.1</td><td>19.7</td></tr><tr><td>LlaVa1.6-Vicuna-7B</td><td>25.8</td><td>14.3</td><td>23.8</td><td>24.9</td><td>19.5</td><td>23.8</td><td>38.0</td><td>15.6</td><td>28.4</td><td>21.5</td><td>11.9</td><td>18.7</td><td>26.5</td><td>15.6</td><td>23.7</td></tr><tr><td>Intern-VL-2.0-8B</td><td>38.5</td><td>27.1</td><td>36.5</td><td>45.7</td><td>29.9</td><td>42.2</td><td>43.4</td><td>27.1</td><td>36.4</td><td>44.9</td><td>22.4</td><td>38.2</td><td>42.7</td><td>26.9</td><td>38.6</td></tr><tr><td>Llama3-Next-8B</td><td>20.3</td><td>5.7</td><td>17.8</td><td>22.4</td><td>16.1</td><td>21.0</td><td>24.0</td><td>21.9</td><td>23.1</td><td>20.9</td><td>16.4</td><td>19.6</td><td>21.6</td><td>15.6</td><td>20.1</td></tr><tr><td>CogVLM2-19B</td><td>24.8</td><td>11.4</td><td>22.5</td><td>28.8</td><td>10.3</td><td>24.8</td><td>27.9</td><td>5.2</td><td>18.2</td><td>24.7</td><td>7.5</td><td>19.6</td><td>26.6</td><td>8.4</td><td>21.9</td></tr><tr><td>LlaVa1.6-YI-34B</td><td>5.8</td><td>10.0</td><td>6.5</td><td>7.7</td><td>4.6</td><td>7.0</td><td>13.2</td><td>5.2</td><td>9.8</td><td>9.5</td><td>6.0</td><td>8.4</td><td>8.1</td><td>6.2</td><td>7.6</td></tr><tr><td>Intern-VL-2.0-40B</td><td>53.3</td><td>42.9</td><td>51.5</td><td>54.3</td><td>37.9</td><td>50.7</td><td>55.8</td><td>37.5</td><td>48.0</td><td>51.9</td><td>20.9</td><td>42.7</td><td>53.8</td><td>35.3</td><td>49.0</td></tr><tr><td colspan="16">Chart Expert Models</td></tr><tr><td>ChartLlama-13B</td><td>7.3</td><td>4.3</td><td>6.8</td><td>7.3</td><td>10.3</td><td>8.0</td><td>21.7</td><td>6.2</td><td>15.1</td><td>13.9</td><td>6.0</td><td>11.6</td><td>10.4</td><td>6.9</td><td>9.5</td></tr><tr><td>ChartAst-S-13B</td><td>12.4</td><td>12.9</td><td>12.5</td><td>14.4</td><td>14.9</td><td>14.5</td><td>14.7</td><td>7.3</td><td>11.6</td><td>13.9</td><td>12.0</td><td>12.0</td><td>13.7</td><td>10.6</td><td>12.9</td></tr><tr><td>ChartIns-Llama2-7B</td><td>17.9</td><td>11.4</td><td>16.8</td><td>16.0</td><td>19.5</td><td>16.8</td><td>27.1</td><td>13.5</td><td>21.3</td><td>13.9</td><td>9.0</td><td>12.4</td><td>17.8</td><td>13.8</td><td>16.8</td></tr><tr><td>ChartIns-FlanT5-3B</td><td>23.6</td><td>24.3</td><td>23.8</td><td>28.4</td><td>16.1</td><td>25.8</td><td>40.3</td><td>19.8</td><td>31.6</td><td>13.9</td><td>19.4</td><td>15.6</td><td>25.9</td><td>19.7</td><td>24.3</td></tr><tr><td>ChartGemma-2B</td><td>33.9</td><td>25.7</td><td>32.5</td><td>29.1</td><td>25.3</td><td>28.2</td><td>36.4</td><td>30.2</td><td>33.8</td><td>32.9</td><td>16.4</td><td>28.0</td><td>32.5</td><td>25.0</td><td>30.6</td></tr><tr><td>TinyChart-3B</td><td>24.5</td><td>15.7</td><td>23.0</td><td>28.4</td><td>17.2</td><td>26.0</td><td>33.3</td><td>15.6</td><td>25.8</td><td>33.5</td><td>17.9</td><td>28.9</td><td>28.6</td><td>16.6</td><td>25.5</td></tr><tr><td>EvoChart-4B</td><td>62.1</td><td>32.9</td><td>57.0</td><td>62.3</td><td>33.3</td><td>56.0</td><td>64.3</td><td>30.2</td><td>49.8</td><td>55.1</td><td>37.3</td><td>49.8</td><td>61.3</td><td>33.1</td><td>54.2</td></tr></table>

Table 2: Experimental results on EvoChart-QA using various open-source or proprietary models. Due to space constraints, abbreviations are used: Dir. refers to Direct, Comp. refers to Complex.

across all models at 54.2%. We summarize our findings as follows:

1) EvoChart-QA presents a substantially more challenging benchmark for evaluating basic chart comprehension. Even without involving numerical reasoning or calculation, these models experience a significant performance drop of 30-50% on EvoChart-QA compared to their scores on ChartQA. This challenge stems from the diversity of charts sourced from 140 websites and the meticulously crafted questions that comprise our dataset.

2) All models demonstrate significantly weaker performance on Complex Retrieval compared to Direct Retrieval, indicating that reasoning over visual information poses a substantially greater challenge than direct extraction of information from charts. Furthermore, nearly all models exhibit lower accuracy on Pie and Scatter chart types compared to their average performance. This suggests that Pie and Scatter charts pose a greater challenge.

3) Open-source general-purpose models exhibit comparable chart comprehension abilities to proprietary models. This suggests that models pretrained on large-scale chart data possess strong generalization capabilities. However, while chart expert models fine-tuned on specific domains achieve impressive scores on the ChartQA dataset, their performance degrades significantly to below 30% when confronted with the entirely OOD EvoChart-QA dataset.

EvoChart Results. Among all proprietary and open-source

<table><tr><td>Model</td><td>ChartQA</td><td>EvoChart-QA</td></tr><tr><td>Gemini-1.5-pro</td><td>81.3</td><td>32.2</td></tr><tr><td>GPT-4-turbo</td><td>62.3</td><td>40.3</td></tr><tr><td>GPT-4o</td><td>85.7</td><td>49.8</td></tr><tr><td>CogVLM2-19B</td><td>81.0</td><td>21.9</td></tr><tr><td>Phi3-Vision-4B</td><td>81.4</td><td>38.6</td></tr><tr><td>Intern-VL-2.0-8B</td><td>81.5</td><td>49.0</td></tr><tr><td>ChartAst-S-13B</td><td>79.9</td><td>12.9</td></tr><tr><td>TinyChart-3B</td><td>83.6</td><td>25.5</td></tr><tr><td>EvoChart-4B</td><td>81.5</td><td>54.2</td></tr></table>

Table 3: Comparison on ChartQA and EvoChart-QA

models evaluated, our proposed EvoChart trained model exhibits significantly superior performance, achieving an accuracy of 54.2%, surpassing GPT-4o 49.8%. We have the following observations:

1) Although the EvoChart model is trained on synthetic data, it achieves SoTA performance on the entirely real-world benchmark EvoChart-QA and exhibits competitive performance on the chart reasoning task ChartQA. This validates the strong generalization ability of the EvoChart.   
2) EvoChart primarily focuses on chart basic comprehension. However, as demonstrated in Table 3, EvoChart remains one of the top-performing chart expert models on

![](images/6a6c3d6a1600da3d5fb1f4d30e4604106b09fa93c073a56bb970d4aba8536d64.jpg)

<details>
<summary>line</summary>

| Year | Military Spending ($B) | Income Inequality (Gini coefficient) |
|------|------------------------|-------------------------------------|
| 1992 | ~500                   | -                                   |
| 1995 | ~600                   | -                                   |
| 2000 | ~700                   | -                                   |
| 2005 | ~800                   | -                                   |
| 2010 | ~900                   | -                                   |
| 2015 | ~1000                  | -                                   |
| 2020 | ~1100                  | -                                   |
| 2023 | ~1200                  | -                                   |
</details>

Figure 3: Four cases from the EvoChart-QA Benchmark. Q1 and Q2 are line charts, Q3 is a scatter chart, and Q4 is a pie chart.

ChartQA. This is an intriguing finding, suggesting that basic chart comprehension serves as a cornerstone for chart reasoning tasks, and training on basic comprehension can enhance performance in chart reasoning tasks.

3) EvoChart's complex retrieval capabilities are inferior to those of InternVL-2.0-40B and GPT4o. This is reasonable, as these models possess significantly larger scales, which confer an inherent advantage in complex visual extraction and reasoning tasks.

# 5.3 Analysis

EvoChart Ablation Study. We conducted comprehensive ablation studies on EvoChart, and the results are summarized in Table 4. We trained and generated EvoChart for 1 to 3 stages. Meanwhile, to verify the effectiveness of Chart Evaluation and Refinement, we set up EvoChart without refinement and trained it for 3 stages, denoted as “w/o refine stage-3.” We observed the following:

1) As the number of EvoChart method Stages increases, the scale of the EvoChart-Dataset expands, and the performance of the EvoChart steadily improves. This highlights EvoChart's effectiveness as a self-training approach.   
2) Despite having access to a larger training dataset, the “w/o refine stage-3.” exhibits significantly lower performance on EvoChart-QA compared to the complete EvoChart method. This indicates the effectiveness of Chart Evaluation and Refinement in enhancing the model’s generalization ability within the EvoChart.

Case Study. To further analyze EvoChart and EvoChart-QA, we selected samples from the EvoChart-QA Benchmark for analysis. Figure 3 presents four cases. More cases are provided in the Appendix. As shown in the figure, overall, EvoChart-QA offers diverse charts and questions, and EvoChart achieves more accurate chart understanding performance compared to GPT4o. Q2 and Q4 demonstrate the effectiveness of our Strict/Flex Metrics. For values explicitly labeled in the image, there should be zero tolerance. However, for estimation questions like Q1, a 5% tolerance is al-

<table><tr><td>Model</td><td>Line</td><td>Bar</td><td>Pie</td><td>Scatter</td><td>Overall</td></tr><tr><td>w/ refine stage-1</td><td>53.2</td><td>51.5</td><td>46.7</td><td>44.9</td><td>50.0</td></tr><tr><td>w/ refine stage-2</td><td>53.5</td><td>54.2</td><td>49.8</td><td>48.0</td><td>52.0</td></tr><tr><td>w/ refine stage-3</td><td>57.0</td><td>56.0</td><td>49.8</td><td>49.8</td><td>54.2</td></tr><tr><td>w/o refine stage-3</td><td>52.5</td><td>50.7</td><td>43.6</td><td>45.3</td><td>49.0</td></tr></table>

Table 4: Ablation Study Results for EvoChart on EvoChart-QA

lowed. Furthermore, Q2 highlights EvoChart's capability for precise chart understanding in complex scenarios.

# 6 Conclusion

In this paper, we introduce EvoChart and EvoChart-QA: a novel approach for enhancing chart comprehension capabilities through self-training and iterative synthetic data generation, and a meticulously crafted real-world chart comprehension benchmark. We aim to provide a new avenue for real-world chart understanding through EvoChart and EvoChart-QA. Through extensive experimentation, we expose the limitations of existing VLMs in chart comprehension and validate the effectiveness of our EvoChart method across multiple datasets. In the future, we will further explore human-free methods in chart comprehension.

# Acknowledgments

This work was supported by National Key Research and Development Program of China (2022YFC3303600), the Key Research and Development Project in Shaanxi Province No. 2022GXLH-01-03, National Natural Science Foundation of China (No. 62137002, 62293553, 62293554, 62450005, 62477036, 62293550, and 62306229), the Shaanxi Provincial Social Science Foundation Project (No. 2024P041), the Natural Science Basic Research Program of Shaanxi (No. 2023-JC-YB-593), the Youth Talent Support Program of Shaanxi Science and Technology Association (20240113), the China Postdoctoral Science Foundation (2024M752585).

# References

Abdin, M.; Jacobs, S. A.; Awan, A. A.; Aneja, J.; and et al., A. A. 2024. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219.

Ahmed, S.; Jawade, B.; Pandey, S.; Setlur, S.; and Govindaraju, V. 2023. RealCQA: Scientific Chart Question Answering as a Test-Bed for First-Order Logic. In ICDAR, 14189: 66–83.

Baechler, G.; Sunkara, S.; Wang, M.; Zubach, F.; Mansoor, H.; Etter, V.; Carbune, V.; Lin, J.; Chen, J.; and Sharma, A. 2024. ScreenAI: A Vision-Language Model for UI and Infographics Understanding. In IJCAI, 3058–3068.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv:2308.12966.

Carbune, V.; Mansoor, H.; Liu, F.; Aralikatte, R.; Baechler, G.; Chen, J.; and Sharma, A. 2024. Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs. In Duh, K.; Gómez-Adorno, H.; and Bethard, S., eds., Findings of NAACL, 989–1004.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; Li, B.; Luo, P.; Lu, T.; Qiao, Y.; and Dai, J. 2023a. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv preprint arXiv:2312.14238.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; Li, B.; Luo, P.; Lu, T.; and Qiao, Y. e. a. 2023b. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv preprint arXiv:2312.14238.

Cheng, Z.; Dai, Q.; and Hauptmann, A. G. 2023. ChartReader: A Unified Framework for Chart Derendering and Comprehension without Heuristic Rules. In ICCV, 22145–22156.

Dai, W.; Li, J.; Li, D.; Tiong, A. M. H.; Zhao, J.; Wang, W.; Li, B.; Fung, P.; and Hoi, S. C. H. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In NeurIPS.

Gülçehre, Ç.; Paine, T. L.; Srinivasan, S.; Konyushkova, K.; Weerts, L.; Sharma, A.; Siddhant, A.; Ahern, A.; Wang, M.; Gu, C.; Macherey, W.; Doucet, A.; Firat, O.; and de Freitas, N. 2023. Reinforced Self-Training (ReST) for Language Modeling. arXiv preprint arXiv:2308.08998.

Han, Y.; Zhang, C.; Chen, X.; Yang, X.; Wang, Z.; Yu, G.; Fu, B.; and Zhang, H. 2023. ChartLlama: A Multimodal LLM for Chart Understanding and Generation. arXiv preprint arXiv:2311.16483.

Kafle, K.; Price, B. L.; Cohen, S.; and Kanan, C. 2018. DVQA: Understanding Data Visualizations via Question Answering. In CVPR, 5648–5656.

Kafle, K.; Shrestha, R.; Price, B. L.; Cohen, S.; and Kanan, C. 2020. Answering Questions about Data Visualizations using Efficient Bimodal Fusion. In WACV, 1487–1496.

Kahou, S. E.; Michalski, V.; Atkinson, A.; Kádár, Á.; Trischler, A.; and Bengio, Y. 2018. FigureQA: An Annotated Figure Dataset for Visual Reasoning. In ICLR.

Li, D.; Mei, H.; Shen, Y.; Su, S.; Zhang, W.; Wang, J.; Zu, M.; and Chen, W. 2018. ECharts: A declarative framework for rapid construction of web-based visualization. VI, 2(2):136–146.

Li, F.; Zhang, R.; Zhang, H.; Zhang, Y.; Li, B.; Li, W.; Ma, Z.; and Li, C. 2024. LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models. arXiv preprint arXiv:2407.07895.

Lin, Z.; Liu, C.; Zhang, R.; Gao, P.; Qiu, L.; Xiao, H.; Qiu, H.; Lin, C.; Shao, W.; Chen, K.; Han, J.; Huang, S.; Zhang, Y.; He, X.; Li, H.; and Qiao, Y. 2023. SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models. arXiv preprint arXiv:2311.07575.

Liu, F.; Piccinno, F.; Krichene, S.; Pang, C.; Lee, K.; Joshi, M.; Altun, Y.; Collier, N.; and Eisenschlos, J. M. 2023a. MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering. In ACL, 12756–12770.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024. Improved baselines with visual instruction tuning. In CVPR, 26296–26306.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023b. Visual Instruction Tuning. In NeurIPS.

Masry, A.; Kavehzadeh, P.; Long, D. X.; Hoque, E.; and Joty, S. 2023. UniChart: A Universal Vision-language Pre-trained Model for Chart Comprehension and Reasoning. In EMNLP, 14662–14684.

Masry, A.; Long, D. X.; Tan, J. Q.; Joty, S. R.; and Hoque, E. 2022. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings of ACL, 2263–2279.

Masry, A.; Shahmohammadi, M.; Parvez, M. R.; Hoque, E.; and Joty, S. 2024a. ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning. arXiv preprint arXiv:2403.09028.

Masry, A.; Thakkar, M.; Bajaj, A.; Kartha, A.; Hoque, E.; and Joty, S. 2024b. ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild. arXiv:2407.04172.

Meng, F.; Shao, W.; Lu, Q.; Gao, P.; Zhang, K.; Qiao, Y.; and Luo, P. 2024. ChartAssistant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning. arXiv preprint arXiv:2401.02384.

Methani, N.; Ganguly, P.; Khapra, M. M.; and Kumar, P. 2020. PlotQA: Reasoning over Scientific Plots. In WACV, 1516–1525.

OpenAI; Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; and et al., F. L. A. 2024. GPT-4 Technical Report. arXiv:2303.08774.

Rose, D.; Himakunthala, V.; Ouyang, A.; He, R.; Mei, A.; Lu, Y.; Saxon, M.; Sonar, C.; Mirza, D.; and Wang, W. Y. 2024. Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings. arXiv:2305.02317.

Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.; and et al., S. M. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530.

Ulmer, D.; Mansimov, E.; Lin, K.; Sun, J.; Gao, X.; and Zhang, Y. 2024. Bootstrapping LLM-based Task-Oriented Dialogue Agents via Self-Talk. arXiv:2401.05033.

Wang, W.; Lv, Q.; Yu, W.; Hong, W.; Qi, J.; Wang, Y.; Ji, J.; Yang, Z.; Zhao, L.; Song, X.; Xu, J.; Xu, B.; Li, J.; Dong, Y.; Ding, M.; and Tang, J. 2023. CogVLM: Visual Expert for Pretrained Language Models. arXiv:2311.03079.

Wang, Z.; Xia, M.; He, L.; Chen, H.; Liu, Y.; Zhu, R.; Liang, K.; Wu, X.; Liu, H.; Malladi, S.; Chevalier, A.; Arora, S.; and Chen, D. 2024. CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs. arXiv preprint arXiv:2406.18521.

Xia, R.; Zhang, B.; Ye, H.; Yan, X.; Liu, Q.; Zhou, H.; Chen, Z.; Dou, M.; Shi, B.; Yan, J.; and Qiao, Y. 2024. ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning. arXiv preprint arXiv:2402.12185.

Xu, F.; Sun, Q.; Cheng, K.; Liu, J.; Qiao, Y.; and Wu, Z. 2024. Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models. arXiv preprint arXiv:2406.11736.

Xu, F.; Wu, Z.; Sun, Q.; Ren, S.; Yuan, F.; Yuan, S.; Lin, Q.; Qiao, Y.; and Liu, J. 2023. Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models. arXiv preprint arXiv:2311.09278.

Yoo, K. M.; Park, D.; Kang, J.; Lee, S.; and Park, W. 2021. GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation. In Findings of EMNLP, 2225–2239.

Zhang, L.; Hu, A.; Xu, H.; Yan, M.; Xu, Y.; Jin, Q.; Zhang, J.; and Huang, F. 2024. TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning. arXiv preprint arXiv: 2404.16635.

Zhu, W.; Agarwal, A.; Joshi, M.; Jia, R.; Thomason, J.; and Toutanova, K. 2024. Efficient End-to-End Visual Document Understanding with Rationale Distillation. In Duh, K.; Gómez-Adorno, H.; and Bethard, S., eds., In NAACL, 8401–8424.

# Appendix

# Modified ChartQA

1) Result of Modified ChartQA. The experimental results on Modified ChartQA using three VLMs, as mentioned in the Introduction, are presented in Table 5. Even the best-performing Chart Expert model exhibits a performance drop of nearly 30%.

<table><tr><td>Model</td><td>ChartQA-Avg</td><td>ChartQA-Modified</td></tr><tr><td>Gemini-1.5-pro</td><td>81.3</td><td>18.1 ↓ 63.2</td></tr><tr><td>GPT-4o</td><td>85.7</td><td>45.8 ↓ 39.9</td></tr><tr><td>Phi-3-Vision</td><td>81.4</td><td>52.6 ↓ 28.8</td></tr><tr><td>TinyChart</td><td>83.6</td><td>45.8 ↓ 37.8</td></tr></table>

Table 5: Comparison results (%) on modified CharQA

2) Case of Modified ChartQA. Examples of the Modified ChartQA cases, as mentioned in the Introduction, are illustrated in Figure 8 and Figure 9. The complete set of 103 questions will be released upon publication of this article.

# EvoChart-QA Detailed Case

An overview of the EvoChart-QA Benchmark is shown in Figure 4. Compared to Charxiv, PlotQA, DVQA, and ChartQA, our proposed EvoChart-QA Benchmark have 140 sources and 1250 manually annotated questions, providing a more realistic evaluation benchmark. This broader scope and meticulous annotation contribute to a more comprehensive and robust assessment of chart comprehension capabilities. Figures 5, 6, and 7 present more detailed examples from the EvoChart-QA dataset.

# EvoChart Action Space Types

Table 6 presents the categories of actions within EvoChart. The three types correspond to the question types mentioned in the main text. Specifically, VaEM represents Value Enhancement Method, and ViEM represents Visual Enhancement Method. For each question, there is a possibility of dropping its corresponding chart. For questions where VaEM or ViEM is selected, a random action will be chosen to modify the corresponding chart.

# Detailed Experiments settings

For all open-source general-purpose VLMs, proprietary VLMs, and chart expert models, we employ a zero-shot prompting approach.

For open-source general-purpose and proprietary models, we use the following prompt: “You will play as a chart reading expert. You should ONLY give the answer STRING or NUMBER, without any units. You should Not Give Any Explanation.” This is because general-purpose models have undergone extensive instruction fine-tuning and alignment with human preferences, thus requiring a more detailed prompt to regulate their output.

For chart expert models, we utilize their respective training instructions. For the EvoChart model, the prompt is as follows: “You will play as a chart reading expert. You should just give the answer, without any explanation or units.” This is because chart expert models have been fine-tuned with specific instructions during their training process.

# EvoChart Method Question Template

We have established 284 distinct QA-Pair Templates, as outlined below:

- Can you tell me the value of {legend\_label} in {xlabel}?   
- I'd like to know the value of {legend\_label} within {xlabel}.   
- Could you provide the value of {legend\_label} found in {xlabel}?   
- What amount does {legend\_label} have in {xlabel}?   
- Please specify the value of {legend\_label} in the context of {xlabel}.   
- What is the value of {legend\_label} in {xlabel}?   
- Can you identify the legend label with a value of {value\_label} at the position marked by {xlabel}?

![](images/263ac7c13df8f1b44bc5d7b994fde1f112032c41909ca929bcf467a51a4ac17c.jpg)  
EvoChart-QA Benchmark

![](images/ae674a5566af4754d7c653ddaf90ecbee686aff6baee5b1864e0731120327c23.jpg)

<details>
<summary>bar</summary>

| Category | Bar Value 1 | Bar Value 2 | Bar Value 3 | Bar Value 4 | Bar Value 5 | Bar Value 6 | Bar Value 7 | Bar Value 8 | Bar Value 9 | Bar Value 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Revenue Growth (Q1) | 1.2 | 1.5 | 1.3 | 1.6 | 1.4 | 1.7 | 1.5 | 1.8 | 1.6 | 1.9 |
| Sales Growth (Q2) | 0.8 | 1.0 | 0.9 | 1.2 | 1.1 | 1.4 | 1.3 | 1.6 | 1.5 | 1.8 |
| Marketing Revenue (Q3) | 0.5 | 0.7 | 0.6 | 0.9 | 0.8 | 1.1 | 1.0 | 1.3 | 1.2 | 1.5 |
| Sales Revenue Growth (Q4) | -0.2 | -0.4 | -0.3 | -0.6 | -0.5 | -0.9 | -0.8 | -1.2 | -1.1 | -1.4 |
| Profitability (Q1) = Sales Revenue Growth (Q1) + Net Sales Growth (Q2) = Net Sales Growth (Q3) + Net Sales Growth (Q4) = Net Sales Growth (Q1) * Net Sales Growth (Q2) * Net Sales Growth (Q3) * Net Sales Growth (Q4) * Net Sales Growth (Q1) * Net Sales Growth (Q2) * Net Sales Growth (Q3) * Net Sales Growth (Q4) * Net Sales Growth (Q1) * Net Sales Growth (Q2) * Net Sales Growth (Q3) * Net Sales Growth (Q4) * Net Sales Growth (Q1) * Net Sales Growth (Q2) * Net Sales Growth (Q3) * Net Sales Growth (Q4) * Net Sales Growth |
| Expenses Growth (Q1) = Expenses Revenue Growth (Q2) + Net Expenses Revenue Growth (Q3) + Net Expenses Revenue Growth (Q4) + Net Expenses Revenue Growth (Q1) * Net Expenses Revenue Growth (Q2) * Net Expenses Revenue Growth (Q3) * Net Expenses Revenue Growth (Q4) * Net Expenses Revenue Growth (Q1) * Net Expenses Revenue Growth (Q2) * Net Expenses Revenue Growth (Q3) * Net Expenses Revenue Growth (Q4) * Net Expenses Revenue Growth (Q1) * Net Expenses Revenue Growth (Q2) * Net Expenses Revenue Growth (Q3) * Net Expenses Revenue Growth (Q4) * Net Expenses Revenue Growth (Q1) * Net Expenses Revenue Growth (Q2) * Net Expenses Revenue Growth (Q3) *Net Expenses Revenue Growth (Q4) *Net Expenses Revenue Growth (Q1) *Net Expenses Revenue Growth (Q2) *Net Expenses Revenue Growth (Q3) *Net Expenses Revenue Growth (Q4) *Net Expenses Revenue Growth (Q1) *Net Expenses Revenue Growth (Q2) *Net Expenses Revenue Growth (Q3) *Net Expenses Revenue Growth (Q4) *Net Expenses Revenue Growth (Q1) *Net Expenses Revenue Growth (Q2) *Net Expenses Revenue Growth (Q3) *Net Expenses Revenue Growth (Q4) *Net Expenses Revenue Growth (Q1) *Net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenue growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth from Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q2 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to Q3 and Q4; net expenses revenues growth in Q1 to qtr of year-end: Net expenses revenue growth from $0.8M in Q2 to $0.9M in Q3, $0.9M in Q4, $0.9M in Q5, $0.9M in Q6, $0.9M in Q7, $0.9M in Q8, $0.9M in Q9, $0.9M in Q10, $0.9M in Q11, $0.9M in Q12, $0.9M in Q13, $0.9M in Q14, $0.9M in Q15, $0.9M in Q16, $0.9M in Q17, $0.9M in Q18, $0.9M in Q19, $0.9M in Q20, $0.9M in Q21, $0.9M in Q22, $0.9M in Q23, $0.9M in Q24, $0.9M in Q25, $0.9M in Q26, $0.9M in Q27, $0.9M in Q28, $0.9M in Q29, $0.9M in Q30, $0.9M in Q31, $0.9M in Q32, $0.9M in Q33, $0.9M in Q34, $0.9M in Q35, $0.9M in Q36, $0.9M in Q37, $0.9M in Q38, $0.9M in Q39, $0.9M in Q40, $0.9M in Q41, $0.9M in Q42, $0.9M in Q43, $0.9M in Q44, $0.9M in Q45, $0.9M in Q46, $0.9M in Q47, $0.9M in Q48, $0.9M in Q49, $0.9M in Q50, $0.9M in Q51, $0.9M in Q52, $0.9M in Q53, $0.9M in Q54, $0.9M in Q55, $0.9M in Q56, $0.9M in Q57, $0.9M in Q58, $0.9M in Q59, $0.9M in Q60, $0.9M in Q61, $0.9M in Q62, $0.9M in Q63, $0.9M in Q64, $0.9M in Q65, $0.9M in Q66, $0.9M in Q67, $0.9M in Q68, $0.9M in Q69, $0.9M in Q70, $0.9M in Q71, $0.9M in Q72, $0.9M in Q73, $0.9M in Q74, $0.9M in Q75, $0.9M in Q76, $0.9M in Q77, $0.9M in Q78, $0.9M in Q79, $0.9M in Q80, $0.9M in Q81, $0.9M in Q82, $0.9M in Q83, $0.9M in Q84, $0.9M in Q85, $0.9M in Q86, $0.9M in Q87, $0.9M in Q88, $0.9M in Q89, $0.9M in Q90, $0.9M in Q91, $0.9M in Q92, $0.9M in Q93, $0.9M in Q94, $0.9M in Q95, $0.9M in Q96, $0.9M in Q97, $0.9M in N/A for the last two quarters of year-end: Net expenses revenue growth from N/A for the first three quarters of year-end: Net expenses revenue growth from N/A for the second three quarters of year-end: Net expenses revenue growth from N/A for the third three quarters of year-end: Net expenses revenue growth from N/A for the fourth three quarters of year-end: Net expenses revenue growth from N/A for the fifth three quarters of year-end: Net expenses revenue growth from N/A for the sixth three quarters of year-end: Net expenses revenue growth from N/A for the seventh three quarters of year-end: Net expenses revenue growth from N/A for the eight three quarters of year-end: Net expenses revenue growth from N/A for the nine three quarters of year-end: Net expenses revenue growth from N/A for the decade's first three quarters of year-end: Net expenses revenue growth from N/A for the decade's second three quarters of year-end: Net expenses revenue growth from N/A for the decade's third three quarters of year-end: Net expenses revenue growth from N/A for the decade's fourth three quarters of year-end: Net expenses revenue growth from N/A for the decade's fifth three quarters of year-end: Net expenses revenue growth from N/A for the decade's sixth three quarters of year-end: Net expenses revenue growth from N/A for the decade's seventh three quarters of year-end: Net expenses revenue growth from N/A for the decade's eight three quarters of year-end: Net expenses revenue growth from N/A for the decade's nine three quarters of year-end: Net expenses revenue growth from N/A for the decade's ten three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue growth from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarter of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow fromN/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters ofyear-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/A for the decade's bottom three quarters of year-end: Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end: Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end: Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end: Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end: Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade's bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses revenue grow from N/Afor the decade’s bottom three quarters of year-end:Net expenses currency value changes are shown on a separate scale.
</details>

ChartQA

![](images/b2427cc79afbb65bebbee6cf7698ef206ccef06f24adae4e997a00857968c427.jpg)  
DVQA

![](images/4107dc52dafceff1068d07de71a5566a102a306cf178d0d8d1d30e14be6ceed6.jpg)

<details>
<summary>line</summary>

| Month | Value |
|-------|-------|
| Jan   | 0.25  |
| Feb   | 0.30  |
| Mar   | 0.35  |
| Apr   | 0.40  |
| May   | 0.45  |
| Jun   | 0.50  |
| Jul   | 0.55  |
| Aug   | 0.60  |
| Sep   | 0.65  |
| Oct   | 0.70  |
| Nov   | 0.75  |
| Dec   | 0.80  |
| Jan   | 0.25  |
| Feb   | 0.30  |
| Mar   | 0.35  |
| Apr   | 0.40  |
| May   | 0.45  |
| Jun   | 0.50  |
| Jul   | 0.55  |
| Aug   | 0.60  |
| Sep   | 0.65  |
|
| Oct   | 0.70  |
| Nov   | 0.75  |
| Dec   | 0.80  |
</details>

CharXiv

![](images/efa8b9730cbbfd267020c4264e29a7ccde19261210eb56c43cb519421d7f7594.jpg)  
PlotQA   
Figure 4: Overview of the EvoChart-QA Benchmark.

<table><tr><td>Type</td><td>Actions</td></tr><tr><td>Is-Chart &amp; Is-Title-Clear</td><td>Drop None</td></tr><tr><td>Label-Value &amp; Value-Label</td><td>Drop None VaEM: Rand Num VaEM: More Legends VaEM: Change Num-Scale</td></tr><tr><td>Label-Visual &amp; Visual-Label</td><td>Drop None ViEM: Shuffle Color ViEM: Change Axis-Scale ViEM: Change Color Schemes ViEM: Switch Legend Position</td></tr></table>

Table 6: List of Actions in the EvoChart Action Space

- What legend label shows a value of {value\_label} at the point {xlabel}?   
- Could you tell me which legend label corresponds to the value {value\_label} at {xlabel}?   
- Which label in the legend has the value {value\_label} at the {xlabel} position?   
- Identify the legend label with a value of {value\_label} at {xlabel}, please.   
- Which legend label has a value of {value\_label} at the position of {xlabel}?   
- List the values at $\{\text{xlabel}\}$ from bottom to top.   
- Give me the values at {xlabel} arranged in a list from bottom to top.   
- Could you provide the values at {xlabel} in a list, starting from the bottom and going to the top?   
- Please provide a list of the values at {xlabel} in order from bottom to top.   
- I'd like the values at {xlabel} in a list format, ordered from bottom to top.   
- Provide the values at {xlabel} in a list format from bottom to top.   
- What are the data values in ascending order on the x-axis tick right before {xlabel}?   
- Can you list the data values from smallest to largest on the x-axis tick just to the left of {xlabel}?   
- Please provide the data values sorted from smallest to largest for the x-axis tick immediately preceding {xlabel}.   
- Could you tell me the data values from smallest to largest at the x-axis tick just before {xlabel}?   
- What are the data values, ordered from smallest to largest, on the x-axis tick directly left of {xlabel}?

Total number of abortion providers down since 1982   
Number of abortion providers in U.S., by type   
![](images/ab371ab400b5854c949e85db54ea2cbc84e039543709eadd01653f47a740cc8d.jpg)

<details>
<summary>bar_stacked</summary>

| Year | Physicians' offices | Hospitals | Other clinics | Abortion clinics |
|------|---------------------|---------|---------------|-----------------|
| 1982 | 714                 | 1,405   | 379           | 410             |
| 1985 | 652                 | 1,191   | 399           | 438             |
| 1988 | 657                 | 1,040   | 409           | 476             |
| 1992 | 636                 | 855     | 441           | 448             |
| 1996 | 470                 | 703     | 452           | 417             |
| 2000 | 383                 | 603     | 386           | 381             |
| 2005 | 367                 | 604     | 435           | 378             |
| 2008 | 332                 | 610     | 473           | 510             |
| 2011 | 286                 | 595     | 329           | 517             |
| 2014 | 244                 | 638     | 272           | 555             |
| 2017 | 261                 | 518     | 253           | 580             |
| 2020 | 266                 | 530     | 227           | 580             |
</details>

QL: What is the label of the longest bar of 1982?

GT:Hospitals (Strict)

EvoChart: Hospitals

GPT-4o:Abortion clinics

InternVL2.0-40B:Physicians' offices

QR: What is the value of Core i5-10400F?

GT:227.5 (Strict)

EvoChart :227.5

GPT-4o:257.5

InternVL2.0-40B:228.5

![](images/80141199cf09e1f8f4a99b76a7ca807068efc0045cc93e8e82d17d77d303d4c0.jpg)

<details>
<summary>bar</summary>

Best CPUs for Gaming (September 2020)
| CPU Model | Gaming Score |
|---|---|
| Core i-1090K | 257.5 |
| Core i7-1090K | 257.3 |
| Core i3-1090K | 256.3 |
| Core i9-990K & 990KF | 250.0 |
| Core i7-990K | 245.0 |
| Ryzyn i 390XT | 243.8 |
| Ryzyn i 390XT | 243.7 |
| Ryzyn i 390XT | 242.4 |
| Ryzyn i 390XT | 241.4 |
| Ryzyn i 390XT | 238.8 |
| Ryzyn i 3700K | 237.5 |
| Ryzyn i 3700K | 232.5 |
| Core i-600K & 600KF | 231.5 |
| Core i-600K & 600KF | 231.5 |
| Ryzyn i 5-900XT | 231.1 |
| Core i-2700K | 230.3 |
| Ryzyn i 390X | 229.5 |
| Ryzyn i 5-900 | 228.8 |
| Core i5-1000P | 228.0 |
| Core i5-100FP | 227.5 |
| Ryzyn i 390X | 227.0 |
| Core i5-990K | 227.0 |
| Core i7-490I | 227.0 |
| Ryzyn i 5-390I | 221.8 |
| Ryzyn i 330X | 221.8 |
| Ryzyn i 2700X | 218.3 |
| Core i5-960I | 216.3 |
| Core i5-440FP & 960I | 212.3 |
| Ryzyn i 9-900X | 210.3 |
| Ryzyn i 9-900Y | 210.0 |
| Ryzyn i 7-270I | 197.9 |
| Core i-4-500I | 195.5 |
| Ryzyn i 5-260I | 194.8 |
| Core i5-530OK | 193.8 |
| Core i5-5310I | 192.8 |
| Ryzyn i 3-310I | 192.3 |
| Core i3-440G | 190.5 |
| Core i3-441FP & 910I | 189.8 |
| Core i5-240I | 189.8 |
| Ryzyn i 330G | 185.0 |
| Core i3-870I | 177.8 |
| Ryzyn i 246QI | 171.0 |
| Ryzyn i 226QI | 168.8 |
| Athloni 266QI | 147.5 |
| Pernium G560I | 145.0 |
| Pernium G560R | 142.5 |
| Athloni 266GE | 141.0 |
</details>

6-Month sales report and forecast   
ACTUAL --- PROJECTED   
![](images/e4d3822cda65e66b4a8e8cccec3476571d92b704ecbf1f68558853c851c87fe1.jpg)

<details>
<summary>line</summary>

| Month     | Sales Volume (in Thousands USD) |
| --------- | ------------------------------- |
| OCT 2019  | 35                              |
| NOV       | 62                              |
| DEC       | 30                              |
| JAN 2020  | 35                              |
| FEB       | 45                              |
| MAR       | 10                              |
| APR       | 40                              |
| MAY       | 62.5                            |
| JUN       | 75                              |
</details>

QL: What is the value of December?

GT:32 (Flex)

EvoChart: 32

GPT-4o: 32

InternVL2.0-40B: 30

QR: How many American adults support the government banning TikTok during September?

GT:38 (Strict)

EvoChart : 38

GPT-4o: 38

InternVL2.0-40B: 38

Support for a U.S. TikTok ban has dropped among adults since March 2023   
% of U.S. adults who say they would \_ the U.S. government banning TikTok   
![](images/4feb4b4cffc6977f90b4af0149eb488b86fb18b94ab08e04b23e681789065018.jpg)

<details>
<summary>line</summary>

| Date       | Support | Not sure | Oppose |
| ---------- | ------- | -------- | ------ |
| March '23  | 50      | 28       | 22     |
| Sept-Oct '23 | 38      | 35       | 27     |
</details>

Note: Those who did not give an answer are not shown.   
Source: Survey of U.S. adults conducted Sept. 25-Oct. 1, 2023.

PEW RESEARCH CENTER   
![](images/ba9f52a03b5b7ea20bfe062a592d6a667b138285e4ded593f3717578ac2cdb98.jpg)

QL: When did the percent of people aged 50-64 who accept same-sex relationships reach 20?

GT: 1995 (Strict)

EvoChart: 1996

GPT-4o: 1997

InternVL2.0-40B: 1997

QR: How many unemployed persons are there in year 2014?

GT: 10 (Strict)

EvoChart: 10.36

GPT-4o: Approximately 9 million.

InternVL2.0-40B: 10

U.S. Job Openings Drop to Lowest Level Since February 2021   
Number of unemployed persons and job openings in the United States, seasonally adjusted   
![](images/752bf693ecdff1c17eaff71b4b376cec837850a25ffe6752878955576b6f2a0e.jpg)

<details>
<summary>line</summary>

| Year | Job openings | Unemployed persons |
|------|---------------|---------------------|
| '05  | ~3M           | ~8M                 |
| '07  | ~4M           | ~8M                 |
| '09  | ~2M           | ~15M                |
| '11  | ~3M           | ~14M                |
| '13  | ~4M           | ~12M                |
| '15  | ~5M           | ~10M                |
| '17  | ~6M           | ~8M                 |
| '19  | ~7M           | ~6M                 |
| '21  | ~10M          | ~10M                |
| Apr '24 | ~8M       | ~6M                 |
</details>

![](images/b2bb31caf3fc396a8edeee3534fe0225e7629267b2b30d195870ff6bb52e8581.jpg)

![](images/003dacdfb8093b9d38a4d75ea1608ca2db5b1fb0ff52ddd63adbe1e7d7910883.jpg)  
Figure 5: Case 1 of EvoChart-QA, “QL” indicates that the corresponding image is located on the left side, while “QR” indicates that the corresponding image is located on the right side.

![](images/6b9215bf763ac047e918a71294a490ead80381f8f409398092396d1f36b73faa.jpg)

<details>
<summary>bar</summary>

Blade Steel Hardness Ratings (HRC)
| Company | Hardness Rating |
| :--- | :--- |
| ZDP-189 | 64 |
| Maxamet | 64 |
| CPM M4 | 63 |
| CPM S110V | 62 |
| Elmax | 62 |
| CruWear | 62 |
| D2 | 62 |
| M390/CPM 20CV | 61 |
| CTS-XHP | 61 |
| CPM S90V | 61 |
| CPM S45VN | 60 |
| CPM S35V | 60 |
| CPM S30V | 60 |
| VG-10 | 60 |
| ATS-34/154CM | 60 |
| H1 | 59 |
| 14C28N | 59 |
| CTS-BD1 | 59 |
| 13C26 | 59 |
| CPM 3V | 59 |
| CPM 10V | 59 |
| N680 | 58 |
| BCr13MoV | 58 |
| 440C | 58 |
| AUS-8 | 58 |
| AUS-6 | 57 |
| 440A | 57 |
| 420HC | 56 |
| 1095 | 55 |
| 420 | 53 |
</details>

# QL: What is the value of the 10th bar?

GT:61 (Strict)

# EvoChart: 55

# GPT-4o: 61

# InternVL2.0-40B: 62

# QR:How many Hispanic Catholic say they have a NET Favorable view of Donald Trump?

GT:32 (Strict)

# EvoChart : 32

# GPT-4o: 32

# InternVL2.0-40B: 32

Two-thirds of White evangelicals see Trump favorably   
![](images/a3d8a520bcdcd350b7a28cccaac88fc18d62578e4a73266a11269cb384a63cf2.jpg)

<details>
<summary>bar</summary>

| Group | NET Unfavorable | Mostly f.v.f. | Very f.v.f. | NET Favorable |
| --- | --- | --- | --- | --- |
| U.S. adults | 60% | 24% | 15% | 39% |
| White evangelical Protestant | 33 | 36 | 30 | 67 |
| White Catholic | 49 | 32 | 19 | 51 |
| White Protestant, not evangelical | 52 | 26 | 22 | 47 |
| Hispanic Protestant | 54 | 27 | 18 | 45 |
| Muslim | 64 | 23 | 17 | 35 |
| Hispanic Catholic | 66 | 19 | 13 | 32 |
| Nothing in particular | 67 | 22 | 10 | 32 |
| Jewish | 79 | 12 | 9 | 21 |
| Black Protestant | 80 | 13 | 4 | 17 |
| Agnostic | 82 | 14 | 3 | 17 |
| Atheist | 88 | 6 | 1 | 12 |
| Among all U.S. adults who are ... Christian | 53 | 27 | 20 | 46 |
| Protestant | 51 | 27 | 21 | 48 |
| Catholic | 57 | 26 | 17 | 42 |
| Religiously unaffiliated (atheist, agnostic or "nothing in particular") | 74 | 17 | 7 | 25 |
</details>

Note: Refused/Never heard of responses not shown. White and Black adults include those who report being only one race and not Hispanic. Hispanics are of any race.   
Source: Survey of U.S. adults conducted Feb. 13-25, 2024.
PEW RESEARCH CENTER   
% of U.S. Catholics expressing a favorable view of Pope Francis and recent popes

Three-quarters of U.S. Catholics rate Pope Francis favorably

![](images/3f9dc73b92599cfb32eb8a362f325d9dea06283690055e6f81a68692f597d52f.jpg)

<details>
<summary>bar_stacked</summary>

| Year | President | Favorable (%) | Mostly favorable (%) | Very favorable (%) |
| :--- | :--- | :--- | :--- | :--- |
| '87 | John Paul II 1978-2005 | 91 | 43 | 48 |
| '90 | Benedict XVI 2005-2013 | 93 | 40 | 53 |
| '96 | Francis 2013-Present | 67 | 74 | 44 |
| '05 | John Paul II 2005-2013 | 74 | 74 | 50 |
| '07 | Benedict XVI 2005-2013 | 83 | 38 | 31 |
| '08 | Francis 2013-Present | 74 | 34 | 36 |
| '13 | John Paul II 2005-2013 | 74 | 41 | 49 |
| '13 | Benedict XVI 2005-2013 | 84 | 41 | 43 |
| '13 | Francis 2013-Present | 79 | 42 | 37 |
| '14 | John Paul II 2005-2013 | 85 | 34 | 51 |
| '15 | Benedict XVI 2005-2013 | 86 | 34 | 57 |
| '15 | Francis 2013-Present | 81 | 20 | 52 |
| '16 | John Paul II 2005-2013 | 87 | 40 | 62 |
| '16 | Benedict XVI 2005-2013 | 84 | 39 | 47 |
| '17 | Francis 2013-Present | 72 | 42 | 30 |
| '18 | John Paul II 2013-2013 | 77 | 42 | 35 |
| '19 | Benedict XVI 2013-2013 | 82 | 49 | 32 |
| '20 | Francis 2013-Present | 82 | 52 | 31 |
| '21 | John Paul II 2013-2013 | 83 | 49 | 34 |
| '24 | Francis 2013-Present | 75 | 49 | 26 |
The chart displays a stacked bar chart with three segments representing net favorability ratings for each president. The data is presented in a table format with the year as the index. The percentages above each segment are explicitly labeled on the bars.
</details>

Note: Based on U.S. Catholics. Figures from February 2020 and later come from American Trends Panel online surveys. Estimates from January 2020 and earlier come from random-digit dial telephone surveys.
Source: Survey of U.S. adults conducted Feb. 13-25, 2024.   
PEW RESEARCH CENTER

# QL: From which year to which year did Pope

Francis Benedict XVI reign?

GT: 2005-2013 (Strict)

# EvoChart: 2005-2013

# √ GPT-4o: 2005-2013

# InternVL2.0-40B: 2005-2013

# QR: How many genres are mentioned under the David Fincher label?

GT: 5 (Strict)

# EvoChart : 10

# GPT-4o: Six

# InternVL2.0-40B: 9

Directors with Greatest Number of Award Wins   
![](images/d6447c76c12ae4b5fa8272a697e005addee7ea1a62f60c8c976dc93c4364bd7a.jpg)

<details>
<summary>bar_stacked</summary>

| director | Action | Adventure | Animation | Biography | Comedy | Crime | Drama | Family | Fantasy | Horror | Mystery | Romance | Sci-Fi | Thriller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ethan Coen, Joel Coen | 0 | 0 | 0 | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 150 |
| Lee Unkrich | 0 | 120 | 100 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Steven Spielberg | 0 | 80 | 30 | 50 | 50 | 20 | 20 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| George Lucas | 0 | 180 | 0 | 0 | 0 | 150 | 0 | 0 | 0 | 0 | 150 | 0 | 0 | 0 |
| David Fincher | 0 | 90 | 0 | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 90 | 0 | 0 | 0 |
| Curtis Hanson | 0 | 70 | 0 | 0 | 0 | 110 | 110 | 0 | 0 | 0 | 70 | 0 | 0 | 0 |
| Ang Lee | 0 | 110 | 0 | 0 | 60 | 0 | 120 | 0 | 60 | 0 | 110 | 0 | 0 | 0 |
| Terrence Malikc | 0 | 130 | 0 | 0 | 110 | 0 | 130 | 0 | 110 | 60 | 130 | 0 | 60 | 0 |
| Benh Zeilinl | 0 | 95 | 0 | 0 | 95 | 0 | 145 | 0 | 95 | 135 | 95 | 65 | 65 | 65 |
| Roberto Benigni | 0 | 85 | 155 | 155 | 175 | 175 | 175 | 175 | 175 | 175 | 65 | 175 | -35 | -35 |
The chart displays a stacked bar chart with each bar representing the sum of awards won for each director. The legend lists genres: Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, Horror, Mystery, Romance, Sci-Fi, and Thriller. The title 'sum (awards.wins)' appears to be a summary label but not part of the data. The x-axis represents directors' names and the y-axis represents the sum of awards won. However, no bars are rendered in the current image. The chart is labeled with the same axes and the color-coded legend for each bar.
</details>

![](images/14b882b83b678e699202abda3e31b507e8d03c9626c856e77446b70e234ae2f5.jpg)

<details>
<summary>bar</summary>

| Age Group | Strongly/somewhat approve | Strongly/somewhat disapprove |
| --------- | ------------------------- | ----------------------------- |
| U.S. adults | 49 | 28 |
| 18-29 | 37 | 49 |
| 30-44 | 43 | 31 |
| 45-64 | 49 | 23 |
| 65+ | 65 | 13 |
| Dem | 40 | 36 |
| Ind | 42 | 27 |
| Rep | 66 | 20 |
1,682 U.S. adults surveyed March 16-19, 2024; missing percentages to 100% indicate 'not sure' responses; Data source: The Economist, YouGov
</details>

# QL: What is the party ID of the lowest red of the party cluster?

GT: Dem (Strict)

# EvoChart: Dem

# GPT-4o: Rep

# InternVL2.0-40B: 27

# QR: Which fruit is the most popular?

GT: Bananas (Strict)

# EvoChart: Saudi Arabia

# GPT-4o: I can't determine.

# InternVL2.0-40B: Apple

Opec oil production cuts continue   
![](images/2fd3eb30f044fa572461ac46b1d020a6feadb1d95e336d55599b464c2c36c965.jpg)

<details>
<summary>bar</summary>

| Country | Dec 2016 (Million barrels per day) | Jan 2017 (Million barrels per day) | Feb 2017 (Million barrels per day) | Mar 2017 (Million barrels per day) | Apr 2017 (Million barrels per day) | May 2017 (Million barrels per day) |
|---|---|---|---|---|---|---|
| Saudi Arabia | 10.5 | 10.3 | 10.2 | 10.1 | 10.0 | 9.9 |
| Iraq | 4.5 | 4.4 | 4.3 | 4.2 | 4.1 | 4.0 |
| Iran | 4.0 | 3.9 | 3.8 | 3.7 | 3.6 | 3.5 |
| UAE | 3.5 | 3.4 | 3.3 | 3.2 | 3.1 | 3.0 |
| Kuwait | 3.0 | 2.9 | 2.8 | 2.7 | 2.6 | 2.5 |
| Venezuela | 2.5 | 2.4 | 2.3 | 2.2 | 2.1 | 2.0 |
| Angola | 2.0 | 1.9 | 1.8 | 1.7 | 1.6 | 1.5 |
| Nigeria | 1.5 | 1.4 | 1.3 | 1.2 | 1.1 | 1.0 |
| Algeria | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
| Qatar | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| Libya | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| Ecuador | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| Gabon | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
Source: Opec
</details>

Figure 6: Case 2 of EvoChart-QA, “QL” indicates that the corresponding image is located on the left side, while “QR” indicates that the corresponding image is located on the right side.

The Media Industries Most Affected by Piracy

Media sector share of global visits to piracy websites in 2022

![](images/8ee242c2662ea5463dfdd7eeff82a2c1ef43de93c745197776c409e0a1f0cf8c.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
|---|---|
| Segment 1 | 46.3 |
| Segment 2 | 27.5 |
| Segment 3 | 12.9 |
| Segment 4 | 7.0 |
| Segment 5 | 6.2 |
</details>

\* e.g. books   
Source: MUSO   
cc =   
statista

QL: Which media looks most affected by piracy?

GT:TV (Strict)

EvoChart: TV

GPT-4o: TV

InternVL2.0-40B: TV

QR: What is the percentage value of the red DOWN segment?

GT: 3.28 (Strict)

EvoChart : 3.42

GPT-4o: 3.42

InternVL2.0-40B: 3.79

Head pose estimations by time of the experiment   
![](images/85998a87b4d4bd971a4c88aca45396c85d5d3a96b5403477eb48c43e1e0a2864.jpg)

<details>
<summary>pie</summary>

| Direction | Time (secs) |
| :--- | :--- |
| UP: 6.67% | 18-20 (secs) |
| DOWN: 3.33% | 14-16 (secs) |
| Left: 3.33% | 14-16 (secs) |
| Right: 3.33% | 14-16 (secs) |
| UP: 3.33% | 10-12 (secs) |
| Down: 3.33% | 10-12 (secs) |
| Left: 3.33% | 10-12 (secs) |
| Right: 3.33% | 10-12 (secs) |
| UP: 6.67% | 4-5 (secs) |
| Down: 3.33% | 4-5 (secs) |
| Left: 3.33% | 4-5 (secs) |
| Right: 3.33% | 4-5 (secs) |
| UP: 6.67% | 6-8 (secs) |
| Down: 3.33% | 6-8 (secs) |
| Left: 3.33% | 6-8 (secs) |
| Right: 3.33% | 6-8 (secs) |
| UP: 6.67% | 8-10 (secs) |
| Down: 3.33% | 8-10 (secs) |
| Left: 3.33% | 8-10 (secs) |
| Right: 3.33% | 8-10 (secs) |
| UP: 6.67% | 10-12 (secs) |
| Down: 3.33% | 10-12 (secs) |
| Left: 3.33% | 10-12 (secs) |
| Right: 3.33% | 10-12 (secs) |
| UP: 6.67% | 12-14 (secs) |
| Down: 3.33% | 12-14 (secs) |
| Left: 3.33% | 12-14 (secs) |
| Right: 3.33% | 12-14 (secs) |
| UP: 6.67% | 14-16 (secs) |
| Down: 3.33% | 14-16 (secs) |
| Left: 3.33% | 14-16 (secs) |
| Right: 3.33% | 14-16 (secs) |
| UP: 6.67% | 16-18 (secs) |
| Down: 3.33% | 16-18 (secs) |
| Left: 3.33% | 16-18 (secs) |
| Right: 3.33% | 16-18 (secs) |
| UP: 6.67% | 18-20 (secs) |
The chart displays a single data series with values ranging from approximately -18 to +16 seconds. The chart includes a legend for the data series and labels indicating 'Head pose' and 'Up'. The values are annotated above each segment.
</details>

![](images/b72d4039f4ca3b61f4e66f8e8174f24cd9506ed5ef5b3526566cd020c26b88be.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
|---|---|
| Managing staff scheduling was a headache | 20 |
| Marketing and reaching new customers | 8 |
| Other | 13 |
</details>

# 33%

Finding and

keeping

competent,

reliable staff

QL: What is the percentage value of the pink loop segment?

GT: 13 (Flex)

EvoChart: 13

GPT-4o: 13

InternVL2.0-40B: 13

QR: What is the label of the dot farthest to the left under the "More than 5 years ago" chart?

GT: Black (Strict)

EvoChart : Total

√ GPT-4o: Black

InternVL2.0-40B: 10

About one-quarter of Black cryptocurrency users report first using it in the past year

Among U.S. adults who say they have ever invested in, traded or used cryptocurrency, % who say they first did this ...

![](images/919d7693ab4e294bead5fb0e5bcf95288f37dfeed1bec63c0434ae1af71d533a.jpg)

<details>
<summary>scatter</summary>

| Category | Within the past year | 1 to 5 years ago | More than 5 years ago |
| :--- | :--- | :--- | :--- |
| Total | 16 | 74 | 10 |
| White | 12 | 75 | 12 |
| Black | 27 | 67 | 6 |
| Hispanic | 21 | 72 | 7 |
| Upper income | 8 | 76 | 15 |
| Middle income | 13 | 76 | 10 |
| Lower income | 31 | 61 | 7 |
</details>

Note: Lines surrounding data points represent the margin of error of each estimate. Family income tiers are based on adjusted 2021 earnings. White and Black adults include those who report being only one race and are not Hispanic. Hispanics are of any race. Those who did not give an answer are not shown.
Source: Survey of U.S. adults conducted March 13-19, 2023,   
PEW RESEARCH CENTER

![](images/7846f283ff03f83ce23046fafc2fb652d96f3ce41f6392956793908b9d016387.jpg)

QL: What is the label of the purple triangle dots?

GT: Wine store (Strict)

EvoChart: Wine store

GPT-4o: Liquor Store

InternVL2.0-40B: Wine store

QR: What is the weight value of the upper black do when x=60?

GT: 165 (Strict)

EvoChart: 154

GPT-4o: 175

InternVL2.0-40B: 165

SCATTERPLOT OF

WEIGHT VS. HEIGHT

There are many situations in which a quantitative variable depends on both quantitative and

qualitative variables. In particular, we will see how the weight of the students in the PULSE data varies with their heights and whether the relationship is different for the two sets.

varies with their heights, and whether the relationship is different for the two

![](images/13fed9cef1aeaa7494827926d4f22bc58b0af1acd60d48f6ea8ae823c749450d.jpg)

<details>
<summary>scatter</summary>

|   X |   Y | Gender |
|----:|----:|--------|
|  60 | 150 | Female |
|  60 | 155 | Male   |
|  61 | 152 | Female |
|  61 | 153 | Male   |
|  62 | 148 | Female |
|  62 | 155 | Male   |
|  63 | 160 | Female |
|  63 | 165 | Male   |
|  64 | 170 | Female |
|  64 | 168 | Male   |
|  65 | 165 | Female |
|  65 | 170 | Male   |
|  66 | 162 | Female |
|  66 | 168 | Male   |
|  67 | 175 | Female |
|  67 | 172 | Male   |
|  68 | 170 | Female |
|  68 | 175 | Male   |
|  69 | 168 | Female |
|  69 | 170 | Male   |
|  70 | 155 | Female |
|  70 | 160 | Male   |
|  71 | 165 | Female |
|  71 | 170 | Male   |
|  72 | 175 | Female |
|  72 | 178 | Male   |
|  73 | 180 | Female |
|  73 | 182 | Male   |
|  74 | 178 | Female |
|  74 | 180 | Male   |
|  75 | 170 | Female |
|  75 | 175 | Male   |
|  76 | 172 | Female |
|  76 | 178 | Male   |
|  77 | 170 | Female |
|  77 | 175 | Male   |
|  78 | nan |       |
</details>

SOURCES   
http://statland.org/Software\_Help/Minitab/MTBpul2.html   
CREATED BY   
f in

Figure 7: Case 3 of EvoChart-QA, “QL” indicates that the corresponding image is located on the left side, while “QR” indicates that the corresponding image is located on the right side.

Obama Job Rating Mostly Steady   
Approve or disapprove of the way Barack Obama is handling his job as president?   
![](images/13113a35294e5dce803da5e5b021328d18d074003ad8dc726bc3a0e065722e58.jpg)

<details>
<summary>line</summary>

| Approval | Disapprove |
| -------- | ---------- |
| 40       | 52         |
| 41       | 51         |
| 46       | 47         |
| 43       | 51         |
| 43       | 49         |
| 46       | 46         |
| 46       | 49         |
| 44       | 49         |
| 43       | 51         |
| 41       | 53         |
| 45       | 49         |
| 43       | 49         |
</details>

Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Jan 2013 2014   
Survey conducted Jan 15-19, 2014.   
PEW RESEARCH CENTER/USA TODAY

# QL: what is the percentage of people who disapprove

Obama is handling his job as president in Sep?

GT: 49

Gemini: 49   
√ GPT-4o: 49   
√ Phi3-Vision: 49

QR: When does the Green line Reach the peak?

GT:Apr

Gemini: Nov   
√ GPT-4o: Apr   
√ Phi3-Vision: Apr

71% of Americans say the worst of the coronavirus outbreak is ‘still to come’   
% of U.S. adults who say, in thinking about the problems the country is facing from the coronavirus outbreak ...   
![](images/d6f71a2402e54cde7eb3690716199ace431a03c3f4f4e0ce948cd3489b787484.jpg)

<details>
<summary>line</summary>

| Category | The worst is still to come | The worst is behind us |
| :--- | :--- | :--- |
| 1 | 73 | 26 |
| 2 | 59 | 40 |
| 3 | 71 | 28 |
</details>

Apr Jun Nov   
Note: Respondents who did not give an answer are not shown.
Source: Survey conducted Nov. 18-29, 2020.
“Intent to Get a COVID-19 Vaccine Rises to 60% as Confidence in Research and Development Process Increases”   
PEW RESEARCH CENTER

Democrats now have more confidence than Republicans in Merkel   
Confidence in German Chancellor Angela Merkel to do the right thing regarding world affairs   
![](images/4b324fea5e9e661b3230e1f7ca1a7004cb66b3888c643f0967c10334ee0f520d.jpg)

<details>
<summary>line</summary>

| Year | Democrats | Republicans |
| ---- | --------- | ----------- |
| 2006 | 35        | 42          |
| 2007 | 46        | 51          |
| 2008 | 38        | 42          |
| 2009 | 49        | 43          |
| 2010 | 45        | 37          |
| 2011 | 41        | 46          |
| 2012 | 45        | 47          |
| 2017 | 64        | 50          |
</details>

Source: Spring 2017 Global Attitudes Survey.   
PEW RESEARCH CENTER

# QL: what is the value of the Red line in 2011?

GT: 46

Gemini: 45   
GPT-4o: 45   
× Phi3-Vision: 47

QR: what is the value of the most light green segment?

GT: 5

Gemini: 826106   
GPT-4o: 13   
× Phi3-Vision: 42

The Selected Reserve   
% of the U.S. Reserves and National Guardforces (826,106) in each branch, 2015   
![](images/409c5c8252cdabd9d211ea136fb2c6b501ba9f0192b8906cc4b731dc7ad25fe6.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Army National Guard | 42 |
| Army Reserve | 24 |
| Air National Guard | 13 |
| Air Force Reserve | 8 |
| Navy Reserve | 7 |
| Marine Corps Reserve | 5 |
| Coast Guard Reserve | 1 |
</details>

Note: An additional 275,247 are members of the Inactive National Guard and the Individual Ready Reserve. These individuals do not train regularly and do not participate in active-duty training exercises with members of the Selected Reserve Defense Department "2015 Demographics: Profile of the Military Community" report.   
PEW RESEARCH CENTER

Where Do You Get Most of your News About National and International Issues?   
![](images/2837d213b525196aa5f5550b41f2724b8f393e18d00e7db8b3ca6c574052a303.jpg)

<details>
<summary>line</summary>

| Year | Television | Newspaper | Internet |
|------|------------|-----------|----------|
| 2001 | 74         | 45        | 13       |
| '02  | 82         | 42        | 14       |
| '03  | 80         | 50        | 20       |
| '04  | 74         | 46        | 24       |
| '05  | 73         | 36        | 20       |
| '06  | 72         | 36        | 24       |
| '07  | 74         | 34        | 24       |
| '08  | 70         | 35        | 40       |
| '09  | 70         | 32        | 35       |
| '10  | 66         | 31        | 41       |
</details>

PEW RESEARCH CENTER Dec 1-5, 2010. Figures add to more than 100% because respondents could volunteer up to two main sources. If asked more than once in a calendar year, trend shows final datapoint from each year.

# QL: What is the legend label of the 3th line in 2009, from bottom to the top?

GT: Internet

Gemini: Radio   
GPT-4o: Radio   
√ Phi3-Vision: Internet

QR: when the dark orange line get its value 50.7, what is the year?

GT: 2010

√ Gemini: 2010   
GPT-4o: 2010   
√ Phi3-Vision: 2010

U.S. Hispanic population reached nearly 61 million in 2019

In millions

![](images/d99e87cf38afe112af0d620199315ff68b1d7ce561fefcb85c6fa4d95d835633.jpg)

<details>
<summary>line</summary>

| Year | Value |
|---|---|
| 1970 | 9.6 |
| 1980 | 14.5 |
| 1990 | 22.6 |
| 2000 | 35.7 |
| 2010 | 50.7 |
| 2019 | 60.6 |
</details>

Note: Population estimates for 1990-2019 are as of July 1 for each year. Hispanics are of any race.   
Source: Pew Research Center analysis of 1970-1980 estimates based on decennial censuses (see 2008 report "U.S. Population Projections: 2005-2050"), U.S. intercensal population estimates for 1990-1999 and 2000-2009, and U.S. Census Bureau Vintage 2019 estimates for 2010-2019.   
PEW RESEARCH CENTER

Figure 8: Case 1 of Modified ChartQA, “QL” indicates that the corresponding image is located on the left side, while “QR” indicates that the corresponding image is located on the right side.

Belief that U.S. considers other countries' interests in foreign policy returns to 2007 level   
In making international policy decisions, to what extent do you think the U.S. takes the interest of your country into account?   
![](images/906b499622bee06d3d29826559fa2eeb14966177da28d15ed391be9e6297e3f9.jpg)

<details>
<summary>line</summary>

| Year | Not too much/not at all | Great deal/fair amount |
| ---- | ------------------------ | ----------------------- |
| 2007 | 71                       | 26                      |
| 2009 | 61                       | 36                      |
| 2013 | 60                       | 37                      |
| 2018 | 72                       | 27                      |
</details>

Note: 14-country median based on Argentina, Canada, France, Germany, Indonesia, Israel, Japan, Kenya, Mexico, Poland, Russia, South Korea, Spain and the UK. Source: Spring 2018 Global Attitudes Survey. Q39.

PEW RESEARCH CENTER

# QL:what is the value of the 2th line a the year 2009?

GT: 61

Gemini: 60   
GPT-4o: 61   
√ Phi3-Vision: 61

QR: what is the value of orange line in 2009?

GT: 65

Gemini: 54   
GPT-4o: 54   
× Phi3-Vision: 43

Since 2014, most Russians have been satisfied with their country's direction   
\_ with the way things are going in our country today   
![](images/893a4a753f29aea54ed48812fc62d6617570d8da87a95412024c8939f8176155.jpg)

<details>
<summary>line</summary>

| Year | Satisfied | Dissatisfied |
|------|-----------|--------------|
| 2002 | 71        | 20           |
| 2004 | 69        | 26           |
| 2006 | 62        | 32           |
| 2008 | 56        | 43           |
| 2010 | 65        | 34           |
| 2012 | 59        | 45           |
| 2014 | 56        | 37           |
| 2017 | 58        | 37           |
</details>

Source: Spring 2017 Global Attitudes Survey. Q2.

PEW RESEARCH CENTER   
Many say financial situation has worsened for average people   
Compared with 20 years ago, the financial situation of average people in our country is ...   
![](images/ef510dfc1c6a92586d564da32bb02f67b53e1f415e4a6a46875e07c3f8b5bb6c.jpg)

<details>
<summary>bar_stacked</summary>

| Country | Worse (%) | No change (%) | Better (%) |
| :--- | :--- | :--- | :--- |
| Greece | 87 | 5 | 7 |
| Italy | 72 | 14 | 10 |
| Spain | 62 | 12 | 24 |
| France | 56 | 22 | 20 |
| UK | 53 | 16 | 28 |
| Germany | 46 | 15 | 36 |
| Hungary | 44 | 21 | 33 |
| Netherlands | 31 | 18 | 45 |
| Sweden | 19 | 13 | 66 |
| Poland | 17 | 11 | 68 |
| MEDIAN | 50 | 15 | 31 |
</details>

Note: Don't know responses not shown.   
Source: Spring 2018 Global Attitudes Survey. Q6.

PEW RESEARCH CENTER

QL: what is the value of NOT Change in the 8th bar

from the top?

GT: 18

Gemini: 15

GPT-4o: 15

√ Phi3-Vision: 18

QR: What is the percentage of Japanese take favorable view of U.S. in 2016?

GT: 72

Gemini: 78

GPT-4o: 72

× Phi3-Vision: 78

Japanese view of the U.S. and confidence in the American president has sharply declined   
![](images/0e7b28ca4c592ccbc96b593a896bf32f151a7b7539bb33d5e4735c269546776e.jpg)

<details>
<summary>line</summary>

| Year | Favorable view of U.S. | Confidence in U.S. President |
|------|------------------------|------------------------------|
| 2006 | 63                     | 32                           |
| 2007 | 61                     | 35                           |
| 2008 | 50                     | 25                           |
| 2009 | 59                     | 85                           |
| 2010 | 66                     | 76                           |
| 2011 | 81                     | 85                           |
| 2012 | 72                     | 74                           |
| 2013 | 69                     | 70                           |
| 2014 | 66                     | 60                           |
| 2015 | 68                     | 66                           |
| 2016 | 72                     | 78                           |
| 2017 | 57                     | 24                           |
</details>

Source: Spring 2017 Global Attitudes Survey. Q12a & Q30a.

PEW RESEARCH CENTER   
Positive economic sentiment increasing since 2016   
The current economic situation in our country is ...   
![](images/35f8fe9abbf5afa32b2dddcdc5e7f3670cd46d70b438f58fbad8168ea430fd50.jpg)

<details>
<summary>line</summary>

| Year | Bad | Good |
|------|-----|------|
| 2010 | 66  | 34   |
| 2012 | 67  | 32   |
| 2014 | 61  | 39   |
| 2016 | 71  | 29   |
| 2018 | 54  | 45   |
</details>

Source: Survey of Nigerian adults conducted June 25-July 30, 2018. Q2.

PEW RESEARCH CENTER

QL: How many people held the opinion that the economic situation in our country was bad in 2015?

GT: 42

Gemini: 61

GPT-4o: 57

× Phi3-Vision: 57

QR: What is the value of Should Not in Turkey?

GT: 55

Gemini: 55

GPT-4o: 55

√ Phi3-Vision: 55

Publics in NATO countries express reluctance on Article 5 obligations   
% who say if Russia got into a serious military conflict with one of its neighboring countries that is our NATO ally, (survey country) \_ use military force to defend that country   
![](images/4520db716a7d089d992422c0da2f4eca58d33d9695506c6d835b280edefe0e47.jpg)

<details>
<summary>bar</summary>

| Country | Should not (%) | Should (%) |
| :--- | :--- | :--- |
| Netherlands | 32 | 64 |
| U.S. | 29 | 60 |
| Canada | 38 | 56 |
| UK | 41 | 55 |
| Lithuania | 34 | 51 |
| France | 53 | 41 |
| Spain | 56 | 41 |
| Poland | 43 | 40 |
| Czech Rep. | 47 | 36 |
| Germany | 60 | 34 |
| Hungary | 43 | 33 |
| Slovakia | 55 | 32 |
| Turkey | 55 | 32 |
| Greece | 63 | 25 |
| Italy | 66 | 25 |
| Bulgaria | 69 | 12 |
16-COUNTRY MEDIAN
</details>

Note: Don't know responses not shown.
Source: Spring 2019 Global Attitudes Survey. Q24.
“NATO Seen Favorably Across Member States”

PEW RESEARCH CENTER

Figure 9: Case 2 of Modified ChartQA, “QL” indicates that the corresponding image is located on the left side, while “QR” indicates that the corresponding image is located on the right side.

- On the x-axis tick immediately to the left of {xlabel}, what are the data values from smallest to largest?   
- What legend label features a {line\_color} {line\_style} line?   
- Which legend key shows a {line\_color} {line\_style} line?   
- Can you identify the legend label with a {line\_color} {line\_style} line?   
- Which label in the legend corresponds to a {line\_color} {line\_style} line?   
- Could you tell me which legend label has a {line\_color} {line\_style} line?   
- Which legend label has a {line\_color} {line\_style} line?   
- Can you tell me the line style for {legend\_label}?   
- What kind of line style does {legend\_label} use?   
- Please specify the line style associated with {legend\_label}.   
- How is the line style defined for {legend\_label}?   
- What's the type of line style for {legend\_label}?   
- What is the line style of {legend\_label}?   
- Can you tell me the value of the {n}th data point from the left on the {line\_color} line that is {line\_style}?   
- What is the value of the {n}th point from the left on the {line\_style} line that is colored {line\_color}?   
- Please provide the value of the {n}th data point from the left on the {line\_color} {line\_style} line.   
- Could you specify the value of the {n}th point from the left on the {line\_style} line in {line\_color}?   
- What value does the {n}th point from the left have on the {line\_color} line with {line\_style}?   
- What is the value of the {n}th data point from left to right on the {line\_style} line of {line\_color} color?   
- When the line labeled {legend\_label} hits the {value\_label} mark at {xlabel}, how many lines is it positioned above?   
- At {xlabel}, when the {legend\_label} line reaches the {value\_label} level, how many other lines is it above?   
- How many lines are beneath the {legend\_label} line when it reaches {value\_label} at {xlabel}?   
- When {legend\_label} hits the value {value\_label} at {xlabel}, how many lines are below it?   
- How many lines does the line labeled {legend\_label} surpass at {xlabel} when it attains the {value\_label} value?   
- When the line represented by {legend\_label} reaches the value {value\_label} at {xlabel}, how many lines is this line above?   
- At which x-label does the line represented by {legend\_label} reach its highest point?   
- Where does the line indicated by {legend\_label} peak on the x-axis?   
- Can you identify the x-label where the line for {legend\_label} reaches its maximum value?   
- Which x-label corresponds to the highest point of the line denoted by {legend\_label}?   
- At what x-label is the peak of the line marked by {legend\_label}?   
- The highest point of the line represented by {legend\_label} is at which x-label?   
- Where on the x-axis does the line labeled {legend\_label} dip to its lowest value?   
- Can you tell me the x-label where the {legend\_label} line hits its minimum?

- At which point on the x-axis does the {legend\_label} line bottom out?   
- What's the x-label where the line for {legend\_label} reaches its lowest point?   
- Where along the x-axis does the {legend\_label} line find its lowest value?   
- At which x-label does the line represented by {legend\_label} reach its lowest point?   
- At which x-label does the line with color {line\_color} and style {line\_style} reach its lowest point?   
- Where along the x-axis does the {line\_color} line, with its {line\_style} style, hit the bottom?   
- Can you tell me the x-label where the {line\_color} line, designed in {line\_style}, drops to its lowest?   
- Which x-label marks the lowest point of the {line\_color} line that's styled as {line\_style}?   
- What's the x-label where the {line\_color} line, styled in {line\_style}, touches its lowest point?   
- Which xlabel does the {line\_color} line, styled as {line\_style}, reach its bottom?   
- At which xlabel does the {line\_color} colored line with {line\_style} reach its peak?   
- Can you tell me the value of {legend\_label} in {xlabel}?   
- I'd like to know the value of {legend\_label} within {xlabel}.   
- Could you provide the value of {legend\_label} found in {xlabel}?   
- What amount does {legend\_label} have in {xlabel}?   
- Please specify the value of {legend\_label} in the context of {xlabel}.   
- What is the value of {legend\_label} in {xlabel}?   
- What is the legend label for the value {value\_label} on the {xlabel} axis?   
- Identify the legend label that corresponds to the value {value\_label} in {xlabel}.   
- Which legend label matches the value {value\_label} in {xlabel}?   
- Find the legend label associated with the value {value\_label} in {xlabel}.   
- Can you tell me the legend label that has the value {value\_label} in {xlabel}?   
- Which legend label have value of {value\_label} in {xlabel}?   
- Which legend label has a value of {value\_label} at the position of {xlabel}?   
- At the position of {xlabel}, which legend label corresponds to the value {value\_label}?   
- Identify the legend label that has a value of {value\_label} at the {xlabel} position.   
- What legend label holds the value {value\_label} at the position indicated by {xlabel}?   
- Determine the legend label with a value of {value\_label} at the {xlabel} location.   
- Which legend label shows a value of {value\_label} at the position marked by {xlabel}?   
- Provide the values at {xlabel} in a list format rating from small to large.   
- Can you list the values at {xlabel} from small to large?   
- Please arrange the values at {xlabel} in a list from small to large.

- List the values at {xlabel} in ascending order.   
- I'd like to see the values at {xlabel} rated from small to large in a list.   
- Can you provide a list of the values at {xlabel} from the smallest to the largest?   
- Could you please give the values at {xlabel} in a list format, ordered from small to large?   
- What is the value of the $\{\mathbf{n}\}$ th data point from bottom to top on the $\{\text{border\_type}\}$ border bar of $\{\text{line\_color}\}$ color?   
- Could you tell me the value of the {n}th data point from the bottom to top on the {border\_type} border bar in {line\_color}?   
- Please provide the value of the {n}th data point from bottom to top on the {line\_color} {border\_type} border bar.   
- What's the value of the $\{\mathbf{n}\}$ th data point from bottom to top on the $\{\text{line\_color}\} \{\text{border\_type}\}$ border bar?   
- I need the value of the {n}th data point from the bottom to the top on the {border\_type} border bar of {line\_color}.   
- Can you find the value of the {n}th data point from bottom to top on the {line\_color} border bar of {border\_type} type?   
- Please tell me the value of the {n}th data point from bottom to top on the {border\_type} border bar that is {line\_color}.   
- What is the value of the longest {line\_color} bar?   
- Can you tell me the value of the tallest {line\_color} bar?   
- I need to know the value of the highest {line\_color} bar.   
- What is the value of the {line\_color} bar with the maximum height?   
- Please provide the value of the largest {line\_color} bar.   
- Could you find out the value of the highest {line\_color} bar?   
- What is the value of the shortest {line\_color} bar?   
- Can you tell me the value of the smallest {line\_color} bar?   
- I need to know the value of the least tall {line\_color} bar.   
- What is the value of the {line\_color} bar with the minimum height?   
- Please provide the value of the {line\_color} bar that is the shortest.   
- Could you find out the value of the shortest {line\_color} bar for me?   
- Can you tell me the value of {legend\_label} in {xlabel}?   
- I'd like to know the value of {legend\_label} within {xlabel}.   
- Could you provide the value of {legend\_label} found in {xlabel}?   
- What amount does {legend\_label} have in {xlabel}?   
- Please specify the value of {legend\_label} in the context of {xlabel}.   
- What is the value of {legend\_label} in {xlabel}?   
- What is the legend label for the value {value\_label} on the {xlabel} axis?   
- Identify the legend label that corresponds to the value {value\_label} in {xlabel}.   
- Which legend label matches the value {value\_label} in {xlabel}?   
- Find the legend label associated with the value {value\_label} in {xlabel}.   
- Can you tell me the legend label that has the value {value\_label} in {xlabel}?   
- Which legend label have value of {value\_label} in {xlabel}?

- Which legend label has a value of {value\_label} at the position of {xlabel}?   
- At the position of {xlabel}, which legend label corresponds to the value {value\_label}?   
- Identify the legend label that has a value of {value\_label} at the {xlabel} position.   
- What legend label holds the value {value\_label} at the position indicated by {xlabel}?   
- Determine the legend label with a value of {value\_label} at the {xlabel} location.   
- Which legend label shows a value of {value\_label} at the position marked by {xlabel}?   
- Provide the values at {xlabel} in a list format rating from small to large.   
- Can you list the values at {xlabel} from small to large?   
- Please arrange the values at {xlabel} in a list from small to large.   
- List the values at $\{\text{xlabel}\}$ in ascending order.   
- I'd like to see the values at {xlabel} rated from small to large in a list.   
- Can you provide a list of the values at {xlabel} from the smallest to the largest?   
- Could you please give the values at {xlabel} in a list format, ordered from small to large?   
- What is the value of the {n}th data point from left on the {border\_type} border bar of {line\_color} color?   
- Could you tell me the value of the {n}th data point from left on the {border\_type} border bar in {line\_color}?   
- Please provide the value of the {n}th data point from left on the {line\_color} {border\_type} border bar.   
- What's the value of the $\{\mathfrak{n}\}$ th data point from left on the $\{\text{line\_color}\} \{\text{border\_type}\}$ border bar?   
- I need the value of the {n}th data point from left on the {border\_type} border bar of {line\_color}.   
- Can you find the value of the {n}th data point from left on the {line\_color} border bar of {border\_type} type?   
- Please tell me the value of the {n}th data point from left on the {border\_type} border bar that is {line\_color}.   
- What is the value of the longest {line\_color} bar?   
- Can you tell me the value of the tallest {line\_color} bar?   
- I need to know the value of the highest {line\_color} bar.   
- What is the value of the {line\_color} bar with the maximum height?   
- Please provide the value of the largest {line\_color} bar.   
- Could you find out the value of the highest {line\_color} bar?   
- What is the value of the shortest {line\_color} bar?   
- Can you tell me the value of the smallest {line\_color} bar?   
- I need to know the value of the least tall {line\_color} bar.   
- What is the value of the {line\_color} bar with the minimum height?   
- Please provide the value of the {line\_color} bar that is the shortest.   
- Could you find out the value of the shortest {line\_color} bar for me?   
• How many sectors are there in total in this pie chart?   
- How many segments are there in total in this pie chart?   
• What is the total number of sectors in this pie chart?

- How many segments in total are present in this pie chart?   
• What is the total count of sectors in this pie chart?   
• How many segments does this pie chart have in total?   
- What is the percentage of 'sector\_label' in 'series\_label' on this chart?   
- How much percent does 'sector\_label' make up in 'series\_label' on this chart?   
- Can you tell me the percentage of 'sector\_label' within 'series\_label' in this chart?   
- What proportion of 'series\_label' does 'sector\_label' represent in this chart?   
- How large is the percentage of 'sector\_label' in the 'series\_label' shown on this chart?   
- In this chart, what percentage does 'sector\_label' constitute in 'series\_label'?   
- What is the number of 'sector\_label' in 'series\_label' on this chart?   
- How many 'sector\_label' are there in 'series\_label' on this chart?   
- Can you find the number of 'sector\_label' within 'series\_label' in this chart?   
- What count of 'sector\_label' does 'series\_label' have in this chart?   
- How many instances of 'sector\_label' are in 'series\_label' on this chart?   
- In this chart, what is the count of 'sector\_label' in 'series\_label'?   
- What percentage of 'series\_label' is made up by 'sector\_label' in this chart?   
- What is the proportion of 'sector\_label' in 'series\_label' on this chart?   
- Could you specify the fraction of 'sector\_label' within 'series\_label' depicted in this chart?   
- What ratio does 'sector\_label' contribute to 'series\_label' as shown in this chart?   
- How large is the share of 'sector\_label' in 'series\_label' on this chart?   
- In this chart, what part of series\_label does 'sector\_label' represent?   
- What is the percentage of the sector with the {name\_color} color?   
- How much percent does the sector with the {name\_color} color represent?   
- Can you tell me the proportion of the sector with the {name\_color} color?   
- What fraction does the sector with the {name\_color} color make up?   
- How large is the percentage of the sector with the {name\_color} color?   
- In this chart, what percentage does the sector with the {name\_color} color constitute?   
- What is the number of the sector with the {name\_color} color?   
- How many sectors are there with the {name\_color} color?   
- Can you tell me the count of the sector with the {name\_color} color?   
- What is the quantity of the sector with the {name\_color} color?   
- How many sectors are labeled with the {name\_color} color?

- In this chart, what is the number of sectors with the {name\_color} color?   
- What is the percentage of the sector with the {name\_color} color?   
- How much of the sector is represented by the {name\_color} color in percentage?   
- Can you tell me the percentage of the sector that is {name\_color}?   
- What fraction of the sector is the {name\_color} color?   
- How large is the share of the sector with the {name\_color} color?   
- What portion of the sector does the {name\_color} color represent in percentage?   
- What is the percentage of the largest sector in 'series\_label' in the pie chart?   
- In the pie chart, what percentage does the largest sector in 'series\_label' represent?   
- What proportion does the largest sector in 'series\_label' hold in the pie chart?   
- How much percentage does the largest sector in 'series\_label' account for in the pie chart?   
- Can you tell me the percentage of the largest sector in 'series\_label' on the pie chart?   
- What is the share of the largest sector in 'series\_label' in the pie chart?   
- What percentage of the pie chart does the smallest sector in 'series\_label' occupy?   
- In 'series\_label', what is the percentage of the smallest sector in the pie chart?   
- What is the fractional representation of the smallest sector in 'series\_label' within the pie chart?   
- How much does the smallest sector in 'series\_label' contribute to the pie chart as a percentage?   
- What is the smallest sector's percentage in the pie chart under 'series\_label'?   
- Within 'series\_label', what is the percentage value of the smallest sector in the pie chart?   
- Can you tell me the {y\_axis\_topic} of {x\_value} {x\_axis\_topic} in {legend\_name}?   
- What is the {y\_axis\_topic} of {x\_value} {x\_axis\_topic} in {legend\_name}?   
- Please provide the {y\_axis\_topic} for {x\_value} {x\_axis\_topic} in {legend\_name}.   
- Could you tell me the {y\_axis\_topic} for {x\_value} {x\_axis\_topic} in {legend\_name}?   
- What {y\_axis\_topic} corresponds to {x\_value} {x\_axis\_topic} in {legend\_name}?   
- Can you provide the {y\_axis\_topic} of {x\_value} {x\_axis\_topic} in {legend\_name}?   
- Could you provide the {x\_axis\_topic} for a {y\_value} {y\_axis\_topic} in {legend\_name}?   
- What is the {x\_axis\_topic} for a {y\_value} {y\_axis\_topic} in {legend\_name}?   
- Can you give the {x\_axis\_topic} for a {y\_value} {y\_axis\_topic} in {legend\_name}?   
- Please provide the {x\_axis\_topic} corresponding to a {y\_value} {y\_axis\_topic} in {legend\_name}.

- Could you tell me the {x\_axis\_topic} for a {y\_value} {y\_axis\_topic} in {legend\_name}?   
- What {x\_axis\_topic} corresponds to a {y\_value} {y\_axis\_topic} in {legend\_name}?   
• How many legend labels are there in the chart?   
• What is the number of legend labels in the chart?   
- In the chart, how many legend labels are present?   
- How many labels are there in the chart's legend?   
- What count of legend labels is shown in the chart?   
- Can you tell how many legend labels are included in the chart?   
- How many different colors of data points are there in the chart?   
- What is the number of different colors of data points in the chart?   
- In the chart, how many different colors of data points can be observed?   
- How many unique colors of data points are present in the chart?   
- What count of different colored data points is shown in the chart?   
- Can you tell how many different colors of data points are in the chart?   
- What is the {x\_axis\_topic} value of the {legend\_name} at the peak {y\_axis\_topic} in this chart?   
- What is the {x\_axis\_topic} value of the {legend\_name} at the peak {y\_axis\_topic} in this chart?   
- What {x\_axis\_topic} value corresponds to the peak {y\_axis\_topic} for the {legend\_name} in this chart?   
- In this chart, what is the {x\_axis\_topic} value when {legend\_name} reaches the peak {y\_axis\_topic}?   
- At the peak {y\_axis\_topic} for {legend\_name} in this chart, what is the {x\_axis\_topic} value?   
- What is the {x\_axis\_topic} value when {legend\_name} has the peak {y\_axis\_topic} in this chart?   
- In this chart, what {x\_axis\_topic} value aligns with the peak {y\_axis\_topic} of {legend\_name}?   
- What is the corresponding {x\_axis\_topic} value when {legend\_name} reaches its highest {y\_axis\_topic}?   
- When {legend\_name} reaches its highest {y\_axis\_topic}, what is the corresponding {x\_axis\_topic} value?   
- What {x\_axis\_topic} value corresponds to the highest {y\_axis\_topic} for {legend\_name}?   
- When {legend\_name} has its highest {y\_axis\_topic}, what is the corresponding {x\_axis\_topic} value?   
- What is the {x\_axis\_topic} value at the highest {y\_axis\_topic} for {legend\_name}?   
- When {legend\_name} hits its highest {y\_axis\_topic}, what is the corresponding {x\_axis\_topic} value?   
- When {legend\_name} reaches its highest point, what is the corresponding {x\_axis\_topic} value?   
- When {legend\_name} is at its highest point, what is the corresponding {x\_axis\_topic} value?   
- What is the {x\_axis\_topic} value when {legend\_name} reaches its highest point?   
- At the highest point of {legend\_name}, what is the corresponding {x\_axis\_topic} value?   
- What {x\_axis\_topic} value corresponds to the highest point of {legend\_name}?

- When {legend\_name} reaches its peak, what is the corresponding {x\_axis\_topic} value?   
- What is the {x\_axis\_topic} value of the {legend\_name} with the lowest {y\_axis\_topic} in this chart?   
- In this chart, what {x\_axis\_topic} value corresponds to the {legend\_name} with the minimum {y\_axis\_topic}?   
- For the {legend\_name} with the lowest {y\_axis\_topic} in this chart, what is the {x\_axis\_topic} value?   
- What {x\_axis\_topic} value is associated with the {legend\_name} that has the lowest {y\_axis\_topic} in this chart?   
- Which $\{\mathrm{x\_axis\_topic}\}$ value corresponds to the lowest $\{\mathrm{y\_axis\_topic}\}$ for the $\{\text{legend\_name}\}$ in this chart?   
- In this chart, what is the {x\_axis\_topic} value for the {legend\_name} with the smallest {y\_axis\_topic}?   
- What is the corresponding {x\_axis\_topic} value when {legend\_name} reaches its lowest {y\_axis\_topic}?   
- When {legend\_name} has its lowest {y\_axis\_topic}, what is the corresponding {x\_axis\_topic} value?   
- What {x\_axis\_topic} value corresponds to the lowest {y\_axis\_topic} for {legend\_name}?   
- For {legend\_name} at its minimum {y\_axis\_topic}, what is the corresponding {x\_axis\_topic} value?   
- Which {x\_axis\_topic} value aligns with {legend\_name} when it has the lowest {y\_axis\_topic}?   
- When the {y\_axis\_topic} is at its lowest for {legend\_name}, what is the corresponding {x\_axis\_topic} value?   
- When {legend\_name} reaches its lowest point, what is the corresponding {x\_axis\_topic} value?   
- What is the {x\_axis\_topic} value when {legend\_name} hits its lowest point?   
- At the lowest point of {legend\_name}, what is the corresponding {x\_axis\_topic} value?   
- What {x\_axis\_topic} value corresponds to the lowest point of {legend\_name}?   
- When {legend\_name} is at its lowest, what is the {x\_axis\_topic} value?   
- What {x\_axis\_topic} value aligns with the lowest point of {legend\_name}?   
- Which legend has a {y\_axis\_topic} equal to {y\_value} when the {x\_axis\_topic} is {x\_value}?   
- Which legend shows a {y\_axis\_topic} of {y\_value} when the {x\_axis\_topic} equals {x\_value}?   
- When the $\{x\_axis\_topic\}$ is $\{x\_value\}$ , which legend corresponds to a $\{y\_axis\_topic\}$ of $\{y\_value\}$ ?   
- At $\{x\_value\}$ on the $\{x\_axis\_topic\}$ , which legend has a $\{y\_axis\_topic\}$ value of $\{y\_value\}$ ?   
- What legend's {y\_axis\_topic} is {y\_value} when the {x\_axis\_topic} reads {x\_value}?   
- When $\{x\_value\}$ is the value of the $\{x\_axis\_topic\}$ , which legend has a $\{y\_axis\_topic\}$ of $\{y\_value\}$ ?   
- For an $\{x\_axis\_topic\}$ of $\{x\_value\}$ , which legend displays a $\{y\_axis\_topic\}$ of $\{y\_value\}$ ?   
- Which legend has a value of $\{y\_value\}$ when the $\{x\_axis\_topic\}$ is $\{x\_value\}$ ?   
- Which legend shows a value of $\{y\_value\}$ when the $\{x\_axis\_topic\}$ equals $\{x\_value\}$ ?

- When the $\{x\_axis\_topic\}$ is $\{x\_value\}$ , which legend corresponds to a value of $\{y\_value\}$ ?   
- At $\{x\_value\}$ on the $\{x\_axis\_topic\}$ , which legend has a value of $\{y\_value\}$ ?   
- What legend's value is {y\_value} when the {x\_axis\_topic} reads {x\_value}?   
- When $\{x\_value\}$ is the value of the $\{x\_axis\_topic\}$ , which legend has a value of $\{y\_value\}$ ?   
- For an $\{x\_axis\_topic\}$ of $\{x\_value\}$ , which legend displays a value of $\{y\_value\}$ ?

# EvoChart-QA Source Websites

We selected 650 images from 140 publicly available websites for academic research purposes only. All sources are listed as follows:

- https://www.beautiful.ai   
- https://www.formsbirds.com   
- https://leanscape.io   
- https://www.investopedia.com   
- https://www.storytellingwithdata.com   
- https://blog.finxter.com   
- https://www.degruyter.com   
- https://www.anychart.com   
- https://www.infragistics.com   
- https://awesomeopensource.com   
- https://fluttercore.com   
- https://www.nicesnippets.com   
- http://20bits.com   
- https://unreasonablegroup.com   
- https://mavink.com   
- https://www.smartsheet.com   
- https://template.wps.com   
- https://learn.microsoft.com   
- https://www.zoho.com   
- https://keski.condesan-ecoandes.org   
- https://www.statmethods.net   
- https://www.theinformationlab.com   
- https://www.pluralsight.com   
- https://www.visualitics.it   
- https://dribbble.com   
- https://infogram.com   
- https://beautifulai-od3.appspot.com   
- https://www.slidesteam.net   
- https://sainsdata.id   
- https://www.elegantthemes.com   
- https://www.polymersearch.com   
- https://blog.csdn.net   
- https://aten.edu.vn   
- https://www.sakuranpost.net   
- https://imagesee.biz   
- https://search.justgulfwon.live   
- https://p.codekk.com   
- https://vitalflux.com   
- https://zebrabi.com

- https://classfullprecisions.z13.web.core.windows.net   
- https://www.infocaptor.com   
- https://www.monkeybreadsoftware.de   
- https://mdpi.com   
- https://www.calxa.com   
- https://byggipedia.se   
- https://www.template.net   
- https://www.devtodev.com   
- https://www.bakertilly.com   
- https://www.researchgate.net   
- https://www.tessresearch.org   
- https://www.tandfonline.com   
- http://ww25.chartexamples.com   
- https://www.exceldemy.com   
- https://in.pinterest.com   
- https://blog.51cto.com   
- https://www.fusioncharts.com   
- https://inforiver.com   
- https://exceljet.net   
- https://x.com   
- https://stevenrattner.com   
- https://www.tillerhq.com   
- https://knifeknowitall.com   
- https://exploratory.io   
- https://www.r-bloggers.com   
- https://jethrojeff.com/   
- https://marcuscalan.blogspot.com   
- https://www.pewresearch.org/   
- https://www.smashingmagazine.com   
- https://chart-studio.plotly.com   
- https://www.mdpi.com   
- https://loganix.com   
- https://www.knowbe4.com   
- https://www.zdnet.com   
- https://goldhartmediation.ca   
- https://wiseinvestments.ca   
- https://laptrinhx.com   
- https://mungfali.com   
- https://data-flair.training   
- https://www.ncl.ac.uk   
- https://www.dummies.com   
- https://georgecarlo.blogspot.com   
- http://www.aploris.com   
- https://respect.international   
- https://www.pinterest.com   
- https://www.educba.com   
- https://www.statista.com/   
- https://slidebazaar.com   
- https://venngage.com   
- https://airfreesm.best   
- https://ar.pinterest.com   
- https://www.conceptdraw.com

- https://www.ft.com   
- https://www.statology.org   
- https://exceljet.net/charts   
- https://www.slidekit.com   
- https://worksheetsploshes.z14.web.core.windows.net   
- https://www.hotzxgirl.com   
- https://www.genekitr.fun   
- https://medium.com   
- https://daydreamingnumbers.com   
- https://www.newsweek.com   
- https://www.mekkographics.com   
- https://nowbam.com   
- https://www.mercurynews.com   
- https://www.everviz.com   
- https://byjus.com   
- https://www.dailyrecord.co.uk   
- https://appfire.com   
- https://www.aiophotoz.com   
- https://socialbarrel.com   
- https://www.canadianpizzamag.com   
- https://edubenchmark.com   
- https://ieltsessentialindia.blogspot.com   
- https://forum.knime.com   
- https://docs.oracle.com   
- https://www.cec.health.nsw.gov.au   
- https://www.linkedin.com   
- https://online.stat.psu.edu   
- https://gmt-tutorials.org   
- https://gitcode.csdn.net   
- https://online.visual-paradigm.com   
- https://python-charts.com   
- https://cloud.tencent.com   
- https://www.cnblogs.com   
- https://help.xlstat.com   
- https://www.listendata.com   
- https://docs.thoughtspot.com   
- https://www.data-to-viz.com   
- https://fahimahmad.netlify.app   
- https://developer.aliyun.com   
- https://visme.co   
- https://pythonspot.com   
- https://data36.com   
- https://bootcamp.uxdesign.cc   
- https://revistaplural.es   
- https://environicsanalytics.com   
- https://mainpackage9.gitlab.io   
- https://en.wikipedia.org   
- https://riset.guru.pubiway.com   
- https://lopezcollege.weebly.com