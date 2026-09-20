# Text-space Graph Foundation Models: Comprehensive Benchmarks and New Insights

Zhikai Chen $^{1}$ , Haitao Mao $^{1}$ , Jingzhe Liu $^{1}$ , Yu Song $^{1}$ , Bingheng Li $^{1}$ , Wei Jin $^{2}$ , Bahare Fatemi $^{3}$ , Anton Tsitsulin $^{3}$ , Bryan Perozzi $^{3}$ , Hui Liu $^{1}$ , Jiliang Tang $^{1}$

$^{1}$ Michigan State University, $^{2}$ Emory University, $^{3}$ Google Research

# Abstract

Given the ubiquity of graph data and its applications in diverse domains, building a Graph Foundation Model (GFM) that can work well across different graphs and tasks with a unified backbone has recently garnered significant interests. A major obstacle to achieving this goal stems from the fact that graphs from different domains often exhibit diverse node features. Inspired by multi-modal models that align different modalities with natural language, the text has recently been adopted to provide a unified feature space for diverse graphs. Despite the great potential of these text-space GFMs, current research in this field is hampered by two problems. First, the absence of a comprehensive benchmark with unified problem settings hinders a clear understanding of the comparative effectiveness and practical value of different text-space GFMs. Second, there is a lack of sufficient datasets to thoroughly explore the methods' full potential and verify their effectiveness across diverse settings. To address these issues, we conduct a comprehensive benchmark providing novel text-space datasets and comprehensive evaluation under unified problem settings. Empirical results provide new insights and inspire future research directions. Our code and data are publicly available from https://github.com/CurryTang/TSGFM.

# 1 Introduction

Foundation Models (FMs) [1] have achieved remarkable success in various domains like computer vision [2, 3] and natural language processing [4, 5]. Through large-scale pre-training on diverse data [6, 7], FMs exhibit several intriguing properties compared to task-specific models trained in an end-to-end manner. First, one model can serve diverse tasks with better effectiveness [7, 8], and second, they present emergent capabilities such as in-context learning [9] and reasoning [10].

Nonetheless, the common practice in today's graph machine learning remains training task-specific models from scratch on each individual dataset [11]. Despite the success of graph models in diverse domains such as social networks [12, 13], e-commerce [14–16], and biology [17], most graph models still necessitate tailored data engineering and specific design for each dataset, which makes it hard to scale up due to limited data available for a single dataset [18].

Feature heterogeneity is the key obstacle for extending graph machine learning to training across data and tasks. [11]. Specifically, it refers to the fact that different graphs present different feature dimensions, where the corresponding dimension may have entirely different semantic meanings. Such a problem makes it impossible to train a GFM. To mitigate this problem, [19, 20] propose transforming different kinds of node attributes into texts and then using a large language model (LLM) to generate embeddings, which provides a unified feature space. This feature space offers two advantages: (1) it can mitigate the feature heterogeneity by mapping diverse node features into the same textual space, and (2) thanks to the rich latent knowledge in LLMs, the generated high-quality

text features may improve model performance $[21, 22]$ . Leveraging these high-quality text features, we can unify different graphs in a manner akin to how multi-modal models unify modalities through text $[2, 23, 24]$ , thus giving rise to text-space GFMs $[19, 20, 25]$ . Text-space GFMs can generalize to diverse graphs $[19]$ and show preliminary success. Such a unified feature space also gives new potential for previous graph machine learning methods such as graph self-supervised learning $[26]$ towards building GFM, which applies to various graphs and domains with a unified backbone.

Despite the considerable potential of text-space GFMs, a comprehensive understanding of their applicability and effectiveness across different application scenarios remains elusive.

a) First, most existing work is evaluated on a small number of datasets, primarily focusing on citation datasets, which makes the observations less representative and fails to reflect the full potential of GFMs.   
b) Second, each work adopts its own GFM problem setting and proposes diverse GFM frameworks, which makes it hard to understand the effectiveness of different methods and hinders the development of a landscape of the whole field.   
c) Third, existing work merely evaluates proposed methods, while understanding text-space GFMs' effectiveness remains elusive.

Contributions. To demystify the design spaces of text-space GFMs and inspire future research directions, we introduce a benchmark designed to illuminate the capabilities and limitations of existing text-space GFMs. Our contributions are multi-folded:

1. Novel Text-Space Datasets: Recognizing the scarcity of existing text-space datasets and evaluation based on text space, we curate and preprocess over 20 datasets spanning academic, E-commerce, biology, and other miscellaneous domains.   
2. Comprehensive Evaluation on Diverse Use Cases: Leveraging data from various tasks, we define four applicable GFM paradigms. We first evaluate different GFM building blocks under each setting and then adopt these building blocks as anchor models to investigate the overall effectiveness of text-space GFMs. Our benchmark provides a more comprehensive GFM setting than existing works.   
3. Novel Insights: Our empirical results allow us to derive novel insights, and the most crucial ones are as follows: Although LLMs offer a feature space with promising initial performance, there still exists gaps across different datasets. The positive transfer observed in text-space GFMs relies on transferable structural patterns and is only effective when combined with appropriate inductive biases designed for downstream tasks.

# 2 Preliminaries

In this section, we introduce the background of our benchmark. First, we present the traditional graph machine learning (GML) pipeline, where a task-specific model is trained from scratch. Then, we delineate the general paradigms of text-space GFMs.

# 2.1 Problem Setting

Traditional GML. The standard GML setting involves training a single model for each task. Given dataset P and downstream task D, a specific model $M_{t}$ is trained on P to address D. Such a pipeline necessitates specific data engineering and model deployment for each task.

Graph Foundation Models. GFMs extend the traditional GML setting across different datasets and tasks. Despite the more diverse settings, most GFMs follow a unified paradigm: transferring the knowledge from training tasks to tackle downstream tasks with a unified model backbone. Given a collection of training datasets $P = \{P_{1}, P_{2}, \cdots, P_{n}\}$ , where each dataset $P_{i}$ may encompass multiple training tasks $T_{i} = \{T_{i1}, T_{i2}, \cdots, T_{ik_{i}}\}$ , a GFM $M_{\theta}$ is trained on the union of all training tasks $T = \bigcup_{i=1}^{n} T_{i}$ using a shared representation encoder $E_{\theta}$ and optional task-specific heads $H = \{H_{11}, H_{12}, \ldots, H_{nk_{n}}\}$ where $k_{n}$ represents the number of tasks for n-th dataset. The trained model $M_{\theta}$ can then be adapted to tackle downstream datasets $D = \{D_{1}, D_{2}, \cdots, D_{m}\}$ , each with its own set of downstream tasks $S_{j} = \{S_{j1}, S_{j2}, \cdots, S_{jl_{j}}\}$ where $l_{j}$ represents the number of tasks for j-th dataset. The adaption requires a unified architecture, which means either the entire model's parameters are shared or the encoder is shared with only tunable task-specific heads.

Categorizing GFMs. In this work, we draw inspiration from the GFM literature $[19, 20, 11]$ to focus on the fine-grained categorization of GFMs in different use cases. We decompose these use cases using a framework of scenarios and tasks. A scenario describes the relationship between training and downstream tasks. In this work, we consider two scenarios: co-training and pre-training: co-training specifies the co-trained model to be applied to the same set of datasets, which means P = D. On the other hand, pre-training considers the case when pre-trained models are applied to novel datasets unseen in the training stage, which means $P \neq D$ . Next, besides categorizing based on train and test data relationships, we use the concept of tasks to consider the relationship between training and downstream tasks. In this work, we consider conventional GML tasks, including node classification (NC), link prediction (LP), and graph classification (GC). Referring to $[11]$ , we categorize existing GFMs into task-specific and cross-tasks models. Task-specific GFMs focus on transferring inside a specific task, which means $T_{1} = \cdots = T_{n} = S_{1} = \cdots = S_{m}$ . Cross-task GFMs target a more challenging setting where the knowledge is transferred across diverse tasks, such as node and graph classifications, which assumes the existence of $T_{i} \neq S_{j}$ .

Based on the tuple (scenarios, tasks), we come up with 4 fine-grained GFM paradigms as shown in Figure 1. We then showcase the practical value of the proposed GFM paradigms.

Practical value of the GFM paradigm. GFMs have two primary strengths that we seek to leverage. First, their efficiency. GFMs aim to solve multiple tasks with one model, increasing developer velocity while decreasing maintenance complexity. Sharing a common model across tasks should allow for additional optimizations that would not be cost-effective in a one-model-per-task setting. In the pre-training scenario, GFMs with shared architecture can effectively adapt to a low-resource

downstream task without tuning parameters. Second, their effectiveness. GFMs have more model capacity and available training data than single-purpose models. Recent results show that increasing the amount of available training data can lead to better performance $[18]$ . Especially in the co-training scenario, GFMs present the potential to improve performance by scaling across datasets and tasks.

![](images/1ec1f38612bb01846d1c3dbb3c51e7e06f99f8939791b69d608c0974691d7122.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Co-training
        A["Paper"] --> B["Book"]
    end
    subgraph Pre-training
        C["Paper"] --> D["Book"]
    end
    E["Paper"] --> F["Molecule"]
    G["Paper"] --> H["Category"]
    I["Paper"] --> J["Category"]
    K["Paper"] --> L["Category"]
    M["Paper"] --> N["Category"]
    O["Paper"] --> P["Category"]
    Q["Paper"] --> Q["Category"]
    R["Paper"] --> S["Category"]
    T["Paper"] --> T["Category"]
    U["Paper"] --> V["Category"]
    W["Paper"] --> X["Category"]
    Y["Paper"] --> Y["Category"]
    Z["Paper"] --> AA["Category"]
    AB["Paper"] --> AC["Category"]
    AD["Paper"] --> AE["Category"]
    AF["Paper"] --> AG["Category"]
    AH["Paper"] --> AI["Category"]
    AJ["Paper"] --> AK["Category"]
    AL["Paper"] --> AM["Category"]
    AN["Paper"] --> AO["Category"]
    AP["Paper"] --> AQ["Category"]
    AR["Paper"] --> AS["Category"]
    AT["Paper"] --> AU["Category"]
    AV["Paper"] --> AW["Category"]
    AX["Paper"] --> AY["Category"]
    AZ["Paper"] --> BA["Category"]
    BB["Paper"] --> BC["Category"]
    BD["Paper"] --> BE["Category"]
    BF["Paper"] --> BG["Category"]
    BH["Paper"] --> BI["Category"]
    BJ["Paper"] --> BK["Category"]
    BL["Paper"] --> BM["Category"]
    BN["Paper"] --> BO["Category"]
    BP["Paper"] --> BP1["Category"]
    BP2["Paper"] --> BP2["Category"]
    BP3["Paper"] --> BP3["Category"]
    BP4["Paper"] --> BP4["Category"]
    BP5["Paper"] --> BP5["Category"]
    BP6["Paper"] --> BP6["Category"]
    BP7["Paper"] --> BP7["Category"]
    BP8["Paper"] --> BP8["Category"]
    BP9["Paper"] --> BP9["Category"]
    BP10["Paper"] --> BP10["Category"]
    BP11["Paper"] --> BP11["Category"]
    BP12["Paper"] --> BP12["Category"]
    BP13["Paper"] --> BP13["Category"]
    BP14["Paper"] --> BP14["Category"]
    BP15["Paper"] --> BP15["Category"]
    BP16["Paper"] --> BP16["Category"]
    BP17["Paper"] --> BP17["Category"]
    BP18["Paper"] --> BP18["Category"]
    BP19["Paper"] --> BP19["Category"]
    BP20["Paper"] --> BP20["Category"]
    BP21["Paper"] --> BP21["Category"]
    BP22["Paper"] --> BP22["Category"]
    BP23["Paper"] --> BP23["Category"]
    BP24["Paper"] --> BP24["Category"]
    BP25["Paper"] --> BP25["Category"]
    BP26["Paper"] --> BP26["Category"]
    BP27["Paper"] --> BP27["Category"]
    BP28["Paper"] --> BP28["Category"]
    BP29["Paper"] --> BP29["Category"]
    BP30["Paper"] --> BP30["Category"]
    BP31["Paper"] --> BP31["Category"]
    BP32["Paper"] --> BP32["Category"]
    BP33["Paper"] --> BP33["Category"]
    BP34["Paper"] --> BP34["Category"]
    BP35["Paper"] --> BP35["Category"]
    BP36["Paper"] --> BP36["Category"]
    BP37["Paper"] --> BP37["Category"]
    BP38["Paper"] --> BP38["Category"]
    BP39["Paper"] --> BP39["Category"]
    BP40["Paper"] --> BP40["Category"]
    BP41["Paper"] --> BP41["Category"]
    BP42["Paper"] --> BP42["Category"]
    BP43["Paper"] --> BP43["Category"]
    BP44["Paper"] --> BP44["Category"]
    BP45["Paper"] --> BP45["Category"]
    BP46["Paper"] --> BP46["Category"]
    BP47["Paper"] --> BP47["Category"]
    BP48["Paper"] --> BP48["Category"]
    BP49["Paper"] --> BP49["Category"]
    BP50["Paper"] --> BP50["Category"]
    NP1[Paper in Task-specific: ? What's the category; ? category in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in Cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-task: ? What's the category; ? category in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in cross-tasks; ? property in across-tasks: ? What's the category: Paper & Book
```
</details>

Figure 1: We come up with four paradigms: (Co-training, task-specific), (Co-training, cross-tasks), (Pre-training, task-specific), (Pre-training, cross-tasks)

# 2.2 Text-space GFM Building Blocks

When introducing general GFM paradigms, we emphasize using a unified model architecture to transfer, which requires a shared feature space across different datasets. To achieve a unified feature space, text-space GFMs adopt LLMs as the feature encoders, based on which various techniques have been proposed to learn transferable knowledge across different datasets and tasks, including graph SSL, graph prompts, and LLM with graph projectors $[27]$ .

Text space as the unified feature space. Text-space GFMs adopt LLMs as encoders to project node attributes into a unified feature space. However, this requires that the original attributes can be represented as texts. For non-text attributes like ones for molecules, text-space GFMs may adopt multi-modal models $[19]$ like text-chemistry models $[24]$ to transform the original attributes into texts. As a result, text-space GFMs can process a wide variety of datasets. We empirically evaluate the performance loss brought by the text-space transformation in Appendix B.

Learning transferrable knowledge across graphs. Building upon the unified feature space, various techniques have been proposed to learn transferrable knowledge across different graphs and tasks.

1. Graph SSL [26] employs a unified self-supervised learning task in the training stage, assuming that this task can learn general representations benefiting different downstream tasks.   
2. Foundational graph prompt [19, 20, 28] transforms diverse tasks into a unified format. As a motivating example, [19] first unifies tasks at different levels by viewing node classification as the ego-graph classification and link prediction as the classification of the node pair-induced subgraph. Then, it inserts tasks' labels as augmented nodes into the subgraph used for prediction, which

converts multi-label classification into multiple binary classification problems, thereby unifying all classification tasks. Graph prompts mainly focus on unifying the formulation of tasks, and they still rely on the inductive bias of the model backbone to transfer across different graphs.

3. LLM with graph projectors [25, 29–31] leverages the inherent multi-task capability of LLMs. It is equipped with a projector from graph to text space [23], enabling natural language to describe different graph tasks and thus achieving a unified task formulation.

# 3 Text-space Dataset

![](images/17ee5f80ce0290d2ef312b85c1b6a5f9d29f282066022289530e70c7a75ef298.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Graph
        A["Cora"] --> B["Arxiv23"]
        C["CiteSeer"] --> B
        D["Arxiv"] --> B
        E["Hiv"] --> F["Bbbp"]
        G["PCba"] --> H["Bace"]
        I["Chemble"] --> J["Tox21"]
        K["Muv"] --> L["Toxcast"]
    end

    subgraph Node, Link
        M["History"] --> N["Child"]
        O["Photo"] --> P["Sportsfit"]
        Q["Physical"] --> R["Products"]
    end

    subgraph Graph
        S["MISC"] --> T["Pubmed"]
        U["WikiCS"] --> V["Tolokers"]
        W["Ratings"]
    end
```
</details>

Figure 2: Our proposed text-space dataset covering 20+ datasets coming from diverse domains.

![](images/a40d8ce646500dc28f1dc6070a51d67fef204a30532a59b99609c9d5307b414a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Transforming Node Features"] --> B["Atomics"]
    A --> C["Categorical"]
    A --> D["Expert prompts"]
    A --> E["Text description"]
    A --> F["Estrogen receptor alpha (ER alpha) is Nuclear hormone..."]
    A --> G["This user passes the background test"]
    H["Transforming Labels"] --> I["Gemini"]
    H --> J["Generate descriptions"]
    H --> K["AI research investigates how to create intelligent systems..."]
    L["Class: AI"] --> M["→"]
    N["Class: AI"] --> O["→"]
    P["Class: AI"] --> Q["→"]
    R["Class: AI"] --> S["→"]
    T["Class: AI"] --> U["→"]
```
</details>

Figure 3: Transforming attributes and labels into text space.

To facilitate a comprehensive evaluation of diverse paradigms (Section 2.1), we introduce over 20 text-space datasets as shown in Figure 2. These are derived from [22, 24, 19, 32–34], with attributes transformed into texts through pre-processing of raw files or the generation of expert descriptions following [24, 19] as shown in Figure 3. We utilize Gemini [35] to generate text descriptions for labels.

These datasets encompass a variety of tasks, including node classification, link prediction, and graph classification. Leveraging MMD $[36]$ as a similarity metric and considering the source of datasets, we categorize them into domains. This yields 4 datasets from the CS citation domain, 6 from e-commerce, and 8 from the molecular domain, with the remaining classified as other domains due to divergence from established ones. This categorization allows us to investigate two key questions: (1) For those in-domain datasets with similar features, can text-space GFM fully address feature heterogeneity and achieve positive transfer? (2) Does increasing the volume of training data, both within and across domains, improve GFM performance, thereby demonstrating neural scaling properties $[37]$ ? We adhere to the original splits $[22, 33, 19, 32]$ to simulate varying dataset sizes in real-world applications. Notably, our contribution lies in the breadth and diversity of text-space datasets across domains, exceeding the scope of prior works $[20, 19, 25]$ . Detailed dataset descriptions are provided in Appendix D.

# 4 Empirical Studies

In this section, we present the empirical studies of text-space GFMs. Based on the problem setting in Section 2, we conduct research from the following two dimensions: (1) In each of the 4 paradigms, we comprehensively evaluate different building blocks of GFMs. (2) Based on the experimental results, the selected datasets and models can be viewed as anchors to reflect the overall effectiveness of text space GFMs in this paradigm. The following subsections will be structured as follows: we first introduce the general experiment configurations and then present the empirical results.

# 4.1 Experiment Configurations

We first present the selected GFM building blocks and evaluation settings.

Models. Following Section 2.2, we adopt the following models for each GFM building block:

1. For Graph SSL, we adopt representative methods including DGI [38], GCC [39], BGRL [40], and also GraphMAE [41]. These methods cover different paradigms such as contrastive learning-based SSL, augmentation-free SSL, and feature reconstruction-based SSL [26]. To train these models across different graphs, we adopt GraphSAINT [42] to extract mini-batches with size 1024.

2. For foundational graph prompt models, we adopt two representative methods, OFA [19] and Prodigy [20], specifically designed for GFM training. The original OFA introduces weights to balance different datasets, which are not proportional to the size of the dataset and require extensive tuning, making them impractical in real-world scenarios. We set all weights to 1 to examine the model's preference across different data and tasks.   
3. For LLM with graph projectors, we adopt LLaGA [25] considering its effectiveness and simplicity. We adopt Mistral-7B [43] as the LLM backbone.   
4. As link prediction can transfer across different graphs with a unified formulation, we consider link prediction-specific models like BUDDY [44] and SEAL [45] for link prediction.

For the LLM encoder, we adopt Sentence-BERT [46] since it can achieve good performance with low computational cost [22]. We discuss how other LLM encoders affect the results in Appendix G.2.

Evaluation settings. We use the performance on downstream tasks to evaluate different GFMs. Specifically, for node-level tasks, we use accuracy as the metric. We use the corresponding metrics used in $[33]$ for graph-level tasks. Notably, we use the hit rate as the metric for link-level tasks. $[19, 20, 25]$ use AUC and accuracy to evaluate link prediction, which has been shown ineffective in differentiating different baselines $[47]$ . For hyper-parameter tuning, different hyper-parameters lead to varying model preferences across datasets. Therefore, we utilize the average validation performance of different datasets to select the optimal model. We present the comprehensive experimental settings and model-specific hyper-parameter searching range in Appendix E.

The following subsections present the empirical evaluation results following four paradigms in Section 2.1. We first present the specific experimental settings and the empirical results. At the end of each paradigm, we highlight the core observations. In this paper, we focus our investigation on the co-training setting for two main reasons: First, co-training is a natural extension of the existing end-to-end learning paradigm on graphs, allowing us to leverage existing principles [11] for understanding and making it an actionable next step. Second, through effective adaptation techniques [48], co-trained models also have the potential to be applied to the pre-training setting.

# 4.2 Case 1: Co-training over the same task

We start from the paradigm (Co-training, Task-specific). This work mainly focuses on three tasks: node classification, link prediction, and graph classification.

# 4.2.1 Co-training for Node Classification

Experiment Settings. For co-training over node-level datasets, we adopt graphs from the CS Citation domain, E-commerce domain, Pubmed, and WikiCS from other domains. For baseline models, we consider all baselines introduced in 4.1 except Prodigy and link prediction-specific methods, which are not applicable. We evaluate models under the following three settings: (1) the model is trained on a specific downstream task from scratch; (2) the model is co-trained on graphs from the same domain; and (3) the model is co-trained on overall available datasets.

Table 1: Performance of node-level co-training. ST refers to “training on a single graph from scratch”. ID refers to “co-training on graphs coming from the same domain”. CD refers to “co-training across all graphs”. “Cit-Avg” records the average performance of citation datasets. “Ecom-Avg” records the average performance of E-commerce datasets. “Avg” records the average performance of all datasets. green and yellow represent the domain of data. Underline represents the case where co-training benefits compared to training from scratch. 

<table><tr><td>Methods</td><td>Setting</td><td>Cora</td><td>CiteSeer</td><td>Arxiv</td><td>Arxiv-2023</td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>Cit-Avg</td><td>Ecom-Avg</td><td>Avg</td></tr><tr><td>GCN</td><td>ST</td><td>82.20</td><td>75.29</td><td>73.10</td><td>74.98</td><td>85.25</td><td>56.62</td><td>82.42</td><td>87.43</td><td>89.37</td><td>88.00</td><td>76.39</td><td>81.52</td><td>79.47</td></tr><tr><td rowspan="3">OFA</td><td>ST</td><td>79.41</td><td>81.35</td><td>73.85</td><td>73.75</td><td>83.33</td><td>53.77</td><td>84.46</td><td>86.48</td><td>92.50</td><td>87.35</td><td>77.09</td><td>81.32</td><td>79.63</td></tr><tr><td>ID</td><td>70.74</td><td>81.66</td><td>72.68</td><td>74.07</td><td>83.30</td><td>56.22</td><td>85.05</td><td>87.83</td><td>92.29</td><td>86.91</td><td>74.79</td><td>81.93</td><td>79.08</td></tr><tr><td>CD</td><td>72.63</td><td>70.19</td><td>72.72</td><td>74.13</td><td>83.88</td><td>56.89</td><td>84.95</td><td>87.65</td><td>92.35</td><td>86.96</td><td>72.42</td><td>82.11</td><td>78.24</td></tr><tr><td rowspan="3">GraphMAE</td><td>ST</td><td>81.00</td><td>74.36</td><td>71.67</td><td>74.40</td><td>83.07</td><td>51.79</td><td>83.27</td><td>83.54</td><td>88.49</td><td>85.90</td><td>75.36</td><td>79.34</td><td>77.75</td></tr><tr><td>ID</td><td>78.09</td><td>68.80</td><td>72.80</td><td>73.30</td><td>83.82</td><td>51.38</td><td>83.47</td><td>83.82</td><td>88.47</td><td>85.90</td><td>73.25</td><td>79.48</td><td>76.99</td></tr><tr><td>CD</td><td>80.27</td><td>70.65</td><td>72.43</td><td>71.02</td><td>83.95</td><td>51.38</td><td>83.00</td><td>83.39</td><td>88.38</td><td>85.88</td><td>73.59</td><td>79.33</td><td>77.04</td></tr><tr><td rowspan="3">DGI</td><td>ST</td><td>81.80</td><td>72.95</td><td>70.36</td><td>72.47</td><td>82.93</td><td>48.34</td><td>83.38</td><td>80.86</td><td>86.28</td><td>84.14</td><td>74.40</td><td>77.66</td><td>76.35</td></tr><tr><td>ID</td><td>80.17</td><td>67.18</td><td>71.39</td><td>72.88</td><td>83.11</td><td>49.45</td><td>81.78</td><td>82.90</td><td>86.77</td><td>85.47</td><td>72.91</td><td>78.25</td><td>76.11</td></tr><tr><td>CD</td><td>81.50</td><td>73.14</td><td>71.84</td><td>72.44</td><td>83.24</td><td>49.64</td><td>83.25</td><td>82.68</td><td>86.67</td><td>85.21</td><td>74.73</td><td>78.45</td><td>76.96</td></tr><tr><td rowspan="3">LLaGA</td><td>ST</td><td>81.25</td><td>68.80</td><td>76.05</td><td>76.00</td><td>82.55</td><td>55.05</td><td>86.00</td><td>87.75</td><td>91.45</td><td>88.85</td><td>75.53</td><td>81.94</td><td>79.38</td></tr><tr><td>ID</td><td>79.10</td><td>68.25</td><td>76.20</td><td>75.80</td><td>83.30</td><td>54.45</td><td>85.40</td><td>87.00</td><td>91.40</td><td>89.00</td><td>74.84</td><td>81.76</td><td>78.99</td></tr><tr><td>CD</td><td>76.45</td><td>63.95</td><td>75.90</td><td>75.10</td><td>81.80</td><td>54.10</td><td>86.60</td><td>86.75</td><td>90.60</td><td>88.80</td><td>72.85</td><td>81.44</td><td>78.01</td></tr></table>

Results. We summarize the performance of each model on individual datasets after co-training in Table 1. As the performance of BGRL and GCC are significantly lower than other methods, we

omit them from the table for visualization clarity. Our results indicate that various GFM methods, regardless of in-domain or cross-domain co-training, still underperform compared to task-specific GCN baselines. Notably, LLaGA and OFA, based on supervised learning, exhibit better overall performance and surpass GCN baselines in the E-commerce domain. Meanwhile, we find that different methods exhibit different characteristics during in-domain and cross-domain co-training as follows: (1) When co-training on the same domain, LLaGA tends to match the performance of training from scratch. Cross-domain co-training on a large number of datasets only negatively impacts LLaGA. A similar phenomenon can also be observed when LLMs with cross-modality projectors are applied to CV [49], which may be related to catastrophic forgetting. (2) We observe that hyperparameter tuning can improve the performance of model training from scratch. However, the optimal hyperparameters vary across datasets, contributing to the underperformance of the unified co-trained model. (3) Co-training can potentially benefit SSL methods. Specifically, DGI demonstrates the potential for performance improvement with increasing data scale. The key observations can be summarized as follows.

Observation 1. Under the task-specific co-training for node classification, GFM methods present a performance gap compared to GCN training from scratch, while certain methods like DGI show potential to improve performance with data scale.

Further Probing. To better understand the ineffectiveness of node-level co-training, we further investigate the design of OneForAll, the model with the best performance. We consider two surrogate models to disentangle the influence of node features and graph structures: (1) replacing OneForAll's backbone with MLP to eliminate the graph structures and (2) replacing OneForAll's GCN-based backbone with SGC [50]-like fixed feature preprocessing. As shown in Table 2, three different sets of data result in three distinct outcomes. For the citation dataset, we observe a decrease in MLP and GCN's performance after co-training, indicating that even without the influence of structure, features in this dataset still lead to negative transfer. For e-commerce datasets, there is no negative transfer for both MLP and GCN. Using SGC to replace the GCN backbone yields better results in all three cases. The primary reasons why we don't observe benefits in node-level co-training are: (1) Stacking more data doesn't exhibit a scaling behavior if we ignore graph structure; (2) When considering graph structure, GCN with learnable aggregation as the backbone does not perform better than SGC [50] with fixed aggregation, indicating that stacking more data does not lead to learning a better aggregation function. Since there is no improvement in either feature or structure aspects, co-training shows no benefits.

Table 2: Average performance is recorded in the table. ST means the model is trained from scratch on a single graph. CT means co-training across different graphs. GCN-\* represents the original OneForAll model, while SGC-\* represents the variants replacing the original GCN backbone with SGC backbone. 

<table><tr><td colspan="5">Co-train: CS + Pubmed (citation)</td><td colspan="5">Co-train: E-commerce</td><td colspan="2">Co-train:All</td></tr><tr><td>GCN-ST</td><td>GCN-CT</td><td>MLP-ST</td><td>MLP-CT</td><td>SGC-CT</td><td>GCN-ST</td><td>GCN-CT</td><td>MLP-ST</td><td>MLP-CT</td><td>SGC-CT</td><td>GCN-CT</td><td>SGC-CT</td></tr><tr><td>75.20</td><td>69.97</td><td>71.62</td><td>68.33</td><td>72.51</td><td>81.32</td><td>81.93</td><td>71.89</td><td>71.83</td><td>82.9</td><td>76.31</td><td>80.01</td></tr></table>

# 4.2.2 Co-training for Link Prediction

Experiment Settings. Following the setting of node-level co-training, we adopt graphs from the CS Citation and E-commerce domains for co-training over link prediction tasks to consider the impact of data quantity and domain. For baseline models, we select GraphMAE as a representative for graph SSL, considering its superior performance compared to other SSL methods. We also include OFA and LLaGA, which apply to this paradigm. Since different datasets share the same task formulation, we also adopt end-to-end GCN, SEAL, and BUDDY for co-training (which can be seen as task-specific GFMs). Considering the efficiency of existing GFM pipelines, we first evaluate all methods under three small-scale datasets: Cora, CiteSeer, and Pubmed. Then, we extend scalable methods to co-train on larger graphs.

Results. The results of different methods on small-scale datasets are presented in Figure 4 and Table 3. GFM methods demonstrate no advantages compared to link prediction-specific models regarding efficiency and effectiveness. Comparing different GFMs, LLaGA achieves the best performance, but there is still a significant gap compared to link prediction-specific models like BUDDY. One reason for this phenomenon is that link prediction requires modeling the task-specific inductive bias revolving on the pairwise structural patterns [11], while these patterns are largely ignored by the GFM with a unified architecture across tasks. This also suggests that designing a task-specific GFM

for link prediction could be a promising direction. Meanwhile, we notice a considerable gap between SEAL and BUDDY, which suggests that properly incorporating structural embeddings is crucial for achieving optimal performance.

We further extend the scalable model BUDDY and GCN to larger graphs for co-training, with the results presented in Table 4. GraphMAE is omitted due to its poor performance on E-commerce datasets. The experimental results demonstrate that co-training significantly benefits models like BUDDY, which leverages suitable structural features. As a comparison, GCN achieves much worse performance and co-training shows no clear benefits compared to training on a single graph from scratch. Experimental results indicate that to achieve positive transferring from co-training on link prediction tasks, models need to incorporate proper inductive bias through structural embeddings. We note that even though SEAL's performance is not satisfying in Table 3, co-training still enhances performance across all downstream datasets. We summarize the aforementioned discussions with the following key observations.

Observation 2. GFM methods show no advantages over link prediction-specific models and struggle to scale to large graphs. Link prediction-specific models like BUDDY show great potential to benefit from co-training, highlighting the importance of proper structural embeddings. This also suggests that designing a task-specific GFM for link prediction could be a promising direction.

![](images/529b212e90a7dafe61b2e963db3e8b548fbc058b3982289da8f28d1a3f6ffbbb.jpg)

<details>
<summary>bar</summary>

| Model | Cora | CiteSeer | Pubmed |
|---|---|---|---|
| GCN | 79 | 87 | 69 |
| BUDDY | 89 | 96 | 84 |
| SEAL | 80 | 80 | 71 |
| MAE | 81 | 90 | 55 |
| OFA | 77 | 78 | 67 |
| LLaGA | 88 | 94 | 73 |
</details>

Figure 4: Comparison of different GFM and link prediction-specific models co-trained on three small-scale graphs. Hits@100 is adopted as the metric.   
Table 3: Comparison of different models' average performance trained on a single graph or co-trained on Cora, CiteSeer, Pubmed. We omit the feature preprocessing time for BUDDY and SEAL. 

<table><tr><td></td><td>Single-task</td><td>Co-train</td><td>Training time (s)</td></tr><tr><td>GCN</td><td>79.39</td><td>78.25</td><td>49</td></tr><tr><td>SEAL</td><td>70.40</td><td>76.78</td><td>3492</td></tr><tr><td>MAE</td><td>80.48</td><td>74.80</td><td>32</td></tr><tr><td>OFA</td><td>75.11</td><td>74.05</td><td>5431</td></tr><tr><td>LLaGA</td><td>81.32</td><td>85.03</td><td>7209</td></tr><tr><td>BUDDY</td><td>90.41</td><td>90.05</td><td>148</td></tr></table>

Table 4: Co-training BUDDY and GCN for link prediction at scale. -S means the model is trained on a single downstream task. -D means the model is trained on data coming from similar domains (shown in green and yellow). We use underline to emphasize the case where co-training benefits compared to training from scratch. 

<table><tr><td></td><td>Cora</td><td>Citeseer</td><td>Arxiv</td><td>Arxiv23</td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>Average</td></tr><tr><td>BUDDY-S</td><td>91.37</td><td>96.57</td><td>86.91</td><td>90.00</td><td>75.57</td><td>58.22</td><td>73.97</td><td>74.44</td><td>77.00</td><td>30.78</td><td>75.48</td></tr><tr><td>BUDDY-D</td><td>88.72</td><td> $\underline{97.28}$ </td><td>67.04</td><td>89.22</td><td> $\underline{91.17}$ </td><td> $\underline{86.25}$ </td><td> $\underline{90.42}$ </td><td> $\underline{88.72}$ </td><td> $\underline{93.6}$ </td><td> $\underline{69.34}$ </td><td> $\underline{86.18}$ </td></tr><tr><td>GCN-S</td><td>83.12</td><td>88.91</td><td>25.13</td><td>75.77</td><td>45.01</td><td>16.41</td><td>36.02</td><td>24.23</td><td>24.65</td><td>13.46</td><td>43.27</td></tr><tr><td>GCN-D</td><td>73.46</td><td>84.73</td><td> $\underline{47.61}$ </td><td> $\underline{81.75}$ </td><td>33.07</td><td>7.95</td><td> $\underline{38.55}$ </td><td> $\underline{41.69}$ </td><td>7.95</td><td>6.18</td><td>42.29</td></tr></table>

# 4.2.3 Co-training for Graph Classification

Table 5: Performance of graph-level co-training. Underline represents the best results on each dataset. 

<table><tr><td></td><td>PCBA</td><td>HIV</td><td>TOX21</td><td>BACE</td><td>BBBP</td><td>MUV</td><td>TOXCAST</td></tr><tr><td>Single (Atom)</td><td>0.202</td><td>75.49</td><td>74.6</td><td>72.4</td><td>65.7</td><td> $\underline{70.7}$ </td><td>61.5</td></tr><tr><td>Single (Text)</td><td>0.174</td><td>74.2</td><td>74.49</td><td>72.25</td><td>67.71</td><td>64.88</td><td>60.24</td></tr><tr><td>OFA</td><td> $\underline{0.236}$ </td><td> $\underline{75.24}$ </td><td> $\underline{82.5}$ </td><td> $\underline{77.32}$ </td><td> $\underline{69.97}$ </td><td>70.39</td><td> $\underline{68.39}$ </td></tr></table>

Experiment Settings. We adopt all available text-space datasets from the molecular domain for graph-level co-training. We adopt OFA, the only applicable model in graph-level co-training, as the baseline model. We compare models co-trained on different datasets with models trained from scratch on single datasets. We consider models trained on original atomic and text

features for the latter. It's important to note that our dataset primarily focuses on tasks related to molecular property prediction, and the results in other domains warrant further investigation.

Results. As shown in Table 5, co-training brings clear benefits for graph-level tasks. After unifying the feature and task formulation, models surpass the single dataset counterpart on all datasets after co-training. We also notice that in the single dataset case, there is a performance gap between the

model using LLM features and the model using original features. This indicates there's still some performance loss by transforming original attributes into text space. After co-training, the gap is eliminated, and the co-trained model based on text features performs better.

Observation 3. Co-training in the text space brings clear performance gain compared to training from scratch for graph classification tasks on molecular datasets.

# 4.3 Case 2: Co-training across tasks

We study the paradigm (Co-training, Cross-task) in this section. This setting is more challenging than task-specific co-training, requiring modeling shared principles across different tasks. We consider two settings: first, cross-task co-training happening on the same set of graphs, corresponding to node-level and link-level co-training, which we relegate to Appendix G.1.1. The second scenario is cross-task co-training across graphs, as shown below.

# 4.3.1 Co-training across node classification, link prediction, and graph classification

Co-training over node classification and link prediction still focuses on the paradigm of cross-task co-training on the same graph. We then investigate node, link, and graph-level co-training across different tasks and different graphs.

Experiment Settings. We adopt all datasets from node classification co-training for node-level datasets (Section 4.2.1, three small-scale datasets for link-level datasets (Section 4.2.2), and all datasets from graph classification co-training (Section 4.2.3) for graph-level datasets. Detailed settings can be found in Appendix G.1.1.

Results. As shown in Table 6, we observe that node and graph co-training, link and graph co-training, or node-link and graph co-training, all significantly improve graph-level performance, but they do not provide benefits for node-level or link-level tasks. Notably, co-training with tasks like link prediction that do not present node-level annotations can also significantly benefit graph-level tasks. However, co-training does not improve and may even degrade node-level performance.

Table 6: Performance of cross-data cross-task co-training. Link-Graph means co-training over link-level and graph-level tasks. The remaining two columns follow the same naming convention. We separate PCBA from the other graph datasets due to the significant difference in the scale of results. Underline means cross-task co-training benefits compared to single-task co-training. 

<table><tr><td colspan="4">Single task</td><td colspan="3">Link-Graph</td><td colspan="3">Node-Graph</td><td colspan="3">Link-Node-Graph</td></tr><tr><td>Link Avg</td><td>Node Avg</td><td>PCBA(G)</td><td>Graph Avg</td><td>Graph Avg</td><td>PCBA(G)</td><td>Link Avg</td><td>Graph Avg</td><td>PCBA(G)</td><td>Node Avg</td><td>Graph Avg</td><td>PCBA(G)</td><td>Node Avg</td></tr><tr><td>74.05</td><td>78.24</td><td>0.233</td><td>72.24</td><td> $\underline{75.04}$ </td><td> $\underline{0.265}$ </td><td>74.04</td><td> $\underline{75.86}$ </td><td> $\underline{0.282}$ </td><td>76.43</td><td> $\underline{75.48}$ </td><td> $\underline{0.279}$ </td><td>76.06</td></tr></table>

Observation 4. When co-training OFA on node classification, link prediction, and graph classification tasks across different datasets, the model's performance in graph classification will improve while its performance at link prediction and node classification may decline.

The primary reason behind this phenomenon is that OFA tends to learn inductive biases that are more suitable for graph-level tasks. In that way, the model leverages structural information in node-and link-level tasks to enhance graph-level performance. However, this emphasis on structure may introduce noise that can negatively impact performance on node-level tasks. We put the detailed discussion in Appendix G.1.2.

# 4.4 Case 3 & 4: Transferring from pre-training to downstream tasks

In this subsection, we consider the pre-training scenario, where the primary distinction from co-training lies in the absence of overlap between the pre-training and downstream datasets. Foundation models in other domains have demonstrated two potential capabilities in this setting: (1) the ability to enhance downstream task performance through a pre-train and fine-tune paradigm [4], and (2) the ability to learn in context [51] on downstream tasks.

General Experiment Settings. To assess the effectiveness of different GFMs, we adopt two evaluation protocols: in-context learning (zero-shot and few-shot) and fine-tuning. For in-context learning, we assume the downstream task has no labels (zero-shot) or only k labels per class (fewshot, k = 3 in this section). For fine-tuning, we assume the same labeling rate as in co-training. Specifically, we utilize the three largest node-level datasets, Arxiv, Sportsfit, and Products, as the pre-training data. We then employ Cora, History, and Amazon ratings to evaluate node classification downstream task performance, PubMed for link prediction task performance, and HIV for graph classification task

performance. For graph SSL, we use the level of the labels provided by the train data as the level for pre-training because the learned representation will conform to the corresponding inductive bias [52].

Table 7: Performance of transferring across the same task. The N/A in the table indicate that the model is not applicable to this setting. 

<table><tr><td></td><td colspan="3">Cora</td><td colspan="3">History</td><td colspan="3">Ratings</td></tr><tr><td></td><td>0 shot</td><td>3 shot</td><td>FT</td><td>0 shot</td><td>3 shot</td><td>FT</td><td>0 shot</td><td>3 shot</td><td>FT</td></tr><tr><td>GraphMAE</td><td>N/A</td><td>72.49</td><td>81.8</td><td>N/A</td><td>39.15</td><td>83.68</td><td>N/A</td><td>31.68</td><td>41.06</td></tr><tr><td>LLaGA</td><td>18.25</td><td>60.7</td><td>80.45</td><td>22.05</td><td>36.45</td><td>82.55</td><td>23.15</td><td>23.45</td><td>28.2</td></tr><tr><td>OFA</td><td>30.42</td><td>52.49</td><td>74.27</td><td>22.98</td><td>39.36</td><td>83.53</td><td>21.72</td><td>29.08</td><td>51.44</td></tr><tr><td>OFA-FS</td><td>20.3</td><td>42.1</td><td>N/A</td><td>13.82</td><td>17.5</td><td>N/A</td><td>21.5</td><td>20.5</td><td>N/A</td></tr><tr><td>Prodigy (MAG240M)</td><td>N/A</td><td>73.96</td><td>N/A</td><td>N/A</td><td>54.45</td><td>N/A</td><td>N/A</td><td>71.28</td><td>N/A</td></tr><tr><td>Prodigy</td><td>N/A</td><td>71.60</td><td>N/A</td><td>N/A</td><td>52.66</td><td>N/A</td><td>N/A</td><td>67.41</td><td>N/A</td></tr><tr><td>Simple SBERT (no pretrain)</td><td>67.41</td><td>68.42</td><td>82.2</td><td>59.25</td><td>51.25</td><td>85.3</td><td>27.39</td><td>20.95</td><td>48.46</td></tr></table>

Table 8: Performance of transferring across different tasks. We use Hits@100 for link prediction and AUC-ROC for graph classification.

<table><tr><td rowspan="2"></td><td colspan="2">Pubmed</td><td colspan="3">HIV</td></tr><tr><td>0 shot</td><td>FT</td><td>0 shot</td><td>3 shot</td><td>FT</td></tr><tr><td>GraphMAE</td><td>N/A</td><td>36.83</td><td>N/A</td><td>56.68</td><td>65.39</td></tr><tr><td>LLaGA</td><td>14</td><td>74</td><td>NN/AA</td><td>NN/AA</td><td>NN/AA</td></tr><tr><td>OFA</td><td>0.49</td><td>68.7</td><td>49.73</td><td>50.41</td><td>75.35</td></tr><tr><td>OFA-FS</td><td>4.56</td><td>N/A</td><td>47.8</td><td>50.56</td><td>N/A</td></tr><tr><td>Simple SBERT (no pretrain)</td><td>49.28</td><td>66.14</td><td>NN/AA</td><td>NN/AA</td><td>74.2</td></tr></table>

# 4.4.1 Case 3: Transferring across the same tasks.

Experiment Settings. We start from the case where transferring happens between the same task. Since the pre-training dataset contains node-level labels, we evaluate the node-classification task. Following the general settings in Section 4.4, we select baseline models GraphMAE, LLaGA, OFA, and Prodigy applicable to the transferring settings. For OFA, we consider the normal prompt version and one designed for few-shot inference. For Prodigy, we consider the version pre-trained on the same dataset as other baselines or the one trained on MAG240M as in the original paper. To evaluate the effectiveness of pre-training, we consider the “simple SBERT” method, which conducts no pre-training. For zero-shot inference, it first propagates the features according to graph structures and then uses cosine similarity between propagated and label embeddings to do the inference. For few-shot and fine-tuning, it trains a task-specific GCN from scratch.

Results. The results are shown in Table 7; we first notice that the “pre-training, fine-tuning” paradigm doesn’t present clear benefits with only marginal gain on heterophilous dataset Amazon ratings, which may be explained by the fact that LLM embeddings are powerful enough to achieve good downstream task performance with only a small number of labels, rendering pre-training not that helpful for providing additional information. The most surprising phenomenon lies in the in-context learning setting, where we observe a large gap between OFA and Prodigy, which are both based on graph prompt designs. The key difference lies in whether cosine similarity is adopted in the inference stage, which can be seen from the zero-shot performance of “simple SBERT”. This method works well and surpasses OFA by a large margin on Cora and History. On the heterophilous dataset Amazon ratings, it still presents a large gap compared to Prodigy, which indicates:

Observation 5. Relying on textural space, LLM embedding, and cosine similarity are the keys to effective in-context learning. Graph prompt-based in-context learning demonstrates great potential for tackling heterophilous graphs.

# 4.4.2 Case 4: Transferring across tasks.

Experiment Settings. We then study the paradigm where transferring happens across different tasks. We select GraphMAE, LLaGA, OFA, and our proposed simple baselines in Section 4.4.1.

Results. As shown in Table 8, we observe that (1) existing GFMs present limited capability in cross-task in-context learning. (2) After fine-tuning, we observe positive transferring from node classification to graph classification, which is consistent with our observation in Table 6.

Observation 6. We are still far away from cross-graph, cross-task in-context learning.

# 5 Conclusion

This paper presents a novel benchmark designed for developing text-space GFMs, which comprise novel datasets, comprehensive evaluation under diverse settings, and novel insights. Our key findings can be summarized as follows: the effectiveness of text-space GFM is built on three conditions: (1) LLM embeddings provide a feature space mitigating severe negative transferring. (2) GFM models can extract transferrable patterns across different graphs. (3) GFM backbones present appropriate inductive biases designed for downstream tasks. Insights from our work can potentially inspire research in diverse areas, including E-commerce, social networks, and natural science. We present a thorough discussion on broader impacts in Appendix I.

# References

[1] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. Cited on page 1.   
[2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. Cited on pages 1 and 2.   
[3] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015-4026, 2023. Cited on page 1.   
[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. Cited on pages 1 and 8.   
[5] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023. Cited on page 1.   
[6] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019. Cited on page 1.   
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. Cited on page 1.   
[8] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Yuan Yao, Ao Zhang, Liang Zhang, et al. Pre-trained models: Past, present and future. AI Open, 2:225–250, 2021. Cited on page 1.   
[9] Haitao Mao, Guangliang Liu, Yao Ma, Rongrong Wang, and Jiliang Tang. A data generation perspective to the mechanism of in-context learning. arXiv preprint arXiv:2402.02212, 2024. Cited on page 1.   
[10] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022. Cited on page 1.   
[11] Haitao Mao, Zhikai Chen, Wenzhuo Tang, Jianan Zhao, Yao Ma, Tong Zhao, Neil Shah, Michael Galkin, and Jiliang Tang. Graph foundation models. arXiv preprint arXiv:2402.02216, 2024. Cited on pages 1, 3, 5, 6, and 18.   
[12] Jonathan Halcrow, Alexandru Mosoi, Sam Ruth, and Bryan Perozzi. Grale: Designing networks for graph learning. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2523–2532, 2020. Cited on page 1.   
[13] Shangbin Feng, Herun Wan, Ningnan Wang, and Minnan Luo. Botrgcn: Twitter bot detection with relational graph convolutional networks. In Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, pages 236–239, 2021. Cited on page 1.

[14] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 974–983, 2018. Cited on page 1.   
[15] Fedor Borisyuk, Shihai He, Yunbo Ouyang, Morteza Ramezani, Peng Du, Xiaochen Hou, Chengming Jiang, Nitin Pasumarthy, Priya Bannur, Birjodh Tiwana, et al. Lignn: Graph neural networks at linkedin. arXiv preprint arXiv:2402.11139, 2024. No citations.   
[16] Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. Graph neural networks for social recommendation. In The world wide web conference, pages 417–426, 2019. Cited on page 1.   
[17] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? In Thirty-Fifth Conference on Neural Information Processing Systems, 2021. Cited on page 1.   
[18] Jingzhe Liu, Haitao Mao, Zhikai Chen, Tong Zhao, Neil Shah, and Jiliang Tang. Neural scaling laws on graphs. arXiv preprint arXiv:2402.02054, 2024. Cited on pages 1 and 3.   
[19] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. One for all: Towards training one graph model for all classification tasks. arXiv preprint arXiv:2310.00149, 2023. Cited on pages 1, 2, 3, 4, 5, 17, 18, 21, 22, and 23.   
[20] Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, and Jure Leskovec. Prodigy: Enabling in-context learning over graphs. arXiv preprint arXiv:2305.12600, 2023. Cited on pages 1, 2, 3, 4, 5, 17, 18, and 22.   
[21] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi. Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning, 2023. Cited on pages 2, 19, and 21.   
[22] Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Haifang Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, and Jiliang Tang. Exploring the potential of large language models (llms) in learning on graphs. ArXiv, abs/2307.03393, 2023. Cited on pages 2, 4, 5, 19, 21, and 23.   
[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. Cited on pages 2 and 4.   
[24] Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, and Qi Liu. Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning. bioRxiv, pages 2023–05, 2023. Cited on pages 2, 3, 4, and 21.   
[25] Runjin Chen, Tong Zhao, Ajay Jaiswal, Neil Shah, and Zhangyang Wang. Llaga: Large language and graph assistant. arXiv preprint arXiv:2402.08170, 2024. Cited on pages 2, 4, 5, 17, 18, 22, and 23.   
[26] Yixin Liu, Ming Jin, Shirui Pan, Chuan Zhou, Yu Zheng, Feng Xia, and S Yu Philip. Graph self-supervised learning: A survey. IEEE Transactions on Knowledge and Data Engineering, 35(6):5879–5900, 2022. Cited on pages 2, 3, and 4.   
[27] Wenqi Fan, Shijie Wang, Jiani Huang, Zhikai Chen, Yu Song, Wenzhuo Tang, Haitao Mao, Hui Liu, Xiaorui Liu, Dawei Yin, et al. Graph machine learning in the era of large language models (llms). arXiv preprint arXiv:2404.14928, 2024. Cited on pages 3 and 18.   
[28] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In Proceedings of the ACM Web Conference 2023, pages 417–428, 2023. Cited on page 3.   
[29] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. arXiv preprint arXiv:2310.13023, 2023. Cited on pages 4 and 17.

[30] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. Graphllm: Boosting graph reasoning ability of large language model. arXiv preprint arXiv:2310.05845, 2023. No citations.   
[31] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862, 2024. Cited on pages 4 and 18.   
[32] Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, et al. A comprehensive study on text-attributed graphs: Benchmarking and rethinking. Advances in Neural Information Processing Systems, 36:17238–17264, 2023. Cited on pages 4, 19, and 21.   
[33] Bharath Ramsundar, Peter Eastman, Patrick Walters, Vijay Pande, Karl Leswing, and Zhenqin Wu. Deep Learning for the Life Sciences. O'Reilly Media, 2019. https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837. Cited on pages 4, 5, and 21.   
[34] Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of gnns under heterophily: Are we really making progress? arXiv preprint arXiv:2302.11640, 2023. Cited on pages 4 and 21.   
[35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Cited on pages 4 and 20.   
[36] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(25):723–773, 2012. Cited on page 4.   
[37] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. Cited on page 4.   
[38] Petar Veličković, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. Deep graph infomax. arXiv preprint arXiv:1809.10341, 2018. Cited on pages 4 and 23.   
[39] Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, and Jie Tang. Gcc: Graph contrastive coding for graph neural network pre-training. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1150–1160, 2020. Cited on page 4.   
[40] Shantanu Thakoor, Corentin Tallec, Mohammad Gheshlaghi Azar, Mehdi Azabou, Eva L Dyer, Remi Munos, Petar Veličković, and Michal Valko. Large-scale representation learning on graphs via bootstrapping. arXiv preprint arXiv:2102.06514, 2021. Cited on pages 4 and 23.   
[41] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. Graphmae: Self-supervised masked graph autoencoders. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 594–604, 2022. Cited on pages 4 and 23.   
[42] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graphsaint: Graph sampling based inductive learning method. arXiv preprint arXiv:1907.04931, 2019. Cited on page 4.   
[43] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. Cited on page 5.   
[44] Benjamin Paul Chamberlain, Sergey Shirobokov, Emanuele Rossi, Fabrizio Frasca, Thomas Markovich, Nils Hammerla, Michael M Bronstein, and Max Hansmire. Graph neural networks for link prediction with subgraph sketching. arXiv preprint arXiv:2209.15486, 2022. Cited on pages 5 and 23.

[45] Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, and Long Jin. Labeling trick: A theory of using graph neural networks for multi-node representation learning. Advances in Neural Information Processing Systems, 34:9061–9073, 2021. Cited on pages 5 and 23.   
[46] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019. Cited on pages 5 and 24.   
[47] Juanhui Li, Harry Shomer, Haitao Mao, Shenglai Zeng, Yao Ma, Neil Shah, Jiliang Tang, and Dawei Yin. Evaluating graph neural networks for link prediction: Current pitfalls and new benchmarking. arXiv preprint arXiv:2306.10453, 2023. Cited on page 5.   
[48] Boshen Shi, Yongqing Wang, Fangda Guo, Bingbing Xu, Huawei Shen, and Xueqi Cheng. Graph domain adaptation: Challenges, progress and prospects. arXiv preprint arXiv:2402.00904, 2024. Cited on page 5.   
[49] Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language models. arXiv preprint arXiv:2309.10313, 2023. Cited on page 6.   
[50] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In International conference on machine learning, pages 6861–6871. PMLR, 2019. Cited on page 6.   
[51] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022. Cited on page 8.   
[52] Nian Liu, Xiao Wang, Deyu Bo, Chuan Shi, and Jian Pei. Revisiting graph contrastive learning from the perspective of graph spectrum. Advances in Neural Information Processing Systems, 35:2972–2983, 2022. Cited on page 9.   
[53] Yufei He and Bryan Hooi. Unigraph: Learning a cross-domain graph foundation model from natural language. arXiv preprint arXiv:2402.13630, 2024. Cited on page 17.   
[54] Jiawei Liu, Cheng Yang, Zhiyuan Lu, Junze Chen, Yibo Li, Mengmei Zhang, Ting Bai, Yuan Fang, Lichao Sun, Philip S Yu, et al. Towards graph foundation models: A survey and beyond. arXiv preprint arXiv:2310.11829, 2023. Cited on page 18.   
[55] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation models for knowledge graph reasoning. arXiv preprint arXiv:2310.04562, 2023. Cited on page 18.   
[56] Luis Müller, Mikhail Galkin, Christopher Morris, and Ladislav Rampášek. Attending to graph transformers. arXiv preprint arXiv:2302.04181, 2023. Cited on pages 18 and 26.   
[57] Chen Cai, Truong Son Hy, Rose Yu, and Yusu Wang. On the connection between mpnn and graph transformer. In International Conference on Machine Learning, pages 3408–3430. PMLR, 2023. Cited on page 18.   
[58] Jiarong Xu, Renhong Huang, Xin Jiang, Yuxuan Cao, Carl Yang, Chunping Wang, and Yang Yang. Better with less: A data-active perspective on pre-training graph neural networks. arXiv preprint arXiv:2311.01038, 2023. Cited on page 18.   
[59] Alex O Davies, Riku W Green, Nirav S Ajmeri, et al. Its all graph to me: Foundational topology models with contrastive learning on multiple domains. arXiv preprint arXiv:2311.03976, 2023. Cited on page 18.   
[60] Jun Xia, Chengshuai Zhao, Bozhen Hu, Zhangyang Gao, Cheng Tan, Yue Liu, Siyuan Li, and Stan Z. Li. Mole-BERT: Rethinking pre-training graph neural networks for molecules. In The Eleventh International Conference on Learning Representations, 2023. Cited on page 18.

[61] Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena, Dávid P. Kovács, Janosh Riebesell, Xavier R. Advincula, Mark Asta, William J. Baldwin, Noam Bernstein, Arghya Bhowmik, Samuel M. Blau, Vlad Cărare, James P. Darby, Sandip De, Flaviano Della Pia, Volker L. Deringer, Rokas Elijošius, Zakariya El-Machachi, Edvin Fako, Andrea C. Ferrari, Annalena Genreith-Schriever, Janine George, Rhys E. A. Goodall, Clare P. Grey, Shuang Han, Will Handley, Hendrik H. Heenen, Kersti Hermansson, Christian Holm, Jad Jaafar, Stephan Hofmann, Konstantin S. Jakob, Hyunwook Jung, Venkat Kapil, Aaron D. Kaplan, Nima Karimitari, Namu Kroupa, Jolla Kullgren, Matthew C. Kuner, Domantas Kuryla, Guoda Liepuoniute, Johannes T. Margraf, Ioan-Bogdan Magdău, Angelos Michaelides, J. Harry Moore, Aakash A. Naik, Samuel P. Niblett, Sam Walton Norwood, Niamh O'Neill, Christoph Ortner, Kristin A. Persson, Karsten Reuter, Andrew S. Rosen, Lars L. Schaaf, Christoph Schran, Eric Sivonxay, Tamás K. Stenczel, Viktor Svahn, Christopher Sutton, Cas van der Oord, Eszter Varga-Umbrich, Tejs Vegge, Martin Vondrák, Yangshuai Wang, William C. Witt, Fabian Zills, and Gábor Csányi. A foundation model for atomistic materials chemistry, 2023. Cited on page 18.   
[62] Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. arXiv preprint arXiv:2405.18512, 2024. Cited on page 18.   
[63] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A comprehensive survey. arXiv preprint arXiv:2312.02783, 2023. Cited on page 18.   
[64] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. In The Twelfth International Conference on Learning Representations, 2024. Cited on page 18.   
[65] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv preprint arXiv:2402.07630, 2024. Cited on page 18.   
[66] Mengmei Zhang, Mingwei Sun, Peng Wang, Shen Fan, Yanhu Mo, Xiaoxiao Xu, Hong Liu, Cheng Yang, and Chuan Shi. Graphtranslator: Aligning graph model to large language model for open-ended tasks. In Proceedings of the ACM on Web Conference 2024, pages 1003–1014, 2024. Cited on page 18.   
[67] Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, and Julian McAuley. Instructgraph: Boosting large language models via graph-centric instruction tuning and preference alignment. arXiv preprint arXiv:2402.08785, 2024. Cited on page 18.   
[68] Eli Chien, Wei-Cheng Chang, Cho-Jui Hsieh, Hsiang-Fu Yu, Jiong Zhang, Olgica Milenkovic, and Inderjit S Dhillon. Node feature extraction by self-supervised multi-scale neighborhood prediction. arXiv preprint arXiv:2111.00064, 2021. Cited on page 19.   
[69] Keyu Duan, Qian Liu, Tat-Seng Chua, Shuicheng Yan, Wei Tsang Ooi, Qizhe Xie, and Junxian He. Simteg: A frustratingly simple approach improves textual graph learning. arXiv preprint arXiv:2308.02565, 2023. Cited on page 19.   
[70] Michihiro Yasunaga, Jure Leskovec, and Percy Liang. Linkbert: Pretraining language models with document links. arXiv preprint arXiv:2203.15827, 2022. Cited on page 19.   
[71] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. Learning on large-scale text-attributed graphs via variational inference. arXiv preprint arXiv:2210.14709, 2022. Cited on page 19.   
[72] Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, and Xing Xie. Graphformers: Gnn-nested transformers for representation learning on textual graph. Advances in Neural Information Processing Systems, 34:28798–28810, 2021. Cited on page 19.   
[73] Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D. Manning, Percy Liang, and Jure Leskovec. Deep bidirectional language-knowledge graph pretraining. In Neural Information Processing Systems (NeurIPS), 2022. Cited on page 19.

[74] Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, and Jiliang Tang. Label-free node classification on graphs with large language models (llms). arXiv preprint arXiv:2310.04668, 2023. Cited on page 19.   
[75] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In International conference on machine learning, pages 40–48. PMLR, 2016. Cited on page 21.   
[76] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118–22133, 2020. Cited on page 21.   
[77] Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 188–197, 2019. Cited on page 21.   
[78] Haitao Mao, Zhikai Chen, Wei Jin, Haoyu Han, Yao Ma, Tong Zhao, Neil Shah, and Jiliang Tang. Demystifying structural disparity in graph neural networks: Can one size fit all? arXiv preprint arXiv:2306.01323, 2023. Cited on page 21.   
[79] Ming Ji, Yizhou Sun, Marina Danilevsky, Jiawei Han, and Jing Gao. Graph regularized transductive classification on heterogeneous information networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 570–586. Springer, 2010. Cited on page 21.   
[80] Qifang Zhao, Weidong Ren, Tianyu Li, Xiaoxiao Xu, and Hong Liu. Graphgpt: Graph learning with generative pre-trained transformers. arXiv preprint arXiv:2401.00529, 2023. Cited on page 26.

# Appendix

# Table of Contents

A More backgrounds of our benchmark 17

A.1 Components of our benchmark 17   
A.2 Comparison between our benchmark and existing works 17

B An Empirical Investigation into Performance Degradation from Projecting into Text Space 17

C Comprehensive Related Works 18

C.1 Graph Foundation Models (GFMs) 18   
C.2 Learning over Text-attributed Graphs 18

D Datasets 19

D.1 Inspection of Node-level and Link-level datasets 20   
D.2 Dataset Introduction 20

E Detailed Experimental Settings 21

E.1 Hyperparameter Settings. 22

F Implementations 22   
G Extended Experimental Results 23

G.1 More results for co-training setting 23

G.2 Effects of different LLM Encoders 24   
G.3 Extended results of co-training over link-prediction tasks ..... 24   
G.4 Co-training on heterophilous graphs 24   
G.5 Effects of dataset scales on co-training 25

H Limitations and future works 25

I Broader Impacts 26

# A More backgrounds of our benchmark

# A.1 Components of our benchmark

As shown in 5, our benchmarks compose diverse datasets, implementation of GFM building blocks, and comprehensive evaluation.

![](images/5d80a9ee3041c7e267f09d60ab293a10ddb9fc69826c87a60f86e0d9c2ee1237.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Academic Graphs"] --> B["Ecommerce Graphs"]
    C["Molecular Graphs"] --> D["Wiki Graphs"]
    E["Text-space Dataset"] --> F["Text-space GFM Benchmark"]
    G["Label Space"] --> H["Graph Prompts"]
    I["Graph SSL"] --> J["Response Language model"]
    K["Projector"] --> L["Graph Encoder"]
    M["Graph LLM"] --> N["Language"]
    O["GFM Building Blocks"] --> P["Scenarios"]
    Q["Co-training"] --> R["Pre-training"]
    S["Task levels"] --> T["Node"]
    S --> U["Link"]
    S --> V["Graph"]
    W["Settings"] --> X["Zero-shot"]
    W --> Y["Few-shot"]
    W --> Z["Fine-tune"]
    AA["Evaluation"] --> AB["Evaluation"]
```
</details>

Figure 5: Our benchmark comprises three main components: (1) Diverse text-space datasets: Covering 23 text-space datasets from diverse domains; (2) GFM building block: Implementation of mainstream techniques to build GFMs; (3) Comprehensive Evaluation: We propose four use cases to evaluate the performance of GFMs thoroughly.

# A.2 Comparison between our benchmark and existing works

Table 9: Comparison between our benchmark and existing works: We present many more text-space datasets, based on which we consider comprehensive problem settings of GFM. We adopt graph SSL and link prediction-specific methods, which have often been overlooked in other works. We also employ reasonable experimental settings, such as comparing GFMs with GNNs using LLM embeddings and ensuring no test edge leakage in link prediction evaluation. Finally, we propose new understandings based on reliable experimental results.

<table><tr><td></td><td>Diverse Datasets</td><td>Comprehensive settings</td><td>Comprehensive Baselines</td><td>Comprehensive Evaluation</td><td>Understanding</td></tr><tr><td>GraphGPT [29]</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td></tr><tr><td>LLaGA [25]</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td></tr><tr><td>Prodigy [20]</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td></tr><tr><td>OneForAll [19]</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td></tr><tr><td>UniGraph [53]</td><td> $\times$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\times$ </td><td> $\times$ </td></tr><tr><td>Ours</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td></tr></table>

# B An Empirical Investigation into Performance Degradation from Projecting into Text Space

In this section, we empirically evaluate the performance loss when transforming diverse kinds of attributes into text space. Specifically, we evaluate the following cases as shown in Table 10.

Table 10: Performance comparison between models trained on original attribute space and text space 

<table><tr><td></td><td>Original Attributes</td><td>Task</td><td>Metric</td><td>Original Performance</td><td>Text-space Performance</td></tr><tr><td>Arxiv</td><td>Word2Vec</td><td>Node Classification</td><td>Accuracy</td><td>71.53</td><td>73.10</td></tr><tr><td>HIV</td><td>Atomic Numbers</td><td>Graph Classification</td><td>AUC-ROC</td><td>75.52</td><td>74.20</td></tr><tr><td>Tolokers</td><td>Categorical</td><td>Node Classification</td><td>Accuracy</td><td>83.25</td><td>78.16</td></tr><tr><td>Pubmed</td><td>TF-IDF</td><td>Link Prediction</td><td>Hits@100</td><td>53.05</td><td>66.13</td></tr></table>

The experimental results demonstrate:

- For text attributes, LLMs can generate better-quality embeddings and empower tasks like node classification and link prediction.   
- For non-text attributes like atomic numbers and categorical values, high-quality text prompts can achieve comparable performance.

# C Comprehensive Related Works

# C.1 Graph Foundation Models (GFMs)

Despite the diverse definitions and scopes of existing GFMs $[54, 55, 11]$ , one core criterion defining a GFM is the capability to empower a series of graph-related tasks with a unified backbone. This unified backbone can either be trained from scratch, making it a graph-centric GFM $[54]$ ; or it can be adapted from an existing foundation model (mostly LLM), making it an LLM-induced GFM $[27]$ .

Graph-centric GFM's scope is mostly focused on traditional graph machine learning tasks, and its core philosophy is to unify diverse data and tasks to enlarge the training data, thereby empowering models' capability and also enabling "one model serves all". The core challenge lies in the diversity of graph structures and node features. Tackling diverse graph structures is a trending topic in today's graph machine learning, with models focused on proposing backbones [56] with more flexible inductive biases or enhancing existing backbones to tackle more diverse structures [57]. However, feature heterogeneity has been studied less, and there is currently no solution that can handle all different scenarios well. At the same time, feature heterogeneity is so critical that without a unified feature space, it's impossible to train a GFM. To tackle this issue, some approaches have either ignored feature information altogether, leading to significant performance drops on text-attributed graphs [58, 59], or constructed domain-specific feature spaces for knowledge graphs or molecules, limiting their generalizability [55, 60, 61]. In contrast, Text-space GFM, which transforms diverse attributes into texts and then adopts large language models (LLMs) as encoders, [20, 19, 25], provides a unified feature space that can generalize to a wide range of graphs [19] and demonstrate impressive performance. Despite the preliminary success, our understanding of text-space GFM is still pretty limited. For instance, the tasks on which they have better effectiveness and under what circumstances they can achieve positive transfer, these gaps in understanding motivate us to conduct this benchmark. Moreover, the text space also presents limitations. While most node features can be converted into text, sometimes, this can lead to significant performance loss. Additionally, how to model the interaction between graph structure and LLM features remains an open research question. Another problem lies in the capability of the backbone. Despite the wide applicability of message-passing NN in diverse applications, they still present fundamental capacity limitations, which are addressed by Transformer architectures [62].

Apart from graph-centric GFM, LLM-induced GFM $[63]$ aims to handle various tasks through the inherent multi-task processing capability of LLMs and leveraging language and next token prediction as a natural medium to unify diverse tasks. Their primary focus is on language-centric tasks with certain graph structures, such as graph-based QA tasks, like the GraphQA dataset $[64]$ . These works, like $[65, 31, 64]$ focus on graph-related QA tasks and adapt existing LLMs to answer a series of questions related to graph structures. Specifically, these models demonstrate task generalization capabilities to unseen tasks. $[66]$ adopts a plugin module to "translate" graphs into text, after which LLMs can answer structure-aware open-ended tasks, including traditional node classification and GraphQA tasks. $[67]$ goes one step further by unifying all graphs into texts. It then adopts instruction tuning to align LLMs with these graph representations better, enabling them to tackle a wide range of graph-related tasks. Despite the general capability of these models, they still exhibit limited capability on traditional graph machine learning tasks, especially those where structure plays an important role, like link prediction and graph classification.

# C.2 Learning over Text-attributed Graphs

After unifying diverse features in the text space, the augmented graph naturally becomes a text-attributed graph (TAG), making relevant techniques for TAG applicable to text-space GFMs. The core challenge of learning over TAG lies in integrating node features and graph structures. LLM is adopted from the feature side due to its superior performance in text processing. From the structure

side, graph neural networks have become the de facto approach for handling graph-structured data. As a result, the main research objective for learning over TAGs is how to integrate these two models.

One basic approach to address this problem is to cascade the two models, forming a cascading structure $[68, 22, 69, 70]$ . Specifically, embeddings are generated through an LLM, whose parameters are then fixed, followed by training a GNN. To better adapt the LLM to specific data, some works propose using domain adaptive pretraining $[68, 69]$ to generate embeddings more aligned with the downstream task. The drawback of the cascading structure is the tenuous connection between the LLM and GNN, as LLM does not consider the influence of graph structure when generating embeddings. Therefore, structure-aware joint learning has been proposed $[71–73]$ . $[72]$ introduces a framework for co-training LLM and graph GNN by leveraging each other's generated embeddings. $[71]$ extends this approach by incorporating pseudo-labels generated by both LLMs and GNNs into the optimization process, thus further enhancing the co-training capabilities of the two model types. To better understand the effectiveness of joint learning structure, $[32]$ conducts a benchmark to evaluate different approaches to joint LLM-GNN learning. Despite the claiming superiority of joint learning over cascading structures, experimental results $[32, 22, 69]$ show that with proper LLM selections, cascading structures can achieve better performance with significantly lower computational overhead. This has led to cascading structures becoming the widely adopted design in text-space GFM.

Beyond model-centric research, [21] enhances the effectiveness of learning over TAGs from a data perspective. Specifically, it augments the original node features by generating additional explanation through an LLM. In Section 4.2.1, we observe that co-training has limited improvement on node classification, suggesting that data augmentation may be an effective means further to enhance GFM's performance in node classification tasks. [74] focuses on addressing zero-shot learning on TAGs. Combining the inherent zero-shot capabilities of LLMs with an active learning framework can effectively solve node classification problems on TAGs without manual annotation.

# D Datasets

Details of adopted datasets are presented in Table 11, for the number of edges, we consider all graph as undirected graph and remove all self-loops.

Table 11: Details of our selected datasets. For Products, we sample a subset of the original dataset since subgraph-based GFM methods are hard to scale to datasets with millions of nodes. Datasets with \* are not adopted in co-training and transferring experiments. 

<table><tr><td>Name</td><td>#Graphs</td><td>#Nodes</td><td>#Edges</td><td>Domains</td><td>Tasks</td><td>#Classes</td><td>Metrics</td></tr><tr><td>Cora</td><td>1</td><td>2708</td><td>10556</td><td>CS Citation</td><td>Node, Link</td><td>7</td><td>Accuracy, Hits@100</td></tr><tr><td>CiteSeer</td><td>1</td><td>3186</td><td>8450</td><td>CS Citation</td><td>Node, Link</td><td>6</td><td>Accuracy, Hits@100</td></tr><tr><td>Arxiv</td><td>1</td><td>169343</td><td>2315598</td><td>CS Citation</td><td>Node, Link</td><td>40</td><td>Accuracy, Hits@100</td></tr><tr><td>Arxiv23</td><td>1</td><td>46198</td><td>77726</td><td>CS Citation</td><td>Node, Link</td><td>40</td><td>Accuracy, Hits@100</td></tr><tr><td>History</td><td>1</td><td>41551</td><td>503180</td><td>E-commerce</td><td>Node, Link</td><td>12</td><td>Accuracy, Hits@100</td></tr><tr><td>Child</td><td>1</td><td>76875</td><td>2325044</td><td>E-commerce</td><td>Node, Link</td><td>24</td><td>Accuracy, Hits@100</td></tr><tr><td>Computers</td><td>1</td><td>87229</td><td>1256548</td><td>E-commerce</td><td>Node, Link</td><td>10</td><td>Accuracy, Hits@100</td></tr><tr><td>Photo</td><td>1</td><td>48362</td><td>873782</td><td>E-commerce</td><td>Node, Link</td><td>12</td><td>Accuracy, Hits@100</td></tr><tr><td>Sportsfit</td><td>1</td><td>173055</td><td>3020134</td><td>E-commerce</td><td>Node, Link</td><td>13</td><td>Accuracy, Hits@100</td></tr><tr><td>Products</td><td>1</td><td>316513</td><td>19337722</td><td>E-commerce</td><td>Node, Link</td><td>39</td><td>Accuracy, Hits@100</td></tr><tr><td>Amazon Ratings</td><td>1</td><td>24492</td><td>186100</td><td>E-commerce</td><td>Node, Link</td><td>5</td><td>Accuracy, Hits@100</td></tr><tr><td>Pubmed</td><td>1</td><td>19717</td><td>88648</td><td>Bio Citation</td><td>Node, Link</td><td>3</td><td>Accuracy, Hits@100</td></tr><tr><td>WikiCS</td><td>1</td><td>11701</td><td>431726</td><td>Knowledge</td><td>Node, Link</td><td>10</td><td>Accuracy, Hits@100</td></tr><tr><td>Tolokers(*)</td><td>1</td><td>11758</td><td>1038000</td><td>Anomaly</td><td>Node, Link</td><td>2</td><td>Accuracy, Hits@100</td></tr><tr><td>DBLP(*)</td><td>1</td><td>14376</td><td>431326</td><td>CS Citation</td><td>Node, Link</td><td>4</td><td>Accuracy, Hits@100</td></tr><tr><td>CheMBL</td><td>365065</td><td>26</td><td>112</td><td>Biology</td><td>Graph</td><td>1048</td><td>Not used for downstream tasks</td></tr><tr><td>PCBA</td><td>437092</td><td>26</td><td>56</td><td>Biology</td><td>Graph</td><td>128</td><td>AP</td></tr><tr><td>HIV</td><td>41127</td><td>26</td><td>55</td><td>Biology</td><td>Graph</td><td>2</td><td>ROC-AUC</td></tr><tr><td>Tox21</td><td>7831</td><td>19</td><td>39</td><td>Biology</td><td>Graph</td><td>12</td><td>ROC-AUC</td></tr><tr><td>Bace</td><td>1513</td><td>34</td><td>74</td><td>Biology</td><td>Graph</td><td>2</td><td>ROC-AUC</td></tr><tr><td>Bbbp</td><td>2039</td><td>24</td><td>52</td><td>Biology</td><td>Graph</td><td>2</td><td>ROC-AUC</td></tr><tr><td>Muv</td><td>93087</td><td>24</td><td>53</td><td>Biology</td><td>Graph</td><td>17</td><td>ROC-AUC</td></tr><tr><td>Toxcast</td><td>8575</td><td>19</td><td>39</td><td>Biology</td><td>Graph</td><td>588</td><td>ROC-AUC</td></tr></table>

# D.1 Inspection of Node-level and Link-level datasets

In this section, we demonstrate the inspection of text feature similarity of different node-level and link-level datasets.

For the first group of inspection, we compare CS citation graphs, Pubmed, and WikiCS. For the second group of inspection, we compare E-commerce graphs and Amazon ratings.

![](images/6990241c7035bf91d471855d6f68081003c2328f164783fced9ef75530fb6c24.jpg)

<details>
<summary>heatmap</summary>

Average Pairwise Feature Similarity (MMD)
| Dataset | cora_n | citese | pubmed | arxiv | arxiv2 | wikics |
|---|---|---|---|---|---|---|
| cora_n | 0.00 | -0.20 | -0.94 | -0.14 | -0.23 | -0.49 |
| citese | -0.20 | 0.00 | -0.91 | -0.18 | -0.24 | -0.34 |
| pubmed | -0.94 | -0.91 | 0.00 | -0.80 | -0.82 | -0.94 |
| arxiv | -0.14 | -0.18 | -0.80 | 0.00 | -0.09 | -0.40 |
| arxiv2 | -0.23 | -0.24 | -0.82 | -0.09 | 0.00 | -0.51 |
| wikics | -0.49 | -0.34 | -0.94 | -0.40 | -0.51 | 0.00 |
</details>

Figure 6: Heatmap of the first set of datasets

![](images/bf60b183795360bc2f0389db3a15e8ae067254ba8618816de8929b4f3803d368.jpg)

<details>
<summary>heatmap</summary>

| Dataset | bookhi | bookch | elecom | elepho | sports | produc | amazon |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| bookhi | 0.00 | -0.29 | -0.66 | -0.70 | -1.07 | -0.44 | -1.00 |
| bookch | -0.29 | 0.00 | -0.62 | -0.66 | -0.98 | -0.34 | -0.82 |
| elecom | -0.66 | -0.62 | 0.00 | -0.23 | -0.85 | -0.34 | -0.67 |
| elepho | -0.70 | -0.66 | -0.23 | 0.00 | -0.80 | -0.35 | -0.76 |
| sports | -1.07 | -0.98 | -0.85 | -0.80 | 0.00 | -0.56 | -0.89 |
| produc | -0.44 | -0.34 | -0.34 | -0.35 | -0.56 | 0.00 | -0.47 |
| amazon | -1.00 | -0.82 | -0.67 | -0.76 | -0.89 | -0.47 | 0.00 |
</details>

Figure 7: Heatmap of the second set of datasets

We then inspect the homophily ratio of each dataset, shown in Table 12.

Table 12: Homophily ratio for node-level datasets 

<table><tr><td>Dataset</td><td>Homophily Ratio</td></tr><tr><td>Cora</td><td>0.81</td></tr><tr><td>Citeseer</td><td>0.78</td></tr><tr><td>Arxiv</td><td>0.66</td></tr><tr><td>Arxiv23</td><td>0.65</td></tr><tr><td>History</td><td>0.66</td></tr><tr><td>Child</td><td>0.42</td></tr><tr><td>Computers</td><td>0.83</td></tr><tr><td>Photo</td><td>0.75</td></tr><tr><td>Sportsfit</td><td>0.90</td></tr><tr><td>Products</td><td>0.81</td></tr><tr><td>Pubmed</td><td>0.80</td></tr><tr><td>WikiCS</td><td>0.65</td></tr><tr><td>Tolokers</td><td>0.59</td></tr><tr><td>Amazon ratings</td><td>0.38</td></tr></table>

Finally, we demonstrate the feature space plot in Figure 8.

We don't show the feature space plot for graph-level datasets because their features are all based on prompts for elements that are shared.

# D.2 Dataset Introduction

In this section, we introduce the dataset we use. We need to convert the original node features and labels into natural languages to construct a text-space dataset. We adopt Gemini [35] to generate

![](images/54f51ae6dd42791297c57fbefaf41102d8324c4576c229047538af324ce0d6ce.jpg)

<details>
<summary>scatter</summary>

| Dataset       | UMAP Dimension 1 | UMAP Dimension 2 |
| ------------- | ---------------- | ---------------- |
| cora_node     | [value]          | [value]          |
| pubmed_node   | [value]          | [value]          |
| citeseer_node | [value]          | [value]          |
| arxiv         | [value]          | [value]          |
| arxiv23       | [value]          | [value]          |
| bookhis       | [value]          | [value]          |
| bookchild     | [value]          | [value]          |
| elecomp       | [value]          | [value]          |
| elephoto      | [value]          | [value]          |
| sportsfit     | [value]          | [value]          |
| products      | [value]          | [value]          |
| wikics        | [value]          | [value]          |
</details>

Figure 8: Feature space of the node-level datasets

corresponding descriptions. All datasets mentioned below are under the MIT License unless otherwise specified.

Cora, CiteSeer, Pubmed. These datasets are originally adopted in [75]. In the original version, only processed TF-IDF features are provided. So, we follow [21, 22] to extract the original text attributes.

Arxiv. This dataset is originally provided in $[76]$ . We adopt the text-space version from $[19]$ .

Arxiv23. This dataset is originally provided in [21]. The original link is https://github.com/XiaoxinHe/tape\_arxiv\_2023. We then transform it into the text-space dataset.

History, Child, Computers, Photo, Sportsfit, These datasets are originally adopted in [32]. They are extracted from [77].

Products This dataset is originally provided in [76]. We adopt the original text features provided in the official library. Considering the size of the original dataset, it takes too much time for subgraph-based methods like OneForAll to train on this dataset. As a result, we extract a subgraph with 316513 nodes using torch-geometric's NeighborLoader.

Amazon Ratings, Tolokers. These datasets are originally proposed in [34]. Compared to other datasets, these don't follow the commonly adopted homophily assumption for node classification tasks [78]. We crawl the original attributes of these datasets and transform them into texts.

DBLP. This dataset is originally proposed in [79]. We consider the paper co-author relationship and turn it into a four-way classification.

CheMBL, PCBA, HIV, Tox21, Bace, Bbbp, Muv, Toxcast. These datasets are originally proposed in [33]. Following [19] and [24], we extract the expert prompt to convert the original attributes into texts.

# E Detailed Experimental Settings

Computational Environments. Our experiments are conducted on a single server with 8 A6000 GPUs.

# E.1 Hyperparameter Settings.

Co-training. The hyper-parameter settings for co-training phase are as follows:

1. For GraphMAE, we use

```python
num_heads=4, num_out_heads=1, num_layers=3, num_hidden=1024,
residual=True, in_drop=0.5, attn_drop=0.5, norm='batchnorm',
lr=0.01, weight_decay=1e-05, negative_slope=0.2, activation='prelu',
mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.2,
scheduler='cosine', warmup=true 
```

2. For DGI, we use

```python
num_layers=3, num_hidden=512, residual=True, in_drop=0.5, attn_drop=0.5, norm='batchnorm', lr=0.001, weight_decay=0.0005, activation='relu', scheduler = 'none' 
```

3. For OneForAll, we adopt the following set of hyperparameters for node-level and link-level tasks.

```python
num_layers=5, num_hidden=384, lr=0.0001, weight_decay=0, JK='none', activation='relu' 
```

For graph-level tasks, we set the num\_layers=7.

4. For LLaGA, we follow the hyper-parameter settings in the original paper [25].

5. For BUDDY and SEAL, we generally follow the hyper-parameter settings in the repo https://github.com/melifluos/subgraph-sketching. For BUDDY, the only parameter we tune is the max\_hash\_hops, and we set it to 2 on small-scale graphs Cora, CiteSeer, and Pubmed. We set it to 3 for the rest of the graphs. For SEAL, we set num\_hops to 2 on small-scale graphs Cora, CiteSeer, and Pubmed. Similarly, we set it to 3 for the rest of the graphs.

For BGRL and GCC, we fail to find a set of hyper-parameters working well for the co-training after searching for a large set of hyperparameters.

Pre-training. During the pretraining phase, we adopt hyperparameter settings similar to those in the co-training setup. Therefore, we primarily focus on introducing the relevant settings for pretraining below.

1. For GraphMAE, we pre-train models 10 epochs on the combination of Arxiv, Products, and Sportsfit datasets. Then, we conduct linear probing on downstream tasks.   
2. For LLaGA, we pre-train models 1 epoch on the combination of Arxiv, Products, and Sportsfit datasets. Then, we directly output the trained model's prediction for zero-shot inference. For few-shot and fine-tuning cases, we tune the projector with downstream data.   
3. For OneForAll, we co-train models on the pre-training datasets for 20 epochs. Then, we directly output the trained model's prediction for zero-shot inference. For few-shot and fine-tuning cases, we tune the models with downstream data.   
4. For OneForAll-FS, we pre-train the OneForAll with the few-shot version on the combination of Arxiv, Products, and Sportsfit datasets for 20 epochs. Then, we adopt the trained model for zero-shot and few-shot inference.   
5. For Prodigy, we either pre-train it on MAG240M or Arxiv. We construct a 30-way classification problem for both pre-training and generate 20000 randomly selected in-context learning samples.

# F Implementations

In this paper, we mainly implement the following groups of GFM building blocks. We detail their implementations as follows. All implementations mentioned below are under the MIT License unless otherwise specified. We use a unified data interface for the following methods to pack them for a comprehensive benchmark tool.

Graph prompts models. We mainly include two representative models: OneForAll [19] and Prodigy [20]. Their original implementation can be found via https://github.com/LechengKong/OneForAll and https://github.com/snap-stanford/prodigy.

LLM with graph projectors. We mainly include LLaGA [25] as the baseline for this category for its simplicity and reproducibility. The original implementation can be found via https://github.com/VITA-Group/LLaGA.

Graph SSL. We mainly include GraphMAE [41], DGI [38], BGRL [40] as the baseline for this category. For GraphMAE, we follow the implementation from https://github.com/THUDM/GraphMAE. For the other baselines, we follow the implementation from PyGCL https://github.com/PyGCL/PyGCL.

Link prediction-specific models. We mainly include BUDDY [44] and SEAL [45] as two baselines. The original implementation can be found from https://github.com/melifluos/subgraph-sketching.

# G Extended Experimental Results

# G.1 More results for co-training setting

# G.1.1 Co-training across node classification and link prediction

Table 13: “Node->Link (Acc)” means removing the test edge and then evaluating link prediction. -TS represents “task-specific”, which refers to the model trained with a single task. -CT represents “cross-task”, which refers to the model trained across tasks. Underline represents the case that cross-task co-training benefits compared to task-specific co-training. 

<table><tr><td>Average performance</td><td>OneForAll-TS</td><td>LLaGA-TS</td><td>OneForAll-CT</td><td>LLaGA-CT</td></tr><tr><td>Link-&gt;Node (Acc)</td><td>70.57</td><td>74.65</td><td>74.03</td><td>73.45</td></tr><tr><td>Node-&gt;Link (Hits@100)</td><td>74.05</td><td>85.03</td><td>79.30</td><td>84.10</td></tr></table>

Experiment Settings. Considering the efficiency of GFMs for link prediction, we adopt three small-scale datasets as in Section 4.2.2. We adopt OneForAll and LLaGA, which can share knowledge between node-level and link-level tasks with a unified model architecture. Since link prediction requires deleting test edges during training, different graph structures are required to evaluate node classification and link prediction tasks. Therefore, we investigate two cases in which node classification or link prediction is adopted as the downstream task. It should be noted that OneForAll's evaluation in the original paper [19] when co-training node and link-level tasks is potentially problematic since they don't remove the test edges for node-level tasks.

Results. As shown in Table 13, we observe that OneForAll achieves positive gain after cross-task co-training compared to task-specific co-training, while LLaGA shows no benefits. For “node->link” gain, the possible reason is that link prediction on these datasets requires strong semantic information (as shown in Appendix G.3). In node classification, node features are usually strongly correlated with labels on text-attributed graphs [22]. Therefore, it may help those link prediction tasks requiring strong semantic information. “Link->Node” performance gain is probably related to the dataset imbalance issue (a more thorough discussion can be found in Appendix G.5. In node-level datasets, we observe that negative transferring mainly happens on those small-scale datasets (see Table 1). Under co-training, the smaller the amount of data, the more likely it is to be influenced by other datasets. Introducing link prediction is equivalent to adding self-supervision to the same dataset, thereby reducing negative transfer.

Observation 7. Co-training across link prediction and node classification can benefit each other with proper GFM designs.

# G.1.2 Co-training across node classification, link prediction, and graph classification

Comprehensive Experiment Settings. We adopt all datasets from node classification co-training for node-level datasets (Section 4.2.1, three small-scale datasets for link-level datasets (Section 4.2.2), and all datasets from graph classification co-training (Section 4.2.3) for graph-level datasets. We adopt OneForAll, which supports cross-task and cross-graph training. Specifically, we consider the following three cases: (1) Co-training across link-level and graph-level datasets; (2) Co-training across node-level and graph-level datasets; and (3) Co-training across all tasks and datasets. For each

dataset, we train the model using the corresponding task. When a dataset contains node and link information, we employ multi-task training, and due to the limited amount of link-level datasets, we focus on node-level performance.

Table 14: Performance of OneForAll on graph classification using MLP and SGC backbone models after conducting node-graph co-training. ST means "single-graph training", CT means "co-training". 

<table><tr><td>MLP-ST-PCBA</td><td>MLP-CT-PCBA</td><td>SGC-ST-PCBA</td><td>SGC-CT-PCBA</td></tr><tr><td>0.081</td><td>0.074</td><td>0.092</td><td>0.087</td></tr><tr><td>MLP-ST-Avg</td><td>MLP-CT-Avg</td><td>SGC-ST-Avg</td><td>SGC-CT-Avg</td></tr><tr><td>65.33</td><td>65.28</td><td>69.51</td><td>68.38</td></tr></table>

Further Probing. Existing work rarely explores enhancing graph-level task performance through node-level datasets, making our findings somewhat surprising. To better understand this phenomenon, we replace OneForAll's backbone model with MLP and SGC as what we have done for node-level co-training in Section 4.2.1. As shown in Table 14, neither model benefits from co-training this time. This suggests that the positive gain primarily stems from the structural aspect.

When using a GCN backbone and incorporating graph-level co-training, the model tends to learn inductive biases more suitable for graph-level tasks, meaning it makes judgments based on higher-order structures rather than simply relying on augmented features. However, this increased reliance on structure naturally weakens the model's performance in node classification, especially in text-space datasets where features contain strong semantic information.

# G.2 Effects of different LLM Encoders

In this section, we further study the influence of different LLM encoders. Due to the size of datasets and computing resource restriction, we limit our scope to medium-scale language model encoder minim and mpnet [46]. We adopt OFA as the anchor model to compare these two encoders, where the results are shown in Table 15.

The results show that with a more powerful LLM encoder, mpnet, the performance of OneForAll co-trained across node, link, and graph-level tasks clearly improves on node-level tasks.

This result suggests that with the emergence of better LLM encoders, we have reason to believe that stronger LLMs can provide a better feature space, addressing the feature heterogeneity issue across different datasets at the feature level. Using stronger LLM encoders is an effective way to improve node-level performance.

Table 15: Performance comparison of different LLM encoders 

<table><tr><td rowspan="4">minilm</td><td>Cora</td><td>CiteSeer</td><td>Arxiv</td><td>Arxiv23</td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>WikiCS</td><td>Pubmed</td></tr><tr><td>67.55</td><td>78.37</td><td>71.79</td><td>72.96</td><td>82.96</td><td>53.39</td><td>84.5</td><td>86.32</td><td>84.5</td><td>85.58</td><td>72.16</td><td>72.59</td></tr><tr><td>pcba</td><td>hiv</td><td>tox21</td><td>bace</td><td>bbbp</td><td>muv</td><td>toxcast</td><td>Node Avg</td><td>Graph Avg</td><td></td><td></td><td></td></tr><tr><td>27.9</td><td>77.69</td><td>83.25</td><td>83.23</td><td>68.14</td><td>70.78</td><td>69.79</td><td>76.06</td><td>75.48</td><td></td><td></td><td></td></tr><tr><td rowspan="4">mpnet</td><td>Cora</td><td>CiteSeer</td><td>Arxiv</td><td>Arxiv23</td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>WikiCS</td><td>Pubmed</td></tr><tr><td>74.03</td><td>79.15</td><td>71.97</td><td>74.39</td><td>84.17</td><td>56.02</td><td>84.53</td><td>86.63</td><td>92.05</td><td>85.6</td><td>75.47</td><td>76.37</td></tr><tr><td>pcba</td><td>hiv</td><td>tox21</td><td>bace</td><td>bbbp</td><td>muv</td><td>toxcast</td><td>Node Avg</td><td>Graph Avg</td><td></td><td></td><td></td></tr><tr><td>27.32</td><td>78.18</td><td>83.52</td><td>82.11</td><td>70.03</td><td>70.05</td><td>68.54</td><td>78.37</td><td>75.41</td><td></td><td></td><td></td></tr></table>

# G.3 Extended results of co-training over link-prediction tasks

We test more different baseline models in Table 16. Here, each model is trained from scratch on a single graph. The experimental results indicate that a strong correlation exists between node features and ground truth labels for Cora and Citeseer. Even an MLP without structural information can perform well and surpass GCN.

# G.4 Co-training on heterophilous graphs

In Table 2, we observe that OneForAll with an SGC backbone can outperform one with a GCN backbone. However, it should be noted that this only applies to homophilous graphs. Here, we try co-training OneForAll with SGC backbone on graphs from E-commerce and Amazon ratings.

Table 16: Performance of different link prediction backbones trained from scratch on a single graph 

<table><tr><td></td><td>Cora</td><td>CiteSeer</td><td>Pubmed</td></tr><tr><td>MLP</td><td>83.22</td><td>91.95</td><td>49.88</td></tr><tr><td>GCN</td><td>83.12</td><td>88.91</td><td>66.14</td></tr><tr><td>SIGN</td><td>87.77</td><td>84.73</td><td>43.12</td></tr><tr><td>BUDDY</td><td>91.37</td><td>96.57</td><td>83.29</td></tr></table>

Table 17: Performance of OneForAll after co-training on graphs from E-commerce and Amazon Ratings. 

<table><tr><td></td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>Ratings</td></tr><tr><td>GCN-backbone</td><td>83.3</td><td>56.22</td><td>85.05</td><td>87.83</td><td>92.29</td><td>86.91</td><td>48.21</td></tr><tr><td>SGC-backbone</td><td>84.2</td><td>56.6</td><td>85.8</td><td>87.7</td><td>91.8</td><td>87.1</td><td>43.88</td></tr></table>

As shown in Table 17, the experimental results demonstrate that while the SGC backbone performs well on datasets conforming to the homophily assumption, its inductive bias does not effectively generalize to heterophilous graphs.

# G.5 Effects of dataset scales on co-training

In OneForAll, the authors address the issue of negative transfer by adding weights to different datasets. This raises the question: is the negative transfer between datasets caused by dataset imbalance? To answer this, we first focus on the E-commerce dataset, which doesn't exhibit significant negative transfer. Unlike the original split, we adopt the same low labeling rate as in Cora and Citeseer for co-training, with 20 samples per class in the training set.

As shown in Table 18, we observe that the occurrence of negative transfer appears to be unrelated to dataset ratio but rather depends on the inherent characteristics of the datasets. Interestingly, datasets within the E-commerce domain are closer in feature space compared to those in the CS citation domain, yet we observe better transferability between the former.

Table 18: In this table, we change every dataset's training ratio to 20 samples per class. Unless Table, we use fixed hyper-parameter setting for single-graph training. 

<table><tr><td></td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>Avg</td></tr><tr><td>Single</td><td>69.10</td><td>22.70</td><td>61.10</td><td>63.30</td><td>57.50</td><td>61.30</td><td>55.83</td></tr><tr><td>Co-train</td><td>61.60</td><td>25.90</td><td>62.10</td><td>58.30</td><td>67.20</td><td>63.50</td><td>56.43</td></tr><tr><td></td><td>Cora</td><td>Citeseer</td><td>Arxiv</td><td>Arxiv23</td><td>Avg</td><td></td><td></td></tr><tr><td>Single</td><td>80.13</td><td>81.35</td><td>59.40</td><td>58.30</td><td>69.80</td><td></td><td></td></tr><tr><td>Co-train</td><td>60.10</td><td>82.10</td><td>55.30</td><td>47.90</td><td>61.35</td><td></td><td></td></tr></table>

It's worth noting that we also notice that when each individual dataset has a sufficient number of data points, negative transfer rarely occurs. For instance, we try removing all small-scale datasets from the node-level experiments, and the co-training results are as Table 19.

From the table, We observe that even though these datasets come from diverse domains, negative transfer does not occur.

# H Limitations and future works

Limitations of experiments. Due to computational constraints, we do not utilize multiple seeds to reduce experimental variance in most experiments, as a single full training run of models like

Table 19 

<table><tr><td></td><td>Arxiv</td><td>Arxiv23</td><td>History</td><td>Child</td><td>Photo</td><td>Computers</td><td>Sports</td><td>Products</td><td>Avg</td></tr><tr><td>Single</td><td>73.85</td><td>73.75</td><td>83.33</td><td>53.77</td><td>84.46</td><td>86.48</td><td>92.50</td><td>87.35</td><td>79.44</td></tr><tr><td>Co-train</td><td>72.50</td><td>73.40</td><td>83.42</td><td>55.99</td><td>84.67</td><td>87.39</td><td>92.23</td><td>86.84</td><td>79.56</td></tr></table>

OneForAll and LLaGA takes several days. Conducting multiple trials is a potential avenue for future work.

Limitations of scopes. In this paper, we primarily focus on the issue of feature heterogeneity. To address existing structural heterogeneity, we mainly adopt current solutions within GFM, such as directly employing the classic MPNN. The effectiveness of newer architectures, like graph transformers $[56, 80]$ , and their ability to generalize across diverse tasks remain open questions for further investigation.

# I Broader Impacts

In this paper, we provide empirical investigation for the development of graph foundation models, which may empower diverse applications including E-commerce, social network, and natural science. GFM has the potential to significantly reduce the resource consumption associated with training numerous task-specific models. Additionally, it can drastically minimize the need for manual annotation, especially in domains like molecular property prediction. We believe our contributions will accelerate ongoing efforts to develop the next generation of versatile and equitable graph foundation models.

A potential negative impact of GFM is that due to its unified backbone pre-trained on massive data, some popular biases reflected in the datasets may be present in GFM's predictions, which requires user attention.