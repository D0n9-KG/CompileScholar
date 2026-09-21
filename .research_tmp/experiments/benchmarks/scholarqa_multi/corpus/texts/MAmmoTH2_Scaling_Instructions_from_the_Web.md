# MAmmoTH2: Scaling Instructions from the Web

◇Xiang Yue\*, ♠Tuney Zheng\*, ♠Ge Zhang\*, ♠Wenhu Chen\*

◇Carnegie Mellon University, ♠University of Waterloo

xyue2@andrew.cmu.edu wenhuchen@uwaterloo.ca

https://tiger-ai-lab.github.io/MAmmoTH2/

# Abstract

Instruction tuning improves the reasoning abilities of large language models (LLMs), with data quality and scalability being the crucial factors. Most instruction tuning data come from human crowd-sourcing or GPT-4 distillation. We propose a paradigm to efficiently harvest 10 million naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. Our approach involves (1) recalling relevant documents, (2) extracting instruction-response pairs, and (3) refining the extracted pairs using open-source LLMs. Fine-tuning base LLMs on this dataset, we build MAmmoTH2 models, which significantly boost performance on reasoning benchmarks. Notably, MAmmoTH2-7B's (Mistral) performance increases from $11\%$ to $36.7\%$ on MATH and from $36\%$ to $68.4\%$ on GSM8K without training on any in-domain data. Further training MAmmoTH2 on public instruction tuning datasets yields MAmmoTH2-Plus, achieving state-of-the-art performance on several reasoning and chatbot benchmarks. Our work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data.

![](images/9f447bec939e26c753312ae42895209c4fe0831668338b6be4b6d31d42b4af35.jpg)  
Figure 1: Overview of MAmmoTH2-Plus results. The MAmmoTH2-8x7B-Plus variant outperforms Mixtral-Instruct on reasoning benchmarks, matching Qwen-1.5-110B with only 13B active parameters. It also surpasses Mixtral-Instruct by around 10 points on general code and chatbot benchmarks.

# 1 Introduction

Reasoning is a fundamental aspect of human cognition and problem-solving (Clark et al., 2018; Hendrycks et al., 2021a; Cobbe et al., 2021; Rein et al., 2023; Yue et al., 2023a). Proficiency in

![](images/7ea0b42139ac508ce139f7444e1c2d8cc24a6587d2e33af5a242e4380ea5b518.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph "Previous Methods"
        A["Annotators"] --> B["Annotated"]
        B --> C["Seed Data"]
        C --> D["Synthetic"]
        E["Web"] --> F["Recall"]
        F --> G["Raw Doc"]
        G --> H["Extract"]
        H --> I["Extracted QA"]
        I --> J["Refine"]
        J --> K["Refined QA"]
    end

    subgraph "Our Method"
        L["WebInstruct: 10M instruction data from the web"] --> M["Recall"]
        M --> N["Raw Doc"]
        N --> O["Extract"]
        O --> P["Extracted QA"]
        P --> Q["Refine"]
        Q --> R["Refined QA"]
    end

    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#fcc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#ffc,stroke:#333
    style N fill:#ffc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#ffc,stroke:#333
    style Q fill:#ffc,stroke:#333
    style R fill:#ffc,stroke:#333
```
</details>

Figure 2: Comparison between our dataset curation method and previous studies.

reasoning is essential for advancing scientific knowledge, developing new technologies, and making informed decisions in various contexts. Recently, large language models (LLMs) (Brown et al., 2020; Ouyang et al., 2022; Touvron et al., 2023a,b; Achiam et al., 2023; Team et al., 2023) have shown remarkable progress in various NLP tasks. However, their ability to perform complex reasoning tasks (Lin et al., 2024) in the domains of mathematics, science, and engineering is still limited.

Recent studies have extensively explored how to enhance base LLMs' reasoning abilities. The two main approaches are continued training and instruction tuning. Continued training trains LLMs on large-scale filtered documents (Lewkowycz et al., 2022; Taylor et al., 2022; Azerbayev et al., 2023; Shao et al., 2024; Ying et al., 2024). Instruction tuning seeks to employ supervised fine-tuning loss on, usually small-scale, high-quality instruction-response pairs (Ouyang et al., 2022; Chung et al., 2024). While human-annotated instruction datasets (Cobbe et al., 2021; Hendrycks et al., 2021b; Amini et al., 2019) are often limited in scale, recent studies (Yu et al., 2023; Yue et al., 2023b; Toshniwal et al., 2024; Li et al., 2024a; Tang et al., 2024) attempt to prompt GPT-4 with seed data to increase the scalability. However, the synthesized instruction data becomes highly biased, not diverse, and prone to a high degree of hallucination.

To address these limitations, we propose to discover naturally existing instruction data from the web (Figure 2). We argue that the pre-training corpus (e.g., Common Crawl) already contains a vast amount of high-quality instruction data for LLM reasoning. For example, the web corpus contains a large amount of educational materials in the form of instruction-following pairs. These documents range across various domains like math, science, engineering, and humanities. Such readily available instruction data is not only diverse but also of high quality. However, such instruction data is highly dispersed across the corpus, which makes it particularly challenging to discover.

In this paper, we aim to mine these instruction-response pairs from the web using a three-step pipeline. (1) Recall step: We create a diverse seed dataset by crawling several quiz websites. We use this seed data to train a fastText model (Joulin et al., 2016) and employ it to recall documents from Common Crawl (Computer, 2023). GPT-4 is used to trim down the recalled documents by their root URL. We obtain 18M documents through this step. (2) Extract step: We utilize open-source LLMs like Mixtral (Jiang et al., 2024) to extract Q-A pairs from these documents, producing roughly 5M candidate Q-A pairs. (3) Refine step: After extraction, we further employ Mixtral-8×7B (Jiang et al., 2024) and Qwen-72B (Bai et al., 2023) to refine (Zheng et al., 2024b) these candidate Q-A pairs. This refinement operation aims to remove unrelated content, fix formality, and add missing explanations to the candidate Q-A pairs. This refinement operation is pivotal to maintaining the quality of the mined Q-A pairs. Eventually, we harvest a total of 10M instruction-response pairs through these steps. Unlike existing instruction-tuning dataset, our dataset WEBINSTRUCT is purely mined from the Web without any human crowdsourcing or GPT-4 distillation.

We validate the effectiveness of WEBINSTRUCT by training MAmmoTH2 on various base models (Figure 1), including Mistral-7B (Jiang et al., 2023), Llama3-8B (Meta, 2024), Mixtral-8×7B (Jiang et al., 2024), and Yi-34B (Young et al., 2024). MAmmoTH2 significantly outperforms the base models on seven held-out reasoning benchmarks: TheoremQA (Chen et al., 2023b), GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), ARC-C (Clark et al., 2018), MMLU-STEM (Hendrycks et al., 2021b), GPQA (Rein et al., 2023), and BBH (Suzgun et al., 2022). MAmmoTH2-7B improves Mistral-7B's performance by an average of 14 absolute points, while MAmmoTH2-34B enhances Yi-34B's performance by an average of 5.8 absolute points. Notably, Mistral-7B's MATH accuracy can rise from 11.2% to 36.7% after training on WEBINSTRUCT. As our dataset contains no in-domain data from our evaluation benchmarks, this highlights the models' strong generalization ability.

We further enhance MAmmoTH2's performance on code generation, math reasoning, and instruction-following tasks by tuning it on open-source instruction datasets, including OpenHermes2.5 (Teknium,

2023), Code-Feedback (Zheng et al., 2024c), and Math-plus. The resulting model, MAmmoTH2-Plus, excels on seven reasoning benchmarks and other general tasks. MAmmoTH2-7B-Plus and MAmmoTH2-8B-Plus achieve state-of-the-art performance on TheoremQA, ARC-C, MMLU-STEM, GPQA, and BBH, and competitive results on MATH (45%) and GSM8K (85%). MAmmoTH2-Plus also performs well on general tasks, with MAmmoTH2-7B-Plus showing promising results on HumanEval and MBPP, and MAmmoTH2-8×7B leading the AlpacaEval 2.0 and Arena Hard leaderboards.

Interestingly, MAmmoTH2-8B-Plus and Llama-3-8B-Instruct, both tuned from Llama-3-base using datasets of the same size (10M), provide an apple-to-apple comparison. The only distinction is that Llama-3-8B-Instruct is trained on 10M human-annotated dataset while we do not require any human annotation. MAmmoTH2-8B-Plus outperforms Llama-3-Instruct by 6 points on reasoning tasks while matching its performance on general tasks, reflecting WEBINSTRUCT's cost-effectiveness advantage. MAmmoTH2-Plus consistently surpasses official instruction models like Mixtral-Instruct on chat benchmarks. These results demonstrate the effectiveness of our approach to scale up instruction data from the web and offer a new perspective for future instruction tuning studies.

# 2 WEBINSTRUCT

In this section, we outline the process of constructing WEBINSTRUCT. Specifically, we divide the data collection pipeline into three stages: (1) relevant document recall from the web corpus, (2) Q-A pair extraction from recalled document, and (3) Q-A pair refinement. The full pipeline is depicted in Figure 3 and an example for extraction and refinement is provided in Figure 4.

# 2.1 Recall from Common Crawl

In contrast to previous math-centric approaches (Paster et al., 2023; Wang et al., 2023c; Shao et al., 2024), we aim for broad coverage of disciplines such as math, science, engineering, etc. Therefore, careful balancing of the seed data is necessary to ensure diversity. However, publicly available training datasets are mostly limited to mathematics. To address this issue, we propose to crawl new

![](images/d09c8f467e77a303bb4d7faf89c3fdba9374953a4d11091263da16a2c2630cd7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Seed"] --> B["Fast-Text"]
    B --> C["CC"]
    C --> D["URL Filter"]
    D --> E["Document"]
    E --> F["Fast-Text"]
    F --> G["CC"]
    G --> H["URL Select"]
    H --> I["Document"]
    
    J["Pre-process"] --> K["LLM Extraction"]
    K --> L["Q A"]
    L --> M["Decontaminate"]
    M --> N["LLM Refinement"]
    N --> O["Document with gear icon"]
    
    P["Recall from Web"] --> B
    Q["Q-A Pair Extraction"] --> K
    R["Q-A Pair Refinement"] --> N
```
</details>

Figure 3: Step 1: Recall relevant documents from Common Crawl. Step 2: Extracting Q-A pairs. Step 3: Refine with the extracted Q-A pairs.

![](images/3fd19e7cb22824286a8e1c572277f18aaa8e3b208dd69538fd366089c490151b.jpg)

<details>
<summary>text_image</summary>

Raw Docs
Unformatted Text, Site Information, Ads

Topics Science\nAnatomy&Physiology\nAstronomy\nAstrophysics
\nBiology\nChemistry\n...Socratic Meta...Featured Answers
How do you simplify #((u^4v^3)/(u^2v^-1)^4)^0# and write it using only positive exponents?
Answer by NickTheTurtle (Apr 1, 2017)
Explanation:\nAnything raised to the \{0^{th}\} power is simply 1.
\nRelated Questions\ What is the quotient of powers property?\n\n\n How do you simplify expressions using the quotient rule?...\nImpact of this question\n1274 views around the world #Apps\niOS\nAndroid\nLinks\n[Privacy](#)\n[Terms](#)\n[Help](#)

Extracted QA
Formatted QA but lacking detailed solutions

Question: How do you simplify (u^4 v^3/(u^2 v^-1)^4)^0 and write it using only positive exponents?
Answer: Explanation: Anything to the 0th power is just simply 1.

Rewritten QA
Formatted QA augmented with detailed solutions

Question:
How do you simplify (u^4 v^3/(u^2 v^-1)^4)^0 and write it using only positive exponents?
Answer:
To simplify the expression (u^4 v^3/(u^2 v^-1)^4)^0 and rewrite it using only positive exponents, we start by evaluating the expression from the innermost operation outward.
First, consider the exponent of zero on the entire fraction:
Any expression raised to the power of zero is equal to 1 (provided the base is not zero). Therefore, (u^4 v^3/(u^2 v^-1)^4)^0 = 1
This simplification makes the other calculations unnecessary because raising any non-zero expression to the power of zero will always result in 1.
So, the simplified expression is simply: 1
</details>

Figure 4: An illustrating example from WEBINSTRUCT for the extraction and refinement step.

exam problems from several educational websites. These sites contain diverse problems from various disciplines, helping to ensure diversity. We crawled 100K seed data as positive training examples and randomly selected 100K negative documents from CC (Computer, 2023) to train a fastText model (Joulin et al., 2016). The trained fastText model is used to recall relevant documents. We employ the open-source fastText library with a vector dimension of 256 to train the model for 3 epochs, with a learning rate of 0.1, a maximum n-gram length of 3, and a maximum number of word occurrences of 3. We recalled 100B tokens using the trained fasttext model from an internal CC. These raw web documents are further grouped by their domains (root URL) and only domains with more than 1000 documents are retained. We extracted roughly 600K domains from the recalled documents. We then prompt GPT-3.5 to scan through the domains and automatically select those that might contain instruction data. Around 50K domains are further labeled as positive samples by GPT-3.5. Note that all the recalled documents in the first round are not kept for further usage in Q-A Pair Extraction and Refinement. Next, we sample documents from the selected domains as positive examples, and documents from the non-selected domains and general Common Crawl as negative examples to re-train an improved fastText classifier. The newly trained fastText classifier is used to recall documents. We recalled 40B tokens using the newly trained fastText model. We prompt GPT-4 to sift through the recalled domains again, ultimately leading to 18M raw documents, primarily originating from the desired websites.

# 2.2 Q-A Pair Extraction

We observe that a significant number of naturally existing Q-A pairs are present in the 18M documents. However, these Q-A pairs are interspersed with a high volume of noise such as ads, markups, boilerplate, etc. Our preliminary training on these raw documents only yields limited gains.

First, we carefully pre-process the HTML to pre-extract useful content from the recalled documents. This is mostly rule-based filtering to clean site information, ads, HTML boilerplate, etc. This step significantly reduces the document length for the next stage. We then prompt Qwen-72B (Bai et al., 2023) to identify the question and answer pairs from the preprocessed documents. Specifically, we provide a few in-context examples to help the model understand what to extract. We also allow the model to return void if no natural question-answer pairs exist. In this stage, only 30% of the recalled documents were identified as containing naturally existing Q-A pairs, resulting in roughly 5M Q-A pairs as our candidates for the next step. However, these candidates still contain a substantial amount of unrelated content and formality issues. Besides that, a large portion of the extracted Q-A pairs also lack explanations for how the answer is derived. Therefore, we propose to perform another round of refinement to increase the data quality.

To avoid contamination, we follow previous work (Shao et al., 2024) and filter out web pages containing questions or answers to all of our evaluation benchmarks. Specifically, we filter out all web pages that contain n-grams (n = 10) string matches with either the questions or answers.

# 2.3 Q-A Pair Refinement

To further improve the extracted Q-A pair candidates, we propose refining them using LLMs. In this step, we prompt Mixtral-22B×8 (Jiang et al., 2024) and Qwen-72B (Bai et al., 2023) to reformat the extracted Q-A pairs. If the answer does not contain any explanation, these two LLMs will attempt to complete the intermediate reasoning steps leading to the given answer. We adopt two models to increase the diversity of our dataset. Eventually, we harvest 10M Q-A pairs as our final instruction-tuning dataset WEBINSTRUCT.

# 2.4 Dataset Statistics

To better distinguish our dataset from the existing ones, we include a summarization table in Table 1. It can be observed that most SFT datasets contain less than 1M samples but are of high quality. XwinMath (Li et al., 2024a) is the largest dataset, scaling up to over 1M samples through GPT4 synthesis, while OpenMathInstruct (Toshniwal et al., 2024) has not been generated using GPT-4 but instead uses Mixtral-8x7B Jiang et al. (2024). However, the seed data for both datasets is only based on GSM and MATH, leading to narrow domain coverage. In contrast, continue-training (CT) datasets are normally filtered from the web with much larger size, often exceeding 10B tokens and even rising to 120B tokens. However, continued pre-training on these massive datasets can be not only expensive

Table 1: The list of existing supervise-fine-tuning (SFT) and continue-training (CT) datasets. SFT datasets are primarily from academic NLP sources or synthesized by GPT-3.5/4 using seed data. CT datasets are larger but nosier. Our dataset falls between these two types. 

<table><tr><td>Dataset</td><td>#Pairs</td><td>Domain</td><td>Format</td><td>Dataset Source</td></tr><tr><td>FLAN V2 (Chung et al., 2024)</td><td>100K</td><td>General</td><td>SFT</td><td>NLP data + Human CoT</td></tr><tr><td>Self-Instruct (Wang et al., 2023b)</td><td>82K</td><td>General</td><td>SFT</td><td>Generated by GPT3</td></tr><tr><td>GPT4-Alpaca (Taori et al., 2023)</td><td>52K</td><td>General</td><td>SFT</td><td>Generated by GPT4</td></tr><tr><td>SuperNI (Wang et al., 2022)</td><td>96K</td><td>General</td><td>SFT</td><td>NLP Datasets</td></tr><tr><td>Tora (Gou et al., 2023)</td><td>16K</td><td>Math</td><td>SFT</td><td>GSM+MATH Synthesis by GPT4</td></tr><tr><td>WizardMath (Luo et al., 2023)</td><td>96K</td><td>Math</td><td>SFT</td><td>GSM+MATH Synthesis by GPT4</td></tr><tr><td>MathInstruct (Yue et al., 2023b)</td><td>262K</td><td>Math</td><td>SFT</td><td>Math datasets Synthesis by GPT4</td></tr><tr><td>MetaMathQA (Yu et al., 2023)</td><td>395K</td><td>Math</td><td>SFT</td><td>GSM+MATH Synthesis by GPT3.5</td></tr><tr><td>XwinMath (Li et al., 2024a)</td><td>1.4M</td><td>Math</td><td>SFT</td><td>GSM+MATH Synthesis by GPT4</td></tr><tr><td>OpenMathInstruct (Toshniwal et al., 2024)</td><td>1.8M</td><td>Math</td><td>SFT</td><td>GSM+MATH Synthesis by Mixtral</td></tr><tr><td>Dataset</td><td>#Tokens</td><td>Domain</td><td>Format</td><td>Dataset Source</td></tr><tr><td>OpenWebMath (Paster et al., 2023)</td><td>12B</td><td>Math</td><td>LM</td><td>Filtered from Web</td></tr><tr><td>MathPile (Wang et al., 2023c)</td><td>10B</td><td>Math</td><td>LM</td><td>Filtered from Web</td></tr><tr><td>Cosmopeida (Ben Allal et al., 2024)</td><td>25B</td><td>General</td><td>LM</td><td>Synthesized by Mixtral</td></tr><tr><td>MINERVA (Lewkowycz et al., 2022)</td><td>38B</td><td>Math</td><td>LM</td><td>Filtered from Web</td></tr><tr><td>Proof-Pile-2 (Azerbayev et al., 2023)</td><td>55B</td><td>Math</td><td>LM</td><td>OpenWebMath+Arxiv+Code</td></tr><tr><td>Galactica (Taylor et al., 2022)</td><td>106B</td><td>Math &amp; Sci.</td><td>LM</td><td>Filtered from Web</td></tr><tr><td>DeepseekMath (Shao et al., 2024)</td><td>120B</td><td>Math</td><td>LM</td><td>Recalled from Web</td></tr><tr><td>WEBINSTRUCT</td><td>(10M) 5B</td><td>Math &amp; Sci.</td><td>SFT</td><td>Recall and Extracted from Web</td></tr></table>

but also ineffective due to the high noise ratio. WEBINSTRUCT, with roughly 5B tokens, strikes a good balance between scalability and quality. It approaches the scalability of common CT datasets while maintaining high quality through the three-step construction pipeline. This makes our dataset unique compared to other alternatives.

# 2.5 Additional Public Instruction Datasets

To further enhance the diversity and quality of our dataset, we fine-tune MAmmoTH2 on several open-source instruction tuning datasets. These datasets are carefully selected based on their relevance to different reasoning subjects. Additionally, we consider some chat datasets to balance reasoning ability and general chat ability. The open-source datasets we incorporate are OpenHermes 2.5 (Teknium, 2023), Code-Feedback (Zheng et al., 2024c) and our Math-Plus, which is an augmented version of MetaMathQA (395K) (Yu et al., 2023) and Orca-Math (200K) (Mitra et al., 2024). More details of the public datasets can be found in Appendix A.

# 3 Experimental Setup

# 3.1 Training Setup

We unify all the samples in our instruction dataset to conform to the structure of a multi-turn instruction tuning dataset. This standardization ensures that the fine-tuned models can process data consistently, regardless of the original dataset formats. We select the open-source models Mistral 7B (Jiang et al., 2023), Mixtral 8×7B (Jiang et al., 2024), Llama-3 8B (Meta, 2024), and Yi-34B (Young et al., 2024) as our base models. We fine-tune these models to validate our WEBINSTRUCT at multiple scales using the LLaMA-Factory (Zheng et al., 2024d) library. We use a learning rate of 5e-6 for Mistral 7B and 1e-5 for Mixtral, Llama-3 8B, and Yi 34B. The global batch size is set to 512 with a maximum sequence length of 4096. We employ a cosine scheduler with a 3% warm-up period for 2 epochs. To efficiently train the models, we utilize DeepSpeed (Rasley et al., 2020) with the ZeRO-3 stage. All the models are trained with 32 A100 GPUs.

# 3.2 Evaluation Datasets

To rigorously assess the capabilities of models in reasoning abilities across different domains, we utilize several widely used datasets, GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), TheoremQA (Chen et al., 2023b), BIG-Bench Hard (BBH) (Suzgun et al., 2022), ARC-C (Clark

Table 2: Main results on reasoning datasets. Models without the '-Instruct' suffix refer to the released base models. Results are taken from official papers or blogs when available; otherwise, we use our own evaluation script. Underscored results represent the best baseline scores under the size constraint. All models are inferred with few-shot CoT: TheoremQA (5-shot), MATH (4-shot), GSM8K (4-shot), GPQA (5-shot), MMLU-STEM (5-shot), BBH (3-shot), and ARC-C (8-shot). 

<table><tr><td>Model</td><td>TheoremQA</td><td>MATH</td><td>GSM8K</td><td>GPQA</td><td>MMLU-ST</td><td>BBH</td><td>ARC-C</td><td>AVG</td></tr><tr><td>GPT-4-Turbo-0409</td><td>48.4</td><td>69.2</td><td>94.5</td><td>46.2</td><td>76.5</td><td>86.7</td><td>93.6</td><td>73.6</td></tr><tr><td colspan="9">Parameter Size between 20B and 110B</td></tr><tr><td>Qwen-1.5-110B</td><td>34.9</td><td>49.6</td><td>85.4</td><td>35.9</td><td>73.4</td><td>74.8</td><td>91.6</td><td>63.6</td></tr><tr><td>Qwen-1.5-72B</td><td>29.3</td><td>46.8</td><td>77.6</td><td>36.3</td><td>68.5</td><td>68.0</td><td>92.2</td><td>59.8</td></tr><tr><td>Deepseek-LM-67B</td><td>25.3</td><td>15.9</td><td>66.5</td><td>31.8</td><td>57.4</td><td>71.7</td><td>86.8</td><td>50.7</td></tr><tr><td>Yi-34B</td><td>23.2</td><td>15.9</td><td>67.9</td><td>29.7</td><td>62.6</td><td>66.4</td><td>89.5</td><td>50.7</td></tr><tr><td>Llemma-34B</td><td>21.1</td><td>25.0</td><td>71.9</td><td>29.2</td><td>54.7</td><td>48.4</td><td>69.5</td><td>45.7</td></tr><tr><td>Mixtral-8×7B</td><td>23.2</td><td>28.4</td><td>74.4</td><td>29.7</td><td>59.7</td><td>66.8</td><td>84.7</td><td>52.4</td></tr><tr><td>Mixtral-8×7B-Instruct</td><td>25.3</td><td>22.1</td><td>71.7</td><td>32.4</td><td>61.4</td><td>57.3</td><td>84.7</td><td>50.7</td></tr><tr><td>Intern-Math-20B</td><td>17.1</td><td>37.7</td><td>82.9</td><td>28.9</td><td>50.1</td><td>39.3</td><td>68.6</td><td>46.4</td></tr><tr><td colspan="9">Trained only with WEBINSTRUCT (All evaluations are held-out)</td></tr><tr><td>MAmmoTH2-34B</td><td>30.4</td><td>35.0</td><td>75.6</td><td>31.8</td><td>64.5</td><td>68.0</td><td>90.0</td><td>56.4</td></tr><tr><td>Δ over Yi</td><td>+7.2</td><td>+19.1</td><td>+7.7</td><td>+2.1</td><td>+2.9</td><td>+1.2</td><td>+0.5</td><td>+5.8</td></tr><tr><td>MAmmoTH2-8x7B</td><td>32.2</td><td>39.0</td><td>75.4</td><td>36.8</td><td>67.4</td><td>71.1</td><td>87.5</td><td>58.9</td></tr><tr><td>Δ over Mixtral</td><td>+9.2</td><td>+10.6</td><td>+1.0</td><td>+7.1</td><td>+7.4</td><td>+3.3</td><td>+2.8</td><td>+6.5</td></tr><tr><td colspan="9">Continue trained with additional instruction datasets (All held-out except MATH and GSM8K)</td></tr><tr><td>MAmmoTH2-8x7B-Plus</td><td>34.1</td><td>47.0</td><td>86.4</td><td>37.8</td><td>72.4</td><td>74.1</td><td>88.4</td><td>62.9</td></tr><tr><td>Δ over Qwen-1.5-110B</td><td>-0.8</td><td>-2.6</td><td>+1.0</td><td>+1.5</td><td>-1.0</td><td>-0.7</td><td>-4.0</td><td>-0.7</td></tr><tr><td colspan="9">Parameter Size = 7B or 8B</td></tr><tr><td>Deepseek-7B</td><td>15.7</td><td>6.4</td><td>17.4</td><td>25.7</td><td>43.1</td><td>42.8</td><td>47.8</td><td>28.4</td></tr><tr><td>Qwen-1.5-7B</td><td>14.2</td><td>13.3</td><td>54.1</td><td>26.7</td><td>45.4</td><td>45.2</td><td>75.6</td><td>39.2</td></tr><tr><td>Mistral-7B</td><td>19.2</td><td>11.2</td><td>36.2</td><td>24.7</td><td>50.1</td><td>55.7</td><td>74.2</td><td>38.8</td></tr><tr><td>Gemma-7B</td><td>21.5</td><td>24.3</td><td>46.4</td><td>25.7</td><td>53.3</td><td>57.4</td><td>72.5</td><td>43.0</td></tr><tr><td>Llemma-7B</td><td>17.2</td><td>18.0</td><td>36.4</td><td>23.2</td><td>45.2</td><td>44.9</td><td>50.5</td><td>33.6</td></tr><tr><td>WizardMath-7B-1.1</td><td>11.7</td><td>33.0</td><td>83.2</td><td>28.7</td><td>52.7</td><td>56.7</td><td>76.9</td><td>49.0</td></tr><tr><td>Abel-7B-002</td><td>19.3</td><td>29.5</td><td>83.2</td><td>30.3</td><td>29.7</td><td>32.7</td><td>72.5</td><td>42.5</td></tr><tr><td>Intern-Math-7B</td><td>13.2</td><td>34.6</td><td>78.1</td><td>22.7</td><td>41.1</td><td>48.1</td><td>59.8</td><td>42.5</td></tr><tr><td>Rho-1-Math-7B</td><td>21.0</td><td>31.0</td><td>66.9</td><td>29.2</td><td>53.1</td><td>57.7</td><td>72.7</td><td>47.3</td></tr><tr><td>Deepseek-Math-7B</td><td>25.3</td><td>34.0</td><td>64.2</td><td>29.2</td><td>56.4</td><td>59.5</td><td>67.8</td><td>48.0</td></tr><tr><td>Deepseek-Math-Instruct</td><td>23.7</td><td>44.3</td><td>82.9</td><td>31.8</td><td>59.3</td><td>55.4</td><td>70.1</td><td>52.5</td></tr><tr><td>Llama-3-8B</td><td>20.1</td><td>21.3</td><td>54.8</td><td>27.2</td><td>55.6</td><td>61.1</td><td>78.6</td><td>45.5</td></tr><tr><td>Llama-3-8B-Instruct</td><td>22.8</td><td>30.0</td><td>79.5</td><td>34.5</td><td>60.2</td><td>66.0</td><td>80.8</td><td>53.4</td></tr><tr><td colspan="9">Trained only with WEBINSTRUCT (All evaluations are held-out)</td></tr><tr><td>MAmmoTH2-7B</td><td>29.0</td><td>36.7</td><td>68.4</td><td>32.4</td><td>62.4</td><td>58.6</td><td>81.7</td><td>52.8</td></tr><tr><td>Δ over Mistral</td><td>+9.8</td><td>+25.5</td><td>+32.2</td><td>+7.7</td><td>+12.3</td><td>+2.9</td><td>+7.5</td><td>+14.0</td></tr><tr><td>MAmmoTH2-8B</td><td>32.2</td><td>35.8</td><td>70.4</td><td>35.2</td><td>64.2</td><td>62.1</td><td>82.2</td><td>54.3</td></tr><tr><td>Δ over Llama3</td><td>+12.2</td><td>+14.5</td><td>+15.6</td><td>+8.0</td><td>+8.6</td><td>+1.0</td><td>+3.6</td><td>+8.8</td></tr><tr><td colspan="9">Continue trained with additional instruction datasets (All held-out except MATH and GSM8K)</td></tr><tr><td>MAmmoTH2-7B-Plus</td><td>29.2</td><td>45.0</td><td>84.7</td><td>36.8</td><td>64.5</td><td>63.1</td><td>83.0</td><td>58.0</td></tr><tr><td>MAmmoTH2-8B-Plus</td><td>32.5</td><td>42.8</td><td>84.1</td><td>37.3</td><td>65.7</td><td>67.8</td><td>83.4</td><td>59.1</td></tr><tr><td>Δ over best baseline</td><td>+7.2</td><td>+0.7</td><td>+1.5</td><td>+2.8</td><td>+5.5</td><td>+1.8</td><td>+2.6</td><td>+5.7</td></tr></table>

et al., 2018), GPQA (Rein et al., 2023), MMLU-STEM (Hendrycks et al., 2021a). These datasets collectively enable a comprehensive assessment of language models' reasoning prowess across a spectrum of complexity and realism. The details of the evaluation datasets are in Appendix B.

We further evaluate the models on additional code generation tasks (including HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021) and their augmented version (Liu et al., 2024)), general LLM benchmarks like MMLU (Hendrycks et al., 2021a) and its recent robust and challenging version MMLU-Pro (TIGER-Lab, 2024). We also consider chat benchmarks like MT-Bench (Zheng et al., 2024a), AlpacaEval 2.0 (Li et al., 2023), and Arena Hard (Li et al., 2024b) to demonstrate the generalizability of WEBINSTRUCT and WEBINSTRUCT-PLUS on more general LLM benchmarks.

# 4 Main Results

# 4.1 Experimental Results on Reasoning Benchmarks

Table 2 presents our main results, with existing models partitioned into two tracks based on their parameter size. For 7B parameter models, we observe that our model trained solely with WEBINSTRUCT achieves significant improvements over the base models. For instance, MAmmoTH2-7B boosts the performance of Mistral-7B by an average of 14 points. Notably, WEBINSTRUCT does not contain any training data from these evaluation benchmarks, making all evaluations essentially held-out. The substantial performance gains demonstrate the strong generalization capabilities of MAmmoTH2-7B. Similarly, MAmmoTH2-8B boosts the performance of Llama-3-8B-base by an average of 8.8 points. We also experiment with larger models like Yi-34B and Mixtral to show that the performance gains are consistent across the board. Notably, Yi-34B's performance on MATH also increases by $19\%$ after training on WEBINSTRUCT.

Further tuning on several additional public datasets also significantly enhances performance. The MAmmoTH2-Plus model family achieves state-of-the-art results across the board. For example, MAmmoTH2-Plus's performance on TheoremQA, GPQA, and ARC-C represents the best-known results for any model under 10B parameters. MAmmoTH2-7B-Plus's performance on MATH and GSM is also close to the best-known results. We also show the results of the models solely trained on the additional public datasets in Appendix E.

An interesting comparison is between MAmmoTH2-8B-Plus and Llama3-Instruct, as both models are trained from the Llama3-base. Llama-3-instruct was trained on a 10M human-annotated instruction dataset along with public datasets, similar to WEBINSTRUCT combined with additional public datasets. Therefore, these two models are highly comparable. Our experiments show that MAmmoTH2-8B-Plus outperforms Llama3-Instruct by an average of 6% across the benchmarks. This substantial gain indicates that WEBINSTRUCT is highly cost-effective. For larger models, we found that MAmmoTH2-8x7B-Plus can even match the performance of Qwen-1.5-110B with only 13B active parameters. These results demonstrate the effectiveness of our scalable instruction tuning approach.

# 4.2 Additional Experimental Results

To further demonstrate the capabilities of our models beyond the reasoning benchmarks presented in Table 2, we conduct additional experiments to evaluate their performance on code generation, general language understanding, and instruction-following tasks. Table 3 showcases the results of various models on code generation tasks. The MAmmoTH2-7B-Plus model exhibits strong performance, achieving the highest average scores of 66.1 and 58.2 on HumanEval(+) and MBPP(+) datasets, respectively. It outperforms the official instruct counterparts like Mistral-7B-Instruct-v0.2 on these metrics, indicating its superior code generation abilities.

To assess the general language understanding and instruction-following capabilities of our models, we evaluate them on a range of benchmarks, as shown in Table 3. The MAmmoTH2-Plus models exhibit strong performance across these tasks, showcasing their versatility and robustness. For example, MAmmoTH2-8×7B-Plus achieves the highest scores on AlpacaEval 2.0 and Arena Hard leaderboards, surpassing competitive models like GPT-3.5-Turbo and Tulu-2-DPO-70B (Ivison et al., 2023).

![](images/7328f066fb69f1d8bcdbbc3be1d6844eb1a886d003851e689db4fb260e6863ce.jpg)

<details>
<summary>line</summary>

| # of Instructions | MATH Extracted QA (LM Loss) | MATH Refined QA (LM Loss) | MATH Refined QA (SFT Loss) | TheoremQA Extracted QA (LM Loss) | TheoremQA Refined QA (LM Loss) | TheoremQA Refined QA (SFT Loss) | ARC-C Extracted QA (LM Loss) | ARC-C Refined QA (LM Loss) | ARC-C Refined QA (SFT Loss) |
| ----------------- | --------------------------- | -------------------------- | -------------------------- | -------------------------------- | ------------------------------ | ------------------------------ | --------------------------- | -------------------------- | -------------------------- |
| 2M                | 17.5                        | 20.0                       | 21.0                       | 14.5                             | 16.0                           | 19.0                           | 73.0                        | 75.0                       | 77.0                       |
| 4M                | 21.0                        | 24.0                       | 25.0                       | 16.0                             | 17.5                           | 22.0                           | 75.0                        | 78.0                       | 80.0                       |
| 6M                | 23.0                        | 26.0                       | 27.0                       | 17.5                             | 19.0                           | 23.0                           | 76.0                        | 79.0                       | 81.0                       |
| 8M                | 25.0                        | 28.0                       | 29.0                       | 18.5                             | 21.0                           | 24.0                           | 77.0                        | 80.0                       | 82.0                       |
| 10M               | 26.0                        | 29.5                       | 31.0                       | 19.5                             | 22.5                           | 24.5                           | 77.5                        | 81.0                       | 83.0                       |
</details>

Figure 5: Mistral-7B model reasoning performance improves with scaling instructions. Additionally, SFT Loss is a more effective learning approach compared to LM Loss.

Table 3: Evaluation of code generation, instruction-following and MMLU(-Pro) performance for various models. We report the average of HumanEval(+) and MBPP (+) accuracy as the code generation performance (breakdown results are in Appendix D). Baseline scores are sourced from the original papers or the EvalPlus, MT-Bench, AlpacaEval 2.0, Arena Hard and MMLU-Pro leaderboards. (“-”) indicates that the score is not available from the sources. MAmmoTH2-Plus exhibits strong general conversational ability and excels at multitask language understanding across a wide range of domains compared to their official instruct counterparts and larger models. 

<table><tr><td></td><td>Code Generation</td><td>MT-Bench</td><td>Alpaca Eval 2.0</td><td>Arena Hard</td><td>MMLU</td><td>MMLU-Pro</td></tr><tr><td>GPT-4-1106-preview</td><td>85.6 (77.5)</td><td>9.32</td><td>50.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GPT-3.5-Turbo-1106</td><td>79.7 (70.2)</td><td>8.32</td><td>19.3</td><td>18.9</td><td>-</td><td>-</td></tr><tr><td>GPT-3.5-Turbo-0301</td><td>-</td><td>7.94</td><td>18.1</td><td>18.1</td><td>70.0</td><td>-</td></tr><tr><td>Tulu-2-DPO-70B</td><td>51.2 (43.0)</td><td>7.89</td><td>21.2</td><td>15.0</td><td>67.8</td><td>40.5</td></tr><tr><td>Llama-2-70b-chat</td><td>31.4 (26.5)</td><td>6.86</td><td>14.7</td><td>11.6</td><td>63.0</td><td>33.6</td></tr><tr><td>Yi-34B-Chat</td><td>38.7 (32.6)</td><td>7.86</td><td>27.2</td><td>23.1</td><td>73.5</td><td>42.1</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>43.4 (36.5)</td><td>7.60</td><td>17.1</td><td>12.6</td><td>60.8</td><td>30.8</td></tr><tr><td>Llama-3-8B-Instruct</td><td>65.8 (58.0)</td><td>8.02</td><td>22.9</td><td>20.6</td><td>67.2</td><td>40.9</td></tr><tr><td>Mixtral-8×7B-Instruct-v0.1</td><td>52.3 (44.7)</td><td>8.30</td><td>23.7</td><td>23.4</td><td>70.6</td><td>41.0</td></tr><tr><td>MAmmoTH2-7B-Plus</td><td>66.1 (58.2)</td><td>7.88</td><td>23.4</td><td>14.6</td><td>63.3</td><td>40.9</td></tr><tr><td>MAmmoTH2-8B-Plus</td><td>61.9 (53.3)</td><td>7.95</td><td>18.5</td><td>16.6</td><td>64.6</td><td>43.4</td></tr><tr><td>MAmmoTH2-8x7B-Plus</td><td>63.3 (55.3)</td><td>8.20</td><td>33.8</td><td>32.6</td><td>68.3</td><td>50.4</td></tr></table>

The strong performance of MAmmoTH2 on code generation and general language understanding tasks, as evidenced by Table 3, demonstrates that our method does not overfit to the reasoning benchmarks. Instead, it shows the models' ability to generalize well to a wide range of tasks, highlighting their versatility and robustness. These additional experiments further validate the effectiveness of our WEBINSTRUCT in developing powerful and flexible language models.

# 5 Ablation Study

# 5.1 Scaling Effect of Instructions

We first investigate the impact of model scaling and loss functions on the performance of language models across three representative tasks: MATH, TheoremQA, and ARC-C. We train models with varying training samples (1M to 10M) using extracted QA and refined QA data, and compare the effectiveness of two training losses: LM loss and SFT loss. Figure 5 shows that increasing model size and using SFT Loss with synthetic data consistently improves accuracy across all tasks. These findings demonstrate the importance of model scaling and supervised fine-tuning with synthetic data for enhancing language model performance in various domains.

# 5.2 Comparison of Two Refined Models

To assess the effectiveness of the Q-A pair refinement process by different LLMs, we conducted experiments by training three mistral-7B models: one on the data refined by Mixtral-22B×8, another on the data refined by Qwen-72B, and a third on the merged samples refined by both models. For a fair comparison, we trained the models with the same 9000 steps and a global batch size of 512. Our results show that the model trained on Mixtral-22B×8 refined data achieves comparable performance to the one trained on Qwen-72B refined data. The model trained on the merged samples consistently outperforms the models trained on data refined by individual LLMs. This demonstrates the effectiveness of using multiple LLMs for refinement, as it leads to a more diverse and comprehensive dataset.

Table 4: Comparison of the two data-refining LLMs. We train the three models with the same steps. 

<table><tr><td>Data</td><td>GSM</td><td>MATH</td><td>MMLU-S</td><td>Theo.</td><td>ARC.</td></tr><tr><td>Mixtral</td><td>62.9</td><td>29.1</td><td>56.5</td><td>26.1</td><td>78.3</td></tr><tr><td>Qwen</td><td>65.4</td><td>28.9</td><td>60.6</td><td>23.5</td><td>80.8</td></tr><tr><td>Merged</td><td>65.6</td><td>31.0</td><td>60.5</td><td>24.8</td><td>81.8</td></tr></table>

# 5.3 Comparison of Different Domains and Sources.

To understand how each domain (e.g., math, science, others) and data source (e.g., forum websites and education websites) contribute to the training, we train Mistral 7B on the subsets of different domains and data sources. Details of how we obtain domain labels can be found in Appendix G.

As shown in Table 5, training on different domains and data sources leads to varied performance across the evaluation benchmarks. The education website data source consistently outperforms the forum data source, indicating the higher quality of educational questions. Interestingly, while the math domain excels on MATH, it does not lead to significant improvements on GSM8K, another math-focused dataset, suggesting that training on a single math dataset may not gener-

Table 5: Impact of different data domains and sources. 

<table><tr><td>Data Source</td><td>GSM</td><td>MATH</td><td>MMLU-S</td><td>Theo.</td><td>ARC.</td></tr><tr><td>Base</td><td>47.4</td><td>15.7</td><td>51.4</td><td>17.3</td><td>77.6</td></tr><tr><td>Forum</td><td>51.0</td><td>24.0</td><td>54.7</td><td>21.0</td><td>78.2</td></tr><tr><td>Education</td><td>58.0</td><td>24.8</td><td>54.3</td><td>23.2</td><td>79.5</td></tr><tr><td>Math</td><td>52.9</td><td>27.3</td><td>51.6</td><td>21.7</td><td>74.1</td></tr><tr><td>Science</td><td>54.4</td><td>23.7</td><td>58.9</td><td>21.0</td><td>83.6</td></tr><tr><td>Other</td><td>59.4</td><td>20.8</td><td>55.3</td><td>21.1</td><td>79.4</td></tr></table>

alize well to other math benchmarks. Furthermore, training solely on the math domain does not yield substantial gains on science and STEM benchmarks, highlighting the need for diverse training. In contrast, the "Other" domain, which includes a diverse range of subjects, achieves the highest score on GSM8K, emphasizing the importance of diversity in the training data.

# 5.4 Case Study

We further conduct a case study examining the quality of extracted and refined QA pairs from the dataset. We showcase some good and bad cases in Appendix J. We observe that the question/answer pairs extracted from well-formed exam and education websites are of high quality. The common issue is that a large portion of extracted answers do not contain intermediate rationale (chain-of-thought). This issue could lead to worse generalization.

Therefore, we prompt Mixtral and Qwen-72B to complete the intermediate steps. We observe that the success rate of such completion is relatively high. However, there are cases where the extracted question/answer pairs contain serious formatting issues, which pose challenges for the following refinement step. Besides these issues, we also observe that LLMs can sometimes modify the intention of the originally extracted content, causing hallucinations.

![](images/c1e5d312b1e45e83ceb6b671324b41092dbd9f7a6f4e72fdd0a86a4200f81125.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Wrong Answer | 2 |
| Wrong Question | 4 |
| Wrong CoT | 4 |
| Errors: | 10 |
| Correct but Unchanged: | 12 |
| Fully Correct & Enhanced: | 78 |
</details>

Figure 6: Quality distribution of 50 sampled refined QA examples.

To quantify the error percentages, we randomly sample 50 refined QA examples and ask the human annotators to compare whether the refined examples are correct and significantly better than the extracted ones in terms of format and intermediate solutions. As we can see from Figure 6, 78% examples have been improved after refinement and only 10% examples introduce hallucinations after refinement. Overall, our case study reveals that the harvested instruction tuning dataset is generally accurate with a low error rate.

# 6 Conclusion

In this paper, we argue that the web corpus contains a vast amount of high-quality instruction data across various domains. To mine this data, we develop a three-step pipeline consisting of recall, extraction, and refinement steps. Through this pipeline, we harvest WEBINSTRUCT, a total of 10M diverse, high-quality instruction-response pairs and train language models. Our experiments demonstrate that MAmmoTH2 exhibits significantly enhanced science reasoning abilities compared to the baseline models. Our work showcases the potential of harnessing the vast amount of instruction data in the web corpus to democratize the development of LLMs with enhanced reasoning capabilities.

# References

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. ArXiv preprint, abs/2303.08774, 2023. URL https://arxiv.org/abs/2303.08774.   
A. Amini, S. Gabriel, S. Lin, R. Koncel-Kedziorski, Y. Choi, and H. Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, 2019. doi:10.18653/v1/N19-1245. URL https://aclanthology.org/N19-1245.   
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. ArXiv preprint, abs/2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.   
Z. Azerbayev, H. Schoelkopf, K. Paster, M. Dos Santos, S. M. McAleer, A. Q. Jiang, J. Deng, S. Biderman, and S. Welleck. Llemma: An open language model for mathematics. In The Twelfth International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=4WnqRR915j.   
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. ArXiv preprint, abs/2309.16609, 2023. URL https://arxiv.org/abs/2309.16609.   
L. Ben Allal, A. Lozhkov, G. Penedo, T. Wolf, and L. von Werra. Cosmopedia, 2024. URL https://huggingface.co/datasets/HuggingFaceTB/cosmopedia.   
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfbcb4967418bfb8ac142f64a-Abstract.html.   
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. ArXiv preprint, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.   
W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023a. URL https://openreview.net/forum?id=YfZ4ZPt8zd.   
W. Chen, M. Yin, M. Ku, P. Lu, Y. Wan, X. Ma, J. Xu, X. Wang, and T. Xia. Theoremqa: A theorem-driven question answering dataset. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023b. URL https://aclanthology.org/2023.emnlp-main.489/.   
H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. URL https://arxiv.org/pdf/2210.11416.   
P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv preprint, abs/1803.05457, 2018. URL https://arxiv.org/abs/1803.05457.   
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. ArXiv preprint, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.   
T. Computer. Redpajama: an open dataset for training large language models, 2023. URL https://github.com/togethercomputer/RedPajama-Data.

L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023. URL https://arxiv.org/pdf/2211.10435.   
Z. Gou, Z. Shao, Y. Gong, Y. Yang, M. Huang, N. Duan, W. Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. ArXiv preprint, abs/2309.17452, 2023. URL https://arxiv.org/abs/2309.17452.   
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ.   
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https://openreview.net/forum?id=7Bywt2mQsCe.   
H. Ivison, Y. Wang, V. Pyatkin, N. Lambert, M. Peters, P. Dasigi, J. Jang, D. Wadden, N. A. Smith, I. Beltagy, et al. Camels in a changing climate: Enhancing lm adaptation with tulu 2. ArXiv preprint, abs/2311.10702, 2023. URL https://arxiv.org/pdf/2311.10702.   
A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. ArXiv preprint, abs/2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.   
A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. ArXiv preprint, abs/2401.04088, 2024. URL https://arxiv.org/abs/2401.04088.   
A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, and T. Mikolov. Fasttext. zip: Compressing text classification models. ArXiv preprint, abs/1612.03651, 2016. URL https://arxiv.org/abs/1612.03651.   
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022. URL https://openreview.net/forum?id=IFXTZERXdM7.   
C. Li, W. Wang, J. Hu, Y. Wei, N. Zheng, H. Hu, Z. Zhang, and H. Peng. Common 7b language models already possess strong math capabilities. ArXiv preprint, abs/2403.04706, 2024a. URL https://arxiv.org/abs/2403.04706.   
T. Li, W.-L. Chiang, L. D. Evan Frick, B. Zhu, J. E. Gonzalez, and I. Stoica. From live data to high-quality benchmarks: The arena-hard pipeline, April 2024b. URL https://lmsys.org/blog/2024-04-19-arena-hard/.   
X. Li, T. Zhang, Y. Dubois, R. Taori, I. Gulrajani, C. Guestrin, P. Liang, and T. B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval, 2023.   
Z. Lin, Z. Gou, T. Liang, R. Luo, H. Liu, and Y. Yang. Criticbench: Benchmarking llms for critique-correct reasoning. ArXiv preprint, abs/2402.14809, 2024. URL https://arxiv.org/abs/2402.14809.   
J. Liu, C. S. Xia, Y. Wang, and L. Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36, 2024. URL https://openreview.net/pdf?id=1qvx610Cu7.   
H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. ArXiv preprint, abs/2308.09583, 2023. URL https://arxiv.org/abs/2308.09583.

Meta. Introducing meta llama 3: The most capable openly available llm to date. https://ai.meta.com/blog/meta-llama-3/, April 2024.   
A. Mitra, H. Khanpour, C. Rosset, and A. Awadallah. Orca-math: Unlocking the potential of slms in grade school math. ArXiv preprint, abs/2402.14830, 2024. URL https://arxiv.org/abs/2402.14830.   
M. Nye, A. J. Andreassen, G. Gur-Ari, H. Michalewski, J. Austin, D. Bieber, D. Dohan, A. Lewkowycz, M. Bosma, D. Luan, et al. Show your work: Scratchpads for intermediate computation with language models. In Deep Learning for Code Workshop, 2022. URL https://openreview.net/forum?id=iedYJm92o0a.   
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.   
K. Paster, M. Dos Santos, Z. Azerbayev, and J. Ba. Openwebmath: An open dataset of high-quality mathematical web text. In The Twelfth International Conference on Learning Representations, 2023. URL https://openreview.net/pdf?id=jKHmjlpViu.   
B. Peng, C. Li, P. He, M. Galley, and J. Gao. Instruction tuning with gpt-4. ArXiv preprint, abs/2304.03277, 2023. URL https://arxiv.org/abs/2304.03277.   
J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In R. Gupta, Y. Liu, J. Tang, and B. A. Prakash, editors, KDD '20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020, pages 3505–3506, 2020. URL https://dl.acm.org/doi/10.1145/3394486.3406703.   
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. ArXiv preprint, abs/2311.12022, 2023. URL https://arxiv.org/abs/2311.12022.   
V. Sanh, A. Webson, C. Raffel, S. H. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey, M. S. Bari, C. Xu, U. Thakker, S. S. Sharma, E. Szczechla, T. Kim, G. Chhablani, N. V. Nayak, D. Datta, J. Chang, M. T. Jiang, H. Wang, M. Manica, S. Shen, Z. X. Yong, H. Pandey, R. Bawden, T. Wang, T. Neeraj, J. Rozen, A. Sharma, A. Santilli, T. Févry, J. A. Fries, R. Teehan, T. L. Scao, S. Biderman, L. Gao, T. Wolf, and A. M. Rush. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4.   
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv preprint, abs/2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.   
A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum?id=uyTL5Bvosj.   
L. Sun, Y. Han, Z. Zhao, D. Ma, Z. Shen, B. Chen, L. Chen, and K. Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. ArXiv preprint, abs/2308.13149, 2023. URL https://arxiv.org/abs/2308.13149.   
M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, , and J. Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. ArXiv preprint, abs/2210.09261, 2022. URL https://arxiv.org/abs/2210.09261.   
Z. Tang, X. Zhang, B. Wan, and F. Wei. Mathscale: Scaling instruction tuning for mathematical reasoning. ArXiv preprint, abs/2403.02884, 2024. URL https://arxiv.org/abs/2403.02884.

R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
R. Taylor, M. Kardas, G. Cucurull, T. Scialom, A. Hartshorn, E. Saravia, A. Poulton, V. Kerkez, and R. Stojnic. Galactica: A large language model for science. ArXiv preprint, abs/2211.09085, 2022. URL https://arxiv.org/abs/2211.09085.   
G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: a family of highly capable multimodal models. ArXiv preprint, abs/2312.11805, 2023. URL https://arxiv.org/abs/2312.11805.   
Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. URL https://huggingface.co/datasets/teknium/OpenHermes-2.5.   
TIGER-Lab. Mmlu professional dataset. Hugging Face Dataset Hub, 2024. URL https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro.   
S. Toshniwal, I. Moshkov, S. Narenthiran, D. Gitman, F. Jia, and I. Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. ArXiv preprint, abs/2402.10176, 2024. URL https://arxiv.org/abs/2402.10176.   
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. ArXiv preprint, abs/2302.13971, 2023a. URL https://arxiv.org/abs/2302.13971.   
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288, 2023b. URL https://arxiv.org/abs/2307.09288.   
X. Wang, Z. Hu, P. Lu, Y. Zhu, J. Zhang, S. Subramaniam, A. Loomba, S. Zhang, Y. Sun, and W. Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. In The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS'23, 2023a. URL https://openreview.net/forum?id=u6jbcaCHq0.   
Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap, E. Pathak, G. Karamanolakis, H. Lai, I. Purohit, I. Mondal, J. Anderson, K. Kuznia, K. Doshi, K. K. Pal, M. Patel, M. Moradshahi, M. Parmar, M. Purohit, N. Varshney, P. R. Kaza, P. Verma, R. S. Puri, R. Karia, S. Doshi, S. K. Sampat, S. Mishra, S. Reddy A, S. Patro, T. Dixit, and X. Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, 2022. URL https://aclanthology.org/2022.emnlp-main.340.   
Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, 2023b. URL https://arxiv.org/abs/2212.10560.   
Z. Wang, R. Xia, and P. Liu. Generative ai for math: Part i–mathpile: A billion-token-scale pretraining corpus for math. ArXiv preprint, abs/2312.17120, 2023c. URL https://arxiv.org/abs/2312.17120.   
J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022a. URL https://openreview.net/forum?id=gEZrGCozdqR.   
J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b. URL https://openreview.net/pdf?id=\_VjQlMeSB\_J.

C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang. Wizardlm: Empowering large language models to follow complex instructions. ArXiv preprint, abs/2304.12244, 2023. URL https://arxiv.org/abs/2304.12244.   
H. Ying, S. Zhang, L. Li, Z. Zhou, Y. Shao, Z. Fei, Y. Ma, J. Hong, K. Liu, Z. Wang, et al. Internlm-math: Open math large language models toward verifiable reasoning. ArXiv preprint, abs/2402.06332, 2024. URL https://arxiv.org/abs/2402.06332.   
A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, H. Li, J. Zhu, J. Chen, J. Chang, et al. Yi: Open foundation models by 01. ai. ArXiv preprint, abs/2403.04652, 2024. URL https://arxiv.org/abs/2403.04652.   
L. Yu, W. Jiang, H. Shi, Y. Jincheng, Z. Liu, Y. Zhang, J. Kwok, Z. Li, A. Weller, and W. Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=N8N0hgNDRt.   
L. Yuan, G. Cui, H. Wang, N. Ding, X. Wang, J. Deng, B. Shan, H. Chen, R. Xie, Y. Lin, et al. Advancing llm reasoning generalists with preference trees. ArXiv preprint, abs/2404.02078, 2024. URL https://arxiv.org/abs/2404.02078.   
X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. ArXiv preprint, abs/2311.16502, 2023a. URL https://arxiv.org/abs/2311.16502.   
X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. Mammoth: Building math generalist models through hybrid instruction tuning. In The Twelfth International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=yLC1Gs770I.   
Y. Zhang, Y. Luo, Y. Yuan, and A. C.-C. Yao. Automathtext: Autonomous data selection with language models for mathematical texts. ArXiv preprint, abs/2402.07625, 2024. URL https://arxiv.org/abs/2402.07625.   
L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2024a. URL https://openreview.net/forum?id=uccHPGdlao.   
T. Zheng, S. Guo, X. Qu, J. Guo, W. Zhang, X. Du, C. Lin, W. Huang, W. Chen, J. Fu, et al. Kun: Answer polishment for chinese self-alignment with instruction back-translation. ArXiv preprint, abs/2401.06477, 2024b. URL https://arxiv.org/abs/2401.06477.   
T. Zheng, G. Zhang, T. Shen, X. Liu, B. Y. Lin, J. Fu, W. Chen, and X. Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. ArXiv preprint, abs/2402.14658, 2024c. URL https://arxiv.org/abs/2402.14658.   
Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, and Y. Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. ArXiv preprint, abs/2403.13372, 2024d. URL https://arxiv.org/abs/2403.13372.

# A Details of Additional Public Instruction Tuning Datasets

- OpenHermes 2.5 (Teknium, 2023): The OpenHermes-2.5 dataset is a comprehensive collection of diverse data sources for instruction tuning, including 1M examples from datasets in math, science, and coding, alongside synthetic and chat-based data. It incorporates diverse sources such as Airoboros 2.2, CamelAI Domain Expert Datasets, ChatBot Arena, Collective Cognition, Evol Instruct, Glaive Code Assistant, GPT4-LLM, GPTeacher, Medical Tasks, MetaMath, SlimOrca, Platypus, ShareGPT, and Unnatural Instructions GPT4. We remove TheoremQA from Platypus as it is one of our test sets.   
- Code-Feedback (Zheng et al., 2024c): The Code-Feedback dataset is a multi-turn code generation and refinement dataset, containing 68,000 multi-turn interactions between users, code generation models, and compiler systems. It includes initial user instructions followed by compiler and user feedback. This dataset significantly enhances the model's multi-turn interaction coding ability.   
- Math-Plus: This dataset combines public datasets such as MetaMathQA (395K) (Yu et al., 2023) and Orca-Math (200K) (Mitra et al., 2024). Both of these datasets are generated by GPT-3.5/4 using GSM/MATH and other math datasets as the seed data. To further augment the dataset, we prompt GPT-4 to rewrite Q-A pairs from MATH training sets, adding an additional 300K examples to enhance the challenging problems. The total size of the Math-Plus dataset is 894K examples.

To ensure consistency and compatibility, we carefully align the format and structure of these additional datasets with WEBINSTRUCT. These supplementary datasets provide a rich resource for training models to answer questions and provide explanations across a wide range of topics, enhancing their versatility and applicability in real-world scenarios.

# B Details of Evaluation Datasets

To rigorously assess the capabilities of models in reasoning abilities across different domains, we utilize several widely used datasets. Each of these datasets is designed to challenge the models in different aspects of reasoning.

- GSM8K (Cobbe et al., 2021): This test dataset contains 1.32K diverse grade school math problems, intended to test basic arithmetic and reasoning ability in an educational context.   
- MATH (Hendrycks et al., 2021b): Comprising 5000 intricate competition-level problems to evaluate the models' ability to perform complex mathematical reasoning.   
- TheoremQA (Chen et al., 2023b): Focused on applying mathematical theorems to solve advanced problems in fields such as mathematics, physics, and engineering, TheoremQA includes 800 questions that test the theoretical reasoning capabilities.   
- BIG-Bench Hard (BBH) (Suzgun et al., 2022): Consisting of 23 tasks previously found challenging for language models from BIG-Bench (Srivastava et al., 2023), BBH contains a total of 6511 challenging problems examining the capability of LLMs to solve them.   
- ARC-C (Clark et al., 2018): ARC includes questions derived from various grade-level science exams, testing models' ability to handle both straightforward and complex scientific queries. We use the challenge subset, which contains 1,172 test questions.   
- GPQA (Rein et al., 2023): This dataset provides "Google-proof" questions in biology, physics, and chemistry, designed to test deep domain expertise and reasoning under challenging conditions. We use the diamond subset containing 198 hard problems.   
- MMLU-STEM (Hendrycks et al., 2021a): Spanning 57 subjects across multiple disciplines, MMLU evaluates the breadth and depth of a model's knowledge in a manner akin to academic and professional testing environments. We select the STEM subset of MMLU with 3.13K problems.

These datasets collectively enable a comprehensive assessment of language models' reasoning prowess across a spectrum of complexity and realism. We further evaluate the models on additional code generation tasks (including HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021) and their augmented version (Liu et al., 2024)), general LLM benchmarks like MMLU (Hendrycks et al., 2021a), and chat benchmarks like MT-Bench (Zheng et al., 2024a), AlpacaEval 2.0 (Li et al., 2023), and Arena Hard (Li et al., 2024b) to demonstrate the generalizability of WEBINSTRUCT and WEBINSTRUCT-PLUS on more general LLM benchmarks.

# C Related Work

Instruction Tuning. Instruction tuning is crucial for aligning large language models (LLMs) with end tasks. There are two main types of instruction tuning data: (1) human-written data, such as FLAN (Wei et al., 2022a), T0 (Sanh et al., 2022), and SuperNI (Wang et al., 2022), which assemble large instruction-tuning datasets from existing human-labeled datasets; and (2) synthesized data, like Self-Instruct (Wang et al., 2023b), WizardLM (Xu et al., 2023), and GPT4-Alpaca (Peng et al., 2023), which create instruction-tuning datasets by synthesizing from powerful LLMs like GPT-4 (Achiam et al., 2023). Both types of instruction-tuning data have advantages and disadvantages. Human-written data is limited in size due to the high cost and in task diversity because existing human-labeled datasets mostly focus on a few NLP tasks. Although synthesized data can be generated at any scale, the high rate of hallucination can lead to significant quality degradation. Moreover, the diversity of synthesized data is heavily influenced by the seed data. Without a diverse seed dataset, the synthesized data will lack domain coverage.

Mathematics Reasoning. In recent years, there has been a growing interest in enhancing the mathematical reasoning abilities of large language models (LLMs). Three main approaches have been proposed to improve LLMs' mathematical reasoning skills:

- Prompting: Chain-of-thought-prompting (CoT) (Nye et al., 2022; Wei et al., 2022b) elicits LLMs' inherent reasoning ability by demonstrating intermediate reasoning steps. Program-of-thoughts-prompting (PoT) (Chen et al., 2023a; Gao et al., 2023) utilizes tools to further augment LLMs' math reasoning abilities. Subsequent work (Gou et al., 2023; Toshniwal et al., 2024; Yuan et al., 2024) combines CoT and PoT to maximize LLMs' reasoning ability.   
- Continued Training: Enabling LLMs to solve mathematical problems has been a long-standing challenge. MINERVA (Lewkowycz et al., 2022) and Galactica (Taylor et al., 2022) were pioneers in continued training of LLMs to adapt to scientific domains for math and science reasoning. Open-source models like Llemma (Azerbayev et al., 2023), DeepSeek-Math (Shao et al., 2024), and Intern-Math (Ying et al., 2024) have surpassed MINERVA and Galactica on math benchmarks. These approaches mainly rely on using an efficient classifier to recall documents from Common Crawl to retrieve a massive high-quality math-related corpus (>100B tokens) to enhance math reasoning.   
- Instruction Tuning: Instruction tuning aims to enhance LLMs' math reasoning skills by efficiently training on human-annotated public datasets like GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), and MathQA (Amini et al., 2019). However, these datasets are often insufficient in size and diversity. Therefore, recent work (Yu et al., 2023; Yue et al., 2023b; Toshniwal et al., 2024; Luo et al., 2023; Li et al., 2024a) proposes augmenting them with strong commercial LLMs like GPT-4 (Achiam et al., 2023). These methods can significantly boost LLMs' performance on in-domain math benchmarks but may fall short of generalization.

Our work combines continued training with instruction tuning to exploit the benefits of both approaches. Specifically, our dataset is recalled from Common Crawl like DeepSeekMath. However, due to the significant level of noise in the raw corpus, we utilize a strong LLM to filter and clean the corpus to extract the instruction tuning pairs.

Science Reasoning. In addition to mathematical reasoning, there is growing interest in improving LLMs' general scientific reasoning ability in subjects like physics, biology, chemistry, computer science, etc. Several benchmarks, such as MMLU (Hendrycks et al., 2021a), TheoremQA (Chen et al., 2023b), Sci-Bench (Wang et al., 2023a), SciEval (Sun et al., 2023), and GPQA (Rein et al., 2023), have been developed to measure LLMs' reasoning ability on tasks beyond math. However, there has been less effort in curating high-quality training data for the science domain. Most datasets, like OpenWebMath (Paster et al., 2023), Proof-Pile (Azerbayev et al., 2023), and MathPile (Zhang et al., 2024), are heavily biased towards mathematics. In this work, we aim to generalize the pre-training data to broader subjects through our newly curated science seed data.

# D Code Generation Results

We report the code generation results of our models and baselines in Table 6.

<table><tr><td></td><td>HumanEval</td><td>HumanEval+</td><td>MBPP</td><td>MBPP+</td><td>Average</td><td>Average+</td></tr><tr><td>Mistral-7B</td><td>28.7</td><td>23.8</td><td>51.9</td><td>42.1</td><td>40.3</td><td>33.0</td></tr><tr><td>Gemma-7B</td><td>26.8</td><td>20.1</td><td>52.6</td><td>43.4</td><td>39.7</td><td>31.8</td></tr><tr><td>Llama-3-8B</td><td>33.5</td><td>29.3</td><td>61.4</td><td>51.6</td><td>47.5</td><td>40.5</td></tr><tr><td>Gemma-1.1-7B-Instruct</td><td>42.7</td><td>35.4</td><td>57.1</td><td>45.0</td><td>49.9</td><td>40.2</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>75.0</td><td>70.1</td><td>44.7</td><td>37.0</td><td>59.9</td><td>53.6</td></tr><tr><td>Llama-3-8B-Instruct</td><td>61.6</td><td>56.7</td><td>70.1</td><td>59.3</td><td>65.9</td><td>58.0</td></tr><tr><td>Mixtral-8×7B-Instruct-v0.1</td><td>45.1</td><td>39.6</td><td>59.5</td><td>49.7</td><td>52.3</td><td>44.7</td></tr><tr><td>MAmmoTH2- 7B-Plus</td><td>72.1</td><td>65.9</td><td>60.1</td><td>50.4</td><td>66.1</td><td>58.2</td></tr><tr><td>MAmmoTH2- 8B-Plus</td><td>63.4</td><td>57.9</td><td>60.4</td><td>48.6</td><td>61.9</td><td>53.3</td></tr><tr><td>MAmmoTH2- 8×7B-Plus</td><td>57.9</td><td>53.7</td><td>68.7</td><td>56.9</td><td>63.3</td><td>55.3</td></tr></table>

Table 6: Code generation results of different models. Baseline results are copied from the EvalPlus (Liu et al., 2024) leaderboard.

# E Impact of Additional Public Instruction Tuning Datasets

In Appendix A, we introduce additional public instruction tuning datasets to further boost the model's reasoning performance. Here, we show the three setups of models trained on: 1) WEBINSTRUCT only; 2) Additional Public Datasets only; 3) WEBINSTRUCT + Additional Public Datasets (which we first trained on WEBINSTRUCT and then continued training on additional public datasets). The results are shown in Table 7.

<table><tr><td>Data</td><td>TheoremQA</td><td>MATH</td><td>GSM8K</td><td>GPQA</td><td>MMLU-ST</td><td>BBH</td><td>ARC-C</td><td>AVG</td></tr><tr><td colspan="9">Mistral 7B Base</td></tr><tr><td>WEBINSTRUCT</td><td>29.0</td><td>36.7</td><td>68.4</td><td>32.4</td><td>62.4</td><td>58.6</td><td>81.7</td><td>52.8</td></tr><tr><td>PUBLIC DATASETS</td><td>22.6</td><td>37.9</td><td>83.5</td><td>29.3</td><td>57.6</td><td>62.7</td><td>79.9</td><td>53.4</td></tr><tr><td>WEBINS.+PUBLIC.</td><td>29.2</td><td>45.0</td><td>84.7</td><td>36.8</td><td>64.5</td><td>63.1</td><td>83.0</td><td>58.0</td></tr><tr><td colspan="9">Mixtral 8x7B Base</td></tr><tr><td>WEBINSTRUCT</td><td>32.2</td><td>39.0</td><td>75.4</td><td>36.8</td><td>67.4</td><td>71.1</td><td>87.5</td><td>58.9</td></tr><tr><td>PUBLIC DATASETS</td><td>31.3</td><td>45.1</td><td>85.3</td><td>37.4</td><td>69.4</td><td>73.2</td><td>88.1</td><td>61.4</td></tr><tr><td>WEBINS.+PUBLIC.</td><td>34.1</td><td>47.0</td><td>86.4</td><td>37.8</td><td>72.4</td><td>74.1</td><td>88.4</td><td>62.9</td></tr></table>

Table 7: Impact of additional public instruction tuning datasets.

# F Distributions of Website Domains in WEBINSTRUCT

Figure 7 show the distribution of the top websites in WEBINSTRUCT.

![](images/74e680556c8b7f4999e2dd70814e63889ce0a3bf63bfa83ba0a31c2109e3a3ca.jpg)

<details>
<summary>bar</summary>

Top 25 Website Usage Distribution
| Website | Percentage of Total (%) |
| :--- | :--- |
| biology.stackexchange.com | 0.1 |
| chemistry.stackexchange.com | 0.2 |
| www.indiabix.com | 0.3 |
| physics.stackexchange.com | 0.4 |
| socratic.org | 0.5 |
| cs.stackexchange.com | 1.2 |
| enotes.com | 1.9 |
| proofwiki.org | 1.9 |
| transtutors.com | 2.4 |
| sharemylesson.com | 2.7 |
| physicsforums.com | 3.2 |
| brainmass.com | 3.3 |
| www.khanacademy.org | 3.4 |
| gmatclub.com | 3.6 |
| weegy.com | 3.8 |
| www.chegg.com | 4.0 |
| studypool.com | 4.2 |
| zbmath.org | 5.7 |
| answers.everydaycalculation.com | 6.3 |
| homework.study.com | 6.4 |
| www.brainly.com | 6.9 |
| jiskha.com | 8.5 |
| answers.com | 8.8 |
| math.stackexchange.com | 9.2 |
| coursehero.com | 9.9 |
</details>

Figure 7: The distribution of the top websites in our instruction dataset.

# G Domain Distribution of WEBINSTRUCT

Figure 8 presents a breakdown of the WEBINSTRUCT by subject domains and data sources, providing insights into the composition and diversity of the mined instruction-response pairs. The subject labels are automatically annotated using the Llama-3-8B-Instruct model, while the distribution between education and forum data is obtained by analyzing the source URLs of the samples. The pie chart reveals that WEBINSTRUCT is predominantly composed of science-related subjects, with 81.69% of the data falling under the broad "Science" category. Within this category, Mathematics takes up the largest share at 68.36%, followed by Physics, Chemistry, and Biology. This highlights the dataset's strong emphasis on mathematical problem-solving and scientific reasoning. The remaining non-science categories, such as Business, Art & Design, and Health & Medicine, contribute to the diversity of the dataset. In terms of data sources, the vast majority (86.73%) of the instruction-response pairs come from exam-style questions, while forum discussions make up the remaining 13.27%. This source breakdown indicates that WEBINSTRUCT primarily consists of well-structured, educational content, supplemented by real-world discussions and inquiries from forums. The diverse subject coverage and the combination of education and forum data enable WEBINSTRUCT to capture a wide range of reasoning tasks and problem-solving scenarios.

![](images/5d2d43caa94b8fb0f0737050e35fa73eca03848bf9ed2132bf5f385057f2814a.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Humanities & Social Science | 1.49 |
| Tech & Engineering | 1.32 |
| Health & Medicine | 0.85 |
| Art & Design | 7.37 |
| Business | 10.21 |
| Science | 68.36 |
| Mathematics | 81.69 |
| Education | 86.73 |
Forum: 13.27 |
| Chemistry | 3.77 |
</details>

Figure 8: Breakdown of WEBINSTRUCT by subject domains and data sources.

# H Limitations of WEBINSTRUCT

Despite employing a three-step pipeline to ensure the quality of the mined instruction-response pairs, there may still be some noise and inaccuracies in the dataset, as mentioned in Figure 6. The extraction and refinement steps rely on the performance of the LLMs used, which may introduce biases and errors. Future work could explore more advanced techniques for data cleaning and validation, such as human-in-the-loop approaches or training a data selection model for filtering. Furthermore, while WEBINSTRUCT covers a wide range of subjects, including math, science, and engineering, there may be specific subdomains or advanced topics that are underrepresented, such as humanities and other daily chat topics. Expanding the coverage of the seed data to include broader and more diverse instruction-response pairs could further enhance the reasoning capabilities of LLMs in different areas.

# I Broader Social Impact

The development of MAmmoTH2 and the WEBINSTRUCT has the potential to positively impact education by providing the community with access to a large-scale, diverse set of instruction-response pairs across various subjects, particularly in mathematics and science. MAmmoTH2 can assist students in their learning process by offering step-by-step solutions, explanations, and targeted feedback, thus enhancing their understanding of complex concepts and problem-solving skills. Furthermore, the dataset and models can serve as valuable resources for educators to create more effective and engaging educational content, assignments, and assessments. However, it is important to consider potential limitations and biases in the dataset and the model. The quality and accuracy of the instruction-response pairs in WEBINSTRUCT may vary, and the model's outputs may reflect these inconsistencies. Therefore, it is crucial to employ rigorous data filtering and quality control measures to ensure the reliability and trustworthiness of the dataset and the model.

# J Case Study

In this section, we aim to review some of the instances in WEBINSTRUCT. The data extracted from the website contains many issues. For example, the most common issue is that the answer is relatively short without any intermediate rationale, which could occur frequently from the quiz website with multi-choice questions. Another common issue is that the extraction destroys important formatting in the document, which leads to unrecoverable errors.

# Good Case 1

# Extracted:

Question: Find the distance from the point $\frac{3}{2}$ , $\frac{3}{2}$ to the line $-\frac{37x}{10} - \frac{21y}{10} - \frac{7}{2} = 0$ .

Answer: 61 $\sqrt{\frac{2}{905}}$ .

# Refined:

Question: Calculate the distance from the point $\left(\frac{3}{2},\frac{3}{2}\right)$ to the line defined by the equation $-\frac{37}{10}x-\frac{21}{10}y-\frac{7}{2}=0.$

Answer: To find the distance 'd' from a point $(x_0, y_0)$ to a line given by $Ax + By + C = 0$ , we use the formula:

$$
d = \frac {| A x _ {0} + B y _ {0} + C |}{\sqrt {A ^ {2} + B ^ {2}}}.
$$

For the point $\left(\frac{3}{2},\frac{3}{2}\right)$ and the line $-\frac{37}{10} x - \frac{21}{10} y - \frac{7}{2} = 0$ , substitute the coordinates into the formula:

$$
d = \frac {\left| - \frac {3 7}{1 0} \cdot \frac {3}{2} - \frac {2 1}{1 0} \cdot \frac {3}{2} - \frac {7}{2} \right|}{\sqrt {\left(- \frac {3 7}{1 0}\right) ^ {2} + \left(- \frac {2 1}{1 0}\right) ^ {2}}}.
$$

Simplifying further, the exact simplified distance 'd' is $61\sqrt{\frac{2}{905}}$ . Thus, the correct distance from the point to the line is $61\sqrt{\frac{2}{905}}$ .

Figure A1: In this Refined version, the expression of both the question and the answer is clearer and more precise. It employs standardized mathematical symbols and terms, such as representing the coordinates of the point and the equation of the line as $(x_{0}, y_{0})$ and $Ax + By + C = 0$ , respectively. Moreover, it explicitly provides the formula for calculating the distance. Through these revisions, readers can more easily understand how to compute the distance from a point to a line, making the solution process more transparent and easier to follow.

# Good Case 2

# Extracted:

Question: What are the two main ways that minerals are classified?

Answer: They can be classified by luster, streak, breakage, and hardness.

# Refined:

Question: What are the primary criteria used for classifying minerals in mineralogy?

Answer: Minerals are primarily classified based on two main categories: physical properties and chemical properties. The physical properties include luster, streak, crystal form, hardness, density, and cleavage or breakage. Chemical properties involve the mineral's chemical composition and reactions with other substances. These characteristics help identify and distinguish different minerals in mineralogy.

Figure A2: Refining the extracted response provides a more comprehensive and accurate explanation of mineral classification in mineralogy. By expanding on the initial response to include both physical and chemical properties, the refined version offers a deeper understanding of how minerals are categorized and identified. This enriched content can enhance the model's ability to generate informative responses related to mineralogy.

# Good Case 3

# Extracted:

Question: How do you find the asymptotes for $Q(x) = (2x^{2}) / (x^{2} - 5x - 6)$ ?

Answer: vertical asymptotes x=-1, x=6/n horizontal asymptote y=2 /n /n /n /n /n /n /n /n /n

# Refined:

Question: How do you determine the vertical and horizontal asymptotes for the function $Q(x) = \frac{2x^2}{x^2 - 5x - 6}$ ?

Answer: To find the vertical asymptotes, we need to identify the values of $x$ that make the denominator equal to zero:

$$
x ^ {2} - 5 x - 6 = (x - 6) (x + 1) = 0
$$

This gives us the vertical asymptotes at $x = -1$ and $x = 6$ .

For the horizontal asymptote, as $x$ approaches infinity or negative infinity, the dominant terms in the numerator and denominator are both $x^2$ . Thus, we compare their coefficients:

\- Numerator's highest degree coefficient: 2 - Denominator's highest degree coefficient: 1

The horizontal asymptote is given by the ratio of these coefficients:

$$
y = \frac {2}{1} = 2
$$

Therefore, the vertical asymptotes are at $x = -1$ and $x = 6$ , and the horizontal asymptote is $y = 2$ .

Figure A3: The refined answer provides a more detailed explanation for finding the asymptotes. It includes the steps for solving for the vertical asymptotes by factoring the denominator and determining where it equals zero. It also explains how to find the horizontal asymptote by comparing the leading coefficients of the highest-degree terms in both the numerator and the denominator. The use of LaTeX formatting makes the mathematical notation clearer and easier to understand.

# Good Case 4

# Extracted:

Question: Which of the following is not an example of biased language? a. “You are too short to reach that.” b. “Are you deaf?” c. “Move along, grandpa!” d. “I think I’ve been gyped.

Answer: D. I think I've been gypped.

# Refined:

Question: Which of the following statements does not exhibit biased language?

a. "You are too short to reach that."   
b. "Are you deaf?"   
c. "Move along, grandpa!"   
d. “I think I’ve been gypped.”

Answer: The statement that does not exhibit biased language is:

d. "I think I've been gypped."

Explanation: Biased language refers to words or phrases that stereotype, discriminate, or show prejudice towards a particular group. In this case, options a, b, and c contain biases towards people's physical abilities (height), hearing ability, and age, respectively. However, "I think I've been gyped" is an informal expression meaning to be cheated or deceived, which, while potentially offensive, does not directly target a specific group based on attributes like race, gender, or disabilities.

Figure A4: The refined version of the input presents the question and answers in a clearer, more structured format that improves readability and comprehension. Specifically, using a bulleted list for the options makes them distinct and easier to evaluate individually. Moreover, the addition of an explanation provides context and justification for the selected answer, enhancing the educational value of the content. This refined structure and the inclusion of an explanatory note make it a superior training example for models, as it helps in understanding not just the answer but the rationale behind it, promoting deeper learning and better pattern recognition in language processing tasks.

# Good Case 5

# Extracted:

Question: I am stuck on the following challenge: "If the line determined by two distinct points $(x_{1},y_{1})$ and $(x_{2},y_{2})$ is not vertical, and therefore has slope $(y_{2} - y_{1}) / (x_{2} - x_{1})$ , show that the point-slope form of its equation is the same regardless of which point is used as the given point." Okay, we can separate $(x_0,y_0)$ from the form to get:

$$
y (x _ {2} - x _ {1}) - x (y _ {2} - y _ {1}) = y _ {0} (x _ {2} - x _ {1}) - x _ {0} (y _ {2} - y _ {1})
$$

But how exclude this point $(x_0, y_0)$ and leave only $x, y, x_1, y_1, x_2, y_2$ in the equation? UPDATE: There is a solution for this challenge:

$$
(y _ {1} - y _ {2}) x + (x _ {2} - x _ {1}) y = x _ {2} y _ {1} - x _ {1} y _ {2}
$$

From the answer I found that

$$
y _ {2} (x - x _ {1}) - y _ {1} (x - x _ {2}) = y (x _ {2} - x _ {1})
$$

... but why this is true?

Answer: Thanks to saulspatz, the solution is to simply show that, whether we are using $(x_{1},y_{1})$ or $(x_{2},y_{2})$ as the given point, the equation does not change.

So both equations:

$$
y - y _ {1} = m (x - x _ {1})
$$

$$
y - y _ {2} = m (x - x _ {2})
$$

reduce to:

$$
(y _ {1} - y _ {2}) x + (x _ {2} - x _ {1}) y = x _ {2} y _ {1} - x _ {1} y _ {2}.
$$

# Refined:

Question: Demonstrate that the point-slope form of the equation of a line determined by two distinct points $(x_{1},y_{1})$ and $(x_{2},y_{2})$ , where the line is not vertical, remains consistent regardless of which point is chosen as the reference. Why does the form of the equation not change when different reference points are used?

Answer: To address this challenge, we need to confirm that the point-slope equation of a line remains consistent regardless of the point chosen to derive it. Given two points $(x_{1},y_{1})$ and $(x_{2},y_{2})$ with a non-vertical line passing through them, the slope $m$ of the line is $\frac{y_2 - y_1}{x_2 - x_1}$ .

The point-slope form of a line's equation based on point $(x_{1},y_{1})$ is:

$$
y - y _ {1} = m (x - x _ {1})
$$

Similarly, using point $(x_{2}, y_{2})$ gives:

$$
y - y _ {2} = m (x - x _ {2})
$$

These can be rewritten and compared to show their equivalence. Expanding both equations: - From the first equation: $y - y_{1} = \frac{y_{2} - y_{1}}{x_{2} - x_{1}} (x - x_{1})$ - From the second equation: $y - y_{2} = \frac{y_{2} - y_{1}}{x_{2} - x_{1}} (x - x_{2})$

We simplify these to find a common form. Multiplying both sides by $(x_{2} - x_{1})$ and rearranging terms, we find:

$$
y _ {2} (x - x _ {1}) - y _ {1} (x - x _ {2}) = y (x _ {2} - x _ {1})
$$

This confirms that both forms collapse to the same linear equation, showing the consistency of the point-slope form irrespective of the chosen reference point. Thus, it is demonstrated that the structural form of the equation remains unchanged, ensuring that the mathematical properties of the line—defined by its slope and points through which it passes—are invariant to the choice of the reference point used in its expression. This uniformity stems from the linearity and the constant slope defined by any two points on a straight line.

Figure A5: The original answer contains an irrelevant introductory sentence, which makes it less focused. In contrast, the refined version provides a clearer question by explicitly stating the problem of demonstrating the invariance of the point-slope form. The explanation in the answer is expanded to show step-by-step equivalence between the two forms and includes detailed reasoning about why the consistency holds. The added explanation reinforces the understanding of linear equations and the concept of constant slope, making the solution more comprehensive and pedagogically valuable.

# Bad Case 1

# Extracted:

Question: Sort $\left\{-3\pi, 9, \frac{17}{\sqrt{3}}, -4, \frac{47}{7}, \frac{3}{e}, \frac{3}{\sqrt{5}}\right\}$ .

Answer: $\left\{-3\pi, -4, \frac{3}{e}, \frac{3}{\sqrt{5}}, \frac{47}{7}, 9, \frac{17}{\sqrt{3}}\right\}$ .

# Refined:

Question: Sort the following numbers in ascending order: $-3\pi, 9, \frac{17}{\sqrt{3}}, -4, \frac{47}{7}, \frac{3}{e}, \frac{3}{\sqrt{5}}$ .

Answer: The sorted numbers in ascending order are: $-3\pi$ , -4, $\frac{3}{e}$ , $\frac{3}{\sqrt{5}}$ , $\frac{47}{7}$ , 9, $\frac{17}{\sqrt{3}}$ .

Figure A6: The refined example does not enhance the clarity or quality of the information. The original formatting and question structure were already clear and concise, and the transformation into a slightly more verbose question-and-answer format does not add significant value. Additionally, both versions use proper LaTeX formatting for mathematical expressions, so there is no improvement in readability or technical accuracy. Essentially, the refinement introduces additional words without enhancing comprehension, which could potentially dilute the effectiveness of the model training by focusing on unnecessary stylistic alterations rather than content clarity or quality.

# Bad Case 2

# Extracted:

Question: Which curve (AD or SRAS) will be affected and which direction (leftward or rightward) will it shift when Congress goes to war against Canada?

Answer: When Congress falls into a war with Canada, the aggregate demand curve will be affected by moving to the left. Suppose Congress fights against Canada, then, it is likely that Congress will pass laws that advocate for increased tariffs on imports.

Spending on exports is an element of aggregate demand. Thus, its curve shifts rightwards, when its components, which are, spending on consumptions, government spending, or spending on investment increase.

On the other hand, Canada will not do business with the USA as usual when they fall out with Congress. For example, the expenses on exports to Canada will reduce.

Thus, the aggregate demand will fail to increase and shift on the left if there is less consumption from households, falls of investments from firms or when the demand for exports from a country like Canada falls.

# Refined:

Question: How does a war between Congress and Canada affect the Aggregate Demand (AD) curve, and in which direction will it shift? Answer: When Congress declares war against Canada, the Aggregate Demand (AD) curve shifts to the left because Congress will implement increased tariffs on imports. Spending on exports, which is a component of aggregate demand, decreases due to these tariffs. The Short Run Aggregate Supply (SRAS) curve, however, is not directly affected by this situation. It mainly reflects changes in production costs and the availability of resources, which are not directly influenced by international conflicts unless they lead to disruptions in the domestic production process. In summary, the AD curve shifts left due to decreased exports and potentially lower consumer and business spending, while the SRAS curve remains relatively stable unless there are indirect impacts on domestic production.

Figure A7: The refined answer, while more structured and clearer in terms of the specific economic curves discussed, introduces inaccuracies that were not present in the original. Specifically, the refined version inaccurately states that spending on exports decreases due to tariffs, which contradicts the fundamental economic concept that tariffs affect imports rather than exports. Furthermore, the original answer provides a broader analysis of potential economic impacts, such as changes in household consumption and firm investment, which are omitted in the refined version. These omissions reduce the depth and accuracy of the economic analysis provided in response to the effects of war on economic curves.

# Bad Case 3

# Extracted:

Question: Is there a tool (LaTeX, JavaScript, Mathematica..) that allows one to draw Venn diagram automatically from subsets relations, e.g. $A \subset A + B$ , $A \subset C$ , $C \subset C + D$ , $B \not\subset C$ would yield:

![](images/6b006417343d1d345e7ef166f08a8448f598335c60077acdc1f52bf94268c467.jpg)

<details>
<summary>text_image</summary>

A+B
A
C+D
</details>

(Sorry for the pitiful Ascii drawing)

Answer: https://tex.stackexchange.com/questions/9681/how-to-draw-venn-diagrams-especially-complement-s-in-latex answers it in latex but I couldn't get their answer to work in my answer, as it used tikzpicture ( seems unsupported here).

# Refined:

Question: Is there any software for generating diagrams to show relationships among set members, for example, $A \subset A + B$ ?

Answer: One might try using LaTeX, which supports drawing with the TikZ package, although some features may not work in certain settings. For more details, check out this link: https://tex.stackexchange.com/questions/9681/how-to-draw-venn-diagrams-especially-complement-s-in-latex. If problems persist, updating the software or consulting the community might help.

Figure A8: The refined example's quality becomes bad because it omits most of the specific subset relations provided in the original query, making it unclear which relationships need to be visualized. The answer shifts the focus towards potential troubleshooting rather than directly addressing the user's need to generate diagrams based on specific subset conditions. Moreover, the original question included a sample ASCII diagram for clarity, which is entirely dropped in the refined version, removing a helpful virtual context.