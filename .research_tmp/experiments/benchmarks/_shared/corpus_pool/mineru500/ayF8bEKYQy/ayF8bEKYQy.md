# OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI

Zhen Huang $^{3,4}$ , Zengzhi Wang $^{1,4}$ , Shijie Xia $^{1,4}$ , Xuefeng Li $^{1,4}$ , Haoyang Zou $^{4}$ , Ruijie Xu $^{1,4}$ , Run-Ze Fan $^{1,4}$ , Lyumanshan Ye $^{1,4}$ , Ethan Chern $^{1,4}$ , Yixin Ye $^{1,4}$ , Yikai Zhang $^{1,4}$ , Yuqing Yang $^{4}$ , Ting Wu $^{4}$ , Binjie Wang $^{4}$ , Shichao Sun $^{4}$ , Yang Xiao $^{4}$ , Yiyuan Li $^{4}$ , Fan Zhou $^{1,4}$ , Steffi Chern $^{4}$ , Yiwei Qin $^{4}$ , Yan Ma $^{4}$ , Jiadi Su $^{4}$ , Yixiu Liu $^{1,4}$ , Yuxiang Zheng $^{1,4}$ , Shaoting Zhang $^{2}$ , Dahua Lin $^{2}$ , Yu Qiao $^{2}$ , Pengfei Liu $^{1,2,4}$

$^{1}$ Shanghai Jiao Tong University, $^{2}$ Shanghai Artificial Intelligence Laboratory, $^{3}$ Soochow University, $^{4}$ Generative AI Research Lab (GAIR)

![](images/5d5819445ae7586b585b51f1e98da0345291e71c10a1b983131c0e5897d911d6.jpg)

<details>
<summary>text_image</summary>

CHEMISTRY
CZBORYN
OlympicArena
ASIKONOMY
COMPUTER SCIENCE
PHYSICS
BIOLOGY
MATH
</details>

Figure 1: AI participates in the Olympics from the Gaokao [57] venue.

![](images/f140be37b70455517bab6e807a4a47c6880a0e9da3920cd47212acf7294d1392.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Comprehensive"] --> B["7 Disciplines"]
    A --> C["34 Branches"]
    A --> D["62 Olympiads"]
    A --> E["Multimodal"]
    A --> F["13 Answer Types"]
    G["Data Leakage Detection"] --> H["Rigorous"]
    I["Highly-challenging"] --> J["Cognitive Reasoning"]
    I --> K["8 Logical"]
    I --> L["5 Visual"]
    I --> M["Olympic"]
    N["Fine-grained Evaluation"] --> O["Discipline"]
    N --> P["Cognitive"]
    Q["Answer"] --> R["Process"]
    S["Data Leakage Detection"] --> T["Olympic Arena"]
    U["Step 1: Calculate the Duration of a Solar Eclipse for the Spaceship: The time it takes for the spaceship to experience a total solar eclipse can be calculated using the Earth's rotational angular velocity. The Earth's angular velocity is given by: ω = 2π/τ₀. The duration of the &quot;total eclipse&quot; for the spaceship is: t = α/ω = αT₀/2π"]
    V["Step 2: Verify the Spaceship's Orbital Period Formula: The orbital period T...."]
    W["Step 3: Derive the Orbital Radius: The orbital radius r of the spaceship is ...."]
    X["So the final answers are (αT₀/2π)·2π√(R³/(αMsin²/σ))"]
    Y["A spaceship orbits the Earth in a circular motion with a period T. Due to the Earth blocking the sunlight, it experiences a &quot;total solar eclipse&quot; (astronauts on the spaceship cannot see the sun) as shown in [figure1"]. Given that the Earth's radius is R, Earth's mass is M, the gravitational constant is G, the Earth's rotation period is T₀, and the sunlight can be considered parallel light, astronauts on the spaceship measure the angle α to the Earth at point A. Calculate the duration of each "total solar eclipse" process for the spaceship and the period of the spaceship.]
    X --> Z["Step 1: Calculate the Duration of a Solar Eclipse for the Spaceship: The time it takes for the spaceship to experience a total solar eclipse can be calculated using the Earth's rotational angular velocity. The Earth's angular velocity is given by: ω = 2π/τ₀. The duration of the &quot;total eclipse&quot; for the spaceship is: t = α/ω = αT₀/2π"]
    X --> AA["Step 2: Verify the Spaceship's Orbital Period Formula: The orbital period T...."]
    X --> AB["Step 3: Derive the Orbital Radius: The orbital radius r of the spaceship is ...."]
    X --> AC["So the final answers are (αT₀/2π)·2π√(R³/(αMsin²/σ))"]
```
</details>

Figure 2: The overview of our OlympicArena benchmark.

# Abstract

The evolution of Artificial Intelligence (AI) has been significantly accelerated by advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), gradually showcasing potential cognitive reasoning abilities in problem-solving and scientific discovery (i.e., AI4Science) once exclusive to human intellect. To comprehensively evaluate current models' performance in cognitive reasoning abilities, we introduce OlympicArena, which includes 11,163 bilingual problems across both text-only and interleaved text-image modalities. These challenges encompass a wide range of disciplines spanning seven fields and 62 international Olympic competitions, rigorously examined for data leakage. We argue that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries. Beyond evaluating performance across various disciplines using answer-only criteria, we conduct detailed experiments and analyses from multiple perspectives. We delve into the models' cognitive reasoning abilities, their performance across different modalities, and their outcomes in process-level evaluations, which are vital for tasks requiring complex reasoning with lengthy solutions. Our extensive evaluations reveal that even advanced models like GPT-4o only achieve a 39.97% overall accuracy (28.67% for mathematics and 29.71% for physics), illustrating current AI limitations in complex reasoning and multimodal integration. Through the OlympicArena, we aim to advance AI towards superintelligence, equipping it to address more complex challenges in science and beyond. We also provide a comprehensive set of resources to support AI research, including a benchmark dataset, an open-source annotation platform, a detailed evaluation tool, and a leaderboard with automatic submission features. $^{1}$

# 1 Introduction

The landscape of Artificial Intelligence (AI) has undergone a transformative evolution with advances in technologies like Large Language Models $[2, 3]$ and Large Multimodal Models (LMMs) $[31]$ . These models represent significant milestones on the path to Artificial General Intelligence (AGI) $[47, 15]$ , demonstrating remarkable cognitive reasoning abilities, which represent drawing meaningful conclusions from incomplete and inconsistent knowledge to solve problems in complex scenarios $[16, 34]$ . They adeptly handle tasks ranging from simple grade school math problems $[13, 56, 59, 64]$ to complex challenges like those presented at the International Mathematical Olympiad (IMO) $[46, 42]$ . Furthermore, they are progressively being applied to intricate real-world scenarios, such as using AI agents for software development $[37]$ , collaborating on complex decision-making processes $[11]$ and even boosting the field of scientific research (i.e., AI4Science) $[50]$ .

These applications highlight AI's growing proficiency in cognitive reasoning, a crucial element in the pursuit of AGI and, potentially, superintelligence [35]. Therefore, how to benchmark these abilities has sparked extensive research. Existing benchmarks [18, 22, 26, 63, 44, 62] utilize multidisciplinary exam problems to assess the problem-solving skills of LLMs, but these problems are predominantly knowledge-intensive which has become relatively easy for current LLMs. Also, these benchmarks primarily focus on text-only modalities. Although some benchmarks begin to target college-level problems [52, 40] and incorporate multimodal assessments [58, 60, 61], they still predominantly focus on knowledge-intensive tasks or simple concept applications (shown in Table 1). Concurrent to our work, He et al. [17] introduces an Olympic-level benchmark yet it is limited to only mathematics and physics. Furthermore, all the above benchmarks lack a systematic and fine-grained evaluation of various cognitive reasoning abilities. For example, they mostly do the evaluation only based on answers, neglecting potential errors in the reasoning process. This underscores the need for more comprehensive evaluations that not only cover a broader range of disciplines but also focus on higher levels of cognitive reasoning as well as fine-grained evaluation.

In this paper, we introduce OlympicArena, a comprehensive, highly-challenging, and rigorously curated benchmark featuring a detailed, fine-grained evaluation mechanism designed to assess

advanced AI capabilities across a broad spectrum of Olympic-level challenges (as illustrated in Figure 2). We extensively select, collect, and process problems from seven disciplines—mathematics, physics, chemistry, biology, geography, astronomy, and computer science—encompassing 62 different Olympic-level competitions. This extensive collection has culminated in a benchmark comprising 11,163 problems, categorized into 13 types of answers (e.g., expression, interval). Importantly, OlympicArena enhances its evaluation framework by incorporating process-level evaluations that scrutinize the step-by-step reasoning processes of AI models. This approach is critical for understanding the depth of cognitive reasoning beyond correct answers [29, 53], allowing us to identify and rectify gaps in AI reasoning pathways and ensuring more robust AI capabilities. The benchmark is bilingual, featuring both English and Chinese, to enhance its accessibility and global applicability. Additionally, it supports two modalities: text-only and interleaved text and images, catering to the evolving complexity of tasks that modern AI systems must handle. We also perform data leakage detection experiments [54] on some mainstream models to validate our benchmark's effectiveness.

We conduct a series of experiments across existing top-performing LMMs, encompassing both proprietary models (e.g., GPT-4o [36]) and open-source models (e.g., LLaVa-NeXT [31]). Additionally, we evaluate various types of LLMs (e.g., GPT-3.5) in two settings: text-only and image-caption and conduct comprehensive evaluations from both the answer-level and process-level perspectives. For answer-level evaluations, we combine rule-based and model-based (GPT-4V $^{2}$ in this paper) methods to cover a more diverse range of answer types. For process-level evaluations, we score each reasoning step of the model output, which we consider quite critical in reasoning scenarios. Additionally, we perform fine-grained evaluations and analyses on different types of cognitive reasoning, from both logical and visual perspectives to better interpret the current capabilities of AI.

Our observations from the $OlympicArena$ benchmark are summarized as follows: (1) Even the most advanced model, GPT-4o, achieves only a 39.97% overall accuracy, while other open-source models struggle to reach a 20% overall accuracy, underscoring current models' limitations in handling complex, multidisciplinary problems that require advanced cognitive reasoning—key aspects of scientific discovery. (2) Through more fine-grained analysis § 4.4, we find that LMMs are particularly weak in handling complex, decompositional reasoning problems and exhibit poor spatial and geometric perception visual abilities, as well as difficulties in understanding abstract symbols. (3) Additionally, we discover that current LMMs seem to struggle significantly in leveraging interleaved visual information for complex cognitive reasoning problems. Various LMMs fail to show notable enhancements compared to their text-only counterparts. (4) The process-level evaluation also indicates that most models can correctly execute some reasoning steps in spite of providing incorrect answers, demonstrating the models' significant potential. (5) Through data leakage detection, we find that instances of data leakage in our benchmark are exceedingly rare. Even on the infrequent occasions when leakage does occur, the corresponding models do not consistently solve these problems correctly. This suggests the need for more advanced training strategies to enhance cognitive reasoning capabilities. These observations highlight the immense value of the $OlympicArena$ benchmark in advancing our understanding of AI's capabilities and limitations.

# 2 Related Work

Benchmark AI Intelligence How to benchmark AI intelligence has always been a challenging problem. Initially, the Turing Test $[47]$ provided a conceptual framework for evaluating AI Intelligence. However, limitations in past AI technology lead researchers to focus on specialized domains. In computer vision, benchmarks like MNIST $[25]$ and ImageNet $[14]$ catalyze progress, while in natural language processing, GLUE $[49]$ and XTREME $[21]$ set the standard for evaluating linguistic capabilities across tasks and languages. The success of pretrained language models $[38, 23]$ particularly recent LLMs emphasizes the evaluation of foundational knowledge and innate abilities as shown in Figure 2. This leads to the creation of benchmarks such as MMLU $[18]$ , AGIEval $[63]$ , C-Eval $[22]$ , and CMMLU $[26]$ , which pushed the limits of language models with multidisciplinary, multilingual, and knowledge-intensive tasks. However, the rapid progress of LLMs has rendered these benchmarks insufficient to fully assess the models' growing capabilities.

Table 1: Comparison of various benchmarks. “Subjects”: ■ Math, ■ Physics, ■ Chemistry, ■ Biology, ■ Geography, ■ Astronomy, ■ Computer Science. “Multimodal” indicates whether the benchmark contains visual information. “Language”: “EN” for English and “ZH” for Chinese. “Size” represents the number of test problems. “#Answer” shows the number of answer types (from Appendix A.2). “Eval.” details evaluation methods: ■ rule-based, ■ model-based, ■ answer-level, ■ process-level. “Leak Det.” indicates if data leakage detection is conducted. “Difficulty” shows problem proportions at difficulty levels: ■ Knowledge Recall, ■ Concept Application, ■ Cognitive Reasoning. "#Logic." indicates the average logical reasoning abilities per question, and "#Visual." indicates the average visual reasoning abilities per multimodal question. Cognitive reasoning abilities are detailed in § 3.3.

<table><tr><td>Benchmark</td><td>Subjects</td><td>Multimodal</td><td>Language</td><td>Size</td><td>#Answer</td><td>Eval.</td><td>Leak Det.</td><td>Difficulty</td><td>#Logic.</td><td>#Visual.</td></tr><tr><td>SciBench</td><td></td><td>√</td><td>EN</td><td>789</td><td>1</td><td></td><td>×</td><td></td><td>0.39</td><td>2.35</td></tr><tr><td>CMMLU</td><td></td><td>×</td><td>ZH</td><td>1594</td><td>1</td><td></td><td>×</td><td></td><td>0.36</td><td>-</td></tr><tr><td>MMLU</td><td></td><td>×</td><td>EN</td><td>2554</td><td>1</td><td></td><td>×</td><td></td><td>0.44</td><td>-</td></tr><tr><td>C-Eval</td><td></td><td>×</td><td>ZH</td><td>3362</td><td>1</td><td></td><td>×</td><td></td><td>0.6</td><td>-</td></tr><tr><td>MMMU</td><td></td><td>√</td><td>EN</td><td>3007</td><td>2</td><td></td><td>×</td><td></td><td>0.25</td><td>2.75</td></tr><tr><td>SciEval</td><td></td><td>×</td><td>EN</td><td>15901</td><td>4</td><td></td><td>×</td><td></td><td>1.12</td><td>-</td></tr><tr><td>AGIEval</td><td></td><td>×</td><td>EN &amp; ZH</td><td>3300</td><td>2</td><td></td><td>×</td><td></td><td>1.07</td><td>-</td></tr><tr><td>GPQA</td><td></td><td>×</td><td>EN</td><td>448</td><td>1</td><td></td><td>×</td><td></td><td>2.24</td><td>-</td></tr><tr><td>JEEBench</td><td></td><td>×</td><td>EN</td><td>515</td><td>3</td><td></td><td>×</td><td></td><td>2.41</td><td>-</td></tr><tr><td>OlympiadBench</td><td></td><td>√</td><td>EN &amp; ZH</td><td>8952</td><td>7</td><td></td><td>×</td><td></td><td>2.26</td><td>2.96</td></tr><tr><td>OlympicArena</td><td></td><td>√</td><td>EN &amp; ZH</td><td>11163</td><td>13</td><td></td><td>√</td><td></td><td>2.73</td><td>3.15</td></tr></table>

Cognitive Reasoning is crucial as it allows AI systems to apply prior knowledge and logical principles to complex tasks in a more human-like manner, ensuring better robustness and generalization in real-world applications $[43]$ . Thus, more attention is paid to more intricate reasoning tasks, benchmarks like GSM8K $[13]$ focused on grade-school mathematical reasoning problems, while MATH $[20]$ introduced high-school level mathematical competition tasks. Furthermore, benchmarks such as JEEBench $[4]$ , SciBench $[52]$ , GPQA $[40]$ and MMMU $[58]$ have expanded the scope by incorporating multidisciplinary university-level subjects and even multimodal tasks. To further challenge AI systems, researchers have turned to problems from some of the most difficult competitions, specifically International Olympiads $[17, 46, 30]$ and algorithmic challenges $[28, 19, 41]$ . Nevertheless, there is currently no Olympic-level, multidisciplinary benchmark that comprehensively evaluates comprehensive problem-solving abilities to fully test all-rounded AI's cognitive ability. Table 1 presents a comparison of several related scientific benchmarks.

Rigorous Evaluation for Reasoning While curating comprehensive and appropriate data is crucial in benchmarks, adopting rigorous evaluation methodologies is equally important. Most existing benchmarks, as mentioned above, primarily focus on answer-level evaluation (i.e., only comparing the model's output with the standard answer). Recently, some works have started to focus on the models' intermediate reasoning steps. Some of them $[48, 29, 51]$ explore using process supervision to train better reward models. Lanham et al. $[24]$ delves into the faithfulness of the chain-of-thought reasoning process, while Xia et al. $[53]$ trains models specifically designed to evaluate the validity and redundancy of reasoning steps for mathematical problems. However, in the evaluation methodologies of existing benchmarks as listed in Table 1, few of them incorporate process-level evaluation. This insufficient evaluation often neglects the reliability and faithfulness of AI models, especially in complex cognitive reasoning scenarios requiring lengthy solutions. In this work, the introduced OlympicArena is equipped with a more fine-grained evaluation methodology (i.e., process-level evaluation), allowing developers to better understand the true reasoning behaviors of models.

# 3 The OlympicArena Benchmark

# 3.1 Overview

We introduce the OlympicArena, an Olympic-level, multidisciplinary benchmark designed to rigorously assess the cognitive reasoning abilities of LLMs and LMMs. Our benchmark features a combination of text-only and interleaved text-image modalities, presented bilingually to promote accessibility and inclusivity. It spans seven core disciplines: mathematics, physics, chemistry, biology, geography, astronomy, and computer science, encompassing a total of 34 specialized branches (details are in Appendix A.1) which represent fundamental scientific fields. The benchmark

includes a comprehensive set of 11,163 problems from 62 distinct Olympic competitions, structured with 13 answer types (shown in Appendix A.2) from objective types (e.g., multiple choice and fill-in-the-blanks) to subjective types (e.g., short answers and programming tasks), which distinguishes it from many other benchmarks that primarily focus on objective problems. Detailed statistics of OlympicArena are described in Table 2. Also, to identify potential data leakage, we conduct specialized data leakage detection experiments on several models.

Furthermore, in pursuit of a granular analysis of model performance, we categorize cognitive reasoning into 8 types of logical reasoning abilities and 5 types of visual reasoning abilities. This comprehensive categorization aids in the detailed evaluation of the diverse and complex reasoning skills that both LLMs and LMMs can exhibit. Additionally, we specifically investigate all multimodal problems to compare the performance of LMMs against their text-based counterparts, aiming to better assess LMMs' capabilities in handling visual information. Finally, we evaluate the correctness and efficiency of the reasoning process, not just limited to an answer-based assessment.

Table 2: Benchmark Statistics 

<table><tr><td>Statistic</td><td>Number</td></tr><tr><td>Total Problems</td><td>11163</td></tr><tr><td>Total Competitions</td><td>62</td></tr><tr><td>Total Subjects/Subfields</td><td>7/34</td></tr><tr><td>Total Answer Types</td><td>13</td></tr><tr><td>Problems with Solutions</td><td>7904</td></tr><tr><td>Language (EN: ZH)</td><td>7054: 4109</td></tr><tr><td>Total Images</td><td>7571</td></tr><tr><td>Problems with Images</td><td>4960</td></tr><tr><td>Image Types</td><td>5</td></tr><tr><td>Cognitive Complexity Levels</td><td>3</td></tr><tr><td>Logical Reasoning Abilities</td><td>8</td></tr><tr><td>Visual Reasoning Abilities</td><td>5</td></tr><tr><td>Average Problem Tokens</td><td>244.8</td></tr><tr><td>Average Solution Tokens</td><td>417.1</td></tr></table>

# 3.2 Data Collection

To ensure comprehensive coverage of Olympic-level problems across various disciplines, we begin by collecting URLs of various competitions where problems are publicly available for download in PDF format. Then, we utilize the Mathpix $^{3}$ tool to convert these PDF documents into markdown format, making them compatible with input requirements for models. Specifically, for the programming problems of Computer Science, we additionally collect corresponding test cases. We strictly adhere to copyright and licensing considerations, ensuring compliance with all relevant regulations.

# 3.3 Data Annotation

Problem Extraction and Annotation. To extract individual problems from the markdown format of the test papers, we employ about 30 students with background in science and engineering. We have developed a user interface for annotating multimodality data, which has been released. $^{4}$ To facilitate further research and the process-level evaluation of models, we annotate meta-information like solutions if provided. To ensure data quality, we implement a multi-step validation process after the initial annotation is completed. More details can be seen in Appendix B.1. After collecting all the problems, we perform deduplication within each competition based on model embeddings to remove repeated problems that may appear in multiple test papers from the same year. To further demonstrate that our benchmark emphasizes cognitive reasoning more than most other benchmarks, we categorize the difficulty of the problems into three levels and make comparison with other related benchmarks. Specifically, we classify all problems into: knowledge recall, concept application and cognitive reasoning. We utilize GPT-4V as the annotator for categorizing different difficulty levels $^{5}$ (detailed definitions and specific prompts can be found in Appendix B.2). $^{6}$

Annotation of Cognitive Reasoning Abilities. To facilitate better fine-grained analysis, we categorize cognitive reasoning abilities from both logical and visual perspectives $[16, 43]$ . The logical reasoning abilities encompass Deductive Reasoning (DED), Inductive Reasoning (IND), Abductive Reasoning (ABD), Analogical Reasoning (ANA), Cause-and-Effect Reasoning (CAE), Critical Thinking (CT), Decompositional Reasoning (DEC), and Quantitative Reasoning (QUA). Meanwhile, the visual reasoning abilities include Pattern Recognition (PR), Spatial Reasoning (SPA), Diagrammatic Reasoning (DIA), Symbol Interpretation (SYB), and Comparative Visualization (COM). We also utilize GPT-4V as the annotator for categorizing different cognitive abilities (detailed definitions and specific prompts can be found in Appendix B.3). $^{6}$ With these annotations, we can conduct a more fine-grained analysis of the current cognitive reasoning abilities of AI.

# 3.4 Data Splitting

Our benchmark includes 11,163 problems, with 548 designated for model-based evaluation as OlympicArena-ot. We sample 638 problems across subjects to create OlympicArena-val for hyperparameter tuning or small-scale testing. OlympicArena-val problems have step-by-step solutions, supporting research like process-level evaluation. The remaining problems form OlympicArena-test, the official test set with unreleased answers for formal testing. The results in this paper are based on the entire benchmark dataset, including OlympicArena-ot, OlympicArena-val, and OlympicArena-test.

# 4 Experiments

# 4.1 Experimental Setup

To comprehensively evaluate the capabilities of LLMs and LMMs (selected models are listed in Appendix C.2) across different modalities, we design our experiments to include three distinct settings: multimodal, image-caption, and text-only. In the multimodal setting, we assess the ability of LMMs to leverage visual information by interleaving text and images, simulating real-world scenarios. For models unable to handle interleaved inputs, we concatenate multiple images into a single input. For LMMs requiring necessary image inputs, their text-based counterparts handle text-only problems. In the image-caption setting, we explore whether textual descriptions of images enhance the problem-solving capabilities of LLMs. Using InternVL-Chat-V1.5 $^{7}$ [12], we generate captions for all images based on prompts detailed in Appendix C.1. These captions replace the original image inputs. In the text-only setting, we evaluate the performance of LLMs without any visual information, serving as a baseline to compare against the multimodal and image-caption settings. All experiments use zero-shot prompts, tailored to each answer type and specifying output formats to facilitate answer extraction and rule-based matching. It also minimizes biases typically associated with few-shot learning [32, 33]. Detailed prompt designs are provided in Appendix C.3.

# 4.2 Evaluation

Answer-level Evaluation We combine rule-based and model-based methods to cover a diverse range of problems. For problems with fixed answers, we extract the final answer and perform rule-based matching according to the answer type. For code generation tasks, we use the unbiased pass@k metric $[10]$ to test all test cases. For problems with answer types categorized as "others" which are difficult to be evaluated using rule-based matching (e.g., chemical equation writing problems), we employ GPT-4V as an evaluator to assess the responses. To ensure the reliability of GPT-4V as an evaluator, we manually sample and check the correctness. See Appendix C.5 for more details.

Process-level Evaluation To further investigate the correctness of the reasoning steps, ensuring a rigorous assessment of the cognitive abilities of models, we conduct the process-level evaluation. We first sample 96 problems with reference solutions from OlympicArena. We employ GPT-4 to convert both the references (i.e., gold solutions) and the model-generated solutions into a structured step-by-step format. We then provide these solutions to GPT-4V and score each step for its correctness on a scale ranging from 0 to 1. $^{8}$ The experimental details can be seen in Appendix C.6. To validate the consistency with human judgment, we obtain some samples for human annotations. The results indicate that our model-based evaluation method is highly accurate, with an 83% inter-annotator agreement.

# 4.3 Main Results

Table 3 presents the evaluation results of various LMMs and LLMs on OlympicArena. We obtain the following observations: (1) Even the most advanced large model, GPT-4o, achieves only a 39.97% overall accuracy, while other open-source models struggle to reach a 20% overall accuracy. This stark contrast highlights the significant difficulty and rigor of our benchmark, demonstrating its effectiveness in pushing the boundaries of current AI capabilities. (2) Furthermore, compared to subjects like biology and geography, we observe that mathematics and physics remain the two most

Table 3: Experimental results on OlympicArena, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. We use the pass@k metric (Equation 1) for CS problems. When calculating the overall accuracy, for code generation problems, if any generated code for a problem passes all test cases, the problem is considered correct. 

<table><tr><td rowspan="2">Model</td><td>Math</td><td>Physics</td><td>Chemistry</td><td>Biology</td><td>Geography</td><td>Astronomy</td><td>CS</td><td>Overall</td></tr><tr><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Pass@1</td><td>Accuracy</td></tr><tr><td colspan="9">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>1.58</td><td>3.74</td><td>7.01</td><td>7.31</td><td>4.53</td><td>5.48</td><td>0</td><td>4.31</td></tr><tr><td>Yi-34B-Chat</td><td>3.06</td><td>9.77</td><td>23.53</td><td>32.67</td><td>35.03</td><td>18.15</td><td>0.17</td><td>17.31</td></tr><tr><td>Internlm2-20B-Chat</td><td>5.88</td><td>9.48</td><td>18.36</td><td>31.90</td><td>32.14</td><td>16.03</td><td>0.60</td><td>16.62</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>9.65</td><td>14.54</td><td>29.84</td><td>38.58</td><td>40.69</td><td>28.05</td><td>0.51</td><td>23.69</td></tr><tr><td>GPT-3.5</td><td>7.27</td><td>10.92</td><td>23.03</td><td>31.19</td><td>31.13</td><td>16.93</td><td>3.85</td><td>18.27</td></tr><tr><td>Claude3 Sonnet</td><td>7.76</td><td>17.24</td><td>29.46</td><td>38.25</td><td>40.94</td><td>24.04</td><td>1.62</td><td>23.02</td></tr><tr><td>GPT-4</td><td>19.46</td><td>24.77</td><td>42.52</td><td>46.47</td><td>44.97</td><td>33.44</td><td>7.78</td><td>32.37</td></tr><tr><td>GPT-4o</td><td>28.33</td><td>29.54</td><td>46.24</td><td>49.42</td><td>48.36</td><td>43.25</td><td>8.46</td><td>38.17</td></tr><tr><td colspan="9">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>1.76</td><td>3.56</td><td>6.75</td><td>7.83</td><td>7.17</td><td>6.87</td><td>0</td><td>4.89</td></tr><tr><td>Yi-34B-Chat</td><td>3.01</td><td>9.94</td><td>21.45</td><td>31.26</td><td>34.78</td><td>17.33</td><td>0.17</td><td>16.72</td></tr><tr><td>Internlm2-20B-Chat</td><td>5.94</td><td>10.40</td><td>20.25</td><td>31.00</td><td>32.52</td><td>16.93</td><td>0.73</td><td>17.07</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>9.56</td><td>14.31</td><td>29.84</td><td>38.51</td><td>40.75</td><td>27.2</td><td>0.60</td><td>23.43</td></tr><tr><td>GPT-3.5</td><td>7.16</td><td>14.48</td><td>23.97</td><td>30.94</td><td>33.52</td><td>18.56</td><td>4.70</td><td>18.83</td></tr><tr><td>Claude3 Sonnet</td><td>7.52</td><td>18.10</td><td>29.84</td><td>38.77</td><td>41.14</td><td>22.65</td><td>2.39</td><td>23.10</td></tr><tr><td>GPT-4</td><td>19.46</td><td>26.21</td><td>41.58</td><td>45.89</td><td>48.18</td><td>35</td><td>7.63</td><td>33.00</td></tr><tr><td>GPT-4o</td><td>28.27</td><td>29.71</td><td>45.87</td><td>51.16</td><td>49.12</td><td>43.17</td><td>9.57</td><td>38.50</td></tr><tr><td colspan="9">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>1.73</td><td>4.25</td><td>8.64</td><td>12.13</td><td>13.77</td><td>7.85</td><td>0</td><td>6.90</td></tr><tr><td>Yi-VL-34B</td><td>2.94</td><td>9.94</td><td>19.81</td><td>27.73</td><td>25.16</td><td>16.60</td><td>0</td><td>14.49</td></tr><tr><td>InternVL-Chat-V1.5</td><td>6.03</td><td>9.25</td><td>19.12</td><td>30.39</td><td>32.96</td><td>15.94</td><td>0.38</td><td>16.63</td></tr><tr><td>LLaVA-NeXT-34B</td><td>3.03</td><td>10.06</td><td>21.45</td><td>33.18</td><td>36.92</td><td>18.15</td><td>0.18</td><td>17.38</td></tr><tr><td>Qwen-VL-Max</td><td>6.93</td><td>12.36</td><td>23.79</td><td>36</td><td>40.19</td><td>23.39</td><td>0.77</td><td>20.65</td></tr><tr><td>Gemini Pro Vision</td><td>6.28</td><td>12.47</td><td>28.14</td><td>37.48</td><td>37.42</td><td>20.20</td><td>1.45</td><td>20.97</td></tr><tr><td>Claude3 Sonnet</td><td>7.52</td><td>18.16</td><td>29.27</td><td>38.96</td><td>40.13</td><td>25.02</td><td>1.45</td><td>23.13</td></tr><tr><td>GPT-4V</td><td>19.27</td><td>24.83</td><td>41.45</td><td>46.79</td><td>49.62</td><td>32.46</td><td>7.00</td><td>32.76</td></tr><tr><td>GPT-4o</td><td>28.67</td><td>29.71</td><td>46.69</td><td>52.18</td><td>56.23</td><td>43.91</td><td>9.00</td><td>39.97</td></tr></table>

challenging disciplines, likely due to their reliance on complex reasoning abilities. (3) Computer programming competitions also prove to be highly difficult, with some open-source models failing to solve any of them, indicating current models' poor abilities to design efficient algorithms to solve complex problems.

# 4.4 Fine-grained Analysis

To achieve a more fine-grained analysis of the experimental results, we conduct further evaluations based on different modalities and reasoning abilities. Additionally, we also conduct an analysis of the process-level evaluation. Key findings are as follows:

Models exhibit varied performance across different logical and visual reasoning abilities. As shown in Figure 3, almost all models demonstrate similar performance trends across different logical reasoning abilities. They tend to excel in Abductive Reasoning and Cause-and-Effect Reasoning, doing well in identifying causal relationships from the provided information. Conversely, models perform poorly in Inductive Reasoning and Decompositional Reasoning. This is due to the diverse and unconventional nature of Olympic-level problems, which require the ability to break down complex problems into smaller sub-problems. In terms of visual reasoning abilities, models tend to be better at Pattern Recognition and Comparative Visualization. However, they struggle with tasks involving spatial and geometric reasoning as well as those need to understand abstract symbols. The completed results are presented in Appendix D.1.

Most LMMs are still not proficient at utilizing visual information. As displayed in Figure 4a, only a few LMMs (such as GPT-4o and Qwen-VL-Chat) show significant improvement with image inputs compared to their text-based counterpart. Many LMMs do not exhibit enhanced performance

![](images/f9979437918d2ca25b9611d06033fc641c5bb3d7cd0a2e914dcfb0752643295f.jpg)

<details>
<summary>radar</summary>

| Category | GPT-4o | GPT-4V | Claude3-sonnet | LLaVa-NeXT-34B | InternVL-Chat-V1.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DED | 40 | 32 | 22 | 18 | 16 |
| IND | 38 | 28 | 20 | 16 | 14 |
| ABD | 45 | 42 | 30 | 28 | 26 |
| ANA | 35 | 38 | 25 | 22 | 20 |
| CAE | 42 | 40 | 28 | 25 | 23 |
| CT | 48 | 38 | 30 | 27 | 25 |
| DEC | 35 | 30 | 22 | 18 | 16 |
| QUA | 38 | 28 | 20 | 16 | 14 |
| PR | 45 | 35 | 25 | 22 | 20 |
| SYB | 40 | 30 | 22 | 18 | 16 |
| COM | 42 | 32 | 25 | 20 | 18 |
| DIA | 40 | 30 | 28 | 22 | 20 |
| SPA | 38 | 28 | 25 | 18 | 16 |
| DED | 40 | 32 | 28 | 20 | 18 |
| IND | 35 | 25 | 20 | 16 | 14 |
| ABD | 45 | 40 | 30 | 25 | 22 |
| ANA | 35 | 38 | 25 | 20 | 16 |
| CAE | 40 | 40 | 28 | 22 | 18 |
| CT | 45 | 38 | 30 | 25 | 20 |
| DEC | 35 | 30 | 22 | 18 | 16 |
| QUA | 38 | 28 | 20 | 16 | 14 |
| PR | 45 | 35 | 25 | 20 | 18 |
| SYB | 40 | 30 | 22 | 18 | 16 |
| COM | 42 | 32 | 25 | 20 | 18 |
| DIA | 40 | 30 | 28 | 22 | 20 |
| SPA | 38 | 28 | 25 | 18 | 16 |
| DED | -5 | -5 | -5 | -5 | -5 |
| IND | -10 | -10 | -10 | -10 | -10 |
| ABD | -15 | -15 | -15 | -15 | -15 |
| ANA | -10 | -10 | -10 | -10 | -10 |
| CAE | -15 | -15 | -15 | -15 | -15 |
| CT | -10 | -10 | -10 | -10 | -10 |
| DEC | -15 | -15 | -15 | -15 | -15 |
| QUA | -10 | -10 | -10 | -10 | -10 |
| PR | -5 | -5 | -5 | -5 | -5 |
| SYB | -10 | -10 | -10 | -10 | -10 |
| COM | -15 | -15 | -15 | -15 | -15 |
| DIA | -20 | -20 | -20 | -20 | -20 |
| SPA | -25 | -25 | -25 | -25 | -25 |
| DED | -30 | -30 | -30 | -30 | -30 |
| IND | -35 | -35 | -35 | -35 | -35 |
| ABD | -40 | -40 | -40 | -40 | -40 |
| ANA | -35 | -35 | -35 | -35 | -35 |
| CAE | -40 | -40 | -40 | -40 | -40 |
| CT | -35 | -35 | -35 | -35 | -35 |
| DEC | -40 | -40 | -40 | -40 | -40 |
| QUA | -35 | -35 | -35 | -35 | -35 |
| PR at visual level: GPT-4o; GPT-4V; Claude3-sonnet; LLaVa-NeXT-34B; InternVL-Chat-V1.5; all values are estimated based on the chart's visual legend. The y-axis represents a numerical scale from approximately to 5. The x-axis is labeled with the same categories and the y-axis is labeled with the same label 'PR'. The chart displays a single data series for each model. The visual legend includes the same color coding for each model.
</details>

Figure 3: Performance of various models on logical and visual reasoning abilities. Logical reasoning abilities: Deductive Reasoning (DED), Inductive Reasoning (IND), Abductive Reasoning (ABD), Analogical Reasoning (ANA), Cause-and-Effect Reasoning (CAE), Critical Thinking (CT), Decompositional Reasoning (DEC), and Quantitative Reasoning (QUA). Visual reasoning abilities: Pattern Recognition (PR), Spatial Reasoning (SPA), Diagrammatic Reasoning (DIA), Symbol Interpretation (SYB), and Comparative Visualization (COM).

with image inputs and some even show decreased effectiveness when handling images. Possible reasons include: (1) When text and images are input together, LMMs may focus more on the text, neglecting the information in the images. This conclusion has also been found in some other works $[61, 9]$ . (2) Some LMMs, while training their visual capabilities based on their text-based models, may lose some of their inherent language abilities (e.g., reasoning abilities), which is particularly evident in our scenarios. (3) Our problems use a complex interleaved text and image format, which some models do not support well, leading to difficulties in processing and understanding the positional information of images embedded within the text. $^{9}$

![](images/84543f1852d26090b4a9bd9336d33529273f8b508ff1477dcd374e12466841c2.jpg)

<details>
<summary>bar</summary>

| Model | LMMs (%) | Image caption + LLMs (%) | LLMs (%) |
|---|---|---|---|
| Qwen-VL-Chat | 7.5 | 6.0 | 5.0 |
| InternVL-Chat-V1.5 | 18.0 | 17.0 | 18.0 |
| LLaVA-NeXT-34B | 20.0 | 19.0 | 20.0 |
| Claude3-Sonnet | 23.0 | 22.0 | 23.0 |
| GPT-4v | 31.0 | 32.0 | 31.0 |
| GPT-4o | 38.0 | 34.0 | 34.0 |
</details>

(a)

![](images/353a573e73fbae7004a9724b5a4b874ee1e9c57cede975f7dbd751d926684700.jpg)

<details>
<summary>scatter</summary>

| Answer-level Accuracy | Process-level Score |
| --------------------- | ------------------- |
| 5                     | 30                  |
| 7                     | 28                  |
| 10                    | 32                  |
| 12                    | 34                  |
| 14                    | 36                  |
| 16                    | 38                  |
| 18                    | 40                  |
| 20                    | 42                  |
| 22                    | 44                  |
| 24                    | 46                  |
| 26                    | 48                  |
| 28                    | 50                  |
| 30                    | 52                  |
| 32                    | 54                  |
| 34                    | 56                  |
| 36                    | 58                  |
| 38                    | 60                  |
| 40                    | 62                  |
</details>

(b)

![](images/31cf54c2b04696af4a04f3695ea34908d2e6248377608f4872573ffae6780ec6.jpg)

<details>
<summary>bar</summary>

| Location of Incorrect Steps | Percentage |
| --------------------------- | ---------- |
| 0.0                         | 0.5        |
| 0.1                         | 2.0        |
| 0.2                         | 3.0        |
| 0.3                         | 4.5        |
| 0.4                         | 5.5        |
| 0.5                         | 6.0        |
| 0.6                         | 7.0        |
| 0.7                         | 8.0        |
| 0.8                         | 7.5        |
| 0.9                         | 5.5        |
| 1.0                         | 17.5       |
</details>

(c)   
Figure 4: (a) Comparison of different LMMs and their corresponding LLMs across three different experimental settings. For details on the corresponding LLMs for each LMM, refer to the Appendix C.2. (b) The correlation between answer-level and process-level scores of all the models over all the sampled problems. (c) Distribution of the locations of incorrect steps, represented as the proportion of steps from left to right in the entire process, over all the sampled problems.

Analysis of process-level evaluation results Through process-level evaluation (complete results are in Table 14), we discover following insights: (1) There is generally a high consistency between process-level evaluation and answer-level evaluation. When a model produces a correct answer, the quality of the reasoning process tends to be higher most of the time (see Figure 4b). (2) The accuracy at the process-level is often higher than at the answer-level. This indicates that even for very

complex problems, the model can correctly perform some of the intermediate steps. Therefore, the model likely has significant untapped potential for cognitive reasoning, which opens new avenues for researchers to explore. We also find that in a few disciplines, some models that perform well at the answer level fall behind at the process level. We speculate that this is because models sometimes tend to overlook the reasonableness of intermediate steps when generating answers, even though these steps may not be crucial to the final result. (3) Additionally, we conduct a statistical analysis of the location distribution of error steps (see Figure 4c). We identify that a higher proportion of errors occur in the later stages. This suggests that models are more prone to making mistakes as reasoning accumulates, indicating a need for improvement in handling long chains of logical deductions.

Error analysis To further concretize models' performance, we sample incorrect responses from GPT-4V (16 problems per subject, with 8 text-only and 8 multimodal) and have human evaluators analyze and annotate the reasons for these errors. As depicted in Figure 5, reasoning errors (both logical and visual) constitute the largest category, indicating that our benchmark effectively highlights the current models' deficiencies in cognitive reasoning abilities. Additionally, a significant portion of errors stem from knowledge deficits, suggesting that current models still lack expert-level domain knowledge and the ability to leverage this knowledge to assist in reasoning. Another category of errors arise from understanding biases, which can be attributed to the models' misinterpretation of context and difficulties in integrating complex language structures and multimodal information. More relevant cases are shown in Appendix F.1.

![](images/88483382be03536faeafd1838ab118b165b0befa33b1bf02c36228a57e636bc9.jpg)

<details>
<summary>pie</summary>

| Error Type | Percentage (%) |
| :--- | :--- |
| Logical Reasoning Error | 48.21 |
| Visual Reasoning Error | 9.82 |
| Instruction Following Error | 1.79 |
| Understanding Error | 13.39 |
| Knowledge Deficit Error | 16.07 |
| Partial Response | 6.25 |
| Judgment Error | 2.68 |
| Annotation Error | 1.79 |
</details>

Figure 5: Error types distribution for sampled error problems from GPT-4V.

# 4.5 Efforts on Data Leakage Detection

Given the increasing scale of pre-training corpora, it is crucial to detect potential benchmark leakage. The opacity of pre-training often makes this task challenging. To this end, we employ a recently proposed instance-level leakage detection metric, N-gram Prediction Accuracy $[54]$ . This metric uniformly samples several starting points for each instance, predicts the next n-gram for each starting point, and checks whether all predicted n-grams are correct, indicating that the model has potentially encountered this instance. We apply this metric to all available base or text-only chat models of the evaluated models. As shown in Figure 6, it is surprising yet reasonable that some base models or text-only chat models behind these evaluated models have potentially encountered a few benchmark instances, although the number is negligible compared to the complete benchmark. For instance, the base model of Qwen1.5-32B-Chat has potentially encountered 43 benchmark instances. Furthermore, this raises a natural question: can the model correctly answer these instances? Interestingly, the corresponding text-only chat models and multimodal chat models can correctly answer even fewer of these instances. These results demonstrate that our benchmark has minimal leakage $^{10}$ and is sufficiently challenging, as the models cannot correctly answer most of the leaked instances. See Appendix E for more results and analysis.

![](images/db159582f5cfbefbfc3ff201f8f591d6e44e4675d36ac15e1ffe8f43f21b8161.jpg)

<details>
<summary>bar_stacked</summary>

| Category | # Leakage | # Correct in Text-only Chat Model | # Correct in MM Chat Model |
| :--- | :--- | :--- | :--- |
| InternLM2-20B | 19 | 0 | 3 |
| InternLM2-30B Chat | 19 | 0 | 3 |
| Yi-34B | 12 | 0 | 3 |
| Yi-34B Chat | 2 | 0 | 0 |
| Nous-Hermes-2-Yi-34B | 0 | 0 | 0 |
| Qwen-7B | 11 | 0 | 1 |
| Qwen1.5-32B | 43 | 14 | 0 |
| Qwen1.5-32B Chat | 30 | 8 | 0 |
| GPT-4o | 0 | 0 | 0 |
</details>

Figure 6: Detected number of leaked samples and the number of correct responses by corresponding text-only and multimodal chat models on these samples.

# 5 Conclusion

In this work, we introduce OlympicArena, a comprehensive benchmark for evaluating the cognitive reasoning abilities of LMMs and LLMs on Olympic-level problems. Through our detailed experiments, we find that even the most powerful model at present, GPT-4o, does not perform well in applying cognitive reasoning abilities to solve complex problems. We hope that our OlympicArena benchmark serves as a valuable stepping stone for future advancements in AI for science and engineering.

# Acknowledgements

We sincerely appreciate all the laboratory members for their contributions in data annotation, project discussions, and providing valuable suggestions. Additionally, we extend our gratitude to Teacher Xiaoxia Yu from Hefei No. 168 Middle School for providing us with extensive information on various subjects. We also thank everyone who helps annotate the data for our benchmark dataset.

# References

[1] Gpt-4v(ision) system card. 2023. URL https://api.semanticscholar.org/CorpusID:263218031.   
[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[3] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 2024.   
[4] Daman Arora, Himanshu Gaurav Singh, et al. Have llms advanced enough? a challenging problem solving benchmark for large language models. arXiv preprint arXiv:2305.15074, 2023.   
[5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
[6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.   
[7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023.   
[8] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.   
[9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.   
[10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.   
[11] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=EHg5GDnyq1.   
[12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.

[13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
[15] Edward A Feigenbaum, Julian Feldman, et al. Computers and thought. New York McGraw-Hill, 1963.   
[16] Ulrich Furbach, Steffen Hölldobler, Marco Ragni, Claudia Schon, and Frieder Stolzenburg. Cognitive reasoning: A personal view. KI-Künstliche Intelligenz, 33:209–217, 2019.   
[17] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.   
[18] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.   
[19] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.   
[20] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.   
[21] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In International Conference on Machine Learning, pp. 4411–4421. PMLR, 2020.   
[22] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36, 2024.   
[23] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pp. 4171–4186, 2019.   
[24] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.   
[25] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
[26] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.   
[27] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multimodal code large language models with visually rich programming problems. arXiv preprint arXiv:2404.09486, 2024.   
[28] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

[29] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023.   
[30] Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, et al. Fimo: A challenge formal dataset for automated theorem proving. arXiv preprint arXiv:2309.04295, 2023.   
[31] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024.   
[32] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786, 2021.   
[33] Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, and Bingzhe Wu. Fairness-guided few-shot prompting for large language models. Advances in Neural Information Processing Systems, 36, 2024.   
[34] Meredith Ringel Morris, Jascha Sohl-dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Levels of agi: Operationalizing progress on the path to agi. arXiv preprint arXiv:2311.02462, 2023.   
[35] OpenAI. Introducing superalignment. OpenAI Blog, 2023. URL https://openai.com/superalignment.   
[36] OpenAI. Hello gpt-4o. OpenAI Blog, 2024. URL https://openai.com/index/hello-gpt-4o/.   
[37] Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.   
[38] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.   
[39] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.   
[40] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.   
[41] Quan Shi, Michael Tang, Karthik Narasimhan, and Shunyu Yao. Can language models solve olympiad programming? arXiv preprint arXiv:2404.10952, 2024.   
[42] Shiven Sinha, Ameya Prabhu, Ponnurangam Kumaraguru, Siddharth Bhat, and Matthias Bethge. Wu's method can boost symbolic ai to rival silver medalists and alphageometry to outperform gold medalists at imo geometry. arXiv preprint arXiv:2404.06405, 2024.   
[43] Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. A survey of reasoning with foundation models. arXiv preprint arXiv:2312.11562, 2023.   
[44] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19053–19061, 2024.   
[45] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[46] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.   
[47] Alan M Turing and J Haugeland. Computing machinery and intelligence. The Turing Test: Verbal Behavior as the Hallmark of Intelligence, pp. 29–56, 1950.   
[48] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.   
[49] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 353–355, 2018.   
[50] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.   
[51] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. arXiv preprint arXiv:2312.08935, 2023.   
[52] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023.   
[53] Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. Evaluating mathematical reasoning beyond accuracy. arXiv preprint arXiv:2404.05692, 2024.   
[54] Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824, 2024.   
[55] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.   
[56] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.   
[57] Weizhe Yuan and Pengfei Liu. restructured pre-training. arXiv preprint arXiv:2206.11147, 2022.   
[58] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.   
[59] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.   
[60] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, et al. Cmmu: A chinese massive multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2401.11944, 2024.   
[61] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.   
[62] Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474, 2023.

[63] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.   
[64] Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921, 2023.

# A Detailed Statistics of the Benchmark

# A.1 Distribution of Problems

Our benchmark collects data from various competitions. The detailed list can be found in Table 4. Note that a small portion of the problems are sampled from other related benchmarks which are marked in the table. The subfields covered by each competition subject are shown in Table 5. Additionally, the distribution information of our benchmark across different languages and modalities is presented in Table 6.

# A.2 Answer Types

Through extensive observation of a large number of problems and a thorough examination of multiple previous benchmarks, we have finally distilled 13 comprehensive answer types. These types are designed to cover as many problems as possible. The specific definitions for each answer type are provided in Table 7.

# A.3 Image Types

We categorize and summarize the five most common types of images in our multimodal scientific problems. The definitions of these types can be found in Table 8, and examples are provided in Figure 7. The distribution of different image types in our benchmark is shown in Figure 8

![](images/c75fe75dca6dd7cb1def327cc0c59686723094cf08fc47e6220f88248401f37d.jpg)  
Figure 7: Examples of Image Types

Table 4: List of competitions included in OlympicArena. Competitions marked with \* are partially sourced from OlympiadBench [17], and those marked with † are partially sourced from MMcode [27]. 

<table><tr><td>Competition Name</td><td>Abbreviation</td><td>Subject</td><td># Problems</td></tr><tr><td>UK Senior Kangaroo</td><td>UKMT_SK</td><td>Math</td><td>20</td></tr><tr><td>Math Majors of America Tournament for High Schools</td><td>MMATHS</td><td>Math</td><td>47</td></tr><tr><td>Math Kangaroo</td><td>MK</td><td>Math</td><td>35</td></tr><tr><td>Euclid Mathematics Contest</td><td>EMC</td><td>Math</td><td>215</td></tr><tr><td>Canadian Open Mathematics Challenge</td><td>COMC</td><td>Math</td><td>26</td></tr><tr><td>Johns Hopkins Mathematics Tournament</td><td>JHMT</td><td>Math</td><td>100</td></tr><tr><td>Berkeley Math Tournament</td><td>BMT</td><td>Math</td><td>93</td></tr><tr><td>Stanford Mathematics Tournament</td><td>SMT</td><td>Math</td><td>473</td></tr><tr><td>Chinese High School Mathematics League (Pre Round)</td><td>ZH_Math_PRE</td><td>Math</td><td>546</td></tr><tr><td>Chinese High School Mathematics League (1st&amp;2nd Round)</td><td>ZH_Math_12</td><td>Math</td><td>279</td></tr><tr><td>Duke University Math Meet</td><td>DMM</td><td>Math</td><td>107</td></tr><tr><td>The Princeton University Mathematics Competition</td><td>PUMaC</td><td>Math</td><td>296</td></tr><tr><td>Harvard-MIT Mathematics Tournament</td><td>HMMT</td><td>Math</td><td>392</td></tr><tr><td>William Lowell Putnam Mathematics Competition</td><td>Putnam</td><td>Math</td><td>136</td></tr><tr><td>International Mathematical Olympiad*</td><td>IMO</td><td>Math</td><td>79</td></tr><tr><td>Romanian Master of Mathematics*</td><td>RMM</td><td>Math</td><td>8</td></tr><tr><td>American Regions Mathematics League*</td><td>ARML</td><td>Math</td><td>374</td></tr><tr><td>Euclid Mathematics Competition*</td><td>EMC</td><td>Math</td><td>215</td></tr><tr><td>European Girls’ Mathematical Olympiad*</td><td>EGMO</td><td>Math</td><td>7</td></tr><tr><td>F=MA</td><td>FMA</td><td>Physics</td><td>122</td></tr><tr><td>Intermediate Physics Challenge (Y11)</td><td>BPhO_IPC</td><td>Physics</td><td>50</td></tr><tr><td>Senior Physics Challenge</td><td>BPhO_SPC</td><td>Physics</td><td>38</td></tr><tr><td>Australian Science Olympaids Physics</td><td>ASOP</td><td>Physics</td><td>48</td></tr><tr><td>European Physics Olympiad</td><td>EPhO</td><td>Physics</td><td>15</td></tr><tr><td>Nordic-Baltic Physics Olympiad</td><td>NBPhO</td><td>Physics</td><td>102</td></tr><tr><td>World Physics Olympics</td><td>WoPhO</td><td>Physics</td><td>38</td></tr><tr><td>Asian Physics Olympiad</td><td>APhO</td><td>Physics</td><td>126</td></tr><tr><td>International Physics Olympiad</td><td>IPhO</td><td>Physics</td><td>307</td></tr><tr><td>Canadian Association of Physicists</td><td>CAP</td><td>Physics</td><td>100</td></tr><tr><td>Physics Bowl</td><td>PB</td><td>Physics</td><td>100</td></tr><tr><td>USA Physics Olympiad</td><td>USAPhO</td><td>Physics</td><td>188</td></tr><tr><td>Chinese Physics Olympiad</td><td>CPhO</td><td>Physics</td><td>462</td></tr><tr><td>Physics Challenge (Y13)</td><td>PCY13</td><td>Physics</td><td>44</td></tr><tr><td>Chinese High School Biology Challenge</td><td>GAOKAO_Bio</td><td>Biology</td><td>652</td></tr><tr><td>International Biology Olympiad</td><td>IBO</td><td>Biology</td><td>300</td></tr><tr><td>The USA Biology Olympiad</td><td>USABO</td><td>Biology</td><td>96</td></tr><tr><td>Indian Biology Olympiad</td><td>INBO</td><td>Biology</td><td>86</td></tr><tr><td>Australian Science Olympiad Biology</td><td>ASOB</td><td>Biology</td><td>119</td></tr><tr><td>British Biology Olympiad</td><td>BBO</td><td>Biology</td><td>82</td></tr><tr><td>New Zealand Biology Olympiad</td><td>NZIBO</td><td>Biology</td><td>223</td></tr><tr><td>Chem 13 News</td><td>Chem13News</td><td>Chemistry</td><td>56</td></tr><tr><td>Avogadro</td><td>Avogadro</td><td>Chemistry</td><td>55</td></tr><tr><td>U.S. National Chemistry Olympiad (local)</td><td>USNCO (local)</td><td>Chemistry</td><td>54</td></tr><tr><td>U.S. National Chemistry Olympiad</td><td>USNCO</td><td>Chemistry</td><td>98</td></tr><tr><td>Chinese High School Chemistry Challenge</td><td>GAOKAO_Chem</td><td>Chemistry</td><td>568</td></tr><tr><td>Canadian Chemistry Olympic</td><td>CCO</td><td>Chemistry</td><td>100</td></tr><tr><td>Australian Science Olympiad Chemistry</td><td>ASOC</td><td>Chemistry</td><td>91</td></tr><tr><td>Cambridge Chemistry Challenge</td><td>C3H6</td><td>Chemistry</td><td>61</td></tr><tr><td>UK Chemistry Olympiad</td><td>UKChO</td><td>Chemistry</td><td>100</td></tr><tr><td>International Chemistry Olympiad</td><td>IChO</td><td>Chemistry</td><td>402</td></tr><tr><td>Chinese High School Geography Challenge</td><td>GAOKAO_Geo</td><td>Geography</td><td>862</td></tr><tr><td>US Earth Science Organization</td><td>USESO</td><td>Geography</td><td>301</td></tr><tr><td>Australian Science Olympiad Earth Science</td><td>ASOE</td><td>Geography</td><td>100</td></tr><tr><td>The International Geography Olympiad</td><td>IGeO</td><td>Geography</td><td>327</td></tr><tr><td>Chinese High School Astronomy Challenge</td><td>GAOKAO_Astro</td><td>Astronomy</td><td>740</td></tr><tr><td>The International Astronomy and Astrophysics Competition</td><td>IAAC</td><td>Astronomy</td><td>50</td></tr><tr><td>USA Astronomy and Astrophysics Organization</td><td>USAAAO</td><td>Astronomy</td><td>100</td></tr><tr><td>British Astronomy and Astrophysics Olympiad Challenge</td><td>BAAO_challenge</td><td>Astronomy</td><td>148</td></tr><tr><td>British Astronomy and Astrophysics Olympiad-round2</td><td>BAAO</td><td>Astronomy</td><td>185</td></tr><tr><td>USA Computing Olympiad</td><td>USACO</td><td>CS</td><td>48</td></tr><tr><td>Atcoder</td><td>Atcoder</td><td>CS</td><td>48</td></tr><tr><td>Codeforces†</td><td>CF</td><td>CS</td><td>138</td></tr></table>

Table 5: Subfields of each subject included in OlympicArena. 

<table><tr><td>Subject</td><td>Subfields</td></tr><tr><td>Math</td><td>Algebra, Geometry, Number Theory, Combinatorics</td></tr><tr><td>Physics</td><td>Mechanics, Electricity and Magnetism, Waves and Optics, Thermodynamics, Modern Physics, Fluid Mechanics</td></tr><tr><td>Chemistry</td><td>General Chemistry, Organic Chemistry, Inorganic Chemistry, Analytical Chemistry, Physical Chemistry, Environmental Chemistry</td></tr><tr><td>Biology</td><td>Cell biology, Plant Anatomy and Physiology, Animal Anatomy and Physiology, Ethology, Genetics and Evolution, Ecology , Biosystematics</td></tr><tr><td>Geography</td><td>Physical Geography, Human Geography, Regional Geography, Environmental Geography, Geospatial Techniques</td></tr><tr><td>Astronomy</td><td>Fundamentals of Astronomy, Stellar Astronomy, Galactic and Extragalactic Astronomy, Astrophysics</td></tr><tr><td>CS</td><td>Data Structures, Algorithm</td></tr></table>

Table 6: Statistics of OlympicArena benchmark across different disciplines and modalities. 

<table><tr><td></td><td>Mathematics</td><td>Physics</td><td>Chemistry</td><td>Biology</td><td>Geography</td><td>Astronomy</td><td>CS</td></tr><tr><td>EN &amp; text</td><td>2215</td><td>632</td><td>782</td><td>352</td><td>211</td><td>219</td><td>90</td></tr><tr><td>EN &amp; multi-modal</td><td>193</td><td>646</td><td>235</td><td>554</td><td>517</td><td>264</td><td>144</td></tr><tr><td>ZH &amp; text</td><td>780</td><td>164</td><td>124</td><td>312</td><td>58</td><td>264</td><td>0</td></tr><tr><td>ZH &amp; multi-modal</td><td>45</td><td>298</td><td>444</td><td>340</td><td>804</td><td>476</td><td>0</td></tr><tr><td>Total EN</td><td>2408</td><td>1278</td><td>1017</td><td>906</td><td>728</td><td>483</td><td>234</td></tr><tr><td>Total ZH</td><td>825</td><td>462</td><td>568</td><td>652</td><td>862</td><td>740</td><td>0</td></tr><tr><td>Total text</td><td>2995</td><td>796</td><td>906</td><td>664</td><td>269</td><td>483</td><td>90</td></tr><tr><td>Total multi-modal</td><td>238</td><td>944</td><td>679</td><td>894</td><td>1321</td><td>740</td><td>144</td></tr><tr><td>Grand Total</td><td>3233</td><td>1740</td><td>1585</td><td>1558</td><td>1590</td><td>1223</td><td>234</td></tr></table>

![](images/f69f7fd47f253a2404849fd5365977504a8abb186d39b456c128bdcc4a7b2161.jpg)

<details>
<summary>pie</summary>

| Image Types | Percentage (%) | Count |
|---|---|---|
| Geometric and Mathematical Diagrams | 39.16 | 2965 |
| Statistical and Data Representation | 9.19 | 696 |
| Natural and Environmental Images | 17.12 | 1296 |
| Scientific and Technical Diagrams | 31.36 | 2374 |
| Abstract and Conceptual Visuals | 3.17 | 240 |
</details>

Figure 8: Distribution of Image Types

Table 7: Answer Types and Definitions 

<table><tr><td>Answer Type</td><td>Definition</td></tr><tr><td>Single Choice (SC)</td><td>Problems with only one correct option (e.g., one out of four, one out of five, etc.).</td></tr><tr><td>Multiple-choice (MC)</td><td>Problems with multiple correct options (e.g., two out of four, two out of five, two out of six, etc.).</td></tr><tr><td>True/False (TF)</td><td>Problems where the answer is either True or False.</td></tr><tr><td>Numerical Value (NV)</td><td>Problems where the answer is a numerical value, including special values like  $\pi$ ,  $e$ ,  $\sqrt{7}$ ,  $\log_2 9$ , etc., represented in LaTeX.</td></tr><tr><td>Set (SET)</td><td>Problems where the answer is a set, such as {1, 2, 3}.</td></tr><tr><td>Interval (IN)</td><td>Problems where the answer is a range of values, represented as an interval in LaTeX.</td></tr><tr><td>Expression (EX)</td><td>Problems requiring an expression containing variables, represented in LaTeX.</td></tr><tr><td>Equation (EQ)</td><td>Problems requiring an equation containing variables, represented in LaTeX.</td></tr><tr><td>Tuple (TUP)</td><td>Problems requiring a tuple, usually representing a pair of numbers, such as (x, y).</td></tr><tr><td>Multi-part Value (MPV)</td><td>Problems requiring multiple quantities to be determined within a single sub-problem, such as solving both velocity and time in a physics problem.</td></tr><tr><td>Multiple Answers (MA)</td><td>Problems with multiple solutions for a single sub-problem, such as a math fill-in-the-blank problem with answers 1 or -2.</td></tr><tr><td>Code Generation (CODE)</td><td>Problems where the answer is a piece of code, requiring the generation of functional code snippets or complete programs to solve the given task.</td></tr><tr><td>Others (OT)</td><td>Problems that do not fit into the above categories, such as writing chemical equations or explaining reasons, which require human expert evaluation.</td></tr></table>

Table 8: Definitions and examples of five image types in our multi-modal scientific problems. 

<table><tr><td>Image Type</td><td>Definition</td></tr><tr><td>Geometric and Mathematical Diagrams</td><td>Includes diagrams representing mathematical concepts, such as 2D and 3D shapes, mathematical notations, function plots.</td></tr><tr><td>Statistical and Data Representation</td><td>Visualizations for statistical or data information, including multivariate plots, tables, charts (histograms, bar charts, line plots), and infographics.</td></tr><tr><td>Natural and Environmental Images</td><td>Images of natural scenes or phenomena, including environmental studies visualizations, geological and geographical maps, and satellite images.</td></tr><tr><td>Scientific and Technical Diagrams</td><td>Diagrams used in science, such as cell structures and genetic diagrams in Biology, molecular structures and reaction pathways in Chemistry, force diagrams, circuit diagrams, and astrophysical maps in Physics and Astronomy.</td></tr><tr><td>Abstract and Conceptual Visuals</td><td>Visuals explaining theories and concepts, including flowcharts, algorithms, logic models, and symbolic diagrams.</td></tr></table>

# B Data Annotation

# B.1 Problem Extraction and Annotation

We develop a simple and practical annotation interface using Streamlit $^{11}$ (as shown in Figure 9). Approximately 30 university students are employed to use this interface for annotation. We provide each annotator with a wage higher than the local average hourly rate. The specific fields annotated are shown in Figure 10. We use image URLs to represent pictures, which allows for efficient storage and easy access without embedding large image files directly in the dataset. Each annotated problem is ultimately stored as a JSON file, facilitating subsequent processing. It is worth mentioning that we embed several rule-based checks and filtering mechanisms in the annotation interface to minimize noise from the annotations. When the following situations arise, we promptly identify and correct the annotations:

1) When the answer type is Numerical Value, and the annotated answer contains a variable.   
2) When the answer type is not Numerical Value, but the annotated answer can be parsed as a numerical value.   
3) When the answer type is Expression, and the annotated answer contains an equals sign.   
4) When the answer type is Equation, and the annotated answer does not contain an equals sign.   
5) When the annotated answer contains images that should not be present.   
6) When the annotated answer contains units (since units are a separate field according to Figure 10, we compile a list of common units and manually check and correct answers when suspected units are detected).   
7) When the annotated image links cannot be previewed properly.

Additionally, we implement a multi-step validation process after the initial annotation is completed. First, we conduct a preliminary check using predefined rules to identify any error data, which is then corrected. Following this, a secondary review is performed by different annotators to further check and correct any errors in the annotations. This cross-checking mechanism helps ensure the accuracy and consistency of the annotations.

![](images/2aeae4e2185fe51749d996438c70adf1230dd1b2dd3dd3eef44ac4cb634ca3c1.jpg)

<details>
<summary>text_image</summary>

HMMT February 2020
February 15, 2020
Algebra and Number Theory
1. Let P(x) = x^a + z^b - r^c = 200 in a polynomial with roots x, t. What is P(1)?
2. Find the square part of positive integers (x, y) with x = 1 for which
3. Let n = 256. Find the square and number x > x^4 such that
4. Let log101(k)
5. Let log101(k)
6. Let log101(k)
7. Let log101(k)
8. Let log101(k)
9. Let log101(k)
10. Let log101(k)
11. Let log101(k)
12. Let log101(k)
13. Let log101(k)
14. Let log101(k)
15. Let log101(k)
16. Let log101(k)
17. Let log101(k)
18. Let log101(k)
19. Let log101(k)
20. Let log101(k)
21. Let log101(k)
22. Let log101(k)
23. Let log101(k)
24. Let log101(k)
25. Let log101(k)
26. Let log101(k)
27. Let log101(k)
28. Let log101(k)
29. Let log101(k)
30. Let log101(k)
31. Let log101(k)
32. Let log101(k)
33. Let log101(k)
34. Let log101(k)
35. Let log101(k)
36. Let log101(k)
37. Let log101(k)
38. Let log101(k)
39. Let log101(k)
40. Let log101(k)
41. Let log101(k)
42. Let log101(k)
43. Let log101(k)
44. Let log101(k)
45. Let log101(k)
46. Let log101(k)
47. Let log101(k)
48. Let log101(k)
49. Let log101(k)
50. Let log101(k)
51. Let log101(k)
52. Let log101(k)
53. Let log101(k)
54. Let log101(k)
55. Let log101(k)
56. Let log101(k)
57. Let log101(k)
58. Let log101(k)
59. Let log101(k)
60. Let log101(k)
61. Let log101(k)
62. Let log101(k)
63. Let log101(k)
64. Let log101(k)
65. Let log101(k)
66. Let log101(k)
67. Let log101(k)
68. Let log101(k)
69. Let log101(k)
70. Let log101(k)
71. Let log101(k)
72. Let log101(k)
73. Let log101(k)
74. Let log101(k)
75. Let log101(k)
76. Let log101(k)
77. Let log101(k)
78. Let log101(k)
79. Let log101(k)
80. Let log101(k)
81. Let log101(k)
82. Let log101(k)
83. Let log101(k)
84. Let log101(k)
85. Let log101(k)
86. Let log101(k)
87. Let log101(k)
88. Let log101(k)
89. Let log101(k)
90. Let log101(k)
91. Let log101(k)
92. Let log101(k)
93. Let log101(k)
94. Let log101(k)
95. Let log101(k)
96. Let log101(k)
97. Let log101(k)
98. Let log101(k)
99. Let log101(k)
$HMMT February 2020 <br> February 2020, 2020
## Algebra and Number Theory
Let $P(x)-x^a-3[x^a-2]-x^2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2]-x^2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2}-x^2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2}-x^2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2}-x^-2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2}-x^-2] x-2X-2X-2 be a polynomial with roots $r, s, t$. What is $P(2)$.?
$P(x)-x^a-3[x^a-2}-x^-2]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^3[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x^a-3]^4[x\log_*(k)^*]
$P(x)-x^a-3[x^a-2}-x^-2] x-2X-2X-2 = 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x
$P(x)-x^a-3[x^a-2}-x^-2] x - 2X - 2X - 2 = 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x
$P(x)-x^a-3[x^a-2}-x^-2] x - 2X - 2X - 2= 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x - 2x
$P(x)-x^a-3[x^a-2}-x^-2] x - 2X - 2X - 2= 2x - 2x = 2x - 2x = 2x - 2x = 2x - 2x = 2x - 2x = 2x - 2x = 2x
$P(x)-x^a-3[x^a-2}-x^-2] x - 2X - 2X - 2= 2x - 2x = 2x = 2x = 2x = 2x = 2x = 2x = 2x = 2x
$P(x)-x^a-3[x^a-2}-x^-4] x - 4X - (the set of all possible values of $P(x)) = (the set of all possible values of x) with coefficients in P(X) (the set of all possible values of x) with coefficients in P(X) (the set of all possible values of x) with coefficients in P(X) (the set of all possible values of x). For example, the number of points of P(X) for each integer of degree (e.g., (n) or (n+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+ e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e), (n+e+e),<nl>
$P(x)-x^a-3[x^a-3] x - k = k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k + k +
$P(x)-p(x) = p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^*p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* p(x)^* P(X)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)( N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)(N)( N)(N)(N)(N)(N)(N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N)( N}( N)\text{if } \text{and} \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if} \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{if } \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of} \text{otherwise}. \text{of}
</details>

Figure 9: Annotation Page

# B.2 Annotation for Difficulty Levels

The definitions of three levels of difficulty are as follows:

```json
{
    "answer_type": "SC",
    "context": null,
    "problem": "The numbers from 1 to 9 are to be distributed to the nine squares in the diagram according to the following rules: There is to be one number in each square. The sum of three adjacent numbers is always a multiple of 3. The numbers 7 and 9 are already written in. How many ways are there to insert the remaining numbers?\n\n[figure1]",
    "options": {
    "A": "9",
    "B": "12",
    "C": "15",
    "D": "18",
    "E": "24"
    },
    "answer": "E",
    "unit": null,
    "answer_sequence": null,
    "type_sequence": null,
    "solution": null,
    "figure_urls": {
    "1": {
    "url": "https://i.postimg.cc/mgXQHsKB/image.png",
    "type": "1",
    "caption": "The image displays a simple, black and white representation of a bar graph. The graph consists of a single horizontal bar that is divided into two sections. The left section of the bar is labeled with the number \7\ and the right section is labeled with the number \9\". The numbers are placed inside the respective sections of the bar, indicating their values. The bar graph does not contain any additional text or elements. The style of the image is minimalistic, with a clear focus on the numerical data represented by the bar."
    }
    },
    "subject": "Math",
    "competition": "MK",
    "file_name": "2023_Student",
    "language": "EN",
    "modality": "multi-modal"
} 
```  
Figure 10: Example of a json-formatted representation of an annotated problem.

1) Knowledge Recall: This involves the direct recall of factual information and well-defined procedures. It examines the memory of simple knowledge points, i.e., whether certain information is known.   
2) Concept Application: This category covers the very basic use of simple concepts to solve easy problems or perform straightforward calculations. It involves applying known information to situations without any complex or multi-step reasoning. The focus is on straightforward application rather than reasoning.   
3) Cognitive Reasoning: This involves the use of logical reasoning or visual reasoning to solve problems. It includes problems that require clear thinking and problem-solving techniques. It focuses on the ability to reason and analyze to understand and address the issues.

The prompt we use for categorizing each problem is shown in Figure 11

# B.3 Cognitive Reasoning Abilities Annotation

We provide detailed definitions for each of these cognitive reasoning abilities.

The logical reasoning abilities:

1) Deductive Reasoning involves starting with a general principle or hypothesis and logically deriving specific conclusions. This process ensures that the conclusion necessarily follows from the premises.   
2) Inductive Reasoning involves making broad generalizations from specific observations. This type of reasoning infers general principles from specific instances, enhancing our confidence in the generality of certain phenomena.   
3) Abductive Reasoning starts with incomplete observations and seeks the most likely explanation. It is used to form hypotheses that best explain the available data.   
4) Analogical Reasoning involves using knowledge from one situation to solve problems in a similar situation by drawing parallels.

Problem description:

{problem}

Answer:

{answer}

Solution:

{solution}\*

# Classification Categories:

1. Knowledge Recall: Direct recall of factual information and well-defined procedures. This category examines the memory of simple knowledge points, i.e., whether certain information is known.   
2. Concept Application: Very basic use of simple concepts to solve easy problems or do straightforward calculations. This involves applying known information to situations without any complex or multi-step reasoning. The focus is on straightforward application rather than reasoning.   
3. Cognitive Reasoning: Use of logical reasoning or visual reasoning to solve problems. This category includes problems that require clear thinking and problem-solving techniques. It focuses on the ability to reason and analyze to understand and address the issues.

Instructions for Classification: Please classify the above problem by selecting the most appropriate category that best represents the type of thinking and approach required to address the problem. Consider the complexity, the need for creativity, and the depth of knowledge required. You should conclude your response with "So, the problem can be categorized as ANSWER.", where ANSWER should be one of the indexes in 1, 2, 3.

Figure 11: The prompt template used for annotating the difficulty level of problems. The "solution" part marked with \* is optional.

5) Cause-and-Effect Reasoning identifies the reasons behind occurrences and their consequences. This reasoning establishes causal relationships between events.   
6) Critical Thinking involves objectively analyzing and evaluating information to form a reasoned judgment. It encompasses questioning assumptions and considering alternative explanations.   
7) Decompositional Reasoning breaks down complex problems or information into smaller, more manageable parts for detailed analysis.   
8) Quantitative Reasoning involves using mathematical skills to handle quantities and numerical concepts, essential for interpreting data and performing calculations.

The visual reasoning abilities:

1) Pattern Recognition is the ability to identify and understand repeating forms, structures, or recurring themes, especially when presented visually. This skill is critical in subjects like Chemistry for recognizing molecular structures, Biology for identifying cellular components, and Geography for interpreting topographic maps.   
2) Spatial Reasoning is the ability to understand objects in both two and three-dimensional terms and draw conclusions about them with limited information. This skill is often applied in subjects like Math.

Two-Dimensional Examples: Plane geometry, segments, lengths.

Three-Dimensional Examples: Solid geometry, spatial visualization

3) Diagrammatic Reasoning represents the capability to solve problems expressed in diagrammatic form, understanding the logical connections between shapes, symbols, and texts.

Examples: Reading various forms of charts and graphs, obtaining and analyzing statistical information from diagrams.

4) Symbol Interpretation is the ability to decode and understand abstract and symbolic visual information. Examples: Understanding abstract diagrams, interpreting symbols, including representations of data structures such as graphs and linked lists   
5) Comparative Visualization represents comparing and contrasting visual elements to discern differences or similarities, often required in problem-solving to determine the relationship between variable components.

The prompt we use for annotating different logical reasoning abilities and visual reasoning abilities are shown separately in Figure 12 and Figure 13.

Problem description:

{problem}

Answer:

{answer}

Solution:

{solution}\*

You need to identify and select the specific types of logical reasoning abilities required to solve the question from the list provided below.

# Logical Reasoning Abilities:

1. Deductive Reasoning: Deductive reasoning involves starting with a general principle or hypothesis and logically deriving specific conclusions. This process ensures that the conclusion necessarily follows from the premises.   
2. Inductive Reasoning: Inductive reasoning involves making broad generalizations from specific observations. This type of reasoning infers general principles from specific instances, enhancing our confidence in the generality of certain phenomena.   
3. Abductive Reasoning: Abductive reasoning starts with incomplete observations and seeks the most likely explanation. It is used to form hypotheses that best explain the available data.   
4. Analogical Reasoning: Analogical reasoning involves using knowledge from one situation to solve problems in a similar situation by drawing parallels.   
5. Cause-and-Effect Reasoning: Cause-and-effect reasoning identifies the reasons behind occurrences and their consequences. This reasoning establishes causal relationships between events.   
6. Critical Thinking: Critical thinking involves objectively analyzing and evaluating information to form a reasoned judgment. It encompasses questioning assumptions and considering alternative explanations.   
7. Decompositional Reasoning: Decompositional reasoning breaks down complex problems or information into smaller, more manageable parts for detailed analysis.   
8. Quantitative Reasoning: Quantitative reasoning involves using mathematical skills to handle quantities and numerical concepts, essential for interpreting data and performing calculations.

Analyze the question, its answer and explanation (if provided) to determine which of the above reasoning abilities are necessary. Conclude by clearly stating which reasoning abilities are involved in solving the question using "So, the involved reasoning abilities are ABILITIES", where "ABILITIES" represents the numbers corresponding to the list above, separated by commas if multiple abilities are relevant.

Figure 12: The prompt template used for annotating different logical reasoning abilities of problems. The "solution" part marked with \* is optional.

Problem description:

{problem}

Answer:

{answer}

Solution:

{solution}\*

You need to identify and select the specific types of visual reasoning abilities required to solve the question from the list provided below.

# Visual Reasoning Abilities:

1. Pattern Recognition: The ability to identify and understand repeating forms, structures, or recurring themes, especially when presented visually. This skill is critical in subjects like Chemistry for recognizing molecular structures, Biology for identifying cellular components, and Geography for interpreting topographic maps.   
2. Spatial Reasoning: Spatial reasoning is the ability to understand objects in both two and three-dimensional terms and draw conclusions about them with limited information. This skill is often applied in subjects like Math. Two-Dimensional Examples: Plane geometry, segments, lengths Three-Dimensional Examples: Solid geometry, spatial visualization   
3. Diagrammatic Reasoning: The capability to solve problems expressed in diagrammatic form, understanding the logical connections between shapes, symbols, and texts. Examples: Reading various forms of charts and graphs, obtaining and analyzing statistical information from diagrams   
4. Symbol Interpretation: The ability to decode and understand abstract and symbolic visual information. Examples: Understanding abstract diagrams, interpreting symbols, including representations of data structures such as graphs and linked lists   
5. Comparative Visualization: Comparing and contrasting visual elements to discern differences or similarities, often required in problem-solving to determine the relationship between variable components.

Analyze the question, its answer, and any explanation provided to determine which of the above reasoning abilities are necessary. Conclude by clearly stating which reasoning abilities are involved in solving the question using "So, the involved reasoning abilities are ABILITIES", where "ABILITIES" represents the numbers corresponding to the list above, separated by commas if multiple abilities are relevant.

Figure 13: The prompt template used for annotating different visual reasoning abilities of problems which have multi-modal inputs. The "solution" part marked with \* is optional.

# C Experiment Details

# C.1 Prompt for Image Caption

The prompt we use for captioning each image in the benchmark for LMMs is shown in Figure 14.

# C.2 Models

In our experiments, we evaluate a range of both open-source and proprietary LMMs and LLMs. For LMMs, we select the newly released GPT-4o [36] and the powerful GPT-4V [1] from OpenAI. Additionally, we include Claude3 Sonnet [3] from Anthropic, and Gemini Pro Vision $^{12}$ [45] from Google, and Qwen-VL-Max [6] from Alibaba. We also evaluate several open-source models, includ-

[Image]

Describe the fine-grained content of the image or figure, including scenes, objects, relationships, and any text present.

Figure 14: The prompt template used for image caption.   
Table 9: LMMs and their corresponding LLMs. 

<table><tr><td>LMM</td><td>LLM</td></tr><tr><td>GPT-4o</td><td>GPT-4o</td></tr><tr><td>GPT-4v</td><td>GPT-4</td></tr><tr><td>Claude3 Sonnet</td><td>Claude3 Sonnet</td></tr><tr><td>Gemini Pro Vision</td><td>Gemini Pro</td></tr><tr><td>LLaVA-NeXT-34B</td><td>Nous-Hermes-2-Yi-34B</td></tr><tr><td>InternVL-Chat-V1.5</td><td>InternLM2-20B-Chat</td></tr><tr><td>Yi-VL-34B</td><td>Yi-34B-Chat</td></tr><tr><td>Qwen-VL-Chat</td><td>Qwen-7B-Chat</td></tr></table>

ing LLaVA-NeXT-34B [31], InternVL-Chat-V1.5 [12], Yi-VL-34B [55], and Qwen-VL-Chat [7]. For LLMs, we primarily select the corresponding text models of the aforementioned LMMs, such as GPT-4 [2]. Additionally, we include open-source models like Qwen-7B-Chat, Qwen1.5-32B-Chat [5], Yi-34B-Chat [55], and InternLM2-Chat-20B [8]. Table 9 shows the relationship between LMMs and their corresponding LLMs. For the proprietary models, we call the APIs, while for the open-source models, we run them on an 8-card A800 cluster.

# C.3 Evaluation Prompts

We meticulously design the prompts used for model input during experiments. These prompts are tailored to different answer types, with specific output formats specified for each type. The detailed prompt templates are shown in Figure 15, and the different instructions for each answer type are provided in Table 10.

You are participating in an international {subject} competition and need to solve the following question.

{answer type description}

Here is some context information for this question, which might assist you in solving it: {context}\*

Problem: {problem}

All mathematical formulas and symbols you output should be represented with LaTeX. You can solve it step by step and please end your response with: {answer format instruction}.

Figure 15: The prompt template used for problem input. The "context" part marked with \* is optional and refers to supplementary information provided during manual annotation when the problem relies on conclusions from previous questions. The {answer type description} and {answer format instruction} are specified in Table 10.

# C.4 Model Hyperparameters

For all models, we set the maximum number of output tokens to 2048 and the temperature to 0.0. When performing code generation (CODE) tasks, the temperature is set to 0.2.

# C.5 Answer-level Evaluation Protocols

Rule-based Evaluation For problems with fixed answers, we extract the final answer enclosed in "\boxed{}" (using prompts to instruct models to conclude their final answers with boxes) and perform rule-based matching according to the answer type.

1) For numerical value (NV) answers, we handle units by explicitly stating them in the prompts provided to the model, if applicable. During evaluation, we assess only the numerical value output by the model, disregarding the unit. In cases where numerical answers are subject to estimation, such as in physics or chemistry problems, we convert both the model's output and the correct answer to scientific notation. If the exponent of 10 is the same for both, we allow a deviation of 0.1 in the coefficient before the exponent, accounting for minor estimation errors in the model's calculations.   
2) For problems where the answer type is an expression (EX) or an equation (EQ), we use the SymPy $^{13}$ library for comparison. This allows us to accurately assess the equivalence of algebraic expressions and equations by symbolic computation.   
3) For problems requiring the solution of multiple quantities (MPV), our evaluation strictly follows the order of output specified in the prompt, ensuring consistency and correctness in the sequence of results.   
4) In the case of problems with multiple answers (MA), we require the model to output all possible answers, adequately considering various scenarios.   
5) For problems where the answer type is an interval (IN), we strictly compare the open and closed intervals as well as the boundary values of the endpoints.   
6) For problems where the answer type is a set (SET), we compare the set output by the model with the standard answer set to ensure they are completely identical. For problems where the answer type is a tuple (TUP), we compare the tuple output by the model with the standard answer tuple to ensure that each corresponding position is exactly equal.   
7) For code generation (CODE) problems, we extract the code output by the model and test it through all provided test cases. Specifically, we use the unbiased pass@k metric,

$$
\text { pass } @ k := \underset {\text { Problems }} {\mathbb {E}} \left[ 1 - \frac {\binom {n - c} {k}}{\binom {n} {k}} \right] \tag {1}
$$

where we set k = 1 and n = 5, and c indicates the number of correct samples that pass all test cases.

Model-based Evaluation To deal with those problems with answer types that cannot be appropriately evaluated using rule-based matching, we employ model-based evaluation. In this approach, we utilize GPT-4V as the evaluator. We design prompts that include the problem, the correct answer, the solution (if provided), and the response from the model being tested (see Figure 16 for details). The evaluator model then judges the correctness of the tested model's response.

To further ensure the reliability of using a model as an evaluator, we uniformly sampled 100 problems across various subjects that involved model evaluation. We have several students with backgrounds in science and engineering independently conduct manual evaluations. It turns out that out of the 100 sampled problems, there is nearly 80% agreement between the human evaluations and the model evaluations. Considering that problems requiring model-based evaluation account for approximately 5% of the total, the error rate can be controlled at around $20\% \times 5\%$ , which is approximately 1%. Therefore, we consider this method to be reliable.

You are an experienced teacher tasked with grading an Olympic-level {subject} exam paper.

The problem's context: {context}\*

Problem: {problem}

The student's answer: {the tested model's response}

The reference answer: {the reference answer}

The reference solution: {the reference solution} \*

# Note:

(1) You can tolerate some markdown formatting issues.   
(2) You need to make judgments based on the provided reference answer and reference solution (if provided).

You can analyze the answer step by step, and then output correct or incorrect at the end to express your final judgment.

Figure 16: The prompt used for model-based evaluation. The "context" and "the reference solution" parts marked with \* are optional.

# C.6 Process-level Evaluation Protocols

To conduct the process-level evaluation, we utilize a method based on GPT-4V. First, we reformat both the gold solution and the model-generated solution for the sampled problems into a neat step-by-step format using GPT-4. Then, we employ a carefully designed prompt(see Figure 17) to guide GPT-4V using the reformatted gold solution to evaluate the correctness of each step in the model's output, assigning a score of 0 for incorrect and 1 for correct steps. The final process-level score for each problem is determined by averaging the scores of all the steps.

# D Fine-grained Results

# D.1 Results across Logical and Visual Reasoning Abilities

Table 11 and Table 12 show the performance of different models across various logical and visual reasoning abilities separately.

# D.2 Results on Multimodal Problems

Table 13 shows the performance of different models on multimodal problems across different subjects.

# D.3 Process-level Evaluation Results

Table 14 shows process-level results of different models across different subjects.

# D.4 Results across Different Languages

Table 15 shows results of different models in different languages.

You are a teacher skilled in evaluating the intermediate steps of a student's solution to a given problem.

You are given two types of step-by-step solutions: one from the reference answer and the other from the student. Your task is to evaluate the correctness of each step in the student's solutions using binary scoring: assign a score of 1 for correct steps and 0 for incorrect steps.

Use the reference solutions to guide your evaluation.

Follow the format:

Step 1: ...

Step 2: ...

Step 3: ...

Please provide the results directly, omitting any introductory or concluding remarks.

\# The given question

{the question}

\# The reference solution

{the reference solution}

\# The student's solution

{the model's solution}

\# Your scores for each step of the student's solutions

Figure 17: The prompt used for process-level evaluation.

# E Data Leakage Detection Details

We combine the questions and detailed solutions (or answers if there are no steps) of the problems, then use the n-gram prediction accuracy metric. Specifically, for each sample, we sample k starting points and predict the next 5-gram each time. To evaluate whether the n-gram prediction is correct, we use exact match and more lenient metrics such as edit distance and ROUGE-L. Here, we consider a prediction correct if either the edit distance or ROUGE-L similarity exceeds 75%, to mitigate some reformatting issues during pre-training. We take the union of instances detected by different metrics to obtain the final set of detected instances.

As shown in Tables 16, 17, and 18, the experimental results reveal that indeed, different models exhibit minor leakage across different subjects. An interesting observation is that some leakages detected by the base model are no longer detectable when using the chat model based on the same base model. We hypothesize that optimization for dialogue capabilities potentially impacts the model's ability and performance on the next token prediction. Another similar observation is that leakages detected by text-only chat models tend to decrease when evaluated on multimodal chat models based on the same chat models. Figure 18 presents a data leakage case from Qwen1.5-32B-Chat.

# An exact match case of Qwen1.5-32B-Chat

Text: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-31} ... 1.6 \times 10^{-19} ... \mathrm{\~J}\]

Prompt: The fractional

Prediction: The fractional quantum Hall effect (F

Prompt: The fractional quantum Hall effect (FQHE) was ... fraction

Prediction: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasip

Prompt: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times

Prediction: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-3}

Prompt: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-31} ... 1.

Prediction: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-31} ... 1.6 \times 1

Prompt: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-31} ... 1.6 \times 10^{-19} ... \mathrm

Prediction: The fractional quantum Hall effect (FQHE) was ... fractionally charged quasiparticles ... \times 10^{-31} ... 1.6 \times 10^{-19} ... \mathrm{\~J}\$\$

Figure 18: A potential data leakage case of Qwen1.5-32B-Chat which is presented with the original problem and solution concatenated, separated by a space.

Table 10: Descriptions of answer types and corresponding format instructions included in the problem input prompts. Specifically, {unit description} indicates: "Remember, your answer should be calculated in the unit of {unit}, but do not include the unit in your final answer." 

<table><tr><td>Answer Type</td><td>Answer Type Description</td><td>Answer Format Instruction</td></tr><tr><td>SC</td><td>This is a multiple choice question (only one correct answer).</td><td>Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER should be one of the options: {the options of the problem}.</td></tr><tr><td>MC</td><td>This is a multiple choice question (more than one correct answer).</td><td>Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER should be two or more of the options: {the options of the problem}.</td></tr><tr><td>TF</td><td>This is a True or False question.</td><td>Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER should be either &quot;True&quot; or &quot;False&quot;.</td></tr><tr><td>NV</td><td>The answer to this question is a numerical value.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is the numerical value without any units.</td></tr><tr><td>SET</td><td>The answer to this question is a set.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is the set of all distinct answers, each expressed as a numerical value without any units, e.g. ANSWER = {3, 4, 5}.</td></tr><tr><td>IN</td><td>The answer to this question is a range interval.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is an interval without any units, e.g. ANSWER = (1, 2]∪[7, +∞).</td></tr><tr><td>EX</td><td>The answer to this question is an expression.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is an expression without any units and equals signs, e.g. ANSWER =  $\frac{1}{2}gt^2$ .</td></tr><tr><td>EQ</td><td>The answer to this question is an equation.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is an equation without any units, e.g. ANSWER =  $\frac{x^2}{4} + \frac{y^2}{2} = 1$ .</td></tr><tr><td>TUP</td><td>The answer to this question is a tuple.</td><td>{unit instruction} Please end your response with: &quot;The final answer is ANSWER&quot;, where ANSWER is a tuple without any units, e.g. ANSWER=(3, 5).</td></tr><tr><td>MPV</td><td>This question involves multiple quantities to be determined.</td><td>Your final quantities should be output in the following order: {the ordered sequence of the name of multiple quantities}. Their units are, in order, {the ordered sequence of the units}, but units shouldn&#x27;t be included in your concluded answer. Their answer types are, in order, {the ordered sequence of answer types}. Please end your response with: &quot;The final answers are ANSWER&quot;, where ANSWER should be the sequence of your final answers, separated by commas, for example: 5, 7, 2.5.</td></tr><tr><td>MA</td><td>This question has more than one correct answer, you need to include them all.</td><td>Their units are, in order, {the ordered sequence of the units}, but units shouldn&#x27;t be included in your concluded answer. Their answer types are, in order, {the ordered sequence of answer types}. Please end your response with: &quot;The final answers are ANSWER&quot;, where ANSWER should be the sequence of your final answers, separated by commas, for example: 5, 7, 2.5.</td></tr><tr><td>CODE</td><td>Write a Python program to solve the given competitive programming problem using standard input and output methods. Pay attention to time and space complexities to ensure efficiency.</td><td>Notes: (1) Your solution must handle standard input and output. Use input() for reading input and print() for output. (2) Be mindful of the problem&#x27;s time and space complexity. The solution should be efficient and designed to handle the upper limits of input sizes within the given constraints. (3) It&#x27;s encouraged to analyze and reason about the problem before coding.You can think step by step, and finally output your final code in the following format:Your Python code here</td></tr><tr><td>OT</td><td>-</td><td>-</td></tr></table>

Table 11: Experimental results across different logical reasoning abilities on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. DED: Deductive Reasoning, IND: Inductive Reasoning, ABD: Abductive Reasoning, ANA: Analogical Reasoning, CAE: Cause-and-Effect Reasoning, CT: Critical Thinking, DEC: Decompositional Reasoning, QUA: Quantitative Reasoning. 

<table><tr><td rowspan="2">Model</td><td>DED</td><td>IND</td><td>ABD</td><td>ANA</td><td>CAE</td><td>CT</td><td>DEC</td><td>QUA</td></tr><tr><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td colspan="9">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>4.85</td><td>4.18</td><td>4.84</td><td>5.29</td><td>5.54</td><td>5.16</td><td>4.09</td><td>4.64</td></tr><tr><td>Yi-34B-Chat</td><td>19.65</td><td>13.84</td><td>26.82</td><td>18.73</td><td>26.51</td><td>25.71</td><td>15.00</td><td>15.55</td></tr><tr><td>Internlm2-20B-Chat</td><td>17.43</td><td>13.12</td><td>24.74</td><td>16.30</td><td>22.81</td><td>22.51</td><td>13.03</td><td>13.42</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>25.94</td><td>21.20</td><td>33.39</td><td>24.87</td><td>32.33</td><td>31.82</td><td>20.19</td><td>22.19</td></tr><tr><td>GPT-3.5</td><td>19.38</td><td>13.19</td><td>26.64</td><td>16.30</td><td>23.32</td><td>24.31</td><td>14.43</td><td>17.35</td></tr><tr><td>Claude3 Sonnet</td><td>25.40</td><td>17.88</td><td>34.78</td><td>23.28</td><td>30.64</td><td>31.15</td><td>18.59</td><td>22.67</td></tr><tr><td>GPT-4</td><td>33.93</td><td>24.80</td><td>40.66</td><td>33.33</td><td>38.48</td><td>39.32</td><td>26.84</td><td>31.72</td></tr><tr><td>GPT-4o</td><td>39.1</td><td>30.14</td><td>43.43</td><td>37.78</td><td>42.89</td><td>44.06</td><td>31.79</td><td>36.56</td></tr><tr><td colspan="9">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>5.66</td><td>4.69</td><td>7.27</td><td>6.88</td><td>6.66</td><td>6.02</td><td>4.60</td><td>4.81</td></tr><tr><td>Yi-34B-Chat</td><td>19.08</td><td>13.34</td><td>29.24</td><td>20.11</td><td>25.53</td><td>24.91</td><td>13.79</td><td>14.64</td></tr><tr><td>Internlm2-20B-Chat</td><td>18.25</td><td>12.69</td><td>28.37</td><td>17.67</td><td>23.84</td><td>23.35</td><td>13.40</td><td>14.52</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>25.50</td><td>20.55</td><td>35.81</td><td>26.35</td><td>31.35</td><td>31.39</td><td>19.55</td><td>21.51</td></tr><tr><td>GPT-3.5</td><td>20.71</td><td>13.91</td><td>29.76</td><td>17.78</td><td>25.72</td><td>26.01</td><td>15.74</td><td>17.73</td></tr><tr><td>Claude3 Sonnet</td><td>25.69</td><td>19.11</td><td>35.12</td><td>24.02</td><td>30.88</td><td>31.55</td><td>18.71</td><td>22.89</td></tr><tr><td>GPT-4</td><td>35.06</td><td>24.44</td><td>41.35</td><td>34.39</td><td>40.17</td><td>40.72</td><td>27.26</td><td>32.47</td></tr><tr><td>GPT-4o</td><td>39.26</td><td>30.93</td><td>45.50</td><td>39.37</td><td>43.17</td><td>44.19</td><td>31.35</td><td>36.56</td></tr><tr><td colspan="9">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>7.87</td><td>6.06</td><td>12.80</td><td>8.68</td><td>9.90</td><td>9.92</td><td>5.29</td><td>6.29</td></tr><tr><td>Yi-VL-34B</td><td>16.30</td><td>10.60</td><td>21.11</td><td>16.40</td><td>21.35</td><td>20.76</td><td>11.89</td><td>13.42</td></tr><tr><td>InternVL-Chat-V1.5</td><td>17.65</td><td>12.55</td><td>30.28</td><td>17.25</td><td>22.06</td><td>22.70</td><td>12.56</td><td>14.37</td></tr><tr><td>LLaVA-NeXT-34B</td><td>19.72</td><td>14.70</td><td>30.62</td><td>19.37</td><td>27.40</td><td>25.39</td><td>13.62</td><td>14.79</td></tr><tr><td>Qwen-VL-Max</td><td>22.97</td><td>16.87</td><td>33.91</td><td>21.38</td><td>29.28</td><td>28.51</td><td>17.26</td><td>18.13</td></tr><tr><td>Gemini Pro Vision</td><td>22.45</td><td>17.16</td><td>35.47</td><td>21.59</td><td>25.67</td><td>27.54</td><td>17.24</td><td>18.91</td></tr><tr><td>Claude3 Sonnet</td><td>25.59</td><td>18.89</td><td>36.51</td><td>23.70</td><td>29.89</td><td>31.71</td><td>18.99</td><td>22.87</td></tr><tr><td>GPT-4V</td><td>34.59</td><td>25.59</td><td>46.54</td><td>33.33</td><td>39.61</td><td>41.15</td><td>26.47</td><td>30.79</td></tr><tr><td>GPT-4o</td><td>41.18</td><td>32.73</td><td>50.35</td><td>40.53</td><td>45.94</td><td>47.12</td><td>33.17</td><td>37.58</td></tr></table>

Table 12: Experimental results across different visual reasoning abilities on Olympic Arena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. PR: Pattern Recognition, SPA: Spatial Reasoning, DIA: Diagrammatic Reasoning, SYB: Symbol Interpretation, COM: Comparative Visualization. 

<table><tr><td rowspan="2">Model</td><td>PR</td><td>SPA</td><td>DIA</td><td>SYB</td><td>COM</td></tr><tr><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td colspan="6">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>4.59</td><td>2.64</td><td>4.26</td><td>4.01</td><td>4.66</td></tr><tr><td>Yi-34B-Chat</td><td>23.70</td><td>13.58</td><td>19.56</td><td>17.61</td><td>22.37</td></tr><tr><td>Internlm2-20B-Chat</td><td>22.89</td><td>13.06</td><td>18.63</td><td>15.73</td><td>21.16</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>28.93</td><td>17.94</td><td>24.67</td><td>22.18</td><td>27.83</td></tr><tr><td>GPT-3.5</td><td>22.33</td><td>13.27</td><td>18.40</td><td>16.05</td><td>21.05</td></tr><tr><td>Claude3 Sonnet</td><td>26.88</td><td>17.60</td><td>22.86</td><td>20.49</td><td>25.98</td></tr><tr><td>GPT-4</td><td>33.65</td><td>23.99</td><td>30.09</td><td>27.94</td><td>32.54</td></tr><tr><td>GPT-4o</td><td>35.96</td><td>28.71</td><td>33.29</td><td>31.54</td><td>35.00</td></tr><tr><td colspan="6">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>5.96</td><td>4.11</td><td>5.21</td><td>5.11</td><td>6.30</td></tr><tr><td>Yi-34B-Chat</td><td>21.69</td><td>21.19</td><td>18.01</td><td>14.92</td><td>20.48</td></tr><tr><td>Internlm2-20B-Chat</td><td>22.97</td><td>12.75</td><td>18.27</td><td>15.49</td><td>21.05</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>28.59</td><td>17.81</td><td>23.90</td><td>20.95</td><td>26.73</td></tr><tr><td>GPT-3.5</td><td>23.96</td><td>15.26</td><td>19.72</td><td>17.34</td><td>22.30</td></tr><tr><td>Claude3 Sonnet</td><td>27.60</td><td>17.03</td><td>22.84</td><td>20.17</td><td>26.28</td></tr><tr><td>GPT-4</td><td>34.29</td><td>26.07</td><td>31.07</td><td>28.61</td><td>33.11</td></tr><tr><td>GPT-4o</td><td>37.08</td><td>29.10</td><td>33.60</td><td>31.22</td><td>35.91</td></tr><tr><td colspan="6">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>9.90</td><td>4.93</td><td>7.46</td><td>6.48</td><td>8.91</td></tr><tr><td>Yi-VL-34B</td><td>16.72</td><td>9.60</td><td>13.78</td><td>12.10</td><td>15.09</td></tr><tr><td>InternVL-Chat-V1.5</td><td>22.85</td><td>12.11</td><td>17.68</td><td>15.11</td><td>21.27</td></tr><tr><td>LLaVA-NeXT-34B</td><td>24.69</td><td>12.75</td><td>19.72</td><td>16.38</td><td>22.90</td></tr><tr><td>Qwen-VL-Max</td><td>27.43</td><td>16.26</td><td>22.35</td><td>19.47</td><td>26.01</td></tr><tr><td>Gemini Pro Vision</td><td>28.98</td><td>14.83</td><td>21.65</td><td>19.79</td><td>26.13</td></tr><tr><td>Claude3 Sonnet</td><td>27.18</td><td>17.55</td><td>22.43</td><td>20.84</td><td>25.56</td></tr><tr><td>GPT-4V</td><td>35.28</td><td>23.91</td><td>30.25</td><td>27.70</td><td>34.40</td></tr><tr><td>GPT-4o</td><td>41.49</td><td>30.65</td><td>36.98</td><td>33.91</td><td>40.58</td></tr></table>

Table 13: Experimental results on multimodal problems on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. 

<table><tr><td rowspan="2">Model</td><td>Math</td><td>Physics</td><td>Chemistry</td><td>Biology</td><td>Geography</td><td>Astronomy</td><td>CS</td><td>Overall</td></tr><tr><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Accuracy</td><td>Pass@1</td><td>Accuracy</td></tr><tr><td colspan="9">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>1.26</td><td>2.54</td><td>6.92</td><td>5.59</td><td>4.16</td><td>2.70</td><td>0</td><td>4.01</td></tr><tr><td>Yi-34B-Chat</td><td>5.04</td><td>6.14</td><td>19.15</td><td>27.40</td><td>33.91</td><td>10.00</td><td>0.28</td><td>19.54</td></tr><tr><td>Internlm2-20B-Chat</td><td>6.30</td><td>6.46</td><td>15.76</td><td>26.96</td><td>31.87</td><td>9.19</td><td>0.97</td><td>18.51</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>7.98</td><td>8.90</td><td>23.86</td><td>32.21</td><td>39.74</td><td>18.65</td><td>0.83</td><td>24.58</td></tr><tr><td>GPT-3.5</td><td>6.30</td><td>7.20</td><td>15.46</td><td>26.85</td><td>30.66</td><td>11.62</td><td>6.25</td><td>18.79</td></tr><tr><td>Claude3 Sonnet</td><td>8.82</td><td>11.76</td><td>19.59</td><td>31.99</td><td>38.00</td><td>15.68</td><td>2.64</td><td>23.79</td></tr><tr><td>GPT-4</td><td>16.81</td><td>18.43</td><td>32.11</td><td>39.71</td><td>41.26</td><td>23.92</td><td>12.50</td><td>31.05</td></tr><tr><td>GPT-4o</td><td>21.85</td><td>21.82</td><td>32.11</td><td>42.17</td><td>44.28</td><td>30.68</td><td>12.78</td><td>34.11</td></tr><tr><td colspan="9">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>3.78</td><td>2.22</td><td>6.19</td><td>6.49</td><td>7.34</td><td>5.00</td><td>0</td><td>5.32</td></tr><tr><td>Yi-34B-Chat</td><td>5.04</td><td>6.57</td><td>14.29</td><td>24.94</td><td>33.61</td><td>8.78</td><td>0.28</td><td>18.25</td></tr><tr><td>Internlm2-20B-Chat</td><td>6.72</td><td>6.89</td><td>16.35</td><td>25.39</td><td>31.26</td><td>10.27</td><td>1.18</td><td>18.41</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>6.72</td><td>8.69</td><td>21.80</td><td>32.10</td><td>39.82</td><td>17.30</td><td>0.97</td><td>23.99</td></tr><tr><td>GPT-3.5</td><td>4.20</td><td>12.39</td><td>18.11</td><td>25.73</td><td>32.32</td><td>9.73</td><td>7.64</td><td>20.04</td></tr><tr><td>Claude3 Sonnet</td><td>5.46</td><td>13.35</td><td>20.47</td><td>32.89</td><td>38.23</td><td>13.38</td><td>3.89</td><td>23.97</td></tr><tr><td>GPT-4</td><td>16.81</td><td>20.44</td><td>29.90</td><td>38.7</td><td>45.12</td><td>26.62</td><td>12.26</td><td>32.36</td></tr><tr><td>GPT-4o</td><td>21.01</td><td>22.14</td><td>31.22</td><td>45.19</td><td>45.19</td><td>30.54</td><td>14.58</td><td>34.86</td></tr><tr><td colspan="9">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>3.36</td><td>2.65</td><td>6.63</td><td>9.84</td><td>13.85</td><td>5.27</td><td>0</td><td>7.82</td></tr><tr><td>Yi-VL-34B</td><td>3.36</td><td>6.46</td><td>9.13</td><td>18.79</td><td>22.03</td><td>7.43</td><td>0</td><td>13.00</td></tr><tr><td>InternVL-Chat-V1.5</td><td>7.56</td><td>6.25</td><td>16.05</td><td>24.94</td><td>32.55</td><td>9.73</td><td>0.62</td><td>18.43</td></tr><tr><td>LLaVA-NeXT-34B</td><td>4.62</td><td>6.46</td><td>14.43</td><td>28.30</td><td>36.11</td><td>10.00</td><td>0.28</td><td>19.66</td></tr><tr><td>Qwen-VL-Max</td><td>6.30</td><td>7.63</td><td>17.82</td><td>28.86</td><td>40.05</td><td>15.14</td><td>1.25</td><td>22.38</td></tr><tr><td>Gemini Pro Vision</td><td>7.56</td><td>9.11</td><td>24.30</td><td>32.55</td><td>35.81</td><td>11.22</td><td>2.36</td><td>22.58</td></tr><tr><td>Claude3 Sonnet</td><td>5.46</td><td>13.45</td><td>19.15</td><td>33.22</td><td>37.02</td><td>17.30</td><td>2.36</td><td>24.05</td></tr><tr><td>GPT-4V</td><td>13.87</td><td>18.22</td><td>29.31</td><td>40.27</td><td>46.86</td><td>22.43</td><td>11.25</td><td>31.81</td></tr><tr><td>GPT-4o</td><td>26.47</td><td>22.14</td><td>33.14</td><td>46.98</td><td>53.75</td><td>31.76</td><td>13.61</td><td>38.17</td></tr></table>

Table 14: Results of the process-level evaluation on our comprehensive OlympicArena benchmark. Each step of every problem is assigned a score of 0 (indicating incorrect) or 1 (indicating correct), with the highest score in each setting underlined and the highest scores across all settings highlighted in bold. The subject of computer science is neglected in this part due to the lack of solutions. 

<table><tr><td rowspan="2">Model</td><td>Math</td><td>Physics</td><td>Chemistry</td><td>Biology</td><td>Geography</td><td>Astronomy</td><td>Overall</td></tr><tr><td>Score</td><td>Score</td><td>Score</td><td>Score</td><td>Score</td><td>Score</td><td>Score</td></tr><tr><td colspan="8">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>18.7</td><td>43.7</td><td>35.1</td><td>18.9</td><td>34.5</td><td>31.5</td><td>30.4</td></tr><tr><td>Yi-34B-Chat</td><td>30.2</td><td>51.0</td><td>54.0</td><td>31.9</td><td>36.5</td><td>40.3</td><td>40.7</td></tr><tr><td>Internlm2-20B-Chat</td><td>21.2</td><td>35.0</td><td>51.2</td><td>22.7</td><td>32.9</td><td>33.3</td><td>32.7</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>32.0</td><td>44.0</td><td>61.1</td><td>32.0</td><td>45.2</td><td>48.6</td><td>43.8</td></tr><tr><td>GPT-3.5</td><td>37.6</td><td>46.9</td><td>32.7</td><td>30.2</td><td>38.7</td><td>26.7</td><td>35.4</td></tr><tr><td>Claude3 Sonnet</td><td>40.8</td><td>42.7</td><td>65.3</td><td>30.8</td><td>52.6</td><td>50.5</td><td>47.1</td></tr><tr><td>GPT-4</td><td>57.0</td><td>53.8</td><td>73.6</td><td>50.0</td><td>50.1</td><td>65.0</td><td>58.2</td></tr><tr><td>GPT-4o</td><td>59.9</td><td>65.9</td><td>67.4</td><td>49.6</td><td>61.4</td><td>69.5</td><td>62.3</td></tr><tr><td colspan="8">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>23.0</td><td>42.6</td><td>34.6</td><td>17.4</td><td>34.4</td><td>32.3</td><td>30.7</td></tr><tr><td>Yi-34B-Chat</td><td>26.3</td><td>45.6</td><td>49.5</td><td>20.0</td><td>45.7</td><td>42.0</td><td>38.2</td></tr><tr><td>Internlm2-20B-Chat</td><td>27.7</td><td>42.6</td><td>46.3</td><td>19.4</td><td>25.5</td><td>43.1</td><td>34.1</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>35.9</td><td>49.7</td><td>56.8</td><td>33.5</td><td>43.6</td><td>51.4</td><td>45.1</td></tr><tr><td>GPT-3.5</td><td>32.1</td><td>46.7</td><td>51.2</td><td>29.1</td><td>38.4</td><td>38.2</td><td>39.3</td></tr><tr><td>Claude3 Sonnet</td><td>50.7</td><td>51.7</td><td>66.1</td><td>33.4</td><td>55.8</td><td>52.2</td><td>51.7</td></tr><tr><td>GPT-4</td><td>61.4</td><td>53.8</td><td>62.7</td><td>51.1</td><td>52.0</td><td>62.2</td><td>57.2</td></tr><tr><td>GPT-4o</td><td>54.3</td><td>63.3</td><td>71.8</td><td>58.6</td><td>56.6</td><td>72.6</td><td>62.9</td></tr><tr><td colspan="8">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>14.3</td><td>41.7</td><td>35.7</td><td>21.0</td><td>31.0</td><td>23.6</td><td>27.9</td></tr><tr><td>Yi-VL-34B</td><td>28.9</td><td>41.0</td><td>44.2</td><td>18.7</td><td>30.2</td><td>40.3</td><td>33.9</td></tr><tr><td>InternVL-Chat-V1.5</td><td>26.6</td><td>40.5</td><td>42.7</td><td>29.4</td><td>43.1</td><td>44.8</td><td>37.8</td></tr><tr><td>LLaVA-NeXT-34B</td><td>30.2</td><td>47.1</td><td>50.1</td><td>19.0</td><td>40.6</td><td>47.1</td><td>39.0</td></tr><tr><td>Qwen-VL-Max</td><td>27.5</td><td>52.4</td><td>65.5</td><td>24.3</td><td>36.0</td><td>48.4</td><td>42.3</td></tr><tr><td>Gemini Pro Vision</td><td>28.5</td><td>46.4</td><td>45.2</td><td>19.9</td><td>33.5</td><td>40.5</td><td>35.7</td></tr><tr><td>Claude3 Sonnet</td><td>47.3</td><td>46.8</td><td>63.2</td><td>24.2</td><td>43.2</td><td>48.1</td><td>45.5</td></tr><tr><td>GPT-4V</td><td>49.9</td><td>54.0</td><td>71.1</td><td>51.4</td><td>56.3</td><td>64.3</td><td>57.8</td></tr><tr><td>GPT-4o</td><td>60.2</td><td>54.8</td><td>72.2</td><td>51.6</td><td>59.6</td><td>74.4</td><td>62.1</td></tr></table>

Table 15: Experimental results across different languages (English and Chinese) on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. 

<table><tr><td rowspan="2">Model</td><td>English</td><td>Chinese</td></tr><tr><td>Accuracy</td><td>Accuracy</td></tr><tr><td colspan="3">LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>4.17</td><td>4.55</td></tr><tr><td>Yi-34B-Chat</td><td>16.37</td><td>18.89</td></tr><tr><td>Internlm2-20B-Chat</td><td>16.56</td><td>16.62</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>22.73</td><td>25.29</td></tr><tr><td>GPT-3.5</td><td>19.83</td><td>15.50</td></tr><tr><td>Claude3 Sonnet</td><td>25.73</td><td>18.20</td></tr><tr><td>GPT-4</td><td>35.13</td><td>27.31</td></tr><tr><td>GPT-4o</td><td>40.65</td><td>33.66</td></tr><tr><td colspan="3">Image caption + LLMs</td></tr><tr><td>Qwen-7B-Chat</td><td>4.71</td><td>5.21</td></tr><tr><td>Yi-34B-Chat</td><td>16.96</td><td>16.26</td></tr><tr><td>Internlm2-20B-Chat</td><td>17.40</td><td>16.43</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>22.93</td><td>24.24</td></tr><tr><td>GPT-3.5</td><td>20.56</td><td>15.77</td></tr><tr><td>Claude3 Sonnet</td><td>26.31</td><td>17.43</td></tr><tr><td>GPT-4</td><td>36.08</td><td>27.40</td></tr><tr><td>GPT-4o</td><td>41.50</td><td>33.07</td></tr><tr><td colspan="3">LMMs</td></tr><tr><td>Qwen-VL-Chat</td><td>7.70</td><td>5.55</td></tr><tr><td>Yi-VL-34B</td><td>17.34</td><td>14.68</td></tr><tr><td>InternVL-Chat-V1.5</td><td>17.07</td><td>15.82</td></tr><tr><td>LLaVA-NeXT-34B</td><td>17.74</td><td>16.74</td></tr><tr><td>Qwen-VL-Max</td><td>20.14</td><td>21.49</td></tr><tr><td>Gemini Pro Vision</td><td>21.61</td><td>18.76</td></tr><tr><td>Claude3 Sonnet</td><td>26.52</td><td>17.21</td></tr><tr><td>GPT-4V</td><td>36.18</td><td>26.55</td></tr><tr><td>GPT-4o</td><td>43.04</td><td>34.39</td></tr></table>

Table 16: Full results of Data Leakage Detection on the base models or text-only chat models behind the evaluated models (continued). The “Correspondence” column indicates the text-only chat model and multimodal (MM) chat model corresponding to the model being detected. “# Leak.” denotes the number of leakage instances. “# T” represents the number of instances correctly answered among these leaks by the text-only chat model, while “# MM” represents the number of instances correctly answered among these leaks by the multimodal chat model. 

<table><tr><td rowspan="2">Model to-be-detected</td><td colspan="2">Correspondence</td><td colspan="3">Math</td><td colspan="3">Physics</td><td colspan="3">Chemistry</td></tr><tr><td>Text-only Chat Model</td><td>MM Chat Model</td><td># Leak.</td><td># T</td><td># MM</td><td># Leak.</td><td># T</td><td># MM</td><td># Leak.</td><td># T</td><td># MM</td></tr><tr><td>InternLM2-20B</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat-1.5</td><td>14</td><td>1</td><td>2</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>internLM2-20B-Chat</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat1.5</td><td>17</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Yi-34B</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>10</td><td>2</td><td>2</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Yi-34B-Chat</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Nous-Hermes-2-Yi-34B</td><td>-</td><td>LLaVA-NeXT-34B</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>0</td></tr><tr><td>Qwen-7B</td><td>Qwen-7B-Chat</td><td>Qwen-VL-Chat</td><td>8</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Qwen1.5-32B</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>24</td><td>3</td><td>-</td><td>3</td><td>1</td><td>-</td><td>1</td><td>0</td><td>-</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>19</td><td>2</td><td>-</td><td>4</td><td>2</td><td>-</td><td>3</td><td>1</td><td>-</td></tr><tr><td>GPT-4o</td><td>GPT-4o</td><td>GPT-4o</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="2">Model to-be-detected</td><td colspan="2">Correspondence</td><td colspan="3">Biology</td><td colspan="3">Geography</td><td colspan="3">Astronomy</td></tr><tr><td>Text-only Chat Model</td><td>MM Chat Model</td><td># Leak.</td><td># T</td><td># MM</td><td># Leak.</td><td># T</td><td># MM</td><td># Leak.</td><td># T</td><td># MM</td></tr><tr><td>InternLM2-20B</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat-1.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>InternLM2-20B-Chat</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat-1.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Yi-34B</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Yi-34B-Chat</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Nous-Hermes-2-Yi-34B</td><td>-</td><td>LLaVA-NeXT-34B</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>0</td></tr><tr><td>Qwen-7B</td><td>Qwen-7B-Chat</td><td>Qwen-VL-Chat</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Qwen1.5-32B</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>1</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>5</td><td>1</td><td>-</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>1</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td><td>0</td><td>0</td><td>-</td></tr><tr><td>GPT-4o</td><td>GPT-4o</td><td>GPT-4o</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 18: Full results of Data Leakage Detection on the base models or text-only chat models behind the evaluated models (continued). The “Correspondence” column indicates the text-only chat model and multimodal (MM) chat model corresponding to the model being detected. “# Leak.” denotes the number of leakage instances. “# T” represents the number of instances correctly answered among these leaks by the text-only chat model, while “# MM” represents the number of instances correctly answered among these leaks by the multimodal chat model. 

<table><tr><td rowspan="2">Model to-be-detected</td><td colspan="2">Correspondence</td><td colspan="3">CS</td><td colspan="3">Overall</td></tr><tr><td>Text-only Chat Model</td><td>MM Chat Model</td><td># Leak.</td><td># T</td><td># MM</td><td># Leak.</td><td># T</td><td># MM</td></tr><tr><td>InternLM2-20B</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat-1.5</td><td>1</td><td>1</td><td>1</td><td>19</td><td>2</td><td>3</td></tr><tr><td>InternLM2-20B-Chat</td><td>InternLM2-20B-Chat</td><td>InternVL-Chat-1.5</td><td>1</td><td>1</td><td>1</td><td>19</td><td>3</td><td>2</td></tr><tr><td>Yi-34B</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>0</td><td>0</td><td>0</td><td>12</td><td>2</td><td>2</td></tr><tr><td>Yi-34B-Chat</td><td>Yi-34B-Chat</td><td>Yi-VL-34B</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td></tr><tr><td>Nous-Hermes-2-Yi-34B</td><td>-</td><td>LLaVA-NeXT-34B</td><td>0</td><td>-</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Qwen-7B</td><td>Qwen-7B-Chat</td><td>Qwen-VL-Chat</td><td>1</td><td>1</td><td>1</td><td>11</td><td>2</td><td>1</td></tr><tr><td>Qwen1.5-32B</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>9</td><td>-</td><td>-</td><td>43</td><td>14</td><td>0</td></tr><tr><td>Qwen1.5-32B-Chat</td><td>Qwen1.5-32B-Chat</td><td>-</td><td>3</td><td>3</td><td>-</td><td>30</td><td>8</td><td>0</td></tr><tr><td>GPT-4o</td><td>GPT-4o</td><td>GPT-4o</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

# F Case Study

# F.1 Cases for Error Analysis

From Figure 19 to Figure 25, we showcase examples of various error types across different disciplines.

# Problem:

In the diagram, rectangle PQRS is placed inside rectangle ABCD in two different ways: first, with Q at B and R at C; second, with P on AB, Q on BC, R on CD, and S on DA. [figure1]. If AB = 718 and PQ = 250, determine the length of BC.

![](images/96639fc981eb9bc49772a88981a71e0ebaa0d81777a578a257cefbea062de338.jpg)  
[figure1]

# Solution:

Let $BC = x$ , $PB = b$ , and $BQ = a$ . Since $BC = x$ , then $AD = PS = QR = x$ . Since $BC = x$ and $BQ = a$ , then $QC = x - a$ . Since $AB = 718$ and $PB = b$ , then $AP = 718 - b$ . Note that $PQ = SR = 250$ . Let $\angle BQP = \theta$ . Since $\triangle PBQ$ is right-angled at $B$ , then $\angle BPQ = 90^\circ - \theta$ . Since $BQC$ is a straight angle and $\angle PQR = 90^\circ$ , then $\angle RQC = 180^\circ - 90^\circ - \theta = 90^\circ - \theta$ . Since $APB$ is a straight angle and $\angle SPQ = 90^\circ$ , then $\angle APS = 180^\circ - 90^\circ - (90^\circ - \theta) = \theta$ . Since $\triangle SAP$ and $\triangle QCR$ are each right-angled and have another angle in common with $\triangle PBQ$ , then these three triangles are similar. Continuing in the same way, we can show that $\triangle RDS$ is also similar to these three triangles. Since $RS = PQ$ , then $\triangle RDS$ is actually congruent to $\triangle PBQ$ (angle-side-angle). Similarly, $\triangle SAP$ is congruent to $\triangle QCR$ . In particular, this means that $AS = x - a$ , $SD = a$ , $DR = b$ , and $RC = 718 - b$ . Since $\triangle SAP$ and $\triangle PBQ$ are similar, then $\frac{SA}{PB} = \frac{AP}{BQ} = \frac{SP}{PQ}$ . Thus, $\frac{x - a}{b} = \frac{718 - b}{a} = \frac{x}{250}$ . Also, by the Pythagorean Theorem in $\triangle PBQ$ , we obtain $a^2 + b^2 = 250^2$ . By the Pythagorean Theorem in $\triangle SAP$ ,

$$
x ^ {2} = (x - a) ^ {2} + (7 1 8 - b) ^ {2}
$$

$$
x ^ {2} = x ^ {2} - 2 a x + a ^ {2} + (7 1 8 - b) ^ {2}
$$

$$
0 = - 2 a x + a ^ {2} + (7 1 8 - b) ^ {2}
$$

Since $a^2 + b^2 = 250^2$ , then $a^2 = 250^2 - b^2$ . Since $\frac{718 - b}{a} = \frac{x}{250}$ , then $ax = 250(718 - b)$ . Therefore, substituting into (\*), we obtain

$$
0 = - 2 (2 5 0) (7 1 8 - b) + 2 5 0 ^ {2} - b ^ {2} + (7 1 8 - b) ^ {2}
$$

$$
b ^ {2} = 2 5 0 ^ {2} - 2 (2 5 0) (7 1 8 - b) + (7 1 8 - b) ^ {2}
$$

$$
b ^ {2} = ((7 1 8 - b) - 2 5 0) ^ {2} \quad \left(\text { since } y ^ {2} - 2 y z + z ^ {2} = (y - z) ^ {2}\right)
$$

$$
b ^ {2} = (4 6 8 - b) ^ {2}
$$

$$
b = 4 6 8 - b \quad (\text { since } b \neq b - 4 6 8)
$$

$$
2 b = 4 6 8
$$

$$
b = 2 3 4
$$

Therefore, $a^2 = 250^2 - b^2 = 250^2 - 234^2 = (250 + 234)(250 - 234) = 484 \cdot 16 = 22^2 \cdot 4^2 = 88^2$ and so $a = 88$ .

Finally, $x = \frac{250(718 - b)}{a} = \frac{250 \cdot 484}{88} = 1375$ . Therefore, $BC = \boxed{1375}$ .

# GPT 4V's output:

To solve this problem, we can use the Pythagorean theorem. In the first placement of rectangle PQRS inside rectangle ABCD, we see that PQ is parallel to AB and QR is parallel to BC. Since PQ = 250 and AB = 718, we can denote the length of BC as x. In the second placement, the rectangle PQRS is rotated such that its corners touch the sides of rectangle ABCD. The diagonal PR of rectangle PQRS becomes the diameter of an inscribed circle within rectangle ABCD. This means that the diagonal PR is the hypotenuse of a right triangle with sides PQ and QR. Since PQRS is a rectangle, PQ = RS and PS = QR. Therefore, the length of QR is also x. Now we have a right triangle with sides 250 and x, and the hypotenuse is the diagonal PR. Using the Pythagorean theorem, we have:

$$
P R ^ {2} = P Q ^ {2} + Q R ^ {2}
$$

$$
P R ^ {2} = 2 5 0 ^ {2} + x ^ {2}
$$

$$
P R ^ {2} = 6 2 5 0 0 + x ^ {2}
$$

The diagonal $PR$ also bisects rectangle $ABCD$ into two right triangles, each with sides $AB$ and $BC$ . Therefore, the length of the diagonal $PR$ can also be calculated using the Pythagorean theorem with sides $AB$ and $BC$ :

$$
P R ^ {2} = A B ^ {2} + B C ^ {2}
$$

$$
P R ^ {2} = 7 1 8 ^ {2} + x ^ {2}
$$

$$
P R ^ {2} = 5 1 5 5 2 4 + x ^ {2}
$$

Since both expressions represent $PR^{2}$ , we can set them equal to each other:

$$
6 2 5 0 0 + x ^ {2} = 5 1 5 5 2 4 + x ^ {2}
$$

Subtract $x^{2}$ from both sides:

$$
6 2 5 0 0 = 5 1 5 5 2 4
$$

Now we see that there is a mistake in the calculation because the left side should not equal the right side directly. We need to subtract 62500 from 515524 to find the value of $x^2$ :

$$
x ^ {2} = 5 1 5 5 2 4 - 6 2 5 0 0
$$

$$
x ^ {2} = 4 5 3 0 2 4
$$

Now we take the square root of both sides to find $x$ :

$$
x = \sqrt {4 5 3 0 2 4}
$$

$$
x = 6 7 3
$$

Therefore, the length of $BC$ is 673. The final answer is 673.

Figure 19: An example of a math problem with a logical reasoning error.

# Physics - Logical Reasoning Error

# Problem:

A bead is placed on a horizontal rail, along which it can slide frictionlessly. It is attached to the end of a rigid, massless rod of length R. A ball is attached at the other end. Both the bead and the ball have mass M. The system is initially stationary, with the ball directly above the bead. The ball is then given an infinitesimal push, parallel to the rail.[figure1] Assume that the rod and ball are designed in such a way (not shown explicitly in the diagram) so that they can pass through the rail without hitting it. In other words, the rail only constrains the motion of the bead. Two subsequent states of the system are shown below.[figure2] Derive an expression for the force in the rod when the ball is directly below the bead, as shown at right above.

![](images/ae21cc06abb36aa2de4caa323fed17443a9dc03ee5a53053e925a4872ef16a28.jpg)

<details>
<summary>text_image</summary>

M
θ
R
M
</details>

[figure1]

![](images/1cdfa6696d1a9f8802f250a7dca031401d9bf0da37661c3bfbe0fb698fbf4821.jpg)

<details>
<summary>text_image</summary>

θ = π/2
θ = π
</details>

[figure2]

# Solution:

At this point the released gravitational potential energy is 2MgR, and both masses are moving horizontally with speed v, where

$$
2 \frac {1}{2} M v ^ {2} = 2 M g R \Rightarrow v = \sqrt {2 g R}
$$

Work in the frame moving to the right with speed v. In this frame the bead is stationary and the ball has velocity 2v and is instantaneously rotating about the bead, so it must be experiencing a centripetal force

$$
\frac {M (2 v) ^ {2}}{R} = 8 M g
$$

Unlike in part (a), there are no additional contribution from the acceleration of the rotation center, because the bead can only ever accelerate horizontally, and the force in the rod at this moment is vertical. Since the ball also experiences a downward force of Mg due to gravity, the force in the rod is a tension

$$
T = \boxed {9 M g}
$$

# GPT 4V's output:

To derive an expression for the force in the rod when the ball is directly below the bead, we can employ the conservation of energy and principles from circular motion.

Since the bead and ball are on a frictionless rail and assuming no other non-conservative forces are acting on the system, the total mechanical energy is conserved. The initial potential energy of the ball when it is given an infinitesimal push will convert into kinetic energy as the ball swings down.

At the top position (initial state), the potential energy (PE) is maximum and kinetic energy (KE) is zero. We define the gravitational potential energy to be zero at the lowest point. At the highest point, the potential energy is MgR because the height of the ball above the lowest point is R.

Hence, $PE_{initial} = MgR.$

At the bottom position (final state), when the ball is directly below the bead, all the potential energy will have been converted to kinetic energy (assuming conservation of energy). If v is the velocity of the ball at the bottom, then $KE_{final} = \frac{1}{2} M v^{2}$ .

Since energy is conserved, $PE_{initial} = KE_{final}$ . Thus,

$$
M g R = \frac {1}{2} M v ^ {2}
$$

$$
2 g R = v ^ {2}
$$

$$
v = \sqrt {2 g R}
$$

Now, let us consider the forces acting on the ball at the bottom position. We have the gravitational force $Mg$ acting downwards and the force in the rod $F$ acting upwards. When the ball is at the bottom, it is in circular motion and the centripetal force required to keep it in circular motion is provided entirely by the force in the rod, which is the centripetal force $F_{\mathrm{centripetal}}$ , since gravity acts in the same direction as the centripetal acceleration at this point.

The centripetal force is given by $F_{centripetal} = \frac{Mv^{2}}{R}$ .

Substituting $v = \sqrt{2gR}$ into the above equation, we get

$$
F _ {\text { centripetal }} = \frac {M (2 g R)}{R}
$$

$$
F _ {\text { centripetal }} = 2 M g
$$

Since the centripetal force is provided by the force in the rod and directed towards the center of the circular motion (upwards towards the ball), at this moment, the rod is under tension, not compression.

Thus, the force in the rod when the ball is directly below the bead and in tension is F = 2Mg.

The final answer is 2Mg.

Figure 20: An example of a physics problem with a logical reasoning error.

# Chemistry - Visual Reasoning Error

# Problem:

At temperature $t^{\circ}C$ , a solution is prepared such that $c\left(\mathrm{H}_{2}\mathrm{CO}_{3}\right)+c\left(\mathrm{HCO}_{3}^{-}\right)+c\left(\mathrm{CO}_{3}^{2-}\right)=1.000\times10^{-3}\;\mathrm{mol}\cdot\mathrm{L}^{-1}$ . This solution is a mixture of $H_{2}CO_{3}$ and HCl or $H_{2}CO_{3}$ and NaOH. The negative logarithms of the concentrations of some particles in the solution ( $-lgc$ ) versus pH are shown in the figure. Which of the following statements are incorrect:

![](images/2ce4a063e993fd87dc6064a3bc3d9bdf74e6be5a9b34cb27523173374a1e5914.jpg)

<details>
<summary>line</summary>

| pH   | -lgC  | Species     |
| ---- | ----- | ----------- |
| 6.3  | 4.0   | H₂CO₃       |
| 10.3 | 4.0   | HCO₃⁻       |
| 10.3 | 4.0   | OH⁻         |
| 12.3 | 6.0   | CO₂²⁻       |
</details>

[figure1]

A: In the solution with pH = a: c (HCO $_{3}^{-}$ ) > c (H $_{2}$ CO $_{3}$ ) > c (CO $_{3}^{2-}$ ) > c (H $^{+}$ )   
B: In the solution with pH = 7: c (Na $^{+}$ ) > c (H $_{2}$ CO $_{3}$ )   
C: In the solution with $\mathrm{pH} = 10.3$ : c (Na $^{+}$ ) $< 1.000 \times 10^{-3}$ mol · L $^{-1}$   
D: At $25^{\circ}\mathrm{C}$ , the equilibrium constant for the reaction $\mathrm{H}_2\mathrm{CO}_3 + \mathrm{CO}_3^{2-} \rightleftharpoons 2\mathrm{HCO}_3^-$ is $1.0 \times 10^4$

# Solution:

A. According to the information from the figure, in the solution with $\mathrm{pH} = \mathrm{a}$ , $c\left(\mathrm{CO}_3^{2-}\right) > c\left(\mathrm{H}_2\mathrm{CO}_3\right)$ , so statement A is incorrect.   
B. In the solution with $\mathrm{pH} = 7$ , according to the information from the figure, $c\left(\mathrm{HCO}_3^-\right) > c\left(\mathrm{H}_2\mathrm{CO}_3\right) > c\left(\mathrm{CO}_3^{2-}\right)$ . According to charge conservation and $\mathrm{pH} = 7$ , $c\left(\mathrm{Na}^+\right) = c\left(\mathrm{HCO}_3^-\right) + 2c\left(\mathrm{CO}_3^{2-}\right)$ , so $c\left(\mathrm{Na}^+\right) > c\left(\mathrm{H}_2\mathrm{CO}_3\right)$ . Therefore, statement B is correct.   
C. At $25^{\circ}$ C, when pH = 6.3, $c(\mathrm{H}_{2}\mathrm{CO}_{3}) = c(\mathrm{HCO}_{3}^{-})$ , then $K_{a}(\mathrm{H}_{2}\mathrm{CO}_{3}) = c(\mathrm{H}^{+}) = 1.000 \times 10^{-6.3} \, \mathrm{mol} \cdot \mathrm{L}^{-1}$ . In the solution with pH = 10.3, $c(\mathrm{CO}_{3}^{2-}) = c(\mathrm{HCO}_{3}^{-})$ , $K_{a}(\mathrm{HCO}_{3}^{-}) = c(\mathrm{H}^{+}) = 1.000 \times 10^{-10.3} \, \mathrm{mol} \cdot \mathrm{L}^{-1}$ . Then

$$
K _ {a} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right) = \frac {\mathrm{c} \left(\mathrm{HCO} _ {3} ^ {-}\right) \times \mathrm{c} \left(\mathrm{H} ^ {+}\right)}{\mathrm{c} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right)} = \frac {\mathrm{c} \left(\mathrm{HCO} _ {3} ^ {-}\right) \times 1 0 ^ {- 1 0 . 3}}{\mathrm{c} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right)} = 1 0 ^ {- 6. 3}. \text {When pH} = 1 0. 3, \mathrm{c} \left(\mathrm{CO} _ {3} ^ {2 -}\right) = \mathrm{c} \left(\mathrm{HCO} _ {3} ^ {-}\right),
$$

so $\frac{c\left(\mathrm{HCO}_3^-\right) + c\left(\mathrm{CO}_3^{2-}\right)}{c\left(\mathrm{H}_2\mathrm{CO}_3\right)} = 2\times 10^4$ , $c\left(\mathrm{H}_2\mathrm{CO}_3\right) + c\left(\mathrm{HCO}_3^- \right) + c\left(\mathrm{CO}_3^{2-}\right) = 1.000\times 10^{-3}\mathrm{mol}\cdot \mathrm{L}^{-1}$ . Therefore, $c\left(\mathrm{H}_2\mathrm{CO}_3\right) = 5.000\times 10^{-8}\mathrm{mol}\cdot \mathrm{L}^{-1}$ , and $2c\left(\mathrm{CO}_3^{2-}\right) = 1.000\times 10^{-3}\mathrm{mol}\cdot \mathrm{L}^{-1} - 5.000\times 10^{-8}\mathrm{mol}\cdot \mathrm{L}^{-1}$ . According to charge conservation, $c\left(\mathrm{Na}^+\right) > c\left(\mathrm{HCO}_3^- \right) + 2c\left(\mathrm{CO}_3^{2-}\right)$ , i.e., $c\left(\mathrm{Na}^+\right) > 3c\left(\mathrm{CO}_3^{2-}\right)$ , and $c\left(\mathrm{Na}^+\right) > 1.000\times 10^{-3}\mathrm{mol}\cdot \mathrm{L}^{-1}$ . Therefore, statement C is incorrect.

D. At $25^{\circ}C$ , when pH = 6.3, $c(\mathrm{H}_{2}\mathrm{CO}_{3}) = c(\mathrm{HCO}_{3}^{-})$ , then $K_{a}(\mathrm{H}_{2}\mathrm{CO}_{3}) = c(\mathrm{H}^{+}) = 1.000 \times 10^{-6.3} \, \mathrm{mol} \cdot \mathrm{L}^{-1}$ . Similarly, at pH = 10.3, $c(\mathrm{CO}_{3}^{2-}) = c(\mathrm{HCO}_{3}^{-})$ , then $K_{a}(\mathrm{HCO}_{3}^{-}) = c(\mathrm{H}^{+}) = 1.000 \times 10^{-10.3} \, \mathrm{mol} \cdot \mathrm{L}^{-1}$ . The equilibrium constant for the reaction $H_{2}CO_{3} + CO_{3}^{2-} \rightleftharpoons 2HCO_{3}^{-}$ is:

$$
K = \frac {\mathrm{c} ^ {2} \left(\mathrm{HCO} _ {3} ^ {-}\right)}{\mathrm{c} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right) \times \mathrm{c} \left(\mathrm{CO} _ {3} ^ {2 -}\right)} = \frac {\mathrm{c} \left(\mathrm{HCO} _ {3} ^ {-}\right) \times \mathrm{c} \left(\mathrm{HCO} _ {3} ^ {-}\right) \times \mathrm{c} \left(\mathrm{H} ^ {+}\right)}{\mathrm{c} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right) \times \mathrm{c} \left(\mathrm{CO} _ {3} ^ {2 -}\right) \times \mathrm{c} \left(\mathrm{H} ^ {+}\right)} = \frac {K _ {a} \left(\mathrm{H} _ {2} \mathrm{CO} _ {3}\right)}{K _ {a} \left(\mathrm{HCO} _ {3} ^ {-}\right)} = \frac {1 0 ^ {- 6 . 3}}{1 0 ^ {- 1 0 . 3}} = 1 \times 1 0 ^ {4}
$$

Therefore, statement D is correct.

In conclusion, the answer is $A, C$ .

# GPT 4V's output:

First, we need to analyze the chart, which shows the variation of the negative logarithm of concentrations (-lgc) of $H_{2}CO_{3}$ , $HCO_{3}^{-}$ , $CO_{3}^{2-}$ , and $H^{+}$ at different pH values. This chart is a typical acid-base titration curve where $H_{2}CO_{3}$ is a diprotic weak acid that can dissociate into $HCO_{3}^{-}$ and $CO_{3}^{2-}$ .

Option A: At pH = a, according to the chart, the concentration of $HCO_{3}^{-}$ is the highest, followed by $H_{2}CO_{3}$ , then $CO_{3}^{2-}$ , and the concentration of $H^{+}$ is the lowest. This is consistent with the chart.

Option B: At pH = 7, the concentration of $H_{2}CO_{3}$ should be relatively low because it is a weak acid and does not dissociate completely at neutral pH. The concentration of $Na^{+}$ depends on whether NaOH has been added. If NaOH has been added, the concentration of $Na^{+}$ will increase. However, this option does not provide enough information to determine the exact concentration relationship between $Na^{+}$ and $H_{2}CO_{3}$ , so we cannot confirm if this option is correct. Option C: At pH = 10.3, according to the chart, the concentration of $CO_{3}^{2-}$ is the highest, and the concentration of $HCO_{3}^{-}$ is very low at this pH. If NaOH is added to the solution, the concentration of $Na^{+}$ will increase. However, since the total concentration $c(\mathrm{H}_{2}\mathrm{CO}_{3}) + c(\mathrm{HCO}_{3}^{-}) + c(\mathrm{CO}_{3}^{2-}) = 1.000 \times 10^{-3} \, \mathrm{mol\cdot L^{-1}}$ , the concentration of $Na^{+}$ cannot be lower than this value. Therefore, this option is incorrect.

Option D: The equilibrium constant K for the reaction $H_{2}CO_{3} + CO_{3}^{2-} \rightleftharpoons 2HCO_{3}^{-}$ can be expressed by the following equation:

$$
K = \frac {[ \mathrm{HCO} _ {3} ^ {-} ] ^ {2}}{[ \mathrm{H} _ {2} \mathrm{CO} _ {3} ] [ \mathrm{CO} _ {3} ^ {2 -} ]}
$$

According to the chart, at $\mathrm{pH} = 10.3$ , the concentrations of $\mathrm{HCO}_3^-$ and $\mathrm{CO}_3^{2-}$ are equal, which means $K = 1.0 \times 10^4$ is impossible because this would require $[\mathrm{HCO}_3^-]$ to be much greater than $[\mathrm{CO}_3^{2-}]$ . Therefore, this option is also incorrect.

In summary, options C and D are incorrect. So the final answer is $\boxed{C, D}$

Figure 21: An example of a chemistry problem with a visual reasoning error.

# Biology - Logical Reasoning Error

# Problem:

The bacterium Bradyrhizobium japonicum can infect soybean (Glycine max) roots and form nodules. The nitrogen fixation catalyzed by nitrogenase occurs in the nodules and the nitrogenase activity can be measured easily by acetylene reduction instead of nitrogen reduction. Scientists generated a defective mutation of $NAD^{+}$ -dependent malic enzyme, the enzyme that generates pyruvate and NADH, and infected soybean seedling roots with wildtype and mutant bacteria. The seedlings were grown in nitrogen-free media. After 14 and 28 days of inoculation, the number and weight of nodules in the seedlings and acetylene reduction activity were recorded [figure1]. Which of the following statements are correct:

![](images/306e3cbbf8f0bb6b67c4acd0199a6e95d169bf02e27a39f6d21284d684893bef.jpg)

![](images/8b38e328fdb1ad67b62b55fe4c69f361b0830c987855039af12a93a20140a201.jpg)

<details>
<summary>bar</summary>

| Days after inoculation | Acetylene reduction activity (μmol·hr⁻¹·g·indole dry weight) |
|---|---|
| 14 | 27 |
| 14 | 21 |
| 28 | 36 |
| 28 | 25 |
</details>

![](images/2b3206b0dc2ca7da7c7c91d32e53288b7889f4275fde723847c77e5678b339ea.jpg)

<details>
<summary>bar</summary>

| Days after inoculation | Acetylene reduction activity (μ mol·hr⁻¹·nodule⁻¹) |
|---|---|
| 14 | 0.03 |
| 28 | 0.14 |
D: The white bar represents the mean value, while the black bar represents the mean value with error bars. Error bars indicate variability. The y-axis is labeled in scientific notation (µ mol·hr⁻¹·nodule⁻¹).
</details>

[figure1]: Nodule number and dry weight and acetylene reduction acitivity of soybean. Soybean nodules infected with wild-type B. japonicum (open bars) and the dme mutant (solid bars) are presented.

A: Nitrogen fixation activity in nodules of the same treatment at 28 days after inoculation is higher than that at 14 days after inoculation.   
B: Both number and size of nodules increase with time from 14 to 28 days after inoculation with B. japonicum.   
C: The reduction in nitrogen-fixing activity of nodules infected by the mutant at 28 days after inoculation compared to those at 14 days after inoculation is due to the reduction of nitrogenase activity and nodule formation.   
D: Nitrogen fixation in B. japonicum -induced nodule is down-regulated by $\mathrm{NAD}^+$ dependent malic enzyme.

# Solution:

A. True. As shown in the figure, in both wildtype and mutant treatments, acetylene reduction activity in nodules at 28 days after inoculation is higher than that at 14 days after inoculation, indicating higher nitrogen fixation activity.   
B. True. In both treatments, number and size of nodules are higher at 28 days compared to those at 14 days after inoculation   
C. False. Number of nodules at 28 days after inoculation is higher than at 14 days after inoculation   
D. False. The mutation of $NAD^{+}$ -dependent malic enzyme in the bacteria results in a reduction of acetylene reduction, indicating that the enzyme (in wildtype) up-regulates the nitrogen fixation.

So we choose $A, B$ .

# GPT 4V's output:

Let's analyze the data presented in Fig.Q13 step by step:

A: We need to determine if nitrogen fixation activity in nodules of the same treatment at 28 days after inoculation is higher than that at 14 days after inoculation. Looking at the graphs C and D, which represent acetylene reduction activity (a proxy for nitrogen fixation activity), we can see that for both wild-type (open bars) and mutant (solid bars) B. japonicum, the activity is higher at 28 days than at 14 days. This is true for both per gram of nodule dry weight (graph C) and per nodule (graph D). Therefore, statement A is correct.   
B: To assess this statement, we need to look at graphs A and B. Graph A shows the nodule number per plant, and graph B shows the nodule dry weight. For both wild-type and mutant B. japonicum, the number and size (weight) of nodules increase from 14 to 28 days after inoculation. Therefore, statement B is correct. C: This statement suggests that the reduction in nitrogen-fixing activity of nodules infected by the mutant at 28 days after inoculation compared to those at 14 days is due to the reduction of nitrogenase activity and nodule formation. From graph C, we can see that the acetylene reduction activity per gram of nodule dry weight for the mutant does decrease from 14 to 28 days, indicating a reduction in nitrogenase activity. However, graph A shows that the number of nodules actually increases for the mutant from 14 to 28 days, which contradicts the second part of the statement. Therefore, statement C is not entirely correct.   
D: The statement suggests that nitrogen fixation in B. japonicum-induced nodules is down-regulated by $\mathrm{NAD + }$ -dependent malic enzyme. The mutant, which has a defective mutation in the $\mathrm{NAD + }$ -dependent malic enzyme, shows lower acetylene reduction activity compared to the wild-type in both graphs C and D at both time points. This suggests that the enzyme plays a role in nitrogen fixation, and its absence (or defectiveness) leads to down-regulation of nitrogen fixation. Therefore, statement D is correct.

The final answer is $\boxed{A,B,D}$ .

Figure 22: An example of a biology problem with a logical reasoning error.

# Geography - Knowledge Deficit Error

# Problem:

From December 17 to 20, 2020, the Beijing team went to Tengchong, Yunnan to participate in the study tour course design competition, using the Beihai Wetland in Tengchong, known for its plateau volcanic dammed lakes, as the theme of the study tour course design. Based on this, complete the following small questions. During the study tour, the team members found that compared to Beijing, Tengchong has ( ).

A. The tree shadows at sunrise are more towards the north.   
B. The temperature difference between day and night is larger.   
C. The sunset time is earlier.   
D. The tree shadows at noon are shorter.

# Solution:

According to the material, Tengchong (25°N, 98°E) belongs to Yunnan and is located to the southwest of Beijing (39°N, 116°E). From December 17 to 20, 2020, it was winter, around the winter solstice, with the sun rising in the southeast and the shadow at sunrise being in the northwest. Beijing is at a higher latitude than Tengchong, causing a greater shadow deviation, so A is incorrect. Tengchong is located near a plateau volcanic dammed lake, rich in geothermal energy, resulting in high nighttime temperatures and smaller temperature differences between day and night, so B is incorrect. Tengchong, being at a lower latitude, has longer daylight hours and a later sunset time, making C incorrect. Tengchong's lower latitude also means a larger solar altitude angle at noon, leading to shorter tree shadows, so D is correct.

# GPT 4V's output:

First, we need to understand the geographical locations of Beijing and Tengchong. Beijing is in the northern part of China, while Tengchong is in the southwestern part of China, near the Myanmar border. This means that Tengchong is to the southwest of Beijing.

A. The tree shadows at sunrise are more towards the north - Due to the Earth's rotation, the direction of tree shadows at sunrise is mainly determined by latitude, not by the east-west position. Therefore, this option is incorrect.   
B. The temperature difference between day and night is larger - Tengchong is located in a plateau area, while Beijing is in a plain area. Generally, the temperature difference between day and night in plateau areas is larger than in plain areas because high-altitude areas cool down quickly at night and receive strong solar radiation during the day. Therefore, this option might be correct.   
C. The sunset time is earlier - The sunset time is mainly determined by longitude, as the Earth rotates from west to east. Tengchong is to the southwest of Beijing, but this does not necessarily mean that the sunset time is earlier. This option requires more information to determine.   
D. The tree shadows at noon are shorter - The solar altitude angle at noon depends on latitude; the lower the latitude, the higher the solar altitude angle, and the shorter the tree shadows. Tengchong has a lower latitude than Beijing, so the tree shadows at noon should be shorter.   
In summary, options B and D are both possible. However, we need to determine which one is most related to the "plateau volcanic dammed lake" mentioned in the problem. Since this description emphasizes the plateau characteristics of Tengchong, we can reasonably infer that the larger temperature difference between day and night is most relevant to this characteristic. Therefore, the final answer is $\boxed{B}$ .

Figure 23: An example of a geography problem with a knowledge deficit error.

# Astronomy - Incomplete Response

# Problem:

In order to move a spacecraft between orbits we must apply a thrust using rockets, which changes the velocity of the spacecraft by $\Delta v$ . In this question we will ignore changes in the mass of the spacecraft due to the burning of fuel. For an object of mass m in a circular orbit of radius r around an object with mass M (where $m \ll M$ ) the orbital velocity, $v_{orb}$ , is given by the formula $v_{orb} = \sqrt{\frac{GM}{r}}$ . Approximating Mars' orbit as circular with a radius of 1.52AU, calculate the $\triangle v$ to go from Earth LEO to Mars i.e. $\Delta v = |\Delta v_{A}| + |\Delta v_{B}|$ . Compare your answer to the $\Delta v$ to reach Earth LEO.

# Solution:

$$
\begin{array}{l} \Delta v _ {\mathrm{A}} = \sqrt {\frac {6 . 6 7 \times 1 0 ^ {- 1 1} \times 1 . 9 9 \times 1 0 ^ {3 0}}{1 . 4 9 \times 1 0 ^ {1 1}}} \left(\sqrt {\frac {2 \times 1 . 5 2}{1 + 1 . 5 2}} - 1\right) \\ = 2 9 3 5 \mathrm{ms} ^ {- 1} \left(= 2. 9 4 \mathrm{km} \mathrm{s} ^ {- 1}\right) \\ \end{array}
$$

$$
\Delta v _ {\mathrm{B}} = \sqrt {\frac {6 . 6 7 \times 1 0 ^ {- 1 1} \times 1 . 9 9 \times 1 0 ^ {3 0}}{1 . 5 2 \times 1 . 4 9 \times 1 0 ^ {1 1}}} \left(1 - \sqrt {\frac {2 \times 1}{1 + 1 . 5 2}}\right)
$$

$$
= 2 6 4 2 \mathrm{ms} ^ {- 1} \left(= 2. 6 4 \mathrm{km} \mathrm{s} ^ {- 1}\right)
$$

$$
{\Delta v} {= 2. 9 4 + 2. 6 4}
$$

$$
= \boxed {5. 5 8} \mathrm{km} \mathrm{s} ^ {- 1}
$$

[The $\Delta v$ calculated here would be for the spacecraft to enter a circular orbit around the Sun at the same distance as Mars, but this would not constitute landing - the extra $\Delta v$ to get to the Martian surface increases the total for the whole transfer to roughly 8 km s $^{-1}$ (same as to get into LEO). Other, more complicated routes can be taken, some of which offer substantial efficiencies so the $\Delta v$ for the trip can be much lower (although the time taken to complete the manoeuvre will be longer, and the $\Delta v$ to reach LEO will still be the biggest single step)]

# GPT 4V's output:

To calculate the $\Delta v$ to go from Earth Low Earth Orbit (LEO) to Mars, we need to consider two main maneuvers: the transfer from Earth's orbit to the transfer orbit (Hohmann transfer orbit), and then from the transfer orbit to Mars' orbit.

1. The first part of the $\Delta v$ , denoted as $\Delta v_{A}$ , is the change in velocity needed to move from Earth's orbit to the transfer orbit. This is the velocity at perihelion of the transfer orbit minus the Earth's orbital velocity.

2. The second part, $\Delta v_{B}$ , is the change in velocity needed to move from the transfer orbit to Mars' orbit. This is Mars' orbital velocity minus the velocity at aphelion of the transfer orbit.

The Hohmann transfer orbit is an elliptical orbit that touches Earth's orbit at its closest point to the Sun (perihelion) and Mars' orbit at its farthest point from the Sun (aphelion).

The semi-major axis of the transfer orbit $(a_{t})$ is the average of the Earth's orbital radius $(r_E)$ and Mars' orbital radius $(r_M)$ . Given that 1 AU (Astronomical Unit) is approximately $1.496\times 10^{8}\mathrm{km}$ , we have:

$$
r _ {E} = 1 \times 1. 4 9 6 \times 1 0 ^ {8} \mathrm{km(Earth'sorbitradius)} r _ {M} = 1. 5 2 \times 1. 4 9 6 \times 1 0 ^ {8} \mathrm {km(Mars^ {\prime} orbit radius)}
$$

$$
a _ {t} = \frac {r E + r M}{2}
$$

The orbital velocity of an object is given by $v_{\mathrm{orb}} = \sqrt{\frac{GM}{r}}$ , where $G$ is the gravitational constant and $M$ is the mass of the central object (in this case, the Sun).

The velocity at perihelion of the transfer orbit $(v_{p,t})$ is given by:

$$
v _ {p, t} = \sqrt {\frac {G M}{r _ {E}}} \times \sqrt {\frac {2 r _ {M}}{r _ {E} + r _ {M}}}
$$

The velocity at aphelion of the transfer orbit $(v_{a,t})$ is given by:

$$
v _ {a, t} = \sqrt {\frac {G M}{r _ {M}}} \times \sqrt {\frac {2 r _ {E}}{r _ {E} + r _ {M}}}
$$

The Earth's orbital velocity $(v_{E})$ is:

$$
v _ {E} = \sqrt {\frac {G M}{r _ {E}}}
$$

Mars' orbital velocity $(v_{M})$ is:

$$
v _ {M} = \sqrt {\frac {G M}{r _ {M}}}
$$

Now we can calculate $\Delta v_{A}$ and $\Delta v_{B}$ :

$\Delta v_{A} = v_{p,t} - v_{E}\Delta v_{B} = v_{M} - v_{a,t}$

And the total $\Delta v$ is:

$\Delta v = |\Delta v_A| + |\Delta v_B|$

Let's calculate these values step by step. We will use the gravitational constant $G = 6.674 \times 10^{-20} \, \mathrm{km}^3\mathrm{kg}^{-1}\mathrm{s}^{-2}$ and the mass of the Sun $M = 1.989 \times 10^{30}\mathrm{kg}$ .

First, we calculate the semi-major axis of the transfer orbit:

$$
a _ {t} = \frac {1 \times 1 . 4 9 6 \times 1 0 ^ {8} + 1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}{2}
$$

Now, we calculate the velocities:

$$
v _ {p, t} = \sqrt {\frac {6 . 6 7 4 \times 1 0 ^ {- 2 0} \times 1 . 9 8 9 \times 1 0 ^ {3 0}}{1 . 4 9 6 \times 1 0 ^ {8}}} \times \sqrt {\frac {2 \times 1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}{1 . 4 9 6 \times 1 0 ^ {8} + 1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}}
$$

$$
v _ {a, t} = \sqrt {\frac {6 . 6 7 4 \times 1 0 ^ {- 2 0} \times 1 . 9 8 9 \times 1 0 ^ {3 0}}{1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}} \times \sqrt {\frac {2 \times 1 . 4 9 6 \times 1 0 ^ {8}}{1 . 4 9 6 \times 1 0 ^ {8} + 1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}}
$$

$$
v _ {E} = \sqrt {\frac {6 . 6 7 4 \times 1 0 ^ {- 2 0} \times 1 . 9 8 9 \times 1 0 ^ {3 0}}{1 . 4 9 6 \times 1 0 ^ {8}}}
$$

$$
v _ {M} = \sqrt {\frac {6 . 6 7 4 \times 1 0 ^ {- 2 0} \times 1 . 9 8 9 \times 1 0 ^ {3 0}}{1 . 5 2 \times 1 . 4 9 6 \times 1 0 ^ {8}}}
$$

Now we can calculate $\Delta v_{A}$ and $\Delta v_{B}$ :

$\Delta v_{A} = v_{p,t} - v_{E}$

$\Delta v_{B} = v_{M} - v_{a,t}$

And finally, the total $\Delta v$ :

$\Delta v = |\Delta v_A| + |\Delta v_B|$

Let's calculate these values.

Figure 24: An example of an astronomy problem with an incomplete response.

# Problem:

A girl named Xenia has a cupboard that looks like an arc from ahead. The arc is made of a semicircle with radius r (the cupboard's top) and two walls of height h (the cupboard's sides). The cupboard's depth is r, that is, it looks like a rectangle with base r and height h+r from the sides. The figure below shows what the cupboard looks like (the front view is on the left, the side view is on the right). Xenia got lots of balloons for her birthday. The girl hates the mess, so she wants to store the balloons in the cupboard. Luckily, each balloon is a sphere with radius. Help Xenia calculate the maximum number of balloons she can put in her cupboard. [figure1] You can say that a balloon is in the cupboard if you can't see any part of the balloon on the left or right view. The balloons in the cupboard can touch each other. It is not allowed to squeeze the balloons or deform them in any way. You can assume that the cupboard's walls are negligibly thin.

![](images/d4f8512d2e7f6db5963123e19c1d48343415b395c309f108976c2c13afe80eef.jpg)

<details>
<summary>text_image</summary>

r
h
r
</details>

[figure1]

# Input

The single line contains two integers r, h (1 <= r, h <= $10^{7}$ )

# Output

Print a single integer — the maximum number of balloons Xenia can put in the cupboard.

# Examples

# Input

1 1

# Output

3

# Input

1

# Output

5

# Input

21

# Output

2

time limit: 2 seconds

memory limit: 256 megabytes

# GPT 4V's output:

# import math

\# Function to calculate the maximum number of balloons

def max\_balloons(r, h):

\# Calculate the number of balloons that fit in the height of the walls

balloons\_in\_height = (2 \* h) // r

\# Calculate the number of balloons that fit in the semicircle on top

\# The diameter of the semicircle is r, so it can fit one balloon of radius r/2

\# The remaining height in the semicircle is r/2, which can fit another balloon

balloons\_in\_semicircle = 1 + (h \* 2 + r) // r

\# Calculate the number of additional balloons that can fit in the corners of the semicircle

\# This is the number of full r/2 radii that fit into the remaining arc length after placing

\# the first balloon in the semicircle

remaining\_arclength = (math.pi \* r / 2) - (r / 2)

additional\_balloons = int(remaining\_arc\_length // (r / 2))

\# The total number of balloons is the sum of the above calculated values

total\_balloons = balloons\_in\_height + balloons\_in\_semicircle + additional\_balloons

return total\_balloons

\# Read input

r, h = map(int, input().split())

\# Calculate and print the result

print(max\_balloons(r, h))

This program defines a function "max balloons" that calculates the maximum number of balloons that can fit in the cupboard based on the given dimensions. It then reads the input values for "r" and "h", calls the function with these values, and prints the result. The program takes into account the balloons that fit in the height of the walls, the semicircle on top, and additional balloons that can fit in the corners of the semicircle.

Figure 25: An example of a programming problem with an understanding error.

# G Consideration for Social Impact

Certainly, it is essential to point out that as AI performs increasingly well on our benchmark, potentially even surpassing human capabilities, there are some potential ethical and moral risks that require collective oversight.

# H Limitations and Future Work

Despite the value of this benchmark, there remains work to be done in the future. Firstly, our benchmark inevitably introduces some noisy problems, we will actively utilize community feedback to continuously refine it. Additionally, we aim to release new versions of the benchmark annually to mitigate issues related to data leakage. Moreover, this benchmark is currently limited to evaluating models' abilities to solve complex problems. In the future, we aspire for AI to assist with complex tasks and demonstrate value in real-world applications such as AI4Science and AI4Engineering rather than just problem-solving. This will be the goal of our future benchmark designs for evaluating AI capabilities. Nonetheless, at present, OlympicArena plays an essential role as a catalyst for further advancements.