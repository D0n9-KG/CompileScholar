# ASSESSING THE CREATIVITY OF LLMs IN PROPOSING NOVEL SOLUTIONS TO MATHEMATICAL PROBLEMS

Junyi Ye $^{\dagger}$ , Jingyi Gu $^{\dagger}$ , Xinyun Zhao $^{\dagger}$ , Wenpeng Yin $^{\P}$ , Guiling Wang $^{\dagger}$

$^{\dagger}$ New Jersey Institute of Technology, ¶ Pennsylvania State University

{jy394, jg95, xz43, guiling.wang}@njit.edu; wenpeng@psu.edu

# ABSTRACT

The mathematical capabilities of AI systems are complex and multifaceted. Most existing research has predominantly focused on the correctness of AI-generated solutions to mathematical problems. In this work, we argue that beyond producing correct answers, AI systems should also be capable of, or assist humans in, developing novel solutions to mathematical challenges. This study explores the creative potential of Large Language Models (LLMs) in mathematical reasoning, an aspect that has received limited attention in prior research. We introduce a novel framework and benchmark, CREATIVEMATH, which encompasses problems ranging from middle school curricula to Olympic-level competitions, designed to assess LLMs' ability to propose innovative solutions after some known solutions have been provided. Our experiments demonstrate that, while LLMs perform well on standard mathematical tasks, their capacity for creative problem-solving varies considerably. Notably, the Gemini-1.5-Pro model outperformed other LLMs in generating novel solutions. This research opens a new frontier in evaluating AI creativity, shedding light on both the strengths and limitations of LLMs in fostering mathematical innovation, and setting the stage for future developments in AI-assisted mathematical discovery $^{1}$ .

# 1 INTRODUCTION

In recent years, artificial intelligence has made significant strides, particularly in the development of Large Language Models (LLMs) capable of tackling complex problem-solving tasks. Models like GPT-4 and Gemini-1.5-Pro have demonstrated impressive proficiency on rigorous mathematical benchmarks (Ahn et al., 2024) such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021a), underscoring the evolving role of LLMs from simple text generators to sophisticated tools capable of engaging with high-level mathematical challenges. Beyond solving student-oriented math problems, leading mathematicians have begun exploring the use of LLMs to assist in tackling unresolved mathematical challenges (Romera-Paredes et al., 2024; Trinh et al., 2024). Despite these models' success in achieving high accuracy on existing mathematical datasets, their potential for creative problem-solving remains largely underexplored.

Mathematical creativity goes beyond solving problems correctly; it involves generating novel solutions, applying unconventional techniques, and offering deep insights—areas traditionally associated with human ingenuity. Yet, most studies have focused primarily on correctness and efficiency, paying little attention to the innovative approaches LLMs might employ. Furthermore, creativity in mathematical problem-solving is rarely integrated into existing benchmarks, limiting our understanding of LLMs' full potential. The current research landscape lacks a comprehensive framework that evaluates both the accuracy and the creative capacity of LLMs. This gap highlights the need for new methodologies and benchmarks specifically designed to assess and cultivate the creative problem-solving abilities of LLMs in mathematics, which is the focus of this paper.

We created the dataset CREATIVEMATH, a comprehensive math benchmark that includes problems from middle school to Olympic-level competitions, each accompanied by multiple high-quality solutions ranging from straightforward to highly innovative approaches. Additionally, we designed a

multi-stage framework to rigorously evaluate the creativity of LLMs in generating novel math solutions. This evaluation spans closed-source, open-source, and math-specialized LLMs, assessing both the correctness and novelty of their solutions based on different reference prior solutions.

Our evaluation revealed several interesting key insights: (1) Gemini-1.5-Pro excelled in generating unique solutions, with most correct answers being distinct from the provided references, while smaller and math-specialized models struggled with novelty. (2) Providing more reference solutions generally improved accuracy, with Gemini-1.5-Pro achieving perfect accuracy with four prior solutions. However, increased references made it harder for models to generate unique solutions, indicating a trade-off between leveraging existing knowledge and fostering creativity. (3) As math problem difficulty increased, LLM accuracy declined, but successful solutions were more likely to be innovative, suggesting that tougher problems encourage creativity. (4) Analysis of solution similarity among different LLMs showed that models like Llama-3-70B and Yi-1.5-34B explored diverse approaches, while others like Mixtral-8x22B produced more similar solutions, highlighting the value of using a diverse set of LLMs to enhance originality.

This study lays the groundwork for future advancements in LLM math creativity. The major contributions include: (1) Introducing a new task—evaluating LLMs’ mathematical creativity, (2) Creating the CREATIVEMATH dataset, (3) Developing a framework for assessing mathematical creativity in LLMs, and (4) Evaluating state-of-the-art LLMs, revealing key insights into their strengths and limitations.

# 2 RELATED WORK

LLMs have demonstrated significant advancements in both mathematical reasoning and creative capabilities, making them increasingly powerful tools in a variety of domains. In the realm of mathematical reasoning, techniques such as prompt engineering, Chain-of-Thought (CoT) prompting, and program-aided language modeling have notably enhanced LLMs' abilities to solve complex problems (Brown, 2020; Wei et al., 2022; Zhou et al., 2023). These approaches enable models to break down problems into more manageable steps, thereby improving their accuracy and reasoning depth. Moreover, specialized models like MathVerse (Zhang et al., 2024) and Internlm-Math (Ying et al., 2024b), which are trained on extensive mathematical corpora, have achieved significant improvements in mathematical problem-solving performance (Lewkowycz et al., 2022; Ying et al., 2024b). Benchmarks such as GSM8K and MATH further provide a structured means to evaluate and compare these advancements, highlighting the continuous progress in this area (Cobbe et al., 2021; Hendrycks et al., 2021b).

In terms of creativity, LLMs have shown remarkable prowess across diverse fields. They have excelled in generating high-quality, human-like content, ranging from code generation (Ni et al., 2023; Liu et al., 2024a) and music composition (Yuan et al., 2024) to literature (Gómez-Rodríguez & Williams, 2023; Liu et al., 2024b) and educational tools (Lan & Chen, 2024; Orenstrakh et al., 2023). Creativity in LLMs is often evaluated using frameworks like Margaret Boden's criteria (Franceschelli & Musolesi, 2023) and the Torrance Tests of Creative Thinking (TTCT) (Torrance, 1966), where they have demonstrated high fluency, originality, and flexibility. However, the applicability of these traditional creativity metrics to AI systems is still a topic of debate, as they were originally designed to assess human creativity (Zhao et al., 2024). Techniques such as associative thinking have been employed to enhance the creative output of LLMs further, although challenges remain in ensuring that these models can meaningfully integrate unrelated concepts (Mehrotra et al., 2024). The ethical and legal implications of AI-generated creativity continue to be a significant area of concern, underscoring the need for ongoing research to refine evaluation methods and address societal impacts (Lofstead, 2023).

# 3 CREATIVEMATH CURATION

This section details the creation, collection, and processing of our dataset CreativeMath, which comprises high-quality mathematical problems from various competitions and their numerous solutions. The dataset is diverse, encompassing a broad range of mathematical topics and problem types, and covers difficulty levels from middle school to Olympiad level. It includes problems from

![](images/b0f5e510ab8e000395e46fed5a1d44880d17d24eff8d3420c0d95da1d807d39c.jpg)

<details>
<summary>heatmap</summary>

| Math Category | AMC 8 | AMC 10 | AMC 12 | AHSME | AIME | USAJMO | USAMO | IMO |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Algebra | 216 | 386 | 437 | 853 | 273 | 16 | 75 | 113 |
| Arithmetic | 220 | 80 | 54 | 66 | 4 | 0 | 0 | 1 |
| Counting | 82 | 100 | 84 | 36 | 104 | 10 | 18 | 15 |
| Geometry | 253 | 326 | 323 | 530 | 222 | 34 | 87 | 133 |
| Number Theory | 99 | 144 | 128 | 104 | 171 | 20 | 63 | 68 |
| Probability | 51 | 94 | 83 | 36 | 73 | 0 | 6 | 2 |
| Others | 39 | 21 | 26 | 41 | 15 | 3 | 20 | 22 |
</details>

Figure 1: Distribution of problems across different math categories and competitions in the CreativeMath dataset.

eight major US competitions: AMC 8, AMC 10, AMC 12, AHSME, AIME, USAJMO, USAMO, and IMO.

Data Collection. The dataset was sourced from the Art of Problem Solving (AoPS) $^{2}$ , a platform offering the most comprehensive collection of problems from various math competitions, along with multiple solutions contributed by participants over the years. As the most popular and sought-after resource for math competitors, AoPS effectively functions as a natural crowdsourcing platform. It uniquely approximates the complete set of viable human solutions for each problem, with later contributors often building on earlier ones.

We meticulously scraped data from eight competitions, ranging from middle school level to Olympic-level, to capture the breadth of mathematical challenges and the depth of solution strategies available.

Data Cleaning. To ensure the integrity and reliability of the dataset, we conducted a rigorous data cleaning procedure. We accurately extracted LaTeX-formatted problems and solutions from HTML, ensuring their correct representation. Irrelevant comments were removed to make each problem and solution clear and self-sufficient. Samples with images, problems without solutions, or incomplete entries were manually removed from the dataset. After this process, the dataset comprises 6,469 mathematical problems and 14,223 solutions. Each problem in the dataset is tagged with detailed metadata, including difficulty level, math category, and problem type. Difficulty levels and problem types were assigned based on official competition data, while the math category were determined using the Llama-3-70B model.

Dataset Analysis. As shown in Figure 1, the problem distribution inside CreativeMath reveals that Algebra and Geometry are the most represented categories across all competitions. The number of solutions across different competitions, as depicted in Figure 2, reflects the varying complexity of the problems. Medium-difficulty competitions like AMC 10, AMC 12, and AIME typically have a larger number of solutions, as these problems allow for a variety of approaches. In contrast, simpler competitions like AMC 8 tend to have fewer solutions due to the straightforward nature of the problems, which often have limited methods of solving. Olympic-level competitions such as

![](images/0697d5d69e39cb4851d8c075b5a9258518f77972558727c58e8086b14ec9c552.jpg)

<details>
<summary>histogram</summary>

| Category | Number of Problems |
| -------- | ------------------ |
| AMC 8    | 1                  |
| AMC 8    | 2                  |
| AMC 8    | 3                  |
| AMC 8    | 4                  |
| AMC 8    | 5                  |
| AMC 8    | 6                  |
| AMC 8    | 7                  |
| AMC 8    | 8                  |
| AMC 10   | 1                  |
| AMC 10   | 2                  |
| AMC 10   | 3                  |
| AMC 10   | 4                  |
| AMC 10   | 5                  |
| AMC 10   | 6                  |
| AMC 10   | 7                  |
| AMC 10   | 8                  |
| AMC 12   | 1                  |
| AMC 12   | 2                  |
| AMC 12   | 3                  |
| AMC 12   | 4                  |
| AMC 12   | 5                  |
| AMC 12   | 6                  |
| AMC 12   | 7                  |
| AMC 12   | 8                  |
| AHSME    | 1                  |
| AHSME    | 2                  |
| AHSME    | 3                  |
| AHSME    | 4                  |
| AHSME    | 5                  |
| AHSME    | 6                  |
| AHSME    | 7                  |
| AHSME    | 8                  |
</details>

![](images/c81d2ac11ac32bc1eafc87b15dda2660ccf13b9a74305f52ec560b8a5508d37e.jpg)

<details>
<summary>bar</summary>

| Solutions | AIME | USAJMO | USAMO | IMO |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 130 | 260 | 130 | 250 |
| 2 | 125 | 240 | 75 | 50 |
| 3 | 130 | 100 | 40 | 20 |
| 4 | 110 | 120 | 15 | 10 |
| 5 | 70 | 50 | 5 | 5 |
| 6 | 65 | 60 | 5 | 5 |
| 7 | 60 | 65 | 5 | 5 |
| 8 | 55 | 5 | 5 | 5 |
</details>

Figure 2: Distribution of the number of solutions per problem across different competitions.   
![](images/911377519f273dbbb7bb8c9a34dca349844cf5202a5419c8e8ca4420dcdc9272.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Math problem"] --> B["Reference Solution 1"]
    A --> C["Reference Solution 2"]
    A --> D["..."]
    B --> E["LLM"]
    C --> E
    D --> E
    E --> F["New Solution"]
    F --> G["LLM Evaluators"]
    G --> H["Correct Novel"]
    G --> I["Correct Novel"]
    G --> J["Correct Novel"]
    H --> K["Final Decision"]
    I --> K
    J --> K
    K --> L["Correct But Not a Novel Solution"]
    style A fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#fcc,stroke:#333
    style J fill:#fcc,stroke:#333
```
</details>

![](images/9821799b28b5cdccccc8c99e0acd1c849d20b2bfd4ab7db24f23b9440d3122fd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["New Solution"] --> B{Is correct?}
    B -->|YES| C["Correct"]
    B -->|NO| D["Incorrect"]
    C --> E{Distinct from k solutions?}
    E -->|YES| F["Novel"]
    E -->|NO| G["Not Novel"]
    F --> H{Distinct from n solutions?}
    H -->|YES| I["Novel-Unknown"]
    H -->|NO| J["Novel-Known"]
    I --> K["End"]
    J --> L["End"]
```
</details>

Figure 3: The framework includes solution generation (left) and the evaluation pipeline (middle). The flowchart of the detailed evaluation pipeline is illustrated on the right.

USAJMO, USAMO, and IMO also see fewer solutions, likely due to the high complexity of the problems, which limits the number of viable solving strategies.

# 4 METHODS

Our approach consists of a multi-stage pipeline designed to evaluate the novelty of mathematical solutions generated by an LLM. The methodology is structured into four key stages: Novel Solution Generation, Correctness Evaluation, Coarse-Grained Novelty Assessment, and Fine-Grained Novelty Assessment. This comprehensive pipeline illustrated in Figure 3 ensures that the generated solutions are not only correct but also exhibit a meaningful degree of novelty relative to the reference solutions. The sample prompts and LLMs' responses are provided in the Appendix.

# 4.1 NOVEL SOLUTION GENERATION

The first stage of the methodology aims to generate novel solutions for the given mathematical problem using LLM. For each problem, a subset of k reference solutions (where k ranges from 1 to n, with n representing the total number of available reference solutions) is sequentially selected based on the order in which competitors uploaded their solutions on the website. Earlier solutions are often the most common and intuitive, while later ones may build on previous methods, offer improvements, or introduce entirely novel algorithms. Consequently, as k increases, the difficulty in generating new and innovative solutions also increases.

To ensure clarity and consistency in both prompting and evaluating the novelty of generated solutions, we define a set of criteria agreed upon in consultation with several mathematicians. These criteria guide both the generation and the evaluation process and are used to assess the distinctiveness of the solutions. The criteria are as follows:

# Criteria for evaluating the difference between two mathematical solutions include:

1. If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;   
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;   
3. If two solutions rely on different assumptions or conditions, they are likely to be distinct;   
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;   
5. If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem: {problem}

And some typical solutions: {solutions}

Please output a novel solution distinct from the given ones for this math problem.

Figure 4: The prompt template for generating novel solution.

- Methodological Differences: If the methods used to arrive at the solutions are fundamentally different (e.g., algebraic manipulation versus geometric reasoning), the solutions are considered distinct.   
- Intermediate Step Variation: Even if the final results are identical, if the intermediate steps or processes involved in reaching those solutions differ significantly, the solutions are considered novel.   
- Assumptions and Conditions: Solutions that rely on different assumptions, initial conditions, or constraints are treated as distinct.   
- Generalization: A solution that generalizes to a broader class of problems is considered novel compared to one that is specific to certain conditions.   
- Complexity: If one solution is notably simpler or more complex than another, they are regarded as different, even if they lead to the same final result.

These criteria, also illustrated in Figure 4, are embedded into the prompt used to guide the LLM in generating novel solutions. The reference solutions provided to the model aim to capture a variety of approaches, and the LLM is instructed to output a new solution that is distinct according to the defined criteria. The prompt emphasizes generating solutions that use different problem-solving methods, distinct intermediate steps, and variations in assumptions or generalizability.

As part of this process, to avoid influencing the judgment of evaluators during the subsequent evaluation stage, transition sentences and justifications explaining why the new solution is distinct from the reference solutions are manually removed. Only the newly generated solution is presented for evaluation.

# 4.2 CORRECTNESS AND NOVELTY EVALUATION

To rigorously evaluate the correctness and novelty of the generated solutions, we employ three leading LLMs—GPT-4, Claude 3.5 Sonnet, and Gemini 1.5 Pro—as LLM Evaluators, recognized among the strongest models available. These LLM Evaluators collaboratively assess the solutions following the framework illustrated in Figure 3 (middle). Each LLM Evaluator adheres to the flowchart depicted in Figure 3 (right) to systematically evaluate the generated solutions across three dimensions:

- Correctness: The solution must first be validated for correctness, ensuring it produces the correct result for the problem. Only correct solutions proceed to the novelty assessment stages.   
- Coarse-Grained Novelty: If the solution is correct, it is then evaluated for novelty against a subset of $k$ reference solutions. A solution is deemed novel if it is distinct from these $k$ solutions.   
- Fine-Grained Novelty: A solution deemed novel in the coarse-grained assessment undergoes further evaluation against the entire set of $n$ human-provided solutions. This stage distinguishes between:

- Novel-Unknown: A solution that is distinct from all $n$ human-generated solutions, representing a truly original contribution.   
- Novel-Known: A solution that is distinct from the $k$ reference solutions but similar to others in the remaining $n - k$ solutions.

# 4.2.1 EVALUATION STRATEGY

We apply different strategies for correctness and novelty evaluation to ensure both rigor and practicality. For correctness, only solutions unanimously deemed correct by all LLM Evaluators proceed to the novelty assessment, ensuring that only fully reliable solutions are considered. Given the subjective nature of assessing novelty, we use a majority voting strategy, which balances diverse perspectives and effectively identifies genuinely innovative solutions without being overly restrictive.

# 4.2.2 CORRECTNESS EVALUATION

Once a solution is generated, the first essential step is to verify its correctness. The newly generated solution, along with the original problem and a set of reference solutions, is evaluated by the LLM Evaluators using the prompt shown in Figure 5, top. The LLM Evaluators determine if the solution leads to the correct outcome, with responses of “YES” indicating correctness and “NO” indicating otherwise. Only solutions unanimously validated as correct by all LLM Evaluators advance to the novelty assessment stages.

# 4.2.3 COARSE-GRAINED

# NOVELTY ASSESSMENT

Given the following mathematical problem: {problem}

Reference solutions: {solutions}

New solution: {new solution}

Please output YES if the new solution leads to the same result as the reference solutions; otherwise, output NO.

# Criteria for evaluating the novelty of a new mathematical solution include:

1. If the new solution used to arrive at the solutions is fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;

Given the following mathematical problem: {problem}

Reference solutions: {solutions}

New solution: {new solution}

Please output YES if the new solution is a novel solution; otherwise, output NO.

Figure 5: The prompt templates for evaluating the correctness (top) and novelty (bottom) of the generated solution. The criteria for evaluating the novelty are rephrased from the same criteria applied during the novel solution generation process to ensure alignment.

After correctness is established, the next step is to evaluate the solution's novelty at a coarse level. This involves comparing the generated solution against the $k$ reference solutions. The LLM Evaluators assess whether the solution employs distinct approaches or methods that differentiate it from the provided references, using the prompt (Figure 5, bottom). If the solution is considered novel relative to the $k$ reference solutions, it is marked as “YES” and proceeds to the fine-grained novelty assessment.

# 4.2.4 FINE-GRAINED NOVELTY ASSESSMENT

In the final stage, the solution undergoes a fine-grained novelty evaluation to determine its originality in comparison to all n human-generated solutions. This assessment uses the same prompt as the coarse-grained novelty assessment but changes the reference solutions from the subset 1 to k to

the complementary set $k + 1$ to n. The evaluation focuses on whether the solution introduces new insights, methods, or approaches that surpass existing human solutions in terms of innovation, complexity, or generalizability. The outcome categorizes the solution as either a unique contribution or as similar to existing human-generated solutions.

# 5 EXPERIMENT

In this section, we conduct extensive experiments and analyses to show the performance of ten the-state-of-the-art LLMs in math problem solving. We also address several research questions.

# 5.1 DATASET

We selected a subset from our CreativeMath dataset for this study. For each competition, 50 samples were randomly chosen to ensure a representative evaluation of the LLMs' performance. The datasets were meticulously curated to ensure that when the problem and all reference solutions were included in the novel solution generation prompt, the total token count did not exceed 3K tokens. This approach allowed for 1K tokens to be reserved for generation, accommodating the token limits of models like DeepSeek-Math-7B-RL, which has a 4K-token capacity. In total, the dataset comprises 400 math problems and 605 solutions, forming 605 distinct samples with k varying from 1 to 5.

# 5.2 LARGE LANGUAGE MODELS

In this study, we explore the ability of various LLMs to generate novel and creative solutions in mathematical problem-solving. The LLMs selected for this research have demonstrated superior performance on key mathematical benchmarks, such as GSM8K and the MATH dataset, outperforming other models of similar parameter scale. We include three leading close-sourced models—GPT-4o (Version 2024-05-13) (OpenAI, 2024), Claude-3-Opus (Version 2024-02-29) (Anthropic, 2024), and Gemini-1.5-Pro (Reid et al., 2024)—which are renowned for their excellence in complex mathematical reasoning. To ensure a comprehensive evaluation, we also incorporate five top-ranking open-source instruction-tuned LLMs in math reasoning: Llama-3-70B (Meta AI, 2024), Qwen1.5-72B (Bai et al., 2023), Yi-1.5-34B (Young et al., 2024), Mixtral-8x22B-v0.1 (Mistral AI, 2024), and DeepSeek-V2 (DeepSeek-AI, 2024). Furthermore, two specialized mathematical instruction LLMs, DeepSeek-Math-7B-RL (Shao et al., 2024) and Internlm2-Math-20B (Ying et al., 2024a), are included for their advanced capabilities in mathematical reasoning. By selecting these models, we aim to gain a comprehensive understanding of whether their demonstrated excellence in math benchmarks also reflects an enhanced capacity for generating novel solutions.

# 5.3 IMPLEMENTATION DETAILS

For the closed-source LLMs and DeepSeek-V2, we utilized API calls provided by their respective platforms. Open-source LLMs were run using the Hugging Face library on one to four NVIDIA A100 (80G) GPUs, depending on the model's memory requirements. To ensure reproducibility, all experiments were conducted using the greedy decoding strategy, adhering to the recommended settings provided on the official Hugging Face pages or the models' respective papers. The system prompt followed the guidelines outlined in the models' documentation, with the maximum number of new tokens set to 1024. This standardized approach ensures consistent and reliable evaluation across all models used in our study.

# 5.4 EVALUATION METRICS

To assess the effectiveness of LLMs in generating novel solutions, we define several evaluation metrics, as outlined in Table 1. These metrics capture key aspects of the solutions, including correctness, different levels of novelty, and the relationship between novelty and correctness. Importantly, novelty is only considered if the solution is correct, and the Correctness Ratio, Novelty Ratio, and Novel-Unknown Ratio are calculated based on all generated solutions to ensure a consistent evaluation.

<table><tr><td>Symbol</td><td>Metric Definition</td></tr><tr><td> $C$ </td><td>Correctness Ratio: The proportion of solutions that are valid and can solve the problem correctly.</td></tr><tr><td> $N$ </td><td>Novelty Ratio: The proportion of solutions that are both correct and distinct from the provided  $k$  reference solutions.</td></tr><tr><td> $N_{u}$ </td><td>Novel-Unknown Ratio: The proportion of solutions that are both correct and unique compared to all known human-produced solutions  $n$ .</td></tr><tr><td> $N/C$ </td><td>Novelty-to-Correctness Ratio: The ratio of novel solutions to all correct solutions.</td></tr><tr><td> $N_{u}/N$ </td><td>Novel-Unknown-to-Novelty Ratio: The ratio of Novel-Unknown solutions to all available novel solutions.</td></tr></table>

Table 1: Evaluation Metrics and Their Definitions.

<table><tr><td>Source</td><td>Model</td><td>C(%)↑</td><td>N(%)↑</td><td>N/C(%)↑</td><td>Nu(%)↑</td><td>Nu/N(%)↑</td><td>MATH(%)↑</td></tr><tr><td rowspan="3">Closed Source</td><td>Gemini-1.5-Pro</td><td>69.92</td><td>66.94</td><td>95.75</td><td>65.45</td><td>97.78</td><td>58.5</td></tr><tr><td>Claude-3-Opus</td><td>59.84</td><td>44.63</td><td>74.59</td><td>42.98</td><td>96.30</td><td>60.1</td></tr><tr><td>GPT-4o</td><td>60.83</td><td>30.08</td><td>49.46</td><td>27.60</td><td>91.76</td><td>76.6</td></tr><tr><td rowspan="7">Open Source</td><td>Llama-3-70B</td><td>58.84</td><td>48.76</td><td>82.87</td><td>46.94</td><td>96.27</td><td>50.4</td></tr><tr><td>Qwen1.5-72B</td><td>47.44</td><td>33.06</td><td>69.69</td><td>32.40</td><td>98.00</td><td>41.4</td></tr><tr><td>DeepSeek-V2</td><td>63.47</td><td>30.91</td><td>48.70</td><td>29.09</td><td>94.12</td><td>43.6</td></tr><tr><td>Yi-1.5-34B</td><td>42.98</td><td>29.09</td><td>67.69</td><td>28.43</td><td>97.73</td><td>50.1</td></tr><tr><td>Mixtral-8x22B</td><td>56.03</td><td>27.27</td><td>48.67</td><td>25.62</td><td>93.94</td><td>41.8</td></tr><tr><td>Deepseek-Math-7B-RL</td><td>38.35</td><td>12.56</td><td>32.76</td><td>11.57</td><td>92.11</td><td>51.7</td></tr><tr><td>Internlm2-Math-20B</td><td>40.17</td><td>11.90</td><td>29.63</td><td>11.07</td><td>93.06</td><td>37.7</td></tr></table>

Table 2: Experimental results for various closed-source and open-source LLMs on the MultiMath subset ( $\uparrow$ indicates that higher is better). The best-performing models in the open-source and closed-source categories for each evaluation metric are respectively highlighted. MATH column represents the accuracy on MATH datasets with 4-shot (CoT) setting as reported by the corresponding papers or blogs of the LLMs. Refer to Table 1 for detailed definitions of the evaluation metrics used.

# 5.5 RESULTS & DISCUSSIONS

We introduce our results in the context of each of our four research questions and discuss our main findings.

$\mathbb{Q}_1$ : Given a math problem with $n$ known solutions, and an LLM provided with the problem along with $k$ of those solutions, how effectively can the LLM generate a novel solution?

Analysis of Coarse-Grained Novelty Table 2 demonstrates the superior performance of Gemini-1.5-Pro across all evaluated metrics, particularly in its ability to generate novel solutions. With a Novelty Ratio $(N)$ of $66.94\%$ and a Correctness Ratio $(C)$ of $69.92\%$ , Gemini-1.5-Pro not only generates a high number of correct solutions but also ensures that most of these are novel. The model's Novelty-to-Correctness Ratio $(N / C)$ of $95.75\%$ indicates that nearly all correct solutions it produces are distinct from the provided reference solutions.

Llama-3-70B and Claude-3-Opus also perform well in terms of N, with Llama-3-70B achieving a noteworthy N/C of 82.87%. This contrasts sharply with models like GPT-4o, DeepSeek-V2, and Mixtral-8x22B, which, despite similar C values, have N/C ratios below 50%. This discrepancy highlights significant differences in the ability of LLMs to generate novel solutions, even when their correctness levels are comparable. Notably, Llama-3-70B outperforms closed-source models Claude-3-Opus and GPT-4o, suggesting that open-source LLMs can achieve competitive novelty generation capabilities.

In contrast, smaller models like Yi-1.5-34B and specialized math-tuned models such as Deepseek-Math-7B-RL and Internlm2-Math-20B exhibit lower C and N/C ratios. This outcome is consistent with scaling laws (Kaplan et al., 2020), where large models generally outperform compared to small ones. The low N/C in these math-specialized models suggests that their fine-tuning for mathe-

Table 3: Correctness Ratio (C) across different models with varying numbers of reference solutions (k). Sample sizes for k = 1 to k = 4 are 400, 154, 42, and 8, respectively. 

<table><tr><td>Model</td><td>k=1</td><td>k=2</td><td>k=3</td><td>k=4</td></tr><tr><td>Gemini-1.5-Pro</td><td>68.00</td><td>70.78</td><td>78.57</td><td>100</td></tr><tr><td>Llama-3-70B</td><td>55.00</td><td>66.23</td><td>64.29</td><td>75.00</td></tr><tr><td>Claude-3-Opus</td><td>55.00</td><td>66.88</td><td>76.19</td><td>75.00</td></tr><tr><td>Qwen1.5-72B</td><td>43.75</td><td>55.19</td><td>57.14</td><td>37.50</td></tr><tr><td>DeepSeek-V2</td><td>61.00</td><td>66.88</td><td>71.32</td><td>75.00</td></tr><tr><td>GPT-4o</td><td>58.25</td><td>64.94</td><td>66.67</td><td>75.00</td></tr><tr><td>Yi-1.5-34B</td><td>42.75</td><td>42.21</td><td>47.62</td><td>50.00</td></tr><tr><td>Mixtral-8x22B</td><td>53.50</td><td>60.39</td><td>64.28</td><td>62.50</td></tr><tr><td>Deepseek-Math-7B-RL</td><td>35.50</td><td>40.91</td><td>52.38</td><td>50.00</td></tr><tr><td>Internlm2-Math-20B</td><td>38.00</td><td>42.21</td><td>47.62</td><td>62.50</td></tr></table>

matical tasks may limit their adaptability in generating novel solutions outside of their specialized domain.

Analysis of Fine-Grained Novelty The high average Novel-Unknown to Novelty Ratio $(N_{u}/N)$ of 95% across models indicates that the vast majority of novel solutions generated are distinct from any available human solutions. This suggests a substantial potential for these models to contribute genuinely original and innovative solutions that extend beyond the existing human knowledge base. The ability to produce solutions that are not only correct but also novel, surpassing human ingenuity, underscores the LLMs' capacity to explore new solution spaces. This makes them powerful tools for advancing fields that demand creative problem-solving.

Distinctions Between Novel Solution Generation and Math Problem Solving Novel solution generation and traditional math problem-solving differ fundamentally in their structure and evaluation criteria. In traditional math problem solving, typically using few-shot settings with k = 4 fixed reference examples, the task is to solve a new, unseen problem with correctness as the sole criterion. The provided examples consist of different problems and their solutions, with the solution to the target problem being unknown.

In contrast, novel solution generation involves a fixed problem where the model is given varying numbers of reference solutions with $1 \leq k \leq n$ . Here, the solution is known, and the model must not only solve the problem correctly but also generate solutions that are distinct from the provided references. This requirement for distinctiveness adds a layer of complexity, challenging the model's ability to innovate beyond mere correctness.

This distinction is evident in the evaluation metrics. For example, while GPT-4o achieves a high accuracy of 76.6% on the MATH benchmark in Table 2, it performs poorly on novelty metrics, indicating a limited ability to generate distinct solutions despite its problem-solving accuracy. This contrast underscores the more stringent demands of novel solution generation, where models must demonstrate creativity and innovation in addition to correctness.

$\mathbb{Q}_2$ : How does the number of provided solutions, $k$ , affect the LLM's performance in generating new solutions?

Impact of k on Correctness This section examines how increasing the number of provided reference solutions $(k)$ affects the correctness of generated solutions, as shown in Table 3. Across most models, there is a clear trend of improved correctness with larger k values. For example, Gemini-1.5-Pro reaches 100% correctness at k = 4, demonstrating its ability to effectively utilize additional examples. This trend is consistent with findings in few-shot learning, where more examples typically lead to better model performance (Brown, 2020). Models like Llama-3-70B and DeepSeek-V2 show moderate improvements with increased k, though the gains are less pronounced compared to Gemini-1.5-Pro. In contrast, models like Qwen1.5-72B and Yi-1.5-34B show minimal increases in correctness, potentially due to variability introduced by smaller sample sizes at higher k values.

Impact of the Degree of Solution Availability $(n - k)$ on Novelty The degree of solution availability, denoted by n - k, represents the gap between the total available solutions and those provided

Table 4: Novelty-to-Correctness Ratio $(N / C)$ for different models based on the degree of solution availability $(n - k)$ . Higher values of $n - k$ indicate scenarios with fewer provided solutions, which are easier for the LLM. 

<table><tr><td>Model</td><td>n-k=2</td><td>n-k=1</td><td>n-k=0</td></tr><tr><td>Gemini-1.5-Pro</td><td>100</td><td>95.92</td><td>95.10</td></tr><tr><td>Llama-3-70B</td><td>87.50</td><td>85.26</td><td>81.03</td></tr><tr><td>Claude-3-Opus</td><td>91.67</td><td>72.94</td><td>73.68</td></tr><tr><td>Qwen1.5-72B</td><td>85.00</td><td>70.15</td><td>68.37</td></tr><tr><td>DeepSeek-V2</td><td>36.00</td><td>54.17</td><td>47.84</td></tr><tr><td>GPT-4o</td><td>57.69</td><td>53.33</td><td>47.35</td></tr><tr><td>Yi-1.5-34B</td><td>52.38</td><td>52.87</td><td>46.43</td></tr><tr><td>Mixtral-8x22B</td><td>33.33</td><td>35.48</td><td>56.07</td></tr><tr><td>Deepseek-Math-7B-RL</td><td>27.78</td><td>25.86</td><td>35.10</td></tr><tr><td>Internlm2-Math-20B</td><td>15.00</td><td>27.69</td><td>32.89</td></tr></table>

Table 5: Average Correctness (C) and Novelty-to-Correctness Ratio (N/C) for all LLMs when solving math problems of varying difficulty levels, with k = 1 across all competitions. 

<table><tr><td>Competition</td><td>Difficulty</td><td>k</td><td>Average C</td><td>Average N/C</td></tr><tr><td>AMC 8</td><td>1-1.5</td><td>1</td><td>71.80</td><td>55.39</td></tr><tr><td>AMC 10</td><td>1-3</td><td>1</td><td>67.20</td><td>59.96</td></tr><tr><td>AHSME</td><td>1-4</td><td>1</td><td>65.08</td><td>63.11</td></tr><tr><td>AMC 12</td><td>2-4</td><td>1</td><td>60.40</td><td>54.05</td></tr><tr><td>AIME</td><td>3-6</td><td>1</td><td>35.80</td><td>55.55</td></tr><tr><td>USAJMO</td><td>6-7</td><td>1</td><td>37.00</td><td>77.23</td></tr><tr><td>USAMO</td><td>7-9</td><td>1</td><td>35.00</td><td>83.01</td></tr><tr><td>IMO</td><td>5.5-10</td><td>1</td><td>35.60</td><td>78.86</td></tr></table>

to the model. A higher n - k means fewer distinct solutions are given, leaving more room for the model to explore and innovate. This typically results in fewer constraints, facilitating the generation of novel outputs. As k increases and n - k decreases, the model is exposed to more reference solutions, tightening the constraints and making it harder to generate novel solutions. This pattern is evident in Table 4, where models generally show higher Novelty-to-Correctness Ratios (N/C) at higher n - k values, with Gemini-1.5-Pro achieving a perfect N/C at n - k = 2. However, as n - k decreases, the ability to produce novel solutions diminishes. This mirrors human problem-solving, where creativity often diminishes when more examples are provided, as the model (or individual) must work within tighter constraints.

$\mathbb{Q}_3$ : How does the creativity of LLMs vary when solving math problems of varying difficulty levels?

We analyzed the correctness $(C)$ and Novelty-to-Correctness Ratio $(N/C)$ of all LLMs across competitions of different difficulty levels, focusing on problems where k = 1 to ensure consistency. As shown in Table 5, as problem difficulty increases, the correctness of LLMs consistently decreases, dropping from 71.80% on AMC 8 problems to around 35% on more challenging competitions like USAMO and IMO. Conversely, the N/C ratio increases with difficulty, from 55.39% on easier problems to 83.01% on the most difficult ones. This suggests that while LLMs struggle with accuracy on harder problems, they are more likely to generate novel solutions when they do succeed. The observed trend indicates a shift in the balance between familiarity and innovation: as problem difficulty rises, LLMs are pushed to rely less on familiar strategies and more on creative problem-solving. This complex interplay between familiarity and innovation becomes more pronounced with increasing problem difficulty, leading to a higher likelihood of novel solutions.

$\mathbb{Q}_{4}$ : When different LLMs are given the same math problem and k solutions, how likely are the new solutions generated by these LLMs to be identical or distinct? Additionally, how does the pairwise similarity between the solutions generated by different LLMs inform us about their tendencies to produce similar outputs?

To explore the tendency of different LLMs to generate novel solutions, we first measured pairwise similarity between the outputs of various models. We conducted an experiment using 17 samples where all included LLMs were capable of generating novel solutions. Math-specialized LLMs were excluded due to their low novelty ratios. For each pair of LLMs, we used the same prompt as in the novelty assessment, but replaced the reference solution with the solution generated by one LLM and the new solution with that generated by another LLM. The pairwise similarity was determined based on whether the solutions were distinct (“YES”) or similar (“NO”). The similarity score for each LLM pair was computed as the ratio of similar solutions to the total number of samples (17).

![](images/7a9321d57a3eb429d23e302b91c2c3a89938e22f4794fff107e60a9a5ec26600.jpg)

<details>
<summary>scatter</summary>

| Model           | MDS Dimension 1 | MDS Dimension 2 |
| --------------- | --------------- | --------------- |
| GPT-4o          | -0.3            | 0.3             |
| Llama-3-70B     | -0.6            | -0.1            |
| Mixtral-8x22B   | -0.1            | 0.0             |
| Qiwen1.5-72B    | -0.3            | -0.5            |
| Claude-3-Opus   | 0.2             | -0.1            |
| Yi-1.5-34B      | 0.4             | -0.5            |
| DeepSeek-V2     | 0.5             | 0.2             |
</details>

Figure 6: Similarity map between the novel solutions generated by different LLMs.

We applied Multidimensional Scaling (MDS) to the pairwise similarity matrix, mapping the LLMs into a two-dimensional space. As illustrated in Figure 6, the similarity map reveals a general trend of low similarity between the novel solutions generated by different LLMs. The most distinct pair, Llama-3-70B and Yi-1.5-34B, shows only a 6% similarity, indicating that these models explore vastly different solution spaces. On the other hand, the most similar pairs—Mixtral-8x22B with GPT-4o and Mixtral-8x22B with Claude-3-Opus—each show a 47% similarity. Mixtral-8x22B, positioned centrally in the similarity map, tends to produce solutions that are slightly more similar to those of other models. This analysis suggests that leveraging multiple LLMs positioned on the periphery of the similarity map could be a promising approach to generate diverse novel solutions. These models, exploring vastly different solution spaces, are likely to enhance the efficiency and breadth of problem-solving strategies.

# 6 CONCLUSION

In this study, we introduced the CreativeMath dataset and developed a comprehensive framework that encompasses both the generation of novel solutions by LLMs and their rigorous evaluation. This framework is designed to assess the creative potential of LLMs in mathematical problem-solving, systematically distinguishing between solutions that are merely correct and those that offer genuinely innovative approaches. Our findings reveal significant variability in the creative abilities of state-of-the-art LLMs, emphasizing the importance of advancing AI systems that not only solve problems accurately but also contribute original insights. We encourage future research to delve deeper into methodologies for uncovering and assessing the creative capabilities of LLMs, particularly in complex and abstract domains like mathematics.

# REFERENCES

Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. In Neele Falk, Sara Papi, and Mike Zhang (eds.), Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2024: Student Research Workshop, St. Julian's, Malta, March 21-22, 2024, pp. 225–237. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.eacl-srw.17.   
Anthropic. The claude 3 model family: Opus, sonnet, haiku, 2024. URL https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model\_Card\_Claude\_3.pdf. Accessed: 2024-08-13.   
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
Tom B Brown. Language models are few-shot learners. arXiv preprint ArXiv:2005.14165, 2020.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.   
Giorgio Franceschelli and Mirco Musolesi. On the creativity of large language models. arXiv preprint arXiv:2304.00008, 2023.   
Carlos Gómez-Rodríguez and Paul Williams. A confederacy of models: A comprehensive evaluation of llms on creative writing. arXiv preprint arXiv:2310.08433, 2023.   
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021a.   
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
Yu-Ju Lan and Nian-Shing Chen. Teachers' agency in the era of llm and generative ai. Educational Technology & Society, 27(1):I-XVIII, 2024.   
Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models, 2022. URL https://arxiv.org/abs/2206.14858, 2022.   
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chat-gpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36, 2024a.   
Yiren Liu, Si Chen, Haocong Cheng, Mengxia Yu, Xiao Ran, Andrew Mo, Yiliu Tang, and Yun Huang. How ai processing delays foster creativity: Exploring research question co-creation with an llm-based agent. In Proceedings of the CHI Conference on Human Factors in Computing Systems, pp. 1–25, 2024b.

Jay Lofstead. Economic, societal, legal, and ethical considerations for large language models. In 2023 Fifth International Conference on Transdisciplinary AI (TransAI), pp. 155–162. IEEE, 2023.   
Pronita Mehrotra, Aishni Parab, and Sumit Gulwani. Enhancing creativity in large language models through associative thinking strategies. arXiv preprint arXiv:2405.06715, 2024.   
Meta AI. Meta llama 3. https://ai.meta.com/blog/meta-llama-3/, 2024. Accessed: 22-October-2024.   
Mistral AI. Mixtral 8x22b: The new frontier in ai models, 2024. URL https://mistral.ai/news/mixtral-8x22b/. Accessed: 2024-08-13.   
Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. Lever: Learning to verify language-to-code generation with execution. In International Conference on Machine Learning, pp. 26106–26128. PMLR, 2023.   
OpenAI. Hello gpt-4o!, 2024. URL https://openai.com/index/hello-gpt-4o/. Accessed: 2024-08-13.   
Michael Sheinman Orenstrakh, Oscar Karnalim, Carlos Anibal Suarez, and Michael Liut. Detecting llm-generated text in computing education: A comparative study for chatgpt cases. arXiv preprint arXiv:2307.07411, 2023.   
Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.   
Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nat., 625(7995):468–475, 2024. doi: 10.1038/S41586-023-06924-6. URL https://doi.org/10.1038/s41586-023-06924-6.   
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
E Paul Torrance. Torrance tests of creative thinking. Educational and psychological measurement, 1966.   
Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nat., 625(7995):476–482, 2024. doi: 10.1038/S41586-023-06747-5. URL https://doi.org/10.1038/s41586-023-06747-5.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. Internlm-math: Open math large language models toward verifiable reasoning, 2024a.   
Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, et al. Internlm-math: Open math large language models toward verifiable reasoning. arXiv preprint arXiv:2402.06332, 2024b.   
Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.

Ruibin Yuan, Hanfeng Lin, Yi Wang, Zeyue Tian, Shangda Wu, Tianhao Shen, Ge Zhang, Yuhang Wu, Cong Liu, Ziya Zhou, et al. Chatmusician: Understanding and generating music intrinsically with llm. arXiv preprint arXiv:2402.16153, 2024.   
Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.   
Yunpu Zhao, Rui Zhang, Wenyi Li, Di Huang, Jiaming Guo, Shaohui Peng, Yifan Hao, Yuanbo Wen, Xing Hu, Zidong Du, et al. Assessing and understanding creativity in large language models. arXiv preprint arXiv:2401.12491, 2024.   
Jiajun Zhou, Shijie Rao, Liang Gao, Chunjiang Zhang, Hongtao Tang, Yun Li, and Felix TS Chan. Solving many-task optimization problems via online intertask learning. Expert Systems with Applications, 225:120110, 2023.

# A APPENDIX

# A.1 DIFFERENT PROMPTS AND LLM RESPONSES

To assess the creativity and reasoning capabilities of LLMs, we designed a series of prompts that require the generation of novel solutions to mathematical problems. The prompts were crafted to test various aspects of the models' reasoning processes, including correctness and novelty. Below is a detailed description of the prompts used and the corresponding responses from different LLMs.

# A.1.1 PROMPT 1. NOVEL SOLUTION GENERATION

Task: The LLM is provided with a mathematical problem and a set of reference solutions. The task is to generate a new, distinct solution that is novel compared to the given reference solutions.

Below is the prompt for generating a novel solution for an AMC 8 problem with k = 4 reference solutions. Typically, the subsequent solutions may be modifications of the previous ones; for example, solution 2 is a slightly different version of solution 1.

# Criteria for evaluating the difference between two mathematical solutions include:

i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;   
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;   
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;   
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct; v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

# Given the following mathematical problem:

What is the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ ?

(A) 8

(B) 16

(C) 32

(D) 64

(E) 128

# And some typical solutions:

# Solution 1:

First, we use difference of squares on $13^{4} - 11^{4} = (13^{2})^{2} - (11^{2})^{2}$ to get $13^{4} - 11^{4} = (13^{2} + 11^{2})(13^{2} - 11^{2})$ . Using difference of squares again and simplifying, we get $(169 + 121)(13 + 11)(13 - 11) = 290 \cdot 24 \cdot 2 = (2 \cdot 8 \cdot 2) \cdot (3 \cdot 145)$ . Realizing that we don't need the right-hand side because it doesn't contain any factor of 2, we see that the greatest power of 2 that is a divisor $13^{4} - 11^{4}$ is (C) 32.

# Solution 2:

Just like in the above solution, we use the difference-of-squares factorization, but only once to get $13^{4} - 11^{4} = (13^{2} - 11^{2})(13^{2} + 11^{2})$ . We can then compute that this is equal to $48 \cdot 290$ . Note that $290 = 2 \cdot 145$ (we don't need to factorize any further as 145 is already odd) thus the largest power of 2 that divides 290 is only $2^{1} = 2$ , while $48 = 2^{4} \cdot 3$ , so the largest power of 2 that divides 48 is $2^{4} = 16$ . Hence, the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2 \cdot 16 = \boxed{(\mathbf{C})\,32}$ .

# Solution 3:

Let $n = 13^4 - 11^4$ . We wish to find the largest power of 2 that divides $n$ .

Denote $v_{p}(k)$ as the largest exponent of $p$ in the prime factorization of $n$ . In this problem, we have $p = 2$ .

By the Lifting the Exponent Lemma on n,

$$
\begin{array}{l} v _ {2} (1 3 ^ {4} - 1 1 ^ {4}) = v _ {2} (1 3 - 1 1) + v _ {2} (4) + v _ {2} (1 3 + 1 1) - 1 \\ = v _ {2} (2) + v _ {2} (4) + v _ {2} (2 4) - 1 \\ = 1 + 2 + 3 - 1 = 5. \\ \end{array}
$$

Therefore, exponent of the largest power of 2 that divides $13^{4} - 11^{4}$ is 5, so the largest power of 2 that divides this number is $2^{5} = \boxed{(\mathbf{C})32}$ .

# Solution 4:

We can simply take 13 to the 4th power, which is 28561. We subtract that by 11 to the 4th power, which is 14641 (You can use Pascal's Triangle to find this). Finally, subtract the numbers to get 13920.

To test the options, since we need the largest one, we can go from top down. Testing, we see that both D and E are decimals,

and 32 works. So, our answer is (C) 32.

Please output a novel solution distinct from the given ones for this math problem.

The new solution generated by Yi-1.5-34B.

We can rewrite the expression $13^4 - 11^4$ as $(13^2)^2 - (11^2)^2$ . This is a difference of squares, which factors as $(13^2 + 11^2)(13^2 - 11^2)$ .

Now, notice that $13^{2}-11^{2}$ can be factored further as $(13+11)(13-11)$ , which simplifies to $24\cdot2$ .

The expression now looks like this: $(13^{2} + 11^{2})(24 \cdot 2)$ .

We can see that $13^{2} + 11^{2}$ is clearly greater than 24, so the largest power of 2 that divides $13^{2} + 11^{2}$ must be at least $2^{2} = 4$ .

Since 24 is divisible by $2^3 = 8$ , and 2 is divisible by $2^1 = 2$ , the largest power of 2 that divides the entire expression is $2^3 \cdot 2^1 = 2^4 = 16$ .

However, we are looking for the largest power of 2 that divides $13^{4} - 11^{4}$ , not the entire expression. We know that $13^{4} - 11^{4} = (13^{2} + 11^{2})(24 \cdot 2)$ , and since $13^{2} + 11^{2}$ is not divisible by 2, the largest power of 2 that divides $13^{4} - 11^{4}$ is the same as the largest power of 2 that divides $24 \cdot 2$ , which is $2^{3} \cdot 2^{1} = 2^{4} = 16$ .

But we are not done yet. Notice that $24 \cdot 2 = 48$ , and 48 is divisible by $2^4 = 16$ . Therefore, the largest power of 2 that divides $13^4 - 11^4$ is $2^4 = 16$ .

However, we need to consider that the question asks for the largest power of 2 that is a divisor, not just any power of 2. Since 16 is a divisor of 48, and 48 is a divisor of $13^{4} - 11^{4}$ , the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2^{4} = 16$ .

But we are still not done. We need to consider that the question asks for the largest power of 2 that is a divisor, not just any power of 2. Since 16 is a divisor of 48, and 48 is a divisor of $13^{4} - 11^{4}$ , the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2^{4} = 16$ .

Therefore, the answer is (C) 32.

# A.1.2 PROMPT 2. CORRECTNESS EVALUATION

Task: The LLM is provided with a mathematical problem and a set of reference solutions. The task is to determine whether a newly generated solution matches the correctness of the reference solutions.

Below is the prompt to evaluate the correctness of the solution generated above by Yi-1.5-34B. The first two reference solutions are provided to the LLM for determination.

# Given the following mathematical problem:

What is the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ ?

(A) 8

(B) 16

(C) 32

(D) 64

(E) 128

# Reference solutions:

# Solution 1:

First, we use difference of squares on $13^{4} - 11^{4} = (13^{2})^{2} - (11^{2})^{2}$ to get $13^{4} - 11^{4} = (13^{2} + 11^{2})(13^{2} - 11^{2})$ . Using difference of squares again and simplifying, we get $(169 + 121)(13 + 11)(13 - 11) = 290 \cdot 24 \cdot 2 = (2 \cdot 8 \cdot 2) \cdot (3 \cdot 145)$ . Realizing that we don't need the right-hand side because it doesn't contain any factor of 2, we see that the greatest power of 2 that is a divisor $13^{4} - 11^{4}$ is (C) 32.

# Solution 2:

Just like in the above solution, we use the difference-of-squares factorization, but only once to get $13^{4} - 11^{4} = (13^{2} - 11^{2})(13^{2} + 11^{2})$ . We can then compute that this is equal to $48 \cdot 290$ . Note that $290 = 2 \cdot 145$ (we don't need to factorize any further as 145 is already odd) thus the largest power of 2 that divides 290 is only $2^{1} = 2$ , while $48 = 2^{4} \cdot 3$ , so the largest power of 2 that divides 48 is $2^{4} = 16$ . Hence, the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2 \cdot 16 = \boxed{(\mathbf{C})\ 32}$ .

# New solution:

We can rewrite the expression $13^4 - 11^4$ as $(13^2)^2 - (11^2)^2$ . This is a difference of squares, which factors as $(13^2 + 11^2)(13^2 - 11^2)$ .

Now, notice that $13^{2}-11^{2}$ can be factored further as $(13+11)(13-11)$ , which simplifies to $24\cdot2$ .

The expression now looks like this: $(13^{2} + 11^{2})(24 \cdot 2)$ .

We can see that $13^{2} + 11^{2}$ is clearly greater than 24, so the largest power of 2 that divides $13^{2} + 11^{2}$ must be at least $2^{2} = 4$ .

Since 24 is divisible by $2^3 = 8$ , and 2 is divisible by $2^1 = 2$ , the largest power of 2 that divides the entire expression is $2^3 \cdot 2^1 = 2^4 = 16$ .

However, we are looking for the largest power of 2 that divides $13^{4} - 11^{4}$ , not the entire expression. We know that $13^{4} - 11^{4} = (13^{2} + 11^{2})(24 \cdot 2)$ , and since $13^{2} + 11^{2}$ is not divisible by 2, the largest power of 2 that divides $13^{4} - 11^{4}$ is the same as the largest power of 2 that divides $24 \cdot 2$ , which is $2^{3} \cdot 2^{1} = 2^{4} = 16$ .

But we are not done yet. Notice that $24 \cdot 2 = 48$ , and 48 is divisible by $2^4 = 16$ . Therefore, the largest power of 2 that divides $13^4 - 11^4$ is $2^4 = 16$ .

However, we need to consider that the question asks for the largest power of 2 that is a divisor, not just any power of 2. Since 16 is a divisor of 48, and 48 is a divisor of $13^{4} - 11^{4}$ , the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2^{4} = 16$ .

But we are still not done. We need to consider that the question asks for the largest power of 2 that is a divisor, not just any power of 2. Since 16 is a divisor of 48, and 48 is a divisor of $13^{4} - 11^{4}$ , the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2^{4} = 16$ .

Therefore, the answer is (C) 32.

Please output YES if the new solution leads to the same result as the reference solutions; otherwise, output NO.

The new solution by Yi-1.5-34B provides the correct answer but with flawed reasoning. GPT-4, Gemini-1.5-Pro, and Claude-3.5-Sonnet all give “NO”, as all three LLM Evaluators determined that the generated solution is incorrect.

Now, let us consider the new generated solution by Gemini-1.5-Pro, which is based on modular arithmetic.

# Given the following mathematical problem:

What is the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ ?

(A) 8

(B) 16

(C) 32

(D) 64

(E) 128

# Reference solutions:

# Solution 1:

First, we use difference of squares on $13^{4} - 11^{4} = (13^{2})^{2} - (11^{2})^{2}$ to get $13^{4} - 11^{4} = (13^{2} + 11^{2})(13^{2} - 11^{2})$ . Using difference of squares again and simplifying, we get $(169 + 121)(13 + 11)(13 - 11) = 290 \cdot 24 \cdot 2 = (2 \cdot 8 \cdot 2) \cdot (3 \cdot 145)$ . Realizing that we don't need the right-hand side because it doesn't contain any factor of 2, we see that the greatest power of 2 that is a divisor $13^{4} - 11^{4}$ is (C) 32.

# Solution 2:

Just like in the above solution, we use the difference-of-squares factorization, but only once to get $13^{4} - 11^{4} = (13^{2} - 11^{2})(13^{2} + 11^{2})$ . We can then compute that this is equal to $48 \cdot 290$ . Note that $290 = 2 \cdot 145$ (we don't need to factorize any further as 145 is already odd) thus the largest power of 2 that divides 290 is only $2^{1} = 2$ , while $48 = 2^{4} \cdot 3$ , so the largest power of 2 that divides 48 is $2^{4} = 16$ . Hence, the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2 \cdot 16 = \boxed{(\mathbf{C})\ 32}$ .

# New solution:

We can solve this problem by considering the remainders when powers of 2 divide 13 and 11.

\* Notice that $13 \equiv 1 \pmod{8}$ (13 leaves a remainder of 1 when divided by 8).

\* Therefore, $13^{4} \equiv 1^{4} \equiv 1 \pmod{8}$ .

\* Similarly, 11 ≡ 3 (mod 8), so 11 $^{4}$ ≡ 3 $^{4}$ ≡ 81 ≡ 1 (mod 8).

Since both $13^{4}$ and $11^{4}$ have the same remainder (1) when divided by 8, their difference is divisible by 8:

\* $13^{4} - 11^{4}\equiv 1 - 1\equiv 0$ (mod 8)

Now, let's check divisibility by 16:

\* $13 \equiv -3 \pmod{16}$ so $13^{4} \equiv (-3)^{4} \equiv 81 \equiv 1 \pmod{16}$

\* 11 ≡ -5 (mod 16) so 11 $^{4}$ ≡ (-5) $^{4}$ ≡ 625 ≡ 1 (mod 16)

Again, the difference is divisible by 16:

\* $13^{4} - 11^{4}\equiv 1 - 1\equiv 0$ (mod 16)

Finally, let's check divisibility by 32:

\* $13^{2} \equiv 9 \pmod{32}$ so $13^{4} \equiv 9^{2} \equiv 81 \equiv 17 \pmod{32}$

\* $11^{2} \equiv 5 \pmod{32}$ so $11^{4} \equiv 5^{2} \equiv 25 \pmod{32}$

This time, the difference is not divisible by 32:

\* $13^{4} - 11^{4}\equiv 17 - 25\equiv -8$ (mod 32)

Since the difference is divisible by 16 but not by 32, the largest power of 2 that divides $13^{4} - 11^{4}$ is (C) 32.

Please output YES if the new solution leads to the same result as the reference solutions; otherwise, output NO.

The solution by Gemini-1.5-Pro is correct, as verified by all LLM Evaluators. After the correctness evaluation, the prompt for novelty evaluation is used.

# A.1.3 PROMPT 3. NOVELTY EVALUATION

Task: The LLM is asked to evaluate whether a newly generated solution is novel compared to existing reference solutions.

Criteria for evaluating the novelty of a new mathematical solution include: 1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;

2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;   
3. If the new solution relies on different assumptions or conditions, it should be considered novel;   
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;   
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

# Given the following mathematical problem:

What is the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ ?

(A) 8

(B) 16

(C) 32

(D) 64

(E) 128

# Reference solutions:

# Solution 1:

First, we use difference of squares on $13^4 - 11^4 = (13^2)^2 - (11^2)^2$ to get $13^4 - 11^4 = (13^2 + 11^2)(13^2 - 11^2)$ . Using difference of squares again and simplifying, we get $(169 + 121)(13 + 11)(13 - 11) = 290 \cdot 24 \cdot 2 = (2 \cdot 8 \cdot 2) \cdot (3 \cdot 145)$ . Realizing that we don't need the right-hand side because it doesn't contain any factor of 2, we see that the greatest power of 2 that is a divisor $13^4 - 11^4$ is (C) 32.

# Solution 2:

Just like in the above solution, we use the difference-of-squares factorization, but only once to get $13^{4} - 11^{4} = (13^{2} - 11^{2})(13^{2} + 11^{2})$ . We can then compute that this is equal to $48 \cdot 290$ . Note that $290 = 2 \cdot 145$ (we don't need to factorize any further as 145 is already odd) thus the largest power of 2 that divides 290 is only $2^{1} = 2$ , while $48 = 2^{4} \cdot 3$ , so the largest power of 2 that divides 48 is $2^{4} = 16$ . Hence, the largest power of 2 that is a divisor of $13^{4} - 11^{4}$ is $2 \cdot 16 = \boxed{(\mathbf{C})\ 32}$ .

# Solution 3:

Let $n = 13^4 - 11^4$ . We wish to find the largest power of 2 that divides $n$ .

Denote $v_{p}(k)$ as the largest exponent of $p$ in the prime factorization of $n$ . In this problem, we have $p = 2$ .

By the Lifting the Exponent Lemma on n,

$$
\begin{array}{l} v _ {2} (1 3 ^ {4} - 1 1 ^ {4}) = v _ {2} (1 3 - 1 1) + v _ {2} (4) + v _ {2} (1 3 + 1 1) - 1 \\ = v _ {2} (2) + v _ {2} (4) + v _ {2} (2 4) - 1 \\ = 1 + 2 + 3 - 1 = 5. \\ \end{array}
$$

Therefore, exponent of the largest power of 2 that divides $13^{4} - 11^{4}$ is 5, so the largest power of 2 that divides this number is $2^{5} = \boxed{(\mathbf{C})32}$ .

# Solution 4:

We can simply take 13 to the 4th power, which is 28561. We subtract that by 11 to the 4th power, which is 14641 (You can use Pascal's Triangle to find this). Finally, subtract the numbers to get 13920.

To test the options, since we need the largest one, we can go from top down. Testing, we see that both D and E are decimals, and 32 works. So, our answer is (C) 32.

# New solution:

We can solve this problem by considering the remainders when powers of 2 divide 13 and 11.

\* Notice that $13 \equiv 1 \pmod{8}$ (13 leaves a remainder of 1 when divided by 8).   
\* Therefore, $13^{4} \equiv 1^{4} \equiv 1 \pmod{8}$ .   
\* Similarly, $11 \equiv 3 \pmod{8}$ , so $11^{4} \equiv 3^{4} \equiv 81 \equiv 1 \pmod{8}$ .   
Since both $13^{4}$ and $11^{4}$ have the same remainder (1) when divided by 8, their difference is divisible by 8:

$$
* 1 3 ^ {4} - 1 1 ^ {4} \equiv 1 - 1 \equiv 0 (\mathrm{mod} 8)
$$

Now, let's check divisibility by 16:

$$
\begin{array}{l} * 1 3 \equiv - 3 (\mathrm{mod} 1 6) \text {so} 1 3 ^ {4} \equiv (- 3) ^ {4} \equiv 8 1 \equiv 1 (\mathrm{mod} 1 6) \\ * 1 1 \equiv - 5 (\mathrm{mod} 1 6) \text {so} 1 1 ^ {4} \equiv (- 5) ^ {4} \equiv 6 2 5 \equiv 1 (\mathrm{mod} 1 6) \\ \end{array}
$$

Again, the difference is divisible by 16:

$$
* 1 3 ^ {4} - 1 1 ^ {4} \equiv 1 - 1 \equiv 0 (\mathrm{mod} 1 6)
$$

Finally, let's check divisibility by 32:

$$
\begin{array}{l} * 1 3 ^ {2} \equiv 9 (\mathrm{mod} 3 2) \text {so} 1 3 ^ {4} \equiv 9 ^ {2} \equiv 8 1 \equiv 1 7 (\mathrm{mod} 3 2) \\ * 1 1 ^ {2} \equiv 5 (\mathrm{mod} 3 2) \text {so} 1 1 ^ {4} \equiv 5 ^ {2} \equiv 2 5 (\mathrm{mod} 3 2) \\ \end{array}
$$

This time, the difference is not divisible by 32:

$$
* 1 3 ^ {4} - 1 1 ^ {4} \equiv 1 7 - 2 5 \equiv - 8 (\mathrm{mod} 3 2)
$$

Since the difference is divisible by 16 but not by 32, the largest power of 2 that divides $13^{4} - 11^{4}$ is $\boxed{(\mathbf{C})\ 32}$ .

# Please output YES if the new solution is a novel solution; otherwise, output NO.

The new solution by Gemini-1.5-Pro is distinct from all reference solutions, which has been classified as a novel solution by all three LLM Evaluators. The reasoning and decision why Claude-3.5-Connect classified this solution as a novel solution are illustrated as follows.

To evaluate the novelty of the new solution, let's compare it against the criteria provided and the reference solutions:

1. The new solution uses modular arithmetic and congruences, which is fundamentally different from the algebraic manipulations and factorizations used in the reference solutions. This approach is distinct from the methods used in the reference solutions.   
2. The intermediate steps in the new solution involve checking remainders when dividing by powers of 2, which is significantly different from the processes in the reference solutions.   
3. The new solution relies on the properties of modular arithmetic and congruences, which are different assumptions from those used in the reference solutions.   
4. While the new solution doesn't necessarily generalize to a broader class of problems, it does provide a different perspective on approaching such problems.   
5. The new solution is neither significantly simpler nor more complex than the reference solutions, but it does offer a distinct approach to solving the problem.   
Given these considerations, particularly the fundamentally different approach (criterion 1) and the distinct intermediate steps (criterion 2), we can conclude that this new solution meets the criteria for novelty.

# Output: YES